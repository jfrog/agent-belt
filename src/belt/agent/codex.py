# (c) JFrog Ltd. (2026)

"""CodexAgentAdapter - drives the OpenAI Codex CLI through evaluation scenarios.

Targets the native ``codex-cli 0.130+`` binary. The first turn invokes
``codex exec --json --skip-git-repo-check -o <reply-file> ... -- <message>``
and the subsequent turns invoke ``codex exec resume <session_id> --json
-o <reply-file> -m <model> -c model_provider="<provider>"
-c projects.<cwd>.trust_level="trusted" ... -- <message>``. JSONL events
are streamed to capture per-turn timing (``ttfe``, ``ttft``, ``ttlt``)
and tool-call extraction; the canonical reply text is read from the file
written by ``--output-last-message`` so transient ``error`` events and
reconnect retries cannot poison it.

Authentication
--------------
Per the ``check_available()`` contract, this agent does not probe
authentication state (see ``BaseAgentAdapter`` docstring). Codex caches
login at ``~/.codex/auth.json`` and additionally honours provider-scoped
env vars named in ``~/.codex/config.toml``'s
``[model_providers.<name>] env_key`` (typically ``OPENAI_API_KEY`` for
OpenAI direct, ``AZURE_OPENAI_API_KEY`` for Azure deployments configured
via the Microsoft Foundry recipe). The adapter passes those keys plus
``OPENAI_BASE_URL``, ``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_BASE_URL``,
and ``CODEX_HOME`` through belt's scrubbed subprocess env so users do
not need ``BELT_ALLOW_FULL_ENV=1`` for Azure routing.

References:
    - https://github.com/openai/codex
    - https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/codex
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from loguru import logger

from belt.agent.base import (
    AgentNotAvailableError,
    AgentOption,
    BaseAgentAdapter,
    _drain_stderr,
    _kill_process_tree,
    _sanitize_stderr,
    iter_bounded_stream,
    resolve_binary,
)
from belt.agent.error_types import UNKNOWN, normalize_error_type
from belt.entities import ToolCall, TurnOutput, TurnTiming
from belt.parser.ndjson import parse_ndjson
from belt.runner.entities import AgentConfig
from belt.scenario import GroupConfig

_MIN_VERSION: tuple[int, int, int] = (0, 130, 0)
_VERSION_PATTERN = re.compile(r"codex-cli\s+(\d+)\.(\d+)\.(\d+)")
_INSTALL_HINT = "Install: brew install --cask codex (macOS) or npm install -g @openai/codex"
_UPGRADE_HINT = "Upgrade: brew install --cask codex (macOS) or npm install -g @openai/codex"


def _has_sandbox_flag(flags: list[str]) -> bool:
    """True iff ``flags`` declares an explicit ``-s`` / ``--sandbox``."""
    for flag in flags:
        key = flag.split("=", 1)[0]
        if key in ("-s", "--sandbox"):
            return True
    return False


def _parse_version(text: str) -> tuple[int, int, int] | None:
    """Extract ``(major, minor, patch)`` from a ``codex-cli X.Y.Z`` banner."""
    match = _VERSION_PATTERN.search(text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


class CodexAgentAdapter(BaseAgentAdapter):
    """Agent for the OpenAI ``codex-cli`` 0.130 (or newer) binary."""

    CREDENTIAL_ENV = ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")
    CREDENTIAL_PATHS = (Path.home() / ".codex" / "auth.json",)

    @classmethod
    def supported_output_fields(cls) -> frozenset[str]:
        return frozenset({"tool_sequence", "llm_turn_count"})

    @classmethod
    def denied_flags(cls) -> frozenset[str]:
        # ``--dangerously-bypass-approvals-and-sandbox`` is the wholesale
        # escape hatch (disables both approval prompts and the sandbox).
        # ``--sandbox=danger-full-access`` reaches the same outcome on the
        # sandbox layer alone; we deny that single value (both
        # ``--sandbox=danger-full-access`` and the two-token form
        # ``--sandbox danger-full-access``) while letting safer values
        # (``read-only``, ``workspace-write``) pass through. The
        # value-specific entries are matched by ``_check_denied_flags``.
        return frozenset(
            {
                "--dangerously-bypass-approvals-and-sandbox",
                "--sandbox=danger-full-access",
                "-s=danger-full-access",
            }
        )

    @classmethod
    def check_available(cls) -> None:
        bin_path = resolve_binary(("codex",))
        if not bin_path:
            raise AgentNotAvailableError("codex", "codex CLI not found on PATH", _INSTALL_HINT)

        try:
            result = subprocess.run(
                [bin_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise AgentNotAvailableError("codex", f"codex --version failed: {exc}", _INSTALL_HINT) from exc

        version = _parse_version(result.stdout) or _parse_version(result.stderr)
        if version is None:
            banner = (result.stdout + result.stderr).strip().replace("\n", " ")[:200]
            raise AgentNotAvailableError(
                "codex",
                f"codex --version returned unparseable output: {banner!r}",
                "Expected 'codex-cli X.Y.Z'. " + _INSTALL_HINT,
            )

        if version < _MIN_VERSION:
            installed = ".".join(str(p) for p in version)
            required = ".".join(str(p) for p in _MIN_VERSION)
            raise AgentNotAvailableError(
                "codex",
                f"codex-cli {installed} is older than the required {required}",
                _UPGRADE_HINT,
            )

    @classmethod
    def cli_options(cls) -> list[AgentOption]:
        return [
            AgentOption(
                name="model",
                help="Model override (passed as ``-m`` to ``codex exec`` and replayed on resume)",
                env_var="CODEX_DEFAULT_MODEL",
            ),
            AgentOption(
                name="profile",
                help=(
                    "config.toml profile to use (passed as ``-p`` on the first turn). "
                    "Resume turns translate the profile to ``-c model_provider=...`` because "
                    "``codex exec resume`` does not accept ``-p``."
                ),
                env_var="CODEX_DEFAULT_PROFILE",
            ),
        ]

    @classmethod
    def required_env_vars(cls) -> frozenset[str]:
        # Codex resolves provider auth through entries in
        # ``~/.codex/config.toml``. Azure deployments configured via the
        # Microsoft Foundry recipe name ``AZURE_OPENAI_API_KEY`` in
        # ``[model_providers.azure].env_key``; without this override the
        # default scrubbed env strips it and codex falls back to auth
        # that is not configured for the deployment.
        names = set(super().required_env_vars())
        names.update(
            {
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_BASE_URL",
                "OPENAI_BASE_URL",
                "CODEX_HOME",
            }
        )
        return frozenset(names)

    @classmethod
    def display_info(cls) -> str:
        bin_path = resolve_binary(("codex",))
        if not bin_path:
            return "CodexAgentAdapter (codex CLI not found)"
        try:
            result = subprocess.run(
                [bin_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            banner = (result.stdout or result.stderr).strip().split("\n")[0]
            return f"CodexAgentAdapter ({banner or 'codex'})"
        except Exception:
            return "CodexAgentAdapter (codex)"

    @classmethod
    def runtime_info(cls) -> dict[str, Any]:
        info = super().runtime_info()
        bin_path = resolve_binary(("codex",))
        if bin_path:
            info["cli_binary_path"] = bin_path
            info["cli_version"] = cls._capture_cli_version([bin_path, "--version"])
        return info

    def __init__(self, *, model: str | None = None, profile: str | None = None) -> None:
        self._session_id: str | None = None
        self._model: str | None = model
        self._profile: str | None = profile
        self._workspace_dir: str | None = None
        self._ttfe: float | None = None
        self._ttft: float | None = None
        self._ttlt: float | None = None
        self._last_message: str = ""

    # ── Group lifecycle (no-op) ──

    def setup_group(self, group_config: GroupConfig, group_dir: Path) -> Any:
        return None

    # ── Scenario lifecycle ──

    def setup(self, config: AgentConfig) -> None:
        self._session_id = None
        self._workspace_dir = config.workspace_dir
        self._ttfe = None
        self._ttft = None
        self._ttlt = None
        self._last_message = ""

    def execute(self, message: str, flags: list[str]) -> str:
        with tempfile.NamedTemporaryFile(prefix="codex-last-", suffix=".txt", delete=False) as handle:
            reply_file = handle.name
        try:
            cmd = self._build_command(message, flags, reply_file)
            return self._execute_streaming(cmd, reply_file)
        finally:
            try:
                Path(reply_file).unlink(missing_ok=True)
            except OSError:
                pass

    def _build_command(self, message: str, flags: list[str], reply_file: str) -> list[str]:
        filtered = self.filter_flags(flags)
        if self._session_id is None:
            cmd = ["codex", "exec", "--json", "--skip-git-repo-check", "-o", reply_file]
            if self._model:
                cmd.extend(["-m", self._model])
            if self._profile:
                cmd.extend(["-p", self._profile])
            if self._workspace_dir:
                cmd.extend(["-C", self._workspace_dir])
            # codex's default sandbox is ``read-only``, which silently
            # rejects every file write the model attempts and surfaces as
            # a generic refusal in ``reply_text``. Belt already isolates
            # each scenario in its own per-scenario git worktree, so the
            # next-tier sandbox should match that scope rather than
            # block it. Default to ``workspace-write`` and let scenarios
            # opt into the stricter ``read-only`` (or, via
            # ``--allow-unsafe-flags`` plus the explicit value, the
            # forbidden ``danger-full-access``) by passing ``--sandbox``
            # in the scenario's ``flags``.
            if not _has_sandbox_flag(filtered):
                cmd.extend(["-s", "workspace-write"])
        else:
            # ``codex exec resume`` rejects ``-p`` and ``--skip-git-repo-check``;
            # replicate the profile via ``-m`` plus a ``-c model_provider=...``
            # override, and mark the workspace as trusted with a ``-c`` so codex
            # does not abort with "Not inside a trusted directory".
            cmd = ["codex", "exec", "resume", self._session_id, "--json", "-o", reply_file]
            if self._model:
                cmd.extend(["-m", self._model])
            if self._profile:
                cmd.extend(["-c", f'model_provider="{self._profile}"'])
            if self._workspace_dir:
                cmd.extend(["-c", f'projects.{json.dumps(self._workspace_dir)}.trust_level="trusted"'])
        cmd.extend(filtered)
        # ``--`` ends option parsing for the codex CLI so a message that
        # begins with ``-`` cannot be mistaken for a flag.
        cmd.append("--")
        cmd.append(message)
        return cmd

    def _execute_streaming(self, cmd: list[str], reply_file: str) -> str:
        start = time.monotonic()
        self._ttfe = None
        self._ttft = None
        self._ttlt = None

        env = self.make_subprocess_env()
        cwd = self._workspace_dir if self._workspace_dir else None
        logger.debug("Running: {} (cwd={})", " ".join(cmd[:6]) + " ...", cwd)
        # ``codex exec`` reads additional context from stdin until EOF: when
        # stdin is inherited from a TTY-backed parent, the process completes
        # the turn and then sits idle waiting for the EOF that never comes.
        # Closing stdin via ``DEVNULL`` lets codex exit promptly after
        # ``turn.completed``.
        proc = self._spawner.popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )
        stderr_thread = _drain_stderr(proc)

        lines: list[str] = []
        try:
            if proc.stdout is None:
                from belt.errors import AgentExecutionError

                raise AgentExecutionError("codex Popen stdout is None")
            for line in iter_bounded_stream(proc.stdout):
                now = time.monotonic()
                lines.append(line)

                if self._stream_sink is not None:
                    self._stream_sink.write(line)
                    self._stream_sink.flush()

                if self._ttfe is None:
                    self._ttfe = now - start

                stripped = line.strip()
                if not stripped:
                    continue
                if self._ttft is None:
                    try:
                        event = json.loads(stripped)
                    except (json.JSONDecodeError, ValueError):
                        event = None
                    if isinstance(event, dict) and event.get("type") == "item.completed":
                        item = event.get("item")
                        if isinstance(item, dict) and item.get("type") == "agent_message":
                            self._ttft = now - start
                self._ttlt = now - start

            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            raise
        except Exception:
            logger.exception("Error reading codex stdout")

        stderr_thread.join(timeout=5)
        raw_output = "".join(lines)
        stderr = "".join(stderr_thread.lines)  # type: ignore[attr-defined]

        if proc.returncode != 0:
            logger.warning("codex returned rc={}: {}", proc.returncode, _sanitize_stderr(stderr))
            raw_output = raw_output + "\n" + stderr

        try:
            self._last_message = Path(reply_file).read_text(encoding="utf-8")
        except OSError:
            self._last_message = ""

        return raw_output

    def fetch_results(self, raw_output: str) -> TurnOutput:
        events = parse_ndjson(raw_output)

        agent_message_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        tool_sequence: list[str] = []
        session_id: str | None = None
        usage: dict[str, Any] | None = None
        terminal_failure: dict[str, Any] | None = None
        llm_turn_count = 0

        for event in events:
            etype = event.get("type", "")

            if etype == "thread.started":
                tid = event.get("thread_id")
                if isinstance(tid, str) and tid:
                    session_id = tid

            elif etype == "item.completed":
                item = event.get("item")
                if not isinstance(item, dict):
                    continue
                itype = item.get("type", "")
                if itype == "agent_message":
                    text = item.get("text") or ""
                    if text:
                        agent_message_parts.append(text)
                        llm_turn_count += 1
                elif itype == "command_execution":
                    args: dict[str, Any] = {
                        "command": item.get("command", ""),
                        "exit_code": item.get("exit_code"),
                        "status": item.get("status"),
                    }
                    aggregated = item.get("aggregated_output")
                    if aggregated is not None:
                        args["aggregated_output"] = aggregated
                    tc = ToolCall(name="shell", call_id=str(item.get("id", "")), args=args)
                    tool_calls.append(tc)
                    tool_sequence.append(tc.name)

            elif etype == "turn.completed":
                u = event.get("usage")
                if isinstance(u, dict):
                    usage = u

            elif etype == "turn.failed":
                err = event.get("error")
                if isinstance(err, dict):
                    terminal_failure = err
                else:
                    terminal_failure = {"message": str(err)}

        if session_id and self._session_id is None:
            self._session_id = session_id

        # Canonical reply: the file written by ``--output-last-message``.
        # Fall back to concatenated ``agent_message`` items when the file
        # is empty (process killed before terminal event).
        reply_text = self._last_message.rstrip("\n")
        if not reply_text:
            reply_text = "\n".join(part for part in agent_message_parts if part)

        timing: TurnTiming | None = None
        if self._ttfe is not None or self._ttlt is not None:
            timing = TurnTiming(
                ttfe=self._ttfe,
                ttft=self._ttft,
                ttlt=self._ttlt,
                total=self._ttlt,
            )

        if terminal_failure is not None:
            has_error: bool | None = True
            error_type: str | None = (
                normalize_error_type(None, str(terminal_failure.get("message", "")), raw_output) or UNKNOWN
            )
        elif reply_text.strip():
            has_error = False
            error_type = None
        else:
            classified = normalize_error_type(None, raw_output)
            has_error = classified is not None
            error_type = classified

        return TurnOutput(
            raw_cli=raw_output,
            reply_text=reply_text,
            tool_calls=tool_calls,
            tool_sequence=tool_sequence,
            has_reply=bool(reply_text.strip()),
            has_error=has_error,
            error_type=error_type,
            timing=timing,
            cost_usd=None,
            llm_turn_count=llm_turn_count if llm_turn_count > 0 else None,
            usage=usage,
        )

    def teardown(self) -> None:
        self._session_id = None
        self._last_message = ""

    def metadata(self) -> dict[str, Any] | None:
        meta: dict[str, Any] = {}
        if self._session_id:
            meta["session_id"] = self._session_id
        return meta or None
