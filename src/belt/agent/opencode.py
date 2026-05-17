# (c) JFrog Ltd. (2026)

"""OpenCodeAgentAdapter - drives OpenCode CLI through evaluation scenarios.

OpenCode (`opencode run --format json`) outputs NDJSON with a nested part.*
structure. Event types: step_start, tool_use, text, step_finish, error.

Cost and tokens are distributed across multiple step_finish events per turn
(one per LLM reasoning step) - we sum them. Multi-turn sessions use
`--session <ses_XXX> --continue`.

Reference: https://dev.opencode.ai/docs/cli/
NDJSON schema: https://takopi.dev/reference/runners/opencode/stream-json-cheatsheet/
"""

from __future__ import annotations

import subprocess
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
from belt.agent.error_types import UNKNOWN, classify_error, normalize_error_type
from belt.entities import ToolCall, TurnOutput, TurnTiming
from belt.parser.ndjson import parse_ndjson
from belt.runner.entities import AgentConfig
from belt.scenario import GroupConfig


class OpenCodeAgentAdapter(BaseAgentAdapter):
    """Agent for the OpenCode CLI.

    Streams NDJSON output to capture per-turn timing (ttfe, ttft, ttlt)
    alongside structured tool call parsing and multi-turn session support.

    Authentication
    --------------
    Per the ``check_available()`` contract, this agent does not probe
    authentication state (see ``BaseAgentAdapter`` docstring). OpenCode is multi-provider
    so we don't pin a single env var (each provider's key would be a noisy
    false-positive); the canonical positive signal is the auth file that
    ``opencode auth login`` writes:
    ``~/.local/share/opencode/auth.json``.
    """

    CREDENTIAL_PATHS = (Path.home() / ".local" / "share" / "opencode" / "auth.json",)

    @classmethod
    def supported_output_fields(cls) -> frozenset[str]:
        return frozenset({"tool_sequence", "llm_turn_count"})

    @classmethod
    def denied_flags(cls) -> frozenset[str]:
        # OpenCode ships two equivalent escape hatches for "approve every
        # permission prompt unconditionally": ``--yolo`` (the named alias)
        # and ``--dangerously-skip-permissions`` (the explicit form, same
        # spelling as the Claude Code flag). Both bypass the only safety
        # surface the CLI exposes between the agent and the host. Block
        # injection from scenarios so the operator opts in deliberately
        # via ``_allow_unsafe_flags`` rather than inheriting it from a
        # third-party scenario JSON.
        return frozenset({"--yolo", "--dangerously-skip-permissions"})

    @classmethod
    def check_available(cls) -> None:
        if not resolve_binary(("opencode",)):
            raise AgentNotAvailableError(
                "opencode",
                "opencode CLI not found on PATH",
                "Install: curl -fsSL https://opencode.ai/install | bash",
            )

    @classmethod
    def cli_options(cls) -> list[AgentOption]:
        return [
            AgentOption(
                name="model",
                help="Model override in provider/model format (passed as --model flag)",
                env_var="OPENCODE_DEFAULT_MODEL",
            ),
        ]

    @classmethod
    def display_info(cls) -> str:
        bin_path = resolve_binary(("opencode",))
        if not bin_path:
            return "OpenCodeAgentAdapter (opencode CLI not found)"
        try:
            result = subprocess.run([bin_path, "--version"], capture_output=True, text=True, timeout=10)
            version = result.stdout.strip().split("\n")[0] if result.returncode == 0 else "unknown"
        except Exception:
            version = "unknown"
        return f"OpenCodeAgentAdapter (opencode {version})"

    @classmethod
    def runtime_info(cls) -> dict[str, Any]:
        info = super().runtime_info()
        bin_path = resolve_binary(("opencode",))
        if bin_path:
            info["cli_binary_path"] = bin_path
            info["cli_version"] = cls._capture_cli_version([bin_path, "--version"])
        return info

    def __init__(self, *, model: str | None = None) -> None:
        self._session_id: str | None = None
        self._model: str | None = model
        self._workspace_dir: str | None = None
        self._ttfe: float | None = None
        self._ttft: float | None = None
        self._ttlt: float | None = None

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

    def execute(self, message: str, flags: list[str]) -> str:
        cmd = ["opencode", "run", "--format", "json"]
        if self._model:
            cmd.extend(["--model", self._model])
        if self._session_id:
            cmd.extend(["--session", self._session_id, "--continue"])
        cmd.extend(self.filter_flags(flags))
        # ``--`` ends option parsing so a message starting with ``-`` is
        # passed through as the prompt rather than being parsed as a flag.
        cmd.append("--")
        cmd.append(message)

        return self._execute_streaming(cmd)

    def _execute_streaming(self, cmd: list[str]) -> str:
        """Run opencode CLI with streaming to capture timing metrics."""
        start = time.monotonic()
        self._ttfe = None
        self._ttft = None
        self._ttlt = None

        cwd = self._workspace_dir if self._workspace_dir else None
        logger.debug("Running: {} (cwd={})", " ".join(cmd[:6]) + "...", cwd)
        proc = self._spawner.popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.make_subprocess_env(),
            cwd=cwd,
            start_new_session=True,
        )
        stderr_thread = _drain_stderr(proc)

        lines: list[str] = []
        try:
            if proc.stdout is None:
                from belt.errors import AgentExecutionError

                raise AgentExecutionError("opencode Popen stdout is None")
            for line in iter_bounded_stream(proc.stdout):
                t = time.monotonic()
                lines.append(line)

                if self._stream_sink is not None:
                    self._stream_sink.write(line)
                    self._stream_sink.flush()

                if self._ttfe is None:
                    self._ttfe = t - start

                stripped = line.strip()
                if self._ttft is None and stripped:
                    if '"type"' in stripped and '"text"' in stripped:
                        self._ttft = t - start

                if stripped:
                    self._ttlt = t - start

            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            raise
        except Exception:
            logger.exception("Error reading opencode stdout")

        stderr_thread.join(timeout=5)
        raw_output = "".join(lines)
        stderr = "".join(stderr_thread.lines)  # type: ignore[attr-defined]

        if proc.returncode != 0:
            logger.warning("opencode returned rc={}: {}", proc.returncode, _sanitize_stderr(stderr))
            raw_output = raw_output + "\n" + stderr

        return raw_output

    def fetch_results(self, raw_output: str) -> TurnOutput:
        events = parse_ndjson(raw_output)

        reply_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        tool_sequence: list[str] = []
        session_id: str | None = None
        is_error: bool | None = None
        error_type: str | None = None
        total_cost: float = 0.0
        has_cost = False
        llm_turn_count = 0

        for event in events:
            etype = event.get("type", "")
            part = event.get("part", {})
            if not isinstance(part, dict):
                part = {}

            if etype == "step_start":
                if session_id is None:
                    session_id = event.get("sessionID")

            elif etype == "text":
                text = part.get("text", "")
                if text:
                    reply_parts.append(text)

            elif etype == "tool_use":
                state = part.get("state", {})
                if not isinstance(state, dict):
                    state = {}
                tc = ToolCall(
                    name=part.get("tool", ""),
                    call_id=part.get("callID", ""),
                    args=state.get("input", {}),
                    result={"output": state.get("output", "")} if state.get("output") else None,
                )
                tool_calls.append(tc)
                tool_sequence.append(tc.name)

            elif etype == "step_finish":
                llm_turn_count += 1
                cost = part.get("cost")
                if cost is not None and isinstance(cost, (int, float)):
                    total_cost += cost
                    has_cost = True

            elif etype == "error":
                is_error = True
                err = event.get("error", {})
                if isinstance(err, dict):
                    error_type = err.get("name")
                    if not error_type:
                        data = err.get("data", {})
                        if isinstance(data, dict):
                            error_type = data.get("message")
                elif isinstance(err, str):
                    error_type = err

        if session_id:
            self._session_id = session_id

        reply_text = "\n".join(reply_parts)

        timing = None
        wall_total = self._ttlt if self._ttlt is not None else None
        if wall_total is not None:
            timing = TurnTiming(
                ttfe=self._ttfe,
                ttft=self._ttft,
                ttlt=self._ttlt,
                total=wall_total,
            )

        has_error: bool | None
        if is_error is not None:
            has_error = is_error
        elif reply_text.strip():
            has_error = False
        else:
            # ``None`` (rather than ``False``) preserves the historical
            # "we don't know" state when nothing matches - the opencode
            # adapter declines to assert a clean turn from raw output
            # alone, since opencode events occasionally lack a result
            # marker on success too.
            classified = classify_error(raw_output)
            has_error = True if classified else None
            if classified and not error_type:
                error_type = classified

        if has_error:
            error_type = normalize_error_type(error_type, reply_text, raw_output) or UNKNOWN

        return TurnOutput(
            raw_cli=raw_output,
            reply_text=reply_text,
            tool_calls=tool_calls,
            tool_sequence=tool_sequence,
            has_reply=bool(reply_text.strip()),
            has_error=has_error,
            error_type=error_type,
            timing=timing,
            cost_usd=total_cost if has_cost else None,
            llm_turn_count=llm_turn_count if llm_turn_count > 0 else None,
        )

    def teardown(self) -> None:
        self._session_id = None

    @staticmethod
    def parse_stream_event(event: dict) -> tuple[str, str] | None:
        etype = event.get("type", "")
        part = event.get("part", {})
        if not isinstance(part, dict):
            return None

        if etype == "tool_use":
            name = part.get("tool", "?")
            state = part.get("state", {})
            if isinstance(state, dict):
                args = state.get("input", {})
                if isinstance(args, dict):
                    args_str = ", ".join(f"{k}={v}" for k, v in args.items())
                    if len(args_str) > 80:
                        args_str = args_str[:77] + "…"
                    return "🔧", f"{name}({args_str})"
            return "🔧", f"{name}()"

        if etype == "text":
            text = part.get("text", "")
            if text:
                display = text.replace("\n", " ")
                if len(display) > 120:
                    display = display[:117] + "…"
                return "💬", display
            return None

        if etype == "step_start":
            return BaseAgentAdapter.SUPPRESS_EVENT

        if etype == "step_finish":
            reason = part.get("reason", "")
            if reason == "stop":
                cost = part.get("cost")
                if cost is not None:
                    return "✅", f"done (${cost:.4f})"
                return "✅", "done"
            return BaseAgentAdapter.SUPPRESS_EVENT

        if etype == "error":
            err = event.get("error", {})
            if isinstance(err, dict):
                msg = err.get("name", "")
                data = err.get("data", {})
                if isinstance(data, dict) and data.get("message"):
                    msg = f"{msg}: {data['message']}" if msg else data["message"]
                return "❌", msg or "error"
            return "❌", str(err)[:80] if err else "error"

        return None

    def metadata(self) -> dict[str, Any] | None:
        meta: dict[str, Any] = {}
        if self._session_id:
            meta["session_id"] = self._session_id
        return meta or None
