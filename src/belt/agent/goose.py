# (c) JFrog Ltd. (2026)

"""GooseAgentAdapter - drives the Goose CLI (Block) through evaluation scenarios.

Goose (`goose run --output-format stream-json`) outputs NDJSON with these event types:
- message: contains Message with role + content array (text, toolRequest, toolResponse, thinking)
- notification: MCP extension log/progress events
- model_change: provider/model switch events
- error: error string
- complete: final event with total_tokens count

Multi-turn sessions use `--resume --name <name> -t <message>`.

Reference: https://block.github.io/goose/docs/guides/goose-cli-commands/
Source: https://github.com/block/goose/blob/main/crates/goose-cli/src/session/mod.rs
"""

from __future__ import annotations

import subprocess
import time
import uuid
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


class GooseAgentAdapter(BaseAgentAdapter):
    """Agent for the Goose CLI (Block).

    Streams NDJSON output to capture per-turn timing (ttfe, ttft, ttlt)
    alongside structured tool call parsing and multi-turn session support.

    Authentication
    --------------
    Per the ``check_available()`` contract, this agent does not probe
    authentication state (see ``BaseAgentAdapter`` docstring). Goose is multi-provider
    (configured per-install via ``goose configure``) so we don't pin a single
    env var; the canonical positive signal is the config file goose writes:
    ``~/.config/goose/config.yaml`` (see
    https://block.github.io/goose/docs/guides/config-files).
    """

    CREDENTIAL_PATHS = (Path.home() / ".config" / "goose" / "config.yaml",)

    @classmethod
    def supported_output_fields(cls) -> frozenset[str]:
        return frozenset({"tool_sequence", "llm_turn_count"})

    @classmethod
    def denied_flags(cls) -> frozenset[str]:
        # ``--with-extension`` and ``--with-streamable-http-extension`` let
        # a flag value attach an arbitrary subprocess command (with env
        # vars) or remote HTTP MCP endpoint to the running agent. Either
        # is a capability-broadening surface that bypasses everything
        # the scenario author can audit. Same threat model as Codex's
        # ``--dangerously-bypass-approvals-and-sandbox`` and Copilot's
        # ``--remote``: block at the framework layer; require an explicit
        # ``_allow_unsafe_flags`` opt-in from scenarios that need it.
        return frozenset({"--with-extension", "--with-streamable-http-extension"})

    @classmethod
    def check_available(cls) -> None:
        if not resolve_binary(("goose",)):
            raise AgentNotAvailableError(
                "goose",
                "goose CLI not found on PATH",
                "Install: brew install block-goose-cli  OR  https://block.github.io/goose/docs/getting-started/installation/",
            )

    @classmethod
    def cli_options(cls) -> list[AgentOption]:
        return [
            AgentOption(
                name="model",
                help="Model override (passed as --model flag)",
                env_var="GOOSE_MODEL",
            ),
            AgentOption(
                name="provider",
                help="Provider override (passed as --provider flag)",
                env_var="GOOSE_PROVIDER",
            ),
        ]

    @classmethod
    def display_info(cls) -> str:
        bin_path = resolve_binary(("goose",))
        if not bin_path:
            return "GooseAgentAdapter (goose CLI not found)"
        try:
            # ``goose --version`` is the supported version probe; the
            # ``goose version`` subcommand is rejected with
            # ``unrecognized subcommand 'version'`` (exit 2). A drift to
            # the subcommand form would surface as ``cli_version: null``
            # in the benchmark card.
            result = subprocess.run([bin_path, "--version"], capture_output=True, text=True, timeout=10)
            version = result.stdout.strip().split("\n")[0] if result.returncode == 0 else "unknown"
        except Exception:
            version = "unknown"
        return f"GooseAgentAdapter (goose {version})"

    @classmethod
    def runtime_info(cls) -> dict[str, Any]:
        info = super().runtime_info()
        bin_path = resolve_binary(("goose",))
        if bin_path:
            info["cli_binary_path"] = bin_path
            info["cli_version"] = cls._capture_cli_version([bin_path, "--version"])
        return info

    def __init__(self, *, model: str | None = None, provider: str | None = None) -> None:
        self._session_id: str | None = None
        self._model: str | None = model
        self._provider: str | None = provider
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
        cmd = ["goose", "run", "--output-format", "stream-json"]
        if self._model:
            cmd.extend(["--model", self._model])
        if self._provider:
            cmd.extend(["--provider", self._provider])
        if self._session_id:
            cmd.extend(["--resume", "--name", self._session_id])
        else:
            session_name = f"belt-{uuid.uuid4().hex[:12]}"
            cmd.extend(["--name", session_name])
            self._session_id = session_name
        cmd.extend(self.filter_flags(flags))
        cmd.extend(["-t", message])

        return self._execute_streaming(cmd)

    def _execute_streaming(self, cmd: list[str]) -> str:
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

                raise AgentExecutionError("goose Popen stdout is None")
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
                    if '"type":"message"' in stripped.replace(" ", "") or '"type": "message"' in stripped:
                        self._ttft = t - start

                if stripped:
                    self._ttlt = t - start

            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            raise
        except Exception:
            logger.exception("Error reading goose stdout")

        stderr_thread.join(timeout=5)
        raw_output = "".join(lines)
        stderr = "".join(stderr_thread.lines)  # type: ignore[attr-defined]

        if proc.returncode != 0:
            logger.warning("goose returned rc={}: {}", proc.returncode, _sanitize_stderr(stderr))
            raw_output = raw_output + "\n" + stderr

        return raw_output

    def fetch_results(self, raw_output: str) -> TurnOutput:
        events = parse_ndjson(raw_output)

        reply_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        tool_sequence: list[str] = []
        is_error: bool | None = None
        error_type: str | None = None
        assistant_message_count = 0

        for event in events:
            etype = event.get("type", "")

            if etype == "message":
                msg = event.get("message", {})
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                if role == "assistant":
                    assistant_message_count += 1
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue

                for item in content:
                    if not isinstance(item, dict):
                        continue
                    ctype = item.get("type", "")

                    if ctype == "text":
                        text = item.get("text", "")
                        if text and role == "assistant":
                            reply_parts.append(text)

                    elif ctype == "toolRequest":
                        tool_call_data = item.get("toolCall", {})
                        if not isinstance(tool_call_data, dict):
                            tool_call_data = {}
                        # Goose nests the tool name and arguments under
                        # ``toolCall.value`` (e.g. {"status": "success",
                        # "value": {"name": "analyze", "arguments": {...}}}).
                        # Older flat shapes are tolerated as a fallback.
                        value = tool_call_data.get("value")
                        if not isinstance(value, dict):
                            value = {}
                        name = value.get("name") or tool_call_data.get("name", "")
                        args = value.get("arguments")
                        if not isinstance(args, dict):
                            args = tool_call_data.get("arguments", {})
                            if not isinstance(args, dict):
                                args = {}
                        tc = ToolCall(
                            name=name,
                            call_id=item.get("id", ""),
                            args=args,
                        )
                        tool_calls.append(tc)
                        tool_sequence.append(tc.name)

                    elif ctype == "toolResponse":
                        call_id = item.get("id", "")
                        for tc in tool_calls:
                            if tc.call_id == call_id:
                                tool_result = item.get("toolResult", {})
                                tc.result = tool_result if isinstance(tool_result, dict) else {"raw": tool_result}
                                break

                    elif ctype == "thinking":
                        thinking = item.get("thinking", "")
                        if thinking:
                            thinking_parts.append(thinking)

            elif etype == "error":
                is_error = True
                error_type = event.get("error", "unknown")
                if not isinstance(error_type, str):
                    error_type = str(error_type)[:200]

        # Streamed text blocks are concatenated verbatim. Adjacent ``text`` items
        # are token deltas (e.g. ["Cla", "ude", "Code"]) when goose drives a
        # streaming backend such as Ollama; injecting separators here would
        # fracture words. Any genuine line breaks were emitted by the model
        # inside the ``text`` payload and are preserved by ``str.join("")``.
        reply_text = "".join(reply_parts)
        thinking_text = "".join(thinking_parts) if thinking_parts else None

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
            # "we don't know" state when nothing matches - goose events
            # occasionally lack an explicit result marker even on
            # success.
            classified = classify_error(raw_output)
            has_error = True if classified else None
            if classified and not error_type:
                error_type = classified

        if has_error:
            error_type = normalize_error_type(error_type, reply_text, raw_output) or UNKNOWN

        return TurnOutput(
            raw_cli=raw_output,
            reply_text=reply_text,
            thinking_text=thinking_text,
            tool_calls=tool_calls,
            tool_sequence=tool_sequence,
            has_reply=bool(reply_text.strip()),
            has_error=has_error,
            error_type=error_type,
            timing=timing,
            llm_turn_count=assistant_message_count if assistant_message_count > 0 else None,
        )

    def teardown(self) -> None:
        self._session_id = None

    @staticmethod
    def parse_stream_event(event: dict) -> tuple[str, str] | None:
        etype = event.get("type", "")

        if etype == "message":
            msg = event.get("message", {})
            if not isinstance(msg, dict):
                return None
            role = msg.get("role", "")
            content = msg.get("content", [])
            if not isinstance(content, list):
                return None

            for item in content:
                if not isinstance(item, dict):
                    continue
                ctype = item.get("type", "")

                if ctype == "toolRequest":
                    tool_call_data = item.get("toolCall", {})
                    if isinstance(tool_call_data, dict):
                        # Goose nests name + arguments under ``toolCall.value``;
                        # tolerate older flat shapes as a fallback.
                        value = tool_call_data.get("value")
                        if not isinstance(value, dict):
                            value = {}
                        name = value.get("name") or tool_call_data.get("name", "?")
                        args = value.get("arguments")
                        if not isinstance(args, dict):
                            args = tool_call_data.get("arguments", {})
                        if isinstance(args, dict):
                            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
                            if len(args_str) > 80:
                                args_str = args_str[:77] + "…"
                            return "🔧", f"{name}({args_str})"
                    return "🔧", "tool()"

                if ctype == "toolResponse":
                    return BaseAgentAdapter.SUPPRESS_EVENT

                if ctype == "text" and role == "assistant":
                    text = item.get("text", "")
                    if text:
                        display = text.replace("\n", " ")
                        if len(display) > 120:
                            display = display[:117] + "…"
                        return "💬", display

                if ctype == "thinking":
                    return "🧠", "thinking..."

            if role == "user":
                return BaseAgentAdapter.SUPPRESS_EVENT

            return None

        if etype == "notification":
            log_data = event.get("log", {})
            if isinstance(log_data, dict):
                msg = log_data.get("message", "")
                if msg:
                    if len(msg) > 100:
                        msg = msg[:97] + "…"
                    return "📋", msg
            return BaseAgentAdapter.SUPPRESS_EVENT

        if etype == "model_change":
            model = event.get("model", "?")
            return "🔄", f"model → {model}"

        if etype == "error":
            err = event.get("error", "error")
            return "❌", str(err)[:80] if err else "error"

        if etype == "complete":
            tokens = event.get("total_tokens")
            if tokens is not None:
                return "✅", f"done ({tokens:,} tokens)"
            return "✅", "done"

        return None

    def metadata(self) -> dict[str, Any] | None:
        meta: dict[str, Any] = {}
        if self._session_id:
            meta["session_id"] = self._session_id
        return meta or None
