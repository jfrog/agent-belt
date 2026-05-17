# (c) JFrog Ltd. (2026)

"""CursorAgentAdapter - drives the Cursor Agent CLI through evaluation scenarios.

Cursor Agent headless mode: ``agent -p --output-format stream-json <prompt>``
(or equivalently ``cursor-agent -p ...``) outputs newline-delimited JSON events.
Multi-turn via ``--resume <chatId>``.

Binary discovery
----------------
Two install layouts are supported in the wild:

  - **Standalone CLI** (``curl https://cursor.com/install | bash``):
    extracts the binary into ``~/.local/share/cursor-agent/versions/<v>/``
    and creates two symlinks in ``~/.local/bin/``:
    ``agent`` (the primary command name the installer recommends) and
    ``cursor-agent`` (the legacy alias kept for backwards compatibility).
    Invoked as ``agent <args>`` or ``cursor-agent <args>`` -- both resolve
    to the same binary.
  - **Cursor IDE bundle**: provides a ``cursor`` binary with an ``agent``
    subcommand. Invoked as ``cursor agent <args>``.

The agent picks whichever is present, preferring the more specific names
(``cursor-agent``, ``cursor``) over the bare ``agent`` symlink. ``agent``
is checked last because it is a generic name that could collide with
unrelated tools on the user's PATH; the specific names are unambiguous.

Authentication
--------------
Per the ``check_available()`` contract, this agent does not probe
authentication state (see ``BaseAgentAdapter`` docstring). Cursor accepts both
``CURSOR_API_KEY`` (CI) and ``agent login`` (browser flow); detecting which
one the user has set up requires either invoking the agent (forbidden - costs
credits) or parsing version-specific status strings (forbidden - fragile).
Auth failures surface at eval time with a clear error from the agent itself.

Reference:
  - https://cursor.com/docs/cli/reference/parameters
  - https://cursor.com/docs/cli/reference/authentication
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from loguru import logger

from belt.agent.base import (
    AgentNotAvailableError,
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

_BINARY_CANDIDATES: tuple[str, ...] = ("cursor-agent", "cursor", "agent")
_EXTRA_PATHS: tuple[Path, ...] = (Path.home() / ".local" / "bin",)


class CursorAgentAdapter(BaseAgentAdapter):
    """Agent for the Cursor Agent CLI.

    Injects ``--force`` and ``--approve-mcps`` by default to prevent
    interactive hangs in headless mode.  Unlike Claude Code (which
    auto-approves in ``-p`` mode), Cursor prompts for tool approval
    even when headless, causing evaluation runs to hang indefinitely.
    """

    CREDENTIAL_ENV = ("CURSOR_API_KEY",)
    CREDENTIAL_PATHS = (Path.home() / ".cursor",)

    @classmethod
    def supported_output_fields(cls) -> frozenset[str]:
        return frozenset({"tool_sequence", "thinking_text"})

    @classmethod
    def denied_flags(cls) -> frozenset[str]:
        # ``--yolo`` is documented in cursor-agent --help as "Alias for
        # --force (Run Everything)". The adapter injects --force itself
        # for headless reasons (one named tool-confirmation override),
        # so denying --yolo from scenarios costs no functionality but
        # blocks the broader "skip every safeguard" intent that the
        # --yolo name signals to scenario authors. Same pattern as the
        # gemini deny-list: name-as-signal, even when the alias is
        # mechanically equivalent today.
        return frozenset({"--yolo"})

    @classmethod
    def _resolve_cli(cls) -> str | None:
        """Return absolute path of the Cursor agent binary, or None if not found."""
        return resolve_binary(_BINARY_CANDIDATES, _EXTRA_PATHS)

    @staticmethod
    def _build_cmd(bin_path: str, *args: str) -> list[str]:
        """Build the agent invocation for either the standalone CLI or the IDE bundle.

        Standalone CLI (``cursor-agent`` / ``agent``): ``<bin> <args>``
        IDE bundle      (``cursor``)                 : ``cursor agent <args>``

        The standalone CLI ships two symlinks to the same binary
        (``~/.local/bin/agent`` -- primary, ``~/.local/bin/cursor-agent`` --
        legacy alias). Both are invoked positionally; only the IDE bundle's
        ``cursor`` binary needs the explicit ``agent`` subcommand.
        """
        name = Path(bin_path).name
        if name == "cursor":
            return [bin_path, "agent", *args]
        return [bin_path, *args]

    @classmethod
    def check_available(cls) -> None:
        if not cls._resolve_cli():
            raise AgentNotAvailableError(
                "cursor",
                "cursor CLI not found on PATH",
                "Install: curl -fsSL https://cursor.com/install | bash && " 'export PATH="$HOME/.local/bin:$PATH"',
            )

    @classmethod
    def display_info(cls) -> str:
        bin_path = cls._resolve_cli()
        if not bin_path:
            return "CursorAgentAdapter (cursor CLI not found)"
        try:
            result = subprocess.run(
                cls._build_cmd(bin_path, "about"),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().splitlines()
                return " | ".join(line.strip() for line in lines[:3] if line.strip())
        except Exception:
            pass
        return f"CursorAgentAdapter ({Path(bin_path).name})"

    @classmethod
    def runtime_info(cls) -> dict[str, Any]:
        info = super().runtime_info()
        bin_path = cls._resolve_cli()
        if bin_path:
            info["cli_binary_path"] = bin_path
            info["cli_version"] = cls._capture_cli_version(cls._build_cmd(bin_path, "--version"))
        return info

    def __init__(self) -> None:
        self._chat_id: str | None = None
        self._last_duration: float | None = None
        self._workspace_dir: str | None = None
        self._cli_path: str | None = None

    # ── Group lifecycle (no-op) ──

    def setup_group(self, group_config: GroupConfig, group_dir: Path) -> Any:
        return None

    # ── Scenario lifecycle ──

    def setup(self, config: AgentConfig) -> None:
        self._chat_id = None
        self._workspace_dir = config.workspace_dir
        self._cli_path = self._resolve_cli()

    def execute(self, message: str, flags: list[str]) -> str:
        # Resolve once and cache. If unresolved (e.g. test that mocks subprocess.Popen
        # without installing cursor), fall back to the canonical name and let Popen
        # surface FileNotFoundError naturally - matches behavior of other agents
        # and keeps check_available() as the single source of truth for "is it installed".
        bin_path = self._cli_path or self._resolve_cli() or _BINARY_CANDIDATES[0]
        self._cli_path = bin_path

        args: list[str] = ["-p", "--output-format", "stream-json", "--force", "--approve-mcps"]
        if self._chat_id:
            args.extend(["--resume", self._chat_id])
        args.extend(self.filter_flags(flags))
        # ``--`` ends option parsing so a message starting with ``-`` is
        # passed through as the positional prompt instead of being parsed
        # as a flag by the Cursor agent CLI.
        args.append("--")
        args.append(message)

        cmd = self._build_cmd(bin_path, *args)
        logger.debug("Running: {}", " ".join(cmd[:8]) + "...")
        return self._execute_streaming(cmd)

    def _execute_streaming(self, cmd: list[str]) -> str:
        start = time.monotonic()

        cwd = self._workspace_dir if self._workspace_dir else None
        proc = self._spawner.popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=self.make_subprocess_env(),
            start_new_session=True,
        )
        stderr_thread = _drain_stderr(proc)

        lines: list[str] = []
        try:
            if proc.stdout is None:
                from belt.errors import AgentExecutionError

                raise AgentExecutionError("cursor Popen stdout is None")
            for line in iter_bounded_stream(proc.stdout):
                lines.append(line)

                if self._stream_sink is not None:
                    self._stream_sink.write(line)
                    self._stream_sink.flush()

            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            logger.error("cursor agent timed out after 300s")
            raise

        stderr_thread.join(timeout=5)
        self._last_duration = time.monotonic() - start

        raw_output = "".join(lines)
        stderr = "".join(stderr_thread.lines)  # type: ignore[attr-defined]

        if proc.returncode != 0:
            logger.warning("cursor agent returned rc={}: {}", proc.returncode, _sanitize_stderr(stderr))
            raw_output = raw_output + "\n" + stderr
        return raw_output

    def fetch_results(self, raw_output: str) -> TurnOutput:
        events = parse_ndjson(raw_output)

        reply_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        chat_id: str | None = None
        is_error: bool | None = None
        duration_ms: float | None = None

        for event in events:
            etype = event.get("type", "")

            if etype == "thinking":
                # Cursor reasoning arrives as
                # ``{"type":"thinking","subtype":"delta","text":...}``
                # deltas; concatenate every event carrying a ``text``
                # field so subtype shape changes do not silently drop
                # content.
                text = event.get("text", "")
                if isinstance(text, str) and text:
                    thinking_parts.append(text)
                continue

            if etype in ("message", "assistant", "text"):
                content = event.get("content", "")
                # Cursor wraps content as event.message.content (nested)
                if not content and isinstance(event.get("message"), dict):
                    content = event["message"].get("content", "")
                if isinstance(content, str) and content.strip():
                    reply_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            reply_parts.append(block.get("text", ""))
                        elif isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_calls.append(
                                ToolCall(
                                    name=block.get("name", ""),
                                    call_id=block.get("id", ""),
                                    args=block.get("input", {}),
                                )
                            )
                text_field = event.get("text", "")
                if isinstance(text_field, str) and text_field.strip():
                    reply_parts.append(text_field)

            elif etype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        name=event.get("name", ""),
                        call_id=event.get("id", ""),
                        args=event.get("input", {}),
                    )
                )

            elif etype == "tool_call":
                _apply_tool_call_event(event, tool_calls)

            elif etype == "result":
                chat_id = event.get("chat_id") or event.get("conversation_id") or event.get("session_id")
                is_error = event.get("is_error", False)
                duration_ms = event.get("duration_ms")
                result_text = event.get("result", "")
                if isinstance(result_text, str) and result_text.strip() and not reply_parts:
                    reply_parts.append(result_text)

        if chat_id:
            self._chat_id = chat_id

        reply_text = "\n".join(p for p in reply_parts if p.strip())
        thinking_text = "".join(thinking_parts) if thinking_parts else None
        has_error = is_error if is_error is not None else ("error" in raw_output.lower() and not reply_text)

        # Cursor's stream-json events don't carry a structured error_type
        # field; classify from text so the framework still produces a
        # stable label for downstream surfaces.
        error_type: str | None = None
        if has_error:
            error_type = normalize_error_type(None, reply_text, raw_output) or UNKNOWN

        timing = None
        if duration_ms is not None:
            timing = TurnTiming(total=duration_ms / 1000.0)
        elif self._last_duration is not None:
            timing = TurnTiming(total=self._last_duration)

        return TurnOutput(
            raw_cli=raw_output,
            reply_text=reply_text,
            thinking_text=thinking_text,
            tool_calls=tool_calls,
            tool_sequence=[tc.name for tc in tool_calls],
            has_reply=bool(reply_text.strip()),
            has_error=has_error,
            error_type=error_type,
            timing=timing,
        )

    def teardown(self) -> None:
        self._chat_id = None

    def metadata(self) -> dict[str, Any] | None:
        if self._chat_id:
            return {"chat_id": self._chat_id}
        return None

    @staticmethod
    def parse_stream_event(event: dict) -> tuple[str, str] | None:
        etype = event.get("type", "")
        if etype != "tool_call":
            return None
        tc_data = event.get("tool_call", {})
        if not isinstance(tc_data, dict) or not tc_data:
            return None
        if event.get("subtype") == "completed":
            return _render_cursor_tool_result(tc_data)
        name, args = _extract_cursor_tool(tc_data)
        if not name:
            return None
        args_str = ", ".join(f"{k}={v}" for k, v in args.items()) if isinstance(args, dict) else str(args)
        if len(args_str) > 80:
            args_str = args_str[:77] + "…"
        return "🔧", f"{name}({args_str})"


_CURSOR_TOOL_TYPE_MAP: dict[str, str] = {
    "readToolCall": "Read",
    "shellToolCall": "Shell",
    "grepToolCall": "Grep",
    "globToolCall": "Glob",
    "editToolCall": "Edit",
    "listToolCall": "List",
    "searchToolCall": "Search",
}


def _extract_cursor_tool(tc_data: dict) -> tuple[str, dict]:
    """Extract (tool_name, args) from a Cursor stream-json tool_call payload.

    Cursor nests the real tool info under a subtype key:
      mcpToolCall  → args.toolName has the MCP tool name
      readToolCall → implicit "Read", shellToolCall → "Shell", etc.
    """
    for key, implicit_name in _CURSOR_TOOL_TYPE_MAP.items():
        if key in tc_data:
            inner = tc_data[key]
            return implicit_name, inner.get("args", {})

    if "mcpToolCall" in tc_data:
        inner = tc_data["mcpToolCall"]
        inner_args = inner.get("args", {})
        name = inner_args.get("toolName", inner_args.get("name", "mcp_tool"))
        return name, inner_args.get("args", {})

    return "", {}


def _extract_cursor_tool_result(tc_data: dict) -> dict[str, Any] | None:
    """Extract the result payload from a completed tool_call event.

    Returns the raw result dict if found, or None.
    """
    for inner in tc_data.values():
        if not isinstance(inner, dict):
            continue
        result = inner.get("result")
        if isinstance(result, dict):
            return result
    return None


def _render_cursor_tool_result(tc_data: dict) -> tuple[str, str]:
    """Render a Cursor completed tool_call as a brief result line."""
    name, _ = _extract_cursor_tool(tc_data)
    if not name:
        name = "tool"

    for _key, inner in tc_data.items():
        if not isinstance(inner, dict):
            continue
        result = inner.get("result", {})
        if not isinstance(result, dict):
            continue
        if "error" in result:
            err = result["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            if len(msg) > 60:
                msg = msg[:57] + "…"
            return "❌", f"{name} → {msg}"
        if "success" in result:
            return "✅", f"{name} → ok"

    return "✅", f"{name} → done"


def _apply_tool_call_event(event: dict, tool_calls: list[ToolCall]) -> None:
    """Apply a Cursor ``tool_call`` stream event to the running tool_calls list.

    ``subtype == "completed"`` events attach a result to the matching prior call;
    other events (started/in-progress) append a new ToolCall.
    """
    call_id = event.get("call_id", "")
    tc_data = event.get("tool_call", {})
    if event.get("subtype") == "completed":
        result = _extract_cursor_tool_result(tc_data)
        if result is not None:
            for tc in tool_calls:
                if tc.call_id == call_id:
                    tc.result = result
                    break
        return
    name, args = _extract_cursor_tool(tc_data)
    tool_calls.append(ToolCall(name=name, call_id=call_id, args=args))
