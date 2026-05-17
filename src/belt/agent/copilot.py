# (c) JFrog Ltd. (2026)

"""CopilotAgentAdapter - drives the GitHub Copilot CLI through evaluation scenarios.

Stateful per scenario: stores ``sessionId`` from the first turn's ``result``
event for multi-turn ``--resume`` support.

Copilot CLI emits namespaced JSONL when invoked with ``--output-format json``
(``copilot -p <PROMPT>``). Programmatic mode requires ``--allow-all-tools``
(or the ``COPILOT_ALLOW_ALL=true`` env var) so the agent can actually run
tools without an interactive permission prompt.

Event schema:

  - ``user.message`` - prompt echo (skipped)
  - ``session.*`` - startup/MCP/skills loading (ignored)
  - ``assistant.turn_start`` / ``assistant.turn_end`` - turn boundaries;
    count of ``turn_start`` is reported as ``llm_turn_count``
  - ``assistant.message_start`` / ``assistant.message_delta`` - streaming
    frame and text deltas (used for ttft only; consolidated text comes
    from ``assistant.message``)
  - ``assistant.message`` - completed model message; carries
    ``data.content`` (final text) and ``data.toolRequests[]`` of
    ``{toolCallId, name, arguments, type}``
  - ``tool.execution_start`` / ``tool.execution_complete`` - tool calls
    and their results (used as a fallback source for tool calls; deduped
    by ``toolCallId``)
  - ``result`` - terminator with ``sessionId``, ``exitCode``, and
    ``usage.{totalApiDurationMs, sessionDurationMs, premiumRequests}``

The agent is thin plumbing - it translates ``execute()`` into a subprocess
call and parses output into ``TurnOutput``. Policy choices (model, allowed
tools, working directory) are controlled by scenario flags, not the agent.

References:
- https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference
- https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-programmatic-reference
"""

from __future__ import annotations

import json
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
from belt.agent.error_types import UNKNOWN, normalize_error_type
from belt.entities import ToolCall, TurnOutput, TurnTiming
from belt.parser.ndjson import parse_ndjson
from belt.runner.entities import AgentConfig
from belt.scenario import GroupConfig


class CopilotAgentAdapter(BaseAgentAdapter):
    """Agent for the GitHub Copilot CLI (``copilot``).

    Streams JSONL output to capture per-turn timing (ttfe, ttft, ttlt)
    alongside the structured output parsing.

    Authentication
    --------------
    Per the ``check_available()`` contract, this agent does not probe
    authentication state (see ``BaseAgentAdapter`` docstring). Copilot CLI
    accepts ``COPILOT_GITHUB_TOKEN``/``GH_TOKEN``/``GITHUB_TOKEN`` (in that
    precedence order) for headless use, or stored credentials in
    ``~/.copilot/`` from interactive ``copilot login`` - the doctor displays
    whichever signal it finds.

    Programmatic mode
    -----------------
    Copilot CLI in non-interactive (``-p``) mode needs ``--allow-all-tools``
    to actually run tools without an interactive permission prompt. Without
    it, scenarios that require shell/read/write fail silently because the
    agent cannot proceed past the first tool call. The agent passes the
    flag by default; scenarios that need stricter sandboxing can use the
    framework's path-allowlist or per-tool ``--allow-tool`` flags.
    """

    CREDENTIAL_ENV = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
    CREDENTIAL_PATHS = (Path.home() / ".copilot",)

    @classmethod
    def supported_output_fields(cls) -> frozenset[str]:
        return frozenset({"tool_sequence", "thinking_text", "llm_turn_count"})

    @classmethod
    def denied_flags(cls) -> frozenset[str]:
        # Block scenario-injected flags that would broaden the agent's
        # permissions beyond the tool-execution opt-in this agent passes
        # by default, or expose the session to external steering.
        # ``--allow-all`` / ``--yolo`` are the wholesale escape hatches;
        # ``--allow-all-tools`` / ``--allow-all-paths`` / ``--allow-all-urls``
        # weaken the tools / path / URL guards independently. ``--remote``
        # opens the session for external control; ``--connect`` attaches
        # to one. Selective ``--allow-tool`` / ``--allow-url`` are not
        # denied so scenarios can still grant narrow capabilities.
        return frozenset(
            {
                "--allow-all",
                "--allow-all-tools",
                "--allow-all-paths",
                "--allow-all-urls",
                "--yolo",
                "--remote",
                "--connect",
            }
        )

    @classmethod
    def check_available(cls) -> None:
        if not resolve_binary(("copilot",)):
            raise AgentNotAvailableError(
                "copilot",
                "copilot CLI not found on PATH",
                "Install: npm install -g @github/copilot",
            )

    @classmethod
    def cli_options(cls) -> list[AgentOption]:
        return [
            AgentOption(name="model", help="Model override (passed as --model flag)", env_var="COPILOT_MODEL"),
        ]

    @classmethod
    def required_env_vars(cls) -> frozenset[str]:
        # Extend the framework default with Copilot-specific runtime vars so
        # the minimal subprocess env still carries auth tokens, model
        # selection, the user's config dir, and the headless-permission
        # opt-in. Without this, ``build_subprocess_env`` would strip
        # ``COPILOT_GITHUB_TOKEN`` and the CLI would fail to authenticate.
        names = set(super().required_env_vars())
        names.update(
            {
                "COPILOT_GITHUB_TOKEN",
                "GH_TOKEN",
                "GITHUB_TOKEN",
                "COPILOT_MODEL",
                "COPILOT_HOME",
                "COPILOT_ALLOW_ALL",
                "COPILOT_GH_HOST",
                "GH_HOST",
            }
        )
        return frozenset(names)

    @classmethod
    def display_info(cls) -> str:
        bin_path = resolve_binary(("copilot",))
        if not bin_path:
            return "CopilotAgentAdapter (copilot CLI not found)"
        try:
            result = subprocess.run([bin_path, "--version"], capture_output=True, text=True, timeout=10)
            version = result.stdout.strip().split("\n")[0] if result.returncode == 0 else "unknown"
        except Exception:
            version = "unknown"
        return f"CopilotAgentAdapter (copilot {version})"

    @classmethod
    def runtime_info(cls) -> dict[str, Any]:
        info = super().runtime_info()
        bin_path = resolve_binary(("copilot",))
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

    def setup_group(self, group_config: GroupConfig, group_dir: Path) -> Any:
        return None

    def setup(self, config: AgentConfig) -> None:
        self._session_id = None
        self._workspace_dir = config.workspace_dir
        self._ttfe = None
        self._ttft = None
        self._ttlt = None

    def execute(self, message: str, flags: list[str]) -> str:
        cmd = [
            "copilot",
            "--output-format",
            "json",
            "--allow-all-tools",
            "--no-auto-update",
        ]
        if self._model:
            cmd.extend(["--model", self._model])
        if self._session_id:
            cmd.extend(["--resume", self._session_id])
        cmd.extend(self.filter_flags(flags))
        # ``-p`` takes the prompt as a flag value (Shape B). The message is a
        # separate argv element so a prompt starting with ``-``/``--`` cannot
        # be reparsed as an option by the Copilot CLI.
        cmd.extend(["-p", message])

        logger.debug("Running: {}", " ".join(cmd[:6]) + "...")
        return self._execute_streaming(cmd)

    def _execute_streaming(self, cmd: list[str]) -> str:
        """Run copilot CLI with streaming to capture timing metrics."""
        start = time.monotonic()
        self._ttfe = None
        self._ttft = None
        self._ttlt = None

        cwd = self._workspace_dir if self._workspace_dir else None
        env = self.make_subprocess_env({"COPILOT_ALLOW_ALL": "true"})
        proc = self._spawner.popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        stderr_thread = _drain_stderr(proc)

        lines: list[str] = []
        try:
            if proc.stdout is None:
                from belt.errors import AgentExecutionError

                raise AgentExecutionError("copilot Popen stdout is None")
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
                    try:
                        event = json.loads(stripped)
                        etype = event.get("type", "")
                        # ``assistant.message_start`` / ``assistant.message_delta``
                        # are the first events carrying real model text;
                        # ``assistant.turn_start`` is just a frame boundary.
                        if etype.startswith("assistant.") and etype != "assistant.turn_start":
                            self._ttft = t - start
                    except (json.JSONDecodeError, ValueError):
                        pass

                if stripped:
                    self._ttlt = t - start

            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            raise

        stderr_thread.join(timeout=5)
        raw_output = "".join(lines)
        stderr = "".join(stderr_thread.lines)  # type: ignore[attr-defined]

        if proc.returncode != 0:
            logger.warning("copilot returned rc={}: {}", proc.returncode, _sanitize_stderr(stderr))
            raw_output = raw_output + "\n" + stderr

        return raw_output

    def fetch_results(self, raw_output: str) -> TurnOutput:
        """Parse Copilot CLI's namespaced JSONL event stream.

        See module docstring for the full schema; this method extracts:

        - ``reply_text`` from ``assistant.message`` ``data.content``
        - ``tool_calls`` from ``assistant.message`` ``data.toolRequests[]``,
          deduplicated against ``tool.execution_start`` events
        - ``thinking_text`` from any ``thinking`` content block (reasoning models)
        - ``llm_turn_count`` from ``assistant.turn_start`` events
        - ``timing.total`` from ``result.usage.totalApiDurationMs``
        - ``has_error`` / ``error_type`` from ``result`` ``exitCode`` / ``error``
        - ``session_id`` from ``result.sessionId`` (used for ``--resume``)

        Defensive paths handle legacy/Claude-style flat shapes
        (``assistant`` / ``tool_use`` / ``function_call``) so a future schema
        change degrades gracefully rather than dropping data.
        """
        events = parse_ndjson(raw_output)

        reply_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        tool_sequence: list[str] = []
        seen_tool_call_ids: set[str] = set()
        session_id: str | None = None
        is_error: bool | None = None
        error_type: str | None = None
        duration_ms: float | None = None
        cost_usd: float | None = None
        llm_turn_count = 0

        def _add_tool_call(name: str, call_id: str, args: Any) -> None:
            if not name:
                return
            if call_id and call_id in seen_tool_call_ids:
                return
            if call_id:
                seen_tool_call_ids.add(call_id)
            tool_calls.append(ToolCall(name=name, call_id=call_id, args=args or {}))
            tool_sequence.append(name)

        for event in events:
            etype = event.get("type", "")
            data = event.get("data") if isinstance(event.get("data"), dict) else {}

            if etype == "assistant.turn_start":
                llm_turn_count += 1

            elif etype == "assistant.message":
                content = data.get("content", "")
                if isinstance(content, str) and content:
                    reply_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type", "")
                        if btype in ("text", "output_text"):
                            reply_parts.append(block.get("text", ""))
                        elif btype == "thinking":
                            thinking_parts.append(block.get("thinking", "") or block.get("text", ""))
                for req in data.get("toolRequests", []) or []:
                    if not isinstance(req, dict):
                        continue
                    _add_tool_call(req.get("name", ""), req.get("toolCallId", ""), req.get("arguments", {}))

            elif etype == "tool.execution_start":
                # Some message variants may omit ``toolRequests``; this event
                # is the authoritative per-call signal in that case. Deduped
                # against ``toolRequests`` by ``toolCallId``.
                _add_tool_call(data.get("toolName", ""), data.get("toolCallId", ""), data.get("arguments", {}))

            elif etype == "result":
                session_id = data.get("sessionId") or event.get("sessionId") or session_id
                exit_code = event.get("exitCode")
                if exit_code is None:
                    exit_code = data.get("exitCode")
                if exit_code is not None:
                    is_error = exit_code != 0
                usage = event.get("usage") if isinstance(event.get("usage"), dict) else data.get("usage", {})
                if isinstance(usage, dict):
                    duration_ms = usage.get("totalApiDurationMs") or usage.get("sessionDurationMs") or duration_ms
                # Copilot exposes premium-request quota, not USD cost; leave
                # ``cost_usd`` unset unless a future schema adds it.
                if "totalCostUsd" in event:
                    cost_usd = event.get("totalCostUsd")
                err = event.get("error") or data.get("error")
                if err:
                    error_type = str(err) if isinstance(err, str) else err.get("type")
                    if error_type:
                        is_error = True

            elif etype in ("assistant", "message"):
                role = event.get("role", "")
                if role == "user":
                    continue
                if etype == "assistant" or role == "assistant":
                    llm_turn_count += 1
                content = event.get("content", [])
                if isinstance(content, str) and content:
                    reply_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type", "")
                        if btype in ("text", "output_text"):
                            reply_parts.append(block.get("text", ""))
                        elif btype == "thinking":
                            thinking_parts.append(block.get("thinking", "") or block.get("text", ""))
                        elif btype == "tool_use":
                            _add_tool_call(block.get("name", ""), block.get("id", ""), block.get("input", {}))

            elif etype == "tool_use":
                _add_tool_call(event.get("name", ""), event.get("id", ""), event.get("input", {}))

            elif etype == "function_call":
                args: Any = event.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {"raw": args}
                _add_tool_call(
                    event.get("name", ""),
                    event.get("call_id", "") or event.get("id", ""),
                    args,
                )

        if session_id:
            self._session_id = session_id

        reply_text = "\n".join(p for p in reply_parts if p.strip())
        thinking_text = "\n".join(thinking_parts) if thinking_parts else None

        timing = None
        wall_total = self._ttlt if self._ttlt is not None else None
        api_secs = duration_ms / 1000.0 if duration_ms is not None else None
        if wall_total is not None or api_secs is not None:
            timing = TurnTiming(
                ttfe=self._ttfe,
                ttft=self._ttft,
                ttlt=self._ttlt,
                total=wall_total or api_secs,
            )

        has_error: bool | None
        if is_error is not None:
            has_error = is_error
        elif reply_text.strip():
            has_error = False
        else:
            has_error = "error" in raw_output.lower()

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
            cost_usd=cost_usd,
            thinking_text=thinking_text,
            llm_turn_count=llm_turn_count if llm_turn_count > 0 else None,
        )

    def teardown(self) -> None:
        self._session_id = None

    def metadata(self) -> dict[str, Any] | None:
        meta: dict[str, Any] = {}
        if self._session_id:
            meta["session_id"] = self._session_id
        return meta or None

    @staticmethod
    def parse_stream_event(event: dict) -> tuple[str, str] | None:
        """Render a Copilot JSONL event as (icon, summary) for live progress."""
        etype = event.get("type", "")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}

        if etype == "tool.execution_start":
            name = data.get("toolName", "?")
            args = data.get("arguments", {})
            if isinstance(args, dict) and args:
                args_str = ", ".join(f"{k}={v}" for k, v in args.items())
                if len(args_str) > 80:
                    args_str = args_str[:77] + "…"
                return "🔧", f"{name}({args_str})"
            return "🔧", f"{name}()"

        if etype == "assistant.message":
            content = data.get("content", "")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") in ("text", "output_text")
                ]
                text = " ".join(p for p in parts if p)
            if text:
                display = text.replace("\n", " ")
                if len(display) > 120:
                    display = display[:117] + "…"
                return "💬", display
            return None

        return None
