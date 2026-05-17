# (c) JFrog Ltd. (2026)

"""ClaudeCodeAgentAdapter - drives the Claude Code CLI through evaluation scenarios.

Stateful per scenario: stores session_id from the first turn's result event
for multi-turn `--resume` support.

Claude Code streams NDJSON (`--output-format stream-json`). Key event types:
  - assistant: contains reply text, tool_use blocks, and thinking blocks
  - result: final summary with session_id, cost, duration, is_error

The agent is thin plumbing - it translates execute() into a subprocess call
and parses output into TurnOutput. Policy choices (model, allowed tools,
working directory, timeout) are controlled by scenario flags, not the agent.
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


def _coerce_tool_result(block: dict[str, Any]) -> dict[str, Any]:
    """Flatten a Claude Code ``tool_result`` content block into a dict that
    `_iter_result_strings` (in ``belt.scorer.rules.helpers``) can match against.

    The Anthropic tool_result shape varies by tool:

    - Most tools return ``content`` as a list of typed items, where each text
      item is ``{"type": "text", "text": "..."}``. MCP tools follow this
      shape: the server's JSON payload arrives as one or more text items.
    - Some tools return ``content`` as a bare string (e.g. simple skill
      launch confirmations).
    - ``is_error`` may be set on either the block or a nested item to flag
      tool failures.

    We project all three into a flat ``{"text": ..., "content": ..., "is_error": ...}``
    dict so the scorer's substring / regex matchers see the server payload
    directly without authors having to know the underlying shape.
    """
    raw_content = block.get("content")
    result: dict[str, Any] = {}

    # Always preserve the raw content so non-text items remain reachable
    # through `_iter_result_strings`' json-dump fallback.
    if raw_content is not None:
        result["content"] = raw_content

    # Project a flat "text" key for the common case where the server returned
    # one or more text items: this is what `tool_result_contains` /
    # `tool_result_pattern` are designed to match against without escaping.
    if isinstance(raw_content, str):
        result["text"] = raw_content
    elif isinstance(raw_content, list):
        text_parts: list[str] = []
        for item in raw_content:
            if isinstance(item, dict) and item.get("type") == "text":
                t = item.get("text")
                if isinstance(t, str):
                    text_parts.append(t)
        if text_parts:
            result["text"] = "\n".join(text_parts)

    is_error = block.get("is_error")
    if is_error is not None:
        result["is_error"] = is_error

    return result


class ClaudeCodeAgentAdapter(BaseAgentAdapter):
    """Agent for the Claude Code CLI (`claude`).

    Streams NDJSON output to capture per-turn timing (ttfe, ttft, ttlt)
    alongside the structured output parsing.

    Authentication
    --------------
    Per the ``check_available()`` contract, this agent does not probe
    authentication state (see ``BaseAgentAdapter`` docstring). Claude Code accepts the
    ``ANTHROPIC_API_KEY`` env var or stored credentials in ``~/.claude/``
    (interactive ``claude login``) - the doctor displays whichever it finds.
    """

    CREDENTIAL_ENV = ("ANTHROPIC_API_KEY",)
    CREDENTIAL_PATHS = (Path.home() / ".claude.json", Path.home() / ".claude")

    @classmethod
    def supported_output_fields(cls) -> frozenset[str]:
        return frozenset({"tool_sequence", "thinking_text", "llm_turn_count"})

    @classmethod
    def denied_flags(cls) -> frozenset[str]:
        return frozenset({"--dangerously-skip-permissions"})

    @classmethod
    def check_available(cls) -> None:
        if not resolve_binary(("claude",)):
            raise AgentNotAvailableError(
                "claude-code",
                "claude CLI not found on PATH",
                "Install: npm install -g @anthropic-ai/claude-code",
            )

    @classmethod
    def required_env_vars(cls) -> frozenset[str]:
        # ``claude`` reads several env vars directly (model selection, base
        # URL, tiered defaults, auth alternatives, Foundry/Azure routing).
        # The agent class doesn't consume them itself - this override just
        # keeps them in the subprocess env allow-list so the CLI sees what the
        # user set in their shell. Auth keys come from the framework default
        # (``_DEFAULT_PROVIDER_ENV_VARS``).
        names = set(super().required_env_vars())
        names.update(
            {
                "ANTHROPIC_MODEL",
                "ANTHROPIC_DEFAULT_SONNET_MODEL",
                "ANTHROPIC_DEFAULT_OPUS_MODEL",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                "ANTHROPIC_AUTH_TOKEN",
                "CLAUDE_CODE_USE_FOUNDRY",
                "ANTHROPIC_FOUNDRY_BASE_URL",
                "AZURE_FOUNDRY_RESOURCE",
                "AZURE_FOUNDRY_API_KEY",
                "DISABLE_AUTOUPDATER",
                "DEFAULT_CLAUDE_MODEL",
            }
        )
        return frozenset(names)

    @classmethod
    def display_info(cls) -> str:
        bin_path = resolve_binary(("claude",))
        if not bin_path:
            return "ClaudeCodeAgentAdapter (claude CLI not found)"
        try:
            result = subprocess.run([bin_path, "--version"], capture_output=True, text=True, timeout=10)
            version = result.stdout.strip().split("\n")[0] if result.returncode == 0 else "unknown"
        except Exception:
            version = "unknown"
        return f"ClaudeCodeAgentAdapter (claude {version})"

    @classmethod
    def runtime_info(cls) -> dict[str, Any]:
        info = super().runtime_info()
        bin_path = resolve_binary(("claude",))
        if bin_path:
            info["cli_binary_path"] = bin_path
            info["cli_version"] = cls._capture_cli_version([bin_path, "--version"])
        return info

    def __init__(self) -> None:
        self._session_id: str | None = None
        self._ttfe: float | None = None
        self._ttft: float | None = None
        self._ttlt: float | None = None
        self._workspace_dir: str | None = None

    # ── Group lifecycle (no-op) ──

    def setup_group(self, group_config: GroupConfig, group_dir: Path) -> Any:
        return None

    # ── Scenario lifecycle ──

    def setup(self, config: AgentConfig) -> None:
        self._session_id = None
        self._ttfe = None
        self._ttft = None
        self._ttlt = None
        self._workspace_dir = config.workspace_dir

    def execute(self, message: str, flags: list[str]) -> str:
        cmd = ["claude", "-p", "--verbose", "--output-format", "stream-json"]
        if self._session_id:
            cmd.extend(["--resume", self._session_id])
        cmd.extend(self.filter_flags(flags))
        # ``--`` ends option parsing so a message starting with ``-``/``--``
        # is treated as a positional prompt by the Claude Code CLI rather
        # than as an unknown flag (or, worse, a recognised one).
        cmd.append("--")
        cmd.append(message)

        logger.debug("Running: {}", " ".join(cmd[:6]) + "...")
        return self._execute_streaming(cmd)

    def _execute_streaming(self, cmd: list[str]) -> str:
        """Run claude CLI with streaming to capture timing metrics."""
        start = time.monotonic()
        self._ttfe = None
        self._ttft = None
        self._ttlt = None

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

                raise AgentExecutionError("claude Popen stdout is None")
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
                    if '"type"' in stripped and '"assistant"' in stripped:
                        self._ttft = t - start

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
            logger.warning("claude returned rc={}: {}", proc.returncode, _sanitize_stderr(stderr))
            raw_output = raw_output + "\n" + stderr

        return raw_output

    def fetch_results(self, raw_output: str) -> TurnOutput:
        events = parse_ndjson(raw_output)

        reply_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        tool_calls_by_id: dict[str, ToolCall] = {}
        tool_sequence: list[str] = []
        session_id = None
        is_error: bool | None = None
        error_type: str | None = None
        duration_ms: float | None = None
        cost_usd: float | None = None
        llm_turn_count = 0

        def _record_tool_call(tc: ToolCall) -> None:
            tool_calls.append(tc)
            tool_sequence.append(tc.name)
            if tc.call_id:
                tool_calls_by_id[tc.call_id] = tc

        for event in events:
            etype = event.get("type", "")

            if etype == "assistant":
                llm_turn_count += 1
                content = event.get("content", [])
                if not content and "message" in event:
                    msg = event.get("message")
                    content = msg.get("content", []) if isinstance(msg, dict) else []
                if not isinstance(content, list):
                    content = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        reply_parts.append(block.get("text", ""))
                    elif btype == "thinking":
                        thinking_parts.append(block.get("thinking", "") or block.get("text", ""))
                    elif btype == "tool_use":
                        _record_tool_call(
                            ToolCall(
                                name=block.get("name", ""),
                                call_id=block.get("id", ""),
                                args=block.get("input", {}),
                            )
                        )

            elif etype == "tool_use":
                _record_tool_call(
                    ToolCall(
                        name=event.get("name", ""),
                        call_id=event.get("id", ""),
                        args=event.get("input", {}),
                    )
                )

            elif etype == "user":
                # Tool results are carried back to the model on the next
                # ``user`` turn as ``tool_result`` content blocks. Each block
                # references the matching ``tool_use`` via ``tool_use_id`` so
                # we can attach the server-side payload onto ``ToolCall.result``,
                # the field the scorer's ``tool_result_contains`` /
                # ``tool_result_pattern`` rules read from. Without this, every
                # Claude Code tool call (MCP, Skill, ToolSearch, Read, Bash,
                # ...) ships with ``result=None`` and reply-side scoring is
                # the only working signal.
                msg = event.get("message")
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_result":
                        continue
                    call_id = block.get("tool_use_id")
                    if not isinstance(call_id, str):
                        continue
                    target = tool_calls_by_id.get(call_id)
                    if target is None:
                        continue
                    target.result = _coerce_tool_result(block)

            elif etype == "result":
                session_id = event.get("session_id")
                is_error = event.get("is_error", False)
                duration_ms = event.get("duration_ms")
                cost_usd = event.get("total_cost_usd") or event.get("cost_usd")
                if event.get("error"):
                    error_type = str(event["error"]) if isinstance(event["error"], str) else event["error"].get("type")

        if session_id:
            self._session_id = session_id

        reply_text = "\n".join(reply_parts)
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
        else:
            has_error = "error" in raw_output.lower()

        # Claude Code's ``result`` event sets ``is_error=true`` for auth
        # failures but does not populate ``error`` - the 401/login text
        # lands in the assistant ``text`` block (and therefore in
        # ``reply_text``). Normalise so the framework's ``error_type``
        # is never silently null when ``has_error=true`` and so any raw
        # vendor token gets projected onto the canonical taxonomy.
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
