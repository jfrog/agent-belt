# (c) JFrog Ltd. (2026)

"""GeminiAgentAdapter - drives the Gemini CLI through evaluation scenarios.

Gemini CLI (`gemini`) supports structured NDJSON output (`--output-format stream-json`)
and multi-turn sessions (`--resume`).

Key event types in stream-json:
  - init: session_id, model
  - message: user/assistant messages with structured content blocks
    (assistant role is sometimes emitted as ``"model"`` in older CLI builds;
    the parser accepts both)
  - tool_call / tool_result: structured tool invocations and results
  - result: final summary with stats (tokens, duration_ms, tool_calls)

The agent is thin plumbing - it translates execute() into a subprocess call
and parses output into TurnOutput. Policy choices (model, sandbox, yolo mode)
are controlled by scenario flags, not the agent.

Reference: https://github.com/google-gemini/gemini-cli
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
from belt.agent.error_types import UNKNOWN, classify_error, normalize_error_type
from belt.entities import ToolCall, TurnOutput, TurnTiming
from belt.parser.ndjson import parse_ndjson
from belt.runner.entities import AgentConfig
from belt.scenario import GroupConfig


class GeminiAgentAdapter(BaseAgentAdapter):
    """Agent for the Google Gemini CLI.

    Streams NDJSON output to capture per-turn timing (ttfe, ttft, ttlt)
    alongside structured tool call parsing and multi-turn session support.

    Authentication
    --------------
    Per the ``check_available()`` contract, this agent does not probe
    authentication state (see ``BaseAgentAdapter`` docstring). Gemini accepts both
    ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` env vars and OAuth via
    ``~/.gemini/settings.json``. Auth failures surface at eval time with a
    clear error from the agent itself.

    The previous implementation invoked ``gemini -p ping`` with a 60s timeout
    as part of ``doctor``, which (a) consumed model credits, (b) made
    ``doctor`` slow on the happy path, and (c) was version-fragile.
    """

    CREDENTIAL_ENV = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    CREDENTIAL_PATHS = (Path.home() / ".gemini" / "settings.json",)

    @classmethod
    def supported_output_fields(cls) -> frozenset[str]:
        return frozenset({"tool_sequence"})

    @classmethod
    def denied_flags(cls) -> frozenset[str]:
        # ``--yolo`` is the published Gemini CLI flag that disables every
        # confirmation prompt; injecting it from a scenario lets that
        # scenario silently auto-approve destructive operations the
        # operator never reviewed. Block at the framework layer so the
        # decision to opt in stays explicit (override via
        # ``_allow_unsafe_flags``).
        return frozenset({"--yolo"})

    @classmethod
    def check_available(cls) -> None:
        if not resolve_binary(("gemini",)):
            raise AgentNotAvailableError(
                "gemini",
                "gemini CLI not found on PATH",
                "Install: npm install -g @google/gemini-cli",
            )

    @classmethod
    def display_info(cls) -> str:
        bin_path = resolve_binary(("gemini",))
        if not bin_path:
            return "GeminiAgentAdapter (gemini CLI not found)"
        try:
            result = subprocess.run([bin_path, "--version"], capture_output=True, text=True, timeout=10)
            version = result.stdout.strip().split("\n")[0] if result.returncode == 0 else "unknown"
        except Exception:
            version = "unknown"
        return f"GeminiAgentAdapter (gemini {version})"

    @classmethod
    def runtime_info(cls) -> dict[str, Any]:
        info = super().runtime_info()
        bin_path = resolve_binary(("gemini",))
        if bin_path:
            info["cli_binary_path"] = bin_path
            info["cli_version"] = cls._capture_cli_version([bin_path, "--version"])
        return info

    def __init__(self) -> None:
        self._session_id: str | None = None
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
        cmd = ["gemini", "--output-format", "stream-json"]
        if self._session_id:
            cmd.extend(["--resume", self._session_id])
        cmd.extend(self.filter_flags(flags))
        cmd.extend(["-p", message])

        logger.debug("Running: {}", " ".join(cmd[:6]) + "...")
        return self._execute_streaming(cmd)

    def _execute_streaming(self, cmd: list[str]) -> str:
        """Run gemini CLI with streaming to capture timing metrics."""
        start = time.monotonic()
        self._ttfe = None
        self._ttft = None
        self._ttlt = None

        cwd = self._workspace_dir if self._workspace_dir else None
        logger.debug("Running: {} (cwd={})", " ".join(cmd[:6]) + "...", cwd)
        proc = self._spawner.popen(
            cmd,
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

                raise AgentExecutionError("gemini Popen stdout is None")
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
                    if (
                        '"type"' in stripped
                        and '"message"' in stripped
                        and (
                            '"role":"model"' in stripped
                            or '"role": "model"' in stripped
                            or '"role":"assistant"' in stripped
                            or '"role": "assistant"' in stripped
                        )
                    ):
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
            logger.warning("gemini returned rc={}: {}", proc.returncode, _sanitize_stderr(stderr))
            raw_output = raw_output + "\n" + stderr

        return raw_output

    def fetch_results(self, raw_output: str) -> TurnOutput:
        events = parse_ndjson(raw_output)

        reply_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        tool_sequence: list[str] = []
        session_id = None
        is_error: bool | None = None
        error_type: str | None = None
        duration_ms: float | None = None

        for event in events:
            etype = event.get("type", "")

            if etype == "init":
                session_id = event.get("session_id")

            elif etype == "message" and event.get("role") in ("model", "assistant"):
                content = event.get("content")
                if isinstance(content, str):
                    reply_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type", "")
                        if btype == "text":
                            reply_parts.append(block.get("text", ""))
                        elif btype == "functionCall":
                            tc = ToolCall(
                                name=block.get("name", ""),
                                call_id=block.get("id", ""),
                                args=block.get("args", {}),
                            )
                            tool_calls.append(tc)
                            tool_sequence.append(tc.name)

            elif etype == "tool_call":
                tc = ToolCall(
                    name=event.get("name", ""),
                    call_id=event.get("id", ""),
                    args=event.get("args") or event.get("input", {}),
                )
                tool_calls.append(tc)
                tool_sequence.append(tc.name)

            elif etype == "result":
                stats = event.get("stats")
                if not isinstance(stats, dict):
                    stats = {}
                duration_ms = stats.get("duration_ms")
                status = event.get("status", "")
                if status == "error":
                    is_error = True
                    err = event.get("error", {})
                    if isinstance(err, dict):
                        error_type = err.get("type")
                    elif isinstance(err, str):
                        error_type = err
                elif status == "success":
                    is_error = False

        if session_id:
            self._session_id = session_id

        reply_text = "\n".join(reply_parts)

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
            # No structured signal: probe the raw output for known
            # failure shapes so an auth/refused/timeout is still
            # labelled. ``None`` (rather than ``False``) preserves the
            # historical "we don't know" state when nothing matches -
            # downstream surfaces treat it the same as a clean turn.
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
        )

    def teardown(self) -> None:
        self._session_id = None

    def metadata(self) -> dict[str, Any] | None:
        meta: dict[str, Any] = {}
        if self._session_id:
            meta["session_id"] = self._session_id
        return meta or None
