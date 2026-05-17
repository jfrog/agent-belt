# (c) JFrog Ltd. (2026)

"""Minimal belt agent - echoes input back as output.

Install:   pip install -e examples/custom-agent
Run:       belt eval examples/custom-agent/scenarios --agent echo --modes rules
"""

from __future__ import annotations

import time
from importlib import metadata as importlib_metadata
from typing import Any

from belt import AgentConfig, BaseAgentAdapter, ToolCall, TurnOutput, TurnTiming


class EchoAgentAdapter(BaseAgentAdapter):
    """Trivial agent that returns the input message as the reply.

    Useful as a template for writing real agents. The implementation
    populates the universal ``TurnOutput`` fields plus a synthesized
    ``echo`` tool call so scenarios can exercise tool-trajectory checks
    against this agent without depending on a remote LLM.
    """

    @classmethod
    def runtime_info(cls) -> dict[str, Any]:
        """Demonstrate the ``runtime_info`` override pattern.

        Real agents return their resolved CLI binary path and version;
        this template has no external CLI, so it reports the bundled
        plugin's package version instead. The base class default is also
        valid - override only when you have meaningful identity to add.
        """
        info = super().runtime_info()
        try:
            info["cli_version"] = importlib_metadata.version("belt-echo")
        except importlib_metadata.PackageNotFoundError:
            pass
        return info

    def __init__(self) -> None:
        self._t0: float | None = None

    def setup(self, config: AgentConfig) -> None:
        self._t0 = time.monotonic()

    def execute(self, message: str, flags: list[str]) -> str:
        return f"echo: {message}"

    def fetch_results(self, raw_output: str) -> TurnOutput:
        elapsed = time.monotonic() - (self._t0 or time.monotonic())
        return TurnOutput(
            raw_cli=raw_output,
            reply_text=raw_output.strip(),
            tool_calls=[ToolCall(name="echo", call_id="echo-0", args={"text": raw_output.strip()})],
            tool_sequence=["echo"],
            has_reply=bool(raw_output.strip()),
            has_error=False,
            timing=TurnTiming(total=elapsed),
        )

    def teardown(self) -> None:
        self._t0 = None

    @classmethod
    def display_info(cls, **kwargs: Any) -> str:
        return "EchoAgentAdapter v0.2 (example)"
