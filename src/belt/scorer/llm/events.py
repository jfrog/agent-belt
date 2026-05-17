# (c) JFrog Ltd. (2026)

"""Score events emitted during LLM scoring for live progress display.

Internal to the scorer phase - never imported by runner or aggregator.
The ``format_score_event`` function converts events to pre-formatted strings
so that ``progress.py`` only deals with plain strings (respecting P1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_MAX_REASONING = 120


@dataclass(slots=True)
class ScoreEvent:
    """A single event emitted during LLM scoring."""

    kind: str  # "start" | "cache_hit" | "verdict" | "done"
    scenario: str
    dimension: str = ""
    # Any token from :class:`belt.scorer.entities.ScoreLevel` - the
    # comment is deliberately not enumerated to avoid drift from the
    # enum.
    score: str = ""
    reasoning: str = ""
    judge: str = ""
    passed: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for NDJSON output."""
        d: dict[str, Any] = {"kind": self.kind, "scenario": self.scenario}
        if self.dimension:
            d["dimension"] = self.dimension
        if self.score:
            d["score"] = self.score
        if self.reasoning:
            d["reasoning"] = self.reasoning
        if self.judge:
            d["judge"] = self.judge
        if self.passed is not None:
            d["passed"] = self.passed
        if self.extra:
            d.update(self.extra)
        return d


def _truncate(text: str, max_len: int = _MAX_REASONING) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def format_score_event(event: ScoreEvent) -> str:
    """Format a ScoreEvent into a Rich-markup string for display.

    All event-derived fields (judge, dimension, score, reasoning) are
    treated as untrusted: the judge name comes from configuration, but the
    reasoning text is the LLM judge's verbatim response and dimension may
    flow through from rule scorers that quote agent output. Each value
    passes through ``rich_safe`` so the only Rich markup interpreted is
    what this formatter itself emits.
    """
    from belt._safe import rich_safe
    from belt.scorer.display import VERDICT_DISPLAY

    judge_prefix = f"[dim]{rich_safe(event.judge)}:[/dim] " if event.judge else ""

    if event.kind == "start":
        return f"  🎯 {judge_prefix}scoring…"

    if event.kind == "cache_hit":
        return f"  ⚡ {judge_prefix}cache hit"

    if event.kind == "verdict":
        display = VERDICT_DISPLAY.get(event.score)
        style = display.color if display else "dim"
        snippet = _truncate(event.reasoning) if event.reasoning else ""
        reason_part = f' · "{rich_safe(snippet)}"' if snippet else ""
        return (
            f"  📊 {judge_prefix}{rich_safe(event.dimension)}: "
            f"[{style}]{rich_safe(event.score)}[/{style}]{reason_part}"
        )

    if event.kind == "error":
        return f"  ⚠️  {judge_prefix}[red]API error - no verdict[/red]"

    if event.kind == "done":
        icon = "✅" if event.passed else "❌"
        return f"  {icon} {judge_prefix}done"

    return f"  ❓ {judge_prefix}{rich_safe(event.kind)}"
