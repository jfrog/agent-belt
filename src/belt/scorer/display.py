# (c) JFrog Ltd. (2026)

"""Single source of truth for verdict-token display (icon + color).

Renderers and exporters that surface LLM judge verdicts to a human
(terminal result panel, ``belt view``, GitHub step-summary markdown,
JUnit XML, etc.) import the dictionaries below so the icon-and-colour
story stays consistent across surfaces. Adding a new verdict token or
re-skinning an existing one requires editing this module - and only
this module - rather than chasing copies through every renderer
(Design Principle 9).

The verdict tokens themselves are defined by
:class:`belt.scorer.entities.ScoreLevel`; this module pairs each token
with its rendering attributes without re-spelling the literals.
"""

from __future__ import annotations

from typing import NamedTuple

from belt.scorer.entities import ScoreLevel


class VerdictDisplay(NamedTuple):
    """Rendering attributes for a single verdict token.

    ``icon`` is a short unicode glyph for table cells and panels;
    ``color`` is a Rich-compatible style name used inside panel bodies
    (``[green]``, ``[yellow]``, ``[red]``). Renderers that only need
    the icon discard the colour.
    """

    icon: str
    color: str


VERDICT_DISPLAY: dict[str, VerdictDisplay] = {
    ScoreLevel.HIGH.value: VerdictDisplay("✅", "green"),
    ScoreLevel.MEDIUM.value: VerdictDisplay("⚠️", "yellow"),
    ScoreLevel.LOW.value: VerdictDisplay("❌", "red"),
    ScoreLevel.PASS.value: VerdictDisplay("✅", "green"),
    ScoreLevel.FAIL.value: VerdictDisplay("❌", "red"),
    ScoreLevel.INCONCLUSIVE.value: VerdictDisplay("❓", "yellow"),
}
"""Verdict token → (icon, Rich color). Keys are :class:`ScoreLevel`
values so a typo in a renderer fails fast rather than silently using
the fallback."""

UNKNOWN_VERDICT_DISPLAY: VerdictDisplay = VerdictDisplay("?", "white")
"""Fallback for renderers that may encounter a future verdict token.
Today every emitted verdict is defined in :class:`ScoreLevel`, so a
fallback hit indicates either a forward-compat read of an older
artifact or a bug."""


def verdict_label(score: str) -> str:
    """Return ``"<icon> <score>"`` for terminal cells that show both.

    Falls back to ``"? <score>"`` for unknown tokens rather than
    raising; the caller is rendering, not validating.
    """
    display = VERDICT_DISPLAY.get(score, UNKNOWN_VERDICT_DISPLAY)
    return f"{display.icon} {score}"
