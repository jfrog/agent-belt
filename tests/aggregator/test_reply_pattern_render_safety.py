# (c) JFrog Ltd. (2026)

"""Rich-markup safety for ``reply_pattern`` failure details.

The ``reply_pattern`` failure path emits ``CheckEntry.details`` of the
form ``f"reply did not match: {output.reply_text[:200]!r}"``. When the
agent's reply contains literal Rich-markup tokens (``[bold red]``,
``[/]``), an unescaped interpolation into the terminal panel would let
agent stdout style or corrupt the renderer output. ``rich_safe`` in
``render_terminal.print_terminal`` is supposed to neutralise this; this
file is the regression pin.

Rich's escape mechanism works at the *parser* level:
``Text.from_markup("[bold red]X")`` consumes ``[bold red]`` as a style
tag (the brackets disappear from the output, even when no color system
is active). ``Text.from_markup("\\[bold red]X")`` treats the brackets
as literal text - they survive into the rendered output. So the
load-bearing assertion is "the literal markup tokens survive": if they
do, the escape worked; if they're missing, Rich consumed them.
"""

from __future__ import annotations

import io

from rich.console import Console
from rich.text import Text

from belt._safe import rich_safe
from belt.aggregator.render_terminal import print_terminal
from belt.entities import ScenarioScore
from belt.scorer.payloads import CheckEntry, RulesPayload


def _failed_score_with_reply_pattern_details(reply_snippet: str) -> ScenarioScore:
    rules = RulesPayload(
        passed=False,
        checks=[
            CheckEntry(
                dimension="response",
                check=r"reply_pattern(^ORDER\|)",
                passed=False,
                # Mirror what ``response.check_response`` writes: a
                # ``repr()``-quoted snippet of the agent reply, embedded
                # in a fixed prefix.
                details=f"reply did not match: {reply_snippet!r}",
                turn_idx=0,
            )
        ],
    )
    return ScenarioScore(
        scenario_name="malicious_reply",
        group="hardening",
        tags=["real-runnable"],
        scores={"rules": rules},
        overall_pass=False,
    )


def _render(score: ScenarioScore) -> str:
    buf = io.StringIO()
    # ``force_terminal=False`` + ``color_system=None`` keeps the rendered
    # bytes deterministic across CI runners and free of ANSI noise.
    console = Console(file=buf, force_terminal=False, color_system=None, width=200)
    print_terminal([score], run_label="rich-safety-test", console=console)
    return buf.getvalue()


def test_rich_text_from_markup_consumes_unescaped_brackets_baseline() -> None:
    """Baseline: prove the threat model.

    Without escaping, ``Text.from_markup("[bold red]X")`` consumes the
    ``[bold red]`` token entirely - the brackets disappear from the
    plain-text output even when no color system is attached. Pinning
    this baseline locks the danger that ``rich_safe`` defends against.
    """
    plain = Text.from_markup("[bold red]ORDER|42[/]").plain
    assert plain == "ORDER|42"  # markup tokens were consumed by the parser
    # And: ``rich_safe`` neutralises the same input so the brackets survive.
    plain_safe = Text.from_markup(rich_safe("[bold red]ORDER|42[/]")).plain
    assert "[bold red]" in plain_safe
    assert "ORDER|42" in plain_safe


def test_reply_pattern_details_preserve_rich_markup_tokens_literally() -> None:
    """If ``rich_safe`` works inside ``print_terminal``, the markup tokens
    survive into the rendered output instead of being consumed by the parser."""
    rendered = _render(_failed_score_with_reply_pattern_details("[bold red]ORDER|42[/]"))
    assert "[bold red]" in rendered
    assert "ORDER|42" in rendered
    assert "[/]" in rendered


def test_reply_pattern_details_preserve_closing_markup_token() -> None:
    """Closing tokens (``[/]``, ``[/bold]``) are equally dangerous - they
    can terminate a Rich panel's own styling and bleed into siblings."""
    rendered = _render(_failed_score_with_reply_pattern_details("ORDER|[/]"))
    assert "ORDER|" in rendered
    assert "[/]" in rendered


def test_reply_pattern_details_does_not_crash_on_unbalanced_brackets() -> None:
    """An unbalanced ``[`` must not propagate :class:`MarkupError` up
    through ``print_terminal`` - the renderer has to survive any agent
    reply, including malformed ones."""
    rendered = _render(_failed_score_with_reply_pattern_details("ORDER|[unclosed"))
    assert "ORDER|" in rendered
    assert "[unclosed" in rendered
