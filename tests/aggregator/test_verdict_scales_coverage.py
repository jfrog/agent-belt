# (c) JFrog Ltd. (2026)

"""End-to-end coverage matrix for the verdict-scale feature.

This module is the "scenario plan" for the configurable-verdict-scale
feature made concrete: each test exercises one cell of the matrix
(``token`` × ``aggregator path``) without depending on an LLM judge, so
the verification is deterministic and fast.

Matrix axes:

* **Verdict token** - every value of :class:`ScoreLevel`
  (``high`` / ``medium`` / ``low`` / ``pass`` / ``fail`` / ``inconclusive``).
* **Downstream surface** - histogram bucketing, headline pass-rate
  recomputation, ``overall_pass`` truth-table, ``INCONCLUSIVE_CEILING_PCT``
  alarm, ``--llm-fail-on`` threshold gating, terminal/markdown/JUnit
  renderers.

The end-to-end LLM run (Ollama gemma3) covers ``low``/``fail``/``pass`` for
the rendered ``examples/scenarios/showcase/verdict-scales/`` group; the
remaining three tokens (``medium``/``high``/``inconclusive``) are exercised
here by synthesising :class:`LLMPayload` directly. Together the two layers
prove every token round-trips through every aggregator surface.
"""

from __future__ import annotations

import pytest

from belt.aggregator.stats import _VERDICT_BUCKETS, INCONCLUSIVE_CEILING_PCT, build_bottom_line, build_stats
from belt.aggregator.thresholds import VALID_LLM_FAIL_ON, ThresholdEnforcer
from belt.entities import ScenarioScore
from belt.scorer.entities import (
    ALL_VERDICT_TOKENS,
    DEFAULT_FAIL_LEVELS,
    DEFAULT_LLM_FAIL_ON,
    DEFAULT_LLM_FAIL_ON_STR,
    DOWNGRADE_VERDICT_SET,
    DOWNGRADE_VERDICTS,
    ScoreLevel,
)
from belt.scorer.payloads import LLMDimensionVerdict, LLMPayload

_ALL_TOKENS: tuple[str, ...] = ALL_VERDICT_TOKENS


def _llm_score(name: str, verdicts: dict[str, str]) -> ScenarioScore:
    """Build a ScenarioScore whose LLM payload carries the given verdicts.

    ``overall_pass`` is computed using the production rule so this
    helper matches what the LLM scorer would actually persist: any
    failing verdict flips it to ``false``.
    """
    dims = {dim: LLMDimensionVerdict(score=score, reasoning=f"reason for {dim}") for dim, score in verdicts.items()}
    overall = not any(v.score in DEFAULT_FAIL_LEVELS for v in dims.values())
    payload = LLMPayload(overall_pass=overall, dimensions=dims)
    return ScenarioScore(
        scenario_name=name,
        group="verdict-coverage",
        scores={"llm": payload},
        overall_pass=overall,
    )


# ── Constants drift: derived constants stay coherent ──


def test_constants_derive_from_score_level_enum() -> None:
    """All verdict constants are derived from :class:`ScoreLevel` so a
    new enum value propagates without separate edits to histograms,
    CLI validators, or threshold defaults (Principle 9)."""
    assert set(_ALL_TOKENS) == {level.value for level in ScoreLevel}
    assert _VERDICT_BUCKETS == _ALL_TOKENS
    assert VALID_LLM_FAIL_ON == set(_ALL_TOKENS)
    assert DEFAULT_FAIL_LEVELS == frozenset(DEFAULT_LLM_FAIL_ON)
    assert DEFAULT_LLM_FAIL_ON_STR == ",".join(DEFAULT_LLM_FAIL_ON)
    assert DOWNGRADE_VERDICT_SET == frozenset(DOWNGRADE_VERDICTS)
    # ``medium`` is the only verdict downgrade-but-not-fail; if this
    # ever changes, audit the renderers that surface reasoning blocks.
    assert DOWNGRADE_VERDICT_SET - DEFAULT_FAIL_LEVELS == {"medium"}


# ── Histogram: every token buckets correctly ──


@pytest.mark.parametrize("token", _ALL_TOKENS)
def test_histogram_buckets_every_verdict_token(token: str) -> None:
    """``build_stats`` puts each emitted token into its own bucket and
    leaves all other buckets at zero."""
    scores = [_llm_score(f"s_{token}", {"dim": token})]
    stats = build_stats(scores)
    hist = stats["llm"]["dim"]
    assert hist[token] == 1
    assert hist["total"] == 1
    for other in _ALL_TOKENS:
        if other != token:
            assert hist[other] == 0, f"unexpected count in {other!r} bucket"


def test_histogram_initialised_with_all_buckets_at_zero() -> None:
    """Even when only a subset of tokens are emitted in the run, the
    histogram still exposes every bucket so JSON consumers see a
    stable shape."""
    scores = [_llm_score("s1", {"dim": "high"}), _llm_score("s2", {"dim": "pass"})]
    hist = build_stats(scores)["llm"]["dim"]
    for token in _ALL_TOKENS:
        assert token in hist, f"bucket {token!r} missing from histogram"


# ── Headline pass-rate: inconclusive cannot earn a pass ──


@pytest.mark.parametrize("failing_token", ["low", "fail", "inconclusive"])
def test_failing_tokens_flip_overall_pass_to_false(failing_token: str) -> None:
    """``low``, ``fail``, and ``inconclusive`` are hardcoded as
    failures in the headline pass-rate (cannot be overridden by
    ``--llm-fail-on``; see SCORING.md §2.5)."""
    score = _llm_score("s", {"dim": failing_token})
    assert score.overall_pass is False


@pytest.mark.parametrize("passing_token", ["high", "medium", "pass"])
def test_passing_tokens_keep_overall_pass_true(passing_token: str) -> None:
    """``high``, ``medium``, and ``pass`` do not flip the headline."""
    score = _llm_score("s", {"dim": passing_token})
    assert score.overall_pass is True


def test_inconclusive_alongside_pass_still_fails_headline() -> None:
    """A single inconclusive verdict is enough to fail the scenario
    even when every other dimension passes."""
    score = _llm_score("s", {"correctness": "pass", "safety": "inconclusive"})
    assert score.overall_pass is False


# ── Inconclusive-ceiling alarm ──


def _inconclusive_run(count_inconclusive: int, count_pass: int) -> list[ScenarioScore]:
    return [_llm_score(f"i_{i}", {"safety": "inconclusive"}) for i in range(count_inconclusive)] + [
        _llm_score(f"p_{i}", {"safety": "pass"}) for i in range(count_pass)
    ]


def test_inconclusive_ceiling_alarm_fires_above_threshold() -> None:
    """When the inconclusive ratio exceeds
    :data:`INCONCLUSIVE_CEILING_PCT`, the aggregator emits a warning
    that surfaces in renderers and CI step summaries."""
    scores = _inconclusive_run(count_inconclusive=3, count_pass=7)
    assert (3 / 10) * 100 > INCONCLUSIVE_CEILING_PCT
    stats = build_stats(scores)
    warnings = stats.get("llm_inconclusive_warnings", [])
    assert warnings, "ceiling exceeded but no warning emitted"
    assert any("safety" in w for w in warnings)


def test_inconclusive_ceiling_alarm_silent_below_threshold() -> None:
    """A low inconclusive ratio is normal noise and should not
    trigger the warning - the ceiling is a heuristic for *broken*
    rubrics, not a hair-trigger."""
    scores = _inconclusive_run(count_inconclusive=1, count_pass=19)
    assert (1 / 20) * 100 < INCONCLUSIVE_CEILING_PCT
    stats = build_stats(scores)
    assert not stats.get("llm_inconclusive_warnings", [])


# ── Threshold gating: --llm-fail-on respects every token ──


@pytest.mark.parametrize("fail_set", [{"low"}, {"fail"}, {"inconclusive"}, {"low", "fail", "inconclusive"}, {"medium"}])
def test_threshold_enforcer_respects_llm_fail_on(fail_set: set[str]) -> None:
    """Per-dimension failure counts under ``--llm-fail-on`` are exactly
    the count of verdicts whose token is in the configured set."""
    scores = [_llm_score(f"s{i}", {"dim": t}) for i, t in enumerate(_ALL_TOKENS)]
    enforcer = ThresholdEnforcer(scores, llm_fail_on=fail_set)
    failed, total = enforcer.llm_failures["dim"]
    assert total == len(_ALL_TOKENS)
    expected = sum(1 for t in _ALL_TOKENS if t in fail_set)
    assert failed == expected


def test_threshold_enforcer_default_matches_default_fail_levels() -> None:
    """Constructing without an explicit set falls back to
    :data:`DEFAULT_FAIL_LEVELS` so CLI and library callers agree on
    "no override = strict mode"."""
    scores = [_llm_score(f"s{i}", {"dim": t}) for i, t in enumerate(_ALL_TOKENS)]
    enforcer = ThresholdEnforcer(scores)
    failed, _ = enforcer.llm_failures["dim"]
    assert failed == sum(1 for t in _ALL_TOKENS if t in DEFAULT_FAIL_LEVELS)


# ── Bottom-line text: each failing verdict yields a separate reason ──


def test_bottom_line_groups_failures_by_verdict_class() -> None:
    """The aggregator emits distinct lines for ``low`` / ``fail`` /
    ``inconclusive`` so reviewers see at a glance which kind of
    failure dominates."""
    scores = [
        _llm_score("a", {"quality": "low"}),
        _llm_score("b", {"correctness": "fail"}),
        _llm_score("c", {"safety": "inconclusive"}),
    ]
    bottom_line = build_bottom_line(scores)
    text = "\n".join(bottom_line)
    assert "low" in text
    assert "fail" in text
    assert "inconclusive" in text


# ── Renderers: every token survives a round-trip to display ──


@pytest.mark.parametrize("token", _ALL_TOKENS)
def test_terminal_renderer_displays_every_verdict_token(token: str, tmp_path) -> None:
    """The terminal panel renders each verdict token without falling
    through to the ``?`` fallback (proof the display table covers
    every emitted verdict)."""
    from io import StringIO

    from rich.console import Console

    from belt.aggregator.render_terminal import print_terminal

    scores = [_llm_score("s", {"dim": token})]
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=160, color_system=None)
    print_terminal(scores, run_label="test", outcomes_root=tmp_path, console=console)
    out = buf.getvalue()
    assert token in out
    assert "? dim" not in out


@pytest.mark.parametrize("token", DOWNGRADE_VERDICT_SET)
def test_markdown_renderer_surfaces_downgrade_tokens_in_failed_scenarios(token: str) -> None:
    """The GitHub step-summary markdown surfaces the token + reasoning
    for every verdict in :data:`DOWNGRADE_VERDICT_SET` *when the
    scenario fails*. ``medium`` is in the set so a failed scenario's
    ``medium`` dimension still gets explained, even though ``medium``
    on its own does not flip ``overall_pass``."""
    from belt.aggregator.render_markdown import build_markdown

    # Pair the downgrade verdict with a guaranteed failing verdict so
    # the scenario reaches the failure block; otherwise overall_pass
    # is true and the renderer renders no failures at all.
    scores = [_llm_score("s", {"focus": token, "anchor_fail": "fail"})]
    md = build_markdown(scores)
    assert "Failures" in md
    assert token in md, f"verdict {token!r} expected in markdown failure block"


@pytest.mark.parametrize("token", [t for t in ALL_VERDICT_TOKENS if t not in DOWNGRADE_VERDICT_SET])
def test_markdown_renderer_omits_passing_tokens_from_failure_block(token: str) -> None:
    """``high`` and ``pass`` never appear in the failure block; their
    role is purely positive headline contributions."""
    from belt.aggregator.render_markdown import build_markdown

    # Pair with a failure so the failure block is rendered; the
    # passing-side token still shouldn't appear in it.
    scores = [_llm_score("s", {"focus": token, "anchor_fail": "fail"})]
    md = build_markdown(scores)
    assert "Failures" in md
    # The token may legitimately appear elsewhere in the document
    # (header counts, thresholds row); restrict the assertion to the
    # failures bullet list.
    failures_block = md.split("### Failures", 1)[-1]
    assert f"**{token}**" not in failures_block


@pytest.mark.parametrize("token", _ALL_TOKENS)
def test_junit_exporter_emits_failure_for_failing_tokens(token: str, tmp_path) -> None:
    """JUnit XML emits a ``<failure>`` element exactly for failing
    verdicts (``low`` / ``fail`` / ``inconclusive``) and a clean
    ``<testcase>`` for passing ones."""
    from belt.entities import AggregatedResults
    from belt.exporter.entities import ExportContext
    from belt.exporter.junit import JUnitExporter

    scores = [_llm_score("scenario", {"dim": token})]
    results = AggregatedResults(
        total=1,
        passed=(token not in DEFAULT_FAIL_LEVELS),
        failed=(1 if token in DEFAULT_FAIL_LEVELS else 0),
        pass_rate=1.0 if token not in DEFAULT_FAIL_LEVELS else 0.0,
    )
    out_path = tmp_path / "junit.xml"
    ctx = ExportContext(run_dir=tmp_path, scores=scores, results=results)
    JUnitExporter().export(ctx, out_path, options={})
    xml = out_path.read_text()
    if token in DEFAULT_FAIL_LEVELS:
        assert "<failure" in xml
        assert "below pass" in xml
    else:
        assert "<failure" not in xml


# ── Display constants: every token resolves to a non-fallback icon ──


@pytest.mark.parametrize("token", _ALL_TOKENS)
def test_verdict_display_covers_every_token(token: str) -> None:
    """Each :class:`ScoreLevel` value resolves to a defined entry in
    :data:`VERDICT_DISPLAY`; missing here means the renderer falls
    back to a ``?`` icon at runtime."""
    from belt.scorer.display import UNKNOWN_VERDICT_DISPLAY, VERDICT_DISPLAY

    display = VERDICT_DISPLAY.get(token)
    assert display is not None
    assert display != UNKNOWN_VERDICT_DISPLAY
    assert display.icon and display.color
