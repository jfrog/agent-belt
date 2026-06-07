# (c) JFrog Ltd. (2026)

"""Multi-judge non-consensus runs surface each judge in `stats["scorers"]`.

A ``--scorer-config`` declaring two independent judges (no
``consensus:`` key) writes ``scores["primary_judge"]`` +
``scores["adversarial_judge"]`` on each :class:`ScenarioScore`.
``build_stats`` indexes the resulting per-dimension histograms by
scorer key so the aggregator, result table, markdown step-summary,
CSV, JUnit per-``(scenario, scorer_key, dim)`` testcases, ``belt
view``, ``belt compare``, and the benchmark card all surface every
judge attributably.
"""

from __future__ import annotations

from belt.aggregator.stats import build_stats
from belt.entities import ScenarioScore
from belt.scorer.payloads import LLMDimensionVerdict, LLMPayload


def _payload(score: str) -> LLMPayload:
    return LLMPayload(
        overall_pass=score == "high",
        dimensions={"correctness": LLMDimensionVerdict(score=score, reasoning="ok")},
    )


def _scenario_with_two_judges(score_a: str, score_b: str) -> ScenarioScore:
    """A scenario scored independently by two LLM judges (non-consensus mode)."""
    return ScenarioScore(
        scenario_name="s",
        group="g",
        overall_pass=score_a == "high" and score_b == "high",
        scores={
            "primary_judge": _payload(score_a),
            "adversarial_judge": _payload(score_b),
        },
    )


class TestPerScorerStats:
    def test_both_judges_appear_in_stats_scorers(self) -> None:
        scores = [
            _scenario_with_two_judges("high", "high"),
            _scenario_with_two_judges("low", "high"),
            _scenario_with_two_judges("low", "low"),
        ]
        stats = build_stats(scores)

        scorers = stats["scorers"]
        assert "primary_judge" in scorers, scorers
        assert "adversarial_judge" in scorers, scorers
        # Each scorer carries its own verdict histogram.
        assert scorers["primary_judge"]["correctness"]["total"] == 3
        assert scorers["primary_judge"]["correctness"]["low"] == 2
        assert scorers["adversarial_judge"]["correctness"]["total"] == 3
        assert scorers["adversarial_judge"]["correctness"]["low"] == 1

    def test_top_level_keys_are_canonical(self) -> None:
        """`stats` exposes ``pass_rate`` and ``scorers`` as the only
        top-level histogram surface. No legacy ``llm`` / ``rules``
        siblings."""
        scores = [_scenario_with_two_judges("high", "high")]
        stats = build_stats(scores)
        assert "scorers" in stats
        assert "llm" not in stats
        assert "rules" not in stats
