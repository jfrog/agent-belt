# (c) JFrog Ltd. (2026)

"""Consensus scorer: majority-vote across multiple LLM judges.

Wraps N ``LLMScorer`` instances, runs all on the same scenario, then
majority-votes shared dimensions. Non-shared dimensions pass through
from whichever judge scored them.

Activated via ``consensus: majority`` in ``--scorer-config`` YAML.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

from loguru import logger

from belt.entities import TurnOutput
from belt.scenario import Scenario
from belt.scorer.base import BaseScorer
from belt.scorer.entities import ScoreLevel, ScorerResult
from belt.scorer.llm.events import ScoreEvent
from belt.scorer.llm.scorer import LLMScorer
from belt.scorer.payloads import ConsensusMeta, LLMDimensionVerdict, LLMPayload, UsageStats

CONSENSUS_STRATEGIES = {"majority", "unanimous", "any"}


def _all_judges_errored_payload(errored: dict[str, ScorerResult]) -> LLMPayload:
    """Build a merged ``judge_errored=True`` payload when every sub-judge errored.

    Picks the most-common error type across sub-judges as the headline
    ``judge_error_type`` (ties break alphabetically so the choice is
    deterministic across runs). Per-judge error types are preserved in
    ``individual_verdicts`` so a postmortem reader can see exactly which
    provider produced which token.
    """
    type_counts: Counter[str] = Counter()
    individual: dict[str, Any] = {}
    for name, result in errored.items():
        payload = result.data
        assert isinstance(payload, LLMPayload)
        if payload.judge_error_type:
            type_counts[payload.judge_error_type] += 1
        individual[name] = payload.model_dump(mode="json")
    headline_type = sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if type_counts else "other"
    return LLMPayload(
        overall_pass=False,
        dimensions={},
        judge_errored=True,
        judge_error_type=headline_type,  # type: ignore[arg-type]
        individual_verdicts=individual,
    )


# Rank used for pessimistic tie-breaking and for unanimous/any strategies.
# Cross-scale comparison is conceptually a config bug (two judges on the
# same dimension should agree on its kind), but the rank still has to
# total-order every ``ScoreLevel`` so a stray mixed vote degrades safely
# rather than crashing. ``INCONCLUSIVE`` ranks below every real verdict
# so any judge returning it pulls the merged verdict downward, matching
# the headline pass-rate semantics (inconclusive counts as failure).
_LEVEL_RANK = {
    ScoreLevel.INCONCLUSIVE: -1,
    ScoreLevel.FAIL: 0,
    ScoreLevel.LOW: 0,
    ScoreLevel.MEDIUM: 1,
    ScoreLevel.HIGH: 2,
    ScoreLevel.PASS: 2,
}


def _majority_vote_level(votes: list[ScoreLevel]) -> ScoreLevel:
    """Pick the majority ScoreLevel. Ties break pessimistically (lower wins)."""
    counts = Counter(votes)
    max_count = max(counts.values())
    candidates = [level for level, count in counts.items() if count == max_count]
    return min(candidates, key=lambda lv: _LEVEL_RANK[lv])


def _majority_vote_bool(votes: list[bool]) -> bool:
    """Majority vote on booleans. Ties break pessimistically (False wins)."""
    true_count = sum(votes)
    return true_count > len(votes) / 2


class ConsensusScorer(BaseScorer):
    """Wraps N LLMScorers and majority-votes their verdicts."""

    def __init__(
        self,
        judges: list[LLMScorer],
        strategy: str = "majority",
    ):
        if len(judges) < 2:
            from belt.errors import ConfigError

            raise ConfigError("ConsensusScorer requires at least 2 judges")
        if strategy not in CONSENSUS_STRATEGIES:
            from belt.errors import ConfigError

            valid = ", ".join(sorted(CONSENSUS_STRATEGIES))
            raise ConfigError(f"Unknown consensus strategy '{strategy}'. Valid: {valid}")

        self._judges = judges
        self._strategy = strategy
        self._dimension_warning_emitted = False

    @property
    def name(self) -> str:
        return "llm"

    @property
    def judges(self) -> list[LLMScorer]:
        return self._judges

    @property
    def consensus_strategy(self) -> str:
        return self._strategy

    def is_available(self) -> bool:
        return all(j.is_available() for j in self._judges)

    def set_on_event(self, callback: Callable[[ScoreEvent], None] | None) -> None:
        """Propagate event callback to each judge, injecting judge name."""
        for judge in self._judges:
            if callback is None:
                judge.on_event = None
            else:
                name = judge.judge_name

                def _wrap(cb: Callable[[ScoreEvent], None], jn: str) -> Callable[[ScoreEvent], None]:
                    def _inner(event: ScoreEvent) -> None:
                        event.judge = jn
                        cb(event)

                    return _inner

                judge.on_event = _wrap(callback, name)

    def emit_dimension_warnings(self) -> None:
        """Check dimension overlap across judges and warn if not identical.

        Call once at startup (not per-scenario) to avoid log spam.
        """
        if self._dimension_warning_emitted:
            return
        self._dimension_warning_emitted = True

        dim_sets = []
        for j in self._judges:
            dim_sets.append(set(j.strategy.dimension_names))

        if not dim_sets:
            return

        all_same = all(ds == dim_sets[0] for ds in dim_sets)
        if all_same:
            return

        shared = set.intersection(*dim_sets)
        all_dims = set.union(*dim_sets)
        unique = all_dims - shared

        logger.warning(
            "Consensus judges have different dimensions - majority vote applies "
            "only to shared dimensions ({}). Judge-specific dimensions pass through "
            "without voting: {}",
            ", ".join(sorted(shared)) or "none",
            ", ".join(sorted(unique)) or "none",
        )

        for j in self._judges:
            judge_dims = set(j.strategy.dimension_names)
            j_unique = judge_dims - shared
            if j_unique:
                logger.warning(
                    "  Judge '{}': unique dimensions [{}]",
                    j.judge_name,
                    ", ".join(sorted(j_unique)),
                )

    def score(
        self,
        scenario: Scenario,
        turn_outputs: list[TurnOutput],
    ) -> ScorerResult | None:
        if not turn_outputs:
            return None

        all_results: dict[str, ScorerResult] = {}
        for judge in self._judges:
            result = judge.score(scenario, turn_outputs)
            if result is not None:
                all_results[judge.judge_name] = result

        if not all_results:
            return None

        # Partition sub-judges into "produced a real verdict" vs "errored on
        # infra". A single sub-judge dropping out is recoverable - the
        # consensus proceeds with the survivors and records the error so a
        # consistently flaky provider is still visible. Only when every
        # sub-judge errored does the merged payload itself carry
        # judge_errored=True, so the aggregator partitions exactly the
        # scenarios where no judge actually voted.
        real_verdicts: dict[str, ScorerResult] = {}
        errored: dict[str, ScorerResult] = {}
        for name, result in all_results.items():
            payload = result.data
            if isinstance(payload, LLMPayload) and payload.judge_errored:
                errored[name] = result
            else:
                real_verdicts[name] = result

        if not real_verdicts:
            return ScorerResult(passed=False, data=_all_judges_errored_payload(errored))

        if len(real_verdicts) == 1 and not errored:
            return next(iter(real_verdicts.values()))

        return self._merge_verdicts(real_verdicts, errored)

    def _merge_verdicts(
        self,
        verdicts: dict[str, ScorerResult],
        errored: dict[str, ScorerResult] | None = None,
    ) -> ScorerResult:
        # Each judge's data is already a typed LLMPayload; reach into
        # ``payload.dimensions`` directly rather than walking ``data`` as a
        # raw dict.
        payloads: dict[str, LLMPayload] = {}
        for name, result in verdicts.items():
            assert isinstance(
                result.data, LLMPayload
            ), f"Consensus expects LLMPayload from sub-judge '{name}', got {type(result.data).__name__}"
            payloads[name] = result.data

        dim_sets = [set(p.dimensions.keys()) for p in payloads.values()]
        all_dims = set.union(*dim_sets) if dim_sets else set()
        shared_dims = set.intersection(*dim_sets) if dim_sets else set()

        merged_dimensions: dict[str, LLMDimensionVerdict] = {}
        disagreements: list[dict[str, Any]] = []

        for dim in sorted(all_dims):
            votes: list[tuple[str, ScoreLevel, str]] = []
            for judge_name, payload in payloads.items():
                verdict = payload.dimensions.get(dim)
                if verdict is None:
                    continue
                try:
                    level = ScoreLevel(verdict.score)
                except ValueError:
                    continue
                votes.append((judge_name, level, verdict.reasoning))

            if not votes:
                continue

            if dim in shared_dims and len(votes) >= 2:
                levels = [v[1] for v in votes]
                winner = self._apply_strategy(levels)
                vote_counts = dict(Counter(lv.value for lv in levels))
                unanimous = len(set(levels)) == 1

                winning_reasonings = [v[2] for v in votes if v[1] == winner]
                reasoning = winning_reasonings[0] if winning_reasonings else votes[0][2]

                if not unanimous:
                    vote_summary = ", ".join(f"{v[0]}={v[1].value}" for v in votes)
                    reasoning = f"[Consensus {self._strategy} ({vote_summary})] {reasoning}"
                    disagreements.append({"dimension": dim, "votes": vote_counts})

                merged_dimensions[dim] = LLMDimensionVerdict(score=winner.value, reasoning=reasoning)
            else:
                merged_dimensions[dim] = LLMDimensionVerdict(score=votes[0][1].value, reasoning=votes[0][2])

        pass_votes = [r.passed for r in verdicts.values()]
        overall = self._apply_strategy_bool(pass_votes)

        total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for payload in payloads.values():
            if payload.usage is None:
                continue
            usage_dict = payload.usage.model_dump(mode="json", exclude_none=True)
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                try:
                    total_usage[field] += int(usage_dict.get(field, 0) or 0)
                except (TypeError, ValueError):
                    pass
        merged_usage = UsageStats(**total_usage) if total_usage["total_tokens"] > 0 else None

        # Record errored sub-judges alongside the real ones so reviewers can
        # see when a single provider was flaky even though consensus still
        # produced a valid verdict from the survivors.
        individual: dict[str, Any] = {
            judge_name: result.data.model_dump(mode="json") for judge_name, result in verdicts.items()
        }
        if errored:
            for judge_name, result in errored.items():
                individual[judge_name] = result.data.model_dump(mode="json")

        merged_payload = LLMPayload(
            overall_pass=overall,
            dimensions=merged_dimensions,
            usage=merged_usage,
            consensus_meta=ConsensusMeta(
                strategy=self._strategy,
                judges=list(verdicts.keys()),
                shared_dimensions=sorted(shared_dims),
                disagreements=disagreements,
            ),
            individual_verdicts=individual,
        )

        return ScorerResult(passed=overall, data=merged_payload)

    def _apply_strategy(self, levels: list[ScoreLevel]) -> ScoreLevel:
        if self._strategy == "majority":
            return _majority_vote_level(levels)
        elif self._strategy == "unanimous":
            return min(levels, key=lambda lv: _LEVEL_RANK[lv])
        elif self._strategy == "any":
            return max(levels, key=lambda lv: _LEVEL_RANK[lv])
        return _majority_vote_level(levels)

    def _apply_strategy_bool(self, votes: list[bool]) -> bool:
        if self._strategy == "majority":
            return _majority_vote_bool(votes)
        elif self._strategy == "unanimous":
            return all(votes)
        elif self._strategy == "any":
            return any(votes)
        return _majority_vote_bool(votes)
