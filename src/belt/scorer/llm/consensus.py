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
from belt.scorer.payloads import (
    ConsensusMeta,
    LLMDimensionVerdict,
    LLMPayload,
    PerTurnLLMPayload,
    TurnVerdict,
    UsageStats,
)

CONSENSUS_STRATEGIES = {"majority", "unanimous", "any"}


def _all_judges_errored_payload(
    errored: dict[str, ScorerResult],
    *,
    resolution: str = "scenario",
    n_turns: int = 0,
) -> LLMPayload | PerTurnLLMPayload:
    """Build a merged ``judge_errored=True`` payload when every sub-judge errored.

    Picks the most-common error type across sub-judges as the headline
    ``judge_error_type`` (ties break alphabetically so the choice is
    deterministic across runs). Per-judge error types are preserved in
    ``individual_verdicts`` so a postmortem reader can see exactly which
    provider produced which token.

    *resolution* controls the merged payload's shape so per-turn
    consensus runs keep emitting the ``per_turn_llm.v1`` schema
    discriminator even when no sub-judge produced a verdict (a
    silent fall-back to ``llm.v1`` would break consumers that
    dispatch on ``isinstance`` over the per-turn payload).
    """
    type_counts: Counter[str] = Counter()
    individual: dict[str, Any] = {}
    for name, result in errored.items():
        payload = result.data
        assert isinstance(
            payload, (LLMPayload, PerTurnLLMPayload)
        ), f"Consensus expects an LLM payload from sub-judge '{name}', got {type(payload).__name__}"
        if payload.judge_error_type:
            type_counts[payload.judge_error_type] += 1
        individual[name] = payload.model_dump(mode="json")
    headline_type = sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if type_counts else "other"
    if resolution == "turn":
        return PerTurnLLMPayload(
            overall_pass=False,
            turns=[
                TurnVerdict(
                    turn_idx=i,
                    dimensions={},
                    judge_errored=True,
                    judge_error_type=headline_type,  # type: ignore[arg-type]
                )
                for i in range(n_turns)
            ],
            judge_errored=True,
            judge_error_type=headline_type,  # type: ignore[arg-type]
            individual_verdicts=individual,
        )
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

        # All sub-judges in a consensus block MUST agree on resolution.
        # Per-turn and scenario-level produce structurally different
        # payloads (`PerTurnLLMPayload` vs `LLMPayload`), so merging them
        # is undefined. Fail fast at build-time with the offending
        # configuration in the message so the author can fix YAML
        # before any judge call happens.
        resolutions = {j.resolution for j in judges}
        if len(resolutions) > 1:
            from belt.errors import ConfigError

            per_judge = ", ".join(f"{j.judge_name}={j.resolution}" for j in judges)
            raise ConfigError(
                f"ConsensusScorer requires all judges to share resolution; got mixed "
                f"resolutions [{per_judge}]. Split into separate scorer-config files or "
                f"set ``resolution`` uniformly across the consensus block."
            )

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

    @property
    def resolution(self) -> str:
        """Uniform resolution across sub-judges (enforced at ``__init__``)."""
        return self._judges[0].resolution

    @property
    def evidence_scope(self) -> str:
        """First sub-judge's evidence scope (consumers usually only need resolution)."""
        return self._judges[0].evidence_scope

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

        For per-turn judges the warning is necessarily approximate: a
        ``TurnJudgeOverride.dimensions`` may extend or replace the
        judge's configured dimensions only for specific turns. We
        compare configured dimensions here as a startup heuristic; the
        runtime per-turn merge in :meth:`_merge_per_turn_verdicts`
        handles whichever dimensions actually appear per turn.
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

        per_turn_note = " (per-turn overrides may further diverge per turn)" if self.resolution == "turn" else ""
        logger.warning(
            "Consensus judges have different dimensions{} - majority vote applies "
            "only to shared dimensions ({}). Judge-specific dimensions pass through "
            "without voting: {}",
            per_turn_note,
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

    @staticmethod
    def _subjudge_usable(payload: object) -> bool:
        """Whether a sub-judge result still contributes to the merge.

        - Scenario-level (:class:`LLMPayload`): usable iff the judge
          produced a verdict (``judge_errored=False``).
        - Per-turn (:class:`PerTurnLLMPayload`): usable iff at least one
          turn carries a real verdict, even when other turns errored.
          This preserves per-turn graceful degradation - a partially
          errored judge's good turns must still reach
          :meth:`_merge_per_turn_verdicts`, where per-turn coverage is
          decided. Using the payload-level ``judge_errored`` (the OR
          across turns) here would drop the judge wholesale.
        - Anything else: usable (defensive default; should not occur).
        """
        if isinstance(payload, PerTurnLLMPayload):
            return any(t.dimensions for t in payload.turns)
        if isinstance(payload, LLMPayload):
            return not payload.judge_errored
        return True

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

        # Partition sub-judges into "still contributes to the merge" vs
        # "no usable verdict". A single sub-judge dropping out is
        # recoverable - the consensus proceeds with the survivors and
        # records the error so a consistently flaky provider is still
        # visible. Only when no sub-judge contributes any usable verdict
        # does the merged payload itself carry judge_errored=True, so the
        # aggregator partitions exactly the scenarios where no judge
        # actually voted.
        #
        # For per-turn judging the partition is by usable *turns* rather
        # than the payload-level ``judge_errored`` flag (which is the OR
        # across turns): a judge that errored on one turn but voted on
        # the rest must still reach :meth:`_merge_per_turn_verdicts`, so
        # two judges that each error on a *different* turn still cover the
        # scenario between them. Dropping them wholesale would discard
        # good per-turn verdicts and regress per-turn graceful
        # degradation.
        real_verdicts: dict[str, ScorerResult] = {}
        errored: dict[str, ScorerResult] = {}
        for name, result in all_results.items():
            if self._subjudge_usable(result.data):
                real_verdicts[name] = result
            else:
                errored[name] = result

        if not real_verdicts:
            return ScorerResult(
                passed=False,
                data=_all_judges_errored_payload(errored, resolution=self.resolution, n_turns=len(turn_outputs)),
            )

        if len(real_verdicts) == 1 and not errored:
            return next(iter(real_verdicts.values()))

        if self.resolution == "turn":
            return self._merge_per_turn_verdicts(real_verdicts, errored)
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

    def _merge_per_turn_verdicts(
        self,
        verdicts: dict[str, ScorerResult],
        errored: dict[str, ScorerResult] | None = None,
    ) -> ScorerResult:
        """Per-turn analogue of :meth:`_merge_verdicts`.

        For each turn index that appears in any sub-judge payload:

        - Build a :class:`TurnVerdict` whose dimensions are the merged
          per-dimension consensus of every sub-judge that voted on
          that ``(turn, dimension)``. Shared dimensions use
          :meth:`_apply_strategy`; non-shared dimensions pass through
          from the first voting judge (same heuristic as the
          scenario-level merge).
        - Per-turn ``judge_errored`` is set when every sub-judge
          errored on that turn; partial errors degrade to the
          surviving judges.

        Payload-level ``judge_errored`` is the OR over merged turns: any
        turn left uncovered (every sub-judge errored on it) makes the
        merged payload ``judge_errored=True`` so the pipeline forces a
        fail and the aggregator charges it to ``env_failed_judge`` rather
        than agent task quality. ``overall_pass`` is the AND across turns
        AND merged dimensions, and is ``False`` whenever any turn is
        uncovered. Empty merged turns (every sub-judge skipped) do not
        vacuously pass: the guard records ``judge_errored=True`` with
        ``judge_error_type="all_turns_skipped"``.
        """
        payloads: dict[str, PerTurnLLMPayload] = {}
        for name, result in verdicts.items():
            assert isinstance(result.data, PerTurnLLMPayload), (
                f"Per-turn consensus expects PerTurnLLMPayload from sub-judge "
                f"'{name}', got {type(result.data).__name__}"
            )
            payloads[name] = result.data

        # Union of turn indices voted on by any sub-judge - keeps the
        # merged payload aligned to scenario turn numbering even if one
        # sub-judge skipped a turn the others voted on.
        all_turn_indices: set[int] = set()
        for payload in payloads.values():
            for t in payload.turns:
                all_turn_indices.add(t.turn_idx)

        merged_turns: list[TurnVerdict] = []
        disagreements: list[dict[str, Any]] = []
        shared_dim_union: set[str] = set()

        for turn_idx in sorted(all_turn_indices):
            # Collect per-judge entries for this turn.
            per_judge_turn: dict[str, TurnVerdict] = {}
            for name, payload in payloads.items():
                for t in payload.turns:
                    if t.turn_idx == turn_idx:
                        per_judge_turn[name] = t
                        break
            if not per_judge_turn:
                continue

            # If every judge that touched this turn errored on it, mark
            # the merged turn as errored.
            voting = {n: t for n, t in per_judge_turn.items() if not t.judge_errored and t.dimensions}
            if not voting:
                # Pick the most common error type across judges that errored on this turn.
                err_types: Counter[str] = Counter()
                for t in per_judge_turn.values():
                    if t.judge_errored and t.judge_error_type:
                        err_types[t.judge_error_type] += 1
                headline = sorted(err_types.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if err_types else None
                merged_turns.append(
                    TurnVerdict(
                        turn_idx=turn_idx,
                        dimensions={},
                        judge_errored=bool(err_types),
                        judge_error_type=headline,  # type: ignore[arg-type]
                    )
                )
                continue

            # Compute per-dimension merged verdicts within this turn.
            dim_sets = [set(t.dimensions.keys()) for t in voting.values()]
            all_dims = set.union(*dim_sets) if dim_sets else set()
            shared_dims = set.intersection(*dim_sets) if dim_sets else set()
            shared_dim_union.update(shared_dims)

            merged_dims: dict[str, LLMDimensionVerdict] = {}
            for dim in sorted(all_dims):
                votes: list[tuple[str, ScoreLevel, str]] = []
                for judge_name, turn in voting.items():
                    verdict = turn.dimensions.get(dim)
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
                        disagreements.append({"turn_idx": turn_idx, "dimension": dim, "votes": vote_counts})
                    merged_dims[dim] = LLMDimensionVerdict(score=winner.value, reasoning=reasoning)
                else:
                    merged_dims[dim] = LLMDimensionVerdict(score=votes[0][1].value, reasoning=votes[0][2])
            merged_turns.append(TurnVerdict(turn_idx=turn_idx, dimensions=merged_dims))

        # Aggregate token usage across sub-judges (each payload already
        # summed its own per-turn usage).
        total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for payload in payloads.values():
            if payload.usage is None:
                continue
            usage_dict = payload.usage.model_dump(mode="json", exclude_none=True)
            for fld in ("prompt_tokens", "completion_tokens", "total_tokens"):
                try:
                    total_usage[fld] += int(usage_dict.get(fld, 0) or 0)
                except (TypeError, ValueError):
                    pass
        merged_usage = UsageStats(**total_usage) if total_usage["total_tokens"] > 0 else None

        individual: dict[str, Any] = {n: r.data.model_dump(mode="json") for n, r in verdicts.items()}
        if errored:
            for n, r in errored.items():
                individual[n] = r.data.model_dump(mode="json")

        from belt.scorer.entities import DEFAULT_FAIL_LEVELS

        # A merged turn is errored only when EVERY sub-judge that touched
        # it errored (the per-turn ``voting`` filter above) - i.e. no
        # judge covered that turn. Propagate that up: if any merged turn
        # is uncovered the consensus verdict is incomplete, so the
        # payload is judge_errored (routes to env_failed_judge) and
        # cannot pass - identical to the single-judge per-turn path and
        # the scenario-level guarantee. ``all_turns_skipped`` is the
        # headline only for the pure all-skip degenerate case (no error,
        # no vote).
        errored_turns = [t for t in merged_turns if t.judge_errored]
        any_voted = any(t.dimensions for t in merged_turns)
        judge_errored = bool(errored_turns) or not any_voted

        headline_error: str | None = None
        if errored_turns:
            etypes: Counter[str] = Counter(t.judge_error_type for t in errored_turns if t.judge_error_type)
            headline_error = (
                sorted(etypes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if etypes else "all_turns_skipped"
            )
        elif not any_voted:
            headline_error = "all_turns_skipped"

        has_fail = any(
            v.score in DEFAULT_FAIL_LEVELS
            for turn in merged_turns
            if not turn.judge_errored
            for v in turn.dimensions.values()
        )
        overall_pass = any_voted and not judge_errored and not has_fail

        return ScorerResult(
            passed=overall_pass,
            data=PerTurnLLMPayload(
                overall_pass=overall_pass,
                turns=merged_turns,
                usage=merged_usage,
                consensus_meta=ConsensusMeta(
                    strategy=self._strategy,
                    judges=list(verdicts.keys()),
                    shared_dimensions=sorted(shared_dim_union),
                    disagreements=disagreements,
                ),
                individual_verdicts=individual,
                judge_errored=judge_errored,
                judge_error_type=headline_error,  # type: ignore[arg-type]
            ),
        )

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
