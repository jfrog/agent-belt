# (c) JFrog Ltd. (2026)

"""Rule-based scorer: coordinates check categories per turn.

Completely agent-agnostic - reads only TurnOutput fields populated by the
agent's fetch_results(). No agent-specific parsing (emojis, thread state,
CLI format) happens here.
"""

from __future__ import annotations

from loguru import logger

from belt.entities import Scenario, ScorerResult, StateExpectation, TurnExpectation, TurnOutput
from belt.scenario import VerifySpec
from belt.scorer.base import BaseScorer
from belt.scorer.payloads import CheckEntry, RulesPayload
from belt.scorer.rules.efficiency import check_cost, check_efficiency
from belt.scorer.rules.execution import check_execution
from belt.scorer.rules.file_diff import check_file_diff, has_file_diff_checks
from belt.scorer.rules.performance import check_performance
from belt.scorer.rules.response import check_response
from belt.scorer.rules.state import check_state, has_state_checks
from belt.scorer.rules.trajectory import check_trajectory
from belt.scorer.rules.verify import check_verify, has_verify


class RuleBasedScorer(BaseScorer):
    @property
    def name(self) -> str:
        return "rules"

    def is_available(self) -> bool:
        return True

    def score(
        self,
        scenario: Scenario,
        turn_outputs: list[TurnOutput],
    ) -> ScorerResult | None:
        all_checks: list[CheckEntry] = []
        for i, turn in enumerate(scenario.turns):
            to = turn_outputs[i] if i < len(turn_outputs) else TurnOutput(raw_cli="")
            se = turn.state_expect if has_state_checks(turn.state_expect) else None
            checks = _check_turn(i, to, turn.expect, state_expect=se, verify=turn.verify)
            all_checks.extend(checks)
            failed = [c for c in checks if not c.passed]
            if failed:
                logger.debug("Turn {} rule failures: {}", i, [c.check for c in failed])

        # Per-scenario (end-of-conversation) verify: a turn-less check that
        # reads the result recorded by the runner on the final turn. Emitted
        # even when no turns ran (skipped) so the dimension is always present.
        if scenario.verify is not None:
            last_result = turn_outputs[-1].scenario_verify_result if turn_outputs else None
            all_checks.extend(check_verify(scenario.verify, last_result, turn_idx=None))

        payload = RulesPayload(checks=all_checks, passed=all(c.passed for c in all_checks))
        return ScorerResult(passed=payload.passed, data=payload)


def _check_turn(
    turn_idx: int,
    output: TurnOutput,
    expect: TurnExpectation,
    state_expect: StateExpectation | None = None,
    verify: VerifySpec | None = None,
) -> list[CheckEntry]:
    results: list[CheckEntry] = []
    results.extend(check_execution(turn_idx, output, expect))
    results.extend(check_trajectory(turn_idx, output, expect))
    results.extend(check_response(turn_idx, output, expect))
    results.extend(check_efficiency(turn_idx, output, expect))
    results.extend(check_performance(turn_idx, output, expect))
    results.extend(check_cost(turn_idx, output, expect))
    if has_file_diff_checks(expect):
        results.extend(check_file_diff(turn_idx, output, expect))
    if state_expect is not None:
        results.extend(check_state(turn_idx, output, state_expect))
    if has_verify(verify):
        results.extend(check_verify(verify, output.verify_result, turn_idx=turn_idx))
    return results
