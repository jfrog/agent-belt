# (c) JFrog Ltd. (2026)

"""Threshold parsing, validation, and per-dimension enforcement.

A *threshold* is a budget on the percentage of scenarios that may fail in a
given mode/dimension before the run as a whole is considered failed.  The
flags ``--threshold rules/execution:0`` and ``--threshold llm/execution:5``
arrive as CLI strings; this module:

1. Parses them into ``(mode, dimension, percent)`` tuples.
2. Validates known modes and (when LLM dimensions are dynamically
   discoverable from scores) flags unknown dimension names early.
3. Counts failures per dimension and compares against the budget via
   ``ThresholdEnforcer`` - which lazily computes rules and LLM failures so
   call sites that only need one stay cheap.

Library-only.  ``commands/aggregate.py`` is the sole CLI consumer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from belt.entities import ScenarioScore
from belt.scorer.entities import ALL_VERDICT_TOKENS, DEFAULT_FAIL_LEVELS
from belt.scorer.payloads import RulesPayload, iter_llm_payloads, iter_llm_verdicts

VALID_LLM_FAIL_ON: set[str] = set(ALL_VERDICT_TOKENS)
"""Tokens accepted by ``--llm-fail-on``. Derived from
:data:`belt.scorer.entities.ALL_VERDICT_TOKENS` so adding a new
verdict to :class:`ScoreLevel` propagates without a separate edit
here (Design Principle 9)."""


def parse_threshold(raw: str) -> tuple[str, str, float]:
    """Parse 'mode/dimension:percent' → (mode, dimension, percent)."""
    if ":" not in raw or "/" not in raw.split(":")[0]:
        raise argparse.ArgumentTypeError(
            f"Invalid threshold '{raw}' - expected format: mode/dimension:percent (e.g. rules/execution:0)"
        )
    path, pct_str = raw.rsplit(":", 1)
    mode, dimension = path.split("/", 1)
    try:
        pct = float(pct_str)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid percentage '{pct_str}' in threshold '{raw}'") from e
    if not 0 <= pct <= 100:
        raise argparse.ArgumentTypeError(f"Percentage must be 0-100, got {pct} in threshold '{raw}'")
    return mode, dimension, pct


def validate_thresholds(
    thresholds: dict[str, dict[str, float]],
    known_llm_dims: set[str] | None = None,
) -> None:
    """Validate threshold modes; rules dims are validated at enforcement time."""
    errors = []
    for mode, dims in thresholds.items():
        if mode == "rules":
            pass
        elif mode == "llm":
            if known_llm_dims is not None:
                for dim in dims:
                    if dim not in known_llm_dims:
                        errors.append(f"llm/{dim}: unknown dimension (valid: {', '.join(sorted(known_llm_dims))})")
        else:
            errors.append(f"Unknown mode '{mode}' (valid: rules, llm)")
    if errors:
        from belt.errors import ConfigError

        raise ConfigError("; ".join(errors))


def discover_llm_dimensions(scores: list[ScenarioScore]) -> list[str]:
    """Discover LLM dimension names from actual score data.

    Walks every LLM-shaped payload (``LLMPayload`` and
    ``PerTurnLLMPayload``, including non-default scorer names from
    ``--scorer-config``) so a per-turn judge or a renamed multi-judge
    setup contributes its dimensions to threshold validation.
    """
    dims: set[str] = set()
    for s in scores:
        for _name, payload in iter_llm_payloads(s):
            for dim, _score_token, _reasoning in iter_llm_verdicts(payload):
                dims.add(dim)
    return sorted(dims)


@dataclass
class ThresholdCheck:
    """Result of a single threshold enforcement check."""

    dimension: str
    actual_pct: float
    max_pct: float
    passed: bool

    def display_line(self) -> str:
        icon = "✅" if self.passed else "❌"
        return f"  {self.dimension}: {self.actual_pct:.1f}% failed (max {self.max_pct:.0f}%) {icon}"


class ThresholdEnforcer:
    """Counts failures by dimension and enforces per-dimension failure budgets."""

    def __init__(self, scores: list[ScenarioScore], llm_fail_on: set[str] | None = None):
        self._scores = scores
        # Default fail set covers every "did not pass" verdict across
        # both ternary and binary scales, plus inconclusive. Authors
        # who want to treat inconclusive as informational only can
        # override with --llm-fail-on low,fail (omitting inconclusive).
        self._llm_fail_on = llm_fail_on or set(DEFAULT_FAIL_LEVELS)
        self._rules_failures: dict[str, tuple[int, int]] | None = None
        self._llm_failures: dict[str, tuple[int, int]] | None = None

    @property
    def rules_failures(self) -> dict[str, tuple[int, int]]:
        """(failed_scenarios, total_scenarios) per rules dimension."""
        if self._rules_failures is None:
            self._rules_failures = self._count_rules_failures()
        return self._rules_failures

    @property
    def llm_failures(self) -> dict[str, tuple[int, int]]:
        """(failed_scenarios, total_scenarios) per LLM dimension."""
        if self._llm_failures is None:
            self._llm_failures = self._count_llm_failures()
        return self._llm_failures

    def _count_rules_failures(self) -> dict[str, tuple[int, int]]:
        dim_scenarios: dict[str, dict[str, bool]] = {}
        for s in self._scores:
            rules = s.scores.get("rules")
            if not isinstance(rules, RulesPayload):
                continue
            scenario_key = f"{s.group}/{s.scenario_name}"
            for check in rules.checks:
                dim = check.dimension or "unknown"
                dim_scenarios.setdefault(dim, {})
                if scenario_key not in dim_scenarios[dim]:
                    dim_scenarios[dim][scenario_key] = True
                if check.passed is False:
                    dim_scenarios[dim][scenario_key] = False

        result: dict[str, tuple[int, int]] = {}
        for dim, scenarios in dim_scenarios.items():
            total = len(scenarios)
            failed = sum(1 for passed in scenarios.values() if not passed)
            result[dim] = (failed, total)
        return result

    def _count_llm_failures(self) -> dict[str, tuple[int, int]]:
        # Walk every LLM-shaped payload per scenario so multi-judge
        # (judge_a / judge_b) and per-turn (PerTurnLLMPayload) verdicts
        # all contribute to threshold gating. ``iter_llm_verdicts``
        # applies the per-turn worst-of-turns rollup so a single
        # failing turn marks the whole scenario as failing the dim.
        dimensions = discover_llm_dimensions(self._scores)
        result: dict[str, tuple[int, int]] = {}
        for dim in dimensions:
            total = 0
            failed = 0
            for s in self._scores:
                verdict_for_dim: str | None = None
                for _name, payload in iter_llm_payloads(s):
                    for d, score_token, _reasoning in iter_llm_verdicts(payload):
                        if d != dim:
                            continue
                        # Take the worst verdict across all judges /
                        # turns for this scenario+dim so one passing
                        # judge cannot mask a failing one.
                        if verdict_for_dim is None or (
                            score_token in self._llm_fail_on and verdict_for_dim not in self._llm_fail_on
                        ):
                            verdict_for_dim = score_token
                if verdict_for_dim is None:
                    continue
                total += 1
                if verdict_for_dim in self._llm_fail_on:
                    failed += 1
            if total > 0:
                result[dim] = (failed, total)
        return result

    def enforce(self, thresholds: dict[str, dict[str, float]]) -> tuple[list[str], bool, list[ThresholdCheck]]:
        """Check thresholds, return (display_lines, all_passed, structured_checks)."""
        checks: list[ThresholdCheck] = []
        all_passed = True

        for mode, dims in sorted(thresholds.items()):
            failures = self.rules_failures if mode == "rules" else self.llm_failures
            for dim, max_pct in sorted(dims.items()):
                failed, total = failures.get(dim, (0, 0))
                actual_pct = (failed / total * 100) if total > 0 else 0.0
                passed = actual_pct <= max_pct
                checks.append(ThresholdCheck(f"{mode}/{dim}", actual_pct, max_pct, passed))
                if not passed:
                    all_passed = False

        lines = [c.display_line() for c in checks]
        return lines, all_passed, checks


# Module-level convenience functions for stateless callers - delegate to
# ThresholdEnforcer. These are part of the public aggregator API; pick
# whichever entry point matches the call site (class for batched access,
# free function for a single counted check).


def count_rules_failures(scores: list[ScenarioScore]) -> dict[str, tuple[int, int]]:
    return ThresholdEnforcer(scores).rules_failures


def count_llm_failures(scores: list[ScenarioScore], fail_on: set[str]) -> dict[str, tuple[int, int]]:
    return ThresholdEnforcer(scores, llm_fail_on=fail_on).llm_failures


def enforce_thresholds(
    thresholds: dict[str, dict[str, float]],
    rules_failures: dict[str, tuple[int, int]],
    llm_failures: dict[str, tuple[int, int]],
) -> tuple[list[str], bool, list[ThresholdCheck]]:
    enforcer = ThresholdEnforcer([])
    enforcer._rules_failures = rules_failures
    enforcer._llm_failures = llm_failures
    return enforcer.enforce(thresholds)
