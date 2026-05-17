# (c) JFrog Ltd. (2026)

"""Efficiency and cost checks - LLM turns, tool calls, cost budgets."""

from __future__ import annotations

from belt.entities import TurnExpectation, TurnOutput
from belt.scorer.payloads import CheckEntry


def check_efficiency(ti: int, output: TurnOutput, expect: TurnExpectation) -> list[CheckEntry]:
    results: list[CheckEntry] = []

    if expect.max_llm_turns is not None:
        count = output.llm_turn_count if output.llm_turn_count is not None else 0
        results.append(
            CheckEntry(
                dimension="efficiency",
                check=f"max_llm_turns(<={expect.max_llm_turns})",
                passed=count <= expect.max_llm_turns,
                details=f"counted {count}",
                turn_idx=ti,
            )
        )

    if expect.max_tool_calls is not None:
        count = len(output.tool_calls)
        results.append(
            CheckEntry(
                dimension="efficiency",
                check=f"max_tool_calls(<={expect.max_tool_calls})",
                passed=count <= expect.max_tool_calls,
                details=f"counted {count}",
                turn_idx=ti,
            )
        )

    return results


def check_cost(ti: int, output: TurnOutput, expect: TurnExpectation) -> list[CheckEntry]:
    results: list[CheckEntry] = []

    if expect.max_cost_usd is not None:
        actual = output.cost_usd
        passed = actual is not None and actual <= expect.max_cost_usd
        results.append(
            CheckEntry(
                dimension="cost",
                check=f"max_cost_usd(<=${expect.max_cost_usd})",
                passed=passed,
                details=f"actual: ${actual:.4f}" if actual is not None else "cost not reported",
                turn_idx=ti,
            )
        )

    return results
