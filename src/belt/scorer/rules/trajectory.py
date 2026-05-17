# (c) JFrog Ltd. (2026)

"""Trajectory checks - tool invocation, ordering, thinking."""

from __future__ import annotations

from belt.entities import TurnExpectation, TurnOutput
from belt.scorer.payloads import CheckEntry
from belt.scorer.rules.helpers import (
    args_match,
    is_subsequence,
    result_contains,
    result_matches,
    skill_invoked,
    tool_name_in_cli,
)


def check_trajectory(ti: int, output: TurnOutput, expect: TurnExpectation) -> list[CheckEntry]:
    results: list[CheckEntry] = []
    tool_names = {tc.name for tc in output.tool_calls}

    for tool in expect.tools_invoked:
        in_tool_calls = tool in tool_names
        in_cli = tool_name_in_cli(tool, output.raw_cli) if not in_tool_calls else False
        found = in_tool_calls or in_cli
        results.append(
            CheckEntry(
                dimension="trajectory",
                check=f"tool_invoked({tool})",
                passed=found,
                details="via tool_calls" if in_tool_calls else ("via cli fallback" if in_cli else ""),
                turn_idx=ti,
            )
        )

    for alternatives in expect.tools_invoked_any:
        matched = [t for t in alternatives if t in tool_names or tool_name_in_cli(t, output.raw_cli)]
        label = "|".join(alternatives)
        results.append(
            CheckEntry(
                dimension="trajectory",
                check=f"tool_invoked_any({label})",
                passed=bool(matched),
                details=f"matched: {matched[0]}" if matched else "",
                turn_idx=ti,
            )
        )

    if expect.tools_invoked_in_order:
        seq = output.tool_sequence or [tc.name for tc in output.tool_calls]
        passed = is_subsequence(expect.tools_invoked_in_order, seq)
        label = "→".join(expect.tools_invoked_in_order)
        results.append(
            CheckEntry(
                dimension="trajectory",
                check=f"tools_invoked_in_order({label})",
                passed=passed,
                details=f"actual sequence: {'→'.join(seq[:20])}" if not passed else "",
                turn_idx=ti,
            )
        )

    if expect.only_used_tools:
        allowed = set(expect.only_used_tools)
        violations = sorted(tool_names - allowed)
        results.append(
            CheckEntry(
                dimension="trajectory",
                check=f"only_used_tools({','.join(sorted(allowed))})",
                passed=not violations,
                details=f"violations: {','.join(violations)}" if violations else "",
                turn_idx=ti,
            )
        )

    if expect.forbidden_tools:
        forbidden = set(expect.forbidden_tools)
        used_forbidden = sorted(tool_names & forbidden)
        results.append(
            CheckEntry(
                dimension="trajectory",
                check=f"forbidden_tools({','.join(sorted(forbidden))})",
                passed=not used_forbidden,
                details=f"used: {','.join(used_forbidden)}" if used_forbidden else "",
                turn_idx=ti,
            )
        )

    for tool_name, expected_args in expect.tool_args_contain.items():
        matching_calls = [tc for tc in output.tool_calls if tc.name == tool_name]
        if not matching_calls:
            results.append(
                CheckEntry(
                    dimension="trajectory",
                    check=f"tool_args_contain({tool_name})",
                    passed=False,
                    details=f"tool {tool_name} not invoked",
                    turn_idx=ti,
                )
            )
        else:
            found = any(args_match(tc.args, expected_args) for tc in matching_calls)
            results.append(
                CheckEntry(
                    dimension="trajectory",
                    check=f"tool_args_contain({tool_name})",
                    passed=found,
                    details="" if found else f"no call matched {expected_args}",
                    turn_idx=ti,
                )
            )

    for tool_name, expected_substring in expect.tool_result_contains.items():
        matching_calls = [tc for tc in output.tool_calls if tc.name == tool_name]
        if not matching_calls:
            results.append(
                CheckEntry(
                    dimension="trajectory",
                    check=f"tool_result_contains({tool_name})",
                    passed=False,
                    details=f"tool {tool_name} not invoked",
                    turn_idx=ti,
                )
            )
        else:
            found = any(result_contains(tc, expected_substring) for tc in matching_calls)
            results.append(
                CheckEntry(
                    dimension="trajectory",
                    check=f"tool_result_contains({tool_name})",
                    passed=found,
                    details="" if found else f"no result contained {expected_substring!r}",
                    turn_idx=ti,
                )
            )

    for tool_name, expected_pattern in expect.tool_result_pattern.items():
        # Consume the compiled-regex cache populated by
        # ``TurnExpectation.model_post_init`` so the runtime never
        # recompiles (and so a stale string can never reach
        # ``result_matches`` ahead of validation).
        compiled = expect._compiled_tool_patterns[tool_name]
        matching_calls = [tc for tc in output.tool_calls if tc.name == tool_name]
        if not matching_calls:
            results.append(
                CheckEntry(
                    dimension="trajectory",
                    check=f"tool_result_pattern({tool_name})",
                    passed=False,
                    details=f"tool {tool_name} not invoked",
                    turn_idx=ti,
                )
            )
        else:
            found = any(result_matches(tc, compiled) for tc in matching_calls)
            results.append(
                CheckEntry(
                    dimension="trajectory",
                    check=f"tool_result_pattern({tool_name})",
                    passed=found,
                    details="" if found else f"no result matched {expected_pattern!r}",
                    turn_idx=ti,
                )
            )

    for skill in expect.skills_invoked:
        found, detail = skill_invoked(skill, output.tool_calls, output.raw_cli)
        results.append(
            CheckEntry(
                dimension="trajectory",
                check=f"skill_invoked({skill})",
                passed=found,
                details=detail,
                turn_idx=ti,
            )
        )

    if expect.has_thinking is not None:
        has_thinking = bool(output.thinking_text and output.thinking_text.strip())
        results.append(
            CheckEntry(
                dimension="trajectory",
                check="has_thinking",
                passed=has_thinking == expect.has_thinking,
                details=(
                    ""
                    if has_thinking == expect.has_thinking
                    else ("thinking absent" if expect.has_thinking else "thinking unexpectedly present")
                ),
                turn_idx=ti,
            )
        )

    return results
