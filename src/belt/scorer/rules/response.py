# (c) JFrog Ltd. (2026)

"""Response quality checks - reply presence and content matching."""

from __future__ import annotations

from belt.entities import TurnExpectation, TurnOutput
from belt.scorer.payloads import CheckEntry


def check_response(ti: int, output: TurnOutput, expect: TurnExpectation) -> list[CheckEntry]:
    results: list[CheckEntry] = []
    cli_lower = output.raw_cli.lower()

    if expect.has_reply:
        results.append(CheckEntry(dimension="response", check="has_reply", passed=output.has_reply, turn_idx=ti))

    for expected in expect.contains:
        reply_lower = output.reply_text.lower()
        results.append(
            CheckEntry(
                dimension="response",
                check=f"contains({expected})",
                passed=expected.lower() in reply_lower or expected.lower() in cli_lower,
                turn_idx=ti,
            )
        )

    # ``reply_pattern`` is the strict-format sibling of ``contains``:
    # regex-based, ALL must match, and matched against ``reply_text``
    # only - no ``raw_cli`` fallback. CLI noise (debug output,
    # agent-trace dumps, stderr) cannot turn the assertion green; if
    # the agent did not put the token in its reply, the check fails.
    # Compiled patterns come from ``TurnExpectation.model_post_init``
    # so the runtime never recompiles.
    for pattern, compiled in zip(expect.reply_pattern, expect._compiled_reply_patterns):
        passed = compiled.search(output.reply_text) is not None
        results.append(
            CheckEntry(
                dimension="response",
                check=f"reply_pattern({pattern})",
                passed=passed,
                details="" if passed else f"reply did not match: {output.reply_text[:200]!r}",
                turn_idx=ti,
            )
        )

    return results
