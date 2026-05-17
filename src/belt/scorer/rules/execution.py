# (c) JFrog Ltd. (2026)

"""Execution checks - error detection and forbidden content."""

from __future__ import annotations

from belt.constants import ERROR_PATTERNS
from belt.entities import TurnExpectation, TurnOutput
from belt.scorer.payloads import CheckEntry


def check_execution(ti: int, output: TurnOutput, expect: TurnExpectation) -> list[CheckEntry]:
    results: list[CheckEntry] = []
    cli_lower = output.raw_cli.lower()

    if expect.no_errors:
        if output.has_error is not None:
            has_error = output.has_error
        else:
            has_error = any(p in cli_lower for p in ERROR_PATTERNS)
        results.append(
            CheckEntry(dimension="execution", check="no_errors", passed=not has_error, details="", turn_idx=ti)
        )

    for forbidden in expect.not_contains:
        results.append(
            CheckEntry(
                dimension="execution",
                check=f"not_contains({forbidden})",
                passed=forbidden.lower() not in cli_lower,
                turn_idx=ti,
            )
        )

    if expect.error_type_is is not None:
        actual = output.error_type or ""
        results.append(
            CheckEntry(
                dimension="execution",
                check=f"error_type_is({expect.error_type_is})",
                passed=actual.lower() == expect.error_type_is.lower(),
                details=f"actual: {actual}" if actual else "no error_type",
                turn_idx=ti,
            )
        )

    return results
