# (c) JFrog Ltd. (2026)

"""Performance checks - timing thresholds (ttfe, ttft, ttlt, total)."""

from __future__ import annotations

from belt.entities import TurnExpectation, TurnOutput
from belt.scorer.payloads import CheckEntry

_TIMING_FIELDS = {
    "performance_ttfe": "ttfe",
    "performance_ttft": "ttft",
    "performance_ttlt": "ttlt",
    "performance_total": "total",
}


def check_performance(ti: int, output: TurnOutput, expect: TurnExpectation) -> list[CheckEntry]:
    results: list[CheckEntry] = []

    perf_checks = [
        ("performance_ttfe", "max_ttfe_seconds", expect.max_ttfe_seconds),
        ("performance_ttft", "max_ttft_seconds", expect.max_ttft_seconds),
        ("performance_ttlt", "max_ttlt_seconds", expect.max_ttlt_seconds),
        ("performance_total", "max_total_seconds", expect.max_total_seconds),
    ]
    if not any(max_val is not None for _, _, max_val in perf_checks):
        return results

    timing = output.timing
    for dim, check_name, max_val in perf_checks:
        if max_val is None:
            continue
        field = _TIMING_FIELDS[dim]
        actual = getattr(timing, field, None) if timing else None
        passed = actual is not None and actual <= max_val
        results.append(
            CheckEntry(
                dimension=dim,
                check=f"{check_name}(<={max_val}s)",
                passed=passed,
                details=f"actual: {actual:.1f}s" if actual is not None else "not parsed",
                turn_idx=ti,
            )
        )

    return results
