# (c) JFrog Ltd. (2026)

"""File-diff checks - git diff and modified-files assertions from workspace isolation."""

from __future__ import annotations

from belt.entities import TurnExpectation, TurnOutput
from belt.scorer.payloads import CheckEntry


def check_file_diff(ti: int, output: TurnOutput, expect: TurnExpectation) -> list[CheckEntry]:
    results: list[CheckEntry] = []

    if expect.files_modified_any:
        matched = [p for p in expect.files_modified_any if p in output.files_modified]
        results.append(
            CheckEntry(
                dimension="file_diff",
                check=f"files_modified_any({','.join(expect.files_modified_any)})",
                passed=bool(matched),
                details=f"matched: {', '.join(matched)}" if matched else "none matched",
                turn_idx=ti,
            )
        )

    if expect.files_modified_exact:
        actual = set(output.files_modified)
        expected = set(expect.files_modified_exact)
        ok = actual == expected
        details = ""
        if not ok:
            missing = expected - actual
            extra = actual - expected
            parts = []
            if missing:
                parts.append(f"missing: {', '.join(sorted(missing))}")
            if extra:
                parts.append(f"extra: {', '.join(sorted(extra))}")
            details = "; ".join(parts)
        results.append(
            CheckEntry(
                dimension="file_diff",
                check="files_modified_exact",
                passed=ok,
                details=details,
                turn_idx=ti,
            )
        )

    for path in expect.files_not_modified:
        modified = path in output.files_modified
        results.append(
            CheckEntry(
                dimension="file_diff",
                check=f"file_not_modified({path})",
                passed=not modified,
                details="" if not modified else "file was modified",
                turn_idx=ti,
            )
        )

    diff_text = output.git_diff or ""
    for substring in expect.git_diff_contains:
        found = substring in diff_text
        results.append(
            CheckEntry(
                dimension="file_diff",
                check=f"git_diff_contains({substring})",
                passed=found,
                details="" if found else "not found in diff",
                turn_idx=ti,
            )
        )

    return results


def has_file_diff_checks(expect: TurnExpectation) -> bool:
    return bool(
        expect.files_modified_any
        or expect.files_modified_exact
        or expect.files_not_modified
        or expect.git_diff_contains
    )
