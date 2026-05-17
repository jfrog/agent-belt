# (c) JFrog Ltd. (2026)

"""Workspace state checks - filesystem assertions after each turn."""

from __future__ import annotations

from belt.entities import StateExpectation, TurnOutput
from belt.scorer.payloads import CheckEntry


def check_state(ti: int, output: TurnOutput, se: StateExpectation) -> list[CheckEntry]:
    results: list[CheckEntry] = []
    ws = output.workspace_files

    for path in se.files_exist:
        exists = path in ws and ws[path] is not None
        results.append(
            CheckEntry(
                dimension="state",
                check=f"file_exists({path})",
                passed=exists,
                details="" if exists else "file not found",
                turn_idx=ti,
            )
        )

    for path, substring in se.files_contain.items():
        content = ws.get(path)
        if content is None:
            results.append(
                CheckEntry(
                    dimension="state",
                    check=f"file_contains({path})",
                    passed=False,
                    details="file not found",
                    turn_idx=ti,
                )
            )
        else:
            found = substring in content
            results.append(
                CheckEntry(
                    dimension="state",
                    check=f"file_contains({path})",
                    passed=found,
                    details="" if found else f"'{substring}' not in file",
                    turn_idx=ti,
                )
            )

    for path in se.files_not_exist:
        exists = path in ws and ws[path] is not None
        results.append(
            CheckEntry(
                dimension="state",
                check=f"file_not_exists({path})",
                passed=not exists,
                details="" if not exists else "file unexpectedly exists",
                turn_idx=ti,
            )
        )

    return results


def has_state_checks(se: StateExpectation) -> bool:
    return bool(se.files_exist or se.files_contain or se.files_not_exist)
