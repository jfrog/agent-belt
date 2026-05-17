# (c) JFrog Ltd. (2026)

"""End-to-end threshold enforcement test for ``belt aggregate``.

Unit tests in ``tests/commands/test_aggregate.py`` cover ``parse_threshold``,
``validate_thresholds``, and the ``ThresholdEnforcer`` API. This module
exercises the full subprocess CLI surface - ``--threshold`` parsing, exit
code, and the "no threshold == always exit 0" contract - by feeding the
binary a synthetic outcomes directory.

Subprocess-based because the contract CI consumers depend on is the binary's
exit code; unit-testing the enforcer in isolation can pass while the CLI
wiring is broken.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_BELT_BIN = shutil.which("belt")


def _write_score(dir_path: Path, *, passed: bool, judge_errored: bool = False) -> None:
    """Write a minimal score.json under <dir_path>/<group>/<scenario>/.

    ``judge_errored=True`` produces a force-failed scenario whose llm
    payload mirrors the shape the scorer pipeline writes when an LLM
    judge is unreachable (rate-limit, timeout, parse, auth).
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    rules_passed = passed and not judge_errored
    overall_pass = passed and not judge_errored
    checks = [
        {
            "dimension": "execution",
            "check": "no_errors",
            "passed": passed,
            "details": "",
            "turn_idx": 0,
        }
    ]
    if judge_errored:
        checks.append(
            {
                "dimension": "execution",
                "check": "llm_scorer_ran",
                "passed": False,
                "details": "llm scorer judge-infra failure: rate_limited",
                "turn_idx": None,
            }
        )
    score = {
        "schema_version": "1",
        "scenario_name": dir_path.name,
        "group": dir_path.parent.name,
        "scores": {
            "rules": {
                "schema_version": "rules.v1",
                "checks": checks,
                "passed": rules_passed,
            }
        },
        "overall_pass": overall_pass,
    }
    if judge_errored:
        score["scores"]["llm"] = {
            "schema_version": "llm.v1",
            "overall_pass": False,
            "dimensions": {},
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": None,
                "cached": False,
            },
            "consensus_meta": None,
            "individual_verdicts": None,
            "judge_errored": True,
            "judge_error_type": "rate_limited",
        }
    (dir_path / "score.json").write_text(json.dumps(score))


def _run_aggregate(*args: str, run_dir: Path) -> subprocess.CompletedProcess:
    if not _BELT_BIN:
        pytest.skip("belt console script not on PATH")
    return subprocess.run(
        [_BELT_BIN, "aggregate", "--run-dir", str(run_dir), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.fixture
def outcomes_with_one_failure(tmp_path: Path) -> Path:
    """An outcomes dir with one passing and one failing scenario (50% failure)."""
    run = tmp_path / "outcomes" / "20260503-150000"
    _write_score(run / "demo" / "passes_basic", passed=True)
    _write_score(run / "demo" / "fails_basic", passed=False)
    return run


def test_no_threshold_flag_exits_zero_even_with_failures(outcomes_with_one_failure: Path) -> None:
    """The documented contract: ``aggregate`` without ``--threshold`` is report-only."""
    proc = _run_aggregate(run_dir=outcomes_with_one_failure)
    assert proc.returncode == 0, (
        f"aggregate without --threshold should always exit 0; got rc={proc.returncode}\n"
        f"stdout: {proc.stdout[:500]}\nstderr: {proc.stderr[:500]}"
    )


def test_zero_tolerance_threshold_exits_one_when_any_fails(outcomes_with_one_failure: Path) -> None:
    proc = _run_aggregate(
        "--threshold",
        "rules/execution:0",
        run_dir=outcomes_with_one_failure,
    )
    assert proc.returncode == 1, (
        f"--threshold rules/execution:0 should fail when any rules/execution check fails; "
        f"got rc={proc.returncode}\nstdout: {proc.stdout[:500]}"
    )


def test_loose_threshold_passes_when_failure_rate_is_within_budget(
    outcomes_with_one_failure: Path,
) -> None:
    """One-of-two = 50% failure; budget of 60% must pass (50 ≤ 60)."""
    proc = _run_aggregate(
        "--threshold",
        "rules/execution:60",
        run_dir=outcomes_with_one_failure,
    )
    assert proc.returncode == 0, (
        f"--threshold rules/execution:60 should pass at 50% failure rate; "
        f"got rc={proc.returncode}\nstdout: {proc.stdout[:500]}"
    )


def test_aggregate_writes_results_json_with_thresholds_passed_field(
    outcomes_with_one_failure: Path,
) -> None:
    """results.json carries the thresholds_passed field for downstream tooling."""
    proc = _run_aggregate(
        "--threshold",
        "rules/execution:60",
        run_dir=outcomes_with_one_failure,
    )
    assert proc.returncode == 0, proc.stderr
    results = json.loads((outcomes_with_one_failure / "results.json").read_text())
    assert results["thresholds_passed"] is True
    assert results["total"] == 2
    assert results["failed"] == 1


def test_aggregate_marks_thresholds_passed_false_when_threshold_fails(
    outcomes_with_one_failure: Path,
) -> None:
    proc = _run_aggregate(
        "--threshold",
        "rules/execution:0",
        run_dir=outcomes_with_one_failure,
    )
    assert proc.returncode == 1
    results = json.loads((outcomes_with_one_failure / "results.json").read_text())
    assert results["thresholds_passed"] is False


def test_judge_infra_failure_exits_nonzero_without_threshold(tmp_path: Path) -> None:
    """A run with any judge-infra failure must exit non-zero even with no
    ``--threshold`` flag.

    The aggregator already partitions judge-infra failures into a dedicated
    axis, marks the scenario ``overall_pass=false``, and prints "do not
    treat passing rules as a green" in ``bottom_line``. Without a non-zero
    exit, CI shell-checking ``$?`` would still mark the run green - which
    is exactly the false-green failure mode the partition was built to
    eliminate.
    """
    run = tmp_path / "outcomes" / "20260512-000000"
    _write_score(run / "demo" / "good", passed=True)
    _write_score(run / "demo" / "judge_dropped", passed=True, judge_errored=True)
    proc = _run_aggregate(run_dir=run)
    assert proc.returncode == 1, (
        f"aggregate must exit non-zero on judge-infra failure even without --threshold; "
        f"got rc={proc.returncode}\nstdout: {proc.stdout[:600]}"
    )
    results = json.loads((run / "results.json").read_text())
    assert results["judge_errors"] is not None
    assert results["judge_errors"]["scenarios_with_errors"] == 1


def test_judge_infra_failure_does_not_block_when_no_judge_errors(tmp_path: Path) -> None:
    """Negative control: a clean run (rules-only failure, no judge errors)
    keeps the existing "exit 0 without --threshold" contract.

    Guards against over-broadening the judge-infra exit gate into "any
    failure exits non-zero", which would silently break the
    test_no_threshold_flag_exits_zero_even_with_failures contract.
    """
    run = tmp_path / "outcomes" / "20260512-000001"
    _write_score(run / "demo" / "passes", passed=True)
    _write_score(run / "demo" / "fails_on_rules", passed=False)
    proc = _run_aggregate(run_dir=run)
    assert proc.returncode == 0, (
        f"clean rules-only failure (no judge errors) must keep advisory exit-0; "
        f"got rc={proc.returncode}\nstdout: {proc.stdout[:600]}"
    )
    results = json.loads((run / "results.json").read_text())
    assert results["judge_errors"] is None


def test_invalid_threshold_format_is_rejected_at_parse_time(
    outcomes_with_one_failure: Path,
) -> None:
    proc = _run_aggregate(
        "--threshold",
        "this-is-not-a-threshold",
        run_dir=outcomes_with_one_failure,
    )
    assert proc.returncode != 0
    assert "threshold" in (proc.stderr + proc.stdout).lower()
