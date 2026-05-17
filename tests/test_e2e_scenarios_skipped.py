# (c) JFrog Ltd. (2026)

"""End-to-end test for the ``scenarios_skipped`` signal.

Drives the full thread:

1. ``parse_and_filter`` counts malformed scenarios.
2. ``commands/run.py`` persists the count to ``run_meta.json``.
3. ``commands/aggregate.main`` reads it back via ``_read_scenarios_skipped``
   and surfaces it on ``AggregatedResults.scenarios_skipped`` in
   ``results.json``.

Skipping the actual orchestrator/scorer keeps the test hermetic - we
seed a synthetic run directory with one ``score.json`` and a
``run_meta.json`` that already carries the count, then assert the
aggregate writes it through. Step 1 is verified directly in
``tests/runner/phases/test_parse_filter.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from belt.commands.aggregate import main as aggregate_main
from belt.constants import RESULTS_FILE, RUN_META_FILE, SCHEMA_VERSION


def _seed_run_dir(run_dir: Path, *, scenarios_skipped: int) -> None:
    """Write the minimum artifacts ``aggregate_main`` needs to succeed.

    One passing ``score.json`` plus a ``run_meta.json`` carrying the
    count we want to thread through.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_META_FILE).write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "scenarios_root": str(run_dir),
                "agent": "stub",
                "env": {},
                "scenarios_skipped": scenarios_skipped,
            }
        )
    )
    score_dir = run_dir / "g" / "valid"
    score_dir.mkdir(parents=True)
    (score_dir / "score.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario_name": "valid",
                "group": "g",
                "scores": {},
                "overall_pass": True,
            }
        )
    )


def test_scenarios_skipped_threads_through_to_results_json(tmp_path: Path) -> None:
    """End-to-end: a non-zero count in ``run_meta.json`` lands on
    ``AggregatedResults.scenarios_skipped`` in ``results.json``."""
    run_dir = tmp_path / "run-001"
    _seed_run_dir(run_dir, scenarios_skipped=2)

    rc = aggregate_main(["--run-dir", str(run_dir)])
    assert rc == 0

    results = json.loads((run_dir / RESULTS_FILE).read_text())
    assert results["scenarios_skipped"] == 2
    # The passing score still aggregates normally - the new field is
    # purely additive and does not affect pass/fail accounting.
    assert results["total"] == 1
    assert results["passed"] == 1
    assert results["overall_pass"] is True


def test_scenarios_skipped_defaults_to_zero_on_clean_run(tmp_path: Path) -> None:
    """Clean run (no malformed files) → ``scenarios_skipped == 0`` in
    the on-disk ``results.json``. Guards against a future regression
    where the field accidentally becomes ``None`` or is omitted."""
    run_dir = tmp_path / "run-clean"
    _seed_run_dir(run_dir, scenarios_skipped=0)

    rc = aggregate_main(["--run-dir", str(run_dir)])
    assert rc == 0

    results = json.loads((run_dir / RESULTS_FILE).read_text())
    assert results["scenarios_skipped"] == 0


def test_scenarios_skipped_run_meta_without_field_loads_as_zero(
    tmp_path: Path,
) -> None:
    """``scenarios_skipped`` is optional in the v1 run-meta contract:
    a ``run_meta.json`` that omits the key surfaces ``0`` through the
    aggregator rather than crashing."""
    run_dir = tmp_path / "run-no-skipped"
    run_dir.mkdir(parents=True)
    (run_dir / RUN_META_FILE).write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "scenarios_root": str(run_dir),
                "agent": "stub",
                "env": {},
            }
        )
    )
    score_dir = run_dir / "g" / "valid"
    score_dir.mkdir(parents=True)
    (score_dir / "score.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario_name": "valid",
                "group": "g",
                "scores": {},
                "overall_pass": True,
            }
        )
    )

    rc = aggregate_main(["--run-dir", str(run_dir)])
    assert rc == 0

    results = json.loads((run_dir / RESULTS_FILE).read_text())
    assert results["scenarios_skipped"] == 0
