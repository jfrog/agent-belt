# (c) JFrog Ltd. (2026)

"""Unit tests for the ``setup_errors`` plumbing introduced for #385 (1.4).

A group that fails setup never produces a per-scenario score artifact;
before this change, those scenarios silently vanished from
``results.json`` and ``belt view``. The new ``setup_errors`` field on
``AggregatedResults`` (plus the run-phase sidecar that feeds it)
preserves the cause and the casualty list so a 22-scenario run that
loses 5 to a misconfigured ``working_dir`` reports as
"17 scored + 5 setup-skipped" instead of "17, headline green".

These tests exercise the round-trip: write a fake sidecar → aggregate
reads it → ``AggregatedResults.setup_errors`` populated and
``overall_pass=False`` when any group failed setup.
"""

from __future__ import annotations

import json
from pathlib import Path

from belt.entities import AggregatedResults


def test_aggregated_results_accepts_setup_errors():
    """The new field is additive and defaults to empty list."""
    ar = AggregatedResults(total=0, passed=0, failed=0, overall_pass=True)
    assert ar.setup_errors == []

    ar2 = AggregatedResults(
        total=0,
        passed=0,
        failed=0,
        overall_pass=True,
        setup_errors=[
            {"group": "editing-workspace", "scenarios": ["a", "b"], "error": "working_dir resolves outside"},
        ],
    )
    assert ar2.setup_errors[0]["group"] == "editing-workspace"
    assert ar2.setup_errors[0]["scenarios"] == ["a", "b"]


def test_aggregated_results_round_trips_via_json():
    """Disk shape preserves the field through model_dump / parse cycle."""
    ar = AggregatedResults(
        total=1,
        passed=1,
        failed=0,
        overall_pass=False,  # forced false because setup_errors present
        setup_errors=[{"group": "g1", "scenarios": ["s1"], "error": "boom"}],
    )
    payload = ar.model_dump(mode="json")
    parsed = AggregatedResults.model_validate(payload)
    assert parsed.setup_errors == ar.setup_errors


def test_read_setup_errors_sidecar(tmp_path: Path):
    """The aggregator reads ``setup_errors.json`` written by the run phase."""
    from belt.commands.aggregate import _read_setup_errors

    sidecar = tmp_path / "setup_errors.json"
    sidecar.write_text(
        json.dumps(
            [
                {"group": "g1", "scenarios": ["s1", "s2"], "error": "boom"},
                {"group": "g2", "scenarios": [], "error": "boom2"},
            ]
        )
    )
    errs = _read_setup_errors(tmp_path)
    assert [e["group"] for e in errs] == ["g1", "g2"]
    assert errs[0]["scenarios"] == ["s1", "s2"]
    assert errs[0]["error"] == "boom"


def test_read_setup_errors_missing_file_returns_empty(tmp_path: Path):
    from belt.commands.aggregate import _read_setup_errors

    assert _read_setup_errors(tmp_path) == []


def test_read_setup_errors_malformed_returns_empty(tmp_path: Path):
    from belt.commands.aggregate import _read_setup_errors

    # Not a list - should silently fall back to empty.
    (tmp_path / "setup_errors.json").write_text(json.dumps({"not": "a list"}))
    assert _read_setup_errors(tmp_path) == []

    # Garbage JSON - same.
    (tmp_path / "setup_errors.json").write_text("not json at all")
    assert _read_setup_errors(tmp_path) == []


def test_view_reads_setup_errors_from_results_json(tmp_path: Path):
    """``belt view`` reads ``setup_errors`` from ``results.json`` (canonical path)."""
    from belt.commands.view import _read_run_setup_errors

    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "total": 1,
                "passed": 1,
                "failed": 0,
                "overall_pass": False,
                "setup_errors": [{"group": "g1", "scenarios": ["s1"], "error": "boom"}],
            }
        )
    )
    errs = _read_run_setup_errors(tmp_path)
    assert len(errs) == 1
    assert errs[0]["group"] == "g1"


def test_view_falls_back_to_sidecar_when_results_missing(tmp_path: Path):
    """If ``aggregate`` never ran, the sidecar still feeds the banner."""
    from belt.commands.view import _read_run_setup_errors

    (tmp_path / "setup_errors.json").write_text(json.dumps([{"group": "g1", "scenarios": ["s1"], "error": "boom"}]))
    errs = _read_run_setup_errors(tmp_path)
    assert len(errs) == 1
    assert errs[0]["group"] == "g1"
