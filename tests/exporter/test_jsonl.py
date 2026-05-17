# (c) JFrog Ltd. (2026)

"""``jsonl`` exporter behaviour."""

from __future__ import annotations

import json
from pathlib import Path

from belt.exporter.entities import ExportContext
from belt.exporter.jsonl import JsonlExporter


def test_one_object_per_line(export_context: ExportContext, tmp_path: Path):
    out = tmp_path / "results.jsonl"
    JsonlExporter().export(export_context, out, {})
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        # Each line is a self-contained JSON object - the streaming contract.
        obj = json.loads(line)
        assert "scenario_name" in obj
        assert "cost_timing" in obj


def test_summary_granularity_collapses_trials(trial_export_context: ExportContext, tmp_path: Path):
    out = tmp_path / "summary.jsonl"
    JsonlExporter().export(trial_export_context, out, {"granularity": "summary"})
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["scenario"] == "g1/alpha"
    assert obj["trials"] == 2
    assert obj["trials_passed"] == 1
    assert obj["overall_pass"] is False


def test_default_granularity_emits_per_trial(trial_export_context: ExportContext, tmp_path: Path):
    out = tmp_path / "results.jsonl"
    JsonlExporter().export(trial_export_context, out, {})
    lines = out.read_text().splitlines()
    # Default mode iterates ctx.scores directly -> trial expanded.
    assert len(lines) == 2


def test_keys_are_sorted_for_diff_friendliness(export_context: ExportContext, tmp_path: Path):
    out = tmp_path / "results.jsonl"
    JsonlExporter().export(export_context, out, {})
    line = out.read_text().splitlines()[0]
    obj = json.loads(line)
    # ``json.dumps(sort_keys=True)`` lexicographically sorts top-level keys;
    # if a future refactor flips that flag, runs from different machines diff
    # spuriously.
    assert list(obj.keys()) == sorted(obj.keys())
