# (c) JFrog Ltd. (2026)

"""Tests for ``_read_scenarios_skipped`` - the aggregate-side reader
that surfaces ``parse_and_filter``'s skipped count on
``AggregatedResults.scenarios_skipped`` via ``run_meta.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from belt.commands.aggregate import _read_scenarios_skipped
from belt.constants import RUN_META_FILE


def _write_meta(run_dir: Path, body: dict | None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if body is not None:
        (run_dir / RUN_META_FILE).write_text(json.dumps(body))


class TestReadScenariosSkipped:
    def test_reads_field_from_run_meta(self, tmp_path: Path) -> None:
        _write_meta(tmp_path, {"scenarios_skipped": 4})
        assert _read_scenarios_skipped(tmp_path) == 4

    def test_zero_when_field_missing(self, tmp_path: Path) -> None:
        # Common case: a clean run with no malformed scenarios. The
        # writer side still emits ``scenarios_skipped: 0``, but the
        # reader must also handle older runs that pre-date the field.
        _write_meta(tmp_path, {"agent": "stub"})
        assert _read_scenarios_skipped(tmp_path) == 0

    def test_zero_when_run_meta_missing(self, tmp_path: Path) -> None:
        # Aggregate run on a directory without a ``run_meta.json`` at
        # all. Falls back to ``0`` rather than crashing the aggregate.
        assert _read_scenarios_skipped(tmp_path) == 0

    def test_zero_when_run_meta_unreadable(self, tmp_path: Path) -> None:
        (tmp_path / RUN_META_FILE).write_text("not-json{{{")
        assert _read_scenarios_skipped(tmp_path) == 0

    def test_zero_when_field_is_non_numeric(self, tmp_path: Path) -> None:
        # Defensive: a corrupted writer (or a hand-edited file) putting
        # a string in the field must not crash the aggregate.
        _write_meta(tmp_path, {"scenarios_skipped": "lots"})
        assert _read_scenarios_skipped(tmp_path) == 0

    def test_negative_value_clamped_to_zero(self, tmp_path: Path) -> None:
        # ``parse_and_filter`` never produces a negative count, but we
        # clamp on read so a hand-edited file cannot make downstream
        # consumers (CSV exporter, dashboard math) misbehave.
        _write_meta(tmp_path, {"scenarios_skipped": -3})
        assert _read_scenarios_skipped(tmp_path) == 0

    def test_meta_root_not_a_dict_returns_zero(self, tmp_path: Path) -> None:
        (tmp_path / RUN_META_FILE).write_text(json.dumps([1, 2, 3]))
        assert _read_scenarios_skipped(tmp_path) == 0
