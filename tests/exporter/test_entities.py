# (c) JFrog Ltd. (2026)

"""Contract checks on the export-phase Pydantic models.

These shapes are the contract between the export driver and exporter
plugins. They MUST stay strict (``extra='forbid'`` where applicable) so a
typo in user-facing YAML surfaces as a clear validation error rather than
a silent no-op.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from belt.entities import AggregatedResults
from belt.exporter.entities import ExportContext, ExporterEntry, ExporterFile


class TestExporterEntry:
    def test_minimal_entry(self):
        e = ExporterEntry(name="csv", path="results.csv")
        assert e.options == {}

    def test_unknown_field_rejected(self):
        # ``extra='forbid'`` - typos like ``paht:`` (instead of ``path:``)
        # in the YAML must fail loudly.
        with pytest.raises(ValidationError):
            ExporterEntry(name="csv", paht="results.csv")  # type: ignore[call-arg]

    def test_missing_required_fields_rejected(self):
        with pytest.raises(ValidationError):
            ExporterEntry(name="csv")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            ExporterEntry(path="results.csv")  # type: ignore[call-arg]


class TestExporterFile:
    def test_empty_file(self):
        f = ExporterFile()
        assert f.exporters == []

    def test_unknown_top_level_key_rejected(self):
        with pytest.raises(ValidationError):
            ExporterFile.model_validate({"exporters": [], "unknown": "value"})

    def test_round_trip_via_model_validate(self):
        f = ExporterFile.model_validate(
            {
                "exporters": [
                    {"name": "csv", "path": "out.csv"},
                    {"name": "junit", "path": "out.xml", "options": {"suite_name": "x"}},
                ]
            }
        )
        assert [e.name for e in f.exporters] == ["csv", "junit"]
        assert f.exporters[1].options == {"suite_name": "x"}


class TestExportContext:
    def test_constructed_from_real_aggregated_results(self, tmp_path: Path):
        results = AggregatedResults(schema_version="1", total=0, passed=0, failed=0, overall_pass=True)
        ctx = ExportContext(run_dir=tmp_path, results=results, scores=[])
        assert ctx.benchmark_card is None
        assert ctx.scores == []

    def test_run_dir_is_pathlike(self, tmp_path: Path):
        # Pydantic accepts a string for Path fields; confirm the round-trip
        # does not silently coerce away the type.
        results = AggregatedResults(schema_version="1")
        ctx = ExportContext(run_dir=str(tmp_path), results=results, scores=[])
        assert isinstance(ctx.run_dir, Path)
