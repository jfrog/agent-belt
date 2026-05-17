# (c) JFrog Ltd. (2026)

"""``commands/export.py`` driver behaviour: parsing, slot assembly, failure isolation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from belt.commands import export as export_cmd
from belt.entities import AggregatedResults, ScenarioScore
from belt.errors import ConfigError
from belt.exporter.base import BaseExporter
from belt.exporter.entities import ExportContext

# ── Spec parsing ──


class TestParseToSpec:
    def test_basic(self):
        assert export_cmd.parse_to_spec("csv:results.csv") == ("csv", "results.csv")

    def test_path_with_colon(self):
        # ``parse_threshold`` uses the same convention; we split on the first
        # colon so a Windows-style ``C:\out.csv`` survives.
        assert export_cmd.parse_to_spec("csv:C:/dir/out.csv") == ("csv", "C:/dir/out.csv")

    @pytest.mark.parametrize("bad", ["csv", "csv:", ":path.csv", ""])
    def test_bad_specs_rejected(self, bad: str):
        with pytest.raises(ConfigError):
            export_cmd.parse_to_spec(bad)


# ── YAML config parsing ──


class TestExportConfig:
    def test_valid_yaml(self, tmp_path: Path):
        cfg = tmp_path / "exporters.yaml"
        cfg.write_text(
            "exporters:\n"
            "  - name: csv\n"
            "    path: results.csv\n"
            "  - name: junit\n"
            "    path: report.xml\n"
            "    options:\n"
            "      suite_name: matrix-1\n"
        )
        loaded = export_cmd.load_export_config(cfg)
        assert [e.name for e in loaded.exporters] == ["csv", "junit"]
        assert loaded.exporters[1].options == {"suite_name": "matrix-1"}

    def test_missing_file_raises_configerror(self, tmp_path: Path):
        with pytest.raises(ConfigError):
            export_cmd.load_export_config(tmp_path / "nope.yaml")

    def test_unknown_top_level_key_rejected(self, tmp_path: Path):
        cfg = tmp_path / "exporters.yaml"
        cfg.write_text("exporters: []\nunexpected: value\n")
        with pytest.raises(ConfigError):
            export_cmd.load_export_config(cfg)

    def test_invalid_yaml_rejected(self, tmp_path: Path):
        cfg = tmp_path / "exporters.yaml"
        cfg.write_text("not:\n  - valid: [unbalanced\n")
        with pytest.raises(ConfigError):
            export_cmd.load_export_config(cfg)


# ── Failure isolation ──


class _AlwaysRaisingExporter(BaseExporter):
    @property
    def name(self) -> str:
        return "_test_raise"

    def export(self, ctx: ExportContext, output: Path, options: dict[str, Any]) -> None:
        raise RuntimeError("simulated plugin failure")


class _NoOpExporter(BaseExporter):
    @property
    def name(self) -> str:
        return "_test_noop"

    def export(self, ctx: ExportContext, output: Path, options: dict[str, Any]) -> None:
        output.write_text("ok")


@pytest.fixture
def patched_registry(monkeypatch):
    """Inject two test exporters into the built-in registry without touching disk."""
    from belt.exporter import registry

    test_registry = {
        **registry._EXPORTER_REGISTRY,
        "_test_raise": "tests.exporter.test_command_export:_AlwaysRaisingExporter",
        "_test_noop": "tests.exporter.test_command_export:_NoOpExporter",
    }
    monkeypatch.setattr(registry, "_EXPORTER_REGISTRY", test_registry)
    return test_registry


class TestFailureIsolation:
    def test_one_exporter_failing_does_not_abort_others(
        self,
        patched_registry,
        export_context: ExportContext,
        tmp_path: Path,
        capsys,
    ):
        out_ok = tmp_path / "ok.txt"
        out_fail = tmp_path / "fail.txt"
        rc = export_cmd.run_exporters(
            run_dir=export_context.run_dir,
            results=export_context.results,
            scores=export_context.scores,
            to_specs=[
                f"_test_raise:{out_fail}",
                f"_test_noop:{out_ok}",
            ],
            config_path=None,
        )
        # At least one exporter succeeded -> rc == 0 per the spec.
        assert rc == 0
        # The successful exporter wrote its file; the failing one did not.
        assert out_ok.read_text() == "ok"
        assert not out_fail.exists()

    def test_all_failing_yields_nonzero_exit(
        self,
        patched_registry,
        export_context: ExportContext,
        tmp_path: Path,
    ):
        rc = export_cmd.run_exporters(
            run_dir=export_context.run_dir,
            results=export_context.results,
            scores=export_context.scores,
            to_specs=[f"_test_raise:{tmp_path / 'a.txt'}"],
            config_path=None,
        )
        assert rc == 1


# ── No-op when nothing requested ──


class TestNothingRequested:
    def test_empty_specs_returns_nonzero(self, export_context: ExportContext):
        rc = export_cmd.run_exporters(
            run_dir=export_context.run_dir,
            results=export_context.results,
            scores=export_context.scores,
            to_specs=[],
            config_path=None,
        )
        assert rc == 1


# ── End-to-end: export a real run via CLI driver ──


def _seed_run_dir(tmp_path: Path, scores: list[ScenarioScore]) -> Path:
    """Write the on-disk shape ``commands/export.main`` expects to read."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results = AggregatedResults(
        schema_version="1",
        total=len(scores),
        passed=sum(1 for s in scores if s.overall_pass),
        failed=sum(1 for s in scores if not s.overall_pass),
        overall_pass=all(s.overall_pass for s in scores),
        scenarios=[s.model_dump(mode="json") for s in scores],
    )
    (run_dir / "results.json").write_text(json.dumps(results.model_dump(mode="json"), indent=2) + "\n")
    for s in scores:
        scenario_dir = run_dir / s.group / s.scenario_name
        scenario_dir.mkdir(parents=True)
        (scenario_dir / "score.json").write_text(s.model_dump_json())
    return run_dir


class TestEndToEndCli:
    def test_main_writes_csv_against_real_run_dir(self, tmp_path: Path, mixed_scores: list[ScenarioScore]):
        run_dir = _seed_run_dir(tmp_path, mixed_scores)
        out = tmp_path / "out.csv"
        rc = export_cmd.main([str(run_dir), "--to", f"csv:{out}"])
        assert rc == 0
        assert out.is_file()
        content = out.read_text()
        assert "scenario" in content.splitlines()[0]

    def test_main_errors_when_results_missing(self, tmp_path: Path, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        rc = export_cmd.main([str(empty), "--to", "csv:/tmp/x.csv"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "results.json" in err
