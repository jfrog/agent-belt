# (c) JFrog Ltd. (2026)

"""``csv`` exporter behaviour."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from belt.exporter.csv import CsvExporter
from belt.exporter.entities import ExportContext


class TestScenarioGranularity:
    def test_one_row_per_scenario(self, export_context: ExportContext, tmp_path: Path):
        out = tmp_path / "results.csv"
        CsvExporter().export(export_context, out, {})
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2  # alpha + beta
        names = {r["scenario"] for r in rows}
        assert names == {"alpha", "beta"}

    def test_failed_rule_checks_column(self, export_context: ExportContext, tmp_path: Path):
        out = tmp_path / "results.csv"
        CsvExporter().export(export_context, out, {})
        with out.open() as f:
            rows = list(csv.DictReader(f))
        beta = next(r for r in rows if r["scenario"] == "beta")
        assert "execution/exited_zero" in beta["failed_rule_checks"]

    def test_costs_round_trip(self, export_context: ExportContext, tmp_path: Path):
        out = tmp_path / "results.csv"
        CsvExporter().export(export_context, out, {})
        with out.open() as f:
            rows = list(csv.DictReader(f))
        alpha = next(r for r in rows if r["scenario"] == "alpha")
        # 0.001234 -> "0.001234" after the rstrip path
        assert alpha["agent_cost_usd"] == "0.001234"


class TestTrialGranularity:
    def test_one_row_per_trial(self, trial_export_context: ExportContext, tmp_path: Path):
        out = tmp_path / "results.csv"
        CsvExporter().export(trial_export_context, out, {"granularity": "trial"})
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        # Both trials carry the same "passed" of their individual outcome.
        assert {r["passed"] for r in rows} == {"true", "false"}

    def test_collapsed_by_default(self, trial_export_context: ExportContext, tmp_path: Path):
        out = tmp_path / "results.csv"
        CsvExporter().export(trial_export_context, out, {})
        with out.open() as f:
            rows = list(csv.DictReader(f))
        # Default granularity == scenario -> 1 row per base scenario.
        assert len(rows) == 1
        assert rows[0]["scenario"] == "alpha"
        assert rows[0]["trials"] == "2"
        assert rows[0]["trials_passed"] == "1"
        # Mixed pass/fail across trials -> overall failure.
        assert rows[0]["passed"] == "false"


class TestOptions:
    def test_custom_delimiter(self, export_context: ExportContext, tmp_path: Path):
        out = tmp_path / "results.tsv"
        CsvExporter().export(export_context, out, {"delimiter": "\t"})
        text = out.read_text()
        assert "\t" in text.splitlines()[0]

    def test_invalid_delimiter_falls_back_to_comma(self, export_context: ExportContext, tmp_path: Path):
        # Multi-char delimiters silently fall back; we don't want to raise
        # because YAML deserialisation could legitimately produce odd values.
        out = tmp_path / "results.csv"
        CsvExporter().export(export_context, out, {"delimiter": "BAD"})
        text = out.read_text()
        assert "," in text.splitlines()[0]


@pytest.mark.parametrize("ext", ["csv", "tsv"])
def test_creates_parent_directories(export_context: ExportContext, tmp_path: Path, ext: str):
    out = tmp_path / "nested" / "deeper" / f"results.{ext}"
    CsvExporter().export(export_context, out, {})
    assert out.is_file()


class TestFormulaInjection:
    """OWASP-style hardening: cells starting with =/+/-/@/\\t/\\r are sanitized.

    Without this, an LLM judge can produce ``reasoning`` like
    ``=cmd|'/c calc'!A1`` which executes when the CSV is opened in Excel.
    The sanitizer prepends ``'`` so the spreadsheet renders the literal
    string instead of evaluating it as a formula.
    """

    @pytest.mark.parametrize("malicious", ["=cmd", "+1+1", "-cmd", "@SUM(A1)", "\tinjected"])
    def test_risky_leading_chars_are_prefixed(self, malicious: str, tmp_path: Path):
        from belt.entities import AggregatedResults
        from belt.exporter.entities import ExportContext

        score = ScenarioScoreFactory(scenario_name=malicious)
        results = AggregatedResults(
            schema_version="1",
            total=1,
            passed=0,
            failed=1,
            overall_pass=False,
            scenarios=[score.model_dump(mode="json")],
        )
        ctx = ExportContext(run_dir=tmp_path, results=results, scores=[score])
        out = tmp_path / "results.csv"
        CsvExporter().export(ctx, out, {})
        content = out.read_text()
        # The malicious string appears in the data row but the leading
        # character is prefixed with a single quote so it is rendered as
        # literal text, not evaluated.
        assert "'" + malicious in content
        # Header line must NEVER carry the prefix - column names start
        # with safe ASCII letters so the sanitizer is a no-op there.
        header = content.splitlines()[0]
        assert "'" not in header

    def test_leading_safe_char_is_unchanged(self, export_context, tmp_path: Path):
        """The fixture's 'alpha'/'beta' scenario names are safe and must
        round-trip without modification."""
        out = tmp_path / "results.csv"
        CsvExporter().export(export_context, out, {})
        text = out.read_text()
        # No spurious ' prefixes on cells that were always safe.
        assert ",'alpha" not in text
        assert ",'beta" not in text


def ScenarioScoreFactory(scenario_name: str):
    """Factory for the formula-injection test - keeps the parametrize body short."""
    from belt.entities import ScenarioScore
    from belt.scorer.payloads import RulesPayload

    return ScenarioScore(
        scenario_name=scenario_name,
        group="g",
        tags=[],
        scores={"rules": RulesPayload(checks=[], passed=False)},
        overall_pass=False,
    )
