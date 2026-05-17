# (c) JFrog Ltd. (2026)

"""``markdown`` exporter behaviour: structure + untrusted-text escaping."""

from __future__ import annotations

from pathlib import Path

from belt.exporter.entities import ExportContext
from belt.exporter.markdown import MarkdownExporter


def test_failed_run_uses_failure_heading(export_context: ExportContext, tmp_path: Path):
    out = tmp_path / "summary.md"
    MarkdownExporter().export(export_context, out, {})
    text = out.read_text()
    assert "Evaluation failed" in text
    assert "## Failures" in text
    assert "g1 / beta" in text


def test_untrusted_llm_reasoning_is_md_safe(export_context: ExportContext, tmp_path: Path):
    out = tmp_path / "summary.md"
    MarkdownExporter().export(export_context, out, {})
    raw = out.read_text()
    # The fixture's reasoning embeds ``<script>alert(1)</script>`` - md_safe
    # must neutralise the leading angle bracket so the GitHub Markdown
    # renderer does not interpret it as raw HTML.
    assert "<script>" not in raw


def test_trial_run_renders_reliability_block(trial_export_context: ExportContext, tmp_path: Path):
    out = tmp_path / "summary.md"
    MarkdownExporter().export(trial_export_context, out, {})
    text = out.read_text()
    assert "## Reliability" in text
    assert "pass@1" in text


def test_passing_run_uses_neutral_heading(tmp_path: Path):
    from belt.entities import AggregatedResults

    results = AggregatedResults(schema_version="1", total=0, passed=0, failed=0, overall_pass=True)
    ctx = ExportContext(run_dir=tmp_path, results=results, scores=[])
    out = tmp_path / "summary.md"
    MarkdownExporter().export(ctx, out, {})
    assert "Evaluation report" in out.read_text()
