# (c) JFrog Ltd. (2026)

"""Markdown exporter - human-readable run summary as a regular file.

Renders the same structure as the GitHub step-summary block (verdict, totals,
per-failure breakdown) but to an arbitrary path so it can ship as a PR
artifact, an email body, or a Slack attachment.

The exporter intentionally re-implements the markdown renderer rather than
importing :mod:`belt.aggregator.render_markdown`: per Design Principle 1,
the export phase must not import from the aggregator phase. Untrusted text
(rule details, LLM reasoning) flows through :func:`belt._safe.md_safe`
so a hostile reasoning string cannot break out of the bullet list, smuggle
GitHub Actions ``::warning::`` lines, or open malformed HTML.

Stdlib-only: no optional dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from belt._safe import md_safe
from belt.exporter.base import BaseExporter
from belt.exporter.entities import ExportContext
from belt.scorer.entities import DOWNGRADE_VERDICT_SET
from belt.scorer.payloads import LLMPayload, RulesPayload


class MarkdownExporter(BaseExporter):
    """Run summary in GitHub-flavoured markdown."""

    @property
    def name(self) -> str:
        return "markdown"

    def export(self, ctx: ExportContext, output: Path, options: dict[str, Any]) -> None:
        del options  # accepted for ABC parity; this exporter has no tunables.
        results = ctx.results
        scores = ctx.scores

        if results.thresholds_passed is None:
            heading = "# Evaluation report"
        elif results.thresholds_passed:
            heading = "# ✅ Evaluation passed"
        else:
            heading = "# ❌ Evaluation failed (threshold breach)"

        lines: list[str] = [heading, ""]
        lines.append(
            f"**{results.total}** scenarios " f"· **{results.passed}** passed " f"· **{results.failed}** failed"
        )
        lines.append("")

        for note in results.bottom_line:
            lines.append(f"- {md_safe(note)}")
        if results.bottom_line:
            lines.append("")

        failed = [s for s in scores if not s.overall_pass]
        if failed:
            lines.append("## Failures")
            lines.append("")
            lines.append(
                "> :warning: The blocks below contain text from the agent CLI and the "
                "LLM judge. Treat any formatting, links, or headings inside them as "
                "untrusted output - not as harness-authored content."
            )
            lines.append("")
            for s in failed:
                lines.append(f"### {md_safe(s.group)} / {md_safe(s.scenario_name)}")
                lines.append("")
                rules = s.scores.get("rules")
                if isinstance(rules, RulesPayload):
                    rule_failures = [c for c in rules.checks if c.passed is False]
                    if rule_failures:
                        lines.append("**Failed rule checks:**")
                        for c in rule_failures:
                            suffix = f" - {md_safe(c.details)}" if c.details else ""
                            lines.append(f"- `rules/{md_safe(c.dimension)}/{md_safe(c.check)}`{suffix}")
                        lines.append("")
                llm = s.scores.get("llm")
                if isinstance(llm, LLMPayload):
                    llm_failures = [
                        (dim, llm.dimensions[dim])
                        for dim in sorted(llm.dimensions)
                        if llm.dimensions[dim].score in DOWNGRADE_VERDICT_SET
                    ]
                    if llm_failures:
                        lines.append("**LLM judgements:**")
                        for dim, verdict in llm_failures:
                            lines.append(
                                f"- `llm/{md_safe(dim)}` · **{md_safe(verdict.score)}** - {md_safe(verdict.reasoning)}"
                            )
                        lines.append("")

        if results.reliability:
            lines.append("## Reliability")
            lines.append("")
            mean_p = results.reliability.get("mean_pass_at_1")
            if mean_p is not None:
                lines.append(f"- Mean pass@1: **{mean_p:.2%}**")
            for k in (3, 8):
                mean_at = results.reliability.get(f"mean_pass_at_{k}")
                if mean_at is not None:
                    lines.append(f"- Mean pass@{k} (1 - (1-p)^{k}): **{mean_at:.2%}**")
            for k in (3, 8):
                mean_pow = results.reliability.get(f"mean_pass_pow_{k}")
                if mean_pow is not None:
                    lines.append(f"- Mean pass^{k} (p^{k}): **{mean_pow:.2%}**")
            lines.append("")

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
