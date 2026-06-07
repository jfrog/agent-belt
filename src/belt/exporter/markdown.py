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
from belt.scorer.payloads import PerTurnLLMPayload, RulesPayload, iter_llm_payloads, iter_llm_verdicts


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
                # Walk every LLM-shaped payload so multi-judge and
                # per-turn downgrades both appear in the exporter
                # markdown. Per-judge namespace keeps the bullets
                # attributable when the run has more than one judge.
                for name, payload in iter_llm_payloads(s):
                    rows = [
                        (dim, score_token, reasoning)
                        for dim, score_token, reasoning in iter_llm_verdicts(payload)
                        if score_token in DOWNGRADE_VERDICT_SET
                    ]
                    if not rows:
                        continue
                    label = "LLM judgements" if name == "llm" else f"LLM judgements ({md_safe(name)})"
                    lines.append(f"**{label}:**")
                    prefix = "llm" if name == "llm" else f"llm[{md_safe(name)}]"
                    for dim, score_token, reasoning in rows:
                        lines.append(f"- `{prefix}/{md_safe(dim)}` · **{md_safe(score_token)}** - {md_safe(reasoning)}")
                    lines.append("")

                    # Per-turn nested detail block under the rolled-up
                    # rubric: surface which turn(s) dragged the
                    # dimension down so a reader can attribute the
                    # downgrade without diving into score.json.
                    if isinstance(payload, PerTurnLLMPayload):
                        lines.append(f"<details><summary>Per-turn detail ({md_safe(name)})</summary>")
                        lines.append("")
                        for tv in payload.turns:
                            if not tv.dimensions:
                                if tv.judge_errored:
                                    etype = md_safe(tv.judge_error_type or "other")
                                    lines.append(f"- Turn {tv.turn_idx}: judge errored (`{etype}`)")
                                continue
                            for dim, vd in tv.dimensions.items():
                                lines.append(
                                    f"- Turn {tv.turn_idx} · `{md_safe(dim)}` · "
                                    f"**{md_safe(vd.score)}** - {md_safe(vd.reasoning)}"
                                )
                        lines.append("")
                        lines.append("</details>")
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
