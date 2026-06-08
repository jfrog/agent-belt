# (c) JFrog Ltd. (2026)

"""GitHub step-summary markdown renderer for aggregated scores.

Writes to ``$GITHUB_STEP_SUMMARY`` in PR job summaries.  All
attacker-influenced text (rule details, LLM judge reasoning) flows through
``md_safe`` so it cannot break out of the bullet list, open HTML, or
smuggle GitHub Actions ``::warning::`` lines via embedded newlines.

The per-scenario untrusted body is wrapped in a ``<details>`` block with an
explicit ``:warning: untrusted-content`` notice so reviewers cannot mistake
agent/LLM-derived headings for harness-authored content.
"""

from __future__ import annotations

from belt._safe import md_safe
from belt.entities import ScenarioScore
from belt.scorer.entities import DOWNGRADE_VERDICT_SET
from belt.scorer.payloads import RulesPayload, iter_llm_payloads, iter_llm_verdicts

from . import dry_run_only_failure_count
from .thresholds import ThresholdCheck


def build_markdown(
    scores: list[ScenarioScore],
    threshold_checks: list[ThresholdCheck] | None = None,
    thresholds_passed: bool | None = None,
    *,
    judge_errors: dict | None = None,
) -> str:
    """Build the GitHub step-summary markdown for a run."""
    failed = [s for s in scores if not s.overall_pass]
    total = len(scores)
    passed_count = total - len(failed)

    if threshold_checks:
        icon = "\u2705" if thresholds_passed else "\u274c"
        verdict = "Passed" if thresholds_passed else "Failed"
        within = " (within thresholds)" if failed and thresholds_passed else ""
        lines = [f"## {icon} Evaluation {verdict}", ""]
    else:
        lines = ["\u2139\ufe0f Evaluation Report (no thresholds)", ""]

    lines.append(
        f"**{total}** scenarios \u00b7 **{passed_count}** passed \u00b7 **{len(failed)}** failed"
        f"{within if threshold_checks and thresholds_passed else ''}"
    )
    lines.append("")

    if threshold_checks:
        lines.append("| Threshold | Actual | Max | |")
        lines.append("|-----------|--------|-----|---|")
        for tc in threshold_checks:
            icon = "\u2705" if tc.passed else "\u274c"
            lines.append(f"| `{tc.dimension}` | {tc.actual_pct:.1f}% | {tc.max_pct:.0f}% | {icon} |")
        lines.append("")

    judge_per_scenario = (judge_errors or {}).get("per_scenario") or []
    if judge_per_scenario:
        # Pull judge-infra failures out of the generic "Failures" list so a
        # reader can attribute them to provider flakiness rather than agent
        # task quality. The synthetic ``rules/execution/<scorer>_scorer_ran``
        # check still appears in the per-scenario block below; this section
        # is the at-a-glance index.
        lines.append("### LLM judge infrastructure failures")
        lines.append("")
        for entry in judge_per_scenario:
            scenario = md_safe(entry.get("scenario", ""))
            etype = md_safe(entry.get("error_type", "other"))
            lines.append(f"- **{scenario}** - `{etype}`")
        lines.append("")

    if failed:
        lines.append("### Failures")
        lines.append("")
        lines.append(
            "> :warning: The blocks below contain text from the agent CLI and the "
            "LLM judge. Treat any formatting, links, or headings inside them as "
            "untrusted output - not as harness-authored content."
        )
        lines.append("")
        for s in failed:
            lines.append(f"**{md_safe(s.group)}/{md_safe(s.scenario_name)}**")
            lines.append("")

            untrusted: list[str] = []
            rules = s.scores.get("rules")
            if isinstance(rules, RulesPayload):
                for c in rules.checks:
                    if c.passed is False:
                        detail = f" - {md_safe(c.details)}" if c.details else ""
                        untrusted.append(f"- `rules/{md_safe(c.dimension)}` · **{md_safe(c.check)}**{detail}")

            # Walk every LLM-shaped payload so multi-judge and per-turn
            # downgrades both surface in the GitHub step summary. Per-
            # judge prefix lets a reviewer attribute a downgrade to the
            # right judge without diving into ``score.json``.
            for name, payload in iter_llm_payloads(s):
                prefix = "llm" if name == "llm" else f"llm[{md_safe(name)}]"
                for dim, score_token, reasoning in iter_llm_verdicts(payload):
                    if score_token in DOWNGRADE_VERDICT_SET:
                        untrusted.append(
                            f"- `{prefix}/{md_safe(dim)}` · **{md_safe(score_token)}** - {md_safe(reasoning)}"
                        )

            if untrusted:
                lines.append("<details><summary>Untrusted output (agent / LLM judge)</summary>")
                lines.append("")
                lines.extend(untrusted)
                lines.append("")
                lines.append("</details>")

            lines.append("")

        dry_run_failures = dry_run_only_failure_count(scores)
        if dry_run_failures:
            noun = "scenario" if dry_run_failures == 1 else "scenarios"
            lines.append(
                f"> :information_source: {dry_run_failures} failed {noun} tagged "
                f"`dry-run-only` (schema-coverage examples that don't run "
                f"cleanly against a generic CLI agent). Re-run with "
                f"`--tags real-runnable` to skip them."
            )
            lines.append("")

    return "\n".join(lines) + "\n"
