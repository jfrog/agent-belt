# (c) JFrog Ltd. (2026)

"""Markdown renderer for the benchmark card.

The card is appended to ``$GITHUB_STEP_SUMMARY`` (and rendered
standalone), so every cell that may carry externally-influenced content
- agent ``--version`` output captured from any binary on the runner's
``PATH``, fixture working-dir paths, judge model strings echoed back
from configuration, redacted environment values, free-form scenario
filter strings - is routed through :func:`belt._safe.md_inline`
for table-cell code spans or :func:`belt._safe.md_safe` for prose.
Without this the value ``"`# Inject heading`"`` in (e.g.) a malicious
``--version`` output would close the surrounding inline-code span and
emit a real H1 in the rendered Step Summary; an embedded pipe or
newline would split the table row.

This file is the only Markdown emitter in the benchmark-card package;
the escape policy itself lives in :mod:`belt._safe` so the
aggregator's other Markdown sinks share the same guarantees.
"""

from __future__ import annotations

from typing import Any, Iterable

from belt._safe import md_inline, md_safe
from belt.constants import BENCHMARK_CARD_JSON_FILE

from .entities import BenchmarkCard


def _md_kv_table(rows: list[tuple[str, str]]) -> str:
    """Two-column key/value table; values rendered as sanitised inline code."""
    return _emit_table(["Field", "Value"], [(md_safe(k), md_inline(v)) for k, v in rows])


def _emit_table(headers: list[str], rows: Iterable[tuple[str, ...]]) -> str:
    """Emit a GitHub-flavoured Markdown table.

    Headers are written verbatim (callers control them); cells are
    written verbatim too - the caller is responsible for routing
    user-influenced content through :func:`belt._safe.md_inline`
    or :func:`belt._safe.md_safe` before passing it in.
    Centralises the
    ``"| header | header |\\n|---|---|\\n| ..."`` boilerplate that
    repeated five times across the renderer.
    """
    rows_list = list(rows)
    if not rows_list:
        return ""
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows_list:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _format_optional(val: Any) -> str:
    """Render ``None`` / empty as a literal ``-`` placeholder."""
    if val is None:
        return "-"
    s = str(val)
    return s if s else "-"


def _format_pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def _format_secs(s: float | None) -> str:
    if s is None:
        return "-"
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s // 60)}m{int(s % 60):02d}s"


def render_markdown(card: BenchmarkCard) -> str:
    """Render a human-friendly snapshot of the card.

    Optimised for ``$GITHUB_STEP_SUMMARY``: tables for dense data,
    plain headings for navigation, and the headline pass/fail
    prominently at the top so a glance at the PR's CI summary tells the
    story.

    Every value that may carry externally-influenced content (agent
    ``--version`` output, working-dir paths, judge model strings,
    redacted env values, scenario filter strings) is routed through
    :func:`belt._safe.md_inline` or :func:`belt._safe.md_safe`
    before it reaches the rendered output. This is the same
    defence-in-depth contract used by every other Markdown sink in the
    project.
    """
    lines: list[str] = []

    status_emoji = "✅" if card.summary.overall_pass else "❌"
    lines.append(f"# Benchmark Card {status_emoji}")
    lines.append("")
    lines.append(
        f"**Run** {md_inline(card.run_id)} - {card.summary.passed}/{card.summary.total} "
        f"scenarios passed ({_format_pct(card.summary.pass_rate)}) in "
        f"{_format_secs(card.cost_timing.total_seconds)}."
    )
    lines.append("")

    lines.append("## Run identity")
    lines.append(
        _md_kv_table(
            [
                ("belt", card.belt.version),
                ("install kind", card.belt.install_kind),
                ("git SHA", _format_optional(card.belt.git_sha)),
                ("git dirty", _format_optional(card.belt.git_dirty)),
                ("started", card.started_at),
                ("ended", card.ended_at),
                ("host OS", card.host.os),
                ("machine", card.host.machine),
                ("python", f"{card.host.python_implementation} {card.host.python_version}"),
            ]
        )
    )
    lines.append("")

    lines.append("## Invocation")
    argv_rendered = " ".join(card.invocation.argv) if card.invocation.argv else "-"
    lines.append(f"- **Command**: {md_inline(argv_rendered)}")
    lines.append(f"- **Working dir**: {md_inline(_format_optional(card.invocation.cwd))}")
    if card.invocation.env:
        env_rows = [(k, v) for k, v in sorted(card.invocation.env.items())]
        lines.append("")
        lines.append("**Environment** (allow-listed, secret values redacted to `<set>`):")
        lines.append("")
        lines.append(_md_kv_table(env_rows))
    lines.append("")

    lines.append("## Scenarios")
    lines.append(f"- **Root**: {md_inline(card.scenarios.scenarios_root)}")
    if card.scenarios.selected_groups:
        lines.append(f"- **Selected groups**: {md_inline(', '.join(card.scenarios.selected_groups))}")
    if card.scenarios.selected_tags:
        lines.append(f"- **Selected tags**: {md_inline(', '.join(card.scenarios.selected_tags))}")
    if card.scenarios.excluded_tags:
        lines.append(f"- **Excluded tags**: {md_inline(', '.join(card.scenarios.excluded_tags))}")
    if card.scenarios.scenario_files:
        lines.append(
            f"- **Scenario files**: {len(card.scenarios.scenario_files)} files hashed "
            f"(see `{BENCHMARK_CARD_JSON_FILE}` for SHA-256 list)"
        )
    lines.append("")

    if card.fixtures:
        lines.append("## Fixtures")
        lines.append("")
        lines.append(
            _emit_table(
                ["Group", "Working dir", "Tracked", "Git SHA", "Ref", "Auto-init", "Dirty"],
                [
                    (
                        md_inline(f.group),
                        md_inline(_format_optional(f.working_dir)),
                        "yes" if f.tracked else "no",
                        md_inline(f.git_sha[:12] if f.git_sha else "-"),
                        md_inline(_format_optional(f.git_ref)),
                        "yes" if f.auto_initialized else "no",
                        str(f.dirty_files),
                    )
                    for f in card.fixtures
                ],
            )
        )
        lines.append("")

    if card.agents:
        lines.append("## Agents")
        lines.append("")
        lines.append(
            _emit_table(
                ["Group", "Agent", "Adapter", "CLI version", "CLI path", "Auth"],
                [
                    (
                        md_inline(a.group),
                        md_inline(a.agent.name),
                        md_inline(a.agent.adapter_class),
                        md_inline(_format_optional(a.cli.version)),
                        md_inline(_format_optional(a.cli.binary_path)),
                        md_safe(", ".join(a.agent.auth_signals) if a.agent.auth_signals else "-"),
                    )
                    for a in card.agents
                ],
            )
        )
        for a in card.agents:
            if a.agent.args:
                pretty = ", ".join(f"{k}={v}" for k, v in sorted(a.agent.args.items()))
                lines.append(f"- **{md_safe(a.group)} agent args**: {md_inline(pretty)}")
        lines.append("")

    lines.append("## Scoring")
    lines.append(f"- **Modes**: {md_inline(', '.join(card.scoring.modes) if card.scoring.modes else '-')}")
    if card.scoring.consensus:
        lines.append(f"- **Consensus**: {md_inline(card.scoring.consensus)}")
    if card.scoring.thresholds:
        thr = ", ".join(f"{k}={v}" for k, v in sorted(card.scoring.thresholds.items()))
        lines.append(f"- **Thresholds**: {md_inline(thr)}")
    if card.scoring.judges:
        lines.append("")
        lines.append("**Judges**:")
        lines.append("")
        lines.append(
            _emit_table(
                ["Provider", "Model", "Base URL", "Dimensions"],
                [
                    (
                        md_inline(j.provider),
                        md_inline(j.model),
                        md_inline(_format_optional(j.base_url)),
                        md_safe(", ".join(j.dimensions) if j.dimensions else "-"),
                    )
                    for j in card.scoring.judges
                ],
            )
        )
    lines.append("")

    lines.append("## Runtime")
    lines.append(
        _md_kv_table(
            [
                ("workers", str(card.runtime.workers)),
                ("trials", str(card.runtime.trials)),
                ("streaming", "yes" if card.runtime.streaming else "no"),
                ("scenario_delay_s", str(card.runtime.scenario_delay_s)),
            ]
        )
    )
    lines.append("")

    if card.agent_errors:
        # Card consumers (CI dashboards, regression trackers) can rely on
        # this section's presence to flag a run where the agent itself
        # did not really run - rules-only "pass rate" headlines should be
        # ignored or annotated when this block appears.
        lines.append("## Agent errors")
        lines.append("")
        ae = card.agent_errors
        kv_rows = [
            ("scenarios with errors", f"{ae.scenarios_with_errors}/{ae.scenarios_total}"),
            ("vacuous passes", str(ae.vacuous_passes)),
        ]
        if ae.task_quality is not None:
            tq = ae.task_quality
            pct_str = f"{tq.pct}%" if tq.pct is not None else "N/A"
            kv_rows.append(("task quality (passed/completed)", f"{tq.passed}/{tq.completed} ({pct_str})"))
            kv_rows.append(("environmental failures", str(tq.env_failed)))
            kv_rows.append(("  agent-axis env failures", str(tq.env_failed_agent)))
            kv_rows.append(("  judge-axis env failures", str(tq.env_failed_judge)))
            kv_rows.append(("agent task failures", str(tq.task_failed)))
        type_rows = [(md_safe(t), str(c)) for t, c in sorted(ae.by_error_type.items(), key=lambda kv: (-kv[1], kv[0]))]
        lines.append(_md_kv_table(kv_rows))
        lines.append("")
        lines.append("**By error type**:")
        lines.append("")
        lines.append(_emit_table(["Error type", "Scenarios"], type_rows))
        if ae.remediation:
            lines.append("")
            lines.append(f"> {md_safe(ae.remediation)}")
        lines.append("")

    if card.judge_errors:
        # Surfaced as a sibling block to ``agent_errors`` so a dashboard
        # can attribute environmental failures to the right backend
        # (provider key vs judge key, agent CLI auth vs judge backend
        # auth) without collapsing them into a single env_failed bucket.
        lines.append("## Judge errors")
        lines.append("")
        je = card.judge_errors
        kv_rows = [
            ("scenarios with errors", f"{je.scenarios_with_errors}/{je.scenarios_total}"),
        ]
        type_rows = [(md_safe(t), str(c)) for t, c in sorted(je.by_error_type.items(), key=lambda kv: (-kv[1], kv[0]))]
        lines.append(_md_kv_table(kv_rows))
        lines.append("")
        lines.append("**By error type**:")
        lines.append("")
        lines.append(_emit_table(["Error type", "Scenarios"], type_rows))
        lines.append("")

    lines.append("## Summary")
    lines.append(
        _md_kv_table(
            [
                ("total", str(card.summary.total)),
                ("passed", str(card.summary.passed)),
                ("failed", str(card.summary.failed)),
                ("pass rate", _format_pct(card.summary.pass_rate)),
                ("overall pass", "yes" if card.summary.overall_pass else "no"),
                ("thresholds passed", _format_optional(card.summary.thresholds_passed)),
                ("agent cost (USD)", _format_optional(card.cost_timing.agent_cost_usd)),
                ("judge cost (USD)", _format_optional(card.cost_timing.judge_cost_usd)),
                ("total cost (USD)", _format_optional(card.cost_timing.total_cost_usd)),
                ("total wall time", _format_secs(card.cost_timing.total_seconds)),
                ("mean per scenario", _format_secs(card.cost_timing.mean_seconds)),
            ]
        )
    )
    lines.append("")

    if card.links:
        lines.append("## Artifacts")
        for name, path in sorted(card.links.items()):
            # ``name`` is a fixed artifact filename (run_meta.json,
            # results.json, eval.log) - safe to render as literal.
            # ``path`` is the run_dir path which may contain
            # user-controlled segments; route it through the
            # inline-code sanitiser.
            lines.append(f"- `{name}`: {md_inline(path)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
