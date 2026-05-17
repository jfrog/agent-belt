# (c) JFrog Ltd. (2026)

"""Rich/terminal renderer for aggregated scores.

Builds the compact result panel (one row per scenario) plus the failure
detail block.  All untrusted strings (group/scenario names, rule details,
LLM judge reasoning) pass through ``rich_safe`` so they cannot inject
Rich markup.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from belt._logging import resolve_terminal_level
from belt._safe import rich_safe
from belt.constants import ERROR_PATTERNS, MAX_TURNS_PER_SCENARIO, TURN_CLI_TEMPLATE, TURN_OUTPUT_TEMPLATE
from belt.entities import ScenarioScore, TurnOutput
from belt.scorer.display import VERDICT_DISPLAY, verdict_label
from belt.scorer.entities import DOWNGRADE_VERDICT_SET
from belt.scorer.payloads import LLMPayload, RulesPayload

# Loguru level names at or below which the renderer should show the
# inline judge reasoning, the response tail, and the verbose rule-by-rule
# trajectory diagnostics. Everything above this stays in
# ``<run_dir>/eval.log`` and the per-scenario ``score.json`` - reachable
# via ``belt view`` and a tail of the log file.
_VERBOSE_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "TRACE"})


def _is_verbose() -> bool:
    """Return True iff the terminal handler is at INFO/DEBUG/TRACE level."""
    return resolve_terminal_level(0) in _VERBOSE_LEVELS


from . import dry_run_only_failure_count
from .stats import compute_partial_score
from .thresholds import discover_llm_dimensions

SCORE_EMOJI: dict[str, str] = {token: verdict_label(token) for token in VERDICT_DISPLAY}
"""``token → "<icon> <token>"`` table for result-panel cells. Derived
from :data:`belt.scorer.display.VERDICT_DISPLAY` so icons stay in
sync across the terminal renderer, ``belt view``, and the markdown
step-summary."""


def _rules_summary(score: ScenarioScore) -> str:
    rules = score.scores.get("rules")
    if not isinstance(rules, RulesPayload):
        return "-"
    total = len(rules.checks)
    passed = sum(1 for c in rules.checks if c.passed is True)
    return f"{passed}/{total}"


def _llm_dim(score: ScenarioScore, dim: str) -> str:
    llm = score.scores.get("llm")
    if not isinstance(llm, LLMPayload):
        return "-"
    verdict = llm.dimensions.get(dim)
    if verdict is None:
        return "-"
    return SCORE_EMOJI.get(verdict.score, "-")


def build_result_table(scores: list[ScenarioScore]) -> list[str]:
    """Build a compact result table showing all scenarios with rules + LLM scores."""
    if not scores:
        return []

    has_rules = any("rules" in s.scores for s in scores)
    llm_dims = discover_llm_dimensions(scores)

    rows: list[tuple[str, ...]] = []
    for s in scores:
        name = f"{s.group}/{s.scenario_name}"
        icon = "✅" if s.overall_pass else "❌"
        cells: list[str] = [f"{icon} {name}"]
        if has_rules:
            cells.append(_rules_summary(s))
        for dim in llm_dims:
            cells.append(_llm_dim(s, dim))
        rows.append(tuple(cells))

    headers: list[str] = ["Scenario"]
    if has_rules:
        headers.append("Rules")
    headers.extend(llm_dims)

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    lines: list[str] = []
    header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    lines.append(f"  {header_line}")
    sep_line = "  ".join("─" * col_widths[i] for i in range(len(headers)))
    lines.append(f"  {sep_line}")
    for row in rows:
        row_line = "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
        lines.append(f"  {row_line}")
    return lines


def _extract_error_lines(cli_output: str, max_lines: int = 12) -> list[str]:
    """Return the lines at and after the first error pattern match."""
    lines = cli_output.splitlines()
    for i, line in enumerate(lines):
        if any(p in line.lower() for p in ERROR_PATTERNS):
            return [stripped for line in lines[i : i + max_lines] if (stripped := line.strip())]
    return []


def _load_turn_output(outcome_dir: Path, turn_idx: int) -> TurnOutput | None:
    """Load TurnOutput from the persisted output JSON file."""
    output_path = outcome_dir / TURN_OUTPUT_TEMPLATE.format(turn_idx)
    if not output_path.exists():
        return None
    try:
        return TurnOutput.model_validate_json(output_path.read_text())
    except Exception:
        return None


def _agent_error_context(outcome_dir: Path) -> list[str]:
    """Surface ``has_error`` / ``error_type`` / ``reply_text`` from each erroring turn.

    Rendered before any rule-failure context so a user reading top-down
    sees the agent's own error message (e.g. ``"Not logged in · Please
    run /login"``) before the rule-by-rule diagnostics.
    """
    lines: list[str] = []
    for i in range(MAX_TURNS_PER_SCENARIO):
        path = outcome_dir / TURN_OUTPUT_TEMPLATE.format(i)
        if not path.exists():
            break
        try:
            to = TurnOutput.model_validate_json(path.read_text())
        except Exception:
            continue
        if not to.has_error:
            continue
        etype = rich_safe(to.error_type or "unknown")
        lines.append(f"       [red]agent error:[/red] {etype}")
        if to.reply_text and to.reply_text.strip():
            first_line = to.reply_text.strip().splitlines()[0]
            # Cap at 200 chars so a multi-KB reply (full streaming tail)
            # doesn't blow out the panel; the full text is one ``→`` away.
            if len(first_line) > 200:
                first_line = first_line[:200] + "..."
            lines.append(f'         "{rich_safe(first_line)}"')
    return lines


def _failure_context(s: ScenarioScore, outcome_dir: Path) -> list[str]:
    """Diagnostic lines for each failing check, reading turn files directly."""
    rules = s.scores.get("rules")
    if not isinstance(rules, RulesPayload):
        return []
    failing = [c for c in rules.checks if c.passed is False]
    if not failing:
        return []

    lines: list[str] = []
    by_turn: dict[int, list] = {}
    for c in failing:
        if c.turn_idx is not None:
            by_turn.setdefault(c.turn_idx, []).append(c)

    for ti, checks in sorted(by_turn.items()):
        cli_path = outcome_dir / TURN_CLI_TEMPLATE.format(ti)
        cli_output = cli_path.read_text() if cli_path.exists() else ""
        turn_output = _load_turn_output(outcome_dir, ti)

        for c in checks:
            check = c.check
            dim = c.dimension

            if check == "no_errors":
                error_lines = _extract_error_lines(cli_output)
                if error_lines:
                    lines.append(f"              turn {ti} error:")
                    for el in error_lines:
                        lines.append(f"                {el}")

            elif dim == "trajectory" and check.startswith("tool_invoked"):
                if turn_output:
                    actual = sorted({tc.name for tc in turn_output.tool_calls})
                else:
                    actual = []
                label = ", ".join(actual) if actual else "(no tools)"
                lines.append(f"              turn {ti} tools invoked: {label}")

            elif dim == "response" and check.startswith("contains("):
                snippet = cli_output.strip()[-500:] if cli_output.strip() else "(empty)"
                lines.append(f"              turn {ti} response (last 500 chars):")
                lines.append(f"                {snippet}")

    return lines


def _evidence_paths(s: ScenarioScore, outcome_dir: Path) -> list[Path]:
    """Existing evidence files (output JSON + CLI log) for failing turns."""
    rules = s.scores.get("rules")
    if not isinstance(rules, RulesPayload):
        return []
    failing = [c for c in rules.checks if c.passed is False]
    turn_indices = sorted({c.turn_idx for c in failing if c.turn_idx is not None})

    paths: list[Path] = []
    for ti in turn_indices:
        for template in (TURN_OUTPUT_TEMPLATE, TURN_CLI_TEMPLATE):
            p = outcome_dir / template.format(ti)
            if p.exists():
                paths.append(p)
    return paths


def print_terminal(
    scores: list[ScenarioScore],
    run_label: str,
    threshold_lines: list[str] | None = None,
    outcomes_root: Path | None = None,
    cost_timing: dict | None = None,
    reliability: dict | None = None,
    agent_errors: dict | None = None,
    judge_errors: dict | None = None,
    console: Console | None = None,
) -> None:
    """Render the full terminal report (panel + failures + thresholds).

    ``agent_errors`` (typically from
    :func:`belt.aggregator.stats.collect_agent_errors`) drives the
    "Passed with agent errors" section that lists scenarios whose rules
    passed despite an erroring turn (vacuous pass).

    ``judge_errors`` (from :func:`belt.aggregator.stats.collect_judge_errors`)
    drives the analogous "Judge infrastructure failures" section so a
    scenario whose rules passed while the LLM judge timed out / was rate
    limited is rendered as an environmental failure rather than buried
    under "All N passed".
    """
    from belt.progress import phase_header

    # All UI goes to stderr. Stdout stays empty for future ``--output``
    # modes (json/markdown piping) and so ``belt eval ... > /dev/null``
    # keeps the human-readable UI visible. Matches the convention used
    # by ``gh``, ``kubectl``, ``inspect``, ``promptfoo``.
    con = console or Console(stderr=True)

    phase_header(con, "Results")

    failed_scores = [s for s in scores if not s.overall_pass]

    table_lines = build_result_table(scores)

    footer_parts: list[str] = []
    partial = compute_partial_score(scores)
    if partial is not None:
        pct = partial["partial_score"] * 100
        cp, ct = partial["checks_passed"], partial["checks_total"]
        # If agent errors were detected, the partial-score number is
        # actively misleading (rules pass vacuously when the agent never
        # ran). Annotate inline so the headline number can't be read in
        # isolation.
        if agent_errors:
            footer_parts.append(f"[yellow]{cp}/{ct} checks ({pct:.0f}%) - agent errors detected[/yellow]")
        else:
            footer_parts.append(f"{cp}/{ct} checks ({pct:.0f}%)")
    if cost_timing:
        agent_cost = cost_timing.get("agent_cost_usd")
        judge_cost = cost_timing.get("judge_cost_usd")
        total_cost_val = cost_timing.get("total_cost_usd")
        if agent_cost is not None and judge_cost is not None:
            footer_parts.append(
                f"Agent: [green]${agent_cost:.4f}[/green] · "
                f"Judge: [green]${judge_cost:.4f}[/green] · "
                f"Total: [green]${total_cost_val:.4f}[/green]"
            )
        elif agent_cost is not None:
            footer_parts.append(f"Agent: [green]${agent_cost:.4f}[/green]")
        elif judge_cost is not None:
            footer_parts.append(f"Judge: [green]${judge_cost:.4f}[/green]")
        elif total_cost_val is not None:
            footer_parts.append(f"Total: [green]${total_cost_val:.4f}[/green]")
        if "total_seconds" in cost_timing:
            secs = cost_timing["total_seconds"]
            mins, rem = divmod(secs, 60)
            footer_parts.append(f"{int(mins)}m{rem:.0f}s" if mins else f"{secs:.1f}s")

    if reliability:
        p1 = reliability["mean_pass_at_1"] * 100
        at3 = reliability["mean_pass_at_3"] * 100
        at8 = reliability["mean_pass_at_8"] * 100
        pow3 = reliability["mean_pass_pow_3"] * 100
        pow8 = reliability["mean_pass_pow_8"] * 100
        footer_parts.append(
            f"pass@1={p1:.0f}% pass@3={at3:.0f}% pass@8={at8:.0f}% " f"pass^3={pow3:.0f}% pass^8={pow8:.0f}%"
        )

    panel_lines = list(table_lines)
    if footer_parts:
        panel_lines.append("")
        panel_lines.append(f"  [dim]{' · '.join(footer_parts)}[/dim]")

    panel_content = "\n".join(panel_lines) if panel_lines else "  No scenarios scored."

    try:
        panel = Panel(
            Text.from_markup(panel_content),
            border_style="dim",
            padding=(0, 1),
        )
    except Exception:
        panel = Panel(panel_content, border_style="dim", padding=(0, 1))
    con.print(panel)

    if failed_scores:
        con.print("  [bold]Failed:[/bold]")
        verbose = _is_verbose()
        for i, s in enumerate(failed_scores, 1):
            # ``group`` / ``scenario_name`` are regex-bound by the schema, but
            # we still escape defensively so any future call site that bypasses
            # validation can't inject Rich markup into the panel.
            rules = s.scores.get("rules")
            llm = s.scores.get("llm")

            # Collapse the failed rules and downgraded LLM dims onto the same
            # line as the scenario name. The pre-compaction layout printed
            # scenario / rule / llm on three lines each (six lines for two
            # failed scenarios) which dominated the terminal surface for
            # what is, semantically, two facts per scenario.
            line_parts: list[str] = []
            failed_rules: list[str] = []
            if isinstance(rules, RulesPayload):
                for c in rules.checks:
                    if c.passed is False:
                        # Rule details may quote agent stdout - always escape.
                        detail = f" ({rich_safe(c.details)})" if c.details else ""
                        failed_rules.append(f"{rich_safe(c.dimension)}/{rich_safe(c.check)}{detail}")

            llm_downgraded: list[str] = []
            if isinstance(llm, LLMPayload):
                llm_downgraded = sorted(
                    f"{dim}={llm.dimensions[dim].score}"
                    for dim in llm.dimensions
                    if llm.dimensions[dim].score in DOWNGRADE_VERDICT_SET
                )

            if failed_rules:
                line_parts.append(f"[dim]rule:[/dim] {'; '.join(failed_rules)}")
            if llm_downgraded:
                line_parts.append(f"[dim]llm:[/dim] {', '.join(rich_safe(x) for x in llm_downgraded)}")

            scenario_label = rich_safe(s.scenario_name)
            if line_parts:
                con.print(f"    {i}. {scenario_label}  {'   '.join(line_parts)}")
            else:
                con.print(f"    {i}. {scenario_label}")

            # Verbose-only: judge reasoning prose, one paragraph per failed
            # dimension. Heaviest single contributor to terminal noise on a
            # failing run; one ``belt view`` away in ``score.json`` otherwise.
            if verbose and isinstance(llm, LLMPayload):
                for dim in sorted(llm.dimensions):
                    verdict = llm.dimensions[dim]
                    if verdict.score in DOWNGRADE_VERDICT_SET:
                        con.print(
                            f"       [dim]llm {rich_safe(dim)} ({rich_safe(verdict.score)}):[/dim] "
                            f"{rich_safe(verdict.reasoning)}"
                        )

            if isinstance(llm, LLMPayload) and llm.consensus_meta and llm.consensus_meta.disagreements:
                disagreements = llm.consensus_meta.disagreements
                dims = ", ".join(rich_safe(d.get("dimension", "")) for d in disagreements if isinstance(d, dict))
                con.print(f"       [dim]consensus:[/dim] {len(disagreements)} disagreement(s) [{dims}]")

            if outcomes_root is not None:
                outcome_dir = outcomes_root / s.group / s.scenario_name
                # Lead with the agent's own error message - if the
                # subprocess crashed or returned 401, the rule-level
                # context below (rule names, response tails) is noise
                # until the user knows the agent didn't really run.
                for ctx_line in _agent_error_context(outcome_dir):
                    con.print(ctx_line)
                # ``_failure_context`` includes the last 500 chars of CLI
                # output per failing turn. That's exactly the kind of
                # high-volume debug text that drove the original
                # ~840-line failure surface; keep it gated on the
                # explicit ``-v`` opt-in.
                if verbose:
                    for ctx_line in _failure_context(s, outcome_dir):
                        con.print(ctx_line)
                evidence = _evidence_paths(s, outcome_dir)
                if evidence and verbose:
                    for ep in evidence:
                        con.print(f"       [dim]->[/dim] {ep}")

        dry_run_failures = dry_run_only_failure_count(scores)
        if dry_run_failures:
            noun = "scenario" if dry_run_failures == 1 else "scenarios"
            # Compact one-liner. The "schema-coverage examples..." rationale
            # is documented in WRITING-SCENARIOS.md; surfacing it inline
            # tripled the line count for what is fundamentally a hint.
            con.print(
                f"  [dim]Note: {dry_run_failures} [/dim][bold]dry-run-only[/bold]"
                f"[dim] {noun}. Re-run with [/dim][bold]--tags real-runnable"
                f"[/bold][dim] to skip.[/dim]"
            )
        con.print()

    # Vacuous passes: scenarios whose ``overall_pass=true`` rules score
    # masked an underlying agent error. Without this block, a 5/7-pass
    # run looks "decent partial" while every turn 401'd.
    vacuous_scenarios = [
        s.get("scenario") for s in (agent_errors or {}).get("per_scenario", []) if s.get("vacuous_pass")
    ]
    if vacuous_scenarios and outcomes_root is not None:
        con.print("  [yellow bold]Passed with agent errors (vacuous):[/yellow bold]")
        score_by_path = {f"{s.group}/{s.scenario_name}": s for s in scores}
        for i, path in enumerate(vacuous_scenarios, 1):
            score = score_by_path.get(path)
            if score is None:
                continue
            con.print(
                f"    {i}. {rich_safe(score.group)}/{rich_safe(score.scenario_name)} "
                "[dim](rules passed but agent errored)[/dim]"
            )
            outcome_dir = outcomes_root / score.group / score.scenario_name
            for ctx_line in _agent_error_context(outcome_dir):
                con.print(ctx_line)
        con.print()

    # Judge-infra failures: scenarios where the LLM judge backend itself
    # produced no verdict (rate-limit, timeout, network, parse failure).
    # Surface the affected scenarios so a reader can see WHICH ones lost
    # their judge verdict, not just an aggregate count in the headline.
    judge_per_scenario = (judge_errors or {}).get("per_scenario") or []
    if judge_per_scenario:
        con.print("  [yellow bold]Judge infrastructure failures:[/yellow bold]")
        for i, entry in enumerate(judge_per_scenario, 1):
            scenario = rich_safe(entry.get("scenario", ""))
            etype = rich_safe(entry.get("error_type", "other"))
            con.print(f"    {i}. {scenario} [dim]({etype})[/dim]")
        con.print()

    if threshold_lines:
        con.print("  [bold]Thresholds:[/bold]")
        for line in threshold_lines:
            con.print(line)
        con.print()

    # Single artifact pointer. `belt view` opens the canonical viewer, the
    # one place all per-scenario artifacts (results.json, score.json,
    # benchmark-card, turn outputs, eval.log) are reachable. We used to print
    # each of those paths on its own line; the SOTA (inspect view /
    # promptfoo view) is one viewer pointer, mirrored here. The `-v` hint
    # tags on inline only when the terminal was kept quiet AND something
    # interesting was actually hidden (a failed scenario).
    hint = "  [dim](pass -v for details)[/dim]" if failed_scores and not _is_verbose() else ""
    con.print(f"  [dim]→ belt view {run_label}[/dim]{hint}")
