# (c) JFrog Ltd. (2026)

"""Terminal-based results viewer for belt outcome artifacts.

Reads filesystem artifacts only - no runner/scorer imports (Principle 1: Phase independence).

Subcommand: ``belt view [outcomes-dir]``

Artifacts read:
    results.json       - top-level run summary
    score.json         - per-scenario scoring (rules + LLM dimensions)
    turn_N_output.json - TurnOutput: reply, tool calls, cost, timing
    turn_N_cli.txt     - raw CLI output for a turn

Navigation:
    Summary table → pick a scenario number → per-turn detail panels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from belt._safe import rich_safe
from belt.constants import (
    MAX_TURNS_PER_SCENARIO,
    OUTCOMES_ROOT,
    RESULTS_FILE,
    SCORE_FILE,
    TURN_CLI_TEMPLATE,
    TURN_OUTPUT_TEMPLATE,
)
from belt.entities import ScenarioScore, TurnOutput
from belt.scorer.display import UNKNOWN_VERDICT_DISPLAY, VERDICT_DISPLAY
from belt.scorer.payloads import RulesPayload, iter_llm_payloads, iter_llm_verdicts

# ── LLM score display ──

# Verdict-token → (icon, Rich color). Single source of truth lives in
# :mod:`belt.scorer.display`; the local alias keeps the call sites
# below terse without re-spelling the mapping.
_SCORE_STYLE = VERDICT_DISPLAY


# ── Directory discovery ──


def find_latest_outcomes_dir(base: Path) -> Optional[Path]:
    """Return the most recently modified subdirectory under *base*.

    Mirrors the pattern in ``watch._find_latest_run``.  Returns ``None`` if
    *base* does not exist or contains no subdirectories.
    """
    try:
        if not base.is_dir():
            return None
        candidates = sorted(
            (p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None
    return candidates[0] if candidates else None


# ── Data loading ──


def _load_results_cost_timing(outcomes_dir: Path) -> dict[str, tuple[Optional[float], Optional[float]]]:
    """Index aggregator's per-scenario cost/timing by ``"<group>/<scenario_name>"``.

    Returns an empty dict if ``results.json`` is missing or malformed - callers
    fall back to walking ``turn_*_output.json`` files.  Reading the aggregator
    output once is cheaper than re-parsing every turn file with Pydantic on
    every ``view`` invocation (a 100-scenario x 5-turn run is ~500 parses).
    """
    results_path = outcomes_dir / RESULTS_FILE
    try:
        results = json.loads(results_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    index: dict[str, tuple[Optional[float], Optional[float]]] = {}
    for entry in (results.get("cost_timing") or {}).get("scenarios") or []:
        if not isinstance(entry, dict):
            continue
        key = entry.get("scenario")
        if not isinstance(key, str):
            continue
        index[key] = (entry.get("agent_cost_usd"), entry.get("total_seconds"))
    return index


def _read_run_setup_errors(outcomes_dir: Path) -> list[dict[str, Any]]:
    """Read ``setup_errors`` from ``results.json`` (or the run-phase sidecar).

    ``belt aggregate`` is the canonical writer; the ``setup_errors.json``
    sidecar is the fallback path that keeps the banner visible when the
    aggregator never ran (e.g. all groups failed setup, no scenarios to
    score). Both forms share the same record shape - ``{"group",
    "scenarios", "error"}`` - so the renderer doesn't branch.
    """
    results_json = outcomes_dir / "results.json"
    if results_json.exists():
        try:
            data = json.loads(results_json.read_text())
            if isinstance(data, dict):
                errs = data.get("setup_errors") or []
                if isinstance(errs, list):
                    return [e for e in errs if isinstance(e, dict)]
        except Exception:
            pass
    sidecar = outcomes_dir / "setup_errors.json"
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text())
            if isinstance(data, list):
                return [e for e in data if isinstance(e, dict)]
        except Exception:
            pass
    return []


def _render_setup_errors_banner(setup_errors: list[dict[str, Any]], console: Console) -> None:
    """Print a one-banner-per-failed-group block above the results table.

    Setup-failed groups are otherwise invisible in ``belt view`` because
    they have no per-scenario score.json. Rendering them here closes the
    visibility gap reported in #385 (1.4) without inventing fake score
    rows.
    """
    total_skipped = sum(len(e.get("scenarios") or []) for e in setup_errors)
    header = (
        f"[red bold]Setup errors:[/red bold] "
        f"[red]{len(setup_errors)} group(s), {total_skipped} scenario(s) skipped[/red]"
    )
    console.print()
    console.print(header)
    for entry in setup_errors:
        group = entry.get("group", "?")
        scenarios = entry.get("scenarios") or []
        err = entry.get("error", "")
        names = ", ".join(scenarios) if scenarios else "-"
        console.print(f"  [red]\u2717[/red] [bold]{group}[/bold]: {err}")
        console.print(f"      [dim]skipped: {names}[/dim]")
    console.print()


def load_summary(outcomes_dir: Path) -> list[dict[str, Any]]:
    """Read all score.json files under *outcomes_dir* and return a list of summary dicts.

    Cost/timing prefer the aggregator's ``results.json`` (one read per run).
    If ``results.json`` is missing - e.g. user ran ``score`` without
    ``aggregate`` - falls back to summing ``turn_*_output.json`` per scenario.

    Each dict contains:
        scenario     - "group/name"
        pass         - bool
        rules        - "N/M" or "-"
        llm_dims     - dict[dim, score_str]  (e.g. {"execution": "high"})
        cost_usd     - float | None
        total_secs   - float | None
        tags         - list[str]
        group        - str
        name         - str
        outcome_dir  - Path
    """
    summaries: list[dict[str, Any]] = []
    cost_timing_index = _load_results_cost_timing(outcomes_dir)

    for score_path in sorted(outcomes_dir.rglob(SCORE_FILE)):
        try:
            score = ScenarioScore.model_validate_json(score_path.read_text())
        except Exception:
            continue

        outcome_dir = score_path.parent
        cost_usd: Optional[float] = None
        total_secs: Optional[float] = None

        index_key = f"{score.group}/{score.scenario_name}"
        if index_key in cost_timing_index:
            cost_usd, total_secs = cost_timing_index[index_key]
        else:
            for turn_idx in range(MAX_TURNS_PER_SCENARIO):
                output_path = outcome_dir / TURN_OUTPUT_TEMPLATE.format(turn_idx)
                if not output_path.exists():
                    break
                try:
                    to = TurnOutput.model_validate_json(output_path.read_text())
                    if to.cost_usd is not None:
                        cost_usd = (cost_usd or 0.0) + to.cost_usd
                    if to.timing and to.timing.total is not None:
                        total_secs = (total_secs or 0.0) + to.timing.total
                except Exception:
                    continue

        rules = score.scores.get("rules")
        if isinstance(rules, RulesPayload):
            checks = rules.checks
            total_checks = len(checks)
            passed_checks = sum(1 for c in checks if c.passed is True)
            rules_str = f"{passed_checks}/{total_checks}" if total_checks else "-"
        else:
            rules_str = "-"

        # Walk every LLM-shaped payload so per-judge and per-turn
        # verdicts both populate the summary table. Same-dim from
        # multiple judges: keep the last write (insertion order matches
        # config order).
        llm_dims: dict[str, str] = {}
        for _name, payload in iter_llm_payloads(score):
            for dim, score_token, _r in iter_llm_verdicts(payload):
                if score_token:
                    llm_dims[dim] = score_token

        # Tags: attempt to read from a sibling scenario JSON file
        tags: list[str] = _load_tags(outcome_dir, score.scenario_name)

        summaries.append(
            {
                "scenario": f"{score.group}/{score.scenario_name}",
                "pass": score.overall_pass,
                "rules": rules_str,
                "llm_dims": llm_dims,
                "cost_usd": cost_usd,
                "total_secs": total_secs,
                "tags": tags,
                "group": score.group,
                "name": score.scenario_name,
                "outcome_dir": outcome_dir,
            }
        )

    return summaries


def _load_tags(outcome_dir: Path, scenario_name: str) -> list[str]:
    """Best-effort read of tags from a scenario JSON file adjacent to the outcome dir.

    The scenario file is not guaranteed to exist (runs may have been moved),
    so any failure returns an empty list.
    """
    # Scenario JSON may live in siblings like scenarios/group/scenario_name.json
    # We can't reliably locate it without running the parser, so we skip gracefully.
    return []


def _count_turns(outcome_dir: Path) -> int:
    """Count how many turn_N_output.json files exist in *outcome_dir*."""
    count = 0
    for i in range(MAX_TURNS_PER_SCENARIO):
        if (outcome_dir / TURN_OUTPUT_TEMPLATE.format(i)).exists():
            count += 1
        else:
            break
    return count


# ── Display helpers ──


def _pass_icon(passed: bool) -> Text:
    token = "pass" if passed else "fail"
    display = VERDICT_DISPLAY[token]
    return Text(f"{display.icon} {token}", style=display.color)


def _fmt_cost(cost: Optional[float]) -> str:
    if cost is None:
        return "-"
    return f"${cost:.4f}"


def _fmt_secs(secs: Optional[float]) -> str:
    if secs is None:
        return "-"
    if secs >= 60:
        m, s = divmod(secs, 60)
        return f"{int(m)}m{s:.0f}s"
    return f"{secs:.1f}s"


def _fmt_llm_dims(llm_dims: dict[str, str]) -> str:
    if not llm_dims:
        return "-"
    parts = []
    for dim, score_val in sorted(llm_dims.items()):
        icon, _ = _SCORE_STYLE.get(score_val, UNKNOWN_VERDICT_DISPLAY)
        parts.append(f"{icon} {dim}")
    return "  ".join(parts)


# ── Summary table ──


def show_summary_table(summaries: list[dict[str, Any]], console: Optional[Console] = None) -> None:
    """Render a Rich summary table for all scenarios in *summaries*."""
    con = console or Console(stderr=True)

    table = Table(
        title="belt results",
        show_lines=False,
        header_style="bold",
        border_style="dim",
        pad_edge=True,
    )
    table.add_column("#", style="dim", width=3, no_wrap=True)
    table.add_column("Scenario", min_width=24, no_wrap=False)
    table.add_column("Result", width=8, no_wrap=True)
    table.add_column("Rules", width=7, no_wrap=True, justify="right")
    table.add_column("LLM", min_width=14, no_wrap=False)
    table.add_column("Cost", width=9, no_wrap=True, justify="right")
    table.add_column("Time", width=7, no_wrap=True, justify="right")

    for idx, s in enumerate(summaries, 1):
        result_text = _pass_icon(s["pass"])
        llm_str = _fmt_llm_dims(s["llm_dims"])
        table.add_row(
            str(idx),
            s["scenario"],
            result_text,
            s["rules"],
            llm_str,
            _fmt_cost(s["cost_usd"]),
            _fmt_secs(s["total_secs"]),
        )

    con.print()
    con.print(table)


# ── Per-turn detail ──


def show_scenario_detail(
    outcomes_dir: Path,
    scenario_name: str,
    console: Optional[Console] = None,
) -> None:
    """Render per-turn detail panels for *scenario_name* inside *outcomes_dir*.

    Displays:
        - Score summary (rules checks, LLM dimensions with reasoning)
        - Per-turn: message context, tool calls, reply text, raw CLI output (collapsed)
    """
    con = console or Console(stderr=True)

    # Locate outcome dir: look for group/scenario_name structure
    outcome_dir: Optional[Path] = None
    score: Optional[ScenarioScore] = None

    for score_path in sorted(outcomes_dir.rglob(SCORE_FILE)):
        try:
            candidate = ScenarioScore.model_validate_json(score_path.read_text())
        except Exception:
            continue
        full_name = f"{candidate.group}/{candidate.scenario_name}"
        if full_name == scenario_name or candidate.scenario_name == scenario_name:
            outcome_dir = score_path.parent
            score = candidate
            break

    if outcome_dir is None or score is None:
        con.print(f"[red]Scenario not found:[/red] {scenario_name}")
        return

    con.print()
    con.print(
        Panel(
            f"[bold]{rich_safe(score.group)}[/bold] / [bold]{rich_safe(score.scenario_name)}[/bold]",
            border_style="blue",
        )
    )

    # Score summary panel
    _show_score_summary(con, score)

    # Per-turn panels
    num_turns = _count_turns(outcome_dir)
    if num_turns == 0:
        con.print("[dim]No turn output files found.[/dim]")
        return

    for turn_idx in range(num_turns):
        _show_turn_detail(con, outcome_dir, turn_idx)


def _show_score_summary(con: Console, score: ScenarioScore) -> None:
    """Render a scoring summary panel for the scenario."""
    lines: list[str] = []

    overall_icon = "✅" if score.overall_pass else "❌"
    lines.append(f"{overall_icon} Overall: {'pass' if score.overall_pass else 'fail'}")
    lines.append("")

    rules = score.scores.get("rules")
    if isinstance(rules, RulesPayload):
        checks = rules.checks
        passed = sum(1 for c in checks if c.passed is True)
        lines.append(f"[bold]Rules:[/bold] {passed}/{len(checks)} checks passed")
        for c in checks:
            icon = "✅" if c.passed is True else "❌"
            # Rule details may quote agent stdout - escape before letting them
            # through ``Text.from_markup``.
            detail = f" - {rich_safe(c.details)}" if c.details else ""
            dim = rich_safe(c.dimension or "?")
            chk = rich_safe(c.check or "?")
            turn_label = f" (turn {c.turn_idx})" if c.turn_idx is not None else ""
            lines.append(f"  {icon} \\[{dim}] {chk}{detail}{turn_label}")

    # Walk every LLM-shaped payload so the LLM section surfaces all
    # judges (multi-judge: judge_a + judge_b) and all per-turn rollups.
    # Per-judge header keeps the rubrics attributable.
    for name, payload in iter_llm_payloads(score):
        header = "LLM scoring" if name == "llm" else f"LLM scoring [{rich_safe(name)}]"
        lines.append("")
        lines.append(f"[bold]{header}:[/bold]")
        for dim, score_token, reasoning in iter_llm_verdicts(payload):
            icon, style = _SCORE_STYLE.get(score_token, UNKNOWN_VERDICT_DISPLAY)
            lines.append(f"  {icon} [bold]{rich_safe(dim)}[/bold] - " f"[{style}]{rich_safe(score_token)}[/{style}]")
            if reasoning:
                truncated = reasoning[:300] + ("…" if len(reasoning) > 300 else "")
                lines.append(f"       [dim]{rich_safe(truncated)}[/dim]")

    content = "\n".join(lines)
    try:
        panel = Panel(Text.from_markup(content), title="Score Summary", border_style="dim", padding=(0, 1))
    except Exception:
        panel = Panel(content, title="Score Summary", border_style="dim", padding=(0, 1))
    con.print(panel)


def _show_turn_detail(con: Console, outcome_dir: Path, turn_idx: int) -> None:
    """Render a Rich panel for a single turn."""
    output_path = outcome_dir / TURN_OUTPUT_TEMPLATE.format(turn_idx)
    cli_path = outcome_dir / TURN_CLI_TEMPLATE.format(turn_idx)

    turn_output: Optional[TurnOutput] = None
    if output_path.exists():
        try:
            turn_output = TurnOutput.model_validate_json(output_path.read_text())
        except Exception:
            pass

    cli_text = ""
    if cli_path.exists():
        try:
            cli_text = cli_path.read_text()
        except OSError:
            pass

    lines: list[str] = []

    if turn_output is not None:
        # Cost / timing
        meta_parts: list[str] = []
        if turn_output.cost_usd is not None:
            meta_parts.append(f"cost: [green]{_fmt_cost(turn_output.cost_usd)}[/green]")
        if turn_output.timing and turn_output.timing.total is not None:
            meta_parts.append(f"time: {_fmt_secs(turn_output.timing.total)}")
        if meta_parts:
            lines.append("  ".join(meta_parts))
            lines.append("")

        # Tool calls
        if turn_output.tool_calls:
            lines.append(f"[bold]Tool calls ({len(turn_output.tool_calls)}):[/bold]")
            for tc in turn_output.tool_calls:
                args_preview = _fmt_args(tc.args)
                lines.append(f"  🔧 [cyan]{tc.name}[/cyan]({args_preview})")
            lines.append("")

        # Reply text
        if turn_output.reply_text:
            preview = turn_output.reply_text.strip()[:500]
            if len(turn_output.reply_text.strip()) > 500:
                preview += "…"
            lines.append("[bold]Reply:[/bold]")
            lines.append(f"  {preview}")
            lines.append("")

        # Error flag
        if turn_output.has_error:
            lines.append("[red]⚠ Turn reported an error[/red]")
            if turn_output.error_type:
                lines.append(f"  Error type: {turn_output.error_type}")
            lines.append("")

    # CLI output (truncated)
    if cli_text.strip():
        cli_preview = cli_text.strip()
        # Show last 800 chars of CLI output - most relevant content is at the end
        if len(cli_preview) > 800:
            cli_preview = "…\n" + cli_preview[-800:]
        lines.append("[bold]CLI output (tail):[/bold]")
        # Escape markup in raw CLI text
        lines.append(cli_preview.replace("[", "\\["))

    content = "\n".join(lines) if lines else "[dim](no data)[/dim]"

    try:
        panel = Panel(
            Text.from_markup(content),
            title=f"Turn {turn_idx}",
            border_style="blue",
            padding=(0, 1),
        )
    except Exception:
        panel = Panel(content, title=f"Turn {turn_idx}", border_style="blue", padding=(0, 1))
    con.print(panel)


def _fmt_args(args: dict[str, Any]) -> str:
    """Format tool args as a compact key=value string (120 chars max)."""
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        vs = json.dumps(v) if not isinstance(v, str) else v
        vs = vs.replace("\n", " ")
        if len(vs) > 60:
            vs = vs[:57] + "…"
        parts.append(f"{k}={vs}")
    result = ", ".join(parts)
    if len(result) > 120:
        result = result[:117] + "…"
    return result


# ── Interactive navigation ──


def _interactive_loop(outcomes_dir: Path, summaries: list[dict[str, Any]], console: Console) -> None:
    """Show summary table, prompt user to pick a scenario, show detail, repeat."""
    while True:
        show_summary_table(summaries, console)

        if not summaries:
            break

        console.print(f"\n  Enter a scenario number (1-{len(summaries)}) to drill down, or [bold]q[/bold] to quit.")
        try:
            choice = Prompt.ask("  Choice", console=console, default="q")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        choice = choice.strip()
        if choice.lower() in ("q", "quit", "exit", ""):
            break

        try:
            idx = int(choice)
        except ValueError:
            console.print(f"[red]Invalid choice:[/red] {choice!r}")
            continue

        if not (1 <= idx <= len(summaries)):
            console.print(f"[red]Out of range:[/red] enter 1-{len(summaries)}")
            continue

        selected = summaries[idx - 1]
        show_scenario_detail(outcomes_dir, selected["scenario"], console)

        console.print("\n  Press [bold]Enter[/bold] to return to summary…")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break


# ── Main entry point ──


def view_cmd(outcomes_dir_arg: Optional[str] = None, *, non_interactive: bool = False) -> int:
    """Main entry point for ``belt view``.

    Args:
        outcomes_dir_arg: Path to an outcomes run directory.  If ``None``, the
            most recently modified subdirectory under ``OUTCOMES_ROOT`` is used.
        non_interactive: When ``True``, print the summary table and exit without
            prompting (useful for testing and piping).

    Returns:
        Integer exit code (0 = success, 1 = error).
    """
    console = Console(stderr=True)

    if outcomes_dir_arg:
        outcomes_dir = Path(outcomes_dir_arg)
    else:
        outcomes_dir = find_latest_outcomes_dir(OUTCOMES_ROOT)  # type: ignore[assignment]
        if outcomes_dir is None:
            console.print(
                "[red]No outcomes directory found.[/red] " "Run [bold]belt eval[/bold] first, or pass a directory path."
            )
            return 1
        console.print(f"  [dim]Using latest outcomes dir: {outcomes_dir}[/dim]")

    if not outcomes_dir.is_dir():
        console.print(f"[red]Directory not found:[/red] {outcomes_dir}")
        return 1

    summaries = load_summary(outcomes_dir)
    setup_errors = _read_run_setup_errors(outcomes_dir)

    if not summaries and not setup_errors:
        console.print(f"[yellow]No scored results found[/yellow] under [bold]{outcomes_dir}[/bold]")
        console.print("  Run [bold]belt score[/bold] first to generate score.json files.")
        return 1

    if non_interactive:
        if setup_errors:
            _render_setup_errors_banner(setup_errors, console)
        if summaries:
            show_summary_table(summaries, console)
        return 0

    _interactive_loop(outcomes_dir, summaries, console)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point - parse ``belt view`` arguments and invoke ``view_cmd``."""
    parser = argparse.ArgumentParser(
        prog="belt view",
        description="Browse evaluation results in the terminal.",
    )
    # Positional ``run-dir`` is the single run directory to view (a child
    # of the outcomes root). The metavar is ``run-dir`` for consistency
    # with the renamed ``--run-dir`` flag on ``score`` / ``aggregate``;
    # the dest is also ``run_dir`` so help text and Python attribute
    # match. Old positional name (``outcomes-dir``) was confusable with
    # ``--outcomes-dir`` (the parent root).
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        metavar="run-dir",
        help="Run directory to view (default: most recent under outcomes/)",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        default=False,
        help="Print summary table and exit without prompting",
    )
    args = parser.parse_args(argv)
    return view_cmd(args.run_dir, non_interactive=args.non_interactive)
