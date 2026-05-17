#!/usr/bin/env python3
# (c) JFrog Ltd. (2026)

"""
Score evaluation outcomes using configurable scorers.

This module is the thin CLI entry point for ``belt score``. Library
logic lives in:

- ``belt.scorer.pipeline`` - ``score_scenario``, ``build_scorers``,
  ``validate_scorers``, ``discover_outcome_dirs``
- ``belt.scorer.scenario_map`` - outcome-dir → scenario file mapping,
  scoring strategy resolution
- ``belt.scorer.event_sink`` - NDJSON writer + judge → progress fan-out
- ``belt.scorer.dotenv_safety`` - owner-checked ``.env`` loader
- ``belt.scorer.dry_run`` - ``--dry-run`` payload printer

Usage:
    belt score
    belt score --modes rules
    belt score --modes rules,llm --scorer-arg model=<provider>/<model> --scorer-arg temperature=0.0
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from belt import _internal_envvars, envvars
from belt._ui import eprint
from belt.constants import EXAMPLE_LLM_MODEL, LOG_FILE, SCORE_FILE
from belt.entities import ScenarioScore
from belt.progress import ScorerProgress
from belt.scorer import ConsensusScorer, LLMScorer, ScoreCache
from belt.scorer.dotenv_safety import load_dotenv_safely
from belt.scorer.dry_run import handle_dry_run
from belt.scorer.event_sink import NdjsonWriter, build_event_sink, wire_event_callbacks
from belt.scorer.pipeline import (
    build_judge_config_from_scorer_args,
    build_scorers,
    discover_outcome_dirs,
    load_scorer_config,
    resolve_scorer_args,
    score_scenario,
    validate_scorers,
)
from belt.scorer.scenario_map import map_to_scenario, parse_dimension_defs, resolve_scoring_strategy, scenarios_root

# Aliases re-exported for tests and downstream callers that import these
# names from ``commands.score`` rather than from the underlying helpers.
_NdjsonWriter = NdjsonWriter
_load_dotenv_safely = load_dotenv_safely
_build_event_sink = build_event_sink
_wire_event_callbacks = wire_event_callbacks
_handle_dry_run = handle_dry_run
_build_scorers = build_scorers
_load_scorer_config = load_scorer_config
_build_judge_config_from_scorer_args = build_judge_config_from_scorer_args
_scenarios_root = scenarios_root
_resolve_scoring_strategy = resolve_scoring_strategy
_parse_dimension_defs = parse_dimension_defs
_resolve_scorer_args = resolve_scorer_args


@dataclass
class ScorerOption:
    """Declaration of a scorer-specific CLI option for help text and validation."""

    name: str
    help: str
    required: bool = False
    default: str | None = None


LLM_SCORER_OPTIONS = [
    ScorerOption(name="model", help=f"LLM model (required; prefix routing: {EXAMPLE_LLM_MODEL})"),
    ScorerOption(name="temperature", help="Sampling temperature", default="0.0"),
    ScorerOption(name="seed", help="Sampling seed", default="2008"),
    ScorerOption(name="max_tokens", help="Max response tokens", default="4096"),
    ScorerOption(name="max_prompt_chars", help="Max chars for dynamic message (~4 chars/token)", default="100000"),
    ScorerOption(name="no_cache", help="Disable response cache", default="false"),
    ScorerOption(name="max_retries", help="Max retries on 429", default="5"),
]


def _select_dry_run_outcome(
    outcome_dirs: list[Path],
    outcomes_root: Path,
    selector: str,
) -> Path | None:
    """Pick the outcome dir to preview with ``--dry-run``.

    Empty selector → first outcome (preserves the original ``outcome_dirs[0]``
    behaviour).  Non-empty selector matches a substring of the outcome's path
    relative to ``outcomes_root`` so users can write ``--dry-run simple_math``
    instead of the full ``experience/tasktracker-claude/l2_fix_formatter_bug``.

    Prints an error + returns ``None`` on no-match / multi-match so the caller
    can return exit code 1 without raising.
    """
    if not selector:
        return outcome_dirs[0]
    needle = selector.strip()
    matches = [d for d in outcome_dirs if needle in str(d.relative_to(outcomes_root))]
    if not matches:
        eprint(f"  ❌ --dry-run: no scenario matches '{needle}'")
        return None
    if len(matches) > 1:
        eprint(f"  ❌ --dry-run: '{needle}' matches {len(matches)} scenarios; be more specific:")
        for d in matches[:10]:
            eprint(f"     - {d.relative_to(outcomes_root)}")
        return None
    return matches[0]


def _configure_logging(log_path: Path | None) -> None:
    """Attach the score-phase transcript log handler.

    Symmetric with :func:`belt.commands.run._configure_logging`. The
    terminal handler is configured separately by
    :func:`belt._logging.configure_terminal_logging`; this function is
    additive so the score phase, when invoked as a step of ``belt eval``,
    keeps the terminal handler put in place by the parent command.
    """
    if log_path:
        logger.add(log_path, level="DEBUG", rotation=None)
        os.environ[_internal_envvars.LOG_FILE] = str(log_path)


def _add_common_score_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared between ``belt score`` and ``belt eval``."""
    scoring = parser.add_argument_group("Scoring")
    scoring.add_argument(
        "--modes",
        default="rules,llm",
        help="Comma-separated scoring modes (default: rules,llm). "
        "Built-in: rules, llm. Third-party scorers registered via belt.scorers entry points are also accepted.",
    )
    scoring.add_argument(
        "-S",
        "--scorer-arg",
        action="append",
        dest="scorer_args",
        metavar="KEY=VALUE",
        help=(
            "Scorer-specific option (repeatable). Passed to the scorer's constructor. "
            f"E.g.: --scorer-arg model={EXAMPLE_LLM_MODEL} --scorer-arg temperature=0.0 --scorer-arg seed=42"
        ),
    )
    scoring.add_argument(
        "--scorer-config",
        metavar="PATH",
        help="YAML config for multi-judge scoring. Each named judge gets its own model, "
        "temperature, dimensions, and optional system preamble.",
    )


def main(argv: list[str] | None = None, *, chained: bool = False) -> int:
    # ``chained=True`` is passed by ``commands.eval`` when this main is
    # called in-process as part of ``belt eval``'s run -> score ->
    # aggregate pipeline. It propagates to ``ScorerProgress.summary`` so
    # the score phase suppresses its pass-count headline (the aggregator
    # prints the canonical "X/N checks (Y%)" footer immediately after).
    # Standalone ``belt score`` invocations leave it ``False`` and keep
    # the full summary.
    parser = argparse.ArgumentParser(
        prog="belt score",
        description="Score evaluation outcomes",
        epilog="Note: --scenarios filters scenarios during `belt run` and `belt eval`. "
        "The score command operates on all outcomes in the --run-dir directory.\n\n"
        "Scorer-specific options are passed via --scorer-arg KEY=VALUE. "
        "Available LLM scorer options: model, temperature, seed, max_tokens, no_cache, max_retries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )

    inp = parser.add_argument_group("Input")
    inp.add_argument("--run-dir", help="Outcomes run directory (default: latest)")

    _add_common_score_args(parser)

    # Within this group, ``add_argument`` calls are alphabetised by long
    # flag name. Enforced by ``tests/test_cli_order.py``.
    exe = parser.add_argument_group("Execution")
    exe.add_argument(
        "--allow-arbitrary-agent",
        action="store_true",
        default=False,
        help=(
            "Allow loading agent metadata via dotted import path. Off by default - "
            "only registered agents are recognised. ``score`` reads agent info "
            f"from outcome dirs, so the gate belongs here too. Same effect as {envvars.ALLOW_ARBITRARY_AGENT}=1."
        ),
    )
    exe.add_argument(
        "--allow-arbitrary-scorer",
        action="store_true",
        default=False,
        help=(
            "Allow ``--modes`` to resolve to a dotted import path. Off by default - "
            "only built-in scorers and entry-point-registered ones are loaded."
        ),
    )
    exe.add_argument(
        "--allow-insecure-base-url",
        action="store_true",
        default=False,
        help=(
            "Permit plaintext (http://) custom LLM base URLs. Off by default - "
            "https:// is required for any non-default base URL. Same effect as "
            f"{envvars.ALLOW_INSECURE_BASE_URL}=1."
        ),
    )
    exe.add_argument(
        "--dry-run",
        nargs="?",
        const="",
        default=None,
        metavar="SCENARIO",
        help=(
            "Preview what would happen without executing. With --modes llm: "
            "prints the LLM payload. With --modes rules: lists scenarios + "
            "rule-check categories. Optional SCENARIO selector matches a "
            "substring of the outcome path (e.g. 'simple_math'); defaults to "
            "the first scenario."
        ),
    )
    exe.add_argument(
        "--progress",
        choices=["rich", "plain", "live"],
        default="rich",
        help="Progress display: rich (bars), plain (CI), live (panel bars matching run TUI)",
    )
    exe.add_argument("--workers", type=int, default=1, help="Max parallel scoring tasks (default: 1)")
    args = parser.parse_args(argv)

    from belt._logging import configure_terminal_logging

    configure_terminal_logging(0)

    envvars.forward_security_toggles(args)

    load_dotenv_safely()

    if args.run_dir:
        outcomes_root = Path(args.run_dir)
    else:
        from belt.manifest import Manifest

        latest = Manifest().latest_run
        if not latest:
            eprint("No previous run found - run runner first or pass --run-dir")
            return 1
        outcomes_root = Path(latest)

    if not outcomes_root.is_dir():
        eprint(f"No outcomes directory: {outcomes_root}")
        return 1

    _configure_logging(outcomes_root / LOG_FILE)

    scorer_args = resolve_scorer_args(args)
    no_cache = scorer_args.get("no_cache", "false").lower() in ("true", "1", "yes")

    outcome_dirs = discover_outcome_dirs(outcomes_root)
    if not outcome_dirs:
        eprint(f"No outcomes found under {outcomes_root}")
        return 1

    scorer_config_path = getattr(args, "scorer_config", None)
    is_dry_run = args.dry_run is not None
    try:
        scorers, llm_scorers = build_scorers(args.modes, scorer_args, scorer_config_path, skip_availability=is_dry_run)
    except Exception as e:
        eprint(f"\n  ❌ {e}")
        return 1

    score_cache = None if no_cache else ScoreCache(outcomes_root / ".score_cache")
    for s in llm_scorers:
        s.cache = score_cache

    first_llm = llm_scorers[0] if llm_scorers else None

    if is_dry_run:
        selected = _select_dry_run_outcome(outcome_dirs, outcomes_root, args.dry_run)
        if selected is None:
            return 1
        return handle_dry_run(
            llm_scorer=first_llm,
            outcome_dir=selected,
            outcomes_root=outcomes_root,
            scorers=scorers,
        )

    mode_str = " + ".join(s.name for s in scorers) if scorers else "none"
    workers = min(args.workers, len(outcome_dirs))
    max_retries = int(scorer_args.get("max_retries", "5"))

    is_live = args.progress == "live"
    progress = ScorerProgress(plain=args.progress == "plain", live=is_live)
    progress.start(len(outcome_dirs), mode_str, workers, max_retries=max_retries)

    ndjson_writer = NdjsonWriter.from_path(outcomes_root / "score_stream.ndjson") if llm_scorers else None
    event_sink = build_event_sink(progress, ndjson_writer) if is_live else None
    if ndjson_writer and not is_live:
        event_sink = build_event_sink(None, ndjson_writer)

    wire_event_callbacks(scorers, event_sink)

    def _score_one(outcome_dir: Path) -> tuple[Path, ScenarioScore]:
        strategy = resolve_scoring_strategy(outcome_dir, outcomes_root)
        effective_scorers = scorers
        if strategy is not None:
            effective_scorers = []
            for s in scorers:
                if isinstance(s, ConsensusScorer):
                    new_judges = [
                        LLMScorer(j.config, j.max_retries, strategy=strategy, cache=j.cache, on_event=j.on_event)
                        for j in s.judges
                    ]
                    for orig, new in zip(s.judges, new_judges):
                        new.judge_name = orig.judge_name
                    effective_scorers.append(ConsensusScorer(new_judges, strategy=s.consensus_strategy))
                elif isinstance(s, LLMScorer):
                    effective_scorers.append(
                        LLMScorer(s.config, s.max_retries, strategy=strategy, cache=s.cache, on_event=s.on_event)
                    )
                else:
                    effective_scorers.append(s)
        score = score_scenario(outcome_dir, outcomes_root, effective_scorers)

        eff_llm = [s for s in effective_scorers if isinstance(s, LLMScorer)]
        if not eff_llm:
            from belt.scorer.llm.consensus import ConsensusScorer as _CS

            for s in effective_scorers:
                if isinstance(s, _CS):
                    eff_llm.extend(s.judges)
        if eff_llm:
            pt = sum(s.total_prompt_tokens for s in eff_llm)
            ct = sum(s.total_completion_tokens for s in eff_llm)
            score.scorer_prompt_tokens = pt
            score.scorer_completion_tokens = ct
            costs = [s.total_cost_usd for s in eff_llm if s.total_cost_usd is not None]
            if costs:
                score.judge_cost_usd = sum(costs)

        score_path = outcome_dir / SCORE_FILE
        score_path.write_text(score.model_dump_json(indent=2) + "\n")
        return outcome_dir, score

    results: list[tuple[Path, ScenarioScore]] = []

    if workers <= 1:
        for outcome_dir in outcome_dirs:
            try:
                od, score = _score_one(outcome_dir)
                results.append((od, score))
                relative = str(od.relative_to(outcomes_root))
                progress.scored(relative, score.overall_pass)
            except Exception as e:
                logger.error("Scoring failed for {}: {}", outcome_dir, e)
                relative = str(outcome_dir.relative_to(outcomes_root))
                progress.scored(relative, False)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_score_one, od): od for od in outcome_dirs}
            for future in as_completed(futures):
                try:
                    od, score = future.result()
                    results.append((od, score))
                    relative = str(od.relative_to(outcomes_root))
                    progress.scored(relative, score.overall_pass)
                except Exception as e:
                    od = futures[future]
                    logger.error("Scoring failed for {}: {}", od, e)

    progress.stop()
    if ndjson_writer is not None:
        ndjson_writer.close()

    cache_hits = score_cache.hits if score_cache else 0
    cache_misses = score_cache.misses if score_cache else 0
    prompt_tokens = sum(s.total_prompt_tokens for s in llm_scorers)
    completion_tokens = sum(s.total_completion_tokens for s in llm_scorers)
    judge_costs = [s.total_cost_usd for s in llm_scorers if s.total_cost_usd is not None]
    judge_cost = sum(judge_costs) if judge_costs else None

    progress.summary(
        results,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        judge_cost_usd=judge_cost,
        chained=chained,
    )

    return 0


__all__ = [
    "LLM_SCORER_OPTIONS",
    "ScorerOption",
    "build_scorers",
    "discover_outcome_dirs",
    "main",
    "map_to_scenario",
    "score_scenario",
    "validate_scorers",
]


if __name__ == "__main__":
    sys.exit(main())
