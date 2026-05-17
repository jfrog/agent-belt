# (c) JFrog Ltd. (2026)

"""Phase 1 - discover groups under the scenarios root and apply CLI filters.

Returns a list of ``MatchedGroup`` records (one per scenario group that
passed the ``--scenarios`` / ``--tags`` filters and the per-group agent
override).  ``commands/run.py`` calls this before ``build_run_context``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from loguru import logger

from belt import _internal_envvars
from belt._ui import eprint
from belt.filter import ScenarioFilter
from belt.parser.scenario import ScenarioLoader
from belt.runner.context import MatchedGroup


def _discover_groups(scenarios_root: Path) -> list[Path]:
    """Find every directory containing a ``_config.json`` under ``scenarios_root``."""
    return sorted(p.parent.resolve() for p in scenarios_root.rglob("_config.json"))


def parse_and_filter(args: argparse.Namespace) -> tuple[Path, list[MatchedGroup], int] | int:
    """Parse CLI args, discover groups, filter scenarios.

    Returns ``(scenarios_root, matched_groups, scenarios_skipped)`` on
    success, or an int exit code on failure (printed message already
    produced).

    ``scenarios_skipped`` is the count of scenario JSON files that failed
    to parse across every matched group. Always printed in the summary
    line below; threaded into ``run_meta.json`` so the aggregator can
    surface it on ``AggregatedResults.scenarios_skipped``.

    When ``--strict`` is set and any scenario was skipped, the function
    prints the full error list and returns ``1`` so CI fails loudly
    instead of running on a silently-shrunken fleet.
    """
    scenarios_root = Path(args.path).resolve()
    if not scenarios_root.is_dir():
        eprint(f"Not a directory: {scenarios_root}")
        return 1
    os.environ[_internal_envvars.SCENARIOS_ROOT] = str(scenarios_root)

    groups = _discover_groups(scenarios_root)
    if not groups:
        eprint(f"No groups found under {scenarios_root}")
        return 1

    try:
        scenario_filter = ScenarioFilter.from_cli_args(
            scenarios_root,
            tags=args.tags,
            scenarios=args.scenarios,
        )
    except Exception as e:
        eprint(f"\n  ❌ {e}")
        return 1

    strict_config = bool(getattr(args, "strict_config", False))

    matched_groups: list[MatchedGroup] = []
    # ``(group_label, scenario_file_error)`` for each scenario that failed
    # to parse - printed verbatim under ``--strict`` so the operator can
    # fix the offending file without re-running.
    skipped: list[tuple[str, str]] = []
    # ``--strict-config`` rejections on a group's own ``_config.json`` are
    # collected here. Group-config errors abort the run unconditionally
    # (independent of ``--strict``) because every scenario in the group
    # would otherwise inherit a misconfigured agent / working_dir / etc.
    group_config_errors: list[tuple[str, str]] = []
    for group_dir in groups:
        matched, allowed_names = scenario_filter.matches_group(group_dir)
        if not matched:
            continue
        try:
            group_config = ScenarioLoader.load_group_config(group_dir, strict_config=strict_config)
            scenarios, load_errors = ScenarioLoader.load_group_scenarios(group_dir, strict_config=strict_config)
        except Exception as e:
            logger.error("Failed to load group {}: {}", group_dir, e)
            if strict_config:
                # Surface group-config validation failures to the operator
                # immediately - per-scenario errors otherwise hide the root
                # cause behind a "no scenarios loaded" summary.
                group_config_errors.append((group_dir.name, str(e)))
            continue
        for err in load_errors:
            logger.warning("Skipping malformed scenario in {}: {}", group_dir.name, err)
            skipped.append((group_dir.name, err))
        if allowed_names:
            available = {s.name for s in scenarios}
            unknown = allowed_names - available
            if unknown:
                group_label = group_dir.resolve().relative_to(scenarios_root)
                eprint(f"Unknown scenario(s) in {group_label}: {', '.join(sorted(unknown))}")
                eprint(f"  Available: {', '.join(sorted(available))}")
                return 1
            scenarios = [s for s in scenarios if s.name in allowed_names]
        scenarios = [s for s in scenarios if scenario_filter.matches_scenario(s, group_config)]

        if args.agent:
            group_config.agent = args.agent

        if scenarios:
            # Group identity is the path relative to ``scenarios_root``.
            # When the user points the runner directly at a single group
            # (``scenarios_root == group_dir``), the relative path
            # collapses to ``"."``; that is a poor display name and an
            # inconsistent identity (downstream sinks like the benchmark
            # card's fixtures table fall back to ``group_dir.name``,
            # which would diverge). Normalise to the directory's own
            # basename in that case so every consumer sees the same
            # human-readable label.
            relative = group_dir.resolve().relative_to(scenarios_root)
            name = group_dir.name if str(relative) == "." else str(relative)
            matched_groups.append(
                MatchedGroup(group_dir=group_dir, config=group_config, scenarios=scenarios, name=name)
            )

    total_scenarios = sum(len(mg.scenarios) for mg in matched_groups)
    total_skipped = len(skipped)

    # Always-on summary so a CI run shrunken by a typo is visible without
    # tailing the log. Stays terse when no scenarios were dropped.
    if total_skipped:
        eprint(
            f"Loaded {total_scenarios} scenarios across {len(matched_groups)} groups "
            f"({total_skipped} malformed skipped)"
        )
    else:
        eprint(f"Loaded {total_scenarios} scenarios across {len(matched_groups)} groups")

    if group_config_errors:
        # Group-config errors under ``--strict-config`` are unconditionally
        # fatal: every scenario in an offending group would otherwise run
        # against a misconfigured agent or workspace.
        eprint("\n  ❌ --strict-config: group config validation failed:")
        for group_label, err in group_config_errors:
            eprint(f"     - {group_label}: {err}")
        return 1

    if total_skipped and (strict_config or getattr(args, "strict", False)):
        # ``--strict-config`` is unconditionally fatal on per-scenario
        # rejections too: the operator opted in to "fail fast on typos",
        # so a partial run defeats the purpose. ``--strict`` keeps its
        # existing semantics.
        flag = "--strict-config" if strict_config else "--strict"
        eprint(f"\n  ❌ {flag}: refusing to run with malformed scenarios:")
        for group_label, err in skipped:
            eprint(f"     - {group_label}: {err}")
        return 1

    return scenarios_root, matched_groups, total_skipped
