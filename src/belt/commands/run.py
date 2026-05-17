#!/usr/bin/env python3
# (c) JFrog Ltd. (2026)

"""
Run evaluation scenarios via agent-agent-driven CLI agents.

This module is the thin CLI entry point for ``belt run``. Library
logic lives in ``belt.runner.context`` (shared dataclasses, agent
construction) and ``belt.runner.phases.*`` (one module per phase).

Usage:
    belt run examples/scenarios/
    belt run examples/scenarios/ --tags interrupt
    belt run examples/scenarios/ --agent claude-code
    belt run examples/scenarios/ --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from belt import _internal_envvars, envvars
from belt._io import write_json
from belt._redact import safe_environ, scrub_argv, scrub_kv_list
from belt._ui import eprint
from belt.benchmark_card import (
    collect_belt_provenance,
    collect_host_provenance,
    collect_invocation,
    hash_scenario_files,
)
from belt.constants import EVAL_DIR, LOG_FILE, REPO_ROOT, RUN_META_FILE, SCHEMA_VERSION
from belt.runner.context import (
    MatchedGroup,
    RunContext,
    _resolve_env_var_defaults,
    build_run_context,
    create_agent,
    resolve_outcomes_root,
)
from belt.runner.phases.parse_filter import parse_and_filter
from belt.runner.phases.run_scenarios import run_scenarios
from belt.runner.phases.setup_groups import cleanup_orphans, setup_groups
from belt.runner.phases.teardown import teardown_groups

# Re-export library symbols so callers that imported them from this module
# keep working. Stabilizes the public surface for forks and in-tree tests
# (``tests/test_security.py`` and friends).
__all__ = [
    "MatchedGroup",
    "RunContext",
    "_add_common_run_args",
    "_resolve_env_var_defaults",
    "build_run_context",
    "cleanup_orphans",
    "create_agent",
    "main",
    "parse_and_filter",
    "resolve_outcomes_root",
    "run_scenarios",
    "setup_groups",
    "teardown_groups",
]


def _configure_logging(log_path: Path | None) -> None:
    """Attach the run's transcript log handler.

    The terminal handler is attached separately (and earlier) by
    :func:`belt._logging.configure_terminal_logging` so the terminal log
    level is independent of the transcript log level. This function does
    not remove existing handlers - it only adds the always-on DEBUG-level
    file handler that produces ``<run_dir>/eval.log``.

    When the run dir is set up by ``belt eval``, the terminal handler is
    already in place at the level resolved from ``-v`` / ``BELT_LOG_LEVEL``;
    standalone ``belt run`` invocations have no terminal handler by
    design (the run phase is non-interactive infrastructure - the user
    inspects results via ``belt score`` / ``belt aggregate`` /
    ``belt view`` afterwards).
    """
    if log_path:
        logger.add(log_path, level="DEBUG", rotation=None)
        # Surface the active log file to the top-level error path so an
        # unhandled exception can print "-> logs: <path>" and point the
        # user at the file. Read by ``belt.cli._report_error``.
        os.environ[_internal_envvars.LOG_FILE] = str(log_path)


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared between ``belt run`` and ``belt eval``.

    Within each ``add_argument_group(...)`` block, ``add_argument`` calls are
    ordered alphabetically by long flag name. Positional args precede flags.
    The ordering is enforced by ``tests/test_cli_order.py``.
    """
    sel = parser.add_argument_group("Scenario selection")
    # See ``belt eval --bundled`` for the rationale. Keeping the flag
    # available on ``belt run`` (and the ``path`` positional optional)
    # so the two subcommands stay symmetrical - users can mix-and-match
    # the lower-level command exactly the same way.
    sel.add_argument(
        "path",
        nargs="?",
        help="Root scenarios directory. Omit when using --bundled.",
    )
    sel.add_argument(
        "--agent", dest="agent", help="Override agent from _config.json (run 'belt agent list' to see options)"
    )
    sel.add_argument(
        "--bundled",
        nargs="?",
        const="",
        metavar="NAME",
        help=(
            "Run scenarios shipped inside the agent-belt wheel "
            "(e.g. ``--bundled showcase``). Mutually exclusive with the "
            "positional path."
        ),
    )
    sel.add_argument(
        "--scenarios",
        help="Comma-separated filter paths, relative to the scenarios root. "
        "Examples: --scenarios showcase/correctness (whole group), "
        "--scenarios showcase/correctness/correctness_basic (single scenario). "
        "When the path argument IS itself a group directory, you may pass a bare scenario "
        "name (no slash) - the runner resolves it inside that group.",
    )
    sel.add_argument("--tags", help="Comma-separated tags to require (AND logic)")

    agent = parser.add_argument_group("Agent options")
    agent.add_argument(
        "-X",
        "--agent-arg",
        action="append",
        dest="agent_args",
        metavar="KEY=VALUE",
        help="Agent-specific option (repeatable). Passed to the agent's constructor. " "E.g.: -X max_tokens=4096",
    )

    exe = parser.add_argument_group("Execution")
    exe.add_argument(
        "--allow-arbitrary-agent",
        action="store_true",
        default=False,
        help=(
            "Allow ``--agent`` to resolve to an arbitrary dotted import path "
            "(e.g. ``mypackage.MyAgentAdapter``). Off by default - only built-in agents "
            "and entry-point-registered ones are loaded. Same effect as setting "
            f"{envvars.ALLOW_ARBITRARY_AGENT}=1."
        ),
    )
    exe.add_argument(
        "--allow-arbitrary-scorer",
        action="store_true",
        default=False,
        help=(
            "Allow scorers to resolve via dotted import path. Off by default - "
            "only registered scorers are loadable. ``run`` preflight loads scorer "
            f"metadata, so the gate belongs here too. Same effect as {envvars.ALLOW_ARBITRARY_SCORER}=1."
        ),
    )
    exe.add_argument(
        "--allow-external-working-dir",
        action="store_true",
        default=False,
        help=(
            "Permit a group's working_dir to resolve outside the scenarios root (the "
            "positional path argument). Off by default - belt treats this as a "
            "configuration error since git-worktree isolation auto-initialises the "
            "resolved path as a repo, which is unsafe for arbitrary host paths."
        ),
    )
    exe.add_argument(
        "--allow-full-env",
        action="store_true",
        default=False,
        help=(
            "Pass belt's full os.environ to every CLI agent subprocess. Off by "
            "default; agents only inherit a curated allow-list (PATH, HOME, locale, "
            "proxy, TLS, common provider API keys). Same effect as setting "
            f"{envvars.ALLOW_FULL_ENV}=1."
        ),
    )
    exe.add_argument(
        "--allow-inplace",
        action="store_true",
        default=False,
        help=(
            "Permit groups whose ``_config.json`` declares "
            '``workspace_isolation: "none"``. Off by default - belt refuses '
            "such groups so an agent never operates without per-scenario "
            "worktree isolation by accident. Opt in only when the scenario "
            "deliberately edits the harness CWD. Same effect as setting "
            f"{envvars.ALLOW_INPLACE}=1."
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
    exe.add_argument("--dry-run", action="store_true", help="List matching scenarios without executing")
    exe.add_argument(
        "--no-cleanup", action="store_true", default=False, help="Skip group teardown after run (default: clean up)"
    )
    exe.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Disable writing live NDJSON stream files per turn (default: stream enabled)",
    )
    exe.add_argument(
        "--progress",
        choices=["rich", "plain", "live"],
        default="rich",
        help="Progress display: rich (bars), plain (CI), live (integrated TUI with agent stream)",
    )
    exe.add_argument(
        "--progress-live-lines",
        type=int,
        default=30,
        help="Max event lines in --progress live panel (default: 30)",
    )
    exe.add_argument(
        "--sandbox",
        metavar="PROVIDER",
        default=None,
        help=(
            "Override the sandbox provider for this run. Names are resolved via the "
            "``belt.sandbox_providers`` entry-point group; the framework ships "
            "``host`` (default; agent runs on the host with the invoking user's "
            "privileges and no isolation) and ``docker`` (each agent subprocess "
            "in a container with --cap-drop=ALL, --read-only rootfs, the worktree as "
            "the only writable mount, env passthrough by exact name). Per-scenario "
            "image / hosts / passthrough are still read from the group's "
            "``_config.json``. Same effect as setting "
            f"{envvars.SANDBOX_PROVIDER}=PROVIDER. See SANDBOXING.md."
        ),
    )
    exe.add_argument(
        "--scenario-delay",
        type=float,
        default=0,
        help="Seconds to wait between dispatching parallel scenarios (default: 0)",
    )
    exe.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help=(
            "Abort if any group's agent check fails OR if any scenario fails to parse " "(default: skip and continue)"
        ),
    )
    exe.add_argument(
        "--strict-config",
        action="store_true",
        default=False,
        help=(
            "Reject scenarios and group configs containing keys not declared on the "
            "Pydantic model and not registered as a plugin extension. Catches typos like "
            "'tools_invoke' instead of 'tools_invoked' that would otherwise silently produce "
            "zero coverage. Implies fail-fast on the affected file (default: permissive)."
        ),
    )
    exe.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Run each scenario N times for reliability measurement (pass^k). Default: 1",
    )
    exe.add_argument("--workers", type=int, default=1, help="Max parallel scenarios (default: 1)")

    out = parser.add_argument_group("Output")
    out.add_argument(
        "--outcomes-dir",
        metavar="PATH",
        help="Root directory for outcomes (default: $BELT_OUTCOMES_DIR or cwd/outcomes)",
    )


def show_header(ctx: RunContext) -> int | None:
    """Display the run header and dry-run table. Returns exit code if dry-run, else None."""
    args = ctx.args
    try:
        run_label = str(ctx.run_dir.relative_to(EVAL_DIR)) if not args.dry_run else "-"
    except ValueError:
        run_label = str(ctx.run_dir) if not args.dry_run else "-"

    unique_agents = sorted({mg.config.agent for mg in ctx.matched_groups})
    tags = args.tags or "-"

    scorer_descriptions: list[str] | None = None
    _scorer_json = os.environ.get(_internal_envvars.SCORER_DESCS, "")
    if _scorer_json:
        try:
            scorer_descriptions = json.loads(_scorer_json)
        except (json.JSONDecodeError, TypeError):
            scorer_descriptions = None

    ctx.progress.header(
        total_scenarios=ctx.total_scenarios,
        total_groups=len(ctx.matched_groups),
        total_turns=ctx.total_turns,
        agents=unique_agents,
        tags=tags,
        run_label=run_label,
        dry_run=args.dry_run,
        scorer_descriptions=scorer_descriptions,
        workspace=str(ctx.workspace),
    )

    if args.dry_run:
        groups_for_display = [(mg.group_dir, mg.config, mg.scenarios) for mg in ctx.matched_groups]
        ctx.progress.dry_run_table(groups_for_display, ctx.scenarios_root)
        return 0
    return None


def _redact_parsed_args(args: argparse.Namespace) -> dict[str, Any]:
    """Render ``args`` as a JSON-friendly dict with secret-looking values dropped.

    Argparse stores the parsed namespace as plain Python values; the only
    user-supplied strings of interest are the agent-arg pairs (already
    redacted by :func:`belt._redact.safe_agent_args` when the
    runner records them per scenario). Everything else here is structured
    runner config (paths, ints, bools, scorer settings) which is safe by
    construction. ``--agent-arg`` is the one field that takes raw strings;
    each ``key=value`` element is rewritten via the canonical
    :func:`belt._redact.scrub_kv_list` so the secret-name regex and
    the parsing logic live in exactly one place.
    """
    out: dict[str, Any] = {}
    for k, v in vars(args).items():
        if k == "agent_args" and isinstance(v, list):
            out[k] = scrub_kv_list([str(x) for x in v], mark="<set>")
            continue
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = repr(v)
    return out


def _resolve_run_agent(ctx: RunContext) -> str | list[str] | None:
    """Resolve the agent name(s) used for this run, for ``run_meta.json``.

    Returns:
        - A single agent name string when the run uses one agent uniformly
          (either via ``--agent`` or because every group's ``_config.json``
          declares the same agent).
        - A sorted list of agent names when groups disagree (heterogeneous
          runs - e.g. one suite of scenarios per agent).
        - ``None`` when no agent can be resolved (no groups, or every
          group lacks an ``agent`` field). Downstream readers should
          treat ``None`` the same as a missing key.

    Recording this at run-init time means downstream consumers (the
    aggregator's headline-remediation lookup, the watch command, future
    cross-run analytics) read one well-known field instead of
    re-deriving the answer from per-scenario sidecars.
    """
    cli_agent = getattr(getattr(ctx, "args", None), "agent", None)
    if isinstance(cli_agent, str) and cli_agent:
        return cli_agent
    matched_groups = getattr(ctx, "matched_groups", None) or []
    seen = sorted({mg.config.agent for mg in matched_groups if getattr(mg.config, "agent", None)})
    if not seen:
        return None
    if len(seen) == 1:
        return seen[0]
    return seen


def _matched_group_provenance(ctx: RunContext) -> list[dict[str, Any]]:
    """Per-group effective config snapshot for ``run_meta.json``.

    Records the fields downstream tools need to render group-level
    badges (agent, isolation mode, fixture refs) without re-reading
    every group's ``_config.json``. Stable shape; new fields append.
    """
    out: list[dict[str, Any]] = []
    for mg in getattr(ctx, "matched_groups", None) or []:
        gc = mg.config
        entry: dict[str, Any] = {
            "name": mg.name,
            "agent": gc.agent,
            "workspace_isolation": gc.workspace_isolation,
        }
        if gc.working_dir:
            entry["working_dir"] = gc.working_dir
        if gc.fixture_repo:
            # ``fixture_ref`` is only meaningful alongside ``fixture_repo``
            # (defaults to ``"HEAD"`` whenever the field is parsed); record
            # the pair together to keep the snapshot self-describing.
            entry["fixture_repo"] = gc.fixture_repo
            entry["fixture_ref"] = gc.fixture_ref
        out.append(entry)
    return out


def initialize_run_dir(ctx: RunContext, argv: list[str] | None = None) -> int | None:
    """Create the run directory, write run_meta.json, configure logging.

    ``run_meta.json`` is the input-side reproducibility record: everything
    we know at the start of the run, before any scenario executes. The
    aggregator pairs it with per-scenario sidecars and ``results.json`` to
    produce ``benchmark-card.json`` after scoring.

    Returns exit code on failure, None on success.
    """
    is_live = ctx.args.progress == "live"
    try:
        if not is_live:
            # Owner-only run dir - keeps stdout / NDJSON / CLI logs out of
            # other users' reach on shared hosts.
            ctx.run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as e:
        eprint(f"\n  ❌ Cannot create outcomes directory: {ctx.run_dir}\n     {e}")
        return 1
    os.environ[_internal_envvars.LAST_RUN_DIR] = str(ctx.run_dir)

    # Always write the minimal ``run_meta.json`` first - this block has
    # near-zero failure surface, so downstream phases (and the security
    # tests that pass a stub ``Ctx``) always see the env allow-list snapshot.
    # The enrichment block below is best-effort and may fail on partial
    # contexts without taking the minimal file with it.
    minimal_meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scenarios_root": str(ctx.scenarios_root.resolve()),
        "workspace": str(ctx.workspace),
        "agent": _resolve_run_agent(ctx),
        "env": safe_environ(),
        # Captured during parse_and_filter, persisted here so the aggregate
        # phase can surface the count even on runs where every malformed
        # file was silently skipped (no score.json sidecar to count from).
        "scenarios_skipped": int(getattr(ctx, "scenarios_skipped", 0) or 0),
    }
    if not write_json(ctx.run_dir / RUN_META_FILE, minimal_meta, sort_keys=False):
        logger.warning("Failed to write run_meta.json")

    # Enrich ``run_meta.json`` with full benchmark-card provenance: belt
    # identity, host runtime, the user's invocation, scenario hashes,
    # runtime knobs, and scoring config. ``safe_environ`` only records
    # the documented allow-list of operational env vars (CI markers +
    # BELT_* knobs); secret-shaped names degrade to "<set>" and
    # *_BASE_URL is redacted to scheme+host. When ``belt eval``
    # invoked us, prefer its stashed original argv so the card records the
    # user-facing command, not the synthesised ``run`` argv.
    try:
        original = os.environ.get(_internal_envvars.ORIGINAL_ARGV, "")
        argv_in: list[str]
        if original:
            try:
                parsed_orig = json.loads(original)
                argv_in = (
                    [str(x) for x in parsed_orig]
                    if isinstance(parsed_orig, list)
                    else (list(argv) if argv is not None else list(sys.argv))
                )
            except (TypeError, ValueError):
                argv_in = list(argv) if argv is not None else list(sys.argv)
        else:
            argv_in = list(argv) if argv is not None else list(sys.argv)
        scenario_paths = _scenario_files_from_groups(ctx)
        scenario_files = hash_scenario_files(ctx.scenarios_root, scenario_paths)
        belt_block = collect_belt_provenance(REPO_ROOT)
        host_block = collect_host_provenance()
        invocation_block = collect_invocation(
            argv=scrub_argv(argv_in),
            parsed_args=_redact_parsed_args(ctx.args),
            cwd=str(ctx.workspace),
        )
        # ``--modes`` is parsed by the ``eval`` and ``score`` commands, not
        # by ``run``. When the parent is ``eval``, the modes are stashed in
        # an internal env var so the benchmark card reflects the user's
        # intent. Direct ``belt run`` invocations leave the field
        # empty - that command does no scoring on its own.
        modes_raw = os.environ.get(_internal_envvars.SCORING_MODES, "")
        modes_list = [m.strip() for m in modes_raw.split(",") if m.strip()]
        scoring_block = {
            "modes": modes_list,
            "thresholds": _collect_threshold_args(ctx.args),
        }
        runtime_block = {
            "workers": int(getattr(ctx.args, "workers", 1) or 1),
            "trials": int(getattr(ctx.args, "trials", 1) or 1),
            "streaming": not bool(getattr(ctx.args, "no_stream", False)),
            "scenario_delay_s": float(getattr(ctx.args, "scenario_delay", 0.0) or 0.0),
        }
        enriched_meta = dict(minimal_meta)
        enriched_meta.update(
            {
                # ``initialize_run_dir`` is the first thing the live run
                # touches, so ``now()`` here is a tight upper bound on the
                # actual evaluation start. Recording it as a real UTC
                # timestamp avoids labelling the local-time-encoded run
                # directory name as if it were UTC, which would skew the
                # ``started_at`` field by the host's local-UTC offset.
                "started_at": _now_iso_utc(),
                "belt": belt_block.model_dump(),
                "host": host_block.model_dump(),
                "invocation": invocation_block.model_dump(),
                "scenarios": {
                    "scenarios_root": str(ctx.scenarios_root.resolve()),
                    "selected_groups": _selected_groups(ctx.args),
                    "selected_tags": _split_csv(getattr(ctx.args, "tags", None)),
                    "excluded_tags": _split_csv(getattr(ctx.args, "exclude_tags", None)),
                    "scenario_files": [sf.model_dump() for sf in scenario_files],
                    # Per-group effective config the runner actually loaded.
                    # Lets a downstream reader (results.json, dashboards,
                    # CI gates) tell from artifacts alone whether a given
                    # group ran with workspace_isolation off, without
                    # re-loading the source ``_config.json``.
                    "matched_groups": _matched_group_provenance(ctx),
                },
                "runtime": runtime_block,
                "scoring": scoring_block,
            }
        )
        write_json(ctx.run_dir / RUN_META_FILE, enriched_meta)
    except Exception as e:
        logger.warning("run_meta.json provenance enrichment incomplete: {}", e)

    _configure_logging(ctx.run_dir / LOG_FILE)
    return None


def _scenario_files_from_groups(ctx: RunContext) -> list[Path]:
    """Resolve absolute scenario-JSON paths for every scenario in the run.

    ``ScenarioLoader`` already resolved scenario JSONs during
    :func:`parse_and_filter`; we recompute the path here from
    ``group_dir / "<scenario_name>.json"``. This matches the loader's
    contract (one file per scenario, file basename equals scenario name)
    and avoids piping the loader's intermediate state through the runner.
    """
    paths: list[Path] = []
    for mg in ctx.matched_groups:
        for s in mg.scenarios:
            candidate = mg.group_dir / f"{s.name}.json"
            if candidate.is_file():
                paths.append(candidate)
    return paths


def _collect_threshold_args(args: argparse.Namespace) -> dict[str, Any]:
    """Snapshot threshold-related CLI flags for the card.

    Names match the argparse destinations (``min_pass_rate``,
    ``llm_fail_on``, ...) so the card field is a faithful copy of what
    the user passed - readable without cross-referencing argparse output.
    """
    out: dict[str, Any] = {}
    for name in ("min_pass_rate", "llm_fail_on", "fail_under", "max_cost_usd"):
        val = getattr(args, name, None)
        if val is not None:
            out[name] = val
    return out


def _split_csv(raw: str | list[str] | None) -> list[str]:
    """Split a comma-separated CLI value into a list of trimmed, non-empty entries.

    ``--tags`` and ``--exclude-tags`` are documented as comma-separated
    strings; iterating the raw value with ``list(...)`` would produce a
    per-character list and corrupt the reproducibility manifest.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [t.strip() for t in str(raw).split(",") if t.strip()]


def _selected_groups(args: argparse.Namespace) -> list[str]:
    """Return ``--scenarios`` filter values, normalised to a flat list of strings."""
    raw = getattr(args, "scenarios_filter", None) or getattr(args, "scenarios", None)
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return [str(raw)]


def _now_iso_utc() -> str:
    """Lazy-import wrapper around the ISO-UTC helper.

    ``benchmark_card`` is imported lazily because it pulls in
    ``constants`` and ``_redact``; importing it eagerly here would
    invert the dependency for tests that monkey-patch either module.
    """
    from belt.benchmark_card.io import iso_utc

    return iso_utc()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="belt run",
        description="Run evaluation scenarios via CLI agents",
        epilog="Agent-specific options are passed via --agent-arg KEY=VALUE. "
        "Use --agent NAME --help to see available options.",
        # ``allow_abbrev=False`` so ``--outcomes`` is not silently abbreviated
        # to ``--outcomes-dir`` (the root, not a specific run dir).
        allow_abbrev=False,
    )
    _add_common_run_args(parser)
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error(f"--workers must be >= 1, got {args.workers}")

    # ``--bundled`` rewrites ``args.path`` so every downstream consumer
    # (parse_and_filter, the manifest, the artifacts headline, the
    # ``-X repo_root=...`` injection) keeps working with a plain
    # filesystem path - it never has to know the run originated from
    # the bundled tree.
    if args.bundled is not None and args.path:
        parser.error(
            f"--bundled and the positional path are mutually exclusive. "
            f"Got --bundled={args.bundled!r} and path={args.path!r}."
        )
    if args.bundled is not None:
        from belt._bundled import bundled_groups, resolve_bundled_path

        resolved = resolve_bundled_path(args.bundled)
        if resolved is None:
            groups = bundled_groups()
            available = ", ".join(groups) if groups else "(none reachable)"
            parser.error(
                f"--bundled: '{args.bundled or '<root>'}' is not a bundled "
                f"scenarios path. Available groups: {available}."
            )
        args.path = str(resolved)
    if not args.path:
        parser.error("provide a scenarios path, or use --bundled <NAME> to run a bundled group.")

    # Idempotent terminal-log configuration. When called from ``belt eval``
    # the env var was set above; standalone invocations fall back to the
    # ``WARNING`` default. The file handler is wired up later from
    # :func:`initialize_run_dir` once the run directory exists.
    from belt._logging import configure_terminal_logging

    configure_terminal_logging(0)

    # Propagate --allow-full-env / --allow-arbitrary-agent via env vars so
    # agents loaded as entry points (and the registry resolver) see the
    # same toggle.
    envvars.forward_security_toggles(args)

    # ``--sandbox`` is read by the orchestrator at scenario-setup time via
    # ``BELT_SANDBOX_PROVIDER`` so the override survives subprocess hops
    # (``belt eval`` -> ``belt run``) without an extra plumbing parameter.
    # Validate the name now so a typo fails before any group setup.
    sandbox_choice = getattr(args, "sandbox", None)
    if sandbox_choice:
        from belt.runner.sandbox import available_sandbox_providers, get_sandbox_provider

        try:
            get_sandbox_provider(sandbox_choice)
        except KeyError:
            available = ", ".join(available_sandbox_providers()) or "(none)"
            parser.error(f"--sandbox: unknown provider '{sandbox_choice}'. Available: {available}")
        os.environ[envvars.SANDBOX_PROVIDER] = sandbox_choice

    result = parse_and_filter(args)
    if isinstance(result, int):
        return result
    scenarios_root, matched_groups, scenarios_skipped = result

    if not matched_groups:
        eprint("No matching scenarios found.")
        return 1

    ctx = build_run_context(args, scenarios_root, matched_groups)
    # Threaded into ``run_meta.json`` and re-read by ``commands/aggregate.py``
    # so ``AggregatedResults.scenarios_skipped`` reflects the run's true
    # input fleet, not just the scenarios that made it past the parser.
    ctx.scenarios_skipped = scenarios_skipped

    exit_code = show_header(ctx)
    if exit_code is not None:
        return exit_code

    exit_code = initialize_run_dir(ctx, argv=argv)
    if exit_code is not None:
        return exit_code

    cleanup_orphans(ctx)

    exit_code = setup_groups(ctx)
    if exit_code is not None:
        return exit_code

    all_results, interrupted = run_scenarios(ctx)

    teardown_groups(ctx)

    if interrupted:
        return 130
    # ``r.error`` is harness/subprocess failure; ``r.agent_errors`` is the
    # agent's own runtime failure (auth, refused, etc.). Either category
    # makes the run untrustworthy, so both contribute to the exit code -
    # see the run-phase footer in :mod:`belt.progress` for the
    # parallel UX change.
    if any(r.error for r in all_results) or any(r.agent_errors for r in all_results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
