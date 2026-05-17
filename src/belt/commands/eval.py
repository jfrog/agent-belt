# (c) JFrog Ltd. (2026)

"""belt eval - unified run + score + aggregate command.

The primary entry point for evaluation. Chains the three phases:
1. Run scenarios via agent (commands/run.py)
2. Score outcomes with rules + LLM judge (commands/score.py)
3. Aggregate and report with threshold enforcement (commands/aggregate.py)

Accepts flags from all three phases; each phase ignores flags it doesn't recognize.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from belt import envvars
from belt._ui import eprint
from belt.constants import EXAMPLE_LLM_MODEL
from belt.scorer.entities import DEFAULT_LLM_FAIL_ON_STR


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="belt eval",
        description="Run + score + aggregate evaluation in one shot",
        # ``allow_abbrev=False`` keeps the rename ``--outcomes`` → ``--run-dir``
        # honest. With abbreviation on, ``--outcomes`` would silently match
        # ``--outcomes-dir`` (different semantics: root vs specific run dir).
        allow_abbrev=False,
    )

    # ── Scenario selection ──
    sel = parser.add_argument_group("Scenario selection")
    # ``path`` is optional because ``--bundled`` is the
    # pip-install-friendly alternative: ``belt eval --bundled showcase``
    # resolves to the wheel-included tree (``belt/_bundled_examples/
    # scenarios/showcase``). One of the two must be set; passing both
    # is rejected after parsing so the error message can name the
    # specific values the user gave.
    sel.add_argument(
        "path",
        nargs="?",
        help="Root scenarios directory (positional). Omit when using --bundled.",
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
            "Run scenarios shipped inside the agent-belt wheel. "
            "``--bundled showcase`` is the recommended first run for "
            "``pip install agent-belt`` users. Pass a deeper path "
            "(``--bundled showcase/correctness``) to scope further, or "
            "use ``--bundled`` alone for the full bundled tree."
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

    # ── Agent options ──
    agent = parser.add_argument_group("Agent options")
    agent.add_argument(
        "-X",
        "--agent-arg",
        action="append",
        dest="agent_args",
        metavar="KEY=VALUE",
        help=(
            "Agent-specific option (repeatable, passthrough). "
            f"E.g.: --agent-arg model={EXAMPLE_LLM_MODEL} --agent-arg server_id=abc123"
        ),
    )

    # ── Scoring ──
    scoring = parser.add_argument_group("Scoring")
    scoring.add_argument("--modes", default="rules,llm", help="Scoring modes: rules, llm (default: rules,llm)")
    scoring.add_argument(
        "-S",
        "--scorer-arg",
        action="append",
        dest="scorer_args",
        metavar="KEY=VALUE",
        help=(
            "Scorer-specific option (repeatable, passthrough). "
            f"E.g.: --scorer-arg model={EXAMPLE_LLM_MODEL} --scorer-arg temperature=0.0 --scorer-arg seed=42"
        ),
    )
    scoring.add_argument(
        "--scorer-config",
        metavar="PATH",
        help="YAML config for multi-judge scoring",
    )

    # ── Thresholds ──
    agg = parser.add_argument_group("Thresholds")
    agg.add_argument(
        "--llm-fail-on",
        default=DEFAULT_LLM_FAIL_ON_STR,
        help=(
            f"Comma-separated LLM verdicts that count as failures. "
            f"Default ``{DEFAULT_LLM_FAIL_ON_STR}`` covers ternary 'low', "
            f"binary 'fail', and inconclusive verdicts on either scale. Pass "
            f"e.g. ``low,fail`` to treat inconclusive as informational only."
        ),
    )
    agg.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="MODE/DIM:PCT",
        help="Per-dimension failure threshold (repeatable). Example: --threshold rules/execution:0",
    )

    # ── Execution ──
    # Within this group, ``add_argument`` calls are alphabetised by long
    # flag name. Enforced by ``tests/test_cli_order.py``.
    exe = parser.add_argument_group("Execution")
    exe.add_argument(
        "--allow-arbitrary-agent",
        action="store_true",
        default=False,
        help=(
            "Allow --agent to resolve to a dotted import path. By default "
            "only agents registered as ``belt.agents`` entry points "
            "are loadable."
        ),
    )
    exe.add_argument(
        "--allow-arbitrary-exporter",
        action="store_true",
        default=False,
        help=(
            "Allow --export to resolve to a dotted import path. By default "
            "only built-in exporters and exporters registered as "
            "``belt.exporters`` entry points are loadable."
        ),
    )
    exe.add_argument(
        "--allow-arbitrary-scorer",
        action="store_true",
        default=False,
        help=(
            "Allow scorers to resolve to a dotted import path. By default "
            "only scorers registered as ``belt.scorers`` entry points "
            "are loadable."
        ),
    )
    exe.add_argument(
        "--allow-external-working-dir",
        action="store_true",
        default=False,
        help=(
            "Permit a group's working_dir to resolve outside the scenarios "
            "root (the positional path argument). By default this is rejected "
            "to prevent scenarios from reaching into unrelated parts of the "
            "host filesystem."
        ),
    )
    exe.add_argument(
        "--allow-full-env",
        action="store_true",
        default=False,
        help=(
            "Pass belt's full os.environ to every CLI agent subprocess. "
            "By default agents only see a curated minimal env (PATH, locale, "
            "proxy/TLS, and provider keys the agent declares as required)."
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
            "worktree isolation by accident."
        ),
    )
    exe.add_argument(
        "--allow-insecure-base-url",
        action="store_true",
        default=False,
        help=(
            "Permit LLM judge calls over plain http:// to a non-loopback host. "
            "Off by default - belt refuses with ConfigError."
        ),
    )
    exe.add_argument("--dry-run", action="store_true", help="List matching scenarios without executing")
    exe.add_argument("--no-cleanup", action="store_true", default=False, help="Skip group teardown after run")
    exe.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Disable per-turn NDJSON stream files (turn_N_stream.ndjson). Slightly less disk I/O.",
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
            "Override the sandbox provider for this run. ``host`` (default; agent "
            "runs on the host with the invoking user's privileges and no isolation) "
            "or ``docker`` (each agent subprocess in a container with --cap-drop=ALL, "
            "--read-only rootfs, the worktree as the only writable mount, env "
            "passthrough by exact name). Per-scenario image / hosts / passthrough are "
            "still read from the group's ``_config.json``. Forwarded to ``belt run`` "
            "and recorded in the benchmark card. See SANDBOXING.md."
        ),
    )
    exe.add_argument(
        "--scenario-delay",
        type=float,
        default=0,
        help="Seconds between dispatching parallel scenarios (default: 0)",
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
        help="Run each scenario N times for reliability measurement (default: 1). "
        "When >1 the aggregator reports pass@1 / pass^k metrics.",
    )
    exe.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=(
            "Increase terminal log verbosity. Default prints WARNING and "
            "above. ``-v`` bumps to INFO (per-scenario judge reasoning and "
            "trajectory diagnostics shown inline), ``-vv`` to DEBUG. The "
            "transcript log at ``<run_dir>/eval.log`` always runs at DEBUG "
            "regardless of this flag. Same effect as setting "
            f"{envvars.LOG_LEVEL}=info|debug."
        ),
    )
    exe.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Max parallel tasks (default: 1). Note: concurrent runs sharing the same "
        "outcomes directory are not coordinated - use external locking or separate "
        "output dirs when running multiple eval processes.",
    )

    # Within this group, ``add_argument`` calls are alphabetised by long flag
    # name. Enforced by ``tests/test_cli_order.py``.
    out = parser.add_argument_group("Output")
    out.add_argument(
        "--export",
        action="append",
        default=[],
        metavar="NAME:PATH",
        help=(
            "Run an exporter after aggregation, writing to PATH. Repeatable. "
            "NAME is any name shown by `belt doctor` under Exporters. "
            "Example: --export csv:results.csv --export junit:report.xml."
        ),
    )
    out.add_argument(
        "--export-config",
        metavar="PATH",
        help="YAML file describing exporter entries (name + path + options).",
    )
    out.add_argument(
        "--outcomes-dir",
        metavar="PATH",
        help="Root directory for outcomes (default: $BELT_OUTCOMES_DIR or cwd/outcomes)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error(f"--workers must be >= 1, got {args.workers}")

    # ``--bundled`` is mutually exclusive with the positional path. Both
    # forms feed the same ``args.path`` downstream so every existing
    # call site (run/score/aggregate, log lines, error messages) keeps
    # working unchanged; only the resolution shifts from
    # "user-supplied directory" to "wheel-included tree".
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

    # Configure terminal logging once at the top, before any phase emits
    # log records. ``-v`` is mirrored into ``BELT_LOG_LEVEL`` so that
    # nested calls into ``belt run`` / ``belt score`` / ``belt aggregate``
    # (which each call :func:`configure_terminal_logging` from their own
    # ``main()``) pick the same level up from the env. Returns the
    # resolved level so the post-run banner can point the user at ``-v``
    # when something interesting was hidden.
    from belt._logging import configure_terminal_logging

    if args.verbose >= 2:
        os.environ[envvars.LOG_LEVEL] = "DEBUG"
    elif args.verbose == 1:
        os.environ[envvars.LOG_LEVEL] = "INFO"
    configure_terminal_logging(args.verbose)

    if args.threshold:
        from belt.aggregator.thresholds import parse_threshold

        for raw in args.threshold:
            try:
                parse_threshold(raw)
            except Exception as e:
                parser.error(str(e))

    # Stash the user's original eval invocation so the benchmark card
    # records ``belt eval ...`` rather than the synthesised
    # ``belt run ...`` argv we forward to ``run_main`` further down.
    # Reads any caller-provided ``argv``; falls back to ``sys.argv`` so
    # a plain console-script invocation also produces a faithful card.
    import json as _json
    import sys as _sys

    from belt import _internal_envvars

    original_argv = list(argv) if argv is not None else list(_sys.argv)
    os.environ[_internal_envvars.ORIGINAL_ARGV] = _json.dumps(original_argv)

    # Plumb scoring modes through to ``commands/run.py`` so the benchmark
    # card's scoring block reflects the user's intent. ``--modes`` is
    # parsed by ``eval`` (and ``score``), never by ``run`` itself, so
    # ``run.args.modes`` would be ``None`` without this hand-off.
    os.environ[_internal_envvars.SCORING_MODES] = args.modes

    # Propagate every ``--allow-*`` flag to the subprocess-env builder and
    # the agent / scorer registries via environment variables so code
    # paths loaded later (entry points, subcommands invoked by name) see
    # the same toggle.
    envvars.forward_security_toggles(args)

    agent_args = list(args.agent_args or [])
    scorer_args = list(args.scorer_args or [])

    # ── Preflight: validate scorer modes early (even dry-run), full availability only for live runs ──
    from belt.scorer.registry import available_scorers, get_scorer_class

    mode_set = {m.strip() for m in args.modes.split(",") if m.strip()}
    for mode_name in sorted(mode_set):
        try:
            get_scorer_class(mode_name)
        except Exception:
            all_scorers = available_scorers()
            eprint(f"\n  ❌ Unknown scoring mode '{mode_name}'. Available: {', '.join(all_scorers)}")
            return 1

    if not args.dry_run:
        # Live runs probe judge model callability so a wrong key /
        # typo'd model / project-scoped key without access fails in
        # under a second, before the (expensive) agent phase. Dry-run
        # is intentionally skipped because it must be offline-safe:
        # air-gapped CI runs ``belt eval --dry-run`` to lint scenarios
        # without spending money or touching the network, and a
        # mandatory probe would break that contract.
        try:
            from belt.commands.score import validate_scorers

            scorer_descriptions = validate_scorers(
                args.modes,
                scorer_args,
                getattr(args, "scorer_config", None),
                probe_api=True,
            )
            import json

            from belt._internal_envvars import SCORER_DESCS

            os.environ[SCORER_DESCS] = json.dumps(scorer_descriptions)
        except Exception as e:
            eprint(f"\n  ❌ Scorer preflight failed - aborting before running scenarios.\n\n  {e}")
            return 1

    # ── Phase 1: Run ──
    run_argv = [args.path]
    if args.scenarios:
        run_argv += ["--scenarios", args.scenarios]
    if args.tags:
        run_argv += ["--tags", args.tags]
    if args.agent:
        run_argv += ["--agent", args.agent]
    for xa in agent_args:
        run_argv += ["-X", xa]
    if args.workers > 1:
        run_argv += ["--workers", str(args.workers)]
    if args.no_cleanup:
        run_argv += ["--no-cleanup"]
    if args.no_stream:
        run_argv += ["--no-stream"]
    if args.scenario_delay > 0:
        run_argv += ["--scenario-delay", str(args.scenario_delay)]
    if args.trials > 1:
        run_argv += ["--trials", str(args.trials)]
    run_argv += ["--progress", args.progress]
    if args.progress == "live":
        # Always forward - the previous "only if != 30" guard silently
        # dropped the flag when the user explicitly passed --progress-live-lines 30.
        run_argv += ["--progress-live-lines", str(args.progress_live_lines)]
    if args.dry_run:
        run_argv += ["--dry-run"]
    if args.strict:
        run_argv += ["--strict"]
    if args.strict_config:
        run_argv += ["--strict-config"]
    if args.outcomes_dir:
        run_argv += ["--outcomes-dir", args.outcomes_dir]
    # Forward every ``--allow-*`` flag the user passed so the run subprocess
    # parses the same args. The env var is also set above for agents /
    # registries that read it directly; forwarding the flag keeps the run
    # phase's argparse view consistent.
    if getattr(args, "allow_arbitrary_agent", False):
        run_argv += ["--allow-arbitrary-agent"]
    if getattr(args, "allow_arbitrary_scorer", False):
        run_argv += ["--allow-arbitrary-scorer"]
    if getattr(args, "allow_external_working_dir", False):
        run_argv += ["--allow-external-working-dir"]
    if getattr(args, "allow_full_env", False):
        run_argv += ["--allow-full-env"]
    if getattr(args, "allow_inplace", False):
        run_argv += ["--allow-inplace"]
    if getattr(args, "allow_insecure_base_url", False):
        run_argv += ["--allow-insecure-base-url"]
    if getattr(args, "sandbox", None):
        run_argv += ["--sandbox", args.sandbox]

    from belt.commands.run import main as run_main

    run_rc = run_main(run_argv)

    if args.dry_run:
        return run_rc

    from belt._internal_envvars import LAST_RUN_DIR
    from belt.constants import TURN_OUTPUT_TEMPLATE

    run_dir_str = os.environ.get(LAST_RUN_DIR)
    if not run_dir_str or not Path(run_dir_str).is_dir():
        eprint("\n  No run directory produced by run phase - skipping score and aggregate.")
        return 1

    # ``turn_*_output.json`` is written by ``orchestrator._persist_turn_output``
    # only when the agent actually produced a turn. ``turn_*_cli.txt`` is
    # also written as a sentinel when a scenario raises before the agent
    # is invoked, so testing it would let an all-setup-failed run waste
    # LLM judge tokens scoring its own error messages. The output JSON is
    # the source of truth for "a real turn happened here".
    run_dir = Path(run_dir_str)
    has_real_turns = any(run_dir.rglob(TURN_OUTPUT_TEMPLATE.format("*")))
    if not has_real_turns:
        eprint("\n  Score/aggregate skipped - no scenario produced a real turn." f"\n  → belt view {run_dir}")
        return run_rc if run_rc != 0 else 1

    # The "proceeding to score partial outcomes" banner only makes sense
    # when there *are* outcomes to score. Reading turn-output presence
    # before this branch keeps the message honest: if every scenario
    # short-circuited at setup, we already routed through the "no real
    # turn" return above; reaching here means at least one turn landed.
    if run_rc != 0:
        eprint(f"\n  ⚠ Run phase exited with code {run_rc} - proceeding to score partial outcomes.")

    # ── Phase 2: Score ──
    score_argv: list[str] = ["--run-dir", run_dir_str]
    score_argv += ["--modes", args.modes]
    for sa in scorer_args:
        score_argv += ["-S", sa]
    if getattr(args, "scorer_config", None):
        score_argv += ["--scorer-config", args.scorer_config]
    if args.workers > 1:
        score_argv += ["--workers", str(args.workers)]
    score_argv += ["--progress", args.progress]
    # Forward scorer / agent / base-url gates so the score subprocess's
    # argparse view matches what the user originally passed to ``eval``.
    # The env vars set above are still authoritative at the registry layer.
    if getattr(args, "allow_arbitrary_agent", False):
        score_argv += ["--allow-arbitrary-agent"]
    if getattr(args, "allow_arbitrary_scorer", False):
        score_argv += ["--allow-arbitrary-scorer"]
    if getattr(args, "allow_insecure_base_url", False):
        score_argv += ["--allow-insecure-base-url"]

    from belt.commands.score import main as score_main

    # ``chained=True`` tells the scorer's progress reporter to drop its
    # pass-count headline + judge-cost subpart so the aggregator panel
    # below is the single canonical scoreboard for ``belt eval``. See
    # belt.progress.ScorerProgress.summary.
    score_rc = score_main(score_argv, chained=True)
    if score_rc != 0:
        eprint(f"\n  ⚠ Score phase exited with code {score_rc} - proceeding to aggregate partial results.")

    # ── Phase 3: Aggregate ──
    agg_argv: list[str] = ["--run-dir", run_dir_str]
    for t in args.threshold:
        agg_argv += ["--threshold", t]
    if args.llm_fail_on != DEFAULT_LLM_FAIL_ON_STR:
        agg_argv += ["--llm-fail-on", args.llm_fail_on]
    for spec in args.export or []:
        agg_argv += ["--export", spec]
    if getattr(args, "export_config", None):
        agg_argv += ["--export-config", args.export_config]

    from belt.commands.aggregate import main as agg_main

    # The aggregator's results panel ends with a single ``→ belt view
    # <run_dir>`` line (plus a ``(pass -v for details)`` hint when failures
    # were hidden by the WARNING-default verbosity). No additional banner
    # here: each extra line repeats the run directory the reader already
    # has, and the SOTA (inspect view, promptfoo view) is one viewer
    # pointer not three.
    return agg_main(agg_argv)
