#!/usr/bin/env python3
# (c) JFrog Ltd. (2026)

"""belt - Evaluation harness for real headless CLI agents.

Subcommands:
    eval       Run + score + aggregate in one shot (the common case)
    run        Execute evaluation scenarios via agent-driven CLI agents
    score      Score evaluation outcomes using rule + LLM scorers
    aggregate  Aggregate scores into summary reports with threshold enforcement
    export     Emit a completed run to one or more configured destinations
    compare    Compare two evaluation result sets side-by-side
    watch      Watch live agent output during evaluation runs
    view       Browse evaluation results in the terminal
    gc         Prune old run directories under outcomes/
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from belt._ui import eprint


def _report_error(exc: Exception, *, unexpected: bool = False) -> None:
    """Render a user-facing error to stderr."""
    from belt.errors import BeltError

    if unexpected:
        label = f"{type(exc).__name__}: {exc}"
    elif isinstance(exc, BeltError):
        label = str(exc)
    else:
        label = f"{type(exc).__name__}: {exc}"

    print(f"\n  ❌ {label}", file=sys.stderr)

    from belt import _internal_envvars, envvars

    log_path = os.environ.get(_internal_envvars.LOG_FILE, "")
    if log_path:
        print(f"     → logs: {log_path}", file=sys.stderr)

    if unexpected:
        print(f"     → set {envvars.DEBUG}=1 for full traceback", file=sys.stderr)

    if envvars.is_truthy(envvars.DEBUG):
        traceback.print_exc(file=sys.stderr)


def _dispatch() -> int:
    """Parse top-level args and dispatch to the appropriate subcommand."""
    parser = argparse.ArgumentParser(
        prog="belt",
        description="Evaluation harness for real headless CLI agents",
    )
    try:
        from importlib.metadata import version as pkg_version

        ver = pkg_version("agent-belt")
    except Exception:
        ver = "0.0.0"
    parser.add_argument("--version", action="version", version=f"belt {ver}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("eval", help="Run + score + aggregate (the common case)", add_help=False)
    subparsers.add_parser("run", help="Run evaluation scenarios", add_help=False)
    subparsers.add_parser("score", help="Score evaluation outcomes", add_help=False)
    subparsers.add_parser("aggregate", help="Aggregate scores into reports", add_help=False)
    subparsers.add_parser("export", help="Emit a completed run to one or more configured destinations", add_help=False)
    subparsers.add_parser("compare", help="Compare two result sets", add_help=False)
    subparsers.add_parser("watch", help="Watch live agent output during runs", add_help=False)
    subparsers.add_parser("view", help="Browse evaluation results in the terminal", add_help=False)
    subparsers.add_parser("agent", help="Agent management (list, info)")
    subparsers.add_parser("doctor", help="Check agent and LLM scoring readiness", add_help=False)
    subparsers.add_parser("quickstart", help="Validate an agent and run a first scenario", add_help=False)
    subparsers.add_parser("gc", help="Prune old run directories under outcomes/", add_help=False)
    subparsers.add_parser("version", help="Print version and exit")

    args, remaining = parser.parse_known_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "eval":
        from belt.commands.eval import main as eval_main

        return eval_main(remaining)
    elif args.command == "run":
        from belt.commands.run import main as run_main

        return run_main(remaining)
    elif args.command == "score":
        from belt.commands.score import main as score_main

        return score_main(remaining)
    elif args.command == "aggregate":
        from belt.commands.aggregate import main as agg_main

        return agg_main(remaining)
    elif args.command == "export":
        from belt.commands.export import main as export_main

        return export_main(remaining)
    elif args.command == "compare":
        from belt.commands.compare import main as compare_main

        return compare_main(remaining)
    elif args.command == "watch":
        from belt.commands.watch import main as watch_main

        return watch_main(remaining)
    elif args.command == "view":
        from belt.commands.view import main as view_main

        return view_main(remaining)
    elif args.command == "agent":
        return _agent_subcommand(remaining)
    elif args.command == "doctor":
        from belt.commands.doctor import main as doctor_main

        return doctor_main(remaining)
    elif args.command == "quickstart":
        from belt.commands.quickstart import main as quickstart_main

        return quickstart_main(remaining)
    elif args.command == "gc":
        from belt.commands.gc import main as gc_main

        return gc_main(remaining)
    elif args.command == "version":
        print(f"belt {ver}")
        return 0
    else:
        parser.print_help()
        return 1


def main() -> int:
    """Entry point with top-level exception boundary (Principle 7).

    Sets a restrictive umask (0o077) before any I/O so files belt
    creates default to owner-only read/write. This complements the explicit
    ``mode=0o700`` we set on directories that hold scenario stdout, NDJSON
    streams, and the score cache. Users who intentionally want
    world-readable artefacts can opt out via ``envvars.NO_UMASK=1``
    (e.g. shared CI cache dirs they manage).
    """
    from belt import envvars

    if not envvars.is_truthy(envvars.NO_UMASK):
        os.umask(0o077)
    try:
        return _dispatch()
    except KeyboardInterrupt:
        eprint("\nInterrupted.")
        return 130
    except Exception as e:
        from belt.errors import BeltError

        _report_error(e, unexpected=not isinstance(e, BeltError))
        return 1


def _agent_subcommand(argv: list[str]) -> int:
    """Handle `belt agent {list,info}`."""
    parser = argparse.ArgumentParser(prog="belt agent")
    sub = parser.add_subparsers(dest="agent_command")
    sub.add_parser("list", help="List available agents")
    info_parser = sub.add_parser("info", help="Show details for a specific agent")
    info_parser.add_argument("name", help="Agent name")

    args = parser.parse_args(argv)

    if args.agent_command == "list":
        return _agent_list()
    elif args.agent_command == "info":
        return _agent_info(args.name)
    else:
        parser.print_help()
        return 0


def _agent_list() -> int:
    from belt.agent.registry import available_agents, get_agent_class

    agents = available_agents()
    if not agents:
        eprint("No agents registered.")
        return 0
    eprint(f"{'Name':<16} {'Class':<30} {'Available'}")
    eprint(f"{'─' * 16} {'─' * 30} {'─' * 10}")
    for name in agents:
        try:
            cls = get_agent_class(name)
            class_name = cls.__name__
            try:
                cls.check_available()
                status = "✓"
            except Exception:
                status = "✗"
        except Exception as e:
            class_name = f"(error: {e})"
            status = "✗"
        eprint(f"{name:<16} {class_name:<30} {status}")
    return 0


def _agent_info(name: str) -> int:
    from belt.agent.base import AgentNotAvailableError
    from belt.agent.registry import get_agent_class
    from belt.errors import BeltError

    try:
        cls = get_agent_class(name)
    except (ValueError, BeltError) as e:
        eprint(f"Error: {e}")
        return 1

    eprint(f"Agent: {name}")
    eprint(f"Class: {cls.__module__}.{cls.__name__}")

    try:
        cls.check_available()
        eprint("Available: ✓")
    except AgentNotAvailableError as e:
        eprint(f"Available: ✗ ({e.reason})")
        if e.suggestion:
            eprint(f"  → {e.suggestion}")

    options = cls.cli_options()
    if options:
        eprint("\nOptions (-X):")
        _SECRET_MARKERS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")
        for opt in options:
            req = " (required)" if opt.required else ""
            env = ""
            if opt.env_var:
                current = os.environ.get(opt.env_var)
                is_secret = any(m in opt.env_var.upper() for m in _SECRET_MARKERS)
                if current:
                    display = "***" if is_secret else current
                    env = f" [env: {opt.env_var}={display}]"
                else:
                    env = f" [env: {opt.env_var} (unset)]"
            eprint(f"  {opt.name:<24} {opt.help}{req}{env}")
    else:
        eprint("\nNo agent-specific options.")

    info = cls.display_info()
    if info and info != cls.__name__:
        eprint(f"\nInfo: {info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
