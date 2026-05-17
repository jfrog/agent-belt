# (c) JFrog Ltd. (2026)

"""belt quickstart - zero-to-first-run in one command.

Validates that a specific agent is ready, runs the simplest built-in scenario
for that agent, and prints next-step guidance.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from belt.agent.base import AgentNotAvailableError
from belt.agent.registry import available_agents, get_agent_class
from belt.constants import EXAMPLE_LLM_MODEL

# Quickstart runs a single, agent-agnostic, no-fixture scenario whose only
# job is to answer "does the agent reply at all?". Living under
# ``showcase/correctness/`` keeps the quickstart aligned with the same
# capability matrix the rest of the documentation walks through.
_QUICKSTART_GROUP = "showcase/correctness"
_QUICKSTART_SCENARIO = "correctness_basic"


def _find_examples_dir() -> Path | None:
    """Locate a scenarios directory containing the quickstart group.

    Thin wrapper around :func:`belt._bundled.bundled_scenarios_root` -
    the shared helper used by ``belt eval --bundled`` to resolve the
    same tree from a single source of truth.
    """
    from belt._bundled import bundled_scenarios_root

    root = bundled_scenarios_root()
    if root is None:
        return None
    if not (root / _QUICKSTART_GROUP).is_dir():
        return None
    return root


def _validate_agent(name: str, console: Console) -> bool:
    """Check agent availability and print result. Returns True if ready."""
    if name not in available_agents():
        all_names = ", ".join(available_agents())
        console.print(f"\n  [red]Unknown agent '{name}'.[/red] Available: {all_names}")
        console.print("  Run [cyan]belt doctor[/cyan] to see which agents are ready.\n")
        return False

    console.print(f"\nChecking {name}...", end=" ")
    try:
        cls = get_agent_class(name)
        cls.check_available()
    except AgentNotAvailableError as e:
        console.print("[red]✗[/red]")
        console.print(f"  {e.reason}")
        if e.suggestion:
            console.print(f"  [dim]→ {e.suggestion}[/dim]")
        console.print(f"\n  Fix the issue above, then re-run: [cyan]belt quickstart {name}[/cyan]\n")
        return False
    except Exception as e:
        console.print(f"[red]✗[/red] {e}")
        return False

    try:
        info = cls.display_info()
    except Exception:
        info = "ready"
    console.print(f"[green]✓[/green] {info}")
    return True


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``belt quickstart``."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="belt quickstart",
        description="Validate an agent and run the simplest scenario to verify your setup",
    )
    ap.add_argument(
        "agent",
        nargs="?",
        help="Agent name (run 'belt agent list' to see options). " "If omitted, uses the first available agent.",
    )
    args = ap.parse_args(argv)

    console = Console(stderr=True)

    agent_name = args.agent
    if not agent_name:
        for name in available_agents():
            try:
                cls = get_agent_class(name)
                cls.check_available()
                agent_name = name
                break
            except Exception:
                continue
        if not agent_name:
            console.print("\n  [red]No agents available.[/red] Run [cyan]belt doctor[/cyan] to diagnose.\n")
            return 1

    if not _validate_agent(agent_name, console):
        return 1

    scenarios_dir = _find_examples_dir()
    if scenarios_dir is None:
        console.print("\n  [red]Cannot find examples/scenarios/ directory.[/red]")
        console.print("  Run from the belt repo root, or install with: pip install -e '.[dev]'\n")
        return 1

    group_dir = scenarios_dir / _QUICKSTART_GROUP
    scenario_file: Path | None = None
    candidate = group_dir / f"{_QUICKSTART_SCENARIO}.json"
    if candidate.is_file():
        scenario_file = candidate
    if scenario_file is None and group_dir.is_dir():
        jsons = sorted(p for p in group_dir.glob("*.json") if not p.name.startswith("_"))
        scenario_file = jsons[0] if jsons else None
    if not scenario_file:
        console.print("\n  [red]No quickstart scenarios found.[/red]")
        console.print(f"  Expected: {group_dir}/\n")
        return 1

    console.print(
        f"\nRunning: [bold]{scenario_file.stem}[/bold] with [bold]{agent_name}[/bold] (rules-only, no LLM key needed)\n"
    )

    from belt.commands.eval import main as eval_main

    # Filter is relative to scenarios_dir. The quickstart scenario lives in
    # ``showcase/correctness/`` and is agent-agnostic - we override its
    # configured agent with --agent so the same scenario verifies any agent.
    filter_arg = f"{_QUICKSTART_GROUP}/{scenario_file.stem}"
    eval_argv = [
        str(scenarios_dir),
        "--scenarios",
        filter_arg,
        "--agent",
        agent_name,
        "--modes",
        "rules",
    ]
    rc = eval_main(eval_argv)

    console.print()
    if rc == 0:
        # The ``--bundled <GROUP>`` form is the portable next step for
        # pip-installed users: their copy of the showcase lives inside
        # the wheel and has no human-meaningful absolute path. The flag
        # resolves to the same on-disk tree that the quickstart just
        # used (``scenarios_dir``), so any of these one-liners produce
        # the same wiring regardless of how ``agent-belt`` was installed.
        console.print("[green bold]You're set![/green bold] Next steps:\n")
        console.print(f"  [cyan]belt eval --bundled showcase --agent {agent_name} --modes rules[/cyan]")
        console.print(f"       Run the full schema-feature showcase against {agent_name}\n")
        console.print("  [cyan]belt eval --bundled showcase --modes rules --workers 3[/cyan]")
        console.print("       All scenarios in parallel\n")
        console.print("  [cyan]belt eval --bundled showcase --scorer-arg model=ollama/gemma4[/cyan]")
        console.print("       Add LLM judge scoring with a local model (no API key needed)\n")
        console.print(f"  [cyan]belt eval --bundled showcase --scorer-arg model={EXAMPLE_LLM_MODEL}[/cyan]")
        console.print("       Add LLM judge scoring with OpenAI (needs BELT_OPENAI_API_KEY)\n")
        console.print("  [cyan]belt eval --bundled showcase --progress live --workers 3[/cyan]")
        console.print("       Live TUI with progress + agent output\n")
    else:
        console.print("[yellow]The scenario run had issues.[/yellow] Debug with:\n")
        console.print("  [cyan]belt doctor[/cyan]                  Check your setup")
        console.print("  [cyan]ls outcomes/[/cyan]                      Inspect artifacts\n")

    return rc
