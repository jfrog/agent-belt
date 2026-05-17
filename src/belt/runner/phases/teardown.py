# (c) JFrog Ltd. (2026)

"""Phase 4 - agent teardown + manifest unregister.

Skipped when ``--no-cleanup`` is set or when no groups completed setup.
``KeyboardInterrupt`` mid-teardown leaves the manifest entry in place; the
next run's ``cleanup_orphans`` will pick it up.
"""

from __future__ import annotations

import os

from belt.agent.registry import get_agent_class
from belt.runner.context import RunContext, create_agent


def teardown_groups(ctx: RunContext) -> None:
    """Tear down agent groups and unregister from manifest."""
    if not ctx.group_states or ctx.args.no_cleanup:
        return
    ctx.progress.console.print(f"Cleaning up [bold]{len(ctx.group_states)}[/bold] group(s)...", end=" ")
    try:
        for name, shared_state in ctx.group_states.items():
            try:
                gc = ctx.group_config_by_name[name]
                agent_cls = get_agent_class(gc.agent)
                agent = create_agent(agent_cls, ctx.agent_args)
                agent.teardown_group(shared_state)
            except Exception:
                pass
        ctx._manifest.unregister_run(os.getpid())  # type: ignore[attr-defined]
        ctx.progress.console.print("[green]done[/green]")
    except KeyboardInterrupt:
        ctx.progress.console.print("[yellow]skipped[/yellow]")
