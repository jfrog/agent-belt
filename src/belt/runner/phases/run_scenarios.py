# (c) JFrog Ltd. (2026)

"""Phase 3 - execute scenarios via the thread pool.

The runner parallelises at the *scenario* level: group resources are set up
once (phase 2), then individual scenarios are dispatched to a thread pool so
that slow scenarios in one group don't block faster scenarios in another.
``--trials N>1`` repeats each scenario into a sibling outcome dir suffixed
``__trial_N`` (see ``constants.TRIAL_SUFFIX_RE`` for the canonical pattern).
"""

from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from loguru import logger

from belt.agent.registry import get_agent_class
from belt.constants import TURN_CLI_TEMPLATE
from belt.runner.context import RunContext, create_agent
from belt.runner.entities import ScenarioResult
from belt.runner.orchestrator import build_agent_config, run_scenario_turns
from belt.runner.workspace import WorkspaceError, WorkspaceManager
from belt.scenario import GroupConfig, Scenario


def emit_implicit_default_model_warnings(ctx: RunContext) -> list[str]:
    """Warn (once per agent) when an agent will fall back to its CLI's built-in default model.

    Returns the list of agent names that received a warning, mostly for
    test introspection.  When neither ``-X model=...`` nor the agent's
    ``env_var`` is set, the agent CLI silently uses *its own* internal
    default - which may be a model the user's account can't access.  The
    failure that surfaces is whatever the provider returns ("model not
    available", "401", "404", "insufficient quota") - opaque enough that
    users (and debugging agents) commonly blame their own changes.

    This proactive nudge is the first thing the user sees on stderr / in
    ``eval.log`` when they run with no model selection, so the cause is
    obvious before any failure occurs.
    """
    warned: list[str] = []
    seen_agents: set[str] = set()
    for mg in ctx.matched_groups:
        agent_name = mg.config.agent
        if agent_name in seen_agents:
            continue
        seen_agents.add(agent_name)
        try:
            agent_cls = get_agent_class(agent_name)
        except Exception:
            continue
        model_opt = next((o for o in agent_cls.cli_options() if o.name == "model"), None)
        if model_opt is None:
            continue
        if "model" in ctx.agent_args:
            continue
        if model_opt.env_var and os.environ.get(model_opt.env_var):
            continue
        env_hint = (
            f"    export {model_opt.env_var}=<your-model>"
            if model_opt.env_var
            else "    (set via the agent's documented env var)"
        )
        logger.warning(
            "Agent '{agent}' will run without an explicit model - falling back to {agent}'s built-in default. "
            "If runs fail with 'model not available' / 'unauthorized' / '404' / 'insufficient quota', set one of:\n"
            "    belt eval ... -X model=<your-model>\n"
            "{env_hint}",
            agent=agent_name,
            env_hint=env_hint,
        )
        warned.append(agent_name)
    return warned


def run_scenarios(ctx: RunContext) -> tuple[list[ScenarioResult], bool]:
    """Execute all scenarios via the thread pool. Returns (results, interrupted)."""
    args = ctx.args
    n_trials = ctx.n_trials
    is_live = args.progress == "live"
    workers = min(args.workers, ctx.total_scenarios)

    emit_implicit_default_model_warnings(ctx)

    all_results: list[ScenarioResult] = []
    scenario_tasks: list[tuple[Scenario, Any, Path, GroupConfig, str, int]] = []
    n = 0
    for mg in ctx.matched_groups:
        if mg.name in ctx.failed_groups:
            for s in mg.scenarios:
                for _trial in range(n_trials):
                    n += 1
                    result = ScenarioResult(
                        scenario_name=s.name,
                        group_path=mg.name,
                        error="group setup failed",
                    )
                    all_results.append(result)
                    # No ``scenario_done`` call: the run progress was
                    # started over *non-failed* groups (see
                    # ``setup_groups`` end), so advancing here would
                    # either no-op on an unknown task (live mode) or
                    # push ``_completed`` past the live ``_total``
                    # (plain mode). ``setup_errors`` and the
                    # ``ScenarioResult`` above carry the casualty count.
            continue

        for s in mg.scenarios:
            for trial in range(n_trials):
                n += 1
                suffix = f"__trial_{trial}" if n_trials > 1 else ""
                outcome_dir = ctx.run_dir / mg.group_dir.resolve().relative_to(ctx.scenarios_root) / f"{s.name}{suffix}"
                scenario_tasks.append((s, ctx.group_states[mg.name], outcome_dir, mg.config, mg.name, n))

    def _run_one(task: tuple) -> ScenarioResult:
        scenario, shared_state, outcome_dir, group_config, group_name, num = task
        logger.info("[{}/{}] {}/{}", num, ctx.total_scenarios, group_name, scenario.name)
        try:
            agent_cls = get_agent_class(group_config.agent)
            agent = create_agent(agent_cls, ctx.agent_args)
            config = build_agent_config(group_config, scenario, shared_state)
            do_stream = is_live or not args.no_stream

            ws_manager: WorkspaceManager | None = None
            base_repo: Path | None = None
            workspace_ref = group_config.workspace_ref
            fixture_dir = ctx.group_fixtures.get(group_name)
            if fixture_dir is not None:
                # ``setup_groups`` already cloned the fixture and rejected any
                # group that combined ``fixture_repo`` with ``working_dir``.
                # The clone lives under the run dir, so the external-working-dir
                # check does not apply.
                base_repo = fixture_dir
                workspace_ref = group_config.fixture_ref
            elif group_config.working_dir and group_config.workspace_isolation == "git-worktree":
                # ``working_dir`` resolution against the scenarios root is
                # validated in ``setup_groups._check_external_working_dir_gate``
                # so any group reaching here either lives inside the scenarios
                # root or carries an explicit ``--allow-external-working-dir``
                # opt-in. The check stays a group property because the field
                # itself is one - per-scenario raises would just duplicate the
                # same failure N times.
                mg = next((m for m in ctx.matched_groups if m.name == group_name), None)
                group_dir = mg.group_dir if mg else Path.cwd()
                base_repo = (group_dir / group_config.working_dir).resolve()

            if base_repo is not None and group_config.workspace_isolation == "git-worktree":
                try:
                    ws_manager = WorkspaceManager(base_repo, ref=workspace_ref)
                except WorkspaceError as e:
                    logger.error("Workspace setup failed for {}/{}: {}", group_name, scenario.name, e)
                    raise

            resource_locks: list[dict] | None = list() if group_config.resources else None
            mg = next((m for m in ctx.matched_groups if m.name == group_name), None)
            scenario_group_dir = mg.group_dir if mg else None

            result = run_scenario_turns(
                agent,
                scenario,
                outcome_dir,
                config,
                workspace=ctx.workspace,
                workspace_manager=ws_manager,
                stream=do_stream,
                group_name=group_name,
                resource_locks=resource_locks,
                group_dir=scenario_group_dir,
            )
        except Exception as e:
            logger.error("Scenario {}/{} failed: {}", group_name, scenario.name, e)
            tb = traceback.format_exc()
            try:
                outcome_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                sentinel = outcome_dir / TURN_CLI_TEMPLATE.format(0)
                if not sentinel.exists():
                    sentinel.write_text(f"RuntimeError: scenario setup failed - {e}\n\n{tb}")
            except Exception:
                pass
            result = ScenarioResult(scenario_name=scenario.name, group_path=group_name, error=str(e))
        ctx.progress.scenario_done(group_name, scenario.name, agent_cost_usd=result.agent_cost_usd)
        return result

    scenario_delay = args.scenario_delay
    interrupted = False
    try:
        if workers <= 1:
            for i, task in enumerate(scenario_tasks):
                if scenario_delay > 0 and i > 0:
                    time.sleep(scenario_delay)
                all_results.append(_run_one(task))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures: dict = {}
                for i, t in enumerate(scenario_tasks):
                    if scenario_delay > 0 and i > 0:
                        time.sleep(scenario_delay)
                    futures[executor.submit(_run_one, t)] = t
                for future in as_completed(futures):
                    try:
                        all_results.append(future.result())
                    except Exception as e:
                        task = futures[future]
                        logger.error("Unexpected failure for {}: {}", task[0].name, e)
                        ctx.progress.console.print(f"  [red]\u2717[/red] {task[4]}/{task[0].name}: {e}")
    except KeyboardInterrupt:
        interrupted = True

    ctx.progress.stop()
    if interrupted:
        done = len(all_results)
        ctx.progress.console.print(f"\n[yellow bold]Interrupted[/yellow bold] - {done}/{ctx.total_scenarios} completed")
    else:
        ctx.progress.summary(all_results)

    return all_results, interrupted
