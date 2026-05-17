# (c) JFrog Ltd. (2026)

"""Run-phase shared context and agent construction.

Library module - used by ``commands/run.py`` and ``runner/phases/*``.

Holds:

- ``MatchedGroup`` / ``RunContext`` dataclasses that flow through every phase.
- Outcome-root and agent-args resolution helpers (precedence: CLI > env > default).
- ``create_agent`` - agent instantiation with fail-fast validation of
  ``-X key=value`` flags. Used by setup, scenario dispatch, teardown, and
  orphan cleanup; centralised here so all four sites apply identical
  validation against the agent's declared options.
"""

from __future__ import annotations

import argparse
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from belt.agent.base import AgentArgError, BaseAgentAdapter
from belt.cli_utils import parse_kv_args
from belt.constants import OUTCOMES_DIR_ENV, REPO_ROOT
from belt.progress import RunnerProgress
from belt.scenario import GroupConfig, Scenario


@dataclass
class MatchedGroup:
    """A scenario group that passed CLI filters and is ready to run."""

    group_dir: Path
    config: GroupConfig
    scenarios: list[Scenario]
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = str(self.group_dir)


@dataclass
class RunContext:
    """Shared state across run phases - parsed once, used everywhere."""

    args: argparse.Namespace
    scenarios_root: Path
    matched_groups: list[MatchedGroup]
    agent_args: dict[str, str]
    outcomes_root: Path
    run_dir: Path
    workspace: Path
    progress: RunnerProgress
    n_trials: int = 1
    total_scenarios: int = 0
    total_turns: int = 0
    # Count of scenario JSON files that failed to parse during
    # ``parse_and_filter``. Persisted to ``run_meta.json`` so the
    # aggregate phase can surface it on ``AggregatedResults`` even though
    # the failed files never produced a ``score.json``.
    scenarios_skipped: int = 0

    group_states: dict[str, Any] = field(default_factory=dict)
    failed_groups: set[str] = field(default_factory=set)
    # Per-group setup-failure cause, captured at the moment a group is
    # marked failed. Lives next to ``failed_groups`` so the wording the
    # user saw in the setup banner is preserved verbatim for the
    # ``setup_errors`` sidecar and ``AggregatedResults.setup_errors``.
    # Keys mirror ``failed_groups``; missing keys mean "rejection
    # message was not recorded" (a defensive fallback used by the
    # sidecar writer, never a normal path).
    setup_errors: dict[str, str] = field(default_factory=dict)
    group_config_by_name: dict[str, GroupConfig] = field(default_factory=dict)
    # Maps group name -> cloned fixture directory. Populated by
    # ``setup_groups`` for groups that declare ``GroupConfig.fixture_repo``.
    # Read in ``run_scenarios`` to point ``WorkspaceManager`` at the cached
    # clone instead of the in-tree ``working_dir`` placeholder.
    group_fixtures: dict[str, Path] = field(default_factory=dict)


def resolve_outcomes_root(cli_flag: str | None = None) -> Path:
    """Resolve outcomes root: --outcomes-dir > $BELT_OUTCOMES_DIR > cwd/outcomes."""
    if cli_flag:
        return Path(cli_flag).resolve()
    env_val = os.environ.get(OUTCOMES_DIR_ENV)
    if env_val:
        return Path(env_val)
    return Path.cwd() / "outcomes"


def _resolve_env_var_defaults(
    agent_cls: type[BaseAgentAdapter],
    agent_args: dict[str, str],
) -> dict[str, str]:
    """Inject env-var fallbacks for agent options not already set by -X flags.

    Precedence: -X flag > env var > agent default.
    """
    resolved = dict(agent_args)
    for opt in agent_cls.cli_options():
        if opt.env_var and opt.name not in resolved:
            val = os.environ.get(opt.env_var)
            if val:
                resolved[opt.name] = val
    return resolved


_FRAMEWORK_KEYS: frozenset[str] = frozenset({"repo_root"})


def create_agent(
    agent_cls: type[BaseAgentAdapter],
    agent_args: dict[str, str],
) -> BaseAgentAdapter:
    """Create an agent with fail-fast validation of -X args.

    Framework-injected keys (``_FRAMEWORK_KEYS``) are stripped before invoking
    the agent constructor - agents declare what they actually use, and the
    framework does not smuggle bookkeeping fields through their __init__.

    Empty ``cli_options()`` is treated as the explicit "this agent accepts
    no -X options" contract. Parameterless agents are a deliberate design
    point (no per-call configuration; pin behaviour via scenario flags or
    env vars). Any -X arg passed to such an agent is rejected here with a
    message that points at the blessed paths (the scenario's ``flags``
    field, or env vars listed in ``required_env_vars()``) instead of
    leaking the rewrapped Python ``TypeError`` to the user.
    """
    agent_args = _resolve_env_var_defaults(agent_cls, agent_args)
    known_options = agent_cls.cli_options()
    known_names = {opt.name for opt in known_options}
    unknown = sorted(set(agent_args) - known_names - _FRAMEWORK_KEYS)
    if unknown:
        if known_names:
            accepted = ", ".join(sorted(known_names))
            raise AgentArgError(
                f"Agent '{agent_cls.__name__}' does not accept: {', '.join(unknown)}.\n"
                f"  Accepted options: {accepted}"
            )
        raise AgentArgError(
            f"Agent '{agent_cls.__name__}' does not accept any -X options "
            f"(got: {', '.join(unknown)}).\n"
            f"  This agent is parameterless by design. Pass per-scenario "
            f'flags via the scenario\'s "flags" field (e.g. "flags": '
            f'["--model", "X"]), or set agent-specific env vars listed '
            f"in `required_env_vars()`."
        )
    constructor_kwargs = {k: v for k, v in agent_args.items() if k not in _FRAMEWORK_KEYS}
    try:
        instance = agent_cls(**constructor_kwargs)
    except TypeError as e:
        # Belt-and-braces: the validation above should now catch every
        # unknown-kwarg case for built-in agents. This branch survives only
        # to surface bugs in third-party agents whose ``cli_options()``
        # disagrees with their ``__init__`` signature.
        raise AgentArgError(f"Agent '{agent_cls.__name__}': {e}") from e
    # Stash a copy of the resolved (post-env-var-fallback, framework-key-stripped)
    # agent args on the instance so the orchestrator can record them in the
    # per-scenario ``_runtime_info.json`` sidecar without an extra parameter
    # on ``run_scenario_turns``. Read by ``orchestrator._write_runtime_info_sidecar``.
    instance._captured_agent_args = dict(constructor_kwargs)  # type: ignore[attr-defined]
    return instance


def _resolve_agent_args(args: argparse.Namespace) -> dict[str, str]:
    """Parse -X key=value args and inject framework defaults."""
    agent_args = parse_kv_args(args.agent_args)
    agent_args.setdefault("repo_root", str(REPO_ROOT))
    return agent_args


def build_run_context(
    args: argparse.Namespace,
    scenarios_root: Path,
    matched_groups: list[MatchedGroup],
) -> RunContext:
    """Build the RunContext with all derived values needed by subsequent phases."""
    agent_args = _resolve_agent_args(args)
    n_trials = args.trials
    total_scenarios = sum(len(mg.scenarios) for mg in matched_groups) * n_trials
    total_turns = sum(len(s.turns) for mg in matched_groups for s in mg.scenarios)
    outcomes_root = resolve_outcomes_root(getattr(args, "outcomes_dir", None))
    run_dir = outcomes_root / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"
    workspace = Path.cwd()

    is_live = args.progress == "live"
    progress: RunnerProgress
    if is_live:
        # 0o700 keeps scenario stdout / NDJSON / CLI logs owner-only on shared
        # hosts. The umask in ``cli.main()`` handles files; this mode is
        # required because ``mkdir(parents=True)`` does not apply the same
        # mode to intermediate parents.
        run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        from belt.progress import LiveProgress

        progress = LiveProgress(run_dir=run_dir, max_lines=args.progress_live_lines)
    else:
        progress = RunnerProgress(plain=args.progress == "plain")

    ctx = RunContext(
        args=args,
        scenarios_root=scenarios_root,
        matched_groups=matched_groups,
        agent_args=agent_args,
        outcomes_root=outcomes_root,
        run_dir=run_dir,
        workspace=workspace,
        progress=progress,
        n_trials=n_trials,
        total_scenarios=total_scenarios,
        total_turns=total_turns,
    )
    for mg in matched_groups:
        ctx.group_config_by_name[mg.name] = mg.config
    return ctx
