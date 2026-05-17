# (c) JFrog Ltd. (2026)

"""Outcome-dir → scenario-file mapping and scoring-strategy resolution.

The scorer never writes scenarios; it only reads them.  Given an outcome
directory (``<run>/<group>/<scenario>``), this module finds the matching
``<scenarios_root>/<group>/<scenario>.json`` source file and resolves the
``ScoringStrategy`` (LLM-judge dimensions + agent-context preamble) for
that group.

Trial suffixes (``__trial_N``) are stripped via ``constants.TRIAL_SUFFIX_RE``
- the same regex the runner used to write the directory name and the
aggregator uses to compute pass^k.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from belt import _internal_envvars
from belt.constants import SCENARIOS_DIR, TRIAL_SUFFIX_RE

if TYPE_CHECKING:
    from belt.agent.scoring import ScoringStrategy


def scenarios_root(outcomes_root: Path | None = None) -> Path:
    """Return the scenarios root.

    Resolution order:
    1. ``_BELT_SCENARIOS_ROOT`` env var (set by runner during ``eval``).
    2. ``run_meta.json`` inside the outcomes run directory (persisted by runner).
    3. Fallback to ``SCENARIOS_DIR`` constant.
    """
    override = os.environ.get(_internal_envvars.SCENARIOS_ROOT)
    if override:
        return Path(override)

    if outcomes_root is not None:
        meta = outcomes_root / "run_meta.json"
        if meta.exists():
            try:
                data = json.loads(meta.read_text())
                return Path(data["scenarios_root"])
            except Exception:
                pass

    return SCENARIOS_DIR


def map_to_scenario(outcome_dir: Path, outcomes_root: Path) -> Path:
    """Map an outcome directory to its source scenario JSON file."""
    relative = outcome_dir.relative_to(outcomes_root)
    parts = list(relative.parts)
    scenario_name = TRIAL_SUFFIX_RE.sub("", parts[-1])
    return scenarios_root(outcomes_root) / Path(*parts[:-1]) / f"{scenario_name}.json"


def parse_dimension_defs(dims: list) -> list:
    """Parse dimension definitions from YAML - supports string shorthand and full dict.

    Dict entries route through :meth:`DimensionDef.from_config_dict` so
    the JSON key ``pass`` (Python keyword) and unknown keys are
    handled with a clear error rather than ``TypeError`` from the
    dataclass constructor.
    """
    from belt.agent.scoring import DimensionDef

    parsed: list[DimensionDef] = []
    for d in dims:
        if isinstance(d, str):
            parsed.append(DimensionDef(name=d, description=d.replace("_", " ").title()))
        elif isinstance(d, dict):
            parsed.append(DimensionDef.from_config_dict(d))
        else:
            parsed.append(DimensionDef(name=str(d), description=str(d)))
    return parsed


def _instantiate_agent(agent_cls: type) -> object | None:
    """Instantiate an agent class without running its setup hooks.

    The resolver only needs to call ``scoring_strategy()`` on an
    instance. Some concrete agents perform heavyweight work in
    ``__init__`` (subprocess probes, env validation), so an unparameterised
    ``agent_cls()`` call is unreliable as a probe. Falling back to
    ``__new__`` produces an uninitialised but usable object whenever the
    constructor refuses to run, while still letting overridden
    ``scoring_strategy`` definitions resolve via the normal MRO.
    """
    try:
        return agent_cls()
    except Exception:
        try:
            return agent_cls.__new__(agent_cls)
        except Exception:
            return None


def _agent_default_strategy(agent_name: str) -> "ScoringStrategy | None":
    """Return the agent's declared scoring strategy, or ``None`` on failure.

    Always returns the result of ``instance.scoring_strategy()`` for the
    resolved agent class - including the framework default when the agent
    inherits ``BaseAgentAdapter.scoring_strategy`` unchanged. Used by the
    ``llm_dimensions_extend_defaults`` path, where the caller explicitly
    wants the agent's full strategy as the base list to merge against.
    """
    try:
        agent_cls = __import__("belt.agent.registry", fromlist=["get_agent_class"]).get_agent_class(agent_name)
        instance = _instantiate_agent(agent_cls)
        if instance is None:
            return None
        return instance.scoring_strategy()
    except Exception:
        return None


def _agent_override_strategy(agent_name: str) -> "ScoringStrategy | None":
    """Return the agent's scoring strategy only when the class overrides it.

    The contract is intentionally narrower than ``_agent_default_strategy``:
    when the agent class inherits ``BaseAgentAdapter.scoring_strategy``
    unchanged, this function returns ``None`` so the caller leaves any
    judge-level strategy (for example, dimensions configured via
    ``--scorer-config``) intact. The framework default is then supplied by
    ``LLMScorer`` itself, which substitutes ``default_scoring_strategy()``
    when no explicit strategy was passed.

    Identity comparison against ``BaseAgentAdapter.scoring_strategy`` is
    sufficient: a subclass that rebinds the method has declared an
    intentional scoring strategy and is honoured here, while a subclass
    that inherits the method as-is is treated as opting out of agent-level
    customisation.
    """
    try:
        agent_cls = __import__("belt.agent.registry", fromlist=["get_agent_class"]).get_agent_class(agent_name)
        from belt.agent.base import BaseAgentAdapter

        if agent_cls.scoring_strategy is BaseAgentAdapter.scoring_strategy:
            return None
        instance = _instantiate_agent(agent_cls)
        if instance is None:
            return None
        return instance.scoring_strategy()
    except Exception:
        return None


def resolve_scoring_strategy(outcome_dir: Path, outcomes_root: Path) -> "ScoringStrategy | None":
    """Resolve the per-outcome scoring strategy, or ``None`` to keep the judge's own.

    Precedence (highest first):

    1. The group ``_config.json`` ``llm_dimensions`` field. Resolved
       under the group directory whether the run targets a parent path
       (``<root>/<group>/<scenario>/``) or the group directly
       (``<root>/<scenario>/``, where the group dir IS the scenarios root).
    2. An agent class that overrides ``BaseAgentAdapter.scoring_strategy``.
    3. The judge's build-time strategy (typically populated from
       ``--scorer-config``); preserved by returning ``None`` from this
       function so the caller does not rebuild the judge.
    4. The framework default, supplied by ``LLMScorer`` when no explicit
       strategy was set.

    When ``llm_dimensions_extend_defaults`` is true on the group config,
    its dimensions are merged onto the agent's full default strategy
    rather than replacing it, so groups can extend whatever the agent
    declared without restating it.
    """
    from belt.agent.scoring import ScoringStrategy

    relative = outcome_dir.relative_to(outcomes_root)
    group_parts = relative.parts[:-1]
    # When the user runs ``belt eval <single-group-path>`` directly,
    # ``scenarios_root`` is set to that group directory and outcomes
    # land at ``<run>/<scenario>/`` with no group sub-directory. The
    # group's ``_config.json`` then sits at the scenarios-root itself,
    # not under ``Path(*group_parts)``.
    sroot = scenarios_root(outcomes_root)
    group_dir = sroot / Path(*group_parts) if group_parts else sroot
    group_config_path = group_dir / "_config.json"
    if not group_config_path.exists():
        return None
    try:
        from belt.scenario import GroupConfig

        gc = GroupConfig.model_validate_json(group_config_path.read_text())
        agent_name = gc.agent
    except Exception:
        return None

    if gc.llm_dimensions:
        parsed = parse_dimension_defs(gc.llm_dimensions)
        if gc.llm_dimensions_extend_defaults:
            base = _agent_default_strategy(agent_name)
            base_dims = base.dimensions if base else []
            existing_names = {d.name for d in parsed}
            merged = [d for d in base_dims if d.name not in existing_names] + parsed
            return ScoringStrategy(dimensions=merged, agent_context=base.agent_context if base else "")
        return ScoringStrategy(dimensions=parsed)

    return _agent_override_strategy(agent_name)
