# (c) JFrog Ltd. (2026)

"""Runner-only entities - agent config and scenario run results.

Cross-phase entities (ToolCall, TurnTiming, TurnOutput) live in
``belt.entities`` because they are consumed by scorer and aggregator.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from belt.scenario import GroupConfig


class AgentConfig(BaseModel):
    """Per-scenario configuration passed to agent.setup()."""

    group_config: GroupConfig
    scenario_name: str
    shared_state: Any = None
    scenario_options: dict[str, Any] = Field(default_factory=dict)
    workspace_dir: Optional[str] = Field(
        default=None,
        description="Isolated workspace directory path, set by the orchestrator when workspace isolation is active.",
    )


class ScenarioResult(BaseModel):
    """Result of running a single scenario.

    ``error`` records *infrastructure* failures (subprocess crash,
    workspace setup error, exception in the orchestrator). It is the
    "the harness fell over" signal.

    ``agent_errors`` records *agent runtime* failures - one entry per
    turn whose ``TurnOutput.has_error`` was true. Each entry is the
    ``error_type`` token for that turn (see
    :mod:`belt.agent.error_types`). It is the "the agent didn't
    really run" signal, distinct from ``error`` and from rule failures.
    """

    scenario_name: str
    group_path: str
    turns_completed: int = 0
    agent_cost_usd: Optional[float] = None
    error: Optional[str] = None
    agent_errors: list[str] = Field(default_factory=list)
    outcome_dir: Optional[str] = None
    agent_metadata: Optional[dict[str, Any]] = None
