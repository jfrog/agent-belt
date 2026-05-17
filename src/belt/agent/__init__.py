# (c) JFrog Ltd. (2026)

"""Agent protocol and implementations for evaluating CLI agents."""

from belt.agent.base import (
    AGENT_SPECIFIC_FIELDS,
    UNIVERSAL_OUTPUT_FIELDS,
    AgentArgError,
    AgentNotAvailableError,
    AgentOption,
    BaseAgentAdapter,
)
from belt.agent.registry import get_agent_class
from belt.agent.scoring import ScoringStrategy

__all__ = [
    "AGENT_SPECIFIC_FIELDS",
    "AgentArgError",
    "AgentNotAvailableError",
    "AgentOption",
    "BaseAgentAdapter",
    "ScoringStrategy",
    "UNIVERSAL_OUTPUT_FIELDS",
    "get_agent_class",
]
