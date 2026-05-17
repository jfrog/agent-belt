# (c) JFrog Ltd. (2026)

"""Agent registry - maps agent names to classes.

Discovery order:
1. Built-in registry (public agents only - no vendor-specific entries)
2. Python entry points under ``belt.agents`` group
3. Direct dotted import path (escape hatch: ``--agent mypackage.MyAgentAdapter``)

Vendor-specific agents register via entry points in
their own ``pyproject.toml`` - they are NOT hardcoded here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from belt.agent.base import BaseAgentAdapter

_AGENT_REGISTRY: dict[str, str] = {
    "claude-code": "belt.agent.claude_code.ClaudeCodeAgentAdapter",
    "cursor": "belt.agent.cursor.CursorAgentAdapter",
    "codex": "belt.agent.codex.CodexAgentAdapter",
    "copilot": "belt.agent.copilot.CopilotAgentAdapter",
    "gemini": "belt.agent.gemini.GeminiAgentAdapter",
    "opencode": "belt.agent.opencode.OpenCodeAgentAdapter",
    "goose": "belt.agent.goose.GooseAgentAdapter",
}


_EP_CACHE: dict[str, str] | None = None


def _discover_entry_points() -> dict[str, str]:
    """Discover agents registered via Python entry points (cached after first call)."""
    global _EP_CACHE
    if _EP_CACHE is not None:
        return _EP_CACHE
    try:
        from importlib.metadata import entry_points

        from belt.constants import ENTRY_POINT_GROUP_AGENTS

        eps = entry_points()
        if hasattr(eps, "select"):
            found = eps.select(group=ENTRY_POINT_GROUP_AGENTS)
        else:
            found = eps.get(ENTRY_POINT_GROUP_AGENTS, [])
        _EP_CACHE = {ep.name: ep.value for ep in found}
    except Exception:
        _EP_CACHE = {}
    return _EP_CACHE


def get_agent_class(name: str) -> type[BaseAgentAdapter]:
    """Resolve an agent name to its class.

    Checks built-in registry first, then entry points, then - only when
    explicitly opted in - falls back to direct dotted import. The escape
    hatch lets users register a third-party agent without packaging an
    entry point, but it also lets a hostile config import arbitrary
    modules at process start. Gating it behind
    ``envvars.ALLOW_ARBITRARY_AGENT=1`` (or the runner's
    ``--allow-arbitrary-agent`` flag) closes the gap by default while
    keeping the ergonomic pathway for power users.
    """
    from belt import envvars

    dotted = _AGENT_REGISTRY.get(name)

    if dotted is None:
        ep_registry = _discover_entry_points()
        dotted = ep_registry.get(name)

    if dotted is None:
        if not envvars.is_truthy(envvars.ALLOW_ARBITRARY_AGENT):
            all_names = sorted(set(_AGENT_REGISTRY) | set(_discover_entry_points()))
            from belt.errors import ConfigError

            raise ConfigError(
                f"Unknown agent '{name}'. Available: {', '.join(all_names)}. "
                f"To use a dotted import path, pass --allow-arbitrary-agent "
                f"(or set {envvars.ALLOW_ARBITRARY_AGENT}=1)."
            )
        dotted = name

    if "." not in dotted and ":" not in dotted:
        all_names = sorted(set(_AGENT_REGISTRY) | set(_discover_entry_points()))
        from belt.errors import ConfigError

        raise ConfigError(
            f"Unknown agent '{name}'. Available: {', '.join(all_names)}. "
            f"For third-party agents, use a dotted import path (e.g., 'mypackage.MyAgentAdapter')."
        )

    if ":" in dotted:
        module_path, class_name = dotted.split(":", 1)
    else:
        module_path, class_name = dotted.rsplit(".", 1)
    try:
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        from belt.errors import ConfigError

        raise ConfigError(f"Failed to load agent '{name}' ({dotted}): {e}") from e

    from belt.agent.base import BaseAgentAdapter

    if not (isinstance(cls, type) and issubclass(cls, BaseAgentAdapter)):
        from belt.errors import ConfigError

        raise ConfigError(f"'{dotted}' is not a BaseAgentAdapter subclass")

    return cls


def available_agents() -> list[str]:
    """Return all registered agent names (built-in + entry points)."""
    return sorted(set(_AGENT_REGISTRY) | set(_discover_entry_points()))
