# (c) JFrog Ltd. (2026)

"""Scorer registry - maps scorer names to classes.

Discovery order:
1. Built-in registry (rules, llm)
2. Python entry points under ``belt.scorers`` group
3. Direct dotted import path (escape hatch)

Third-party scorers register via entry points in their own
``pyproject.toml`` - they are NOT hardcoded here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from belt.scorer.base import BaseScorer

_SCORER_REGISTRY: dict[str, str] = {
    "rules": "belt.scorer.rules:RuleBasedScorer",
    "llm": "belt.scorer.llm:LLMScorer",
}

_EP_CACHE: dict[str, str] | None = None


def _discover_entry_points() -> dict[str, str]:
    """Discover scorers registered via Python entry points (cached)."""
    global _EP_CACHE
    if _EP_CACHE is not None:
        return _EP_CACHE
    try:
        from importlib.metadata import entry_points

        from belt.constants import ENTRY_POINT_GROUP_SCORERS

        eps = entry_points()
        if hasattr(eps, "select"):
            found = eps.select(group=ENTRY_POINT_GROUP_SCORERS)
        else:
            found = eps.get(ENTRY_POINT_GROUP_SCORERS, [])
        _EP_CACHE = {ep.name: ep.value for ep in found}
    except Exception:
        _EP_CACHE = {}
    return _EP_CACHE


def get_scorer_class(name: str) -> type[BaseScorer]:
    """Resolve a scorer name to its class.

    Checks built-in registry first, then entry points, then - only when
    explicitly opted in via ``envvars.ALLOW_ARBITRARY_SCORER=1`` - falls
    back to direct dotted import. The escape hatch is gated for the same
    reason as :func:`belt.agent.registry.get_agent_class`: to
    prevent a hostile config from importing arbitrary modules at process
    start.
    """
    from belt import envvars

    dotted = _SCORER_REGISTRY.get(name)

    if dotted is None:
        ep_registry = _discover_entry_points()
        dotted = ep_registry.get(name)

    if dotted is None:
        if not envvars.is_truthy(envvars.ALLOW_ARBITRARY_SCORER):
            all_names = sorted(set(_SCORER_REGISTRY) | set(_discover_entry_points()))
            from belt.errors import ConfigError

            raise ConfigError(
                f"Unknown scorer '{name}'. Available: {', '.join(all_names)}. "
                f"To use a dotted import path, pass --allow-arbitrary-scorer "
                f"(or set {envvars.ALLOW_ARBITRARY_SCORER}=1)."
            )
        dotted = name

    if "." not in dotted and ":" not in dotted:
        all_names = sorted(set(_SCORER_REGISTRY) | set(_discover_entry_points()))
        from belt.errors import ConfigError

        raise ConfigError(
            f"Unknown scorer '{name}'. Available: {', '.join(all_names)}. "
            f"For third-party scorers, use a dotted import path (e.g., 'mypackage:MyScorer')."
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

        raise ConfigError(f"Failed to load scorer '{name}' ({dotted}): {e}") from e

    from belt.scorer.base import BaseScorer

    if not (isinstance(cls, type) and issubclass(cls, BaseScorer)):
        from belt.errors import ConfigError

        raise ConfigError(f"'{dotted}' is not a BaseScorer subclass")

    return cls


def available_scorers() -> list[str]:
    """Return all registered scorer names (built-in + entry points)."""
    return sorted(set(_SCORER_REGISTRY) | set(_discover_entry_points()))
