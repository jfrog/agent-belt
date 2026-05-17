# (c) JFrog Ltd. (2026)

"""Sandbox provider registry -- entry-point loader plus built-in defaults.

Mirrors the agent / scorer / exporter registries: built-ins are wired in this
module; third-party providers register through the
``belt.sandbox_providers`` entry-point group in their own ``pyproject.toml``.
The registry resolves names at runtime so the closed Literal in
:class:`belt.scenario.SandboxProfile` can be rebuilt to include any
plugin-registered provider before scenario parsing runs.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from belt.runner.sandbox.base import BaseSandboxProvider


_ENTRY_POINT_GROUP = "belt.sandbox_providers"


def _load_builtins() -> dict[str, type["BaseSandboxProvider"]]:
    """Return the framework's built-in providers keyed by short name.

    Imported lazily inside the function so the registry module itself stays
    free of side effects on import (matches the pattern in
    ``belt.scorer.registry``).
    """
    from belt.runner.sandbox.docker import DockerSandboxProvider
    from belt.runner.sandbox.host import HostSandboxProvider

    return {
        HostSandboxProvider.name(): HostSandboxProvider,
        DockerSandboxProvider.name(): DockerSandboxProvider,
    }


def _load_entry_points() -> dict[str, type["BaseSandboxProvider"]]:
    """Discover third-party providers registered under ``belt.sandbox_providers``.

    Failures during loading are logged and skipped so a misbehaving plugin
    cannot derail every belt invocation.
    """
    from loguru import logger

    discovered: dict[str, type["BaseSandboxProvider"]] = {}
    try:
        eps = importlib_metadata.entry_points(group=_ENTRY_POINT_GROUP)
    except Exception as e:
        logger.debug("entry_points({}) failed: {}", _ENTRY_POINT_GROUP, e)
        return discovered
    for ep in eps:
        try:
            cls = ep.load()
        except Exception as e:
            logger.warning("sandbox provider entry point '{}' failed to load: {}", ep.name, e)
            continue
        discovered[ep.name] = cls
    return discovered


def available_sandbox_providers() -> list[str]:
    """Return the sorted list of registered provider names (built-in + plugin)."""
    names = set(_load_builtins().keys()) | set(_load_entry_points().keys())
    return sorted(names)


def get_sandbox_provider(name: str) -> type["BaseSandboxProvider"]:
    """Resolve a provider name to its class.

    Raises ``KeyError`` with the available-names list when ``name`` is
    unknown so the user sees an actionable error.
    """
    registry = _load_builtins()
    registry.update(_load_entry_points())
    if name not in registry:
        available = ", ".join(sorted(registry)) or "(none)"
        raise KeyError(f"unknown sandbox provider '{name}'. Available: {available}")
    return registry[name]


__all__ = ["available_sandbox_providers", "get_sandbox_provider"]
