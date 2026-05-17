# (c) JFrog Ltd. (2026)

"""Exporter registry - maps exporter names to classes.

Discovery order mirrors :mod:`belt.agent.registry` and
:mod:`belt.scorer.registry`:

1. Built-in registry (``_EXPORTER_REGISTRY`` below - the canonical list of
   names that ship with core; whatever it lists is what
   :func:`available_exporters` reports).
2. Python entry points under the
   :data:`belt.constants.ENTRY_POINT_GROUP_EXPORTERS` group (vendor
   plugins).
3. Direct dotted import path (escape hatch, default-deny per Design
   Principle 8 - gated behind ``BELT_ALLOW_ARBITRARY_EXPORTER=1`` /
   ``--allow-arbitrary-exporter``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from belt.exporter.base import BaseExporter

_EXPORTER_REGISTRY: dict[str, str] = {
    "csv": "belt.exporter.csv:CsvExporter",
    "jsonl": "belt.exporter.jsonl:JsonlExporter",
    "junit": "belt.exporter.junit:JUnitExporter",
    "markdown": "belt.exporter.markdown:MarkdownExporter",
}


_EP_CACHE: dict[str, str] | None = None


def _discover_entry_points() -> dict[str, str]:
    """Discover exporters registered via Python entry points (cached)."""
    global _EP_CACHE
    if _EP_CACHE is not None:
        return _EP_CACHE
    try:
        from importlib.metadata import entry_points

        from belt.constants import ENTRY_POINT_GROUP_EXPORTERS

        eps = entry_points()
        if hasattr(eps, "select"):
            found = eps.select(group=ENTRY_POINT_GROUP_EXPORTERS)
        else:
            found = eps.get(ENTRY_POINT_GROUP_EXPORTERS, [])
        _EP_CACHE = {ep.name: ep.value for ep in found}
    except Exception:
        _EP_CACHE = {}
    return _EP_CACHE


def get_exporter_class(name: str) -> type[BaseExporter]:
    """Resolve an exporter name to its class.

    Same tiered discovery as :func:`belt.agent.registry.get_agent_class`
    (built-in registry, then entry points, then dotted-import escape hatch).
    The escape hatch is gated behind
    :data:`belt.envvars.ALLOW_ARBITRARY_EXPORTER` so a hostile
    ``--export-config`` cannot import arbitrary modules at process start.
    """
    from belt import envvars

    dotted = _EXPORTER_REGISTRY.get(name)

    if dotted is None:
        ep_registry = _discover_entry_points()
        dotted = ep_registry.get(name)

    if dotted is None:
        if not envvars.is_truthy(envvars.ALLOW_ARBITRARY_EXPORTER):
            all_names = sorted(set(_EXPORTER_REGISTRY) | set(_discover_entry_points()))
            from belt.errors import ConfigError

            raise ConfigError(
                f"Unknown exporter '{name}'. Available: {', '.join(all_names)}. "
                f"To use a dotted import path, pass --allow-arbitrary-exporter "
                f"(or set {envvars.ALLOW_ARBITRARY_EXPORTER}=1)."
            )
        dotted = name

    if "." not in dotted and ":" not in dotted:
        all_names = sorted(set(_EXPORTER_REGISTRY) | set(_discover_entry_points()))
        from belt.errors import ConfigError

        raise ConfigError(
            f"Unknown exporter '{name}'. Available: {', '.join(all_names)}. "
            f"For third-party exporters, use a dotted import path "
            f"(e.g., 'mypackage:MyExporter')."
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

        raise ConfigError(f"Failed to load exporter '{name}' ({dotted}): {e}") from e

    from belt.exporter.base import BaseExporter

    if not (isinstance(cls, type) and issubclass(cls, BaseExporter)):
        from belt.errors import ConfigError

        raise ConfigError(f"'{dotted}' is not a BaseExporter subclass")

    return cls


def available_exporters() -> list[str]:
    """Return all registered exporter names (built-in + entry points)."""
    return sorted(set(_EXPORTER_REGISTRY) | set(_discover_entry_points()))
