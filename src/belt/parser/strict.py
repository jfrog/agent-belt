# (c) JFrog Ltd. (2026)

"""Schema-driven validation for ``--strict-config``.

The default scenario loader is permissive on two surfaces:

- ``TurnExpectation`` uses ``extra="allow"`` so plugins can attach
  bespoke expectation keys (e.g. ``max_handoffs``, ``review_prompted``).
- ``GroupConfig`` uses Pydantic's default ``extra="ignore"`` so plugins
  can declare per-group config (e.g. ``mcp_servers``) without
  coordinating with core.

Permissive parsing is a feature: scenarios authored for a plugin still
load against agents that lack it. But it is also a typo trap. ``"tools_invoke": [...]``
silently produces zero coverage; ``"agnet": "claude"`` becomes "key
``agent`` is missing, falling back to default" at runtime - both
appear green in CI.

This module provides a strict validator that, when enabled via
``--strict-config``, recursively compares each loaded JSON document
against the union of:

- the Pydantic model's declared fields, and
- a process-local registry of plugin extension keys
  (``register_plugin_scenario_key``).

Unknown keys are reported with a difflib-driven "did you mean" hint
and a fully-qualified JSON path. The check is opt-in (default OFF)
so existing scenarios continue to load; CI dashboards opt in for
fail-fast.

The validator is read-only and process-local: it never reads from the
filesystem, the network, or environment variables, and the registry
is never persisted across processes (plugins re-register on each
import). Adding new known keys is monotonic - older plugins keep
working while a new key joins the allow-list.
"""

from __future__ import annotations

import difflib
import re
import types
import typing
from typing import Iterable, Optional

from pydantic import BaseModel

# ── Plugin extension key registry ──
#
# Plugins register their per-scenario / per-group extension keys here
# at import time. The registry is a plain module-level dict: no
# persistence, no IPC, no env-var driven registration. A plugin not
# imported in this process is invisible to the validator, which is
# the correct fail-closed behaviour - if you forgot to install the
# plugin, ``--strict-config`` should reject scenarios that reference
# its keys.
_PLUGIN_KEYS: dict[type[BaseModel], set[str]] = {}

# Reserved-key safety net. Plugins MUST NOT shadow these names; they
# are the framework-controlled surface and a plugin that registers
# them would bypass core validation. Enforced in
# ``register_plugin_scenario_key``.
_RESERVED_KEYS: frozenset[str] = frozenset({"name", "description", "tags", "turns", "agent", "schema_version"})

# Plugin-key shape. Lower-case ASCII identifier, optional dot or
# hyphen separators, max 64 chars. Mirrors the shape of declared
# Pydantic field names so the registry surface looks indistinguishable
# from core for downstream tooling.
_PLUGIN_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:[.\-][a-z0-9_]+)*$")
_PLUGIN_KEY_MAX_LEN = 64


def register_plugin_scenario_key(model: type[BaseModel], key: str) -> None:
    """Register a plugin extension key for ``--strict-config``.

    Plugins call this at import time once per extension key they
    expect to read from a scenario or group config (e.g. a
    multi-agent framework registering ``"max_handoffs"`` on
    :class:`belt.scenario.TurnExpectation`).

    Without registration, ``--strict-config`` rejects the key as a
    typo. With registration, the key passes regardless of whether
    the underlying Pydantic model uses ``extra="allow"`` or
    ``extra="ignore"`` - the strict validator runs before Pydantic.

    Idempotent: registering the same ``(model, key)`` twice is a
    no-op. Registration is process-local; never persisted.

    Raises ``ValueError`` if ``key`` shadows a framework-controlled
    name (see ``_RESERVED_KEYS``) or has the wrong shape, and
    ``TypeError`` if ``model`` is not a Pydantic ``BaseModel``
    subclass.
    """
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise TypeError(f"register_plugin_scenario_key: model must be a pydantic BaseModel subclass, got {model!r}")
    if not isinstance(key, str):
        raise TypeError(f"register_plugin_scenario_key: key must be str, got {type(key).__name__}")
    if key in _RESERVED_KEYS:
        raise ValueError(
            f"register_plugin_scenario_key: {key!r} is a framework-reserved key and cannot be re-registered. "
            f"Pick a different name (e.g. plugin-prefixed like 'myplugin_{key}')."
        )
    if len(key) > _PLUGIN_KEY_MAX_LEN or not _PLUGIN_KEY_PATTERN.match(key):
        raise ValueError(
            f"register_plugin_scenario_key: {key!r} has an unsupported shape. "
            f"Use a lowercase identifier (letters, digits, underscores; optional '.' or '-' separators), "
            f"max {_PLUGIN_KEY_MAX_LEN} chars."
        )
    _PLUGIN_KEYS.setdefault(model, set()).add(key)


def registered_plugin_scenario_keys(model: type[BaseModel]) -> frozenset[str]:
    """Read-only view of registered keys for a model.

    Returns ``frozenset()`` if no keys have been registered. Public so
    plugins can introspect their own registrations and so the test
    suite can pin the registry against unexpected drift.
    """
    return frozenset(_PLUGIN_KEYS.get(model, set()))


# ── Validator ──


class StrictConfigError(ValueError):
    """Raised when ``--strict-config`` finds unknown keys.

    The exception carries a list of ``errors`` (one per unknown key)
    so the caller can format them as it sees fit. The string form is
    the joined newline-separated error list, ready to print.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


def _resolve_submodel(annotation: typing.Any) -> Optional[type[BaseModel]]:
    """Extract a ``BaseModel`` subclass from a field annotation.

    Handles ``X``, ``Optional[X]``, ``list[X]``, ``dict[str, X]``,
    and unions of nested forms. Returns ``None`` when no embedded
    BaseModel subclass is found - leaf scalars and free-form
    ``dict[str, Any]`` fields recurse no further.
    """
    if annotation is None:
        return None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = typing.get_origin(annotation)
    if origin is None:
        return None
    args = typing.get_args(annotation)
    # ``Optional[X]``, ``X | Y``: try each arg in order. ``Union[X, Y]``
    # with two BaseModel subclasses is unusual in this code base - the
    # first one wins, callers that need polymorphic dispatch should
    # carry a discriminator field and not lean on the strict validator.
    if origin in (typing.Union, types.UnionType):
        for arg in args:
            sub = _resolve_submodel(arg)
            if sub is not None:
                return sub
        return None
    # ``list[X]``: recurse into the element type.
    if origin in (list, tuple, set, frozenset):
        for arg in args:
            sub = _resolve_submodel(arg)
            if sub is not None:
                return sub
        return None
    # ``dict[str, X]``: recurse into the value type.
    if origin in (dict,):
        if len(args) >= 2:
            return _resolve_submodel(args[1])
        return None
    return None


def _allowed_keys(model: type[BaseModel]) -> frozenset[str]:
    """Union of declared Pydantic fields and registered plugin keys for ``model``."""
    return frozenset(model.model_fields.keys()) | registered_plugin_scenario_keys(model)


def _did_you_mean(unknown: str, allowed: Iterable[str]) -> Optional[str]:
    """Return the closest valid key name, or None if no good match."""
    matches = difflib.get_close_matches(unknown, list(allowed), n=1, cutoff=0.6)
    return matches[0] if matches else None


def _format_path(path: tuple[str, ...], key: str) -> str:
    """Render a dotted JSON path for an offending key (e.g. ``turns[0].expect.tools_invoke``)."""
    if not path:
        return key
    return f"{'.'.join(path)}.{key}"


def validate_strict(
    raw: typing.Any,
    model: type[BaseModel],
    *,
    source: str,
    _path: tuple[str, ...] = (),
) -> list[str]:
    """Recursively validate ``raw`` against ``model``'s schema.

    Returns a list of human-readable error messages, empty when the
    document is well-formed under strict mode. Each message is
    prefixed with ``source`` (typically the file path) and qualified
    with a dotted JSON path so authors can find the offending key
    without re-reading the whole file.

    ``raw`` is expected to be the parsed JSON (``dict`` / ``list`` /
    primitives), not a Pydantic instance: the validator runs *before*
    Pydantic so it sees keys that ``extra="ignore"`` would silently
    drop.

    Non-dict roots (e.g. a top-level array) are not rejected here -
    Pydantic's own validation will catch them with a clearer message.
    """
    errors: list[str] = []
    if not isinstance(raw, dict):
        return errors

    allowed = _allowed_keys(model)
    for key in raw:
        if key in allowed:
            continue
        # ``__pydantic_extra__`` and similar dunder keys are never in
        # legitimate scenario JSON; surface them as ordinary unknowns
        # rather than treating them as some kind of "advanced" escape
        # hatch.
        suggestion = _did_you_mean(key, allowed)
        hint = f" Did you mean {suggestion!r}?" if suggestion else ""
        errors.append(f"{source}: unknown key {_format_path(_path, key)!r}.{hint}")

    for fname, finfo in model.model_fields.items():
        if fname not in raw:
            continue
        sub_cls = _resolve_submodel(finfo.annotation)
        if sub_cls is None:
            continue
        value = raw[fname]
        if isinstance(value, dict):
            errors.extend(validate_strict(value, sub_cls, source=source, _path=_path + (fname,)))
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, dict):
                    errors.extend(validate_strict(item, sub_cls, source=source, _path=_path + (f"{fname}[{idx}]",)))
    return errors


__all__ = [
    "StrictConfigError",
    "register_plugin_scenario_key",
    "registered_plugin_scenario_keys",
    "validate_strict",
]
