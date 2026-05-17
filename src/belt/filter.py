# (c) JFrog Ltd. (2026)

"""Scenario filtering - shared logic for selecting scenarios by path, name, and tag.

Used by commands/run.py today; available for score, aggregate, and watch to adopt
as filtering needs grow. Single source of truth for all filter semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from belt.errors import ScenarioError
from belt.scenario import GroupConfig, Scenario


@dataclass(frozen=True)
class ScenarioFilter:
    """Immutable filter specification for selecting scenarios.

    Constructed from CLI args via ``from_cli_args``, then applied via
    ``matches_group`` (coarse, per-group) and ``matches_scenario`` (fine,
    per-scenario).  Both must pass for a scenario to be selected.
    """

    scenarios_root: Path
    tags: frozenset[str] = field(default_factory=frozenset)
    parsed_paths: tuple[tuple[Path, str | None], ...] | None = None

    # ── Construction ──

    @classmethod
    def from_cli_args(
        cls,
        scenarios_root: Path,
        *,
        tags: str | None = None,
        scenarios: str | None = None,
    ) -> ScenarioFilter:
        """Build a filter from raw CLI flag values.

        Args:
            scenarios_root: Resolved path to the scenarios directory.
            tags: Comma-separated tag string (AND logic).  None = no tag filter.
            scenarios: Comma-separated path filter (group or group/scenario).
                       None = match all.

        Raises:
            ScenarioError: If a path filter references a non-existent group.
        """
        tag_set = frozenset(t.strip() for t in tags.split(",") if t.strip()) if tags else frozenset()
        parsed = _parse_path_filter(scenarios, scenarios_root) if scenarios else None
        return cls(
            scenarios_root=scenarios_root,
            tags=tag_set,
            parsed_paths=tuple(parsed) if parsed else None,
        )

    # ── Group-level filtering ──

    def matches_group(self, group_dir: Path) -> tuple[bool, frozenset[str]]:
        """Check whether a group directory passes the path filter.

        Returns:
            (matched, allowed_scenario_names)
            - ``allowed_scenario_names`` is empty when all scenarios are allowed.
        """
        if self.parsed_paths is None:
            return True, frozenset()

        g = group_dir.resolve()
        allowed: set[str] = set()
        matched = False
        for filter_path, scenario_name in self.parsed_paths:
            if g == filter_path or _is_descendant(g, filter_path) or _is_descendant(filter_path, g):
                matched = True
                if scenario_name and filter_path == g:
                    allowed.add(scenario_name)
                elif scenario_name is None or filter_path != g:
                    return True, frozenset()
        return matched, frozenset(allowed)

    # ── Scenario-level filtering ──

    def matches_scenario(self, scenario: Scenario, group_config: GroupConfig) -> bool:
        """Check whether a scenario passes the tag filter.

        Tags from the scenario and from the group's ``default_tags`` are merged;
        all required tags must be present (AND logic).
        """
        if not self.tags:
            return True
        all_tags = set(scenario.tags) | set(group_config.default_tags)
        return self.tags <= all_tags

    # ── Convenience: full check ──

    @property
    def has_path_filter(self) -> bool:
        return self.parsed_paths is not None

    @property
    def has_tag_filter(self) -> bool:
        return bool(self.tags)

    @property
    def is_empty(self) -> bool:
        """True when no filters are active (matches everything)."""
        return not self.has_path_filter and not self.has_tag_filter


# ── Private helpers ──


def _parse_path_filter(scenarios_filter: str, scenarios_root: Path) -> list[tuple[Path, str | None]]:
    """Parse ``--scenarios`` into (resolved_group_path, scenario_name | None) pairs.

    A scenario_name of None means "all scenarios in that group".

    When ``scenarios_root`` is itself a group (i.e. it contains ``_config.json``),
    two additional shapes are accepted so a user who points the runner directly
    at a group directory does not have to know whether the group is the root
    or a child of the root:

    1. Bare scenario name (no ``/``): ``--scenarios validation_21`` resolves
       to ``(scenarios_root, "validation_21")``.
    2. ``<root_basename>/<scenario>``: ``--scenarios mygroup/scenario_x`` is
       treated as ``(scenarios_root, "scenario_x")`` when ``scenarios_root``
       is named ``mygroup``. This matches the natural mental model of
       referring to a scenario by the group it visibly belongs to.

    Raises ScenarioError if a filter entry references a non-existent group directory.
    """
    root_is_group = (scenarios_root / "_config.json").is_file()
    result: list[tuple[Path, str | None]] = []
    for raw in scenarios_filter.split(","):
        f = raw.strip()
        if not f:
            continue

        if root_is_group:
            stripped = _strip_root_basename_prefix(f, scenarios_root)
            if stripped is not None and "/" not in stripped and "\\" not in stripped:
                result.append((scenarios_root, stripped.removesuffix(".json")))
                continue

        p = (scenarios_root / f).resolve()
        if p.is_dir():
            result.append((p, None))
            continue

        # Not a directory. A bare name with no path separator means the user
        # is naming what they think is a group; if it doesn't resolve we treat
        # it as a typo and fail loudly. This catches the common confusion
        # after a layout change (``--scenarios claude-code`` when the real
        # path is now ``agents/claude-code``); without this check the old
        # code silently fell through to ``(scenarios_root, "claude-code")``
        # and ran no scenarios.
        if "/" not in f and "\\" not in f:
            raise ScenarioError(_format_filter_error(f, scenarios_root, root_is_group))

        # Path with a separator → ``<group>/<scenario>`` form. The parent must
        # exist as a directory (the discovery layer filters out parents that
        # have no ``_config.json``); we require existence here only to fail
        # fast on typos.
        group_dir = p.parent
        if not group_dir.is_dir():
            raise ScenarioError(_format_filter_error(f, scenarios_root, root_is_group))
        result.append((group_dir, p.name.removesuffix(".json")))
    return result or []


def _strip_root_basename_prefix(filter_str: str, scenarios_root: Path) -> str | None:
    """If ``filter_str`` already includes ``scenarios_root.name`` as its first
    segment, return the remainder. Otherwise return ``filter_str`` unchanged
    when it has no separator (it could be a bare scenario name), or ``None``
    when it has a separator and does not start with the root's name (a real
    nested filter that must be parsed by the regular path).
    """
    if "/" not in filter_str and "\\" not in filter_str:
        return filter_str
    head, _, tail = filter_str.partition("/")
    if "\\" in head and "/" not in head:
        head, _, tail = filter_str.partition("\\")
    if head == scenarios_root.name and tail:
        return tail
    return None


def _format_filter_error(filter_str: str, scenarios_root: Path, root_is_group: bool) -> str:
    """Human-readable error for a filter that does not resolve.

    When ``scenarios_root`` is itself a group, the error explains how to filter
    a single scenario in that group (the most common pitfall - pointing the
    runner at a group and prefixing the filter with the group name, doubling
    the prefix on disk).
    """
    available = _list_available_groups(scenarios_root, root_is_group=root_is_group)
    lines = [
        f"No such group directory for filter '{filter_str}'.",
        f"  Your scenarios root is: {scenarios_root}",
    ]
    if root_is_group:
        prefixed = f"{scenarios_root.name}/<scenario>"
        lines.extend(
            [
                f"  This root IS itself a group ({scenarios_root.name}).",
                "  To filter a single scenario in it, pass the scenario name without a path:",
                "    --scenarios <scenario_name>",
                f"    --scenarios {prefixed}    (the redundant prefix is stripped)",
                "  Alternatively, pass the parent directory as the path and use the full filter.",
            ]
        )
    else:
        lines.append("  Filters must be relative to that root.")
    lines.append(f"  Available groups: {', '.join(available) or '(none)'}")
    return "\n".join(lines)


def _list_available_groups(scenarios_root: Path, *, root_is_group: bool | None = None) -> list[str]:
    """List group paths users can pass to ``--scenarios``.

    A "group" is any directory that contains a ``_config.json``; we walk the
    tree and return paths relative to ``scenarios_root`` so the suggestion is
    a drop-in replacement for the user's bad filter argument. We deliberately
    do not return intermediate "category" directories (e.g. ``experience/``,
    ``showcase/``) because they are not valid as the *group* a
    scenario filter resolves to - they are useful only as path prefixes that
    select multiple groups, which the descendant check in ``matches_group``
    already handles correctly.

    The bare ``"."`` (scenarios_root being itself a group) is replaced with a
    self-describing label so error output is not cryptic. ``root_is_group`` is
    accepted to avoid recomputing the ``_config.json`` stat in callers that
    already know.

    If no ``_config.json`` files exist (typical of a bare scratch directory in
    unit tests), fall back to listing top-level subdirectories so callers
    still see a useful "what's here" suggestion.
    """
    if root_is_group is None:
        root_is_group = (scenarios_root / "_config.json").is_file()

    groups: list[str] = []
    for cfg in scenarios_root.rglob("_config.json"):
        rel = cfg.parent.relative_to(scenarios_root)
        rel_str = str(rel)
        if rel_str == ".":
            groups.append(f"(this directory itself: {scenarios_root.name})")
        else:
            groups.append(rel_str)
    if groups:
        return sorted(groups)
    if scenarios_root.is_dir():
        return sorted(d.name for d in scenarios_root.iterdir() if d.is_dir() and not d.name.startswith("_"))
    return []


def _is_descendant(child: Path, parent: Path) -> bool:
    """True if child is a descendant of parent (no string-prefix false positives)."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False
