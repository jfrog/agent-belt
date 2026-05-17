# (c) JFrog Ltd. (2026)

"""Doc-code parity for bundled agents.

Every bundled agent (entry point under ``[project.entry-points."belt.agents"]``)
must be listed in ``docs/glossary/AGENT-FEATURES.md`` and conversely.

Additionally, each agent must be reachable via the registry - registering an
entry point that points at a missing class is a packaging bug ``agent list``
would surface; this test fails earlier (in CI, before publish).

Only agents registered by the ``belt`` distribution itself are checked -
third-party plugins and reference examples (e.g. ``examples/custom-agent``)
also register under the same entry-point group but are out of scope for the
framework's own feature matrix.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_FEATURES_DOC = REPO_ROOT / "docs" / "glossary" / "AGENT-FEATURES.md"
FRAMEWORK_DIST = "agent-belt"


def _entry_point_agents() -> set[str]:
    """Entry-point names registered for agents by the framework distribution.

    Filters out agents contributed by other installed distributions
    (third-party plugins, reference examples) so the parity check stays
    deterministic regardless of what else happens to be installed in the
    developer's environment.
    """
    from importlib.metadata import entry_points

    eps = entry_points(group="belt.agents")
    return {ep.name for ep in eps if getattr(ep, "dist", None) is not None and ep.dist.name == FRAMEWORK_DIST}


def test_every_entry_point_agent_is_in_doc() -> None:
    code_agents = _entry_point_agents()
    assert code_agents, "No agents registered via entry points - packaging is broken"
    doc = AGENT_FEATURES_DOC.read_text()
    missing = sorted(name for name in code_agents if f"`{name}`" not in doc)
    assert not missing, (
        f"Agents registered via entry points but missing from AGENT-FEATURES.md: {missing}. "
        "Add a row to the Quick Reference table and the Feature Matrix."
    )


def test_documented_agents_have_entry_points() -> None:
    """Every agent name in AGENT-FEATURES.md's Quick Reference table is registered."""
    import re

    doc = AGENT_FEATURES_DOC.read_text()
    # Quick Reference rows look like: ``| `claude-code` | `claude` | NDJSON | ...``
    quick_ref_names = set(re.findall(r"^\|\s*`([a-z][a-z-]+)`\s*\|", doc, flags=re.MULTILINE))
    code_agents = _entry_point_agents()
    orphans = sorted(quick_ref_names - code_agents)
    assert not orphans, (
        f"AGENT-FEATURES.md lists agents without a registered entry point: {orphans}. "
        'Add to pyproject.toml [project.entry-points."belt.agents"].'
    )


@pytest.mark.parametrize("name", sorted(_entry_point_agents()))
def test_entry_point_agent_class_is_loadable(name: str) -> None:
    """Each registered agent's class must actually import (catches packaging bugs)."""
    from belt.agent.registry import get_agent_class

    cls = get_agent_class(name)
    assert cls is not None, f"agent {name!r} resolved to None"
    # Must be a BaseAgentAdapter subclass (legacy name retained as the framework
    # base class - the user-facing term is still ``agent``).
    from belt.agent.base import BaseAgentAdapter

    assert issubclass(cls, BaseAgentAdapter), f"agent {name!r} class {cls.__name__} is not a BaseAgentAdapter subclass"
