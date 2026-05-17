# (c) JFrog Ltd. (2026)

"""Drift test for multi-turn templating coverage.

``runner/orchestrator.py`` declares the closed set of supported template
fields as ``_TEMPLATE_FIELDS = ("reply_text", "git_diff", "tool_sequence")``.
Without a parity check, a contributor can add a new field to the runner
plus a unit test for it and ship -- with no example in
``examples/scenarios/showcase/``. Authors writing scenarios learn the
contract from the showcase, so a missing example silently narrows the
public surface.

This test scans every showcase scenario JSON for ``{{prev.X}}`` /
``{{turn_N.X}}`` placeholders and asserts that every declared field
appears in at least one scenario. Style mirrors ``test_scenarios_layout.py``
-- a mechanical drift guard, not a content check.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from belt.runner.orchestrator import _TEMPLATE_FIELDS

REPO_ROOT = Path(__file__).resolve().parent.parent
SHOWCASE_ROOT = REPO_ROOT / "examples" / "scenarios" / "showcase"

# Same shape as ``orchestrator._PLACEHOLDER_RE`` but field-only -- we do
# not care about the scope (``prev`` vs ``turn_N``) for coverage purposes.
_FIELD_RE = re.compile(r"\{\{\s*(?:prev|turn_\d+)\.(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def _collect_fields_used_in_showcase() -> set[str]:
    used: set[str] = set()
    for scenario_path in SHOWCASE_ROOT.rglob("*.json"):
        if scenario_path.name.startswith("_"):
            continue  # group ``_config.json`` files
        try:
            data = json.loads(scenario_path.read_text())
        except json.JSONDecodeError:
            continue
        for turn in data.get("turns", []):
            message = turn.get("message", "")
            used.update(m.group("field") for m in _FIELD_RE.finditer(message))
    return used


def test_every_declared_template_field_has_a_showcase_example() -> None:
    """Every entry in ``_TEMPLATE_FIELDS`` must appear in at least one
    showcase scenario placeholder.

    Add a new field to ``_TEMPLATE_FIELDS`` -> add a showcase example
    in the same PR. The test failure points the author at the
    ``examples/scenarios/showcase/`` tree.
    """
    declared = set(_TEMPLATE_FIELDS)
    used = _collect_fields_used_in_showcase()
    missing = declared - used
    assert not missing, (
        f"Declared template fields without a showcase example: {sorted(missing)}. "
        f"Add a `{{{{prev.<field>}}}}` reference under examples/scenarios/showcase/ "
        f"(see editing-workspace/multi_turn_templating_workspace.json or "
        f"correctness/multi_turn_templating.json for the established pattern)."
    )


def test_showcase_only_uses_declared_template_fields() -> None:
    """The reverse direction: a showcase scenario referencing an
    undeclared field would fail at runtime with ``unsupported template
    field`` -- catch it statically so a typo (``{{prev.cost}}`` vs
    ``{{prev.cost_usd}}``) fails CI instead of every user.
    """
    declared = set(_TEMPLATE_FIELDS)
    used = _collect_fields_used_in_showcase()
    undeclared = used - declared
    assert not undeclared, (
        f"Showcase scenarios reference undeclared template fields: "
        f"{sorted(undeclared)}. Either add them to "
        f"`runner/orchestrator.py::_TEMPLATE_FIELDS` (and `_format_field`) "
        f"or fix the scenario placeholder."
    )
