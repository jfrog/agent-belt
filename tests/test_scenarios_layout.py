# (c) JFrog Ltd. (2026)

"""Drift tests for the bundled scenarios layout.

The bundled ``examples/scenarios/`` follows a deliberate two-axis shape:

    examples/scenarios/
    ├── experience/<project>/         # full code-editing campaigns per fixture
    └── showcase/<capability>/        # capability-per-group schema demonstrations

Without an enforced layout, regressions creep in: a forgotten capability
sub-grouping makes a namespace look complete when it isn't, ad-hoc
``<agent>-editing/`` directories reappear at the top level, or a domain
campaign lands under ``showcase/`` and dilutes the meaning of the namespace.

These tests don't validate scenario *content* (that's other suites' job) -
only that the directory shape is what the docs and tooling assume.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_ROOT = REPO_ROOT / "examples" / "scenarios"
EXPERIENCE_ROOT = SCENARIOS_ROOT / "experience"
SHOWCASE_ROOT = SCENARIOS_ROOT / "showcase"


def _showcase_capability_dirs() -> list[Path]:
    return sorted(d for d in SHOWCASE_ROOT.iterdir() if d.is_dir() and not d.name.startswith("_"))


def test_top_level_layout_is_experience_and_showcase() -> None:
    """``examples/scenarios/`` should contain exactly the ``experience/`` and ``showcase/`` namespaces.

    Anything else at the top level (``adapters/``, ``agents/``, ``claude-code/``,
    ``claude-code-editing/``, ``bookstore-claude/``, …) means the layout
    migration regressed.
    """
    top = sorted(d.name for d in SCENARIOS_ROOT.iterdir() if d.is_dir() and not d.name.startswith("_"))
    assert top == ["experience", "showcase"], (
        f"Unexpected top-level entries in examples/scenarios/: {top}. " f"Expected exactly ['experience', 'showcase']."
    )


def test_no_legacy_layout_directories() -> None:
    """No top-level ``adapters/``, ``agents/``, ``<agent>-editing/``, or bare ``<agent>/`` sibling.

    The legacy layout used these patterns; they must not reappear as a
    quick fix when adding new scenarios.
    """
    legacy_patterns = [
        d.name
        for d in SCENARIOS_ROOT.iterdir()
        if d.is_dir()
        and (
            d.name in {"adapters", "agents"}
            or d.name.endswith("-editing")
            or d.name in {"claude-code", "codex", "cursor", "gemini", "goose", "opencode", "copilot"}
        )
    ]
    assert legacy_patterns == [], f"Legacy sibling directories found: {legacy_patterns}"


@pytest.mark.parametrize("capability_dir", _showcase_capability_dirs(), ids=lambda d: d.name)
def test_showcase_capability_has_config_and_scenarios(capability_dir: Path) -> None:
    """Every ``showcase/<capability>/`` group owns a ``_config.json`` and at least one scenario."""
    cfg = capability_dir / "_config.json"
    assert cfg.is_file(), f"Missing {cfg.relative_to(REPO_ROOT)}"
    scenarios = [p for p in capability_dir.glob("*.json") if p.name != "_config.json" and not p.name.startswith("_")]
    assert scenarios, (
        f"showcase/{capability_dir.name}/ has no scenario JSON files - every capability "
        f"group must demonstrate at least one expectation field with a real scenario."
    )


def test_experience_groups_are_flat_and_have_config() -> None:
    """``experience/<project>/`` groups have no nested sub-grouping; each owns a ``_config.json``."""
    for project_dir in sorted(d for d in EXPERIENCE_ROOT.iterdir() if d.is_dir() and not d.name.startswith("_")):
        cfg = project_dir / "_config.json"
        assert cfg.is_file(), f"Missing {cfg.relative_to(REPO_ROOT)}"
        nested_dirs = [d for d in project_dir.iterdir() if d.is_dir()]
        assert nested_dirs == [], (
            f"experience/{project_dir.name}/ should be a flat group, found sub-directories: "
            f"{[d.name for d in nested_dirs]}"
        )


def test_showcase_covers_capability_matrix() -> None:
    """Each capability subgroup from the showcase matrix exists under ``showcase/``.

    The matrix is the contract documented in
    ``examples/README.md``'s "Schema features by example" table; if you remove
    or rename a subgroup, update both surfaces in the same change.
    """
    expected = {
        "correctness",
        "tool-trajectory",
        "budgets-latency",
        "agent-capabilities",
        "error-types",
        "editing-workspace",
        "external-fixture",
        "group-config-fields",
        "verdict-scales",
    }
    actual = {d.name for d in _showcase_capability_dirs()}
    missing = expected - actual
    assert not missing, (
        f"showcase/ is missing capability subgroup(s): {sorted(missing)}. "
        f"Add them or update both this test and examples/README.md."
    )


def _model_field_names(model_name: str) -> set[str]:
    """Source-of-truth: the Pydantic model defines the schema; tests follow."""
    from belt import scenario as scenario_mod

    cls = getattr(scenario_mod, model_name)
    return set(cls.model_fields.keys())


def _showcase_used_keys() -> tuple[set[str], set[str], set[str]]:
    """Crawl every showcase scenario JSON and collect the keys actually used."""
    import json

    turn_keys: set[str] = set()
    state_keys: set[str] = set()
    group_keys: set[str] = set()

    for jf in SHOWCASE_ROOT.rglob("*.json"):
        if jf.name.startswith("_") and jf.name != "_config.json":
            continue
        data = json.loads(jf.read_text())
        if jf.name == "_config.json":
            group_keys.update(data.keys())
            continue
        for turn in data.get("turns", []):
            turn_keys.update(turn.get("expect", {}).keys())
            state_keys.update(turn.get("state_expect", {}).keys())

    return turn_keys, state_keys, group_keys


def test_showcase_demonstrates_every_turn_expectation_field() -> None:
    """Every ``TurnExpectation`` field must be exercised by at least one showcase scenario.

    Without this, contributors add fields to the Pydantic model that no example
    documents - readers of ``examples/README.md`` then can't tell what the field
    looks like in practice. Add a scenario (real-runnable or dry-run-only) that
    sets the field, even if only to demonstrate the schema shape.
    """
    declared = _model_field_names("TurnExpectation")
    turn_keys, _, _ = _showcase_used_keys()
    missing = declared - turn_keys
    assert not missing, (
        f"Showcase missing TurnExpectation field(s): {sorted(missing)}. "
        "Add a scenario under examples/scenarios/showcase/ that uses each "
        "missing field - tag with `dry-run-only` if it can't be reliably "
        "exercised by every agent."
    )


def test_showcase_demonstrates_every_state_expectation_field() -> None:
    """Every ``StateExpectation`` field must be exercised by at least one showcase scenario."""
    declared = _model_field_names("StateExpectation")
    _, state_keys, _ = _showcase_used_keys()
    missing = declared - state_keys
    assert not missing, (
        f"Showcase missing StateExpectation field(s): {sorted(missing)}. "
        "Add or extend a scenario under examples/scenarios/showcase/editing-workspace/."
    )


def test_showcase_demonstrates_every_group_config_field() -> None:
    """Every ``GroupConfig`` field must appear in at least one showcase ``_config.json``."""
    declared = _model_field_names("GroupConfig")
    _, _, group_keys = _showcase_used_keys()
    missing = declared - group_keys
    assert not missing, (
        f"Showcase missing GroupConfig field(s): {sorted(missing)}. "
        "Add the field to examples/scenarios/showcase/group-config-fields/_config.json "
        "or another existing showcase _config.json that demonstrates it."
    )


def test_showcase_demonstrates_verify_field() -> None:
    """`Turn.verify` and `Scenario.verify` must each be exercised by a showcase scenario.

    The crawls above only cover ``expect`` / ``state_expect`` / ``_config.json``
    keys, so the `Turn`/`Scenario`-level ``verify`` field needs its own guard -
    otherwise the deterministic exec-test grader could ship with no worked
    example. Mirrors the per-field coverage the other guards enforce.
    """
    import json

    turn_verify = False
    scenario_verify = False
    for jf in SHOWCASE_ROOT.rglob("*.json"):
        if jf.name == "_config.json" or jf.name.startswith("_"):
            continue
        data = json.loads(jf.read_text())
        if "verify" in data:
            scenario_verify = True
        for turn in data.get("turns", []):
            if "verify" in turn:
                turn_verify = True

    assert turn_verify, (
        "No showcase scenario exercises `Turn.verify`. Add one under "
        "examples/scenarios/showcase/verify/ that sets `verify` on a turn."
    )
    assert scenario_verify, (
        "No showcase scenario exercises `Scenario.verify`. Add one under "
        "examples/scenarios/showcase/verify/ that sets a top-level `verify`."
    )
