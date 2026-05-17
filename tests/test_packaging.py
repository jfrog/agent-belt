# (c) JFrog Ltd. (2026)

"""Regression guards for packaging - wheel must ship runtime assets.

The hatchling ``force-include`` entry in ``pyproject.toml`` copies
``examples/scenarios/showcase`` into ``belt/_bundled_examples``
inside the wheel. Without it, ``pip install agent-belt`` produces a
binary that cannot find its own quickstart scenarios, because editable
installs resolve ``examples/`` from the source tree but wheel installs
have no source tree to fall back to.

These tests fail fast if anyone removes the force-include or the
quickstart scenario disappears from the source tree, keeping the wheel
viable for public PyPI users.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# These constants are duplicated intentionally - quickstart.py imports them too.
# The duplication is the contract: if either side moves, this test catches it.
_QUICKSTART_GROUP = "showcase/correctness"
_QUICKSTART_SCENARIO = "correctness_basic.json"


def _hatch_force_include() -> dict[str, str]:
    pyproject = REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    return data["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]


def test_force_include_ships_bundled_examples():
    mapping = _hatch_force_include()
    src = "examples/scenarios/showcase"
    dst = "belt/_bundled_examples/scenarios/showcase"
    assert mapping.get(src) == dst, (
        f"pyproject.toml [tool.hatch.build.targets.wheel.force-include] "
        f"must map {src!r} -> {dst!r}. "
        "Removing this entry will break `pip install agent-belt` for every "
        "public user - quickstart cannot find scenarios in the wheel without it."
    )


def test_force_include_source_paths_exist():
    for src in _hatch_force_include().keys():
        path = REPO_ROOT / src
        assert path.is_dir(), (
            f"pyproject.toml force-includes {src!r} but the directory does not "
            "exist on disk - the wheel build will fail."
        )


def test_quickstart_scenario_is_in_force_included_tree():
    """The exact scenario quickstart needs must live under the bundled tree."""
    mapping = _hatch_force_include()
    assert any(
        "showcase" in dst for dst in mapping.values()
    ), "Expected at least one force-include entry copying showcase/"
    quickstart_file = REPO_ROOT / "examples" / "scenarios" / _QUICKSTART_GROUP / _QUICKSTART_SCENARIO
    assert quickstart_file.is_file(), (
        f"quickstart scenario {quickstart_file} not found - quickstart will fail "
        "for wheel users even though `pip install agent-belt` succeeds."
    )


def test_no_python_pre_311_packaging():
    """tomllib used above is stdlib only on 3.11+. Project declares this minimum."""
    assert sys.version_info >= (3, 11), (
        "Project declares requires-python >=3.11 in pyproject.toml - bump the " "test if that changes."
    )
