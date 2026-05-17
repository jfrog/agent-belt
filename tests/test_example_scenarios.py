# (c) JFrog Ltd. (2026)

"""Validate that all example scenario files parse without errors."""

from __future__ import annotations

import glob
from pathlib import Path

import pytest

from belt.parser.scenario import ScenarioLoader

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "scenarios"
SCENARIO_FILES = sorted(glob.glob(str(EXAMPLES_DIR / "**" / "*.json"), recursive=True))
CONFIG_FILES = sorted(glob.glob(str(EXAMPLES_DIR / "**" / "_config.json"), recursive=True))


@pytest.mark.parametrize("path", SCENARIO_FILES, ids=lambda p: str(Path(p).relative_to(EXAMPLES_DIR)))
def test_example_scenario_is_valid(path):
    p = Path(path)
    if p.name.startswith("_"):
        pytest.skip("group-internal asset (group config or plugin-specific files), not a scenario")
    scenario = ScenarioLoader.load_scenario(p)
    assert scenario.name == p.stem
    assert len(scenario.turns) >= 1


@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: str(Path(p).relative_to(EXAMPLES_DIR)))
def test_example_group_config_is_valid(path):
    config = ScenarioLoader.load_group_config(Path(path).parent)
    assert config.agent  # must have an agent name
