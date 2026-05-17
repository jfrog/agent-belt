# (c) JFrog Ltd. (2026)

"""Load and validate scenario and group config files."""

from __future__ import annotations

import json
from pathlib import Path

from belt.constants import GROUP_CONFIG_FILE, NON_SCENARIO_FILES
from belt.parser.strict import StrictConfigError, validate_strict
from belt.scenario import GroupConfig, Scenario


class ScenarioLoader:
    """Loads scenario JSON files and group configs from disk.

    Both loaders accept ``strict_config``: when True, the parser runs
    a schema-driven check that rejects keys not declared on the
    target Pydantic model and not registered as a plugin extension
    (see :mod:`belt.parser.strict`). Default OFF so existing
    permissive scenarios continue to load.
    """

    @staticmethod
    def load_group_config(group_dir: Path, *, strict_config: bool = False) -> GroupConfig:
        config_path = group_dir / GROUP_CONFIG_FILE
        text = config_path.read_text()
        if strict_config:
            raw = json.loads(text)
            errors = validate_strict(raw, GroupConfig, source=str(config_path))
            if errors:
                raise StrictConfigError(errors)
        return GroupConfig.model_validate_json(text)

    @staticmethod
    def load_scenario(path: Path, *, strict_config: bool = False) -> Scenario:
        text = path.read_text()
        if strict_config:
            raw = json.loads(text)
            errors = validate_strict(raw, Scenario, source=str(path))
            if errors:
                raise StrictConfigError(errors)
        scenario = Scenario.model_validate_json(text)
        # Stash the scenario's on-disk location so the LLM scorer can resolve
        # ``llm_scorer_evidence_files`` against it (and reject paths that
        # escape it). Resolved up front so symlinks at the parent are
        # canonicalised once and the traversal guard sees a stable root.
        scenario._source_dir = path.parent.resolve()
        return scenario

    @staticmethod
    def load_group_scenarios(group_dir: Path, *, strict_config: bool = False) -> tuple[list[Scenario], list[str]]:
        """Returns (valid_scenarios, error_messages). Malformed files are skipped."""
        scenarios = []
        errors = []
        for p in sorted(group_dir.glob("*.json")):
            # Files prefixed with ``_`` are group-internal assets (group config
            # or any plugin-specific files), never scenarios.
            if p.name in NON_SCENARIO_FILES or p.name.startswith("_"):
                continue
            try:
                scenarios.append(ScenarioLoader.load_scenario(p, strict_config=strict_config))
            except Exception as e:
                errors.append(f"{p.name}: {e}")
        return scenarios, errors
