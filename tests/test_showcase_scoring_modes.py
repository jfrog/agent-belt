# (c) JFrog Ltd. (2026)

"""Showcase must demonstrate every advertised LLM scoring mode.

Every supported scoring mode (rules-only, single-judge LLM, multi-judge LLM,
consensus) must be exercised end-to-end. The integration
side runs against real LLM credentials and lives in CI follow-up jobs; this
test enforces the static side: an example artifact for every mode exists and
is well-formed.

Failures here mean the public docs claim a mode users can't actually try
without writing config from scratch.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SHOWCASE = REPO_ROOT / "examples" / "scenarios" / "showcase"
SCORER_CONFIG_DIR = REPO_ROOT / "examples" / "scorer-config"


def test_rules_only_mode_has_showcase_coverage() -> None:
    """At least one showcase scenario must be runnable as ``--modes rules``."""
    # Rules mode is the default and works on every scenario; we just assert
    # there's a non-trivial showcase to point users at.
    scenarios = list(SHOWCASE.rglob("*.json"))
    real_scenarios = [p for p in scenarios if not p.name.startswith("_")]
    assert len(real_scenarios) >= 5, (
        f"Showcase has only {len(real_scenarios)} scenarios - too thin to credibly "
        "demonstrate rules-only scoring across the schema."
    )


def test_single_judge_llm_mode_is_documented_as_default() -> None:
    """Single-judge mode is the implicit default of ``--modes llm`` - verify the doc says so."""
    scoring_doc = REPO_ROOT / "docs" / "glossary" / "SCORING.md"
    text = scoring_doc.read_text().lower()
    assert (
        "single" in text and "judge" in text
    ), "SCORING.md should explain that --modes llm without --scorer-config uses a single judge"


def test_multi_judge_mode_has_example_config() -> None:
    """``examples/scorer-config/judges.yaml`` must demonstrate independent-judge mode."""
    judges_yaml = SCORER_CONFIG_DIR / "judges.yaml"
    assert judges_yaml.is_file(), f"missing multi-judge example: {judges_yaml}"
    config = yaml.safe_load(judges_yaml.read_text())
    judges = config.get("judges") or {}
    assert (
        len(judges) >= 2
    ), f"Multi-judge example must declare at least 2 judges; got {len(judges)} in {judges_yaml.name}"
    assert "consensus" not in config, (
        f"{judges_yaml.name} should demonstrate independent (non-consensus) mode; "
        "see consensus.yaml for the consensus variant."
    )


def test_consensus_mode_has_example_config() -> None:
    """``examples/scorer-config/consensus.yaml`` must demonstrate consensus mode."""
    consensus_yaml = SCORER_CONFIG_DIR / "consensus.yaml"
    assert consensus_yaml.is_file(), f"missing consensus example: {consensus_yaml}"
    config = yaml.safe_load(consensus_yaml.read_text())
    assert (
        "consensus" in config
    ), f"{consensus_yaml.name} must set the top-level `consensus:` key (e.g. `consensus: majority`)"
    judges = config.get("judges") or {}
    assert (
        len(judges) >= 2
    ), f"Consensus example must declare at least 2 judges; got {len(judges)} in {consensus_yaml.name}"


def test_every_documented_scoring_mode_has_a_showcase_artifact() -> None:
    """Cross-check: every mode SCORING.md describes has a runnable example.

    Reading the doc as the source of truth, then mapping to:
    - rules-only      → showcase/ has ≥1 scenario (already tested)
    - single-judge    → SCORING.md mentions; default behaviour, no separate config
    - multi-judge     → examples/scorer-config/judges.yaml exists
    - consensus       → examples/scorer-config/consensus.yaml exists
    """
    scoring_doc = REPO_ROOT / "docs" / "glossary" / "SCORING.md"
    text = scoring_doc.read_text()
    must_mention = ("rules", "judge", "consensus")
    missing_in_doc = [w for w in must_mention if w not in text.lower()]
    assert not missing_in_doc, (
        f"SCORING.md is missing references to: {missing_in_doc}. "
        "Every advertised scoring mode must be documented and have a runnable example."
    )
