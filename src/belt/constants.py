# (c) JFrog Ltd. (2026)

"""Shared constants for the evaluation framework.

Path resolution
~~~~~~~~~~~~~~~
``OUTCOMES_ROOT`` resolves via: ``--outcomes-dir`` CLI flag (runtime override in
``commands/run.py`` and ``commands/eval.py``) > ``BELT_OUTCOMES_DIR`` env var >
``cwd/outcomes`` default.

The default is relative to the current working directory - outcomes are
write-only output and belong where the user runs the command. This avoids a bug
where ``Path(__file__).resolve()`` follows editable-install symlinks to a
different workspace, causing outcomes to land in the wrong tree.

``SCENARIOS_DIR`` and ``ENV_FILE`` are relative to the repo root when running
from source, but resolve against cwd when installed as a package.
"""

import os
import re
from pathlib import Path

from belt import envvars

SRC_DIR = Path(__file__).resolve().parent
EVAL_DIR = SRC_DIR.parent
REPO_ROOT = EVAL_DIR.parent

# Public env var for customizing outcomes location.
# Re-exported from ``envvars`` for back-compat with existing imports.
OUTCOMES_DIR_ENV = envvars.OUTCOMES_DIR

# Outcomes directory: env var > default (cwd/outcomes).
# CLI --outcomes-dir overrides both (applied at runtime in commands/run.py and commands/eval.py).
_outcomes_env = os.environ.get(OUTCOMES_DIR_ENV)
OUTCOMES_ROOT = Path(_outcomes_env) if _outcomes_env else Path.cwd() / "outcomes"

# When installed as a package, EVAL_DIR is inside site-packages - not a valid
# location for scenarios or .env. Detect and fall back to cwd-relative paths.
_installed_in_site_packages = "site-packages" in EVAL_DIR.parts

SCENARIOS_DIR = Path.cwd() / "scenarios" if _installed_in_site_packages else REPO_ROOT / "examples"
ENV_FILE = Path.cwd() / ".env" if _installed_in_site_packages else REPO_ROOT / ".env"

MANIFEST_FILE = ".manifest.json"
LOG_FILE = "eval.log"
SCORE_FILE = "score.json"
RESULTS_FILE = "results.json"
GROUP_CONFIG_FILE = "_config.json"
RUN_META_FILE = "run_meta.json"

# Per-scenario sidecar written by the orchestrator after ``agent.setup()``.
# Captures the agent's runtime identity (CLI binary path + version, auth
# signals, redacted ``-X`` args). The benchmark-card builder reads every
# sidecar under a run directory and deduplicates them per group.
RUNTIME_INFO_FILE = "_runtime_info.json"

# Reproducibility manifest ("benchmark card") emitted by ``belt aggregate``.
# JSON is the machine-readable contract; the Markdown sibling is what we
# append to ``$GITHUB_STEP_SUMMARY`` and what humans read in PR comments.
BENCHMARK_CARD_JSON_FILE = "benchmark-card.json"
BENCHMARK_CARD_MD_FILE = "benchmark-card.md"

TURN_CLI_TEMPLATE = "turn_{}_cli.txt"
TURN_STATE_TEMPLATE = "turn_{}_state.json"
TURN_OUTPUT_TEMPLATE = "turn_{}_output.json"
TURN_STREAM_TEMPLATE = "turn_{}_stream.ndjson"

DEFAULT_API_VERSION = "2024-10-21"

DEFAULT_MAX_MESSAGE_ARG_LEN = 100_000  # overridable via runner --max-message-len

# Schema cap on ``Turn.message``. Set far above any realistic per-turn payload
# (multi-PR release diffs, full-file pastes, log-bundle analyses are all
# comfortably under 1 MB) so it acts as a crash-prevention guardrail rather
# than as a coverage limit. The agent adapter and the LLM judge prompt builder
# enforce their own boundary-specific caps independently.
TURN_MESSAGE_MAX_CHARS = 10_000_000

# Trial-suffix encoding: ``<scenario>__trial_N`` is the on-disk marker the
# runner writes when ``--trials N>1``.  The scorer strips it to find the
# scenario JSON; the aggregator captures N to compute pass^k.  One regex,
# one source of truth - keeps the on-disk contract from drifting between
# producer and consumers.
TRIAL_SUFFIX_RE = re.compile(r"__trial_(\d+)$")

# Hard upper bound on per-scenario turn discovery loops.  Mirrors
# ``Scenario.turns: max_length=100`` in the entity schema; the aggregator and
# scorer use this constant to bound their ``turn_*_output.json`` walks (each
# loop breaks on the first missing file, so the cap only matters if a
# scenario somehow reaches it).
MAX_TURNS_PER_SCENARIO = 100

# ── Canonical example LLM model ──
#
# Single source of truth for "the model name we recommend in CLI help, error
# messages, and the doctor diagnostic." Bumping this constant updates every
# code-side surface in one place. Standalone markdown examples may use
# different model names - they are illustrative, not normative.
#
# Why ``gpt-5.4-mini``: cheaper than gpt-4.1 ($0.75/M in vs $2/M), more
# capable as a judge per OpenAI's released benchmarks, and a representative
# new-family model that exercises the ``max_completion_tokens`` code path.
EXAMPLE_LLM_MODEL = "openai/gpt-5.4-mini"

# Output artifact schema version.  Bump when TurnOutput, ScenarioScore,
# run_meta.json, results.json, or benchmark-card.json formats change in
# backward-incompatible ways. Adding optional fields is additive and does
# not require a bump - readers fall back to defaults for missing fields.
SCHEMA_VERSION = "1"

# ── Plugin entry-point group names ──
#
# Single source of truth for the strings that ``importlib.metadata`` looks up
# when discovering third-party plugins (Design Principle 9). Registry code
# imports these constants; it never re-spells the literal. A typo in a
# constant import is a hard ImportError, where a typo in an inline literal
# silently looks up an empty plugin group.
ENTRY_POINT_GROUP_AGENTS = "belt.agents"
ENTRY_POINT_GROUP_SCORERS = "belt.scorers"
ENTRY_POINT_GROUP_EXPORTERS = "belt.exporters"

NON_SCENARIO_FILES = {GROUP_CONFIG_FILE}

ERROR_PATTERNS = [
    "traceback (most recent call last)",
    "runtimeerror: ",
    "attributeerror: ",
    "keyerror: ",
    "typeerror: ",
    "nameerror: ",
    "valueerror: ",
    "assertionerror: ",
    "importerror: ",
    "oserror: ",
    "filenotfounderror: ",
    "permissionerror: ",
    "indexerror: ",
    "zerodivisionerror: ",
    "validation error",
    "error in stream",
]
