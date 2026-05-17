# (c) JFrog Ltd. (2026)

"""Single source of truth for ``_BELT_*`` (private) env-var names.

These names are used **only** for state hand-off between the ``run``,
``score``, and ``aggregate`` phases of a single ``belt eval``
invocation. They are not part of the public API and must not be set or
read by users.

The leading underscore is the API-stability signal: anything in this
module may be renamed or removed without a deprecation cycle. Public
names live in :mod:`belt.envvars`.

Why a separate module
~~~~~~~~~~~~~~~~~~~~~
``envvars.py`` advertises and tests every ``BELT_*`` (no leading
underscore) name. Mixing private names into that module would either
(a) widen the public surface or (b) force the typo-detection test to
special-case underscore prefixes. A separate module gives each surface
its own ``ALL_*`` registry and its own typo test.
"""

from __future__ import annotations

from typing import Final

# Common prefix for every private handoff variable in this module.
# Every ``_BELT_*`` constant below uses this as a literal prefix;
# external code that needs to match (or strip) the prefix in bulk - e.g.
# the subprocess-env builder that filters private vars before forwarding
# to a child agent CLI - imports this name rather than re-typing the
# literal. Renaming the prefix is intentionally a one-line change.
PREFIX: Final[str] = "_BELT_"

# Run directory path. Set by ``commands/run.py`` after creating the run
# directory; read by ``commands/eval.py`` to chain the score and
# aggregate phases on the same run.
LAST_RUN_DIR: Final[str] = "_BELT_LAST_RUN_DIR"

# Absolute path to ``run_dir/eval.log``. Set by ``commands/run.py`` and
# ``commands/score.py`` after configuring the logger; read by
# ``cli.py``'s error path so ``❌ ...  → logs: <path>`` can point at the
# file the subcommand wrote. Internal because it is only useful for the
# top-level ``cli.py`` error handler within the same Python process.
LOG_FILE: Final[str] = "_BELT_LOG_FILE"

# Resolved scenarios root path. Set by ``commands/run.py`` from the
# user-provided positional path; read by ``commands/score.py`` to map
# outcome directories back to scenario files.
SCENARIOS_ROOT: Final[str] = "_BELT_SCENARIOS_ROOT"

# JSON-serialised scorer descriptions. Set by ``commands/eval.py``
# after scorer preflight; read by ``commands/run.py`` to display scorer
# info in progress headers without re-importing scorer modules.
SCORER_DESCS: Final[str] = "_BELT_SCORER_DESCS"

# JSON-serialised original ``argv`` of the top-level ``belt`` invocation.
# Set by ``commands/eval.py`` (and any future composite command) before it
# delegates to ``commands/run.py``; read by ``run.initialize_run_dir`` so
# the benchmark card records the user's actual command line rather than the
# synthesised ``run_argv`` that ``eval`` builds internally.
ORIGINAL_ARGV: Final[str] = "_BELT_ORIGINAL_ARGV"

# Comma-separated list of scoring modes (``rules``, ``llm``) selected by the
# top-level ``belt eval`` invocation. Set by ``commands/eval.py``
# before delegating to ``commands/run.py``; read by ``run.initialize_run_dir``
# so the benchmark card records which scorer modes were active. ``--modes``
# is owned by the ``score`` and ``eval`` parsers (not ``run``), so the value
# is unavailable to ``run.args`` and must be plumbed explicitly.
SCORING_MODES: Final[str] = "_BELT_SCORING_MODES"

# Used by tests to assert no ``_BELT_*`` literal exists outside this
# module. A typo'd handoff variable would silently break the
# run -> score -> aggregate chain; the regression test catches it at the
# literal-string level.
ALL_INTERNAL_NAMES: Final[frozenset[str]] = frozenset(
    {
        LAST_RUN_DIR,
        LOG_FILE,
        ORIGINAL_ARGV,
        SCENARIOS_ROOT,
        SCORER_DESCS,
        SCORING_MODES,
    }
)
