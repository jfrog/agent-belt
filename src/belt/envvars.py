# (c) JFrog Ltd. (2026)

"""Single source of truth for BELT_* environment variable names.

Every public ``BELT_*`` env var is declared once here, as a
``Final[str]``. Consumers must import from this module rather than
spelling literals inline. A typo in a literal is silently a different
env var; a typo in a constant import is a hard import error.

Internal handoffs use the ``_BELT_*`` prefix (underscore-leading) and
are *not* part of the public surface; they exist only so the ``run`` /
``score`` / ``aggregate`` phases of a single ``belt eval`` invocation
can hand state to one another. Underscore-prefixed names live in
``constants.py`` because they are tightly coupled to file-layout constants
(``OUTCOMES_ROOT`` etc.).

Three accessor helpers (``is_truthy``, ``get_int``, ``get_str``) replace
the repeated ``os.environ.get(name, "") in ("1", "true", "yes")`` idiom
that was scattered across the codebase.
"""

from __future__ import annotations

import os
from typing import Final

# Public namespace prefix. Mirrors ``belt._internal_envvars.PREFIX`` ("_BELT_")
# for the private handoff space. Callers that need to filter the env by
# namespace (notably ``belt.agent.base.build_subprocess_env``) should import
# this constant rather than spelling the literal "BELT_" inline.
#
# Leading-underscore name: this is a meta-constant (the prefix, not an env
# var), so the ``test_envvars`` parity contract exempts it the same way it
# exempts ``ALL_NAMES`` / accessor helpers.
_PREFIX: Final[str] = "BELT_"

# ── Public toggles (boolean) ───────────────────────────────────────────────

ALLOW_FULL_ENV: Final[str] = "BELT_ALLOW_FULL_ENV"
ALLOW_ARBITRARY_AGENT: Final[str] = "BELT_ALLOW_ARBITRARY_AGENT"
ALLOW_ARBITRARY_EXPORTER: Final[str] = "BELT_ALLOW_ARBITRARY_EXPORTER"
ALLOW_ARBITRARY_SCORER: Final[str] = "BELT_ALLOW_ARBITRARY_SCORER"
SILENCE_CUSTOM_BASE_URL_WARNING: Final[str] = "BELT_SILENCE_CUSTOM_BASE_URL_WARNING"
ALLOW_INSECURE_BASE_URL: Final[str] = "BELT_ALLOW_INSECURE_BASE_URL"
ALLOW_EXTERNAL_WORKING_DIR: Final[str] = "BELT_ALLOW_EXTERNAL_WORKING_DIR"
ALLOW_INPLACE: Final[str] = "BELT_ALLOW_INPLACE"
ALLOW_VERIFY_EXEC: Final[str] = "BELT_ALLOW_VERIFY_EXEC"
SANDBOX_PROVIDER: Final[str] = "BELT_SANDBOX_PROVIDER"
NO_DOTENV: Final[str] = "BELT_NO_DOTENV"
NO_UMASK: Final[str] = "BELT_NO_UMASK"
DEBUG: Final[str] = "BELT_DEBUG"

# Terminal log verbosity for ``belt`` commands. Accepts the same level
# names as loguru (``TRACE``/``DEBUG``/``INFO``/``SUCCESS``/``WARNING``/
# ``ERROR``/``CRITICAL``), case-insensitive. Mirrors ``-v`` / ``-vv``
# on the user-facing CLI; the file handler that writes
# ``<run_dir>/eval.log`` is independent and stays at ``DEBUG`` so the
# transcript always carries full forensics regardless of terminal noise.
LOG_LEVEL: Final[str] = "BELT_LOG_LEVEL"

# ── LLM provider credentials and URLs ──────────────────────────────────────

OPENAI_API_KEY: Final[str] = "BELT_OPENAI_API_KEY"
OPENAI_BASE_URL: Final[str] = "BELT_OPENAI_BASE_URL"

ANTHROPIC_API_KEY: Final[str] = "BELT_ANTHROPIC_API_KEY"
ANTHROPIC_BASE_URL: Final[str] = "BELT_ANTHROPIC_BASE_URL"

AZURE_OPENAI_ENDPOINT: Final[str] = "BELT_AZURE_OPENAI_ENDPOINT"
AZURE_OPENAI_API_KEY: Final[str] = "BELT_AZURE_OPENAI_API_KEY"
AZURE_OPENAI_API_VERSION: Final[str] = "BELT_AZURE_OPENAI_API_VERSION"
AZURE_CLIENT_ID: Final[str] = "BELT_AZURE_CLIENT_ID"
AZURE_CLIENT_SECRET: Final[str] = "BELT_AZURE_CLIENT_SECRET"
AZURE_TENANT_ID: Final[str] = "BELT_AZURE_TENANT_ID"

OLLAMA_BASE_URL: Final[str] = "BELT_OLLAMA_BASE_URL"

# ── Disk budgets and stream caps (integer bytes) ───────────────────────────
#
# Three caps, named for *what* they cap:
#
#   * SUBPROCESS_STDOUT_MAX_BYTES   - total bytes captured from a single
#     agent subprocess stdout
#   * SUBPROCESS_STDOUT_LINE_MAX    - max bytes per line in the same
#     stdout (lines exceeding this are split for the runner's parser)
#   * TURN_NDJSON_MAX_BYTES         - cap on the per-turn live NDJSON
#     stream artifact written to the run directory.

CACHE_MAX_BYTES: Final[str] = "BELT_CACHE_MAX_BYTES"
SUBPROCESS_STDOUT_LINE_MAX: Final[str] = "BELT_SUBPROCESS_STDOUT_LINE_MAX"
SUBPROCESS_STDOUT_MAX_BYTES: Final[str] = "BELT_SUBPROCESS_STDOUT_MAX_BYTES"
TURN_NDJSON_MAX_BYTES: Final[str] = "BELT_TURN_NDJSON_MAX_BYTES"

# ── Public paths and overrides ─────────────────────────────────────────────

OUTCOMES_DIR: Final[str] = "BELT_OUTCOMES_DIR"

# ── LLM auto-config (used by ``belt.config``) ─────────────────────────

LLM_MODEL: Final[str] = "BELT_LLM_MODEL"
LLM_PROVIDER: Final[str] = "BELT_LLM_PROVIDER"
# Per-probe timeout for ``BaseJudgeBackend.preflight_model`` (seconds).
# Defaults to 10s when unset / invalid; see ``_preflight_timeout`` in
# ``belt.scorer.llm.backend``.
LLM_PREFLIGHT_TIMEOUT: Final[str] = "BELT_LLM_PREFLIGHT_TIMEOUT"

PRICING_FILE: Final[str] = "BELT_PRICING_FILE"


# ── Documented public allow-list ───────────────────────────────────────────
# Used by ``_redact.safe_environ`` to decide which BELT_* names
# may appear verbatim in ``run_meta.json``. Every entry is a non-secret
# feature toggle or non-secret URL. API keys and tokens are NOT here -
# the secret-name regex in ``_redact`` redacts them to ``"<set>"``.
#
# Sorted alphabetically so drift (additions, removals) is visible in PR
# diffs and enforceable by ``tests/test_envvars.py``.
PUBLIC_ALLOW: Final[frozenset[str]] = frozenset(
    {
        ALLOW_ARBITRARY_AGENT,
        ALLOW_ARBITRARY_EXPORTER,
        ALLOW_ARBITRARY_SCORER,
        ALLOW_EXTERNAL_WORKING_DIR,
        ALLOW_FULL_ENV,
        ALLOW_INPLACE,
        ALLOW_INSECURE_BASE_URL,
        ALLOW_VERIFY_EXEC,
        ANTHROPIC_BASE_URL,
        DEBUG,
        LOG_LEVEL,
        NO_DOTENV,
        NO_UMASK,
        OPENAI_BASE_URL,
        PRICING_FILE,
        SANDBOX_PROVIDER,
        SILENCE_CUSTOM_BASE_URL_WARNING,
    }
)

# ── All known names ────────────────────────────────────────────────────────
# Used by tests to detect rogue inline string literals: any ``BELT_*``
# literal in ``src/`` that is not in this set is a centralization regression.
#
# Sorted alphabetically by constant identifier (and verified by
# ``tests/test_envvars.py``) so a new contributor cannot quietly insert a
# name out of order.
ALL_NAMES: Final[frozenset[str]] = frozenset(
    {
        ALLOW_ARBITRARY_AGENT,
        ALLOW_ARBITRARY_EXPORTER,
        ALLOW_ARBITRARY_SCORER,
        ALLOW_EXTERNAL_WORKING_DIR,
        ALLOW_FULL_ENV,
        ALLOW_INPLACE,
        ALLOW_INSECURE_BASE_URL,
        ALLOW_VERIFY_EXEC,
        ANTHROPIC_API_KEY,
        ANTHROPIC_BASE_URL,
        AZURE_CLIENT_ID,
        AZURE_CLIENT_SECRET,
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_API_VERSION,
        AZURE_OPENAI_ENDPOINT,
        AZURE_TENANT_ID,
        CACHE_MAX_BYTES,
        DEBUG,
        LLM_MODEL,
        LLM_PREFLIGHT_TIMEOUT,
        LLM_PROVIDER,
        LOG_LEVEL,
        NO_DOTENV,
        NO_UMASK,
        OLLAMA_BASE_URL,
        OPENAI_API_KEY,
        OPENAI_BASE_URL,
        OUTCOMES_DIR,
        PRICING_FILE,
        SANDBOX_PROVIDER,
        SILENCE_CUSTOM_BASE_URL_WARNING,
        SUBPROCESS_STDOUT_LINE_MAX,
        SUBPROCESS_STDOUT_MAX_BYTES,
        TURN_NDJSON_MAX_BYTES,
    }
)


# ── Typed accessors ────────────────────────────────────────────────────────


def is_truthy(name: str, default: bool = False) -> bool:
    """Return True iff ``os.environ[name]`` is one of ``"1"``, ``"true"``, ``"yes"``.

    Mirrors the idiom that was repeated across nine modules. Use this
    helper instead of inlining the comparison so that the truthy-string
    set has a single owner.
    """
    raw = os.environ.get(name, "")
    if not raw:
        return default
    return raw in ("1", "true", "yes")


def get_int(name: str, default: int) -> int:
    """Parse ``os.environ[name]`` as int; returns ``default`` on missing or invalid."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_str(name: str, default: str = "") -> str:
    """Return ``os.environ[name]`` or ``default`` if unset."""
    return os.environ.get(name, default)


# ── Cross-process forwarding helpers ──────────────────────────────────────


# Mapping from the argparse destination on the user-facing CLI command to
# the canonical env var name read by downstream phases. Centralised here
# so a new ``--allow-*`` flag is wired up in one place: add the flag to
# the parser, add the constant above, add the entry here. Sorted to
# match the order of the constants block at the top of this file so
# drift is visible in PR diffs.
_SECURITY_TOGGLE_FLAG_TO_ENV: Final[tuple[tuple[str, str], ...]] = (
    ("allow_full_env", ALLOW_FULL_ENV),
    ("allow_arbitrary_agent", ALLOW_ARBITRARY_AGENT),
    ("allow_arbitrary_exporter", ALLOW_ARBITRARY_EXPORTER),
    ("allow_arbitrary_scorer", ALLOW_ARBITRARY_SCORER),
    ("allow_insecure_base_url", ALLOW_INSECURE_BASE_URL),
    ("allow_verify_exec", ALLOW_VERIFY_EXEC),
)


def forward_security_toggles(args: object) -> tuple[str, ...]:
    """Forward parsed ``--allow-*`` flags from ``args`` into the process env.

    The ``eval`` / ``run`` / ``score`` commands all accept the same set
    of safety opt-ins. When set on the command line, each must reach the
    downstream phases (which may run in subprocesses or read the env
    later in the same process) via the canonical ``BELT_ALLOW_*``
    env var rather than threading a flag through every internal API.

    The CLI flag wins: when the user passes ``--allow-X``, the env
    var is set to ``"1"`` even if the shell had it set to ``"0"`` (a
    later truthy check would otherwise silently ignore the flag). When
    the flag is absent, the env var is left untouched - so a shell
    export from CI still applies.

    Returns the tuple of env var names that were forwarded (useful for
    tests asserting a specific subset was applied; production callers
    ignore the return value).
    """
    forwarded: list[str] = []
    for attr, env_name in _SECURITY_TOGGLE_FLAG_TO_ENV:
        if getattr(args, attr, False):
            os.environ[env_name] = "1"
            forwarded.append(env_name)
    return tuple(forwarded)
