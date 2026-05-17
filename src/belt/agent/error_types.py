# (c) JFrog Ltd. (2026)

"""Agent-layer classifier and remediation hints for runtime failures.

The *taxonomy* (the set of stable token values - ``AUTHENTICATION_FAILED``
etc.) lives in :mod:`belt.entities` because it is part of the
:attr:`TurnOutput.error_type` cross-phase contract. This module owns the
*agent-layer behaviour* on top of that taxonomy:

- :func:`classify_error` - turn arbitrary CLI output text into a token.
  Adapters call this in ``fetch_results`` as a fallback when their
  structured event stream did not carry a typed error.
- :func:`remediation_for` - per-(error_type, agent_name) one-line hint
  surfaced by the aggregator headline and the benchmark card.

The constants are re-exported so existing
``from belt.agent.error_types import AUTHENTICATION_FAILED, ...``
imports keep working.

The pattern lists are deliberately conservative: we'd rather classify as
``UNKNOWN`` than mislabel a real failure. ``classify_error`` is
case-insensitive substring matching - cheap, regex-free, and resilient
to minor wording changes across CLI versions.
"""

from __future__ import annotations

from belt.entities import (  # noqa: F401  (re-exported)
    AUTHENTICATION_FAILED,
    ENVIRONMENTAL_ERROR_TYPES,
    ERROR_TYPES,
    MODEL_UNAVAILABLE,
    RATE_LIMITED,
    REFUSED,
    TASK_ERROR_TYPES,
    TIMEOUT,
    UNKNOWN,
)

# Patterns are case-insensitive substring matches. Order matters: first
# match wins. Each tuple is scanned in turn against the lowercased input.
_AUTH_PATTERNS: tuple[str, ...] = (
    "not logged in",
    "please run /login",
    "please run `claude login`",
    "failed to authenticate",
    "invalid authentication credentials",
    "authentication_error",
    "authentication failed",
    "authentication required",
    "401 unauthorized",
    "http 401",
    " 401 ",
    "unauthorized: invalid",
    "expired token",
    "token expired",
    "has expired",
    "please log in",
    "please login",
    "not authenticated",
)

_RATE_LIMIT_PATTERNS: tuple[str, ...] = (
    "rate_limit",
    "rate limit",
    "rate-limit",
    "rate limited",
    " 429 ",
    "http 429",
    "too many requests",
    "quota exceeded",
    "insufficient quota",
)

_TIMEOUT_PATTERNS: tuple[str, ...] = (
    "request timed out",
    "deadline exceeded",
    "operation timed out",
    "context deadline exceeded",
    "504 gateway timeout",
)

# Distinct from auth patterns: the credentials worked, the requested
# model did not. Two failure shapes are common:
#   1. OpenAI-style ``model_not_found`` (provider does not know the
#      model name at all)
#   2. Project-entitlement gaps phrased as ``does not have access to
#      model X`` (OpenAI org/project keys without entitlement)
# Conflating either of these with ``authentication_failed`` would steer
# users at re-login instead of at a model swap.
_MODEL_UNAVAILABLE_PATTERNS: tuple[str, ...] = (
    "model_not_found",
    "model not found",
    "does not have access to model",
    "no access to model",
    "model is not available",
    "the model `",
    "unknown model",
    "unsupported model",
)

_REFUSED_PATTERNS: tuple[str, ...] = (
    "i can't help with that",
    "i cannot help with",
    "i won't help with",
    "i'm not able to help",
    "model refused",
)


def classify_error(text: str | None) -> str | None:
    """Classify an agent failure from arbitrary output text.

    Scans ``text`` for known failure patterns (auth, rate-limit, timeout,
    refusal). Returns one of the :data:`ERROR_TYPES` tokens
    (excluding :data:`UNKNOWN`), or ``None`` if no pattern matched.

    Callers should treat ``None`` as "no specific signal in this text"
    and fall back to :data:`UNKNOWN` only when ``has_error`` is true and
    every available signal returned ``None`` (use
    :func:`normalize_error_type` for that combined logic).
    """
    if not text:
        return None
    lowered = text.lower()
    # Model-availability patterns are scanned before auth so that
    # ``model_not_found: invalid authentication`` (a real OpenAI shape)
    # routes to the entitlement bucket, not the credentials bucket.
    for pat in _MODEL_UNAVAILABLE_PATTERNS:
        if pat in lowered:
            return MODEL_UNAVAILABLE
    for pat in _AUTH_PATTERNS:
        if pat in lowered:
            return AUTHENTICATION_FAILED
    for pat in _RATE_LIMIT_PATTERNS:
        if pat in lowered:
            return RATE_LIMITED
    for pat in _TIMEOUT_PATTERNS:
        if pat in lowered:
            return TIMEOUT
    for pat in _REFUSED_PATTERNS:
        if pat in lowered:
            return REFUSED
    return None


def normalize_error_type(raw_error_type: str | None, *texts: str | None) -> str | None:
    """Project a vendor-specific error label onto the canonical token set.

    Adapters often pull a structured error label out of their CLI's
    NDJSON (Goose's ``"AuthError"``, Gemini's ``"auth_error"``,
    OpenCode's ``"NotLoggedIn"``, ...). These vendor strings are not
    portable: aggregator code keys off the canonical tokens in
    :data:`ERROR_TYPES`, and a downstream consumer that sees
    ``"AuthError"`` cannot tell which framework category it belongs to.

    Resolution order:

    1. If ``raw_error_type`` is already a canonical token, keep it.
    2. Otherwise, scan ``raw_error_type`` and every ``*texts`` argument
       (typically ``reply_text`` and ``raw_output``) for a known
       failure pattern via :func:`classify_error` and return the first
       hit.
    3. Return ``None`` if neither path produced a token. The caller is
       responsible for the final ``or UNKNOWN`` fallback when
       ``has_error`` is true.

    Adapters call this in their ``fetch_results`` after deciding
    ``has_error`` so vendor labels never leak into the cross-phase
    contract.
    """
    if raw_error_type and raw_error_type in ERROR_TYPES:
        return raw_error_type
    for candidate in (raw_error_type, *texts):
        token = classify_error(candidate)
        if token:
            return token
    return None


# Per-(error_type, agent_name) remediation hint surfaced in the
# aggregator headline. ``"*"`` is the agent-agnostic fallback. Hints are
# single-line, imperative, and end with a period.
_REMEDIATION: dict[str, dict[str, str]] = {
    AUTHENTICATION_FAILED: {
        "claude-code": "Re-authenticate the Claude Code CLI: run `claude login`.",
        "codex": "Re-authenticate the Codex CLI: run `codex login`.",
        "gemini": "Re-authenticate the Gemini CLI: run `gemini auth login`.",
        "copilot": "Re-authenticate GitHub Copilot CLI: run `gh auth login` or `copilot auth login`.",
        "goose": "Re-authenticate Goose: check `~/.config/goose/config.yaml`.",
        "opencode": "Re-authenticate OpenCode CLI.",
        "cursor": "Re-authenticate Cursor CLI: run `cursor login`.",
        "*": "Re-authenticate the agent CLI and retry.",
    },
    RATE_LIMITED: {
        "*": "Wait for the rate-limit window to reset, or switch to a different API key.",
    },
    TIMEOUT: {
        "*": "Increase the per-turn timeout or reduce scenario complexity.",
    },
    MODEL_UNAVAILABLE: {
        # The two production failure modes the user can fix:
        # - typo in the model name -> switch to a known one
        # - org/project key without entitlement -> request access or
        #   switch keys/projects
        # We surface both because the agent CLI's wrapping text is too
        # vendor-specific to reliably distinguish.
        "*": "Switch to a model your project key is entitled to, or request access. Use ``belt agent info <agent>`` to see what the adapter expects.",
    },
    REFUSED: {
        "*": "Model refused. Adjust the scenario assistant persona or the user prompt.",
    },
    UNKNOWN: {
        "*": "Open the per-turn output JSON for the underlying error message.",
    },
}


def remediation_for(error_type: str, agent_name: str | None = None) -> str | None:
    """Return a one-line remediation hint, or ``None`` if no hint is registered.

    Per-agent hints take precedence over the agent-agnostic ``"*"`` entry.
    """
    if error_type not in _REMEDIATION:
        return None
    table = _REMEDIATION[error_type]
    if agent_name and agent_name in table:
        return table[agent_name]
    return table.get("*")
