# (c) JFrog Ltd. (2026)

"""User-facing hint formatter for fatal LLM-judge HTTP errors.

A single formatter so the message a user sees is identical whether the
failure surfaces from preflight (``scorer/llm/preflight.py``) or from a
runtime ``_call_api`` retry (``scorer/llm/scorer.py``). Without this,
the two raise sites would drift on every provider-error-shape change
and the user could see a 404-style "model name wrong" hint for a 403
``model_not_found`` (project-scoped key without access).

The mapping intentionally branches on both the HTTP status *and* the
provider-specific ``error.code`` token because the same HTTP status
carries different root causes across providers - notably OpenAI uses
**403** + ``model_not_found`` for a project-scoped key without access
to the model and **404** + ``model_not_found`` for a typo'd model.
"""

from __future__ import annotations

import json
import re
from typing import Final

# OpenAI / OpenAI-compatible servers return ``{"error": {"code": ...}}``
# in the JSON body. Anthropic returns ``{"error": {"type": ...}}``. Both
# may also be wrapped in upstream-gateway HTML if a proxy sits in front,
# which is why we fall back to a regex over the raw body.
_OPENAI_ERROR_CODE_RE: Final = re.compile(r'"code"\s*:\s*"([^"]+)"')
_ANTHROPIC_ERROR_TYPE_RE: Final = re.compile(r'"type"\s*:\s*"([^"]+)"')

# Provider-specific error tokens we recognise. Other tokens fall through
# to the generic per-status hint - we never claim a root cause we are
# not sure of.
_MODEL_NOT_FOUND_TOKENS: Final = frozenset(
    {
        "model_not_found",  # OpenAI / OpenAI-compatible
        "not_found_error",  # Anthropic (their "not_found" type)
        "DeploymentNotFound",  # Azure (capitalised, sigh)
    }
)
_INVALID_KEY_TOKENS: Final = frozenset(
    {
        "invalid_api_key",  # OpenAI
        "authentication_error",  # Anthropic
        "Unauthorized",  # Azure
    }
)


def _extract_error_code(body: str) -> str:
    """Best-effort extraction of the provider-specific error token.

    Tries JSON parse first (handles nested error envelopes correctly,
    e.g. OpenAI's ``{"error": {"code": ..., "message": ...}}``), then
    falls back to a permissive regex so a gateway-mangled or partial
    body still yields a usable token. Returns ``""`` if nothing matches
    - the caller must handle the empty case as "no provider hint".
    """
    if not body:
        return ""
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        err = parsed.get("error")
        if isinstance(err, dict):
            for key in ("code", "type"):
                value = err.get(key)
                if isinstance(value, str) and value:
                    return value
    for pattern in (_OPENAI_ERROR_CODE_RE, _ANTHROPIC_ERROR_TYPE_RE):
        match = pattern.search(body)
        if match:
            return match.group(1)
    return ""


def format_judge_error_hint(status: int, body: str) -> str:
    """Render the one-line hint for a fatal judge HTTP error.

    Returns a string starting with ``"Hint: "`` so callers can append it
    to the upstream error body without further formatting. The branching
    is on (status, error_code) because the same HTTP code means different
    things across providers - see module docstring.
    """
    code = _extract_error_code(body)

    if status == 401 or (code and code in _INVALID_KEY_TOKENS):
        return (
            "Hint: 401 / invalid_api_key - check BELT_OPENAI_API_KEY "
            "(or BELT_ANTHROPIC_API_KEY / BELT_AZURE_OPENAI_API_KEY for "
            "your provider). The credential is missing, malformed, or revoked."
        )

    if status == 403:
        if code in _MODEL_NOT_FOUND_TOKENS:
            return (
                "Hint: 403 + model_not_found - your project / API key is "
                "configured but not authorised to call this model. Check "
                "the provider dashboard's project / organisation / "
                "deployment permissions, or pick a model your key can call."
            )
        return (
            "Hint: 403 - the provider accepted the credentials but rejected "
            "this specific request (project scope, region restriction, or "
            "content policy). Inspect the upstream error body above."
        )

    if status == 404:
        if code in _MODEL_NOT_FOUND_TOKENS:
            return (
                "Hint: 404 + model_not_found - the model name is wrong. "
                "Check --scorer-arg model=..., belt.yaml -> llm.model, or "
                "the BELT_LLM_MODEL env var. Common typos: missing provider "
                "prefix (use 'openai/gpt-4o-mini', not 'gpt-4o-mini'), "
                "stale model name, or a deployment alias that does not exist."
            )
        return (
            "Hint: 404 - endpoint or deployment not found. For Azure this "
            "usually means the deployment name (passed as --scorer-arg "
            "model=...) does not exist in your BELT_AZURE_OPENAI_ENDPOINT. "
            "For OpenAI-compatible servers, check BELT_OPENAI_BASE_URL."
        )

    if status == 429:
        return (
            "Hint: 429 - rate limited or quota exceeded. Wait, lower "
            "--workers / --trials, or pick a model with higher quota."
        )

    return (
        f"Hint: unexpected fatal HTTP {status}. Inspect the upstream "
        "error body above; this is not a status the judge preflight "
        "knows how to map."
    )
