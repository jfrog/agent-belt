# (c) JFrog Ltd. (2026)

"""Unit tests for the agent runtime-error taxonomy.

Per :mod:`belt.agent.error_types`, the framework treats these
tokens as a stable contract (renaming is breaking, adding is additive).
The tests below pin down that contract and the classifier's behaviour
on the failure shapes the project has actually seen in CI logs.
"""

from __future__ import annotations

import pytest

from belt.agent.error_types import (
    AUTHENTICATION_FAILED,
    ERROR_TYPES,
    MODEL_UNAVAILABLE,
    RATE_LIMITED,
    REFUSED,
    TIMEOUT,
    UNKNOWN,
    classify_error,
    remediation_for,
)


class TestStableTokens:
    """The token set is a public schema; pin it down."""

    def test_token_values(self) -> None:
        # Token values are persisted to ``turn_*_output.json`` and
        # ``benchmark-card.json``; a value change is breaking.
        assert AUTHENTICATION_FAILED == "authentication_failed"
        assert RATE_LIMITED == "rate_limited"
        assert TIMEOUT == "timeout"
        assert MODEL_UNAVAILABLE == "model_unavailable"
        assert REFUSED == "refused"
        assert UNKNOWN == "unknown"

    def test_error_types_membership(self) -> None:
        assert AUTHENTICATION_FAILED in ERROR_TYPES
        assert RATE_LIMITED in ERROR_TYPES
        assert TIMEOUT in ERROR_TYPES
        assert MODEL_UNAVAILABLE in ERROR_TYPES
        assert REFUSED in ERROR_TYPES
        assert UNKNOWN in ERROR_TYPES
        assert len(ERROR_TYPES) == 6


class TestClassifyError:
    """Classifier returns the documented token or ``None``."""

    @pytest.mark.parametrize(
        "text",
        [
            # The exact text Claude Code emits when the user is logged out
            # (the case that motivated the feature).
            "Not logged in · Please run /login",
            "Please run `claude login` to authenticate",
            "API Error: 401 unauthorized",
            '{"error":{"type":"authentication_error","message":"..."}}',
            "HTTP 401: Unauthorized",
            "Failed to authenticate with the API",
            "Your token has expired",
            "Please log in to continue",
        ],
    )
    def test_classifies_authentication(self, text: str) -> None:
        assert classify_error(text) == AUTHENTICATION_FAILED

    @pytest.mark.parametrize(
        "text",
        [
            "rate_limit_error: too many requests",
            "Error 429: Too Many Requests",
            "Quota exceeded for this minute",
        ],
    )
    def test_classifies_rate_limit(self, text: str) -> None:
        assert classify_error(text) == RATE_LIMITED

    @pytest.mark.parametrize(
        "text",
        [
            "Operation timed out after 60 seconds",
            "context deadline exceeded",
            "504 Gateway Timeout",
        ],
    )
    def test_classifies_timeout(self, text: str) -> None:
        assert classify_error(text) == TIMEOUT

    @pytest.mark.parametrize(
        "text",
        [
            # OpenAI's literal token + the surrounding message phrasing.
            "model_not_found",
            "The model `gpt-9000` does not exist or you do not have access to it.",
            "Project proj_abc123 does not have access to model gpt-5.",
            "no access to model gpt-5",
            "Unknown model: gpt-9000",
            "The model is not available in your region",
            "unsupported model: claude-9",
        ],
    )
    def test_classifies_model_unavailable(self, text: str) -> None:
        # Verifies the F7 fix: the agent CLI surfaces project-entitlement
        # gaps and model-name typos as ``model_unavailable`` rather than
        # ``unknown`` (which would steer the user at re-running the
        # scenario instead of fixing the model selection).
        assert classify_error(text) == MODEL_UNAVAILABLE

    def test_model_unavailable_preempts_auth_when_both_present(self) -> None:
        # Real OpenAI 404 bodies bundle ``invalid_request`` style auth-ish
        # language with ``model_not_found``; we must route to the
        # entitlement bucket so the remediation hint points at the
        # model, not the credential.
        text = "model_not_found: 401 unauthorized - invalid authentication"
        assert classify_error(text) == MODEL_UNAVAILABLE

    @pytest.mark.parametrize(
        "text",
        [
            "I can't help with that.",
            "Sorry, I cannot help with this request.",
            "Model refused to respond",
        ],
    )
    def test_classifies_refusal(self, text: str) -> None:
        assert classify_error(text) == REFUSED

    @pytest.mark.parametrize("text", ["", None])
    def test_empty_returns_none(self, text: str | None) -> None:
        assert classify_error(text) is None

    def test_unknown_returns_none(self) -> None:
        # Generic prose that doesn't match any pattern - caller is
        # responsible for falling back to UNKNOWN when has_error=true.
        assert classify_error("hello world, the agent finished") is None

    def test_first_match_wins(self) -> None:
        # Auth pattern appears first in the scan order; even if a
        # rate-limit string is also present, we surface auth.
        text = "401 unauthorized - also rate_limit_error"
        assert classify_error(text) == AUTHENTICATION_FAILED

    def test_case_insensitive(self) -> None:
        assert classify_error("NOT LOGGED IN") == AUTHENTICATION_FAILED


class TestRemediationFor:
    """Remediation hints exist for every documented agent + the agnostic case."""

    @pytest.mark.parametrize(
        "agent",
        ["claude-code", "codex", "gemini", "copilot", "goose", "opencode", "cursor"],
    )
    def test_per_agent_auth_hint(self, agent: str) -> None:
        hint = remediation_for(AUTHENTICATION_FAILED, agent)
        assert hint is not None and len(hint) > 0

    def test_unknown_agent_falls_back_to_wildcard(self) -> None:
        hint = remediation_for(AUTHENTICATION_FAILED, "some-unregistered-agent")
        assert hint is not None
        assert "agent CLI" in hint.lower() or "re-authenticate" in hint.lower()

    def test_no_hint_for_invalid_type(self) -> None:
        assert remediation_for("not-a-real-error", "claude-code") is None

    def test_rate_limit_has_wildcard_only(self) -> None:
        hint = remediation_for(RATE_LIMITED, "claude-code")
        assert hint is not None
        assert "rate-limit" in hint.lower() or "api key" in hint.lower()

    def test_model_unavailable_remediation_points_at_model(self) -> None:
        hint = remediation_for(MODEL_UNAVAILABLE, "codex")
        # The hint must steer the user toward fixing the model
        # selection (entitlement / typo), not toward re-authenticating.
        assert hint is not None
        lowered = hint.lower()
        assert "model" in lowered
        assert "re-authenticate" not in lowered
