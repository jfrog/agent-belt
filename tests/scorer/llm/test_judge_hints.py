# (c) JFrog Ltd. (2026)

"""Tests for :func:`belt.scorer.llm.judge_hints.format_judge_error_hint`.

The hint formatter is the single source of truth for the user-facing
"why did the judge fail" message across the preflight raise site
(:mod:`belt.scorer.llm.preflight`) and the runtime 4xx raise site
(:mod:`belt.scorer.llm.scorer` ``_call_api``). Without these tests, the
two paths could drift on the same input - so a user could see a
"model name wrong" hint for a 403 ``model_not_found`` (which actually
means a project-scoped key without access).

The tests pin the branching on (HTTP status, provider error code) pairs
because the same status carries different root causes across providers
- notably OpenAI uses **403** + ``model_not_found`` for a project-
scoped key without access and **404** + ``model_not_found`` for a
typo'd model. A test that only checked status would let that bug back in.
"""

from __future__ import annotations

from belt.scorer.llm.judge_hints import format_judge_error_hint


class TestStatusBranching:
    """Each HTTP status the preflight policy distinguishes gets its own hint."""

    def test_401_mentions_api_key(self) -> None:
        hint = format_judge_error_hint(401, "")
        assert "Hint:" in hint
        assert "401" in hint
        assert "BELT_OPENAI_API_KEY" in hint

    def test_403_without_code_is_distinct_from_404(self) -> None:
        # Bare 403 (no provider error code in body) gets a 403-specific
        # hint, NOT the 404 "model name wrong" hint. Treating these
        # two as the same case is the precise mistake that misled
        # users about project-scope failures.
        hint = format_judge_error_hint(403, "")
        assert "403" in hint
        assert "404" not in hint
        assert "model name" not in hint.lower() or "model_not_found" in hint

    def test_404_without_code_mentions_endpoint(self) -> None:
        hint = format_judge_error_hint(404, "")
        assert "404" in hint
        # Generic 404 (no provider code) should NOT claim model typo.
        assert "model_not_found" not in hint

    def test_429_mentions_rate_limit(self) -> None:
        hint = format_judge_error_hint(429, "")
        assert "429" in hint
        assert "rate limited" in hint.lower() or "quota" in hint.lower()

    def test_unknown_status_falls_back_explicitly(self) -> None:
        hint = format_judge_error_hint(418, "")
        assert "418" in hint
        assert "unexpected" in hint.lower()


class TestProviderCodeBranching:
    """When the body carries a provider error code, the hint becomes more specific."""

    def test_openai_403_model_not_found(self) -> None:
        # Project-scoped API key that lacks access to the requested
        # model: status=403, code=model_not_found. Must NOT hint at
        # a typo (that's the 404 + same code case).
        body = (
            '{"error":{"message":"Project `proj_x` does not have access to model `gpt-4o-mini`",'
            '"type":"invalid_request_error","code":"model_not_found"}}'
        )
        hint = format_judge_error_hint(403, body)
        assert "403" in hint
        assert "model_not_found" in hint
        assert "project" in hint.lower() or "dashboard" in hint.lower() or "authorised" in hint.lower()

    def test_openai_404_model_not_found(self) -> None:
        # The typo case: same code, different status, different hint.
        body = '{"error":{"message":"unknown model","type":"invalid_request_error","code":"model_not_found"}}'
        hint = format_judge_error_hint(404, body)
        assert "404" in hint
        assert "model_not_found" in hint
        assert "typo" in hint.lower() or "wrong" in hint.lower()

    def test_anthropic_404_not_found_error(self) -> None:
        body = '{"type":"error","error":{"type":"not_found_error","message":"model not found"}}'
        hint = format_judge_error_hint(404, body)
        # Anthropic surfaces ``type`` not ``code``; the formatter falls
        # through to the JSON-walk path and pulls ``not_found_error``.
        assert "model_not_found" in hint or "model" in hint.lower()

    def test_azure_404_deployment_not_found(self) -> None:
        body = '{"error":{"code":"DeploymentNotFound","message":"deployment not found"}}'
        hint = format_judge_error_hint(404, body)
        assert "DeploymentNotFound" in hint or "model_not_found" in hint or "typo" in hint.lower()

    def test_invalid_api_key_token_routes_to_401_hint(self) -> None:
        # Some providers return ``invalid_api_key`` with a non-401
        # status; the token-based branch still routes to the API-key hint.
        body = '{"error":{"code":"invalid_api_key","message":"bad key"}}'
        hint = format_judge_error_hint(403, body)
        assert "BELT_OPENAI_API_KEY" in hint or "BELT_ANTHROPIC_API_KEY" in hint


class TestBodyParsing:
    """The formatter must tolerate non-JSON, partial, and gateway-mangled bodies."""

    def test_empty_body_falls_back_to_status_only(self) -> None:
        hint = format_judge_error_hint(403, "")
        assert "403" in hint
        # Empty body, no provider code → generic 403 hint.

    def test_html_gateway_body_does_not_crash(self) -> None:
        body = "<html><body>502 Bad Gateway</body></html>"
        hint = format_judge_error_hint(404, body)
        assert "404" in hint

    def test_truncated_json_falls_back_to_regex(self) -> None:
        # Body cut mid-stream; JSON parse fails, regex still finds the code.
        body = '{"error":{"code":"model_not_found","message":"truncated'
        hint = format_judge_error_hint(403, body)
        assert "model_not_found" in hint
