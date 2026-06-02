# (c) JFrog Ltd. (2026)

"""Unit tests for scorer.llm.backend - BaseJudgeBackend implementations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from belt.entities import JudgeConfig
from belt.scorer.llm.backend import (
    AnthropicBackend,
    AzureBackend,
    OllamaBackend,
    OpenAIBackend,
    parse_model_spec,
    resolve_backend,
)

# ── parse_model_spec ──


class TestParseModelSpec:
    def test_openai_prefix(self):
        provider, model = parse_model_spec("openai/gpt-4.1")
        assert provider == "openai"
        assert model == "gpt-4.1"

    def test_anthropic_prefix(self):
        provider, model = parse_model_spec("anthropic/claude-sonnet-4-5")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-5"

    def test_azure_prefix(self):
        provider, model = parse_model_spec("azure/my-deployment")
        assert provider == "azure"
        assert model == "my-deployment"

    def test_ollama_prefix_maps_to_ollama(self):
        provider, model = parse_model_spec("ollama/llama3")
        assert provider == "ollama"
        assert model == "llama3"

    def test_no_prefix(self):
        provider, model = parse_model_spec("gpt-4.1")
        assert provider is None
        assert model == "gpt-4.1"

    def test_unknown_prefix_returns_none(self):
        provider, model = parse_model_spec("xyzprovider/some-model")
        assert provider is None
        assert model == "xyzprovider/some-model"

    def test_empty_string(self):
        provider, model = parse_model_spec("")
        assert provider is None
        assert model == ""


# ── OpenAIBackend ──


class TestOpenAIBackend:
    def test_available_with_key(self):
        with patch.dict("os.environ", {"BELT_OPENAI_API_KEY": "sk-test"}):
            assert OpenAIBackend().is_available() is True

    def test_not_available_without_key(self):
        with patch.dict("os.environ", {}, clear=True):
            assert OpenAIBackend().is_available() is False

    def test_available_with_base_url_no_key(self):
        with patch.dict("os.environ", {"BELT_OPENAI_BASE_URL": "http://localhost:8000/v1"}, clear=True):
            assert OpenAIBackend().is_available() is True

    def test_build_request_default_base(self):
        with patch.dict("os.environ", {"BELT_OPENAI_API_KEY": "sk-test"}, clear=True):
            config = JudgeConfig(model="gpt-4.1")
            url, headers, body = OpenAIBackend().build_request(
                config, [{"role": "system", "content": "hi"}], {"type": "object"}
            )
            assert url == "https://api.openai.com/v1/chat/completions"
            assert headers["Authorization"] == "Bearer sk-test"
            assert body["model"] == "gpt-4.1"

    def test_build_request_custom_base_url(self):
        env = {
            "BELT_OPENAI_API_KEY": "sk-test",
            "BELT_OPENAI_BASE_URL": "http://localhost:11434/v1",
        }
        with patch.dict("os.environ", env):
            url, _, _ = OpenAIBackend().build_request(
                JudgeConfig(model="openai/gpt-5.4-mini"),
                [{"role": "system", "content": "hi"}],
                {},
            )
            assert url == "http://localhost:11434/v1/chat/completions"

    def test_provider_name(self):
        assert OpenAIBackend().provider_name() == "OpenAI"


# ── AzureBackend ──


class TestAzureBackend:
    def test_available_with_key(self):
        env = {"BELT_AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com", "BELT_AZURE_OPENAI_API_KEY": "key"}
        with patch.dict("os.environ", env):
            assert AzureBackend().is_available() is True

    def test_available_with_service_principal(self):
        env = {
            "BELT_AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com",
            "BELT_AZURE_CLIENT_ID": "id",
            "BELT_AZURE_CLIENT_SECRET": "secret",
            "BELT_AZURE_TENANT_ID": "tenant",
        }
        with patch.dict("os.environ", env):
            assert AzureBackend().is_available() is True

    def test_not_available_without_endpoint(self):
        with patch.dict("os.environ", {"BELT_AZURE_OPENAI_API_KEY": "key"}, clear=True):
            assert AzureBackend().is_available() is False

    def test_build_request_api_key(self):
        env = {
            "BELT_AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com",
            "BELT_AZURE_OPENAI_API_KEY": "my-key",
        }
        with patch.dict("os.environ", env):
            config = JudgeConfig(model="gpt-4.1-deploy")
            url, headers, body = AzureBackend().build_request(
                config, [{"role": "system", "content": "hi"}], {"type": "object"}
            )
            assert "gpt-4.1-deploy" in url
            assert "api-version=" in url
            assert headers["api-key"] == "my-key"

    def test_build_request_prefixed_gpt5_deployment_uses_max_completion_tokens(self):
        """Team / org-prefixed Azure aliases (``myteam-gpt-5.2``) must
        flip to ``max_completion_tokens`` - anchored matching would silently
        send ``max_tokens`` and the deployment would reject the request with
        HTTP 400 ``unsupported_parameter``.
        """
        env = {
            "BELT_AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com",
            "BELT_AZURE_OPENAI_API_KEY": "my-key",
        }
        with patch.dict("os.environ", env):
            _, _, body = AzureBackend().build_request(
                JudgeConfig(model="myteam-gpt-5.2", max_tokens=1024),
                [{"role": "user", "content": "hi"}],
                {"type": "object"},
            )
            assert "max_completion_tokens" in body
            assert body["max_completion_tokens"] == 1024
            assert "max_tokens" not in body

    def test_build_request_legacy_deployment_keeps_max_tokens(self):
        """Regression guard: Azure aliases that don't embed a new-family
        family token (``my-gpt4o``, ``production-judge``) keep the old
        ``max_tokens`` field - 3rd-party OpenAI-compatible servers behind
        Azure endpoints may not accept ``max_completion_tokens`` yet.
        """
        env = {
            "BELT_AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com",
            "BELT_AZURE_OPENAI_API_KEY": "my-key",
        }
        with patch.dict("os.environ", env):
            for legacy in ("my-gpt4o", "azure-gpt-4-turbo", "production-judge"):
                _, _, body = AzureBackend().build_request(
                    JudgeConfig(model=legacy, max_tokens=512),
                    [{"role": "user", "content": "hi"}],
                    {"type": "object"},
                )
                assert "max_tokens" in body, legacy
                assert body["max_tokens"] == 512, legacy
                assert "max_completion_tokens" not in body, legacy

    def test_provider_name(self):
        assert AzureBackend().provider_name() == "Azure OpenAI"


# ── AnthropicBackend ──


class TestAnthropicBackend:
    def test_available_with_key(self):
        with patch.dict("os.environ", {"BELT_ANTHROPIC_API_KEY": "sk-ant-test"}):
            assert AnthropicBackend().is_available() is True

    def test_not_available_without_key(self):
        with patch.dict("os.environ", {}, clear=True):
            assert AnthropicBackend().is_available() is False

    def test_build_request_default_base(self):
        with patch.dict("os.environ", {"BELT_ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
            config = JudgeConfig(model="claude-sonnet-4-5")
            messages = [
                {"role": "system", "content": "You are an evaluator."},
                {"role": "system", "content": "Score this scenario."},
            ]
            url, headers, body = AnthropicBackend().build_request(config, messages, {"type": "object"})

            assert url == "https://api.anthropic.com/v1/messages"
            assert headers["x-api-key"] == "sk-ant-test"
            assert headers["anthropic-version"] == "2023-06-01"
            assert body["model"] == "claude-sonnet-4-5"
            assert body["system"] == "You are an evaluator.\n\nScore this scenario."
            assert len(body["messages"]) == 1
            assert body["messages"][0]["role"] == "user"
            assert body["tools"][0]["name"] == "judge_verdict"
            assert body["tool_choice"] == {"type": "tool", "name": "judge_verdict"}

    def test_build_request_custom_base(self):
        # Custom corporate-proxy base URLs over plaintext require an explicit
        # opt-in (JSEC-18900 K-2). The legitimate dev case here is a non-loopback
        # http:// host, so we must set ``BELT_ALLOW_INSECURE_BASE_URL=1``.
        env = {
            "BELT_ANTHROPIC_API_KEY": "sk-ant-test",
            "BELT_ANTHROPIC_BASE_URL": "http://proxy:8080",
            "BELT_ALLOW_INSECURE_BASE_URL": "1",
        }
        with patch.dict("os.environ", env):
            url, _, _ = AnthropicBackend().build_request(
                JudgeConfig(model="anthropic/claude-sonnet-4-5"),
                [{"role": "user", "content": "hi"}],
                {},
            )
            assert url == "http://proxy:8080/v1/messages"

    def test_provider_name(self):
        assert AnthropicBackend().provider_name() == "Anthropic"


# ── resolve_backend ──


class TestResolveBackend:
    def test_openai_prefix(self):
        with patch.dict("os.environ", {"BELT_OPENAI_API_KEY": "sk-x"}, clear=True):
            backend, model = resolve_backend("openai/gpt-4.1")
            assert isinstance(backend, OpenAIBackend)
            assert model == "gpt-4.1"

    def test_anthropic_prefix(self):
        with patch.dict("os.environ", {"BELT_ANTHROPIC_API_KEY": "sk-ant-x"}, clear=True):
            backend, model = resolve_backend("anthropic/claude-sonnet-4-5")
            assert isinstance(backend, AnthropicBackend)
            assert model == "claude-sonnet-4-5"

    def test_azure_prefix(self):
        env = {"BELT_AZURE_OPENAI_ENDPOINT": "https://x", "BELT_AZURE_OPENAI_API_KEY": "key"}
        with patch.dict("os.environ", env, clear=True):
            backend, model = resolve_backend("azure/my-deploy")
            assert isinstance(backend, AzureBackend)
            assert model == "my-deploy"

    def test_ollama_prefix_uses_ollama_backend(self):
        with patch.dict("os.environ", {}, clear=True):
            backend, model = resolve_backend("ollama/llama3")
            assert isinstance(backend, OllamaBackend)
            assert model == "llama3"

    def test_env_var_provider(self):
        env = {"BELT_LLM_PROVIDER": "anthropic", "BELT_ANTHROPIC_API_KEY": "sk-ant-x"}
        with patch.dict("os.environ", env, clear=True):
            backend, model = resolve_backend("claude-sonnet-4-5")
            assert isinstance(backend, AnthropicBackend)
            assert model == "claude-sonnet-4-5"

    def test_openai_new_param_pattern_matches_gpt5_and_o_series(self):
        """New OpenAI models reject ``max_tokens`` and require ``max_completion_tokens``.

        See ``_OPENAI_NEW_PARAM_PATTERN`` in backend.py; this test pins which
        names route to the new param so a regex regression is caught immediately.
        """
        from belt.scorer.llm.backend import _openai_uses_max_completion_tokens

        # Bare upstream names - must use max_completion_tokens
        assert _openai_uses_max_completion_tokens("gpt-5.4-mini")
        assert _openai_uses_max_completion_tokens("gpt-5-turbo")
        assert _openai_uses_max_completion_tokens("o1")
        assert _openai_uses_max_completion_tokens("o1-mini")
        assert _openai_uses_max_completion_tokens("o3")
        assert _openai_uses_max_completion_tokens("o3-mini")
        assert _openai_uses_max_completion_tokens("O1-PREVIEW")  # case-insensitive

        # Team / org-prefixed Azure deployment aliases - must also match.
        # Anchored ``re.match`` would silently misroute these to ``max_tokens``
        # and the API rejects the request with HTTP 400 ``unsupported_parameter``.
        assert _openai_uses_max_completion_tokens("myteam-gpt-5.2")
        assert _openai_uses_max_completion_tokens("myteam-gpt-5-mini")
        assert _openai_uses_max_completion_tokens("prod-gpt-5-mini")
        assert _openai_uses_max_completion_tokens("myteam-o3-mini")
        assert _openai_uses_max_completion_tokens("prod-o3")
        assert _openai_uses_max_completion_tokens("dev-o4-experimental")

        # Legacy / pre-2025 - must keep max_tokens for backwards compatibility
        assert not _openai_uses_max_completion_tokens("gpt-4.1")
        assert not _openai_uses_max_completion_tokens("gpt-4.1-mini")
        assert not _openai_uses_max_completion_tokens("gpt-4o")
        assert not _openai_uses_max_completion_tokens("gpt-4o-mini")
        assert not _openai_uses_max_completion_tokens("gpt-4-turbo")
        assert not _openai_uses_max_completion_tokens("gpt-3.5-turbo")
        # Common prefixed legacy aliases that should not be promoted.
        assert not _openai_uses_max_completion_tokens("gpt-4o-2024-11")
        assert not _openai_uses_max_completion_tokens("azure-gpt-4-turbo")
        # Bare "o" without digit shouldn't false-positive (Ollama models e.g. ``orca``).
        assert not _openai_uses_max_completion_tokens("orca")
        assert not _openai_uses_max_completion_tokens("ollama-host")
        # Letter-adjacent collisions: only a non-alphanumeric boundary (or
        # position 0) allows the family token to match. ``mongo3`` / ``solo3``
        # / ``nemo3`` / ``hippo3000`` embed ``o3``-shaped substrings inside
        # unrelated words and must not be promoted; ``xgpt-5-mini`` embeds
        # ``gpt-5`` after a letter and likewise must not match.
        assert not _openai_uses_max_completion_tokens("xgpt-5-mini")
        assert not _openai_uses_max_completion_tokens("mongo3")
        assert not _openai_uses_max_completion_tokens("solo3-thing")
        assert not _openai_uses_max_completion_tokens("nemo3-foo")
        assert not _openai_uses_max_completion_tokens("hippo3000")
        # Out-of-range family digit (only o1-o9 are reasoning models today).
        assert not _openai_uses_max_completion_tokens("gpt-50-imaginary")
        assert not _openai_uses_max_completion_tokens("production-judge")

    def test_openai_build_request_uses_max_completion_tokens_for_gpt5(self):
        """Sanity check that the field name flips when the model name flips."""
        env = {"BELT_OPENAI_API_KEY": "sk-test"}
        with patch.dict("os.environ", env, clear=True):
            _, _, body = OpenAIBackend().build_request(
                JudgeConfig(model="openai/gpt-5.4-mini", max_tokens=1024),
                [{"role": "user", "content": "hi"}],
                {"type": "object"},
            )
        assert "max_completion_tokens" in body
        assert body["max_completion_tokens"] == 1024
        assert "max_tokens" not in body

    def test_openai_build_request_keeps_max_tokens_for_legacy(self):
        env = {"BELT_OPENAI_API_KEY": "sk-test"}
        with patch.dict("os.environ", env, clear=True):
            _, _, body = OpenAIBackend().build_request(
                JudgeConfig(model="openai/gpt-4.1", max_tokens=1024),
                [{"role": "user", "content": "hi"}],
                {"type": "object"},
            )
        assert "max_tokens" in body
        assert body["max_tokens"] == 1024
        assert "max_completion_tokens" not in body

    def test_openai_build_request_handles_prefixed_deployment_alias(self):
        """The centralized token-limit helper feeds both OpenAIBackend and
        AzureBackend, so the prefix-friendly heuristic must also work when
        a prefixed deployment-style name is routed through an OpenAI-
        compatible endpoint (3rd-party gateways such as OpenRouter or
        self-hosted vLLM commonly mirror Azure-style aliasing).
        """
        env = {"BELT_OPENAI_API_KEY": "sk-test"}
        with patch.dict("os.environ", env, clear=True):
            _, _, body = OpenAIBackend().build_request(
                JudgeConfig(model="openai/myteam-gpt-5.2", max_tokens=1024),
                [{"role": "user", "content": "hi"}],
                {"type": "object"},
            )
        assert "max_completion_tokens" in body
        assert body["max_completion_tokens"] == 1024
        assert "max_tokens" not in body

    def test_judge_config_requires_explicit_model(self):
        """``JudgeConfig.model`` is required - no built-in default.

        Constructing without a model raises a Pydantic ``ValidationError`` so
        the failure surfaces at the boundary instead of silently defaulting to
        OpenAI for Azure/Anthropic/Ollama users.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            JudgeConfig()  # no overrides - now invalid

        # The missing field must be ``model``; future fields with defaults
        # should not regress this signal.
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("model",) and err["type"] == "missing" for err in errors), errors

    def test_explicit_openai_prefixed_model_routes_to_openai_backend(self):
        """Once the user supplies an ``openai/...`` model, prefix routing works without provider env."""
        config = JudgeConfig(model="openai/gpt-5.4-mini")
        with patch.dict("os.environ", {"BELT_OPENAI_API_KEY": "sk-x"}, clear=True):
            backend, model = resolve_backend(config.model)
            assert isinstance(backend, OpenAIBackend)
            assert "/" not in model

    def test_raises_without_prefix_or_provider(self):
        from belt.errors import ConfigError

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ConfigError, match="no provider prefix"):
                resolve_backend("gpt-4.1")


class TestOllamaBackend:
    """Tests for OllamaBackend."""

    def test_provider_name(self):
        assert OllamaBackend().provider_name() == "Ollama"

    def test_build_request_format(self):
        config = JudgeConfig(model="gemma4", temperature=0.1, seed=42)
        messages = [{"role": "user", "content": "hello"}]
        schema = {"type": "object", "properties": {"score": {"type": "number"}}}

        url, headers, body = OllamaBackend().build_request(config, messages, schema)

        assert url == "http://localhost:11434/api/chat"
        assert body["model"] == "gemma4"
        assert body["stream"] is False
        assert body["format"] == schema
        assert "response_format" not in body
        assert body["options"]["temperature"] == 0.1
        assert body["options"]["seed"] == 42
        assert body["messages"] == messages

    def test_build_request_custom_base_url(self):
        # Pointing Ollama at a non-loopback http:// host requires the explicit
        # insecure-base-url opt-in (JSEC-18900 K-2). The default (and the
        # documented dev case) is ``http://localhost:11434``, which is allowed
        # without any opt-in - see ``test_resolve_base_url_default``.
        env = {
            "BELT_OLLAMA_BASE_URL": "http://remote:11434",
            "BELT_ALLOW_INSECURE_BASE_URL": "1",
        }
        with patch.dict("os.environ", env):
            url, _, _ = OllamaBackend().build_request(JudgeConfig(model="m"), [{"role": "user", "content": "hi"}], {})
            assert url == "http://remote:11434/api/chat"

    def test_available_when_server_running(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("httpx.get", return_value=mock_resp):
            assert OllamaBackend().is_available() is True

    def test_not_available_when_server_down(self):
        import httpx as _httpx

        with patch("httpx.get", side_effect=_httpx.ConnectError("refused")):
            assert OllamaBackend().is_available() is False

    def test_not_available_on_timeout(self):
        import httpx as _httpx

        with patch("httpx.get", side_effect=_httpx.TimeoutException("timeout")):
            assert OllamaBackend().is_available() is False


class TestAnthropicAuthRetryHeaders:
    """Anthropic backend offers Authorization: Bearer as a fallback for x-api-key."""

    def test_returns_bearer_when_original_used_x_api_key(self):
        from belt.scorer.llm.backend import AnthropicBackend

        backend = AnthropicBackend()
        original = {
            "x-api-key": "jwt-token-value",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        retry = backend.auth_retry_headers(original)
        assert retry is not None
        assert retry["Authorization"] == "Bearer jwt-token-value"
        assert "x-api-key" not in retry
        # Other headers are preserved verbatim.
        assert retry["Content-Type"] == "application/json"
        assert retry["anthropic-version"] == "2023-06-01"

    def test_returns_none_when_already_bearer(self):
        """Once we've switched to Bearer (cached), don't loop back to x-api-key."""
        from belt.scorer.llm.backend import AnthropicBackend

        backend = AnthropicBackend()
        original = {
            "Authorization": "Bearer jwt-token-value",
            "Content-Type": "application/json",
        }
        assert backend.auth_retry_headers(original) is None

    def test_returns_none_when_no_auth_header(self):
        from belt.scorer.llm.backend import AnthropicBackend

        backend = AnthropicBackend()
        assert backend.auth_retry_headers({"Content-Type": "application/json"}) is None


class TestAnthropicHeaderCache:
    """Once Bearer succeeded, build_request emits Bearer directly on next call."""

    def test_build_request_uses_bearer_after_record_auth_retry_success(self):
        import os
        from unittest.mock import patch

        from belt.scorer.entities import JudgeConfig
        from belt.scorer.llm.backend import AnthropicBackend

        backend = AnthropicBackend()
        config = JudgeConfig(
            model="anthropic/claude-sonnet-4-5",
            temperature=0.0,
            seed=2008,
            max_tokens=4096,
        )
        # Default: x-api-key.
        with patch.dict(os.environ, {"BELT_ANTHROPIC_API_KEY": "jwt-tok"}, clear=False):
            _, headers, _ = backend.build_request(config, [{"role": "user", "content": "hi"}], {"type": "object"})
            assert headers.get("x-api-key") == "jwt-tok"
            assert "Authorization" not in headers

            # Caller marks Bearer as the working style after a successful retry.
            backend.record_auth_retry_success()

            _, headers2, _ = backend.build_request(config, [{"role": "user", "content": "hi"}], {"type": "object"})
            assert headers2.get("Authorization") == "Bearer jwt-tok"
            assert "x-api-key" not in headers2


class TestPreflightAuthRetry:
    """Preflight 401/403 also retries with backend-supplied fallback headers."""

    def test_preflight_403_retries_with_bearer(self):
        import os
        from unittest.mock import patch

        import httpx

        from belt.scorer.entities import JudgeConfig
        from belt.scorer.llm.backend import AnthropicBackend

        captured_headers: list[dict[str, str]] = []

        request = httpx.Request("GET", "https://gw.example/anthropic/v1/models/claude-sonnet-4-5")
        resp_403 = httpx.Response(403, request=request, text="<H1>403</H1>")
        resp_200 = httpx.Response(
            200,
            request=request,
            json={"id": "claude-sonnet-4-5", "type": "model"},
        )

        def fake_get(url, headers, timeout):
            captured_headers.append(dict(headers))
            return resp_200 if len(captured_headers) >= 2 else resp_403

        backend = AnthropicBackend()
        env = {"BELT_ANTHROPIC_API_KEY": "jwt-token", "BELT_ANTHROPIC_BASE_URL": "https://gw.example"}
        with patch.dict(os.environ, env, clear=False), patch("belt.scorer.llm.backend.httpx.get", side_effect=fake_get):
            config = JudgeConfig(
                model="anthropic/claude-sonnet-4-5",
                temperature=0.0,
                seed=2008,
                max_tokens=4096,
            )
            # No exception means preflight succeeded.
            backend.preflight_model(config, timeout=5)

        assert len(captured_headers) == 2
        assert captured_headers[0].get("x-api-key") == "jwt-token"
        assert captured_headers[1].get("Authorization") == "Bearer jwt-token"
        assert backend._use_bearer is True

    def test_preflight_403_then_401_raises_scorer_error_with_both_bodies(self):
        import os

        import httpx

        from belt.errors import ScorerError
        from belt.scorer.entities import JudgeConfig
        from belt.scorer.llm.backend import AnthropicBackend

        request = httpx.Request("GET", "https://gw.example/anthropic/v1/models/claude-sonnet-4-5")
        resp_403 = httpx.Response(403, request=request, text="primary-body")
        resp_401 = httpx.Response(401, request=request, text="retry-body")

        call_count = [0]

        def fake_get(url, headers, timeout):
            call_count[0] += 1
            return resp_403 if call_count[0] == 1 else resp_401

        backend = AnthropicBackend()
        env = {"BELT_ANTHROPIC_API_KEY": "tok", "BELT_ANTHROPIC_BASE_URL": "https://gw.example"}
        patch_env = patch.dict(os.environ, env, clear=False)
        patch_get = patch("belt.scorer.llm.backend.httpx.get", side_effect=fake_get)
        with patch_env, patch_get:
            config = JudgeConfig(model="anthropic/claude-sonnet-4-5", temperature=0.0, seed=0, max_tokens=4096)
            with pytest.raises(ScorerError) as exc_info:
                backend.preflight_model(config, timeout=5)

        msg = str(exc_info.value)
        assert "403+401" in msg
        assert "primary-body" in msg
        assert "retry-body" in msg

    def test_preflight_403_then_5xx_returns_silently(self):
        import os

        import httpx

        from belt.scorer.entities import JudgeConfig
        from belt.scorer.llm.backend import AnthropicBackend

        request = httpx.Request("GET", "https://gw.example/anthropic/v1/models/claude-sonnet-4-5")
        resp_403 = httpx.Response(403, request=request, text="auth-err")
        resp_503 = httpx.Response(503, request=request, text="service-unavailable")

        call_count = [0]

        def fake_get(url, headers, timeout):
            call_count[0] += 1
            return resp_403 if call_count[0] == 1 else resp_503

        backend = AnthropicBackend()
        env = {"BELT_ANTHROPIC_API_KEY": "tok", "BELT_ANTHROPIC_BASE_URL": "https://gw.example"}
        patch_env = patch.dict(os.environ, env, clear=False)
        patch_get = patch("belt.scorer.llm.backend.httpx.get", side_effect=fake_get)
        with patch_env, patch_get:
            config = JudgeConfig(model="anthropic/claude-sonnet-4-5", temperature=0.0, seed=0, max_tokens=4096)
            # 5xx on retry → transient, should return silently
            backend.preflight_model(config, timeout=5)

        assert call_count[0] == 2

    def test_preflight_403_then_timeout_returns_silently(self):
        import os

        import httpx

        from belt.scorer.entities import JudgeConfig
        from belt.scorer.llm.backend import AnthropicBackend

        request = httpx.Request("GET", "https://gw.example/anthropic/v1/models/claude-sonnet-4-5")
        resp_403 = httpx.Response(403, request=request, text="auth-err")

        call_count = [0]

        def fake_get(url, headers, timeout):
            call_count[0] += 1
            if call_count[0] == 1:
                return resp_403
            raise httpx.TimeoutException("timed out", request=request)

        backend = AnthropicBackend()
        env = {"BELT_ANTHROPIC_API_KEY": "tok", "BELT_ANTHROPIC_BASE_URL": "https://gw.example"}
        patch_env = patch.dict(os.environ, env, clear=False)
        patch_get = patch("belt.scorer.llm.backend.httpx.get", side_effect=fake_get)
        with patch_env, patch_get:
            config = JudgeConfig(model="anthropic/claude-sonnet-4-5", temperature=0.0, seed=0, max_tokens=4096)
            # Timeout on retry → transient, should return silently
            backend.preflight_model(config, timeout=5)

        assert call_count[0] == 2

    def test_preflight_403_no_retry_headers_raises_scorer_error_with_primary_body(self):
        """auth_retry_headers returning None (non-Anthropic backend) raises immediately."""
        import os

        import httpx

        from belt.errors import ScorerError
        from belt.scorer.entities import JudgeConfig
        from belt.scorer.llm.backend import OpenAIBackend

        request = httpx.Request("GET", "https://api.openai.com/v1/models/gpt-4.1")
        resp_403 = httpx.Response(403, request=request, text="openai-403-body")

        def fake_get(url, headers, timeout):
            return resp_403

        env = {
            "BELT_OPENAI_API_KEY": "sk-test",
            "BELT_OPENAI_BASE_URL": "https://api.openai.com",
        }
        patch_env = patch.dict(os.environ, env, clear=False)
        patch_get = patch("belt.scorer.llm.backend.httpx.get", side_effect=fake_get)
        with patch_env, patch_get:
            backend = OpenAIBackend()
            config = JudgeConfig(model="openai/gpt-4.1", temperature=0.0, seed=0, max_tokens=4096)
            with pytest.raises(ScorerError) as exc_info:
                backend.preflight_model(config, timeout=5)

        msg = str(exc_info.value)
        assert "openai-403-body" in msg
        # Only one attempt — no retry
        assert "+" not in msg.split("HTTP")[1].split("\n")[0]

    def test_preflight_404_does_not_retry_and_aborts_with_model_hint(self):
        """A 404 (wrong model name) must NOT trigger an auth retry — it is a
        config bug, not an auth failure. The abort reports a single HTTP 404
        and never the misleading 'auth retry didn't help' message."""
        import os

        import httpx

        from belt.errors import ScorerError
        from belt.scorer.entities import JudgeConfig
        from belt.scorer.llm.backend import AnthropicBackend

        request = httpx.Request("GET", "https://gw.example/anthropic/v1/models/does-not-exist")
        resp_404 = httpx.Response(404, request=request, text="model not found")

        call_count = [0]

        def fake_get(url, headers, timeout):
            call_count[0] += 1
            return resp_404

        backend = AnthropicBackend()
        env = {"BELT_ANTHROPIC_API_KEY": "tok", "BELT_ANTHROPIC_BASE_URL": "https://gw.example"}
        patch_env = patch.dict(os.environ, env, clear=False)
        patch_get = patch("belt.scorer.llm.backend.httpx.get", side_effect=fake_get)
        with patch_env, patch_get:
            config = JudgeConfig(model="anthropic/does-not-exist", temperature=0.0, seed=0, max_tokens=4096)
            with pytest.raises(ScorerError) as exc_info:
                backend.preflight_model(config, timeout=5)

        # Exactly one GET — the Bearer auth retry must not fire on a 404.
        assert call_count[0] == 1
        assert backend._use_bearer is False
        msg = str(exc_info.value)
        assert "HTTP 404" in msg
        assert "auth retry didn't help" not in msg
        assert "404+" not in msg

    def test_preflight_403_then_429_returns_silently(self):
        """A 429 on the auth retry is transient — proceed and let the runtime
        judge-infra path handle real failures, mirroring the primary attempt."""
        import os

        import httpx

        from belt.scorer.entities import JudgeConfig
        from belt.scorer.llm.backend import AnthropicBackend

        request = httpx.Request("GET", "https://gw.example/anthropic/v1/models/claude-sonnet-4-5")
        resp_403 = httpx.Response(403, request=request, text="auth-err")
        resp_429 = httpx.Response(429, request=request, text="rate-limited")

        call_count = [0]

        def fake_get(url, headers, timeout):
            call_count[0] += 1
            return resp_403 if call_count[0] == 1 else resp_429

        backend = AnthropicBackend()
        env = {"BELT_ANTHROPIC_API_KEY": "tok", "BELT_ANTHROPIC_BASE_URL": "https://gw.example"}
        patch_env = patch.dict(os.environ, env, clear=False)
        patch_get = patch("belt.scorer.llm.backend.httpx.get", side_effect=fake_get)
        with patch_env, patch_get:
            config = JudgeConfig(model="anthropic/claude-sonnet-4-5", temperature=0.0, seed=0, max_tokens=4096)
            # 429 on retry → transient, should return silently
            backend.preflight_model(config, timeout=5)

        assert call_count[0] == 2
