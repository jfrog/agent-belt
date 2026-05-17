# (c) JFrog Ltd. (2026)

"""Per-backend tests for :meth:`BaseJudgeBackend.preflight_model`.

The preflight contract is the single behaviour that distinguishes the
three "config bug" cases (wrong key / wrong model / project-scoped key
without model access) from "transient hiccup" (5xx, timeout, rate-limit).
These tests pin that contract per backend so a future refactor cannot
silently regress one provider's preflight (e.g. forgetting to raise on
Azure's ``DeploymentNotFound``, which would re-introduce wasted agent-
phase spend on every misconfigured run).

Strategy:
- Patch ``httpx.get`` (and ``httpx.post`` for Ollama) so no network
  is touched.
- Status 200 ⇒ no exception, returns ``None``.
- Status 401/403/404 ⇒ raises :class:`ScorerError` with a hint that
  matches the provider-specific error code in the body.
- Status 5xx / 429 / timeout / connection error ⇒ no exception
  (transient; the runtime ``JudgeInfraError`` path handles it).
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from belt.errors import ScorerError
from belt.scorer.entities import JudgeConfig
from belt.scorer.llm.backend import AnthropicBackend, AzureBackend, OllamaBackend, OpenAIBackend

# Each backend's env-var prerequisites; the fixture below sets them so
# ``preflight_model`` builds a well-formed request.
_ENV_FIXTURES = {
    OpenAIBackend: {"BELT_OPENAI_API_KEY": "sk-test"},
    AzureBackend: {
        "BELT_AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "BELT_AZURE_OPENAI_API_KEY": "az-test",
    },
    AnthropicBackend: {"BELT_ANTHROPIC_API_KEY": "ant-test"},
    OllamaBackend: {},  # no creds required; uses default loopback URL
}

_MODEL_FIXTURES = {
    OpenAIBackend: "openai/gpt-4o-mini",
    AzureBackend: "my-deployment",  # Azure uses deployment alias, no prefix
    AnthropicBackend: "anthropic/claude-3-5-sonnet-latest",
    OllamaBackend: "ollama/gemma2",
}


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every BELT_* env var so each test starts from a known state."""
    import os

    for key in list(os.environ):
        if key.startswith("BELT_"):
            monkeypatch.delenv(key, raising=False)


def _setup_env(monkeypatch: pytest.MonkeyPatch, backend_cls: type) -> None:
    for k, v in _ENV_FIXTURES[backend_cls].items():
        monkeypatch.setenv(k, v)


def _make_response(status: int, body: str = "") -> httpx.Response:
    req = httpx.Request("GET", "https://judge.example/v1/models/x")
    return httpx.Response(status_code=status, text=body, request=req)


def _patch_http(
    monkeypatch: pytest.MonkeyPatch,
    *,
    get_response: httpx.Response | Exception | None = None,
    post_response: httpx.Response | Exception | None = None,
) -> dict[str, list[Any]]:
    """Patch ``httpx.get`` and ``httpx.post`` in the backend module.

    Returns a dict with call lists so tests can assert the right URL
    was hit. Passing an ``Exception`` as a response simulates a
    network-level error (timeout, connection refused). When only one
    verb's response is set, the other defaults to the same value so
    cross-backend parametrized tests do not have to know whether a
    given backend probes via GET or POST.
    """
    if post_response is None and get_response is not None:
        post_response = get_response
    elif get_response is None and post_response is not None:
        get_response = post_response

    calls: dict[str, list[Any]] = {"get": [], "post": []}

    def fake_get(url: str, **kwargs: Any) -> httpx.Response:
        calls["get"].append((url, kwargs))
        if isinstance(get_response, Exception):
            raise get_response
        assert get_response is not None, "test did not configure a GET response"
        return get_response

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        calls["post"].append((url, kwargs))
        if isinstance(post_response, Exception):
            raise post_response
        assert post_response is not None, "test did not configure a POST response"
        return post_response

    monkeypatch.setattr("belt.scorer.llm.backend.httpx.get", fake_get)
    monkeypatch.setattr("belt.scorer.llm.backend.httpx.post", fake_post)
    return calls


@pytest.mark.parametrize(
    "backend_cls",
    [OpenAIBackend, AzureBackend, AnthropicBackend, OllamaBackend],
    ids=lambda c: c.__name__,
)
class TestPreflightSuccess:
    """A 200 response must NOT raise and must NOT spend more than one HTTP call."""

    def test_returns_silently_on_200(self, backend_cls: type, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup_env(monkeypatch, backend_cls)
        ok = _make_response(200, "{}")
        calls = _patch_http(monkeypatch, get_response=ok, post_response=ok)
        cfg = JudgeConfig(model=_MODEL_FIXTURES[backend_cls])
        backend_cls().preflight_model(cfg, timeout=1.0)
        total_calls = len(calls["get"]) + len(calls["post"])
        assert total_calls == 1, f"expected exactly one probe, got {total_calls}"


@pytest.mark.parametrize(
    "backend_cls",
    [OpenAIBackend, AzureBackend, AnthropicBackend],
    ids=lambda c: c.__name__,
)
class TestPreflightConfigBugs:
    """401 / 403 / 404 ⇒ ScorerError with a hint that mentions the upstream code."""

    def test_401_invalid_api_key_raises_with_key_hint(self, backend_cls: type, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup_env(monkeypatch, backend_cls)
        body = '{"error":{"code":"invalid_api_key","message":"bad key"}}'
        _patch_http(monkeypatch, get_response=_make_response(401, body))
        cfg = JudgeConfig(model=_MODEL_FIXTURES[backend_cls])
        with pytest.raises(ScorerError) as exc:
            backend_cls().preflight_model(cfg, timeout=1.0)
        msg = str(exc.value)
        assert "401" in msg
        assert "Hint:" in msg
        assert "BELT_" in msg and "API_KEY" in msg

    def test_403_model_not_found_raises_with_project_hint(
        self, backend_cls: type, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Project-scoped key without model access: HTTP 403 +
        # ``code: model_not_found``. Distinct from 404 + same code
        # (typo), and the hint must reflect that distinction.
        body = (
            '{"error":{"code":"model_not_found","type":"invalid_request_error",'
            '"message":"Project does not have access to model"}}'
        )
        _setup_env(monkeypatch, backend_cls)
        _patch_http(monkeypatch, get_response=_make_response(403, body))
        cfg = JudgeConfig(model=_MODEL_FIXTURES[backend_cls])
        with pytest.raises(ScorerError) as exc:
            backend_cls().preflight_model(cfg, timeout=1.0)
        msg = str(exc.value)
        assert "403" in msg
        assert "model_not_found" in msg
        # The 403 hint MUST NOT claim "model name is wrong" - that's the 404 case.
        assert "404" not in msg

    def test_404_model_not_found_raises_with_typo_hint(
        self, backend_cls: type, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = '{"error":{"code":"model_not_found","message":"unknown model"}}'
        _setup_env(monkeypatch, backend_cls)
        _patch_http(monkeypatch, get_response=_make_response(404, body))
        cfg = JudgeConfig(model=_MODEL_FIXTURES[backend_cls])
        with pytest.raises(ScorerError) as exc:
            backend_cls().preflight_model(cfg, timeout=1.0)
        msg = str(exc.value)
        assert "404" in msg
        assert "model_not_found" in msg
        # 404 hint MUST talk about typo / model name.
        assert "typo" in msg.lower() or "wrong" in msg.lower() or "scorer-arg" in msg.lower()


@pytest.mark.parametrize(
    "backend_cls",
    [OpenAIBackend, AzureBackend, AnthropicBackend],
    ids=lambda c: c.__name__,
)
class TestPreflightTransientHiccups:
    """5xx / 429 / timeout / connection-error ⇒ no exception (runtime handles it)."""

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_transient_status_does_not_raise(
        self, backend_cls: type, monkeypatch: pytest.MonkeyPatch, status: int
    ) -> None:
        _setup_env(monkeypatch, backend_cls)
        _patch_http(monkeypatch, get_response=_make_response(status, "transient"))
        cfg = JudgeConfig(model=_MODEL_FIXTURES[backend_cls])
        # Must NOT raise.
        backend_cls().preflight_model(cfg, timeout=1.0)

    def test_timeout_does_not_raise(self, backend_cls: type, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup_env(monkeypatch, backend_cls)
        _patch_http(
            monkeypatch,
            get_response=httpx.TimeoutException("simulated timeout"),
        )
        cfg = JudgeConfig(model=_MODEL_FIXTURES[backend_cls])
        backend_cls().preflight_model(cfg, timeout=1.0)

    def test_connection_error_does_not_raise(self, backend_cls: type, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup_env(monkeypatch, backend_cls)
        _patch_http(monkeypatch, get_response=httpx.ConnectError("connection refused"))
        cfg = JudgeConfig(model=_MODEL_FIXTURES[backend_cls])
        backend_cls().preflight_model(cfg, timeout=1.0)


# ── Ollama: tests the POST /api/show path, distinct from the GET-based backends ──


class TestOllamaPreflight:
    """Ollama probes via ``POST /api/show``; 404 = model not pulled."""

    def test_200_returns_silently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_http(monkeypatch, post_response=_make_response(200, '{"modelfile":"..."}'))
        cfg = JudgeConfig(model="ollama/gemma2")
        OllamaBackend().preflight_model(cfg, timeout=1.0)

    def test_404_raises_with_ollama_pull_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = "model 'gemma2' not found, try pulling it first"
        _patch_http(monkeypatch, post_response=_make_response(404, body))
        cfg = JudgeConfig(model="ollama/gemma2")
        with pytest.raises(ScorerError) as exc:
            OllamaBackend().preflight_model(cfg, timeout=1.0)
        msg = str(exc.value)
        assert "404" in msg
        # Ollama-specific guidance: tell the user to ``ollama pull``.
        assert "ollama pull" in msg

    def test_5xx_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_http(monkeypatch, post_response=_make_response(503, "loading"))
        cfg = JudgeConfig(model="ollama/gemma2")
        OllamaBackend().preflight_model(cfg, timeout=1.0)

    def test_connection_error_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_http(monkeypatch, post_response=httpx.ConnectError("refused"))
        cfg = JudgeConfig(model="ollama/gemma2")
        OllamaBackend().preflight_model(cfg, timeout=1.0)


# ── Azure: URL must NEVER carry the ``azure/`` provider prefix ──


class TestAzurePreflightStripsPrefix:
    """The Azure deployment URL is ``…/deployments/<alias>/chat/completions?…``;
    the ``azure/`` provider prefix is metadata, never a path segment.
    The OpenAI / Anthropic / Ollama preflights strip it via
    :func:`parse_model_spec`; this guards that Azure does the same on
    both preflight and runtime build paths."""

    def test_preflight_url_uses_bare_deployment_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup_env(monkeypatch, AzureBackend)
        calls = _patch_http(monkeypatch, post_response=_make_response(400, "{}"))
        cfg = JudgeConfig(model="azure/my-deployment")
        AzureBackend().preflight_model(cfg, timeout=1.0)
        assert len(calls["post"]) == 1
        url = calls["post"][0][0]
        assert "/deployments/my-deployment/chat/completions?" in url
        assert "/deployments/azure/" not in url

    def test_build_request_url_uses_bare_deployment_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup_env(monkeypatch, AzureBackend)
        cfg = JudgeConfig(model="azure/my-deployment")
        url, _headers, _body = AzureBackend().build_request(cfg, [{"role": "user", "content": "x"}], {"type": "object"})
        assert "/deployments/my-deployment/chat/completions?" in url
        assert "/deployments/azure/" not in url


class TestAzurePreflightPostProbe:
    """Azure preflight probes ``POST /chat/completions`` with an empty
    ``messages`` body. Azure routes by URL first, so the response code
    cleanly distinguishes deployment-typo / auth-bug / reachable cases."""

    def test_400_validation_error_passes_preflight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Working deployment + intentionally invalid body returns 400.
        The deployment is reachable; preflight must not abort."""
        _setup_env(monkeypatch, AzureBackend)
        body = '{"error":{"message":"Invalid messages: empty array","type":"invalid_request_error"}}'
        _patch_http(monkeypatch, post_response=_make_response(400, body))
        cfg = JudgeConfig(model="azure/my-deployment")
        AzureBackend().preflight_model(cfg, timeout=1.0)

    def test_404_deployment_not_found_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _setup_env(monkeypatch, AzureBackend)
        body = '{"error":{"code":"DeploymentNotFound","type":"invalid_request_error","message":"The API deployment for this resource does not exist."}}'
        _patch_http(monkeypatch, post_response=_make_response(404, body))
        cfg = JudgeConfig(model="azure/typo-deployment")
        with pytest.raises(ScorerError) as exc:
            AzureBackend().preflight_model(cfg, timeout=1.0)
        assert "404" in str(exc.value)
        assert "DeploymentNotFound" in str(exc.value)


# ── Default-backend ABC: no-op for plugins that don't override ──


class TestBaseBackendDefault:
    """A backend that doesn't override ``preflight_model`` must be a no-op.

    This keeps third-party plugin backends working without forcing a
    breaking change when ``preflight_model`` lands. The default raises
    no exception and does not call the network.
    """

    def test_default_is_no_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # If the default contacted the network, this would crash.
        def boom(*_: Any, **__: Any) -> Any:
            raise AssertionError("default preflight_model must not touch the network")

        monkeypatch.setattr("belt.scorer.llm.backend.httpx.get", boom)
        monkeypatch.setattr("belt.scorer.llm.backend.httpx.post", boom)

        from belt.scorer.llm.backend import BaseJudgeBackend

        class _StubBackend(BaseJudgeBackend):
            def is_available(self) -> bool:
                return True

            def build_request(self, config, messages, schema):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            def provider_name(self) -> str:
                return "stub"

        cfg = JudgeConfig(model="stub/whatever")
        _StubBackend().preflight_model(cfg, timeout=1.0)  # must not raise
