# (c) JFrog Ltd. (2026)

"""Tests for ``BaseJudgeBackend.classify_error`` and the scorer's
judge-infra error path.

The classification table is the single source of truth for how a
provider exception becomes a :data:`JUDGE_ERROR_TYPES` token. Every
HTTP-based backend (OpenAI, Azure, Anthropic, Ollama, custom OpenAI-
compatible) shares the default ``httpx``-based mapping defined on
:class:`BaseJudgeBackend`; backends with provider-specific exception
shapes override only the cases they need to map differently. The tests
below pin the default mapping per-backend so a future regression that
drops 401/403/404 from ``auth_failed`` or moves 429 out of
``rate_limited`` fails loudly.

The scorer-side tests verify the contract that a transient infra failure
results in a non-verdict ``LLMPayload`` (``judge_errored=True``,
``dimensions={}``, ``overall_pass=False``), so downstream consumers can
key off the typed field without parsing strings.
"""

from __future__ import annotations

import httpx
import pytest

from belt.entities import TurnOutput
from belt.errors import JudgeInfraError, ScorerError
from belt.scenario import Scenario, Turn, TurnExpectation
from belt.scorer.entities import JUDGE_ERROR_TYPES, JudgeConfig
from belt.scorer.llm.backend import AnthropicBackend, AzureBackend, BaseJudgeBackend, OllamaBackend, OpenAIBackend
from belt.scorer.llm.scorer import LLMScorer
from belt.scorer.payloads import LLMPayload


@pytest.fixture
def turn_outputs() -> list[TurnOutput]:
    return [TurnOutput(raw_cli="reply: 42")]


@pytest.fixture
def scenario() -> Scenario:
    return Scenario(
        name="t",
        description="judge-infra classification test scenario",
        turns=[Turn(message="what is 6*7?", expect=TurnExpectation(has_reply=True))],
    )


def _fake_response(status: int, text: str = "") -> httpx.Response:
    req = httpx.Request("POST", "https://judge.example/api")
    return httpx.Response(status_code=status, text=text, request=req)


def _status_error(status: int) -> httpx.HTTPStatusError:
    resp = _fake_response(status, text=f"{status} body")
    return httpx.HTTPStatusError("http error", request=resp.request, response=resp)


# ── classify_error: shared default mapping ──


@pytest.mark.parametrize(
    "backend_cls",
    [OpenAIBackend, AzureBackend, AnthropicBackend, OllamaBackend],
    ids=lambda c: c.__name__,
)
class TestDefaultClassification:
    """Every built-in backend inherits the same ``httpx``-based mapping."""

    def test_429_maps_to_rate_limited(self, backend_cls: type[BaseJudgeBackend]) -> None:
        assert backend_cls().classify_error(_status_error(429)) == "rate_limited"

    @pytest.mark.parametrize("status", [401, 403, 404])
    def test_auth_codes_map_to_auth_failed(self, backend_cls: type[BaseJudgeBackend], status: int) -> None:
        assert backend_cls().classify_error(_status_error(status)) == "auth_failed"

    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    def test_server_errors_map_to_other(self, backend_cls: type[BaseJudgeBackend], status: int) -> None:
        assert backend_cls().classify_error(_status_error(status)) == "other"

    def test_timeout_maps_to_timeout(self, backend_cls: type[BaseJudgeBackend]) -> None:
        assert backend_cls().classify_error(httpx.ReadTimeout("slow")) == "timeout"

    def test_connect_timeout_maps_to_timeout(self, backend_cls: type[BaseJudgeBackend]) -> None:
        assert backend_cls().classify_error(httpx.ConnectTimeout("slow connect")) == "timeout"

    def test_connect_error_maps_to_other(self, backend_cls: type[BaseJudgeBackend]) -> None:
        # ``ConnectError`` is an ``httpx.HTTPError`` but not a timeout;
        # the default mapping classifies it as "other" (network).
        assert backend_cls().classify_error(httpx.ConnectError("refused")) == "other"

    def test_unknown_exception_returns_none(self, backend_cls: type[BaseJudgeBackend]) -> None:
        # An exception the backend does not recognise as infra is not
        # silently swallowed - returning None tells the scorer to let
        # it propagate or treat it as ``other``.
        assert backend_cls().classify_error(ValueError("boom")) is None


class TestClassificationTokensExhaustive:
    """Every token in :data:`JUDGE_ERROR_TYPES` has at least one shape that maps to it."""

    @pytest.mark.parametrize(
        "exc,expected",
        [
            (_status_error(429), "rate_limited"),
            (httpx.ReadTimeout("slow"), "timeout"),
            (_status_error(401), "auth_failed"),
            (_status_error(502), "other"),
        ],
    )
    def test_every_token_reachable(self, exc: BaseException, expected: str) -> None:
        assert expected in JUDGE_ERROR_TYPES
        assert OpenAIBackend().classify_error(exc) == expected


# ── scorer.score returns judge_errored payload on infra failure ──


class _ErroringBackend(BaseJudgeBackend):
    """Test double that always raises a configured ``httpx`` exception.

    Lets us exercise :meth:`LLMScorer.score`'s infra-failure path without
    touching the network and without the matrix-runner ``echo`` agent
    that the empirical fixtures use.
    """

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def provider_name(self) -> str:
        return "erroring"

    def is_available(self) -> bool:
        return True

    def build_request(self, config, messages, schema):
        # Return a syntactically valid request - the scorer never sees
        # the response because httpx.post is monkeypatched to raise.
        return "https://erroring.example/v1/chat", {}, {}


def _post_raising(exc: BaseException):
    def _f(*_a, **_kw):
        raise exc

    return _f


@pytest.mark.parametrize(
    "exc,expected_type",
    [
        (_status_error(429), "rate_limited"),
        (httpx.ReadTimeout("slow"), "timeout"),
        (_status_error(502), "other"),
        (httpx.ConnectError("refused"), "other"),
    ],
)
def test_score_returns_judge_errored_payload_for_infra_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    scenario: Scenario,
    turn_outputs: list[TurnOutput],
    exc: BaseException,
    expected_type: str,
) -> None:
    """The four documented transient failures all yield a non-verdict payload."""
    backend = _ErroringBackend(exc)
    monkeypatch.setattr("belt.scorer.llm.scorer.httpx.post", _post_raising(exc))
    scorer = LLMScorer(
        config=JudgeConfig(model="openai/gpt-test"),
        backend=backend,
        cache=None,
        skip_availability=True,
        max_retries=1,
    )

    result = scorer.score(scenario, turn_outputs)

    assert result is not None, "score must return a payload, not None, on infra failure"
    assert isinstance(result.data, LLMPayload)
    assert result.data.judge_errored is True
    assert result.data.judge_error_type == expected_type
    assert result.data.dimensions == {}
    assert result.data.overall_pass is False
    assert result.passed is False


def test_score_raises_scorer_error_on_auth_failure(
    monkeypatch: pytest.MonkeyPatch,
    scenario: Scenario,
    turn_outputs: list[TurnOutput],
) -> None:
    """401/403/404 must abort the run, not produce N silent errored verdicts.

    Auth errors are user-actionable: the user has the wrong key, the
    wrong model name, or the wrong endpoint. Continuing past the first
    one wastes time and quota; aborting is faster feedback.
    """
    exc = _status_error(401)
    backend = _ErroringBackend(exc)
    monkeypatch.setattr("belt.scorer.llm.scorer.httpx.post", _post_raising(exc))
    scorer = LLMScorer(
        config=JudgeConfig(model="openai/gpt-test"),
        backend=backend,
        cache=None,
        skip_availability=True,
        max_retries=1,
    )
    with pytest.raises(ScorerError):
        scorer.score(scenario, turn_outputs)


def test_judge_infra_error_carries_typed_error_type() -> None:
    """``JudgeInfraError.error_type`` is the typed payload field for callers.

    The aggregator and the renderer key off this attribute (via
    :func:`belt.scorer.llm.scorer._judge_errored_payload`), so the
    constructor signature is part of the API surface and must stay stable.
    """
    err = JudgeInfraError("rate_limited", "429 from provider")
    assert err.error_type == "rate_limited"
    assert "rate_limited" in str(err) or "429" in str(err)
