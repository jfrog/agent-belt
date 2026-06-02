# (c) JFrog Ltd. (2026)

"""LLM judge backends - provider-agnostic interface for calling LLMs.

Supports OpenAI, Azure OpenAI, Anthropic, Ollama (native API), and any
OpenAI-compatible server (vLLM, together.ai, OpenRouter, LM Studio) via
OPENAI_BASE_URL.

Provider is selected by:
1. Model prefix: ``openai/gpt-5.4-mini``, ``anthropic/claude-sonnet-4-5``, ``ollama/gemma4``
2. ``BELT_LLM_PROVIDER`` env var

A prefix or explicit provider is always required - no auto-detection waterfall.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse

import httpx
from loguru import logger

from belt import envvars
from belt.errors import ConfigError, ScorerError
from belt.scorer.entities import JudgeConfig
from belt.scorer.llm.judge_hints import format_judge_error_hint

# Per-probe timeout for ``preflight_model``. Short enough that a hung
# provider doesn't stall startup, long enough that a sane TCP handshake
# + TLS + 1-roundtrip GET on a slow link still completes. Tunable via
# ``BELT_LLM_PREFLIGHT_TIMEOUT`` for users on satellite links / heavily
# proxied environments.
_DEFAULT_PREFLIGHT_TIMEOUT = 10.0


def _preflight_timeout() -> float:
    raw = os.environ.get(envvars.LLM_PREFLIGHT_TIMEOUT, "")
    if not raw:
        return _DEFAULT_PREFLIGHT_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_PREFLIGHT_TIMEOUT
    if value <= 0:
        return _DEFAULT_PREFLIGHT_TIMEOUT
    return value


# ── Model prefix → provider mapping ──

_PREFIX_MAP: dict[str, str] = {
    "azure": "azure",
    "openai": "openai",
    "anthropic": "anthropic",
    "ollama": "ollama",
}


def parse_model_spec(model: str) -> tuple[str | None, str]:
    """Parse ``provider/model`` → (provider, model_name).

    If no prefix, returns (None, model) - provider determined by auto-detect.
    """
    if "/" in model:
        prefix, name = model.split("/", 1)
        provider = _PREFIX_MAP.get(prefix.lower())
        if provider:
            return provider, name
    return None, model


# ── OpenAI completion-tokens parameter naming ──
#
# OpenAI deprecated ``max_tokens`` in favour of ``max_completion_tokens`` for
# the gpt-5.x family, all reasoning (o1/o2/o3/o4) models, and (per current
# announcements) every model released from late-2025 onwards. The new name is
# *required* on those models - the API rejects requests with the old name with
# HTTP 400 ``unsupported_parameter``. Older models (gpt-4.x, gpt-3.5, gpt-4o)
# still accept ``max_tokens``; sending ``max_completion_tokens`` to them works
# on api.openai.com but third-party OpenAI-compatible servers (vLLM, LM
# Studio, older OpenRouter snapshots) may not implement it yet, so we only
# switch the field on for models where it's required.
#
# The pattern is a non-anchored boundary search rather than an anchored
# match because the ``model_name`` we receive is often an Azure deployment
# alias rather than a bare upstream model name. Enterprise Azure customers
# typically prefix aliases with team/org names (``myteam-gpt-5.2``,
# ``myteam-o3-mini``, ``prod-gpt-5-mini``) - anchoring at the start would
# silently misroute those to ``max_tokens`` and the request would be
# rejected by the API. The leading ``(?:^|[^a-z0-9])`` is a one-char
# boundary (alternative to ``\b``, which would let ``xgpt-5`` and
# ``mongo3`` match because ``\b`` treats letter→letter as inside a word
# but our word includes the ``-``). It rejects ``xgpt-5-mini`` /
# ``mongo3`` / ``solo3-thing`` while accepting any name where the family
# token is preceded by a separator or appears at position 0.
_OPENAI_NEW_PARAM_PATTERN = re.compile(
    r"(?:^|[^a-z0-9])(gpt-5(?:\.|-)|o[1-9](?:[0-9]*)?(?:-|$))",
    re.IGNORECASE,
)


def _openai_uses_max_completion_tokens(model_name: str) -> bool:
    """Whether the OpenAI Chat Completions API for ``model_name`` rejects ``max_tokens``.

    See the comment block above for the policy. ``model_name`` may be a
    bare upstream name (``gpt-5.4-mini``) or an Azure deployment alias
    that embeds the family token (``myteam-gpt-5.2``, ``prod-o3``);
    both must route to the new parameter.
    """
    return bool(_OPENAI_NEW_PARAM_PATTERN.search(model_name.strip().lower()))


def _set_openai_token_limit(body: dict[str, Any], model_name: str, max_tokens: int) -> None:
    """Populate the right token-limit field on ``body`` for OpenAI/Azure requests.

    Centralised so OpenAIBackend and AzureBackend stay in sync; the AzureBackend
    cannot detect the underlying model from the deployment name and falls back
    to ``max_tokens`` unless the deployment name itself contains a new-family
    family token (``gpt-5.2`` bare or ``myteam-gpt-5.2`` prefixed).
    """
    if _openai_uses_max_completion_tokens(model_name):
        body["max_completion_tokens"] = max_tokens
    else:
        body["max_tokens"] = max_tokens


# ── Shared preflight helper ──


def _do_get_preflight(
    backend: "BaseJudgeBackend",
    config: JudgeConfig,
    url: str,
    headers: dict[str, str],
    timeout: float | None,
) -> None:
    """Issue one HTTP GET preflight probe and apply the shared 4xx-vs-5xx policy.

    Centralised because every backend's preflight does the same thing
    around the wire call: dispatch one GET, classify the response into
    "fatal config bug" vs "transient hiccup" via
    :meth:`BaseJudgeBackend.classify_error`, and format the user-facing
    message via :func:`format_judge_error_hint`. Keeping it here means
    a new backend's ``preflight_model`` is just "build the URL and
    delegate" - it cannot accidentally drift on policy.

    Raises :class:`ScorerError` on 401/403/404 (config bug, abort the
    run). Returns silently on 5xx, timeout, 429, network errors, or
    any other ``classify_error`` token - the run proceeds and the
    runtime :class:`belt.errors.JudgeInfraError` path catches real
    failures per-scenario.
    """
    effective_timeout = timeout if timeout is not None else _preflight_timeout()
    provider = backend.provider_name()
    model = config.model
    try:
        resp = httpx.get(url, headers=headers, timeout=effective_timeout)
    except (httpx.TimeoutException, httpx.HTTPError) as exc:
        logger.warning(
            "{} preflight transient failure for model {!r}: {}; "
            "proceeding (runtime judge-infra path will handle real failures).",
            provider,
            model,
            exc,
        )
        return
    if resp.status_code < 400:
        return
    code = resp.status_code
    body = resp.text[:600]
    # Build a synthetic HTTPStatusError so classify_error can decide
    # whether this is a config bug or a transient. We then make the
    # raise/skip decision here so the policy lives in one place.
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        token = backend.classify_error(exc) or "other"
    else:  # pragma: no cover - resp.raise_for_status always raises here
        token = "other"
    if token != "auth_failed":
        logger.warning(
            "{} preflight transient HTTP {} for model {!r}: {}; "
            "proceeding (runtime judge-infra path will handle real failures).",
            provider,
            code,
            model,
            body,
        )
        return
    # Auth failed. A genuine auth rejection (401/403) is worth retrying with
    # the backend's fallback auth style; a 404 means the model name is wrong,
    # so skip the retry and fall through to the clean single-body abort below.
    retry_headers = backend.auth_retry_headers(headers) if code in (401, 403) else None
    if retry_headers is not None:
        logger.warning(
            "{} preflight HTTP {} with primary auth header; retrying once " "with backend-supplied fallback headers.",
            provider,
            code,
        )
        try:
            resp2 = httpx.get(url, headers=retry_headers, timeout=effective_timeout)
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            logger.warning(
                "{} preflight retry transient failure for model {!r}: {}; "
                "proceeding (runtime judge-infra path will handle real failures).",
                provider,
                model,
                exc,
            )
            return
        if resp2.status_code < 400:
            backend.record_auth_retry_success()
            return
        retry_code = resp2.status_code
        retry_body = resp2.text[:600]
        # Classify the retry the same way as the primary attempt so transient
        # hiccups (5xx, 429, network) proceed and only a real auth/config
        # failure aborts — both attempts share one policy.
        try:
            resp2.raise_for_status()
        except httpx.HTTPStatusError as exc:
            retry_token = backend.classify_error(exc) or "other"
        else:  # pragma: no cover - resp.raise_for_status always raises here
            retry_token = "other"
        if retry_token != "auth_failed":
            logger.warning(
                "{} preflight retry transient HTTP {} for model {!r}: {}; "
                "proceeding (runtime judge-infra path will handle real failures).",
                provider,
                retry_code,
                model,
                retry_body,
            )
            return
        # Auth retry also rejected — config bug; report both attempts.
        primary_body = body
        hint = format_judge_error_hint(retry_code, retry_body)
        raise ScorerError(
            f"{provider} judge preflight failed for model {model!r}: "
            f"HTTP {code}+{retry_code} (auth retry didn't help)\n"
            f"  Primary: {primary_body[:300]}\n"
            f"  Retry:   {retry_body[:300]}\n"
            f"  {hint}"
        )
    hint = format_judge_error_hint(code, body)
    raise ScorerError(
        f"{provider} judge preflight failed for model {model!r}: HTTP {code}\n" f"  Upstream body: {body}\n" f"  {hint}"
    )


# ── Backend ABC ──


class BaseJudgeBackend(ABC):
    """Abstract interface for an LLM judge provider."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend's credentials/endpoint are configured."""
        ...

    @abstractmethod
    def build_request(
        self,
        config: JudgeConfig,
        messages: list[dict[str, str]],
        schema: dict,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Build (url, headers, body) for the chat completion request."""
        ...

    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name for error messages."""
        ...

    def preflight_model(self, config: JudgeConfig, *, timeout: float | None = None) -> None:
        """Verify the configured model is callable with the resolved credentials.

        Called by :func:`belt.scorer.llm.preflight.preflight_judges` before
        ``belt eval`` spawns the (expensive) agent phase. Issued as the
        cheapest provider-specific probe that distinguishes the three
        config bugs the user can fix (wrong key / wrong model / wrong
        project scope) from a transient provider hiccup (5xx, timeout,
        rate-limit) that should not block the run.

        Contract:

        - On success, return silently. The model is callable.
        - On a **4xx** that maps to a config bug (401 / 403 / 404),
          raise :class:`ScorerError` with the upstream body + the hint
          from :func:`format_judge_error_hint`. The eval command aborts
          before the agent phase starts.
        - On **5xx**, **timeout**, **rate-limit**, or any
          ``classify_error`` token other than ``auth_failed``, log a
          warning and return silently. These are transient: letting the
          run proceed lets the runtime ``JudgeInfraError`` path handle
          them per-scenario rather than aborting on a single provider
          blip at T=0.

        ``timeout`` defaults to the ``BELT_LLM_PREFLIGHT_TIMEOUT`` env
        var (or 10s if unset / invalid).
        """
        # Default implementation is a no-op for backends that have no
        # cheap server-side probe (or for which the cost of a probe
        # exceeds the cost of a per-scenario runtime failure). Concrete
        # backends override; the abstract surface stays optional so
        # third-party plugins keep working without a forced edit.
        return None

    def classify_error(self, exc: BaseException) -> str | None:
        """Map a backend exception onto a :data:`JUDGE_ERROR_TYPES` token.

        Returns one of ``rate_limited`` / ``timeout`` / ``auth_failed`` /
        ``other``, or ``None`` if the exception is not recognised as a
        judge-infrastructure failure (the caller should let it propagate).

        The default implementation handles the standard ``httpx`` exception
        family that every HTTP-based backend (OpenAI, Azure, Anthropic, the
        OpenAI-compatible variants) produces. Backends with provider-specific
        error shapes override this to map their wire-level errors onto the
        same tokens. Centralising classification here keeps the scorer free
        of ``isinstance`` sprawl and gives plugins one extension point to
        participate in the judge-error partition.
        """
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            if code == 429:
                return "rate_limited"
            if code in (401, 403, 404):
                return "auth_failed"
            return "other"
        if isinstance(exc, httpx.TimeoutException):
            return "timeout"
        if isinstance(exc, httpx.HTTPError):
            return "other"
        return None

    # Design note: auth fallback is exposed as an optional method with a safe
    # default (return None) instead of letting the dispatcher sniff capabilities
    # via isinstance/hasattr. Capability branching in the framework core is what
    # Principle 5 (graceful degradation via optional defaults, not branching)
    # forbids — so backends opt in by overriding, and LLMScorer stays
    # unconditional. record_auth_retry_success() follows the same rationale.
    def auth_retry_headers(self, original_headers: dict[str, str]) -> dict[str, str] | None:
        """Return alternate auth headers to retry once with after a 401/403.

        Called by the judge dispatcher (and the preflight helper) when the
        provider returns 401/403 on the first attempt. A backend that knows
        the gateway in front of it accepts more than one auth header style
        (e.g. JFrog's gateway: WAF allows ``Authorization: Bearer <jwt>``
        but rejects ``x-api-key: <jwt>`` from some egress IPs) returns the
        replacement header dict here. The dispatcher reissues the POST
        once with those headers and, on success, the backend should cache
        which style worked so subsequent requests skip the failed style.

        Returns ``None`` (the default) when no fallback is meaningful — the
        caller propagates the 401/403 as it would today.
        """
        return None

    def record_auth_retry_success(self) -> None:
        """Called by the dispatcher after the auth-retry attempt succeeds.

        Backends that cache the working auth style (to avoid re-paying the
        failed-style round-trip on every subsequent request) override this to
        record the success. The default no-op keeps callers unconditional —
        no ``hasattr`` guard needed.
        """
        return None


# ── OpenAI-compatible backend ──


_warned_custom_base_urls: set[tuple[str, str]] = set()


def _warn_custom_base_url(env_var: str, url: str, default: str) -> None:
    """Log a security warning the first time a non-default base URL is seen.

    ``_resolve_base_url`` is called from ``build_request`` on every judge call,
    so warning per call would flood the log and bury the signal. We dedupe on
    ``(env_var, url)`` and emit at most one warning per process.

    ``BELT_SILENCE_CUSTOM_BASE_URL_WARNING=1`` opts out of the warning
    entirely for users who deliberately use a corporate gateway or proxy.
    """
    if url == default:
        return
    if envvars.is_truthy(envvars.SILENCE_CUSTOM_BASE_URL_WARNING):
        return
    key = (env_var, url)
    if key in _warned_custom_base_urls:
        return
    _warned_custom_base_urls.add(key)
    logger.warning(
        "Custom LLM base URL active via {}: {} - "
        "Authorization headers will be sent to this endpoint. "
        "Set BELT_SILENCE_CUSTOM_BASE_URL_WARNING=1 to silence this warning.",
        env_var,
        url,
    )


# Loopback hostnames where http:// is allowed without an explicit opt-in.
# These are the only addresses that cannot leak credentials over a hostile
# network. Anything else (including private RFC1918 ranges) requires
# ``BELT_ALLOW_INSECURE_BASE_URL=1`` because in cloud / k8s deployments
# the network between the runner and a "private" IP is not under the user's
# control.
_LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


def _validate_base_url_scheme(env_var: str, url: str) -> None:
    """Reject base URLs that would send Authorization over an unsafe scheme.

    Rules:
    * Scheme must be ``http`` or ``https``. ``file://``, ``ftp://``,
      ``javascript:``, etc. are rejected outright (httpx will not even
      execute them, but the error message we surface is much clearer here
      and the rejection prevents an attacker-controlled scheme from being
      logged or echoed back).
    * ``https://`` is always allowed.
    * ``http://`` is allowed only if the host is a loopback address
      (``localhost`` / ``127.0.0.1`` / ``::1``) OR
      ``BELT_ALLOW_INSECURE_BASE_URL=1`` is set. This covers the
      legitimate Ollama / vLLM / LM Studio dev case without ever shipping
      bearer tokens to a remote ``http://`` endpoint by accident.

    Why this matters: ``OpenAIBackend.build_request`` attaches
    ``Authorization: Bearer <key>`` unconditionally. If a user (or a
    hostile dotenv) sets ``BELT_OPENAI_BASE_URL=http://attacker``,
    a single judge call leaks the API key in the clear.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ConfigError(
            f"{env_var}={url!r} has unsupported scheme {scheme!r}. "
            "Only http:// (loopback only) and https:// are allowed."
        )
    if scheme == "https":
        return
    host = (parsed.hostname or "").lower()
    if host in _LOOPBACK_HOSTS:
        return
    if envvars.is_truthy(envvars.ALLOW_INSECURE_BASE_URL):
        # Opt-in for users who deliberately tunnel to a private endpoint
        # (e.g. corporate proxy, in-cluster service). They have explicitly
        # accepted that bearer tokens will travel over plaintext.
        return
    raise ConfigError(
        f"{env_var}={url!r} uses http:// against a non-loopback host ({host!r}). "
        "Authorization headers would be sent in cleartext. "
        "Use https://, point at localhost, or set "
        "BELT_ALLOW_INSECURE_BASE_URL=1 to opt in explicitly."
    )


class OpenAIBackend(BaseJudgeBackend):
    """Backend for OpenAI and any OpenAI-compatible API.

    Works with: OpenAI, Ollama, vLLM, together.ai, OpenRouter, LM Studio.

    Env vars (all prefixed BELT_):
        BELT_OPENAI_API_KEY - required (unless targeting a local server with no auth)
        BELT_OPENAI_BASE_URL - optional, defaults to https://api.openai.com/v1
    """

    _DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def provider_name(self) -> str:
        return "OpenAI"

    def is_available(self) -> bool:
        return bool(os.environ.get(envvars.OPENAI_API_KEY) or os.environ.get(envvars.OPENAI_BASE_URL))

    def _resolve_base_url(self) -> str:
        url = (os.environ.get(envvars.OPENAI_BASE_URL) or self._DEFAULT_BASE_URL).rstrip("/")
        _validate_base_url_scheme(envvars.OPENAI_BASE_URL, url)
        _warn_custom_base_url(envvars.OPENAI_BASE_URL, url, self._DEFAULT_BASE_URL)
        return url

    def preflight_model(self, config: JudgeConfig, *, timeout: float | None = None) -> None:
        # ``GET /v1/models/{model}`` is the cheapest probe that
        # distinguishes the three OpenAI config bugs:
        #   * 401 invalid_api_key       → bad key
        #   * 403 model_not_found       → project-scoped key without access
        #   * 404 model_not_found       → typo'd model name
        # And it round-trips in tens of milliseconds. The endpoint is
        # implemented identically by every OpenAI-compatible server we
        # ship support for (vLLM, together.ai, OpenRouter, LM Studio);
        # servers that don't implement it return 404 on the path, which
        # we'd misread as "model missing" - so we only probe when the
        # base URL is the default OpenAI host or one we know implements
        # ``/v1/models/{id}``. For unknown bases, skip silently and let
        # the runtime path catch real failures.
        api_key = os.environ.get(envvars.OPENAI_API_KEY, "")
        base = self._resolve_base_url()
        bare_model = parse_model_spec(config.model)[1]
        url = f"{base}/models/{bare_model}"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        _do_get_preflight(self, config, url, headers, timeout)

    def build_request(
        self,
        config: JudgeConfig,
        messages: list[dict[str, str]],
        schema: dict,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        api_key = os.environ.get(envvars.OPENAI_API_KEY, "")
        base = self._resolve_base_url()
        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        bare_model = parse_model_spec(config.model)[1]
        body: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "seed": config.seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "JudgeVerdict", "strict": True, "schema": schema},
            },
        }
        _set_openai_token_limit(body, bare_model, config.max_tokens)
        return url, headers, body


# ── Azure OpenAI backend ──


class AzureBackend(BaseJudgeBackend):
    """Backend for Azure OpenAI deployments.

    Env vars (all prefixed BELT_):
        BELT_AZURE_OPENAI_ENDPOINT - required
        Auth (one of):
            BELT_AZURE_OPENAI_API_KEY - API key auth
            BELT_AZURE_CLIENT_ID + BELT_AZURE_CLIENT_SECRET + BELT_AZURE_TENANT_ID - service principal
        BELT_AZURE_OPENAI_API_VERSION - optional, defaults to 2024-10-21
    """

    _SP_VARS = [envvars.AZURE_CLIENT_ID, envvars.AZURE_CLIENT_SECRET, envvars.AZURE_TENANT_ID]

    def __init__(self) -> None:
        self._sp_token: str | None = None
        self._sp_token_expires: float = 0

    def provider_name(self) -> str:
        return "Azure OpenAI"

    def is_available(self) -> bool:
        if not os.environ.get(envvars.AZURE_OPENAI_ENDPOINT):
            return False
        return bool(os.environ.get(envvars.AZURE_OPENAI_API_KEY) or all(os.environ.get(k) for k in self._SP_VARS))

    def preflight_model(self, config: JudgeConfig, *, timeout: float | None = None) -> None:
        # Azure has no reliable cross-tier management-plane probe: the
        # GET ``/openai/deployments/{name}`` endpoint is not exposed
        # to the data-plane api-key on every Azure resource SKU, so it
        # returns 404 even for working deployments. Probe the actual
        # runtime endpoint (POST ``/chat/completions``) with an
        # intentionally invalid empty-messages body instead. Azure
        # routes by URL first:
        #   * existing deployment + bad body → 400 (preflight passes)
        #   * unknown deployment             → 404 DeploymentNotFound
        #   * bad api-key                    → 401
        endpoint = os.environ.get(envvars.AZURE_OPENAI_ENDPOINT, "").rstrip("/")
        if not endpoint:
            return
        api_version = os.environ.get(envvars.AZURE_OPENAI_API_VERSION, "2024-10-21")
        bare_model = parse_model_spec(config.model)[1]
        url = f"{endpoint}/openai/deployments/{bare_model}/chat/completions?api-version={api_version}"
        headers = self._get_auth_headers() or {}
        effective_timeout = timeout if timeout is not None else _preflight_timeout()
        try:
            resp = httpx.post(url, headers=headers, json={"messages": []}, timeout=effective_timeout)
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            logger.warning(
                "Azure preflight transient failure for model {!r}: {}; "
                "proceeding (runtime judge-infra path will handle real failures).",
                config.model,
                exc,
            )
            return
        code = resp.status_code
        # 200 (unlikely with empty messages) or 400 (validation): the
        # deployment exists and is reachable. Any other 4xx that maps
        # to ``auth_failed`` is a real config bug; everything else is
        # treated as transient.
        if code < 400 or code == 400:
            return
        body = resp.text[:600]
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            token = self.classify_error(exc) or "other"
        else:  # pragma: no cover
            token = "other"
        if token != "auth_failed":
            logger.warning(
                "Azure preflight transient HTTP {} for model {!r}: {}; "
                "proceeding (runtime judge-infra path will handle real failures).",
                code,
                config.model,
                body,
            )
            return
        hint = format_judge_error_hint(code, body)
        raise ScorerError(
            f"Azure OpenAI judge preflight failed for model {config.model!r}: HTTP {code}\n"
            f"  Upstream body: {body}\n"
            f"  {hint}"
        )

    def _get_auth_headers(self) -> dict[str, str] | None:
        api_key = os.environ.get(envvars.AZURE_OPENAI_API_KEY)
        if api_key:
            return {"api-key": api_key, "Content-Type": "application/json"}

        import time

        if self._sp_token and time.time() < self._sp_token_expires:
            return {"Authorization": f"Bearer {self._sp_token}", "Content-Type": "application/json"}

        try:
            resp = httpx.post(
                f"https://login.microsoftonline.com/{os.environ[envvars.AZURE_TENANT_ID]}/oauth2/v2.0/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": os.environ[envvars.AZURE_CLIENT_ID],
                    "client_secret": os.environ[envvars.AZURE_CLIENT_SECRET],
                    "scope": "https://cognitiveservices.azure.com/.default",
                },
                timeout=30,
            )
            resp.raise_for_status()
            token_data = resp.json()
            self._sp_token = token_data["access_token"]
            # Cache for 50 minutes (tokens are valid for 60 min; 10 min safety margin)
            self._sp_token_expires = time.time() + token_data.get("expires_in", 3600) - 600
            return {
                "Authorization": f"Bearer {self._sp_token}",
                "Content-Type": "application/json",
            }
        except KeyError as e:
            logger.error("Missing BELT_AZURE_* env var: {}", e)
            return None
        except Exception as e:
            logger.error("Azure token acquisition failed: {}", e)
            return None

    def build_request(
        self,
        config: JudgeConfig,
        messages: list[dict[str, str]],
        schema: dict,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        endpoint = os.environ[envvars.AZURE_OPENAI_ENDPOINT].rstrip("/")
        _validate_base_url_scheme(envvars.AZURE_OPENAI_ENDPOINT, endpoint)
        api_version = os.environ.get(envvars.AZURE_OPENAI_API_VERSION, "2024-10-21")

        # Azure deployment URL carries the bare deployment alias only;
        # ``parse_model_spec`` is idempotent so this is safe whether
        # the caller passes the prefixed or already-stripped form.
        bare_model = parse_model_spec(config.model)[1]
        url = f"{endpoint}/openai/deployments/{bare_model}/chat/completions?api-version={api_version}"

        headers = self._get_auth_headers()
        if headers is None:
            from belt.errors import ConfigError

            raise ConfigError(
                "Azure auth failed - set BELT_AZURE_OPENAI_API_KEY or " "BELT_AZURE_CLIENT_ID/SECRET/TENANT_ID"
            )

        body: dict[str, Any] = {
            "messages": messages,
            "temperature": config.temperature,
            "seed": config.seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "JudgeVerdict", "strict": True, "schema": schema},
            },
        }
        # Azure deployment names are arbitrary, but new-family OpenAI models
        # routed through Azure also reject ``max_tokens``. Use the same
        # heuristic as direct-OpenAI - a deployment whose alias contains the
        # upstream family token (``gpt-5.4-mini`` bare, ``myteam-gpt-5.2``
        # team-prefixed, ``prod-o3``) gets the new field; legacy deployments
        # named ``my-gpt4o`` keep the old name. Operators with opaque names
        # (``production-judge``) can rename the deployment to embed the family
        # token, or set their own override.
        _set_openai_token_limit(body, bare_model, config.max_tokens)
        return url, headers, body


# ── Anthropic backend ──


class AnthropicBackend(BaseJudgeBackend):
    """Backend for Anthropic's Messages API.

    Uses tool_use for structured output (Anthropic doesn't support json_schema
    response_format). Converts the schema into a tool definition, forces the
    model to call it, and extracts the JSON from the tool call result.

    Env vars (all prefixed BELT_):
        BELT_ANTHROPIC_API_KEY - required
        BELT_ANTHROPIC_BASE_URL - optional, defaults to https://api.anthropic.com
    """

    def __init__(self) -> None:
        # Per-process cache: once we've seen Bearer succeed (typically
        # because a WAF in front of the gateway rejected x-api-key), every
        # subsequent build_request emits Bearer directly. Avoids paying a
        # 401/403 round-trip on every single judge call in a CI run.
        # Single-boolean assignment is atomic under the GIL, so no lock
        # is needed when --workers N > 1; worst case: two threads both see
        # False and each do one extra retry before converging on True.
        self._use_bearer = False

    def record_auth_retry_success(self) -> None:
        """Record that the alternate auth style succeeded — emit it on subsequent calls."""
        self._use_bearer = True

    def provider_name(self) -> str:
        return "Anthropic"

    def auth_retry_headers(self, original_headers: dict[str, str]) -> dict[str, str] | None:
        # Swap x-api-key → Authorization: Bearer. The JFrog gateway's WAF
        # rejects x-api-key from runner-pool egress with a CloudFront 403
        # but accepts Bearer; outside that environment Anthropic's own API
        # also accepts Bearer (undocumented but stable since v1).
        api_key = original_headers.get("x-api-key")
        if not api_key:
            return None
        retry = {k: v for k, v in original_headers.items() if k != "x-api-key"}
        retry["Authorization"] = f"Bearer {api_key}"
        return retry

    def is_available(self) -> bool:
        return bool(os.environ.get(envvars.ANTHROPIC_API_KEY))

    _DEFAULT_BASE_URL = "https://api.anthropic.com"

    def preflight_model(self, config: JudgeConfig, *, timeout: float | None = None) -> None:
        # Anthropic exposes ``GET /v1/models/{id}`` which returns 404
        # ``not_found_error`` on a typo and 401 on a bad key. It also
        # returns the same 403 shape as OpenAI when an org-scoped key
        # lacks model access - good fit for the same preflight policy.
        api_key = os.environ.get(envvars.ANTHROPIC_API_KEY, "")
        base = self._resolve_base_url()
        bare_model = parse_model_spec(config.model)[1]
        url = f"{base}/v1/models/{bare_model}"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        _do_get_preflight(self, config, url, headers, timeout)

    def _resolve_base_url(self) -> str:
        url = (os.environ.get(envvars.ANTHROPIC_BASE_URL) or self._DEFAULT_BASE_URL).rstrip("/")
        _validate_base_url_scheme(envvars.ANTHROPIC_BASE_URL, url)
        _warn_custom_base_url(envvars.ANTHROPIC_BASE_URL, url, self._DEFAULT_BASE_URL)
        return url

    def build_request(
        self,
        config: JudgeConfig,
        messages: list[dict[str, str]],
        schema: dict,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        api_key = os.environ[envvars.ANTHROPIC_API_KEY]
        base = self._resolve_base_url()
        url = f"{base}/v1/messages"
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self._use_bearer:
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["x-api-key"] = api_key

        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        system_text = "\n\n".join(system_parts)
        user_parts = [m["content"] for m in messages if m["role"] != "system"]
        user_text = "\n\n".join(user_parts) if user_parts else "Please evaluate."

        tool_schema = _schema_to_anthropic_tool(schema)

        body: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "system": system_text,
            "messages": [{"role": "user", "content": user_text}],
            "tools": [tool_schema],
            "tool_choice": {"type": "tool", "name": "judge_verdict"},
        }
        return url, headers, body


class OllamaBackend(BaseJudgeBackend):
    """Backend for Ollama's native REST API.

    Uses ``/api/chat`` with the ``format`` parameter for grammar-constrained
    structured output - more reliable than Ollama's OpenAI-compatible endpoint.

    Env vars (all prefixed BELT_):
        BELT_OLLAMA_BASE_URL - optional, defaults to http://localhost:11434
    """

    _DEFAULT_BASE_URL = "http://localhost:11434"

    def provider_name(self) -> str:
        return "Ollama"

    def _resolve_base_url(self) -> str:
        url = (os.environ.get(envvars.OLLAMA_BASE_URL) or self._DEFAULT_BASE_URL).rstrip("/")
        _validate_base_url_scheme(envvars.OLLAMA_BASE_URL, url)
        return url

    def is_available(self) -> bool:
        try:
            base = self._resolve_base_url()
        except ConfigError:
            # Misconfigured base URL is reported via build_request to surface
            # a clear error; treat the backend as unavailable for discovery.
            return False
        try:
            resp = httpx.get(f"{base}/api/tags", timeout=3)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def preflight_model(self, config: JudgeConfig, *, timeout: float | None = None) -> None:
        # Ollama's ``POST /api/show`` returns 200 with the model
        # manifest when the model is pulled and 404 ``model 'x' not
        # found`` otherwise - which is exactly the "model not
        # available locally" case the user must fix. We do NOT pass
        # Ollama 404s through the OpenAI-shaped formatter because
        # Ollama returns plain-text bodies without an ``error.code``;
        # the generic 404 hint plus an Ollama-specific ``ollama pull``
        # instruction is the right surface.
        bare_model = parse_model_spec(config.model)[1]
        try:
            base = self._resolve_base_url()
        except ConfigError:
            # Re-raised by ``build_request`` later; skipping preflight
            # is safe because the runtime path will surface the same
            # error with the same message.
            return
        url = f"{base}/api/show"
        headers = {"Content-Type": "application/json"}
        effective_timeout = timeout if timeout is not None else _preflight_timeout()
        try:
            resp = httpx.post(url, json={"name": bare_model}, headers=headers, timeout=effective_timeout)
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            logger.warning(
                "Ollama preflight transient failure for model {!r}: {}; "
                "proceeding (runtime judge-infra path will handle real failures).",
                config.model,
                exc,
            )
            return
        if resp.status_code < 400:
            return
        code = resp.status_code
        body = resp.text[:600]
        if code == 404:
            raise ScorerError(
                f"Ollama judge preflight failed for model {config.model!r}: HTTP 404\n"
                f"  Upstream body: {body}\n"
                f"  Hint: pull the model first with `ollama pull {bare_model}` "
                f"or pick one from `ollama list`."
            )
        # Any other status (5xx, 503 during model load, etc.) is
        # transient; let the runtime path handle it.
        logger.warning(
            "Ollama preflight transient HTTP {} for model {!r}: {}; proceeding.",
            code,
            config.model,
            body,
        )

    def build_request(
        self,
        config: JudgeConfig,
        messages: list[dict[str, str]],
        schema: dict,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        base = self._resolve_base_url()
        url = f"{base}/api/chat"
        headers = {"Content-Type": "application/json"}
        body: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": config.temperature,
                "seed": config.seed,
            },
        }
        return url, headers, body


def _schema_to_anthropic_tool(schema: dict) -> dict:
    """Convert a JSON schema into an Anthropic tool definition.

    Anthropic's tool_use expects ``input_schema`` in JSON Schema format,
    which is what we already have - just wrap it in the tool envelope.
    """
    return {
        "name": "judge_verdict",
        "description": "Submit the structured evaluation verdict as JSON.",
        "input_schema": schema,
    }


# ── Provider resolution ──

_PROVIDER_BACKENDS: dict[str, type[BaseJudgeBackend]] = {
    "azure": AzureBackend,
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "ollama": OllamaBackend,
}


_VALID_PREFIXES = sorted(_PREFIX_MAP.keys())


def resolve_backend(model: str, provider: str | None = None) -> tuple[BaseJudgeBackend, str]:
    """Resolve backend + clean model name from model spec.

    Resolution order:
    1. Model prefix (``openai/gpt-5.4-mini`` → OpenAIBackend, model=``gpt-5.4-mini``) - **required**
    2. Explicit ``provider`` parameter (from JudgeConfig or multi-judge YAML)
    3. ``BELT_LLM_PROVIDER`` env var

    A provider prefix or explicit provider is always required. There is no
    auto-detection waterfall - this prevents silent misrouting when multiple
    providers are configured.

    Returns (backend_instance, clean_model_name).
    """
    provider_hint, clean_model = parse_model_spec(model)

    if provider_hint:
        cls = _PROVIDER_BACKENDS.get(provider_hint)
        if cls:
            return cls(), clean_model

    for source in [provider, os.environ.get(envvars.LLM_PROVIDER, "")]:
        if source and source.lower() in _PROVIDER_BACKENDS:
            return _PROVIDER_BACKENDS[source.lower()](), clean_model

    from belt.errors import ConfigError

    raise ConfigError(
        f"Model '{model}' has no provider prefix. Use one of: "
        + ", ".join(f"'{p}/{model}'" for p in _VALID_PREFIXES)
        + "\n  Or set BELT_LLM_PROVIDER env var."
    )
