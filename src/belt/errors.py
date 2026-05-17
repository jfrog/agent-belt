# (c) JFrog Ltd. (2026)

"""Typed error hierarchy for user-facing error messages.

All errors that should render as clean one-line messages (not raw tracebacks)
inherit from ``BeltError``. The top-level boundary in ``cli.py`` catches
these and formats them for the terminal. Errors are user-facing communication:
they tell the user what went wrong and what to do next, not what stack frame
raised what.
"""

from __future__ import annotations


class BeltError(Exception):
    """Base for all belt errors. Message is user-facing."""


class ConfigError(BeltError):
    """Bad config file, invalid YAML, missing required env var."""


class AgentExecutionError(BeltError):
    """Agent execution or parsing failure."""


class ScorerError(BeltError):
    """Scoring failure that should abort the run (fatal config, auth, response parse).

    ``ScorerError`` signals a problem the user must fix before any further
    scoring can succeed - typically an HTTP 401/403/404 from the judge
    backend (wrong API key, wrong model name, wrong endpoint). The top-level
    handler catches it and stops the run rather than silently producing
    "judge errored" verdicts for every remaining scenario.
    """


class JudgeInfraError(BeltError):
    """Transient LLM-judge infrastructure failure (rate-limit, timeout, network).

    Distinguished from :class:`ScorerError` by recoverability: the user has
    not done anything wrong - the judge provider is rate-limiting, timing
    out, or otherwise transiently unavailable. The scorer catches this,
    records ``judge_errored=True`` plus a typed ``error_type`` on the
    scenario's :class:`belt.scorer.payloads.LLMPayload`, and lets the run
    continue. The aggregator partitions the scenario out of the headline
    pass-rate via the "judge environment failures" axis.

    ``error_type`` is one of the tokens in
    :data:`belt.scorer.entities.JUDGE_ERROR_TYPES`; classification lives at
    the :class:`belt.scorer.llm.backend.BaseJudgeBackend.classify_error`
    boundary so each backend can map its provider-specific error shapes
    to the same tokens.
    """

    def __init__(self, error_type: str, message: str = "") -> None:
        super().__init__(message or error_type)
        self.error_type = error_type


class ScenarioError(BeltError):
    """Bad scenario JSON, missing files, validation error."""
