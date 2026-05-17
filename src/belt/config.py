# (c) JFrog Ltd. (2026)

"""Layered configuration for belt.

Precedence (highest to lowest):
1. CLI flags (applied by the caller after loading config)
2. Environment variables (BELT_* prefix)
3. Config file (belt.yaml in project root or --config path)

There is no implicit default for ``llm.model`` - if none of the three layers
supplies a model, ``load_judge_config`` raises ``ConfigError`` with the
three-source instruction. Silently picking an OpenAI default would mis-route
Azure/Anthropic/Ollama users.

Example belt.yaml:

    llm:
      model: openai/<your-model>   # prefix required: openai/, azure/, anthropic/, ollama/
      temperature: 0.0
      seed: 2008
      max_tokens: 4096

The recommended example value lives in ``belt.constants.EXAMPLE_LLM_MODEL``
and is rendered into user-facing CLI help and error messages from there.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from belt import envvars
from belt.constants import EXAMPLE_LLM_MODEL
from belt.errors import ConfigError
from belt.scorer.entities import JudgeConfig

ModelSource = Literal["cli", "env", "yaml", "unset"]

_ENV_MAP: dict[str, tuple[str, type]] = {
    envvars.LLM_MODEL: ("model", str),
    envvars.LLM_PROVIDER: ("provider", str),
}

_NO_MODEL_HINT = (
    "No LLM judge model configured. Set one of:\n"
    f"  --scorer-arg model={EXAMPLE_LLM_MODEL}   (CLI flag, highest precedence)\n"
    f"  {envvars.LLM_MODEL}={EXAMPLE_LLM_MODEL}   (env var)\n"
    f"  belt.yaml -> llm.model: {EXAMPLE_LLM_MODEL}   (config file)\n"
    "Provider prefixes: openai/, azure/, anthropic/, ollama/"
)


def _find_config_file(start_dir: Path | None = None) -> Path | None:
    """Walk up from start_dir looking for belt.yaml."""
    search = start_dir or Path.cwd()
    for d in [search, *search.parents]:
        candidate = d / "belt.yaml"
        if candidate.is_file():
            return candidate
        candidate = d / "belt.yml"
        if candidate.is_file():
            return candidate
    return None


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        import yaml

        return yaml.safe_load(path.read_text()) or {}
    except ImportError:
        logger.warning("pyyaml not installed - skipping config file {}", path)
        return {}
    except Exception as e:
        logger.warning("Failed to parse config file {}: {}", path, e)
        return {}


def load_judge_config(
    config_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> JudgeConfig:
    """Build JudgeConfig from layered sources.

    Collects values from all layers into a single dict, then constructs
    JudgeConfig once - ensuring Pydantic validation runs on the final values.

    Precedence (highest wins):
    1. CLI flag overrides
    2. Environment variables (BELT_LLM_*)
    3. Config file (belt.yaml -> llm section)

    Raises:
        ConfigError: When no layer supplies ``model``. The error message
            lists all three sources so the user can fix it without diving
            into docs.
    """
    kwargs, _source_map = _collect_judge_kwargs(config_path, cli_overrides)

    if "model" not in kwargs or not str(kwargs["model"]).strip():
        raise ConfigError(_NO_MODEL_HINT)

    return JudgeConfig(**kwargs)


def resolve_judge_model_source(
    config_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> tuple[str | None, ModelSource]:
    """Return ``(model, source)`` for diagnostic display.

    ``source`` is one of ``"cli"``, ``"env"``, ``"yaml"``, or ``"unset"``.
    ``model`` is ``None`` when no layer supplied one. This helper does not
    raise - ``belt doctor`` calls it to display "(not set)" without
    aborting the whole report when llm scoring is not configured.
    """
    kwargs, source_map = _collect_judge_kwargs(config_path, cli_overrides)
    model = kwargs.get("model")
    source = source_map.get("model", "unset")
    return (model, source) if model else (None, "unset")


def _collect_judge_kwargs(
    config_path: Path | None,
    cli_overrides: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, ModelSource]]:
    """Walk the three layers and return (merged kwargs, per-field source map).

    The source map records *which layer* each final value came from, so the
    diagnostic surface (``doctor``, error messages) can attribute provenance
    accurately. Walks lowest-precedence first, overwriting as we climb.
    """
    kwargs: dict[str, Any] = {}
    sources: dict[str, ModelSource] = {}

    path = config_path or _find_config_file()
    if path:
        data = _load_yaml_file(path)
        llm_section = data.get("llm", {})
        if isinstance(llm_section, dict):
            config_keys = (
                "model",
                "temperature",
                "seed",
                "max_tokens",
                "provider",
                "cost_per_prompt_token",
                "cost_per_completion_token",
            )
            for key in config_keys:
                if key in llm_section:
                    kwargs[key] = llm_section[key]
                    sources[key] = "yaml"

    for env_key, (attr, cast) in _ENV_MAP.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        # Empty values are almost certainly a misconfiguration ("export
        # BELT_LLM_MODEL=" leaves the variable set-but-empty). Raising
        # here surfaces the env var as the source of the problem; silently
        # dropping it would let a downstream "no model" error name all
        # three config layers without pointing at the real culprit.
        if val == "":
            raise ConfigError(f"{env_key} is set but empty. Either unset it or provide a value.")
        try:
            kwargs[attr] = cast(val)
            sources[attr] = "env"
        except (ValueError, TypeError) as exc:
            # Bad casts (e.g. ``BELT_LLM_TEMPERATURE=foo``) raise here
            # so the user fixes the source they actually set, instead of
            # silently falling back to YAML or to the missing-model hint.
            raise ConfigError(f"{env_key}={val!r} is not a valid {cast.__name__}: {exc}") from exc

    if cli_overrides:
        for key, val in cli_overrides.items():
            if val is not None:
                kwargs[key] = val
                sources[key] = "cli"

    return kwargs, sources
