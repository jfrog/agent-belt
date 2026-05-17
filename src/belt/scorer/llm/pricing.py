# (c) JFrog Ltd. (2026)

"""Model pricing data for LLM judge cost calculation.

Resolution order in :func:`compute_cost`:

1. Explicit per-judge override (``cost_per_prompt_token`` /
   ``cost_per_completion_token`` on ``JudgeConfig``).
2. Pricing table loaded once at import: bundled ``pricing.toml`` by
   default, fully replaced (not merged) when ``BELT_PRICING_FILE`` is set.
3. ``None`` (unknown model; caller renders tokens only).

Prices are per-token in USD.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from loguru import logger

from belt.envvars import PRICING_FILE

_BUNDLED_PRICING_PATH = Path(str(files("belt.scorer.llm") / "pricing.toml"))

_warned_models: set[str] = set()


@dataclass(frozen=True)
class ModelPricing:
    """Per-token rates plus optional provenance metadata loaded from the pricing TOML."""

    input_per_token: float
    output_per_token: float
    valid_from: str | None = None
    source_url: str | None = None


# Ollama runs locally; zero-cost by definition. Kept programmatic so a bad
# override file can't accidentally remove the rule.
_OLLAMA_ZERO_COST = ModelPricing(0.0, 0.0)


def _load_toml_pricing(path: Path) -> tuple[dict[str, ModelPricing], dict[str, str]]:
    """Parse a pricing TOML file and return ``(models, aliases)``.

    Raises ``ValueError`` (with file path context) on malformed structure
    so a typo in ``BELT_PRICING_FILE`` surfaces loudly at import.
    """
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ValueError(f"failed to parse pricing TOML at {path}: {e}") from e

    models_section = raw.get("models", {})
    aliases_section = raw.get("aliases", {})
    if not isinstance(models_section, dict) or not isinstance(aliases_section, dict):
        raise ValueError(f"pricing TOML at {path} must define [models] and (optionally) [aliases] tables")

    models: dict[str, ModelPricing] = {}
    for name, row in models_section.items():
        if not isinstance(row, dict):
            raise ValueError(f"pricing entry '{name}' in {path} must be a table")
        try:
            inp = float(row["input_per_token"])
            out = float(row["output_per_token"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"pricing entry '{name}' in {path} requires numeric " f"'input_per_token' and 'output_per_token': {e}"
            ) from e
        models[name.lower().strip()] = ModelPricing(
            input_per_token=inp,
            output_per_token=out,
            valid_from=_opt_str(row.get("valid_from")),
            source_url=_opt_str(row.get("source_url")),
        )

    aliases: dict[str, str] = {}
    for alias, target in aliases_section.items():
        if not isinstance(target, str):
            raise ValueError(f"alias '{alias}' in {path} must map to a string canonical name")
        aliases[alias.lower().strip()] = target.lower().strip()

    return models, aliases


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _resolve_pricing_path() -> tuple[Path, str]:
    """Return ``(path, source_label)`` for the pricing TOML to load."""
    override = os.environ.get(PRICING_FILE, "").strip()
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            # Fail loudly: silently falling through to the bundled table
            # under a typo'd override would produce wrong cost numbers.
            raise ValueError(f"{PRICING_FILE}={override!r} does not point to a readable file")
        return path, "env-override"
    return _BUNDLED_PRICING_PATH, "bundled"


def _load_default_table() -> tuple[dict[str, ModelPricing], dict[str, str]]:
    path, _ = _resolve_pricing_path()
    return _load_toml_pricing(path)


_PRICING_TABLE, _ALIASES = _load_default_table()


def reload_pricing_table() -> None:
    """Re-read the pricing TOML and refresh module-level state.

    Test-only escape hatch; production code resolves once at import.
    """
    global _PRICING_TABLE, _ALIASES, _warned_models
    _PRICING_TABLE, _ALIASES = _load_default_table()
    _warned_models = set()


def lookup_pricing(model: str) -> ModelPricing | None:
    """Look up pricing for a model name.

    Tries exact match, then alias, then prefix-stripped fallback.
    Returns ``None`` for unknown models (caller should warn once).
    """
    key = model.lower().strip()

    if key in _PRICING_TABLE:
        return _PRICING_TABLE[key]

    canonical = _ALIASES.get(key)
    if canonical and canonical in _PRICING_TABLE:
        return _PRICING_TABLE[canonical]

    if "/" in key:
        prefix, bare = key.split("/", 1)
        # Ollama models are local inference - zero cost by definition.
        if prefix == "ollama":
            return _OLLAMA_ZERO_COST
        if bare in _PRICING_TABLE:
            return _PRICING_TABLE[bare]

    return None


def compute_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cost_per_prompt_token: float | None = None,
    cost_per_completion_token: float | None = None,
) -> float | None:
    """Compute cost in USD for a given token usage.

    See module docstring for the resolution order. Returns ``None`` for
    unknown models; logs a one-time warning per model name.
    """
    if cost_per_prompt_token is not None and cost_per_completion_token is not None:
        return prompt_tokens * cost_per_prompt_token + completion_tokens * cost_per_completion_token

    pricing = lookup_pricing(model)
    if pricing is not None:
        return prompt_tokens * pricing.input_per_token + completion_tokens * pricing.output_per_token

    if model not in _warned_models:
        _warned_models.add(model)
        logger.warning(
            "Unknown model '{}' - cannot compute cost. "
            "Set cost_per_prompt_token / cost_per_completion_token in scorer config, "
            "or add the model to the pricing TOML pointed at by ${}.",
            model,
            PRICING_FILE,
        )
    return None
