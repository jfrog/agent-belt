# (c) JFrog Ltd. (2026)

"""Tests for layered configuration loading."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from belt.config import _find_config_file, load_judge_config, resolve_judge_model_source
from belt.errors import ConfigError


class TestFindConfigFile:
    def test_finds_yaml_in_cwd(self, tmp_path):
        (tmp_path / "belt.yaml").write_text("llm:\n  model: gpt-4.1\n")
        result = _find_config_file(tmp_path)
        assert result == tmp_path / "belt.yaml"

    def test_finds_yml_variant(self, tmp_path):
        (tmp_path / "belt.yml").write_text("llm:\n  model: gpt-4.1\n")
        result = _find_config_file(tmp_path)
        assert result == tmp_path / "belt.yml"

    def test_walks_up_to_parent(self, tmp_path):
        child = tmp_path / "a" / "b"
        child.mkdir(parents=True)
        (tmp_path / "belt.yaml").write_text("llm:\n  model: gpt-4.1\n")
        result = _find_config_file(child)
        assert result == tmp_path / "belt.yaml"

    def test_returns_none_when_not_found(self, tmp_path):
        result = _find_config_file(tmp_path)
        assert result is None

    def test_yaml_preferred_over_yml(self, tmp_path):
        (tmp_path / "belt.yaml").write_text("llm:\n  model: from-yaml\n")
        (tmp_path / "belt.yml").write_text("llm:\n  model: from-yml\n")
        result = _find_config_file(tmp_path)
        assert result.name == "belt.yaml"


class TestLoadJudgeConfig:
    """``model`` is required - no built-in default."""

    @pytest.fixture(autouse=True)
    def _clear_llm_env(self):
        # Strip any BELT_LLM_* the test runner inherited so layered-config
        # behaviour is deterministic regardless of the developer's shell state.
        with patch.dict(os.environ, {}, clear=False):
            for k in [k for k in os.environ if k.startswith("BELT_LLM_")]:
                os.environ.pop(k, None)
            yield

    def test_no_layer_supplies_model_raises_config_error(self, tmp_path):
        with pytest.raises(ConfigError) as exc_info:
            load_judge_config(config_path=tmp_path / "nonexistent.yaml")
        msg = str(exc_info.value)
        assert "--scorer-arg model=" in msg
        assert "BELT_LLM_MODEL" in msg
        assert "belt.yaml" in msg
        assert "openai/" in msg and "ollama/" in msg

    def test_yaml_only_model_succeeds(self, tmp_path):
        """Yaml-only path - matches the proposal's missed-coverage gap."""
        cfg = tmp_path / "belt.yaml"
        cfg.write_text("llm:\n  model: openai/gpt-5.4-mini\n")
        config = load_judge_config(config_path=cfg)
        assert config.model == "openai/gpt-5.4-mini"

    def test_env_only_model_succeeds(self, tmp_path):
        with patch.dict(os.environ, {"BELT_LLM_MODEL": "openai/gpt-5.4-mini"}):
            config = load_judge_config(config_path=tmp_path / "nonexistent.yaml")
        assert config.model == "openai/gpt-5.4-mini"

    def test_cli_only_model_succeeds(self, tmp_path):
        config = load_judge_config(
            config_path=tmp_path / "nonexistent.yaml",
            cli_overrides={"model": "openai/gpt-5.4-mini"},
        )
        assert config.model == "openai/gpt-5.4-mini"

    def test_config_file_sets_model(self, tmp_path):
        cfg = tmp_path / "belt.yaml"
        cfg.write_text("llm:\n  model: openai/gpt-5.4-mini\n  temperature: 0.5\n")
        config = load_judge_config(config_path=cfg)
        assert config.model == "openai/gpt-5.4-mini"
        assert config.temperature == 0.5

    def test_env_overrides_config_file(self, tmp_path):
        cfg = tmp_path / "belt.yaml"
        cfg.write_text("llm:\n  model: from-file\n")
        with patch.dict(os.environ, {"BELT_LLM_MODEL": "from-env"}):
            config = load_judge_config(config_path=cfg)
        assert config.model == "from-env"

    def test_cli_overrides_env(self, tmp_path):
        cfg = tmp_path / "belt.yaml"
        cfg.write_text("llm:\n  model: from-file\n")
        with patch.dict(os.environ, {"BELT_LLM_MODEL": "from-env"}):
            config = load_judge_config(config_path=cfg, cli_overrides={"model": "from-cli"})
        assert config.model == "from-cli"

    def test_cli_overrides_alone(self):
        config = load_judge_config(
            cli_overrides={"model": "openai/gpt-5.4-mini", "temperature": 0.1},
        )
        assert config.model == "openai/gpt-5.4-mini"
        assert config.temperature == 0.1

    def test_empty_env_var_treated_as_unset(self, tmp_path):
        """Whitespace/empty model string in env should fail-fast, not produce a phantom config."""
        with patch.dict(os.environ, {"BELT_LLM_MODEL": "   "}):
            with pytest.raises(ConfigError):
                load_judge_config(config_path=tmp_path / "nonexistent.yaml")

    def test_empty_env_var_names_the_var(self, tmp_path):
        """Empty env-var values should raise with the env-var name in the message."""
        with patch.dict(os.environ, {"BELT_LLM_MODEL": ""}):
            with pytest.raises(ConfigError) as exc_info:
                load_judge_config(config_path=tmp_path / "nonexistent.yaml")
        msg = str(exc_info.value)
        assert "BELT_LLM_MODEL" in msg
        assert "empty" in msg.lower()

    def test_invalid_env_var_cast_raises_with_var_name(self, tmp_path):
        """Bad casts (e.g. a hypothetical numeric env var with non-numeric value)
        should raise with the env-var name in the message, not silently warn-and-drop.

        ``_ENV_MAP`` currently only registers string-typed vars (``MODEL``,
        ``PROVIDER``), so we patch in a ``float`` cast for this regression
        test - the protection is the contract, not which casts are wired today.
        """
        from belt import config as config_mod

        patched_map = dict(config_mod._ENV_MAP)
        patched_map["BELT_LLM_TEMPERATURE_TEST"] = ("temperature", float)
        with patch.dict(config_mod._ENV_MAP, patched_map, clear=True):
            with patch.dict(
                os.environ,
                {
                    "BELT_LLM_MODEL": "openai/gpt-5.4-mini",
                    "BELT_LLM_TEMPERATURE_TEST": "foo",
                },
            ):
                with pytest.raises(ConfigError) as exc_info:
                    load_judge_config(config_path=tmp_path / "nonexistent.yaml")
        msg = str(exc_info.value)
        assert "BELT_LLM_TEMPERATURE_TEST" in msg
        assert "foo" in msg

    def test_malformed_yaml_falls_through_to_other_layers(self, tmp_path):
        """A bad yaml shouldn't crash; it should fall through, leaving us at no-model -> ConfigError."""
        cfg = tmp_path / "belt.yaml"
        cfg.write_text("not: valid: yaml: {{{{")
        with pytest.raises(ConfigError):
            load_judge_config(config_path=cfg)

    def test_empty_yaml_falls_through_to_other_layers(self, tmp_path):
        cfg = tmp_path / "belt.yaml"
        cfg.write_text("")
        with pytest.raises(ConfigError):
            load_judge_config(config_path=cfg)


class TestResolveJudgeModelSource:
    """Provenance reporting for ``belt doctor``."""

    @pytest.fixture(autouse=True)
    def _clear_llm_env(self):
        with patch.dict(os.environ, {}, clear=False):
            for k in [k for k in os.environ if k.startswith("BELT_LLM_")]:
                os.environ.pop(k, None)
            yield

    def test_unset_when_no_layer_supplies_model(self, tmp_path):
        model, source = resolve_judge_model_source(config_path=tmp_path / "nonexistent.yaml")
        assert model is None
        assert source == "unset"

    def test_yaml_attribution(self, tmp_path):
        cfg = tmp_path / "belt.yaml"
        cfg.write_text("llm:\n  model: openai/gpt-5.4-mini\n")
        model, source = resolve_judge_model_source(config_path=cfg)
        assert model == "openai/gpt-5.4-mini"
        assert source == "yaml"

    def test_env_attribution(self, tmp_path):
        with patch.dict(os.environ, {"BELT_LLM_MODEL": "openai/gpt-5.4-mini"}):
            model, source = resolve_judge_model_source(config_path=tmp_path / "nonexistent.yaml")
        assert model == "openai/gpt-5.4-mini"
        assert source == "env"

    def test_cli_attribution(self, tmp_path):
        model, source = resolve_judge_model_source(
            config_path=tmp_path / "nonexistent.yaml",
            cli_overrides={"model": "openai/gpt-5.4-mini"},
        )
        assert model == "openai/gpt-5.4-mini"
        assert source == "cli"

    def test_cli_wins_over_env_in_attribution(self, tmp_path):
        cfg = tmp_path / "belt.yaml"
        cfg.write_text("llm:\n  model: from-yaml\n")
        with patch.dict(os.environ, {"BELT_LLM_MODEL": "from-env"}):
            model, source = resolve_judge_model_source(
                config_path=cfg,
                cli_overrides={"model": "from-cli"},
            )
        assert model == "from-cli"
        assert source == "cli"

    def test_env_wins_over_yaml_in_attribution(self, tmp_path):
        cfg = tmp_path / "belt.yaml"
        cfg.write_text("llm:\n  model: from-yaml\n")
        with patch.dict(os.environ, {"BELT_LLM_MODEL": "from-env"}):
            model, source = resolve_judge_model_source(config_path=cfg)
        assert model == "from-env"
        assert source == "env"
