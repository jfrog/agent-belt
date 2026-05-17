# (c) JFrog Ltd. (2026)

"""Tests for scorer/llm/pricing.py - model cost lookup, computation, and TOML loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from belt.envvars import PRICING_FILE
from belt.scorer.llm.pricing import compute_cost, lookup_pricing, reload_pricing_table


class TestLookupPricing:
    def test_known_openai_model(self):
        p = lookup_pricing("gpt-4.1")
        assert p is not None
        assert p.input_per_token == pytest.approx(2.0e-6)
        assert p.output_per_token == pytest.approx(8.0e-6)

    def test_known_anthropic_model(self):
        p = lookup_pricing("claude-sonnet-4-6")
        assert p is not None
        assert p.input_per_token == pytest.approx(3.0e-6)

    def test_case_insensitive(self):
        p = lookup_pricing("GPT-4.1-MINI")
        assert p is not None
        assert p.input_per_token == pytest.approx(0.4e-6)

    def test_provider_prefixed(self):
        p = lookup_pricing("openai/gpt-4.1")
        assert p is not None
        assert p.input_per_token == pytest.approx(2.0e-6)

    def test_unknown_model_returns_none(self):
        assert lookup_pricing("my-custom-azure-deployment") is None

    def test_azure_prefix_unknown_deploy(self):
        assert lookup_pricing("azure/my-deploy-name") is None

    def test_prefix_stripped_to_known(self):
        p = lookup_pricing("azure/gpt-4.1")
        assert p is not None
        assert p.input_per_token == pytest.approx(2.0e-6)


class TestComputeCost:
    def test_known_model(self):
        cost = compute_cost("gpt-4.1", prompt_tokens=1000, completion_tokens=500)
        assert cost is not None
        expected = 1000 * 2.0e-6 + 500 * 8.0e-6
        assert cost == pytest.approx(expected)

    def test_override_takes_priority(self):
        cost = compute_cost(
            "gpt-4.1",
            prompt_tokens=1000,
            completion_tokens=500,
            cost_per_prompt_token=0.00001,
            cost_per_completion_token=0.00005,
        )
        expected = 1000 * 0.00001 + 500 * 0.00005
        assert cost == pytest.approx(expected)

    def test_unknown_model_no_override_returns_none(self):
        cost = compute_cost("azure/my-fancy-deploy", prompt_tokens=1000, completion_tokens=500)
        assert cost is None

    def test_unknown_model_with_override(self):
        cost = compute_cost(
            "azure/my-fancy-deploy",
            prompt_tokens=1000,
            completion_tokens=500,
            cost_per_prompt_token=0.000003,
            cost_per_completion_token=0.000015,
        )
        expected = 1000 * 0.000003 + 500 * 0.000015
        assert cost == pytest.approx(expected)

    def test_zero_tokens(self):
        cost = compute_cost("gpt-4.1", prompt_tokens=0, completion_tokens=0)
        assert cost == pytest.approx(0.0)

    def test_partial_override_falls_to_table(self):
        cost = compute_cost(
            "gpt-4.1",
            prompt_tokens=1000,
            completion_tokens=500,
            cost_per_prompt_token=0.00001,
            cost_per_completion_token=None,
        )
        expected = 1000 * 2.0e-6 + 500 * 8.0e-6
        assert cost == pytest.approx(expected)


class TestPricingTomlMetadata:
    """Provenance fields loaded from the bundled pricing TOML."""

    def test_bundled_entry_has_valid_from_and_source(self):
        p = lookup_pricing("gpt-5.4-mini")
        assert p is not None
        assert p.valid_from == "2026-03-17"
        assert p.source_url and p.source_url.startswith("https://")


class TestEnvOverride:
    """``BELT_PRICING_FILE`` fully replaces the bundled table."""

    def _write_override(self, tmp_path: Path) -> Path:
        path = tmp_path / "override.toml"
        path.write_text(
            """
[models."enterprise-gpt"]
input_per_token = 0.0000001
output_per_token = 0.0000004
valid_from = "2026-04-01"
source_url = "https://internal.example.com/rates"

[aliases]
"acme/enterprise-gpt" = "enterprise-gpt"
""",
            encoding="utf-8",
        )
        return path

    def test_override_replaces_bundled_table(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        path = self._write_override(tmp_path)
        monkeypatch.setenv(PRICING_FILE, str(path))
        try:
            reload_pricing_table()
            # Override-only model resolves.
            p = lookup_pricing("enterprise-gpt")
            assert p is not None
            assert p.valid_from == "2026-04-01"
            # Bundled models do NOT resolve; override is a full replace.
            assert lookup_pricing("gpt-4.1") is None
            # Override aliases work.
            assert lookup_pricing("acme/enterprise-gpt") is not None
        finally:
            monkeypatch.delenv(PRICING_FILE, raising=False)
            reload_pricing_table()

    def test_missing_override_path_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(PRICING_FILE, str(tmp_path / "does-not-exist.toml"))
        try:
            with pytest.raises(ValueError, match=PRICING_FILE):
                reload_pricing_table()
        finally:
            monkeypatch.delenv(PRICING_FILE, raising=False)
            reload_pricing_table()

    def test_malformed_toml_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        bad = tmp_path / "bad.toml"
        bad.write_text('[models."x"]\ninput_per_token = "not-a-number"\n', encoding="utf-8")
        monkeypatch.setenv(PRICING_FILE, str(bad))
        try:
            with pytest.raises(ValueError):
                reload_pricing_table()
        finally:
            monkeypatch.delenv(PRICING_FILE, raising=False)
            reload_pricing_table()
