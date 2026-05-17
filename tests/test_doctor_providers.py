# (c) JFrog Ltd. (2026)

"""Doctor must enumerate every advertised LLM provider regardless of host config.

External consumers (CI gates, scripts) need to know *which* providers exist
so they can decide what credentials to inject. ``doctor --json`` must therefore
report all four canonical providers (OpenAI, Anthropic, Azure OpenAI, Ollama),
even on a host with zero credentials configured - the ``ok`` flag tells the
caller whether the provider is *currently* usable; the *enumeration* tells
them what's available to configure.

This test runs ``doctor --json`` in a child process with all relevant env
vars cleared, parses the JSON, and asserts the four-provider invariant.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

_BELT_BIN = shutil.which("belt")


def _run_doctor_json_clean() -> dict:
    """Run ``doctor --json`` with provider env vars cleared."""
    if not _BELT_BIN:
        pytest.skip("belt console script not on PATH (install with `pip install -e .`)")
    # Strip every var that would change the provider readiness state, so the
    # assertion is about the *enumeration*, not the host's local config.
    env = {k: v for k, v in os.environ.items() if not k.startswith("BELT_")}
    # Force provider checks to run even if Ollama is reachable on localhost -
    # we pin the URL to a non-routable port so the check always returns "not
    # running" without making the assertion depend on the host.
    env["BELT_OLLAMA_BASE_URL"] = "http://127.0.0.1:1"  # unreachable
    proc = subprocess.run(
        [_BELT_BIN, "doctor", "--json"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    # Doctor exits 1 when no agents are configured; that's expected on a
    # CI host. We only care that valid JSON came out.
    if not proc.stdout.strip():
        raise RuntimeError(f"doctor --json produced no stdout (rc={proc.returncode}); stderr: {proc.stderr[:500]}")
    return json.loads(proc.stdout)


def test_doctor_json_enumerates_all_advertised_providers() -> None:
    data = _run_doctor_json_clean()
    providers = {p["name"] for p in data["llm_providers"]}
    expected = {"OpenAI", "Anthropic", "Azure OpenAI", "Ollama"}
    missing = sorted(expected - providers)
    assert not missing, (
        f"doctor --json failed to enumerate advertised LLM providers: missing {missing}. "
        f"Got: {sorted(providers)}. ADVERTISED_LLM_PROVIDERS must stay in sync with "
        "_check_llm_providers()."
    )


def test_doctor_json_provider_entries_have_env_vars_field() -> None:
    """Every provider entry surfaces the env-var names it reads."""
    data = _run_doctor_json_clean()
    for p in data["llm_providers"]:
        if p["name"] == "Judge Model":
            # Synthetic informational entry, not a real provider - has no
            # env vars of its own (it consumes the others' models).
            continue
        assert "env_vars" in p, f"provider {p['name']!r} entry missing env_vars: {p}"
        assert isinstance(p["env_vars"], list), p
        if p["name"] in {"OpenAI", "Anthropic", "Azure OpenAI", "Ollama"}:
            assert p["env_vars"], f"provider {p['name']!r} should declare at least one env var; got {p['env_vars']!r}"
            for var in p["env_vars"]:
                assert var.startswith("BELT_"), f"env var {var!r} for provider {p['name']!r} should use BELT_ prefix"


def test_doctor_json_provider_entries_have_suggestion_when_unconfigured() -> None:
    data = _run_doctor_json_clean()
    for p in data["llm_providers"]:
        if not p["ok"] and p["name"] != "Judge Model":
            assert p.get("suggestion"), f"provider {p['name']!r} is not configured but exposes no suggestion: {p}"


def test_advertised_providers_constant_matches_backend_subclasses() -> None:
    """The hardcoded enumeration must match the actual ``BaseJudgeBackend`` subclasses.

    Adding a backend without updating ADVERTISED_LLM_PROVIDERS would mean
    ``doctor --json`` silently ignores the new provider - caught here.
    """
    import inspect

    from belt.commands.doctor import ADVERTISED_LLM_PROVIDERS
    from belt.scorer.llm import backend

    code_providers = set()
    for _, obj in inspect.getmembers(backend):
        if inspect.isclass(obj) and issubclass(obj, backend.BaseJudgeBackend) and obj is not backend.BaseJudgeBackend:
            inst = obj.__new__(obj)
            code_providers.add(obj.provider_name(inst))

    advertised = set(ADVERTISED_LLM_PROVIDERS)
    missing_from_advertised = sorted(code_providers - advertised)
    extras_in_advertised = sorted(advertised - code_providers)
    assert (
        not missing_from_advertised
    ), f"ADVERTISED_LLM_PROVIDERS missing backend subclasses: {missing_from_advertised}"
    assert (
        not extras_in_advertised
    ), f"ADVERTISED_LLM_PROVIDERS lists providers without a backend: {extras_in_advertised}"
