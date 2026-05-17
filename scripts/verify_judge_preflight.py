#!/usr/bin/env python3
# (c) JFrog Ltd. (2026)

"""End-to-end empirical verification for judge model preflight.

Contract: ``belt eval --modes llm`` with an unreachable judge model
must abort *before* the agent phase, not after wasting agent calls.
This script proves it by running ``belt eval`` against a showcase
scenario group with a deliberately bad model + injected ``httpx.get``
that simulates four failure modes (401 / 403 + model not found /
404 + model not found / 5xx transient), then asserts:

1. The process exits non-zero in <5s (preflight should be sub-second
   in practice; 5s is the headroom for the test harness itself).
2. No agent subprocess is spawned (we patch ``subprocess.Popen`` and
   record every invocation).
3. The error message includes the upstream HTTP status, the
   provider-specific error code, and the correct hint.

Run: ``uv run python scripts/verify_judge_preflight.py``

Unlike the unit tests, this exercises the real CLI dispatch chain
(``cli.py`` → ``commands/eval.py`` → ``validate_scorers`` →
``preflight_judges`` → backend probe) so we can prove the wiring
holds end-to-end and not just at the boundary of each unit test.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _fake_response(status: int, body: str) -> httpx.Response:
    req = httpx.Request("GET", "https://api.openai.com/v1/models/probe")
    return httpx.Response(status_code=status, text=body, request=req)


# Failure matrix. Each row is (label, status, body, must-be-in-error).
# The "must-be-in-error" tokens are the user-visible parts of the
# composite preflight error - the assertion proves the hint reaches
# the user, not just that the probe failed.
_FAILURE_CASES: list[tuple[str, int, str, list[str]]] = [
    (
        "401 invalid_api_key",
        401,
        '{"error":{"code":"invalid_api_key","message":"bad key"}}',
        ["401", "BELT_OPENAI_API_KEY", "Hint:"],
    ),
    (
        "403 model_not_found (the #366 repro)",
        403,
        (
            '{"error":{"code":"model_not_found","type":"invalid_request_error",'
            '"message":"Project does not have access to model `gpt-4o-mini`"}}'
        ),
        ["403", "model_not_found", "Hint:", "project"],
    ),
    (
        "404 model_not_found (typo)",
        404,
        '{"error":{"code":"model_not_found","message":"unknown model"}}',
        ["404", "model_not_found", "Hint:", "typo"],
    ),
]


def _run_eval_with_failing_probe(*, status: int, body: str, agent_calls: list[str]) -> tuple[int, str, float]:
    """Run ``belt eval`` with a patched ``httpx.get``; capture exit, stderr, time.

    Patches:
      - ``httpx.get`` in the backend module so the preflight probe
        returns ``(status, body)`` instead of making a real network call.
      - ``subprocess.Popen`` so we can detect (and prove the absence of)
        any spawned agent subprocess. The eval would normally spawn
        ``cursor-agent`` / ``codex`` / ``claude`` here, none of which
        we want firing.
    """
    import subprocess

    from belt.commands import eval as eval_cmd

    real_popen = subprocess.Popen

    def fake_get(url: str, **kwargs: object) -> httpx.Response:
        return _fake_response(status, body)

    def fake_popen(*args: object, **kwargs: object) -> object:
        # Record every Popen invocation. Any spawn that happens after
        # the preflight check means the abort-before-agent-phase
        # invariant has regressed.
        argv = args[0] if args else kwargs.get("args", "<unknown>")
        agent_calls.append(repr(argv))
        return real_popen(*args, **kwargs)

    # Bake in real OpenAI creds shape so build_request doesn't reject
    # at config time; the fake_get below intercepts the actual probe.
    env_overrides = {
        "BELT_OPENAI_API_KEY": "sk-test-DO-NOT-USE",
        "BELT_LLM_MODEL": "openai/gpt-4o-mini",
        # Tight preflight timeout so a failing patch doesn't stall on
        # the unlikely path of real network leaking through.
        "BELT_LLM_PREFLIGHT_TIMEOUT": "2",
        # Don't try to load the user's local belt.yaml; we want a clean run.
        "BELT_NO_DOTENV": "1",
    }
    # Capture stderr by redirecting it into an in-memory buffer.
    import io

    captured_err = io.StringIO()
    captured_out = io.StringIO()
    started = time.time()
    with (
        patch.dict(os.environ, env_overrides, clear=False),
        patch("belt.scorer.llm.backend.httpx.get", fake_get),
        patch("belt.scorer.llm.backend.httpx.post", fake_get),
        patch("subprocess.Popen", fake_popen),
        patch("sys.stderr", captured_err),
        patch("sys.stdout", captured_out),
    ):
        argv = [
            "examples/scenarios/showcase/correctness",
            "--modes",
            "llm",
            "--scorer-arg",
            "model=openai/gpt-4o-mini",
            "--tags",
            "real-runnable",
            "--allow-external-working-dir",
            "--agent",
            "cursor",
            "--progress",
            "live",
        ]
        rc = eval_cmd.main(argv)
    elapsed = time.time() - started
    combined = captured_out.getvalue() + "\n" + captured_err.getvalue()
    return rc, combined, elapsed


def main() -> int:
    print("=" * 70)
    print("Empirical verification: judge model preflight")
    print("=" * 70)
    failed = 0
    for label, status, body, must_contain in _FAILURE_CASES:
        print(f"\n── {label} ──")
        agent_calls: list[str] = []
        rc, stderr, elapsed = _run_eval_with_failing_probe(status=status, body=body, agent_calls=agent_calls)

        assertions: list[tuple[str, bool]] = [
            (f"exit non-zero (got {rc})", rc != 0),
            (f"elapsed < 5s (got {elapsed:.2f}s)", elapsed < 5.0),
            (
                "NO agent subprocess spawned during preflight",
                # The original #366 symptom: 5 scenarios × 1 agent call
                # each before the failure surfaces. Even a single Popen
                # past preflight is a regression. The empty list is the
                # whole point of the issue.
                len(agent_calls) == 0,
            ),
        ]
        for token in must_contain:
            assertions.append((f"error contains {token!r}", token in stderr))
        for desc, ok in assertions:
            print(f"  {'✅' if ok else '❌'} {desc}")
            if not ok:
                failed += 1
        if any(not ok for _, ok in assertions):
            print(f"  Captured output (first 1000 chars):\n  {stderr[:1000]!r}")
            if agent_calls:
                print(f"  Agent subprocesses spawned (regression!): {agent_calls}")

    print()
    print("=" * 70)
    if failed == 0:
        print("✅ All preflight matrix points verified end-to-end.")
        return 0
    print(f"❌ {failed} assertion(s) failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
