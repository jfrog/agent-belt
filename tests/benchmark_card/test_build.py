# (c) JFrog Ltd. (2026)

"""End-to-end ``build_card`` tests over on-disk fixtures.

These exercise the full collector pipeline (run_meta + per-scenario
sidecars + score.json + run_fixtures.json) and verify that
``build_card`` projects them into a well-formed
:class:`BenchmarkCard`. Per-collector behaviour is covered in
``test_collect.py``; this suite focuses on the assembly contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from belt.benchmark_card import build_card
from belt.constants import RUN_META_FILE, RUNTIME_INFO_FILE, SCHEMA_VERSION, SCORE_FILE


class TestBuildCard:
    def test_builds_from_disk_fixtures(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "20260101-000000-abc12345"
        run_dir.mkdir()
        run_meta = {
            "schema_version": SCHEMA_VERSION,
            "started_at": "2026-01-01T00:00:00Z",
            "scenarios_root": str(tmp_path / "scn"),
            "workspace": "/tmp",
            "env": {},
            "belt": {"version": "9.9.9", "install_kind": "wheel"},
            "host": {
                "os": "Linux 6.0",
                "machine": "x86_64",
                "python_version": "3.12.0",
                "python_implementation": "CPython",
                "package_versions": {"belt": "9.9.9"},
            },
            "invocation": {
                "argv": ["belt", "eval"],
                "parsed_args": {"modes": "rules"},
                "cwd": "/tmp",
                "env": {},
            },
            "scenarios": {
                "scenarios_root": str(tmp_path / "scn"),
                "selected_groups": [],
                "selected_tags": [],
                "excluded_tags": [],
                "scenario_files": [],
            },
            "runtime": {"workers": 2, "trials": 1, "streaming": True, "scenario_delay_s": 0.0},
            "scoring": {"modes": ["rules"], "thresholds": {}},
        }
        (run_dir / RUN_META_FILE).write_text(json.dumps(run_meta))

        # One group + one scenario with a runtime sidecar and a score.json.
        group_dir = run_dir / "showcase"
        scn_dir = group_dir / "scenario_one"
        scn_dir.mkdir(parents=True)
        (scn_dir / RUNTIME_INFO_FILE).write_text(
            json.dumps(
                {
                    # Sidecar is unversioned by design.
                    "group": "showcase",
                    "agent": {
                        "name": "echo",
                        "adapter_class": "EchoAgentAdapter",
                        "args": {"model": "gpt-4"},
                        "auth_signals": ["env:ECHO_TOKEN"],
                    },
                    "cli": {
                        "binary_path": "/usr/local/bin/echo-agent",
                        "version": "0.0.1",
                    },
                }
            )
        )
        (scn_dir / SCORE_FILE).write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "scores": {
                        "llm": {
                            "dimensions": ["correctness"],
                            "usage": {
                                "backends": [
                                    {
                                        "provider": "openai",
                                        "model": "gpt-4",
                                        "base_url": "https://api.openai.com",
                                    }
                                ]
                            },
                        }
                    },
                }
            )
        )

        # Fixture provenance file from setup_groups.
        (run_dir / "run_fixtures.json").write_text(
            json.dumps(
                [
                    {
                        "group": "showcase",
                        "working_dir": "/tmp/fixture",
                        "tracked": True,
                        "git_sha": "abcdef0123456789",
                        "git_ref": "HEAD",
                        "auto_initialized": False,
                        "dirty_files": 0,
                    }
                ]
            )
        )

        results = {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "overall_pass": True,
            "thresholds_passed": True,
            "cost_timing": {
                "agent_cost_usd": 0.01,
                "judge_cost_usd": 0.003,
                "total_cost_usd": 0.013,
                "total_seconds": 12.5,
                "mean_seconds": 12.5,
                "scenarios": [{"scenario": "showcase/echo", "agent_cost_usd": 0.01}],
            },
        }

        card = build_card(run_dir, results)

        assert card.run_id == run_dir.name
        assert card.belt.version == "9.9.9"
        assert card.summary.total == 1 and card.summary.passed == 1
        assert card.summary.pass_rate == 1.0
        assert card.cost_timing.agent_cost_usd == pytest.approx(0.01)
        assert card.cost_timing.judge_cost_usd == pytest.approx(0.003)
        assert card.cost_timing.total_cost_usd == pytest.approx(0.013)
        assert len(card.agents) == 1 and card.agents[0].cli.version == "0.0.1"
        assert card.agents[0].agent.name == "echo"
        assert card.agents[0].agent.args == {"model": "gpt-4"}
        assert card.agents[0].agent.auth_signals == ["env:ECHO_TOKEN"]
        assert len(card.scoring.judges) == 1
        assert card.scoring.judges[0].model == "gpt-4"
        assert len(card.fixtures) == 1
        assert card.fixtures[0].git_sha == "abcdef0123456789"

    def test_missing_run_meta_yields_safe_defaults(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "20260101-000000-deadbeef"
        run_dir.mkdir()
        card = build_card(run_dir, {})
        assert card.belt.install_kind == "unknown"
        assert card.summary.total == 0
        # ``started_at`` falls back to the directory's ``mtime`` (a real
        # UTC timestamp) when ``run_meta.json`` has no recorded value.
        # The run-dir name is local-time-encoded and cannot be safely
        # promoted to UTC, so the fallback uses ``mtime`` instead.
        assert card.started_at.endswith("Z")
        assert "T" in card.started_at

    def test_started_at_is_taken_from_run_meta_not_directory_name(self, tmp_path: Path) -> None:
        # The run-dir name is local-time encoded; the canonical UTC
        # timestamp lives in ``run_meta.json`` and must take precedence.
        run_dir = tmp_path / "20260101-000000-deadbeef"
        run_dir.mkdir()
        recorded = "2026-06-15T12:34:56Z"
        (run_dir / RUN_META_FILE).write_text(json.dumps({"started_at": recorded}))
        card = build_card(run_dir, {})
        assert card.started_at == recorded

    def test_modes_are_read_from_scoring_block(self, tmp_path: Path) -> None:
        # ``--modes`` is parsed by ``eval``/``score``, never by ``run``;
        # the card must reflect what the user actually selected by
        # reading the persisted ``scoring.modes`` block, not by guessing
        # from ``run.args``.
        run_dir = tmp_path / "20260101-000000-deadbeef"
        run_dir.mkdir()
        (run_dir / RUN_META_FILE).write_text(
            json.dumps(
                {
                    "started_at": "2026-01-01T00:00:00Z",
                    "scoring": {"modes": ["rules", "llm"], "thresholds": {}},
                }
            )
        )
        card = build_card(run_dir, {})
        assert card.scoring.modes == ["rules", "llm"]
