# (c) JFrog Ltd. (2026)

"""On-disk collectors for ``benchmark_card.collect``.

Covers scenario file hashing, per-process provenance lookups, and the
per-group runtime-info sidecar dedup that the card relies on for its
``agents`` section.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from belt.benchmark_card import (
    collect_belt_provenance,
    collect_host_provenance,
    collect_invocation,
    hash_scenario_files,
)
from belt.benchmark_card.collect import collect_runtime_info_sidecars
from belt.constants import RUNTIME_INFO_FILE


class TestHashScenarioFiles:
    def test_hashes_are_deterministic_and_sorted(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        f1 = tmp_path / "a" / "x.json"
        f2 = tmp_path / "b" / "y.json"
        f1.write_text("{}")
        f2.write_text('{"foo":1}')
        out = hash_scenario_files(tmp_path, [f2, f1])
        assert [s.relpath for s in out] == ["a/x.json", "b/y.json"]
        assert all(len(s.sha256) == 64 for s in out)

    def test_unreadable_paths_are_skipped(self, tmp_path: Path) -> None:
        good = tmp_path / "good.json"
        good.write_text("{}")
        missing = tmp_path / "missing.json"
        out = hash_scenario_files(tmp_path, [missing, good])
        assert [s.relpath for s in out] == ["good.json"]


class TestCollectors:
    def test_belt_provenance_uses_metadata(self) -> None:
        prov = collect_belt_provenance()
        assert prov.version
        assert prov.install_kind in {"wheel", "editable", "unknown"}

    def test_host_provenance_records_python_runtime(self) -> None:
        host = collect_host_provenance()
        assert host.python_implementation
        assert host.python_version
        assert host.package_versions.get("pydantic")

    def test_invocation_uses_safe_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Set both an allow-listed name and a clearly-secret one. Only the
        # allow-list passes through; the secret-shaped value never appears.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-this-must-not-leak")
        inv = collect_invocation(["belt", "eval", "x"], {"modes": "rules"}, "/tmp")
        assert inv.argv[0] == "belt"
        assert isinstance(inv.env, dict)
        assert "sk-this-must-not-leak" not in json.dumps(inv.env)


def _write_sidecar(scn_dir: Path, *, group: str, agent_name: str, adapter: str, version: str) -> None:
    """Helper: write a nested-shape ``_runtime_info.json`` sidecar.

    The sidecar is intentionally unversioned; only the versioned card
    it feeds carries ``schema_version``.
    """
    scn_dir.mkdir(parents=True, exist_ok=True)
    (scn_dir / RUNTIME_INFO_FILE).write_text(
        json.dumps(
            {
                "group": group,
                "agent": {
                    "name": agent_name,
                    "adapter_class": adapter,
                    "args": {},
                    "auth_signals": [],
                },
                "cli": {"binary_path": None, "version": version},
            }
        )
    )


class TestCollectRuntimeInfoSidecars:
    """Per-group dedup is the contract that lets multi-agent runs surface
    cleanly on the card. Every test here pins one property of that
    contract; together they ensure a multi-agent run never collapses
    distinct groups into one entry, never duplicates within a group,
    and tolerates a missing or malformed sidecar without aborting the
    aggregate.
    """

    def test_multi_group_multi_agent_yields_one_record_per_group(self, tmp_path: Path) -> None:
        # Two distinct groups, each running a different adapter, each with
        # multiple scenarios. The collector must produce exactly two
        # AgentProvenance records (one per group) - never one merged
        # record, never four (one per scenario).
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_sidecar(
            run_dir / "groupA" / "scn1", group="groupA", agent_name="cursor", adapter="CursorAdapter", version="1.0"
        )
        _write_sidecar(
            run_dir / "groupA" / "scn2", group="groupA", agent_name="cursor", adapter="CursorAdapter", version="1.0"
        )
        _write_sidecar(
            run_dir / "groupB" / "scn1", group="groupB", agent_name="claude", adapter="ClaudeAdapter", version="2.0"
        )
        _write_sidecar(
            run_dir / "groupB" / "scn2", group="groupB", agent_name="claude", adapter="ClaudeAdapter", version="2.0"
        )

        out = collect_runtime_info_sidecars(run_dir)
        assert [a.group for a in out] == ["groupA", "groupB"]  # sorted by group
        assert out[0].agent.name == "cursor"
        assert out[0].cli.version == "1.0"
        assert out[1].agent.name == "claude"
        assert out[1].cli.version == "2.0"

    def test_first_sighting_wins_within_a_group(self, tmp_path: Path) -> None:
        # Scenario directories sort lexicographically; ``scn1`` is read
        # before ``scn2``. If the bug ever flips to "last sighting wins"
        # this assertion catches it.
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_sidecar(run_dir / "g" / "scn1", group="g", agent_name="cursor", adapter="A", version="1.0")
        _write_sidecar(run_dir / "g" / "scn2", group="g", agent_name="cursor", adapter="A", version="2.0-DIFFERENT")
        out = collect_runtime_info_sidecars(run_dir)
        assert len(out) == 1
        assert out[0].cli.version == "1.0"

    def test_missing_sidecars_yield_empty_list_not_error(self, tmp_path: Path) -> None:
        # No sidecars at all - aggregator must still produce a card with
        # an empty agents section, never raise.
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        assert collect_runtime_info_sidecars(run_dir) == []

    def test_malformed_sidecar_is_skipped_other_groups_still_collected(self, tmp_path: Path) -> None:
        # A best-effort capture must not let one bad sidecar bury the
        # rest. Verifies the broad ``except Exception`` in the collector
        # actually swallows per-sidecar failures.
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        bad = run_dir / "bad" / "scn"
        bad.mkdir(parents=True)
        (bad / RUNTIME_INFO_FILE).write_text("{not valid json")
        _write_sidecar(run_dir / "good" / "scn", group="good", agent_name="cursor", adapter="A", version="1.0")
        out = collect_runtime_info_sidecars(run_dir)
        assert [a.group for a in out] == ["good"]
