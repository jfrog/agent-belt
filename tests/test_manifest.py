# (c) JFrog Ltd. (2026)

"""Tests for manifest - concurrent run tracking and orphan cleanup."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from belt.manifest import Manifest


def _grp(agent: str = "test", group_id: str = "") -> dict:
    """Helper to build a group entry dict for register_run."""
    d: dict = {"agent": agent}
    if group_id:
        d["group_id"] = group_id
    return d


@pytest.fixture()
def manifest_path(tmp_path: Path) -> Path:
    return tmp_path / ".manifest.json"


@pytest.fixture()
def manifest(manifest_path: Path) -> Manifest:
    return Manifest(path=manifest_path)


# ── _is_pid_alive ──


def test_is_pid_alive_current_process() -> None:
    assert Manifest._is_pid_alive(os.getpid()) is True


def test_is_pid_alive_dead_pid() -> None:
    assert Manifest._is_pid_alive(999_999_999) is False


def test_is_pid_alive_permission_error() -> None:
    with patch("belt.manifest.os.kill", side_effect=PermissionError):
        assert Manifest._is_pid_alive(12345) is True


# ── Fresh manifest ──


def test_empty_manifest(manifest: Manifest) -> None:
    assert manifest.latest_run is None
    assert manifest.runs == []


def test_register_and_read_back(manifest_path: Path) -> None:
    m = Manifest(path=manifest_path)
    m.register_run(
        1234,
        {"group_a": _grp("test", "aid-1"), "group_b": _grp("test", "aid-2")},
        "outcomes/run1",
    )

    m2 = Manifest(path=manifest_path)
    assert m2.latest_run == "outcomes/run1"
    assert len(m2.runs) == 1
    assert m2.runs[0]["pid"] == 1234
    assert m2.runs[0]["run_dir"] == "outcomes/run1"
    assert len(m2.runs[0]["groups"]) == 2


def test_unregister_run(manifest_path: Path) -> None:
    m = Manifest(path=manifest_path)
    m.register_run(1111, {"g": _grp("test", "a1")}, "outcomes/r1")
    m.register_run(2222, {"g": _grp("test", "a2")}, "outcomes/r2")

    m.unregister_run(1111)

    m2 = Manifest(path=manifest_path)
    assert len(m2.runs) == 1
    assert m2.runs[0]["pid"] == 2222


def test_unregister_last_run_removes_key(manifest_path: Path) -> None:
    m = Manifest(path=manifest_path)
    m.register_run(1111, {"g": _grp("test", "a1")}, "outcomes/r1")
    m.unregister_run(1111)

    raw = json.loads(manifest_path.read_text())
    assert "runs" not in raw


# ── Concurrent runs ──


def test_concurrent_register(manifest_path: Path) -> None:
    m = Manifest(path=manifest_path)
    m.register_run(1111, {"g1": _grp("test", "a1")}, "outcomes/r1")
    m.register_run(2222, {"g2": _grp("test", "a2")}, "outcomes/r2")

    assert len(m.runs) == 2
    assert m.latest_run == "outcomes/r2"


def test_cleanup_skips_live_run(manifest_path: Path) -> None:
    """A run whose PID is alive should NOT be cleaned up."""
    m = Manifest(path=manifest_path)
    live_pid = os.getpid()
    m.register_run(live_pid, {"g": _grp("test", "aid-live")}, "outcomes/r1")

    deleted_entries: list[dict] = []
    count = m.cleanup_orphans(lambda entry: deleted_entries.append(entry))

    assert count == 0
    assert deleted_entries == []
    assert len(m.runs) == 1


def test_cleanup_deletes_dead_run(manifest_path: Path) -> None:
    """A run whose PID is dead should be cleaned up."""
    m = Manifest(path=manifest_path)
    dead_pid = 999_999_999
    m.register_run(
        dead_pid,
        {"g1": _grp("test", "aid-1"), "g2": _grp("test", "aid-2")},
        "outcomes/r1",
    )

    deleted_entries: list[dict] = []
    count = m.cleanup_orphans(lambda entry: deleted_entries.append(entry))

    assert count == 2
    groups = {e["group"] for e in deleted_entries}
    assert groups == {"g1", "g2"}
    assert m.runs == []


def test_cleanup_mixed_live_and_dead(manifest_path: Path) -> None:
    """With both live and dead runs, only dead ones are cleaned."""
    m = Manifest(path=manifest_path)
    m.register_run(os.getpid(), {"live_g": _grp("test", "aid-live")}, "outcomes/live")
    m.register_run(999_999_999, {"dead_g": _grp("test", "aid-dead")}, "outcomes/dead")

    deleted_entries: list[dict] = []
    count = m.cleanup_orphans(lambda entry: deleted_entries.append(entry))

    assert count == 1
    assert deleted_entries[0]["group"] == "dead_g"
    assert len(m.runs) == 1
    assert m.runs[0]["pid"] == os.getpid()


def test_cleanup_tolerates_delete_failure(manifest_path: Path) -> None:
    """If delete_fn raises for one group, others still get processed."""
    m = Manifest(path=manifest_path)
    m.register_run(
        999_999_999,
        {"g1": _grp("test", "aid-1"), "g2": _grp("test", "aid-2")},
        "outcomes/r1",
    )

    call_count = 0

    def flaky_delete(entry: dict) -> None:
        nonlocal call_count
        call_count += 1
        if entry.get("group") == "g1":
            raise RuntimeError("API error")

    count = m.cleanup_orphans(flaky_delete)
    assert call_count == 2
    assert count == 1  # only g2 succeeded
    assert m.runs == []  # dead run entry still removed


def test_cleanup_no_runs(manifest: Manifest) -> None:
    count = manifest.cleanup_orphans(lambda entry: None)
    assert count == 0


# ── Corrupt/edge cases ──


def test_corrupt_manifest_file(manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("not json{{{")

    m = Manifest(path=manifest_path)
    assert m.runs == []
    assert m.latest_run is None


def test_manifest_file_does_not_exist(tmp_path: Path) -> None:
    m = Manifest(path=tmp_path / "nonexistent" / ".manifest.json")
    assert m.runs == []


def test_unrecognised_top_level_shape_collapses_to_empty(manifest_path: Path) -> None:
    """An on-disk manifest with an unrecognised top-level shape (e.g. a flat
    ``{"groups": {...}}`` map instead of the expected ``runs`` list) must not
    crash the reader. ``_validate`` keeps unknown top-level keys but
    ``cleanup_orphans`` only acts on the ``runs`` list, so the manifest
    behaves as if empty until the next ``register_run`` call rewrites it.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "latest_run": "outcomes/old-run",
                "groups": {"g1": "gid-1", "g2": "gid-2"},
            }
        )
    )

    m = Manifest(path=manifest_path)
    assert m.runs == []
    deleted = m.cleanup_orphans(lambda entry: None)
    assert deleted == 0


# ── Persistence round-trip ──


def test_concurrent_unregister_no_stale_overwrite(manifest_path: Path) -> None:
    """Two Manifest instances operating concurrently should not overwrite each other.

    Simulates: Run A registers, Run B registers (separate instance), Run A
    unregisters - Run B's entry must survive because _refresh() re-reads disk.
    """
    run_a = Manifest(path=manifest_path)
    run_a.register_run(1111, {"ga": _grp("test", "a1")}, "outcomes/ra")

    run_b = Manifest(path=manifest_path)
    run_b.register_run(2222, {"gb": _grp("test", "a2")}, "outcomes/rb")

    run_a.unregister_run(1111)

    final = Manifest(path=manifest_path)
    assert len(final.runs) == 1
    assert final.runs[0]["pid"] == 2222


def test_concurrent_register_no_stale_overwrite(manifest_path: Path) -> None:
    """Register from a stale instance should not lose entries added by others."""
    m1 = Manifest(path=manifest_path)
    m1.register_run(1111, {"g1": _grp("test", "a1")}, "outcomes/r1")

    m2 = Manifest(path=manifest_path)
    m2.register_run(2222, {"g2": _grp("test", "a2")}, "outcomes/r2")

    m1.register_run(3333, {"g3": _grp("test", "a3")}, "outcomes/r3")

    final = Manifest(path=manifest_path)
    assert len(final.runs) == 3
    pids = {r["pid"] for r in final.runs}
    assert pids == {1111, 2222, 3333}


def test_full_lifecycle(manifest_path: Path) -> None:
    """register → cleanup (skip live) → unregister → verify empty."""
    m = Manifest(path=manifest_path)
    pid = os.getpid()

    m.register_run(pid, {"g1": _grp("test", "a1"), "g2": _grp("test", "a2")}, "outcomes/r1")
    assert len(m.runs) == 1

    count = m.cleanup_orphans(lambda entry: None)
    assert count == 0
    assert len(m.runs) == 1

    m.unregister_run(pid)
    assert m.runs == []

    m2 = Manifest(path=manifest_path)
    assert m2.runs == []


# ── File locking ──


def test_concurrent_register_via_threads(manifest_path: Path) -> None:
    """Multiple threads registering simultaneously should not lose entries."""
    import threading

    errors: list[Exception] = []

    def register_run(pid: int) -> None:
        try:
            m = Manifest(path=manifest_path)
            m.register_run(pid, {f"g{pid}": _grp("test", f"a{pid}")}, f"outcomes/r{pid}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=register_run, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    final = Manifest(path=manifest_path)
    assert len(final.runs) == 10
    pids = {r["pid"] for r in final.runs}
    assert pids == set(range(10))
