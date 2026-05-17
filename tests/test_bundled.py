# (c) JFrog Ltd. (2026)

"""Unit tests for ``belt._bundled`` and the ``--bundled`` CLI flag.

The launch-readiness audit in #385 (1.1, 1.5) found that the
post-quickstart "next steps" rendered absolute paths into the
``site-packages`` tree and the README advertised a command form that
breaks for any ``pip install agent-belt`` user. ``--bundled`` is the
portable surface that fixes both; these tests pin the contract so a
regression here would re-introduce the broken first-run flow.
"""

from __future__ import annotations

import pytest

from belt import _bundled


def test_bundled_scenarios_root_finds_in_tree_examples():
    """Editable install / source clone path: in-tree ``examples/scenarios/``."""
    root = _bundled.bundled_scenarios_root()
    assert root is not None, "Bundled scenarios should be reachable from a source checkout."
    assert (root / "showcase").is_dir()


def test_bundled_groups_lists_showcase():
    groups = _bundled.bundled_groups()
    assert "showcase" in groups, f"Expected 'showcase' in bundled groups, got {groups!r}"
    # ``_fixtures`` and similar leading-underscore dirs are excluded so
    # ``--bundled`` only suggests user-facing groups.
    assert all(not g.startswith("_") for g in groups)


def test_resolve_bundled_path_root_when_empty():
    """``--bundled`` with no argument resolves to the scenarios root."""
    root = _bundled.resolve_bundled_path("")
    assert root is not None
    assert (root / "showcase").is_dir()


def test_resolve_bundled_path_showcase():
    showcase = _bundled.resolve_bundled_path("showcase")
    assert showcase is not None
    assert showcase.is_dir()
    assert showcase.name == "showcase"


def test_resolve_bundled_path_deep_group():
    """Deeper paths like ``showcase/correctness`` resolve to a group dir."""
    target = _bundled.resolve_bundled_path("showcase/correctness")
    assert target is not None
    assert target.is_dir()
    # ``_config.json`` is the contract marker for a group directory.
    assert (target / "_config.json").is_file()


def test_resolve_bundled_path_missing_returns_none():
    assert _bundled.resolve_bundled_path("nope-not-a-group") is None
    assert _bundled.resolve_bundled_path("showcase/does-not-exist") is None


def test_resolve_bundled_path_escape_attempt_returns_none():
    """``--bundled ../../etc/passwd`` must not escape the bundled tree.

    The resolver normalises with ``Path.resolve()`` and verifies the
    result stays inside the bundled root - covering the obvious
    traversal vector that a user could type by mistake or supply via
    automation.
    """
    assert _bundled.resolve_bundled_path("../../etc") is None
    assert _bundled.resolve_bundled_path("../..") is None


class TestEvalBundledFlag:
    def test_bundled_replaces_positional_path(self):
        from belt.commands.eval import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--bundled", "showcase"])
        assert args.bundled == "showcase"
        assert args.path is None

    def test_bundled_with_no_value(self):
        """``--bundled`` alone (no value) means "the whole bundled tree"."""
        from belt.commands.eval import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--bundled"])
        assert args.bundled == ""

    def test_positional_path_still_works(self):
        """The original positional remains the supported form for source clones."""
        from belt.commands.eval import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["examples/scenarios/showcase"])
        assert args.bundled is None
        assert args.path == "examples/scenarios/showcase"

    def test_bundled_and_path_both_set_errors(self, capsys):
        """Passing both is a user error (ambiguous intent), surfaced at argparse exit."""
        from belt.commands.eval import main

        with pytest.raises(SystemExit) as exc:
            main(["--bundled", "showcase", "/some/path"])
        assert exc.value.code != 0
        captured = capsys.readouterr()
        assert "mutually exclusive" in captured.err

    def test_no_path_no_bundled_errors(self, capsys):
        """At least one of the two scenario sources must be set."""
        from belt.commands.eval import main

        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code != 0
        captured = capsys.readouterr()
        assert "--bundled" in captured.err or "scenarios path" in captured.err

    def test_bundled_unknown_group_errors(self, capsys):
        """Unknown ``--bundled`` value enumerates the available groups."""
        from belt.commands.eval import main

        with pytest.raises(SystemExit) as exc:
            main(["--bundled", "does-not-exist"])
        assert exc.value.code != 0
        captured = capsys.readouterr()
        # The error should at minimum name the flag and list valid options.
        assert "--bundled" in captured.err
        assert "showcase" in captured.err


class TestRunBundledFlag:
    """``belt run --bundled`` parity - same flag, same semantics."""

    def test_bundled_with_value(self):
        import argparse

        # ``run.main`` parses internally; reach the parser the same way
        # ``eval.main`` does for parity-style checking.
        parser = argparse.ArgumentParser()
        from belt.commands.run import _add_common_run_args

        _add_common_run_args(parser)
        args = parser.parse_args(["--bundled", "showcase"])
        assert args.bundled == "showcase"
        assert args.path is None
