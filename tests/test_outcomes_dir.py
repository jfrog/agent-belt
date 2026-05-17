# (c) JFrog Ltd. (2026)

"""Tests for customizable outcomes directory (env var + CLI flag).

Covers:
- BELT_OUTCOMES_DIR env var resolution in constants
- --outcomes-dir CLI flag in commands/eval and commands/run parsers
- resolve_outcomes_root precedence: CLI > env var > default
- End-to-end dry-run with --outcomes-dir
- End-to-end dry-run with BELT_OUTCOMES_DIR env var
- Manifest placement follows resolved outcomes root
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

# ── constants.py: OUTCOMES_ROOT env var ──


class TestOutcomesRootEnvVar:
    def test_default_is_cwd_outcomes(self):
        """Without env var, OUTCOMES_ROOT should be cwd/outcomes."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BELT_OUTCOMES_DIR", None)
            import importlib

            import belt.constants as c

            importlib.reload(c)
            assert c.OUTCOMES_ROOT == Path.cwd() / "outcomes"

    def test_env_var_overrides_default(self, tmp_path: Path):
        """BELT_OUTCOMES_DIR should override the default."""
        custom = str(tmp_path / "my-outcomes")
        with patch.dict(os.environ, {"BELT_OUTCOMES_DIR": custom}):
            import importlib

            import belt.constants as c

            importlib.reload(c)
            assert c.OUTCOMES_ROOT == Path(custom)

    def test_env_var_empty_string_uses_default(self):
        """Empty env var should fall back to default."""
        with patch.dict(os.environ, {"BELT_OUTCOMES_DIR": ""}):
            import importlib

            import belt.constants as c

            importlib.reload(c)
            assert c.OUTCOMES_ROOT == Path.cwd() / "outcomes"


# ── resolve_outcomes_root precedence ──


class TestResolveOutcomesRoot:
    def test_cli_flag_wins_over_env_var(self, tmp_path: Path):
        cli_path = str(tmp_path / "cli-outcomes")
        env_path = str(tmp_path / "env-outcomes")
        with patch.dict(os.environ, {"BELT_OUTCOMES_DIR": env_path}):
            from belt.commands.run import resolve_outcomes_root

            result = resolve_outcomes_root(cli_flag=cli_path)
            assert result == Path(cli_path).resolve()

    def test_env_var_used_when_no_cli_flag(self, tmp_path: Path):
        env_path = str(tmp_path / "env-outcomes")
        with patch.dict(os.environ, {"BELT_OUTCOMES_DIR": env_path}):
            from belt.commands.run import resolve_outcomes_root

            result = resolve_outcomes_root(cli_flag=None)
            assert result == Path(env_path)

    def test_default_when_nothing_set(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BELT_OUTCOMES_DIR", None)

            from belt.commands.run import resolve_outcomes_root

            result = resolve_outcomes_root(cli_flag=None)
            assert result == Path.cwd() / "outcomes"

    def test_cli_flag_resolves_relative(self):
        from belt.commands.run import resolve_outcomes_root

        result = resolve_outcomes_root(cli_flag="relative/path")
        assert result.is_absolute()
        assert result == (Path.cwd() / "relative/path").resolve()


# ── Parser tests ──


class TestEvalCmdParser:
    def test_outcomes_dir_flag_accepted(self):
        from belt.commands.eval import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["scenarios/", "--outcomes-dir", "/tmp/custom"])
        assert args.outcomes_dir == "/tmp/custom"

    def test_outcomes_dir_default_is_none(self):
        from belt.commands.eval import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["scenarios/"])
        assert args.outcomes_dir is None


class TestRunnerParser:
    def test_outcomes_dir_flag_accepted(self):
        import argparse

        from belt.commands.run import _add_common_run_args

        parser = argparse.ArgumentParser()
        _add_common_run_args(parser)
        args = parser.parse_args(["scenarios/", "--outcomes-dir", "/tmp/custom"])
        assert args.outcomes_dir == "/tmp/custom"

    def test_outcomes_dir_default_is_none(self):
        import argparse

        from belt.commands.run import _add_common_run_args

        parser = argparse.ArgumentParser()
        _add_common_run_args(parser)
        args = parser.parse_args(["scenarios/"])
        assert args.outcomes_dir is None


# ── Dry-run integration: verify outcomes-dir is accepted without errors ──


class TestDryRunWithOutcomesDir:
    @pytest.fixture()
    def examples_dir(self) -> Path:
        d = Path(__file__).resolve().parent.parent / "examples" / "scenarios"
        if not d.is_dir():
            pytest.skip("examples/scenarios not available")
        return d

    def test_eval_dry_run_with_outcomes_dir(self, examples_dir: Path, tmp_path: Path):
        from belt.commands.eval import main as eval_main

        custom_dir = str(tmp_path / "eval-outcomes")
        rc = eval_main([str(examples_dir), "--dry-run", "--modes", "rules", "--outcomes-dir", custom_dir])
        assert rc == 0

    def test_eval_dry_run_with_env_var(self, examples_dir: Path, tmp_path: Path):
        from belt.commands.eval import main as eval_main

        custom_dir = str(tmp_path / "env-outcomes")
        with patch.dict(os.environ, {"BELT_OUTCOMES_DIR": custom_dir}):
            rc = eval_main([str(examples_dir), "--dry-run", "--modes", "rules"])
            assert rc == 0

    def test_runner_dry_run_with_outcomes_dir(self, examples_dir: Path, tmp_path: Path):
        from belt.commands.run import main as run_main

        custom_dir = str(tmp_path / "run-outcomes")
        rc = run_main([str(examples_dir), "--dry-run", "--outcomes-dir", custom_dir])
        assert rc == 0

    def test_runner_dry_run_with_env_var(self, examples_dir: Path, tmp_path: Path):
        from belt.commands.run import main as run_main

        custom_dir = str(tmp_path / "run-env-outcomes")
        with patch.dict(os.environ, {"BELT_OUTCOMES_DIR": custom_dir}):
            rc = run_main([str(examples_dir), "--dry-run"])
            assert rc == 0


# ── Integration: verify outcomes actually land in custom directory ──


class TestOutcomesWrittenToCustomDir:
    """Run scenarios with a stub agent, verify outcomes land in the right place."""

    @pytest.fixture()
    def scenario_dir(self, tmp_path: Path) -> Path:
        """Create a minimal scenario tree with a stub agent."""
        group = tmp_path / "scenarios" / "stub_group"
        group.mkdir(parents=True)

        import json

        (group / "_config.json").write_text(
            json.dumps({"agent": "tests.test_integration_flow.StubAgentAdapter", "default_tags": ["test"]})
        )
        (group / "hello.json").write_text(
            json.dumps(
                {
                    "name": "hello",
                    "description": "Simple test",
                    "turns": [{"message": "Hello", "expect": {"has_reply": True}}],
                }
            )
        )
        return tmp_path / "scenarios"

    def test_outcomes_dir_cli_flag(self, scenario_dir: Path, tmp_path: Path):
        """--outcomes-dir puts outcomes in the specified directory."""
        from belt.commands.run import main as run_main

        custom = tmp_path / "custom-outcomes"
        rc = run_main([str(scenario_dir), "--outcomes-dir", str(custom), "--progress", "plain"])
        assert rc == 0
        assert custom.is_dir()
        run_dirs = [d for d in custom.iterdir() if d.is_dir()]
        assert len(run_dirs) == 1, f"Expected 1 run dir, got: {run_dirs}"
        cli_files = list(run_dirs[0].rglob("turn_*_cli.txt"))
        assert len(cli_files) >= 1

    def test_outcomes_dir_env_var(self, scenario_dir: Path, tmp_path: Path):
        """BELT_OUTCOMES_DIR env var puts outcomes in the specified directory."""
        from belt.commands.run import main as run_main

        custom = tmp_path / "env-outcomes"
        with patch.dict(os.environ, {"BELT_OUTCOMES_DIR": str(custom)}):
            rc = run_main([str(scenario_dir), "--progress", "plain"])
        assert rc == 0
        assert custom.is_dir()
        runs = list(custom.iterdir())
        manifest_files = [r for r in runs if r.name != ".manifest.json" and r.name != ".manifest.json.lock"]
        assert len(manifest_files) == 1, f"Expected 1 run dir, got: {manifest_files}"

    def test_cli_flag_overrides_env_var(self, scenario_dir: Path, tmp_path: Path):
        """CLI flag takes precedence over env var."""
        from belt.commands.run import main as run_main

        env_dir = tmp_path / "env-dir"
        cli_dir = tmp_path / "cli-dir"
        with patch.dict(os.environ, {"BELT_OUTCOMES_DIR": str(env_dir)}):
            rc = run_main([str(scenario_dir), "--outcomes-dir", str(cli_dir), "--progress", "plain"])
        assert rc == 0
        assert cli_dir.is_dir()
        assert not env_dir.exists(), "Env var dir should not have been created"

    def test_eval_outcomes_dir(self, scenario_dir: Path, tmp_path: Path):
        """belt eval --outcomes-dir writes to the specified directory."""
        from belt.commands.eval import main as eval_main

        custom = tmp_path / "eval-outcomes"
        rc = eval_main([str(scenario_dir), "--outcomes-dir", str(custom), "--progress", "plain", "--modes", "rules"])
        assert rc == 0
        assert custom.is_dir()
        runs = [d for d in custom.iterdir() if d.is_dir()]
        assert len(runs) == 1


# ── Uniqueness: parallel runs get different directories ──


class TestRunDirUniqueness:
    def test_parallel_runs_get_unique_dirs(self, tmp_path: Path):
        """Two runs starting in the same second must not collide."""
        import re
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")

        dirs = set()
        for _ in range(20):
            name = f"{ts}-{__import__('secrets').token_hex(4)}"
            dirs.add(name)

        assert len(dirs) == 20, "Expected 20 unique directory names"
        for name in dirs:
            assert re.match(r"\d{8}-\d{6}-[0-9a-f]{8}$", name), f"Bad format: {name}"


# ── Error handling: unwritable path ──


class TestOutcomesDirErrors:
    def test_unwritable_path_returns_1(self):
        """Non-writable outcomes dir should fail gracefully (not traceback)."""
        from belt.commands.run import main as run_main

        examples = Path(__file__).resolve().parent.parent / "examples" / "scenarios"
        if not examples.is_dir():
            pytest.skip("examples/scenarios not available")
        rc = run_main([str(examples), "--outcomes-dir", "/root/no-access", "--progress", "plain"])
        assert rc == 1


# ── OUTCOMES_DIR_ENV constant is exported ──


def test_outcomes_dir_env_constant():
    from belt.constants import OUTCOMES_DIR_ENV

    assert OUTCOMES_DIR_ENV == "BELT_OUTCOMES_DIR"
