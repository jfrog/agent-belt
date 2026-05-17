# (c) JFrog Ltd. (2026)

"""Tests for belt quickstart."""

from __future__ import annotations

from unittest.mock import patch

from belt.agent.base import AgentNotAvailableError
from belt.commands.quickstart import _QUICKSTART_GROUP, _QUICKSTART_SCENARIO, _find_examples_dir, _validate_agent, main


class TestFindExamplesDir:
    def test_finds_quickstart_group(self):
        # Whether running from the repo (editable install) or from a wheel install,
        # the resolver MUST find a directory that actually contains the quickstart
        # scenario - otherwise quickstart is broken.
        result = _find_examples_dir()
        assert result is not None, "quickstart scenarios must be discoverable"
        assert result.is_dir()
        assert (result / _QUICKSTART_GROUP / f"{_QUICKSTART_SCENARIO}.json").is_file()

    def test_works_from_arbitrary_cwd(self, tmp_path, monkeypatch):
        # Pip-installed users invoke ``belt quickstart`` from their project
        # root, which has no ``examples/`` directory. The resolver must still
        # find scenarios via ``importlib.resources`` or the package-relative
        # editable-install fallback.
        monkeypatch.chdir(tmp_path)
        result = _find_examples_dir()
        assert result is not None
        assert (result / _QUICKSTART_GROUP / f"{_QUICKSTART_SCENARIO}.json").is_file()


class TestValidateAgent:
    def test_unknown_agent(self, capsys):
        from rich.console import Console

        console = Console(file=open("/dev/null", "w"))
        with patch("belt.commands.quickstart.available_agents", return_value=["claude-code"]):
            result = _validate_agent("nonexistent", console)
        assert result is False

    def test_available_agent(self, capsys):
        from rich.console import Console

        console = Console()
        with (
            patch("belt.commands.quickstart.available_agents", return_value=["test-agent"]),
            patch("belt.commands.quickstart.get_agent_class") as mock_cls,
        ):
            mock_cls.return_value.check_available.return_value = None
            mock_cls.return_value.display_info.return_value = "Test v1"
            result = _validate_agent("test-agent", console)
        assert result is True

    def test_unavailable_agent(self, capsys):
        from rich.console import Console

        console = Console()
        with (
            patch("belt.commands.quickstart.available_agents", return_value=["test-agent"]),
            patch("belt.commands.quickstart.get_agent_class") as mock_cls,
        ):
            mock_cls.return_value.check_available.side_effect = AgentNotAvailableError(
                "test-agent", "not installed", "npm install it"
            )
            result = _validate_agent("test-agent", console)
        assert result is False

    def test_generic_exception(self, capsys):
        from rich.console import Console

        console = Console()
        with (
            patch("belt.commands.quickstart.available_agents", return_value=["test-agent"]),
            patch("belt.commands.quickstart.get_agent_class") as mock_cls,
        ):
            mock_cls.return_value.check_available.side_effect = RuntimeError("network timeout")
            result = _validate_agent("test-agent", console)
        assert result is False

    def test_display_info_throws_still_succeeds(self, capsys):
        from rich.console import Console

        console = Console()
        with (
            patch("belt.commands.quickstart.available_agents", return_value=["test-agent"]),
            patch("belt.commands.quickstart.get_agent_class") as mock_cls,
        ):
            mock_cls.return_value.check_available.return_value = None
            mock_cls.return_value.display_info.side_effect = RuntimeError("boom")
            result = _validate_agent("test-agent", console)
        assert result is True


def _make_quickstart_dir(tmp_path):
    """Create a fake examples/scenarios/showcase/correctness/ tree."""
    scenarios_dir = tmp_path / "examples" / "scenarios"
    group_dir = scenarios_dir / _QUICKSTART_GROUP
    group_dir.mkdir(parents=True)
    (group_dir / "_config.json").write_text('{"agent": "cursor"}')
    (group_dir / f"{_QUICKSTART_SCENARIO}.json").write_text(
        '{"name":"correctness_basic","description":"test","turns":[{"message":"hi"}]}'
    )
    return scenarios_dir


class TestQuickstartMain:
    def test_no_agents_available(self, capsys):
        with patch("belt.commands.quickstart.available_agents", return_value=[]):
            rc = main([])
        assert rc == 1

    def test_unknown_agent_name(self, capsys):
        with patch("belt.commands.quickstart.available_agents", return_value=["claude-code"]):
            rc = main(["nonexistent"])
        assert rc == 1

    def test_auto_detect_first_available(self, capsys, tmp_path):
        """When no agent arg given, picks first available and runs the showcase quickstart."""
        scenarios_dir = _make_quickstart_dir(tmp_path)

        def fake_get(name):
            def _check(cls_self):
                if name != "second-agent":
                    raise AgentNotAvailableError(name, "missing")

            cls = type(
                "Agent",
                (),
                {
                    "check_available": classmethod(lambda c: _check(c)),
                    "display_info": classmethod(lambda c: "ok"),
                },
            )
            return cls

        with (
            patch("belt.commands.quickstart.available_agents", return_value=["first-agent", "second-agent"]),
            patch("belt.commands.quickstart.get_agent_class", side_effect=fake_get),
            patch("belt.commands.quickstart._find_examples_dir", return_value=scenarios_dir),
            patch("belt.commands.eval.main", return_value=0) as mock_eval,
        ):
            rc = main([])

        assert rc == 0
        call_args = mock_eval.call_args[0][0]
        # Must call eval with --agent <selected> and the showcase quickstart filter.
        assert "--agent" in call_args
        assert call_args[call_args.index("--agent") + 1] == "second-agent"
        assert _QUICKSTART_GROUP in call_args[call_args.index("--scenarios") + 1]

    def test_fallback_to_first_json_when_preferred_missing(self, capsys, tmp_path):
        """If the canonical quickstart scenario filename is missing, fall back to first non-underscore JSON."""
        scenarios_dir = tmp_path / "examples" / "scenarios"
        group_dir = scenarios_dir / _QUICKSTART_GROUP
        group_dir.mkdir(parents=True)
        (group_dir / "_config.json").write_text('{"agent": "claude-code"}')
        (group_dir / "alpha_test.json").write_text('{"name":"alpha","turns":[{"message":"hi"}]}')
        (group_dir / "beta_test.json").write_text('{"name":"beta","turns":[{"message":"hi"}]}')

        with (
            patch("belt.commands.quickstart.available_agents", return_value=["claude-code"]),
            patch("belt.commands.quickstart.get_agent_class") as mock_cls,
            patch("belt.commands.quickstart._find_examples_dir", return_value=scenarios_dir),
            patch("belt.commands.eval.main", return_value=0) as mock_eval,
        ):
            mock_cls.return_value.check_available.return_value = None
            mock_cls.return_value.display_info.return_value = "Claude 2.1"
            rc = main(["claude-code"])

        assert rc == 0
        call_args = mock_eval.call_args[0][0]
        assert "alpha_test" in call_args[call_args.index("--scenarios") + 1]

    def test_no_scenarios_dir(self, capsys):
        with (
            patch("belt.commands.quickstart.available_agents", return_value=["claude-code"]),
            patch("belt.commands.quickstart.get_agent_class") as mock_cls,
            patch("belt.commands.quickstart._find_examples_dir", return_value=None),
        ):
            mock_cls.return_value.check_available.return_value = None
            mock_cls.return_value.display_info.return_value = "ok"
            rc = main(["claude-code"])
        assert rc == 1

    def test_no_scenarios_for_quickstart_group(self, capsys, tmp_path):
        scenarios_dir = tmp_path / "examples" / "scenarios"
        scenarios_dir.mkdir(parents=True)

        with (
            patch("belt.commands.quickstart.available_agents", return_value=["claude-code"]),
            patch("belt.commands.quickstart.get_agent_class") as mock_cls,
            patch("belt.commands.quickstart._find_examples_dir", return_value=scenarios_dir),
        ):
            mock_cls.return_value.check_available.return_value = None
            mock_cls.return_value.display_info.return_value = "ok"
            rc = main(["claude-code"])
        assert rc == 1

    def test_delegates_to_eval(self, capsys, tmp_path):
        scenarios_dir = _make_quickstart_dir(tmp_path)

        with (
            patch("belt.commands.quickstart.available_agents", return_value=["claude-code"]),
            patch("belt.commands.quickstart.get_agent_class") as mock_cls,
            patch("belt.commands.quickstart._find_examples_dir", return_value=scenarios_dir),
            patch("belt.commands.eval.main", return_value=0) as mock_eval,
        ):
            mock_cls.return_value.check_available.return_value = None
            mock_cls.return_value.display_info.return_value = "Claude Code 2.1"
            rc = main(["claude-code"])

        assert rc == 0
        mock_eval.assert_called_once()
        call_args = mock_eval.call_args[0][0]
        assert "--modes" in call_args
        assert "rules" in call_args
        assert "--agent" in call_args
        assert call_args[call_args.index("--agent") + 1] == "claude-code"

    def test_eval_failure_returns_nonzero(self, capsys, tmp_path):
        scenarios_dir = _make_quickstart_dir(tmp_path)

        with (
            patch("belt.commands.quickstart.available_agents", return_value=["claude-code"]),
            patch("belt.commands.quickstart.get_agent_class") as mock_cls,
            patch("belt.commands.quickstart._find_examples_dir", return_value=scenarios_dir),
            patch("belt.commands.eval.main", return_value=1),
        ):
            mock_cls.return_value.check_available.return_value = None
            mock_cls.return_value.display_info.return_value = "Claude Code 2.1"
            rc = main(["claude-code"])

        assert rc == 1
        # Quickstart UI is rendered via Rich Console(stderr=True).
        out = capsys.readouterr().err
        assert "issues" in out.lower() or "debug" in out.lower()

    def test_eval_failure_shows_debug_help(self, capsys, tmp_path):
        scenarios_dir = _make_quickstart_dir(tmp_path)

        with (
            patch("belt.commands.quickstart.available_agents", return_value=["claude-code"]),
            patch("belt.commands.quickstart.get_agent_class") as mock_cls,
            patch("belt.commands.quickstart._find_examples_dir", return_value=scenarios_dir),
            patch("belt.commands.eval.main", return_value=1),
        ):
            mock_cls.return_value.check_available.return_value = None
            mock_cls.return_value.display_info.return_value = "ok"
            main(["claude-code"])

        out = capsys.readouterr().err
        assert "belt doctor" in out
