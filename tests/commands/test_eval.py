# (c) JFrog Ltd. (2026)

"""Tests for the belt eval unified command."""

from __future__ import annotations

from pathlib import Path

import pytest

from belt.commands.eval import _build_parser, main


class TestEvalParser:
    def test_required_path_argument(self):
        parser = _build_parser()
        args = parser.parse_args(["scenarios/"])
        assert args.path == "scenarios/"

    def test_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["scenarios/"])
        assert args.modes == "rules,llm"
        assert args.workers == 1
        assert args.progress == "rich"
        assert args.dry_run is False
        assert args.threshold == []
        assert args.tags is None
        assert args.scenarios is None
        assert args.agent_args is None
        assert args.scorer_args is None

    def test_agent_args(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "scenarios/",
                "-X",
                "server_id=abc123",
                "-X",
                "max_message_len=50000",
            ]
        )
        assert args.agent_args == ["server_id=abc123", "max_message_len=50000"]

    def test_scorer_args(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "scenarios/",
                "-S",
                "model=openai/gpt-4.1",
                "-S",
                "temperature=0.0",
            ]
        )
        assert args.scorer_args == ["model=openai/gpt-4.1", "temperature=0.0"]

    def test_all_phase_flags(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "scenarios/",
                "--tags",
                "v2",
                "--scenarios",
                "group/name",
                "--agent",
                "claude-code",
                "--workers",
                "4",
                "--modes",
                "rules",
                "-X",
                "server_id=abc",
                "-S",
                "model=gpt-4.1",
                "--threshold",
                "rules/execution:0",
                "--threshold",
                "rules/trajectory:10",
                "--progress",
                "plain",
                "--dry-run",
                "--outcomes-dir",
                "/tmp/custom-outcomes",
            ]
        )
        assert args.tags == "v2"
        assert args.scenarios == "group/name"
        assert args.agent == "claude-code"
        assert args.workers == 4
        assert args.modes == "rules"
        assert args.agent_args == ["server_id=abc"]
        assert args.scorer_args == ["model=gpt-4.1"]
        assert args.threshold == ["rules/execution:0", "rules/trajectory:10"]
        assert args.progress == "plain"
        assert args.dry_run is True
        assert args.outcomes_dir == "/tmp/custom-outcomes"

    def test_unknown_flag_is_rejected(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["scenarios/", "--server-id", "old123"])

    def test_old_outcomes_flag_is_rejected_not_abbreviated(self):
        """``--outcomes`` was renamed to ``--run-dir`` (different semantics from
        ``--outcomes-dir``). With argparse abbreviation on, ``--outcomes`` would
        silently match ``--outcomes-dir``. ``allow_abbrev=False`` prevents this
        and forces the migration.
        """
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["scenarios/", "--outcomes", "/tmp/existing-run-dir"])

    def test_help_exits_0(self):
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0


class TestProgressLiveLinesForwarding:
    """``--progress-live-lines`` must be forwarded unconditionally when
    ``--progress live`` is selected.  The previous "only if != 30" guard
    silently dropped the flag when the user explicitly set it to 30.
    """

    @pytest.fixture
    def _run_argv(self, monkeypatch):
        """Capture the ``argv`` list ``run_main`` is called with.

        ``commands/eval.py`` imports ``run_main`` lazily inside ``main()``
        (``from belt.commands.run import main as run_main``); patching
        the source module is the only effective hook.
        """
        captured: dict[str, list[str]] = {}

        def fake_run_main(argv: list[str]) -> int:
            captured["argv"] = list(argv)
            return 1  # short-circuit subsequent phases

        monkeypatch.setattr("belt.commands.run.main", fake_run_main)
        return captured

    def _argv_run(self, *extra: str) -> list[str]:
        # ``--modes rules --dry-run`` skips the LLM preflight + scorer
        # availability checks, so the test does not require any LLM env
        # var to exercise the marshalling path.
        return ["scenarios/", "--progress", "live", "--modes", "rules", "--dry-run", *extra]

    def test_default_progress_live_lines_still_forwarded(self, _run_argv) -> None:
        main(self._argv_run())
        argv = _run_argv["argv"]
        assert "--progress-live-lines" in argv
        idx = argv.index("--progress-live-lines")
        assert argv[idx + 1] == "30"

    def test_user_set_to_30_is_forwarded(self, _run_argv) -> None:
        """Regression: was silently dropped when user passed exactly the default."""
        main(self._argv_run("--progress-live-lines", "30"))
        argv = _run_argv["argv"]
        assert "--progress-live-lines" in argv
        idx = argv.index("--progress-live-lines")
        assert argv[idx + 1] == "30"

    def test_user_set_to_non_default_is_forwarded(self, _run_argv) -> None:
        main(self._argv_run("--progress-live-lines", "50"))
        argv = _run_argv["argv"]
        assert "--progress-live-lines" in argv
        idx = argv.index("--progress-live-lines")
        assert argv[idx + 1] == "50"

    def test_not_forwarded_when_progress_not_live(self, _run_argv) -> None:
        main(["scenarios/", "--progress", "plain", "--modes", "rules", "--dry-run"])
        argv = _run_argv["argv"]
        assert "--progress-live-lines" not in argv


class TestTrialsAndNoStreamForwarding:
    """``--trials N`` and ``--no-stream`` are documented on ``belt eval``
    and must be forwarded to the ``run`` phase. ``--trials`` only forwards
    when N > 1 to keep the run-phase argv clean for the common case.
    """

    @pytest.fixture
    def _run_argv(self, monkeypatch):
        captured: dict[str, list[str]] = {}

        def fake_run_main(argv: list[str]) -> int:
            captured["argv"] = list(argv)
            return 1

        monkeypatch.setattr("belt.commands.run.main", fake_run_main)
        return captured

    def _argv(self, *extra: str) -> list[str]:
        return ["scenarios/", "--modes", "rules", "--dry-run", *extra]

    def test_no_stream_forwarded(self, _run_argv) -> None:
        main(self._argv("--no-stream"))
        assert "--no-stream" in _run_argv["argv"]

    def test_no_stream_omitted_by_default(self, _run_argv) -> None:
        main(self._argv())
        assert "--no-stream" not in _run_argv["argv"]

    def test_trials_forwarded_when_greater_than_one(self, _run_argv) -> None:
        main(self._argv("--trials", "3"))
        argv = _run_argv["argv"]
        assert "--trials" in argv
        idx = argv.index("--trials")
        assert argv[idx + 1] == "3"

    def test_trials_default_is_not_forwarded(self, _run_argv) -> None:
        """``--trials 1`` is the implicit default; no need to forward."""
        main(self._argv())
        assert "--trials" not in _run_argv["argv"]

    def test_trials_explicit_one_is_not_forwarded(self, _run_argv) -> None:
        """User explicitly passing 1 should be a no-op for the run phase too."""
        main(self._argv("--trials", "1"))
        assert "--trials" not in _run_argv["argv"]

    def test_trials_and_no_stream_combined(self, _run_argv) -> None:
        main(self._argv("--trials", "5", "--no-stream"))
        argv = _run_argv["argv"]
        assert "--no-stream" in argv
        assert "--trials" in argv
        assert argv[argv.index("--trials") + 1] == "5"


class TestAllowInplaceForwarding:
    """``--allow-inplace`` declared on ``belt eval`` must reach the
    underlying ``belt run`` argv, otherwise the gate in ``setup_groups``
    fires for users who explicitly opted in via the unified command.
    Mirrors the existing ``--allow-external-working-dir`` forwarding."""

    @pytest.fixture
    def _run_argv(self, monkeypatch):
        captured: dict[str, list[str]] = {}

        def fake_run_main(argv: list[str]) -> int:
            captured["argv"] = list(argv)
            return 1  # short-circuit downstream phases

        monkeypatch.setattr("belt.commands.run.main", fake_run_main)
        return captured

    def _argv(self, *extra: str) -> list[str]:
        # ``--modes rules --dry-run`` keeps the forwarding test offline -
        # no LLM preflight, no scorer availability check, no real agent.
        return ["scenarios/", "--modes", "rules", "--dry-run", *extra]

    def test_parser_accepts_allow_inplace(self):
        from belt.commands.eval import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["scenarios/", "--allow-inplace"])
        assert args.allow_inplace is True

    def test_default_is_false(self):
        from belt.commands.eval import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["scenarios/"])
        assert args.allow_inplace is False

    def test_flag_forwarded_to_run(self, _run_argv) -> None:
        main(self._argv("--allow-inplace"))
        assert "--allow-inplace" in _run_argv["argv"]

    def test_omitted_when_unset(self, _run_argv) -> None:
        main(self._argv())
        assert "--allow-inplace" not in _run_argv["argv"]


class TestSandboxForwarding:
    """``--sandbox`` declared on ``belt eval`` must reach the underlying
    ``belt run`` argv so the orchestrator picks the right provider when
    users run the unified command. Mirrors the ``--allow-inplace``
    forwarding above; same fixture pattern."""

    @pytest.fixture
    def _run_argv(self, monkeypatch):
        captured: dict[str, list[str]] = {}

        def fake_run_main(argv: list[str]) -> int:
            captured["argv"] = list(argv)
            return 1  # short-circuit downstream phases

        monkeypatch.setattr("belt.commands.run.main", fake_run_main)
        return captured

    def _argv(self, *extra: str) -> list[str]:
        return ["scenarios/", "--modes", "rules", "--dry-run", *extra]

    def test_parser_accepts_sandbox_host(self):
        from belt.commands.eval import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["scenarios/", "--sandbox", "host"])
        assert args.sandbox == "host"

    def test_parser_accepts_sandbox_docker(self):
        from belt.commands.eval import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["scenarios/", "--sandbox", "docker"])
        assert args.sandbox == "docker"

    def test_default_sandbox_is_none(self):
        from belt.commands.eval import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["scenarios/"])
        # ``None`` means "respect whatever the scenario _config.json says";
        # the override is opt-in.
        assert args.sandbox is None

    def test_host_forwarded_to_run(self, _run_argv) -> None:
        main(self._argv("--sandbox", "host"))
        argv = _run_argv["argv"]
        assert "--sandbox" in argv
        assert argv[argv.index("--sandbox") + 1] == "host"

    def test_docker_forwarded_to_run(self, _run_argv) -> None:
        main(self._argv("--sandbox", "docker"))
        argv = _run_argv["argv"]
        assert "--sandbox" in argv
        assert argv[argv.index("--sandbox") + 1] == "docker"

    def test_omitted_when_unset(self, _run_argv) -> None:
        main(self._argv())
        assert "--sandbox" not in _run_argv["argv"]


class TestEvalMain:
    def test_dry_run_nonexistent_path(self):
        rc = main(["nonexistent/", "--dry-run"])
        assert rc == 1

    def test_dry_run_examples(self):
        examples = Path(__file__).resolve().parent.parent / "examples" / "scenarios"
        if not examples.is_dir():
            pytest.skip("examples/scenarios not available")
        rc = main([str(examples), "--dry-run", "--modes", "rules"])
        assert rc == 0

    def test_dry_run_filtered(self):
        examples = Path(__file__).resolve().parent.parent / "examples" / "scenarios"
        if not examples.is_dir():
            pytest.skip("examples/scenarios not available")
        rc = main([str(examples), "--dry-run", "--modes", "rules", "--scenarios", "agents/claude-code"])
        assert rc == 0

    def test_dry_run_no_matching_scenarios(self):
        examples = Path(__file__).resolve().parent.parent / "examples" / "scenarios"
        if not examples.is_dir():
            pytest.skip("examples/scenarios not available")
        rc = main([str(examples), "--dry-run", "--modes", "rules", "--tags", "nonexistent-tag-xyz"])
        assert rc == 0  # no match = 0 in dry-run


class TestScorerCliReturnsInt:
    """Verify commands/score.py main() returns int exit codes instead of calling sys.exit."""

    def test_nonexistent_run_dir_returns_1(self):
        from belt.commands.score import main as score_main

        rc = score_main(["--run-dir", "/nonexistent/path", "--modes", "rules"])
        assert rc == 1

    def test_empty_run_dir_returns_1(self, tmp_path: Path):
        from belt.commands.score import main as score_main

        rc = score_main(["--run-dir", str(tmp_path), "--modes", "rules"])
        assert rc == 1

    def test_unknown_mode_returns_1(self, tmp_path: Path):
        from belt.commands.score import main as score_main

        rc = score_main(["--run-dir", str(tmp_path), "--modes", "badmode"])
        assert rc == 1


class TestHasRealTurnsGate:
    """Pin the ``has_real_turns`` gate at ``eval.py:472`` end-to-end.

    The gate's contract: skip score/aggregate iff zero scenarios under
    the run directory produced a ``turn_*_output.json``. Sentinel
    ``turn_*_cli.txt`` files written by ``orchestrator._write_sentinel``
    on setup / execute failures do NOT count - that's the whole point of
    the gate (without it, ~14K LLM tokens get wasted scoring agent error
    messages on a fully-failed run).

    Without this test, a bad rebase that swaps ``TURN_OUTPUT_TEMPLATE``
    back to ``TURN_CLI_TEMPLATE`` (one-line regression) would slip
    through CI - the renderer-diet pin in ``test_render_diet.py`` only
    fires once scoring has produced a ``ScenarioScore`` to render.
    """

    @pytest.fixture
    def _stub_phases(self, monkeypatch):
        """Stub ``run_main`` / ``score_main`` / ``agg_main`` and record
        which downstream phases ``eval`` invoked.

        ``run_main`` sets ``LAST_RUN_DIR`` to the fixture-supplied path
        (mirroring what the real ``run.py`` does on success) so the
        gate has a run directory to inspect.
        """
        import os

        from belt._internal_envvars import LAST_RUN_DIR

        called: dict[str, int] = {"run": 0, "score": 0, "aggregate": 0}

        def make_fake_run(run_dir: Path, rc: int):
            def fake_run_main(argv: list[str]) -> int:
                called["run"] += 1
                os.environ[LAST_RUN_DIR] = str(run_dir)
                return rc

            return fake_run_main

        def fake_score_main(argv: list[str], *, chained: bool = False) -> int:
            called["score"] += 1
            # Record so the test below can pin the expectation that ``belt
            # eval`` always calls the scorer with ``chained=True``.
            called["score_chained"] = int(chained)
            return 0

        def fake_agg_main(argv: list[str]) -> int:
            called["aggregate"] += 1
            return 0

        def install(run_dir: Path, run_rc: int = 0) -> dict[str, int]:
            monkeypatch.setattr("belt.commands.run.main", make_fake_run(run_dir, run_rc))
            monkeypatch.setattr("belt.commands.score.main", fake_score_main)
            monkeypatch.setattr("belt.commands.aggregate.main", fake_agg_main)
            # Skip scorer preflight (LLM probe); ``--modes rules`` already
            # avoids the LLM key requirement but the preflight still tries
            # to import the rules scorer registry. Returning ``{}`` keeps
            # us offline.
            monkeypatch.setattr(
                "belt.commands.score.validate_scorers",
                lambda modes, args, cfg, probe_api=False: {},
            )
            return called

        return install

    def _argv(self, scenarios_root: str, *extra: str) -> list[str]:
        return [scenarios_root, "--modes", "rules", *extra]

    def test_skips_score_and_aggregate_when_only_sentinels_exist(
        self, tmp_path: Path, _stub_phases, capsys: pytest.CaptureFixture
    ) -> None:
        # Simulate an all-setup-failed run: one ``turn_*_cli.txt`` sentinel,
        # zero ``turn_*_output.json``. This is the exact shape the
        # bookstore-api-claude reproducer produced.
        run_dir = tmp_path / "outcomes" / "20260514-100000-deadbeef"
        outcome = run_dir / "group" / "scenario"
        outcome.mkdir(parents=True)
        (outcome / "turn_0_cli.txt").write_text("sandbox setup failed: ...\n")

        called = _stub_phases(run_dir, run_rc=1)

        rc = main(self._argv(str(tmp_path)))

        assert called["run"] == 1
        assert called["score"] == 0, "score phase ran on sentinel-only run"
        assert called["aggregate"] == 0, "aggregate phase ran on sentinel-only run"
        # ``run_rc`` was 1; the gate propagates it unchanged.
        assert rc == 1
        err = capsys.readouterr().err
        assert "Score/aggregate skipped" in err
        assert "no scenario produced a real turn" in err

    def test_runs_score_and_aggregate_when_any_real_turn_exists(
        self, tmp_path: Path, _stub_phases, capsys: pytest.CaptureFixture
    ) -> None:
        # One scenario produced a real ``turn_*_output.json``; even with
        # peer scenarios that only wrote sentinels, the gate must open.
        run_dir = tmp_path / "outcomes" / "20260514-100000-deadbeef"
        ok = run_dir / "group" / "ok"
        ok.mkdir(parents=True)
        (ok / "turn_0_output.json").write_text('{"raw_cli": "ok"}\n')

        bad = run_dir / "group" / "bad"
        bad.mkdir(parents=True)
        (bad / "turn_0_cli.txt").write_text("agent crashed\n")

        called = _stub_phases(run_dir, run_rc=0)
        rc = main(self._argv(str(tmp_path)))

        assert called["score"] == 1
        assert called["aggregate"] == 1
        # Single-scoreboard regression guard: when ``belt eval`` chains
        # the score phase in-process it must pass ``chained=True`` so
        # ``ScorerProgress.summary`` suppresses its pass-count headline
        # and lets the aggregator print the canonical scoreboard alone.
        assert (
            called.get("score_chained") == 1
        ), "belt eval must call score_main with chained=True so the screen has a single scoreboard"
        assert rc == 0
        err = capsys.readouterr().err
        assert "Score/aggregate skipped" not in err

    def test_uses_turn_output_template_not_cli_template(self, tmp_path: Path, _stub_phases) -> None:
        """Bad-rebase guard: the gate must read ``TURN_OUTPUT_TEMPLATE``
        (``turn_*_output.json``), never ``TURN_CLI_TEMPLATE``
        (``turn_*_cli.txt``). Swapping them would re-introduce the
        wasted-LLM-tokens bug that motivated the gate.
        """
        # A directory with only ``turn_*_cli.txt`` must NOT open the gate.
        # If the renderer ever reverts to ``TURN_CLI_TEMPLATE.format("*")``,
        # ``score`` would be invoked here and this assertion would fire.
        run_dir = tmp_path / "outcomes" / "r"
        outcome = run_dir / "g" / "s"
        outcome.mkdir(parents=True)
        # Three sentinels - mimics a three-turn scenario that crashed
        # in execute() on turn 0 before fetch_results.
        for i in range(3):
            (outcome / f"turn_{i}_cli.txt").write_text("err\n")

        called = _stub_phases(run_dir, run_rc=1)
        main(self._argv(str(tmp_path)))

        assert called["score"] == 0
        assert called["aggregate"] == 0
