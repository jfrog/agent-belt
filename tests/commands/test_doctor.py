# (c) JFrog Ltd. (2026)

"""Tests for belt doctor."""

from __future__ import annotations

import json
import time
from unittest.mock import patch

from belt.agent.base import AgentNotAvailableError
from belt.commands.doctor import (
    CheckResult,
    DoctorReport,
    _check_agent,
    _check_agents_parallel,
    _check_azure_openai,
    _check_install_integrity,
    _check_judge_model,
    _check_llm_providers,
    _strip_ansi,
    main,
    print_report,
    run_doctor,
    run_doctor_live,
)


class TestCheckResult:
    def test_ok_result(self):
        r = CheckResult(ok=True, label="test", detail="ready")
        assert r.ok
        assert r.label == "test"

    def test_failed_result_with_suggestion(self):
        r = CheckResult(ok=False, label="test", detail="missing", suggestion="install it")
        assert not r.ok
        assert r.suggestion == "install it"

    def test_default_suggestion_is_empty(self):
        r = CheckResult(ok=False, label="x", detail="y")
        assert r.suggestion == ""


class TestDoctorReport:
    def test_agents_ready_count(self):
        report = DoctorReport(
            agent_checks=[
                CheckResult(ok=True, label="a"),
                CheckResult(ok=False, label="b"),
                CheckResult(ok=True, label="c"),
            ]
        )
        assert report.agents_ready == 2

    def test_llm_providers_ready_count(self):
        report = DoctorReport(
            llm_checks=[
                CheckResult(ok=True, label="OpenAI"),
                CheckResult(ok=False, label="Anthropic"),
            ]
        )
        assert report.llm_providers_ready == 1

    def test_empty_report(self):
        report = DoctorReport()
        assert report.agents_ready == 0
        assert report.llm_providers_ready == 0

    def test_all_agents_ready(self):
        report = DoctorReport(agent_checks=[CheckResult(ok=True, label=n) for n in ["a", "b", "c"]])
        assert report.agents_ready == 3

    def test_no_agents_ready(self):
        report = DoctorReport(agent_checks=[CheckResult(ok=False, label=n) for n in ["a", "b"]])
        assert report.agents_ready == 0


class TestCheckInstallIntegrity:
    """Cross-clone shadowing detection - see doctor._check_install_integrity."""

    def test_returns_none_when_cwd_is_not_a_clone(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _check_install_integrity() is None

    def test_returns_none_when_loaded_matches_cwd(self, tmp_path, monkeypatch):
        src_init = tmp_path / "src" / "belt" / "__init__.py"
        src_init.parent.mkdir(parents=True)
        src_init.write_text("# fake")
        monkeypatch.chdir(tmp_path)

        import belt as _ae

        with patch.object(_ae, "__file__", str(src_init)):
            assert _check_install_integrity() is None

    def test_warns_when_loaded_from_different_clone(self, tmp_path, monkeypatch):
        clone_a = tmp_path / "a"
        clone_b = tmp_path / "b"
        for c in (clone_a, clone_b):
            (c / "src" / "belt").mkdir(parents=True)
            (c / "src" / "belt" / "__init__.py").write_text("# fake")

        monkeypatch.chdir(clone_a)
        import belt as _ae

        loaded_b = str(clone_b / "src" / "belt" / "__init__.py")
        with patch.object(_ae, "__file__", loaded_b):
            result = _check_install_integrity()

        assert result is not None
        assert not result.ok
        assert result.label == "Install path"
        assert str(clone_b) in result.detail
        assert "pip install -e" in result.suggestion

    def test_returns_none_when_module_file_is_empty(self, tmp_path, monkeypatch):
        src_init = tmp_path / "src" / "belt" / "__init__.py"
        src_init.parent.mkdir(parents=True)
        src_init.write_text("# fake")
        monkeypatch.chdir(tmp_path)

        import belt as _ae

        with patch.object(_ae, "__file__", ""):
            assert _check_install_integrity() is None


class TestStripAnsi:
    def test_strips_escape_sequences(self):
        raw = "\x1b[2K\x1b[GLoading... | \x1b[2K\x1b[1A\x1b[2K\x1b[GAbout Cursor CLI"
        assert _strip_ansi(raw) == "About Cursor CLI"

    def test_strips_carriage_returns(self):
        assert _strip_ansi("hello\rworld") == "helloworld"

    def test_passthrough_clean_text(self):
        assert _strip_ansi("Claude Code 2.1.81") == "Claude Code 2.1.81"

    def test_empty_string(self):
        assert _strip_ansi("") == ""

    def test_only_escape_sequences(self):
        assert _strip_ansi("\x1b[2K\x1b[G\r") == ""

    def test_color_codes(self):
        assert _strip_ansi("\x1b[32mgreen\x1b[0m") == "green"

    def test_multiple_consecutive_escapes(self):
        raw = "\x1b[1A\x1b[2K\x1b[1A\x1b[2Kfinal output"
        assert _strip_ansi(raw) == "final output"


class TestCheckAgent:
    def test_available_agent(self):
        with patch("belt.commands.doctor.get_agent_class") as mock_cls:
            mock_cls.return_value.check_available.return_value = None
            mock_cls.return_value.display_info.return_value = "Test Agent v1.0"
            result = _check_agent("test-agent")
        assert result.ok
        assert "v1.0" in result.detail

    def test_unavailable_agent(self):
        with patch("belt.commands.doctor.get_agent_class") as mock_cls:
            mock_cls.return_value.check_available.side_effect = AgentNotAvailableError(
                "test", "not found", "install it"
            )
            result = _check_agent("test-agent")
        assert not result.ok
        assert "not found" in result.detail
        assert "install it" in result.suggestion

    def test_unloadable_agent(self):
        with patch("belt.commands.doctor.get_agent_class", side_effect=ImportError("no module")):
            result = _check_agent("broken-agent")
        assert not result.ok
        assert "load error" in result.detail

    def test_display_info_throws_falls_back_to_available(self):
        with patch("belt.commands.doctor.get_agent_class") as mock_cls:
            mock_cls.return_value.check_available.return_value = None
            mock_cls.return_value.display_info.side_effect = RuntimeError("boom")
            result = _check_agent("test-agent")
        assert result.ok
        assert result.detail == "available"

    def test_display_info_ansi_stripped(self):
        with patch("belt.commands.doctor.get_agent_class") as mock_cls:
            mock_cls.return_value.check_available.return_value = None
            mock_cls.return_value.display_info.return_value = "\x1b[2K\x1b[GLoading\x1b[2K\x1b[GCursor v1.0"
            result = _check_agent("test-agent")
        assert result.ok
        assert "\x1b" not in result.detail
        assert "Cursor v1.0" in result.detail

    def test_check_available_generic_exception(self):
        with patch("belt.commands.doctor.get_agent_class") as mock_cls:
            mock_cls.return_value.check_available.side_effect = RuntimeError("timeout")
            result = _check_agent("test-agent")
        assert not result.ok
        assert "timeout" in result.detail
        assert result.suggestion == ""


class TestCheckAgentsParallel:
    def test_runs_all_agents(self):
        with patch("belt.commands.doctor._check_agent") as mock:
            mock.side_effect = lambda n: CheckResult(ok=True, label=n, detail="ok")
            results = _check_agents_parallel(["a", "b", "c"])
        assert len(results) == 3
        assert [r.label for r in results] == ["a", "b", "c"]

    def test_preserves_name_order(self):
        """Results are returned in name order, not completion order."""

        def _slow_then_fast(name):
            if name == "slow":
                time.sleep(0.1)
            return CheckResult(ok=True, label=name, detail="done")

        with patch("belt.commands.doctor._check_agent", side_effect=_slow_then_fast):
            results = _check_agents_parallel(["slow", "fast"])
        assert results[0].label == "slow"
        assert results[1].label == "fast"

    def test_timeout_produces_check_result(self):
        def _hang(name):
            time.sleep(5)
            return CheckResult(ok=True, label=name, detail="never")

        with patch("belt.commands.doctor._check_agent", side_effect=_hang):
            results = _check_agents_parallel(["hanging"], timeout=0.1)
        assert len(results) == 1
        assert not results[0].ok
        assert "timed out" in results[0].detail

    def test_on_result_called_per_agent(self):
        received: list[str] = []

        with patch("belt.commands.doctor._check_agent") as mock:
            mock.side_effect = lambda n: CheckResult(ok=True, label=n, detail="ok")
            _check_agents_parallel(["x", "y"], on_result=lambda c: received.append(c.label))
        assert sorted(received) == ["x", "y"]

    def test_empty_names_returns_empty(self):
        results = _check_agents_parallel([])
        assert results == []

    def test_runs_faster_than_sequential(self):
        def _slow(name):
            time.sleep(0.2)
            return CheckResult(ok=True, label=name, detail="ok")

        with patch("belt.commands.doctor._check_agent", side_effect=_slow):
            t0 = time.monotonic()
            _check_agents_parallel(["a", "b", "c", "d"], timeout=5)
            elapsed = time.monotonic() - t0
        assert elapsed < 0.8, f"Parallel should be ~0.2s not {elapsed:.1f}s"


class TestCheckAzureOpenai:
    """Dedicated tests for the multi-var Azure OpenAI check."""

    def test_nothing_set(self):
        with patch.dict("os.environ", {}, clear=True):
            r = _check_azure_openai()
        assert not r.ok
        assert "BELT_AZURE_OPENAI_ENDPOINT not set" in r.detail
        assert "no auth configured" in r.detail
        assert "API key" in r.suggestion
        assert "service principal" in r.suggestion
        assert "BELT_AZURE_CLIENT_ID" in r.suggestion

    def test_endpoint_plus_api_key(self):
        env = {
            "BELT_AZURE_OPENAI_ENDPOINT": "https://my.openai.azure.com",
            "BELT_AZURE_OPENAI_API_KEY": "secret-key",
        }
        with patch.dict("os.environ", env, clear=True):
            r = _check_azure_openai()
        assert r.ok
        assert "API key" in r.detail
        assert "secret-key" not in r.detail

    def test_endpoint_plus_service_principal(self):
        env = {
            "BELT_AZURE_OPENAI_ENDPOINT": "https://my.openai.azure.com",
            "BELT_AZURE_CLIENT_ID": "id",
            "BELT_AZURE_CLIENT_SECRET": "sec",
            "BELT_AZURE_TENANT_ID": "tid",
        }
        with patch.dict("os.environ", env, clear=True):
            r = _check_azure_openai()
        assert r.ok
        assert "service principal" in r.detail
        for secret in ("id", "sec", "tid"):
            assert secret not in r.detail or "configured" in r.detail

    def test_endpoint_only_no_auth(self):
        env = {"BELT_AZURE_OPENAI_ENDPOINT": "https://my.openai.azure.com"}
        with patch.dict("os.environ", env, clear=True):
            r = _check_azure_openai()
        assert not r.ok
        assert "no auth configured" in r.detail
        assert "BELT_AZURE_OPENAI_ENDPOINT not set" not in r.detail

    def test_endpoint_plus_partial_sp(self):
        env = {
            "BELT_AZURE_OPENAI_ENDPOINT": "https://my.openai.azure.com",
            "BELT_AZURE_CLIENT_ID": "id",
        }
        with patch.dict("os.environ", env, clear=True):
            r = _check_azure_openai()
        assert not r.ok
        assert "BELT_AZURE_CLIENT_SECRET" in r.detail
        assert "BELT_AZURE_TENANT_ID" in r.detail

    def test_no_secret_values_leak(self):
        env = {
            "BELT_AZURE_OPENAI_ENDPOINT": "https://super-secret-resource.openai.azure.com",
            "BELT_AZURE_OPENAI_API_KEY": "my-super-secret-api-key-12345",
        }
        with patch.dict("os.environ", env, clear=True):
            r = _check_azure_openai()
        assert "super-secret-resource" not in r.detail
        assert "my-super-secret-api-key" not in r.detail


class TestCheckLlmProviders:
    def test_no_keys_set(self):
        # Isolate from the developer's local belt.yaml (and any other layered
        # config source) so the judge-model check reflects only the cleared env.
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "belt.commands.doctor._check_ollama",
                return_value=CheckResult(ok=False, label="Ollama", detail="not running"),
            ),
            patch("belt.config.resolve_judge_model_source", return_value=(None, None)),
        ):
            results = _check_llm_providers()
        assert all(not r.ok for r in results)
        # 4 provider checks (OpenAI/Anthropic/Azure/Ollama) + 1 judge-model check
        assert len(results) == 5
        assert results[-1].label == "Judge model"

    def test_openai_key_set(self):
        with patch.dict("os.environ", {"BELT_OPENAI_API_KEY": "sk-test1234567890abcdef"}, clear=True):
            results = _check_llm_providers()
        openai = next(r for r in results if r.label == "OpenAI")
        assert openai.ok
        assert "set" in openai.detail
        assert "sk-" not in openai.detail
        assert "test" not in openai.detail

    def test_multiple_keys_set(self):
        env = {
            "BELT_OPENAI_API_KEY": "sk-secret123",
            "BELT_ANTHROPIC_API_KEY": "sk-ant-secret456",
        }
        with patch.dict("os.environ", env, clear=True):
            results = _check_llm_providers()
        openai = next(r for r in results if r.label == "OpenAI")
        anthropic = next(r for r in results if r.label == "Anthropic")
        azure = next(r for r in results if r.label == "Azure OpenAI")
        assert openai.ok
        assert anthropic.ok
        assert not azure.ok

    def test_no_real_value_leaks(self):
        """Ensure no part of the actual key appears in any check result."""
        secret = "sk-SUPERSECRETKEY1234567890"
        with patch.dict("os.environ", {"BELT_OPENAI_API_KEY": secret}, clear=True):
            results = _check_llm_providers()
        for r in results:
            assert secret not in r.detail
            assert "SUPERSECRET" not in r.detail
            assert "1234567890" not in r.detail

    def test_azure_integrated_in_providers(self):
        """Azure check is included as the third provider in the list."""
        env = {
            "BELT_AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com",
            "BELT_AZURE_OPENAI_API_KEY": "key",
        }
        with patch.dict("os.environ", env, clear=True):
            results = _check_llm_providers()
        azure = next(r for r in results if r.label == "Azure OpenAI")
        assert azure.ok


class TestFormatCheckAuthSignals:
    """Doctor renders positive auth signals when present, never negative ones.
    Agents that declared CREDENTIAL_ENV / CREDENTIAL_PATHS but matched none
    get an `auth: unknown` hint - agents that didn't opt in get nothing.
    """

    def test_ok_with_env_signal(self):
        from belt.commands.doctor import _format_check

        c = CheckResult(
            ok=True,
            label="cursor",
            detail="cursor-agent 2026.04",
            auth_signals=["env CURSOR_API_KEY"],
        )
        line = _format_check(c)
        assert "env CURSOR_API_KEY" in line
        assert "auth:" in line

    def test_ok_with_both_signals_joined(self):
        from belt.commands.doctor import _format_check

        c = CheckResult(
            ok=True,
            label="cursor",
            detail="ok",
            auth_signals=["env CURSOR_API_KEY", "stored login (~/.cursor)"],
        )
        line = _format_check(c)
        assert "env CURSOR_API_KEY + stored login (~/.cursor)" in line

    def test_ok_no_signals_no_credentials_declared(self):
        """Agents without CREDENTIAL_ENV/PATHS get a clean line - no
        `auth: unknown` hint."""
        from belt.commands.doctor import _format_check

        with patch("belt.commands.doctor._has_declared_credential_sources", return_value=False):
            c = CheckResult(ok=True, label="some-agent", detail="ready")
            line = _format_check(c)
        assert "auth:" not in line

    def test_ok_no_signals_credentials_declared_shows_unknown(self):
        """Cursor declares CREDENTIAL_ENV but neither env nor stored login
        is present (e.g. user authenticated via system keychain) - show
        `auth: unknown` so the user knows we couldn't verify."""
        from belt.commands.doctor import _format_check

        with patch("belt.commands.doctor._has_declared_credential_sources", return_value=True):
            c = CheckResult(ok=True, label="cursor", detail="cursor-agent 2026")
            line = _format_check(c)
        assert "auth: unknown" in line
        assert "binary OK" in line

    def test_failed_check_never_shows_auth(self):
        from belt.commands.doctor import _format_check

        c = CheckResult(
            ok=False,
            label="cursor",
            detail="not found",
            suggestion="install it",
            auth_signals=["env CURSOR_API_KEY"],
        )
        line = _format_check(c)
        assert "auth:" not in line
        assert "install it" in line


class TestCheckAgentAuthSignals:
    def test_check_agent_collects_auth_signals(self):
        from belt.commands.doctor import _check_agent

        with patch("belt.commands.doctor.get_agent_class") as mock_cls:
            mock_cls.return_value.check_available.return_value = None
            mock_cls.return_value.display_info.return_value = "v1"
            mock_cls.return_value.auth_signals.return_value = ["env MY_KEY"]
            result = _check_agent("my-agent")
        assert result.ok
        assert result.auth_signals == ["env MY_KEY"]

    def test_auth_signals_failure_does_not_break_check(self):
        """If auth_signals() raises (e.g. weird OS state), the check should
        still succeed with empty signals - auth detection is best-effort."""
        from belt.commands.doctor import _check_agent

        with patch("belt.commands.doctor.get_agent_class") as mock_cls:
            mock_cls.return_value.check_available.return_value = None
            mock_cls.return_value.display_info.return_value = "v1"
            mock_cls.return_value.auth_signals.side_effect = OSError("boom")
            result = _check_agent("my-agent")
        assert result.ok
        assert result.auth_signals == []


class TestPrintReport:
    def test_no_agents_ready_shows_install_message(self, capsys):
        report = DoctorReport(
            belt_version="1.0.0",
            python_version="3.13.0",
            agent_checks=[CheckResult(ok=False, label="test", detail="missing", suggestion="install")],
            llm_checks=[],
        )
        print_report(report)
        out = capsys.readouterr().err
        assert "no agents ready" in out
        assert "install at least one" in out.lower()

    def test_agents_ready_shows_quickstart(self, capsys):
        report = DoctorReport(
            belt_version="1.0.0",
            python_version="3.13.0",
            agent_checks=[CheckResult(ok=True, label="claude-code", detail="ready")],
            llm_checks=[],
        )
        print_report(report)
        out = capsys.readouterr().err
        assert "1 agent ready" in out
        assert "You only need one agent" in out
        assert "belt quickstart claude-code" in out

    def test_llm_optional_message(self, capsys):
        report = DoctorReport(
            belt_version="1.0.0",
            python_version="3.13.0",
            agent_checks=[],
            llm_checks=[CheckResult(ok=False, label="OpenAI", detail="not set")],
        )
        print_report(report)
        out = capsys.readouterr().err
        assert "optional" in out
        assert "rules-only" in out


class TestPrintReportAuthSignals:
    """End-to-end: auth signals threaded from CheckResult → human output.
    Guards against regression where signals were collected but not rendered."""

    def test_human_output_renders_env_signal(self, capsys):
        report = DoctorReport(
            belt_version="1.0.0",
            python_version="3.13.0",
            agent_checks=[
                CheckResult(
                    ok=True,
                    label="cursor",
                    detail="cursor-agent 2026",
                    auth_signals=["env CURSOR_API_KEY"],
                )
            ],
            llm_checks=[],
        )
        print_report(report)
        out = capsys.readouterr().err
        assert "env CURSOR_API_KEY" in out
        assert "auth:" in out

    def test_human_output_renders_unknown_for_declared_no_signals(self, capsys):
        """Agent declared CREDENTIAL_ENV but neither matched - show
        `auth: unknown` so user knows we tried but couldn't verify."""
        report = DoctorReport(
            belt_version="1.0.0",
            python_version="3.13.0",
            agent_checks=[CheckResult(ok=True, label="cursor", detail="cursor-agent 2026", auth_signals=[])],
            llm_checks=[],
        )
        with patch("belt.commands.doctor._has_declared_credential_sources", return_value=True):
            print_report(report)
        out = capsys.readouterr().err
        assert "auth: unknown" in out

    def test_human_output_no_auth_line_when_not_declared(self, capsys):
        """Agents that didn't opt in get a clean line - no noise."""
        report = DoctorReport(
            belt_version="1.0.0",
            python_version="3.13.0",
            agent_checks=[CheckResult(ok=True, label="some-agent", detail="ready", auth_signals=[])],
            llm_checks=[],
        )
        with patch("belt.commands.doctor._has_declared_credential_sources", return_value=False):
            print_report(report)
        out = capsys.readouterr().err
        assert "auth:" not in out


class TestHasDeclaredCredentialSources:
    """The render-time helper that asks the agent class whether it opted in."""

    def test_returns_true_for_agent_with_credential_env(self):
        from belt.commands.doctor import _has_declared_credential_sources

        class FakeAgentAdapter:
            CREDENTIAL_ENV = ("KEY",)
            CREDENTIAL_PATHS = ()

        with patch("belt.commands.doctor.get_agent_class", return_value=FakeAgentAdapter):
            assert _has_declared_credential_sources("fake") is True

    def test_returns_true_for_agent_with_credential_paths(self):
        from belt.commands.doctor import _has_declared_credential_sources

        class FakeAgentAdapter:
            CREDENTIAL_ENV = ()
            CREDENTIAL_PATHS = ("/some/path",)

        with patch("belt.commands.doctor.get_agent_class", return_value=FakeAgentAdapter):
            assert _has_declared_credential_sources("fake") is True

    def test_returns_false_for_agent_without_declarations(self):
        from belt.commands.doctor import _has_declared_credential_sources

        class FakeAgentAdapter:
            pass

        with patch("belt.commands.doctor.get_agent_class", return_value=FakeAgentAdapter):
            assert _has_declared_credential_sources("fake") is False

    def test_returns_false_when_agent_class_unavailable(self):
        """If the agent can't be loaded (e.g. plugin missing), don't blow up
        the doctor render."""
        from belt.commands.doctor import _has_declared_credential_sources

        with patch("belt.commands.doctor.get_agent_class", side_effect=ImportError("nope")):
            assert _has_declared_credential_sources("missing") is False


class TestRunDoctor:
    """Tests that run_doctor() wires up checks correctly - mocked to avoid slow subprocess calls."""

    def test_returns_report(self):
        with patch("belt.commands.doctor.available_agents", return_value=["mock"]):
            with patch("belt.commands.doctor._check_agent") as mock_check:
                mock_check.return_value = CheckResult(ok=True, label="mock", detail="ok")
                report = run_doctor()
        assert isinstance(report, DoctorReport)
        assert report.belt_version
        assert report.python_version
        assert len(report.agent_checks) == 1

    def test_report_has_llm_checks(self):
        with patch("belt.commands.doctor.available_agents", return_value=[]):
            report = run_doctor()
        # 4 provider checks + 1 judge-model check
        assert len(report.llm_checks) == 5


class TestRunDoctorLive:
    def test_returns_report(self, capsys):
        with patch("belt.commands.doctor.available_agents", return_value=["mock-agent"]):
            with patch("belt.commands.doctor._check_agent") as mock:
                mock.return_value = CheckResult(ok=True, label="mock-agent", detail="ready")
                report = run_doctor_live()
        assert isinstance(report, DoctorReport)
        assert report.agents_ready == 1
        out = capsys.readouterr().err
        assert "mock-agent" in out
        assert "checking" in out

    def test_streams_results_to_console(self, capsys):
        with patch("belt.commands.doctor.available_agents", return_value=["a", "b"]):
            with patch("belt.commands.doctor._check_agent") as mock:
                mock.side_effect = lambda n: CheckResult(ok=n == "a", label=n, detail="info")
                run_doctor_live()
        out = capsys.readouterr().err
        assert "✓" in out
        assert "✗" in out


def _mock_report(**overrides):
    defaults = dict(
        belt_version="1.0",
        python_version="3.13",
        agent_checks=[CheckResult(ok=True, label="mock", detail="ok")],
        llm_checks=[CheckResult(ok=True, label="OpenAI", detail="BELT_OPENAI_API_KEY set")],
    )
    defaults.update(overrides)
    return DoctorReport(**defaults)


class TestDoctorMain:
    """Tests for the CLI entry point - all use mocked run_doctor to avoid subprocess calls."""

    def test_json_output(self, capsys):
        with patch("belt.commands.doctor.run_doctor", return_value=_mock_report()):
            main(["--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "belt_version" in data
        assert "agents" in data
        assert "llm_providers" in data
        assert isinstance(data["agents_ready"], int)
        for agent in data["agents"]:
            assert "auth_signals" in agent
            assert isinstance(agent["auth_signals"], list)

    def test_json_includes_auth_signals(self, capsys):
        report = _mock_report(
            agent_checks=[
                CheckResult(
                    ok=True,
                    label="cursor",
                    detail="cursor-agent 2026",
                    auth_signals=["env CURSOR_API_KEY"],
                )
            ]
        )
        with patch("belt.commands.doctor.run_doctor", return_value=report):
            main(["--json"])
        data = json.loads(capsys.readouterr().out)
        assert data["agents"][0]["auth_signals"] == ["env CURSOR_API_KEY"]

    def test_json_no_key_values_leak(self, capsys):
        """JSON output must never contain real env var values."""
        secret = "sk-TOPSECRET9999"
        with patch.dict("os.environ", {"BELT_OPENAI_API_KEY": secret}):
            with patch("belt.commands.doctor.run_doctor", return_value=_mock_report()):
                main(["--json"])
        out = capsys.readouterr().out
        assert "BELT_OPENAI_API_KEY set" in out
        assert "sk-" not in out

    def test_human_output(self, capsys):
        with patch("belt.commands.doctor.run_doctor_live", return_value=_mock_report()):
            main([])
        # run_doctor_live prints to console directly; main just returns exit code,
        # so capsys.readouterr() is intentionally not asserted on here.
        capsys.readouterr()

    def test_exit_code_0_when_agents_ready(self):
        with patch("belt.commands.doctor.run_doctor_live") as mock_run:
            mock_run.return_value = _mock_report(
                agent_checks=[CheckResult(ok=True, label="test", detail="ok")],
            )
            rc = main([])
        assert rc == 0

    def test_exit_code_1_when_no_agents(self):
        with patch("belt.commands.doctor.run_doctor_live") as mock_run:
            mock_run.return_value = _mock_report(
                agent_checks=[CheckResult(ok=False, label="test", detail="missing")],
            )
            rc = main([])
        assert rc == 1

    def test_json_includes_sandbox_block(self, capsys):
        """``--json`` exposes sandbox provider readiness, mirroring exporters.

        The terminal output already prints a Sandbox section; the JSON
        surface must publish the same data so scripted callers (CI gates,
        wrapper tools) can assert ``docker`` readiness without parsing
        Rich-formatted text.
        """
        report = _mock_report(
            sandbox_checks=[
                CheckResult(ok=True, label="host", detail="built-in"),
                CheckResult(ok=True, label="docker", detail="built-in (Docker version 28.4.0)"),
            ]
        )
        with patch("belt.commands.doctor.run_doctor", return_value=report):
            main(["--json"])
        data = json.loads(capsys.readouterr().out)

        assert "sandbox" in data, f"--json output missing 'sandbox' key; got {sorted(data)}"
        assert "sandbox_ready" in data
        assert data["sandbox_ready"] == 2
        names = [c["name"] for c in data["sandbox"]]
        assert names == ["host", "docker"]
        for entry in data["sandbox"]:
            assert {"name", "ok", "detail", "suggestion"}.issubset(entry.keys())

    def test_json_exit_code_matches_agent_status(self, capsys):
        with patch("belt.commands.doctor.run_doctor") as mock_run:
            mock_run.return_value = _mock_report(
                agent_checks=[CheckResult(ok=False, label="none", detail="missing")],
            )
            rc = main(["--json"])
        assert rc == 1
        data = json.loads(capsys.readouterr().out)
        assert data["agents_ready"] == 0


# ── Judge model check ──


class TestCheckJudgeModel:
    """Doctor must surface model + source-of-truth or '(not set)'.

    Tell the operator *which* layer supplied the resolved model so confusion
    ("why is it routing to OpenAI when my yaml says azure?") is one ``doctor``
    invocation away.
    """

    @staticmethod
    def _isolated_env() -> dict[str, str]:
        # The fixture runs every doctor sub-test; explicitly clear the
        # BELT_LLM_* keys so the developer's shell can't pollute results.
        import os as _os

        return {k: "" for k in _os.environ if k.startswith("BELT_LLM_")}

    def test_unset_when_no_layer_supplies_model(self, tmp_path, monkeypatch):
        # Walk up from a clean tmp dir that has no belt.yaml above it.
        monkeypatch.chdir(tmp_path)
        for k in [k for k in list(__import__("os").environ) if k.startswith("BELT_LLM_")]:
            monkeypatch.delenv(k, raising=False)

        result = _check_judge_model()
        assert not result.ok
        assert "(not set)" in result.detail
        assert "BELT_LLM_MODEL" in result.suggestion
        assert "belt.yaml" in result.suggestion
        assert "--scorer-arg" in result.suggestion

    def test_env_source_attribution(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for k in [k for k in list(__import__("os").environ) if k.startswith("BELT_LLM_")]:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("BELT_LLM_MODEL", "openai/gpt-5.4-mini")

        result = _check_judge_model()
        assert result.ok
        assert "openai/gpt-5.4-mini" in result.detail
        assert "from env" in result.detail

    def test_yaml_source_attribution(self, tmp_path, monkeypatch):
        (tmp_path / "belt.yaml").write_text("llm:\n  model: openai/gpt-5.4-mini\n")
        monkeypatch.chdir(tmp_path)
        for k in [k for k in list(__import__("os").environ) if k.startswith("BELT_LLM_")]:
            monkeypatch.delenv(k, raising=False)

        result = _check_judge_model()
        assert result.ok
        assert "openai/gpt-5.4-mini" in result.detail
        assert "from yaml" in result.detail
