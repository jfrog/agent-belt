# (c) JFrog Ltd. (2026)

"""Tests for shared agent helpers - resolve_binary, _detect_auth_signals,
BaseAgentAdapter.auth_signals classmethod, and the bounded
``_capture_cli_version`` adversarial defences."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from belt.agent.base import BaseAgentAdapter, _detect_auth_signals, resolve_binary


class TestResolveBinary:
    """Binary discovery prefers PATH, then falls back to declared install
    locations. Centralizes the pattern so agents don't reinvent it."""

    def test_returns_none_when_no_candidates_found(self):
        with patch("shutil.which", return_value=None):
            assert resolve_binary(["nonexistent-tool"]) is None

    def test_finds_first_candidate_on_path(self):
        def _which(name):
            return f"/usr/bin/{name}" if name == "found" else None

        with patch("shutil.which", side_effect=_which):
            assert resolve_binary(["missing", "found", "also-missing"]) == "/usr/bin/found"

    def test_path_takes_precedence_over_extra_paths(self, tmp_path):
        """When a binary exists both on PATH and in extra_paths, PATH wins;
        this matches user expectation that explicit env config beats discovery."""
        binary = tmp_path / "tool"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)

        with patch("shutil.which", return_value="/usr/bin/tool"):
            result = resolve_binary(["tool"], [str(tmp_path)])
        assert result == "/usr/bin/tool"

    def test_falls_back_to_extra_paths_when_not_on_path(self, tmp_path):
        binary = tmp_path / "tool"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)

        with patch("shutil.which", return_value=None):
            result = resolve_binary(["tool"], [str(tmp_path)])
        assert result == str(binary)

    def test_skips_non_executable_files_in_extra_paths(self, tmp_path):
        """A file with no execute bit shouldn't be returned - guards against
        partial installs leaving a non-executable stub."""
        binary = tmp_path / "tool"
        binary.write_text("not executable")
        binary.chmod(0o644)

        with patch("shutil.which", return_value=None):
            result = resolve_binary(["tool"], [str(tmp_path)])
        assert result is None

    def test_skips_directories_with_matching_name(self, tmp_path):
        (tmp_path / "tool").mkdir()
        with patch("shutil.which", return_value=None):
            assert resolve_binary(["tool"], [str(tmp_path)]) is None

    def test_expands_user_in_extra_paths(self, tmp_path, monkeypatch):
        """`~/.local/bin` style paths must be expanded - the official Cursor
        installer drops binaries there before $HOME/.local/bin is on PATH."""
        monkeypatch.setenv("HOME", str(tmp_path))
        bin_dir = tmp_path / ".local" / "bin"
        bin_dir.mkdir(parents=True)
        binary = bin_dir / "tool"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)

        with patch("shutil.which", return_value=None):
            result = resolve_binary(["tool"], ["~/.local/bin"])
        assert result == str(binary)

    def test_tries_candidates_in_order_within_extra_paths(self, tmp_path):
        """Resolution order matters - `cursor-agent` should be preferred over
        `cursor` even when both exist, to disambiguate from the IDE binary."""
        for name in ("cursor", "cursor-agent"):
            f = tmp_path / name
            f.write_text("#!/bin/sh\n")
            f.chmod(0o755)

        with patch("shutil.which", return_value=None):
            result = resolve_binary(["cursor-agent", "cursor"], [str(tmp_path)])
        assert result == str(tmp_path / "cursor-agent")

    def test_empty_candidates_returns_none(self):
        assert resolve_binary([]) is None

    def test_empty_string_candidate_skipped(self):
        with patch("shutil.which", return_value=None) as mock_which:
            resolve_binary(["", "real"])
        # `which` should only be called once (for "real"), not for the empty string
        assert mock_which.call_count == 1


class TestDetectAuthSignals:
    """Detects positive auth signals only - never claims `not authenticated`,
    because we cannot reliably distinguish "no auth" from "logged in via a
    mechanism we don't catalogue (system keychain, OAuth refresh token)"."""

    def test_no_signals_when_nothing_present(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FAKE_KEY", raising=False)
        assert _detect_auth_signals(["FAKE_KEY"], [tmp_path / "absent"]) == []

    def test_env_var_signal(self, monkeypatch):
        monkeypatch.setenv("MY_AGENT_KEY", "x")
        signals = _detect_auth_signals(["MY_AGENT_KEY"], [])
        assert signals == ["env MY_AGENT_KEY"]

    def test_first_env_var_match_wins(self, monkeypatch):
        """Agents can declare multiple env vars (e.g. GEMINI_API_KEY +
        GOOGLE_API_KEY) - only the first non-empty one is reported, to keep
        the doctor line scannable."""
        monkeypatch.setenv("KEY_A", "x")
        monkeypatch.setenv("KEY_B", "y")
        signals = _detect_auth_signals(["KEY_A", "KEY_B"], [])
        assert signals == ["env KEY_A"]

    def test_empty_env_var_skipped(self, monkeypatch):
        monkeypatch.setenv("EMPTY_KEY", "")
        monkeypatch.setenv("REAL_KEY", "x")
        signals = _detect_auth_signals(["EMPTY_KEY", "REAL_KEY"], [])
        assert signals == ["env REAL_KEY"]

    def test_stored_login_signal(self, tmp_path):
        cred = tmp_path / "auth"
        cred.mkdir()
        signals = _detect_auth_signals([], [cred])
        assert len(signals) == 1
        assert "stored login" in signals[0]
        assert str(cred) in signals[0]

    def test_stored_login_uses_tilde_for_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        cred = tmp_path / ".my-agent"
        cred.mkdir()
        signals = _detect_auth_signals([], [cred])
        assert "~/.my-agent" in signals[0]

    def test_stored_login_first_existing_wins(self, tmp_path):
        """If multiple credential paths are declared, the first existing one
        wins (avoids redundant lines like `stored login (.../session1) +
        stored login (.../session2)`)."""
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        signals = _detect_auth_signals([], [first, second])
        assert len(signals) == 1
        assert "first" in signals[0]
        assert "second" not in signals[0]

    def test_signal_order_env_then_path(self, tmp_path, monkeypatch):
        """Env var listed first because it's what runs in CI and is the most
        likely confusion source (`why is it picking up an old key?`)."""
        monkeypatch.setenv("MY_KEY", "x")
        cred = tmp_path / "auth"
        cred.mkdir()
        signals = _detect_auth_signals(["MY_KEY"], [cred])
        assert signals[0].startswith("env ")
        assert "stored login" in signals[1]

    def test_never_reads_file_contents(self, tmp_path):
        """Per the contract - existence-only check, never read contents.
        If the path is unreadable but exists, we still report it."""
        cred = tmp_path / "auth"
        cred.write_text("secret-token-xxx")
        cred.chmod(0o000)  # unreadable

        try:
            signals = _detect_auth_signals([], [cred])
            assert len(signals) == 1
        finally:
            cred.chmod(0o644)


class TestBaseAgentAuthSignals:
    def test_default_no_signals(self):
        """Agents that don't declare CREDENTIAL_ENV/PATHS report no signals,
        which doctor will surface as `auth: unknown` only if it sees something
        was declared. For unsupported agents, no auth line at all."""

        class MinimalAgentAdapter(BaseAgentAdapter):
            def setup(self, config):
                pass

            def execute(self, message, flags):
                return ""

            def fetch_results(self, raw):
                pass

            def teardown(self):
                pass

        assert MinimalAgentAdapter.auth_signals() == []

    def test_subclass_can_declare_credential_env(self, monkeypatch):
        monkeypatch.setenv("MY_AGENT_TOKEN", "x")

        class MyAgentAdapter(BaseAgentAdapter):
            CREDENTIAL_ENV = ("MY_AGENT_TOKEN",)

            def setup(self, config):
                pass

            def execute(self, message, flags):
                return ""

            def fetch_results(self, raw):
                pass

            def teardown(self):
                pass

        assert MyAgentAdapter.auth_signals() == ["env MY_AGENT_TOKEN"]

    def test_subclass_can_declare_credential_paths(self, tmp_path):
        cred = tmp_path / ".myagent"
        cred.mkdir()

        class MyAgentAdapter(BaseAgentAdapter):
            CREDENTIAL_PATHS = (cred,)

            def setup(self, config):
                pass

            def execute(self, message, flags):
                return ""

            def fetch_results(self, raw):
                pass

            def teardown(self):
                pass

        signals = MyAgentAdapter.auth_signals()
        assert len(signals) == 1
        assert "stored login" in signals[0]


class TestCaptureCliVersionBounded:
    """``_capture_cli_version`` must defend against hostile binaries on PATH.

    Three properties together: bounded memory regardless of how chatty
    the binary is, no leakage of ANSI / control characters into
    downstream renderers, and no hang when the binary refuses to exit.
    The values surface in the benchmark card sidecar; failure to defend
    here would let a malicious binary write whatever it wants into the
    persisted JSON.
    """

    def test_reads_at_most_cap_bytes(self, tmp_path: Path) -> None:
        from belt.agent.base import _VERSION_OUTPUT_CAP_BYTES

        # A hostile binary that emits more than the cap on the first line
        # but small enough to fit in the OS pipe buffer (so it can exit
        # cleanly with a 0 returncode). 32 KiB > the 4 KiB cap and well
        # under typical pipe buffer sizes (64 KiB on Linux, 16 KiB on
        # macOS once you reach EAGAIN). The cap, not the binary's
        # length, must determine our captured size.
        chatty = "X" * (32 * 1024)
        script = tmp_path / "chatty"
        script.write_text(f"#!/usr/bin/env python3\nimport sys\nsys.stdout.write({chatty!r})\nsys.stdout.flush()\n")
        script.chmod(0o755)

        result = BaseAgentAdapter._capture_cli_version([str(script)], timeout=5)
        assert result is not None
        # First-line truncation + read cap bound the result regardless of
        # how much stdout the binary produced.
        assert len(result) <= _VERSION_OUTPUT_CAP_BYTES

    def test_blocking_chatty_binary_is_killed_and_returns_none(self, tmp_path: Path) -> None:
        # Edge case: a binary that wants to emit MORE than the OS pipe
        # buffer will block on its own write once we've stopped reading
        # past the cap. The helper must not hang the runner: ``proc.wait``
        # times out and ``_kill_process_tree`` cleans up. Returns ``None``
        # because returncode != 0.
        script = tmp_path / "huge"
        # 10 MiB - guaranteed to overflow any sensible pipe buffer.
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "for _ in range(10240):\n"
            "    sys.stdout.write('X' * 1024)\n"
            "    sys.stdout.flush()\n"
        )
        script.chmod(0o755)

        import time

        start = time.monotonic()
        result = BaseAgentAdapter._capture_cli_version([str(script)], timeout=2.0)
        elapsed = time.monotonic() - start
        assert result is None
        # Cap + 2s teardown tolerance. A hang would burn the test timeout.
        assert elapsed < 8.0, f"capture took {elapsed:.1f}s - should have killed the process"

    def test_strips_control_chars(self, tmp_path: Path) -> None:
        script = tmp_path / "ansi"
        script.write_text("#!/usr/bin/env python3\nprint('\\x1b[31mfoo\\x1b[0m 1.2')\n")
        script.chmod(0o755)

        result = BaseAgentAdapter._capture_cli_version([str(script)], timeout=5)
        assert result == "foo 1.2"

    def test_timeout_returns_none_without_raising(self, tmp_path: Path) -> None:
        script = tmp_path / "hang"
        script.write_text("#!/usr/bin/env python3\nimport time; time.sleep(30)\n")
        script.chmod(0o755)

        # Tight timeout; the helper must kill the process and return None.
        result = BaseAgentAdapter._capture_cli_version([str(script)], timeout=1.0)
        assert result is None

    def test_missing_binary_returns_none(self) -> None:
        result = BaseAgentAdapter._capture_cli_version(["/no/such/binary"], timeout=1)
        assert result is None
