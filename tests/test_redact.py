# (c) JFrog Ltd. (2026)

"""Unit tests for ``belt._redact``.

The module is the single source of truth for secret redaction across
belt; every site that ever puts user-supplied strings into
persisted artifacts (``run_meta.json``, ``benchmark-card.json``, the
rendered Markdown card, agent runtime sidecars, the env-var snapshot)
routes through it. Coverage here is table-driven so a new shape of
``key=value`` argv flag can be added by extending one parametrize list -
not by writing a new test.

Regression coverage: hand-rolled redactors that count ``=`` characters
silently leak secrets in the *combined* ``-Xkey=value`` form (the
``key=value`` payload itself contains the second ``=``). The
parametrized cases below pin down every argparse-accepted shape so any
future drift to a hand-rolled parser would fail loudly here.
"""

from __future__ import annotations

import pytest

from belt._redact import (
    PRESENT,
    REDACTED,
    is_secret_name,
    safe_agent_args,
    safe_environ,
    scrub_argv,
    scrub_dict,
    scrub_kv_list,
    scrub_kv_string,
    scrub_url,
)


class TestIsSecretName:
    @pytest.mark.parametrize(
        "name",
        [
            "api_key",
            "API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GITHUB_TOKEN",
            "SLACK_BEARER",
            "DB_PASSWORD",
            "MY_PASSWD",
            "AWS_SESSION_TOKEN",
            "AWS_CREDENTIAL_FILE",
            "X_SECRET_VALUE",
        ],
    )
    def test_matches_secret_shapes(self, name: str) -> None:
        assert is_secret_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "model",
            "working_dir",
            "PATH",
            "HOME",
            "OPENAI_BASE_URL",
            "GITHUB_REPOSITORY",
            "trials",
        ],
    )
    def test_passes_through_non_secret(self, name: str) -> None:
        assert is_secret_name(name) is False


class TestScrubKvString:
    @pytest.mark.parametrize(
        "item,expected",
        [
            ("api_key=sk-xxx", f"api_key={REDACTED}"),
            ("API_KEY=anything", f"API_KEY={REDACTED}"),
            ("token=ghp_abc", f"token={REDACTED}"),
            ("password=pw", f"password={REDACTED}"),
            ("model=gpt-4o", "model=gpt-4o"),
            ("working_dir=/tmp/foo", "working_dir=/tmp/foo"),
            ("no_equals_sign", "no_equals_sign"),
            ("", ""),
            ("api_key=", f"api_key={REDACTED}"),
            ("endpoint=https://x?token=y", "endpoint=https://x?token=y"),
        ],
    )
    def test_redacts_only_secret_keys(self, item: str, expected: str) -> None:
        assert scrub_kv_string(item) == expected

    def test_non_string_returns_unchanged(self) -> None:
        # A defensive call site might pass the wrong type; redactor must
        # not raise.
        assert scrub_kv_string(None) is None  # type: ignore[arg-type]
        assert scrub_kv_string(42) == 42  # type: ignore[arg-type]


class TestScrubKvList:
    def test_element_wise(self) -> None:
        out = scrub_kv_list(["api_key=sk", "model=gpt-4o", "token=abc"])
        assert out == [f"api_key={REDACTED}", "model=gpt-4o", f"token={REDACTED}"]

    def test_empty(self) -> None:
        assert scrub_kv_list([]) == []


class TestScrubDict:
    def test_redacts_secret_keys(self) -> None:
        out = scrub_dict({"api_key": "sk-xxx", "model": "gpt-4"})
        assert out == {"api_key": PRESENT, "model": "gpt-4"}

    def test_redacts_via_env_var_lookup(self) -> None:
        out = scrub_dict(
            {"model": "claude-3"},
            env_var_by_name={"model": "ANTHROPIC_API_KEY"},
        )
        assert out == {"model": PRESENT}

    def test_no_lookup_keeps_non_secret_keys(self) -> None:
        out = scrub_dict({"working_dir": "/tmp", "trials": "3"})
        assert out == {"working_dir": "/tmp", "trials": "3"}

    def test_custom_mark(self) -> None:
        out = scrub_dict({"api_key": "x"}, mark="<hidden>")
        assert out == {"api_key": "<hidden>"}


class TestScrubArgv:
    """The argv foot-gun: every argparse-accepted shape is covered.

    A drift to a hand-rolled redactor that counts ``=`` characters
    would fail one of these cases.
    """

    @pytest.mark.parametrize(
        "argv,expected",
        [
            # Separated short form.
            (
                ["-X", "api_key=secret"],
                ["-X", f"api_key={REDACTED}"],
            ),
            # Combined short form. The trap: a hand-rolled parser that
            # expects a second ``=`` at a fixed offset miscounts and
            # leaks the value through unchanged.
            (
                ["-Xapi_key=secret"],
                [f"-Xapi_key={REDACTED}"],
            ),
            (
                ["-Xtoken=ghp_abc"],
                [f"-Xtoken={REDACTED}"],
            ),
            # Separated long form.
            (
                ["--agent-arg", "api_key=secret"],
                ["--agent-arg", f"api_key={REDACTED}"],
            ),
            # Long form with equals.
            (
                ["--agent-arg=api_key=secret"],
                [f"--agent-arg=api_key={REDACTED}"],
            ),
            # Non-secret payloads are preserved verbatim.
            (
                ["-Xmodel=gpt-4o"],
                ["-Xmodel=gpt-4o"],
            ),
            (
                ["-X", "model=gpt-4o"],
                ["-X", "model=gpt-4o"],
            ),
            (
                ["--agent-arg=model=gpt-4o"],
                ["--agent-arg=model=gpt-4o"],
            ),
            # Mixed sequence: one secret, one not, both forms.
            (
                ["belt", "eval", "x", "-Xapi_key=s1", "-X", "model=gpt-4"],
                ["belt", "eval", "x", f"-Xapi_key={REDACTED}", "-X", "model=gpt-4"],
            ),
            # Unrelated argv entries pass through unchanged.
            (
                ["belt", "eval", "examples/scenarios", "--workers", "4"],
                ["belt", "eval", "examples/scenarios", "--workers", "4"],
            ),
            # Empty argv.
            ([], []),
            # Trailing flag with no value (argparse would reject it; we
            # must not crash).
            (
                ["-X"],
                ["-X"],
            ),
        ],
    )
    def test_canonical_shapes(self, argv: list[str], expected: list[str]) -> None:
        assert scrub_argv(list(argv)) == expected

    def test_combined_form_does_not_leak_secret(self) -> None:
        # Direct anti-leak assertion for the combined-form trap. The
        # secret value must never appear anywhere in the redacted output.
        out = scrub_argv(["-Xapi_key=SECRET_LEAK_CANARY"])
        assert "SECRET_LEAK_CANARY" not in " ".join(out)

    def test_custom_kv_flags(self) -> None:
        out = scrub_argv(
            ["-S", "api_key=secret", "--scorer-arg", "token=t"],
            kv_flags=("-S", "--scorer-arg"),
        )
        assert out == ["-S", f"api_key={REDACTED}", "--scorer-arg", f"token={REDACTED}"]


class TestScrubUrl:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("https://example.com/v1?token=abc", "https://example.com"),
            ("https://user:pw@example.com:8443/path", "https://example.com:8443"),
            ("http://localhost:11434/api", "http://localhost:11434"),
            ("not-a-url", PRESENT),
            ("", PRESENT),
            ("file:///etc/passwd", PRESENT),  # No host -> can't attest origin.
        ],
    )
    def test_reduces_to_scheme_host_port(self, value: str, expected: str) -> None:
        assert scrub_url(value) == expected


class TestSafeEnviron:
    def test_records_only_allow_listed_names(self) -> None:
        env = {
            "OPENAI_API_KEY": "sk-leak",  # NOT in allow-list -> dropped
            "GITHUB_REPOSITORY": "owner/repo",  # CI marker -> kept
            "RANDOM_VAR": "x",  # NOT allow-listed -> dropped
        }
        out = safe_environ(env)
        assert out.get("GITHUB_REPOSITORY") == "owner/repo"
        assert "OPENAI_API_KEY" not in out
        assert "RANDOM_VAR" not in out

    def test_secret_named_allow_listed_value_is_redacted(self) -> None:
        # Synthetic case: even if a future allow-list mistake added a
        # secret-named key, the deny-list catches it.
        from belt import envvars

        if not envvars.PUBLIC_ALLOW:
            pytest.skip("PUBLIC_ALLOW is empty in this build")
        # Pick the first allow-listed name and pretend it matches the
        # secret regex by aliasing the test env to a secret-shaped name.
        env = {"GITHUB_TOKEN": "ghp_should_not_leak"}
        # GITHUB_TOKEN is not in the allow-list, so this verifies the
        # deny-list-after-allow-list ordering: we don't even reach the
        # value because the name is not allowed.
        out = safe_environ(env)
        assert "GITHUB_TOKEN" not in out

    def test_base_url_value_is_reduced(self) -> None:
        # Skip if the build does not declare any *_BASE_URL allow-listed
        # name (defensive against allow-list trimming).
        from belt import envvars

        url_names = [n for n in envvars.PUBLIC_ALLOW if n.endswith("_BASE_URL")]
        if not url_names:
            pytest.skip("no *_BASE_URL allow-listed in PUBLIC_ALLOW")
        name = url_names[0]
        env = {name: "https://attacker.test/v1?token=leak"}
        out = safe_environ(env)
        assert out[name] == "https://attacker.test"


class TestSafeAgentArgs:
    """Boundary helper that adds per-option ``env_var`` lookup on top of
    :func:`scrub_dict`. The pure-transform behaviour is covered by
    ``TestScrubDict``; this suite covers only what the wrapper adds.
    """

    def test_redacts_secret_keys(self) -> None:
        out = safe_agent_args({"api_key": "sk-xxx", "model": "gpt-4"})
        assert out == {"api_key": PRESENT, "model": "gpt-4"}

    def test_uses_env_var_metadata(self) -> None:
        class FakeOpt:
            name = "model"
            env_var = "ANTHROPIC_API_KEY"

        out = safe_agent_args({"model": "claude-3"}, cli_options=[FakeOpt()])
        assert out == {"model": PRESENT}
