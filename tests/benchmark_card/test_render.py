# (c) JFrog Ltd. (2026)

r"""Markdown rendering and injection defences for ``benchmark_card.render``.

The card is appended to the same ``$GITHUB_STEP_SUMMARY`` sink the
aggregator targets. A malicious ``--version`` binary on the runner's
PATH or an attacker-controlled path string must not be able to break
out of an inline-code span or table cell. The renderer is the second
line of defence behind ``_capture_cli_version`` (which strips control
chars at capture time); both must hold for the persisted Markdown to
be safe.
"""

from __future__ import annotations

from typing import Any

from belt.benchmark_card import BenchmarkCard, render_markdown

from .conftest import minimal_card


class TestRenderMarkdown:
    def test_includes_all_sections(self) -> None:
        card = minimal_card()
        md = render_markdown(card)
        for heading in (
            "# Benchmark Card",
            "## Run identity",
            "## Invocation",
            "## Scenarios",
            "## Scoring",
            "## Runtime",
            "## Summary",
        ):
            assert heading in md, f"missing heading: {heading}"

    def test_pipe_in_value_is_escaped(self) -> None:
        card = minimal_card()
        card.invocation.argv = ["echo", "a|b"]
        md = render_markdown(card)
        # Pipes inside table cells must be escaped.
        assert "a\\|b" in md
        # Sanity: no malformed table rows from injected pipes.
        assert "| a | b |" not in md


class TestMarkdownInjectionDefenses:
    def _card_with(self, **overrides: Any) -> BenchmarkCard:
        from belt.benchmark_card import AgentIdentity, AgentProvenance, CliIdentity, FixtureProvenance

        card = minimal_card()
        if "agent" in overrides:
            o = overrides["agent"]
            card.agents = [
                AgentProvenance(
                    group=o.get("group", "g"),
                    agent=AgentIdentity(
                        name=o.get("name", o.get("agent", "a")),
                        adapter_class=o.get("adapter_class", "Cls"),
                        args=o.get("args", o.get("agent_args", {})),
                        auth_signals=o.get("auth_signals", []),
                    ),
                    cli=CliIdentity(
                        binary_path=o.get("binary_path", o.get("cli_binary_path")),
                        version=o.get("version", o.get("cli_version")),
                    ),
                )
            ]
        if "fixture" in overrides:
            card.fixtures = [FixtureProvenance(**overrides["fixture"])]
        return card

    def test_backtick_in_cli_version_does_not_close_code_span(self) -> None:
        card = self._card_with(agent={"cli_version": "1.0\n`# Free Money Click Here`"})
        md = render_markdown(card)
        # The hostile backtick-wrapped heading must not survive verbatim
        # (which would close the surrounding code span and emit a real
        # H1 in the rendered Markdown). Backticks are rewritten to ``'``.
        assert "`# Free Money Click Here`" not in md
        # No literal heading at the start of any line - the value lives
        # inside an inline-code span so any ``#`` is interpreted as
        # plain text by GFM, not a heading marker.
        for line in md.splitlines():
            assert not line.startswith("# Free Money"), f"heading injection: {line!r}"
        # The newline is stripped so the value stays in one cell.
        assert "1.0'# Free Money Click Here'" in md

    def test_newline_in_cli_version_does_not_split_table_row(self) -> None:
        card = self._card_with(agent={"cli_version": "1.0\n| INJECTED | ROW |"})
        md = render_markdown(card)
        # The injected row would only render as a separate row if the
        # newline survived. The inline-code sanitiser strips control
        # chars, so the entire string lives inside one cell.
        assert "| INJECTED | ROW |" not in md
        assert "INJECTED" in md  # value is still surfaced, just defanged

    def test_pipe_in_working_dir_does_not_split_row(self) -> None:
        card = self._card_with(
            fixture={
                "group": "demo",
                "working_dir": "/tmp/a|b/c",
                "tracked": False,
                "git_sha": None,
                "git_ref": None,
                "auto_initialized": False,
                "dirty_files": 0,
            }
        )
        md = render_markdown(card)
        assert "/tmp/a\\|b/c" in md

    def test_ansi_escape_byte_is_stripped(self) -> None:
        # The ESC byte (0x1B) drives terminal escape sequences; once it's
        # removed, the residual ``[31m...[0m`` characters render as plain
        # text in Markdown - harmless. The renderer is the second line of
        # defence behind ``_capture_cli_version`` which strips the whole
        # sequence at capture time.
        card = self._card_with(agent={"cli_version": "\x1b[31mEVIL\x1b[0m 1.0"})
        md = render_markdown(card)
        assert "\x1b" not in md
        assert "EVIL" in md  # value still surfaced
        assert "1.0" in md

    def test_nul_byte_in_cli_path_is_stripped(self) -> None:
        card = self._card_with(agent={"cli_binary_path": "/usr/bin/agent\x00malicious"})
        md = render_markdown(card)
        assert "\x00" not in md
