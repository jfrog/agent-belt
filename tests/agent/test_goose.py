# (c) JFrog Ltd. (2026)

"""Tests for GooseAgentAdapter ``fetch_results`` parsing.

These tests pin the contract that adjacent ``text`` blocks in the goose
NDJSON stream are concatenated *verbatim* - without injecting separators
between them. When goose drives a streaming backend (e.g. local Ollama
with ``llama3.2:3b-instruct``), each token arrives as its own ``text``
block. Joining with ``"\\n"`` would shred words like
``["Cla", "ude", "Code"] -> "Cla\\nude\\nCode"``, which is what caused
nine ``goose/*`` scenarios to fail rule checks looking for substrings
like ``ClaudeCodeAgentAdapter``.
"""

from __future__ import annotations

import json

import pytest

from belt.agent.goose import GooseAgentAdapter


@pytest.fixture
def agent() -> GooseAgentAdapter:
    return GooseAgentAdapter()


def _msg_event(*texts: str, role: str = "assistant") -> str:
    """Render one goose-style assistant message with N text blocks."""
    return json.dumps(
        {
            "type": "message",
            "message": {
                "id": "chatcmpl-1",
                "role": role,
                "created": 1,
                "content": [{"type": "text", "text": t} for t in texts],
                "metadata": {"userVisible": True, "agentVisible": True},
            },
        }
    )


def _thinking_event(*chunks: str) -> str:
    return json.dumps(
        {
            "type": "message",
            "message": {
                "id": "chatcmpl-2",
                "role": "assistant",
                "created": 2,
                "content": [{"type": "thinking", "thinking": c} for c in chunks],
                "metadata": {"userVisible": True, "agentVisible": True},
            },
        }
    )


class TestFetchResultsTokenStreaming:
    def test_token_deltas_are_concatenated_without_newlines(self, agent: GooseAgentAdapter) -> None:
        # Single-token text blocks across multiple streamed events -
        # exactly what local Ollama models emit.
        raw = "\n".join(
            [
                _msg_event("Cla"),
                _msg_event("ude"),
                _msg_event("Code"),
                _msg_event("Agent"),
                _msg_event("Adapter"),
            ]
        )

        out = agent.fetch_results(raw)

        assert out.reply_text == "ClaudeCodeAgentAdapter"
        assert "\n" not in out.reply_text

    def test_token_deltas_within_one_message_concatenate(self, agent: GooseAgentAdapter) -> None:
        # Same shred shape but carried inside a single assistant message
        # (some goose builds buffer a whole turn into one event).
        raw = _msg_event("Hello", " ", "world", "!")

        out = agent.fetch_results(raw)

        assert out.reply_text == "Hello world!"

    def test_model_emitted_newlines_are_preserved(self, agent: GooseAgentAdapter) -> None:
        # Newlines that the model itself emitted (inside a single ``text``
        # payload) must still appear in the final reply.
        raw = _msg_event("Line one\nLine two\n\nParagraph two")

        out = agent.fetch_results(raw)

        assert out.reply_text == "Line one\nLine two\n\nParagraph two"

    def test_multi_turn_assistant_messages_concatenate(self, agent: GooseAgentAdapter) -> None:
        # Two complete assistant turns. Concatenating without a separator
        # is the documented behaviour - agents that need explicit
        # paragraph breaks must emit them inside ``text`` payloads.
        raw = "\n".join([_msg_event("First reply."), _msg_event("Second reply.")])

        out = agent.fetch_results(raw)

        assert out.reply_text == "First reply.Second reply."
        assert out.has_reply is True


class TestFetchResultsThinking:
    def test_thinking_token_deltas_are_concatenated(self, agent: GooseAgentAdapter) -> None:
        raw = _thinking_event("Step ", "one", " then ", "step ", "two")

        out = agent.fetch_results(raw)

        assert out.thinking_text == "Step one then step two"

    def test_no_thinking_returns_none(self, agent: GooseAgentAdapter) -> None:
        raw = _msg_event("hi")

        out = agent.fetch_results(raw)

        assert out.thinking_text is None


class TestFetchResultsRuleSubstringExposesShredBug:
    """Regression test for the production failure that motivated the fix.

    The ``goose/grep_pattern`` scenario asserted ``contains(ClaudeCodeAgentAdapter)``
    against the agent's reply. Before the fix, the streamed reply was
    ``"Cla\\nude\\nCode\\nAgent\\nAdapter"`` and the rule failed even though the
    semantic answer was correct. After the fix the substring check passes.
    """

    def test_substring_check_on_streamed_class_name_passes(self, agent: GooseAgentAdapter) -> None:
        raw = "\n".join(
            [
                _msg_event("Here are the agents: "),
                _msg_event("Cla"),
                _msg_event("ude"),
                _msg_event("Code"),
                _msg_event("Agent"),
                _msg_event("Adapter"),
                _msg_event(", "),
                _msg_event("Cod"),
                _msg_event("ex"),
                _msg_event("Agent"),
                _msg_event("Adapter"),
                _msg_event("."),
            ]
        )

        out = agent.fetch_results(raw)

        assert "ClaudeCodeAgentAdapter" in out.reply_text
        assert "CodexAgentAdapter" in out.reply_text


def _tool_request_event(name: str, args: dict, *, call_id: str = "call_1") -> str:
    """Render a goose-style toolRequest event (production schema).

    Goose nests the tool name under ``toolCall.value.name`` rather than at
    the top level. Older flat ``toolCall.name`` shapes are also supported.
    """
    return json.dumps(
        {
            "type": "message",
            "message": {
                "id": "chatcmpl-tr",
                "role": "assistant",
                "created": 3,
                "content": [
                    {
                        "type": "toolRequest",
                        "id": call_id,
                        "toolCall": {"status": "success", "value": {"name": name, "arguments": args}},
                        "_meta": {"goose_extension": name},
                    }
                ],
                "metadata": {"userVisible": True, "agentVisible": True},
            },
        }
    )


class TestFetchResultsToolNameExtraction:
    """Regression test for the goose toolRequest schema.

    Production goose builds nest tool name + arguments inside ``toolCall.value``
    rather than at the top level. Before the fix, every ``ToolCall.name`` was
    captured as ``""`` - which broke ``tools_invoked_any`` rule checks and
    obscured what the agent actually did during a turn.
    """

    def test_tool_name_extracted_from_value_field(self, agent: GooseAgentAdapter) -> None:
        raw = _tool_request_event("rg", {"pattern": "BaseAgentAdapter", "path": "src/"})

        out = agent.fetch_results(raw)

        assert len(out.tool_calls) == 1
        assert out.tool_calls[0].name == "rg"
        assert out.tool_calls[0].args == {"pattern": "BaseAgentAdapter", "path": "src/"}
        assert out.tool_sequence == ["rg"]

    def test_tool_name_falls_back_to_flat_shape(self, agent: GooseAgentAdapter) -> None:
        # Tolerate older/alternate shapes that put ``name`` at the top level.
        raw = json.dumps(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolRequest",
                            "id": "call_flat",
                            "toolCall": {"name": "analyze", "arguments": {"path": "src/"}},
                        }
                    ],
                },
            }
        )

        out = agent.fetch_results(raw)

        assert out.tool_calls[0].name == "analyze"
        assert out.tool_calls[0].args == {"path": "src/"}

    def test_multiple_tools_keep_distinct_names(self, agent: GooseAgentAdapter) -> None:
        raw = "\n".join(
            [
                _tool_request_event("analyze", {"path": "src/"}, call_id="c1"),
                _tool_request_event("rg", {"pattern": "Agent"}, call_id="c2"),
                _tool_request_event("read_file", {"path": "src/x.py"}, call_id="c3"),
            ]
        )

        out = agent.fetch_results(raw)

        names = [tc.name for tc in out.tool_calls]
        assert names == ["analyze", "rg", "read_file"]
        assert out.tool_sequence == ["analyze", "rg", "read_file"]


class TestRuntimeInfoVersionProbe:
    """Pin the version-probe argv shape.

    The ``goose`` CLI accepts ``--version`` but rejects ``goose version``
    as an unrecognized subcommand (exit 2). The wrong shape here would
    surface as ``cli_version: null`` in the benchmark card without any
    other test failing. Pinning the argv means a future drift fails
    loudly in CI rather than silently null-ing the field.
    """

    def test_runtime_info_uses_double_dash_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import shutil

        monkeypatch.setattr(shutil, "which", lambda _: "/opt/homebrew/bin/goose")
        captured: dict[str, list[str]] = {}

        def fake_capture(cls, cmd: list[str], timeout: float = 5.0, env: dict[str, str] | None = None) -> str | None:
            captured["cmd"] = cmd
            return "1.30.0"

        monkeypatch.setattr(GooseAgentAdapter, "_capture_cli_version", classmethod(fake_capture))
        info = GooseAgentAdapter.runtime_info()

        assert info["cli_version"] == "1.30.0"
        assert captured["cmd"] == ["/opt/homebrew/bin/goose", "--version"], (
            "goose only accepts ``--version``; ``version`` returns "
            "``unrecognized subcommand`` and exit 2 in current releases."
        )
