# (c) JFrog Ltd. (2026)

"""Tests for agent capability declaration (supported_output_fields)."""

from __future__ import annotations

from belt.agent.base import AGENT_SPECIFIC_FIELDS, UNIVERSAL_OUTPUT_FIELDS, BaseAgentAdapter


class TestFieldSets:
    def test_universal_fields_are_frozenset(self) -> None:
        assert isinstance(UNIVERSAL_OUTPUT_FIELDS, frozenset)

    def test_agent_specific_fields_are_frozenset(self) -> None:
        assert isinstance(AGENT_SPECIFIC_FIELDS, frozenset)

    def test_no_overlap(self) -> None:
        overlap = UNIVERSAL_OUTPUT_FIELDS & AGENT_SPECIFIC_FIELDS
        assert not overlap, f"Fields in both sets: {overlap}"


class TestBaseDefault:
    def test_base_default_returns_empty(self) -> None:
        assert BaseAgentAdapter.supported_output_fields() == frozenset()


class TestClaudeCode:
    def test_declares_capabilities(self) -> None:
        from belt.agent.claude_code import ClaudeCodeAgentAdapter

        fields = ClaudeCodeAgentAdapter.supported_output_fields()
        assert "tool_sequence" in fields
        assert "thinking_text" in fields
        assert "llm_turn_count" in fields

    def test_only_declares_known_fields(self) -> None:
        from belt.agent.claude_code import ClaudeCodeAgentAdapter

        fields = ClaudeCodeAgentAdapter.supported_output_fields()
        unknown = fields - AGENT_SPECIFIC_FIELDS
        assert not unknown, f"Unknown fields declared: {unknown}"


class TestCodex:
    def test_declares_capabilities(self) -> None:
        # codex-cli 0.130's JSONL stream emits ``agent_message`` and
        # ``command_execution`` items; ``thinking_text`` is not exposed
        # by that surface and is therefore not declared. If a future
        # codex release adds a ``reasoning`` item type, extend
        # ``fetch_results`` to capture it and add ``thinking_text`` here.
        from belt.agent.codex import CodexAgentAdapter

        fields = CodexAgentAdapter.supported_output_fields()
        assert "tool_sequence" in fields
        assert "llm_turn_count" in fields
        assert "thinking_text" not in fields

    def test_only_declares_known_fields(self) -> None:
        from belt.agent.codex import CodexAgentAdapter

        fields = CodexAgentAdapter.supported_output_fields()
        unknown = fields - AGENT_SPECIFIC_FIELDS
        assert not unknown, f"Unknown fields declared: {unknown}"


class TestGemini:
    def test_declares_tool_sequence(self) -> None:
        from belt.agent.gemini import GeminiAgentAdapter

        fields = GeminiAgentAdapter.supported_output_fields()
        assert "tool_sequence" in fields

    def test_does_not_declare_thinking(self) -> None:
        from belt.agent.gemini import GeminiAgentAdapter

        fields = GeminiAgentAdapter.supported_output_fields()
        assert "thinking_text" not in fields


class TestCursor:
    def test_declares_tool_sequence(self) -> None:
        from belt.agent.cursor import CursorAgentAdapter

        fields = CursorAgentAdapter.supported_output_fields()
        assert "tool_sequence" in fields

    def test_only_declares_known_fields(self) -> None:
        from belt.agent.cursor import CursorAgentAdapter

        fields = CursorAgentAdapter.supported_output_fields()
        unknown = fields - AGENT_SPECIFIC_FIELDS
        assert not unknown, f"Unknown fields declared: {unknown}"
