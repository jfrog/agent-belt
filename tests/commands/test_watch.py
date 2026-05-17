# (c) JFrog Ltd. (2026)

"""Tests for the live agent output viewer (watch module)."""

from __future__ import annotations

import json
from pathlib import Path

from belt.agent.base import BaseAgentAdapter
from belt.commands.watch import StreamEvent, StreamParser, _discover_stream_files, _FileWatcher
from belt.entities import AgentConfig, GroupConfig, Scenario, Turn, TurnExpectation, TurnOutput
from belt.runner.orchestrator import build_agent_config, run_scenario_turns


class TestStreamParser:
    """StreamParser handles common NDJSON event shapes from built-in agents."""

    def setup_method(self):
        self.parser = StreamParser()

    def test_empty_line_returns_none(self):
        assert self.parser.parse_line("") is None
        assert self.parser.parse_line("   ") is None

    def test_invalid_json_returns_none(self):
        assert self.parser.parse_line("not json") is None

    def test_non_dict_json_returns_none(self):
        assert self.parser.parse_line("[1, 2, 3]") is None

    # ── Tool events ──

    def test_tool_use_event(self):
        event = json.dumps({"type": "tool_use", "name": "Read", "input": {"path": "src/main.py"}})
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.icon == "🔧"
        assert "Read" in result.summary
        assert "src/main.py" in result.summary

    def test_tool_call_event(self):
        event = json.dumps({"type": "tool_call", "name": "readFile", "args": {"path": "index.ts"}})
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.icon == "🔧"
        assert "readFile" in result.summary

    def test_function_call_event(self):
        event = json.dumps({"type": "function_call", "name": "search", "arguments": {"query": "bug"}})
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.icon == "🔧"
        assert "search" in result.summary

    # ── Assistant / text events ──

    def test_claude_assistant_with_text(self):
        event = json.dumps(
            {
                "type": "assistant",
                "content": [{"type": "text", "text": "I found the issue in auth.py"}],
            }
        )
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.icon == "💬"
        assert "auth.py" in result.summary

    def test_claude_assistant_with_thinking(self):
        event = json.dumps(
            {
                "type": "assistant",
                "content": [{"type": "thinking", "thinking": "Let me analyze the code"}],
            }
        )
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.icon == "💭"
        assert "analyze" in result.summary

    def test_claude_assistant_with_tool_use_block(self):
        event = json.dumps(
            {
                "type": "assistant",
                "content": [{"type": "tool_use", "name": "Grep", "input": {"pattern": "TODO"}}],
            }
        )
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.icon == "🔧"
        assert "Grep" in result.summary

    def test_claude_assistant_empty_content_returns_none(self):
        event = json.dumps({"type": "assistant", "content": []})
        assert self.parser.parse_line(event) is None

    # ── Claude meta-tools (Skill, ToolSearch) - distinct icons ──

    def test_claude_skill_call_uses_wand_icon(self):
        event = json.dumps(
            {
                "type": "assistant",
                "content": [{"type": "tool_use", "name": "Skill", "input": {"skill": "bookstore-assistant"}}],
            }
        )
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.icon == "🪄"
        assert "Skill" in result.summary

    def test_claude_toolsearch_call_uses_magnifier_icon(self):
        event = json.dumps(
            {
                "type": "assistant",
                "content": [{"type": "tool_use", "name": "ToolSearch", "input": {"query": "select:mcp__*"}}],
            }
        )
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.icon == "🔎"
        assert "ToolSearch" in result.summary

    def test_claude_skill_result_is_suppressed(self):
        # The 🪄 Skill(...) call line already carries the skill name; the
        # follow-up "Launching skill: X" tool_result would just duplicate it.
        event = json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [{"type": "tool_result", "content": "Launching skill: bookstore-assistant"}],
                },
                "tool_use_result": {"success": True, "commandName": "bookstore-assistant"},
            }
        )
        assert self.parser.parse_line(event) is None

    def test_claude_toolsearch_result_renders_discovered_tool_names(self):
        event = json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "content": [
                                {"type": "tool_reference", "tool_name": "mcp__folio__search_books"},
                                {"type": "tool_reference", "tool_name": "mcp__folio__check_stock"},
                            ],
                        }
                    ],
                },
                "tool_use_result": {
                    "matches": [],
                    "query": "select:mcp__folio__*",
                    "total_deferred_tools": 56,
                },
            }
        )
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.icon == "🔎"
        # Count must reflect what we list, not the catalog-wide deferred count.
        assert "2 tools" in result.summary
        assert "mcp__folio__search_books" in result.summary
        assert "mcp__folio__check_stock" in result.summary
        assert "56" not in result.summary

    def test_claude_toolsearch_result_with_no_matches(self):
        event = json.dumps(
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "content": []}]},
                "tool_use_result": {"matches": [], "query": "select:nonexistent", "total_deferred_tools": 0},
            }
        )
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.icon == "🔎"
        assert "no matches" in result.summary

    def test_claude_mcp_tool_call_uses_wrench_icon(self):
        # Domain tools (MCP, Read, Edit, Bash, ...) keep the 🔧 icon.
        event = json.dumps(
            {
                "type": "assistant",
                "content": [{"type": "tool_use", "name": "mcp__folio__search_books", "input": {"query": "Stephenson"}}],
            }
        )
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.icon == "🔧"
        assert "mcp__folio__search_books" in result.summary

    def test_claude_mcp_tool_result_uses_paperclip_icon(self):
        event = json.dumps(
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "content": '{"books":[]}'}]},
                "tool_use_result": {"content": '{"books":[]}', "structuredContent": {"books": []}},
            }
        )
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.icon == "📎"
        assert "books" in result.summary

    # ── Gemini events ──

    def test_gemini_model_message_string(self):
        event = json.dumps({"type": "message", "role": "model", "content": "The fix is simple"})
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.icon == "💬"
        assert "fix" in result.summary

    def test_gemini_model_message_function_call(self):
        event = json.dumps(
            {
                "type": "message",
                "role": "model",
                "content": [{"type": "functionCall", "name": "writeFile", "args": {"path": "a.txt"}}],
            }
        )
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.icon == "🔧"
        assert "writeFile" in result.summary

    def test_user_message_ignored(self):
        event = json.dumps({"type": "message", "role": "user", "content": "fix the bug"})
        assert self.parser.parse_line(event) is None

    # ── Result events ──

    def test_result_with_cost(self):
        event = json.dumps({"type": "result", "total_cost_usd": 0.0312, "duration_ms": 5400})
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.icon == "✅"
        assert "$0.0312" in result.summary
        assert "5.4s" in result.summary

    def test_result_error(self):
        event = json.dumps({"type": "result", "is_error": True, "duration_ms": 1000})
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.icon == "❌"
        assert "ERROR" in result.summary

    def test_gemini_result_with_stats(self):
        event = json.dumps({"type": "result", "status": "success", "stats": {"duration_ms": 3200}})
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.icon == "✅"
        assert "3.2s" in result.summary

    def test_gemini_result_error_status(self):
        event = json.dumps({"type": "result", "status": "error", "error": {"type": "timeout"}})
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.icon == "❌"

    # ── Truncation ──

    def test_long_text_truncated(self):
        long_text = "x" * 200
        event = json.dumps({"type": "assistant", "content": [{"type": "text", "text": long_text}]})
        result = self.parser.parse_line(event)
        assert result is not None
        assert len(result.summary) < 200
        assert result.summary.endswith("…")

    def test_long_args_truncated(self):
        event = json.dumps({"type": "tool_use", "name": "Write", "input": {"content": "a" * 200}})
        result = self.parser.parse_line(event)
        assert result is not None
        assert "…" in result.summary

    # ── Unknown event types ──

    def test_unknown_event_type_returns_none(self):
        event = json.dumps({"type": "system", "data": "startup"})
        assert self.parser.parse_line(event) is None

    def test_init_event_returns_none(self):
        event = json.dumps({"type": "init", "session_id": "abc123"})
        assert self.parser.parse_line(event) is None


class TestStreamEvent:
    def test_str_representation(self):
        e = StreamEvent("🔧", "Read(path=src/main.py)")
        assert str(e) == "🔧 Read(path=src/main.py)"

    def test_summary_is_markup_default_false(self):
        e = StreamEvent("🔧", "Read(path=src/main.py)")
        assert e.summary_is_markup is False

    def test_summary_is_markup_explicitly_true(self):
        e = StreamEvent("✅", "result: [green]$0.42[/green]", summary_is_markup=True)
        assert e.summary_is_markup is True


class TestStreamEventTrustContract:
    """Trust contract for ``StreamEvent.summary``:
    framework-built result events carry trusted Rich markup; agent-derived
    summaries are plain text and must be escaped at render time."""

    def setup_method(self):
        self.parser = StreamParser()

    def test_result_event_marked_as_trusted_markup(self):
        event = json.dumps({"type": "result", "total_cost_usd": 0.4173, "duration_ms": 71400})
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.summary_is_markup is True
        assert "[green]" in result.summary  # raw markup is preserved in the field

    def test_result_done_fallback_is_plain_text(self):
        event = json.dumps({"type": "result"})  # no cost, no duration
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.summary == "done"
        assert result.summary_is_markup is False

    def test_result_error_event_marked_as_trusted_markup(self):
        event = json.dumps({"type": "result", "is_error": True, "duration_ms": 1000})
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.summary_is_markup is True

    def test_tool_event_is_plain_text(self):
        event = json.dumps({"type": "tool_use", "name": "Read", "input": {"path": "src/main.py"}})
        result = self.parser.parse_line(event)
        assert result is not None
        assert result.summary_is_markup is False


class TestPrintEventRendering:
    """``_print_event`` must render trusted framework markup as styled text and
    escape agent-controlled markup so it surfaces as literal characters."""

    def _render(self, event: StreamEvent) -> str:
        """Render an event through ``_print_event`` and return the captured text."""
        from io import StringIO

        from rich.console import Console

        from belt.commands.watch import _print_event

        # ``no_color=True`` strips ANSI escape codes from the recorded output
        # but Rich still parses markup; the visible characters are what end up
        # on a real terminal minus colour styling.
        console = Console(file=StringIO(), width=200, force_terminal=True, no_color=True)
        _print_event(console, "scen-label", event)
        return console.file.getvalue()

    def test_result_event_renders_cost_without_markup_leak(self):
        """A framework-built result event must render the cost as styled text
        (no literal ``[green]`` substring should appear in the output)."""
        parser = StreamParser()
        event = json.dumps({"type": "result", "total_cost_usd": 0.4173, "duration_ms": 71400})
        result = parser.parse_line(event)
        assert result is not None
        out = self._render(result)
        assert "[green]" not in out, f"trusted markup leaked as literal text: {out!r}"
        assert "[/green]" not in out, f"trusted markup leaked as literal text: {out!r}"
        assert "$0.4173" in out
        assert "71.4s" in out

    def test_agent_injected_markup_is_escaped(self):
        """An agent-controlled summary containing markup-like substrings must
        appear as literal text - never interpreted by Rich."""
        evt = StreamEvent("🔧", "evil([red]injected[/red])")
        out = self._render(evt)
        # The literal brackets survive because rich_safe escapes them; once
        # ``Text.from_markup`` renders the escaped form, the visible characters
        # include the brackets but no styling was applied.
        assert "[red]injected[/red]" in out, (
            "agent-controlled markup must surface as plain characters, not be parsed; " f"got {out!r}"
        )

    def test_tool_event_renders_args_safely(self):
        """A tool call summary with realistic file-path args renders without errors."""
        parser = StreamParser()
        event = json.dumps({"type": "tool_use", "name": "Read", "input": {"path": "/tmp/x.py"}})
        result = parser.parse_line(event)
        assert result is not None
        out = self._render(result)
        assert "Read" in out
        assert "/tmp/x.py" in out


class TestAgentDelegation:
    """StreamParser delegates to agent's parse_stream_event when provided."""

    def test_cursor_mcp_tool_via_agent(self):
        from belt.agent.cursor import CursorAgentAdapter

        parser = StreamParser(agent_cls=CursorAgentAdapter)
        event = json.dumps(
            {
                "type": "tool_call",
                "subtype": "started",
                "call_id": "toolu_abc",
                "tool_call": {
                    "mcpToolCall": {
                        "args": {
                            "toolName": "get_runtime_environment_status",
                            "args": {"environmentName": "production"},
                        }
                    }
                },
            }
        )
        result = parser.parse_line(event)
        assert result is not None
        assert result.icon == "🔧"
        assert "get_runtime_environment_status" in result.summary
        assert "production" in result.summary

    def test_cursor_builtin_tool_via_agent(self):
        from belt.agent.cursor import CursorAgentAdapter

        parser = StreamParser(agent_cls=CursorAgentAdapter)
        event = json.dumps(
            {
                "type": "tool_call",
                "subtype": "started",
                "tool_call": {"readToolCall": {"args": {"path": "/tmp/foo.py"}}},
            }
        )
        result = parser.parse_line(event)
        assert result is not None
        assert "Read" in result.summary

    def test_cursor_completed_renders_result(self):
        from belt.agent.cursor import CursorAgentAdapter

        parser = StreamParser(agent_cls=CursorAgentAdapter)
        event = json.dumps(
            {
                "type": "tool_call",
                "subtype": "completed",
                "tool_call": {
                    "mcpToolCall": {
                        "args": {"toolName": "get_status"},
                        "result": {"success": {"content": [{"text": "ok"}]}},
                    }
                },
            }
        )
        result = parser.parse_line(event)
        assert result is not None
        assert result.icon == "✅"
        assert "get_status" in result.summary

    def test_cursor_completed_error_renders_error(self):
        from belt.agent.cursor import CursorAgentAdapter

        parser = StreamParser(agent_cls=CursorAgentAdapter)
        event = json.dumps(
            {
                "type": "tool_call",
                "subtype": "completed",
                "tool_call": {
                    "mcpToolCall": {
                        "args": {"toolName": "bad_tool"},
                        "result": {"error": {"message": "not found"}},
                    }
                },
            }
        )
        result = parser.parse_line(event)
        assert result is not None
        assert result.icon == "❌"
        assert "not found" in result.summary

    def test_generic_events_still_work_with_agent(self):
        from belt.agent.cursor import CursorAgentAdapter

        parser = StreamParser(agent_cls=CursorAgentAdapter)
        event = json.dumps({"type": "function_call", "name": "search", "arguments": {"query": "test"}})
        result = parser.parse_line(event)
        assert result is not None
        assert "search" in result.summary

    def test_no_agent_falls_through_generic(self):
        parser = StreamParser()
        event = json.dumps({"type": "tool_call", "name": "readFile", "args": {"path": "x.ts"}})
        result = parser.parse_line(event)
        assert result is not None
        assert "readFile" in result.summary

    def test_base_agent_returns_none(self):
        assert BaseAgentAdapter.parse_stream_event({"type": "tool_call"}) is None


class TestFileWatcher:
    def test_reads_new_lines(self, tmp_path: Path):
        stream_file = tmp_path / "run" / "group" / "turn_0_stream.ndjson"
        stream_file.parent.mkdir(parents=True)
        stream_file.write_text("")

        watcher = _FileWatcher(stream_file, tmp_path / "run")
        assert watcher.read_new_lines() == []

        stream_file.write_text('{"type":"tool_use","name":"Read"}\n')
        lines = watcher.read_new_lines()
        assert len(lines) == 1
        assert "Read" in lines[0]

    def test_incremental_reads(self, tmp_path: Path):
        stream_file = tmp_path / "turn_0_stream.ndjson"
        stream_file.write_text('{"line":1}\n')

        watcher = _FileWatcher(stream_file, tmp_path)
        assert len(watcher.read_new_lines()) == 1

        with open(stream_file, "a") as f:
            f.write('{"line":2}\n')
        lines = watcher.read_new_lines()
        assert len(lines) == 1
        assert '"line":2' in lines[0]

    def test_label_from_path(self, tmp_path: Path):
        run = tmp_path / "run"
        stream = run / "claude-code" / "my_scenario" / "turn_0_stream.ndjson"
        stream.parent.mkdir(parents=True)
        stream.write_text("")
        watcher = _FileWatcher(stream, run)
        assert "claude-code" in watcher.label
        assert "my_scenario" in watcher.label

    def test_missing_file_returns_empty(self, tmp_path: Path):
        stream = tmp_path / "nonexistent" / "turn_0_stream.ndjson"
        watcher = _FileWatcher(stream, tmp_path)
        assert watcher.read_new_lines() == []


class TestDiscoverStreamFiles:
    def test_finds_stream_files(self, tmp_path: Path):
        (tmp_path / "group" / "scenario").mkdir(parents=True)
        (tmp_path / "group" / "scenario" / "turn_0_stream.ndjson").write_text("{}\n")
        (tmp_path / "group" / "scenario" / "turn_1_stream.ndjson").write_text("{}\n")
        (tmp_path / "group" / "scenario" / "turn_0_cli.txt").write_text("raw")

        files = _discover_stream_files(tmp_path)
        assert len(files) == 2
        assert all("stream.ndjson" in f.name for f in files)

    def test_empty_dir_returns_empty(self, tmp_path: Path):
        assert _discover_stream_files(tmp_path) == []


class _StreamingStub(BaseAgentAdapter):
    def setup(self, config: AgentConfig) -> None:
        pass

    def execute(self, message: str, flags: list[str]) -> str:
        if self._stream_sink is not None:
            self._stream_sink.write('{"type":"tool_use","name":"Read"}\n')
            self._stream_sink.flush()
        return '{"type":"result"}'

    def fetch_results(self, raw_output: str) -> TurnOutput:
        return TurnOutput(raw_cli=raw_output, reply_text="ok", has_reply=True)

    def teardown(self) -> None:
        pass


class TestStreamSinkIntegration:
    """Verify stream files are written by the orchestrator."""

    def test_stream_files_created(self, tmp_path: Path):
        agent = _StreamingStub()
        scenario = Scenario(
            name="stream_test",
            description="test",
            turns=[Turn(message="hi", expect=TurnExpectation())],
        )
        outcome = tmp_path / "outcome"
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)

        run_scenario_turns(agent, scenario, outcome, config, stream=True)

        stream_file = outcome / "turn_0_stream.ndjson"
        assert stream_file.exists()
        content = stream_file.read_text()
        assert "Read" in content

    def test_no_stream_files_when_disabled(self, tmp_path: Path):
        agent = _StreamingStub()
        scenario = Scenario(
            name="no_stream_test",
            description="test",
            turns=[Turn(message="hi", expect=TurnExpectation())],
        )
        outcome = tmp_path / "outcome"
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)

        run_scenario_turns(agent, scenario, outcome, config, stream=False)

        stream_file = outcome / "turn_0_stream.ndjson"
        assert not stream_file.exists()

    def test_stream_sink_reset_after_turn(self, tmp_path: Path):
        agent = _StreamingStub()
        scenario = Scenario(
            name="reset_test",
            description="test",
            turns=[
                Turn(message="turn 0", expect=TurnExpectation()),
                Turn(message="turn 1", expect=TurnExpectation()),
            ],
        )
        outcome = tmp_path / "outcome"
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)

        run_scenario_turns(agent, scenario, outcome, config, stream=True)

        assert agent._stream_sink is None
        assert (outcome / "turn_0_stream.ndjson").exists()
        assert (outcome / "turn_1_stream.ndjson").exists()
