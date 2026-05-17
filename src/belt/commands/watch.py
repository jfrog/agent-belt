# (c) JFrog Ltd. (2026)

"""Live agent output viewer - watch NDJSON stream files as agents execute.

Standalone: ``belt watch [run_dir]``
Reusable:   ``StreamParser`` class renders NDJSON events into human-readable lines.

StreamParser is agent-agnostic - it recognises event shapes from Claude Code,
Gemini CLI, and Codex CLI without importing agent code.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console
from rich.text import Text

from belt.constants import OUTCOMES_ROOT

_TOOL_ICONS = {"tool_use": "🔧", "function_call": "🔧", "tool_call": "🔧"}
_MAX_ARG_DISPLAY = 120
_MAX_TEXT_DISPLAY = 120
_MAX_VALUE_DISPLAY = 80


class StreamEvent:
    """A single rendered event from an NDJSON stream.

    Trust contract:
      - ``summary`` is **plain text by default** (``summary_is_markup=False``).
        Rendering sites must escape it with ``rich_safe`` before passing it to
        ``Text.from_markup`` because the content may include attacker-controlled
        text (tool args, file paths, reply text from agent stdout).
      - ``summary_is_markup=True`` declares that the summary was constructed by
        framework code (e.g. ``_render_result``) and contains only
        framework-emitted Rich markup tokens around values that are not
        agent-controlled (cost floats, duration numbers, fixed labels).
        Rendering sites must NOT escape it - escaping turns ``[green]`` into
        ``\\[green]`` which renders as literal text.

    Agent ``parse_stream_event`` overrides should always return events with the
    default ``summary_is_markup=False``: plain text, no Rich markup. The
    framework owns styling.
    """

    __slots__ = ("icon", "summary", "summary_is_markup")

    def __init__(self, icon: str, summary: str, *, summary_is_markup: bool = False):
        self.icon = icon
        self.summary = summary
        self.summary_is_markup = summary_is_markup

    def __str__(self) -> str:
        return f"{self.icon} {self.summary}"


class StreamParser:
    """Parse NDJSON lines into human-readable StreamEvent objects.

    Generic rendering covers common NDJSON event shapes from built-in agents.
    For agent-specific formats, pass an ``agent_cls`` with a
    ``parse_stream_event(event) -> (icon, summary) | None`` static method.
    The agent gets first shot; if it returns None, generic rendering applies.
    """

    def __init__(self, agent_cls: type | None = None):
        self._agent_cls = agent_cls

    def parse_line(self, line: str) -> StreamEvent | None:
        """Parse a single NDJSON line into a StreamEvent, or None if not renderable."""
        line = line.strip()
        if not line:
            return None
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(event, dict):
            return None
        try:
            return self._render_event(event)
        except Exception:
            return None

    def _render_event(self, event: dict) -> StreamEvent | None:
        if self._agent_cls is not None:
            try:
                result = self._agent_cls.parse_stream_event(event)
                if result is not None:
                    if not result[0] and not result[1]:
                        return None
                    return StreamEvent(result[0], result[1])
            except Exception:
                pass

        etype = event.get("type", "")

        if etype == "user_input":
            msg = str(event.get("message", "")).replace("\n", " ")
            if len(msg) > 120:
                msg = msg[:117] + "…"
            return StreamEvent("👤", msg)

        if etype in ("tool_use", "tool_call"):
            return self._render_tool(event)

        if etype == "tool_result":
            return self._render_generic_tool_result(event)

        if etype == "function_call":
            name = event.get("name", "?")
            args_str = self._compact_args(event.get("arguments", {}))
            return StreamEvent("🔧", f"{name}({args_str})")

        if etype == "assistant":
            return self._render_content_blocks(event)

        if etype == "message":
            role = event.get("role", "")
            if role == "model":
                return self._render_gemini_message(event)
            if role in ("assistant", ""):
                return self._render_content_blocks(event)
            return None

        if etype == "result":
            return self._render_result(event)

        if etype == "user":
            return self._render_tool_result(event)

        if etype == "system" and event.get("subtype") == "init":
            model = event.get("model", "")
            tools = event.get("tools", [])
            parts = []
            if model:
                parts.append(f"model: {model}")
            if tools:
                parts.append(f"{len(tools)} tools")
            return StreamEvent("🔌", " · ".join(parts)) if parts else None

        return None

    def _render_tool(self, event: dict) -> StreamEvent | None:
        name = event.get("name") or event.get("tool_name") or "?"
        args = event.get("input") or event.get("args") or event.get("parameters") or {}
        args_str = self._compact_args(args)
        return StreamEvent("🔧", f"{name}({args_str})")

    def _render_generic_tool_result(self, event: dict) -> StreamEvent | None:
        status = event.get("status") or ""
        output = event.get("output") or event.get("result") or ""
        if isinstance(output, (dict, list)):
            output = json.dumps(output, separators=(",", ":"))
        output = str(output).replace("\n", " ").strip()
        is_error = status == "error" or event.get("is_error")
        icon = "❌" if is_error else "📎"
        if status and not output:
            return StreamEvent(icon, f"tool → {status}")
        if output:
            return StreamEvent(icon, _trunc(output, 80))
        return StreamEvent(icon, "tool result")

    def _render_tool_result(self, event: dict) -> StreamEvent | None:
        # Claude Code emits three distinct ``tool_use_result`` schemas
        # (observed in stream-json output, claude-code 2.1.x), each from
        # a different family of tools. We dispatch on the schema so the
        # live UI shows the right semantics for each family:
        #
        #   * Skill   - ``{success, commandName}``
        #       The companion ``user`` event that follows ``Skill(...)``.
        #       Its only signal is "skill X loaded", which the 🪄 call
        #       line already conveys, so we suppress this result row to
        #       avoid a redundant "Launching skill: X" line.
        #
        #   * ToolSearch - ``{matches, query, total_deferred_tools}``
        #       Claude's tool-discovery meta-tool. The result is a list
        #       of tool *names* the agent can call next (no args). We
        #       render with 🔎 and prefix with "→ N tools:" so it's
        #       unambiguous that this is discovery, not invocation.
        #
        #   * Everything else - ``{content, structuredContent,
        #     numLines?, numFiles?}``
        #       Real tool output (MCP, Read, Edit, Bash, ...). Render
        #       with 📎 and the response body.
        tur = event.get("tool_use_result")
        if tur is None:
            return None
        if not isinstance(tur, dict):
            return StreamEvent("📎", _trunc(str(tur), 80))

        # Family 1: Skill result - suppress, the 🪄 Skill(...) call line is enough.
        if "commandName" in tur:
            return None

        msg_blocks = (event.get("message") or {}).get("content") or []

        # Family 2: ToolSearch result - render discovered tool names with 🔎.
        if "matches" in tur and "query" in tur:
            names: list[str] = []
            for block in msg_blocks:
                if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                    continue
                body = block.get("content")
                if not isinstance(body, list):
                    continue
                for b in body:
                    if isinstance(b, dict) and b.get("tool_name"):
                        names.append(b["tool_name"])
            # ``total_deferred_tools`` is the catalog-wide count of tools
            # Claude is aware of, not the number returned by this search.
            # The user-visible count must match the names we actually list.
            if names:
                summary = f"→ {len(names)} tool{'s' if len(names) != 1 else ''}: {', '.join(names)}"
            else:
                summary = "→ no matches"
            return StreamEvent("🔎", _trunc(summary, 80))

        # Family 3: domain tool result.
        num_lines = tur.get("numLines") or 0
        num_files = tur.get("numFiles") or 0
        if num_lines or num_files:
            parts = []
            if num_lines:
                parts.append(f"{num_lines} lines")
            if num_files:
                parts.append(f"{num_files} files")
            return StreamEvent("📎", ", ".join(parts))

        content = tur.get("content", "")
        if content:
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            return StreamEvent("📎", _trunc(str(content).replace("\n", " "), 80))

        # Fallback: render text from the embedded message tool_result block.
        # This catches future schema variants without losing the payload.
        for block in msg_blocks:
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            body = block.get("content")
            if isinstance(body, str) and body.strip():
                return StreamEvent("📎", _trunc(body.replace("\n", " "), 80))

        return None

    def _render_content_blocks(self, event: dict) -> StreamEvent | None:
        content = event.get("content", [])
        if not content and "message" in event:
            msg = event["message"]
            content = msg.get("content", []) if isinstance(msg, dict) else []
        if isinstance(content, str):
            return StreamEvent("💬", _trunc(content, _MAX_TEXT_DISPLAY)) if content.strip() else None
        if not isinstance(content, list):
            return None

        parts: list[StreamEvent] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                text = block.get("text", "").strip()
                if text:
                    parts.append(StreamEvent("💬", _trunc(text, _MAX_TEXT_DISPLAY)))
            elif btype == "thinking":
                text = (block.get("thinking", "") or block.get("text", "")).strip()
                if text:
                    parts.append(StreamEvent("💭", _trunc(text, _MAX_TEXT_DISPLAY)))
            elif btype == "tool_use":
                name = block.get("name", "?")
                args_str = self._compact_args(block.get("input", {}))
                # Claude Code emits two built-in meta-tools alongside
                # real domain tools. Render them with distinct icons so
                # the live UI does not conflate Claude's internal
                # scaffolding (skill loading, tool discovery) with
                # MCP/file/shell calls.
                if name == "Skill":
                    icon = "🪄"
                elif name == "ToolSearch":
                    icon = "🔎"
                else:
                    icon = "🔧"
                parts.append(StreamEvent(icon, f"{name}({args_str})"))
            elif btype == "functionCall":
                name = block.get("name", "?")
                args_str = self._compact_args(block.get("args", {}))
                parts.append(StreamEvent("🔧", f"{name}({args_str})"))

        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return StreamEvent(parts[0].icon, " | ".join(str(p) for p in parts))

    def _render_gemini_message(self, event: dict) -> StreamEvent | None:
        content = event.get("content")
        if isinstance(content, str):
            return StreamEvent("💬", _trunc(content, _MAX_TEXT_DISPLAY)) if content.strip() else None
        if isinstance(content, list):
            parts: list[StreamEvent] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    text = block.get("text", "").strip()
                    if text:
                        parts.append(StreamEvent("💬", _trunc(text, _MAX_TEXT_DISPLAY)))
                elif btype == "functionCall":
                    name = block.get("name", "?")
                    args_str = self._compact_args(block.get("args", {}))
                    parts.append(StreamEvent("🔧", f"{name}({args_str})"))
            if not parts:
                return None
            if len(parts) == 1:
                return parts[0]
            return StreamEvent(parts[0].icon, " | ".join(str(p) for p in parts))
        return None

    def _render_result(self, event: dict) -> StreamEvent:
        # Result-event values (cost, duration, fixed "ERROR" label) are
        # numeric or framework-owned strings, never agent-controlled. The
        # ``[green]...[/green]`` wrapper is framework-emitted styling, so
        # mark the summary as trusted markup; render sites must not
        # double-escape it.
        parts: list[str] = []
        cost = event.get("total_cost_usd") or event.get("cost_usd")
        if cost is not None:
            parts.append(f"[green]${cost:.4f}[/green]")
        duration = event.get("duration_ms")
        if duration is not None:
            parts.append(f"{duration / 1000:.1f}s")
        stats = event.get("stats") or {}
        if isinstance(stats, dict) and stats.get("duration_ms"):
            parts.append(f"{stats['duration_ms'] / 1000:.1f}s")
        is_error = event.get("is_error") or event.get("status") == "error"
        if is_error:
            parts.append("ERROR")
            return StreamEvent("❌", "result: " + ", ".join(parts), summary_is_markup=True)
        if parts:
            return StreamEvent("✅", "result: " + ", ".join(parts), summary_is_markup=True)
        return StreamEvent("✅", "done")

    def _compact_args(self, args: dict | str) -> str:
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                return _trunc(args, _MAX_ARG_DISPLAY)
        if not isinstance(args, dict):
            return str(args)[:_MAX_ARG_DISPLAY]
        pairs = []
        for k, v in args.items():
            vs = json.dumps(v) if not isinstance(v, str) else v
            pairs.append(f"{k}={_trunc_value(vs, _MAX_VALUE_DISPLAY)}")
        return _trunc(", ".join(pairs), _MAX_ARG_DISPLAY)


def _trunc_value(s: str, max_len: int) -> str:
    """Truncate a value, preserving the tail for file paths."""
    s = s.replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    if "/" in s:
        parts = s.rstrip("/").split("/")
        result = "/".join(parts)
        while len(result) > max_len - 1 and len(parts) > 1:
            parts.pop(0)
            result = "/".join(parts)
        return "…/" + result if result != s else s[:max_len] + "…"
    return s[:max_len] + "…"


def _trunc(s: str, max_len: int) -> str:
    s = s.replace("\n", " ").strip()
    return s[:max_len] + "…" if len(s) > max_len else s


# ── Watch command ──


def _find_latest_run(outcomes_root: Path) -> Path | None:
    """Find the most recently created run directory under outcomes/."""
    try:
        if not outcomes_root.is_dir():
            return None
        runs = sorted(outcomes_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return None
    for r in runs:
        if r.is_dir() and not r.name.startswith("."):
            return r
    return None


def _discover_stream_files(run_dir: Path) -> list[Path]:
    """Find all stream NDJSON files under a run directory."""
    try:
        return sorted(run_dir.rglob("turn_*_stream.ndjson"))
    except OSError:
        return []


class _FileWatcher:
    """Track read position for a single stream file."""

    __slots__ = ("path", "offset", "label")

    def __init__(self, path: Path, run_dir: Path):
        self.path = path
        self.offset = 0
        relative = path.relative_to(run_dir)
        parts = relative.parts
        self.label = "/".join(parts[:-1]) if len(parts) > 1 else str(relative)

    def read_new_lines(self) -> list[str]:
        try:
            with open(self.path) as f:
                f.seek(self.offset)
                data = f.read()
                self.offset = f.tell()
        except OSError:
            return []
        if not data:
            return []
        return data.splitlines()


_DEFAULT_IDLE_TIMEOUT = 3.0


def _resolve_agent_from_run(run_dir: Path) -> type | None:
    """Try to resolve the agent class from run_meta.json in a run directory."""
    try:
        meta_path = run_dir / "run_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            agent_name = meta.get("agent")
            if agent_name:
                from belt.agent.registry import get_agent_class

                return get_agent_class(agent_name)
    except Exception:
        pass
    return None


def watch_command(
    run_dir: Path | None = None,
    *,
    poll_interval: float = 0.2,
    scenario_filter: str | None = None,
    follow: bool = False,
    console: Console | None = None,
) -> int:
    """Main watch loop - tail stream files and render events.

    By default, exits after idle_timeout seconds of no new data (run complete).
    With ``follow=True``, polls indefinitely until Ctrl-C (useful when watching
    a run in progress from a second terminal).
    """
    console = console or Console(stderr=True)

    if run_dir is None:
        run_dir = _find_latest_run(OUTCOMES_ROOT)
    if run_dir is None or not run_dir.is_dir():
        console.print("[red]No run directory found. Start a run first.[/red]")
        return 1

    agent_cls = _resolve_agent_from_run(run_dir)
    parser = StreamParser(agent_cls=agent_cls)

    console.print(f"\n[bold]Watching:[/bold] {run_dir}")
    if follow:
        console.print("[dim]Following (Ctrl-C to stop)…[/dim]\n")
    else:
        console.print("")

    watchers: dict[Path, _FileWatcher] = {}
    seen_paths: set[Path] = set()
    last_event_time: float | None = None
    any_events = False
    started_at = time.monotonic()

    try:
        while True:
            new_files = _discover_stream_files(run_dir)
            for path in new_files:
                if path not in seen_paths:
                    seen_paths.add(path)
                    watcher = _FileWatcher(path, run_dir)
                    if scenario_filter and scenario_filter not in watcher.label:
                        continue
                    watchers[path] = watcher

            had_output = False
            for watcher in watchers.values():
                lines = watcher.read_new_lines()
                for line in lines:
                    try:
                        event = parser.parse_line(line)
                    except Exception:
                        continue
                    if event is not None:
                        _print_event(console, watcher.label, event)
                        had_output = True

            if had_output:
                any_events = True
                last_event_time = time.monotonic()
                time.sleep(0.05)
            else:
                if not follow:
                    if any_events and last_event_time is not None:
                        idle = time.monotonic() - last_event_time
                        if idle >= _DEFAULT_IDLE_TIMEOUT:
                            console.print("\n[dim]Run complete.[/dim]")
                            return 0
                    elif time.monotonic() - started_at >= _DEFAULT_IDLE_TIMEOUT:
                        console.print("\n[dim]No live stream data in this run.[/dim]")
                        return 0
                time.sleep(poll_interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Watch stopped.[/dim]")
        return 0


def _print_event(console: Console, label: str, event: StreamEvent) -> None:
    # ``event.summary`` is plain text by default and must be ``rich_safe``-escaped
    # because it may include attacker-controlled content from agent stdout (tool
    # args, reply text). When the framework itself built the summary with trusted
    # Rich markup (``event.summary_is_markup=True``), pass it through unchanged so
    # ``Text.from_markup`` parses the styling. Escaping a trusted markup string
    # turns ``[green]`` into a literal ``\\[green]`` substring at render time.
    from belt._safe import rich_safe

    payload = event.summary if event.summary_is_markup else rich_safe(event.summary)
    text = Text()
    text.append(f"  {label} ", style="cyan dim")
    text.append_text(Text.from_markup(f"{event.icon} {payload}"))
    console.print(text)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``belt watch``."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="belt watch",
        description="Watch live agent output during evaluation runs",
    )
    # Positional first; flags alphabetised by long flag name. Enforced by
    # ``tests/test_cli_order.py``.
    ap.add_argument("run_dir", nargs="?", help="Run directory to watch (default: latest)")
    ap.add_argument("--follow", "-f", action="store_true", help="Keep watching until Ctrl-C (default: exit when idle)")
    ap.add_argument("--poll", type=float, default=0.2, help="Poll interval in seconds (default: 0.2)")
    ap.add_argument("--scenario", help="Filter to scenarios matching this substring")
    args = ap.parse_args(argv)

    run_path = Path(args.run_dir) if args.run_dir else None
    return watch_command(run_path, poll_interval=args.poll, scenario_filter=args.scenario, follow=args.follow)
