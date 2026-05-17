# (c) JFrog Ltd. (2026)

"""Rich-based progress display for evaluation runs."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console, ConsoleRenderable
from rich.console import Group as RichGroup
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from belt._ui import eprint

if TYPE_CHECKING:
    from belt.entities import ScenarioScore
    from belt.runner.entities import ScenarioResult
    from belt.scenario import GroupConfig, Scenario


def phase_header(console: Console, name: str, *, plain: bool = False) -> None:
    """Print a phase section header: ── Name ──────────────────"""
    if plain:
        eprint(f"\n── {name} ──")
    else:
        console.print()
        console.rule(f"[bold]{name}[/bold]", style="dim")


class SpeedColumn(ProgressColumn):
    """Displays avg wall time per completed item (elapsed / completed)."""

    def render(self, task: Task) -> Text:
        if not task.completed or not task.elapsed:
            return Text("-", style="dim")
        secs_per = task.elapsed / task.completed
        if secs_per >= 10:
            return Text(f"{secs_per:.0f}s/sc", style="cyan")
        return Text(f"{secs_per:.1f}s/sc", style="cyan")


class RunnerProgress:
    """Thread-safe progress display for the scenario runner."""

    def __init__(self, console: Console | None = None, plain: bool = False):
        self.console = console or Console(stderr=True)
        self._plain = plain
        self._progress: Progress | None = None
        self._group_tasks: dict[str, int] = {}
        self._start_time: float = 0.0
        self._completed: int = 0
        self._total: int = 0
        self._total_agent_cost: float = 0.0
        self._lock = threading.Lock()

    def header(
        self,
        *,
        total_scenarios: int,
        total_groups: int,
        total_turns: int = 0,
        agents: list[str] | None = None,
        tags: str = "",
        run_label: str = "",
        dry_run: bool = False,
        scorer_descriptions: list[str] | None = None,
        workspace: str = "",
    ) -> None:
        agent_names = agents or []
        label = "agents" if len(agent_names) != 1 else "agent"
        agent_str = ", ".join(agent_names) or "-"
        turns_part = f" · [bold]{total_turns}[/bold] turns" if total_turns else ""
        mode_part = " · [dim]dry-run[/dim]" if dry_run else ""
        lines = [
            f"[bold]{total_scenarios}[/bold] scenarios · [bold]{total_groups}[/bold] groups{turns_part}{mode_part}",
            f"{label}: [cyan]{agent_str}[/cyan] · tags: {tags}",
        ]
        if workspace:
            lines.append(f"workspace: [cyan]{workspace}[/cyan]")
        lines.append(f"run: {run_label}")
        if scorer_descriptions:
            lines.append("")
            lines.append("[bold]Scorers:[/bold]")
            for desc in scorer_descriptions:
                lines.append(f"  [green]✓[/green] {desc}")
        self.console.print()
        self.console.print(Panel("\n".join(lines), title="[bold]Evaluation[/bold]", border_style="blue"))

    def dry_run_table(
        self,
        matched_groups: list[tuple[Path, GroupConfig, list[Scenario]]],
        scenarios_dir: Path,
    ) -> None:
        table = Table(show_header=True, show_lines=False, padding=(0, 1))
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Group", style="cyan")
        table.add_column("Scenario")
        table.add_column("Turns", justify="right", style="dim")
        table.add_column("Tags", style="dim")

        n = 0
        for group_dir, group_config, scenarios in matched_groups:
            relative = str(group_dir.resolve().relative_to(scenarios_dir))
            for si, s in enumerate(scenarios):
                n += 1
                tags = sorted(set(s.tags) | set(group_config.default_tags))
                table.add_row(
                    str(n),
                    relative if si == 0 else "",
                    s.name,
                    str(len(s.turns)),
                    ", ".join(tags),
                )
        self.console.print(table)

    def group_setup_table(
        self,
        summaries: dict[str, str],
        failed: set[str],
    ) -> None:
        """Show group setup results."""
        table = Table(show_header=True, show_lines=False, padding=(0, 1))
        table.add_column("Group", style="cyan")
        table.add_column("Setup", style="dim")
        table.add_column("Status")
        for name in sorted(set(summaries.keys()) | failed):
            if name in failed:
                table.add_row(name, "-", "[red]✗ failed[/red]")
            else:
                table.add_row(name, summaries[name], "[green]✓[/green]")
        self.console.print(table)

    def start(
        self,
        matched_groups: list[tuple[Path, GroupConfig, list[Scenario]]],
        scenarios_dir: Path,
        workers: int,
        scenario_delay: float = 0,
        max_retries: int = 0,
    ) -> None:
        """Start live progress bars - one per group."""
        phase_header(self.console, "Run", plain=self._plain)

        extras = ""
        if scenario_delay > 0:
            extras += f" · delay: [cyan]{scenario_delay:.1f}s[/cyan]"
        if max_retries > 0:
            extras += f" · retries: [cyan]{max_retries}[/cyan]"
        if workers > 1 or extras:
            self.console.print(f"Workers: [bold]{workers}[/bold]{extras}\n")

        self._start_time = time.monotonic()
        self._total = sum(len(scenarios) for _, _, scenarios in matched_groups)

        if self._plain:
            return

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("scenarios"),
            SpeedColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        for group_dir, _, scenarios in matched_groups:
            name = str(group_dir.resolve().relative_to(scenarios_dir))
            self._group_tasks[name] = self._progress.add_task(name, total=len(scenarios))
        self._progress.start()

    def scenario_done(self, group_name: str, scenario_name: str = "", agent_cost_usd: float | None = None) -> None:
        """Advance progress by one scenario."""
        if agent_cost_usd is not None:
            with self._lock:
                self._total_agent_cost += agent_cost_usd
        if self._plain:
            with self._lock:
                self._completed += 1
                elapsed = time.monotonic() - self._start_time
                label = f"{group_name}/{scenario_name}" if scenario_name else group_name
                cost_part = f" ${agent_cost_usd:.4f}" if agent_cost_usd is not None else ""
                eprint(f"  [{self._completed}/{self._total}] {label} ({elapsed:.0f}s{cost_part})")
            return
        if self._progress and group_name in self._group_tasks:
            self._progress.advance(self._group_tasks[group_name])

    def stop(self) -> None:
        if self._progress:
            self._progress.stop()
            self._progress = None

    def summary(self, all_results: list[ScenarioResult]) -> None:
        # "group setup failed" is already surfaced by the per-group ✗
        # banner during setup AND by the structured ``setup_errors``
        # block in ``belt view`` / ``results.json``. Repeating it once
        # per scenario in the run-summary footer made the same fact
        # arrive three times in a row, so we partition it out and only
        # echo the *unique* error message under a single group header
        # (the count goes into the headline; the message goes once).
        group_setup_failed = [r for r in all_results if r.error == "group setup failed"]
        errors = [r for r in all_results if r.error and r.error != "group setup failed"]
        agent_errored = [r for r in all_results if r.agent_errors]
        total = len(all_results)
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        mins, secs = divmod(elapsed, 60)
        time_str = f"{int(mins)}m{secs:.0f}s" if mins else f"{secs:.1f}s"

        cost_parts = [r.agent_cost_usd for r in all_results if r.agent_cost_usd is not None]
        cost_str = f" · [green]agent ${sum(cost_parts):.4f}[/green]" if cost_parts else ""

        self.console.print()
        if errors or group_setup_failed:
            # Infrastructure failure (subprocess crash, harness exception).
            # Listed first because it usually subsumes any agent error and
            # the harness operator can act on it directly.
            total_err = len(errors) + len(group_setup_failed)
            self.console.print(
                f"[red bold]❌ Run: {total} scenarios, {total_err} error(s)[/red bold]{cost_str} · {time_str}"
            )
            for r in errors:
                self.console.print(f"  [red]•[/red] {r.scenario_name}: {r.error}")
            if group_setup_failed:
                # Group-level setup failures are not per-scenario problems;
                # the per-group ✗ banner above already named the root cause.
                # Just count the casualties here.
                by_group: dict[str, int] = {}
                for r in group_setup_failed:
                    by_group[r.group_path] = by_group.get(r.group_path, 0) + 1
                for group, n in sorted(by_group.items()):
                    self.console.print(
                        f"  [red]•[/red] {group}: {n} scenario(s) skipped (group setup failed, see banner above)"
                    )
        elif agent_errored:
            # Subprocesses exited cleanly but the agent itself reported an
            # error in one or more turns (auth, refused, rate-limit). This
            # is the case the framework historically hid behind the
            # misleading ``0 errors`` headline.
            agent_error_types = sorted({et for r in agent_errored for et in r.agent_errors})
            type_summary = ", ".join(agent_error_types)
            self.console.print(
                f"[red bold]❌ Run: {total} scenarios, "
                f"{len(agent_errored)} agent error(s)[/red bold] "
                f"[dim]({type_summary})[/dim]{cost_str} · {time_str}"
            )
            for r in agent_errored:
                first_type = r.agent_errors[0]
                self.console.print(f"  [red]•[/red] {r.scenario_name}: agent error ({first_type})")
        else:
            self.console.print(f"[green bold]✅ Run: {total} scenarios, 0 errors[/green bold]{cost_str} · {time_str}")


_DEFAULT_LIVE_LINES = 30


class LiveProgress(RunnerProgress):
    """Integrated TUI: progress bars + live agent stream in one terminal.

    Uses Rich ``Live`` with a ``Group`` of two renderables:
    - Top: per-group progress table (rebuilt on each update)
    - Bottom: scrolling panel of recent stream events

    A background thread polls ``turn_*_stream.ndjson`` files and feeds
    rendered events into the bottom panel via ``StreamParser``.
    """

    def __init__(self, run_dir: Path, console: Console | None = None, max_lines: int = _DEFAULT_LIVE_LINES):
        super().__init__(console=console, plain=False)
        self._run_dir = run_dir
        self._max_lines = max_lines
        self._live: Live | None = None
        self._scenario_streams: dict[str, list[str]] = {}
        self._scenario_pin_indices: dict[str, set[int]] = {}
        self._scenario_order: list[str] = []
        self._poller_thread: threading.Thread | None = None
        self._poller_stop = threading.Event()
        self._group_completed: dict[str, int] = {}
        self._group_totals: dict[str, int] = {}
        self._scenarios_dir: Path | None = None
        self._current_scenario: str = ""

    def start(
        self,
        matched_groups: list[tuple[Path, GroupConfig, list[Scenario]]],
        scenarios_dir: Path,
        workers: int,
        scenario_delay: float = 0,
        max_retries: int = 0,
    ) -> None:
        extras = ""
        if scenario_delay > 0:
            extras += f" · delay: {scenario_delay:.1f}s"
        if max_retries > 0:
            extras += f" · retries: {max_retries}"
        if workers > 1 or extras:
            self.console.print(f"\nWorkers: [bold]{workers}[/bold]{extras}\n")
        else:
            self.console.print()

        self._start_time = time.monotonic()
        self._total = sum(len(scenarios) for _, _, scenarios in matched_groups)
        self._scenarios_dir = scenarios_dir

        agents = {gc.agent for _, gc, _ in matched_groups}
        self._agent_name: str | None = agents.pop() if len(agents) == 1 else None
        # Per-group agent map so multi-agent runs still get agent-specific
        # stream parsing (cursor's tool_call shape, gemini's functionCall
        # blocks, etc). Without this, runs that mix agents fall back to the
        # generic parser and surface tool calls as ``?()``.
        self._agent_by_group: dict[str, str] = {}

        for group_dir, gc, scenarios in matched_groups:
            name = str(group_dir.resolve().relative_to(scenarios_dir))
            self._group_totals[name] = len(scenarios)
            self._group_completed[name] = 0
            self._agent_by_group[name] = gc.agent

        self._live = Live(self._build_display(), console=self.console, refresh_per_second=4)
        self._live.start()

        self._poller_stop.clear()
        self._poller_thread = threading.Thread(target=self._poll_streams, daemon=True)
        self._poller_thread.start()

    def scenario_done(self, group_name: str, scenario_name: str = "", agent_cost_usd: float | None = None) -> None:
        with self._lock:
            self._completed += 1
            if agent_cost_usd is not None:
                self._total_agent_cost += agent_cost_usd
            if group_name in self._group_completed:
                self._group_completed[group_name] += 1
            self._current_scenario = f"{group_name}/{scenario_name}" if scenario_name else group_name
        self._refresh()

    def stop(self) -> None:
        self._poller_stop.set()
        if self._poller_thread is not None:
            self._poller_thread.join(timeout=2)
            self._poller_thread = None
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _refresh(self) -> None:
        live = self._live
        if live is not None:
            try:
                live.update(self._build_display())
            except Exception:
                pass

    def _build_display(self) -> ConsoleRenderable:
        progress_table = self._build_progress_table()
        stream_panel = self._build_stream_panel()
        return RichGroup(progress_table, stream_panel)

    def _build_progress_table(self) -> Table:
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        mins, secs = divmod(elapsed, 60)
        time_str = f"{int(mins)}m{secs:.0f}s" if mins else f"{secs:.0f}s"
        speed = ""
        if self._completed > 0 and elapsed > 0:
            secs_per = elapsed / self._completed
            speed = f" · {secs_per:.1f}s/sc" if secs_per < 10 else f" · {secs_per:.0f}s/sc"

        table = Table(show_header=False, show_edge=False, padding=(0, 1), expand=True)
        table.add_column("group", style="cyan", ratio=2)
        table.add_column("bar", ratio=4)
        table.add_column("count", justify="right", style="dim")
        table.add_column("time", justify="right", style="dim")

        for name in self._group_totals:
            done = self._group_completed.get(name, 0)
            total = self._group_totals[name]
            frac = done / total if total else 0
            bar_width = 20
            filled = int(frac * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            bar_style = "green" if done == total else "blue"
            table.add_row(name, f"[{bar_style}]{bar}[/{bar_style}]", f"{done}/{total}", "")

        cost = f" · [green]agent ${self._total_agent_cost:.4f}[/green]" if self._total_agent_cost > 0 else ""
        header = f"  [bold]{self._completed}/{self._total}[/bold] scenarios{cost} · {time_str}{speed}"
        return Panel(
            (
                RichGroup(Text.from_markup(header), table)
                if len(self._group_totals) > 1
                else Text.from_markup(header + f"  {self._build_single_bar()}")
            ),
            border_style="blue",
            padding=(0, 1),
        )

    def _build_single_bar(self) -> str:
        if not self._group_totals:
            return ""
        name = next(iter(self._group_totals))
        done = self._group_completed.get(name, 0)
        total = self._group_totals[name]
        frac = done / total if total else 0
        bar_width = 20
        filled = int(frac * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        style = "green" if done == total else "blue"
        return f"[{style}]{bar}[/{style}]"

    def _build_stream_panel(self) -> Panel:
        inner_width = max(40, self.console.width - 4)
        with self._lock:
            if not self._scenario_order:
                content: ConsoleRenderable = Text("Waiting for agent output…", style="dim")
            else:
                text_lines: list[Text] = []
                max_per = max(3, self._max_lines // max(1, len(self._scenario_order)))
                for label in self._scenario_order:
                    events = self._scenario_streams.get(label, [])
                    pin_idx = self._scenario_pin_indices.get(label, set())
                    prefix_len = 4 + len(label) + 1
                    pad = "─" * max(1, inner_width - prefix_len)
                    text_lines.append(
                        self._render_line_safe(f"[cyan]───[/cyan] [bold]{label}[/bold] [cyan]{pad}[/cyan]")
                    )
                    pins = [events[i] for i in sorted(pin_idx) if i < len(events)]
                    for line in self._clamp(pins, inner_width):
                        text_lines.append(self._render_line_safe(line))
                    non_pin = [e for i, e in enumerate(events) if i not in pin_idx]
                    tail = non_pin[-max_per:] if len(non_pin) > max_per else non_pin
                    if len(non_pin) > max_per:
                        text_lines.append(self._render_line_safe(f"  [dim]… {len(non_pin) - max_per} earlier[/dim]"))
                    for line in self._clamp(tail, inner_width):
                        text_lines.append(self._render_line_safe(line))
                content = Text("\n").join(text_lines)
        return Panel(content, title="[bold]Live Output[/bold]", border_style="dim", padding=(0, 1))

    @staticmethod
    def _render_line_safe(line: str) -> Text:
        """Render a single panel line, falling back to literal text on failure.

        Rendering each line independently isolates malformed markup: a single
        broken tag (e.g. unbalanced ``[bold]...`` from a future bug or from
        truncation cutting mid-tag) only affects that one line, instead of
        cascading into the whole panel rendering as plain text.
        """
        try:
            return Text.from_markup(line)
        except Exception:
            return Text(line)

    @staticmethod
    def _clamp(lines: list[str], width: int) -> list[str]:
        """Truncate lines to fit within panel width, accounting for emoji double-width."""
        clamped = []
        for line in lines:
            if len(line) > width:
                clamped.append(line[: width - 1] + "…")
            else:
                clamped.append(line)
        return clamped

    def _poll_streams(self) -> None:
        from belt.agent.registry import get_agent_class
        from belt.commands.watch import StreamParser, _discover_stream_files, _FileWatcher

        # Cache one parser per resolved agent class so we don't re-import on
        # every line. ``None`` key holds the generic (no-agent) parser used
        # when the stream file's group can't be matched to a known agent.
        parsers: dict[str | None, StreamParser] = {}

        def _parser_for(agent_name: str | None) -> StreamParser:
            if agent_name in parsers:
                return parsers[agent_name]
            agent_cls = None
            if agent_name:
                try:
                    agent_cls = get_agent_class(agent_name)
                except Exception:
                    agent_cls = None
            parsers[agent_name] = StreamParser(agent_cls=agent_cls)
            return parsers[agent_name]

        def _agent_for_path(path: Path) -> str | None:
            # Stream file lives at
            # ``<run_dir>/<group_relpath>/<scenario>/turn_N_stream.ndjson``.
            # Walk parents up to the run dir to find a matching group key in
            # ``self._agent_by_group``; supports nested group paths like
            # ``scenarios/cursor`` as well as flat ones.
            try:
                rel = path.relative_to(self._run_dir).parent.parent
            except ValueError:
                return self._agent_name
            current = rel
            while True:
                key = str(current)
                if key in self._agent_by_group:
                    return self._agent_by_group[key]
                if current.parent == current or str(current) in (".", ""):
                    break
                current = current.parent
            return self._agent_name

        watchers: dict[Path, _FileWatcher] = {}
        seen_paths: set[Path] = set()
        path_parsers: dict[Path, StreamParser] = {}

        while not self._poller_stop.is_set():
            new_files = _discover_stream_files(self._run_dir)
            for path in new_files:
                if path not in seen_paths:
                    seen_paths.add(path)
                    watchers[path] = _FileWatcher(path, self._run_dir)
                    path_parsers[path] = _parser_for(_agent_for_path(path))

            had_output = False
            for path, watcher in watchers.items():
                parser = path_parsers[path]
                for line in watcher.read_new_lines():
                    try:
                        event = parser.parse_line(line)
                    except Exception:
                        continue
                    if event is not None:
                        # ``event.summary`` is plain text by default and must be
                        # ``rich_safe``-escaped because agent stdout can smuggle
                        # markup like ``[red]injected[/red]`` through tool args
                        # or reply text. When the framework itself built the
                        # summary with trusted Rich markup (e.g. the cost-styling
                        # in ``_render_result``), pass it through unchanged -
                        # escaping it turns ``[green]`` into a literal
                        # ``\\[green]`` at render time. See ``StreamEvent``.
                        from belt._safe import rich_safe

                        payload = event.summary if event.summary_is_markup else rich_safe(event.summary)
                        formatted = f"  {event.icon} {payload}"
                        with self._lock:
                            if watcher.label not in self._scenario_streams:
                                self._scenario_streams[watcher.label] = []
                                self._scenario_order.append(watcher.label)
                            idx = len(self._scenario_streams[watcher.label])
                            self._scenario_streams[watcher.label].append(formatted)
                            if event.icon == "👤":
                                self._scenario_pin_indices.setdefault(watcher.label, set()).add(idx)
                        had_output = True

            if had_output:
                self._refresh()
            self._poller_stop.wait(0.15)


_DEFAULT_SCORER_LINES = 20


class ScorerProgress:
    """Thread-safe progress display for the scorer CLI."""

    def __init__(
        self,
        console: Console | None = None,
        plain: bool = False,
        live: bool = False,
        max_lines: int = _DEFAULT_SCORER_LINES,
    ):
        self.console = console or Console(stderr=True)
        self._plain = plain
        self._live_mode = live
        self._max_lines = max_lines
        self._progress: Progress | None = None
        self._live: Live | None = None
        self._task_id: int | None = None
        self._start_time: float = 0.0
        self._completed: int = 0
        self._total: int = 0
        self._last_label: str = ""
        self._lock = threading.Lock()
        self._scenario_events: dict[str, list[str]] = {}
        self._scenario_order: list[str] = []

    def start(self, total: int, mode_str: str, workers: int, max_retries: int = 0) -> None:
        phase_header(self.console, "Score", plain=self._plain)

        self._mode_str = mode_str
        self._start_time = time.monotonic()
        self._total = total
        self._has_llm = "llm" in mode_str

        if self._plain:
            return

        if self._live_mode:
            self._live = Live(self._build_panel(), console=self.console, refresh_per_second=4)
            self._live.start()
            return

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            SpeedColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        self._task_id = self._progress.add_task("Scoring", total=total)
        self._progress.start()

    def scored(self, relative: str, passed: bool) -> None:
        icon = "✅" if passed else "❌"
        if self._plain:
            with self._lock:
                self._completed += 1
                eprint(f"  [{self._completed}/{self._total}] {icon} {relative}")
            return
        if self._live_mode:
            with self._lock:
                self._completed += 1
                self._last_label = f"{icon} {relative}"
            self._refresh_live()
            return
        if self._progress and self._task_id is not None:
            self._progress.advance(self._task_id)
            self._progress.update(self._task_id, description=f"{icon} {relative}")

    def add_event(self, scenario: str, formatted_line: str) -> None:
        """Append a pre-formatted event line for a scenario (thread-safe)."""
        with self._lock:
            if scenario not in self._scenario_events:
                self._scenario_events[scenario] = []
                self._scenario_order.append(scenario)
            self._scenario_events[scenario].append(formatted_line)
        self._refresh_live()

    def _refresh_live(self) -> None:
        if self._live is not None:
            try:
                self._live.update(self._build_panel())
            except Exception:
                pass

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
        if self._progress:
            self._progress.stop()
            self._progress = None

    def _build_panel(self) -> ConsoleRenderable:
        with self._lock:
            done = self._completed
            total = self._total
            label = self._last_label
            has_events = bool(self._scenario_order)

        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        mins, secs = divmod(elapsed, 60)
        time_str = f"{int(mins)}m{secs:.0f}s" if mins else f"{secs:.0f}s"
        speed = ""
        if done > 0 and elapsed > 0:
            secs_per = elapsed / done
            speed = f" · {secs_per:.1f}s/sc" if secs_per < 10 else f" · {secs_per:.0f}s/sc"

        frac = done / total if total else 0
        bar_width = 30
        filled = int(frac * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        bar_style = "green" if done == total else "blue"

        header = f"  [bold]{done}/{total}[/bold] scenarios · {time_str}{speed}  [{bar_style}]{bar}[/{bar_style}]"
        header_lines = [header]
        if label:
            header_lines.append(f"  {label}")
        else:
            header_lines.append("  [dim]Scoring…[/dim]")

        try:
            progress_text = Text.from_markup("\n".join(header_lines))
        except Exception:
            progress_text = Text("\n".join(header_lines))

        progress_panel = Panel(progress_text, border_style="blue", padding=(0, 1))

        if not has_events:
            return progress_panel

        stream_panel = self._build_stream_panel()
        return RichGroup(progress_panel, stream_panel)

    def _build_stream_panel(self) -> Panel:
        inner_width = max(40, self.console.width - 4)
        with self._lock:
            if not self._scenario_order:
                content = Text("Waiting for scorer output…", style="dim")
            else:
                parts: list[str] = []
                max_per = max(3, self._max_lines // max(1, len(self._scenario_order)))
                for label in self._scenario_order:
                    events = self._scenario_events.get(label, [])
                    prefix_len = 4 + len(label) + 1
                    pad = "─" * max(1, inner_width - prefix_len)
                    parts.append(f"[cyan]───[/cyan] [bold]{label}[/bold] [cyan]{pad}[/cyan]")
                    tail = events[-max_per:] if len(events) > max_per else events
                    if len(events) > max_per:
                        parts.append(f"  [dim]… {len(events) - max_per} earlier[/dim]")
                    for line in tail:
                        if len(line) > inner_width:
                            parts.append(line[: inner_width - 1] + "…")
                        else:
                            parts.append(line)
                try:
                    content = Text.from_markup("\n".join(parts))
                except Exception:
                    content = Text("\n".join(parts))
        return Panel(content, title="[bold]Scorer Output[/bold]", border_style="dim", padding=(0, 1))

    def summary(
        self,
        results: list[tuple[Path, ScenarioScore]],
        *,
        cache_hits: int = 0,
        cache_misses: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        judge_cost_usd: float | None = None,
        chained: bool = False,
    ) -> None:
        # ``chained`` is set by ``belt eval`` when the aggregator is about
        # to print its own canonical "X/N checks (Y%) - Agent: $a - Judge:
        # $j - Total: $t" footer (see aggregator/render_terminal.py). In
        # that case suppress the pass-count headline and the judge-cost
        # subpart so the screen has a single scoreboard. Cache + token
        # stats are scorer-internal and not surfaced by the aggregator,
        # so they remain useful and stay unconditional.
        passed = sum(1 for _, s in results if s.overall_pass)
        failed = len(results) - passed
        total = len(results)
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        mins, secs = divmod(elapsed, 60)
        time_str = f"{int(mins)}m{secs:.0f}s" if mins else f"{secs:.1f}s"
        mode = getattr(self, "_mode_str", "")

        if not chained:
            self.console.print()
            pass_str = f"{passed}/{total} passed" if failed else "all passed"
            if failed:
                self.console.print(f"[red bold]❌ Score: {pass_str}[/red bold] · {mode} · {time_str}")
            else:
                self.console.print(f"[green bold]✅ Score: {pass_str}[/green bold] · {mode} · {time_str}")

        sub_parts: list[str] = []
        if judge_cost_usd is not None and not chained:
            sub_parts.append(f"[green]${judge_cost_usd:.4f}[/green] judge cost")
        cache_total = cache_hits + cache_misses
        if cache_total > 0:
            sub_parts.append(f"cache: {cache_hits}/{cache_total} hit")
        total_tokens = prompt_tokens + completion_tokens
        if total_tokens > 0:
            sub_parts.append(f"{total_tokens:,} tokens")
        if sub_parts:
            # Pad with a leading blank when the headline was suppressed so
            # the cache/token subline is not glued to the runner footer
            # above it.
            if chained:
                self.console.print()
            self.console.print(f"   [dim]{' · '.join(sub_parts)}[/dim]")
