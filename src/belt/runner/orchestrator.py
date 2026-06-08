# (c) JFrog Ltd. (2026)

"""Agent-agent-agnostic scenario orchestrator.

Executes scenario turns through a BaseAgentAdapter, writing per-turn artifacts
(CLI output, thread state, normalized TurnOutput) to the outcome directory.

Post-turn workspace capture: when a Turn has state_expect, the orchestrator
captures referenced files and optionally git diff into TurnOutput fields so
downstream scorers can verify filesystem side-effects without workspace access.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import traceback
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from loguru import logger

from belt import envvars
from belt._git import run_git
from belt._io import write_json
from belt._redact import safe_agent_args
from belt._sanitize import strip_ansi
from belt.agent.error_types import UNKNOWN
from belt.constants import (
    RUNTIME_INFO_FILE,
    SCHEMA_VERSION,
    TURN_CLI_TEMPLATE,
    TURN_MESSAGE_MAX_CHARS,
    TURN_OUTPUT_TEMPLATE,
    TURN_STATE_TEMPLATE,
    TURN_STREAM_TEMPLATE,
)
from belt.entities import TurnOutput, VerifyResult
from belt.errors import ScenarioError
from belt.parser.ndjson import bounded_json_loads
from belt.runner.entities import AgentConfig, ScenarioResult
from belt.runner.process.spawner import LocalSpawner, SandboxedSpawner, SubprocessRunner
from belt.runner.sandbox import BaseSandboxProvider, SandboxHandle, get_sandbox_provider
from belt.runner.sandbox.base import SandboxContext
from belt.scenario import GroupConfig, SandboxProfile, Scenario, StateExpectation, VerifySpec

if TYPE_CHECKING:
    from belt.agent.base import BaseAgentAdapter
    from belt.runner.workspace import WorkspaceManager

_MAX_FILE_CAPTURE_BYTES = 10_240
_MAX_FILES_PER_TURN = 500

# Per-turn NDJSON-stream artifact cap. A malicious agent that emits gigabytes
# of NDJSON would otherwise fill the disk shared with other CI jobs.
# Override with ``envvars.TURN_NDJSON_MAX_BYTES``.
_DEFAULT_TURN_NDJSON_MAX_BYTES = 50 * 1024 * 1024  # 50 MiB


def _turn_stream_cap() -> int:
    raw = os.environ.get(envvars.TURN_NDJSON_MAX_BYTES)
    if not raw:
        return _DEFAULT_TURN_NDJSON_MAX_BYTES
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer {}={!r}", envvars.TURN_NDJSON_MAX_BYTES, raw)
        return _DEFAULT_TURN_NDJSON_MAX_BYTES
    if value <= 0:
        logger.warning("Ignoring non-positive {}={}", envvars.TURN_NDJSON_MAX_BYTES, value)
        return _DEFAULT_TURN_NDJSON_MAX_BYTES
    return value


class _BoundedStreamWriter:
    """File-like wrapper that drops writes past a byte cap with a sentinel.

    The first write that crosses ``max_bytes`` lands a single ``__truncated``
    NDJSON marker (with the configured cap) and silently swallows everything
    that follows. The wrapper exposes the minimal subset of ``IO[str]`` used
    by agents (``write``, ``flush``, ``close``) so it is a drop-in
    replacement for any plain file handle bound to ``_stream_sink``.
    """

    def __init__(self, fh: IO[str], max_bytes: int) -> None:
        self._fh = fh
        self._max = max_bytes
        self._written = 0
        self._tripped = False

    @classmethod
    def from_path(cls, path: Path, max_bytes: int) -> "_BoundedStreamWriter":
        """Open ``path`` for writing and wrap it with a byte cap.

        Centralises the bare ``open()`` so call sites no longer carry a
        ``# noqa: SIM115`` (file ownership transfers into the writer's
        ``close()``).
        """
        return cls(open(path, "w"), max_bytes=max_bytes)  # noqa: SIM115 - handle owned by writer

    def write(self, data: str) -> int:
        if self._tripped:
            return 0
        size = len(data.encode("utf-8", errors="replace"))
        if self._written + size <= self._max:
            self._written += size
            return self._fh.write(data)
        self._tripped = True
        marker = json.dumps({"type": "__truncated", "max_bytes": self._max}) + "\n"
        try:
            self._fh.write(marker)
            self._fh.flush()
        except Exception as e:
            logger.debug("turn-stream truncation marker write failed: {}", e)
        return 0

    def flush(self) -> None:
        try:
            self._fh.flush()
        except Exception as e:
            logger.debug("turn-stream flush failed: {}", e)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception as e:
            logger.debug("turn-stream close failed: {}", e)


# Multi-turn message templating: closed-set placeholders that scenario authors
# can put inside ``Turn.message`` to reference fields from prior ``TurnOutput``s.
_TEMPLATE_FIELDS = ("reply_text", "git_diff", "tool_sequence")
# Captures both the supported shapes and any ``{{ prev.* }}`` / ``{{ turn_N.* }}``
# typo with an unsupported field (so the latter can fail loudly instead of
# silently passing the literal placeholder through to the agent).
_PLACEHOLDER_RE = re.compile(r"\{\{\s*(?P<scope>prev|turn_(?P<idx>\d+))\.(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def _format_field(turn_output: TurnOutput, field: str) -> str:
    if field == "reply_text":
        return turn_output.reply_text
    if field == "git_diff":
        return turn_output.git_diff or ""
    if field == "tool_sequence":
        return ", ".join(turn_output.tool_sequence)
    # _PLACEHOLDER_RE only routes here for the closed set above; any other
    # field name is rejected upstream in ``_render_turn_message``.
    raise ScenarioError(f"internal: unsupported template field {field!r}")


def _render_turn_message(template: str, prior_outputs: list[TurnOutput]) -> str:
    """Render ``{{prev.X}}`` / ``{{turn_N.X}}`` placeholders in ``template``.

    Supported fields: ``reply_text``, ``git_diff``, ``tool_sequence``.
    Returns ``template`` unchanged when no ``{{`` appears (zero-cost fast path
    for the common single-turn / no-templating case). Raises
    :class:`ScenarioError` on a placeholder that references a future turn,
    a turn-0 ``prev``, an unsupported field (e.g. ``{{prev.foo}}``), or a
    rendered length above ``TURN_MESSAGE_MAX_CHARS``.

    Notes on canonical security helpers:

    - :mod:`belt._safe` and :mod:`belt._sanitize` are intentionally
      not applied here. Their job is to neutralise markup or control bytes
      flowing from agent stdout into a markup-aware sink (Rich panel,
      Markdown card, CSV, JUnit). The rendered string here flows the other
      direction -- into the next ``agent.execute`` call as a plain-text
      prompt -- so any escaping would corrupt the agent's input.
    - ``re.sub`` with a callable replaces all matches in a single
      left-to-right pass over the *original* template. Output text from a
      prior turn that itself contains ``{{...}}`` is therefore not
      re-expanded, so an agent cannot smuggle a placeholder into a later
      turn's render by emitting one in its reply.
    - The post-render length cap mirrors the parser-side
      ``TURN_MESSAGE_MAX_CHARS`` cap on ``Turn.message`` -- a
      crash-prevention guardrail. Without it, a runaway prior-turn
      ``reply_text`` or ``git_diff`` could splice a multi-GB string into
      the next turn's message and OOM the runner.
    """
    if "{{" not in template:
        return template

    def replace(match: re.Match[str]) -> str:
        scope = match.group("scope")
        field = match.group("field")
        if field not in _TEMPLATE_FIELDS:
            supported = ", ".join(_TEMPLATE_FIELDS)
            raise ScenarioError(f"unsupported template field {match.group(0)!r} -- " f"supported fields: {supported}")
        if scope == "prev":
            if not prior_outputs:
                raise ScenarioError(f"template {match.group(0)!r} used on turn 0 -- " f"no previous turn exists")
            return _format_field(prior_outputs[-1], field)
        # turn_N.field
        idx = int(match.group("idx"))
        if idx >= len(prior_outputs):
            raise ScenarioError(
                f"template {match.group(0)!r} references turn {idx}, "
                f"but only turns 0..{len(prior_outputs) - 1} have run "
                f"(current turn index is {len(prior_outputs)})"
            )
        return _format_field(prior_outputs[idx], field)

    rendered = _PLACEHOLDER_RE.sub(replace, template)
    if len(rendered) > TURN_MESSAGE_MAX_CHARS:
        raise ScenarioError(
            f"rendered turn message length {len(rendered)} exceeds "
            f"TURN_MESSAGE_MAX_CHARS={TURN_MESSAGE_MAX_CHARS} after splicing "
            f"prior-turn placeholders -- check that the referenced "
            f"reply_text / git_diff / tool_sequence is not unbounded"
        )
    return rendered


def run_scenario_turns(
    agent: BaseAgentAdapter,
    scenario: Scenario,
    outcome_dir: Path,
    config: AgentConfig,
    workspace: Path | None = None,
    workspace_manager: WorkspaceManager | None = None,
    *,
    stream: bool = True,
    group_name: str | None = None,
    resource_locks: list[dict[str, Any]] | None = None,
    group_dir: Path | None = None,
) -> ScenarioResult:
    """Execute all turns of a scenario through the agent.

    Lifecycle: agent.setup → turns(execute + fetch_results) → agent.teardown.
    Artifacts are written per turn; a sentinel file is written on failure so
    downstream scorers can detect and report the error.

    Args:
        workspace: Root directory for StateExpectation file checks.
                   Defaults to Path.cwd() if not provided.
        workspace_manager: When provided, acquires an isolated git worktree
                          for this scenario and auto-captures git diff/files_modified.
        stream: Write live NDJSON stream files per turn for external observers.
        group_name: Canonical scenario-group identity (e.g.
                   ``"agent-capabilities"``) recorded in the per-scenario
                   runtime sidecar. Required to keep the sidecar correct
                   when ``scenarios_root`` and the group directory are the
                   same path - a layout where ``outcome_dir`` collapses to
                   ``<run_dir>/<scenario_name>`` and the group name can no
                   longer be inferred from the path.
    """
    outcome_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    ws = workspace or Path.cwd()

    isolated_ws: Path | None = None
    if workspace_manager is not None:
        isolated_ws = workspace_manager.acquire(config.scenario_name)
        ws = isolated_ws
        config.workspace_dir = str(isolated_ws)
        logger.info("  workspace isolated: {}", isolated_ws)

    # Resolve the sandbox profile for this scenario and attach the spawner to
    # the agent BEFORE ``setup()`` runs -- agents often spawn subprocesses from
    # ``setup`` (e.g. login probes), and the spawner must already be in place
    # so those calls land inside the sandbox too. Keep ``provider`` / ``handle``
    # in scope so the ``finally`` block can call ``teardown()`` even if
    # ``setup()`` raises.
    sandbox_provider: BaseSandboxProvider | None = None
    sandbox_handle: SandboxHandle | None = None
    try:
        effective_profile = _resolve_effective_sandbox_profile(config.group_config)
        spawner, sandbox_provider, sandbox_handle = _build_sandbox_for_scenario(
            effective_profile,
            agent,
            ws,
            config.scenario_name,
        )
        agent._spawner = spawner
        if effective_profile.provider != "host":
            logger.info(
                "  sandbox: provider={}, image={}",
                effective_profile.provider,
                effective_profile.image,
            )
    except Exception as e:
        logger.error("Sandbox setup failed for {}: {}", config.scenario_name, e)
        _write_sentinel(outcome_dir, 0, f"sandbox setup failed: {e}", tb=traceback.format_exc())
        if workspace_manager is not None and isolated_ws is not None:
            try:
                workspace_manager.release(isolated_ws)
            except Exception as release_err:
                logger.warning("workspace release after sandbox failure: {}", release_err)
        return ScenarioResult(
            scenario_name=scenario.name,
            group_path=str(outcome_dir.parent),
            error=f"sandbox setup failed: {e}",
            outcome_dir=str(outcome_dir),
        )

    if resource_locks is not None and isolated_ws is not None:
        # Install before ``agent.setup`` so the agent sees the resources at
        # session start. Failures abort the scenario with a sentinel CLI
        # output so downstream scorers detect the error.
        from belt.runner.workspace import install_resources as _install_resources

        try:
            installed = _install_resources(
                isolated_ws,
                config.group_config.resources,
                config_dir=group_dir,
            )
        except Exception as e:
            logger.error("Resource install failed for {}: {}", config.scenario_name, e)
            _write_sentinel(outcome_dir, 0, f"resource install failed: {e}", tb=traceback.format_exc())
            return ScenarioResult(
                scenario_name=scenario.name,
                group_path=str(outcome_dir.parent),
                error=f"resource install failed: {e}",
                outcome_dir=str(outcome_dir),
            )
        resource_locks.extend(installed)

    agent.setup(config)

    _write_runtime_info_sidecar(outcome_dir, agent, config, group_name=group_name)

    if resource_locks:
        # Persist alongside scenario artefacts so reviewers can pin
        # "this result corresponds to skill v0.91 sha256:abcd..." without
        # re-deriving from the run directory tree.
        write_json(outcome_dir / "resource_lock.json", resource_locks)

    completed = 0
    agent_cost: float | None = None
    agent_errors: list[str] = []
    # Collected ``TurnOutput`` for every turn that has finished ``fetch_results``
    # so far. Drives ``{{prev.*}}`` / ``{{turn_N.*}}`` rendering for the next
    # turn's message; intentionally not appended on agent failure (the rendered
    # message would be misleading).
    turn_outputs_so_far: list[TurnOutput] = []
    try:
        for i, turn in enumerate(scenario.turns):
            stream_fh = None
            try:
                rendered_message = _render_turn_message(turn.message, turn_outputs_so_far)
                logger.info("  turn {}: {}...", i, rendered_message[:60])

                if stream:
                    stream_path = outcome_dir / TURN_STREAM_TEMPLATE.format(i)
                    stream_fh = _BoundedStreamWriter.from_path(stream_path, max_bytes=_turn_stream_cap())
                    user_event = json.dumps({"type": "user_input", "message": rendered_message})
                    stream_fh.write(user_event + "\n")
                    stream_fh.flush()
                    agent._stream_sink = stream_fh

                raw_output = agent.execute(rendered_message, turn.flags)
            except Exception as e:
                logger.error("Turn {} failed: {}", i, e)
                _write_sentinel(outcome_dir, i, str(e), tb=traceback.format_exc())
                break
            finally:
                agent._stream_sink = None
                if stream_fh is not None:
                    stream_fh.close()

            try:
                turn_output = agent.fetch_results(raw_output)
            except Exception as e:
                logger.error("Turn {} fetch_results failed: {}", i, e)
                _write_sentinel(outcome_dir, i, f"fetch_results error: {e}", tb=traceback.format_exc())
                turn_output = TurnOutput(raw_cli=raw_output, has_error=True)

            if turn_output.cost_usd is not None:
                agent_cost = (agent_cost or 0.0) + turn_output.cost_usd

            # Record per-turn agent failures so the run-phase footer and
            # the aggregator headline can surface "the agent didn't really
            # run" without re-reading turn outputs from disk.
            if turn_output.has_error:
                agent_errors.append(turn_output.error_type or UNKNOWN)

            _capture_workspace_state(turn_output, turn.state_expect, ws)

            if isolated_ws is not None and workspace_manager is not None:
                _capture_git_state(turn_output, workspace_manager, isolated_ws)

            # Per-turn ``verify``: run the author-declared command in the
            # worktree after capture. Skipped when the agent errored on this
            # turn (the worktree state is meaningless) -> the scorer emits a
            # skipped (passed=None) check rather than a false fail. The setup
            # gate already enforced --allow-verify-exec + worktree presence.
            if turn.verify is not None and isolated_ws is not None and not turn_output.has_error:
                logger.info("  turn {}: verify -> {}", i, " ".join(turn.verify.cmd))
                turn_output.verify_result = _run_verify_command(turn.verify, spawner, ws)

            # Record the finalized turn output (post-capture, so any later
            # ``{{prev.git_diff}}`` or ``{{prev.tool_sequence}}`` reference
            # picks up the populated fields).
            turn_outputs_so_far.append(turn_output)

            _write_turn_artifacts(outcome_dir, i, turn_output)

            parts = ["cli \u2713"]
            if turn_output.raw_state:
                parts.append("state \u2713")
            if turn_output.workspace_files:
                parts.append(f"files({len(turn_output.workspace_files)}) \u2713")
            if turn_output.git_diff:
                parts.append(f"diff({len(turn_output.files_modified)} files) \u2713")
            parts.append("output \u2713")
            if turn_output.cost_usd is not None:
                parts.append(f"${turn_output.cost_usd:.4f}")
            logger.info("    captured: {}", ", ".join(parts))
            completed += 1

        # Per-scenario ``verify`` (end-of-conversation): run once after every
        # turn completed, while the worktree still exists. Recorded on the
        # final turn's output under ``scenario_verify_result`` (re-written so
        # the on-disk artifact carries it). Skipped if the conversation did
        # not run to completion (an early break leaves stale workspace state).
        if (
            scenario.verify is not None
            and isolated_ws is not None
            and completed == len(scenario.turns)
            and turn_outputs_so_far
        ):
            logger.info("  scenario verify -> {}", " ".join(scenario.verify.cmd))
            last_output = turn_outputs_so_far[-1]
            last_output.scenario_verify_result = _run_verify_command(scenario.verify, spawner, ws)
            _write_turn_artifacts(outcome_dir, completed - 1, last_output)
    finally:
        try:
            meta = agent.metadata()
        except Exception as e:
            logger.warning("agent.metadata() failed: {}", e)
            meta = None
        try:
            agent.teardown()
        except Exception as e:
            logger.warning("agent.teardown() failed: {}", e)
        if sandbox_provider is not None and sandbox_handle is not None:
            try:
                sandbox_provider.teardown(sandbox_handle)
            except Exception as e:
                logger.warning("sandbox teardown failed: {}", e)
        if workspace_manager is not None and isolated_ws is not None:
            try:
                workspace_manager.release(isolated_ws)
            except Exception as e:
                logger.warning("workspace release failed: {}", e)

    return ScenarioResult(
        scenario_name=scenario.name,
        group_path=str(outcome_dir.parent),
        turns_completed=completed,
        agent_cost_usd=agent_cost,
        agent_errors=agent_errors,
        outcome_dir=str(outcome_dir),
        agent_metadata=meta,
    )


def build_agent_config(
    group_config: GroupConfig,
    scenario: Scenario,
    shared_state: Any,
) -> AgentConfig:
    """Build an AgentConfig for a scenario, extracting agent-relevant options."""
    return AgentConfig(
        group_config=group_config,
        scenario_name=scenario.name,
        shared_state=shared_state,
        scenario_options={},
    )


def _capture_git_state(
    turn_output: TurnOutput,
    workspace_manager: WorkspaceManager,
    worktree: Path,
) -> None:
    """Auto-capture git diff and files_modified from the isolated worktree."""
    try:
        git_diff, files_modified = workspace_manager.capture_diff(worktree)
        turn_output.git_diff = git_diff
        turn_output.files_modified = files_modified
    except Exception as e:
        logger.warning("Failed to capture git state from worktree: {}", e)


# Byte cap on captured ``verify`` stdout. A hostile or runaway test command
# could otherwise emit unbounded output and OOM the runner; the cap bounds
# resident memory regardless of how chatty the command is.
_VERIFY_STDOUT_CAP_BYTES = 1 * 1024 * 1024


def _run_verify_command(spec: VerifySpec, spawner: SubprocessRunner, workspace: Path) -> VerifyResult:
    """Execute a ``VerifySpec`` command in ``workspace`` and capture the result.

    Runs through ``spawner`` so the command lands inside the active sandbox
    (the docker provider mounts only the worktree; the host provider runs it
    in ``workspace`` directly). The environment is the minimal base set with
    NO provider credentials - a verify command is a test runner, not the
    agent, so it has no business seeing API keys. stdout is captured under a
    byte cap, the command is bounded by ``spec.timeout``, and a timeout kills
    the whole process group. Never raises: any failure to spawn collapses to
    a non-zero ``exit_code`` so scoring always has a deterministic result.
    """
    from belt.agent.base import _kill_process_tree, build_subprocess_env

    env = build_subprocess_env()
    start = time.monotonic()
    try:
        proc = spawner.popen(
            list(spec.cmd),
            cwd=str(workspace),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except Exception as e:  # noqa: BLE001 - report any spawn failure as a non-zero verify result
        logger.warning("verify command failed to start: {}", e)
        return VerifyResult(
            cmd=list(spec.cmd), exit_code=127, stdout=f"verify command failed to start: {e}", duration_s=0.0
        )

    # ``communicate(timeout=...)`` reads stdout AND enforces the deadline in one
    # call - reading the pipe to EOF first would block past the timeout. On a
    # timeout we kill the whole process group and drain what was buffered.
    # Captured stdout is untrusted command output, so it is ANSI/OSC-stripped at
    # capture (``strip_ansi``) before storage - a terminal-escape sequence can
    # never reach a future renderer or split an ``output_contains`` match - then
    # truncated to the byte cap so the stored / scored output stays bounded.
    # (Same posture as ``_capture_cli_version``; newlines are preserved so
    # multi-line test output stays readable.)
    try:
        stdout, _ = proc.communicate(timeout=spec.timeout)
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        try:
            stdout, _ = proc.communicate(timeout=5)
        except Exception:  # noqa: BLE001 - best-effort drain after kill
            stdout = ""
        capped = strip_ansi(stdout or "")[:_VERIFY_STDOUT_CAP_BYTES]
        return VerifyResult(
            cmd=list(spec.cmd),
            exit_code=124,
            stdout=capped + f"\n...[verify timed out after {spec.timeout}s]",
            duration_s=time.monotonic() - start,
        )

    rc = proc.returncode if proc.returncode is not None else 1
    return VerifyResult(
        cmd=list(spec.cmd),
        exit_code=rc,
        stdout=strip_ansi(stdout or "")[:_VERIFY_STDOUT_CAP_BYTES],
        duration_s=time.monotonic() - start,
    )


def _safe_resolve(workspace: Path, rel_path: str) -> Path | None:
    """Resolve rel_path inside workspace; return None if it escapes the boundary."""
    try:
        ws_resolved = workspace.resolve()
        full = (ws_resolved / rel_path).resolve()
        full.relative_to(ws_resolved)
        return full
    except (ValueError, OSError):
        logger.warning("Path traversal blocked: '{}' escapes workspace '{}'", rel_path, workspace)
        return None


def _capture_workspace_state(
    turn_output: TurnOutput,
    state_expect: StateExpectation,
    workspace: Path,
) -> None:
    """Capture workspace files and git diff into TurnOutput for downstream scoring."""
    paths_to_check: set[str] = set()
    paths_to_check.update(state_expect.files_exist)
    paths_to_check.update(state_expect.files_contain.keys())
    paths_to_check.update(state_expect.files_not_exist)

    if paths_to_check:
        if len(paths_to_check) > _MAX_FILES_PER_TURN:
            logger.warning(
                "state_expect references {} files, capping at {} to prevent resource exhaustion",
                len(paths_to_check),
                _MAX_FILES_PER_TURN,
            )
        files: dict[str, str | None] = {}
        for rel_path in sorted(paths_to_check)[:_MAX_FILES_PER_TURN]:
            full = _safe_resolve(workspace, rel_path)
            if full is None:
                files[rel_path] = None
                continue
            if full.is_file():
                try:
                    content = full.read_text(errors="replace")
                    if len(content) > _MAX_FILE_CAPTURE_BYTES:
                        content = content[:_MAX_FILE_CAPTURE_BYTES] + "\n... (truncated)"
                    files[rel_path] = content
                except Exception as e:
                    logger.warning("Failed to read {}: {}", full, e)
                    files[rel_path] = None
            else:
                files[rel_path] = None
        turn_output.workspace_files = files

    if state_expect.capture_git_diff and not turn_output.raw_state:
        result = run_git("diff", "--no-color", cwd=workspace, timeout=10)
        if result is not None and result.returncode == 0 and result.stdout.strip():
            turn_output.raw_state = result.stdout
        elif result is not None and result.returncode != 0:
            logger.debug("git diff capture failed (rc={}): {}", result.returncode, result.stderr.strip())


def _write_runtime_info_sidecar(
    outcome_dir: Path,
    agent: BaseAgentAdapter,
    config: AgentConfig,
    *,
    group_name: str | None = None,
) -> None:
    """Persist agent runtime identity for the benchmark card.

    Called once per scenario after :meth:`BaseAgentAdapter.setup`. The
    written file is small (CLI path/version, auth signals, redacted agent
    args) and is read later by ``benchmark_card.build_card`` which
    deduplicates per group. Any error in this best-effort capture is
    debug-logged so a single misbehaving agent cannot derail an evaluation.

    ``group_name`` is the canonical group identity from
    ``MatchedGroup.name`` (a path relative to ``scenarios_root``). Falls
    back to ``outcome_dir.parent.name`` only when the caller did not
    supply one, matching the documented ``run_dir/<group>/<scenario>``
    layout. The fallback is unreliable when ``scenarios_root`` and the
    group directory are the same path (the relative segment collapses to
    ``"."`` and ``outcome_dir.parent`` is the run dir itself), so callers
    inside the framework should always pass ``group_name``.
    """
    try:
        cls = type(agent)
        # ``runtime_info()`` is the adapter-author contract and stays
        # flat (e.g. ``{"adapter_class": ..., "cli_binary_path": ...,
        # "cli_version": ..., "auth_signals": [...]}``). The framework
        # owns the persisted shape, so we project that flat dict into
        # the two-level sidecar schema (``agent`` / ``cli``) here. New
        # adapter authors keep writing simple flat dicts; consumers of
        # the sidecar (and of ``benchmark-card.json``) get a stable,
        # logically-grouped layout.
        flat = cls.runtime_info()
        try:
            cli_options = cls.cli_options()
        except Exception:
            cli_options = []
        captured = getattr(agent, "_captured_agent_args", {}) or {}
        # ``_runtime_info.json`` is intentionally unversioned: it is an
        # internal input to the benchmark card whose fields are absorbed
        # into the versioned card and never surface independently to
        # external readers. The card's own ``schema_version`` covers it.
        sidecar_data = {
            "group": group_name or outcome_dir.parent.name,
            "agent": {
                "name": config.group_config.agent,
                "adapter_class": flat.get("adapter_class", cls.__name__),
                "args": safe_agent_args(
                    {str(k): str(v) for k, v in captured.items()},
                    cli_options=cli_options,
                ),
                "auth_signals": list(flat.get("auth_signals") or []),
            },
            "cli": {
                "binary_path": flat.get("cli_binary_path"),
                "version": flat.get("cli_version"),
            },
        }
        write_json(outcome_dir / RUNTIME_INFO_FILE, sidecar_data)
    except Exception as e:
        logger.debug("runtime_info sidecar write failed: {}", e)


def _write_turn_artifacts(outcome_dir: Path, turn_idx: int, turn_output: TurnOutput) -> None:
    turn_output.schema_version = SCHEMA_VERSION
    (outcome_dir / TURN_CLI_TEMPLATE.format(turn_idx)).write_text(turn_output.raw_cli)

    if turn_output.raw_state:
        state_text = turn_output.raw_state
        try:
            state_text = json.dumps(bounded_json_loads(state_text), indent=2, ensure_ascii=False) + "\n"
        except (json.JSONDecodeError, TypeError, ValueError, RecursionError) as e:
            logger.debug("turn {} raw_state pretty-print skipped: {}", turn_idx, e)
        (outcome_dir / TURN_STATE_TEMPLATE.format(turn_idx)).write_text(state_text)

    (outcome_dir / TURN_OUTPUT_TEMPLATE.format(turn_idx)).write_text(turn_output.model_dump_json(indent=2) + "\n")


def _resolve_effective_sandbox_profile(group_config: GroupConfig) -> SandboxProfile:
    """Apply the runner-level ``--sandbox`` / env override on top of the scenario default.

    Precedence: ``--sandbox`` flag (set by ``commands/run.py`` into
    ``BELT_SANDBOX_PROVIDER``) overrides the group's scenario-level
    ``sandbox.provider`` so an operator can opt into hardened execution
    for a scenario authored without sandbox awareness, or downgrade to
    ``host`` for fast iteration. Image, hosts, and passthrough lists
    stay scenario-owned -- those reflect the scenario's needs and do
    not change with the operator's hardness preference.
    """
    override = os.environ.get(envvars.SANDBOX_PROVIDER, "").strip().lower()
    if not override or override == group_config.sandbox.provider:
        return group_config.sandbox
    return group_config.sandbox.model_copy(update={"provider": override})


def _build_sandbox_for_scenario(
    profile: SandboxProfile,
    agent: BaseAgentAdapter,
    workspace_dir: Path,
    scenario_name: str,
) -> tuple[SubprocessRunner, BaseSandboxProvider | None, SandboxHandle | None]:
    """Resolve the provider, set up a handle, and return the spawner triple.

    Returns ``(spawner, provider, handle)``. ``provider`` and ``handle`` are
    ``None`` when the profile resolves to ``host`` -- ``LocalSpawner`` is
    behaviour-identical to a direct ``Popen`` call so the rest of the
    orchestrator can stay branch-free.

    Provider capability validation runs first, for every provider
    (including ``host``). A profile that asks for isolation the chosen
    provider cannot enforce raises :class:`SandboxConfigError` here, before
    any subprocess spawns -- this is what stops a scenario authored with
    ``provider=host, network_policy=none`` from silently running on the
    host's open network.
    """
    provider_cls = get_sandbox_provider(profile.provider)
    provider = provider_cls()
    ctx = SandboxContext(
        workspace_dir=workspace_dir,
        agent_required_env=type(agent).required_env_vars(),
        scenario_name=scenario_name,
    )
    provider.validate_profile(profile, ctx)
    if profile.provider == "host":
        return LocalSpawner(), None, None
    handle = provider.setup(profile, ctx)
    return SandboxedSpawner(provider, handle), provider, handle


def _write_sentinel(outcome_dir: Path, turn_idx: int, error_msg: str, tb: str | None = None) -> None:
    """Write a sentinel CLI output file so scorers detect the failure.

    Includes the Python traceback when available, so post-mortem debugging
    doesn't require access to the original terminal session.
    """
    try:
        sentinel = outcome_dir / TURN_CLI_TEMPLATE.format(turn_idx)
        if not sentinel.exists():
            text = f"RuntimeError: agent failed on turn {turn_idx} - {error_msg}\n"
            if tb:
                text += f"\n{tb}"
            sentinel.write_text(text)
    except Exception as e:
        logger.warning("Failed to write sentinel file for turn {}: {}", turn_idx, e)
