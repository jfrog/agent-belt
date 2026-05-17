# (c) JFrog Ltd. (2026)

"""Base agent interface for evaluating CLI agents."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Iterable, Iterator

from belt import envvars
from belt._sanitize import sanitize, strip_ansi
from belt.agent.scoring import ScoringStrategy, default_scoring_strategy
from belt.entities import TurnOutput
from belt.runner.entities import AgentConfig
from belt.runner.process.spawner import LocalSpawner, SubprocessRunner
from belt.scenario import GroupConfig

# Module-level singleton shared by every agent that does not get a custom
# spawner injected by the runner (e.g. unit tests that instantiate an agent
# directly). LocalSpawner has no per-instance state, so sharing is safe and
# cheaper than rebuilding it on every agent construction.
_DEFAULT_SPAWNER = LocalSpawner()


@dataclass
class AgentOption:
    """Declaration of an agent-specific CLI option.

    When ``env_var`` is set, the framework auto-reads it as a fallback:
    ``-X flag`` > ``env_var`` > agent default.
    """

    name: str
    help: str
    required: bool = False
    default: str | None = None
    env_var: str | None = None


class AgentNotAvailableError(Exception):
    """Raised when an agent's CLI tool or credentials are not available."""

    def __init__(self, agent_name: str, reason: str, suggestion: str = ""):
        self.agent_name = agent_name
        self.reason = reason
        self.suggestion = suggestion
        msg = f"{agent_name}: {reason}"
        if suggestion:
            msg += f"\n  → {suggestion}"
        super().__init__(msg)


class AgentArgError(Exception):
    """Raised when an invalid -X agent arg is passed."""


# Captured ``--version`` output ceiling. ``--version`` for every agent we
# ship fits in well under 256 bytes (typical: ``"codex 0.42.1"``). The
# cap exists to defend against a malicious binary on PATH that pipes
# unbounded stdout: paired with ``Popen + read(N)`` it bounds the
# runner's resident memory regardless of how chatty the binary is.
_VERSION_OUTPUT_CAP_BYTES = 4096

# Sanitisation regexes live in :mod:`belt._sanitize` so a new
# escape shape (OSC hyperlinks, xterm window-title sequences) is
# handled in one place rather than re-spelled here.


def _kill_process_tree(proc: subprocess.Popen) -> None:  # type: ignore[type-arg]
    """Kill an entire process group (subprocess + its children), falling back to proc.kill()."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        proc.kill()
    proc.wait()


def _drain_stderr(proc: subprocess.Popen) -> threading.Thread:  # type: ignore[type-arg]
    """Start a daemon thread that drains proc.stderr to avoid pipe deadlock.

    Returns the thread. Caller should join() after proc.wait().
    The collected lines are stored on thread.lines (list[str]).
    """
    lines: list[str] = []

    def _reader() -> None:
        if proc.stderr is None:
            return
        for line in proc.stderr:
            lines.append(line)

    t = threading.Thread(target=_reader, daemon=True)
    t.lines = lines  # type: ignore[attr-defined]
    t.start()
    return t


# Stream-bounding caps for a single agent subprocess's stdout. A misbehaving
# or hostile agent could produce unbounded output (e.g. millions of long
# pretty-printed JSON lines) and OOM the host. These caps keep an evaluator's
# memory budget predictable. ``LINE_MAX`` truncates a single line;
# ``MAX_BYTES`` aborts accumulation once total in-memory output crosses the
# threshold. Overrides for known-large outputs:
# ``envvars.SUBPROCESS_STDOUT_LINE_MAX`` / ``envvars.SUBPROCESS_STDOUT_MAX_BYTES``.
_DEFAULT_SUBPROCESS_STDOUT_LINE_MAX = 1 * 1024 * 1024  # 1 MiB per line
_DEFAULT_SUBPROCESS_STDOUT_MAX_BYTES = 256 * 1024 * 1024  # 256 MiB total stream


def _stream_cap_env(name: str, fallback: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return fallback
    try:
        value = int(raw)
    except ValueError:
        return fallback
    return value if value > 0 else fallback


def iter_bounded_stream(
    stream: IO[str],
    *,
    max_line_len: int | None = None,
    max_bytes: int | None = None,
) -> Iterator[str]:
    """Iterate ``stream`` line-by-line under per-line and total-payload caps.

    Drop-in replacement for ``for line in proc.stdout:`` in agents. Each line
    is truncated to ``max_line_len`` characters; iteration stops once the total
    payload reaches ``max_bytes`` so a runaway agent cannot OOM the evaluator.
    A truncation marker is yielded as the final line in either case.

    Caps default to ``envvars.SUBPROCESS_STDOUT_LINE_MAX`` /
    ``envvars.SUBPROCESS_STDOUT_MAX_BYTES`` env overrides, then to module
    defaults (1 MiB per line, 256 MiB total).
    """
    from belt import envvars

    if max_line_len is None:
        max_line_len = _stream_cap_env(envvars.SUBPROCESS_STDOUT_LINE_MAX, _DEFAULT_SUBPROCESS_STDOUT_LINE_MAX)
    if max_bytes is None:
        max_bytes = _stream_cap_env(envvars.SUBPROCESS_STDOUT_MAX_BYTES, _DEFAULT_SUBPROCESS_STDOUT_MAX_BYTES)

    total = 0
    line_truncation_marker = "...[line truncated]\n"
    for line in stream:
        if len(line) > max_line_len:
            line = line[:max_line_len] + line_truncation_marker
        if total + len(line) > max_bytes:
            from loguru import logger

            logger.warning(
                "Stream cap reached at {} bytes; dropping further output.",
                total,
            )
            yield f"...[stream truncated after {max_bytes} bytes]\n"
            try:
                while stream.read(64 * 1024):
                    pass
            except Exception:
                pass
            return
        total += len(line)
        yield line


def resolve_binary(
    candidates: Iterable[str],
    extra_paths: Iterable[str | Path] = (),
) -> str | None:
    """Locate the first available binary for an agent CLI.

    Resolution order:
      1. ``shutil.which(name)`` for each candidate, in order - covers
         anything on ``$PATH``.
      2. ``<extra_path>/<name>`` for each (extra_path, name) combination,
         in order; covers official installer locations not yet on
         ``$PATH`` (e.g. ``~/.local/bin`` right after
         ``curl https://cursor.com/install | bash``).

    The first executable hit wins. Returns the absolute path or
    ``None``.

    Every adapter routes its ``check_available`` / ``display_info`` /
    ``runtime_info`` lookups through this function instead of calling
    bare ``shutil.which`` so that:

      - Aliased binaries (e.g. ``cursor-agent`` vs the legacy IDE-bundled
        ``cursor``) resolve to whichever is present.
      - Installs that haven't run ``source ~/.bashrc`` yet still work in
        CI.
      - There is exactly one site to add caching, instrumentation, or a
        per-host allowlist if those become necessary.

    The runner's worktree manager and the benchmark-card collector use
    :func:`belt._git.git_available` for the same purpose, scoped to
    git.
    """
    for name in candidates:
        if not name:
            continue
        hit = shutil.which(name)
        if hit:
            return hit

    for extra in extra_paths:
        base = Path(extra).expanduser()
        for name in candidates:
            if not name:
                continue
            candidate = base / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def _detect_auth_signals(
    env_vars: Iterable[str] = (),
    paths: Iterable[str | Path] = (),
) -> list[str]:
    """Return positive, verifiable auth signals for ``belt doctor``.

    Detects two classes of signal - both cheap, neither reads file contents:

      - **env var present** → ``f"env {NAME}"``  (first non-empty wins; one entry max)
      - **stored login**    → ``f"stored login ({short_path})"``  (first existing path wins)

    The function is deliberately conservative: it never claims "not authenticated"
    because we can't reliably distinguish "no auth at all" from "logged in via a
    mechanism we don't catalogue (e.g. system keychain, OAuth refresh token)".
    Failure mode is "missing positive signal", never a false negative.

    Returns a list (possibly empty). Display order: env var first (it's what runs
    in CI and is the most likely confusion source), then stored login.
    """
    signals: list[str] = []

    for name in env_vars:
        if name and os.environ.get(name):
            signals.append(f"env {name}")
            break

    for raw in paths:
        p = Path(raw).expanduser()
        if p.exists():
            try:
                short = "~/" + str(p.relative_to(Path.home()))
            except ValueError:
                short = str(p)
            signals.append(f"stored login ({short})")
            break

    return signals


def _sanitize_stderr(text: str, max_chars: int = 200) -> str:
    """Sanitize agent stderr for safe logging: strip ANSI escapes, collapse newlines, truncate."""
    text = strip_ansi(text)
    text = text.replace("\n", " ").replace("\r", "")
    return text[:max_chars]


def _check_denied_flags(flags: list[str], denied: frozenset[str], agent_name: str) -> list[str]:
    """Filter out denied flags from scenario-provided flags, logging warnings.

    Each entry in ``denied`` is one of two shapes:

    - ``"--flag"`` blocks the flag with any value (covers ``--flag``,
      ``--flag=value``, and the two-token form ``--flag value``).
    - ``"--flag=value"`` blocks the flag only when given the listed
      ``value``, in either the equals-form (``--flag=value``) or the
      two-token form (``--flag value``). Other values pass through.

    Value-specific entries let an agent block one dangerous setting on a
    multi-value flag without over-blocking the safe values: codex's
    ``--sandbox=danger-full-access`` is denied while ``--sandbox=read-only``
    and ``--sandbox=workspace-write`` continue to work.
    """
    flag_only: set[str] = {entry for entry in denied if "=" not in entry}
    value_only: dict[str, set[str]] = {}
    for entry in denied:
        if "=" in entry:
            name, value = entry.split("=", 1)
            value_only.setdefault(name, set()).add(value)

    from loguru import logger

    def _log_block(rendered: str) -> None:
        logger.warning(
            "Blocked denied flag '{}' from scenario (agent: {}). "
            "Agents with a non-empty ``denied_flags()`` set strip these by default; "
            "remove the flag from the scenario or override the agent's deny-list to permit it.",
            rendered,
            agent_name,
        )

    clean: list[str] = []
    i = 0
    while i < len(flags):
        flag = flags[i]
        key, eq, eq_value = flag.partition("=")

        if key in flag_only:
            _log_block(flag)
            i += 1
            continue

        if key in value_only:
            if eq:
                if eq_value in value_only[key]:
                    _log_block(flag)
                    i += 1
                    continue
            elif i + 1 < len(flags) and flags[i + 1] in value_only[key]:
                _log_block(f"{flag} {flags[i + 1]}")
                i += 2
                continue

        clean.append(flag)
        i += 1
    return clean


# ── Minimal subprocess environment ──
#
# Inheriting the belt process's full ``os.environ`` would leak
# unrelated secrets (CI tokens, kube creds, AWS keys) into the CLI agent's
# process. Agents instead receive a conservative base set that real-world
# CLI tools need to function (PATH/locale/proxy/TLS/Node tooling) plus each
# agent's explicitly-declared API key vars from ``cli_options()`` and
# ``required_env_vars()``. Set ``BELT_ALLOW_FULL_ENV=1`` (or use the
# runner's ``--allow-full-env`` flag) for the permissive
# ``os.environ.copy()`` mode when the allow-list does not cover a use case.
_BASE_ENV_VARS: frozenset[str] = frozenset(
    {
        # Process basics
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "PWD",
        "TMPDIR",
        "TMP",
        "TEMP",
        # Locale
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "LC_NUMERIC",
        "LC_TIME",
        "LC_COLLATE",
        "LC_MONETARY",
        # Terminal / colour
        "TERM",
        "TERMINFO",
        "COLUMNS",
        "LINES",
        "NO_COLOR",
        "FORCE_COLOR",
        "CLICOLOR",
        "CLICOLOR_FORCE",
        # Proxy / TLS
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "ALL_PROXY",
        "all_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
        # Node / npm tooling
        "NODE_PATH",
        "NODE_OPTIONS",
        "NVM_DIR",
        "NVM_BIN",
        "NPM_TOKEN",
        # Python tooling
        "PYTHONPATH",
        "PIP_INDEX_URL",
        # Optional but harmless to pass through
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "DBUS_SESSION_BUS_ADDRESS",
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
    }
)


def _allow_full_env() -> bool:
    return envvars.is_truthy(envvars.ALLOW_FULL_ENV)


# Common provider auth/config vars. Agents typically require ONE of these to
# work, but auto-detect which is set; including all of them by default avoids
# breaking users who routinely pre-populate multiple providers.
_DEFAULT_PROVIDER_ENV_VARS: frozenset[str] = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
        "MISTRAL_API_KEY",
        "GROQ_API_KEY",
        "DEEPSEEK_API_KEY",
        "TOGETHER_API_KEY",
        "XAI_API_KEY",
        "FIREWORKS_API_KEY",
        "PERPLEXITY_API_KEY",
        "CURSOR_API_KEY",
        "WANDB_API_KEY",
        "AWS_PROFILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        envvars.OPENAI_BASE_URL,
        envvars.ANTHROPIC_BASE_URL,
    }
)


def build_subprocess_env(
    *,
    required: frozenset[str] | set[str] | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Construct a minimal subprocess environment.

    The returned dict contains the always-allowed base set, every variable in
    ``required`` (typically the API keys an agent declared via
    ``cli_options()`` or ``required_env_vars()``), every variable matching one
    of the ``NPM_CONFIG_*`` / ``LC_*`` prefix patterns, and any explicit
    ``extra`` overrides. Variables that aren't set on the parent process are
    quietly skipped.

    When ``BELT_ALLOW_FULL_ENV=1`` the function returns ``os.environ`` minus
    the two namespaces below, so users can recover from any false negatives
    in the allow-list without re-leaking belt-internal state into the
    subprocess.

    Variables matching :data:`belt._internal_envvars.PREFIX` (the
    private ``_BELT_*`` handoff namespace, e.g.
    ``_BELT_ORIGINAL_ARGV``) are *always* stripped, even under
    ``BELT_ALLOW_FULL_ENV=1``. They carry pre-redaction belt
    state that has no consumer in the agent process and would defeat the
    benchmark card's argv redaction if leaked: a hostile agent could
    read its own raw ``sys.argv`` from the child env. Stripping is
    unconditional because there is no legitimate use case for forwarding
    them - they exist solely for parent-to-child handoff between
    belt subcommands.

    Variables in the public ``BELT_*`` namespace (scorer credentials such
    as ``BELT_OPENAI_API_KEY`` / ``BELT_AZURE_CLIENT_SECRET`` and framework
    toggles like ``BELT_LOG_LEVEL`` / ``BELT_PRICING_FILE``) are *also*
    always stripped. The agent talks to its own provider via the
    un-prefixed names (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ...); the
    ``BELT_*`` namespace exists precisely so the scorer's credentials stay
    out of the agent process. ``BELT_ALLOW_FULL_ENV`` is an escape hatch
    for missing allow-list entries, not an opt-in to scorer-credential
    forwarding - that distinction would silently widen the leak surface
    every time a new ``BELT_*_API_KEY`` is added.

    The ``NPM_CONFIG_*`` / ``npm_config_*`` / ``LC_*`` prefix families are
    forwarded unconditionally in minimal-env mode without enumeration so
    locale and registry configuration reach the agent verbatim.
    """
    from belt._internal_envvars import PREFIX as _INTERNAL_PREFIX
    from belt.envvars import _PREFIX as _PUBLIC_PREFIX

    def _is_belt_namespaced(name: str) -> bool:
        # ``_BELT_*`` carries pre-redaction handoff state between belt phases;
        # ``BELT_*`` carries scorer credentials and framework toggles. Neither
        # has a consumer in the agent subprocess; both are stripped under
        # every code path including ``BELT_ALLOW_FULL_ENV=1``.
        return name.startswith(_INTERNAL_PREFIX) or name.startswith(_PUBLIC_PREFIX)

    if _allow_full_env():
        env = {k: v for k, v in os.environ.items() if not _is_belt_namespaced(k)}
        if extra:
            env.update(extra)
        return env

    env: dict[str, str] = {}
    parent = os.environ
    allowed = set(_BASE_ENV_VARS)
    if required:
        allowed.update(required)
    for name in allowed:
        if name in parent and not _is_belt_namespaced(name):
            env[name] = parent[name]
    for name, value in parent.items():
        if name.startswith(("NPM_CONFIG_", "npm_config_", "LC_")):
            env.setdefault(name, value)
    if extra:
        env.update(extra)
    return env


UNIVERSAL_OUTPUT_FIELDS: frozenset[str] = frozenset(
    {
        "raw_cli",
        "reply_text",
        "tool_calls",
        "has_reply",
        "has_error",
        "timing",
        "cost_usd",
        "error_type",
        "workspace_files",
        "schema_version",
    }
)

AGENT_SPECIFIC_FIELDS: frozenset[str] = frozenset(
    {
        "raw_state",
        "llm_turn_count",
        "thinking_text",
        "tool_sequence",
    }
)


class BaseAgentAdapter(ABC):
    """Protocol for driving a headless CLI agent through evaluation scenarios.

    One instance per scenario. The instance persists across all turns in the scenario,
    allowing agents to track inter-turn state (session IDs, thread state, etc.).

    Stream sink: the orchestrator may set ``_stream_sink`` before each ``execute()``
    call.  Agents that stream subprocess output line-by-line should write each
    line to the sink (and flush) so external tools can observe the agent live.
    Non-streaming agents can ignore it.

    ``check_available()`` contract:
      - MUST verify the agent binary exists and is invokable.
      - MUST complete in under 2 seconds on the happy path.
      - MUST NOT invoke a model or perform any network call that consumes credits.
      - SHOULD NOT parse human-readable status strings (fragile across versions).
      - MAY declare ``CREDENTIAL_ENV`` and ``CREDENTIAL_PATHS`` so ``doctor`` can
        surface positive auth signals (never gated on for execution).

    Authentication failures are surfaced at eval time as ``TurnOutput.has_error``,
    not as ``check_available`` failures.
    """

    _stream_sink: IO[str] | None = None
    _allow_unsafe_flags: bool = False
    # Subprocess spawner injected by the runner before ``setup()``. Default is a
    # pass-through ``LocalSpawner`` so agents instantiated outside the runner
    # (unit tests, ``belt agent info`` introspection) keep working without any
    # sandboxing knowledge. The runner replaces this with a ``SandboxedSpawner``
    # when ``--sandbox docker`` (or any plugin provider) is active.
    _spawner: SubprocessRunner = _DEFAULT_SPAWNER

    CREDENTIAL_ENV: tuple[str, ...] = ()
    CREDENTIAL_PATHS: tuple[str | Path, ...] = ()

    @classmethod
    def auth_signals(cls) -> list[str]:
        """Detected positive auth signals for this agent (informational only).

        Reads ``CREDENTIAL_ENV`` and ``CREDENTIAL_PATHS`` declared on the subclass.
        Never used to gate execution - surfaced by ``doctor`` as a hint about
        which credential source the user has set up.
        """
        return _detect_auth_signals(cls.CREDENTIAL_ENV, cls.CREDENTIAL_PATHS)

    @classmethod
    def denied_flags(cls) -> frozenset[str]:
        """Flags that must not be injected from scenario JSON.

        Override in subclasses. Default: empty (no flags denied).
        """
        return frozenset()

    def filter_flags(self, flags: list[str]) -> list[str]:
        """Apply the deny-list to scenario-provided flags."""
        if self._allow_unsafe_flags:
            return flags
        denied = self.denied_flags()
        if not denied:
            return flags
        return _check_denied_flags(flags, denied, type(self).__name__)

    @classmethod
    def supported_output_fields(cls) -> frozenset[str]:
        """Declare which agent-specific TurnOutput fields this agent populates.

        Returns a frozenset of field names beyond the universal set.
        Scenario expectations referencing unsupported fields produce a warning
        rather than a false failure.

        Override in subclasses. Default: empty (universal fields only).
        """
        return frozenset()

    @classmethod
    def check_available(cls) -> None:
        """Verify that the agent's CLI tool and credentials are available.

        Raises AgentNotAvailableError with an actionable message if not.
        Default implementation does nothing (agent is always available).
        Override in subclasses that depend on external CLI tools.
        """

    def health_check(self) -> None:
        """Verify that the agent's backend/service is reachable.

        Called once before group setup begins. Raises AgentNotAvailableError
        with a clear message if the agent cannot serve requests.
        Default: no-op (agent has no backend dependency).
        """

    @classmethod
    def cli_options(cls) -> list[AgentOption]:
        """Declare agent-specific options for help text and validation.

        Override in subclasses to declare accepted -X options.
        """
        return []

    @classmethod
    def required_env_vars(cls) -> frozenset[str]:
        """Return the env-var names this agent needs to reach the CLI agent.

        The default implementation harvests env vars declared via
        ``cli_options()`` and adds the well-known API-key names matching
        common providers (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, …) so
        agents that auto-detect credentials from any of several providers
        keep working even when the user did not pin one via ``-X``.

        Override to add agent-specific runtime vars (e.g. session caches,
        provider config dirs). Used by :func:`build_subprocess_env` to keep
        the child process env minimal.
        """
        names = {opt.env_var for opt in cls.cli_options() if opt.env_var}
        names.update(_DEFAULT_PROVIDER_ENV_VARS)
        return frozenset(names)

    def make_subprocess_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Convenience wrapper: build a minimal env using this agent's allow-list."""
        return build_subprocess_env(required=type(self).required_env_vars(), extra=extra)

    @classmethod
    def display_info(cls) -> str:
        """One-line string for progress header (version, server, etc.).

        Override in subclasses to show CLI version or connection info.
        """
        return cls.__name__

    @classmethod
    def _capture_cli_version(
        cls,
        cmd: list[str],
        timeout: float = 5.0,
        env: dict[str, str] | None = None,
    ) -> str | None:
        """Run ``cmd`` and return the first non-empty line of stdout, else ``None``.

        Helper for :meth:`runtime_info` overrides. Errors of any kind
        (binary missing, timeout, non-zero exit, decode failure) collapse
        to ``None`` so the benchmark card always populates a deterministic
        field. Never raises.

        Output is treated as **untrusted**: a malicious binary on the
        runner's PATH could pipe gigabytes of data, embed ANSI escapes,
        or output Markdown control characters that break out of the
        benchmark card's table cells. We defend against all three:

        - Bound captured stdout to a small ceiling
          (``_VERSION_OUTPUT_CAP_BYTES``) via ``Popen + read(N)`` so a
          chatty binary cannot grow the runner's resident memory
          regardless of how much it writes.
        - Take only the first non-empty line; ``--version`` is always
          single-line for the agents we ship.
        - Strip ANSI escape sequences and ASCII control bytes via
          :func:`belt._sanitize.sanitize` so the value renders as
          plain text in the TUI panel, GitHub Step Summary, and JSON
          sidecar.

        ``env`` overrides the inherited environment for adapters that
        need a curated PATH (e.g. codex needs the bundled Node 22+
        binary directory prepended). ``None`` means "inherit
        ``os.environ``", matching :func:`subprocess.Popen`'s default.
        """
        try:
            # ``start_new_session=True`` puts the child in its own
            # process group so we can kill it (and any descendants) via
            # ``_kill_process_tree`` without taking out the runner's own
            # process group. Without this flag ``os.killpg`` would reach
            # back to the running ``belt`` (or test) process.
            proc = subprocess.Popen(  # noqa: S603 - cmd is a fixed argv list owned by the adapter
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
                env=env,
            )
        except Exception:
            return None
        try:
            try:
                stdout = proc.stdout.read(_VERSION_OUTPUT_CAP_BYTES) if proc.stdout else ""
            except Exception:
                stdout = ""
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_process_tree(proc)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                return None
        finally:
            if proc.stdout is not None:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
        if proc.returncode != 0:
            return None
        first = (stdout or "").strip().split("\n", 1)[0]
        first = sanitize(first).strip()
        return first or None

    @classmethod
    def runtime_info(cls) -> dict[str, Any]:
        """Return per-run agent provenance for the benchmark card.

        Default returns the adapter class name plus the auth signals
        derived from ``CREDENTIAL_ENV`` / ``CREDENTIAL_PATHS``, and leaves
        ``cli_binary_path`` / ``cli_version`` as ``None``. Subclasses
        override to populate the CLI fields by invoking the agent's
        ``--version`` (or equivalent) command - typically with the help of
        :meth:`_capture_cli_version`.

        The returned dict is recorded verbatim into the per-scenario
        ``_runtime_info.json`` sidecar by the runner orchestrator and
        deduplicated per group when the benchmark card is assembled. Keys
        beyond the four canonical ones are preserved but ignored by the
        default card schema; future schema additions can pick them up
        without re-touching every adapter.
        """
        return {
            "adapter_class": cls.__name__,
            "cli_binary_path": None,
            "cli_version": None,
            "auth_signals": list(cls.auth_signals()),
        }

    # ── Group lifecycle (optional) ──

    def setup_group(self, group_config: GroupConfig, group_dir: Path) -> Any:
        """Create shared resources for a scenario group.

        ``group_dir`` is the filesystem path containing ``_config.json`` - agents
        use it to resolve relative assistant/config paths.

        Returns opaque shared_state passed to each scenario's AgentConfig.
        """
        return None

    def teardown_group(self, shared_state: Any) -> None:
        """Clean up shared group resources. Default: no-op."""
        pass

    # ── Scenario lifecycle ──

    @abstractmethod
    def setup(self, config: AgentConfig) -> None:
        """Per-scenario initialization.

        Config includes shared_state from setup_group and scenario_options.
        """
        ...

    @abstractmethod
    def execute(self, message: str, flags: list[str]) -> str:
        """Invoke the CLI agent with a message and flags; return raw output.

        The agent may transform flags before invocation based on
        agent-specific inter-turn context (e.g., expanding bulk decisions
        into per-operation values using stored prior-turn state).
        The orchestrator passes scenario-defined flags as-is.
        """
        ...

    @abstractmethod
    def fetch_results(self, raw_output: str) -> TurnOutput:
        """Normalize raw output into a TurnOutput.

        May perform additional work beyond parsing raw_output - e.g., a
        secondary command invocation to retrieve agent-internal state.
        """
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Per-scenario cleanup."""
        ...

    def metadata(self) -> dict[str, Any] | None:
        """Return optional per-scenario metadata stored in ScenarioResult.

        Examples: session token, thread ID, assistant ID, workspace path.
        """
        return None

    def group_setup_summary(self, shared_state: Any) -> str | None:
        """One-line summary of group setup for progress display.

        Examples: created resource ID, server label, workspace label.
        """
        return None

    def scoring_strategy(self) -> ScoringStrategy:
        """Return the scoring strategy for this agent's LLM judge evaluation."""
        return default_scoring_strategy()

    SUPPRESS_EVENT: tuple[str, str] = ("", "")

    @staticmethod
    def parse_stream_event(event: dict) -> tuple[str, str] | None:
        """Render a raw NDJSON event for live progress display.

        Returns:
          - ``(icon, summary)`` to render the event
          - ``SUPPRESS_EVENT`` (empty tuple pair) to hide the event
          - ``None`` to fall through to the generic renderer

        Override in subclasses to handle agent-specific event formats.
        """
        return None
