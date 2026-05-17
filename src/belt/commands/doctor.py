# (c) JFrog Ltd. (2026)

"""belt doctor - verify that agents and LLM scoring are ready to use.

Checks every layer of the evaluation stack and prints actionable fix suggestions.
Inspired by ``flutter doctor`` and ``brew doctor``.

Agent checks run in parallel with live streaming output so the user sees
results as they arrive instead of staring at a blank terminal.
"""

from __future__ import annotations

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

from rich.console import Console

from belt import envvars
from belt._sanitize import strip_ansi
from belt._ui import pluralize
from belt.agent.base import AgentNotAvailableError
from belt.agent.registry import available_agents, get_agent_class

DOCTOR_AGENT_TIMEOUT = 15


@dataclass
class CheckResult:
    ok: bool
    label: str
    detail: str = ""
    suggestion: str = ""
    auth_signals: list[str] = field(default_factory=list)


@dataclass
class DoctorReport:
    belt_version: str = ""
    python_version: str = ""
    agent_checks: list[CheckResult] = field(default_factory=list)
    llm_checks: list[CheckResult] = field(default_factory=list)
    exporter_checks: list[CheckResult] = field(default_factory=list)
    sandbox_checks: list[CheckResult] = field(default_factory=list)
    security_checks: list[CheckResult] = field(default_factory=list)
    install_check: Optional[CheckResult] = None

    @property
    def agents_ready(self) -> int:
        return sum(1 for c in self.agent_checks if c.ok)

    @property
    def llm_providers_ready(self) -> int:
        return sum(1 for c in self.llm_checks if c.ok)

    @property
    def exporters_ready(self) -> int:
        return sum(1 for c in self.exporter_checks if c.ok)

    @property
    def sandbox_ready(self) -> int:
        return sum(1 for c in self.sandbox_checks if c.ok)

    @property
    def security_warnings(self) -> int:
        return sum(1 for c in self.security_checks if not c.ok)


_SIMPLE_LLM_PROVIDERS: list[tuple[str, str, str]] = [
    ("OpenAI", envvars.OPENAI_API_KEY, f"export {envvars.OPENAI_API_KEY}=sk-..."),
    ("Anthropic", envvars.ANTHROPIC_API_KEY, f"export {envvars.ANTHROPIC_API_KEY}=sk-ant-..."),
]

_AZURE_ENDPOINT_VAR = envvars.AZURE_OPENAI_ENDPOINT
_AZURE_API_KEY_VAR = envvars.AZURE_OPENAI_API_KEY
_AZURE_SP_VARS = [
    envvars.AZURE_CLIENT_ID,
    envvars.AZURE_CLIENT_SECRET,
    envvars.AZURE_TENANT_ID,
]

# Stays in sync with ``BaseJudgeBackend`` subclasses and CONFIGURATION.md;
# parity is enforced by tests/test_doctor_providers.py and tests/test_provider_doc_parity.py.
ADVERTISED_LLM_PROVIDERS: tuple[str, ...] = ("OpenAI", "Anthropic", "Azure OpenAI", "Ollama")

# Surfaced in ``doctor --json`` so CI scripts can read the credentials they
# need to inject without parsing the free-form ``detail`` field.
_PROVIDER_ENV_VARS: dict[str, tuple[str, ...]] = {
    "OpenAI": (envvars.OPENAI_API_KEY,),
    "Anthropic": (envvars.ANTHROPIC_API_KEY,),
    "Azure OpenAI": (
        envvars.AZURE_OPENAI_ENDPOINT,
        envvars.AZURE_OPENAI_API_KEY,
        envvars.AZURE_CLIENT_ID,
        envvars.AZURE_CLIENT_SECRET,
        envvars.AZURE_TENANT_ID,
    ),
    "Ollama": (envvars.OLLAMA_BASE_URL,),
}


def _get_version() -> str:
    try:
        from importlib.metadata import version as pkg_version

        return pkg_version("agent-belt")
    except Exception:
        return "unknown"


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences and carriage returns from CLI output.

    Thin wrapper over :func:`belt._sanitize.strip_ansi` with
    ``simulate_erase=True`` so spinner / progress output from CLIs like
    Cursor resolves to the final displayed text. The trailing ``.strip()``
    is doctor-specific (single-line label rendering) and is not part of
    the canonical sanitiser.
    """
    return strip_ansi(text, simulate_erase=True).strip()


def _check_agent(name: str) -> CheckResult:
    """Check a single agent: loadable, installed, authenticated."""
    try:
        cls = get_agent_class(name)
    except Exception as e:
        return CheckResult(ok=False, label=name, detail=f"load error: {e}")

    try:
        cls.check_available()
    except AgentNotAvailableError as e:
        return CheckResult(ok=False, label=name, detail=e.reason, suggestion=e.suggestion)
    except Exception as e:
        return CheckResult(ok=False, label=name, detail=str(e))

    try:
        info = _strip_ansi(cls.display_info())
    except Exception:
        info = "available"

    try:
        signals = cls.auth_signals()
    except Exception:
        signals = []

    return CheckResult(ok=True, label=name, detail=info, auth_signals=signals)


def _check_agents_parallel(
    names: list[str],
    timeout: float = DOCTOR_AGENT_TIMEOUT,
    on_result: Optional[Callable[[CheckResult], None]] = None,
) -> list[CheckResult]:
    """Check all agents in parallel, returning results in name order.

    The overall timeout caps the wall-clock time for *all* checks.
    ``on_result`` is called as each agent finishes for live feedback.
    """
    if not names:
        return []

    results_by_name: dict[str, CheckResult] = {}
    pool = ThreadPoolExecutor(max_workers=len(names))
    future_to_name = {pool.submit(_check_agent, n): n for n in names}
    try:
        for future in as_completed(future_to_name, timeout=timeout):
            name = future_to_name[future]
            try:
                result = future.result()
            except Exception as exc:
                result = CheckResult(ok=False, label=name, detail=str(exc))
            results_by_name[name] = result
            if on_result:
                on_result(result)
    except TimeoutError:
        pass

    for future, name in future_to_name.items():
        if name not in results_by_name:
            result = CheckResult(
                ok=False,
                label=name,
                detail=f"timed out after {timeout:.0f}s",
                suggestion=f"Run `belt agent info {name}` to debug",
            )
            results_by_name[name] = result
            if on_result:
                on_result(result)
            future.cancel()

    pool.shutdown(wait=False, cancel_futures=True)
    return [results_by_name[n] for n in names]


def _check_azure_openai() -> CheckResult:
    """Check Azure OpenAI readiness: endpoint + one auth method."""
    endpoint = os.environ.get(_AZURE_ENDPOINT_VAR, "")
    has_api_key = bool(os.environ.get(_AZURE_API_KEY_VAR, ""))
    sp_missing = [v for v in _AZURE_SP_VARS if not os.environ.get(v, "")]
    has_sp = len(sp_missing) == 0

    if endpoint and (has_api_key or has_sp):
        auth = "API key" if has_api_key else "service principal"
        return CheckResult(ok=True, label="Azure OpenAI", detail=f"endpoint + {auth} configured")

    parts: list[str] = []
    if not endpoint:
        parts.append(f"{_AZURE_ENDPOINT_VAR} not set")
    has_any_sp = len(sp_missing) < len(_AZURE_SP_VARS)
    if not has_api_key and has_any_sp and sp_missing:
        parts.append(f"missing: {', '.join(sp_missing)}")
    elif not has_api_key and not has_any_sp:
        parts.append("no auth configured")

    suggestion_lines: list[str] = []
    if not endpoint:
        suggestion_lines.append(f"export {_AZURE_ENDPOINT_VAR}=<endpoint-url>")
    if not has_api_key and has_any_sp and sp_missing:
        for v in sp_missing:
            suggestion_lines.append(f"export {v}=<value>")
    elif not (has_api_key or has_sp):
        suggestion_lines.append("Then choose ONE auth method:")
        suggestion_lines.append(f"  API key: export {_AZURE_API_KEY_VAR}=<key>")
        suggestion_lines.append("  - or service principal:")
        for v in _AZURE_SP_VARS:
            suggestion_lines.append(f"    export {v}=<value>")
    suggestion = "\n      ".join(suggestion_lines)

    return CheckResult(ok=False, label="Azure OpenAI", detail="; ".join(parts), suggestion=suggestion)


def _check_ollama() -> CheckResult:
    """Check if Ollama is running and reachable."""
    import httpx as _httpx

    base_url = envvars.get_str(envvars.OLLAMA_BASE_URL, "http://localhost:11434")
    try:
        resp = _httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("models", [])
            names = [m.get("name", "?") for m in models[:5]]
            if models:
                head = ", ".join(names)
                extra = len(models) - len(names)
                suffix = f", +{extra} more" if extra > 0 else ""
                detail = f"{pluralize(len(models), 'model')}: {head}{suffix}"
            else:
                detail = "running, no models pulled"
            return CheckResult(ok=True, label="Ollama", detail=detail)
        return CheckResult(ok=False, label="Ollama", detail=f"HTTP {resp.status_code}")
    except (_httpx.ConnectError, _httpx.TimeoutException):
        return CheckResult(
            ok=False, label="Ollama", detail="not running", suggestion="brew install ollama && ollama serve"
        )
    except Exception as e:
        return CheckResult(ok=False, label="Ollama", detail=str(e))


def _check_llm_providers() -> list[CheckResult]:
    results = []
    for provider_name, env_var, suggestion in _SIMPLE_LLM_PROVIDERS:
        val = os.environ.get(env_var, "")
        if val:
            results.append(CheckResult(ok=True, label=provider_name, detail=f"{env_var} set"))
        else:
            results.append(
                CheckResult(ok=False, label=provider_name, detail=f"{env_var} not set", suggestion=suggestion)
            )
    results.append(_check_azure_openai())
    results.append(_check_ollama())
    results.append(_check_judge_model())
    return results


def _check_judge_model() -> CheckResult:
    """Resolve the judge model from the layered config and report its source.

    There is no built-in default: when none of CLI / env / yaml supplies
    one, this check is **informational red** so the user
    sees explicitly that LLM-mode scoring will fail-fast at preflight.
    Rules-only scoring still works without a judge model.
    """
    try:
        from belt.config import resolve_judge_model_source
    except Exception as exc:
        return CheckResult(
            ok=False,
            label="Judge model",
            detail=f"could not resolve config: {exc}",
        )

    from belt.constants import EXAMPLE_LLM_MODEL

    model, source = resolve_judge_model_source()
    if model is None:
        return CheckResult(
            ok=False,
            label="Judge model",
            detail="(not set) - LLM scoring will fail at preflight; rules-only still works",
            suggestion=(
                "Set one of:\n"
                f"      --scorer-arg model={EXAMPLE_LLM_MODEL}   (CLI flag)\n"
                f"      {envvars.LLM_MODEL}={EXAMPLE_LLM_MODEL}   (env var)\n"
                f"      belt.yaml -> llm.model: {EXAMPLE_LLM_MODEL}"
            ),
        )
    return CheckResult(ok=True, label="Judge model", detail=f"{model}  (from {source})")


_BASE_URL_PATTERN = re.compile(r"^BELT_[A-Z0-9_]+_BASE_URL$")
_SAFE_BASE_URL_PREFIXES = (
    "https://api.openai.com",
    "https://api.anthropic.com",
    "https://generativelanguage.googleapis.com",
    "http://localhost",
    "http://127.0.0.1",
)


def _check_install_integrity() -> Optional[CheckResult]:
    """Detect cross-clone shadowing of an editable install.

    If the user runs ``belt`` from inside an belt clone but the
    imported ``belt`` package resolves to a *different* clone's
    ``src/belt/``, the active install is silently using someone else's
    source tree. This typically happens when the user has multiple clones
    of belt in the same Python environment: ``pip install -e .`` writes
    a single ``.pth`` entry per environment, so whichever clone was installed
    last wins for the entire interpreter - making local changes appear to
    "vanish" and agent files to appear "missing" in the loser's clone.

    Returns ``None`` when no warning is needed (CWD is not an agent-belt
    clone, or the loaded module already matches CWD).
    """
    cwd = os.path.realpath(os.getcwd())
    cwd_src = os.path.join(cwd, "src", "belt", "__init__.py")
    if not os.path.isfile(cwd_src):
        return None

    try:
        import belt as _ae

        loaded_raw = getattr(_ae, "__file__", "") or ""
        loaded = os.path.realpath(loaded_raw) if loaded_raw else ""
    except Exception:
        return None

    if not loaded or os.path.realpath(cwd_src) == loaded:
        return None

    return CheckResult(
        ok=False,
        label="Install path",
        detail=f"loaded from {loaded} (not this clone)",
        suggestion=(
            "Another belt clone is active in this Python env. "
            "To use this clone instead: "
            'run `pip install -e ".\\[dev]"` here, or use a per-clone '
            "venv: `python -m venv .venv && source .venv/bin/activate && "
            'pip install -e ".\\[dev]"`.'
        ),
    )


def _check_exporters() -> list[CheckResult]:
    """Probe every registered exporter, mirroring the agent / LLM-provider sections.

    Built-in exporters (``csv``, ``jsonl``, ``junit``, ``markdown``) are
    filesystem-only and report ready unconditionally. Plugin exporters
    discovered via the ``belt.exporters`` entry-point group call their
    own :meth:`~belt.exporter.base.BaseExporter.is_available` so a
    Langfuse-style plugin can flag a missing API key the same way the
    Anthropic provider check flags a missing ``ANTHROPIC_API_KEY``.

    Errors during instantiation surface as ``ok=False`` with the load error in
    ``detail``; the doctor must never raise just because a third-party
    exporter has a bug at import time.
    """
    from belt.exporter.registry import _EXPORTER_REGISTRY, available_exporters, get_exporter_class

    # Built-in identity is derived from the in-tree registry (not a hard-coded
    # tuple) so adding or renaming a built-in does not require editing this
    # file - the registry is the single source of truth for built-in names.
    builtin_names = frozenset(_EXPORTER_REGISTRY)
    results: list[CheckResult] = []
    for name in available_exporters():
        try:
            exporter = get_exporter_class(name)()
        except Exception as e:
            results.append(
                CheckResult(
                    ok=False,
                    label=name,
                    detail=f"load error: {e}",
                )
            )
            continue
        try:
            ok = bool(exporter.is_available())
        except Exception as e:
            results.append(
                CheckResult(
                    ok=False,
                    label=name,
                    detail=f"is_available() raised: {e}",
                )
            )
            continue
        detail = "built-in" if name in builtin_names else "plugin"
        results.append(CheckResult(ok=ok, label=name, detail=detail))
    return results


def _check_sandbox_providers() -> list[CheckResult]:
    """Probe every registered sandbox provider for runtime readiness.

    Mirrors :func:`_check_exporters`: built-ins (``host``, ``docker``) plus
    any plugin discovered through the ``belt.sandbox_providers`` entry-point
    group. ``host`` is always ready by definition (no-op pass-through).
    ``docker`` reports ready iff a working ``docker`` binary is on PATH;
    ``docker --version`` is captured into ``detail`` so the user sees which
    Docker engine the runner would talk to. Plugin providers fall back to a
    bare "registered" line because the framework cannot probe their
    implementation specifics without coupling.

    Errors during introspection collapse to ``ok=False`` so a misbehaving
    plugin never raises out of ``belt doctor``.
    """
    from belt.runner.sandbox import available_sandbox_providers

    results: list[CheckResult] = []
    for name in available_sandbox_providers():
        if name == "host":
            results.append(
                CheckResult(
                    ok=True,
                    label="host",
                    detail="built-in (no-op pass-through; no isolation)",
                )
            )
            continue
        if name == "docker":
            from belt.runner.sandbox.docker import docker_version

            version = docker_version()
            if version:
                results.append(CheckResult(ok=True, label="docker", detail=f"built-in ({version})"))
            else:
                results.append(
                    CheckResult(
                        ok=False,
                        label="docker",
                        detail="docker CLI not on PATH",
                        suggestion="Install Docker, or stay on --sandbox host",
                    )
                )
            continue
        # Plugin provider: announce registration without attempting a probe.
        results.append(CheckResult(ok=True, label=name, detail="plugin (registered)"))
    return results


def _check_security_env() -> list[CheckResult]:
    """Surface security-relevant environment toggles so users notice rerouted traffic.

    ``BELT_*_BASE_URL`` lets a user point judge calls at a custom endpoint
    (corporate gateway, AWS Bedrock proxy, Ollama, mock). That is intentional -
    but a hostile dotenv or shell init could silently redirect API traffic to
    ``https://evil.example`` and exfiltrate prompts/responses. The doctor
    flags every override so it shows up as a visible warning rather than a
    silent default.

    To accept custom base URLs without per-run warnings, set
    ``BELT_SILENCE_CUSTOM_BASE_URL_WARNING=1``. Note that this only
    suppresses the warning - it does *not* permit insecure traffic; see
    ``BELT_ALLOW_INSECURE_BASE_URL`` for that.
    """
    results: list[CheckResult] = []
    silenced = envvars.is_truthy(envvars.SILENCE_CUSTOM_BASE_URL_WARNING)
    # The toggle names end in ``_BASE_URL`` and would otherwise match the
    # regex; surface them separately as "Base URL policy" below so they
    # don't appear as if they were custom-URL overrides.
    excluded_names = {envvars.SILENCE_CUSTOM_BASE_URL_WARNING}
    overrides = sorted(name for name in os.environ if _BASE_URL_PATTERN.match(name) and name not in excluded_names)
    for name in overrides:
        value = os.environ.get(name, "")
        is_safe_default = any(value.startswith(prefix) for prefix in _SAFE_BASE_URL_PREFIXES)
        if is_safe_default:
            results.append(
                CheckResult(
                    ok=True,
                    label="Base URL",
                    detail=f"{name} → {value} (recognised host)",
                )
            )
        elif silenced:
            results.append(
                CheckResult(
                    ok=True,
                    label="Base URL",
                    detail=(
                        f"{name} → {value} " f"(custom; warning silenced via {envvars.SILENCE_CUSTOM_BASE_URL_WARNING})"
                    ),
                )
            )
        else:
            results.append(
                CheckResult(
                    ok=False,
                    label="Base URL",
                    detail=f"{name} → {value}",
                    suggestion=(
                        "Custom base URLs redirect every judge call. Confirm this host is trusted, "
                        "set BELT_SILENCE_CUSTOM_BASE_URL_WARNING=1 to silence this once, "
                        "or unset the variable to use the provider default."
                    ),
                )
            )
    if envvars.is_truthy(envvars.ALLOW_FULL_ENV):
        results.append(
            CheckResult(
                ok=False,
                label="Env scoping",
                detail=f"{envvars.ALLOW_FULL_ENV}=1",
                suggestion=(
                    "Agents inherit your full os.environ - useful for debugging, risky on shared hosts. "
                    "Unset to fall back to the curated allow-list."
                ),
            )
        )
    if envvars.is_truthy(envvars.ALLOW_ARBITRARY_AGENT):
        results.append(
            CheckResult(
                ok=False,
                label="Agent loading",
                detail=f"{envvars.ALLOW_ARBITRARY_AGENT}=1",
                suggestion="Disable to require entry-point registration for third-party agents.",
            )
        )
    if envvars.is_truthy(envvars.ALLOW_ARBITRARY_SCORER):
        results.append(
            CheckResult(
                ok=False,
                label="Scorer loading",
                detail=f"{envvars.ALLOW_ARBITRARY_SCORER}=1",
                suggestion="Disable to require entry-point registration for third-party scorers.",
            )
        )
    if envvars.is_truthy(envvars.ALLOW_ARBITRARY_EXPORTER):
        results.append(
            CheckResult(
                ok=False,
                label="Exporter loading",
                detail=f"{envvars.ALLOW_ARBITRARY_EXPORTER}=1",
                suggestion="Disable to require entry-point registration for third-party exporters.",
            )
        )
    if silenced:
        # Surface the opt-in itself as a security check so users see the
        # tradeoff (silenced custom-URL warnings → trust the operator).
        results.append(
            CheckResult(
                ok=False,
                label="Base URL policy",
                detail=f"{envvars.SILENCE_CUSTOM_BASE_URL_WARNING}=1",
                suggestion=(
                    "Custom base URLs are accepted silently. Unset this var to fall back "
                    "to the explicit-warning policy."
                ),
            )
        )
    for var in (envvars.TURN_NDJSON_MAX_BYTES, envvars.CACHE_MAX_BYTES):
        # Surface the disk-budget overrides so a CI operator who set them in a
        # forgotten dotenv can find them quickly. These caps bound how much disk
        # the per-turn stream files and the LLM-judge response cache may consume.
        raw = os.environ.get(var, "")
        if raw:
            results.append(
                CheckResult(
                    ok=True,
                    label="Disk budget",
                    detail=f"{var}={raw}",
                )
            )
    return results


def run_doctor(console: Console | None = None) -> DoctorReport:
    """Run all doctor checks and return the report (parallel, no live output)."""
    report = DoctorReport()
    report.belt_version = _get_version()
    report.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    names = list(available_agents())
    report.agent_checks = _check_agents_parallel(names)

    report.llm_checks = _check_llm_providers()
    report.exporter_checks = _check_exporters()
    report.sandbox_checks = _check_sandbox_providers()
    report.security_checks = _check_security_env()
    report.install_check = _check_install_integrity()
    return report


def _hedge_auth_signal(signal: str) -> str:
    """Append a presence-only hedge to ``stored login`` signals.

    ``_detect_auth_signals`` reports two flavours:

    - ``"env <NAME>"`` - the env var is set to a non-empty value. This
      is a value, not a file, so its presence is a stronger (though
      still unverified) auth claim.
    - ``"stored login (<path>)"`` - a credential file exists on disk.
      We did not open it, did not check expiry, did not call the
      provider. The file may contain a long-expired token.

    We hedge only the second flavour. The user's mental model when they
    read ``stored login`` is "I am authenticated"; the hedge replaces
    that with the truth ("a file exists at this path"). The
    ``check_available()`` contract forbids us from doing better without
    explicit opt-in.
    """
    if signal.startswith("stored login "):
        return f"{signal} [italic](presence only - not verified)[/italic]"
    return signal


def _format_check(c: CheckResult) -> str:
    """Format a single check result as a Rich-markup line.

    For OK agent checks with declared credential sources, append a positive auth
    hint (e.g. ``· auth: env CURSOR_API_KEY + stored login``) - never claims
    ``not authenticated`` because we cannot verify negatives without invoking
    the agent (forbidden by the ``check_available()`` contract: no model calls,
    no credit consumption, no version-fragile status string parsing).

    Stored-login signals are rendered with a presence-only hedge (see
    :func:`_hedge_auth_signal`) so the user does not mistake credential
    presence on disk for credential validity.
    """
    if c.ok:
        line = f"  [green]✓[/green] {c.label:<16} {c.detail}"
        if c.auth_signals:
            hedged = [_hedge_auth_signal(s) for s in c.auth_signals]
            line += f"  [dim]· auth: {' + '.join(hedged)}[/dim]"
        elif _has_declared_credential_sources(c.label):
            line += "  [dim]· auth: unknown (binary OK)[/dim]"
        return line
    line = f"  [red]✗[/red] {c.label:<16} {c.detail}"
    if c.suggestion:
        line += f"\n    [dim]→ {c.suggestion}[/dim]"
    return line


def _any_stored_login_signal(report: DoctorReport) -> bool:
    """Whether any agent's auth signals include a ``stored login`` entry."""
    for c in report.agent_checks:
        for s in c.auth_signals:
            if s.startswith("stored login "):
                return True
    return False


def _has_declared_credential_sources(name: str) -> bool:
    """Whether the named agent declared CREDENTIAL_ENV or CREDENTIAL_PATHS.

    Used by doctor to decide whether to render the ``auth: unknown`` hint;
    only meaningful for agents that opted into auth-signal reporting.
    """
    try:
        cls = get_agent_class(name)
    except Exception:
        return False
    return bool(getattr(cls, "CREDENTIAL_ENV", ()) or getattr(cls, "CREDENTIAL_PATHS", ()))


def run_doctor_live(console: Console | None = None) -> DoctorReport:
    """Run doctor with live streaming - print each agent result as it arrives."""
    console = console or Console(stderr=True)
    report = DoctorReport()
    report.belt_version = _get_version()
    report.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    console.print("\n[bold]belt doctor[/bold]")
    console.print(f"{'=' * 40}")
    console.print(f"[green]✓[/green] belt {report.belt_version}")
    console.print(f"[green]✓[/green] Python {report.python_version}")

    report.install_check = _check_install_integrity()
    if report.install_check is not None:
        console.print(_format_check(report.install_check))

    names = list(available_agents())
    console.print(f"\n[bold]Agents[/bold] (you need at least one ✓)  - checking {len(names)} agents...")

    def _on_result(c: CheckResult) -> None:
        console.print(_format_check(c))

    report.agent_checks = _check_agents_parallel(names, on_result=_on_result)
    if _any_stored_login_signal(report):
        console.print(
            "  [dim italic]Auth signals indicate credential presence, not validity. "
            "If a run fails with auth errors, re-authenticate.[/dim italic]"
        )

    report.llm_checks = _check_llm_providers()
    console.print("\n[bold]LLM Scoring[/bold] (optional - rules-only scoring needs no key):")
    for c in report.llm_checks:
        console.print(_format_check(c))

    report.exporter_checks = _check_exporters()
    console.print("\n[bold]Exporters[/bold] (post-aggregation result writers):")
    for c in report.exporter_checks:
        console.print(_format_check(c))

    report.sandbox_checks = _check_sandbox_providers()
    console.print("\n[bold]Sandbox[/bold] (OS-level isolation for agent subprocesses):")
    for c in report.sandbox_checks:
        console.print(_format_check(c))

    report.security_checks = _check_security_env()
    if report.security_checks:
        console.print("\n[bold]Security[/bold] (review any overrides):")
        for c in report.security_checks:
            console.print(_format_check(c))

    _print_summary(report, console)
    return report


def print_report(report: DoctorReport, console: Console | None = None) -> None:
    """Render a pre-collected doctor report to the terminal."""
    console = console or Console(stderr=True)

    console.print("\n[bold]belt doctor[/bold]")
    console.print(f"{'=' * 40}")
    console.print(f"[green]✓[/green] belt {report.belt_version}")
    console.print(f"[green]✓[/green] Python {report.python_version}")

    if report.install_check is not None:
        console.print(_format_check(report.install_check))

    console.print("\n[bold]Agents[/bold] (you need at least one ✓):")
    for c in report.agent_checks:
        console.print(_format_check(c))
    if _any_stored_login_signal(report):
        console.print(
            "  [dim italic]Auth signals indicate credential presence, not validity. "
            "If a run fails with auth errors, re-authenticate.[/dim italic]"
        )

    console.print("\n[bold]LLM Scoring[/bold] (optional - rules-only scoring needs no key):")
    for c in report.llm_checks:
        console.print(_format_check(c))

    if report.exporter_checks:
        console.print("\n[bold]Exporters[/bold] (post-aggregation result writers):")
        for c in report.exporter_checks:
            console.print(_format_check(c))

    if report.sandbox_checks:
        console.print("\n[bold]Sandbox[/bold] (OS-level isolation for agent subprocesses):")
        for c in report.sandbox_checks:
            console.print(_format_check(c))

    if report.security_checks:
        console.print("\n[bold]Security[/bold] (review any overrides):")
        for c in report.security_checks:
            console.print(_format_check(c))

    _print_summary(report, console)


def _print_summary(report: DoctorReport, console: Console) -> None:
    console.print()
    if report.agents_ready > 0:
        agents_str = f"[green]{pluralize(report.agents_ready, 'agent')} ready[/green]"
    else:
        agents_str = "[red]no agents ready - install at least one (see table above)[/red]"
    llm_str = (
        f"{pluralize(report.llm_providers_ready, 'LLM provider')} configured"
        if report.llm_providers_ready
        else "no LLM providers (rules-only scoring still works)"
    )
    parts = [agents_str, llm_str]
    if report.exporter_checks:
        parts.append(f"{pluralize(report.exporters_ready, 'exporter')} ready")
    console.print(f"[bold]Summary:[/bold] {', '.join(parts)}")

    if report.agents_ready > 0:
        ready_name = next(c.label for c in report.agent_checks if c.ok)
        console.print("\n  You only need one agent. Next step:")
        console.print(f"  [cyan]belt quickstart {ready_name}[/cyan]")
    else:
        console.print("\n  Install at least one agent, then re-run [cyan]belt doctor[/cyan]")
    console.print()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``belt doctor``."""
    import argparse

    ap = argparse.ArgumentParser(prog="belt doctor", description="Check agent and LLM scoring readiness")
    ap.add_argument("--json", action="store_true", help="Output as JSON (for scripting)")
    args = ap.parse_args(argv)

    if args.json:
        import json

        report = run_doctor()
        data = {
            "belt_version": report.belt_version,
            "python_version": report.python_version,
            "agents": [
                {
                    "name": c.label,
                    "ok": c.ok,
                    "detail": c.detail,
                    "suggestion": c.suggestion,
                    "auth_signals": c.auth_signals,
                }
                for c in report.agent_checks
            ],
            "llm_providers": [
                {
                    "name": c.label,
                    "ok": c.ok,
                    "detail": c.detail,
                    "suggestion": c.suggestion,
                    "env_vars": list(_PROVIDER_ENV_VARS.get(c.label, ())),
                }
                for c in report.llm_checks
            ],
            "exporters": [
                {
                    "name": c.label,
                    "ok": c.ok,
                    "detail": c.detail,
                    "suggestion": c.suggestion,
                }
                for c in report.exporter_checks
            ],
            "sandbox": [
                {
                    "name": c.label,
                    "ok": c.ok,
                    "detail": c.detail,
                    "suggestion": c.suggestion,
                }
                for c in report.sandbox_checks
            ],
            "security": [
                {"label": c.label, "ok": c.ok, "detail": c.detail, "suggestion": c.suggestion}
                for c in report.security_checks
            ],
            "agents_ready": report.agents_ready,
            "llm_providers_ready": report.llm_providers_ready,
            "exporters_ready": report.exporters_ready,
            "sandbox_ready": report.sandbox_ready,
            "security_warnings": report.security_warnings,
        }
        print(json.dumps(data, indent=2))
        return 0 if report.agents_ready > 0 else 1

    # Human-readable doctor output goes to stdout: the report IS the
    # deliverable for this command (there is no separate "data" stream
    # the way ``belt run`` or ``belt eval`` have), so users expect
    # ``belt doctor > out.txt`` to capture what they saw on screen. The
    # ``--json`` branch above already prints to stdout via ``print``;
    # this brings the non-JSON path into the same shape.
    report = run_doctor_live(console=Console())
    return 0 if report.agents_ready > 0 else 1
