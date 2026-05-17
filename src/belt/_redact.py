# (c) JFrog Ltd. (2026)

"""Single source of truth for secret redaction across belt.

Every site that ever puts user-supplied strings into persisted artifacts
(``run_meta.json``, ``benchmark-card.json``, the rendered Markdown card,
agent runtime sidecars, the env-var snapshot) routes through the helpers
in this module. Concentrating both the secret-shape regex *and* the
parsing of ``key=value`` strings here makes a class of bugs structurally
impossible: callers no longer split on ``=`` themselves, so a future
``-Xkey=value`` style cannot diverge from a ``-X key=value`` style.

Naming convention
~~~~~~~~~~~~~~~~~

Two flavours of helper, distinguished by name on purpose:

- ``scrub_*`` - **pure transform**. Takes arbitrary input, returns the
  scrubbed version. No allow-list, no policy, no ambient context. Easy
  to test, easy to compose. (``scrub_kv_string``, ``scrub_kv_list``,
  ``scrub_dict``, ``scrub_argv``, ``scrub_url``.)
- ``safe_*`` - **curated boundary snapshot**. Combines a deny-list,
  allow-list, and one or more ``scrub_*`` primitives to produce a value
  that is safe to persist as-is. (``safe_environ``, ``safe_agent_args``.)

If a future helper does pure transformation, name it ``scrub_*``. If it
applies policy on top of a transformation, name it ``safe_*``.

Public-internal surface (all underscore-prefixed module; not part of the
distribution's public API):

- :func:`is_secret_name` - boolean test; the only place the regex runs.
- :func:`scrub_kv_string` - rewrites ``"k=v"`` to ``"k=<redacted>"`` if
  ``k`` is secret-shaped; non-``=`` strings are returned unchanged.
- :func:`scrub_kv_list` - element-wise :func:`scrub_kv_string`.
- :func:`scrub_dict` - redacts dict values whose key (or declared
  ``env_var``) is secret-shaped; replaces with ``"<set>"``.
- :func:`scrub_argv` - sanitises a full argv list, handling both the
  combined (``-Xk=v``) and separated (``-X k=v``) forms of ``-X`` /
  ``--agent-arg`` flags.
- :func:`scrub_url` - reduces a URL value to ``scheme://host[:port]``,
  stripping userinfo, path, and query.
- :func:`safe_environ` - returns the allow-listed env-var snapshot used
  by ``run_meta.json`` (CI markers + ``BELT_*`` knobs only,
  sourced from :data:`belt.envvars.PUBLIC_ALLOW`). Names matching
  the secret regex degrade to ``"<set>"``; ``*_BASE_URL`` values are
  reduced via :func:`scrub_url`. The deny-list is checked *after* the
  allow-list so a future allow-list mistake (e.g. someone adding
  ``CURSOR_API_KEY`` to ``PUBLIC_ALLOW``) cannot exfiltrate the secret.
- :func:`safe_agent_args` - thin wrapper over :func:`scrub_dict` that
  extracts ``(name, env_var)`` pairs from an agent's ``cli_options()``
  metadata. Lives here (not in ``benchmark_card``) because the agent
  runner writes the agent-args sidecar at runner-phase time; coupling
  it to the aggregator-phase ``benchmark_card`` package would violate
  the phase-isolation principle.
"""

from __future__ import annotations

import os
import re
from typing import Iterable
from urllib.parse import urlsplit

from belt import envvars

# ── Regex constants ──

# Names matching this regex are *never* persisted verbatim, regardless of
# any allow-list. The single canonical definition; every redactor in the
# codebase is required to test against it (not against an inline copy).
_SECRET_NAME_RE: re.Pattern[str] = re.compile(
    r"(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|BEARER|SESSION)",
    re.IGNORECASE,
)

# Names whose value is a URL we want to redact to scheme+host.
_URL_NAME_RE: re.Pattern[str] = re.compile(r"_BASE_URL$")

# Stable replacement strings - exposed as module constants so tests and
# callers compare against a single source of truth.
REDACTED: str = "<redacted>"
PRESENT: str = "<set>"

# ── Allow-lists for the env-var snapshot ──

# Operational CI markers: GitHub Actions and generic runners.
_CI_ALLOW: frozenset[str] = frozenset(
    {
        "CI",
        "GITHUB_ACTIONS",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_NUMBER",
        "GITHUB_REPOSITORY",
        "GITHUB_REF",
        "GITHUB_REF_NAME",
        "GITHUB_SHA",
        "GITHUB_WORKFLOW",
        "GITHUB_JOB",
        "GITHUB_EVENT_NAME",
        "RUNNER_OS",
        "RUNNER_ARCH",
        "RUNNER_NAME",
    }
)

# Documented BELT_* knobs are sourced live from
# ``envvars.PUBLIC_ALLOW`` inside :func:`safe_environ` (not bound at
# module import) so a test that monkeypatches ``envvars.PUBLIC_ALLOW``
# is observed without having to rebind anything here. Caching a local
# alias would create a second place to keep in sync and a drift risk;
# we read the canonical source on every call instead.


# ── Primitive predicates ──


def is_secret_name(name: str) -> bool:
    """Return True if ``name`` matches the canonical secret-name regex."""
    return bool(_SECRET_NAME_RE.search(name))


# ── String / list redactors ──


def scrub_kv_string(item: str, *, mark: str = REDACTED) -> str:
    """Rewrite a single ``"key=value"`` string with secret-shaped key redacted.

    Non-string inputs and strings without ``=`` are returned unchanged.
    Splits on the *first* ``=`` so values containing ``=`` are preserved
    intact (e.g. ``"endpoint=https://x?token=y"`` keeps the value, but
    ``"endpoint"`` is not secret-shaped so the value also is not
    rewritten - that case is covered by :func:`scrub_url` for URL-typed
    keys).
    """
    if not isinstance(item, str) or "=" not in item:
        return item
    key, _, _val = item.partition("=")
    if is_secret_name(key):
        return f"{key}={mark}"
    return item


def scrub_kv_list(items: Iterable[str], *, mark: str = REDACTED) -> list[str]:
    """Element-wise :func:`scrub_kv_string` over an iterable of strings."""
    return [scrub_kv_string(x, mark=mark) for x in items]


def scrub_dict(
    d: dict[str, str],
    *,
    env_var_by_name: dict[str, str] | None = None,
    mark: str = PRESENT,
) -> dict[str, str]:
    """Return a copy of ``d`` with secret-named entries replaced by ``mark``.

    Two checks:

    1. The dict key itself is secret-shaped (e.g. ``api_key``).
    2. The key's declared ``env_var`` (looked up in ``env_var_by_name``)
       is secret-shaped (e.g. ``ANTHROPIC_API_KEY``).

    ``env_var_by_name`` lets agent-arg captures benefit from per-option
    metadata (``cli_options()`` ``env_var=`` declarations) without the
    redactor having to know about the agent abstraction directly.
    """
    out: dict[str, str] = {}
    lookup = env_var_by_name or {}
    for k, v in d.items():
        if is_secret_name(k) or is_secret_name(lookup.get(k, "")):
            out[k] = mark
        else:
            out[k] = v
    return out


# ── argv redactor (the foot-gun) ──


def scrub_argv(
    argv: list[str],
    *,
    kv_flags: tuple[str, ...] = ("-X", "--agent-arg"),
    mark: str = REDACTED,
) -> list[str]:
    """Redact ``key=value`` payloads of selected flags in an argv list.

    Handles every shape argparse accepts for a short ``-X`` option plus
    its long alias ``--agent-arg``:

    - separated short  : ``["-X", "key=value"]``
    - combined short   : ``["-Xkey=value"]``  (no whitespace)
    - separated long   : ``["--agent-arg", "key=value"]``
    - long with equals : ``["--agent-arg=key=value"]``

    A value whose key matches :func:`is_secret_name` is rewritten to
    ``"<flag><sep>key=<mark>"``. Non-secret pairs and unrelated argv
    entries are returned unchanged. The function never raises; an
    unparseable element is returned verbatim.

    The single point of parsing is intentional: hand-rolling a second
    ``partition("=")`` at a call site is the foot-gun this function
    exists to remove (the combined ``-Xkey=value`` form, in particular,
    silently bypasses ad-hoc redactors that count ``=`` characters).
    Callers must not parse argv themselves; they must call this
    function.
    """
    short_flags = tuple(f for f in kv_flags if not f.startswith("--"))
    long_flags = tuple(f for f in kv_flags if f.startswith("--"))

    out: list[str] = []
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            out.append(scrub_kv_string(arg, mark=mark))
            skip_next = False
            continue

        # Separated forms: ``-X k=v`` / ``--agent-arg k=v``.
        if arg in kv_flags and i + 1 < len(argv):
            out.append(arg)
            skip_next = True
            continue

        # Combined short form: ``-Xkey=value``. ``arg[2:]`` is the
        # ``key=value`` payload.
        matched_short = next(
            (f for f in short_flags if arg.startswith(f) and len(arg) > len(f) and arg[len(f)] != "="),
            None,
        )
        if matched_short is not None:
            payload = arg[len(matched_short) :]
            out.append(f"{matched_short}{scrub_kv_string(payload, mark=mark)}")
            continue

        # Long form with equals: ``--agent-arg=key=value``. Splits on
        # the *first* ``=`` so the payload retains its own ``=``.
        matched_long = next(
            (f for f in long_flags if arg.startswith(f + "=")),
            None,
        )
        if matched_long is not None:
            payload = arg[len(matched_long) + 1 :]
            out.append(f"{matched_long}={scrub_kv_string(payload, mark=mark)}")
            continue

        out.append(arg)
    return out


# ── URL redactor ──


def scrub_url(value: str) -> str:
    """Reduce a URL string to ``scheme://host[:port]``.

    Strips userinfo, path, query, and fragment. Returns ``"<set>"`` for
    unparseable input rather than echoing it back verbatim - the goal is
    to attest "this URL was set" without surfacing any of its potentially
    sensitive segments.
    """
    try:
        parts = urlsplit(value.strip())
    except ValueError:
        return PRESENT
    if not parts.scheme or not parts.hostname:
        return PRESENT
    netloc = parts.hostname
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return f"{parts.scheme}://{netloc}"


# ── Env-var snapshot ──


def safe_environ(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Return the allow-listed env-var snapshot for ``run_meta.json``.

    Only names in the union of :data:`_CI_ALLOW` and
    :data:`belt.envvars.PUBLIC_ALLOW` are considered. Within that set:

    - secret-named values become ``"<set>"`` (deny-list applied
      *after* the allow-list);
    - ``*_BASE_URL`` values are reduced via :func:`scrub_url`;
    - everything else is recorded verbatim.

    The ``PUBLIC_ALLOW`` set is read live from :mod:`belt.envvars`
    on every call; a test that monkeypatches that attribute is observed
    immediately without needing to rebind anything in this module.

    Args:
        environ: Override for ``os.environ`` (mainly for tests).
    """
    src = os.environ if environ is None else environ
    out: dict[str, str] = {}
    allow = _CI_ALLOW | envvars.PUBLIC_ALLOW
    for name in sorted(allow):
        if name not in src:
            continue
        raw = src[name]
        if is_secret_name(name):
            out[name] = PRESENT
            continue
        if _URL_NAME_RE.search(name):
            out[name] = scrub_url(raw)
            continue
        out[name] = raw
    return out


def safe_agent_args(
    agent_args: dict[str, str],
    cli_options: list[object] | None = None,
) -> dict[str, str]:
    """Redact secret-looking values in an agent-arg dict.

    Two checks, both defensive:

    1. Option name itself matches the secret regex (e.g. an agent that
       declared ``api_key`` as an option) - replaced with ``"<set>"``.
    2. Option's declared ``env_var`` matches the secret regex (e.g.
       ``env_var="ANTHROPIC_API_KEY"``) - same replacement.

    ``cli_options`` should be ``agent_cls.cli_options()`` from the
    agent being captured. ``None`` means "no per-option metadata
    available"; the function still applies check (1).

    Lives in :mod:`belt._redact` (not in ``benchmark_card``)
    because the runner-phase orchestrator persists agent args to the
    ``_runtime_info.json`` sidecar; importing from the aggregator-phase
    ``benchmark_card`` would cross the phase-isolation boundary.
    """
    env_var_by_name: dict[str, str] = {}
    if cli_options:
        for opt in cli_options:
            env_var = getattr(opt, "env_var", None)
            name = getattr(opt, "name", None)
            if env_var and name:
                env_var_by_name[name] = env_var
    return scrub_dict(agent_args, env_var_by_name=env_var_by_name)


__all__ = [
    "PRESENT",
    "REDACTED",
    "is_secret_name",
    "safe_agent_args",
    "safe_environ",
    "scrub_argv",
    "scrub_dict",
    "scrub_kv_list",
    "scrub_kv_string",
    "scrub_url",
]
