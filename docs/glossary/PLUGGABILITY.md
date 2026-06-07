# Pluggability

How consumers extend belt. The framework ships three extension
points; each is implemented by subclassing a base class and registering
under a Python entry-point group. This document is the single source of
truth for the plugin contract - discovery, public API, schema boundary,
and per-extension authoring.

## 1. Architecture

belt is a **standalone framework** published to PyPI. Vendor-specific
agents, scorers, and exporters live in separate plugin packages that
register via Python entry points - they are never bundled in the core.

```text
┌──────────────────────────────────────┐
│           belt (core)           │
│                                      │
│  cli, runner, scoring, scenarios     │
│  built-in agents (see registry.py)   │
│                                      │
│  built-in scorers: rules, llm        │
│  built-in exporters: csv, jsonl,     │
│    junit, markdown                   │
│                                      │
│  entry-point discovery:              │
│    belt.agents    → agents      │
│    belt.scorers   → scorers     │
│    belt.exporters → exporters   │
└──────────┬───────────────────────────┘
           │ pip install
           │
┌──────────▼───────────────────────────┐
│   belt-my-agent (plugin)        │
│                                      │
│  my_agent/agent.py   - MyAgentAdapter│
│  my_agent/scorer.py  - MyScorer      │
│  agent-specific scenarios            │
│                                      │
│  pyproject.toml:                     │
│    dependencies = ["belt"]      │
│    [entry-points."belt.agents"] │
│    my-agent = "my_agent:MyAgentAdapter"  │
│    [entry-points."belt.scorers"]│
│    my-scorer = "my_agent:MyScorer"   │
└──────────────────────────────────────┘
```

## 2. Public API boundary

Plugins import from the top-level package only:

```python
from belt import BaseAgentAdapter, BaseScorer, BaseExporter, ScenarioScore  # etc.
```

The canonical list lives in
[`belt._public_api.PUBLIC_API`](../../src/belt/_public_api.py)
(re-exported via `belt/__init__.py`). Adding a symbol there is the
*only* way to expand the public surface; removing or renaming is a
breaking change. `scripts/check_design.py::check_plugin_public_api_only`
fails the build if any code under `plugins/` or `examples/custom-agent/`
imports from a non-public path.

## 3. Schema boundary

The core `GroupConfig` (`_config.json`) and `Scenario` (`<name>.json`)
schemas declare **only fields that core code reads**. Plugin-specific
configuration - for example a plugin's backend URL or a credentials file
path, or any other extra key a plugin's agent needs - does not belong in
core.

### 3.1. What core does

Pydantic models in `src/belt/scenario.py` and `src/belt/entities.py`
accept extra keys on three surfaces, with two different retrieval semantics:

- `TurnExpectation` and `TurnOutput` use `extra="allow"`. Unknown keys
  are accepted and surfaced via Pydantic's `model_extra` dict; plugins
  read them back through `expect.model_extra["..."]` /
  `output.model_extra["..."]`.
- `GroupConfig` uses Pydantic's default `extra="ignore"`. Unknown keys
  in `_config.json` validate silently but are **dropped** from the
  model - they do not appear in `model_extra`. Plugins that need
  per-group config must re-read the raw JSON in their own setup hook
  (see §3.2).

Built-in scorers and the LLM judge prompt only consume the explicitly
declared fields, so a scenario authored for a plugin runs harmlessly
against agents that lack it.

### 3.2. What plugins do

A plugin that needs to read or assert on its own data has three options:

| Approach | When to use |
|---|---|
| Read `expect.model_extra["my_key"]` / `output.model_extra["my_field"]` from a custom scorer registered under `belt.scorers` | Plugin-specific assertions in scenario JSON (e.g. multi-agent handoff counts, conversational suggestions) |
| Read the raw `_config.json` from disk in the plugin's agent setup (`json.loads((group_dir / "_config.json").read_text())`) | Group-level config, one file, one extra field |
| Pass values via `-X key=value` (`agent_args`) or scenario-level `scenario_options` | Per-run overrides, no file change |

The plugin's agent receives the typed `GroupConfig` via `AgentConfig`.
For extension fields on `TurnOutput`, the plugin's `fetch_results()`
simply passes them as kwargs (`TurnOutput(raw_cli=..., my_field=value)`) -
they land in `model_extra` and the plugin's scorer reads them back. For
`TurnExpectation` extras, the plugin's scorer iterates
`expect.model_extra` and emits its own `CheckEntry` rows into a
`RulesPayload`.

**The rule:** a field belongs in core only when at least one core code
path (runner, scoring, scenario loader, manifest) reads it. Otherwise
it lives in the plugin.

### 3.3. Strict-config key registration

The default permissive load is a typo trap (see
[SCENARIOS.md → Strict config validation](SCENARIOS.md#10-strict-config-validation---strict-config)).
Users opt into `--strict-config` to fail-fast on unknown keys, but
without coordination they would also reject every legitimate plugin
extra. Plugins close that gap by registering each extension key they
expect at import time:

```python
from belt import register_plugin_scenario_key
from belt.scenario import TurnExpectation, GroupConfig

# scenario JSON: "expect": { "max_handoffs": 3 }
register_plugin_scenario_key(TurnExpectation, "max_handoffs")

# group config: { "agent": "...", "mcp_servers": [...] }
register_plugin_scenario_key(GroupConfig, "mcp_servers")
```

| Constraint | Rationale |
|---|---|
| Process-local registry | No persistence, no IPC, no env-var registration. A plugin not imported is invisible to the validator - that is the correct fail-closed default. |
| Reserved framework names rejected | `name`, `description`, `tags`, `turns`, `agent`, `schema_version` are framework-controlled; a plugin that registered them would shadow core validation. |
| Lowercase ASCII identifier shape | Plugin keys must look like declared field names (`[a-z][a-z0-9_]*` with optional `.` / `-` separators, ≤ 64 chars) so the registry surface is indistinguishable from core for downstream tooling. |
| Idempotent | Re-registering the same `(model, key)` is a no-op. Safe to call from `__init__.py` on every import. |
| Per-model scope | A key registered on `TurnExpectation` does not leak to `GroupConfig`. Register once per surface where the key may appear. |

The same rule that governs core schema fields applies here: register
a key only when the plugin's scorer or agent actually reads it. A
key registered without a consumer is dead weight that misleads
scenario authors and dilutes the validator's signal.

## 4. Discovery

Agents, scorers, and exporters follow the same three-tier discovery
pattern:

| Step | Agents (`agent/registry.py`) | Scorers (`scorer/registry.py`) | Exporters (`exporter/registry.py`) |
|---|---|---|---|
| 1. Built-in | `_AGENT_REGISTRY` dict | `_SCORER_REGISTRY` dict | `_EXPORTER_REGISTRY` dict |
| 2. Entry points | `belt.agents` group | `belt.scorers` group | `belt.exporters` group |
| 3. Direct import | `--agent mypackage.MyAgentAdapter` | `--modes mypackage:MyScorer` | `--export mypackage:MyExporter:path` |

The entry-point group strings are defined as constants in
`belt.constants` (`ENTRY_POINT_GROUP_AGENTS`,
`ENTRY_POINT_GROUP_SCORERS`, `ENTRY_POINT_GROUP_EXPORTERS`); registry
code imports the constant rather than re-spelling the literal
([principle 9](ARCHITECTURE.md#principle-9-one-source-of-truth-for-public-names)).

No code changes in the core are needed when a new plugin is installed -
entry-point discovery is automatic. Tier 3 (direct dotted import) is
gated behind the `--allow-arbitrary-{agent,scorer,exporter}` flags
(default deny - see [CONFIGURATION.md → Security toggles](CONFIGURATION.md#4-behaviour-gates----allow----default-deny)).

## 5. Using a plugin

```bash
pip install belt my-agent-plugin            # or `pip install -e ./my-agent-plugin/` for editable dev
belt eval examples/scenarios/ --agent my-agent
```

First-party plugin layout (under `plugins/`) is documented in
[CONTRIBUTING.md → First-party plugins](../../CONTRIBUTING.md#first-party-plugins-plugins).

---

## Authoring

The remaining sections are step-by-step guides for each extension type.
A minimal reference agent ships under `examples/custom-agent/`.

## 6. Authoring an agent

Subclass [`BaseAgentAdapter`](../../src/belt/agent/base.py); the
class docstrings are the canonical reference for required methods,
optional overrides, and field semantics. Mirror an existing built-in
(`agent/opencode.py`, `agent/claude_code.py`) - this section covers
only the non-obvious contracts that the base class can't enforce on
its own.

### 6.1. Register and verify

Built-in: add to `_AGENT_REGISTRY` in `agent/registry.py` and a column
to [AGENT-FEATURES.md](AGENT-FEATURES.md). Plugin: declare an
`belt.agents` entry point in `pyproject.toml`. Then:

```bash
belt agent list                                            # shows your agent
belt eval examples/scenarios/experience/<your-group>       # smoke
```

Add example scenarios under `examples/scenarios/{showcase,experience}/`
(see [SCENARIOS.md](SCENARIOS.md)) and unit tests under
`tests/agent/test_<agent>.py`.

### 6.2. `check_available()` contract

The runner's startup gate and `belt doctor` both call
`check_available()`. To stay reliable across upstream CLI version
churn:

1. **MUST** verify the agent binary exists and is invokable.
2. **MUST** complete in under 2s on the happy path.
3. **MUST NOT** invoke a model or make any credit-consuming network call.
4. **SHOULD NOT** parse human-readable status strings (`"Not logged in"`,
   `"Authentication required"`) - they change between releases and
   cause false negatives that block legitimate runs.
5. **MAY** declare `CREDENTIAL_ENV` / `CREDENTIAL_PATHS` so `doctor`
   surfaces positive auth signals (existence checks only - never read
   contents, never gate execution; the framework adds a "presence only
   - not verified" hedge automatically). There is intentionally no
   negative `auth: not authenticated` line: we cannot reliably
   distinguish "no auth" from "auth via mechanism we don't catalogue"
   without violating rule 3.

Authentication failures surface **at eval time** as
`TurnOutput.has_error=True` with `error_type` set - not as
`check_available` failures. `error_type` is the canonical taxonomy in
`belt.entities.ERROR_TYPES` (see
[OUTCOMES.md → Agent runtime errors](OUTCOMES.md#3-agent-runtime-errors-error_type));
agents that read a vendor label from CLI NDJSON MUST pipe it through
`belt.agent.error_types.normalize_error_type`. The parity test
requires every new agent to ship
`tests/agent/fixtures/auth_failure/<name>.ndjson` and pass
`test_classifies_auth_failure`.

For binary discovery use `resolve_binary` from `agent/base.py` (PATH +
declared `EXTRA_PATHS` like `~/.local/bin`), not bare `shutil.which`.
For benchmark-card identity, override `runtime_info()` and use
`_capture_cli_version()` (5s timeout, never raises).

### 6.3. Argv safety

The `message` argument is untrusted scenario content (may come from a
fork PR). A CLI that reparses it as an option can have flags toggled by
the scenario author. There are exactly two safe argv shapes - pick one:

**Shape A - positional message, terminated with `--`** (used by
`claude-code`, `codex`, `cursor`, `opencode`):

```python
cmd = ["my-agent", "--output-format", "stream-json", *self.filter_flags(flags), "--", message]
```

**Shape B - message as the value of a flag** (used by `gemini` /
`goose` / `copilot`):

```python
cmd = ["my-agent", "--mode", "stream", *self.filter_flags(flags), "-p", message]
```

Do **not** add `--` between the flag and its value in Shape B
(`["-p", "--", message]`) - that re-exposes `message` as a free
positional, recreating the vulnerability.

`tests/test_security.py::TestAgentArgvSafetyAutoDiscovery` enforces one
of the two shapes for every agent in `_AGENT_REGISTRY`. Third-party
agents are out of test scope but **must** follow the same convention.

### 6.4. Parity contract

`tests/agent/test_agent_parity.py` parametrises over `_AGENT_REGISTRY`
and is the source of truth for agent behaviour - if the test disagrees
with [AGENT-FEATURES.md](AGENT-FEATURES.md), the matrix is wrong.

| Enforced invariant | Failure mode it catches |
|---|---|
| `config.workspace_dir` is forwarded to `subprocess.Popen(cwd=…)` | Agent silently runs in harness cwd instead of the isolated worktree; editing scenarios then diff against the wrong tree |
| `supported_output_fields()` ⊆ `AGENT_SPECIFIC_FIELDS` | Misspelled field name silently disables every expectation that references it |
| Argv safety (one of the two shapes in §6.3) | Scenario `message` reparsed as a flag |
| Auth-failure classification on the fixture under `tests/agent/fixtures/auth_failure/` | Vendor error string not mapped to canonical `authentication_failed` |
| Constructor rejects unknown kwargs (see §6.5) | Runner's `cli_options` validation bypassed |

Optional surfaces (`CREDENTIAL_ENV`, `parse_stream_event`,
`cli_options`, `denied_flags`, `scoring_strategy`, `health_check`) are
intentionally not enforced - inherit the default if you don't need it.

### 6.5. Constructor ↔ `cli_options()` parity

`__init__` declares only the kwargs it uses - kw-only, typed, with
defaults. **Never** accept `**kwargs`: the runner validates `-X key=value`
against `cli_options()` and a swallowing `**kwargs` makes that
validation a lie.

The mapping is bidirectional:

- Every `cli_options()` entry MUST be a kwarg `__init__` accepts -
  otherwise the runner crashes with `AgentArgError` whenever the
  option's `env_var` is set in the user's environment.
- A parameterless agent MUST return `[]` from `cli_options()` and pin
  policy via scenario `flags` instead.

Both directions are gated by `test_agent_parity.py`.

### 6.6. Gotchas

- **stdin hanging** - some CLIs (OpenCode) hang on a stdin pipe; pass
  `stdin=subprocess.DEVNULL`.
- **stderr deadlock** - drain stderr in a background thread via
  `_drain_stderr` or the 64KB pipe buffer fills.
- **Live streaming** - `subprocess.Popen` (not `run`); write each
  stdout line to `self._stream_sink` when set, otherwise
  `--progress live` is silent.
- **`parse_stream_event` return shape** - `(icon, summary)` where
  `summary` is **plain text**, no Rich markup. The framework
  `rich_safe`-escapes at render time. Surface numeric data through
  `TurnOutput` fields (e.g. `cost_usd`) and let the framework style
  it; see [CLI.md → Progress modes](CLI.md#5-progress-modes---progress).

## 7. Authoring a scorer

Reach for a scorer when neither built-in (`rules`, `llm`) can express
the check you need - structured-output validation, cross-turn linkage,
domain metrics. *Not* for post-run output (exporter) or a new agent CLI
(agent).

Subclass [`BaseScorer`](../../src/belt/scorer/base.py); base
docstrings cover `name` / `is_available` / `score` signatures. The
non-obvious contract is the **payload**.

### 7.1. Register it

Built-in: add to `_SCORER_REGISTRY` in `scorer/registry.py`. Plugin:
declare an `belt.scorers` entry point in `pyproject.toml`.

### 7.2. Payload contract

`ScorerResult.data` MUST be a typed `pydantic.BaseModel` (not a raw
`dict`) with
`schema_version: Literal["<scorer>.v<N>"] = "<scorer>.v<N>"` as the
discriminator - the **only** place that string lives; registration and
dispatch read it back through `model_fields`. Built-ins
(`belt.scorer.payloads`): `RulesPayload` (`rules.v1`),
`LLMPayload` (`llm.v1`), and `PerTurnLLMPayload` (`per_turn_llm.v1` -
aggregates one `TurnVerdict` per scenario turn for per-turn LLM
judging; see [SCORING.md → §2.10](SCORING.md#210-per-turn-llm-judging)).

The 5 contract rules every payload must follow:

1. Pydantic model with the `Literal` discriminator above. Convention
   `"<scorer>.v<int>"` lets you ship a v2 without breaking v1.
2. **Register a payload iterator at module-import time** so every
   downstream reader (exporters, aggregator stats, `view`, `compare`)
   walks it uniformly:

   ```python
   from belt.scorer.payloads import DimensionFeedback, register_payload_type

   def _iter_my_scorer(scorer_name, payload):
       for dim, verdict in payload.dimensions.items():
           yield DimensionFeedback(scorer_name=scorer_name, dimension=dim,
                                   score=verdict.score, comment=verdict.reasoning,
                                   raw=verdict.model_dump(mode="json"))

   register_payload_type(MyScorerPayload, _iter_my_scorer)
   ```

3. Consumers walk payloads via `iter_dimension_feedback(score)` (or
   `isinstance` on the typed classes), **never** by indexing
   `score.scores[...]` as a dict. Enforced by the design check for
   plugin code. LLM-shaped consumers (multi-judge non-consensus,
   per-turn judging) should additionally use
   `iter_llm_payloads(score)` / `iter_llm_verdicts(payload)` to walk
   both `LLMPayload` and `PerTurnLLMPayload` uniformly - hard-coding
   `score.scores["llm"]` silently drops every renamed multi-judge
   key and every per-turn payload.
4. Numeric scores normalise to `0.0-1.0` or `None`. For categorical
   scorers use `belt.scorer.payloads.level_to_score` to map
   `high`/`medium`/`low` consistently with the built-in LLM scorer.
5. Unregistered `schema_version` raises `ValueError` at parse and
   iteration time - the framework refuses to guess at unknown shapes.

## 8. Authoring a custom LLM judge backend

`BaseJudgeBackend` is in `belt._public_api.PUBLIC_API` and therefore a stable
public extension point. Subclass it when you need a provider that the built-in
backends (`OpenAIBackend`, `AzureBackend`, `AnthropicBackend`, `OllamaBackend`)
do not support, or when a gateway in front of an existing provider requires
non-standard auth.

The full abstract interface is documented in the class docstring in
`belt.scorer.llm.backend`. Two hooks are worth calling out here because they
interact with the dispatcher and are the most common customisation points:

### 8.1. `auth_retry_headers(original_headers) → dict | None`

Called by `LLMScorer._call_api` (and the preflight helper) when the first
request returns 401 or 403. Return a replacement header dict to retry once with
that dict instead. Return `None` (the default) to propagate the error
immediately without retrying.

Typical use: a corporate WAF that rejects one of two interchangeable auth
header styles from certain egress IPs (e.g. rejecting `x-api-key` but
accepting `Authorization: Bearer`).

### 8.2. `record_auth_retry_success() → None`

Called by the dispatcher immediately after an `auth_retry_headers` retry
succeeds. The default no-op is sufficient for one-shot retries. Override when
the backend caches the working auth style so subsequent requests emit it
directly, avoiding a 401/403 round-trip on every call in a multi-trial run.

```python
from belt import BaseJudgeBackend

class MyGatewayBackend(BaseJudgeBackend):
    def __init__(self):
        self._use_bearer = False

    def auth_retry_headers(self, original_headers):
        if "x-api-key" not in original_headers:
            return None
        retry = {k: v for k, v in original_headers.items() if k != "x-api-key"}
        retry["Authorization"] = f"Bearer {original_headers['x-api-key']}"
        return retry

    def record_auth_retry_success(self):
        self._use_bearer = True

    def build_request(self, config, messages, schema):
        headers = {...}
        if self._use_bearer:
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["x-api-key"] = api_key
        return url, headers, body
```

Both methods are called unconditionally on every `BaseJudgeBackend` instance —
no `isinstance` or `hasattr` guard is needed. The dispatcher calls
`auth_retry_headers` only on 401/403 and `record_auth_retry_success` only when
the retry actually succeeded.

## 9. Authoring an exporter

Reach for an exporter to land a **completed run** somewhere belt
doesn't write today (CSV, JUnit XML, vendor SaaS, Slack-friendly
Markdown). *Not* for live per-span streaming (use `--progress` /
`belt watch`), a new scoring dimension (scorer), or a new CLI
(agent). Exporters consume `results.json` + `score.json` files and
never re-execute the agent.

Subclass [`BaseExporter`](../../src/belt/exporter/base.py) +
read [`ExportContext`](../../src/belt/exporter/entities.py) for
the typed inputs (`results`, trial-expanded `scores`, `run_dir`,
parsed `benchmark_card`). Signature is closed to modification
([principle 3](ARCHITECTURE.md#principle-3-extend-via-abstractions-never-modify-the-framework-core)).

### 9.1. Register it

Built-in: register in `exporter/registry.py` AND declare an
`belt.exporters` entry point in core's `pyproject.toml`. Plugin:
entry point only. List is `belt doctor` under Exporters - this
guide deliberately doesn't enumerate.

### 9.2. Non-obvious contracts

- **Failure isolation.** `export()` raising surfaces as a one-line
  typed error and the next exporter still runs. Exit code is non-zero
  only when **every** requested exporter failed.
- **`--trials N` expansion.** `ExportContext.scores` has one entry per
  `__trial_K` outcome dir. Per-scenario formats (JUnit, default CSV)
  MUST call `belt.exporter.helpers.collapse_trials(scores)`;
  per-trial streamers iterate `ctx.scores` directly. Reliability
  summaries live on `ctx.results.reliability`.
- **Per-call config** rides on the `options` dict, populated from
  `--export-config` YAML and validated by `ExporterFile`
  (`extra='forbid'`).
- **Env-var namespace.** Plugins SHOULD prefix their package name
  (`MYEXPORTER_*`); plugins MUST NOT introduce `BELT_*`
  literals - the env-var registry test catches that drift in core.
- **Network calls.** Cap retries (the harness won't retry); `_redact.py`
  scrubs core provenance but does not inspect plugin output, so don't
  log credentials yourself.

## 10. Authoring a sandbox provider

Reach for a sandbox provider when neither built-in (`host`,
`docker`) can express the isolation model you need - Firecracker
microVMs, gVisor, Kata, Kubernetes Jobs, or a custom container runtime.
*Not* for adding hardening flags to the docker provider (open a PR
against `src/belt/runner/sandbox/docker.py`) or for environment-level
policy that lives in `SandboxProfile`.

Subclass [`BaseSandboxProvider`](../../src/belt/runner/sandbox/base.py);
base docstrings cover the four method contracts and the
`SandboxContext` / `SandboxHandle` entities. The provider is loaded via
the `belt.sandbox_providers` entry-point group; selection is via
`--sandbox NAME` or `BELT_SANDBOX_PROVIDER`. See
[CONFIGURATION.md → `SandboxProfile`](CONFIGURATION.md#3-environment-variables)
for the per-group schema your provider receives.

### 10.1. Register it

Built-in: register in `src/belt/runner/sandbox/registry.py`. Plugin:
declare a `belt.sandbox_providers` entry point in `pyproject.toml`:

```toml
[project.entry-points."belt.sandbox_providers"]
firecracker = "my_plugin:FirecrackerSandboxProvider"
```

`belt doctor` then enumerates the provider under "Sandbox" alongside
`host` and `docker`.

### 10.2. The four method contracts

```python
from belt import BaseSandboxProvider, SandboxContext, SandboxHandle
from belt.scenario import SandboxProfile

class FirecrackerSandboxProvider(BaseSandboxProvider):
    def validate_profile(self, profile: SandboxProfile, ctx: SandboxContext) -> None:
        # Raise SandboxConfigError if you cannot enforce ``profile`` (e.g.
        # this provider does not implement ``network_policy='none'``).
        # Default: accept everything. Override only when an isolation
        # field is unenforceable.
        ...

    def setup(self, profile: SandboxProfile, ctx: SandboxContext) -> SandboxHandle:
        # Called once per scenario, before any agent subprocess spawns.
        # Build the microVM / pod / container; return a handle carrying
        # the profile, the SandboxContext, and any provider-private state.
        ...

    def wrap(self, handle, *, cmd, cwd, env) -> tuple[list[str], str | None, dict[str, str]]:
        # Pure-policy rewrite: rewrite ``(cmd, cwd, env)`` so subprocess.Popen
        # lands inside the sandbox. Called once per agent subprocess spawn.
        # ``env`` must already be filtered to the union of
        # ``ctx.agent_required_env`` and ``profile.env_passthrough``.
        ...

    def teardown(self, handle: SandboxHandle) -> None:
        # Release resources. Idempotent. Errors are logged and swallowed.
        ...
```

### 10.3. Hard contracts

| Invariant | Why |
|---|---|
| `validate_profile()` is cheap (no I/O, no subprocess) and side-effect-free | The runner calls it inside the per-scenario try/except; an exception aborts only the offending scenario |
| `validate_profile()` raises `SandboxConfigError` for any `SandboxProfile` field the provider cannot enforce | Silently running with weaker isolation than the profile declared is the exact failure mode this typed error exists to prevent |
| `setup()` raises a typed `BeltError` subclass on misconfiguration; nothing else | The runner surfaces one actionable line per failure |
| `teardown()` is idempotent | The runner always calls it from a `finally` block, even after a partial `setup()` failure |
| `wrap()` filters `env` to `ctx.agent_required_env ∪ profile.env_passthrough` | Anything else leaks secrets the scenario did not opt into into the sandbox |
| `wrap()` returns a triple - never mutates the inputs | The spawner forwards it verbatim to `Popen`; mutating the inputs causes subtle cross-scenario state bleed |

### 10.4. Test gates

Two checks the framework enforces uniformly for every registered
provider:

- **Profile-enforcement parity.** If the provider cannot enforce a
  given `SandboxProfile` field, `validate_profile()` MUST raise -
  it is never acceptable to silently relax the request.
- **Agent denylist forwarding.** `BaseAgentAdapter.denied_flags()`
  (the per-agent list of "skip all permission" flags - see §6) is
  read by the runner *before* `wrap()`; the sandbox provider does
  not need to enforce it. Plugins that ship their own agents alongside
  a sandbox provider still implement `denied_flags()` on the agent,
  not on the provider. Tested by `tests/test_security.py::TestAgentDeniedFlagsDefaults`.

For container-style providers, mirror the kernel-invariant style of
`tests/runner/sandbox/test_docker_e2e.py`: every hardening claim in
your README must have a corresponding e2e test that proves it at the
kernel level (read-only rootfs, capability drop, network isolation,
env-passthrough scope, etc.). The framework cannot enforce those
claims for you - if the test does not exist, the hardening does not
exist. [SANDBOXING.md](SANDBOXING.md) documents the canonical invariant
table for the built-in `docker` provider; reuse the same shape.
