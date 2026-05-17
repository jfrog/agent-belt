# Configuration

This document is the canonical reference for every way to configure
belt. It answers three questions:

1. **Which layer wins** when the same setting is set in multiple places.
2. **Every `BELT_*` environment variable**, what it does, and its
   default - exhaustive, sorted, one stop.
3. **What the framework will refuse to do** unless you opt in
   (security toggles + the trust model that justifies the defaults).

For per-flag CLI help, run `belt <command> --help` - that is the
canonical, drift-free flag reference.

## 1. Precedence

Higher layers override lower ones. Add a setting at the lowest layer
that makes sense; promote up only when an invocation needs to override
it.

```text
CLI flags (-S, -X, --allow-*, --outcomes-dir, …)   ← highest priority
    │
Environment variables (BELT_*)
    │
Config file (belt.yaml, judges.yaml)
    │
Built-in defaults                                  ← lowest priority
```

> **`llm.model` has no built-in default.** If none of the three layers
> above sets it, `belt eval --modes llm` (and `belt score
> --modes llm`) fails at preflight with a three-source error message
> listing the CLI flag, env var, and yaml key. `--modes rules` runs
> fine without a model. Rationale: silently routing
> Azure/Anthropic/Ollama users to OpenAI hides genuine misconfiguration.

## 2. CLI flags

Highest priority. Two patterns appear repeatedly across `eval`, `run`,
`score`, `aggregate`, `export`:

| Flag | Purpose |
|---|---|
| `--agent-arg key=value` (`-X`) | Pass a key/value into the active agent (e.g. `-X timeout=120`). |
| `--scorer-arg key=value` (`-S`) | Pass a key/value into the active scorer (e.g. `-S model=openai/gpt-5.4-mini`, `-S temperature=0.0`). |
| `--allow-*` | Opt into a default-deny behaviour for this invocation only. See [§4 Behaviour gates](#4-behaviour-gates---allow----default-deny). |

For everything else, see `belt <command> --help`.

## 3. Environment variables

All public variables use the `BELT_*` prefix to avoid colliding
with provider-native names. Internal handoffs use the `_BELT_*`
prefix and are not part of the public surface.

The source of truth is
[`src/belt/envvars.py`](../../src/belt/envvars.py); a test
asserts every `BELT_*` literal in `src/` is declared there.

### 3.1. LLM routing

| Variable | Description | Default |
|---|---|---|
| `BELT_LLM_MODEL` | Model / deployment name. **Required when `--modes llm`.** Must include a provider prefix (`openai/`, `anthropic/`, `azure/`, `ollama/`). | unset |
| `BELT_LLM_PROVIDER` | Force a provider for unprefixed model names (`openai`, `anthropic`, `azure`, `ollama`). | inferred from prefix |
| `BELT_LLM_PREFLIGHT_TIMEOUT` | Per-probe timeout (seconds) for the judge model preflight that `belt eval` runs before spawning the agent phase. Increase on satellite links / heavily proxied environments; lower on tight CI to fail faster. | `10` |
| `BELT_PRICING_FILE` | Path to a TOML file overriding the bundled judge-cost pricing table. The override **fully replaces** the bundled table (it is not merged) so an invalid or partial file fails loudly at import. Schema mirrors [`src/belt/scorer/llm/pricing.toml`](../../src/belt/scorer/llm/pricing.toml): `[models."<name>"]` rows with `input_per_token`, `output_per_token`, and optional `valid_from` / `source_url` provenance, plus an optional `[aliases]` table. Use this to ship enterprise-negotiated rates or to add models released after the bundled table's `valid_from`. | unset (use bundled) |

Model parameters (`temperature`, `seed`, `max_tokens`, `max_prompt_chars`)
are per-judge settings and live in `belt.yaml` or `judges.yaml`,
not env vars.

### 3.2. LLM provider credentials

Set exactly one group, matching the provider prefix in your model name.

| Provider | Model prefix example | Required | Optional |
|---|---|---|---|
| OpenAI | `openai/gpt-5.4-mini` | `BELT_OPENAI_API_KEY` | `BELT_OPENAI_BASE_URL` |
| Anthropic | `anthropic/claude-sonnet-4-5` | `BELT_ANTHROPIC_API_KEY` | `BELT_ANTHROPIC_BASE_URL` |
| Azure OpenAI (API key) | `azure/my-deployment` | `BELT_AZURE_OPENAI_ENDPOINT`, `BELT_AZURE_OPENAI_API_KEY` | `BELT_AZURE_OPENAI_API_VERSION` |
| Azure OpenAI (Service Principal) | `azure/my-deployment` | `BELT_AZURE_OPENAI_ENDPOINT`, `BELT_AZURE_CLIENT_ID`, `BELT_AZURE_CLIENT_SECRET`, `BELT_AZURE_TENANT_ID` | `BELT_AZURE_OPENAI_API_VERSION` |
| Ollama (local) | `ollama/gemma4` | none - auto-detected | `BELT_OLLAMA_BASE_URL` |
| OpenAI-compatible (vLLM, LM Studio) | `openai/<name>` | `BELT_OPENAI_BASE_URL` | - |

Ollama uses a dedicated native backend (`/api/chat` with
grammar-constrained `format`) rather than its OpenAI-compatible endpoint.
Run `belt doctor` to confirm provider detection.

### 3.3. Output paths and disk budgets

| Variable | What it controls | Default |
|---|---|---|
| `BELT_OUTCOMES_DIR` | Root directory under which run dirs are written. Lower priority than `--outcomes-dir`. | `./outcomes` |
| `BELT_CACHE_MAX_BYTES` | LRU-by-mtime cap on the per-judge response cache (`<run>/.score_cache/`). `0` disables eviction. | 500 MiB |
| `BELT_TURN_NDJSON_MAX_BYTES` | Cap on the per-turn live NDJSON stream artifact (`turn_*_stream.ndjson`). | 50 MiB |
| `BELT_SUBPROCESS_STDOUT_MAX_BYTES` | Cap on total bytes captured from a single agent subprocess's stdout. | 256 MiB |
| `BELT_SUBPROCESS_STDOUT_LINE_MAX` | Cap on the length of a single stdout line within the above. Lines exceeding this are split for the runner's parser. | 1 MiB |

### 3.4. `.env` loading

belt auto-loads `.env` from the working directory via
`python-dotenv` so credentials from a local file reach the runner.

| Variable | What it controls | Default |
|---|---|---|
| `BELT_NO_DOTENV` | Set to `1` to disable `.env` auto-loading (e.g. CI where the env must come from the runner secret store only). | enabled |

### 3.5. Operator diagnostics

| Variable | What it does |
|---|---|
| `BELT_DEBUG` | When truthy, print full Python tracebacks to stderr on unexpected errors. Non-secret; recorded verbatim in `run_meta.json` for reproducibility. |
| `BELT_LOG_LEVEL` | Terminal log level for `belt` commands. Accepts `TRACE`/`DEBUG`/`INFO`/`SUCCESS`/`WARNING`/`ERROR`/`CRITICAL` (case-insensitive). Defaults to `WARNING`: only warnings and errors print to the terminal during a run, results panel and tables are unaffected. Set to `INFO` to see judge reasoning and trajectory diagnostics inline (equivalent to `belt eval -v`); set to `DEBUG` for trace-level output (equivalent to `-vv`). The on-disk transcript log at `<run_dir>/eval.log` always records `DEBUG` regardless of this setting. |

### 3.6. Behaviour gates (`BELT_ALLOW_*`) - default DENY

See [§4 Behaviour gates](#4-behaviour-gates---allow----default-deny) for what each one
permits, why it's off by default, and the matching `--allow-*` CLI
flag (CLI wins over env).

| Variable |
|---|
| `BELT_ALLOW_INSECURE_BASE_URL` |
| `BELT_ALLOW_FULL_ENV` |
| `BELT_ALLOW_INPLACE` |
| `BELT_ALLOW_ARBITRARY_AGENT` |
| `BELT_ALLOW_ARBITRARY_SCORER` |
| `BELT_ALLOW_ARBITRARY_EXPORTER` |
| `BELT_ALLOW_EXTERNAL_WORKING_DIR` |

### 3.7. Warning suppressors (`BELT_SILENCE_*`) - default WARN

| Variable | What it silences |
|---|---|
| `BELT_SILENCE_CUSTOM_BASE_URL_WARNING` | The "Custom LLM base URL active via …" warning that fires once per process when any `BELT_*_BASE_URL` is non-default. Behaviour is unchanged either way. |

> Suppressing a warning never changes behaviour. `SILENCE_CUSTOM_BASE_URL_WARNING`
> does not permit plaintext traffic - that's `ALLOW_INSECURE_BASE_URL`.

### 3.8. Sandbox provider selector

| Variable | What it controls | Default |
|---|---|---|
| `BELT_SANDBOX_PROVIDER` | Override the sandbox provider for the whole run. Names resolve via the `belt.sandbox_providers` entry-point group; the framework ships `host` (no isolation, today's behaviour: agent runs on the host with the invoking user's privileges) and `docker` (each agent subprocess in a container with `--cap-drop=ALL`, `--read-only` rootfs, the worktree as the only writable mount, env passthrough by exact name). The CLI flag `--sandbox PROVIDER` takes precedence. Per-scenario `sandbox.image` / `sandbox.allowed_hosts` / `sandbox.env_passthrough` are still read from each group's `_config.json`. See [SANDBOXING.md](SANDBOXING.md). | `host` |

### 3.9. Discipline overrides (`BELT_NO_*`) - default ENFORCE

| Variable | What it disables |
|---|---|
| `BELT_NO_DOTENV` | `.env` auto-loading (also listed under [§3.4](#34-env-loading)). |
| `BELT_NO_UMASK` | The `0o077` umask belt applies before writing artifacts (so logs, score caches, and run dirs are owner-only). Set when a parent process needs to read belt's output. |

### 3.10. Truthy values

For every boolean variable above, belt treats `1`, `true`, and
`yes` as on. Anything else - including empty string, `TRUE`, `True`,
`0`, `no` - is **unset**. This is deliberate; see
[`is_truthy` in envvars.py](../../src/belt/envvars.py).

### 3.11. Agent-side provider credentials

§3.2 covers the LLM **judge** that runs inside belt. Each CLI agent
has its own credential model owned by the agent's own subprocess;
belt's job is to forward variables through the scrubbed subprocess
env via per-agent `required_env_vars()`.

For codex against Azure OpenAI, follow the
[Microsoft Foundry recipe](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/codex)
to populate `~/.codex/config.toml`, export `AZURE_OPENAI_API_KEY`,
then run `belt eval ... --agent codex -X profile=<name> -X model=<deployment>`.

## 4. Behaviour gates (`--allow-*`) - default DENY

The framework refuses these by default. Setting the flag (or the
matching `BELT_ALLOW_*` env var) opts in for that invocation.

| Flag | Available on | Gates | When to opt in |
|---|---|---|---|
| `--allow-insecure-base-url` | `eval`, `run`, `score` | LLM judge calls over `http://` against a non-loopback host. `https://` and `http://` to loopback (`localhost`/`127.0.0.1`/`::1`) are always allowed. | In-cluster vLLM at `http://gpu-host.cluster:8000` or a corporate plaintext proxy. You're accepting that bearer tokens travel in cleartext. |
| `--allow-full-env` | `eval`, `run` | Whether the agent subprocess inherits the full `os.environ`. Default: belt scrubs the env to a curated allow-list. | Local debugging only. Risky on shared CI hosts because unrelated secrets flow into the agent. |
| `--allow-arbitrary-agent` | `eval`, `run`, `score` | Loading an agent via dotted import path instead of a registered name / entry-point. | Power-user iteration on a third-party agent that isn't packaged yet. A hostile config could otherwise import arbitrary modules. |
| `--allow-arbitrary-scorer` | `eval`, `run`, `score` | Loading a scorer via dotted import path. Same risk profile as the agent toggle. | Same as above, for a custom scorer. |
| `--allow-arbitrary-exporter` | `eval`, `export` | Loading an exporter via dotted import path. | Same as above, for an exporter not yet registered as an `belt.exporters` entry point. |
| `--allow-external-working-dir` | `eval`, `run` | Allowing a scenario's `working_dir` to resolve outside the scenarios-root path argument. | Multi-repo monorepos where scenarios deliberately reference a sibling worktree. |
| `--allow-inplace` | `eval`, `run` | Permitting groups whose `_config.json` declares `workspace_isolation: "none"`. Default DENY: belt refuses such groups so an agent never runs without per-scenario worktree isolation by accident. The schema rejects any other value than `git-worktree` / `none` at parse time, so this gate is the *only* way isolation gets disabled. | Scenarios that deliberately edit the harness CWD (rare; usually you want `git-worktree`). |

> **Per-agent denied flags.** Each agent declares a `denied_flags()`
> set (e.g. `--dangerously-skip-permissions` on Claude Code) that the
> runner strips from scenario `flags` before invoking the CLI. There
> is no public flag to disable this filter; override `denied_flags()`
> in a custom subclass if you have a legitimate need.

## 5. Config file (`belt.yaml`)

Place a `belt.yaml` anywhere in your project tree. belt
walks **up** from the working directory (or scenarios directory)
toward the filesystem root to find it.

```yaml
# belt.yaml
llm:
  model: openai/gpt-5.4-mini   # required when --modes llm; provider prefix mandatory
  # temperature, seed, max_tokens, max_prompt_chars: optional per-judge knobs
```

For multiple LLM judges with different configurations, use a dedicated
file via `--scorer-config`:

```bash
belt eval scenarios/ --scorer-config examples/scorer-config/judges.yaml --modes llm
```

The full multi-judge format lives in [SCORING.md](SCORING.md).

### 5.1. Dimension precedence

Per-judge `dimensions` declared in the YAML are honored unless a
higher-priority source contributes its own:

| Priority | Source | Where it lives |
|---|---|---|
| 1 (highest) | `llm_dimensions` on the scenario group | `<group>/_config.json` |
| 2 | Agent class override | `scoring_strategy()` on a custom `BaseAgentAdapter` subclass |
| 3 | `--scorer-config` YAML | Per-judge `dimensions` block |
| 4 (lowest) | Generic defaults | Built-in `GENERIC_DIMENSIONS` |

A judge that sets `extend_defaults: true` in its YAML entry merges its
dimensions onto the generic defaults; other layers control merge
behaviour with their own flag (`llm_dimensions_extend_defaults` for
the group config).

## 6. Flag vs env: when to use which

Every public knob falls into one of three categories. Use this rule
when adding new ones.

| Category | When to choose it | Examples |
|---|---|---|
| **Flag-only** | Per-invocation choice unlikely to be reused across runs in the same shell. A flag would only be noise in the env. | `--workers`, `--progress`, `--dry-run`, `--threshold`, `--strict`, `--strict-config`, `--llm-fail-on`, `--scenario-delay` |
| **Env-only** | Setting describes the runtime environment, not a per-run choice - secrets, log-noise tuning, host-dependent disk budgets, framework behaviour read at import time. Surfacing a flag would tempt scenario authors to set it per-call. | secrets (`BELT_OPENAI_API_KEY`), discipline overrides (`BELT_NO_DOTENV`, `BELT_NO_UMASK`), warning suppressors (`BELT_SILENCE_*`), disk budgets (`BELT_*_MAX_BYTES`) |
| **Both** | Setting is security-sensitive *or* a frequently-toggled path. The env var is "this CI job opts in for the whole pipeline"; the flag is "I'm opting in for this one invocation." Flag overrides env overrides yaml overrides default. | every `BELT_ALLOW_*` ↔ `--allow-*`, `BELT_OUTCOMES_DIR` ↔ `--outcomes-dir`, `BELT_LLM_MODEL` ↔ `-S model=` |

The parity rule between every `BELT_ALLOW_*` and its `--allow-*`
flag is enforced by `tests/test_envvars.py::TestAllowFlagParity`.
