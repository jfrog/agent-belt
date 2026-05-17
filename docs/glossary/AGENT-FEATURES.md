# Agent Feature Matrix

Which capabilities each built-in agent supports today.

## Quick Reference

| Name | CLI | Output format | Multi-turn session flag |
|------|-----|---------------|------------------------|
| `claude-code` | `claude` | NDJSON | `--resume <id>` |
| `codex` | `codex` | JSONL (`exec --json`) | `exec resume <session_id>` |
| `copilot` | `copilot` | JSONL (`--output-format json`) | `--resume <id>` |
| `cursor` | `cursor-agent` | NDJSON | `--thread-id <id>` |
| `gemini` | `gemini` | NDJSON (`--output-format stream-json`) | `--resume <id>` |
| `opencode` | `opencode` | NDJSON (`run --format json`) | `--session <id> --continue` |
| `goose` | `goose` | NDJSON (`run --output-format stream-json`) | `--resume --name <name>` |

## Feature Matrix

> **Maintenance.** This matrix is hand-maintained and reviewed in PRs alongside
> agent changes. The behavioral parity contract that prevents the most
> impactful drift (workspace cwd propagation, supported-output-fields integrity)
> is enforced by `tests/agent/test_agent_parity.py` - see
> [PLUGGABILITY.md → Parity contract](PLUGGABILITY.md#64-parity-contract).

| Feature | claude-code | codex | copilot | cursor | gemini | opencode | goose |
|---------|:-----------:|:-----:|:-------:|:------:|:------:|:--------:|:-----:|
| **Streaming execution** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Cost tracking** | ✅ | ✅ | ⚠️ defensive | - | - | ✅ | - |
| **Tool call extraction** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Tool sequence / ordering** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Multi-turn sessions** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Timing (ttfe/ttft/ttlt)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Error classification** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Auth-failure parity** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Thinking/reasoning** | ✅ | ✅ | ⚠️ defensive | - | - | - | ✅ |
| **Live progress (`parse_stream_event`)** | - | - | ✅ | ✅ | - | ✅ | ✅ |
| **`display_info()`** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **`cli_options()`** | - | ✅ model | ✅ model | - | - | ✅ model | ✅ model, provider |
| **`check_available` auth** | ✅ | ⚠️ default model only | ⚠️ PATH only | ✅ | ✅ | ⚠️ PATH only | ⚠️ PATH only |
| **`llm_turn_count`** | ✅ | ✅ | ✅ | - | - | ✅ | ✅ |
| **Token tracking** | - | - | - | - | - | ⚠️ data available | ⚠️ in complete event |
| **Workspace `cwd` propagation** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

> `-` for **Live progress** means the agent uses the framework's default
> NDJSON renderer rather than a custom `parse_stream_event` override. Live
> progress is still emitted; it's just generic instead of agent-specific.
> See [CLI.md → Progress modes](CLI.md#5-progress-modes---progress) for the
> full agent x mode matrix and the trust model that keeps untrusted agent
> stdout from being interpreted as Rich markup.
>
> `-` for **`cli_options()`** means the agent is parameterless by design
> (claude-code, cursor, gemini). Policy choices like `model` flow
> through scenario `flags` (e.g. `--model <model-name>`); the harness does
> not inject a constructor kwarg from the environment.

## Known Gaps & Future Work

### Framework-level (all agents)

- Per-scenario timeout config (currently hardcoded 300s)
- Model sweep / matrix runs across models
- MCP tool metadata extraction (server URIs, resource types)

### claude-code

- LLM judge inclusion of thinking text (gated on scorer config)
- Session ID in progress display for multi-worker debugging
- Retry logic for overload errors
- Parameterless constructor by design - to pin a model in a scenario, pass
  `--model <model-name>` via the scenario's `flags`. `ANTHROPIC_MODEL` is
  read by the `claude` CLI itself (not by the agent class).

### codex

- Requires `codex-cli` 0.130+ (native Rust binary); older `0.1.x`
  Node.js builds are rejected by `check_available`.
- Sandbox defaults to `workspace-write` because belt already isolates
  each scenario in its own git worktree. `denied_flags()` blocks only
  `--sandbox=danger-full-access` and `--dangerously-bypass-approvals-and-sandbox`;
  scenarios may pass `--sandbox=read-only` to opt into the stricter mode.
- For Azure OpenAI, see [CONFIGURATION.md §3.11](CONFIGURATION.md#311-agent-side-provider-credentials).

### copilot

- Parser primary path is the namespaced `assistant.*`/`tool.*`/`result`
  schema; legacy/Claude-style flat shapes (`assistant`/`message`/`tool_use`/
  `function_call`) are kept as defensive fallbacks
- Cost/thinking extraction marked "defensive" - fields are wired up but
  whether Copilot's `result` event actually populates `total_cost_usd` /
  whether assistant blocks carry `thinking` is model-dependent (reasoning
  models surface thinking; standard models do not)
- `--allow-all-tools` is passed by default (programmatic mode requires it);
  scenario-level `--allow-all` / `--yolo` / `--allow-all-paths` /
  `--allow-all-urls` / `--remote` / `--connect` are blocked via
  `denied_flags()`; selective `--allow-tool` / `--allow-url` remain permitted
- `check_available` does not probe auth (consistent with framework policy);
  auth failures surface at eval time as `TurnOutput.has_error`
- `--agent` (custom agents like `code-review`, `research`, `explore`) and
  `/fleet` parallel subagents are passable via scenario `flags` but not
  declared `cli_option`s

### cursor

- Workspace isolation framework for edit-and-verify scenarios
- MCP pre-flight checks (`cursor agent mcp list`)
- `check_available` doesn't verify headless mode actually works
- Headless hang detection (zero-output vs empty response)

### gemini

- Cost tracking (not surfaced in `--output-format stream-json` result events)
- `llm_turn_count` aggregation from stream events
- MCP-specific metadata extraction (upstream CLI support needed)
- Parameterless constructor by design - pass `--model <model-name>` via
  scenario `flags` to pin a model.

### opencode

- Token tracking (data in `step_finish.part.tokens`, not yet aggregated)
- `serve` mode for warm starts across scenarios
- `--agent` persona as declared `cli_option`
- `check_available` auth probe (no lightweight probe command exists)

### goose

- Cost tracking (not exposed in stream-json events)
- Token aggregation from `complete.total_tokens` into `TurnOutput` field
- `check_available` auth probe (no lightweight probe command exists)
- MCP extension notification parsing for richer progress display
