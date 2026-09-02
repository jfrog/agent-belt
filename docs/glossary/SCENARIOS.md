# Scenarios

How to author scenarios that belt can run. Schemas are exhaustive
references; everything else is the non-obvious behaviour you can't
infer from the field list.

The Pydantic models in
[`src/belt/scenario.py`](../../src/belt/scenario.py)
(`GroupConfig`, `Scenario`, `TurnExpectation`, `Resource`) are the
canonical shape - field tables below are derived from them.

## 1. Layout

Scenarios live in groups. Each group targets one agent and contains a
`_config.json` plus one or more scenario JSONs. Put them anywhere on
disk and point the CLI at that directory:

```text
my-scenarios/
└── claude-code/
    ├── read-only/                # explanations, lookups, security review
    │   ├── _config.json          # no working_dir
    │   ├── explain_code.json
    │   └── security_analysis.json
    └── editing/                  # bug fixes, refactors, test additions
        ├── _config.json          # working_dir + workspace_isolation: git-worktree
        ├── add_tests.json
        └── fix_divide_bug.json
```

`working_dir` and `workspace_isolation` are **group-level** config;
mixing read-only and editing scenarios in one group is not supported -
split into sibling groups. CLI filtering works at any depth:

```bash
belt eval my-scenarios/                                  # all groups
belt eval my-scenarios/claude-code                       # both groups
belt eval my-scenarios/claude-code/editing               # one group
```

The bundled examples follow this convention under
`examples/scenarios/{showcase,experience}/`. Some bundled groups
reference fixtures outside the scenarios root and require
`--allow-external-working-dir` to execute; some are tagged
`dry-run-only` (pass `--tags real-runnable` to skip them). See
[`examples/scenarios/showcase/README.md`](../../examples/scenarios/showcase/README.md).

## 2. Group config (`_config.json`)

Minimum:

```json
{ "agent": "claude-code", "default_tags": ["smoke"] }
```

`agent` must match a registered agent (`belt agent list`).

| Field | Required | Default | Description |
|---|---|---|---|
| `agent` | Yes | - | Registered agent name. |
| `default_tags` | No | `[]` | Tags applied to every scenario in the group. |
| `llm_dimensions` | No | `[]` | Custom LLM judge dimensions (list of `DimensionDef` dicts or strings). |
| `llm_dimensions_extend_defaults` | No | `false` | When `true`, custom dimensions are appended to defaults instead of replacing them. |
| `working_dir` | No | `null` | Path to a git repo for code-editing scenarios (relative to the group dir). Group-level only. |
| `workspace_isolation` | No | `"git-worktree"` | `git-worktree` or `none`. Validated by Pydantic - any other string is rejected at parse time. `none` requires `--allow-inplace` (or `BELT_ALLOW_INPLACE=1`); see [§6](#6-workspace-isolation). |
| `workspace_ref` | No | `"HEAD"` | Git ref to checkout each worktree at. |
| `fixture_repo` | No | `null` | Git URL or local path cloned **once per group** as the worktree base. Mutually exclusive with `working_dir`. |
| `fixture_ref` | No | `"HEAD"` | Branch / tag / SHA to check out after cloning `fixture_repo`. |
| `resources` | No | `[]` | List of `Resource` entries copied into each scenario's worktree before the agent starts (see below). |

> Plugin agents and scorers may consume extra keys via Pydantic's
> `extra="allow"`. Built-ins ignore them. See
> [PLUGGABILITY.md → Schema boundary](PLUGGABILITY.md#3-schema-boundary).

### 2.1. `Resource`

| Field | Required | Default | Description |
|---|---|---|---|
| `kind` | Yes | - | `"file"` (copy single file or directory) or `"archive"` (extract `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`). |
| `source` | Yes | - | Local path or `file://` / `https://` URL. Other schemes rejected. Remote downloads capped at 256 MiB. |
| `dest` | Yes | - | Destination path **relative to the worktree root**. Absolute paths and `..` segments rejected. |
| `name` | No | basename of `dest` | Free-form label captured in `resource_lock.json`. |
| `version` | No | `null` | Version label captured in `resource_lock.json` for cross-run pinning. |

## 3. Scenario file (`<name>.json`)

Minimum:

```json
{
  "name": "simple_math",
  "description": "Ask a simple math question requiring no tools",
  "tags": ["no-tools", "single-turn"],
  "turns": [
    {
      "message": "What is 17 * 23? Reply with just the number.",
      "expect": { "no_errors": true, "has_reply": true, "contains": ["391"] }
    }
  ]
}
```

`name` MUST match the filename without `.json`.

| Field | Required | Default | Description |
|---|---|---|---|
| `name` | Yes | - | Scenario identifier; matches the filename. |
| `description` | Yes | - | Human-readable description. |
| `tags` | No | `[]` | Combined with group `default_tags` for filtering. |
| `turns` | Yes | - | Ordered list of conversation turns (see §3.1). |
| `llm_scorer_instruction` | No | `""` | Extra instruction appended to the LLM judge prompt. |
| `llm_scorer_raw_transcript` | No | `false` | Append full raw CLI transcript as a low-priority `## Raw CLI Output` section in the judge prompt. See [§8](#8-llm-scoring-opt-ins). |
| `llm_scorer_evidence_files` | No | `[]` | Up to 20 paths (relative to the scenario JSON's directory) read into the judge prompt as ground-truth evidence. Stay outside the worktree so the agent can't peek. |

> `Scenario`, `Turn`, and `StateExpectation` reject unknown keys
> (`extra="forbid"`). Plugin per-scenario extras go inside `expect`
> (read via `expect.model_extra["..."]` from a custom scorer). Plugin
> per-group extras go in `_config.json`: `GroupConfig` uses Pydantic's
> default `extra="ignore"`, so the keys are silently dropped at
> validation - plugins must re-read the JSON in their own setup hook
> to recover them. See [§10 Plugin extras](#10-tags-mcp-plugin-extras).

### 3.1. Turn

| Field | Required | Default | Description |
|---|---|---|---|
| `message` | Yes | - | User message sent to the agent. Multi-turn templating supported - see [§5](#5-multi-turn-templating). |
| `flags` | No | `[]` | Extra CLI flags forwarded to `agent.execute()`. |
| `expect` | No | `{}` | Deterministic rule-based expectations. See [§4](#4-turn-expect-and-state_expect). |
| `state_expect` | No | `{}` | Post-turn workspace filesystem checks. See [§4.2](#42-state_expect). |
| `verify` | No | `null` | Deterministic command run after this turn in the worktree. See [§4.3](#43-verify-deterministic-exec-test). |

## 4. Turn `expect` and `state_expect`

All fields are optional; omitted checks are skipped, **not** assumed
to pass. The full set of rule-based check semantics lives in
[SCORING.md → Rule-based dimensions](SCORING.md#11-dimensions-and-checks); this
table is the on-disk shape.

| Field | Type | Default | Notes |
|---|---|---|---|
| `no_errors` | bool | `true` | Expect no errors in output. |
| `has_reply` | bool | `true` | Expect a non-empty reply. |
| `contains` / `not_contains` | list[str] | `[]` | Substrings that must / must not appear in the reply. Case-insensitive. `contains` matches against either the agent reply text or the raw CLI transcript (the `raw_cli` fallback) - convenient for permissive checks but cannot prove the agent itself produced the token. Use `reply_pattern` when CLI noise must not produce a false green. |
| `reply_pattern` | list[str] | `[]` | Strict reply-format assertions as Python regexes. ALL patterns must match (each via `re.search`, case-insensitive). Matches against `reply_text` only - no `raw_cli` fallback. Single-line semantics by default; opt into per-line matching with the inline `(?m)` flag. Bad regexes are rejected at scenario-load time, not at scoring time (same policy as `tool_result_pattern`). See [SCORING.md → Regex policy](SCORING.md#13-regex-policy-for-scenario-fields). |
| `tools_invoked` | list[str] | `[]` | Tools that must be called. |
| `tools_invoked_any` | list[list[str]] | `[]` | At least one tool from each sublist. |
| `tools_invoked_in_order` | list[str] | `[]` | Tools that must appear as a subsequence in invocation order. |
| `only_used_tools` / `forbidden_tools` | list[str] | `[]` | Allowlist / denylist on tools invoked. |
| `tool_args_contain` | dict[str, dict] | `{}` | Per-tool argument substring assertions. |
| `tool_result_contains` | dict[str, str] | `{}` | At least one call to `tool` must have a result whose stringified value contains the substring (case-insensitive). |
| `tool_result_pattern` | dict[str, str] | `{}` | Regex variant of the above (`re.search`, case-insensitive). Bad regexes are rejected at scenario-load time, not at scoring time. See [SCORING.md → Regex policy](SCORING.md#13-regex-policy-for-scenario-fields). |
| `skills_invoked` | list[str] | `[]` | Agent-agnostic skill check. Detector matches structured `Skill` tool args, then `skills/<name>` paths in any tool arg, then the same path in raw CLI output. |
| `error_type_is` | str? | `null` | Pin a canonical error token (`authentication_failed`, `rate_limited`, `timeout`, `refused`, `unknown`) - see [OUTCOMES.md](OUTCOMES.md#3-agent-runtime-errors-error_type). |
| `has_thinking` | bool? | `null` | Assert thinking blocks were present/absent (agent-specific). |
| `max_cost_usd` | float? | `null` | Per-turn cost cap (agent-specific). |
| `max_llm_turns` / `max_tool_calls` | int? | `null` | Per-turn caps. |
| `max_ttfe_seconds` / `max_ttft_seconds` / `max_ttlt_seconds` / `max_total_seconds` | float? | `null` | Latency caps (streaming-timing fields are agent-specific). |
| `files_modified_any` / `files_modified_exact` / `files_not_modified` | list[str] | `[]` | Diff-aware checks; require workspace isolation (see §6). Entries must be specific file paths - directory-shaped paths (trailing `/`) are rejected at load time because the scorer compares via literal equality and would silently pass. |
| `git_diff_contains` | list[str] | `[]` | Substrings that must appear in `git diff`. |

**Universal vs agent-specific.** Most fields above work with every
agent. The agent-specific ones (`has_thinking`, `max_cost_usd`,
`error_type_is`, `max_ttfe/ttft/ttlt_seconds`) are silently ignored by
agents that don't populate the corresponding `TurnOutput` fields - see
[AGENT-FEATURES.md](AGENT-FEATURES.md). `TurnExpectation` accepts
arbitrary extra keys (`extra="allow"`) so plugin scorers can read their
own fields via `expect.model_extra["my_key"]`.

### 4.1. Common patterns

```json
"expect": { "contains": ["391"], "not_contains": ["password"] }
```

```json
"expect": {
  "tools_invoked_in_order": ["Read", "Edit"],
  "only_used_tools": ["Read", "Edit", "Write"],
  "tool_result_contains": {"Read": "belt"}
}
```

```json
"expect": { "max_llm_turns": 3, "max_tool_calls": 5, "max_total_seconds": 30, "max_cost_usd": 0.10 }
```

```json
// Negative test for auth-failure handling
"expect": { "no_errors": false, "error_type_is": "rate_limited" }
```

### 4.2. `state_expect`

Post-turn filesystem checks. Captured files land in
`TurnOutput.workspace_files` and feed both rule-based scoring and the
LLM judge as ground-truth evidence.

| Field | Type | Default | Description |
|---|---|---|---|
| `files_exist` / `files_not_exist` | list[str] | `[]` | Relative paths that must / must not exist after the turn. |
| `files_contain` | dict[str, str] | `{}` | Path → substring required in file content. |
| `capture_git_diff` | bool | `false` | Capture `git diff` into `TurnOutput.raw_state`. |

### 4.3. `verify` (deterministic exec-test)

`verify` runs an author-declared command in the worktree and gates on its
exit code - the strongest, cheapest grader for code-editing scenarios ("did
the agent's change make the test suite pass?"). Declarable on a **turn**
(`Turn.verify`, runs after that turn) or on the **scenario**
(`Scenario.verify`, runs once after the final turn - the end-of-conversation
case). Both produce checks under the `rules/verify` dimension; a skipped
verify (command did not run) is recorded as a tri-state skip, not a failure.

| Field | Type | Default | Description |
|---|---|---|---|
| `cmd` | list[str] | - | Command argv (no shell). E.g. `["python", "-m", "pytest", "-q"]`. |
| `exit_code` | int | `0` | Expected exit code; the check passes when it matches. |
| `output_contains` | list[str] | `[]` | Plain substrings (not regex) that must all appear in stdout. |
| `timeout` | int | `300` | Max seconds; a timeout fails the check. |

```json
"verify": { "cmd": ["python", "-m", "pytest", "-q"], "exit_code": 0, "output_contains": ["passed"] }
```

`verify` executes an author-supplied command, so it is default-deny: it runs
only with an isolated worktree and only when `--allow-verify-exec` (or
`BELT_ALLOW_VERIFY_EXEC=1`) is set; otherwise the group is refused at setup.
The command runs through the active sandbox provider (inside the container
under `--sandbox docker`) with a minimal, credential-free environment. See
[SECURITY-MODEL.md](SECURITY-MODEL.md#511-deterministic-verify-execution).

## 5. Multi-turn templating

Inside any turn's `message`, reference fields from prior `TurnOutput`s
using a closed set of placeholders. They render before the agent sees
the message - the agent never sees `{{...}}`.

| Placeholder | Renders |
|---|---|
| `{{prev.reply_text}}` | Previous turn's `reply_text`. |
| `{{prev.git_diff}}` | Previous turn's captured `git_diff` (empty when none). |
| `{{prev.tool_sequence}}` | Previous turn's tool names, comma-joined. |
| `{{turn_N.<field>}}` | Explicit prior turn's `reply_text` / `git_diff` / `tool_sequence` (0-based index). |

```json
{
  "turns": [
    { "message": "List the bugs you find in src/parser.py" },
    { "message": "Fix the issues you listed: {{prev.reply_text}}" }
  ]
}
```

Errors:

- Referencing a turn that hasn't run yet (`{{turn_5.*}}` from turn 2,
  or `{{prev.*}}` from turn 0) fails with `ScenarioError` **before**
  `agent.execute` - much faster than discovering hallucinated context
  at score time.
- An unsupported field fails the same way and lists the supported set.
- Anything not matching `{{scope.field}}` is left untouched, so literal
  `{{` / `}}` in JSON snippets are safe.

Authoring awareness: a placeholder splices arbitrary agent-generated
text into the next turn's prompt - wrap references in quotes or fenced
blocks and keep instructions *outside* the rendered text. Rendering is
single-pass left-to-right; rendered messages are still bounded by
`TURN_MESSAGE_MAX_CHARS` (overflow fails the run, doesn't OOM).

## 6. Workspace isolation

For code-editing scenarios, set `working_dir` in `_config.json`:

```json
{
  "agent": "claude-code",
  "working_dir": "../../fixtures/sample-project",
  "workspace_isolation": "git-worktree",
  "workspace_ref": "HEAD"
}
```

Each scenario gets an isolated git worktree at `workspace_ref`. After
each turn the orchestrator captures `TurnOutput.git_diff` (full
`git diff --cached`) and `TurnOutput.files_modified`. These feed the
diff-aware `expect` keys (`files_modified_*`, `git_diff_contains`)
**and** the LLM judge as ground-truth evidence (see
[SCORING.md](SCORING.md#23-workspace-evidence-ground-truth)).

Parallel workers (`--workers N`) each get independent worktrees - no
locking. The original repo is never modified; worktrees are cleaned up
after each scenario (atexit handler covers crashes). See
[`examples/scenarios/showcase/editing-workspace/`](../../examples/scenarios/showcase/editing-workspace/)
for a working example.

### 6.1. Skipping isolation (`workspace_isolation: "none"`)

`workspace_isolation: "none"` disables per-scenario worktrees: the
agent runs in the harness CWD with no isolation, and any edits it
makes are real. Two guardrails apply:

1. **Schema lock.** `workspace_isolation` is a closed enum
   (`git-worktree` / `none`). Any other value - typos like `"None"` or
   `"git-wortree"`, or invented values like `"off"` - is rejected at
   scenario load time with a Pydantic `ValidationError` that lists the
   valid options. A typo cannot silently fall through to "no
   isolation".
2. **Opt-in gate.** Even the spelled-correctly `"none"` value is
   refused by default. Re-run with `--allow-inplace` (or set
   `BELT_ALLOW_INPLACE=1`) to permit it. The runner marks any
   `"none"` group as failed and prints a message naming both opt-ins.

Together these mean disabling isolation requires writing the exact
string `"none"` *and* a conscious opt-in. The flag is value-specific:
passing `--allow-inplace` against a fully `git-worktree` scenarios
root is a no-op.

## 7. External fixtures and resources

`working_dir` is the right tool when the codebase under test is a
sibling of your scenarios. For everything else (third-party repo,
versioned skill payload, downloadable corpus), use `fixture_repo` plus
`resources`:

```json
{
  "agent": "claude-code",
  "fixture_repo": "https://github.com/example/target-repo.git",
  "fixture_ref": "v1.4.0",
  "resources": [
    { "kind": "file",    "source": "../../skills/security-review", "dest": ".skills/security-review", "version": "0.3.1" },
    { "kind": "archive", "source": "https://example.com/test-corpus-2026-04.tar.gz", "dest": ".corpus", "version": "2026-04" }
  ]
}
```

Lifecycle: clone `fixture_repo@fixture_ref` once per group → acquire
isolated worktree per scenario → install `resources` per scenario.
`resource_lock.json` is written next to the scenario's outcomes with
`source_sha256` per entry so reviewers can pin a result to an exact
source SHA.

`fixture_repo` and `working_dir` are mutually exclusive (the runner
rejects groups that set both).

### 7.1. Trust boundaries

- **`fixture_repo` local paths** resolve against the **process CWD**
  with symlink-following enabled (matching `cd` semantics). A hostile
  scenario JSON can plant a symlink to redirect the runner. Run
  scenarios from sources you trust. URL forms (`https://`, `file://`,
  `ssh://`, `git@host:...`) pass through unchanged to `git clone`.
- **`resources[].source` local paths** resolve against the **scenario
  group's `_config.json` directory** - the natural anchor for scenario
  assets. URL forms and absolute paths pass through unchanged.
- **`resources` of `kind: file`** are copied with **symlinks
  preserved** (not followed). The agent sees a symlink, not a copy of
  its target - pair with `workspace_isolation: "git-worktree"` for
  isolation; `resources` alone is not a sandbox.
- **`resources` of `kind: archive`** are extracted with path-traversal
  guards (no `..`, no absolute paths, no oversized members).

## 8. LLM scoring opt-ins

For scenarios where rule-based checks alone aren't sufficient
(subjective quality, security review depth, multi-step reasoning), add
`llm_scorer_instruction` to guide the judge:

```json
{
  "llm_scorer_instruction": "Focus on whether the review identifies the division-by-zero risk and suggests type hints.",
  "turns": [{ "message": "Review this function...", "expect": { "has_reply": true } }]
}
```

Run with `--modes rules,llm`.

By default the judge sees a structured summary built from `TurnOutput`
fields, **not** the agent's raw NDJSON - keeps small judges from
confabulating verdicts about environment events the agent never
produced. Opt in to the raw transcript per scenario when your
evaluation depends on event-level inspection:

```json
{ "llm_scorer_raw_transcript": true, ... }
```

When the rubric is too large for `llm_scorer_instruction` (capped at
10,000 chars) or must stay out of the agent's worktree so the agent
can't peek, attach it via `llm_scorer_evidence_files`. Paths resolve
relative to the scenario JSON's directory and are wrapped as
`<evidence_file path="...">...</evidence_file>` in the judge prompt.
See [SCENARIOS.md → §8 LLM scoring opt-ins](SCENARIOS.md#8-llm-scoring-opt-ins).

## 9. Per-turn LLM judging

When the scenario-level instruction is not granular enough ("in turn 0
the agent must reach tool X with arg Y, in turn 1 it must recover from
the error in turn 0, in turn 2 it must summarise without hallucinating"),
declare a per-turn judge in `--scorer-config` with `resolution: turn`
and attach per-turn overrides via `Turn.llm_judges`:

```json
{
  "name": "per_turn_judging_demo",
  "description": "...",
  "turns": [
    {
      "message": "Read the README and list its sections.",
      "expect": { "has_reply": true },
      "llm_judges": {
        "per_turn_judge": {
          "instruction": "Verify the agent invoked a read tool BEFORE replying."
        }
      }
    },
    {
      "message": "Now summarise the architecture section.",
      "expect": { "has_reply": true },
      "llm_judges": {
        "per_turn_judge": {
          "instruction": "Verify the summary only references content from the README.",
          "dimensions": [{"name": "no_hallucination", "kind": "ternary"}]
        }
      }
    }
  ]
}
```

### 9.1. `Turn.llm_judges` field reference

| Field | Type | Default | Notes |
|---|---|---|---|
| `instruction` | `str` | `null` | Per-turn rubric override. Replaces (does not extend) the scenario-level `llm_scorer_instruction` for this turn only. Capped at 10 000 chars. Closing `</scenario_instruction>` tags are neutralised before reaching the judge. |
| `dimensions` | `list` | `null` | Per-turn dimension override. Accepts both string shorthand (`"correctness"`) and `DimensionDef` dicts. Capped at 50 entries. |
| `extend_default_dimensions` | `bool` | `false` | When `true`, declared `dimensions` extend the judge's default rubric rather than replacing it. |
| `evidence_files` | `list[str]` | `null` | Per-turn evidence override. Paths resolve relative to the scenario JSON's directory; `..` and absolute paths reject with the same traversal guard as the scenario-level path. Closing `</evidence_file>` tags are neutralised. Capped at 20 entries. |
| `skip` | `bool` | `false` | Skip this judge for this turn (no API call). Other turns still score normally. A scenario where every turn skips the only judge rejects at preflight (no vacuous-pass). |

The dict is capped at 10 entries (`turns × llm_judges × dimensions` bounds
the per-scenario judge call count to `100 × 10 × --trials` worst case).

For judging mechanics, evidence scope, cost amplification, and the
threat model see [SCORING.md → §2.10](SCORING.md#210-per-turn-llm-judging).

## 10. Tags, MCP, plugin extras

**Tags.** Combine group `default_tags` with per-scenario `tags`; the
runner matches with AND logic.

```bash
belt eval my-scenarios/ --tags smoke
belt eval my-scenarios/ --tags smoke --tags multi-turn   # both required
```

Two tags carry framework meaning in the bundled showcase: `real-runnable`
(pass with `--modes rules` against a generic CLI agent) and
`dry-run-only` (schema example that doesn't run cleanly). The aggregator
footnote points users at `--tags real-runnable` to skip the latter.

**MCP servers.** belt doesn't configure MCP servers per run. If a
scenario requires the agent to call an MCP tool, the server must be
pre-configured on the host before the eval. Most agents auto-discover
servers from a config file in the worktree root - consult your agent's
docs. Document required servers in the group's `README.md`.

**Plugin extras.** Three core surfaces accept extras, each with
different retrieval semantics:

- `TurnExpectation` and `TurnOutput` use `extra="allow"` - unknown
  keys land in `model_extra` and a custom scorer reads them back via
  `expect.model_extra["..."]` / `output.model_extra["..."]`.
- `GroupConfig` uses Pydantic's default `extra="ignore"` - unknown
  keys validate silently but are dropped from the model. A plugin
  that needs per-group config must re-read the raw JSON in its own
  setup hook (`json.loads((group_dir / "_config.json").read_text())`).

`Scenario`, `Turn`, and `StateExpectation` are `extra="forbid"`: an
unknown top-level key in a scenario JSON fails the loader with `Extra
inputs are not permitted` and the scenario is skipped. Built-in
scorers ignore the allowed extras, so a scenario authored for a plugin
still loads against agents that lack it. See
[PLUGGABILITY.md → Schema boundary](PLUGGABILITY.md#3-schema-boundary).

## 11. Strict config validation (`--strict-config`)

Permissive parsing on `TurnExpectation` and `GroupConfig` is a typo
trap. `"tools_invoke": [...]` (missing `d`) lands in `model_extra`
and produces zero coverage. `"agnet": "claude-code"` is dropped from
`GroupConfig` and the loader falls back to defaults. Both load
silently and CI shows green.

Pass `--strict-config` to opt into a schema-driven validator that
runs before Pydantic and rejects keys that are neither declared on
the model nor explicitly registered as a plugin extension:

```bash
belt eval my-scenarios/ --strict-config
```

```text
❌ --strict-config: refusing to run with malformed scenarios:
   - claude-code/editing: editing/fix_bug.json: unknown key
     'turns[0].expect.tools_invoke'. Did you mean 'tools_invoked'?
```

The validator reports a fully-qualified JSON path and a difflib
suggestion so authors can find and fix the offending key without
re-reading the file. `--strict-config` is unconditionally fatal: any
rejection aborts the run with exit `1`, independent of `--strict`.

Plugins declare their extension keys at import time so they pass
under strict mode without a special case:

```python
from belt import register_plugin_scenario_key
from belt.scenario import TurnExpectation

register_plugin_scenario_key(TurnExpectation, "max_handoffs")
register_plugin_scenario_key(TurnExpectation, "review_prompted")
```

See [PLUGGABILITY.md → Strict-config registration](PLUGGABILITY.md#33-strict-config-key-registration)
for the registration contract and constraints (reserved framework
names cannot be shadowed; keys are process-local, never persisted).

CI pipelines should turn this on the same day they turn on
`--strict`: it is the difference between "we want this run to be
real" and "we want this run to be real *and* well-typed".
