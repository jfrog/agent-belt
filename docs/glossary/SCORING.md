# Scoring

belt has two scoring modes, both optional and composable:

- **`rules`** - deterministic per-turn checks defined in scenario
  `expect` blocks. Always available, no API keys.
- **`llm`** - semantic evaluation by an LLM judge with structured
  per-dimension verdicts. Each dimension declares its own verdict
  scale (`ternary`: `high`/`medium`/`low`, or `binary`: `pass`/`fail`)
  and may opt in to an `inconclusive` verdict for cases where the
  evidence is insufficient to grade. See
  [§2.5 Verdict scales](#25-verdict-scales-binary--ternary--inconclusive)
  for the full matrix. Provider credentials live in
  [CONFIGURATION.md → LLM provider credentials](CONFIGURATION.md#32-llm-provider-credentials).

```bash
belt eval scenarios/ --modes rules        # deterministic only
belt eval scenarios/ --modes rules,llm    # both
```

## 1. Rule-based scoring

### 1.1. Dimensions and checks

This is the canonical map from rule-based dimension to the `expect`
keys that contribute to it. The on-disk `expect` shape lives in
[SCENARIOS.md → §4](SCENARIOS.md#4-turn-expect-and-state_expect).

| Dimension | Checks |
|---|---|
| `execution` | `no_errors`, `not_contains`, `error_type_is` |
| `trajectory` | `tools_invoked`, `tools_invoked_any`, `tools_invoked_in_order`, `only_used_tools`, `forbidden_tools`, `tool_args_contain`, `tool_result_contains`, `tool_result_pattern`, `has_thinking` |
| `response` | `has_reply`, `contains`, `reply_pattern` |
| `efficiency` | `max_llm_turns`, `max_tool_calls` |
| `cost` | `max_cost_usd` |
| `state` | `files_exist`, `files_contain`, `files_not_exist` |
| `file_diff` | `files_modified_any`, `files_modified_exact`, `files_not_modified`, `git_diff_contains` |
| `performance` | `max_ttfe_seconds`, `max_ttft_seconds`, `max_ttlt_seconds`, `max_total_seconds` |

Each check produces a
`CheckEntry(dimension, check, passed, details, turn_idx)` - full
payload contract in
[PLUGGABILITY.md → Payload contract](PLUGGABILITY.md#73-payload-contract).
A scenario passes rules iff every check passes. Plugin scorers can
register additional dimensions via the `belt.scorers` entry-point
group ([PLUGGABILITY.md → §7](PLUGGABILITY.md#7-authoring-a-scorer)).

### 1.2. Strict reply-format assertions (`reply_pattern`)

`contains` matches against either the agent reply text OR the raw CLI
transcript (the `raw_cli` fallback) - permissive, but means CLI noise
(debug prints, agent-trace dumps, stderr) can produce a false green.
`reply_pattern` is the strict-format sibling: regex-based, ALL patterns
must match, matched against `reply_text` only. ALL patterns appear as
individual `CheckEntry` rows on the `response` dimension; mixing
`contains` and `reply_pattern` in the same scenario is supported (they
have different fail-green characteristics by design).

```json
"expect": {
  "reply_pattern": [
    "^ORDER\\|",
    "\\|shipped\\|",
    "\\|1Z999AA10123456784$"
  ]
}
```

### 1.3. Regex policy for scenario fields

Both `reply_pattern` and `tool_result_pattern` route their compile
through `belt._regex_policy.compile_user_regex` - the single source of
truth for default flags (`re.IGNORECASE` only) and the error contract
(raise at scenario load, never silently ship a bad regex to score
time). Per-line `^` / `$` anchoring requires the inline `(?m)` flag;
the default is single-line semantics so `^foo$` matches the full reply.

A bad regex aborts `belt eval` at scenario load with a single error
report listing every offending entry. There is no runtime fail-soft
branch: a typo in a regex is a scenario bug, not a scenario outcome.

### 1.4. Partial credit

Binary pass/fail hides progress - an agent passing 45% of checks but
0% of scenarios looks identical to one passing 0% of checks. The
aggregator reports `checks_passed / checks_total` across all scenarios
in the terminal scoring box, `results.json` stats, and the GitHub step
summary.

## 2. LLM judge scoring

### 2.1. Default dimensions

The generic scoring strategy ships four dimensions:

| Dimension | What it measures |
|---|---|
| `execution` | Clean execution, no errors, all turns completed |
| `trajectory` | Correct tools, efficient approach |
| `response_quality` | Accurate, coherent, well-formatted output |
| `efficiency` | Minimal turns, no redundant actions |

### 2.2. Agent output (structured by default)

For every turn, the judge sees a structured summary built from
`TurnOutput` fields, **not** the agent's raw NDJSON:

```text
### Turn 0
<agent_reply>
The capital of France is Paris.
</agent_reply>
<agent_tools>
tool_sequence: ["Read", "Edit"]
tool_calls (2):
- Read({"file_path": "src/foo.py"})
- Edit({"file_path": "src/foo.py", "new_string": "..."})
</agent_tools>
<agent_metadata>
has_reply: true
has_error: null
error_type: null
llm_turn_count: 1
thinking_text: null
</agent_metadata>
```

This is agent-agnostic (every agent populates these fields) and
keeps small judge models from confabulating verdicts about events the
agent never actually emitted (system/init banners, hook events, MCP
tool catalogues). Tool-call args are JSON-previewed and capped per
call so a single Edit with a giant `new_string` cannot drown the turn.

When event-level inspection is genuinely required, opt in per scenario
with `llm_scorer_raw_transcript: true` - the raw CLI appears as a
low-priority `## Raw CLI Output` section.

### 2.3. Workspace evidence (ground truth)

Beyond the per-turn agent output, the prompt includes up to three
evidence layers per turn:

1. **Thread State** - agent internal state, if `raw_state` is populated.
2. **Git Diff** - actual `git diff` from the isolated worktree
   (`TurnOutput.git_diff`), auto-captured when `working_dir` is set in
   `_config.json`.
3. **Workspace Files** - actual file contents
   (`TurnOutput.workspace_files`), captured when `state_expect`
   includes `files_contain` or `files_exist`.

Diff and workspace files are **ground truth** - the system prompt
instructs the judge that they are authoritative. An agent that claims
"I added type hints" while the diff shows nothing scores `low` on
`response_quality`. Empty sections are omitted, so read-only scenarios
get no extra noise.

### 2.4. Prompt truncation

The dynamic message (all evidence combined) is capped at
`max_prompt_chars` (default 100,000 ≈ 25K tokens). When over budget,
sections are truncated lowest-priority first:

| Priority | Section | Strategy |
|---|---|---|
| 1 (highest) | Scenario JSON | Never truncated |
| 2 | Scenario Instruction | Head-preserve |
| 3 | Evidence Files (`llm_scorer_evidence_files`) | Head-preserve |
| 4 | Agent Output (structured) | Head-preserve |
| 5 | Git Diff | Head-preserve (keeps file headers + first hunks) |
| 6 | Workspace Files | Head-preserve |
| 7 | Thread State | Head-preserve |
| 8 (lowest) | Raw CLI Output (opt-in) | Tail-preserve (keeps final answer + errors) |

A `... (N chars truncated)` marker is injected so the judge knows.
Configure via `-S max_prompt_chars=200000` or `belt.yaml`.

### 2.5. Verdict scales (binary / ternary / inconclusive)

Each dimension declares its verdict scale via `kind` and may opt in
to a fourth `inconclusive` verdict:

| `kind`              | `allow_inconclusive` | Verdicts                              |
|---------------------|----------------------|---------------------------------------|
| `ternary` (default) | `false` (default)    | `low` / `medium` / `high`             |
| `ternary`           | `true`               | + `inconclusive`                      |
| `binary`            | `false`              | `pass` / `fail`                       |
| `binary`            | `true`               | + `inconclusive`                      |

Use `ternary` for graded subjective rubrics, `binary` for
correctness or safety assertions where `medium` is degenerate. The
`pass` JSON key is reserved (Python keyword) and aliased to `pass_`
by the loader. See
[`examples/scenarios/showcase/verdict-scales/`](../../examples/scenarios/showcase/verdict-scales/)
for a runnable example covering all four combinations.

`inconclusive` is hardcoded as a failure in the headline pass-rate
so the judge cannot hedge for a free pass; the aggregator reports it
separately under `stats.llm` so reviewers can distinguish
*agent did it wrong* from *evidence missing*. The judge is instructed
(in the system preamble) to quote the evidence gap in `reasoning` and
to never pick `inconclusive` just because the answer is hard. Abuse
is bounded by `INCONCLUSIVE_CEILING_PCT` in `belt.aggregator.stats`:
when one dimension trips it, the run emits a
`stats.llm_inconclusive_warnings` entry on the next render.

`--llm-fail-on` controls **threshold gating only** - the headline
pass-rate is hardcoded to fail on `low` / `fail` / `inconclusive`
regardless. See §4.

### 2.5.1. Judge infrastructure failures

A scenario whose LLM-judge backend fails for infrastructure reasons
(rate-limit, timeout, network, parse failure) writes a non-verdict
`LLMPayload` with `judge_errored=true` and `judge_error_type` set to one
of `rate_limited` / `timeout` / `auth_failed` / `other`. The pipeline
then appends a synthetic `execution/llm_scorer_ran` check to the rules
payload and forces `overall_pass=false`, so rules cannot silently green-
light a scenario whose judge never voted. The aggregator partitions
these scenarios into a dedicated `env_failed_judge` bucket in the task-
quality split, with their own headline line ("LLM judge infrastructure
failure in N/M scenario(s): rate_limited (2), timeout"). Downstream eval
runners detect them via the typed `LLMPayload.judge_errored` field in
`score.json`, never via string parsing on grader evidence. Auth errors
(`401` / `403` / `404`) instead raise `ScorerError` and abort the run -
they are user-actionable config bugs, not transient failures. Backends
override `BaseJudgeBackend.classify_error(exc)` to map provider-specific
exception shapes onto the same token set. `belt eval` and `belt
aggregate` exit non-zero whenever `judge_errors` is populated, even
without an explicit `--threshold`; a run whose verdicts never arrived
is not actionable, so CI must not mark it green.

### 2.6. Custom dimensions

Three sources, in priority order (first source that contributes wins;
later sources only fill in when earlier ones are absent):

1. **Per-group `_config.json`** - `llm_dimensions` scoped to a group;
   set `llm_dimensions_extend_defaults: true` to append rather than
   replace defaults.
2. **Agent override** - agents can override `scoring_strategy()` in
   Python for agent-specific defaults
   ([PLUGGABILITY.md → §6](PLUGGABILITY.md#6-authoring-an-agent)).
3. **`--scorer-config` YAML** - per-judge `dimensions` block, with
   per-judge `extend_defaults: true` to merge onto generic defaults.

Generic defaults apply when nothing else is configured.

```json
// _config.json
{
  "llm_dimensions": [
    {
      "name": "tool_selection_accuracy",
      "description": "Did the agent pick the correct tool?",
      "high": "correct tool every time, ignored distractors",
      "medium": "mostly correct, one suboptimal pick",
      "low": "used wrong tools or fell for distractors"
    }
  ],
  "llm_dimensions_extend_defaults": true
}
```

```yaml
# judges.yaml - string shorthand also works (`- "policy_adherence"`)
judges:
  my_judge:
    model: openai/gpt-5.4-mini
    extend_defaults: true
    dimensions:
      - name: policy_adherence
        description: "Did the agent stay within scope?"
        high: "strictly scoped"
        low: "significant out-of-scope actions"
```

See `examples/scorer-config/custom-dimensions.yaml` for a full example.

### 2.7. Per-scenario tuning

`llm_scorer_instruction`, `llm_scorer_raw_transcript`, and
`llm_scorer_evidence_files` are all set on the scenario JSON - see
[SCENARIOS.md → §8 LLM scoring opt-ins](SCENARIOS.md#8-llm-scoring-opt-ins).

### 2.8. Judge model preflight

Before the agent phase starts, `belt eval --modes llm` runs one cheap
probe per configured judge model in parallel:

- **OpenAI / OpenAI-compatible**: `GET /v1/models/{id}`
- **Azure**: `GET /openai/deployments/{name}?api-version=…`
- **Anthropic**: `GET /v1/models/{id}`
- **Ollama**: `POST /api/show`

This separates the three "config bug" cases the user must fix from
transient provider hiccups:

| Outcome | What happens |
|---|---|
| 2xx | Run proceeds. |
| 401 / 403 / 404 | Abort in <2s with a hint that branches on `(status, error.code)`. Multi-judge configs aggregate every failure into one composite error. |
| 5xx / 429 / timeout / network | Run proceeds. The runtime [judge infrastructure failure path](#251-judge-infrastructure-failures) catches real failures per-scenario. |

`belt eval --dry-run` skips the network probe (still validates config
shape). The 4xx-only abort policy is deliberate: a single provider
hiccup at T=0 must not block a long-running eval - that's what the
issue-358 partition is for. Timeout knob:
`BELT_LLM_PREFLIGHT_TIMEOUT` (default 10s).

The user-facing hint string is built by a single formatter
(`belt/scorer/llm/judge_hints.py`) used by both the preflight raise
site and the runtime 4xx raise site, so the wording stays in sync.

### 2.9. Response caching

The LLM scorer caches verdicts by content hash - identical inputs
(same scenario outputs, same model/temperature/seed) return cached
results. Disk budget controlled by `BELT_CACHE_MAX_BYTES`
([CONFIGURATION.md → §3.3](CONFIGURATION.md#33-output-paths-and-disk-budgets)).

## 3. Multi-judge scoring

Run multiple LLM judges (different models / temperatures / dimensions
/ personas); results merge by dimension name during aggregation.

```yaml
# judges.yaml
judges:
  correctness:
    model: openai/gpt-5.4-mini
    temperature: 0.0
    dimensions: [factual_accuracy, completeness]
    system_preamble: "Focus on factual correctness."
  safety:
    model: anthropic/claude-sonnet-4-5
    temperature: 0.0
    dimensions: [safety, responsible_disclosure]
    system_preamble: "Assess security-sensitive topics."
```

```bash
belt eval scenarios/ --scorer-config judges.yaml --modes llm
```

## 4. Thresholds

Enforce per-dimension failure thresholds (percentage of checks /
verdicts passing) at aggregation time:

```bash
belt eval scenarios/ --threshold rules/execution:0          # zero failures allowed
belt eval scenarios/ --threshold rules/efficiency:20        # ≤ 20% failure
belt eval scenarios/ --threshold rules/execution:0 --threshold llm/response_quality:10
```

Format: `--threshold MODE/DIMENSION:PERCENT`.

For LLM scoring, control which categorical verdicts count as failures
**toward threshold gates** (the headline pass-rate is hardcoded to
fail on `low` / `fail` / `inconclusive` regardless; see
[§2.5](#25-verdict-scales-binary--ternary--inconclusive)):

```bash
--llm-fail-on low,fail,inconclusive   # default
--llm-fail-on low,fail                 # exempt inconclusive from threshold counts
```

Valid tokens: any from §2.5.

## 5. Reliability (pass@k and pass^k)

Run each scenario N times with `--trials N`:

```bash
belt eval scenarios/ --trials 5 --modes rules
```

Given empirical pass rate `p`, the aggregator reports for k=1, 3, 8:

- **pass@k = 1 - (1-p)^k** - at least one of k trials passes.
- **pass^k = p^k** - all k trials pass.

Per-scenario and mean values land in terminal output, `results.json`,
JUnit `<property>`s, and the Markdown report.

## 6. Score output format

Each scenario produces a `score.json` with typed, versioned payloads
per scorer (outer `ScenarioScore` envelope, inner per-scorer payload
with its own `schema_version`). Walk per-dimension results uniformly
via `iter_dimension_feedback` from `belt.scorer.payloads`.

- On-disk shape: [OUTCOMES.md → Nested scorer payloads](OUTCOMES.md#21-nested-scorer-payloads-inside-scorejson).
- Plugin scorer authoring: [PLUGGABILITY.md → §7](PLUGGABILITY.md#7-authoring-a-scorer).

## 7. Live scorer streaming

With `--progress live`, the Score phase streams real-time per-scenario
events:

| Icon | Event | Meaning |
|---|---|---|
| 🎯 | `start` | LLM judge API call begins |
| ⚡ | `cache_hit` | Result served from cache |
| 📊 | `verdict` | Per-dimension score with reasoning snippet |
| ✅/❌ | `done` | All dimensions scored |

In multi-judge consensus mode, events include the judge name prefix
(`correctness: execution: high`).

When LLM scoring is active, `score_stream.ndjson` is also written
alongside outcome files (one JSON event per line) for post-hoc
debugging:

```json
{"kind": "start",   "scenario": "security_analysis"}
{"kind": "verdict", "scenario": "security_analysis", "dimension": "execution", "score": "high", "reasoning": "..."}
{"kind": "done",    "scenario": "security_analysis", "passed": true}
```
