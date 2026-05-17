# Agent error-type fixtures

One subdirectory per canonical token in
`belt.entities.ERROR_TYPES`, with one file per registered agent
inside (`<agent-name>.ndjson` or `.txt`). Each fixture is the smallest
possible raw output from the corresponding CLI that satisfies both:

1. The adapter's `fetch_results` sets `TurnOutput.has_error = True`.
2. The text contains a recognisable substring (per
   `belt.agent.error_types.classify_error`) so the framework
   classifies it as the matching canonical token.

## Layout

```text
tests/agent/fixtures/
├── auth_failure/   → authentication_failed
├── rate_limited/   → rate_limited
├── timeout/        → timeout
└── refused/        → refused
```

The `unknown` token has no fixture directory: it is the fallback when
`has_error=true` but no signal classified, so there is no canonical
shape to pin per agent.

Fixtures are deliberately minimal - they encode only the result-style
event the adapter needs to flip `is_error` plus an assistant/text
event carrying the recognisable substring. Real-world CLI output is
much noisier; the contract is "any output whose semantic shape
includes these two pieces should classify".

These fixtures back the parity test `test_classifies_error_type` in
`tests/agent/test_agent_parity.py`. Failure to classify any canonical
type is a hard fail of the adapter contract - taxonomy parity is part
of the contract every adapter must satisfy, not optional polish.

## Adding a new agent

For each error-type subdirectory:

1. Read your adapter's `fetch_results` to identify (a) the event(s)
   it uses to set `has_error=true`, and (b) where it accumulates
   `reply_text` from.
2. Hand-craft the smallest input that triggers both. Refer to
   neighbouring fixtures for the canonical substrings (e.g.
   "Not logged in · Please run /login" for `auth_failure`,
   "HTTP 429 Too Many Requests" for `rate_limited`).
3. Add the file as `<agent-name>.ndjson` (or `.txt` for non-NDJSON
   formats). The parity test discovers files by glob.

## Adding a new canonical token

If you add a new token to `belt.entities.ERROR_TYPES`:

1. Create `tests/agent/fixtures/<dirname>/` with one fixture per
   registered agent.
2. Add the `(dirname, token)` pair to the parametrize list in
   `test_classifies_error_type`.
3. Document the token in `docs/glossary/OUTCOMES.md` under the
   "Agent runtime errors" table - the doc-parity test
   (`tests/test_error_types_doc_parity.py`) enforces this.
