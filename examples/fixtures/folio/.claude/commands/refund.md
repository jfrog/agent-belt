---
description: Issue a refund for a Folio order by id with a reason. Calls the Folio MCP server's refund_order tool directly and emits a strict pipe-separated reply that downstream parsers can consume.
---

Issue a refund for the Folio order whose id is given as the first
argument, with the rest of the argument string treated as the
refund reason. Use the `mcp__folio__refund_order` tool exactly
once. Reply with **exactly one line** in this strict format and
nothing else - no preamble, no markdown, no skill marker:

```text
REFUND|<order_id>|<status>|$<amount_usd>|tag=folio-cmd-v1
```

Use the literal string `tag=folio-cmd-v1` at the end - downstream
tooling matches on it to confirm this slash command produced the
reply. Never omit it, never paraphrase it. If the refund fails,
reply with `REFUND|<order_id>|error|<reason>|tag=folio-cmd-v1`.

Argument: $ARGUMENTS
