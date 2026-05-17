---
description: Look up a Folio customer by id and report their order count, in a strict pipe-separated format that downstream parsers can consume.
---

Look up the Folio customer whose id is provided as the argument.
Use `mcp__folio__get_customer` to fetch the customer, then
`mcp__folio__list_customer_orders` to fetch their orders. Reply
with **exactly one line** in this strict format and nothing else -
no preamble, no markdown, no skill marker:

```text
LOOKUP|<customer_id>|<name>|orders=<count>|tag=folio-ops-v1
```

The literal `tag=folio-ops-v1` is what downstream tooling matches
on to confirm this plugin's command produced the reply - never
omit it, never paraphrase it. If the customer is unknown reply
with `LOOKUP|<customer_id>|error|unknown|tag=folio-ops-v1`.

Argument: $ARGUMENTS
