---
description: Generate a one-line aggregate report of every order in orders_db, in a strict colon-separated format that downstream parsers can consume.
---

Use the `mcp__orders_db__list_orders` tool with no arguments to retrieve
all orders. Reply with **exactly one line** in this strict format and
nothing else - no preamble, no markdown, no trailing skill marker:

```text
REPORT|total=<count>|shipped=<n>|processing=<n>|delivered=<n>|tag=orders-pack-v1
```

The literal `tag=orders-pack-v1` at the end is what downstream tooling
matches on to confirm this plugin's command was the one that produced
the report - never omit it and never paraphrase it.
