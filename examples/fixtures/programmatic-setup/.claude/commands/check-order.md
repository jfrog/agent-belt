---
description: Look up a single order by id through the orders_db MCP server and report it in a strict pipe-separated format that downstream parsers can consume.
---

Look up the order whose id is provided as the argument. Use the
`mcp__orders_db__get_order` tool. Reply with **exactly one line** in this
strict pipe-separated format and nothing else - no preamble, no markdown,
no trailing skill marker:

```text
ORDER|<id>|<customer>|<status>|<tracking>
```

Use the value `none` for fields that are null. If no order id was
provided or the lookup failed, reply with `ORDER|error|<reason>`.

Argument: $ARGUMENTS
