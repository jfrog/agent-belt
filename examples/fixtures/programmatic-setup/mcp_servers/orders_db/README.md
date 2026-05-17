# `orders_db` - stdio MCP server

Pure-stdlib Model Context Protocol server (revision 2024-11-05) with a tiny
in-memory orders database. Deterministic fixture data (orders 42, 43, 44)
lets assertions pin to specific reply substrings (`"shipped"`, `"AGB-EVAL-42"`).

| Tool | Arguments | Returns |
|---|---|---|
| `get_order` | `order_id: int` | One order by id |
| `list_orders` | `status: str?` | All orders, optionally filtered |

## Smoke test

```bash
{ echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'; \
  echo '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'; } \
  | python3 server.py
```

Should print two JSON-RPC responses.
