# (c) JFrog Ltd. (2026)

"""orders_db - a stdio MCP server with deterministic in-memory order data.

Implements just enough of the Model Context Protocol (revision 2024-11-05) to
satisfy any compliant client:

  - initialize          -> capabilities + server info
  - notifications/initialized
  - tools/list          -> two tools (get_order, list_orders)
  - tools/call          -> execute the requested tool

JSON-RPC 2.0 messages are line-delimited on stdin/stdout. Pure stdlib, so
the file runs anywhere Python does with no install step.
"""

from __future__ import annotations

import json
import sys
from typing import Any

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "orders_db"
SERVER_VERSION = "1.0.0"

# Deterministic fixture data. Stable so scenario assertions (e.g. "the reply
# must contain 'shipped' for order 42") hold across runs.
_ORDERS: dict[int, dict[str, Any]] = {
    42: {
        "order_id": 42,
        "customer": "Ada Lovelace",
        "status": "shipped",
        "tracking_number": "AGB-EVAL-42",
        "items": ["analytical-engine-mk1"],
    },
    43: {
        "order_id": 43,
        "customer": "Alan Turing",
        "status": "processing",
        "tracking_number": None,
        "items": ["bombe-rotor-set", "enigma-replica"],
    },
    44: {
        "order_id": 44,
        "customer": "Grace Hopper",
        "status": "delivered",
        "tracking_number": "AGB-EVAL-44",
        "items": ["cobol-handbook"],
    },
}

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_order",
        "description": (
            "Look up a single order by its numeric order_id. Returns the order's "
            "customer, status, tracking number, and items. Use this when the user "
            "asks about a specific order."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "integer",
                    "description": "The numeric order id to look up.",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "list_orders",
        "description": (
            "List all orders, optionally filtered by status. Returns an array of "
            "order summaries. Use this when the user asks about orders in general "
            "rather than a specific one."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Optional status filter (e.g. 'shipped').",
                }
            },
        },
    },
]


def _result(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _text_content(payload: Any) -> dict[str, Any]:
    """Wrap a Python value as an MCP `tools/call` content item."""
    return {"content": [{"type": "text", "text": json.dumps(payload, indent=2)}]}


def _handle_initialize(request_id: Any) -> dict[str, Any]:
    return _result(
        request_id,
        {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        },
    )


def _handle_tools_list(request_id: Any) -> dict[str, Any]:
    return _result(request_id, {"tools": _TOOLS})


def _handle_tools_call(request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    args = params.get("arguments") or {}

    if name == "get_order":
        order_id = args.get("order_id")
        if not isinstance(order_id, int):
            return _result(request_id, _text_content({"error": "order_id must be an integer"}))
        order = _ORDERS.get(order_id)
        if order is None:
            return _result(request_id, _text_content({"error": f"order {order_id} not found"}))
        return _result(request_id, _text_content(order))

    if name == "list_orders":
        status = args.get("status")
        orders = list(_ORDERS.values())
        if isinstance(status, str):
            orders = [o for o in orders if o["status"] == status]
        return _result(request_id, _text_content({"orders": orders, "count": len(orders)}))

    return _error(request_id, -32601, f"Unknown tool: {name}")


def _dispatch(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")
    params = request.get("params") or {}

    if method == "initialize":
        return _handle_initialize(request_id)
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return _handle_tools_list(request_id)
    if method == "tools/call":
        return _handle_tools_call(request_id, params)

    if request_id is None:
        return None
    return _error(request_id, -32601, f"Method not found: {method}")


def main() -> int:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = _dispatch(request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
