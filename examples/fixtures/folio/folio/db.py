# (c) JFrog Ltd. (2026)

"""SQLite access for the Folio bookstore SaaS.

The DB is rebuilt from `data/seed.sql` + `data/orders_seed.json` on every
server start so scenario assertions are deterministic regardless of when
the demo is run.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

_FIXTURE_ROOT = Path(__file__).resolve().parent.parent
_SEED_SQL = _FIXTURE_ROOT / "data" / "seed.sql"
_SEED_ORDERS = _FIXTURE_ROOT / "data" / "orders_seed.json"


class FolioDB:
    """Thin wrapper around an in-memory SQLite database.

    A single connection is shared by FastAPI + FastMCP (uvicorn runs both in
    one event loop). SQLite handles concurrent reads in the same connection;
    writes are serialized at the sqlite3 layer which is fine for the demo.
    """

    def __init__(self, path: str = ":memory:") -> None:
        # check_same_thread=False because FastAPI's threadpool may invoke
        # sync handlers from worker threads.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._seed()

    def _seed(self) -> None:
        with self._conn:
            self._conn.executescript(_SEED_SQL.read_text(encoding="utf-8"))
            self._seed_orders()

    def _seed_orders(self) -> None:
        now = datetime.now(timezone.utc)
        with _SEED_ORDERS.open(encoding="utf-8") as handle:
            rows = json.load(handle)
        for row in rows:
            placed = (now - timedelta(days=row["placed_days_ago"])).isoformat()
            delivered = (
                (now - timedelta(days=row["delivered_days_ago"])).isoformat()
                if row["delivered_days_ago"] is not None
                else None
            )
            self._conn.execute(
                """
                INSERT INTO orders
                  (order_id, customer_id, isbn, qty, unit_price_usd, status,
                   placed_at, delivered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["order_id"],
                    row["customer_id"],
                    row["isbn"],
                    row["qty"],
                    row["unit_price_usd"],
                    row["status"],
                    placed,
                    delivered,
                ),
            )

    def reset(self) -> None:
        """Wipe and re-seed the database back to its initial deterministic state.

        Used by the demo's `POST /admin/reset` endpoint (and the
        `python -m folio.reset` CLI) so an eval run starts from a clean
        snapshot regardless of what mutated the in-memory state previously.
        """
        with self._conn:
            for table in ("refunds", "store_credits", "escalations", "orders", "customers", "books"):
                self._conn.execute(f"DROP TABLE IF EXISTS {table}")
        self._seed()

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # ---- read helpers ----------------------------------------------------

    def search_books(self, query: str | None, limit: int) -> list[dict[str, Any]]:
        sql = "SELECT * FROM books"
        params: tuple[Any, ...] = ()
        if query:
            sql += " WHERE title LIKE ? OR author LIKE ? OR category LIKE ?"
            like = f"%{query}%"
            params = (like, like, like)
        sql += " ORDER BY title ASC LIMIT ?"
        params += (limit,)
        with self.cursor() as cur:
            cur.execute(sql, params)
            return [_row_to_dict(r) for r in cur.fetchall()]

    def get_book(self, isbn: str) -> dict[str, Any] | None:
        with self.cursor() as cur:
            cur.execute("SELECT * FROM books WHERE isbn = ?", (isbn,))
            row = cur.fetchone()
        return _row_to_dict(row) if row else None

    def get_customer(self, customer_id: str) -> dict[str, Any] | None:
        with self.cursor() as cur:
            cur.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
            row = cur.fetchone()
        return _row_to_dict(row) if row else None

    def list_customers(self, query: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql = "SELECT * FROM customers"
        params: tuple[Any, ...] = ()
        if query:
            sql += " WHERE name LIKE ? OR email LIKE ? OR customer_id LIKE ?"
            like = f"%{query}%"
            params = (like, like, like)
        sql += " ORDER BY customer_id ASC LIMIT ?"
        params += (limit,)
        with self.cursor() as cur:
            cur.execute(sql, params)
            return [_row_to_dict(r) for r in cur.fetchall()]

    def get_order(self, order_id: int) -> dict[str, Any] | None:
        with self.cursor() as cur:
            cur.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
            row = cur.fetchone()
        return _row_to_dict(row) if row else None

    def list_orders_for_customer(self, customer_id: str) -> list[dict[str, Any]]:
        with self.cursor() as cur:
            cur.execute(
                "SELECT * FROM orders WHERE customer_id = ? ORDER BY placed_at DESC",
                (customer_id,),
            )
            return [_row_to_dict(r) for r in cur.fetchall()]

    # ---- write helpers ---------------------------------------------------

    def place_order(self, isbn: str, qty: int, customer_id: str) -> dict[str, Any]:
        book = self.get_book(isbn)
        if book is None:
            raise LookupError(f"book {isbn!r} not found")
        customer = self.get_customer(customer_id)
        if customer is None:
            raise LookupError(f"customer {customer_id!r} not found")
        if qty <= 0:
            raise ValueError("qty must be > 0")
        if book["stock_qty"] < qty:
            return {
                "status": "out_of_stock",
                "isbn": isbn,
                "requested_qty": qty,
                "available_qty": book["stock_qty"],
            }
        now = datetime.now(timezone.utc).isoformat()
        with self.cursor() as cur:
            cur.execute(
                "UPDATE books SET stock_qty = stock_qty - ? WHERE isbn = ?",
                (qty, isbn),
            )
            cur.execute(
                """
                INSERT INTO orders
                  (customer_id, isbn, qty, unit_price_usd, status, placed_at)
                VALUES (?, ?, ?, ?, 'placed', ?)
                """,
                (customer_id, isbn, qty, book["price_usd"], now),
            )
            order_id = cur.lastrowid
        return {
            "status": "placed",
            "order_id": order_id,
            "isbn": isbn,
            "qty": qty,
            "unit_price_usd": book["price_usd"],
            "total_usd": round(book["price_usd"] * qty, 2),
            "customer_id": customer_id,
            "placed_at": now,
        }

    def refund_order(self, order_id: int, reason: str) -> dict[str, Any]:
        order = self.get_order(order_id)
        if order is None:
            raise LookupError(f"order {order_id} not found")
        # `placed` is accepted because real e-commerce flows let a customer
        # cancel before fulfilment; L7 (place-then-refund) depends on this.
        # Already-resolved statuses cannot be refunded again.
        if order["status"] not in {"placed", "shipped", "delivered"}:
            raise ValueError(f"cannot refund order in status {order['status']!r}")
        amount = round(order["unit_price_usd"] * order["qty"], 2)
        now = datetime.now(timezone.utc).isoformat()
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO refunds (order_id, amount_usd, reason, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (order_id, amount, reason, now),
            )
            refund_id = cur.lastrowid
            cur.execute(
                "UPDATE orders SET status = 'refunded' WHERE order_id = ?",
                (order_id,),
            )
        return {
            "status": "refunded",
            "refund_id": refund_id,
            "order_id": order_id,
            "amount_usd": amount,
            "reason": reason,
            "created_at": now,
        }

    def issue_store_credit(self, order_id: int, amount_usd: float, reason: str) -> dict[str, Any]:
        order = self.get_order(order_id)
        if order is None:
            raise LookupError(f"order {order_id} not found")
        if amount_usd <= 0:
            raise ValueError("amount_usd must be > 0")
        now = datetime.now(timezone.utc).isoformat()
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO store_credits
                  (order_id, customer_id, amount_usd, reason, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (order_id, order["customer_id"], amount_usd, reason, now),
            )
            credit_id = cur.lastrowid
            cur.execute(
                "UPDATE orders SET status = 'credited' WHERE order_id = ?",
                (order_id,),
            )
        return {
            "status": "store_credit_issued",
            "credit_id": credit_id,
            "order_id": order_id,
            "customer_id": order["customer_id"],
            "amount_usd": amount_usd,
            "reason": reason,
            "created_at": now,
        }

    def escalate(self, customer_id: str, reason: str, order_id: int | None) -> dict[str, Any]:
        if not self.get_customer(customer_id):
            raise LookupError(f"customer {customer_id!r} not found")
        if order_id is not None and not self.get_order(order_id):
            raise LookupError(f"order {order_id} not found")
        now = datetime.now(timezone.utc).isoformat()
        with self.cursor() as cur:
            cur.execute(
                """
                INSERT INTO escalations
                  (customer_id, order_id, reason, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (customer_id, order_id, reason, now),
            )
            ticket_id = cur.lastrowid
            if order_id is not None:
                cur.execute(
                    "UPDATE orders SET status = 'escalated' WHERE order_id = ?",
                    (order_id,),
                )
        return {
            "status": "escalated",
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "order_id": order_id,
            "reason": reason,
            "created_at": now,
        }


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None
