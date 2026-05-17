# (c) JFrog Ltd. (2026)

"""Tests for the Folio SQLite layer.

These run independently of the HTTP server - they exercise FolioDB
directly. Time-based assertions depend on order seed offsets being
stable (see data/orders_seed.json).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from folio.db import FolioDB


@pytest.fixture()
def db() -> FolioDB:
    return FolioDB()


def test_seed_loads_books_and_customers(db: FolioDB) -> None:
    books = db.search_books(None, 100)
    assert len(books) >= 30

    customer = db.get_customer("C001")
    assert customer is not None
    assert customer["name"] == "Ada Lovelace"


def test_get_order_returns_known_seed_row(db: FolioDB) -> None:
    order = db.get_order(1001)
    assert order is not None
    assert order["customer_id"] == "C001"
    assert order["isbn"] == "978-0441172719"
    assert order["status"] == "delivered"


def test_order_dates_track_seed_offsets(db: FolioDB) -> None:
    now = datetime.now(timezone.utc)
    order = db.get_order(1003)
    assert order is not None
    placed = datetime.fromisoformat(order["placed_at"])
    delta_days = (now - placed).days
    # Seed says 92 days ago; allow drift from test clock vs server clock.
    assert 90 <= delta_days <= 94


def test_place_order_decrements_stock_and_returns_total(db: FolioDB) -> None:
    isbn = "978-0441172719"  # Dune
    book_before = db.get_book(isbn)
    assert book_before is not None
    before_stock = book_before["stock_qty"]

    result = db.place_order(isbn, 2, "C001")
    assert result["status"] == "placed"
    assert result["qty"] == 2
    assert result["total_usd"] == pytest.approx(book_before["price_usd"] * 2)

    book_after = db.get_book(isbn)
    assert book_after is not None
    assert book_after["stock_qty"] == before_stock - 2


def test_place_order_reports_out_of_stock_without_decrementing(db: FolioDB) -> None:
    isbn = "978-0441478125"  # The Left Hand of Darkness - 3 in stock
    book_before = db.get_book(isbn)
    assert book_before is not None
    assert book_before["stock_qty"] == 3

    result = db.place_order(isbn, 10, "C002")
    assert result["status"] == "out_of_stock"
    assert result["available_qty"] == 3

    book_after = db.get_book(isbn)
    assert book_after is not None
    assert book_after["stock_qty"] == 3


def test_refund_flips_order_status(db: FolioDB) -> None:
    result = db.refund_order(1001, "changed my mind")
    assert result["status"] == "refunded"
    assert result["amount_usd"] == pytest.approx(18.99)
    assert db.get_order(1001)["status"] == "refunded"  # type: ignore[index]


def test_store_credit_flips_order_status_and_records_amount(db: FolioDB) -> None:
    result = db.issue_store_credit(1002, 99.98, "past 30-day window")
    assert result["status"] == "store_credit_issued"
    assert result["customer_id"] == "C002"
    assert db.get_order(1002)["status"] == "credited"  # type: ignore[index]


def test_escalation_records_ticket(db: FolioDB) -> None:
    result = db.escalate("C003", "out of self-serve window", 1003)
    assert result["status"] == "escalated"
    assert result["ticket_id"] >= 1
    assert db.get_order(1003)["status"] == "escalated"  # type: ignore[index]


def test_refund_for_unknown_order_raises(db: FolioDB) -> None:
    with pytest.raises(LookupError):
        db.refund_order(9999, "test")


def test_refund_accepts_just_placed_order(db: FolioDB) -> None:
    # L7 flow: place_order returns status='placed'; the customer
    # immediately changes their mind. The refund must succeed.
    placed = db.place_order("978-0593135204", 1, "C001")
    result = db.refund_order(placed["order_id"], "customer changed mind")
    assert result["status"] == "refunded"
    assert db.get_order(placed["order_id"])["status"] == "refunded"  # type: ignore[index]


def test_refund_rejects_already_resolved_order(db: FolioDB) -> None:
    db.refund_order(1001, "first refund")
    with pytest.raises(ValueError):
        db.refund_order(1001, "second refund attempt")


def test_reset_restores_seed_after_mutation(db: FolioDB) -> None:
    db.refund_order(1001, "test")
    db.issue_store_credit(1002, 99.98, "test")
    db.escalate("C003", "test", 1003)
    assert db.get_order(1001)["status"] == "refunded"  # type: ignore[index]
    assert db.get_order(1002)["status"] == "credited"  # type: ignore[index]
    assert db.get_order(1003)["status"] == "escalated"  # type: ignore[index]

    db.reset()

    assert db.get_order(1001)["status"] == "delivered"  # type: ignore[index]
    assert db.get_order(1002)["status"] == "delivered"  # type: ignore[index]
    assert db.get_order(1003)["status"] == "delivered"  # type: ignore[index]


def test_search_filters_by_query(db: FolioDB) -> None:
    hits = db.search_books("Stephenson", 10)
    titles = {b["title"] for b in hits}
    assert {"Snow Crash", "Anathem"}.issubset(titles)


def test_list_customers_returns_all_seed_rows(db: FolioDB) -> None:
    customers = db.list_customers(None, 100)
    ids = {c["customer_id"] for c in customers}
    assert {"C001", "C002", "C003", "C004", "C005"} == ids


def test_list_customers_filters_by_query(db: FolioDB) -> None:
    hits = db.list_customers("Hopper", 10)
    assert len(hits) == 1
    assert hits[0]["customer_id"] == "C002"

    by_email = db.list_customers("acme-books.example", 10)
    assert len(by_email) == 1
    assert by_email[0]["customer_id"] == "C005"
