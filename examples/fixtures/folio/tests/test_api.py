# (c) JFrog Ltd. (2026)

"""HTTP-level tests for Folio's REST surface.

Uses FastAPI's TestClient (sync) - no live uvicorn process needed.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from folio.server import build_app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(build_app())


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "folio"}


def test_admin_reset_restores_seed(client: TestClient) -> None:
    # mutate, then reset via the admin endpoint, then verify seed state.
    r = client.post("/api/orders/1001/refund", json={"reason": "test"})
    assert r.status_code == 200
    assert client.get("/api/orders/1001").json()["status"] == "refunded"

    r = client.post("/api/admin/reset")
    assert r.status_code == 200
    assert r.json() == {"status": "reset", "service": "folio"}

    assert client.get("/api/orders/1001").json()["status"] == "delivered"


def test_books_filtered(client: TestClient) -> None:
    r = client.get("/api/books", params={"q": "Stephenson", "limit": 5})
    assert r.status_code == 200
    payload = r.json()
    titles = {b["title"] for b in payload["books"]}
    assert "Snow Crash" in titles
    assert "Anathem" in titles


def test_list_customers_returns_all(client: TestClient) -> None:
    r = client.get("/api/customers")
    assert r.status_code == 200
    payload = r.json()
    assert payload["count"] == 5
    ids = {c["customer_id"] for c in payload["customers"]}
    assert {"C001", "C002", "C003", "C004", "C005"} == ids


def test_list_customers_filters_by_q(client: TestClient) -> None:
    r = client.get("/api/customers", params={"q": "Turing"})
    payload = r.json()
    assert payload["count"] == 1
    assert payload["customers"][0]["customer_id"] == "C003"


def test_get_unknown_book_returns_404(client: TestClient) -> None:
    r = client.get("/api/books/nope-such-isbn")
    assert r.status_code == 404


def test_place_order_succeeds(client: TestClient) -> None:
    r = client.post(
        "/api/orders",
        json={"isbn": "978-0441172719", "qty": 1, "customer_id": "C001"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "placed"
    assert body["isbn"] == "978-0441172719"


def test_place_order_out_of_stock(client: TestClient) -> None:
    r = client.post(
        "/api/orders",
        json={"isbn": "978-0441478125", "qty": 50, "customer_id": "C002"},
    )
    assert r.status_code == 201  # 201 because we still return a structured response
    body = r.json()
    assert body["status"] == "out_of_stock"
    assert body["available_qty"] == 3


def test_refund_existing_order(client: TestClient) -> None:
    r = client.post(
        "/api/orders/1001/refund",
        json={"reason": "changed my mind"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "refunded"
    assert body["amount_usd"] == 18.99


def test_refund_unknown_order_returns_404(client: TestClient) -> None:
    r = client.post(
        "/api/orders/9999/refund",
        json={"reason": "test"},
    )
    assert r.status_code == 404


def test_escalation_creates_ticket(client: TestClient) -> None:
    r = client.post(
        "/api/escalations",
        json={
            "customer_id": "C003",
            "reason": "outside policy",
            "order_id": 1003,
        },
    )
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "escalated"
    assert body["ticket_id"] >= 1
