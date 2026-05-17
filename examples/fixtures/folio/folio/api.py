# (c) JFrog Ltd. (2026)

"""FastAPI REST surface for Folio.

Mirrors the MCP tool surface so developers can poke at Folio with curl in
the demo (`curl localhost:8765/api/books?q=dune`) before wiring an agent.
The MCP layer and the REST layer share one `FolioDB` instance.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from folio.db import FolioDB
from pydantic import BaseModel, Field


class PlaceOrderRequest(BaseModel):
    isbn: str
    qty: int = Field(gt=0)
    customer_id: str


class RefundRequest(BaseModel):
    reason: str


class StoreCreditRequest(BaseModel):
    amount_usd: float = Field(gt=0)
    reason: str


class EscalationRequest(BaseModel):
    customer_id: str
    reason: str
    order_id: int | None = None


def build_router(db: FolioDB) -> APIRouter:
    """Return an APIRouter bound to a shared FolioDB instance."""
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "folio"}

    @router.post("/admin/reset")
    def admin_reset() -> dict[str, Any]:
        """Wipe and re-seed the in-memory DB.

        Folio is in-memory and stateful across the process lifetime;
        any MCP client (IDE agent, prior eval, ad-hoc curl) that calls
        a mutating tool (`refund_order`, `issue_store_credit`,
        `escalate_to_human`, `place_order`) leaves the DB in a state
        that breaks scenario assumptions. Call this endpoint before
        every eval invocation to guarantee deterministic seed state.
        """
        db.reset()
        return {"status": "reset", "service": "folio"}

    @router.get("/books")
    def list_books(
        q: str | None = Query(default=None, description="Free-text search."),
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict[str, Any]:
        books = db.search_books(q, limit)
        return {"books": books, "count": len(books)}

    @router.get("/books/{isbn}")
    def get_book(isbn: str) -> dict[str, Any]:
        book = db.get_book(isbn)
        if book is None:
            raise HTTPException(status_code=404, detail=f"book {isbn!r} not found")
        return book

    @router.get("/customers")
    def list_customers(
        q: str | None = Query(default=None, description="Free-text search."),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        customers = db.list_customers(q, limit)
        return {"customers": customers, "count": len(customers)}

    @router.get("/customers/{customer_id}")
    def get_customer(customer_id: str) -> dict[str, Any]:
        customer = db.get_customer(customer_id)
        if customer is None:
            raise HTTPException(status_code=404, detail=f"customer {customer_id!r} not found")
        return customer

    @router.get("/customers/{customer_id}/orders")
    def list_customer_orders(customer_id: str) -> dict[str, Any]:
        if db.get_customer(customer_id) is None:
            raise HTTPException(status_code=404, detail=f"customer {customer_id!r} not found")
        orders = db.list_orders_for_customer(customer_id)
        return {"orders": orders, "count": len(orders)}

    @router.get("/orders/{order_id}")
    def get_order(order_id: int) -> dict[str, Any]:
        order = db.get_order(order_id)
        if order is None:
            raise HTTPException(status_code=404, detail=f"order {order_id} not found")
        return order

    @router.post("/orders", status_code=201)
    def place_order(payload: PlaceOrderRequest) -> dict[str, Any]:
        try:
            return db.place_order(payload.isbn, payload.qty, payload.customer_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/orders/{order_id}/refund")
    def refund_order(order_id: int, payload: RefundRequest) -> dict[str, Any]:
        try:
            return db.refund_order(order_id, payload.reason)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/orders/{order_id}/store-credit")
    def issue_store_credit(order_id: int, payload: StoreCreditRequest) -> dict[str, Any]:
        try:
            return db.issue_store_credit(order_id, payload.amount_usd, payload.reason)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/escalations", status_code=201)
    def escalate(payload: EscalationRequest) -> dict[str, Any]:
        try:
            return db.escalate(payload.customer_id, payload.reason, payload.order_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router
