import logging
import sqlite3
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from app.database import claim_event, connect, init_db
from app.downstream import create_client
from app.processor import MAX_CONTENT_LENGTH, scrub
from app.worker import DeliveryWorker

logger = logging.getLogger(__name__)

MAX_ID_LENGTH = 255


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = await connect()
    client = create_client()
    worker = DeliveryWorker(db, client)
    try:
        await init_db(db)
        await worker.start()  # warms the process pool and re-queues 'pending' rows
        app.state.db = db
        app.state.worker = worker
        yield
    finally:
        # Reverse order. Events still in flight stay 'pending' and are recovered on next start.
        await worker.stop()
        await client.aclose()
        await db.close()


app = FastAPI(title="Webhook Ingestion Service", lifespan=lifespan)


class WebhookPayload(BaseModel):
    event_id: str = Field(min_length=1, max_length=MAX_ID_LENGTH)
    tenant_id: str = Field(min_length=1, max_length=MAX_ID_LENGTH)
    # Bounded so the inline PII scrub stays cheap on the event loop.
    content: str = Field(max_length=MAX_CONTENT_LENGTH)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/webhooks/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest_webhook(
    payload: WebhookPayload,
    request: Request,
    response: Response,
    x_idempotency_key: Annotated[str | None, Header(max_length=MAX_ID_LENGTH)] = None,
):
    """
    Acknowledges fast: the only work before replying is a cheap regex scrub and one
    atomic INSERT. Fingerprinting and downstream delivery happen in the background.
    """
    if not x_idempotency_key or not x_idempotency_key.strip():
        raise HTTPException(status_code=400, detail="Missing X-Idempotency-Key header")

    # Scrub before persisting so raw PII never reaches the database.
    cleaned_text = scrub(payload.content)

    try:
        row_id = await claim_event(
            request.app.state.db,
            x_idempotency_key,
            payload.event_id,
            payload.tenant_id,
            cleaned_text,
        )
    except sqlite3.Error:
        logger.exception("Failed to record webhook %s", payload.event_id)
        # Safe for the sender to retry: the claim is idempotent.
        raise HTTPException(
            status_code=503, detail="Temporarily unable to accept webhook; please retry"
        )

    if row_id is None:
        # Already accepted earlier (or concurrently). Acknowledge without reprocessing.
        response.status_code = status.HTTP_200_OK
        return {"status": "duplicate", "event_id": payload.event_id}

    # The row is committed, so the event survives a crash from here on.
    request.app.state.worker.enqueue(row_id)
    return {"status": "accepted", "event_id": payload.event_id}
