import json
import sqlite3
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.processor import MAX_CONTENT_LENGTH, fingerprint

VALID = {
    "event_id": "evt_1",
    "tenant_id": "tenant_a",
    "content": "Customer jane@example.com SSN 123-45-6789 opened a ticket",
}
HEADERS = {"X-Idempotency-Key": "idem_1"}


def ok_handler(sent):
    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"accepted": True})
    return handler


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "api.db")
    monkeypatch.setenv("WEBHOOK_DB_PATH", path)
    return path


def start_app(monkeypatch, handler):
    """TestClient runs the real lifespan (DB, worker, process pool); only the partner is faked."""
    monkeypatch.setattr(
        main, "create_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    return TestClient(main.app)


@pytest.fixture
def api(db_path, monkeypatch):
    sent = []
    with start_app(monkeypatch, ok_handler(sent)) as client:
        client.sent = sent
        yield client


def rows(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM events ORDER BY id")]
    finally:
        conn.close()


def wait_for_status(db_path, expected, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = [r["status"] for r in rows(db_path)]
        if current and all(s == expected for s in current):
            return
        time.sleep(0.05)
    raise AssertionError(f"rows did not reach {expected!r}: {rows(db_path)}")


def test_health(api):
    assert api.get("/health").json() == {"status": "ok"}


def test_new_event_is_accepted_with_202(api):
    r = api.post("/webhooks/ingest", json=VALID, headers=HEADERS)
    assert r.status_code == 202
    assert r.json() == {"status": "accepted", "event_id": "evt_1"}


def test_duplicate_key_returns_200_and_creates_no_second_row(api, db_path):
    assert api.post("/webhooks/ingest", json=VALID, headers=HEADERS).status_code == 202
    r = api.post("/webhooks/ingest", json=VALID, headers=HEADERS)
    assert r.status_code == 200
    assert r.json() == {"status": "duplicate", "event_id": "evt_1"}
    wait_for_status(db_path, "delivered")
    assert len(rows(db_path)) == 1
    assert len(api.sent) == 1


def test_same_key_from_different_tenants_are_separate_events(api, db_path):
    for tenant in ("tenant_a", "tenant_b"):
        r = api.post("/webhooks/ingest", json={**VALID, "tenant_id": tenant}, headers=HEADERS)
        assert r.status_code == 202
    assert len(rows(db_path)) == 2


def test_pii_is_scrubbed_before_the_ack(api, db_path):
    api.post("/webhooks/ingest", json=VALID, headers=HEADERS)
    stored = rows(db_path)[0]["cleaned_text"]  # read immediately, before background work
    assert "jane@example.com" not in stored and "123-45-6789" not in stored


def test_event_is_fingerprinted_and_delivered_in_background(api, db_path):
    api.post("/webhooks/ingest", json=VALID, headers=HEADERS)
    wait_for_status(db_path, "delivered")
    row = rows(db_path)[0]
    assert row["fingerprint"] == fingerprint(row["cleaned_text"])
    assert row["attempts"] == 1
    assert api.sent == [
        {"event_id": "evt_1", "text": row["cleaned_text"], "fingerprint": row["fingerprint"]}
    ]


@pytest.mark.parametrize("headers", [{}, {"X-Idempotency-Key": ""}, {"X-Idempotency-Key": "   "}])
def test_missing_or_blank_idempotency_key_is_400(api, headers):
    r = api.post("/webhooks/ingest", json=VALID, headers=headers)
    assert r.status_code == 400


@pytest.mark.parametrize(
    "body, headers",
    [
        ({k: v for k, v in VALID.items() if k != "event_id"}, HEADERS),
        ({**VALID, "event_id": ""}, HEADERS),
        ({**VALID, "tenant_id": "t" * 256}, HEADERS),
        ({**VALID, "content": "x" * (MAX_CONTENT_LENGTH + 1)}, HEADERS),
        ({**VALID, "content": 123}, HEADERS),
        (VALID, {"X-Idempotency-Key": "k" * 256}),
    ],
    ids=["missing-field", "empty-event-id", "long-tenant", "oversized-content", "wrong-type", "long-key"],
)
def test_invalid_requests_are_422_and_not_stored(api, db_path, body, headers):
    r = api.post("/webhooks/ingest", json=body, headers=headers)
    assert r.status_code == 422
    assert rows(db_path) == []


def test_content_at_size_limit_is_accepted(api):
    body = {**VALID, "content": "x" * MAX_CONTENT_LENGTH}
    assert api.post("/webhooks/ingest", json=body, headers=HEADERS).status_code == 202


def test_database_error_returns_503_without_internals(api, monkeypatch):
    async def broken_claim(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(main, "claim_event", broken_claim)
    r = api.post("/webhooks/ingest", json=VALID, headers=HEADERS)
    assert r.status_code == 503
    assert "locked" not in r.text


def test_pending_event_is_delivered_after_restart(db_path, monkeypatch):
    # Session 1: the partner call blows up unexpectedly, so the row is left 'pending'
    # (the same state a crash right after the ack would leave).
    def exploding(request):
        raise RuntimeError("process died mid-delivery")

    with start_app(monkeypatch, exploding) as client:
        assert client.post("/webhooks/ingest", json=VALID, headers=HEADERS).status_code == 202
        deadline = time.monotonic() + 10
        while rows(db_path)[0]["fingerprint"] is None and time.monotonic() < deadline:
            time.sleep(0.05)
    assert rows(db_path)[0]["status"] == "pending"

    # Session 2: a fresh process recovers the row at startup and delivers it once.
    sent = []
    with start_app(monkeypatch, ok_handler(sent)):
        wait_for_status(db_path, "delivered")
    assert [e["event_id"] for e in sent] == ["evt_1"]
