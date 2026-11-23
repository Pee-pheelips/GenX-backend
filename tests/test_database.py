import asyncio
import sqlite3

import pytest

from app.database import (
    claim_event,
    connect,
    fetch_pending,
    get_event,
    init_db,
    mark_delivered,
    mark_failed,
    set_fingerprint,
)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "events.db")


@pytest.fixture
async def db(db_path):
    conn = await connect(db_path)
    await init_db(conn)
    yield conn
    await conn.close()


async def _claim(db, key="idem_1", tenant="tenant_a", event_id="evt_1"):
    return await claim_event(db, key, event_id, tenant, "cleaned text")


async def test_connection_pragmas(db):
    async with db.execute("PRAGMA journal_mode") as cur:
        assert (await cur.fetchone())[0] == "wal"
    async with db.execute("PRAGMA busy_timeout") as cur:
        assert (await cur.fetchone())[0] == 5000
    async with db.execute("PRAGMA synchronous") as cur:
        assert (await cur.fetchone())[0] == 1  # NORMAL


async def test_init_db_is_idempotent(db):
    await init_db(db)  # second call must not fail or wipe data
    assert await _claim(db) is not None


async def test_init_db_rejects_legacy_schema(db_path):
    legacy = sqlite3.connect(db_path)
    legacy.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, event_id TEXT)")
    legacy.close()
    conn = await connect(db_path)
    try:
        with pytest.raises(RuntimeError, match="old schema"):
            await init_db(conn)
    finally:
        await conn.close()


async def test_claim_returns_id_then_none_for_duplicate(db):
    first = await _claim(db)
    assert isinstance(first, int)
    assert await _claim(db) is None
    async with db.execute("SELECT COUNT(*) FROM events") as cur:
        assert (await cur.fetchone())[0] == 1


async def test_concurrent_duplicate_claims_have_exactly_one_winner(db):
    results = await asyncio.gather(*(_claim(db) for _ in range(50)))
    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    async with db.execute("SELECT COUNT(*) FROM events") as cur:
        assert (await cur.fetchone())[0] == 1


async def test_concurrent_claims_across_connections_have_one_winner(db, db_path):
    # Separate connections behave like separate processes (e.g. uvicorn --workers 4):
    # the UNIQUE constraint still guarantees a single winner.
    others = [await connect(db_path) for _ in range(3)]
    try:
        conns = [db, *others]
        results = await asyncio.gather(*(_claim(conns[i % 4]) for i in range(40)))
        assert sum(r is not None for r in results) == 1
    finally:
        for c in others:
            await c.close()


async def test_idempotency_key_is_scoped_per_tenant(db):
    assert await _claim(db, key="shared", tenant="tenant_a") is not None
    assert await _claim(db, key="shared", tenant="tenant_b") is not None
    assert await _claim(db, key="shared", tenant="tenant_a") is None


async def test_new_event_defaults(db):
    row = await get_event(db, await _claim(db))
    assert row["status"] == "pending"
    assert row["fingerprint"] is None
    assert row["attempts"] == 0
    assert row["cleaned_text"] == "cleaned text"


async def test_delivery_lifecycle_updates_only_target_row(db):
    a = await _claim(db, key="k1", event_id="evt_same")
    b = await _claim(db, key="k2", event_id="evt_same")  # same event_id, different key

    await set_fingerprint(db, a, "abc123")
    await mark_delivered(db, a, attempts=2)

    row_a, row_b = await get_event(db, a), await get_event(db, b)
    assert (row_a["status"], row_a["fingerprint"], row_a["attempts"]) == ("delivered", "abc123", 2)
    # Regression for the original bug: updates by event_id touched every matching row.
    assert row_b["status"] == "pending"


async def test_mark_failed_records_error_and_truncates(db):
    row_id = await _claim(db)
    await mark_failed(db, row_id, attempts=6, error="x" * 2000)
    row = await get_event(db, row_id)
    assert row["status"] == "failed"
    assert row["attempts"] == 6
    assert len(row["last_error"]) == 500


async def test_fetch_pending_returns_only_pending_in_order(db):
    p1 = await _claim(db, key="k1")
    done = await _claim(db, key="k2")
    failed = await _claim(db, key="k3")
    p2 = await _claim(db, key="k4")
    await mark_delivered(db, done, attempts=1)
    await mark_failed(db, failed, attempts=6, error="boom")
    assert await fetch_pending(db) == [p1, p2]


async def test_status_constraint_rejects_unknown_values(db):
    row_id = await _claim(db)
    with pytest.raises(sqlite3.IntegrityError):
        await db.execute("UPDATE events SET status = 'bogus' WHERE id = ?", (row_id,))
