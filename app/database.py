import os

import aiosqlite

DB_PATH = os.getenv("WEBHOOK_DB_PATH", "webhooks.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key TEXT NOT NULL,
    event_id        TEXT NOT NULL,
    tenant_id       TEXT NOT NULL,
    cleaned_text    TEXT NOT NULL,
    fingerprint     TEXT,  -- NULL until computed in the background, after the ack
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'delivered', 'failed')),
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- The idempotency gate. Scoped per tenant so one tenant cannot suppress
    -- another tenant's event by reusing its key.
    UNIQUE (tenant_id, idempotency_key)
)
"""

MAX_ERROR_LENGTH = 500


async def connect(db_path: str | None = None) -> aiosqlite.Connection:
    """
    Opens the single long-lived connection the service shares.

    isolation_level=None puts sqlite3 in autocommit mode: every statement below is
    its own atomic transaction. That matters because coroutines share this one
    connection; a multi-statement transaction spanning several awaits could have
    another coroutine's statements interleaved into it.
    """
    path = db_path or os.getenv("WEBHOOK_DB_PATH", DB_PATH)
    db = await aiosqlite.connect(path, isolation_level=None)
    db.row_factory = aiosqlite.Row
    # WAL: readers (e.g. a status query or the test harness) never block the writer.
    await db.execute("PRAGMA journal_mode=WAL")
    # Durable across process crashes; only an OS crash/power loss can lose the last commits.
    await db.execute("PRAGMA synchronous=NORMAL")
    # Wait instead of failing with "database is locked" when another connection or
    # process holds the write lock (relevant for --workers > 1).
    await db.execute("PRAGMA busy_timeout=5000")
    return db


async def init_db(db: aiosqlite.Connection) -> None:
    async with db.execute("PRAGMA table_info(events)") as cur:
        columns = {row["name"] for row in await cur.fetchall()}
    if columns and "idempotency_key" not in columns:
        raise RuntimeError(
            "Existing 'events' table uses the old schema (no idempotency_key). "
            "Delete the database file or migrate it before starting the service."
        )
    await db.execute(SCHEMA)


async def claim_event(
    db: aiosqlite.Connection,
    idempotency_key: str,
    event_id: str,
    tenant_id: str,
    cleaned_text: str,
) -> int | None:
    """
    Atomically records a new event. Returns its row id, or None if this
    (tenant_id, idempotency_key) was already claimed.

    SQLite serializes writes and the UNIQUE constraint is checked inside the
    INSERT, so "check if it exists, then insert" is a single indivisible step:
    of any number of concurrent identical deliveries, exactly one gets a row id.
    """
    async with db.execute(
        """
        INSERT INTO events (idempotency_key, event_id, tenant_id, cleaned_text)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (tenant_id, idempotency_key) DO NOTHING
        """,
        (idempotency_key, event_id, tenant_id, cleaned_text),
    ) as cur:
        return cur.lastrowid if cur.rowcount == 1 else None


async def get_event(db: aiosqlite.Connection, row_id: int) -> aiosqlite.Row | None:
    async with db.execute("SELECT * FROM events WHERE id = ?", (row_id,)) as cur:
        return await cur.fetchone()


async def set_fingerprint(db: aiosqlite.Connection, row_id: int, fingerprint: str) -> None:
    await db.execute(
        "UPDATE events SET fingerprint = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (fingerprint, row_id),
    )


async def mark_delivered(db: aiosqlite.Connection, row_id: int, attempts: int) -> None:
    await db.execute(
        """
        UPDATE events
        SET status = 'delivered', attempts = ?, last_error = NULL, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (attempts, row_id),
    )


async def mark_failed(db: aiosqlite.Connection, row_id: int, attempts: int, error: str) -> None:
    await db.execute(
        """
        UPDATE events
        SET status = 'failed', attempts = ?, last_error = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (attempts, error[:MAX_ERROR_LENGTH], row_id),
    )


async def fetch_pending(db: aiosqlite.Connection) -> list[int]:
    """Row ids accepted but not yet delivered or failed (used for recovery at startup)."""
    async with db.execute("SELECT id FROM events WHERE status = 'pending' ORDER BY id") as cur:
        return [row["id"] for row in await cur.fetchall()]
