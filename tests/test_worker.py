import json
import os

import httpx
import pytest

from app import worker as worker_module
from app.database import claim_event, connect, get_event, init_db, mark_delivered, set_fingerprint
from app.processor import fingerprint
from app.worker import DeliveryWorker


def fingerprint_with_pid(text: str, blob_iterations: int = 0) -> str:
    """Stand-in for fingerprint() that reports which process ran it."""
    return f"pid:{os.getpid()}"


@pytest.fixture
async def db(tmp_path):
    conn = await connect(str(tmp_path / "events.db"))
    await init_db(conn)
    yield conn
    await conn.close()


class FakeDownstream:
    """httpx client backed by MockTransport; `outcomes` are consumed per call, the last repeats."""

    def __init__(self, *outcomes):
        self.outcomes = outcomes or (httpx.Response(200, json={"accepted": True}),)
        self.requests: list[dict] = []
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        item = self.outcomes[min(len(self.requests), len(self.outcomes)) - 1]
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def started_workers():
    workers = []

    async def start(db, downstream, **kwargs):
        w = DeliveryWorker(db, downstream.client, processes=1, **kwargs)
        await w.start()
        workers.append(w)
        return w

    yield start
    # Teardown runs in sync context; workers are stopped explicitly in each test below.
    assert all(not w._tasks for w in workers), "test forgot to stop its worker"


async def claim(db, n, text="cleaned text"):
    return await claim_event(db, f"idem_{n}", f"evt_{n}", "tenant_a", text)


async def test_pending_event_is_fingerprinted_and_delivered(db, started_workers):
    downstream = FakeDownstream()
    w = await started_workers(db, downstream)
    try:
        row_id = await claim(db, 1)
        w.enqueue(row_id)
        await w.join()
    finally:
        await w.stop()

    row = await get_event(db, row_id)
    assert row["status"] == "delivered" and row["attempts"] == 1
    assert row["fingerprint"] == fingerprint("cleaned text")
    assert downstream.requests == [
        {"event_id": "evt_1", "text": "cleaned text", "fingerprint": row["fingerprint"]}
    ]


async def test_fingerprint_runs_in_a_separate_process(db, started_workers, monkeypatch):
    monkeypatch.setattr(worker_module, "fingerprint", fingerprint_with_pid)
    w = await started_workers(db, FakeDownstream())
    try:
        row_id = await claim(db, 1)
        w.enqueue(row_id)
        await w.join()
    finally:
        await w.stop()

    fp = (await get_event(db, row_id))["fingerprint"]
    assert fp.startswith("pid:")
    assert fp != f"pid:{os.getpid()}", "fingerprint ran on the event loop's process"


async def test_pending_rows_are_recovered_on_start(db, started_workers):
    # Simulates a crash after acknowledging: rows exist but were never delivered.
    pending = [await claim(db, n) for n in range(3)]
    done = await claim(db, 99)
    await mark_delivered(db, done, attempts=1)

    downstream = FakeDownstream()
    w = await started_workers(db, downstream)
    try:
        await w.join()
    finally:
        await w.stop()

    for row_id in pending:
        assert (await get_event(db, row_id))["status"] == "delivered"
    sent = sorted(r["event_id"] for r in downstream.requests)
    assert sent == ["evt_0", "evt_1", "evt_2"]  # the already-delivered event is not re-sent


async def test_stored_fingerprint_is_reused_after_restart(db, started_workers):
    # Crash between fingerprinting and delivery: don't redo the CPU work.
    row_id = await claim(db, 1)
    await set_fingerprint(db, row_id, "precomputed")
    downstream = FakeDownstream()
    w = await started_workers(db, downstream)
    try:
        await w.join()
    finally:
        await w.stop()
    assert downstream.requests[0]["fingerprint"] == "precomputed"


async def test_transient_failure_is_retried_until_delivered(db, started_workers):
    downstream = FakeDownstream(httpx.Response(503), httpx.Response(200, json={}))
    w = await started_workers(db, downstream)
    try:
        row_id = await claim(db, 1)
        w.enqueue(row_id)
        await w.join()
    finally:
        await w.stop()
    row = await get_event(db, row_id)
    assert row["status"] == "delivered" and row["attempts"] == 2


async def test_permanent_failure_marks_row_failed(db, started_workers):
    downstream = FakeDownstream(httpx.Response(400))
    w = await started_workers(db, downstream)
    try:
        row_id = await claim(db, 1)
        w.enqueue(row_id)
        await w.join()
    finally:
        await w.stop()
    row = await get_event(db, row_id)
    assert row["status"] == "failed" and row["attempts"] == 1
    assert "400" in row["last_error"]
    assert len(downstream.requests) == 1


async def test_non_pending_rows_are_skipped(db, started_workers):
    row_id = await claim(db, 1)
    await mark_delivered(db, row_id, attempts=1)
    downstream = FakeDownstream()
    w = await started_workers(db, downstream)
    try:
        w.enqueue(row_id)  # e.g. a stale duplicate enqueue
        await w.join()
    finally:
        await w.stop()
    assert downstream.requests == []


async def test_unexpected_error_does_not_kill_the_worker(db, started_workers):
    # A non-httpx exception escapes forward_event; with a single worker task, the
    # second event is only delivered if that task survived the first one's error.
    downstream = FakeDownstream(ValueError("boom"), httpx.Response(200, json={}))
    w = await started_workers(db, downstream, concurrency=1)
    try:
        first, second = await claim(db, 1), await claim(db, 2)
        w.enqueue(first)
        w.enqueue(second)
        await w.join()
    finally:
        await w.stop()
    assert (await get_event(db, first))["status"] == "pending"  # left for recovery
    assert (await get_event(db, second))["status"] == "delivered"


async def test_stop_shuts_down_tasks_and_pool(db):
    w = DeliveryWorker(db, FakeDownstream().client, processes=1, concurrency=3)
    await w.start()
    assert len(w._tasks) == 3 and w._pool is not None
    await w.stop()
    assert w._tasks == [] and w._pool is None
