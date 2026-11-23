import asyncio
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import aiosqlite
import httpx

from app import database
from app.downstream import DeliveryError, forward_event
from app.processor import fingerprint

logger = logging.getLogger(__name__)

# Delivery is I/O-bound (~250ms per partner call), so many tasks can wait concurrently.
DELIVERY_CONCURRENCY = int(os.getenv("DELIVERY_CONCURRENCY", "16"))
# Fingerprinting is CPU-bound. Leave one core for the event loop so /health stays fast.
FINGERPRINT_PROCESSES = int(
    os.getenv("FINGERPRINT_PROCESSES", str(max(1, (os.cpu_count() or 2) - 1)))
)


class DeliveryWorker:
    """
    Runs everything that happens after a webhook is acknowledged:
    fingerprint (in a process pool) -> deliver downstream (with retries) -> record outcome.

    The database row is the source of truth; the queue only carries row ids, so
    anything lost from memory (crash, restart) is recovered from rows still 'pending'.
    """

    def __init__(
        self,
        db: aiosqlite.Connection,
        client: httpx.AsyncClient,
        *,
        concurrency: int = DELIVERY_CONCURRENCY,
        processes: int = FINGERPRINT_PROCESSES,
    ):
        self._db = db
        self._client = client
        self._concurrency = concurrency
        self._processes = processes
        self._queue: asyncio.Queue[int] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._pool: ProcessPoolExecutor | None = None

    async def start(self) -> None:
        # "spawn" on every platform: forking a process that already runs threads
        # (aiosqlite's connection thread) can deadlock, and it matches the macOS default.
        self._pool = ProcessPoolExecutor(
            max_workers=self._processes, mp_context=multiprocessing.get_context("spawn")
        )
        await self._warm_up_pool()

        pending = await database.fetch_pending(self._db)
        for row_id in pending:
            self._queue.put_nowait(row_id)
        if pending:
            logger.info("Recovered %d pending event(s) for delivery", len(pending))

        self._tasks = [
            asyncio.create_task(self._run(), name=f"delivery-worker-{i}")
            for i in range(self._concurrency)
        ]

    def enqueue(self, row_id: int) -> None:
        # Unbounded and non-blocking: the ack must never wait on background capacity.
        # Items are just ints, and the DB (not the queue) is the durable record.
        self._queue.put_nowait(row_id)

    async def join(self) -> None:
        """Waits until every queued event has been processed (used by tests)."""
        await self._queue.join()

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        if self._pool is not None:
            # Events interrupted here are still 'pending' and are recovered on next start.
            # Joining the child processes takes ~50-150ms; do it off the event loop.
            await asyncio.to_thread(self._pool.shutdown, wait=True, cancel_futures=True)
            self._pool = None

    async def _warm_up_pool(self) -> None:
        # Processes are spawned lazily on first submit; pay that cost (~100ms+ each)
        # at startup instead of during the first burst.
        loop = asyncio.get_running_loop()
        await asyncio.gather(
            *(loop.run_in_executor(self._pool, fingerprint, "", 1) for _ in range(self._processes))
        )

    async def _run(self) -> None:
        while True:
            row_id = await self._queue.get()
            try:
                await self._process(row_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Never let one bad event kill the worker. The row stays 'pending'
                # and is retried by recovery on the next start.
                logger.exception("Unexpected error processing event row %s", row_id)
            finally:
                self._queue.task_done()

    async def _process(self, row_id: int) -> None:
        row = await database.get_event(self._db, row_id)
        if row is None or row["status"] != "pending":
            return

        fp = row["fingerprint"]
        if fp is None:
            loop = asyncio.get_running_loop()
            fp = await loop.run_in_executor(self._pool, fingerprint, row["cleaned_text"])
            await database.set_fingerprint(self._db, row_id, fp)

        try:
            _, attempts = await forward_event(
                self._client, row["event_id"], row["cleaned_text"], fp, row["idempotency_key"]
            )
        except DeliveryError as exc:
            await database.mark_failed(self._db, row_id, exc.attempts, str(exc))
            logger.error("Delivery of %s failed permanently: %s", row["event_id"], exc)
            return
        await database.mark_delivered(self._db, row_id, attempts)
