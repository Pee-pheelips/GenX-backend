# Architecture & Debugging Decisions

Please fill out the four sections below concisely before submitting your solution.

> All numbers below were measured on a 4-vCPU Linux VM (Python 3.12) using the same 60-request burst as `tests/test_burst_load.py` (30 unique events, each sent twice concurrently with the same key).
>
> | | Original | Refactored |
> |---|---|---|
> | Non-2xx responses | 6 / 60 | 0 / 60 |
> | Ingest p95 | 5,099 ms | 40–63 ms |
> | Max `/health` during burst | 2,141 ms | 26–39 ms |
> | Downstream deliveries (unique) | 54 (30) | 30 (30) |
> | DB rows | 60, all marked `delivered` | 30, all `delivered` |
>
> `tests/test_burst_load.py` passed 10/10 runs on its own and 15/16 runs as part of the full suite. The one failure was a single `/health` probe at 58 ms. For comparison, a FastAPI app whose handler does nothing measured 30–43 ms (one run 70 ms) under the same burst on this VM. The remaining `/health` latency is therefore the uvicorn/FastAPI/test-client floor for ~60 simultaneous new connections, not this service's code.

## 1. Root-Cause Diagnosis
What specific bottlenecks and failure modes did you observe in the original codebase under concurrent load?

- **Event-loop / CPU behavior:** `scrub_and_fingerprint()` (120,000 chained SHA-256 rounds, **~37 ms per call**) was called synchronously inside the `async def` handler, so each request froze the event loop. With 60 concurrent requests that is ~2.2 s of serialized CPU work during which nothing else, including `/health` (max **2,141 ms**), could run. A second, latent problem surfaced during the refactor: the email regex started a match at every `\b` inside a run of local-part characters, which is quadratic. A 16 KB payload such as `"a.a.a…"` took **~200 ms** of event-loop time on its own.
- **Concurrency & race conditions:**
  - **No idempotency at all.** `X-Idempotency-Key` was checked for presence and then ignored: it was never stored, and the table had no unique constraint. Every duplicate produced its own row: **60 rows** for 30 events.
  - **The ack waited for downstream.** The handler awaited the partner (~250 ms) before replying, so the partner's latency and failures were passed straight to the sender. A transient `503` became a `500` (`raise_for_status()`), which makes real senders retry, and each retry was processed again. That is the reported duplicate-event problem: the partner received **54 deliveries for 30 events**.
  - **Lost events were hidden.** The row was inserted before forwarding, so a failed forward left it `pending` with nothing to retry it. `mark_delivered` then updated by `event_id`, so when the duplicate's delivery succeeded it also marked the failed row `delivered`: the DB reported 60/60 delivered while events had actually failed.
  - **Idempotency alone is not enough.** During the refactor, adding only the idempotency gate (with delivery still synchronous and no retries) removed the duplicates, but **lost 2–10 events per run**. The winning request hit a `503`, and the sender's retry was told "duplicate", so the event was never delivered. Idempotency needs reliable retries and recovery behind it.
- **Resource & connection management:** Every DB call opened a new `aiosqlite` connection (each one is a new thread) with SQLite defaults: rollback journal, no `busy_timeout`. That costs a connection and thread per operation, and concurrent writers risk `database is locked` errors (a risk found by reading the code; I did not observe it in the burst). Every forward created and discarded an `httpx.AsyncClient`, so no connection reuse and only httpx's implicit default timeout. Startup used the deprecated `@app.on_event` hook.

## 2. Concurrency & Offloading Strategy
How did you separate CPU-bound preprocessing, HTTP acknowledgment, and downstream delivery? Why did you choose your specific concurrency primitives over alternatives (e.g., `asyncio.to_thread` vs. `ProcessPoolExecutor` vs. FastAPI `BackgroundTasks`)?

**Separation** (a small, in-process "persist first, deliver later" design, with SQLite as the source of truth):

1. **Request path (`app/main.py`):** validate (Pydantic plus length limits) → `scrub()` PII inline → atomic `claim_event()` INSERT → `worker.enqueue(row_id)` → `202 accepted` (`200 duplicate` for a repeated key). `processor.py` was split so only the cheap part stays here: `scrub()` takes ~2 µs on the test payload, and the regex fix bounds it to ~1 ms at the 16 KB content limit. PII is scrubbed before anything is persisted.
2. **CPU work (`app/worker.py`):** `fingerprint()` runs in a `ProcessPoolExecutor` via `loop.run_in_executor`, after the ack. It is never on the request path. Even after deduplication and fully parallel, 30 × 37 ms across 3 processes (~370 ms) could never fit a 250 ms p95.
3. **Delivery (`app/worker.py`, `app/downstream.py`):** an `asyncio.Queue` of row ids consumed by 16 worker tasks. Each worker fingerprints → stores the fingerprint → forwards with retries over one shared `httpx.AsyncClient` → marks the row `delivered` or `failed`. At startup, rows still `pending` are re-queued.

**Why these primitives:**

- **`ProcessPoolExecutor` over `asyncio.to_thread`:** the fingerprint is pure-Python CPU work, and `hashlib` only releases the GIL for inputs of 2 KB or more (each round here hashes 32 bytes). In a thread pool the hashing would hold the GIL, run one call at a time, and compete with the event loop for it. Processes give real parallelism and leave the loop's GIL alone. *(This choice rests on that reasoning; I did not benchmark the thread-pool variant.)*
- **Pool size = CPU count − 1, using `spawn`, warmed at startup.** Measured max event-loop stall while fingerprinting 30 events: inline **118 ms**; pool with 4 processes on 4 cores **90 ms** (the children starve the loop's process of CPU); pool with 3 processes **< 2 ms**. With the full worker (DB, pool and a simulated 250 ms partner) the loop stall stayed at 2–11 ms over 8 runs. Other details:
  - `spawn` is used on every platform, because forking a process that already runs threads (aiosqlite's) can deadlock, and `spawn` is the macOS default anyway.
  - Warm-up moves process start-up cost to startup, before traffic arrives.
  - The pool is shut down via `asyncio.to_thread`, because joining the children blocked the loop for 45–143 ms.
- **Queue + worker tasks over FastAPI `BackgroundTasks`:** `BackgroundTasks` run on the same event loop, so CPU-bound work there still blocks `/health`. They also have no concurrency limit and nothing to recover them after a crash. The explicit queue limits delivery concurrency (16 I/O-bound tasks, cores − 1 CPU jobs), keeps strong references to its tasks, and only carries row ids, because the durable state is the DB row. The queue is unbounded so the ack never waits on background capacity; the items are small ints, and the DB, not the queue, is the record.
- **Retries (`forward_event`):** up to 6 attempts using exponential backoff with full jitter (base 0.2 s, cap 10 s). A partner's `Retry-After` is honoured up to the cap. It retries 5xx, 429 and transport errors, but never other 4xx (they are permanent). Timeouts are explicit: connect 2 s, read 5 s.

## 3. Idempotency & SQLite Concurrency
How does your implementation guarantee zero duplicate processing when two identical webhook retries arrive at the exact same millisecond? What SQLite settings or constraints did you rely on?

- **Constraint:** `UNIQUE (tenant_id, idempotency_key)` on `events`. It is scoped per tenant so one tenant cannot suppress another tenant's event by reusing its key.
- **Atomic claim:** `INSERT … ON CONFLICT (tenant_id, idempotency_key) DO NOTHING`, then `rowcount == 1` decides the winner. SQLite serializes writers, and the uniqueness check happens inside the INSERT, so "check, then insert" is a single indivisible step with no window between them. Of any number of simultaneous identical requests, exactly one gets a row id. Only that request enqueues work, so only one fingerprint and one delivery are ever scheduled; the rest return `200 duplicate`. No application-level lock is needed. Tested: 50 concurrent claims on one connection → 1 winner; 40 claims spread over 4 separate connections (the multi-process case) → 1 winner.
- **Worker guard:** the worker re-reads the row and skips anything not `pending`. A stale enqueue cannot re-deliver a finished event, and all updates are by primary key, never `event_id`.
- **Settings:**
  - **One long-lived connection** opened with `isolation_level=None` (autocommit). Every write is a single atomic statement, so coroutines sharing the connection can never have their statements interleaved inside one transaction.
  - **`PRAGMA journal_mode=WAL`:** readers never block the writer.
  - **`PRAGMA synchronous=NORMAL`:** in WAL mode, a commit survives a process crash; only an OS crash or power loss can lose the most recent commits.
  - **`PRAGMA busy_timeout=5000`:** wait instead of failing when another connection or process holds the write lock.
  - **`rowcount`/`lastrowid`** are used instead of `RETURNING`, for compatibility with older SQLite builds.
- **Limits of the guarantee (stated honestly):**
  - **Duplicates on our side:** guaranteed zero. There is one row and one scheduled delivery per key.
  - **Duplicates at the partner:** delivery is **at-least-once**, not exactly-once. If a request reaches the partner but our read times out (5 s, far above its normal ~250 ms), we retry and the partner may see the event twice. The same applies after a crash between delivery and `mark_delivered` (see §4). We forward `Idempotency-Key` so the partner can deduplicate; the mock ignores it.
  - **What a duplicate means:** a repeated key with a *different* body is treated as a duplicate (first write wins). We don't hash payloads to detect that conflict.
  - **No reprocessing of failed events:** a retry of an event that already ended `failed` is answered `200 duplicate` and is not reprocessed.

## 4. Failure Modes & Production Trade-offs
- **What happens in your design if the downstream service is completely unreachable for 60 seconds?**
  - **Ingestion is unaffected.** Acks don't touch the partner, so webhooks are still recorded and acknowledged in milliseconds with `202`.
  - **Each event gives up well inside 60 s.** It gets 6 attempts. The backoff ceilings between attempts add up to at most 6.2 s (full jitter averages about half that), plus up to 2 s of connect timeout per attempt if packets are silently dropped rather than refused. So each event exhausts its retries within roughly 6–18 s, and events arriving during the outage are marked **`failed`** with `attempts` and `last_error` recorded.
  - **Failed events are not retried automatically.** Startup recovery only re-queues `pending` rows. After the outage they need a manual replay: set them back to `pending` and restart.
  - **Delivery slows during the outage.** The 16 worker tasks spend it sleeping in backoff, so delivery throughput drops while the queue of ids grows in memory.
  - **Production fix:** a longer retry window via a periodic sweeper that re-queues `failed` rows using a stored `next_attempt_at`, plus a circuit breaker so workers stop hammering a dead partner. The `Retry-After` support already lets a partner slow us down.
- **What happens if this service process crashes 50ms after acknowledging a webhook?**
  - **The event survives.** The `202` is sent only after the claim INSERT is committed, and with WAL + `synchronous=NORMAL` that commit survives a process crash. (Only an OS crash or power loss could lose it; `synchronous=FULL` would close that gap at the cost of an fsync per write.)
  - **It is delivered on restart.** At 50 ms the row is almost certainly still `pending`, since the partner call alone takes ~250 ms. On the next start `DeliveryWorker.start()` re-queues every `pending` row and delivers it. If the fingerprint was already stored, it is reused instead of recomputed. This is covered by tests (`test_pending_rows_are_recovered_on_start`, `test_pending_event_is_delivered_after_restart`).
  - **The edge case:** if the crash happens after the partner accepted the event but before `mark_delivered` commits, the event is redelivered on restart. That is the at-least-once trade-off; the forwarded `Idempotency-Key` lets the partner drop the repeat.
  - **Recovery only runs at startup.** There is no sweeper, so a row left `pending` by an unexpected in-process error waits for the next restart.
- **What would break first if we deployed this container with `--workers 4` instead of 1?**
  - **First, under load: CPU oversubscription.** Each uvicorn worker builds its own fingerprint pool sized at CPU count − 1. On a 4-core box that is 4 × 3 = 12 CPU-bound processes plus 4 event loops. We measured that a pool matching the core count already stalls the event loop up to 90 ms, so `/health` and ack latency would degrade on the first burst. Pool size would have to become `cores / workers`.
  - **Second, a correctness bug on restart: duplicate deliveries.** All 4 workers run startup recovery at the same time, each reads the same `pending` rows into its own in-memory queue, and each delivers them, so one event can go out up to 4 times. The per-request idempotency gate still holds (the UNIQUE constraint works across processes, as tested), but recovery has no cross-process claim. The fix is a lease: an atomic `UPDATE … SET status='processing', lease_until=… WHERE id=? AND status='pending'` before a worker processes a row, or running delivery in a single dedicated process.
  - **What doesn't break immediately:** SQLite still allows only one writer at a time, but `busy_timeout` makes the four processes wait for the write lock instead of erroring; at this volume that is fine.
  - **Separate state per worker:** each process has its own queue, connection and HTTP client, so there is no shared view of in-flight work.
- **Other known trade-offs (not implemented, by choice):**
  - **No sender authentication:** no HMAC signature check on incoming webhooks, and no rate limiting.
  - **Body size is checked after reading:** a very large request is read into memory before Pydantic rejects it with `422`; a reverse proxy should cap body size.
  - **A broken process pool isn't rebuilt:** if a fingerprint process dies hard, the pool becomes unusable and affected events stay `pending` until the next restart.
  - **Old database files need deleting:** there is no migration tool. `init_db` refuses to start against the old schema with a clear error instead of failing on the first request.
