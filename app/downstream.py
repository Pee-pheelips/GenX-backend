import asyncio
import logging
import os
import random

import httpx

logger = logging.getLogger(__name__)

DOWNSTREAM_URL = os.getenv("DOWNSTREAM_URL", "http://127.0.0.1:9001/events")

MAX_ATTEMPTS = int(os.getenv("DOWNSTREAM_MAX_ATTEMPTS", "6"))
BACKOFF_BASE = float(os.getenv("DOWNSTREAM_BACKOFF_BASE", "0.2"))  # seconds
BACKOFF_CAP = float(os.getenv("DOWNSTREAM_BACKOFF_CAP", "10"))  # seconds

# Read timeout sits far above the partner's normal latency (~250ms). A timeout is the
# one retry case where the partner may already have accepted the event, so it should
# only fire when something is genuinely wrong.
TIMEOUT = httpx.Timeout(5.0, connect=2.0)
LIMITS = httpx.Limits(max_connections=50, max_keepalive_connections=20)


class DeliveryError(Exception):
    """Delivery did not succeed. `retryable` is False for permanent (4xx) rejections."""

    def __init__(self, message: str, attempts: int, retryable: bool):
        super().__init__(message)
        self.attempts = attempts
        self.retryable = retryable


def create_client() -> httpx.AsyncClient:
    """One shared client for the process: reuses connections instead of opening one per event."""
    return httpx.AsyncClient(timeout=TIMEOUT, limits=LIMITS)


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None  # HTTP-date form is rare for webhooks; fall back to our own backoff


def retry_delay(
    attempt: int,
    retry_after: float | None = None,
    base: float = BACKOFF_BASE,
    cap: float = BACKOFF_CAP,
) -> float:
    """
    Exponential backoff with full jitter: uniform(0, min(cap, base * 2**(attempt-1))).
    Jitter spreads retries out so a recovering partner is not hit by every queued
    event at the same instant. A partner's Retry-After is honoured, up to the cap.
    """
    delay = random.uniform(0, min(cap, base * 2 ** (attempt - 1)))
    if retry_after is not None:
        delay = max(delay, min(retry_after, cap))
    return delay


async def forward_event(
    client: httpx.AsyncClient,
    event_id: str,
    cleaned_text: str,
    fingerprint: str,
    idempotency_key: str,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    backoff_base: float = BACKOFF_BASE,
    backoff_cap: float = BACKOFF_CAP,
) -> tuple[dict, int]:
    """
    Forwards a processed event downstream, retrying transient failures.

    Retries: 5xx, 429 and transport errors (connection failures, timeouts).
    No retry: any other 4xx, which would fail identically every time.
    Returns (response body, attempts used); raises DeliveryError when giving up.
    """
    payload = {"event_id": event_id, "text": cleaned_text, "fingerprint": fingerprint}
    headers = {"Idempotency-Key": idempotency_key}

    for attempt in range(1, max_attempts + 1):
        retry_after = None
        try:
            response = await client.post(DOWNSTREAM_URL, json=payload, headers=headers)
        except httpx.TransportError as exc:
            reason = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        else:
            if response.is_success:
                try:
                    body = response.json()
                except ValueError:
                    body = {}  # accepted is accepted; never retry (and duplicate) over a body we can't parse
                return body, attempt
            reason = f"HTTP {response.status_code}"
            if not _is_retryable_status(response.status_code):
                raise DeliveryError(f"Downstream rejected event: {reason}", attempt, retryable=False)
            retry_after = _retry_after_seconds(response)

        if attempt == max_attempts:
            raise DeliveryError(
                f"Gave up after {attempt} attempts; last error: {reason}", attempt, retryable=True
            )
        delay = retry_delay(attempt, retry_after, backoff_base, backoff_cap)
        logger.warning(
            "Delivery of %s failed (attempt %d/%d, %s); retrying in %.2fs",
            event_id, attempt, max_attempts, reason, delay,
        )
        await asyncio.sleep(delay)

    raise AssertionError("unreachable")  # loop always returns or raises
