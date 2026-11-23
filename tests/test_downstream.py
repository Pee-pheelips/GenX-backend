import json

import httpx
import pytest

from app import downstream
from app.downstream import DeliveryError, create_client, forward_event, retry_delay

NO_WAIT = {"backoff_base": 0.0, "backoff_cap": 0.0}


def make_client(*responses):
    """
    Client backed by httpx.MockTransport. Each item is an httpx.Response to return,
    or an exception class to raise, consumed in order; the last item repeats.
    Returns (client, list of captured requests).
    """
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        item = responses[min(len(calls), len(responses)) - 1]
        if isinstance(item, type) and issubclass(item, Exception):
            raise item("simulated", request=request)
        return item

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), calls


async def send(client, **kwargs):
    return await forward_event(client, "evt_1", "clean text", "fp123", "idem_1", **NO_WAIT, **kwargs)


async def test_success_on_first_attempt_sends_payload_and_idempotency_key():
    client, calls = make_client(httpx.Response(200, json={"accepted": True}))
    body, attempts = await send(client)
    assert (body, attempts) == ({"accepted": True}, 1)
    assert len(calls) == 1
    assert calls[0].headers["Idempotency-Key"] == "idem_1"
    assert json.loads(calls[0].content) == {
        "event_id": "evt_1", "text": "clean text", "fingerprint": "fp123",
    }


@pytest.mark.parametrize(
    "first_failure",
    [
        httpx.Response(503),
        httpx.Response(500),
        httpx.Response(429),
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.RemoteProtocolError,
    ],
    ids=["503", "500", "429", "connect-error", "read-timeout", "protocol-error"],
)
async def test_transient_failure_is_retried_then_succeeds(first_failure):
    client, calls = make_client(first_failure, httpx.Response(200, json={"accepted": True}))
    body, attempts = await send(client)
    assert attempts == 2 and len(calls) == 2
    assert body == {"accepted": True}


@pytest.mark.parametrize("status", [400, 401, 404, 409, 422])
async def test_permanent_4xx_is_not_retried(status):
    client, calls = make_client(httpx.Response(status))
    with pytest.raises(DeliveryError) as exc_info:
        await send(client)
    assert len(calls) == 1
    assert exc_info.value.attempts == 1
    assert exc_info.value.retryable is False
    assert str(status) in str(exc_info.value)


async def test_gives_up_after_max_attempts():
    client, calls = make_client(httpx.Response(503))
    with pytest.raises(DeliveryError) as exc_info:
        await send(client, max_attempts=4)
    assert len(calls) == 4
    assert exc_info.value.attempts == 4
    assert exc_info.value.retryable is True
    assert "503" in str(exc_info.value)


async def test_accepted_response_with_unparseable_body_is_not_retried():
    # Retrying here would deliver the event twice.
    client, calls = make_client(httpx.Response(200, text="OK"))
    assert await send(client) == ({}, 1)
    assert len(calls) == 1


async def test_backoff_sleeps_between_attempts(monkeypatch):
    delays = []

    async def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(downstream.asyncio, "sleep", fake_sleep)
    client, _ = make_client(httpx.Response(503))
    with pytest.raises(DeliveryError):
        await forward_event(client, "e", "t", "f", "k", max_attempts=3, backoff_base=1.0, backoff_cap=10.0)
    assert len(delays) == 2  # sleeps between attempts, never after the last one
    assert 0 <= delays[0] <= 1.0 and 0 <= delays[1] <= 2.0


def test_retry_delay_full_jitter_bounds_and_cap():
    for attempt in range(1, 12):
        ceiling = min(10.0, 0.2 * 2 ** (attempt - 1))
        for _ in range(50):
            assert 0 <= retry_delay(attempt, base=0.2, cap=10.0) <= ceiling


def test_retry_delay_honours_retry_after_up_to_cap():
    assert retry_delay(1, retry_after=3.0, base=0.01, cap=10.0) >= 3.0
    assert retry_delay(1, retry_after=60.0, base=0.01, cap=10.0) == 10.0


async def test_retry_after_header_is_used(monkeypatch):
    delays = []

    async def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(downstream.asyncio, "sleep", fake_sleep)
    client, _ = make_client(
        httpx.Response(503, headers={"Retry-After": "2"}), httpx.Response(200, json={})
    )
    await forward_event(client, "e", "t", "f", "k", backoff_base=0.01, backoff_cap=10.0)
    assert delays == [2.0]


async def test_shared_client_has_explicit_timeouts():
    client = create_client()
    try:
        assert client.timeout.connect == 2.0
        assert client.timeout.read == 5.0
    finally:
        await client.aclose()
