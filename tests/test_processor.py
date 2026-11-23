import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import pytest

from app.processor import (
    MAX_CONTENT_LENGTH,
    fingerprint,
    scrub,
    scrub_and_fingerprint,
)

# Captured from the ORIGINAL app/processor.py before the refactor, to prove the
# split did not change the redaction output or the fingerprint algorithm.
SAMPLE = "Customer user7@example.com SSN 123-45-6789 card 4111 1111 1111 1111 ticket #7"
GOLDEN = (
    "Customer [REDACTED] SSN [REDACTED] card [REDACTED] ticket #7",
    "8f9b073b8437af0748b787541eaeb1a50dd6dddd4cb25ba12ce87c78511340ed",
)
GOLDEN_SHORT = "6c92efed736019d91804be8749edd36be572a345bd51c80f1ac4ea0d932c0dc7"  # "hello", 3 rounds


def test_output_matches_original_implementation():
    assert scrub_and_fingerprint(SAMPLE) == GOLDEN
    assert fingerprint("hello", blob_iterations=3) == GOLDEN_SHORT


def test_split_functions_compose_to_wrapper():
    cleaned = scrub(SAMPLE)
    assert (cleaned, fingerprint(cleaned)) == scrub_and_fingerprint(SAMPLE)


@pytest.mark.parametrize(
    "text, leaked",
    [
        ("SSN 123-45-6789 on file", "123-45-6789"),
        ("mail john.doe+tag@mail.example.co.uk now", "john.doe+tag@mail.example.co.uk"),
        ("(.john@example.com)", "john@example.com"),
        ("card 4111-1111-1111-1111 ok", "4111-1111-1111-1111"),
        ("card 4111111111111111 ok", "4111111111111111"),
    ],
)
def test_scrub_redacts_pii(text, leaked):
    cleaned = scrub(text)
    assert leaked not in cleaned
    assert "[REDACTED]" in cleaned


def test_scrub_leaves_clean_text_untouched():
    text = "Ticket #42 opened by tenant_1 at 10:30"
    assert scrub(text) == text


def test_fingerprint_is_deterministic_sha256_hex():
    a = fingerprint("same input", blob_iterations=10)
    assert a == fingerprint("same input", blob_iterations=10)
    assert a != fingerprint("other input", blob_iterations=10)
    assert len(a) == 64 and int(a, 16) >= 0


def test_fingerprint_runs_in_spawned_process_pool():
    # "spawn" is the macOS default and the strictest mode: the function and its
    # arguments must be importable/picklable. This is the contract the worker relies on.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        assert pool.submit(fingerprint, "hello", 3).result(timeout=30) == GOLDEN_SHORT


@pytest.mark.parametrize(
    "payload",
    [
        "1-" * (MAX_CONTENT_LENGTH // 2),
        "a." * (MAX_CONTENT_LENGTH // 2),
        "a@" + "b." * (MAX_CONTENT_LENGTH // 2 - 1),
        "1 " * (MAX_CONTENT_LENGTH // 2),
        "1" + " " * (MAX_CONTENT_LENGTH - 1),
        ("a" * 60 + "@") * (MAX_CONTENT_LENGTH // 61),
    ],
    ids=["dashed", "dotted", "dotted-domain", "spaced-digits", "digit-then-spaces", "many-at"],
)
def test_scrub_stays_cheap_on_adversarial_input_at_size_cap(payload):
    # scrub() runs on the event loop, so worst-case input must stay far below the
    # 50ms /health budget. Best of 5 runs measures the regex cost itself rather than
    # a one-off scheduler pause; the bound is generous (measured ~1ms, while the old
    # quadratic email regex took ~200ms on the same inputs).
    assert len(payload) <= MAX_CONTENT_LENGTH
    timings = []
    for _ in range(5):
        t0 = time.perf_counter()
        scrub(payload)
        timings.append((time.perf_counter() - t0) * 1000)
    assert min(timings) < 20
