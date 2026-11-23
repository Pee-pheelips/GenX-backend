import hashlib
import re

# Upper bound on webhook content size. Enforced by the API layer; defined here because
# the limit exists to keep `scrub()` (which runs on the event loop) cheap.
MAX_CONTENT_LENGTH = 16_384

FINGERPRINT_ITERATIONS = 120_000

PII_PATTERNS = [
    # US SSN
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Email. The lookbehind (instead of a leading \b) only lets a match start at the
    # beginning of a run of local-part characters. With \b, the engine re-scanned the
    # same run from every position inside it, which is quadratic: ~200ms on a 16KB
    # "a.a.a..." payload, all of it blocking the event loop. Matches are the same or
    # slightly wider (e.g. a leading "." is redacted too), never narrower.
    re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # Payment card numbers (13-16 digits, optional space/dash separators)
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
]


def scrub(raw_text: str) -> str:
    """
    Redacts common PII patterns.

    Cheap (linear-time regexes, well under 1ms for typical payloads), so it is safe to
    run on the event loop for input bounded by MAX_CONTENT_LENGTH. Runs before anything
    is persisted, so raw PII never reaches the database.
    """
    cleaned = raw_text
    for pattern in PII_PATTERNS:
        cleaned = pattern.sub("[REDACTED]", cleaned)
    return cleaned


def fingerprint(cleaned_text: str, blob_iterations: int = FINGERPRINT_ITERATIONS) -> str:
    """
    Computes a deterministic cryptographic fingerprint of already-scrubbed text.

    CPU-bound (~40-60ms per call depending on host CPU) and holds the GIL (hashlib
    only releases it for inputs >= 2KB; each round here hashes 32 bytes), so it must
    not run on the event loop or in a thread. Kept as a module-level function with
    picklable arguments so it can be submitted to a ProcessPoolExecutor.
    """
    digest = cleaned_text.encode("utf-8")
    for _ in range(blob_iterations):
        digest = hashlib.sha256(digest).digest()
    return digest.hex()


def scrub_and_fingerprint(
    raw_text: str, blob_iterations: int = FINGERPRINT_ITERATIONS
) -> tuple[str, str]:
    """Redacts PII and fingerprints the result. Equivalent to the original single-step API."""
    cleaned = scrub(raw_text)
    return cleaned, fingerprint(cleaned, blob_iterations)
