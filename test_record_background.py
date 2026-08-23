"""record(background=) — the latency/correctness trade, made explicit.

A developer review pointed out that making record fire-and-forget by default
buys latency at the cost of a race: prepare for turn N+1 retrieves what has been
written, so a record still in flight means turn N is missing from turn N+1's
context — intermittently, under load, reading as "the memory is flaky" rather
than as a race.

So blocking stays the default and the fast path is opt-in. These tests pin that
choice, because a later refactor that flips the default would be a silent
correctness change.

Run: python3 test_record_background.py
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from khwan import Khwan, Turn  # noqa: E402


def _client():
    return Khwan(api_key="kwk_test", base_url="https://example.invalid")


def _turn():
    return Turn({"messages": [], "turn_token": "tt_1", "allowed": True})


def test_default_is_blocking():
    """The call must have completed before record() returns."""
    kw = _client()
    done = []
    kw._request = lambda *a, **k: (done.append(True), {"ok": True})[1]  # type: ignore[method-assign]

    out = kw.record(_turn(), "an answer")

    assert done == [True], "record() returned before the request was made"
    assert out == {"ok": True}, out
    print("✓ default: the write is finished when record() returns")


def test_background_returns_immediately_and_still_sends():
    kw = _client()
    started = threading.Event()
    finished = threading.Event()

    def slow(*a, **k):
        started.set()
        time.sleep(0.3)
        finished.set()
        return {"ok": True}

    kw._request = slow  # type: ignore[method-assign]

    t0 = time.monotonic()
    out = kw.record(_turn(), "an answer", background=True)
    elapsed = time.monotonic() - t0

    assert out == {"queued": True}, out
    assert elapsed < 0.1, f"background record blocked for {elapsed:.3f}s"
    assert started.wait(1.0), "the request was never dispatched"
    assert finished.wait(1.0), "the request never completed"
    print(f"✓ background: returned in {elapsed * 1000:.0f}ms, request still sent")


def test_background_failure_never_raises():
    """The point of this mode is not delaying a reply — a failed learn costs one
    turn of memory and must not surface as an exception in someone's handler."""
    kw = _client()
    boom = threading.Event()

    def explode(*a, **k):
        boom.set()
        raise RuntimeError("connection reset")

    kw._request = explode  # type: ignore[method-assign]

    out = kw.record(_turn(), "an answer", background=True)
    assert out == {"queued": True}
    assert boom.wait(1.0), "the request was never attempted"
    time.sleep(0.05)  # let the thread finish unwinding
    print("✓ background: a failed write is swallowed, not raised")


def test_blocking_failure_still_raises():
    """The default must NOT swallow — a caller who waited deserves the error."""
    kw = _client()

    def explode(*a, **k):
        raise RuntimeError("connection reset")

    kw._request = explode  # type: ignore[method-assign]

    try:
        kw.record(_turn(), "an answer")
    except RuntimeError:
        print("✓ default: a failed write raises, as it should")
    else:
        raise AssertionError("blocking record swallowed an error")


if __name__ == "__main__":
    test_default_is_blocking()
    test_background_returns_immediately_and_still_sends()
    test_background_failure_never_raises()
    test_blocking_failure_still_raises()
    print("\n✅ record(background=) verified")
