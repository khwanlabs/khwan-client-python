"""flush() — the fire-and-forget write that must not vanish at exit.

record(background=True) sends on a daemon thread, and the interpreter does not
wait for daemons. In a CLI, a serverless handler, or any script that ends soon
after its last turn, the write can be killed mid-flight with no error anywhere —
the turn is simply never learned, and that reads as unreliable memory rather than
as a missing call.

The real proof is the subprocess test: a process that exits immediately after a
background record still lands the write.

Run: python3 test_flush.py
"""

import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import requests  # noqa: E402

from khwan import Khwan, Turn  # noqa: E402


class _Resp:
    status_code, headers, content = 200, {}, b"{}"
    text = "{}"

    def json(self):
        return {}


def test_flush_waits_for_the_write():
    started, finished = threading.Event(), []

    def slow(method, url, **kw):
        started.set()
        time.sleep(0.25)
        finished.append(url)
        return _Resp()

    real, requests.request = requests.request, slow
    try:
        kw = Khwan(api_key="k", base_url="https://example.invalid")
        kw.record(Turn({"turn_token": "t"}), "a", background=True)
        started.wait(2)
        assert finished == [], "precondition: the write should still be in flight"
        pending = kw.flush()
        assert pending == 1, pending
        assert len(finished) == 1, "flush returned before the write landed"
    finally:
        requests.request = real
    print("✓ flush: returns only once the in-flight write has landed")


def test_flush_is_bounded_and_does_not_raise():
    """A hung request must not turn into a hang at exit, or an exception there."""
    def hang(method, url, **kw):
        time.sleep(10)
        return _Resp()

    real, requests.request = requests.request, hang
    try:
        kw = Khwan(api_key="k", base_url="https://example.invalid")
        kw.record(Turn({"turn_token": "t"}), "a", background=True)
        t0 = time.monotonic()
        kw.flush(timeout=0.2)          # returns, does not raise
        assert time.monotonic() - t0 < 2, "flush ignored its timeout"
    finally:
        requests.request = real
    print("✓ flush: a hung write is given up on, not raised and not waited out")


def test_flush_with_nothing_pending_is_a_no_op():
    kw = Khwan(api_key="k", base_url="https://example.invalid")
    assert kw.flush() == 0
    print("✓ flush: nothing in flight → nothing to wait for")


def test_a_process_that_exits_immediately_still_lands_the_write():
    """The actual failure this fixes, in a real interpreter that really exits.

    Without the atexit hook the daemon thread is killed and `landed` stays empty.
    """
    script = r'''
import sys, threading, time
sys.path.insert(0, %r)
import requests
from khwan import Khwan, Turn

landed = []

class R:
    status_code, headers, content, text = 200, {}, b"{}", "{}"
    def json(self): return {}

def slow(method, url, **kw):
    time.sleep(0.3)          # still in flight when the script falls off the end
    landed.append(url)
    with open(%r, "a") as f:
        f.write(url + "\n")
    return R()

requests.request = slow
kw = Khwan(api_key="k", base_url="https://example.invalid")
kw.record(Turn({"turn_token": "t"}), "answer", background=True)
# no flush() call, no sleep — the process ends here, as a CLI would
''' % (str(Path(__file__).parent / "src"), str(Path(__file__).parent / ".flush-probe"))

    probe = Path(__file__).parent / ".flush-probe"
    probe.unlink(missing_ok=True)
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
    assert probe.exists() and "/record" in probe.read_text(), (
        "the process exited before the background write landed — the failure "
        "flush() exists to prevent")
    probe.unlink()
    print("✓ exit: a script that ends immediately still lands its last write")


if __name__ == "__main__":
    test_flush_waits_for_the_write()
    test_flush_is_bounded_and_does_not_raise()
    test_flush_with_nothing_pending_is_a_no_op()
    test_a_process_that_exits_immediately_still_lands_the_write()
    print("\n✅ flush verified")
