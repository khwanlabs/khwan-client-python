"""record(occurred_at=) — dating a packet when the turn actually happened.

An import stamps every packet with the minute it ran. Replaying two months of
transcripts produced packets spanning twelve minutes, so retrieval could not tell
a decision from June from one made this morning — and a recency tiebreak has
nothing to work with on exactly the data most likely to hold a superseded fact.

Run: python3 test_occurred_at.py
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import requests  # noqa: E402

from khwan import Khwan, Turn  # noqa: E402

WHEN = datetime(2026, 6, 22, 11, 50, tzinfo=timezone.utc)


class _Resp:
    status_code, headers, content = 200, {}, b"{}"
    text = "{}"

    def json(self):
        return {}


def _capturing_client():
    seen: list = []

    def capture(method, url, **kw):
        seen.append(kw.get("json") or {})
        return _Resp()

    requests.request = capture
    return Khwan(api_key="k", base_url="https://example.invalid"), seen


def test_absent_unless_asked_for():
    """A normal turn must not start carrying a field it never needed, and an
    older engine must keep seeing the body it always did."""
    real = requests.request
    try:
        kw, seen = _capturing_client()
        kw.record(Turn({"turn_token": "t"}), "an answer")
        assert "occurred_at" not in seen[-1], seen[-1]
    finally:
        requests.request = real
    print("✓ omitted when not given — no new field on an ordinary turn")


def test_sent_as_iso_when_given():
    real = requests.request
    try:
        kw, seen = _capturing_client()
        kw.record(Turn({"turn_token": "t"}), "an answer", occurred_at=WHEN)
        assert seen[-1]["occurred_at"].startswith("2026-06-22T11:50"), seen[-1]
    finally:
        requests.request = real
    print("✓ sent as ISO 8601 when given")


def test_sent_on_the_background_path_too():
    """The background path builds the body once and hands it to a thread; the
    timestamp has to be in that copy, not added afterwards."""
    real = requests.request
    try:
        kw, seen = _capturing_client()
        kw.record(Turn({"turn_token": "t"}), "an answer",
                  background=True, occurred_at=WHEN)
        # Waited for directly rather than via flush(), so this test does not
        # depend on a change that lives in another branch.
        deadline = time.monotonic() + 5
        while not seen and time.monotonic() < deadline:
            time.sleep(0.02)
        assert seen, "the background write never happened"
        assert seen[-1].get("occurred_at", "").startswith("2026-06-22"), seen[-1]
    finally:
        requests.request = real
    print("✓ sent on the background path, where the body is built once")


if __name__ == "__main__":
    test_absent_unless_asked_for()
    test_sent_as_iso_when_given()
    test_sent_on_the_background_path_too()
    print("\n✅ occurred_at verified")
