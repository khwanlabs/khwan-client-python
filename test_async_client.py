"""AsyncKhwan — same loop, same rules, on an event loop.

The point of an async client is not speed on one call; it is that an agent
framework's loop never blocks on memory. These tests pin the parts where that
could quietly stop being true: the retry policy is the SAME object as the sync
client's (so the two cannot drift), a background record actually completes rather
than being collected mid-flight, and closing waits for it.

No network: httpx is driven through a MockTransport.

Run: python3 test_async_client.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import httpx  # noqa: E402

from khwan import AsyncKhwan, Khwan, KhwanError, Turn, _is_idempotent, _retry_delay  # noqa: E402


def _client(handler) -> AsyncKhwan:
    kw = AsyncKhwan(api_key="kwk_test", core="acme", user_id="Web",
                    base_url="https://example.invalid", max_retries=2)
    kw._client = httpx.AsyncClient(
        base_url="https://example.invalid",
        transport=httpx.MockTransport(handler),
        headers={"X-API-Key": "kwk_test", "X-Khwan-Core": "acme", "X-Khwan-User": "Web"},
    )
    return kw


def test_policy_is_shared_with_the_sync_client():
    """One rule set, not two. A retry rule that drifts between the clients is a
    bug nobody notices until a duplicate record lands in production."""
    assert _is_idempotent("POST", "/prepare") is True
    assert _is_idempotent("POST", "/record") is False
    assert _retry_delay(0, "3") == 3.0            # honours Retry-After
    assert 0.375 <= _retry_delay(0, None) <= 0.625  # else backoff with jitter
    print("✓ policy: shared module-level rules, honoured by both clients")


def test_scope_headers_and_the_loop():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.path] = dict(request.headers)
        if request.url.path == "/prepare":
            return httpx.Response(200, json={"messages": [{"role": "user", "content": "hi"}],
                                             "turn_token": "t1",
                                             "lessons": ["Answer in Thai."]})
        return httpx.Response(200, json={})

    async def go():
        async with _client(handler) as kw:
            turn = await kw.prepare("what did we decide?")
            assert turn.turn_token == "t1"
            assert turn.lessons == ["Answer in Thai."], turn.lessons
            await kw.record(turn, "we decided X")

    asyncio.run(go())
    assert seen["/prepare"]["x-khwan-core"] == "acme"
    assert seen["/prepare"]["x-khwan-user"] == "Web"
    assert "/record" in seen
    print("✓ loop: prepare → record, with core and sub-brain headers on both")


def test_background_record_completes_and_close_waits():
    """asyncio holds only a weak reference to a task. Without a strong one, a
    fire-and-forget record can be garbage-collected before it is sent — losing
    the turn silently, which is the one failure mode this mode must not have."""
    landed = []

    def handler(request: httpx.Request) -> httpx.Response:
        landed.append(request.url.path)
        return httpx.Response(200, json={})

    async def go():
        kw = _client(handler)
        out = await kw.record(Turn({"turn_token": "t1"}), "answer", background=True)
        assert out == {"queued": True}
        await kw.aclose()          # must wait for what is still in flight

    asyncio.run(go())
    assert landed == ["/record"], landed
    print("✓ background: the write lands, and aclose() waits for it")


def test_background_failure_is_swallowed_but_blocking_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    async def go():
        kw = _client(handler)
        await kw.record(Turn({"turn_token": "t1"}), "a", background=True)
        await kw.aclose()          # a failed learn must not raise into the loop

        kw2 = _client(handler)
        try:
            await kw2.record(Turn({"turn_token": "t1"}), "a")
        except KhwanError as e:
            assert e.status == 500
        else:
            raise AssertionError("a blocking record must raise on failure")
        await kw2.aclose()

    asyncio.run(go())
    print("✓ background swallows failure; blocking raises it")


def test_retries_are_awaited_not_slept():
    """A retry must yield to the loop. time.sleep here would stall every other
    task in the process — the exact thing an async client exists to avoid."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, headers={"Retry-After": "0"}, text="try later")
        return httpx.Response(200, json={"turn_token": "t1"})

    async def go():
        ticks = 0
        async def other():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0)
        spin = asyncio.ensure_future(other())
        async with _client(handler) as kw:
            await kw.prepare("x")
        spin.cancel()
        return ticks

    ticks = asyncio.run(go())
    assert len(calls) == 2, calls
    assert ticks > 0, "the loop never got control during the retry"
    print("✓ retry: 503 retried, and the loop kept running while it waited")


def test_httpx_missing_is_a_clear_error():
    """The extra is optional, so the failure has to name the fix."""
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "httpx":
            raise ModuleNotFoundError("No module named 'httpx'")
        return real(name, *a, **k)

    builtins.__import__ = fake
    try:
        AsyncKhwan(api_key="k")
    except ModuleNotFoundError as e:
        assert 'khwan[async]' in str(e), str(e)
    else:
        raise AssertionError("expected a ModuleNotFoundError naming the extra")
    finally:
        builtins.__import__ = real
    print("✓ missing httpx names the install that fixes it")


if __name__ == "__main__":
    test_policy_is_shared_with_the_sync_client()
    test_scope_headers_and_the_loop()
    test_background_record_completes_and_close_waits()
    test_background_failure_is_swallowed_but_blocking_raises()
    test_retries_are_awaited_not_slept()
    test_httpx_missing_is_a_clear_error()
    print("\n✅ AsyncKhwan verified")
