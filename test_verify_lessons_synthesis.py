"""verify / lessons / synthesize — the endpoints the SDK was missing.

The website advertises the answer-gate and lesson review; until now neither was
reachable without hand-rolling HTTP. These pin the wiring, and one thing that
matters more than wiring: `synthesize` must never call a model itself. The whole
claim of the BYOM path is that the distillation is yours, so a future refactor
that "helpfully" adds a default model would break the product's promise, not just
a test.

Run: python3 test_verify_lessons_synthesis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from khwan import Khwan, Turn  # noqa: E402


def _client():
    return Khwan(api_key="kwk_test", base_url="https://example.invalid")


def _turn(token="tt_1"):
    return Turn({"messages": [], "turn_token": token, "allowed": True})


def _record_calls(kw, reply=None):
    """Swap _request for a recorder. Returns the list it appends (method, path, body)."""
    calls = []

    def fake(method, path, body=None):
        calls.append((method, path, body))
        return reply(method, path, body) if callable(reply) else (reply or {})

    kw._request = fake  # type: ignore[method-assign]
    return calls


def test_verify_sends_the_draft_and_the_turn():
    kw = _client()
    calls = _record_calls(kw, {"ok": False, "reason": "contradicts a stored preference"})

    out = kw.verify(_turn("tt_9"), "Absolutely, you're completely right.")

    assert calls == [("POST", "/verify",
                      {"answer": "Absolutely, you're completely right.",
                       "turn_token": "tt_9"})], calls
    assert out["ok"] is False
    print("✓ verify posts the draft with its turn token")


def test_verify_without_a_token_still_works():
    """A draft can be checked against the core alone — the token is optional."""
    kw = _client()
    calls = _record_calls(kw, {"ok": True})
    kw.verify(_turn(None), "a draft")
    assert calls[0][2] == {"answer": "a draft"}, calls
    print("✓ verify omits turn_token when the turn has none")


def test_verify_is_retryable_but_synthesize_prepare_is_not():
    """/verify never consumes the token server-side, so retrying it is safe.
    /synthesize/prepare mints a new batch each call — retrying orphans one."""
    kw = _client()
    assert kw._is_idempotent("POST", "/verify") is True
    assert kw._is_idempotent("POST", "/synthesize/prepare") is False
    assert kw._is_idempotent("POST", "/synthesize/record") is False
    print("✓ verify retries; synthesize prepare/record do not")


def test_lessons_unwraps_the_list():
    kw = _client()
    _record_calls(kw, {"lessons": [{"id": "L1", "response_text": "Answer in Thai."}]})
    out = kw.lessons()
    assert isinstance(out, list) and out[0]["id"] == "L1", out
    print("✓ lessons() returns the list, not the envelope")


def test_delete_and_edit_hit_the_right_verbs():
    kw = _client()
    calls = _record_calls(kw, {"status": "ok"})
    kw.delete_lesson("L1")
    kw.edit_lesson("L2", "Answer in Thai, briefly.")
    assert calls[0] == ("DELETE", "/lessons/L1", None), calls[0]
    assert calls[1] == ("PATCH", "/lessons/L2", {"text": "Answer in Thai, briefly."}), calls[1]
    print("✓ delete_lesson → DELETE, edit_lesson → PATCH with the new text")


def test_synthesize_never_calls_a_model_itself():
    """The claim the BYOM path rests on. `distill` is the only thing that may
    reach a model, and the SDK must call nothing else."""
    kw = _client()
    seen = []

    def fake(method, path, body=None):
        if path == "/synthesize/prepare":
            return {"synthesis_token": "st_1", "system": "SYS", "packets_scanned": 9,
                    "clusters": [{"id": "c0", "prompt": "P0"}, {"id": "c1", "prompt": "P1"}]}
        seen.append((method, path, body))
        return {"lessons_created": 1, "skipped": 1, "status": "ok"}

    kw._request = fake  # type: ignore[method-assign]

    distilled = []

    def distill(system, prompt):
        distilled.append((system, prompt))
        return "Answer in Thai." if prompt == "P0" else "NONE"

    out = kw.synthesize(distill=distill)

    assert distilled == [("SYS", "P0"), ("SYS", "P1")], distilled
    assert seen == [("POST", "/synthesize/record",
                     {"synthesis_token": "st_1",
                      "lessons": [{"cluster_id": "c0", "text": "Answer in Thai."},
                                  {"cluster_id": "c1", "text": "NONE"}]})], seen
    assert out["lessons_created"] == 1
    print("✓ synthesize calls YOUR distill per cluster and posts the results — nothing else")


def test_synthesize_with_nothing_to_learn_does_not_post():
    kw = _client()
    posted = []

    def fake(method, path, body=None):
        if path == "/synthesize/prepare":
            return {"synthesis_token": None, "clusters": [], "packets_scanned": 0}
        posted.append(path)
        return {}

    kw._request = fake  # type: ignore[method-assign]

    out = kw.synthesize(distill=lambda s, p: "should never be called")
    assert posted == [], posted
    assert out["status"] == "skipped", out
    print("✓ no token → no record call, and distill is never invoked")


def test_one_bad_cluster_does_not_lose_the_batch():
    kw = _client()
    sent = {}

    def fake(method, path, body=None):
        if path == "/synthesize/prepare":
            return {"synthesis_token": "st_1", "system": "SYS",
                    "clusters": [{"id": "c0", "prompt": "P0"}, {"id": "c1", "prompt": "P1"}]}
        sent.update(body or {})
        return {"lessons_created": 1}

    kw._request = fake  # type: ignore[method-assign]

    def flaky(system, prompt):
        if prompt == "P0":
            raise RuntimeError("model timed out")
        return "Answer in Thai."

    kw.synthesize(distill=flaky)

    assert sent["lessons"] == [{"cluster_id": "c0", "text": None},
                               {"cluster_id": "c1", "text": "Answer in Thai."}], sent
    print("✓ a cluster whose distill raises is skipped; the rest still record")


if __name__ == "__main__":
    test_verify_sends_the_draft_and_the_turn()
    test_verify_without_a_token_still_works()
    test_verify_is_retryable_but_synthesize_prepare_is_not()
    test_lessons_unwraps_the_list()
    test_delete_and_edit_hit_the_right_verbs()
    test_synthesize_never_calls_a_model_itself()
    test_synthesize_with_nothing_to_learn_does_not_post()
    test_one_bad_cluster_does_not_lose_the_batch()
    print("\n✅ verify / lessons / synthesize verified")
