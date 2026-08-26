"""Two credential kinds, and they are not interchangeable at the wire.

An API key is a long-lived account secret sent as `X-API-Key`. A bearer token is
an OAuth access token minted for one user, and the API reads it from
`Authorization`. A JWT put in the api_key slot does not quietly also work — it is
looked up as a key, misses, and 401s before the bearer path is reached. These
tests pin the header each one produces, because that difference is invisible
until a request fails somewhere else.

Offline: constructing a client makes no network call, and the assertions read the
headers the client would send.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "src"))

from khwan import Khwan, _auth_headers  # noqa: E402

JWT = "eyJhbGciOiJSUzI1NiJ9.stub.stub"


# ── which header goes out ─────────────────────────────────────────────────────

def test_api_key_goes_in_the_api_key_header():
    h = Khwan(api_key="kwk_live_x")._headers()
    assert h["X-API-Key"] == "kwk_live_x"
    assert "Authorization" not in h


def test_bearer_goes_in_authorization():
    h = Khwan(bearer_token=JWT)._headers()
    assert h["Authorization"] == f"Bearer {JWT}"
    assert "X-API-Key" not in h


def test_a_bearer_client_never_sends_an_api_key_header():
    """The whole point: X-API-Key resolves as a key and 401s before bearer runs."""
    assert "X-API-Key" not in _auth_headers(None, None, None, JWT)


# ── brain selection is orthogonal to the credential ───────────────────────────

def test_core_and_user_ride_along_with_a_bearer():
    h = Khwan(bearer_token=JWT, core="acme", user_id="web")._headers()
    assert h["Authorization"] == f"Bearer {JWT}"
    assert h["X-Khwan-Core"] == "acme"
    assert h["X-Khwan-User"] == "web"


def test_core_and_user_still_ride_along_with_a_key():
    h = Khwan(api_key="kwk_live_x", core="acme", user_id="web")._headers()
    assert (h["X-API-Key"], h["X-Khwan-Core"], h["X-Khwan-User"]) == (
        "kwk_live_x", "acme", "web")


# ── exactly one credential ────────────────────────────────────────────────────

def test_neither_is_refused():
    with pytest.raises(ValueError, match="exactly one"):
        Khwan()


def test_both_is_refused():
    """Ambiguous rather than harmless — one of them would silently win."""
    with pytest.raises(ValueError, match="exactly one"):
        Khwan(api_key="kwk_live_x", bearer_token=JWT)


def test_an_empty_string_is_not_a_credential():
    with pytest.raises(ValueError, match="exactly one"):
        Khwan(bearer_token="")


# ── the async client agrees ───────────────────────────────────────────────────

def test_async_client_takes_a_bearer_too():
    httpx = pytest.importorskip("httpx")  # noqa: F841
    from khwan import AsyncKhwan
    client = AsyncKhwan(bearer_token=JWT, core="acme")
    assert client._bearer == JWT
    assert _auth_headers(client._key, client.user_id, client.core,
                         client._bearer)["Authorization"] == f"Bearer {JWT}"


def test_async_client_refuses_both():
    pytest.importorskip("httpx")
    from khwan import AsyncKhwan
    with pytest.raises(ValueError, match="exactly one"):
        AsyncKhwan(api_key="kwk_live_x", bearer_token=JWT)


# ── what an error says ────────────────────────────────────────────────────────
# The hint used to be the whole message and the server's text was discarded,
# which hides the only part that separates causes.

from khwan import _error_message  # noqa: E402


def test_the_servers_own_words_survive():
    """`Invalid bearer token` and `Missing credentials` are different problems."""
    rejected = _error_message(401, "Invalid bearer token", True)
    absent = _error_message(401, "Missing credentials (X-API-Key or Authorization: Bearer)", True)
    assert "Invalid bearer token" in rejected
    assert "Missing credentials" in absent
    assert rejected != absent


def test_a_bearer_caller_is_not_sent_looking_for_an_api_key():
    """The wrong turn: an OAuth caller has no API key to check."""
    msg = _error_message(401, "Invalid bearer token", True)
    assert "API key" not in msg
    assert "authorize again" in msg


def test_an_api_key_caller_still_hears_about_the_key():
    msg = _error_message(401, "Invalid API key", False)
    assert "API key" in msg


def test_other_statuses_keep_their_hint_and_gain_the_detail():
    msg = _error_message(402, "per-user memory: the free plan allows 3 sub-brain(s)")
    assert "upgrade" in msg
    assert "free plan allows 3" in msg


def test_an_unmapped_status_is_just_the_server_text():
    assert _error_message(500, "internal error") == "internal error"


def test_no_detail_leaves_the_hint_alone():
    assert _error_message(429, "") == "rate limited / over your plan's limit — retry later"
