"""Khwan hosted client (public).

A thin HTTP wrapper — contains NO Khwan engine code. The Brain (memory,
constitution, coherence, learning) runs on the Khwan server; this client just
connects and hands results back to your app.

Positioning: Khwan is a pure **AI-memory layer** — it never runs a model. You
always call your own model (BYOM). The only loop is
`prepare` → (your model) → `record`:

  kw     = Khwan(api_key="kwk_live_xxx", user_id="alice")
  turn   = kw.prepare("remember I like short answers")   # no LLM on Khwan's side
  answer = your_model(turn.messages)                      # your model, your key
  kw.record(turn, answer)                                 # Khwan learns

Isolated cores — many separate brains within one account; pass ``core=``:

  test = Khwan(api_key="kwk_live_xxx", user_id="alice", core="test")
  kw.cores()   # list the account's cores (default included)

`memory=`/`embedder=` are NOT configurable here — they are server-managed. They
exist only in the on-prem engine (shipped under license).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import random
import time
from email.utils import parsedate_to_datetime

import requests

__version__ = "0.1.0"
DEFAULT_BASE_URL = "https://api.khwan.ai"


class KhwanError(RuntimeError):
    """Raised on a non-2xx response. `.status` is the HTTP code."""

    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(message)


class Turn:
    """Context Khwan prepared for one turn. Feed `.messages` to your own model,
    then pass this object (plus the model's answer) back to `kw.record()`."""

    def __init__(self, data: Dict[str, Any]):
        self._d = data

    @property
    def messages(self) -> List[Dict[str, str]]:
        return self._d.get("messages", [])

    @property
    def coherence(self) -> Optional[float]:
        return self._d.get("coherence")

    @property
    def sources(self) -> List[Any]:
        return self._d.get("sources", [])

    @property
    def allowed(self) -> bool:
        return bool(self._d.get("allowed", True))

    @property
    def reason(self) -> Optional[str]:
        return self._d.get("reason")

    @property
    def turn_token(self) -> Optional[str]:
        return self._d.get("turn_token")

    def raw(self) -> Dict[str, Any]:
        return self._d


class Khwan:
    def __init__(self, *, user_id: Optional[str] = None, api_key: Optional[str] = None,
                 base_url: str = DEFAULT_BASE_URL,
                 model: Optional[str] = None, constitution: Optional[str] = None,
                 core: Optional[str] = None,
                 timeout: int = 60, max_retries: int = 2,
                 memory: Any = None, embedder: Any = None):
        if memory is not None or embedder is not None:
            raise TypeError(
                "memory/embedder are server-managed in the hosted client; they are "
                "only configurable in the on-prem engine (khwan-engine, under license)."
            )
        if not api_key:
            raise ValueError("api_key is required (get one from your Khwan dashboard).")
        # OPTIONAL end-user id. Omit for one shared brain per account/core. Set it to give
        # each of your end-users a fully ISOLATED sub-brain (one key → a private brain per
        # user); requires a paid plan. Combines with `core`: account::<core>::@<user>.
        self.user_id = user_id
        # Selects the isolated core this client targets. Each named core is a fully
        # isolated brain (own memory/identity/learning) within the same account.
        # Omit for the account's default core.
        self.core = core
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        # Auto-retry transient failures (429/502/503/504 honoring Retry-After, plus
        # network errors on idempotent calls) with exponential backoff + jitter.
        self._max_retries = max(0, max_retries)
        # session config forwarded to the server (model may be overridden by the
        # account's dashboard settings; constitution is a named profile reference).
        self._cfg = {k: v for k, v in
                     {"model": model, "constitution": constitution}.items() if v}

    # ---- transport ----
    def _headers(self) -> Dict[str, str]:
        h = {"X-API-Key": self._key}
        if self.user_id:
            h["X-Khwan-User"] = self.user_id  # optional: isolated sub-brain per end-user
        if self.core:
            h["X-Khwan-Core"] = self.core  # select the isolated core
        return h

    # Reads + prepare are safe to retry after an ambiguous network error (the
    # request may already have been processed); record/sync/reset are NOT (replaying
    # could double-apply or hit an already-consumed turn_token). A rejected
    # 429/502/503/504 is safe to retry regardless — it wasn't processed.
    _RETRY_STATUS = frozenset({429, 502, 503, 504})

    def _is_idempotent(self, method: str, path: str) -> bool:
        return method == "GET" or path == "/prepare"

    def _retry_delay(self, attempt: int, resp: "Optional[requests.Response]") -> float:
        """Seconds to wait: honor Retry-After if present, else exponential
        (0.5, 1, 2, … capped 20s) with ±25% jitter."""
        if resp is not None:
            ra = resp.headers.get("Retry-After")
            if ra:
                try:
                    return min(float(ra), 60.0)
                except ValueError:
                    try:
                        dt = parsedate_to_datetime(ra)
                        return max(0.0, dt.timestamp() - time.time())
                    except (TypeError, ValueError):
                        pass
        base = min(0.5 * (2 ** attempt), 20.0)
        return base * (0.75 + random.random() * 0.5)

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        idempotent = self._is_idempotent(method, path)
        attempt = 0
        while True:
            try:
                r = requests.request(method, self._base + path,
                                     headers=self._headers(), json=body,
                                     timeout=self._timeout)
            except requests.RequestException as e:
                # Network/timeout error — retry only idempotent calls.
                if idempotent and attempt < self._max_retries:
                    time.sleep(self._retry_delay(attempt, None))
                    attempt += 1
                    continue
                raise KhwanError(0, f"network error: {e}")

            if r.status_code in self._RETRY_STATUS and attempt < self._max_retries:
                time.sleep(self._retry_delay(attempt, r))
                attempt += 1
                continue

            if r.status_code // 100 != 2:
                msg = {
                    401: "unauthorized — bad or missing API key",
                    402: "payment required — add a payment method / upgrade your plan",
                    429: "rate limited / over your plan's limit — retry later",
                }.get(r.status_code, r.text[:300])
                raise KhwanError(r.status_code, msg)
            return r.json() if r.content else {}

    # ---- the memory loop: prepare → (your model) → record ----
    def prepare(self, user_input: str) -> Turn:
        """Khwan builds the context (memory + constitution + coherence). No LLM call."""
        return Turn(self._request("POST", "/prepare",
                                  {"input": user_input, **self._cfg}))

    def record(self, turn: Turn, answer: str, *, background: bool = False) -> dict:
        """Hand your model's answer back so Khwan can persist + learn.

        Blocking by default, and that default is deliberate. ``prepare`` for the
        next turn retrieves what has been written, so a ``record`` still in flight
        means the turn you just had is missing from the context of the turn after
        it — intermittently, under load, in a way that reads as "the memory is
        flaky" rather than as a race. Correctness first; opt into the latency win.

        ``background=True`` dispatches on a daemon thread and returns immediately
        with ``{"queued": True}``. Use it when the turn is the last one (a one-shot
        job, a webhook reply) or when the next ``prepare`` is far enough away that
        the write will have landed. Failures are swallowed — a lost record costs
        one turn of learning, and the point of this mode is never to delay a reply.

        For a strict sequence at lower latency, prefer ``record`` on a thread you
        join before the next ``prepare`` rather than fire-and-forget.
        """
        if not background:
            return self._request("POST", "/record",
                                 {"turn_token": turn.turn_token, "answer": answer})

        def _send() -> None:
            try:
                self._request("POST", "/record",
                              {"turn_token": turn.turn_token, "answer": answer})
            except Exception:  # noqa: BLE001 — a failed learn must not raise into a thread
                pass

        threading.Thread(target=_send, daemon=True, name="khwan-record").start()
        return {"queued": True}

    # ---- learning / inspection ----
    def sync(self) -> dict:
        return self._request("POST", "/sync")

    def memory(self, limit: int = 20) -> dict:
        return self._request("GET", f"/memory?limit={limit}")

    def metrics(self) -> dict:
        return self._request("GET", "/metrics")

    def cores(self) -> List[Dict[str, Any]]:
        """List the isolated cores on this account. The default core is included,
        with ``is_default`` True."""
        return self._request("GET", "/cores")  # type: ignore[return-value]


__all__ = ["Khwan", "Turn", "KhwanError"]
