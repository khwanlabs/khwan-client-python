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

from typing import Any, Dict, List, Optional

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
    def __init__(self, *, user_id: str, api_key: Optional[str] = None,
                 base_url: str = DEFAULT_BASE_URL,
                 model: Optional[str] = None, constitution: Optional[str] = None,
                 core: Optional[str] = None,
                 timeout: int = 60, memory: Any = None, embedder: Any = None):
        if memory is not None or embedder is not None:
            raise TypeError(
                "memory/embedder are server-managed in the hosted client; they are "
                "only configurable in the on-prem engine (khwan-engine, under license)."
            )
        if not api_key:
            raise ValueError("api_key is required (get one from your Khwan dashboard).")
        self.user_id = user_id
        # Selects the isolated core this client targets. Each named core is a fully
        # isolated brain (own memory/identity/learning) within the same account.
        # Omit for the account's default core.
        self.core = core
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        # session config forwarded to the server (model may be overridden by the
        # account's dashboard settings; constitution is a named profile reference).
        self._cfg = {k: v for k, v in
                     {"model": model, "constitution": constitution}.items() if v}

    # ---- transport ----
    def _headers(self) -> Dict[str, str]:
        h = {"X-API-Key": self._key}
        if self.user_id:
            h["X-Khwan-User"] = self.user_id  # identifies the end user (not a separate brain)
        if self.core:
            h["X-Khwan-Core"] = self.core  # select the isolated core
        return h

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        r = requests.request(method, self._base + path, headers=self._headers(),
                             json=body, timeout=self._timeout)
        if r.status_code // 100 != 2:
            msg = {
                401: "unauthorized — bad or missing API key",
                402: "payment required — add a payment method / upgrade your plan",
                429: "quota exceeded — you are over your plan's limit",
            }.get(r.status_code, r.text[:300])
            raise KhwanError(r.status_code, msg)
        return r.json() if r.content else {}

    # ---- the memory loop: prepare → (your model) → record ----
    def prepare(self, user_input: str) -> Turn:
        """Khwan builds the context (memory + constitution + coherence). No LLM call."""
        return Turn(self._request("POST", "/prepare",
                                  {"input": user_input, **self._cfg}))

    def record(self, turn: Turn, answer: str) -> dict:
        """Hand your model's answer back so Khwan can persist + learn."""
        return self._request("POST", "/record",
                             {"turn_token": turn.turn_token, "answer": answer})

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
