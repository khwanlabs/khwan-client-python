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

import asyncio
import threading
from typing import Any, Callable, Dict, List, Optional

import random
import time
from email.utils import parsedate_to_datetime

import requests

__version__ = "0.3.0"
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
    def lessons(self) -> List[str]:
        """Standing rules synthesis distilled from many past turns.

        Separate from `sources`, which is the raw context retrieved for THIS turn.
        Both are already inside `messages`; they are surfaced so a caller building
        its own context — a recall tool, a subagent brief — can take the distilled
        rules without replaying the whole prepared prompt. Empty against an engine
        that predates them.
        """
        return [str(x) for x in (self._d.get("lessons") or []) if str(x).strip()]

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


# ── Transport policy, shared by both clients ──────────────────────────────────
# Kept at module level rather than on the class so the sync and async clients
# cannot drift: a retry rule that holds for one holds for the other.

# Rejected before processing, so retrying is always safe.
RETRY_STATUS = frozenset({429, 502, 503, 504})


def _is_idempotent(method: str, path: str) -> bool:
    """Whether an AMBIGUOUS failure (network error, timeout) may be retried.

    Reads and /prepare are safe: the request may already have been processed, and
    doing it twice costs nothing. record/sync/reset are not — replaying could
    double-apply or hit an already-consumed turn_token.

    /verify is here because the server documents it as non-destructive: it never
    consumes the token and never persists, so a retry after a dropped connection
    beats failing a gate check. /synthesize/prepare is deliberately absent — each
    call mints a new batch token, and a retry would orphan one.
    """
    return method == "GET" or path in ("/prepare", "/verify")


def _retry_delay(attempt: int, retry_after: Optional[str]) -> float:
    """Seconds to wait: honour Retry-After when the server sent one, else
    exponential (0.5, 1, 2, … capped at 20s) with ±25% jitter."""
    if retry_after:
        try:
            return min(float(retry_after), 60.0)
        except ValueError:
            try:
                dt = parsedate_to_datetime(retry_after)
                return max(0.0, dt.timestamp() - time.time())
            except (TypeError, ValueError):
                pass
    base = min(0.5 * (2 ** attempt), 20.0)
    return base * (0.75 + random.random() * 0.5)


def _error_message(status: int, text: str) -> str:
    """A message that says what to DO, for the statuses with an obvious answer."""
    return {
        401: "unauthorized — bad or missing API key",
        402: "payment required — add a payment method / upgrade your plan",
        404: "not found — check the core in X-Khwan-Core exists",
        429: "rate limited / over your plan's limit — retry later",
    }.get(status, text[:300])


def _auth_headers(api_key: str, user_id: Optional[str], core: Optional[str]) -> Dict[str, str]:
    h = {"X-API-Key": api_key}
    if user_id:
        h["X-Khwan-User"] = user_id   # optional: isolated sub-brain per end-user
    if core:
        h["X-Khwan-Core"] = core      # select the isolated core
    return h


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
        return _auth_headers(self._key, self.user_id, self.core)

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        idempotent = _is_idempotent(method, path)
        attempt = 0
        while True:
            try:
                r = requests.request(method, self._base + path,
                                     headers=self._headers(), json=body,
                                     timeout=self._timeout)
            except requests.RequestException as e:
                # Network/timeout error — retry only idempotent calls.
                if idempotent and attempt < self._max_retries:
                    time.sleep(_retry_delay(attempt, None))
                    attempt += 1
                    continue
                raise KhwanError(0, f"network error: {e}")

            if r.status_code in RETRY_STATUS and attempt < self._max_retries:
                time.sleep(_retry_delay(attempt, r.headers.get("Retry-After")))
                attempt += 1
                continue

            if r.status_code // 100 != 2:
                raise KhwanError(r.status_code, _error_message(r.status_code, r.text))
            return r.json() if r.content else {}

    # ---- the memory loop: prepare → (your model) → record ----
    def prepare(self, user_input: str) -> Turn:
        """Khwan builds the context (memory + constitution + coherence). No LLM call."""
        return Turn(self._request("POST", "/prepare",
                                  {"input": user_input, **self._cfg}))

    def record(self, turn: Turn, answer: str, *, background: bool = False,
               occurred_at: "Optional[datetime]" = None) -> dict:
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

        ``occurred_at`` says when the turn HAPPENED, when that is not now —
        importing history, replaying a transcript. Without it an import stamps
        every packet with the minute it ran, and retrieval cannot tell a decision
        from June from one made this morning. The server clamps it to the present.
        """
        body: Dict[str, Any] = {"turn_token": turn.turn_token, "answer": answer}
        if occurred_at is not None:
            body["occurred_at"] = occurred_at.isoformat()
        if not background:
            return self._request("POST", "/record", body)

        def _send() -> None:
            try:
                self._request("POST", "/record", body)
            except Exception:  # noqa: BLE001 — a failed learn must not raise into a thread
                pass

        threading.Thread(target=_send, daemon=True, name="khwan-record").start()
        return {"queued": True}

    def verify(self, turn: Turn, draft: str) -> dict:
        """Score a draft answer against the brain BEFORE you ship it.

        ``prepare`` gates the turn; this gates the *answer*. Returns
        ``{ok, reason, coherence, contradiction}`` — ship when ``ok`` is true,
        regenerate or route to a human when it is not.

        Non-destructive: it never consumes the turn token, so ``record`` still
        works normally afterwards. Treat a transport failure as ``ok`` rather than
        blocking a reply on a network blip.
        """
        body: Dict[str, Any] = {"answer": draft}
        if turn.turn_token:
            body["turn_token"] = turn.turn_token
        return self._request("POST", "/verify", body)

    # ---- lesson review ----
    def lessons(self, limit: int = 50) -> List[Dict[str, Any]]:
        """The standing rules synthesis has written for this core.

        A lesson is a behaviour rule, not a fact — the most-reinforced ones are
        injected on every turn regardless of relevance. ``source_link`` on each
        entry points back at the turns it was distilled from.
        """
        out = self._request("GET", f"/lessons?limit={limit}")
        return out.get("lessons", [])  # type: ignore[union-attr]

    def delete_lesson(self, lesson_id: str) -> dict:
        """Remove a rule the agent should not have learned.

        Retrieval only reinforces — a lesson that gets used has its expiry
        extended — so a rule that is wrong but relevant never expires on its own.
        This is the only negative signal in the system.
        """
        return self._request("DELETE", f"/lessons/{lesson_id}")

    def edit_lesson(self, lesson_id: str, text: str) -> dict:
        """Correct a rule's wording, keeping its sources and use history.

        The server re-embeds from the new text in the same write, so retrieval
        matches what the rule now says rather than what it used to.
        """
        return self._request("PATCH", f"/lessons/{lesson_id}", {"text": text})

    # ---- BYOM synthesis: the learning loop, with your model in the middle ----
    def synthesize_prepare(self) -> dict:
        """Cluster recent turns and hand them back for YOUR model to distil.

        Returns ``{synthesis_token, system, clusters, packets_scanned}``. No model
        is called — on this path no packet text reaches a provider Khwan chose.
        ``synthesis_token`` is None when there is nothing to learn.

        Feed ``system`` plus each cluster's ``prompt`` to your model; it answers
        with one imperative rule, or the literal ``NONE``.
        """
        return self._request("POST", "/synthesize/prepare")

    def synthesize_record(self, synthesis_token: str,
                          lessons: List[Dict[str, Any]]) -> dict:
        """Store the rules your model distilled. ``lessons`` is a list of
        ``{"cluster_id": ..., "text": ... or None}``; None records that the cluster
        held nothing durable, which is an outcome rather than a failure."""
        return self._request("POST", "/synthesize/record",
                             {"synthesis_token": synthesis_token, "lessons": lessons})

    def synthesize(self, distill: Callable[[str, str], Optional[str]]) -> dict:
        """Run a whole BYOM synthesis pass, with ``distill`` as your model.

        ``distill(system, prompt)`` is called once per cluster and returns the rule,
        or None/"NONE" when there is nothing durable in it. The SDK never calls a
        model itself — this only owns the loop and the token, which is the part that
        is tedious rather than the part that is yours::

            kw.synthesize(lambda system, prompt: my_llm(system, prompt))

        A cluster whose ``distill`` raises is skipped rather than losing the batch;
        the others still record. Returns the server's summary, or a skipped-shaped
        dict when there was nothing to learn.
        """
        plan = self.synthesize_prepare()
        token = plan.get("synthesis_token")
        if not token:
            return {"lessons_created": 0, "skipped": 0,
                    "packets_scanned": plan.get("packets_scanned", 0),
                    "status": "skipped"}

        system = plan.get("system", "")
        out: List[Dict[str, Any]] = []
        for c in plan.get("clusters", []):
            try:
                text = distill(system, c["prompt"])
            except Exception:  # noqa: BLE001 — one bad cluster must not lose the batch
                text = None
            out.append({"cluster_id": c["id"], "text": text})
        return self.synthesize_record(token, out)

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


__all__ = ["Khwan", "AsyncKhwan", "Turn", "KhwanError"]


# ── Async client ──────────────────────────────────────────────────────────────

class AsyncKhwan:
    """The same loop for an async host: ``prepare`` → (your model) → ``record``.

    Every agent framework worth integrating is async — an event loop calling a
    blocking client either stalls it or grows a thread pool to hide the stall, so
    each integration ends up rewriting this transport instead of using one.

    Same surface, same retry rules (they are module-level, so the two cannot
    drift), same errors. Needs httpx::

        pip install "khwan[async]"

    Reuses one connection pool, so hold it open rather than making one per turn::

        async with AsyncKhwan(api_key="kwk_live_xxx", core="acme") as kw:
            turn   = await kw.prepare("what did we decide about billing?")
            answer = await your_model(turn.messages)
            await kw.record(turn, answer)

    Without the context manager, call ``aclose()`` when finished.
    """

    def __init__(self, *, user_id: Optional[str] = None, api_key: Optional[str] = None,
                 base_url: str = DEFAULT_BASE_URL,
                 model: Optional[str] = None, constitution: Optional[str] = None,
                 core: Optional[str] = None,
                 timeout: int = 60, max_retries: int = 2):
        try:
            import httpx  # noqa: F401
        except ModuleNotFoundError as e:  # pragma: no cover - import guard
            raise ModuleNotFoundError(
                'AsyncKhwan needs httpx — install with: pip install "khwan[async]"'
            ) from e
        if not api_key:
            raise ValueError("api_key is required (get one from your Khwan dashboard).")
        self.user_id = user_id
        self.core = core
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max(0, max_retries)
        self._cfg = {k: v for k, v in
                     {"model": model, "constitution": constitution}.items() if v}
        self._client = None  # created lazily, on the loop that will use it
        # Fire-and-forget records are held here: asyncio keeps only a weak
        # reference to a task, so without this they can be collected mid-flight.
        self._pending: set = set()

    async def __aenter__(self) -> "AsyncKhwan":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    def _http(self):
        import httpx
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base,
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                headers=_auth_headers(self._key, self.user_id, self.core),
            )
        return self._client

    async def aclose(self) -> None:
        """Close the pool, after letting any background records finish."""
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        import httpx
        idempotent = _is_idempotent(method, path)
        attempt = 0
        while True:
            try:
                r = await self._http().request(method, path, json=body)
            except httpx.HTTPError as e:
                if idempotent and attempt < self._max_retries:
                    await asyncio.sleep(_retry_delay(attempt, None))
                    attempt += 1
                    continue
                raise KhwanError(0, f"network error: {e}")

            if r.status_code in RETRY_STATUS and attempt < self._max_retries:
                await asyncio.sleep(_retry_delay(attempt, r.headers.get("Retry-After")))
                attempt += 1
                continue

            if r.status_code // 100 != 2:
                raise KhwanError(r.status_code, _error_message(r.status_code, r.text))
            return r.json() if r.content else {}

    # ---- the memory loop ----
    async def prepare(self, user_input: str) -> Turn:
        """Khwan builds the context (memory + constitution + coherence). No LLM call."""
        return Turn(await self._request("POST", "/prepare",
                                        {"input": user_input, **self._cfg}))

    async def record(self, turn: Turn, answer: str, *, background: bool = False,
                     occurred_at: "Optional[datetime]" = None) -> dict:
        """Hand your model's answer back so Khwan can persist + learn.

        Awaited by default, and that default is deliberate: the next ``prepare``
        retrieves what has been written, so a record still in flight means the turn
        you just had is missing from the context of the turn after it —
        intermittently, under load, reading as "the memory is flaky" rather than as
        a race.

        ``background=True`` schedules it and returns ``{"queued": True}``. Use it
        when the turn is the last one, or when the next prepare is far enough away.
        Failures are swallowed; ``aclose()`` waits for what is still in flight.
        """
        body: Dict[str, Any] = {"turn_token": turn.turn_token, "answer": answer}
        if occurred_at is not None:
            body["occurred_at"] = occurred_at.isoformat()
        if not background:
            return await self._request("POST", "/record", body)

        async def _send() -> None:
            try:
                await self._request("POST", "/record", body)
            except Exception:  # noqa: BLE001 — a failed learn must not raise into the loop
                pass

        task = asyncio.ensure_future(_send())
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        return {"queued": True}

    async def verify(self, turn: Turn, draft: str) -> dict:
        """Score a draft answer against the brain BEFORE you ship it.

        ``prepare`` gates the turn; this gates the *answer*. Non-destructive — it
        never consumes the turn token. Treat a transport failure as ``ok`` rather
        than blocking a reply on a network blip.
        """
        body: Dict[str, Any] = {"answer": draft}
        if turn.turn_token:
            body["turn_token"] = turn.turn_token
        return await self._request("POST", "/verify", body)

    # ---- inspection ----
    async def lessons(self, limit: int = 50) -> List[Dict[str, Any]]:
        """The standing rules synthesis has written for this core."""
        return await self._request("GET", f"/lessons?limit={limit}")  # type: ignore[return-value]

    async def memory(self, limit: int = 20) -> dict:
        return await self._request("GET", f"/memory?limit={limit}")

    async def cores(self) -> List[Dict[str, Any]]:
        """List the isolated cores on this account."""
        return await self._request("GET", "/cores")  # type: ignore[return-value]
