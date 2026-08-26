# khwan (Python client)

The Khwan hosted client — a thin HTTP wrapper with **no engine code**. Khwan
is a memory layer (memory + constitutional identity + coherence + learning) that
runs on our server; you bring your own model.

**Khwan never generates text.** It is a pure AI-memory layer — you always call
your own model. The only loop is `prepare` → your model → `record`.

```bash
pip install khwan
```

## The memory loop — you call your own model
```python
from khwan import Khwan

kw = Khwan(api_key="kwk_live_xxx", user_id="alice")

turn   = kw.prepare("remember I prefer short answers in Thai")  # Khwan builds context, no LLM
answer = your_model(turn.messages)                              # YOUR model + key
kw.record(turn, answer)                                          # Khwan persists + learns

# `record` waits by default, on purpose: `prepare` for the next turn reads what
# has been written, so a record still in flight drops this turn from the next
# turn's context — only under load, which makes it read as flaky memory rather
# than as a race. Skip the wait when the turn is the last one:
kw.record(turn, answer, background=True)                         # → {"queued": True}

# The send runs on a daemon thread, and the interpreter does not wait for those.
# In a CLI, a serverless handler, or any script that ends soon after its last
# turn, that write can be killed mid-flight — no error, the turn simply never
# learned. Wait for it before you exit:
kw.flush()                                                       # → how many were in flight
```

A `flush()` also runs automatically at interpreter exit, bounded to five seconds,
so forgetting the call costs latency rather than the turn.

## On an event loop

Every agent framework worth integrating is async, and a blocking client on an
event loop either stalls it or grows a thread pool to hide the stall. `AsyncKhwan`
is the same loop, the same retry rules — they live at module level, so the two
clients cannot drift — and the same errors.

```bash
pip install "khwan[async]"
```

```python
from khwan import AsyncKhwan

# Holds one connection pool, so keep it open rather than building one per turn.
async with AsyncKhwan(api_key="kwk_live_xxx", core="acme", user_id="Web") as kw:
    turn   = await kw.prepare("what did we decide about billing?")
    answer = await your_model(turn.messages)
    await kw.record(turn, answer)

    await kw.record(turn, answer, background=True)   # → {"queued": True}
```

`background=True` schedules the write and returns immediately; `aclose()` — which
`async with` calls for you — waits for anything still in flight, so a fire-and-
forget record is not lost when the process ends.

## What the brain already knew

`prepare` returns the raw turns it retrieved *and* the rules synthesis has
distilled from many past turns. Both are already inside `turn.messages`; they are
also exposed so a caller building its own context — a recall tool, a subagent
brief — can take the distilled rules without replaying the whole prompt.

```python
turn.lessons   # ["Answer in Thai.", …]  standing rules
turn.sources   # the raw turns retrieved for THIS turn, each with a similarity
```

Retrieval applies a relevance floor, so an empty `sources` is an answer: the brain
has nothing close to this question. Read it as "not known here" rather than
reaching for whichever memory was nearest.

## Gate the answer, review what it learned

```python
v = kw.verify(turn, draft)          # score a draft BEFORE you ship it
if not v["ok"]:
    ...                             # regenerate, or route to a human

for l in kw.lessons():              # the standing rules it distilled
    print(l["response_text"], "←", l["source_link"])
kw.delete_lesson(bad_id)            # the only negative signal in the system
```

## Own the learning step

`prepare → your model → record` covers answering. The same shape covers learning —
Khwan clusters the turns, **your** model writes the rule, so no packet text reaches
a provider Khwan chose:

```python
kw.synthesize(distill=lambda system, prompt: my_llm(system, prompt))
```

`turn.messages` is a standard `[{role, content}]` array with Khwan's value baked
into the system prompt (learned lessons + constitution + retrieved memory + coherence).
`your_model` is just your normal LLM call:

```python
import anthropic
client = anthropic.Anthropic(api_key="sk-ant-...")  # your key, Khwan never sees it

def your_model(messages):
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    chat   = [m for m in messages if m["role"] != "system"]
    r = client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               system=system, messages=chat)
    return r.content[0].text
```

## Isolated cores
One account can hold many **cores** — fully separate brains, each with its own
memory, identity, and learning. Point a client at one with `core`:

```python
test    = Khwan(api_key="kwk_live_xxx", user_id="alice", core="test")
client1 = Khwan(api_key="kwk_live_xxx", user_id="alice", core="client1")

kw.cores()   # list the account's cores (the default core is included)
```

`test` and `client1` never share memory. Omit `core` for the account's default brain.

## Two ways to authenticate

An **API key** is a long-lived account secret. It is the right credential when
the process belongs to you — a script, a job, a server you run:

```python
Khwan(api_key="kwk_live_…")          # sent as X-API-Key
```

A **bearer token** is an OAuth access token minted for one end user, short-lived
and scoped to a resource. It is the right credential when you are acting on
someone's behalf and should never hold their key — a remote MCP server, or any
service where the caller authenticated with Khwan rather than with you:

```python
Khwan(bearer_token=access_token)     # sent as Authorization: Bearer
```

Pass exactly one. They are not interchangeable at the wire: a token placed in
`api_key` is looked up as an API key, misses, and 401s — it never reaches the
bearer path, and the error does not say why.

`core` and `user_id` work the same with either.

## On-prem
Same code, point at your instance:
```python
kw = Khwan(api_key="kwk_...", user_id="alice",
               base_url="https://khwan.internal.acme.com")
```

`memory=`/`embedder=` are server-managed and rejected here — they exist only in the
on-prem engine, shipped under license.
