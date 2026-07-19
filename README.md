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

## On-prem
Same code, point at your instance:
```python
kw = Khwan(api_key="kwk_...", user_id="alice",
               base_url="https://khwan.internal.acme.com")
```

`memory=`/`embedder=` are server-managed and rejected here — they exist only in the
on-prem engine, shipped under license.
