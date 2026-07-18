# khwan (Python client)

The Khwan hosted client — a thin HTTP wrapper with **no engine code**. Khwan
is a cognition layer (memory + constitutional identity + coherence + learning) that
runs on our server; you bring your own model.

```bash
pip install khwan
```

## BYOM — you call your own model
```python
from khwan import Khwan

fc = Khwan(api_key="kwk_live_xxx", user_id="alice")

turn   = fc.prepare("remember I prefer short answers in Thai")  # FC builds context, no LLM
answer = my_own_llm(turn.messages)                              # YOUR model + key
fc.record(turn, answer)                                          # FC persists + learns
```

`turn.messages` is a standard `[{role, content}]` array with Khwan's value baked
into the system prompt (learned lessons + constitution + retrieved memory + coherence).
`my_own_llm` is just your normal LLM call:

```python
import anthropic
client = anthropic.Anthropic(api_key="sk-ant-...")  # your key, FC never sees it

def my_own_llm(messages):
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    chat   = [m for m in messages if m["role"] != "system"]
    r = client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                               system=system, messages=chat)
    return r.content[0].text
```

## Convenience (server-side generation, if your plan enables it)
```python
reply = fc.chat("hello")
print(reply.text, reply.coherence, reply.sources)
```

## On-prem
Same code, point at your instance:
```python
fc = Khwan(api_key="kwk_...", user_id="alice",
               base_url="https://khwan.internal.acme.com")
```

`memory=`/`embedder=` are server-managed and rejected here — they exist only in the
on-prem engine, shipped under license.
