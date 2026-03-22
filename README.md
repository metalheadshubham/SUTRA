# S.U.T.R.A

```
 ███████╗ ██╗   ██╗ ████████╗ ██████╗   █████╗
 ██╔════╝ ██║   ██║ ╚══██╔══╝ ██╔══██╗ ██╔══██╗
 ███████╗ ██║   ██║    ██║    ██████╔╝ ███████║
 ╚════██║ ██║   ██║    ██║    ██╔══██╗ ██╔══██║
 ███████║ ╚██████╔╝    ██║    ██║  ██║ ██║  ██║
 ╚══════╝  ╚═════╝     ╚═╝    ╚═╝  ╚═╝ ╚═╝  ╚═╝
```

*Structured Universal Transfer via Retrieval Adaptation*
*by Shubham Kumar*

---

ok so idk how to explain this properly but basically i got tired of paying for API calls and also tired of my laptop dying trying to run a 70B model so i built this thing.

the idea came from a weird place — i was reading about how LLMs do in-context learning and thought, what if instead of trying to make one model smarter, you just... make multiple small models argue with each other and then have a slightly bigger one read the argument and write the answer. turns out that actually works. not always. but enough.

---

## what even is this

it's a pipeline. four stages. runs entirely locally on ollama. never puts two models in RAM at the same time because idk if you noticed but 8GB doesn't go very far.

```
your query
  ↓
small model (temp 0.3)  →  answer A
small model (temp 0.8)  →  answer B   ← same model, different temperature = different answer
small model (temp 0.3)  →  critiques both
large model (temp 0.2)  →  reads everything, writes final answer
```

the temperature thing is important btw. running the same model twice at different temps gives you genuinely different answers — one more conservative, one more exploratory. the critique then finds where they disagree or where both went wrong. the large model then synthesizes something that (hopefully) avoids both failure modes.

is this just fancy prompting? kind of. but it's fancy prompting that i actually benchmarked and the numbers came out real so.

---

## does it work

yeah. i ran it on HumanEval (164 python coding problems, standard benchmark everyone uses).

baseline — just asking `llama-3.3-70b` directly — solved 132/164.

SUTRA — same 70B model but guided by two `llama-3.1-8b` answers + critique — solved 140/164.

that's 8 problems rescued that the 70B couldn't do alone. 25% rescue rate on its own failures. zero regressions (council only ran on problems the baseline failed, so it literally cannot make passing problems fail).

is +4.9% pass@1 huge? not in absolute terms no. but it's real and reproducible and it costs nothing extra in terms of model size.

---

## is it fast

lmaooo no. on CPU with 8GB RAM you're looking at 5-10 minutes per query in council mode. on a machine with a GPU (even a mid-range one) it drops to like 1-2 minutes. on Groq's free API it runs in about 30 seconds total.

i'm aware this is a problem. the architecture is sound though — the slowness is a hardware constraint not a design flaw. `/quick` mode skips the council and just asks the large model directly if you need a fast answer.

---

## install

you need [Ollama](https://ollama.com) running. that's it.

```bash
git clone https://github.com/metalheadshubham/SUTRA
cd SUTRA
pip install -r requirements.txt
ollama pull qwen2.5-coder:3b
ollama pull qwen2.5-coder:7b
python -m agent_handoff
```

Python 3.10+. 8GB RAM minimum. works on Windows, Mac, Linux (Mac is actually faster bc Apple Silicon).

---

## models

default setup that i benchmarked:
- **small model** (stages 1, 2, 3): `qwen2.5-coder:3b`
- **large model** (synthesis): `qwen2.5-coder:7b`

if you have more RAM or a GPU you can use:
- `deepseek-r1:8b` as large model — it does actual chain-of-thought reasoning, significantly better on hard problems but very slow on CPU
- `llama3.1:8b` as large model — good general purpose alternative

the CLI auto-detects your Ollama models and recommends roles based on parameter count.

---

## commands

```
/quick <query>      skip council, ask large model directly (fast)
/council <query>    force council mode explicitly
/save               save last output — asks you where
/load <file>        attach a file to context for next query
/unload             remove attached files
/project            see files saved + loaded this session
/logs               last 5 runs with timing
/plugins            loaded plugins + hooks
/install <url>      install a plugin from a .py URL
/help               all commands
```

multiline input — type `"""` to open a block, paste whatever, `"""` to send. useful for pasting code you want fixed.

---

## workspace

this is the part i'm actually proud of tbh. you can load files into context and the council will read them without you copy-pasting anything.

```bash
# generate some code
❯ implement a FastAPI router for user auth

# save it
❯ /save
  Save to (path or filename): src/auth.py
  ✓ Saved to C:\project\src\auth.py · registered in workspace

# later, load it back
❯ /load auth.py
  ✓ Loaded auth.py (~340 tokens) · attached to context

# now your query has the file as context automatically
❯ add rate limiting to this
```

if the file is large (>800 tokens for the small model) it warns you to switch to a bigger model before continuing.

---

## plugins

drop a `.py` file into `~/.sutra/plugins/`. it gets loaded on startup. hooks available:

```python
def pre_query(query): ...
def post_answer_a(text): ...
def post_answer_b(text): ...
def post_critique(text): ...
def post_synthesis(text): ...
```

built-in logger plugin saves every run to `~/.sutra/logs/YYYY-MM-DD.jsonl` automatically.

---

## roadmap

- difficulty router — auto-detect when to use council vs quick (currently manual)
- MCP server — expose council pipeline as an MCP tool so Claude/Cursor can call it
- math + reasoning benchmarks — HumanEval is coding only, want to test GSM8K
- ablation study — prove the critique step is actually doing something (vs just calling twice)

---

## project structure

```
agent_handoff/
├── cli.py          # the whole terminal UI — where you spend your time
├── handoff.py      # AgentHandoff (old) + CouncilHandoff (new)
├── templates.py    # prompts for each stage
├── protocol.py     # dataclasses — HandoffPacket, CouncilResult etc
├── parser.py       # extracts structured output from model responses
├── cache.py        # SHA-256 keyed cache with TTL
└── utils.py        # helpers
tests/              # 56 tests, all pass
```

---

## license

MIT. do whatever.

---

*built weird, works anyway*
