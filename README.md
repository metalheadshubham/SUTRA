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

## the actual problem nobody talks about

every local LLM wrapper does the same thing.

you type a question. it calls `ollama.generate()`. it prints the response. someone puts it on github and calls it an "AI assistant framework."

that's not a framework. that's a for loop with branding.

the real problem is this: language models are text predictors. they are very good at predicting what a correct answer *looks like*. they are not, by default, good at *being correct*. these are different things. most people don't notice the difference until the model confidently writes a bug, explains it eloquently, and you spend 40 minutes debugging code that was wrong from line one.

SUTRA is an attempt to fix that. not by using a bigger model. by being smarter about how you use the ones you have.

---

## part one — ARC (the new thing)

### what agents actually are, and why they keep failing

an agent is just a loop.
```
while task_not_done:
    decide what to do next
    do it
    observe result
    repeat
```

that's it. every agent framework — LangChain, AutoGPT, Claude Code, all of them — is this loop with different decorations. the loop itself is not the hard part. the hard part is step one: *decide what to do next.*

most agent implementations hand this decision entirely to the model and say "figure it out." the model, being a text predictor, does what text predictors do — it generates plausible-sounding text. sometimes that text is a valid action. sometimes it's a paragraph explaining why the action is complex. sometimes it's a list_dir call when you asked it to write a file. sometimes it's nothing at all wrapped in confident language.

the loop continues. nothing happens. 47 minutes pass. you close the terminal.

this is not an intelligence problem. it is a constraint problem. you gave the model infinite output space and asked it to produce one specific shape of output. it didn't. shocking.

---

### ARC — Action-Read-Check

ARC is a constrained execution environment. the constraint is the entire point.

inside ARC, the model has exactly two legal outputs:
```
TOOL: <CATEGORY> <action> <key="value" ...>
DONE: <message>
```

that's it. no markdown. no explanation. no "I'll now proceed to write the file." no reasoning out loud. no hedging. one line. either a tool call or a termination signal.

anything else is **invalid and rejected.** not parsed. not interpreted. not forgiven. rejected. the model gets two retries per step. after that, the loop aborts.

this sounds strict because it is. that's the point.

here's the insight: **the model does not decide whether to act. it only decides which action to take.** the harness has already decided that action is happening. the model's only job is to specify which one. you've reduced an open-ended generation problem to a constrained classification problem. the model is much better at the second one.
```
User Task
   ↓
Plan Phase  ← model thinks freely here, ≤6 steps, natural language allowed
   ↓
ARC Loop    ← model is now in a box. one tool call per step. nothing else.
   ↓
[TOOL → RESULT] × N
   ↓
DONE
```

the plan phase exists because you need the model to understand the task before executing it. free reasoning is allowed there. once execution starts, the box closes.

---

### the tool system

tools are grouped into three categories based on what they do to the world:

**ACT** — mutations. these change state.
- `write_file(path, content)`
- `delete_file(path)`
- `replace_in_file(path, old, new)`

**CHECK** — observations. these read state without changing it.
- `read_file(path)`
- `list_dir(path)`
- `file_exists(path)`

**VERIFY** — execution. this runs things and reports what happened.
- `run_command(cmd)` — 30 second timeout. blocks `rm -rf` and similar creative decisions.

every tool returns exactly this:
```
ToolResult(
    success: bool,
    output: str,
    error: Optional[str]
)
```

no exceptions propagate out of tools. failures are data, not crashes. the loop always knows what happened. there is no "the tool ran but who knows what it did" state.

---

### workspace isolation

the agent runs inside a workspace directory you choose. every path every tool touches is validated against it before execution. `../` traversal is blocked. absolute paths outside the workspace are blocked.

if you somehow find a way to make it escape the boundary anyway, that's impressive and also your fault.

---

### why 8 steps

because if you can't complete a task in 8 deterministic steps, the problem isn't the step limit. the problem is the plan.

an agent that runs 40 steps to create a file is not a capable agent. it's a confused one with a high patience tax.

---

### what it actually looks like working
```
task: build a website from an 800-word spec file

/load spec.txt
/agent do what the file says

→ plan generated (6 steps)
→ TOOL: ACT write_file path="index.html" content="..."  ✓
→ TOOL: CHECK file_exists path="index.html"             ✓
→ DONE: created index.html, all constraints satisfied

time: 28.6 seconds
model: qwen2.5-coder:3b, local, offline
```

all constraints from the spec were hit. not approximately. exactly. because the model wasn't allowed to approximate — it was only allowed to act.

---

## part two — council (the original thing)

### the problem with asking one model

when you ask a single model a hard question at a fixed temperature, you get one sample from its probability distribution over answers. that sample might be correct. it might be confidently wrong. the model cannot tell you which one it is, because the mechanism that generates the answer is the same mechanism that would evaluate it. you're asking a witness to judge their own testimony.

council runs the same small model twice — once at temperature 0.3 (conservative, low variance) and once at temperature 0.8 (exploratory, higher variance). these are not the same output. different temperatures sample different regions of the distribution. the conservative run finds the safe answer. the exploratory run finds answers the safe run wouldn't. sometimes the exploratory answer is wrong in an interesting way. sometimes it's right in a way the conservative run wasn't.

a third pass critiques both. not to pick a winner — to find where they diverge and why. divergence is the signal. if both answers agree, one of them is probably right. if they disagree, the disagreement tells you exactly where the uncertainty lives.

then the large model reads everything — answer A, answer B, the critique — and synthesizes a final response that, under Bayesian in-context learning theory, should assign higher probability to tokens that survived both sampling runs. this is contrast decoding (Zhang et al., 2023) implemented in natural language without requiring two models in memory simultaneously. which matters a lot when your memory is 8GB.
```
query
  ↓
small model (temp 0.3)  →  answer A
small model (temp 0.8)  →  answer B
small model (temp 0.3)  →  critique
large model (temp 0.2)  →  synthesis
```

four inference calls. one answer. no cloud.

---

### the numbers

<img width="1622" height="608" alt="Benchmark" src="https://github.com/user-attachments/assets/d41e7fe0-8451-4a56-9614-acb54cd3a2f2" />

> SUTRA uses `qwen2.5-coder:3b` as small and `qwen2.5-coder:7b` as large, running locally via Ollama. cloud model scores from published benchmarks. this is not a parameter-count comparison. SUTRA costs 4 inference calls vs 1 and runs on consumer hardware with no API key, no subscription, and no internet. it sits above GPT-4 original. on 8GB RAM.

### hard problems (HumanEval, 20 problems)

| method | pass@1 | solved |
|--------|--------|--------|
| raw `llama-3.3-70b` | 0.600 | 12/20 |
| **SUTRA council** | **0.800** | **16/20** |
| delta | **+0.200** | **+4 problems** |

**+20% on hard problems.**

### full benchmark (164 problems)

| method | pass@1 | solved | delta |
|--------|--------|--------|-------|
| baseline | 0.805 | 132/164 | — |
| **SUTRA council** | **0.854** | **140/164** | **+8 problems** |

council only ran on the 32 baseline failures. rescued 8. zero regressions — council never runs on problems the baseline already solved, so it is mathematically incapable of making those worse.

### easy problems (same 20-problem test)

| method | pass@1 | solved |
|--------|--------|--------|
| raw `llama-3.3-70b` | 0.900 | 18/20 |
| SUTRA council | 0.800 | 16/20 |
| delta | -0.100 | -2 problems |

council makes easy problems slightly worse. the critique adds noise when the large model already knew the answer. this is why `/quick` exists. knowing when not to use your own system is also a feature.

---

## is it fast

no.

council on CPU: 5-10 minutes. on GPU: 1-2 minutes. on Groq's free API: ~30 seconds.  
agent mode: depends. the website took 28.6 seconds.  
quick mode: as fast as your local model. which is not fast. but it costs nothing, so the exchange rate is reasonable.

the slowness is a hardware constraint. a 3B model on 8GB RAM was not designed to be fast. it was designed to fit in 8GB RAM. these are different optimization targets and only one of them is achievable here.

---

## install
```bash
pip install sutra-llm
```

you need [Ollama](https://ollama.com) running locally:
```bash
ollama pull qwen2.5-coder:3b
ollama pull qwen2.5-coder:7b
python -m agent_handoff
```

Python 3.10+. 8GB RAM minimum. Windows, Mac, Linux. Mac is faster due to Apple Silicon. this is not something you can fix on Windows. sorry.

or from source if you enjoy that sort of thing:
```bash
git clone https://github.com/metalheadshubham/SUTRA
cd SUTRA
pip install -r requirements.txt
python -m agent_handoff
```

---

## models

default:
- **small** (council stages 1–3 + agent execution): `qwen2.5-coder:3b` (1.8GB)
- **large** (council synthesis): `qwen2.5-coder:7b` (4.4GB)

if you have more RAM:
- `deepseek-r1:8b` as large — actual chain-of-thought reasoning built in, noticeably better on hard problems, noticeably slower on CPU. pick your suffering.
- `llama3.1:8b` — good general alternative

the CLI detects your pulled models and recommends roles by parameter count. it will not recommend a 3B model as large. it has some standards.

---

## commands
```
/quick <query>      large model, no council, no waiting, no guarantees
/council <query>    full pipeline. use this when correctness matters more than time
/agent <task>       ARC execution mode. use this when you want files to actually exist

/load <file>        attach file to context for the next query
/unload             detach loaded files
/save               save last output somewhere useful
/project            what's loaded, what's been saved this session
/logs               last 5 runs with timing
/plugins            loaded plugins
/install <url>      install a plugin from a .py URL
/help               all of the above, formatted
```

you choose the mode manually. there is no automatic routing. the system does not presume to know what you need. you are capable of reading three command names and picking one.

---

## loading files into context
```bash
❯ /load spec.txt
  ✓ Loaded spec.txt (~800 tokens) · attached to context

❯ /agent do what the file says
```

works with `/council` and `/quick` too. the file gets injected into the prompt before your query runs. if the file exceeds ~800 tokens it warns you to use a larger model. it won't stop you. it just wants you to know.

multiline input: open with `"""`, paste whatever, close with `"""`. useful for code you want fixed and are too tired to explain.

---

## plugins

`.py` file in `~/.sutra/plugins/`. loaded on startup. available hooks:
```python
def pre_query(query): ...
def post_answer_a(text): ...
def post_answer_b(text): ...
def post_critique(text): ...
def post_synthesis(text): ...
```

built-in logger writes every run to `~/.sutra/logs/YYYY-MM-DD.jsonl`. because if something goes wrong at 2am you want receipts.

---

## project structure
```
agent_handoff/
├── cli.py          # 61KB. the terminal UI. it got away from us.
├── agent.py        # ARC loop — the whole thing described above
├── handoff.py      # CouncilHandoff — the reasoning pipeline
├── tools.py        # the 7 tools the agent is allowed to use
├── templates.py    # prompts for all council stages
├── protocol.py     # HandoffPacket, CouncilResult dataclasses
├── parser.py       # structured output parser with fallback
├── cache.py        # SHA-256 keyed cache with TTL
├── benchmark.py    # council vs baseline comparison runner
└── utils.py        # helpers that don't belong anywhere else
tests/              # 96 tests. all pass.
```

---

## honest limitations

- command blocking in `run_command` is pattern-based. it catches the obvious things. it does not catch someone who is trying. this is not a sandbox.
- no parallel tool execution. one step at a time. sequential by design, slow by consequence.
- no memory between sessions. every conversation starts from zero. the model does not remember you. you could find this freeing.
- no auto mode-selection. you pick the mode. this will eventually be a router. for now it's three commands and your judgment.
- cannot match GPT-4o or Sonnet 3.5 on raw quality. a 7B parameter model and a 200B parameter model are not the same thing. no amount of orchestration changes parameter count. what it changes is how intelligently you use what you have.

---

## roadmap

- difficulty router — query comes in, system picks mode, you don't have to think about it
- MCP server — expose council as an MCP tool so Claude/Cursor can call SUTRA as a backend. local reasoning as a service.
- GSM8K benchmark — math and reasoning, not just code
- ablation study — isolate the critique step's contribution vs just calling the model twice
- full sandboxing for `run_command`
- memory that survives sessions

---

## license

MIT. use it however. credit where it's due.

---

*shit works.*
