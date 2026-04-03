# S.U.T.R.A
```
 ███████╗ ██╗   ██╗ ████████╗ ██████╗   █████╗
 ██╔════╝ ██║   ██║ ╚══██╔══╝ ██╔══██╗ ██╔══██╗
 ███████╗ ██║   ██║    ██║    ██████╔╝ ███████║
 ╚════██║ ██║   ██║    ██║    ██╔══██╗ ██╔══██║
 ███████║ ╚██████╔╝    ██║    ██║  ██║ ██║  ██║
 ╚══════╝  ╚═════╝     ╚═╝    ╚═╝  ╚═╝ ╚═╝  ╚═╝
```

**Structured Universal Transfer via Retrieval Adaptation**
*by Shubham Kumar*

> *local model orchestration that beats GPT-4. costs $0. runs on your laptop. yes, really. stop making that face.*

---

## ok what even is this

SUTRA is a framework that makes small, offline, **completely free** language models actually good by being clever about *how* you use them instead of just buying a bigger model like everyone else's solution to every problem ever.

two things it does:

- **Council** — runs the same model multiple times, makes it debate itself, synthesizes a final answer that's genuinely better. like a peer review process, but one that finishes before the heat death of the universe and doesn't come back asking you to restructure the entire paper three days before the deadline.

- **ARC** (Action-Read-Check) — an agent loop where the model cannot philosophize, hedge, apologize, or do literally anything except take an action. no "i'll now proceed to." no vibes. only function calls. the model does the thing or it fails trying.

runs **offline** on **your machine** using **Ollama** (free). no API key. no $20/month. no "you've hit 80% of your context window limit" notification at the worst possible moment.

---

## the problem nobody wants to say out loud

open github. search "AI assistant framework." i'll wait.

every single one of those repos is doing: user asks question → `model.generate()` → print answer → call it "production-ready AI infrastructure." that's a for-loop wearing a blazer. that's automation cosplaying as intelligence. some of these have 4,000 stars.

the deeper problem: LLMs (Large Language Models — the thing everyone's manager is convinced will replace the entire team by Q3) are, at a fundamental level, **extremely expensive autocomplete.** not in the dismissive "it's just autocomplete" sense, in the *technically precise* sense. they learn P(next token | all previous tokens) over enormous datasets and sample from it at inference time. they're superhuman at generating text that *looks like* a correct answer. correctness is a separate question they don't get asked.

"looks correct" ≠ "is correct."

your compiler knows this. your tests know this. the model? the model is operating in a world where producing a confident-sounding token sequence IS the task, and correctness is evaluated by a vibes check. it will write a bug, explain the bug eloquently, defend the bug under questioning, and generate a Stack Overflow answer containing the bug as the accepted solution. all with the confidence of someone who read the entire internet and has formed Strong Opinions.

SUTRA is the "actually check your work" infrastructure.

---

## PART ONE — COUNCIL
### *or: what if we made the model argue with itself and then hired a judge*

---

### the witness problem

imagine a trial where the witness is also the jury, also the judge, and also the defendant. the justice system has feelings about this. the justice system is correct.

that's what happens when you ask an LLM a hard question once and ship the answer. the mechanism that generated the response is the same mechanism that would evaluate whether it's correct. there's no second opinion. no "wait, actually." just one sample from a probability distribution, dressed in confident language, sent directly to your users.

one sample is not enough. this isn't a SUTRA opinion. this is the first year of any statistics curriculum.

---

### how council works

**the ELI5 version:** ask the same question twice with different "creativity" settings. make a third call that criticizes both answers. make a fourth call that synthesizes everything into a final response. four calls, one answer that's actually good.

**the "i know what temperature does" version:** the two sampling runs use T=0.3 and T=0.8 respectively. temperature rescales logits as z_i/T before softmax — low T concentrates mass at the mode (safe, low-variance, picks the most probable path), higher T samples from further down the distribution (exploratory, occasionally finds answers the conservative run missed). these are genuinely different outputs even from the same model, because you're sampling different regions of the token probability distribution.

the critique pass performs implicit posterior inference over which claims in {answer A, answer B} are robust across temperature regimes. claims appearing in both runs have higher implicit correctness probability. claims unique to one run need to justify themselves against active scrutiny. the synthesis step is functionally analogous to contrast decoding (Zhang et al., 2023), implemented in-context via natural language instead of at the logit level. you're getting ensemble-style variance reduction without needing two models in VRAM simultaneously.

on 8GB RAM, "without needing two models simultaneously" is not a technical nicety. it's the difference between running and not running.

is it exactly Bayesian inference? no. is it *directionally* justified by Bayesian in-context learning theory? yes. is it empirically better than asking once and hoping? also yes. the benchmarks are below and they're not subtle.

---

### the benchmarks (where we stop being humble)

> this is the chart that started arguments. SUTRA's 3B+7B local setup, benchmarked against cloud models. look at where it lands.

![SUTRA Council vs Cloud Models Benchmark](https://github.com/user-attachments/assets/d41e7fe0-8451-4a56-9614-acb54cd3a2f2)

sitting above GPT-4 original. on a laptop. offline. with no internet connection and no API bill. if your reaction is "that can't be right" — we felt the same way. then we ran the benchmark again. and again. the numbers held.

---

**Hard Problems — HumanEval subset, 20 problems:**

| method | pass@1 | solved |
|---|---|---|
| raw `llama-3.3-70b` (baseline) | 0.600 | 12/20 |
| **SUTRA Council** | **0.800** | **16/20** |
| delta | **+0.200** | **+4 problems** |

a 3B+7B local setup beat a 70B model by 20 percentage points on hard coding problems. SUTRA said "it's not the size, it's how you use it" and then proved it in a benchmark, which is a sentence i never expected to type and yet here we are.

---

**Full Benchmark — 164 problems, council applied to failures only:**

| method | pass@1 | solved | delta |
|---|---|---|---|
| baseline | 0.805 | 132/164 | — |
| **SUTRA Council** | **0.854** | **140/164** | **+8 problems** |

important note that makes this better than it already looks: Council only ran on the 32 problems the baseline *already failed on.* rescued 8 of them. and because it never ran on the 132 the baseline already solved, making those worse is **mathematically impossible.** not "very unlikely." *impossible.* set theory is doing free consulting work for us here and we appreciate it.

---

**Easy Problems — the honest section, because we have integrity:**

| method | pass@1 | solved |
|---|---|---|
| raw `llama-3.3-70b` | 0.900 | 18/20 |
| SUTRA Council | 0.800 | 16/20 |

Council makes easy problems slightly *worse.* the critique pipeline introduces noise when the model already knew the answer. this is exactly what you'd predict from any ensemble method on low-uncertainty inputs — variance reduction overhead on problems that had no variance to reduce.

this is why `/quick` mode exists. knowing when to not use your own system is a feature. we built the feature. we're proud of it.

*(stat disclaimer: n=20 is not large enough for high-confidence claims. we know. the 164-problem benchmark is more defensible. GSM8K is next. treat the 20-problem results as directionally interesting, not conclusive. we will not gaslight you about statistical significance. we've all read enough papers that do.)*

---

## PART TWO — ARC
### *Action-Read-Check. or: how to stop an AI agent from spending 47 minutes doing absolutely nothing*

---

### the dirty secret of every agent framework

every "agentic AI" product — LangChain agents, AutoGPT, Claude Code, whatever got $50M in funding last week — is, at its core, a while-loop. observe, plan, act, repeat. the while-loop is not the hard part. a CS101 student can write that loop. the hard part is the *planning step* when the model has infinite output space to do the planning in.

what happens when you give an LLM unlimited output space and tell it to "figure out the next step"?

- sometimes it does the step. great.
- sometimes it writes 3 paragraphs about why the step is complex. less great.
- sometimes it lists things it *could* do without committing to any of them. concerning.
- sometimes it just apologizes. for what? unclear.
- sometimes it does a thing *adjacent to* the step but not the step itself. baffling.
- sometimes, in a move that i genuinely respect on a philosophical level, it produces absolutely nothing useful wrapped in extremely confident language.

47 minutes pass. the terminal is full of words. nothing has been created. the loop is technically still running. you close it. you stare at the wall. you consider a career in something with more predictable outputs, like agriculture.

this isn't an intelligence failure. it's a **constraint failure.** you gave a token generation system an open output space and expected structured behavior. you got token generation. this was predictable. everyone who built agent frameworks has been quietly experiencing this and nobody talks about it in the launch posts.

---

### ARC: take away the options

inside ARC, the model has exactly two legal outputs. a tool call in a specific structured format. or a done signal. that's the complete menu. everything else — every explanation, every hedge, every "i'll now proceed to," every markdown header, every thought process rendered in natural language — is **rejected.** not interpreted charitably. not coaxed into compliance. *rejected.* two retries per step, then the loop aborts.

the insight: **the model no longer decides whether to act. it decides which action.** the harness already decided action is happening. the model's job is specifying which one with what arguments. you've gone from "generate any token sequence" (unbounded, hard for reliability) to "choose one of N templates and fill in the slots" (constrained, much easier for reliability).

this is why every structured output paper since 2022 exists. constrained generation is more reliable than open generation when correctness matters. ARC is the application-layer version of that insight — no custom sampling, no logit manipulation, just: wrong format → rejected → retry → abort.

for the researchers: this is chain-of-thought with the scratchpad phase explicitly gated from the execution phase. plan = scratchpad (free generation). ARC loop = structured execution under hard syntactic constraints. you could describe it as constrained decoding via rejection sampling at the harness layer. the tradeoff is generation flexibility for execution reliability. we chose reliability.

---

### two phases, because structure is a love language

**Phase 1 — Plan:** the model thinks freely. natural language, real reasoning, maximum 6 steps. this is the architecture phase. you design the building before construction. you do not hand the foreman a hammer and say "make something load-bearing and figure it out."

*(why 6? because if your plan needs more than 6 steps, your task needs to be decomposed before the agent sees it. an agent running 40 steps to create a text file is not "capable." it's confused, and you've been charged for the confusion.)*

**Phase 2 — ARC Execution:** the box closes. reasoning is illegal. one tool call per step. every step returns a structured result: success or failure, always captured, always readable. no ambiguous states. no "the tool ran and who knows." the agent always knows exactly what happened. always.

---

### the tools (7. not 700. seven.)

**ACT** — things that change the world. write file, delete file, replace text. mutations with side effects. handled with appropriate gravity.

**CHECK** — things that observe the world without disturbing it. read file, list directory, check if file exists. reconnaissance before action.

**VERIFY** — things that run the world. execute a shell command with a 30-second timeout and a blocklist preventing the model from achieving spiritual liberation via destroying your filesystem. it is not a real sandbox. it is a strongly-worded suggestion backed by pattern matching. this is documented because you deserve to know.

every tool returns the same structure: did it work, what was the output, what failed if anything. uniform interface. no surprises. the agent always knows what the world looks like after every step.

---

### workspace isolation

the agent lives in a directory you specify. every path any tool tries to touch is validated against that boundary before execution. `../` traversal: blocked. absolute paths outside the workspace: blocked. you should still not run this pointed at your production database. this is less a technical limitation and more general life advice.

---

### does it work

800-word spec file → full website → 28.6 seconds. offline. `qwen2.5-coder:3b`. all spec constraints satisfied. not approximately — *exactly.* because the model couldn't approximate. it was only allowed to act.

spec went in. file came out. everything in between was structured, auditable, deterministic. which should not be a remarkable achievement in 2025 and yet here we are, writing it up.

---

## IS IT FAST

no.

Council on CPU: 5-10 minutes. with a GPU: 1-2 minutes. Groq's free tier: ~30 seconds.

ARC agent mode: 28.6 seconds for the website. task-dependent.

quick mode: as fast as your local model. which isn't fast. but it's free, offline, and your data never hits anyone's servers. if you're debugging something at 2am that you don't want in a training dataset, that's not nothing. that's actually quite a lot.

the slowness is hardware, not SUTRA. a 3B model on 8GB RAM was optimized to *exist* in 8GB RAM, not to be fast. mac users on Apple Silicon get better throughput because unified memory architecture lets the GPU and CPU share the same pool, eliminating the PCIe bottleneck that makes CPU inference on Windows feel like trying to pour a lake through a garden hose. this is physics. it cannot be configured away.

---

## MODELS

**defaults:**
- **small** (Council stages 1-3, all agent execution): `qwen2.5-coder:3b` — 1.8GB, fits comfortably, genuinely good at code
- **large** (Council synthesis only): `qwen2.5-coder:7b` — 4.4GB, meaningfully better reasoning, worth the disk space

**if you have more RAM:**
- `deepseek-r1:8b` as large: has chain-of-thought reasoning baked into training rather than bolted on top. noticeably better on hard problems. noticeably slower on CPU. this is the "i know what i'm signing up for" option.
- `llama3.1:8b`: solid general-purpose alternative if you're not primarily doing code.

the CLI detects your local models and recommends roles by parameter count. it will not recommend a 3B model as your synthesis model. it has standards. more than some hiring pipelines.

---

## COMMANDS

you pick the mode. manual selection. no auto-router (yet). this is a feature: knowing which tool to reach for is a skill you have. the router is on the roadmap for when we've built enough confidence in what the heuristics should be.

**`/quick <query>`** — large model, no council, immediate answer. for questions you already know are easy. "what's the syntax for X." "how does Y work." adding Council to a question the model already knows is like forming a committee to decide what to have for lunch. wasteful. don't.

**`/council <query>`** — full four-call pipeline. correctness maximized. time spent. use when being right matters more than being fast. budget 5-10 minutes on CPU.

**`/agent <task>`** — ARC mode. use when you want something to *exist* that didn't exist before. the agent will plan (6 steps), execute (8 max), verify, and report. what goes in is a description. what comes out is a done task or a clear failure, never ambiguity.

**`/load <file>`** — attaches a file to context for the next query. contents injected into the prompt before execution. you can point the agent at a spec file and say "do what this says." it does what the file says. multiline input via triple-quote syntax for code you want debugged and lack the energy to explain.

**`/save`** — saves last output. does what it says, unlike most things.

**`/logs`** — last 5 runs with timing. every run is written to a dated JSONL file in your home directory. JSONL because it's line-parseable and survives mid-write crashes, which is not a hypothetical. we have been burned.

**`/plugins`** — shows what's loaded.

**`/install <url>`** — install a plugin from a `.py` URL.

---

## PLUGINS

drop a `.py` file in `~/.sutra/plugins/`. loads on startup. hooks at every pipeline stage: before query, after answer A, after answer B, after critique, after synthesis. log, transform, route, alert, do whatever you need.

the hook system exists because we're not arrogant enough to think we've covered every use case. we've covered ours.

---

## HONEST LIMITATIONS
### *(the section most READMEs skip because it requires self-awareness)*

**the command blocklist is not a sandbox.** it stops obvious bad things. it does not stop someone who's *trying.* adversarial inputs from adversarial environments need the container sandboxing on the roadmap. you've been told. act accordingly.

**no parallel execution.** sequential by design. parallelism needs locking semantics, conflict resolution, rollback. those are harder than they look. we did the simple thing correctly. parallel is roadmapped.

**no inter-session memory.** every conversation starts fresh. the model doesn't know you. persistent memory via RAG (Retrieval-Augmented Generation — searching your past conversations and injecting relevant context into the prompt so the model isn't starting blind) is coming. today, bring your context explicitly.

**no auto mode-selection.** `/quick`, `/council`, `/agent`: you pick. the difficulty router is on the roadmap. for now, read three words, make a decision. you're capable.

**can't match GPT-4o or Claude Sonnet 3.5 on raw quality.** 7B vs 175B+ is a real difference. parameter count correlates with capability on hard reasoning tasks. orchestration changes how efficiently you use what you have, not what you have. the deal: beats GPT-4 original on the benchmarks above, runs offline for free, keeps your data local. take it or leave it. we think it's a good deal.

---

## ROADMAP

**difficulty router** — system reads the query, picks the mode, you don't have to. will be wrong sometimes. "sometimes wrong but automatic" beats "always manual." coming.

**MCP server** — expose Council as an MCP (Model Context Protocol) tool. external AI clients call SUTRA as a local reasoning backend. the cloud calls your machine, your machine thinks, the cloud gets the answer, nobody's API bill increases. we find this funny and are building it entirely for that reason.

**GSM8K benchmark** — math and general reasoning, not just code. we want to know if Council generalizes across task types. "we believe it does but haven't proven it" is the correct epistemic state. we're in it. we're running the experiment.

**ablation study** — isolate the critique step's contribution vs just running twice. we believe the critique matters. "believe" is doing work here. the controlled experiment will settle it one way or another.

**full sandboxing** — container-based. the blocklist is not adequate for adversarial environments. roadmapped.

**persistent memory** — RAG over your own conversation history. sessions that remember sessions. high value, technically tractable, actively being worked on.

---

## INSTALL

two requirements: SUTRA (one pip install) and Ollama (the free local model runner — a well-behaved background process that manages your AI models and, critically, does not mine crypto or send telemetry about your queries to anyone).

pull `qwen2.5-coder:3b` and `qwen2.5-coder:7b` via Ollama. run the entry point. the TUI (Terminal User Interface — like a GUI but for people who've made peace with the terminal) appears. Python 3.10+. 8GB RAM minimum. Windows, Mac, Linux supported.

source install available for people who enjoy the process of cloning repos. GitHub: `metalheadshubham/SUTRA`. requirements file is honest about dependencies, which is a bar more things should clear.

---

## PROJECT STRUCTURE

nine files. not because we're minimalists, because nine files is what it took and not one more. the CLI is 61KB, which got away from us and we've made peace with it. everything else is modular, named for what it does, and documented well enough that you can read it without sending an angry issue. 96 tests. all pass. the number of times they didn't pass before they did is personal information.

---

## LICENSE

MIT. use it, build on it, deploy it, credit where credit's due. don't be weird about it.

---

*shit works.*

---

*SUTRA: because "just use GPT-4" is not an architecture, it's a coping strategy.*
