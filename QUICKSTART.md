# MAS Recon: the 30-minute walkthrough

For anyone running this call with a prospect — no engineering background needed. This is a script, not a technical manual.

## Before the call

Two questions to ask when scheduling (so you know which path you're on):

1. **"What do you build your agents with?"** — LangChain/LangGraph (including deepagents — it's a LangGraph app underneath), AWS Strands (including on Bedrock AgentCore), the Claude Agent SDK, CrewAI, AutoGen/AG2, or something else (custom)?
2. **"Do you already have any tracing/observability on your agents?"** — Langfuse, LangSmith, Datadog, an OpenTelemetry collector, or none?

## Path A: they already have traces somewhere

Nothing to install. Ask them to export or share one trace file from a recent run (most of these tools have an "export" or "download trace" button). Skip to **Step 3** below.

## Path B: no tracing yet, and they use LangChain/LangGraph, AWS Strands, the Claude Agent SDK, CrewAI, or AutoGen

This is the common case. On the call, screen-share and walk through:

### Step 1 — Install (30 seconds)

```bash
pip install "mas-recon[langchain]"   # for LangChain / LangGraph
pip install "mas-recon[strands]"     # for AWS Strands Agents
pip install "mas-recon[claude]"      # for the Claude Agent SDK
pip install "mas-recon[crewai]"      # for CrewAI
pip install "mas-recon[autogen]"     # for AutoGen / AG2
```

### Step 2 — Add two lines to their existing code

They don't write anything new — they wrap the code they already have:

```python
from mas_recon import start_capture

handle = start_capture()          # <-- add this before their agent runs

# ... their existing agent code, completely unchanged ...

handle.stop()                     # <-- add this after it finishes
```

That's it. No OpenTelemetry knowledge needed, no reading this package's source. `start_capture()` detects which framework is installed and wires up the right tracing automatically. If it doesn't recognize what's installed, it says so plainly with a clear next step, rather than handing back an empty report.

Five working, runnable examples are in `examples/`:
- `examples/strands_example.py`
- `examples/claude_agent_sdk_example.py`
- `examples/langgraph_example.py`
- `examples/crewai_example.py`
- `examples/autogen_example.py`

If the prospect wants to see it work before touching their own code, run one of these first.

### Step 3 — Run it

Have them run their normal agent workflow once — a real task, their own tools, their own credentials. Nothing to fake, nothing we need from them except that it runs.

### Step 4 — Read the report

`handle.stop()` already printed a summary and wrote a full markdown report (or, for Path A, hand them the file and run):

```bash
mas-recon <path-to-trace-file>
```

Open the `.md` file it writes next to the trace (or read the terminal summary live on the call). It shows, from the actual run:

- The real agent graph — who called whom, confirmed by execution
- Per-agent model and token usage
- Any **silent failures** — retries or errors the app's own code handled quietly, that never show up as an error anywhere else, which is usually the moment that lands in the conversation

## Path C: they build directly on Claude Code / Claude Agent SDK with no other framework

If there's no LangChain/Strands/CrewAI/AutoGen underneath at all — it's Claude Code itself, or raw Claude Code sessions driving the work — skip the `pip install` and `start_capture()` entirely. Instead:

1. Copy `claude-code-plugin/` into their Claude Code plugins directory (or set `CLAUDE_CODE_PLUGIN_DIRS` to point at it). No code change, no Python package.
2. Have them run their normal Claude Code session once.
3. It writes `.mas-recon/trace-<session-id>.jsonl` on its own. Run `mas-recon` on that file exactly as in Step 4 below — same report, built from hook events instead of OpenTelemetry spans.

## What to say if something goes wrong

- **"It says it doesn't recognize my framework"** — anything outside LangChain/LangGraph (deepagents included), AWS Strands (AgentCore-deployed included), the Claude Agent SDK, CrewAI, and AutoGen/AG2 isn't auto-instrumented yet. A custom, hand-rolled agent builder is the one case this can't see coming — take down what they use; that's real signal for what to build next, not a dead end. Offer Path A instead (point us at any trace export they can produce another way).
- **"The report says 0 agents found"** — ask what the agent actually did during the run. A single LLM call with no sub-agent or tool delegation is a valid result, not a bug — some apps really are that simple under the hood.
- **Anything deeper** — that's an engineering follow-up, not something to debug live on the call.
