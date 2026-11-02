# MAS Recon

Point it at a running multi-agent system; get back the real agent topology and a silent-failure report, from an actual run.

```python
from mas_recon import start_capture

handle = start_capture()   # auto-detects your framework, starts tracing
# ... run your agent exactly as you already do ...
handle.stop()               # writes the trace, prints the report
```

Supported today: **AWS Strands Agents** (including agents deployed on **AWS Bedrock AgentCore** — AgentCore is a deployment wrapper, not a framework, so Strands underneath it is already covered), **the Claude Agent SDK** (including subagent delegation), **LangChain / LangGraph** (including **deepagents**, which is a LangGraph app underneath — confirmed by an actual run, not assumed), **CrewAI**, and **AutoGen / AG2**. Anything else — including a custom, hand-rolled agent builder with none of the above underneath — isn't auto-instrumented yet: tell us what you're using and we'll add it, or point us at an existing trace export if you already have one.

See [QUICKSTART.md](QUICKSTART.md) for the walkthrough, or run one of the scripts in `examples/`.

## Install

```bash
pip install "mas-recon[langchain]"   # LangChain / LangGraph
pip install "mas-recon[strands]"     # AWS Strands Agents
pip install "mas-recon[claude]"      # Claude Agent SDK
pip install "mas-recon[crewai]"      # CrewAI
pip install "mas-recon[autogen]"     # AutoGen / AG2
```

## Already using Claude Code directly?

For a Claude Code session with no Python agent framework underneath it at all, there's a second, portable capture method: `claude-code-plugin/`. It's a Claude Code plugin, not a Python import — it registers hooks (`PreToolUse`, `PostToolUse`, `SubagentStart`, `SubagentStop`, ...) that record tool calls and subagent delegation at the host level, for *any* session, whatever is actually making the model calls. No code change, and it's a pure observer: it only ever exits 0 with no output, so it can't block or modify anything.

Copy `claude-code-plugin/` into your plugins directory (or set `CLAUDE_CODE_PLUGIN_DIRS` to point at it), run your session as usual, then point `mas-recon` at the `.mas-recon/trace-<session-id>.jsonl` file it writes — same report format as everything else above, just built from hook events instead of OpenTelemetry spans.

## What the report shows

- The real agent graph, confirmed by execution
- Per-agent model and token usage
- **Silent failures**: retries or errors the app's own code handled quietly, that never surface anywhere else in the normal output

Every claim the report makes is labeled by how it was obtained — stated directly by the trace, inferred from timing, or auto-detected by a heuristic — never presented with more confidence than the underlying data supports.
