# MAS Recon

Point it at a running multi-agent system; get back the real agent topology and a silent-failure report, from an actual run.

```python
from mas_recon import start_capture

handle = start_capture()   # auto-detects your framework, starts tracing
# ... run your agent exactly as you already do ...
handle.stop()               # writes the trace, prints the report
```

Supported today: **AWS Strands Agents**, **LangChain / LangGraph**, **CrewAI**, and **AutoGen / AG2**. Anything else is told plainly it isn't recognized yet, instead of silently capturing nothing.

See [QUICKSTART.md](QUICKSTART.md) for the walkthrough, or run one of the scripts in `examples/`.

## Install

```bash
pip install "mas-recon[langchain]"   # LangChain / LangGraph
pip install "mas-recon[strands]"     # AWS Strands Agents
pip install "mas-recon[crewai]"      # CrewAI
pip install "mas-recon[autogen]"     # AutoGen / AG2
```

## What the report shows

- The real agent graph, confirmed by execution — not inferred from reading source
- Per-agent model and token usage
- **Silent failures**: retries or errors the app's own code handled quietly, that never surface anywhere else in the normal output

Every claim the report makes is labeled by how it was obtained — stated directly by the trace, inferred from timing, or auto-detected by a heuristic — never presented with more confidence than the underlying data supports.
