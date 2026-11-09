"""Reconstruct the real agent topology + failure signatures from a captured trace.

This is the actual "recon" logic: turn a JSONL span dump into (1) the real
agent/tool graph, grounded in what actually executed, and (2) a list of
observed failure signatures (right now: structured-output / tool-call
validation errors and their retry outcome) -- the kind of thing no amount
of reading source code can tell you, because it only exists at runtime.

Usage:
    mas-recon <file>.jsonl [--out <name>.md]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def load_spans(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def detect_format(path: Path) -> str:
    """'strands' = Strands' native gen_ai.* convention (driver.py). 'monocle' =
    GPT Researcher's built-in tracer, a single JSON array. 'openinference' =
    the OpenInference LangChain/CrewAI/AutoGen instrumentors, same JSONL shell
    as 'strands' (same exporter code) but a completely different attribute
    vocabulary. 'claude_agent_sdk' = the OpenInference Claude Agent SDK
    instrumentor -- also carries openinference.span.kind, but its own distinct
    span-naming convention and attribute placement (real subagent identity
    sits on the PARENT tool-call span, not the agent span itself), confirmed
    only by actually running it -- checked before the generic openinference
    fallback since it would otherwise misclassify. 'claude_code_hooks' = the
    MAS Recon Claude Code plugin's own capture format -- not a span at all,
    just flat hook-event records (hook_event_name, tool_name, agent_id, ...),
    checked first since these records have no "attributes"/"name" shape in
    common with anything else here. Five tracers, five schemas, three of
    them nominally "OpenTelemetry" -- this is the first thing an analyzer has
    to handle, not an edge case."""
    text = path.read_text().strip()
    if text.startswith("["):
        return "monocle"
    lines = [json.loads(line) for line in text.splitlines()]
    if any("hook_event_name" in s for s in lines):
        return "claude_code_hooks"
    # Spans export child-before-parent (SimpleSpanProcessor fires on span end),
    # so a generic TOOL span with no "ClaudeAgentSDK." name can appear BEFORE
    # the identifying "ClaudeAgentSDK.*" line in the file. Scan the whole file
    # for the more specific signal first; only fall back to the generic
    # per-line checks once that's ruled out for every line.
    if any(s.get("name", "").startswith("ClaudeAgentSDK.") for s in lines):
        return "claude_agent_sdk"
    for s in lines:
        attrs = s.get("attributes", {})
        if "openinference.span.kind" in attrs:
            return "openinference"
        if "gen_ai.system" in attrs or "gen_ai.operation.name" in attrs:
            return "strands"
    return "strands"


def build_tree(spans: list[dict]) -> tuple[dict[str, dict], dict[str, list[str]]]:
    by_id = {s["span_id"]: s for s in spans}
    children: dict[str, list[str]] = defaultdict(list)
    for s in spans:
        parent = s["parent_span_id"]
        if parent:
            children[parent].append(s["span_id"])
    return by_id, children


def owning_agent_span_id(span_id: str, by_id: dict, parent_of: dict[str, str]) -> str | None:
    """Walk up the span tree to find the nearest invoke_agent ancestor."""
    current = span_id
    seen = set()
    while current in parent_of and current not in seen:
        seen.add(current)
        current = parent_of[current]
        if by_id.get(current, {}).get("name", "").startswith("invoke_agent"):
            return current
    return None


def extract_handoff_edges(spans: list[dict], by_id: dict, parent_of: dict[str, str]) -> list[dict]:
    """Swarm mode's handoffs are explicit tool calls naming their target -- a real,
    stated edge, not an inference from timing. Graph mode has no such call."""
    edges = []
    handoffs = sorted(
        (s for s in spans if s["attributes"].get("gen_ai.tool.name") == "handoff_to_agent"),
        key=lambda s: s["start_time_ns"],
    )
    for h in handoffs:
        caller_id = owning_agent_span_id(h["span_id"], by_id, parent_of)
        caller = by_id.get(caller_id, {}).get("attributes", {}).get("gen_ai.agent.name") if caller_id else None
        target = None
        for ev in h.get("events", []):
            if ev["name"] == "gen_ai.tool.message":
                try:
                    target = json.loads(ev["attributes"]["content"]).get("agent_name")
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass
        if caller and target:
            edges.append({"from": caller, "to": target, "kind": "handoff-real"})
    return edges


def analyze(spans: list[dict]) -> dict:
    by_id, children = build_tree(spans)
    parent_of = {s["span_id"]: s["parent_span_id"] for s in spans if s["parent_span_id"]}

    agent_spans = sorted(
        (s for s in spans if s["name"].startswith("invoke_agent")),
        key=lambda s: s["start_time_ns"],
    )
    tool_spans = [s for s in spans if s["name"].startswith("execute_tool")]

    agents = []
    for a in agent_spans:
        attrs = a["attributes"]
        agent_tools = [t for t in tool_spans if owning_agent_span_id(t["span_id"], by_id, parent_of) == a["span_id"]]
        errors = []
        for t in agent_tools:
            if t["attributes"].get("gen_ai.tool.status") == "error":
                detail = ""
                for ev in t.get("events", []):
                    if ev["name"] == "gen_ai.choice":
                        detail = ev["attributes"].get("message", "")
                # a retry = same tool name, later start, this same agent, ended in success
                retried_ok = any(
                    t2["attributes"].get("gen_ai.tool.name") == t["attributes"].get("gen_ai.tool.name")
                    and t2["start_time_ns"] > t["start_time_ns"]
                    and t2["attributes"].get("gen_ai.tool.status") == "success"
                    for t2 in agent_tools
                )
                errors.append(
                    {
                        "tool": t["attributes"].get("gen_ai.tool.name"),
                        "detail": detail,
                        "recovered": retried_ok,
                        # Named check: a tool call that errored, was never retried, and whose
                        # owning agent still finished (no crash) is a SILENT failure -- nothing
                        # downstream (not even this span's own parent) reflects it. See
                        # root_status_is_uninformative below for why you can't detect this from
                        # the top-level run status alone.
                        "category": "self-corrected" if retried_ok else "silent-failure",
                    }
                )
        agents.append(
            {
                "name": attrs.get("gen_ai.agent.name"),
                "model": attrs.get("gen_ai.request.model"),
                "tools_available": json.loads(attrs.get("gen_ai.agent.tools", "[]")),
                "tool_calls": [
                    {"name": t["attributes"].get("gen_ai.tool.name"), "status": t["attributes"].get("gen_ai.tool.status")}
                    for t in sorted(agent_tools, key=lambda t: t["start_time_ns"])
                ],
                "tokens": {
                    "input": attrs.get("gen_ai.usage.input_tokens"),
                    "output": attrs.get("gen_ai.usage.output_tokens"),
                    "total": attrs.get("gen_ai.usage.total_tokens"),
                    "cache_read": attrs.get("gen_ai.usage.cache_read.input_tokens"),
                },
                "start": attrs.get("gen_ai.event.start_time"),
                "end": attrs.get("gen_ai.event.end_time"),
                "errors": errors,
            }
        )

    # agents, so far, has one entry PER INVOCATION. That's correct for Strands
    # Graph/Swarm, where each node runs exactly once -- but AutoGen's chat-loop
    # agents can be invoked multiple times in one conversation, and counting
    # each invocation as a separate "agent" silently over-reports the real
    # agent count (caught testing this against AutoGen: a 2-agent conversation
    # reported "3 agents" because one of them spoke twice). Group by name for
    # the agent list; edges still need the ungrouped, time-ordered invocation
    # sequence below, so that's computed first, from `agents` before grouping.

    invocations = agents

    # Prefer real, stated edges (Swarm's explicit handoff_to_agent calls) when
    # present. Fall back to inferring from non-overlapping start/end order
    # between successive invocations (Graph mode has no agent-to-agent call span).
    edges = extract_handoff_edges(spans, by_id, parent_of)
    if not edges:
        for prev, nxt in zip(invocations, invocations[1:]):
            overlap = prev["end"] and nxt["start"] and nxt["start"] < prev["end"]
            edges.append(
                {"from": prev["name"], "to": nxt["name"], "kind": "parallel-overlap" if overlap else "sequential-inferred"}
            )

    by_name: dict[str, dict] = {}
    for inv in invocations:
        g = by_name.setdefault(
            inv["name"],
            {
                "name": inv["name"],
                "model": inv["model"],
                "tools_available": [],
                "tool_calls": [],
                "tokens": {"input": 0, "output": 0, "total": 0, "cache_read": 0},
                "start": inv["start"],
                "end": inv["end"],
                "errors": [],
                "invocations": 0,
            },
        )
        g["invocations"] += 1
        g["tools_available"] = sorted(set(g["tools_available"]) | set(inv["tools_available"]))
        g["tool_calls"] += inv["tool_calls"]
        for k in ("input", "output", "total", "cache_read"):
            g["tokens"][k] = (g["tokens"][k] or 0) + (inv["tokens"].get(k) or 0)
        g["end"] = inv["end"] or g["end"]
        g["errors"] += inv["errors"]
    agents = list(by_name.values())

    all_errors = [{"agent": a["name"], **e} for a in agents for e in a["errors"]]
    silent_failures = [e for e in all_errors if e["category"] == "silent-failure"]

    root = next((s for s in spans if s["name"] in ("invoke_graph", "invoke_swarm")), None)
    root_status = root["status"] if root else None

    return {
        "agents": agents,
        "edges": edges,
        "errors": all_errors,
        "silent_failure_count": len(silent_failures),
        "root_status": root_status,
        "span_count": len(spans),
    }


def render_markdown(result: dict, trace_path: Path, static_scan_note: str | None = None) -> str:
    """``static_scan_note`` is for when you actually have a prior static-analysis
    result to contrast against for one specific repo. It's not a general claim --
    omitted by default, since most traces this analyzes won't have one."""
    lines = [f"# Trace-based fit report: {trace_path.name}", ""]
    if static_scan_note:
        lines += [static_scan_note, ""]
    lines += ["## What the trace actually shows", ""]
    lines.append(f"**{len(result['agents'])} real agents**, {result['span_count']} spans captured.")
    lines.append("")
    lines.append("```mermaid")
    lines.append("flowchart TD")
    for a in result["agents"]:
        lines.append(f'  {a["name"]}["{a["name"]}\\n{a["model"]}"]')
    for e in result["edges"]:
        if e["kind"] == "handoff-real":
            arrow = "-->|handoff|"
        elif e["kind"] == "parallel-overlap":
            arrow = "-.->|parallel/overlap|"
        else:
            arrow = "-->"
        lines.append(f'  {e["from"]} {arrow} {e["to"]}')
    lines.append("```")
    lines.append("")
    lines.append("## Agents (ground truth, from real execution)")
    lines.append("")
    lines.append("| Agent | Invocations | Model | Tools available | Tool calls made | Tokens (in/out/cache-read) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for a in result["agents"]:
        calls = ", ".join(f"{c['name']}({c['status']})" for c in a["tool_calls"])
        tok = a["tokens"]
        lines.append(
            f"| {a['name']} | {a['invocations']} | {a['model']} | {len(a['tools_available'])} | {calls} | "
            f"{tok['input']}/{tok['output']}/{tok['cache_read']} |"
        )
    lines.append("")
    lines.append("## Observed failure signatures")
    lines.append("")
    lines.append(
        f"Top-level run span (`invoke_graph`/`invoke_swarm`) status: **{result['root_status']}**. "
        "This is the same regardless of whether a silent failure occurred below it -- the root span "
        "gives no signal either way. Everything below comes from inspecting individual tool-call spans."
    )
    lines.append("")
    silent = [e for e in result["errors"] if e["category"] == "silent-failure"]
    corrected = [e for e in result["errors"] if e["category"] == "self-corrected"]

    lines.append(f"### Silent failures (unrecovered) -- {len(silent)}")
    lines.append("")
    lines.append("Errored, was never retried, and the run still completed with no downstream signal.")
    lines.append("")
    if not silent:
        lines.append("None in this run.")
    else:
        lines.append("| Agent | Tool | Detail |")
        lines.append("| --- | --- | --- |")
        for e in silent:
            lines.append(f"| {e['agent']} | {e['tool']} | {e['detail']} |")
    lines.append("")
    lines.append(f"### Self-corrected failures (recovered) -- {len(corrected)}")
    lines.append("")
    lines.append("Errored, the agent was told exactly why, and a retry succeeded before the run moved on.")
    lines.append("")
    if not corrected:
        lines.append("None in this run.")
    else:
        lines.append("| Agent | Tool | Detail |")
        lines.append("| --- | --- | --- |")
        for e in corrected:
            lines.append(f"| {e['agent']} | {e['tool']} | {e['detail']} |")
    lines.append("")
    edge_kinds = {e["kind"] for e in result["edges"]}
    if "handoff-real" in edge_kinds:
        lines.append(
            "> Edges above are **real, stated edges**: each is a `handoff_to_agent` tool call naming its target "
            "agent directly, not an inference. This is Swarm mode -- agents route dynamically and the handoff "
            "call is the only ground truth for 'who called whom', which no static scan can see."
        )
    else:
        lines.append(
            "> Edges above are **inferred from execution timestamps** (non-overlapping agent spans => sequential), "
            "not a ground-truth edge the trace states directly. This matches reality for Strands Graph mode, which "
            "has no agent-to-agent call span -- only node start/stop timing."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Monocle (GPT Researcher's built-in tracer) -- a different schema entirely.
#
# Unlike Strands, Monocle spans are named after the literal library call site
# (e.g. "langchain_core...BaseChatModel.ainvoke"), not the host app's agent
# names. There is no span that says "this call belongs to the writer agent".
# The only way to recover that is to fingerprint the system prompt text each
# call actually sent, against prompts read out of this checkout's own agent
# source files. This is a heuristic, not a fact the trace states -- labeled
# as such everywhere it's used. It will also go stale if the repo's prompit
# wording changes, which is itself worth knowing: Strands' approach (bake the
# agent name into the span) does not have that fragility.
# ---------------------------------------------------------------------------

# (label, distinctive substring) -- pulled directly from multi_agents/agents/*.py
# in the exact checkout this was run against. Researcher/planner/browser/human/
# publisher don't define their own inline prompt in that folder; they call into
# gpt_researcher's shared core prompts, which aren't fingerprinted here.
KNOWN_AGENT_PROMPTS = [
    ("writer", "You are a research writer. Your sole purpose is to write a well-written"),
    ("reviewer", "You are an expert research article reviewer."),
    ("reviser", "You are an expert writer. Your goal is to revise drafts based on reviewer notes."),
    ("visualizer", "You are a Data Visualizer Agent."),
    ("fact_checker", "You are a diligent Fact Checker Agent."),
]

PHASE_LABELS = {
    "orchestrator.py": "Main orchestration graph (orchestrator.py)",
    "editor.py": "Editor revision sub-graph (editor.py)",
}


def load_monocle_spans(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def classify_prompt(system_text: str) -> str:
    for label, needle in KNOWN_AGENT_PROMPTS:
        if needle in system_text:
            return label
    return "unclassified (shared gpt_researcher core prompt, not multi_agents-specific)"


def _monocle_system_text(span: dict) -> str:
    for ev in span.get("events", []):
        if ev["name"] == "data.input":
            raw = ev["attributes"].get("input", [])
            for item in raw if isinstance(raw, list) else [raw]:
                try:
                    parsed = json.loads(item) if isinstance(item, str) else item
                    if isinstance(parsed, dict) and "system" in parsed:
                        return parsed["system"]
                except (json.JSONDecodeError, TypeError):
                    continue
    return ""


def _monocle_tokens(span: dict) -> dict | None:
    for ev in span.get("events", []):
        if ev["name"] == "metadata":
            a = ev["attributes"]
            if "total_tokens" in a:
                return {"prompt": a.get("prompt_tokens"), "completion": a.get("completion_tokens"), "total": a.get("total_tokens")}
    return None


def analyze_monocle(spans: list[dict]) -> dict:
    by_id = {s["context"]["span_id"]: s for s in spans}
    parent_of = {s["context"]["span_id"]: s["parent_id"] for s in spans if s.get("parent_id")}

    def owning_phase(span_id: str) -> str | None:
        current = span_id
        seen = set()
        while current in parent_of and current not in seen:
            seen.add(current)
            current = parent_of[current]
            src = by_id.get(current, {}).get("attributes", {}).get("span_source", "")
            for fname, label in PHASE_LABELS.items():
                if fname in src:
                    return label
        return None

    inference_spans = [s for s in spans if s["attributes"].get("span.type") == "inference.framework"]
    retrieval_spans = [s for s in spans if s["attributes"].get("span.type") == "retrieval"]
    error_spans = [s for s in spans if s["status"]["status_code"] != "OK"]

    calls = []
    for s in inference_spans:
        system_text = _monocle_system_text(s)
        calls.append(
            {
                "phase": owning_phase(s["context"]["span_id"]) or "(unattributed)",
                "classification": classify_prompt(system_text),
                "model": s["attributes"].get("entity.2.name"),
                "tokens": _monocle_tokens(s),
            }
        )

    by_classification: dict[str, list[dict]] = defaultdict(list)
    for c in calls:
        by_classification[c["classification"]].append(c)

    return {
        "span_count": len(spans),
        "phases": sorted(set(c["phase"] for c in calls)),
        "inference_call_count": len(inference_spans),
        "retrieval_call_count": len(retrieval_spans),
        "span_level_errors": len(error_spans),
        "by_classification": {
            k: {
                "count": len(v),
                "total_tokens": sum(c["tokens"]["total"] for c in v if c["tokens"]),
                "models": sorted(set(c["model"] for c in v if c["model"])),
            }
            for k, v in by_classification.items()
        },
    }


def render_monocle_markdown(result: dict, trace_path: Path) -> str:
    """Unlike OpenInference, Monocle's spans carry no agent-boundary span at all --
    just the raw LLM/tool calls -- so there's no structural trick that recovers
    agent identity generically (OpenInference's "has an LLM descendant" heuristic
    has nothing to anchor to here). KNOWN_AGENT_PROMPTS and PHASE_LABELS above are
    GPT Researcher-specific, derived from that one checkout's source. A genuinely
    generic Monocle adapter would need a *different* per-app prompt or phase table
    every time -- this is a real, open limitation, not a decision to hide."""
    lines = [f"# Trace-based fit report: {trace_path.name} (Monocle)", ""]
    lines.append("")
    lines.append(f"**{result['span_count']} spans captured** across {len(result['phases'])} graph phase(s): " + ", ".join(result["phases"]))
    lines.append("")
    lines.append("## What this tracer cannot tell you (and Strands could)")
    lines.append("")
    lines.append(
        "Monocle instruments LangChain/LangGraph/OpenAI-SDK call sites, not the host app's own "
        "agent boundaries. No span here says \"this call is the writer agent\" the way Strands' "
        "`gen_ai.agent.name` did for Thrad.ai. Agent attribution below is a **heuristic**: matching "
        "each call's system prompt text against known prompt strings pulled from this checkout's "
        "own source files. Calls that don't match a known prompt are grouped as \"unclassified\" "
        "rather than guessed."
    )
    lines.append("")
    lines.append("## LLM calls by (heuristically) identified agent")
    lines.append("")
    lines.append("| Classification | Calls | Total tokens | Model(s) |")
    lines.append("| --- | --- | --- | --- |")
    for label, data in sorted(result["by_classification"].items(), key=lambda kv: -kv[1]["count"]):
        lines.append(f"| {label} | {data['count']} | {data['total_tokens']} | {', '.join(data['models'])} |")
    lines.append("")
    lines.append(f"Retrieval calls observed: {result['retrieval_call_count']}. Span-level errors: {result['span_level_errors']}.")
    lines.append("")
    lines.append("## A real blind spot this run exposed")
    lines.append("")
    lines.append(
        f"The console log for this run showed **two real DNS failures** (Wikipedia lookups failing "
        "on this network) that the app's own code caught and logged before they ever reached a "
        "LangChain or OpenAI call. Monocle's instrumentation sits at the LangChain/OpenAI boundary, "
        f"so it never saw them: this trace reports **{result['span_level_errors']} span-level errors**, "
        "which is technically accurate and still misses real failures that happened. Zero errors in "
        "a trace is evidence the instrumented layer saw no errors -- not evidence nothing went wrong."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OpenInference (TradingAgents, via the LangChain auto-instrumentor) -- a
# third schema. Unlike Monocle, OpenInference's LangChain instrumentor DOES
# preserve the host app's own LangGraph node names as the span name directly
# ("Bull Researcher", "Trader", ...) -- no prompt-fingerprinting heuristic
# needed here. But "openinference.span.kind" only says the generic operation
# type (LLM/TOOL/CHAIN/PROMPT) -- it does not mark which CHAIN spans are the
# app's own named agent nodes vs. internal LangGraph plumbing ("tools",
# "RunnableSequence", "_tools_or_done"). That list has to come from the repo's
# own source (tradingagents/graph/setup.py's add_node calls), the same
# discipline as GPT Researcher's prompt table: derived from this checkout,
# not guessed, and it will go stale if the graph is restructured.
# ---------------------------------------------------------------------------

# A BLOCKLIST of framework-internal span names generalizes across apps in a way
# an allowlist of agent names never can: these class/function names come from
# LangChain/LangGraph/deepagents/OpenInference themselves, so they repeat
# identically across every app built on those frameworks, whereas agent names
# are app-specific by definition. Anything NOT on this list, of kind CHAIN or
# AGENT, that actually does an LLM call somewhere in its subtree, is treated
# as a real app-level agent/node.
GENERIC_SPAN_NAMES = {
    "RunnableSequence", "RunnableParallel", "RunnableLambda", "RunnableBranch",
    "ChatPromptTemplate", "PromptTemplate", "FewShotPromptTemplate",
    "PydanticToolsParser", "StrOutputParser", "JsonOutputParser", "ToolsOutputParser",
    "ChatOpenAI", "AzureChatOpenAI", "ChatAnthropic", "ChatBedrock", "ChatBedrockConverse",
    "LocalCompatibleChatOpenAI", "ChatGoogleGenerativeAI", "ChatGroq", "ChatMistralAI",
    "LangGraph", "StateGraph", "CompiledStateGraph", "ToolNode",
    "agent", "tools", "task", "model",
}
GENERIC_SPAN_PATTERNS = [
    re.compile(r".*Middleware\.\w+$"),  # e.g. "ModelCallLimitMiddleware.before_model"
    re.compile(r"^should_continue"),     # LangGraph conditional-edge predicates
    re.compile(r"^_"),                   # internal helper functions by convention
    re.compile(r"wrap_up"),
]


def _is_generic_span_name(name: str) -> bool:
    if name in GENERIC_SPAN_NAMES:
        return True
    return any(p.search(name) for p in GENERIC_SPAN_PATTERNS)


def detect_agent_nodes(spans: list[dict]) -> set[str]:
    """Auto-detect which CHAIN/AGENT-kind spans are the host app's own named
    agents, replacing a per-app hardcoded allowlist.

    Heuristic: kind in (CHAIN, AGENT), name isn't known framework plumbing,
    AND it actually does an LLM call somewhere in its own subtree (a real
    agent does something; a router/conditional-edge function does not -- this
    is what separates "Market Analyst" from "should_continue_debate" even
    though both are kind=CHAIN).

    This is a heuristic, not a certainty: a creatively-named internal helper
    could slip through, or a real agent could be named exactly like a generic
    component. Pass an explicit known_agent_nodes set to analyze_openinference
    to override it when that happens.
    """
    by_id = {s["span_id"]: s for s in spans}
    children: dict[str, list[str]] = defaultdict(list)
    for s in spans:
        if s["parent_span_id"]:
            children[s["parent_span_id"]].append(s["span_id"])

    def has_llm_descendant(span_id: str, seen: set[str]) -> bool:
        for cid in children.get(span_id, []):
            if cid in seen:
                continue
            seen.add(cid)
            if by_id[cid]["attributes"].get("openinference.span.kind") == "LLM":
                return True
            if has_llm_descendant(cid, seen):
                return True
        return False

    candidates = set()
    for s in spans:
        kind = s["attributes"].get("openinference.span.kind")
        if kind not in ("CHAIN", "AGENT"):
            continue
        if _is_generic_span_name(s["name"]):
            continue
        # kind=="AGENT" is the instrumentor directly claiming "this is an agent" --
        # trust it outright, since some instrumentors (CrewAI) fold the LLM call
        # itself into that one span with no separate LLM-kind child to find.
        # kind=="CHAIN" has no such explicit signal (LangGraph nodes, conditional-
        # edge functions, and real named agents are all CHAIN alike), so it still
        # needs the "does it actually call an LLM" structural check.
        if kind == "AGENT" or has_llm_descendant(s["span_id"], set()):
            candidates.add(s["name"])
    return candidates


def analyze_openinference(spans: list[dict], known_agent_nodes: set[str] | None = None) -> dict:
    """When ``known_agent_nodes`` is omitted, auto-detects them via
    ``detect_agent_nodes`` (see its docstring for the heuristic and its
    limits). Pass an explicit set to override the heuristic for a specific
    app -- e.g. when it misses a real agent or includes a false positive."""
    auto_detected = known_agent_nodes is None
    known_agent_nodes = known_agent_nodes or detect_agent_nodes(spans)
    by_id = {s["span_id"]: s for s in spans}
    children: dict[str, list[str]] = defaultdict(list)
    for s in spans:
        if s["parent_span_id"]:
            children[s["parent_span_id"]].append(s["span_id"])

    def descendants(span_id: str, stop_at_agent: bool = True) -> list[dict]:
        """BFS down the tree, not crossing into a nested agent node's own subtree."""
        out, queue = [], list(children.get(span_id, []))
        while queue:
            cid = queue.pop(0)
            child = by_id[cid]
            if stop_at_agent and child["name"] in known_agent_nodes:
                continue  # belongs to that nested agent, not this one
            out.append(child)
            queue.extend(children.get(cid, []))
        return out

    agent_spans = sorted(
        (s for s in spans if s["name"] in known_agent_nodes),
        key=lambda s: s["start_time_ns"],
    )

    agents = []
    for a in agent_spans:
        desc = descendants(a["span_id"])
        llm_spans = [d for d in desc if d["attributes"].get("openinference.span.kind") == "LLM"]
        tool_spans = [d for d in desc if d["attributes"].get("openinference.span.kind") == "TOOL"]
        total_tokens = sum(int(l["attributes"].get("llm.token_count.total", 0) or 0) for l in llm_spans)
        models = sorted(set(l["attributes"].get("llm.model_name") for l in llm_spans if l["attributes"].get("llm.model_name")))
        agents.append(
            {
                "name": a["name"],
                "start": a["start_time_ns"],
                "end": a["end_time_ns"],
                "models": models,
                "llm_calls": len(llm_spans),
                "total_tokens": total_tokens,
                "tools_called": sorted(set(t["attributes"].get("tool.name") for t in tool_spans if t["attributes"].get("tool.name"))),
            }
        )

    # Edges inferred from timing, same honesty rule as Strands Graph mode: no
    # explicit agent-to-agent call span exists here either. Overlapping start/end
    # across analysts is the real, observed parallel fan-out, not a guess --
    # the timestamps themselves show it.
    edges = []
    for prev, nxt in zip(agents, agents[1:]):
        overlap = prev["end"] and nxt["start"] and nxt["start"] < prev["end"]
        edges.append({"from": prev["name"], "to": nxt["name"], "kind": "parallel-overlap" if overlap else "sequential-inferred"})

    error_spans = [s for s in spans if s["status"] not in ("OK", "UNSET", None)]

    return {
        "span_count": len(spans),
        "agents": agents,
        "edges": edges,
        "span_level_errors": len(error_spans),
        "agent_detection": "auto" if auto_detected else "manual override",
        "known_agent_nodes": sorted(known_agent_nodes),
    }


# ---------------------------------------------------------------------------
# Claude Agent SDK (via the OpenInference Claude Agent SDK instrumentor) -- a
# fifth schema, and one that needed a real run to understand, not just the
# instrumentor's README. Confirmed empirically against an actual captured
# trace: the root span (no parent) is the orchestrator and embeds its own
# LLM data directly, not via a child span. Each subagent's real identity
# (subagent_type, e.g. "researcher") sits on its PARENT tool-call span's
# tool.parameters JSON -- the subagent's own AGENT span only echoes the
# generic tool name ("Agent") in its agent.name attribute, which is useless
# for identification. And the subagent's own LLM/token data is not captured
# at all this way: its real model calls execute inside the separate `claude`
# CLI subprocess, which doesn't relay per-call telemetry back as spans here --
# only the orchestrator's own top-level call gets real token data.
# ---------------------------------------------------------------------------


def analyze_claude_agent_sdk(spans: list[dict]) -> dict:
    by_id = {s["span_id"]: s for s in spans}

    def llm_data(span: dict) -> dict | None:
        a = span["attributes"]
        if "llm.token_count.total" not in a:
            return None
        return {
            "model": a.get("llm.model_name"),
            "input": a.get("llm.token_count.prompt"),
            "output": a.get("llm.token_count.completion"),
            "total": a.get("llm.token_count.total"),
        }

    agents = []
    for s in spans:
        if not s["name"].startswith("ClaudeAgentSDK.") or s["attributes"].get("openinference.span.kind") != "AGENT":
            continue

        if s["parent_span_id"] is None:
            agents.append({"name": "orchestrator", "role": "main", "tokens": llm_data(s)})
            continue

        parent = by_id.get(s["parent_span_id"])
        subagent_type = None
        if parent and parent["attributes"].get("openinference.span.kind") == "TOOL":
            try:
                params = json.loads(parent["attributes"].get("tool.parameters", "{}"))
                subagent_type = params.get("subagent_type")
            except json.JSONDecodeError:
                pass
        agents.append(
            {
                "name": subagent_type or s["attributes"].get("agent.name") or s["name"],
                "role": "subagent",
                "tokens": llm_data(s),
            }
        )

    orchestrator = next((a for a in agents if a["role"] == "main"), None)
    edges = []
    if orchestrator:
        # Real, stated edges: a subagent span's parent IS the tool call that
        # spawned it -- read directly from the tree, not inferred from timing,
        # unlike every timestamp-based fallback elsewhere in this file.
        for a in agents:
            if a["role"] == "subagent":
                edges.append({"from": orchestrator["name"], "to": a["name"], "kind": "delegation-real"})

    error_spans = [s for s in spans if s["status"] not in ("OK", "UNSET", None)]

    return {
        "span_count": len(spans),
        "agents": agents,
        "edges": edges,
        "span_level_errors": len(error_spans),
    }


def render_claude_agent_sdk_markdown(result: dict, trace_path: Path) -> str:
    lines = [f"# Trace-based fit report: {trace_path.name} (Claude Agent SDK)", ""]
    lines.append(
        "Agent identity here is pieced together from two places, confirmed by reading a "
        "real captured trace, not assumed from the instrumentor's documentation: the root "
        "orchestrator is the span with no parent, and each subagent's real name comes from "
        "`subagent_type` on its **parent** tool-call span -- the subagent's own span only "
        "echoes the generic tool name (\"Agent\"), which carries no identifying information."
    )
    lines.append("")
    lines.append(f"**{len(result['agents'])} agent(s)**, {result['span_count']} spans captured.")
    lines.append("")
    lines.append("```mermaid")
    lines.append("flowchart TD")

    def node_id(name: str) -> str:
        return "n_" + "".join(c if c.isalnum() else "_" for c in name)

    for a in result["agents"]:
        label = a["name"] + (f"\\n{a['tokens']['model']}" if a["tokens"] else "")
        lines.append(f'  {node_id(a["name"])}["{label}"]')
    for e in result["edges"]:
        lines.append(f"  {node_id(e['from'])} -->|delegation| {node_id(e['to'])}")
    lines.append("```")
    lines.append("")
    lines.append(
        "Edges above are a **real, stated delegation**: each subagent span's parent is "
        "literally the tool call that spawned it, read from the span tree -- not inferred "
        "from timestamps the way most other adapters in this tool have to."
    )
    lines.append("")
    lines.append("## Agents")
    lines.append("")
    lines.append("| Agent | Role | Model | Tokens (in/out/total) |")
    lines.append("| --- | --- | --- | --- |")
    for a in result["agents"]:
        t = a["tokens"]
        tok_str = f"{t['input']}/{t['output']}/{t['total']}" if t else "not captured"
        model = t["model"] if t else "-"
        lines.append(f"| {a['name']} | {a['role']} | {model} | {tok_str} |")
    lines.append("")
    missing = [a["name"] for a in result["agents"] if a["role"] == "subagent" and not a["tokens"]]
    if missing:
        lines.append(
            f"> **Token/model data is not captured for subagent(s): {', '.join(missing)}.** "
            "Confirmed empirically, not assumed: a subagent's real LLM calls execute inside "
            "the separate `claude` CLI subprocess, and this instrumentor doesn't relay "
            "per-call telemetry back as spans for that nested work -- only its final "
            "hand-back result reaches the conversation. Claude Code's own native "
            "OpenTelemetry traces (beta) nest real `llm_request` spans under each subagent "
            "with proper token data; this lighter, OpenInference-based path doesn't have "
            "that yet."
        )
        lines.append("")
    lines.append(f"Span-level errors: **{result['span_level_errors']}**.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claude Code hooks (via the MAS Recon plugin) -- not a span tree at all,
# just flat lifecycle events (PreToolUse/PostToolUse/PostToolUseFailure/
# SubagentStart/SubagentStop) correlated by tool_use_id and agent_id. The
# portable alternative to instrumenting a specific library: it works for ANY
# Claude Code session, not just one agent-building framework, because the
# hooks fire at the host level, not inside a Python/JS library's own code.
# ---------------------------------------------------------------------------


def analyze_claude_code_hooks(records: list[dict]) -> dict:
    # Pair Pre/Post(Failure) events sharing a tool_use_id into one call.
    calls: dict[str, dict] = {}
    for r in records:
        tuid = r.get("tool_use_id")
        if not tuid:
            continue
        c = calls.setdefault(
            tuid,
            {"tool_name": r.get("tool_name"), "agent_id": None, "agent_type": None, "status": None, "error": None, "duration_ms": None},
        )
        if r.get("agent_id"):
            c["agent_id"] = r["agent_id"]
            c["agent_type"] = r.get("agent_type")
        if r["hook_event_name"] == "PostToolUse":
            c["status"] = "success"
            c["duration_ms"] = r.get("duration_ms")
        elif r["hook_event_name"] == "PostToolUseFailure":
            c["status"] = "error"
            c["error"] = r.get("error")
            c["duration_ms"] = r.get("duration_ms")

    # Group by agent_type (not agent_id) the same way the AutoGen fix does:
    # the same role can be started more than once in one session, and that's
    # one agent invoked repeatedly, not several distinct agents.
    by_agent: dict[str | None, dict] = {}

    def bucket(agent_type: str | None) -> dict:
        key = agent_type
        return by_agent.setdefault(
            key,
            {"name": agent_type or "orchestrator", "role": "subagent" if agent_type else "main", "invocations": set(), "calls": []},
        )

    for c in calls.values():
        b = bucket(c["agent_type"])
        b["invocations"].add(c["agent_id"])
        b["calls"].append(c)

    # Real, stated subagent-start/stop lifecycle, independent of tool calls --
    # catches a subagent that started but made no recorded tool call at all,
    # and carries the honest "did it hand back a final message" signal.
    subagent_lifecycle: dict[str, dict] = {}
    for r in records:
        if r["hook_event_name"] == "SubagentStart":
            subagent_lifecycle.setdefault(r.get("agent_id"), {"agent_type": r.get("agent_type"), "stopped": False, "last_message": None})
        elif r["hook_event_name"] == "SubagentStop":
            entry = subagent_lifecycle.setdefault(r.get("agent_id"), {"agent_type": r.get("agent_type"), "stopped": False, "last_message": None})
            entry["stopped"] = True
            entry["last_message"] = r.get("last_assistant_message")
            bucket(r.get("agent_type"))["invocations"].add(r.get("agent_id"))

    agents = []
    for a in by_agent.values():
        errors = []
        for c in a["calls"]:
            if c["status"] != "error":
                continue
            retried_ok = any(c2["tool_name"] == c["tool_name"] and c2["status"] == "success" for c2 in a["calls"] if c2 is not c)
            errors.append(
                {
                    "tool": c["tool_name"],
                    "detail": c["error"],
                    "recovered": retried_ok,
                    "category": "self-corrected" if retried_ok else "silent-failure",
                }
            )
        agents.append(
            {
                "name": a["name"],
                "role": a["role"],
                "invocations": len(a["invocations"]) or 1,
                "tool_calls": len(a["calls"]),
                "errors": errors,
            }
        )

    orchestrator = next((a for a in agents if a["role"] == "main"), None)
    edges = []
    if orchestrator:
        for a in agents:
            if a["role"] == "subagent":
                edges.append({"from": orchestrator["name"], "to": a["name"], "kind": "delegation-real"})

    all_errors = [{"agent": a["name"], **e} for a in agents for e in a["errors"]]
    silent_failures = [e for e in all_errors if e["category"] == "silent-failure"]

    return {
        "event_count": len(records),
        "agents": agents,
        "edges": edges,
        "errors": all_errors,
        "silent_failure_count": len(silent_failures),
    }


def render_claude_code_hooks_markdown(result: dict, trace_path: Path) -> str:
    lines = [f"# Trace-based fit report: {trace_path.name} (Claude Code hooks)", ""]
    lines.append(
        "Captured via the MAS Recon Claude Code plugin -- a portable alternative to "
        "instrumenting one specific library: these hooks fire at the host level for "
        "*any* Claude Code session, whatever framework or raw API calls are behind it. "
        "Trade-off: no span timing tree, no token/model data -- just tool-call and "
        "subagent lifecycle events, correlated by `tool_use_id` and `agent_id`."
    )
    lines.append("")
    lines.append(f"**{len(result['agents'])} agent(s)**, {result['event_count']} hook events captured.")
    lines.append("")
    lines.append("```mermaid")
    lines.append("flowchart TD")

    def node_id(name: str) -> str:
        return "n_" + "".join(c if c.isalnum() else "_" for c in name)

    for a in result["agents"]:
        lines.append(f'  {node_id(a["name"])}["{a["name"]}"]')
    for e in result["edges"]:
        lines.append(f"  {node_id(e['from'])} -->|delegation| {node_id(e['to'])}")
    lines.append("```")
    lines.append("")
    lines.append(
        "Edges are a **real, stated delegation**: each subagent's own `SubagentStart` event "
        "is a direct fact, not inferred from timing."
    )
    lines.append("")
    lines.append("## Agents")
    lines.append("")
    lines.append("| Agent | Role | Invocations | Tool calls |")
    lines.append("| --- | --- | --- | --- |")
    for a in result["agents"]:
        lines.append(f"| {a['name']} | {a['role']} | {a['invocations']} | {a['tool_calls']} |")
    lines.append("")
    lines.append("## Failure signatures")
    lines.append("")
    silent = [e for e in result["errors"] if e["category"] == "silent-failure"]
    corrected = [e for e in result["errors"] if e["category"] == "self-corrected"]
    lines.append(f"### Silent failures (unrecovered) -- {len(silent)}")
    lines.append("")
    if not silent:
        lines.append("None in this run.")
    else:
        lines.append("| Agent | Tool | Detail |")
        lines.append("| --- | --- | --- |")
        for e in silent:
            lines.append(f"| {e['agent']} | {e['tool']} | {e['detail']} |")
    lines.append("")
    lines.append(f"### Self-corrected failures (recovered) -- {len(corrected)}")
    lines.append("")
    if not corrected:
        lines.append("None in this run.")
    else:
        lines.append("| Agent | Tool | Detail |")
        lines.append("| --- | --- | --- |")
        for e in corrected:
            lines.append(f"| {e['agent']} | {e['tool']} | {e['detail']} |")
    return "\n".join(lines)


def render_openinference_markdown(result: dict, trace_path: Path) -> str:
    """Generic report for any OpenInference-traced app -- no hardcoded narrative
    about one specific run. Honest about whether agent detection was automatic
    or manually overridden, since that's now a real variable, not a constant."""
    lines = [f"# Trace-based fit report: {trace_path.name} (OpenInference)", ""]
    if result["agent_detection"] == "auto":
        lines.append(
            "Agent nodes below were **auto-detected**: any CHAIN/AGENT-kind span that isn't known "
            "LangChain/LangGraph framework plumbing and does a real LLM call somewhere in its own "
            "subtree. This is a heuristic, not a fact the trace states directly -- it can miss a "
            "real agent with an unusual name, or include a false positive. Detected node names: "
            f"{', '.join(result['known_agent_nodes'])}."
        )
    else:
        lines.append(
            f"Agent nodes below were a **manual override** (not auto-detected): "
            f"{', '.join(result['known_agent_nodes'])}."
        )
    lines.append("")
    lines.append(f"**{len(result['agents'])} agent node(s) executed**, {result['span_count']} spans captured.")
    lines.append("")
    lines.append("```mermaid")
    lines.append("flowchart TD")
    for a in result["agents"]:
        model = a["models"][0] if a["models"] else "?"
        lines.append(f'  n{id(a)}["{a["name"]}\\n{model}"]')
    for e in result["edges"]:
        arrow = "-.->|parallel/overlap|" if e["kind"] == "parallel-overlap" else "-->"
        from_id = next(f"n{id(a)}" for a in result["agents"] if a["name"] == e["from"])
        to_id = next(f"n{id(a)}" for a in result["agents"] if a["name"] == e["to"])
        lines.append(f"  {from_id} {arrow} {to_id}")
    lines.append("```")
    lines.append("")
    lines.append(
        "Edges are inferred from execution-order timing (no explicit agent-to-agent call span exists "
        "for a CHAIN-kind node the way Swarm's `handoff_to_agent` tool call does) -- treat as an "
        "inference, not a stated fact."
    )
    lines.append("")
    lines.append("## Agents (ground truth, from real execution)")
    lines.append("")
    lines.append("| Agent | Model | LLM calls | Total tokens | Tools called |")
    lines.append("| --- | --- | --- | --- | --- |")
    for a in result["agents"]:
        lines.append(
            f"| {a['name']} | {', '.join(a['models']) or '?'} | {a['llm_calls']} | {a['total_tokens']} | "
            f"{', '.join(a['tools_called']) or '-'} |"
        )
    lines.append("")
    if result["agents"] and all(a["llm_calls"] == 0 for a in result["agents"]):
        lines.append(
            "> **No LLM call or token data was found inside any detected agent's own span.** "
            "The agents and topology above are still real (an AGENT/CHAIN-kind span is a direct "
            "claim by the instrumentor, not a guess), but this particular instrumentor version "
            "folds the model call into the agent span itself without exposing a separate LLM-kind "
            "child span or token-usage attributes -- cost/call-count data simply isn't captured at "
            "this granularity here, not that the agent made zero calls."
        )
        lines.append("")
    lines.append(f"Span-level errors: **{result['span_level_errors']}**.")
    return "\n".join(lines)


def run_and_print(trace_path: Path, out_path: Path | None = None, known_agents: str | None = None) -> Path:
    """Analyze one trace file, write the markdown report, and print a short
    summary. This is the one function both the CLI and start_capture()'s
    automatic post-run report call -- so a capture-and-analyze call and a
    `mas-recon analyze` call always produce identical output."""
    trace_path = Path(trace_path)
    fmt = detect_format(trace_path)
    out_path = Path(out_path) if out_path else trace_path.parent / "reports" / f"{trace_path.stem}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "monocle":
        spans = load_monocle_spans(trace_path)
        result = analyze_monocle(spans)
        report = render_monocle_markdown(result, trace_path)
        out_path.write_text(report)

        print(f"format: monocle, spans: {result['span_count']}")
        print(f"inference calls: {result['inference_call_count']}, retrieval calls: {result['retrieval_call_count']}")
        print(f"span-level errors: {result['span_level_errors']} (does not include errors the app's own code swallowed)")
        for label, data in result["by_classification"].items():
            print(f"  - {label}: {data['count']} call(s), {data['total_tokens']} tokens")
    elif fmt == "claude_agent_sdk":
        spans = load_spans(trace_path)
        result = analyze_claude_agent_sdk(spans)
        report = render_claude_agent_sdk_markdown(result, trace_path)
        out_path.write_text(report)

        print(f"format: claude_agent_sdk, spans: {result['span_count']}")
        print(f"agents found: {len(result['agents'])}")
        for a in result["agents"]:
            tok = a["tokens"]["total"] if a["tokens"] else "not captured"
            print(f"  - {a['name']} ({a['role']}): tokens={tok}")
    elif fmt == "openinference":
        spans = load_spans(trace_path)
        known = set(known_agents.split(",")) if known_agents else None
        result = analyze_openinference(spans, known_agent_nodes=known)
        report = render_openinference_markdown(result, trace_path)
        out_path.write_text(report)

        print(f"format: openinference, spans: {result['span_count']}")
        print(f"agent detection: {result['agent_detection']}, agents found: {len(result['agents'])}")
        print(f"span-level errors: {result['span_level_errors']} (does not include errors the app's own code swallowed)")
        for a in result["agents"]:
            print(f"  - {a['name']}: {a['llm_calls']} LLM call(s), {a['total_tokens']} tokens")
    elif fmt == "claude_code_hooks":
        records = load_spans(trace_path)
        result = analyze_claude_code_hooks(records)
        report = render_claude_code_hooks_markdown(result, trace_path)
        out_path.write_text(report)

        print(f"format: claude_code_hooks, hook events: {result['event_count']}")
        print(f"agents found: {len(result['agents'])}")
        print(f"silent (unrecovered) failures: {result['silent_failure_count']}")
        for a in result["agents"]:
            print(f"  - {a['name']} ({a['role']}): {a['tool_calls']} tool call(s)")
    else:
        spans = load_spans(trace_path)
        result = analyze(spans)
        report = render_markdown(result, trace_path)
        out_path.write_text(report)

        print(f"agents found: {len(result['agents'])}")
        print(f"root span status: {result['root_status']} (uninformative either way)")
        print(f"silent (unrecovered) failures: {result['silent_failure_count']}")
        print(f"self-corrected failures: {len(result['errors']) - result['silent_failure_count']}")
        for e in result["errors"]:
            print(f"  - [{e['category']}] {e['agent']}/{e['tool']}: {e['detail']}")

    print(f"report written to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_file")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--known-agents",
        default=None,
        help="OpenInference traces only: comma-separated agent node names to override auto-detection",
    )
    args = parser.parse_args()
    run_and_print(Path(args.trace_file), out_path=args.out, known_agents=args.known_agents)


if __name__ == "__main__":
    main()
