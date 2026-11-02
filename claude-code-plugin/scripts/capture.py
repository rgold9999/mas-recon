#!/usr/bin/env python3
"""MAS Recon's Claude Code hook: reads one event as JSON on stdin, appends a
trimmed-down record to a per-session trace file, and exits. Always exit 0
with no stdout -- a plain command hook with no output is a pure observer in
Claude Code (confirmed against the hooks docs): it can never block, deny, or
modify a tool call, deliberately, so installing this can't change how the
session behaves, only what gets recorded alongside it.

Fields kept, per event (see the hooks docs for the full payload):
  PreToolUse / PostToolUse / PostToolUseFailure: tool_name, tool_use_id,
    tool_input, tool_response (Post only), error (Failure only), duration_ms
  SubagentStart / SubagentStop: agent_id, agent_type, last_assistant_message

agent_id/agent_type are only present when the hook fires INSIDE a subagent's
own context -- their absence on an event means it belongs to the main
session, not a parsing gap. The analyzer relies on exactly that distinction
to attribute tool calls to the right agent.
"""

import json
import os
import sys
import time


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return  # never break the session over a malformed payload

    cwd = event.get("cwd") or os.getcwd()
    session_id = event.get("session_id", "unknown-session")
    out_dir = os.path.join(cwd, ".mas-recon")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:
        return

    record = {
        "captured_at": time.time(),
        "hook_event_name": event.get("hook_event_name"),
        "session_id": session_id,
        "agent_id": event.get("agent_id"),
        "agent_type": event.get("agent_type"),
        "tool_name": event.get("tool_name"),
        "tool_use_id": event.get("tool_use_id"),
        "tool_input": event.get("tool_input"),
        "tool_response": event.get("tool_response"),
        "error": event.get("error"),
        "duration_ms": event.get("duration_ms"),
        "last_assistant_message": event.get("last_assistant_message"),
    }
    record = {k: v for k, v in record.items() if v is not None}

    out_path = os.path.join(out_dir, f"trace-{session_id}.jsonl")
    try:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        return


if __name__ == "__main__":
    main()
