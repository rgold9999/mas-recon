"""ChatDev (the "virtual software company", chatdev1.0 branch) adapter -- a
fourth trace schema, and a different shape entirely from the other three:
ChatDev emits no OpenTelemetry spans at all. Its own run log
(WareHouse/<project>/*.log) is a plain structured text file written as the
run happens, not an export from any tracer -- and ChatDev's own repo already
ships one of these per completed run under WareHouse/. That makes this
purely retrospective: there is no running system to start_capture() against,
only a log that already exists.

What the log gives us, and why it's trustworthy:

  - Real, STATED dialogue edges. Every turn is printed as a literal line like
    "Chief Product Officer: **Chief Product Officer<->Chief Executive Officer
    on : DemandAnalysis, turn 0**" -- naming both participants and the phase
    directly. This is the same category of evidence as Strands Swarm's
    explicit handoff_to_agent call (see analyzer.py's extract_handoff_edges):
    a fact the trace states, not a timing inference.

  - Three independent, genuine "silent failure" signal types, none of which
    ChatDev surfaces as a top-level run failure (it has no such concept --
    a run either produces a WareHouse/ folder or it doesn't):
      1. Runtime tracebacks from actually executing the generated code
         (the Test phase), fed back into another CodeReview/Test cycle.
      2. Unfinished code stubs in the CodeComplete phase, tracked by
         ChatDev's own internal 'num_tried' retry counter per file.
      3. Transient OpenAI API errors (rate limits, connection resets) that
         ChatDev's tenacity-based retry wrapper catches and retries.

Every claim below is labeled by how it was obtained, same discipline as the
rest of this package: a stated edge is marked as such, and a failure is only
ever called "unrecovered" when it is demonstrably the last occurrence of its
signature in the session, never because the top-level run "looks successful"
(ChatDev's own narrative has no failure state to contradict in the first
place -- that is exactly the blind spot being measured here).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

PHASE_HEADER_RE = re.compile(
    r"^\[(?P<ts>[0-9-]+ [0-9:]+) [A-Z]+\] (?:System: )?\*\*\[(?P<label>[^\]]+)\]\*\*"
)
TURN_RE = re.compile(
    r"^\[(?P<ts>[0-9-]+ [0-9:]+) [A-Z]+\] [^:]+: "
    r"\*\*(?P<from_role>[A-Za-z ]+?)<->(?P<to_role>[A-Za-z ]+?) on : (?P<phase>[A-Za-z]+), turn (?P<turn>[0-9]+)\*\*"
)
USAGE_HEADER_RE = re.compile(r"^\[(?P<ts>[0-9-]+ [0-9:]+) [A-Z]+\] \*\*\[OpenAI_Usage_Info Receive\]\*\*")
USAGE_FIELD_RE = re.compile(r"^(prompt_tokens|completion_tokens|total_tokens): (\d+)$")
PLACEHOLDERS_LINE_RE = re.compile(r"\*\*placeholders\*\*\s*\|\s*(?P<blob>.*)\|\s*$")
CYCLE_INDEX_RE = re.compile(r"'cycle_index':\s*(\d+)")
NUM_TRIED_BLOB_RE = re.compile(r"'num_tried':\s*defaultdict\([^,]*,\s*\{(?P<body>[^}]*)\}\)")
NUM_TRIED_PAIR_RE = re.compile(r"'([^']*)':\s*(\d+)")
WARNING_RETRY_RE = re.compile(
    r"^\[(?P<ts>[0-9-]+ [0-9:]+) WARNING\] (?P<detail>.*(?:Retrying|Rate limit reached).*)$"
)
TRACEBACK_START_RE = re.compile(r"^Traceback \(most recent call last\):")
LOG_LINE_START_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} ")
EXCEPTION_LINE_RE = re.compile(r"^(?P<exc>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)): ?(?P<msg>.*)$")
TASK_PROMPT_RE = re.compile(r"\*\*task_prompt\*\*:\s*(?P<task>.*)")
PROJECT_NAME_RE = re.compile(r"\*\*project_name\*\*:\s*(?P<name>.*)")


@dataclass
class ParsedLog:
    project_name: str | None = None
    task_prompt: str | None = None
    turns: list[dict] = field(default_factory=list)
    usage_by_phase: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0}))
    tracebacks: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    num_tried_max: dict[str, int] = field(default_factory=dict)
    phases_seen: list[str] = field(default_factory=list)


def parse_chatdev_log(path: Path) -> ParsedLog:
    """Single forward pass over the log, carrying just enough state (current
    phase, pending usage fields, an open traceback block) to attribute each
    event to its context -- no multi-line Python-literal parsing of the huge
    'placeholders' blob; it's handled with targeted regexes on that one line,
    since the log never actually re-serializes it across multiple physical
    lines."""
    parsed = ParsedLog()
    current_phase = "(preamble)"
    pending_usage_fields: dict[str, int] = {}
    in_traceback: list[str] = []
    lines = path.read_text(errors="replace").splitlines()

    for line in lines:
        if parsed.project_name is None:
            m = PROJECT_NAME_RE.search(line)
            if m:
                parsed.project_name = m.group("name").strip()
        if parsed.task_prompt is None:
            m = TASK_PROMPT_RE.search(line)
            if m:
                parsed.task_prompt = m.group("task").strip()

        if in_traceback:
            # This corpus logs a blank line after every physical line of a
            # traceback block (a \r\n artifact from Windows-run sessions), so
            # a blank line does NOT end the block -- only a genuine new
            # timestamped log line, the terminal exception line, or a safety
            # cap does.
            if LOG_LINE_START_RE.match(line):
                in_traceback = []
            elif line.strip() == "":
                continue
            else:
                in_traceback.append(line)
                m = EXCEPTION_LINE_RE.match(line.strip())
                if m:
                    parsed.tracebacks.append(
                        {"phase": current_phase, "exc": m.group("exc"), "message": m.group("msg").strip()[:200]}
                    )
                    in_traceback = []
                    continue
                if len(in_traceback) > 80:  # safety cap against a malformed/never-closing block
                    in_traceback = []
                else:
                    continue

        m = TURN_RE.match(line)
        if m:
            current_phase = m.group("phase")
            if current_phase not in parsed.phases_seen:
                parsed.phases_seen.append(current_phase)
            parsed.turns.append(
                {
                    "phase": current_phase,
                    "turn": int(m.group("turn")),
                    "from_role": m.group("from_role").strip(),
                    "to_role": m.group("to_role").strip(),
                    "ts": m.group("ts"),
                }
            )
            continue

        m = PHASE_HEADER_RE.match(line)
        if m and m.group("label") not in ("Seminar Conclusion", "Update Codes", "OpenAI_Usage_Info Receive"):
            pass  # phase context comes from TURN_RE; these headers are just markers

        if USAGE_HEADER_RE.match(line):
            pending_usage_fields = {}
            continue
        fm = USAGE_FIELD_RE.match(line.strip())
        if fm:
            key = {"prompt_tokens": "prompt", "completion_tokens": "completion", "total_tokens": "total"}[fm.group(1)]
            pending_usage_fields[key] = int(fm.group(2))
            if len(pending_usage_fields) == 3:
                bucket = parsed.usage_by_phase[current_phase]
                for k, v in pending_usage_fields.items():
                    bucket[k] += v
                pending_usage_fields = {}
            continue

        wm = WARNING_RETRY_RE.match(line)
        if wm:
            parsed.warnings.append({"phase": current_phase, "ts": wm.group("ts"), "detail": wm.group("detail")[:200]})
            continue

        if TRACEBACK_START_RE.match(line.strip()):
            in_traceback = [line]
            continue

        pm = PLACEHOLDERS_LINE_RE.search(line)
        if pm:
            blob = pm.group("blob")
            nt = NUM_TRIED_BLOB_RE.search(blob)
            if nt:
                for fname, count in NUM_TRIED_PAIR_RE.findall(nt.group("body")):
                    count = int(count)
                    if fname and count > parsed.num_tried_max.get(fname, 0):
                        parsed.num_tried_max[fname] = count

    return parsed


def analyze_chatdev_session(parsed: ParsedLog) -> dict:
    """Build the role graph (stated edges only) and the three-category
    failure list for one session's log."""
    roles: set[str] = set()
    edge_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for t in parsed.turns:
        roles.add(t["from_role"])
        roles.add(t["to_role"])
        edge_counts[(t["from_role"], t["to_role"], t["phase"])] += 1
    edges = [
        {"from": f, "to": tt, "phase": ph, "turns": n, "kind": "stated-dialogue"}
        for (f, tt, ph), n in edge_counts.items()
    ]

    by_role_turns: dict[str, int] = defaultdict(int)
    for t in parsed.turns:
        by_role_turns[t["from_role"]] += 1

    # Traceback signatures, in order of first appearance. The LAST signature
    # to occur in the file is the one that shipped unexamined -- ChatDev has
    # no top-level "this run failed" status to contradict, so "it reached the
    # end of the log" is not evidence the bug was fixed.
    seen_order: list[tuple[str, str]] = []
    sig_occurrences: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for tb in parsed.tracebacks:
        sig = (tb["exc"], tb["message"])
        if sig not in sig_occurrences:
            seen_order.append(sig)
        sig_occurrences[sig].append(tb)

    failures = []
    for i, sig in enumerate(seen_order):
        occ = sig_occurrences[sig]
        is_last = i == len(seen_order) - 1
        failures.append(
            {
                "category": "silent-failure" if is_last else "self-corrected",
                "kind": "runtime-traceback",
                "detail": f"{sig[0]}: {sig[1]}",
                "phase": occ[0]["phase"],
                "occurrences": len(occ),
            }
        )

    for fname, max_tries in parsed.num_tried_max.items():
        if max_tries > 0:
            failures.append(
                {
                    "category": "self-corrected",
                    "kind": "codecomplete-retry",
                    "detail": f"{fname}: left an unfinished stub on first attempt, retried {max_tries} time(s) before the run continued",
                    "phase": "CodeComplete",
                    "occurrences": max_tries,
                }
            )

    infra_retry_count = len(parsed.warnings)

    total_tokens = sum(v["total"] for v in parsed.usage_by_phase.values())

    return {
        "project_name": parsed.project_name,
        "roles": sorted(roles),
        "edges": edges,
        "phases_seen": parsed.phases_seen,
        "turns_by_role": dict(by_role_turns),
        "tokens_by_phase": {k: dict(v) for k, v in parsed.usage_by_phase.items()},
        "total_tokens": total_tokens,
        "failures": failures,
        "infra_retry_count": infra_retry_count,
        "silent_failure_count": sum(1 for f in failures if f["category"] == "silent-failure"),
        "self_corrected_count": sum(1 for f in failures if f["category"] == "self-corrected"),
    }


def render_chatdev_markdown(result: dict, log_path: Path) -> str:
    name = result["project_name"] or log_path.parent.name
    lines = [f"# Trace-based fit report: {name} (ChatDev)", ""]
    lines.append(
        f"**{len(result['roles'])} real roles**, {len(result['edges'])} stated dialogue edge(s) across "
        f"{len(result['phases_seen'])} phase(s), {result['total_tokens']} tokens total."
    )
    lines.append("")
    lines.append("## Role graph (ground truth -- each edge is a literal `<->` line the log states directly)")
    lines.append("")
    lines.append("```mermaid")
    lines.append("flowchart TD")
    safe = {r: re.sub(r"\W+", "_", r) for r in result["roles"]}
    for r in result["roles"]:
        lines.append(f'  {safe[r]}["{r}\\n{result["turns_by_role"].get(r, 0)} turn(s)"]')
    for e in result["edges"]:
        lines.append(f'  {safe[e["from"]]} -->|"{e["phase"]} x{e["turns"]}"| {safe[e["to"]]}')
    lines.append("```")
    lines.append("")
    lines.append("## Phases and token usage")
    lines.append("")
    lines.append("| Phase | Tokens (prompt/completion/total) |")
    lines.append("| --- | --- |")
    for phase in result["phases_seen"]:
        tok = result["tokens_by_phase"].get(phase, {"prompt": 0, "completion": 0, "total": 0})
        lines.append(f"| {phase} | {tok['prompt']}/{tok['completion']}/{tok['total']} |")
    lines.append("")
    lines.append("## Observed failure signatures")
    lines.append("")
    lines.append(
        "ChatDev has no top-level \"run failed\" status to check -- a run either produces a WareHouse/ "
        "folder or it doesn't, regardless of what happened inside. Everything below comes from inspecting "
        "individual test-cycle and retry events, the same discipline the OTel-based adapters use."
    )
    lines.append("")
    silent = [f for f in result["failures"] if f["category"] == "silent-failure"]
    corrected = [f for f in result["failures"] if f["category"] == "self-corrected"]
    lines.append(f"### Silent failures (unrecovered) -- {len(silent)}")
    lines.append("")
    lines.append("The last occurrence of this signature in the log -- nothing downstream shows it was ever fixed.")
    lines.append("")
    if not silent:
        lines.append("None in this run.")
    else:
        lines.append("| Phase | Kind | Detail | Times seen |")
        lines.append("| --- | --- | --- | --- |")
        for f in silent:
            lines.append(f"| {f['phase']} | {f['kind']} | {f['detail']} | {f['occurrences']} |")
    lines.append("")
    lines.append(f"### Self-corrected (recovered) -- {len(corrected)}")
    lines.append("")
    if not corrected:
        lines.append("None in this run.")
    else:
        lines.append("| Phase | Kind | Detail | Times seen |")
        lines.append("| --- | --- | --- | --- |")
        for f in corrected:
            lines.append(f"| {f['phase']} | {f['kind']} | {f['detail']} | {f['occurrences']} |")
    lines.append("")
    lines.append(f"### Transient API retries (infra-level, not counted above) -- {result['infra_retry_count']}")
    lines.append("")
    lines.append(
        "Rate limits / connection resets caught and retried by ChatDev's own `tenacity` wrapper. These are "
        "genuinely logged (at WARNING level), unlike the two categories above -- listed separately so they "
        "aren't conflated with failures the app's own code swallowed silently."
    )
    return "\n".join(lines)


def analyze_warehouse(warehouse_dir: Path, out_dir: Path | None = None) -> dict:
    """Batch over every WareHouse/<project> folder: parse its one .log file
    (sessions with none are counted and skipped, not silently dropped),
    write a per-session report, and aggregate the failure counts."""
    warehouse_dir = Path(warehouse_dir)
    out_dir = Path(out_dir) if out_dir else warehouse_dir.parent / "reports" / "chatdev"
    out_dir.mkdir(parents=True, exist_ok=True)

    session_dirs = sorted(p for p in warehouse_dir.iterdir() if p.is_dir())
    sessions_with_logs = 0
    sessions_without_logs = []
    per_session_results = []

    for sdir in session_dirs:
        logs = list(sdir.glob("*.log"))
        if not logs:
            sessions_without_logs.append(sdir.name)
            continue
        log_path = logs[0]
        parsed = parse_chatdev_log(log_path)
        if not parsed.turns:
            sessions_without_logs.append(sdir.name)
            continue
        sessions_with_logs += 1
        result = analyze_chatdev_session(parsed)
        report = render_chatdev_markdown(result, log_path)
        report_path = out_dir / f"{sdir.name}.md"
        report_path.write_text(report)
        per_session_results.append({"session": sdir.name, "report_path": str(report_path), **result})

    total_silent = sum(r["silent_failure_count"] for r in per_session_results)
    total_corrected = sum(r["self_corrected_count"] for r in per_session_results)
    total_infra_events = sum(r["infra_retry_count"] for r in per_session_results)
    sessions_with_infra_retries = sum(1 for r in per_session_results if r["infra_retry_count"] > 0)

    return {
        "warehouse_dir": str(warehouse_dir),
        "sessions_total": len(session_dirs),
        "sessions_with_logs": sessions_with_logs,
        "sessions_without_logs": sessions_without_logs,
        "total_silent_failures": total_silent,
        "total_self_corrected": total_corrected,
        # Deliberately NOT folded into total_findings: a single throttling episode can
        # produce hundreds of individual retry-warning lines (one session here had 833),
        # which would make "total findings" mostly noise from one session's rate limit,
        # not 833 distinct behavioral problems. Reported as its own bucket instead.
        "total_infra_retry_events": total_infra_events,
        "sessions_with_infra_retries": sessions_with_infra_retries,
        "total_findings": total_silent + total_corrected,
        "per_session": per_session_results,
        "out_dir": str(out_dir),
    }


def render_warehouse_summary(agg: dict) -> str:
    lines = [f"# ChatDev WareHouse recon summary", ""]
    lines.append(
        f"**{agg['sessions_with_logs']} of {agg['sessions_total']} sessions** had a parseable `.log` "
        f"({len(agg['sessions_without_logs'])} had none or an empty one -- listed at the bottom, not silently dropped)."
    )
    lines.append("")
    lines.append(
        f"**{agg['total_findings']} behavioral findings**: {agg['total_silent_failures']} silent (unrecovered) failures, "
        f"{agg['total_self_corrected']} self-corrected failures."
    )
    lines.append("")
    lines.append(
        f"Separately: **{agg['sessions_with_infra_retries']} session(s)** hit transient OpenAI API throttling "
        f"({agg['total_infra_retry_events']} retry-warning lines total, concentrated in a handful of sessions -- "
        "not counted in \"behavioral findings\" above since these were visibly logged and retried, not silent)."
    )
    lines.append("")
    lines.append("## Per-session breakdown")
    lines.append("")
    lines.append("| Session | Roles | Silent | Self-corrected | Infra retries | Tokens | Report |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted(agg["per_session"], key=lambda r: -(r["silent_failure_count"] + r["self_corrected_count"])):
        rel = Path(r["report_path"]).name
        lines.append(
            f"| {r['session']} | {len(r['roles'])} | {r['silent_failure_count']} | {r['self_corrected_count']} | "
            f"{r['infra_retry_count']} | {r['total_tokens']} | [{rel}]({rel}) |"
        )
    lines.append("")
    if agg["sessions_without_logs"]:
        lines.append("## Sessions with no parseable log")
        lines.append("")
        for name in agg["sessions_without_logs"]:
            lines.append(f"- {name}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a ChatDev WareHouse/ directory or a single ChatDev .log file.")
    parser.add_argument("path", help="Path to a WareHouse/ directory, or a single .log file")
    parser.add_argument("--out", default=None, help="Output directory (default: <parent-of-input>/reports/chatdev)")
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_dir():
        agg = analyze_warehouse(path, out_dir=Path(args.out) if args.out else None)
        summary = render_warehouse_summary(agg)
        summary_path = Path(agg["out_dir"]) / "SUMMARY.md"
        summary_path.write_text(summary)
        print(f"sessions: {agg['sessions_with_logs']}/{agg['sessions_total']} parsed")
        print(f"behavioral findings: {agg['total_findings']} "
              f"(silent: {agg['total_silent_failures']}, self-corrected: {agg['total_self_corrected']})")
        print(f"infra retries (separate): {agg['sessions_with_infra_retries']} session(s), "
              f"{agg['total_infra_retry_events']} event(s)")
        print(f"summary written to {summary_path}")
    else:
        parsed = parse_chatdev_log(path)
        result = analyze_chatdev_session(parsed)
        report = render_chatdev_markdown(result, path)
        out_path = Path(args.out) if args.out else path.parent / "reports" / f"{path.stem}.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        print(f"roles found: {len(result['roles'])}")
        print(f"silent (unrecovered) failures: {result['silent_failure_count']}")
        print(f"self-corrected failures: {result['self_corrected_count']}")
        print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
