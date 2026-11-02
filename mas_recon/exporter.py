"""The one piece every framework adapter shares: write finished OTel spans to a
local JSONL file. No collector, no OTLP network hop, no account/API key of ours
needed -- the trace never leaves the machine it's captured on unless the person
running it chooses to send the file somewhere."""

from __future__ import annotations

import json
from pathlib import Path

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class JSONLSpanExporter(SpanExporter):
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._file = open(path, "a", encoding="utf-8")  # noqa: SIM115

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            ctx = span.get_span_context()
            parent = span.parent
            record = {
                "name": span.name,
                "trace_id": format(ctx.trace_id, "032x"),
                "span_id": format(ctx.span_id, "016x"),
                "parent_span_id": format(parent.span_id, "016x") if parent else None,
                "start_time_ns": span.start_time,
                "end_time_ns": span.end_time,
                "duration_ms": (span.end_time - span.start_time) / 1e6 if span.end_time and span.start_time else None,
                "status": span.status.status_code.name if span.status else None,
                "attributes": dict(span.attributes or {}),
                "events": [
                    {"name": e.name, "attributes": dict(e.attributes or {}), "timestamp_ns": e.timestamp}
                    for e in span.events
                ],
            }
            self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._file.close()

    @property
    def path(self) -> Path:
        return self._path
