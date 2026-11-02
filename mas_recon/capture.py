"""One import, two lines, no reverse-engineering: start_capture() detects which
supported framework is installed in THIS environment and wires the matching
OpenTelemetry instrumentation automatically, so the person running their own
agent never has to read this package's source or know what OpenTelemetry is.

Usage:
    from mas_recon import start_capture

    handle = start_capture()     # auto-detects framework, starts tracing
    ... run your agent normally, exactly as you already do ...
    handle.stop()                # flushes the trace and prints a report

Supported today:
    - AWS Strands Agents (native OTel -- no extra instrumentation needed)
    - Anything built on langchain_core (LangChain, LangGraph, many deepagents
      apps) via the OpenInference LangChain instrumentor
    - CrewAI, via the OpenInference CrewAI instrumentor
    - AutoGen / AG2 (autogen-agentchat), via the OpenInference AutoGen instrumentor

Anything else: start_capture() tells you plainly that it doesn't recognize
your framework yet, rather than silently capturing nothing.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, field
from pathlib import Path

from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from .exporter import JSONLSpanExporter

# name -> (module to detect, pip extra that installs its instrumentor)
SUPPORTED_FRAMEWORKS = ("strands", "langchain", "crewai", "autogen")

# Detection order matters: an app could have more than one of these installed
# (e.g. a LangChain tool called from a Strands agent, or CrewAI pulling in
# langchain as a transitive dependency of some tool). Each framework's own
# orchestration layer is checked for first, most-specific before most-generic.
_DETECT_MODULES = {
    "strands": "strands",
    "crewai": "crewai",
    "autogen": "autogen_agentchat",  # AG2 / modern AutoGen's importable name
    "langchain": "langchain_core",
}

# name -> (openinference module, class name, pip extra)
_OPENINFERENCE_INSTRUMENTORS = {
    "langchain": ("openinference.instrumentation.langchain", "LangChainInstrumentor", "langchain"),
    "crewai": ("openinference.instrumentation.crewai", "CrewAIInstrumentor", "crewai"),
    "autogen": ("openinference.instrumentation.autogen_agentchat", "AutogenAgentChatInstrumentor", "autogen"),
}


def _installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def detect_framework() -> str | None:
    """Returns one of SUPPORTED_FRAMEWORKS, or None if nothing recognized is
    installed. Order: strands, crewai, autogen, langchain -- most-specific
    orchestration layer first, since an app can have more than one of these
    installed at once (e.g. a LangChain tool called from a Strands agent)."""
    for name in ("strands", "crewai", "autogen", "langchain"):
        if _installed(_DETECT_MODULES[name]):
            return name
    return None


class UnsupportedFrameworkError(RuntimeError):
    pass


@dataclass
class CaptureHandle:
    exporter: JSONLSpanExporter
    framework: str
    _stopped: bool = field(default=False, init=False)

    @property
    def trace_path(self) -> Path:
        return self.exporter.path

    def stop(self, *, analyze: bool = True) -> Path:
        """Flush the trace to disk. By default also runs the analyzer
        immediately and prints a short summary + the full report's path --
        the point is nobody has to know a second command exists."""
        if self._stopped:
            return self.trace_path
        self._stopped = True
        self.exporter.shutdown()
        print(f"[mas-recon] trace captured: {self.trace_path}")

        if analyze:
            from .analyzer import run_and_print

            run_and_print(self.trace_path)

        return self.trace_path


def _start_openinference(framework: str, exporter: JSONLSpanExporter, service_name: str) -> None:
    module_name, class_name, pip_extra = _OPENINFERENCE_INSTRUMENTORS[framework]
    try:
        module = __import__(module_name, fromlist=[class_name])
        instrumentor_cls = getattr(module, class_name)
    except ImportError as exc:
        raise UnsupportedFrameworkError(
            f"Detected a {framework} app, but its instrumentation package isn't "
            f"installed. Run: pip install 'mas-recon[{pip_extra}]'"
        ) from exc
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace_api.set_tracer_provider(provider)
    instrumentor_cls().instrument()


def start_capture(
    output_path: str | Path | None = None,
    *,
    service_name: str = "agent-app",
    framework: str | None = None,
) -> CaptureHandle:
    """Start capturing OpenTelemetry spans from the detected agent framework.

    Call this BEFORE you build/run your agent -- instrumentation has to be in
    place before the first model or graph call happens. Everything after this
    call runs exactly as it already does; this only adds an observer.
    """
    framework = framework or detect_framework()
    if framework is None:
        raise UnsupportedFrameworkError(
            "mas-recon doesn't recognize an installed agent framework "
            f"in this environment (checked: {', '.join(SUPPORTED_FRAMEWORKS)}). "
            "If you're using something else, this version can't auto-instrument "
            "it yet -- tell us what you're using and we'll add it, or point us "
            "at an existing trace export if you already have one (Langfuse, "
            "LangSmith, Datadog, any OTel collector)."
        )

    trace_path = Path(output_path) if output_path else Path.cwd() / "mas-recon" / f"{int(time.time())}-{framework}.jsonl"
    exporter = JSONLSpanExporter(trace_path)

    if framework == "strands":
        from strands.telemetry.config import StrandsTelemetry

        # StrandsTelemetry() with no args builds its own TracerProvider (with
        # Strands' own resource attributes and W3C propagators set up) and
        # sets it as the global provider -- then we attach our exporter to
        # that same provider. This is the exact pattern already proven against
        # a real multi-agent run; a hand-built provider wasn't re-tested here.
        telemetry = StrandsTelemetry()
        telemetry.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    elif framework in _OPENINFERENCE_INSTRUMENTORS:
        _start_openinference(framework, exporter, service_name)
    else:  # pragma: no cover - detect_framework() only returns known values
        raise UnsupportedFrameworkError(f"Unknown framework: {framework}")

    print(f"[mas-recon] capturing ({framework} detected) -> {trace_path}")
    return CaptureHandle(exporter=exporter, framework=framework)
