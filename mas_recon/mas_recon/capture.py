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
    - AWS Strands Agents (native OTel -- no extra instrumentation needed),
      including agents deployed on AWS Bedrock AgentCore: AgentCore is a
      deployment/runtime wrapper, not an agent-building framework, so if the
      underlying agent code is Strands (the common pairing), it's already
      covered -- start_capture() also reuses AgentCore's own tracer provider
      if it set one up before this ran, instead of silently creating an
      unused second one
    - The Claude Agent SDK (claude_agent_sdk), including subagent delegation,
      via the OpenInference Claude Agent SDK + Anthropic instrumentors together
    - Anything built on langchain_core (LangChain, LangGraph, and deepagents --
      deepagents apps ARE LangGraph apps underneath, confirmed by an actual
      run, not assumed) via the OpenInference LangChain instrumentor
    - CrewAI, via the OpenInference CrewAI instrumentor
    - AutoGen / AG2 (autogen-agentchat), via the OpenInference AutoGen instrumentor

Anything else -- including a custom, hand-rolled agent builder with none of
the above underneath -- start_capture() tells you plainly that it doesn't
recognize your framework yet, rather than silently capturing nothing.
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
SUPPORTED_FRAMEWORKS = ("strands", "claude_agent_sdk", "langchain", "crewai", "autogen")

# Detection order matters: an app could have more than one of these installed
# (e.g. a LangChain tool called from a Strands agent, or CrewAI pulling in
# langchain as a transitive dependency of some tool). Each framework's own
# orchestration layer is checked for first, most-specific before most-generic.
_DETECT_MODULES = {
    "strands": "strands",
    "claude_agent_sdk": "claude_agent_sdk",
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
    installed. Order: strands, claude_agent_sdk, crewai, autogen, langchain --
    most-specific orchestration layer first, since an app can have more than
    one of these installed at once (e.g. a LangChain tool called from a
    Strands agent).

    Note on AWS Bedrock AgentCore: it's a deployment/runtime wrapper, not an
    agent-building framework in its own right -- the actual agent logic
    underneath is almost always built with one of the frameworks above
    (Strands, in every real example seen so far). There's deliberately no
    separate "agentcore" entry here: if the underlying framework is Strands,
    detection already finds it regardless of the AgentCore wrapper around it.
    bedrock_agentcore is only checked below, for a clearer error message when
    NEITHER a known framework nor AgentCore is found."""
    for name in ("strands", "claude_agent_sdk", "crewai", "autogen", "langchain"):
        if _installed(_DETECT_MODULES[name]):
            return name
    return None


def _get_or_create_provider(service_name: str) -> tuple[TracerProvider, bool]:
    """Reuse an already-configured global TracerProvider if the app -- or its
    runtime, e.g. AWS Bedrock AgentCore, which can wire up its own tracing
    before user code runs -- already set one up before start_capture() was
    called. Calling set_tracer_provider() a second time is a silent no-op in
    the OTel SDK: the second provider would just sit there, unused, with our
    exporter attached to a provider nobody's actually sending spans through.
    Returns (provider, created_new)."""
    existing = trace_api.get_tracer_provider()
    if isinstance(existing, TracerProvider):
        return existing, False
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    trace_api.set_tracer_provider(provider)
    return provider, True


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
    provider, _ = _get_or_create_provider(service_name)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor_cls().instrument(tracer_provider=provider)


def _start_claude_agent_sdk(exporter: JSONLSpanExporter, service_name: str) -> None:
    """Needs TWO instrumentors on the same provider, not one: the SDK instrumentor
    gives AGENT/TOOL spans (including subagent delegation as a nested
    ClaudeAgentSDK.Task span), but emits no LLM spans itself -- the Anthropic
    instrumentor supplies those (model, tokens) by patching the same underlying
    client calls. Neither alone gives the full picture."""
    try:
        from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
    except ImportError as exc:
        raise UnsupportedFrameworkError(
            "Detected a Claude Agent SDK app, but its instrumentation packages aren't "
            "installed. Run: pip install 'mas-recon[claude]'"
        ) from exc
    provider, _ = _get_or_create_provider(service_name)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    ClaudeAgentSDKInstrumentor().instrument(tracer_provider=provider)
    AnthropicInstrumentor().instrument(tracer_provider=provider)


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
        on_agentcore = _installed("bedrock_agentcore")
        agentcore_note = (
            " AWS Bedrock AgentCore was detected, but AgentCore is a deployment "
            "wrapper, not an agent framework -- none of the actual agent-building "
            "frameworks this tool knows (Strands, in particular) were found "
            "underneath it. If the agent logic isn't Strands-based, that's the "
            "real gap, not a detection bug."
            if on_agentcore
            else ""
        )
        raise UnsupportedFrameworkError(
            "mas-recon doesn't recognize an installed agent framework "
            f"in this environment (checked: {', '.join(SUPPORTED_FRAMEWORKS)})."
            f"{agentcore_note} If you're using something else, this version "
            "can't auto-instrument it yet -- tell us what you're using and "
            "we'll add it, or point us at an existing trace export if you "
            "already have one (Langfuse, LangSmith, Datadog, any OTel collector)."
        )

    trace_path = Path(output_path) if output_path else Path.cwd() / "mas-recon" / f"{int(time.time())}-{framework}.jsonl"
    exporter = JSONLSpanExporter(trace_path)

    if framework == "strands":
        from strands.telemetry.config import StrandsTelemetry

        # StrandsTelemetry() with no args builds its own TracerProvider (with
        # Strands' own resource attributes and W3C propagators set up) and
        # sets it as the global provider -- the pattern already proven against
        # a real multi-agent run. But if something already set a global
        # provider before we got here -- e.g. AWS Bedrock AgentCore's own
        # runtime, which can wire up tracing before user code runs -- handing
        # StrandsTelemetry that SAME provider instead avoids the silent
        # second-provider problem _get_or_create_provider() exists to prevent.
        existing = trace_api.get_tracer_provider()
        if isinstance(existing, TracerProvider):
            telemetry = StrandsTelemetry(tracer_provider=existing)
        else:
            telemetry = StrandsTelemetry()
        telemetry.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    elif framework == "claude_agent_sdk":
        _start_claude_agent_sdk(exporter, service_name)
    elif framework in _OPENINFERENCE_INSTRUMENTORS:
        _start_openinference(framework, exporter, service_name)
    else:  # pragma: no cover - detect_framework() only returns known values
        raise UnsupportedFrameworkError(f"Unknown framework: {framework}")

    print(f"[mas-recon] capturing ({framework} detected) -> {trace_path}")
    return CaptureHandle(exporter=exporter, framework=framework)
