"""OpenTelemetry for the agent: where a card's time goes, span by span.

Semantic Kernel already instruments itself with the GenAI semantic conventions, so most of
the work is turning that on and deciding where spans go:

    f1.card                        this module's root span, one per incident
      invoke_agent steward         SK: the whole agent run
        chat qwen3:4b-instruct     SK: one per model call, i.e. per tool round
        execute_tool search_rules  SK: one per tool call
          retrieve rule            ours: the filtered, fused search inside the tool

SK reads its switch when it is first imported, so this module has to be imported before
anything from `semantic_kernel`. `agent/__init__.py` does that.

Exporters: an in-memory one for the evaluation to read back, and a JSON-lines file under
`data/traces/` for everything else. Prompts and answers go on spans too (the "sensitive"
flag): this is a local, single-user system and the prompts are public regulations.
"""

from __future__ import annotations

import os

os.environ.setdefault("SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS", "true")
os.environ.setdefault("SEMANTICKERNEL_EXPERIMENTAL_GENAI_ENABLE_OTEL_DIAGNOSTICS_SENSITIVE", "true")

from collections import defaultdict  # noqa: E402
from datetime import datetime  # noqa: E402

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import (  # noqa: E402
    BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

import config  # noqa: E402

TRACES = config.DATA / "traces"

_memory: InMemorySpanExporter | None = None


def setup(to_file: bool = True) -> InMemorySpanExporter:
    """Install the tracer provider once. Returns the in-memory exporter."""
    global _memory
    if _memory is not None:
        return _memory

    provider = TracerProvider(resource=Resource.create({"service.name": "f1-race-control-agent"}))
    _memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(_memory))
    if to_file:
        TRACES.mkdir(parents=True, exist_ok=True)
        out = open(TRACES / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl", "a", encoding="utf-8")
        exporter = ConsoleSpanExporter(out=out, formatter=lambda s: s.to_json(indent=None) + "\n")
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return _memory


def tracer() -> trace.Tracer:
    return trace.get_tracer("f1.agent")


def breakdown(spans: list[ReadableSpan], trace_id: int) -> dict[str, float]:
    """Milliseconds per kind of span within one card's trace.

    `chat` is model time, `tool` is SK's tool execution (retrieval included), `card` is the
    whole thing. Model time + tool time falls short of the card by SK's own overhead.
    """
    ms: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    for s in spans:
        if s.context.trace_id != trace_id or s.end_time is None:
            continue
        kind = ("card" if s.name == "f1.card" else
                "chat" if s.name.startswith("chat ") else
                "tool" if s.name.startswith("execute_tool") else None)
        if kind:
            ms[kind] += (s.end_time - s.start_time) / 1e6
            count[kind] += 1
    return {**{f"{k}_ms": round(v, 1) for k, v in ms.items()},
            **{f"{k}_n": v for k, v in count.items()}}
