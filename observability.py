"""
Observability layer (eval Part A).

Wraps ANY LLMClient and records, per call: which operation ran, the model,
input/output tokens, estimated cost, latency, and whether it errored. Traces
are written as JSONL so you can analyse a run afterwards.

Usage - wrap your existing client, change nothing else:

    from observability import InstrumentedClient, RunTracker

    tracker = RunTracker()
    client = InstrumentedClient(OllamaClient(), tracker)
    ...run the pipeline as usual...
    tracker.report()          # printed summary
    tracker.save("traces.jsonl")

This works because InstrumentedClient satisfies the same LLMClient interface as
the real backends - the same swappable-backend pattern used everywhere else in
this project.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from llm_base import LLMClient

# Estimated USD per MILLION tokens (input, output).
# VERIFY against current pricing before quoting these anywhere - model prices
# change, and this table is a convenience, not a source of truth.
PRICING = {
    "claude-haiku-4-5":  (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-8":   (15.00, 75.00),
}
LOCAL_FREE = 0.0  # anything served by Ollama costs nothing


def estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    for name, (pin, pout) in PRICING.items():
        if model and model.startswith(name):
            return (in_tok / 1_000_000) * pin + (out_tok / 1_000_000) * pout
    return LOCAL_FREE


@dataclass
class CallRecord:
    ts: str
    operation: str          # "analysis" | "draft" | "other"
    model: str
    backend: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    ok: bool
    stop_reason: Optional[str] = None
    error: Optional[str] = None


@dataclass
class RunTracker:
    """Collects CallRecords for one run and summarises them."""
    records: List[CallRecord] = field(default_factory=list)

    def add(self, rec: CallRecord):
        self.records.append(rec)

    # ---------------------------------------------------------- aggregates
    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.records)

    def by_operation(self) -> dict:
        out = {}
        for r in self.records:
            b = out.setdefault(r.operation, {"calls": 0, "tokens": 0,
                                             "cost": 0.0, "latency": 0.0,
                                             "errors": 0})
            b["calls"] += 1
            b["tokens"] += r.input_tokens + r.output_tokens
            b["cost"] += r.cost_usd
            b["latency"] += r.latency_s
            b["errors"] += 0 if r.ok else 1
        return out

    def report(self) -> str:
        if not self.records:
            return "No LLM calls recorded."
        lines = ["", "=== RUN SUMMARY ===",
                 f"calls        : {len(self.records)}",
                 f"tokens       : {self.total_tokens:,}",
                 f"est. cost    : ${self.total_cost:.4f}",
                 f"total latency: {sum(r.latency_s for r in self.records):.1f}s",
                 f"errors       : {sum(1 for r in self.records if not r.ok)}",
                 "", "by operation:"]
        for op, b in sorted(self.by_operation().items()):
            avg = b["latency"] / b["calls"] if b["calls"] else 0
            lines.append(
                f"  {op:9} calls={b['calls']:3} tokens={b['tokens']:7,} "
                f"cost=${b['cost']:.4f} avg={avg:5.1f}s errors={b['errors']}")
        text = "\n".join(lines)
        print(text)
        return text

    def save(self, path: str = "traces.jsonl"):
        """Append records as JSON Lines - one object per call."""
        with open(path, "a", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r)) + "\n")
        return path


class InstrumentedClient(LLMClient):
    """Transparent wrapper: same interface, records every call."""

    def __init__(self, inner: LLMClient, tracker: RunTracker):
        self.inner = inner
        self.tracker = tracker
        self.name = f"{inner.name}+traced"

    @property
    def model(self):
        return getattr(self.inner, "model", "?")

    def _call(self, fn, operation, **kwargs):
        start = time.perf_counter()
        ok, err, result = True, None, None
        try:
            result = fn(**kwargs)
            return result
        except Exception as e:
            ok, err = False, f"{type(e).__name__}: {e}"
            raise
        finally:
            latency = time.perf_counter() - start
            u = getattr(self.inner, "last_usage", None) or {}
            in_tok = u.get("input_tokens", 0)
            out_tok = u.get("output_tokens", 0)
            self.tracker.add(CallRecord(
                ts=datetime.now(timezone.utc).isoformat(),
                operation=operation,
                model=self.model,
                backend=getattr(self.inner, "name", "?"),
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=estimate_cost(self.model, in_tok, out_tok),
                latency_s=round(latency, 3),
                ok=ok,
                stop_reason=u.get("stop_reason"),
                error=err,
            ))

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        return self._call(self.inner.complete, "draft",
                          system=system, user=user, max_tokens=max_tokens)

    def complete_json(self, system: str, user: str, max_tokens: int) -> dict:
        return self._call(self.inner.complete_json, "analysis",
                          system=system, user=user, max_tokens=max_tokens)