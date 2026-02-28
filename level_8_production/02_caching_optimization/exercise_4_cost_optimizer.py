"""





























































```python exercise_4_alerting_system.py  # Alerting system exercisepython exercise_3_agent_tracer.py     # Agent tracer exercisepython exercise_2_cost_dashboard.py   # Cost dashboard exercisepython exercise_1_request_logger.py   # Request logger exercisepython main.py              # Interactive demo menu```bash## Running- Tracing for complex multi-agent diagnostic pipelines- Latency monitoring for time-sensitive clinical workflows- Cost tracking per department/provider for hospital budgeting- Audit trails for every AI-assisted clinical decision## Healthcare Relevance4. `exercise_4_alerting_system.py` — Multi-signal alerting with thresholds3. `exercise_3_agent_tracer.py` — Agent execution tracer with tree visualization2. `exercise_2_cost_dashboard.py` — Text-based cost dashboard with budget alerts1. `exercise_1_request_logger.py` — Comprehensive JSONL request logger with querying## Exercises4. **Trace Viewer** — Visualize agent execution as a timed step tree3. **Latency Monitoring** — Track p50/p95/p99 with anomaly detection2. **Cost Tracking** — Track and aggregate costs per model/endpoint/user1. **Request Logging** — Log every LLM request with full metadata to JSON## Demos in main.pyit easy to identify bottlenecks and failures.A trace viewer shows the complete execution path with timing at each step, makingAgent workflows involve chains of LLM calls, tool invocations, and retrieval steps.### Distributed Tracingexperience.using statistical methods. Alert when latency exceeds thresholds that impact userTrack percentile latencies (p50, p95, p99) rather than averages. Detect anomalies### Latency Monitoringusage patterns.Set budget alerts to prevent runaway spending. Project monthly costs from dailyTrack costs at multiple levels: per-model, per-endpoint, per-user, and per-day.### Cost Trackingeasy querying and analysis.tokens, total cost, latency, and the request/response content. JSONL format enablesEvery LLM call should be logged with: timestamp, model, prompt tokens, completion### Request Logging## Key Concepts- **Compliance**: Healthcare requires audit trails of every AI-generated response- **Debugging**: Agent workflows involve multiple LLM calls; tracing is essential- **Latency Tracking**: Users abandon slow responses; p95/p99 latency matters- **Cost Control**: LLM API calls cost real money; unmonitored usage can spike bills## Why LLM Monitoring Mattersinfrastructure from scratch — no external dependencies needed.reliability, control costs, and debug issues. This project builds monitoringProduction LLM applications require comprehensive observability to ensure## OverviewExercise 4 — Complete Cost Optimization System
===============================================
Combine caching + model routing + batch processing into one unified pipeline
and measure end-to-end savings on a set of 50 medical queries.

Pipeline:
    1. Semantic cache check (skip LLM if similar query was answered)
    2. Complexity-based model routing (cheap model for simple queries)
    3. Batch embedding calls where possible

Compare:
    A) No optimization   — every query goes to gpt-4o
    B) Full optimization — semantic cache + routing

Report: total tokens, total cost, cache hit rate, money saved.

Usage:
    python exercise_4_cost_optimizer.py
"""

import os
import math
import time
import hashlib
import random
from dataclasses import dataclass, field
from typing import Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  pip install openai")


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
PRICING = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o":      {"input": 0.0025,  "output": 0.01},
    "o1-mini":     {"input": 0.003,   "output": 0.012},
}


def estimate_cost(model: str, prompt_tok: int, completion_tok: int) -> float:
    p = PRICING.get(model, PRICING["gpt-4o-mini"])
    return (prompt_tok / 1000 * p["input"]) + (completion_tok / 1000 * p["output"])


# ---------------------------------------------------------------------------
# Semantic cache (reused from exercise 1)
# ---------------------------------------------------------------------------
def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class SemanticCache:
    def __init__(self, threshold: float = 0.92):
        self.threshold = threshold
        self._entries: list[tuple[list[float], str, str]] = []
        self.hits = 0
        self.misses = 0

    def lookup(self, emb: list[float]) -> Optional[str]:
        best, best_reply = 0.0, None
        for e, _, r in self._entries:
            s = _cosine(emb, e)
            if s > best:
                best, best_reply = s, r
        if best >= self.threshold and best_reply:
            self.hits += 1
            return best_reply
        self.misses += 1
        return None

    def store(self, emb: list[float], query: str, reply: str):
        self._entries.append((emb, query, reply))

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total else 0.0


# ---------------------------------------------------------------------------
# Model router (simplified from exercise 3)
# ---------------------------------------------------------------------------
COMPLEX_TERMS = {
    "pathophysiology", "pharmacokinetics", "differential diagnosis",
    "contraindication", "comorbidity", "drug interaction",
    "mechanism of action", "clinical staging", "evidence-based",
}


def route_model(query: str) -> str:
    q_lower = query.lower()
    score = 0.0
    for term in COMPLEX_TERMS:
        if term in q_lower:
            score += 1.5
    if len(query.split()) > 40:
        score += 1.5
    if any(kw in q_lower for kw in ["step by step", "reason through", "compare and contrast"]):
        score += 2.0

    if score >= 4.0:
        return "gpt-4o"  # cap at gpt-4o for cost demo
    elif score >= 1.5:
        return "gpt-4o"
    return "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Optimization pipeline
# ---------------------------------------------------------------------------
@dataclass
class QueryResult:
    query: str
    reply: str
    model: str
    tokens: int
    cost: float
    source: str  # "cache" | "api"
    latency_ms: float


@dataclass
class PipelineStats:
    results: list[QueryResult] = field(default_factory=list)
    label: str = ""

    @property
    def total_queries(self): return len(self.results)
    @property
    def total_tokens(self): return sum(r.tokens for r in self.results)
    @property
    def total_cost(self): return sum(r.cost for r in self.results)
    @property
    def cache_hits(self): return sum(1 for r in self.results if r.source == "cache")
    @property
    def avg_latency(self):
        lats = [r.latency_ms for r in self.results]
        return sum(lats) / len(lats) if lats else 0


def run_no_optimization(client: "OpenAI", queries: list[str]) -> PipelineStats:
    """Baseline: send every query to gpt-4o, no cache."""
    stats = PipelineStats(label="No Optimization (all gpt-4o)")
    for q in queries:
        start = time.perf_counter()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a medical assistant."},
                {"role": "user", "content": q},
            ],
            max_tokens=200,
        )
        elapsed = (time.perf_counter() - start) * 1000
        usage = resp.usage
        cost = estimate_cost("gpt-4o", usage.prompt_tokens, usage.completion_tokens)
        stats.results.append(QueryResult(
            query=q[:50], reply=resp.choices[0].message.content[:80],
            model="gpt-4o", tokens=usage.total_tokens, cost=cost,
            source="api", latency_ms=round(elapsed, 1),
        ))
    return stats


def run_optimized(client: "OpenAI", queries: list[str]) -> PipelineStats:
    """Optimized: semantic cache + model routing."""
    stats = PipelineStats(label="Full Optimization (cache + routing)")
    cache = SemanticCache(threshold=0.92)

    for q in queries:
        start = time.perf_counter()

        # Step 1: embed the query
        emb = client.embeddings.create(model="text-embedding-3-small", input=[q]).data[0].embedding

        # Step 2: check semantic cache
        cached = cache.lookup(emb)
        if cached:
            elapsed = (time.perf_counter() - start) * 1000
            stats.results.append(QueryResult(
                query=q[:50], reply=cached[:80],
                model="cache", tokens=0, cost=0.0,
                source="cache", latency_ms=round(elapsed, 1),
            ))
            continue

        # Step 3: route to appropriate model
        model = route_model(q)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a medical assistant."},
                {"role": "user", "content": q},
            ],
            max_tokens=200,
        )
        elapsed = (time.perf_counter() - start) * 1000
        usage = resp.usage
        cost = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
        reply = resp.choices[0].message.content

        cache.store(emb, q, reply)

        stats.results.append(QueryResult(
            query=q[:50], reply=reply[:80],
            model=model, tokens=usage.total_tokens, cost=cost,
            source="api", latency_ms=round(elapsed, 1),
        ))

    return stats


# ---------------------------------------------------------------------------
# Query generator
# ---------------------------------------------------------------------------
def generate_medical_queries(n: int = 50) -> list[str]:
    """Generate a realistic mix of medical queries with duplicates and paraphrases."""
    base_queries = [
        "What are the symptoms of Type 2 diabetes?",
        "How is hypertension treated?",
        "What are common side effects of metformin?",
        "Normal blood pressure range?",
        "What is the recommended dosage of ibuprofen for adults?",
        "Explain the difference between Type 1 and Type 2 diabetes.",
        "What causes high cholesterol?",
        "How often should adults get a flu vaccine?",
        "What are the warning signs of a stroke?",
        "Is aspirin safe to take daily?",
        "What is the normal resting heart rate?",
        "How is asthma diagnosed in children?",
        "What are symptoms of iron deficiency anemia?",
        "Explain the pathophysiology of congestive heart failure.",
        "What antibiotics are used for urinary tract infections?",
    ]

    paraphrases = {
        0: ["What symptoms does Type 2 diabetes cause?", "Tell me about T2D symptoms."],
        1: ["Treatment options for high blood pressure?", "How do you manage hypertension?"],
        2: ["Metformin side effects?", "What are adverse effects of metformin?"],
        3: ["What's a normal BP reading?", "Normal BP range for adults?"],
        10: ["Normal resting heart rate for adults?", "What heart rate is considered normal?"],
    }

    queries = []
    for _ in range(n):
        idx = random.randint(0, len(base_queries) - 1)
        r = random.random()
        if r < 0.3:
            # Exact duplicate
            queries.append(base_queries[idx])
        elif r < 0.6 and idx in paraphrases:
            # Paraphrase
            queries.append(random.choice(paraphrases[idx]))
        else:
            queries.append(base_queries[idx])

    return queries


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_comparison(baseline: PipelineStats, optimized: PipelineStats):
    print(f"\n{'=' * 65}")
    print("  COST OPTIMIZATION COMPARISON")
    print(f"{'=' * 65}")

    headers = f"  {'Metric':<30s} {'Baseline':>15s} {'Optimized':>15s}"
    print(headers)
    print("  " + "─" * 62)

    rows = [
        ("Total queries", str(baseline.total_queries), str(optimized.total_queries)),
        ("Cache hits", "0", str(optimized.cache_hits)),
        ("Cache hit rate", "0.0%", f"{optimized.cache_hits / optimized.total_queries * 100:.1f}%"),
        ("Total tokens", f"{baseline.total_tokens:,}", f"{optimized.total_tokens:,}"),
        ("Total cost", f"${baseline.total_cost:.4f}", f"${optimized.total_cost:.4f}"),
        ("Avg latency (ms)", f"{baseline.avg_latency:.0f}", f"{optimized.avg_latency:.0f}"),
    ]

    for label, b_val, o_val in rows:
        print(f"  {label:<30s} {b_val:>15s} {o_val:>15s}")

    print("  " + "─" * 62)

    if baseline.total_cost > 0:
        cost_savings = baseline.total_cost - optimized.total_cost
        pct = (cost_savings / baseline.total_cost) * 100
        print(f"  💰 Cost savings: ${cost_savings:.4f} ({pct:.1f}%)")
    if baseline.avg_latency > 0:
        lat_savings = baseline.avg_latency - optimized.avg_latency
        pct_lat = (lat_savings / baseline.avg_latency) * 100
        print(f"  ⚡ Latency reduction: {lat_savings:.0f} ms ({pct_lat:.1f}%)")

    # Model distribution for optimized
    model_counts: dict[str, int] = {}
    for r in optimized.results:
        model_counts[r.model] = model_counts.get(r.model, 0) + 1

    print(f"\n  Model distribution (optimized):")
    for model, count in sorted(model_counts.items()):
        print(f"    {model:<16s} {count:>3d} queries")

    print(f"{'=' * 65}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Exercise 4 — Complete Cost Optimization System")
    print("=" * 60)

    if not OPENAI_AVAILABLE:
        print("Install openai to proceed.")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY first.")
        return

    client = OpenAI(api_key=api_key)

    # Use a smaller set for the demo to save cost (set to 50 for full benchmark)
    NUM_QUERIES = 15
    print(f"\nGenerating {NUM_QUERIES} medical queries (duplicates + paraphrases) …")
    queries = generate_medical_queries(NUM_QUERIES)

    for i, q in enumerate(queries, 1):
        print(f"  {i:2d}. {q[:65]}")

    # --- Baseline: no optimization ---
    print(f"\n{'─' * 50}")
    print(f"  Running BASELINE (all queries → gpt-4o) …")
    print(f"{'─' * 50}")
    baseline = run_no_optimization(client, queries)
    for r in baseline.results:
        print(f"  🌐 {r.model:<14s} ${r.cost:.5f}  {r.latency_ms:6.0f} ms  {r.query}")

    # --- Optimized: cache + routing ---
    print(f"\n{'─' * 50}")
    print(f"  Running OPTIMIZED (cache + model routing) …")
    print(f"{'─' * 50}")
    optimized = run_optimized(client, queries)
    for r in optimized.results:
        icon = "⚡" if r.source == "cache" else "🌐"
        print(f"  {icon} {r.model:<14s} ${r.cost:.5f}  {r.latency_ms:6.0f} ms  {r.query}")

    # --- Comparison ---
    print_comparison(baseline, optimized)

    print("\n✅ Cost optimization exercise complete.")
    print("💡 Tip: Increase NUM_QUERIES to 50 for a more representative benchmark.")


if __name__ == "__main__":
    main()
