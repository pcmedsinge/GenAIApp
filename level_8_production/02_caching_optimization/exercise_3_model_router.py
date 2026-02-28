"""







































































- No external monitoring libraries needed — built from scratch- Python 3.9+- OpenAI API key set as `OPENAI_API_KEY`## Prerequisites```python exercise_4_alerting_system.pypython exercise_3_agent_tracer.pypython exercise_2_cost_dashboard.pypython exercise_1_request_logger.py# Individual exercisespython main.py# Main demos```bash## Running| 4 | Alerting System | Threshold alerts, anomaly detection || 3 | Agent Tracer | Execution tree, timing breakdown || 2 | Cost Dashboard | Text dashboard, budget alerts || 1 | Request Logger | JSONL logging, query by date/model ||----------|-------|------------|| Exercise | Focus | Key Skills |## Exercises- **Alerting thresholds** — define SLOs and alert on violations- **Distributed tracing** — follow requests across multiple LLM calls- **Cost attribution** — track spend per model, user, and endpoint- **Percentile tracking** — p50/p95/p99 for realistic latency view- **Structured logging** — JSON logs for machine-parseable analysis## Key Concepts(LLM call → tool call → result) with timing breakdown.Build a trace viewer for agent interactions showing each step### Demo 4: Trace ViewerAlert when latency exceeds configurable thresholds.Track p50/p95/p99 latency percentiles. Detect anomalies automatically.### Demo 3: Latency Monitoringaggregate reports and budget projections.Track costs per model, per endpoint, per user. Generate daily/weekly### Demo 2: Cost Trackingand cost. Store structured logs as JSON for analysis.Log every LLM request with timestamp, model, tokens in/out, latency,### Demo 1: Request Logging## What You'll Build- **Quality drift** — model behavior can change without code changes- **Agent complexity** — multi-step agent runs need end-to-end tracing- **Token budgets** — usage must be tracked to avoid surprise bills- **Unpredictable latency** — responses range from 500ms to 30+ seconds- **Variable costs** — each call costs different amounts based on tokensUnlike traditional APIs, LLM calls have unique monitoring challenges:## Why LLM Monitoring Mattersdependencies needed.observability infrastructure from scratch — no external monitoringreliability, control costs, and maintain performance. This project buildsProduction LLM applications require comprehensive monitoring to ensure## OverviewExercise 3 — Intelligent Model Router
======================================
Build a router that analyses each query's complexity and routes it to the
cheapest OpenAI model that can handle it well:

    simple     → gpt-4o-mini   (cheapest)
    complex    → gpt-4o        (more capable)
    reasoning  → o1-mini       (multi-step reasoning)

Complexity is estimated via a set of heuristics:
    • Token count / sentence length
    • Presence of complex medical jargon
    • Multi-step or comparative reasoning keywords
    • Question depth (e.g. "explain pathophysiology" vs "define X")

Usage:
    python exercise_3_model_router.py
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  pip install openai")


# ---------------------------------------------------------------------------
# Pricing (USD per 1 K tokens, approx. 2025)
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
# Complexity classifier
# ---------------------------------------------------------------------------
COMPLEX_TERMS = {
    "pathophysiology", "pharmacokinetics", "pharmacodynamics",
    "differential diagnosis", "contraindication", "comorbidity",
    "drug interaction", "treatment protocol", "clinical staging",
    "prognostic factors", "histopathology", "etiology",
    "mechanism of action", "adverse event", "therapeutic index",
    "renal clearance", "hepatic metabolism", "evidence-based",
}

REASONING_PATTERNS = [
    r"step.by.step",
    r"reason\s+through",
    r"analyze\s+the\s+case",
    r"compare\s+and\s+contrast",
    r"evaluate\s+the\s+evidence",
    r"work\s+through",
    r"what\s+is\s+the\s+best\s+approach",
    r"given\s+the\s+following\s+labs?",
    r"interpret\s+these\s+results",
    r"multi.?step",
]


@dataclass
class ComplexityResult:
    level: str  # simple | complex | reasoning
    score: float
    reasons: list[str] = field(default_factory=list)
    recommended_model: str = ""


def classify_complexity(query: str) -> ComplexityResult:
    """Score query complexity and recommend a model tier."""
    q_lower = query.lower()
    score = 0.0
    reasons = []

    # 1. Word count / length
    word_count = len(query.split())
    if word_count > 60:
        score += 2.0
        reasons.append(f"long query ({word_count} words)")
    elif word_count > 30:
        score += 1.0
        reasons.append(f"medium-length query ({word_count} words)")

    # 2. Complex medical terms
    found_terms = [t for t in COMPLEX_TERMS if t in q_lower]
    if found_terms:
        score += len(found_terms) * 1.5
        reasons.append(f"complex terms: {', '.join(found_terms[:3])}")

    # 3. Reasoning patterns
    matched_reasoning = [p for p in REASONING_PATTERNS if re.search(p, q_lower)]
    if matched_reasoning:
        score += len(matched_reasoning) * 2.0
        reasons.append(f"reasoning pattern detected")

    # 4. Multiple questions / sub-parts
    question_marks = query.count("?")
    if question_marks > 1:
        score += question_marks * 0.5
        reasons.append(f"multi-part ({question_marks} questions)")

    # 5. Numbered lists (1., 2., …)
    if re.search(r"\b\d+\.\s", query):
        score += 1.0
        reasons.append("enumerated requirements")

    # Determine tier
    if score >= 4.0:
        level = "reasoning"
        model = "o1-mini"
    elif score >= 1.5:
        level = "complex"
        model = "gpt-4o"
    else:
        level = "simple"
        model = "gpt-4o-mini"

    if not reasons:
        reasons.append("straightforward query")

    return ComplexityResult(level=level, score=round(score, 1), reasons=reasons, recommended_model=model)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
@dataclass
class RoutingRecord:
    query: str
    complexity: str
    model_used: str
    tokens: int
    cost: float
    latency_ms: float


class ModelRouter:
    """Route queries to the cheapest appropriate model."""

    def __init__(self, allow_o1: bool = False):
        self.allow_o1 = allow_o1
        self.records: list[RoutingRecord] = []

    def route(self, client: "OpenAI", query: str, max_tokens: int = 256) -> dict:
        cx = classify_complexity(query)
        model = cx.recommended_model

        # Skip o1-mini unless explicitly allowed (it's expensive)
        if model == "o1-mini" and not self.allow_o1:
            model = "gpt-4o"

        start = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a medical assistant. Be concise and accurate."},
                {"role": "user", "content": query},
            ],
            max_tokens=max_tokens,
        )
        elapsed = (time.perf_counter() - start) * 1000

        usage = resp.usage
        cost = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
        record = RoutingRecord(
            query=query[:60],
            complexity=cx.level,
            model_used=model,
            tokens=usage.total_tokens,
            cost=cost,
            latency_ms=round(elapsed, 1),
        )
        self.records.append(record)

        return {
            "reply": resp.choices[0].message.content,
            "model": model,
            "complexity": cx.level,
            "score": cx.score,
            "reasons": cx.reasons,
            "cost": cost,
            "tokens": usage.total_tokens,
            "latency_ms": elapsed,
        }

    def report(self) -> str:
        lines = [
            "",
            "─" * 65,
            "  MODEL ROUTING REPORT",
            "─" * 65,
            f"  {'Query':<35s} {'Complexity':>10s} {'Model':>14s} {'Tok':>5s} {'Cost':>9s}",
            "  " + "─" * 63,
        ]
        total_cost = 0.0
        total_cost_if_gpt4o = 0.0
        for r in self.records:
            total_cost += r.cost
            # Estimate what it would have cost on gpt-4o
            cost_4o = estimate_cost("gpt-4o", r.tokens, 0)  # rough
            total_cost_if_gpt4o += cost_4o * 1.5  # approximate
            lines.append(
                f"  {r.query:<35s} {r.complexity:>10s} {r.model_used:>14s} "
                f"{r.tokens:>5d} ${r.cost:.5f}"
            )
        lines.append("  " + "─" * 63)
        lines.append(f"  Total routed cost:    ${total_cost:.5f}")
        if total_cost_if_gpt4o > 0:
            savings = (1 - total_cost / total_cost_if_gpt4o) * 100
            lines.append(f"  Est. all-gpt-4o cost: ${total_cost_if_gpt4o:.5f}")
            lines.append(f"  Savings from routing: ~{savings:.0f}%")
        lines.append("─" * 65)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Exercise 3 — Intelligent Model Router")
    print("=" * 60)

    if not OPENAI_AVAILABLE:
        print("Install openai to proceed.")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY first.")
        return

    client = OpenAI(api_key=api_key)
    router = ModelRouter(allow_o1=False)

    queries = [
        "What is aspirin?",
        "Normal resting heart rate range?",
        "Explain the pathophysiology of congestive heart failure and its prognostic factors.",
        "What are common side effects of lisinopril?",
        "Compare and contrast ACE inhibitors and ARBs for hypertension treatment. "
        "Include mechanism of action, contraindications, and evidence-based guidelines.",
        "Define tachycardia.",
        "Step by step, reason through the differential diagnosis for a 60-year-old female "
        "presenting with acute dyspnea, bilateral lower extremity edema, and elevated BNP.",
        "What is ibuprofen used for?",
    ]

    print("\nClassifying and routing queries …\n")
    for q in queries:
        cx = classify_complexity(q)
        trimmed = q[:70] + "…" if len(q) > 70 else q
        print(f"  [{cx.level:>9s}] (score={cx.score:4.1f}) → {cx.recommended_model}")
        print(f"             {trimmed}")

    print("\n\nSending queries to routed models …\n")
    for q in queries:
        result = router.route(client, q)
        print(f"  {result['complexity']:>9s} → {result['model']:<14s} "
              f"${result['cost']:.5f}  {result['latency_ms']:6.0f} ms")
        print(f"     {result['reply'][:100]}…\n")

    print(router.report())
    print("\n✅ Routing exercise complete.")


if __name__ == "__main__":
    main()
