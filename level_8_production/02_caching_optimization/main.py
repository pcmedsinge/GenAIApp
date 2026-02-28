"""
Level 8, Project 02: Caching & Optimization — Main Demos
=========================================================
Four demos illustrating caching and model-routing strategies to reduce cost and
latency for healthcare LLM applications.

Demos
-----
1. Exact Match Cache      — dictionary keyed on the raw prompt
2. Semantic Cache         — embedding similarity for near-duplicate detection
3. Tiered Model Routing   — route to the cheapest model that can handle the query
4. Cost Dashboard         — track tokens, cost, cache hits, and savings

Usage
-----
    python main.py
"""

import os
import math
import time
import hashlib
import json
from datetime import datetime
from typing import Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI SDK not installed. Run: pip install openai")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️  numpy not installed. Run: pip install numpy")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_client() -> "OpenAI":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY first.")
    return OpenAI(api_key=api_key)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _embed(client: "OpenAI", text: str) -> list[float]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
    return resp.data[0].embedding


def _chat(client: "OpenAI", message: str, model: str = "gpt-4o-mini", max_tokens: int = 256):
    """Returns (reply_text, usage_dict)."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": message},
        ],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content, {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "total_tokens": resp.usage.total_tokens,
    }


# Per-model pricing (USD per 1K tokens — approximate as of 2025)
MODEL_PRICING = {
    "gpt-4o-mini":  {"input": 0.00015, "output": 0.0006},
    "gpt-4o":       {"input": 0.0025,  "output": 0.01},
    "o1-mini":      {"input": 0.003,   "output": 0.012},
}


def _estimate_cost(model: str, usage: dict) -> float:
    p = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o-mini"])
    return (usage["prompt_tokens"] / 1000 * p["input"]) + (usage["completion_tokens"] / 1000 * p["output"])


# ============================================================
# DEMO 1: Exact Match Cache
# ============================================================
def demo_exact_match_cache():
    """Simple dictionary cache — identical queries return instantly."""
    print("\n🔹 DEMO 1: Exact Match Cache")
    print("=" * 60)

    if not OPENAI_AVAILABLE:
        print("OpenAI SDK required.")
        return

    client = get_client()
    cache: dict[str, str] = {}  # md5(query) → reply

    queries = [
        "What are the symptoms of Type 2 diabetes?",
        "What are common side effects of metformin?",
        "What are the symptoms of Type 2 diabetes?",  # duplicate
        "What are common side effects of metformin?",  # duplicate
        "Explain hypertension treatment guidelines.",
    ]

    hits = 0
    misses = 0
    total_saved_ms = 0.0

    for q in queries:
        key = hashlib.md5(q.encode()).hexdigest()
        if key in cache:
            hits += 1
            print(f"  ⚡ CACHE HIT  ({0:.0f} ms): {q[:60]}")
            total_saved_ms += 1500  # assume ~1.5 s saved per hit
        else:
            start = time.perf_counter()
            reply, _ = _chat(client, q)
            elapsed = (time.perf_counter() - start) * 1000
            cache[key] = reply
            misses += 1
            print(f"  🌐 CACHE MISS ({elapsed:.0f} ms): {q[:60]}")

    print(f"\n📊 Results: {hits} hits / {misses} misses")
    print(f"   Hit rate: {hits / len(queries) * 100:.0f}%")
    print(f"   Estimated time saved: ~{total_saved_ms:.0f} ms")


# ============================================================
# DEMO 2: Semantic Cache
# ============================================================
def demo_semantic_cache():
    """Cache based on embedding similarity — catches paraphrased duplicates."""
    print("\n🔹 DEMO 2: Semantic Cache")
    print("=" * 60)

    if not OPENAI_AVAILABLE:
        print("OpenAI SDK required.")
        return

    client = get_client()
    SIMILARITY_THRESHOLD = 0.92

    # Semantic cache: list of (embedding, query, reply)
    sem_cache: list[tuple[list[float], str, str]] = []

    queries = [
        "What are the symptoms of Type 2 diabetes?",
        "What symptoms does someone with Type 2 diabetes have?",
        "Tell me about Type 2 diabetes symptoms",
        "What are common side effects of ibuprofen?",
        "Side effects of ibuprofen — what should I know?",
        "Explain the pathophysiology of heart failure.",
    ]

    hits = 0
    misses = 0

    for q in queries:
        q_emb = _embed(client, q)

        # Check cache
        best_score = 0.0
        best_reply = None
        best_cached_q = None
        for cached_emb, cached_q, cached_reply in sem_cache:
            score = _cosine_sim(q_emb, cached_emb)
            if score > best_score:
                best_score = score
                best_reply = cached_reply
                best_cached_q = cached_q

        if best_score >= SIMILARITY_THRESHOLD:
            hits += 1
            print(f"  ⚡ SEMANTIC HIT (sim={best_score:.4f}): {q[:55]}")
            print(f"       matched → {best_cached_q[:55]}")
        else:
            start = time.perf_counter()
            reply, _ = _chat(client, q)
            elapsed = (time.perf_counter() - start) * 1000
            sem_cache.append((q_emb, q, reply))
            misses += 1
            print(f"  🌐 MISS (best_sim={best_score:.4f}, {elapsed:.0f} ms): {q[:55]}")

    print(f"\n📊 Results: {hits} hits / {misses} misses")
    print(f"   Hit rate: {hits / len(queries) * 100:.0f}%")
    print(f"   Threshold: {SIMILARITY_THRESHOLD}")


# ============================================================
# DEMO 3: Tiered Model Routing
# ============================================================
def demo_tiered_model_routing():
    """Route queries to the cheapest model that can handle them."""
    print("\n🔹 DEMO 3: Tiered Model Routing")
    print("=" * 60)

    if not OPENAI_AVAILABLE:
        print("OpenAI SDK required.")
        return

    client = get_client()

    # Complexity classifier (simple heuristic)
    COMPLEX_KEYWORDS = [
        "differential diagnosis", "pathophysiology", "contraindications",
        "drug interaction", "treatment protocol", "prognosis", "staging",
        "comorbidity", "pharmacokinetics", "clinical trial",
    ]
    REASONING_KEYWORDS = [
        "step by step", "reason through", "analyze the case",
        "compare and contrast", "evaluate the evidence", "work through",
    ]

    def classify_complexity(query: str) -> str:
        q_lower = query.lower()
        if any(kw in q_lower for kw in REASONING_KEYWORDS):
            return "reasoning"
        if any(kw in q_lower for kw in COMPLEX_KEYWORDS):
            return "complex"
        if len(query.split()) > 50:
            return "complex"
        return "simple"

    MODEL_MAP = {
        "simple": "gpt-4o-mini",
        "complex": "gpt-4o",
        "reasoning": "o1-mini",
    }

    queries = [
        "What is aspirin used for?",
        "Explain the pathophysiology of congestive heart failure.",
        "Step by step, reason through the differential diagnosis for a 55-year-old male with chest pain, dyspnea, and elevated troponin.",
        "What is the normal blood pressure range?",
        "Discuss the drug interaction between warfarin and aspirin, including pharmacokinetics.",
    ]

    total_cost_routed = 0.0
    total_cost_gpt4o = 0.0

    for q in queries:
        complexity = classify_complexity(q)
        model = MODEL_MAP[complexity]
        print(f"\n  Query: {q[:70]}…" if len(q) > 70 else f"\n  Query: {q}")
        print(f"  Complexity: {complexity} → Model: {model}")

        # Only call the routed model (skip o1-mini for cost reasons in demo)
        if model == "o1-mini":
            print(f"  ⏭️  Skipping live call to {model} in demo (expensive)")
            continue

        reply, usage = _chat(client, q, model=model)
        cost = _estimate_cost(model, usage)
        cost_if_gpt4o = _estimate_cost("gpt-4o", usage)
        total_cost_routed += cost
        total_cost_gpt4o += cost_if_gpt4o
        print(f"  Tokens: {usage['total_tokens']}  Cost: ${cost:.5f}  (gpt-4o would be ${cost_if_gpt4o:.5f})")
        print(f"  Reply: {reply[:120]}…")

    if total_cost_gpt4o > 0:
        savings = (1 - total_cost_routed / total_cost_gpt4o) * 100
        print(f"\n📊 Routing saved {savings:.1f}% vs always using gpt-4o")
        print(f"   Routed cost: ${total_cost_routed:.5f}   All-gpt-4o cost: ${total_cost_gpt4o:.5f}")


# ============================================================
# DEMO 4: Cost Dashboard
# ============================================================
def demo_cost_dashboard():
    """Track tokens, cost, cache hit rate, and money saved by caching."""
    print("\n🔹 DEMO 4: Cost Dashboard")
    print("=" * 60)

    if not OPENAI_AVAILABLE:
        print("OpenAI SDK required.")
        return

    client = get_client()

    # Tracking state
    stats = {
        "total_queries": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
        "cost_saved": 0.0,
        "queries": [],
    }

    cache: dict[str, tuple[str, dict]] = {}  # md5 → (reply, usage)

    queries = [
        "What are symptoms of pneumonia?",
        "How is pneumonia diagnosed?",
        "What are symptoms of pneumonia?",   # exact dup
        "Treatment options for hypertension",
        "Treatment options for hypertension", # exact dup
        "What is a normal heart rate?",
        "What are symptoms of pneumonia?",   # exact dup
        "Side effects of lisinopril",
        "What is a normal heart rate?",      # exact dup
        "Explain Type 1 diabetes briefly.",
    ]

    for q in queries:
        stats["total_queries"] += 1
        key = hashlib.md5(q.encode()).hexdigest()

        if key in cache:
            stats["cache_hits"] += 1
            cached_reply, cached_usage = cache[key]
            saved = _estimate_cost("gpt-4o-mini", cached_usage)
            stats["cost_saved"] += saved
            stats["queries"].append({"query": q[:50], "source": "cache", "cost": 0.0})
        else:
            stats["cache_misses"] += 1
            reply, usage = _chat(client, q)
            cost = _estimate_cost("gpt-4o-mini", usage)
            stats["total_tokens"] += usage["total_tokens"]
            stats["total_cost"] += cost
            cache[key] = (reply, usage)
            stats["queries"].append({"query": q[:50], "source": "api", "cost": cost})

    # --- Dashboard output ---
    hit_rate = stats["cache_hits"] / stats["total_queries"] * 100 if stats["total_queries"] else 0

    print(f"\n{'─' * 50}")
    print(f"  📊 COST DASHBOARD")
    print(f"{'─' * 50}")
    print(f"  Total queries:      {stats['total_queries']}")
    print(f"  Cache hits:         {stats['cache_hits']}")
    print(f"  Cache misses:       {stats['cache_misses']}")
    print(f"  Hit rate:           {hit_rate:.1f}%")
    print(f"{'─' * 50}")
    print(f"  Total tokens used:  {stats['total_tokens']:,}")
    print(f"  Total API cost:     ${stats['total_cost']:.5f}")
    print(f"  Cost saved (cache): ${stats['cost_saved']:.5f}")
    total_would_be = stats["total_cost"] + stats["cost_saved"]
    if total_would_be > 0:
        pct = stats["cost_saved"] / total_would_be * 100
        print(f"  Savings:            {pct:.1f}% of what it would have cost")
    print(f"{'─' * 50}")
    print(f"\n  Per-query breakdown:")
    for entry in stats["queries"]:
        icon = "⚡" if entry["source"] == "cache" else "🌐"
        print(f"    {icon} {entry['query']:50s}  ${entry['cost']:.5f}")

    print(f"\n✅ Dashboard complete.")


# ============================================================
# Main menu
# ============================================================
def main():
    demos = {
        "1": ("Exact Match Cache", demo_exact_match_cache),
        "2": ("Semantic Cache", demo_semantic_cache),
        "3": ("Tiered Model Routing", demo_tiered_model_routing),
        "4": ("Cost Dashboard", demo_cost_dashboard),
    }

    while True:
        print("\n" + "=" * 60)
        print("LEVEL 8 · PROJECT 02 — CACHING & OPTIMIZATION")
        print("=" * 60)
        for key, (title, _) in demos.items():
            print(f"  {key}. {title}")
        print("  q. Quit")

        choice = input("\nSelect demo: ").strip().lower()
        if choice == "q":
            print("Goodbye!")
            break
        if choice in demos:
            demos[choice][1]()
        else:
            print("Invalid choice — try again.")


if __name__ == "__main__":
    main()
