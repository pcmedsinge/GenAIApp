"""
Exercise 2 — Cache Strategy Comparison
=======================================
Compare four caching strategies for medical LLM queries:

    1. Exact match  — hash-based, fast, but misses paraphrases
    2. Semantic      — embedding similarity, catches paraphrases
    3. TTL-based     — expire entries after N seconds
    4. LRU           — evict least-recently-used when cache is full

Run the same query set through each strategy and measure:
    • Hit rate
    • Average latency
    • Staleness risk

Usage:
    python exercise_2_cache_strategies.py
"""

import os
import math
import time
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  pip install openai")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _embed(client: "OpenAI", text: str) -> list[float]:
    return client.embeddings.create(model="text-embedding-3-small", input=[text]).data[0].embedding


def _llm(client: "OpenAI", query: str) -> tuple[str, int]:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a concise medical assistant."},
            {"role": "user", "content": query},
        ],
        max_tokens=200,
    )
    return resp.choices[0].message.content, resp.usage.total_tokens


# ---------------------------------------------------------------------------
# Strategy 1: Exact Match
# ---------------------------------------------------------------------------
class ExactMatchCache:
    """Hash the prompt string — only literal duplicates hit."""

    def __init__(self):
        self._store: dict[str, str] = {}
        self.hits = 0
        self.misses = 0

    def query(self, client, q: str) -> tuple[str, str]:
        key = hashlib.sha256(q.encode()).hexdigest()
        if key in self._store:
            self.hits += 1
            return self._store[key], "hit"
        reply, _ = _llm(client, q)
        self._store[key] = reply
        self.misses += 1
        return reply, "miss"


# ---------------------------------------------------------------------------
# Strategy 2: Semantic Cache
# ---------------------------------------------------------------------------
class SemanticCacheStrategy:
    """Embedding-similarity cache with configurable threshold."""

    def __init__(self, threshold: float = 0.92):
        self.threshold = threshold
        self._entries: list[tuple[list[float], str, str]] = []  # (emb, q, reply)
        self.hits = 0
        self.misses = 0

    def query(self, client, q: str) -> tuple[str, str]:
        q_emb = _embed(client, q)
        best_score = 0.0
        best_reply = None
        for emb, _, reply in self._entries:
            score = _cosine(q_emb, emb)
            if score > best_score:
                best_score = score
                best_reply = reply
        if best_score >= self.threshold and best_reply:
            self.hits += 1
            return best_reply, "hit"
        reply, _ = _llm(client, q)
        self._entries.append((q_emb, q, reply))
        self.misses += 1
        return reply, "miss"


# ---------------------------------------------------------------------------
# Strategy 3: TTL (Time-To-Live)
# ---------------------------------------------------------------------------
class TTLCache:
    """Exact-match cache with expiration. Entries older than `ttl` seconds are
    considered stale and evicted on access."""

    def __init__(self, ttl_seconds: float = 10.0):
        self.ttl = ttl_seconds
        self._store: dict[str, tuple[str, float]] = {}  # hash → (reply, timestamp)
        self.hits = 0
        self.misses = 0
        self.expirations = 0

    def query(self, client, q: str) -> tuple[str, str]:
        key = hashlib.sha256(q.encode()).hexdigest()
        now = time.time()
        if key in self._store:
            reply, ts = self._store[key]
            if now - ts <= self.ttl:
                self.hits += 1
                return reply, "hit"
            else:
                self.expirations += 1
                del self._store[key]
        reply, _ = _llm(client, q)
        self._store[key] = (reply, now)
        self.misses += 1
        return reply, "miss"


# ---------------------------------------------------------------------------
# Strategy 4: LRU (Least Recently Used)
# ---------------------------------------------------------------------------
class LRUCache:
    """Exact-match cache with a maximum size. Evicts the least-recently-used
    entry when the cache is full."""

    def __init__(self, max_size: int = 3):
        self.max_size = max_size
        self._store: OrderedDict[str, str] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def query(self, client, q: str) -> tuple[str, str]:
        key = hashlib.sha256(q.encode()).hexdigest()
        if key in self._store:
            self._store.move_to_end(key)
            self.hits += 1
            return self._store[key], "hit"
        reply, _ = _llm(client, q)
        if len(self._store) >= self.max_size:
            self._store.popitem(last=False)
            self.evictions += 1
        self._store[key] = reply
        self.misses += 1
        return reply, "miss"


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(client, strategy, queries: list[str], name: str) -> dict:
    """Run all queries through a strategy and collect stats."""
    print(f"\n{'─' * 50}")
    print(f"  Strategy: {name}")
    print(f"{'─' * 50}")

    latencies = []
    for q in queries:
        start = time.perf_counter()
        _, result_type = strategy.query(client, q)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        icon = "⚡" if result_type == "hit" else "🌐"
        print(f"  {icon} {result_type:4s} {elapsed:7.1f} ms  {q[:55]}")

    total = strategy.hits + strategy.misses
    hit_rate = (strategy.hits / total * 100) if total else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    summary = {
        "name": name,
        "hits": strategy.hits,
        "misses": strategy.misses,
        "hit_rate": hit_rate,
        "avg_latency_ms": avg_lat,
    }
    # Add strategy-specific stats
    if hasattr(strategy, "expirations"):
        summary["expirations"] = strategy.expirations
    if hasattr(strategy, "evictions"):
        summary["evictions"] = strategy.evictions

    print(f"  → Hit rate: {hit_rate:.1f}%   Avg latency: {avg_lat:.0f} ms")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Exercise 2 — Cache Strategy Comparison")
    print("=" * 60)

    if not OPENAI_AVAILABLE:
        print("Install openai to proceed.")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY first.")
        return

    client = OpenAI(api_key=api_key)

    # Query set with exact duplicates and paraphrases
    queries = [
        "What are symptoms of Type 2 diabetes?",
        "How is hypertension treated?",
        "What are symptoms of Type 2 diabetes?",            # exact dup
        "What symptoms does Type 2 diabetes cause?",         # paraphrase
        "How is hypertension treated?",                      # exact dup
        "Treatment options for high blood pressure",         # paraphrase
        "What are the side effects of aspirin?",
        "What are symptoms of Type 2 diabetes?",            # exact dup (3rd time)
        "What are the side effects of aspirin?",            # exact dup
        "Normal blood glucose range?",
    ]

    strategies = [
        (ExactMatchCache(), "Exact Match"),
        (SemanticCacheStrategy(threshold=0.92), "Semantic (threshold=0.92)"),
        (TTLCache(ttl_seconds=5.0), "TTL (5 sec)"),
        (LRUCache(max_size=3), "LRU (max_size=3)"),
    ]

    results = []
    for strategy, name in strategies:
        r = run_benchmark(client, strategy, queries, name)
        results.append(r)

    # Summary table
    print(f"\n{'=' * 60}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Strategy':<28s} {'Hits':>5s} {'Misses':>7s} {'Rate':>6s} {'Avg ms':>8s}")
    print(f"  {'─' * 56}")
    for r in results:
        print(f"  {r['name']:<28s} {r['hits']:>5d} {r['misses']:>7d} {r['hit_rate']:>5.1f}% {r['avg_latency_ms']:>7.0f}")

    # Recommend
    best = max(results, key=lambda r: r["hit_rate"])
    print(f"\n  🏆 Best hit rate: {best['name']} ({best['hit_rate']:.1f}%)")
    print("\n✅ Comparison complete.")


if __name__ == "__main__":
    main()
