"""
Exercise 1 — Semantic Cache for Medical Queries
================================================
Build a semantic cache that stores LLM responses keyed by their embedding
vectors. When a new query arrives, check if a cached query is similar enough
(cosine similarity > threshold) and return the cached answer — skipping the
expensive LLM call entirely.

Features:
    • Embedding-based similarity lookup
    • Configurable similarity threshold
    • Cache stats (hits, misses, savings)
    • Test with paraphrased medical questions

Usage:
    python exercise_1_semantic_cache.py
"""

import os
import math
import time
import hashlib
from typing import Optional
from dataclasses import dataclass, field

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  pip install openai")


# ---------------------------------------------------------------------------
# Semantic cache implementation
# ---------------------------------------------------------------------------
@dataclass
class CacheEntry:
    """Single entry in the semantic cache."""
    query: str
    embedding: list[float]
    reply: str
    model: str
    tokens_used: int
    created_at: float = field(default_factory=time.time)
    access_count: int = 0


class SemanticCache:
    """Embedding-similarity-based LLM response cache."""

    def __init__(self, threshold: float = 0.93, max_entries: int = 500):
        self.threshold = threshold
        self.max_entries = max_entries
        self.entries: list[CacheEntry] = []
        self.stats = {"hits": 0, "misses": 0, "embed_calls": 0, "llm_calls": 0}

    # -- helpers --
    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    def _embed(self, client: "OpenAI", text: str) -> list[float]:
        self.stats["embed_calls"] += 1
        resp = client.embeddings.create(model="text-embedding-3-small", input=[text])
        return resp.data[0].embedding

    def _llm_call(self, client: "OpenAI", query: str, model: str, max_tokens: int):
        self.stats["llm_calls"] += 1
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a medical assistant. Be concise."},
                {"role": "user", "content": query},
            ],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content, resp.usage.total_tokens

    # -- public API --
    def lookup(self, embedding: list[float]) -> Optional[CacheEntry]:
        """Find the best matching cache entry above the threshold."""
        best_entry = None
        best_score = 0.0
        for entry in self.entries:
            score = self._cosine(embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_score >= self.threshold and best_entry:
            best_entry.access_count += 1
            return best_entry
        return None

    def store(self, query: str, embedding: list[float], reply: str, model: str, tokens: int):
        """Add an entry, evicting oldest if at capacity."""
        if len(self.entries) >= self.max_entries:
            self.entries.sort(key=lambda e: e.access_count)
            self.entries.pop(0)
        self.entries.append(CacheEntry(
            query=query, embedding=embedding, reply=reply,
            model=model, tokens_used=tokens,
        ))

    def query(self, client: "OpenAI", user_query: str,
              model: str = "gpt-4o-mini", max_tokens: int = 256) -> dict:
        """Full query pipeline: embed → lookup → (hit or miss+store)."""
        q_emb = self._embed(client, user_query)
        hit = self.lookup(q_emb)

        if hit:
            self.stats["hits"] += 1
            return {
                "reply": hit.reply,
                "source": "cache",
                "matched_query": hit.query,
                "similarity": self._cosine(q_emb, hit.embedding),
            }

        # Cache miss — call the LLM
        self.stats["misses"] += 1
        reply, tokens = self._llm_call(client, user_query, model, max_tokens)
        self.store(user_query, q_emb, reply, model, tokens)
        return {
            "reply": reply,
            "source": "api",
            "tokens": tokens,
        }

    def report(self) -> str:
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total * 100) if total else 0
        lines = [
            "",
            "─" * 50,
            "  SEMANTIC CACHE REPORT",
            "─" * 50,
            f"  Cache entries:   {len(self.entries)}",
            f"  Total queries:   {total}",
            f"  Hits:            {self.stats['hits']}",
            f"  Misses:          {self.stats['misses']}",
            f"  Hit rate:        {hit_rate:.1f}%",
            f"  Embed API calls: {self.stats['embed_calls']}",
            f"  LLM API calls:   {self.stats['llm_calls']}",
            f"  Threshold:       {self.threshold}",
            "─" * 50,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main exercise
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Exercise 1 — Semantic Cache for Medical Queries")
    print("=" * 60)

    if not OPENAI_AVAILABLE:
        print("Install openai to proceed.")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY first.")
        return

    client = OpenAI(api_key=api_key)
    cache = SemanticCache(threshold=0.92)

    # Medical query groups — each group has paraphrased versions of the same question
    query_groups = [
        [
            "What are the symptoms of Type 2 diabetes?",
            "What symptoms does Type 2 diabetes cause?",
            "Tell me about Type 2 diabetes symptoms.",
        ],
        [
            "What are common side effects of metformin?",
            "Metformin side effects — what should patients know?",
            "List the adverse effects of metformin.",
        ],
        [
            "How is hypertension treated?",
            "What treatments are available for high blood pressure?",
            "Explain the management of hypertension.",
        ],
        [
            "What is the normal range for blood glucose?",
        ],
    ]

    print("\nSending queries through the semantic cache …\n")
    for group_idx, group in enumerate(query_groups, 1):
        print(f"  Group {group_idx}:")
        for q in group:
            start = time.perf_counter()
            result = cache.query(client, q)
            elapsed = (time.perf_counter() - start) * 1000

            if result["source"] == "cache":
                sim = result.get("similarity", 0)
                print(f"    ⚡ HIT  (sim={sim:.4f}, {elapsed:6.0f} ms) {q[:55]}")
            else:
                print(f"    🌐 MISS ({elapsed:6.0f} ms) {q[:55]}")
        print()

    print(cache.report())
    print("\n✅ Exercise complete.")


if __name__ == "__main__":
    main()
