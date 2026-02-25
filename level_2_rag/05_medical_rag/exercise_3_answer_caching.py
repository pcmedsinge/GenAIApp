"""
Exercise 3: Answer Caching
Implement a caching system so that repeated or similar questions return
cached responses instantly instead of hitting the LLM again.

Skills practiced:
- Exact-match caching with dictionaries
- Semantic caching (similar but not identical questions)
- Cache TTL (time-to-live) and invalidation
- Measuring cache hit rates and cost savings

Healthcare context:
  In a hospital, many clinicians ask the SAME questions: "What's the
  metformin starting dose?" Caching avoids redundant API calls, reduces
  latency, and cuts costs — especially at scale.
"""

import os
import json
import time
import hashlib
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)


# ============================================================
# Medical Knowledge Base (from main.py)
# ============================================================

MEDICAL_DOCUMENTS = [
    {"id": "htn_def", "text": "Hypertension is defined as blood pressure consistently at or above 130/80 mmHg. Stage 1 is 130-139 systolic or 80-89 diastolic. Stage 2 is 140/90 or higher.", "metadata": {"specialty": "cardiology", "topic": "hypertension", "subtopic": "definition"}},
    {"id": "htn_meds", "text": "First-line antihypertensives: ACE inhibitors (lisinopril 10-40mg, enalapril 5-40mg), ARBs (losartan 50-100mg, valsartan 80-320mg), CCBs (amlodipine 2.5-10mg), thiazides (HCTZ 12.5-25mg).", "metadata": {"specialty": "cardiology", "topic": "hypertension", "subtopic": "medications"}},
    {"id": "hf_pillars", "text": "Heart failure with reduced EF treatment pillars: (1) ARNI (sacubitril-valsartan), (2) Beta-blocker (carvedilol, metoprolol succinate, bisoprolol), (3) MRA (spironolactone, eplerenone), (4) SGLT2i (dapagliflozin, empagliflozin). All four improve survival.", "metadata": {"specialty": "cardiology", "topic": "heart_failure", "subtopic": "medications"}},
    {"id": "dm_dx", "text": "Type 2 Diabetes diagnosis: fasting glucose 126+ mg/dL on two occasions, HbA1c 6.5%+, 2-hour OGTT 200+ mg/dL, or random glucose 200+ with symptoms.", "metadata": {"specialty": "endocrinology", "topic": "diabetes", "subtopic": "diagnosis"}},
    {"id": "dm_firstline", "text": "Metformin is first-line for Type 2 diabetes. Start 500mg once daily with meals, titrate to 2000mg daily. Contraindicated if eGFR below 30.", "metadata": {"specialty": "endocrinology", "topic": "diabetes", "subtopic": "metformin"}},
    {"id": "dm_addon", "text": "Add-on therapy after metformin: GLP-1 agonists (semaglutide, liraglutide) if CV disease or obesity. SGLT2 inhibitors if heart failure or CKD.", "metadata": {"specialty": "endocrinology", "topic": "diabetes", "subtopic": "second_line"}},
    {"id": "dep_tx", "text": "Depression treatment: Mild — psychotherapy. Moderate — SSRI plus CBT. Severe — SSRI plus CBT; SNRI or mirtazapine if SSRI fails.", "metadata": {"specialty": "psychiatry", "topic": "depression", "subtopic": "treatment"}},
    {"id": "ckd_mgmt", "text": "CKD management: BP target less than 130/80. ACE/ARB for proteinuria. Avoid nephrotoxins. Refer to nephrology at Stage 4.", "metadata": {"specialty": "nephrology", "topic": "ckd", "subtopic": "management"}},
]


def setup_knowledge_base():
    """Create ChromaDB collection"""
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_caching", embedding_function=openai_ef
    )
    collection.add(
        ids=[d["id"] for d in MEDICAL_DOCUMENTS],
        documents=[d["text"] for d in MEDICAL_DOCUMENTS],
        metadatas=[d["metadata"] for d in MEDICAL_DOCUMENTS]
    )
    return collection


def retrieve_and_answer(question, collection):
    """Standard RAG: retrieve + generate (no cache)"""
    results = collection.query(query_texts=[question], n_results=3)
    context = "\n\n".join([
        f"[Source {i+1}]: {results['documents'][0][i]}"
        for i in range(len(results["ids"][0]))
    ])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer from provided sources only. Cite [Source X]. Be specific with doses and criteria. Educational purposes only."},
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=300, temperature=0.2
    )
    return {
        "answer": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
        "source_ids": results["ids"][0],
    }


# ============================================================
# Cache Implementations
# ============================================================

class ExactMatchCache:
    """Simple dictionary-based exact match cache with TTL"""

    def __init__(self, ttl_seconds=3600):
        self.cache = {}
        self.ttl = ttl_seconds
        self.stats = {"hits": 0, "misses": 0, "tokens_saved": 0}

    def _make_key(self, question):
        """Normalize and hash the question"""
        normalized = question.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, question):
        """Look up a cached answer"""
        key = self._make_key(question)
        if key in self.cache:
            entry = self.cache[key]
            # Check TTL
            if time.time() - entry["timestamp"] < self.ttl:
                self.stats["hits"] += 1
                self.stats["tokens_saved"] += entry["tokens"]
                return entry
            else:
                # Expired
                del self.cache[key]

        self.stats["misses"] += 1
        return None

    def put(self, question, answer, tokens, source_ids):
        """Store an answer in the cache"""
        key = self._make_key(question)
        self.cache[key] = {
            "question": question,
            "answer": answer,
            "tokens": tokens,
            "source_ids": source_ids,
            "timestamp": time.time(),
        }

    def get_stats(self):
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {**self.stats, "total": total, "hit_rate": hit_rate, "cache_size": len(self.cache)}


class SemanticCache:
    """Embedding-based cache that matches SIMILAR (not just identical) questions"""

    def __init__(self, similarity_threshold=0.90, ttl_seconds=3600):
        self.entries = []  # list of {question, answer, tokens, source_ids, timestamp}
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        self.stats = {"hits": 0, "misses": 0, "tokens_saved": 0}

        # Use a separate ChromaDB collection for cache
        self.chroma_client = chromadb.Client()
        self.cache_collection = self.chroma_client.create_collection(
            name="semantic_cache",
            embedding_function=openai_ef
        )
        self._next_id = 0

    def get(self, question):
        """Find a semantically similar cached question"""
        if self.cache_collection.count() == 0:
            self.stats["misses"] += 1
            return None

        results = self.cache_collection.query(
            query_texts=[question],
            n_results=1
        )

        if results["ids"][0]:
            distance = results["distances"][0][0]
            # Convert distance to similarity (lower distance = more similar)
            # For cosine distance: similarity ≈ 1 - distance/2 (rough approximation)
            similarity = 1 - distance / 2

            if similarity >= self.threshold:
                # Find the corresponding cache entry
                cached_id = results["ids"][0][0]
                for entry in self.entries:
                    if entry["cache_id"] == cached_id:
                        # Check TTL
                        if time.time() - entry["timestamp"] < self.ttl:
                            self.stats["hits"] += 1
                            self.stats["tokens_saved"] += entry["tokens"]
                            return {
                                **entry,
                                "matched_question": entry["question"],
                                "similarity": similarity,
                            }

        self.stats["misses"] += 1
        return None

    def put(self, question, answer, tokens, source_ids):
        """Store an answer with its embedding for semantic matching"""
        cache_id = f"cache_{self._next_id}"
        self._next_id += 1

        self.cache_collection.add(
            ids=[cache_id],
            documents=[question]
        )

        self.entries.append({
            "cache_id": cache_id,
            "question": question,
            "answer": answer,
            "tokens": tokens,
            "source_ids": source_ids,
            "timestamp": time.time(),
        })

    def get_stats(self):
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        return {**self.stats, "total": total, "hit_rate": hit_rate, "cache_size": len(self.entries)}


# ============================================================
# Demo 1: Exact Match Caching
# ============================================================

def demo_exact_cache():
    """Show exact match caching in action"""
    print("\n" + "=" * 70)
    print("DEMO 1: EXACT MATCH CACHING")
    print("=" * 70)

    collection = setup_knowledge_base()
    cache = ExactMatchCache(ttl_seconds=60)

    queries = [
        "What is the starting dose of metformin?",
        "What are the heart failure medication pillars?",
        "What is the starting dose of metformin?",     # exact repeat
        "What are the heart failure medication pillars?", # exact repeat
        "What is the starting dose of METFORMIN?",      # case difference → still matches
        "How is diabetes diagnosed?",                    # new question
        "How is diabetes diagnosed?",                    # repeat
    ]

    print(f"\n   Processing {len(queries)} queries...\n")

    for i, question in enumerate(queries):
        start = time.time()
        cached = cache.get(question)

        if cached:
            elapsed = time.time() - start
            print(f"   {i+1}. ⚡ CACHE HIT ({elapsed*1000:.1f}ms)")
            print(f"      Q: \"{question}\"")
            print(f"      A: \"{cached['answer'][:80]}...\"")
        else:
            result = retrieve_and_answer(question, collection)
            elapsed = time.time() - start
            cache.put(question, result["answer"], result["tokens"], result["source_ids"])
            print(f"   {i+1}. 🔍 CACHE MISS ({elapsed*1000:.0f}ms, {result['tokens']} tokens)")
            print(f"      Q: \"{question}\"")
            print(f"      A: \"{result['answer'][:80]}...\"")
        print()

    stats = cache.get_stats()
    print(f"   {'─' * 50}")
    print(f"   📊 Cache Statistics:")
    print(f"      Total queries:  {stats['total']}")
    print(f"      Cache hits:     {stats['hits']}")
    print(f"      Cache misses:   {stats['misses']}")
    print(f"      Hit rate:       {stats['hit_rate']:.0%}")
    print(f"      Tokens saved:   {stats['tokens_saved']}")
    print(f"      Cache entries:  {stats['cache_size']}")

    print("""
💡 EXACT MATCH CACHE:
   ✅ Instant lookups (~0ms vs ~1000ms for LLM call)
   ✅ Zero cost for cache hits
   ✅ Simple to implement and debug
   ❌ Misses slightly different phrasings ("metformin dose" vs "starting dose of metformin")
""")


# ============================================================
# Demo 2: Semantic Caching
# ============================================================

def demo_semantic_cache():
    """Show semantic caching that matches SIMILAR questions"""
    print("\n" + "=" * 70)
    print("DEMO 2: SEMANTIC CACHING (Similar Question Matching)")
    print("=" * 70)

    collection = setup_knowledge_base()
    cache = SemanticCache(similarity_threshold=0.85, ttl_seconds=300)

    # First populate cache with some questions
    seed_questions = [
        "What is the starting dose of metformin?",
        "What medications treat heart failure?",
        "How do you diagnose Type 2 diabetes?",
    ]

    print(f"\n   📦 Seeding cache with {len(seed_questions)} questions...")
    for q in seed_questions:
        result = retrieve_and_answer(q, collection)
        cache.put(q, result["answer"], result["tokens"], result["source_ids"])
        print(f"      Cached: \"{q}\"")

    # Now test with similar but NOT identical questions
    test_questions = [
        ("What dose should I start metformin at?",                 True,  "Similar to seed"),
        ("Metformin initial dosing for new diabetes patients",     True,  "Similar to seed"),
        ("What are the four pillars of HF treatment?",             True,  "Similar to seed"),
        ("What is the treatment for osteoarthritis?",              False, "NOT similar to any seed"),
        ("How is Type 2 DM diagnosed?",                            True,  "Similar to seed"),
    ]

    print(f"\n   🔍 Testing similar questions:\n")

    for question, should_hit, description in test_questions:
        cached = cache.get(question)

        if cached:
            print(f"   ⚡ CACHE HIT ({description})")
            print(f"      Query:    \"{question}\"")
            print(f"      Matched:  \"{cached['matched_question']}\"")
            print(f"      Similarity: {cached['similarity']:.3f}")
        else:
            print(f"   🔍 CACHE MISS ({description})")
            print(f"      Query: \"{question}\"")
            result = retrieve_and_answer(question, collection)
            cache.put(question, result["answer"], result["tokens"], result["source_ids"])
        print()

    stats = cache.get_stats()
    print(f"   {'─' * 50}")
    print(f"   📊 Semantic Cache Stats:")
    print(f"      Hits: {stats['hits']} | Misses: {stats['misses']} | Hit rate: {stats['hit_rate']:.0%}")
    print(f"      Tokens saved: {stats['tokens_saved']}")

    print("""
💡 SEMANTIC CACHING:
   ✅ Matches similar questions ("metformin dose" ≈ "starting dose of metformin")
   ✅ Higher hit rate than exact match in practice
   ❌ Costs one embedding call per lookup
   ❌ Risk of returning wrong cached answer if threshold too low
   
   THRESHOLD TUNING:
   • 0.95+ = very strict (almost exact match)
   • 0.85-0.95 = balanced (catches rephrasings)
   • <0.85 = risky (might return wrong answers)
""")


# ============================================================
# Demo 3: Cache Economics
# ============================================================

def demo_cache_economics():
    """Calculate the cost savings from caching"""
    print("\n" + "=" * 70)
    print("DEMO 3: CACHE ECONOMICS — How Much You Save")
    print("=" * 70)

    # Simulate a day of queries with different cache hit rates
    scenarios = [
        {"name": "No Cache",        "hit_rate": 0.0},
        {"name": "Exact Cache",     "hit_rate": 0.30},
        {"name": "Semantic Cache",  "hit_rate": 0.55},
        {"name": "Warm Sem. Cache", "hit_rate": 0.75},
    ]

    queries_per_day = 500
    avg_tokens_per_query = 350  # retrieval + generation
    cost_per_token = 0.00000015  # gpt-4o-mini pricing approximation
    avg_latency_llm_ms = 1200
    avg_latency_cache_ms = 50

    print(f"\n   Assumptions: {queries_per_day} queries/day, "
          f"{avg_tokens_per_query} tokens/query, gpt-4o-mini pricing\n")

    print(f"   {'Scenario':<20} {'LLM Calls':>10} {'Daily Cost':>12} {'Avg Latency':>12} {'Savings':>10}")
    print(f"   {'─' * 64}")

    base_cost = queries_per_day * avg_tokens_per_query * cost_per_token

    for s in scenarios:
        llm_calls = int(queries_per_day * (1 - s["hit_rate"]))
        daily_cost = llm_calls * avg_tokens_per_query * cost_per_token
        avg_latency = (1 - s["hit_rate"]) * avg_latency_llm_ms + s["hit_rate"] * avg_latency_cache_ms
        savings = base_cost - daily_cost

        print(f"   {s['name']:<20} {llm_calls:>10} ${daily_cost:>10.4f} {avg_latency:>10.0f}ms ${savings:>8.4f}")

    # Monthly projections
    print(f"\n   📅 Monthly projections (30 days):\n")
    print(f"   {'Scenario':<20} {'Monthly Cost':>14} {'Monthly Savings':>16}")
    print(f"   {'─' * 50}")
    for s in scenarios:
        llm_calls = int(queries_per_day * (1 - s["hit_rate"]))
        monthly = llm_calls * avg_tokens_per_query * cost_per_token * 30
        monthly_savings = (base_cost - llm_calls * avg_tokens_per_query * cost_per_token) * 30
        print(f"   {s['name']:<20} ${monthly:>13.2f} ${monthly_savings:>14.2f}")

    print(f"""
💡 CACHING ECONOMICS:
   • Even 30% hit rate (exact match) cuts costs significantly
   • Semantic caching with warm-up can reach 50-75% hit rate
   • LATENCY reduction is often more valuable than cost savings
   • In healthcare: faster answers = better clinician workflows

   COST SCALES WITH VOLUME:
   • 500 queries/day → savings are modest
   • 50,000 queries/day → caching saves hundreds $/month
   • The ROI of caching INCREASES with scale
""")


# ============================================================
# Demo 4: Cache TTL and Invalidation
# ============================================================

def demo_ttl_invalidation():
    """Demonstrate cache expiration and invalidation strategies"""
    print("\n" + "=" * 70)
    print("DEMO 4: CACHE TTL AND INVALIDATION")
    print("=" * 70)

    collection = setup_knowledge_base()

    # Demo with a very short TTL to show expiration
    cache = ExactMatchCache(ttl_seconds=3)  # 3-second TTL for demo

    question = "What is the starting dose of metformin?"

    # Step 1: First query — cache miss
    print(f"\n   Step 1: First query (cache miss)")
    result = retrieve_and_answer(question, collection)
    cache.put(question, result["answer"], result["tokens"], result["source_ids"])
    print(f"   🔍 MISS → cached answer ({result['tokens']} tokens)")

    # Step 2: Immediate repeat — cache hit
    print(f"\n   Step 2: Immediate repeat (should be cache hit)")
    cached = cache.get(question)
    if cached:
        print(f"   ⚡ HIT — returned cached answer")
    else:
        print(f"   🔍 MISS (unexpected)")

    # Step 3: Wait for TTL to expire
    print(f"\n   Step 3: Waiting 4 seconds for TTL to expire...")
    time.sleep(4)

    # Step 4: Same query — cache miss (expired)
    print(f"\n   Step 4: Same query after TTL expiry")
    cached = cache.get(question)
    if cached:
        print(f"   ⚡ HIT (TTL not expired yet)")
    else:
        print(f"   🔍 MISS — expired! Re-fetching from LLM...")
        result = retrieve_and_answer(question, collection)
        cache.put(question, result["answer"], result["tokens"], result["source_ids"])
        print(f"   Cached fresh answer ({result['tokens']} tokens)")

    # Invalidation strategies
    print(f"""
   {'─' * 50}
   📋 CACHE INVALIDATION STRATEGIES:

   ┌──────────────────────────────────────────────────────────────┐
   │ Strategy           │ When to Use                            │
   ├────────────────────┼────────────────────────────────────────┤
   │ Time-based (TTL)   │ General purpose. Set based on how      │
   │                    │ often knowledge base changes.          │
   │                    │ Guidelines: TTL = 24 hours             │
   │                    │ Drug info: TTL = 7 days                │
   │                    │ Policies: TTL = 1 hour (change often)  │
   ├────────────────────┼────────────────────────────────────────┤
   │ Version-based      │ Clear cache when knowledge base is     │
   │                    │ updated. Tag cache with KB version.     │
   ├────────────────────┼────────────────────────────────────────┤
   │ Topic-based        │ Only invalidate cached answers for     │
   │                    │ topics whose documents changed.         │
   ├────────────────────┼────────────────────────────────────────┤
   │ Manual purge       │ Admin triggers cache clear after a     │
   │                    │ critical guideline update.              │
   └──────────────────────────────────────────────────────────────┘

   🏥 HEALTHCARE CRITICAL:
   • Stale cached answers can be DANGEROUS in healthcare
   • If a drug dosing guideline changes, cached old answers are wrong
   • Always version-tag your cache AND set a reasonable TTL
   • Alert system when cache is serving answers from outdated KB
""")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n⚡ Exercise 3: Answer Caching")
    print("=" * 70)
    print("Don't pay for the same answer twice\n")

    print("Choose a demo:")
    print("1. Exact match caching (dictionary-based)")
    print("2. Semantic caching (similar question matching)")
    print("3. Cache economics (cost and latency savings)")
    print("4. Cache TTL and invalidation strategies")
    print("5. Run all demos in sequence")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_exact_cache()
    elif choice == "2":
        demo_semantic_cache()
    elif choice == "3":
        demo_cache_economics()
    elif choice == "4":
        demo_ttl_invalidation()
    elif choice == "5":
        demo_exact_cache()
        demo_semantic_cache()
        demo_cache_economics()
        demo_ttl_invalidation()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY LEARNINGS — Exercise 3: Answer Caching
{'=' * 70}

1. TWO CACHING APPROACHES:
   • Exact match: Fast, simple, zero risk — but low hit rate
   • Semantic: Higher hit rate, catches rephrasings — but costs one embedding call
   • In practice: use BOTH (check exact first, then semantic)

2. CACHE ECONOMICS:
   • Even 30% hit rate significantly reduces costs
   • Latency reduction often matters more than cost savings
   • ROI increases with query volume
   • In healthcare: faster answers = better clinician adoption

3. TTL AND INVALIDATION:
   • ALWAYS set a TTL — stale answers are dangerous in healthcare
   • Version-tag cache entries with knowledge base version
   • Topic-based invalidation for surgical precision
   • When in doubt, expire sooner rather than later

4. PRODUCTION PATTERN:
   • Layer 1: Exact match cache (dict/Redis) — instant, free
   • Layer 2: Semantic cache (embedding-based) — fast, cheap
   • Layer 3: LLM generation — slow, expensive, fresh answer
   • Log cache hits/misses to optimize thresholds over time

5. HEALTHCARE SAFETY:
   • Stale cache = stale clinical guidance = patient risk
   • Invalidate on guideline updates, drug recalls, protocol changes
   • Display "cached from [timestamp]" so clinicians know freshness
   • Alert admins when cache hit rate drops (KB expansion needed)
""")


if __name__ == "__main__":
    main()
