# Level 8, Project 02: Caching & Optimization

## Overview

Reduce latency and cost for LLM-powered applications through **semantic
caching**, **prompt caching**, **embedding caches**, and **intelligent model
routing**. Every cached hit saves money and returns results in milliseconds
instead of seconds.

## Why This Matters

LLM API calls are expensive and slow compared to a cache lookup:

| Operation | Latency | Cost per query |
|-----------|---------|---------------|
| GPT-4o call | 2–10 s | ~$0.01–0.05 |
| Semantic cache hit | <50 ms | ~$0.00 |
| Embedding lookup | ~200 ms | ~$0.0001 |

A well-tuned caching layer can reduce costs by **60–80 %** on repetitive
medical workloads (patient FAQs, discharge instructions, drug info).

## Prerequisites

```bash
pip install openai numpy
```

You also need an `OPENAI_API_KEY` environment variable.

## Files

| File | Description |
|------|-------------|
| `main.py` | Four demos (exact cache, semantic cache, model routing, cost dashboard) |
| `exercise_1_semantic_cache.py` | Semantic cache for medical queries |
| `exercise_2_cache_strategies.py` | Compare exact / semantic / TTL / LRU caching |
| `exercise_3_model_router.py` | Route queries to the cheapest capable model |
| `exercise_4_cost_optimizer.py` | End-to-end cost optimization pipeline |

## Running

```bash
python main.py
```

## Key Concepts

1. **Exact-match cache** — hash the prompt, return stored answer on collision
2. **Semantic cache** — embed the query, return a cached answer if cosine
   similarity > threshold (e.g., 0.95)
3. **TTL (Time-To-Live)** — expire stale cache entries after N minutes
4. **LRU (Least Recently Used)** — evict the oldest entry when the cache is full
5. **Model routing** — analyze query complexity and pick the cheapest model
   that can handle it (gpt-4o-mini → gpt-4o → o1-mini)

## Healthcare Considerations

- Cached answers may become stale — set appropriate TTLs for clinical content
- Always log whether a response came from cache or a live model
- Model routing must not sacrifice safety — complex clinical queries need
  the most capable model regardless of cost

## Next Steps

After this project, continue with the remaining Level 8 modules to cover
logging, monitoring, containerisation, and CI/CD for production AI systems.
