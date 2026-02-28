"""
Exercise 2: Integrated RAG Service
====================================
Integrate RAG into the platform: document ingestion, cached queries,
evaluation endpoint. Add monitoring to every call.

Requirements:
- Document ingestion with embedding generation
- Cached query endpoint (check cache before calling LLM)
- Evaluation endpoint to score RAG quality
- Full monitoring on every RAG call (tokens, cost, latency)
- Integration with platform monitoring infrastructure

Healthcare Context:
  Hospital knowledge bases contain clinical guidelines, drug formularies,
  and protocol documents. RAG ensures answers are grounded in approved
  sources, not hallucinated.

Usage:
    python exercise_2_integrated_rag.py
"""

from openai import OpenAI
import time
import json
import os
import hashlib
import numpy as np
from datetime import datetime
from collections import defaultdict

client = OpenAI()

MODEL_COSTS = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "text-embedding-3-small": {"input": 0.00002, "output": 0},
}


class RAGMonitor:
    """Monitoring for RAG pipeline operations."""

    def __init__(self):
        self.logs = []
        self.total_cost = 0.0

    def log(self, operation: str, tokens: int, cost: float, latency_ms: float,
            cache_hit: bool = False, **kwargs):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "tokens": tokens,
            "cost_usd": cost,
            "latency_ms": round(latency_ms, 2),
            "cache_hit": cache_hit,
            **kwargs,
        }
        self.logs.append(entry)
        self.total_cost += cost

    def summary(self) -> dict:
        if not self.logs:
            return {"total_operations": 0}
        return {
            "total_operations": len(self.logs),
            "total_cost": round(self.total_cost, 8),
            "total_tokens": sum(l["tokens"] for l in self.logs),
            "cache_hits": sum(1 for l in self.logs if l["cache_hit"]),
            "avg_latency_ms": round(sum(l["latency_ms"] for l in self.logs) / len(self.logs), 1),
        }


class ResponseCache:
    """Simple hash-based response cache."""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self.store = {}

    def make_key(self, query: str) -> str:
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str):
        key = self.make_key(query)
        if key in self.store:
            entry = self.store[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["data"]
            del self.store[key]
        return None

    def set(self, query: str, data: dict):
        key = self.make_key(query)
        self.store[key] = {"data": data, "timestamp": time.time()}


class DocumentStore:
    """Simple in-memory document store with embeddings."""

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def ingest(self, text: str, metadata: dict = None) -> dict:
        """Ingest a document: chunk, embed, and store."""
        # Simple chunking by paragraphs
        chunks = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 20]
        if not chunks:
            chunks = [text]

        ingested = []
        for chunk in chunks:
            # Generate embedding
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk,
            )
            embedding = response.data[0].embedding

            doc = {
                "id": len(self.documents),
                "content": chunk,
                "metadata": metadata or {},
                "embedding": embedding,
                "ingested_at": datetime.now().isoformat(),
            }
            self.documents.append(doc)
            self.embeddings.append(embedding)
            ingested.append(doc["id"])

        return {"chunks_ingested": len(ingested), "doc_ids": ingested}

    def search(self, query: str, top_k: int = 3) -> list:
        """Search for relevant documents using embedding similarity."""
        if not self.documents:
            return []

        # Embed the query
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
        )
        query_embedding = response.data[0].embedding

        # Calculate cosine similarity
        scores = []
        q_arr = np.array(query_embedding)
        q_norm = np.linalg.norm(q_arr)

        for i, emb in enumerate(self.embeddings):
            e_arr = np.array(emb)
            similarity = np.dot(q_arr, e_arr) / (q_norm * np.linalg.norm(e_arr))
            scores.append((i, float(similarity)))

        scores.sort(key=lambda x: -x[1])
        results = []
        for idx, score in scores[:top_k]:
            doc = self.documents[idx]
            results.append({
                "id": doc["id"],
                "content": doc["content"],
                "score": round(score, 4),
                "metadata": doc["metadata"],
            })

        return results


class IntegratedRAGService:
    """RAG service with caching, monitoring, and evaluation."""

    def __init__(self):
        self.doc_store = DocumentStore()
        self.cache = ResponseCache(ttl_seconds=300)
        self.monitor = RAGMonitor()

    def ingest_document(self, text: str, source: str = "unknown") -> dict:
        """Ingest a document into the RAG system."""
        start = time.time()
        result = self.doc_store.ingest(text, metadata={"source": source})
        latency = (time.time() - start) * 1000

        # Estimate embedding cost
        tokens_est = len(text.split()) * 1.3
        cost = (tokens_est / 1000) * MODEL_COSTS["text-embedding-3-small"]["input"]

        self.monitor.log("ingest", int(tokens_est), cost, latency,
                         chunks=result["chunks_ingested"], source=source)
        return result

    def query(self, query: str, user: str = "anonymous", top_k: int = 3) -> dict:
        """Query the RAG system with caching and monitoring."""
        # Check cache first
        cached = self.cache.get(query)
        if cached:
            self.monitor.log("query", 0, 0.0, 0.5, cache_hit=True, user=user)
            return {**cached, "cached": True}

        start = time.time()

        # Retrieve relevant documents
        results = self.doc_store.search(query, top_k=top_k)

        if not results:
            return {"response": "No relevant documents found.", "sources": [], "cached": False}

        context = "\n\n".join(f"[Source {r['id']}] {r['content']}" for r in results)

        # Generate response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a medical RAG assistant. Answer using ONLY the provided context. "
                    "Cite source IDs in brackets [Source X]. If the context doesn't contain "
                    "relevant information, say so. Always add a medical disclaimer."
                )},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        )

        latency = (time.time() - start) * 1000
        pt = response.usage.prompt_tokens
        ct = response.usage.completion_tokens
        rates = MODEL_COSTS["gpt-4o-mini"]
        cost = (pt / 1000) * rates["input"] + (ct / 1000) * rates["output"]

        result = {
            "response": response.choices[0].message.content,
            "sources": results,
            "tokens": pt + ct,
            "cost_usd": round(cost, 8),
            "latency_ms": round(latency, 2),
            "cached": False,
        }

        # Cache the result
        self.cache.set(query, result)
        self.monitor.log("query", pt + ct, cost, latency, user=user)

        return result

    def evaluate_response(self, query: str, response: str, sources: list) -> dict:
        """Evaluate RAG response quality."""
        start = time.time()
        eval_prompt = (
            f"Evaluate this RAG response for quality:\n\n"
            f"Query: {query}\n"
            f"Response: {response}\n"
            f"Sources used: {json.dumps([s['content'][:100] for s in sources])}\n\n"
            f"Rate on a scale of 1-5 for:\n"
            f"1. Relevance (does it answer the query?)\n"
            f"2. Groundedness (is it based on the provided sources?)\n"
            f"3. Completeness (does it fully address the question?)\n"
            f"4. Safety (does it include appropriate disclaimers?)\n\n"
            f"Respond in JSON: {{\"relevance\": X, \"groundedness\": X, \"completeness\": X, \"safety\": X, \"notes\": \"...\"}}"
        )

        eval_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical AI evaluator. Be strict and objective."},
                {"role": "user", "content": eval_prompt},
            ],
        )
        latency = (time.time() - start) * 1000
        self.monitor.log("evaluate", eval_response.usage.total_tokens, 0.0, latency)

        try:
            content = eval_response.choices[0].message.content
            # Extract JSON from response
            json_match = content[content.find("{"):content.rfind("}") + 1]
            scores = json.loads(json_match)
        except (json.JSONDecodeError, ValueError):
            scores = {"relevance": "N/A", "notes": "Failed to parse evaluation"}

        return scores


def main():
    """Run the integrated RAG exercise."""
    print("=" * 60)
    print("  Exercise 2: Integrated RAG Service")
    print("=" * 60)

    rag = IntegratedRAGService()

    # Step 1: Ingest medical documents
    print("\n--- Step 1: Document Ingestion ---")
    documents = [
        {
            "text": (
                "Hypertension Management Guidelines\n\n"
                "Stage 1 hypertension is defined as blood pressure 130-139/80-89 mmHg. "
                "First-line treatments include thiazide diuretics, ACE inhibitors, "
                "ARBs, and calcium channel blockers.\n\n"
                "Stage 2 hypertension is defined as blood pressure >= 140/90 mmHg. "
                "Combination therapy is often required. Regular monitoring every 1-3 months "
                "is recommended until blood pressure is at goal."
            ),
            "source": "clinical_guidelines_hypertension",
        },
        {
            "text": (
                "Diabetes Type 2 Management Protocol\n\n"
                "Metformin is the first-line pharmacological therapy for Type 2 diabetes. "
                "Starting dose is typically 500mg once daily, titrated to 1000mg twice daily.\n\n"
                "HbA1c target is generally < 7.0% for most adults. "
                "SGLT2 inhibitors or GLP-1 receptor agonists are recommended as add-on therapy "
                "for patients with established cardiovascular disease."
            ),
            "source": "clinical_guidelines_diabetes",
        },
        {
            "text": (
                "Drug Interaction: Warfarin\n\n"
                "Warfarin interacts with numerous medications and foods. "
                "NSAIDs increase bleeding risk and should be avoided. "
                "Vitamin K-rich foods (leafy greens) can reduce warfarin effectiveness.\n\n"
                "INR monitoring is required regularly. Target INR is typically 2.0-3.0 "
                "for most indications. Significant interactions include: amiodarone, "
                "fluoroquinolones, metronidazole, and azole antifungals."
            ),
            "source": "drug_interactions_warfarin",
        },
    ]

    for doc in documents:
        result = rag.ingest_document(doc["text"], source=doc["source"])
        print(f"  Ingested '{doc['source']}': {result['chunks_ingested']} chunks")

    print(f"  Total documents in store: {len(rag.doc_store.documents)}")

    # Step 2: Query the system
    print("\n--- Step 2: RAG Queries ---")
    queries = [
        ("What is the first-line treatment for Stage 1 hypertension?", "dr_smith"),
        ("What medications interact with warfarin?", "pharmacist_lee"),
        ("What is the HbA1c target for Type 2 diabetes?", "nurse_chen"),
        ("What is the first-line treatment for Stage 1 hypertension?", "dr_jones"),  # Cache test
    ]

    for query, user in queries:
        print(f"\n  Query: {query}")
        result = rag.query(query, user=user)
        print(f"    Cached:  {result['cached']}")
        if not result["cached"]:
            print(f"    Tokens:  {result.get('tokens', 'N/A')}")
            print(f"    Latency: {result.get('latency_ms', 'N/A')}ms")
            print(f"    Sources: {len(result.get('sources', []))} documents")
        print(f"    Response: {result['response'][:120]}...")

    # Step 3: Evaluate response quality
    print("\n--- Step 3: Response Evaluation ---")
    test_result = rag.query("What is the first-line treatment for Type 2 diabetes?", user="evaluator")
    if not test_result["cached"]:
        scores = rag.evaluate_response(
            "What is the first-line treatment for Type 2 diabetes?",
            test_result["response"],
            test_result.get("sources", []),
        )
        print(f"  Evaluation Scores:")
        for key, val in scores.items():
            print(f"    {key}: {val}")

    # Step 4: Monitoring summary
    print("\n--- Step 4: Monitoring Summary ---")
    summary = rag.monitor.summary()
    for key, val in summary.items():
        print(f"  {key}: {val}")

    print("\nIntegrated RAG exercise complete!")


if __name__ == "__main__":
    main()
