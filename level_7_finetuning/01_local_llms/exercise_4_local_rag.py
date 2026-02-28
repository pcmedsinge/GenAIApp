"""
Exercise 4 — Complete Local RAG System
=======================================

Skills practiced
----------------
* Building a fully local RAG pipeline (zero data leaves the machine)
* Local embeddings with Ollama (nomic-embed-text)
* Local generation with Ollama (llama3 / mistral)
* Vector storage with ChromaDB
* Chunking, retrieval, and prompt engineering for RAG

Healthcare context
------------------
A hospital wants to build a clinical decision-support tool that answers
questions from its internal formulary and treatment guidelines — but NO
patient data or proprietary guidelines may be sent to a cloud API.  This
exercise demonstrates how to achieve that with a 100 % on-premise stack.

Usage
-----
    python exercise_4_local_rag.py

Prerequisites
-------------
    ollama pull llama3
    ollama pull nomic-embed-text
    pip install openai chromadb
"""

import sys
import time
import textwrap
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3"  # fallback order: llama3, mistral, phi3

COLLECTION_NAME = "hospital_guidelines_local"

# ---------------------------------------------------------------------------
# Sample medical guidelines (simulated internal hospital knowledge base)
# ---------------------------------------------------------------------------
GUIDELINES = [
    {
        "id": "abx-cap-1",
        "source": "Hospital Antibiotic Guideline v4.2",
        "text": (
            "Community-Acquired Pneumonia (CAP) — Outpatient, no comorbidities: "
            "Amoxicillin 1 g PO TID for 5 days.  Alternative: Doxycycline 100 mg PO BID "
            "for 5 days or Azithromycin 500 mg day 1 then 250 mg days 2-5."
        ),
    },
    {
        "id": "abx-cap-2",
        "source": "Hospital Antibiotic Guideline v4.2",
        "text": (
            "Community-Acquired Pneumonia (CAP) — Outpatient with comorbidities "
            "(COPD, diabetes, heart failure, renal disease): Amoxicillin-clavulanate "
            "875/125 mg PO BID PLUS Azithromycin 500 mg day 1 then 250 mg days 2-5.  "
            "Alternative: Respiratory fluoroquinolone (levofloxacin 750 mg PO daily "
            "for 5 days)."
        ),
    },
    {
        "id": "abx-cap-3",
        "source": "Hospital Antibiotic Guideline v4.2",
        "text": (
            "Community-Acquired Pneumonia (CAP) — Inpatient, non-ICU: "
            "Ceftriaxone 1-2 g IV daily PLUS Azithromycin 500 mg IV/PO daily.  "
            "If penicillin allergy: Respiratory fluoroquinolone monotherapy."
        ),
    },
    {
        "id": "abx-uti-1",
        "source": "Hospital Antibiotic Guideline v4.2",
        "text": (
            "Uncomplicated urinary tract infection (UTI) in women: "
            "Nitrofurantoin 100 mg PO BID for 5 days (first-line).  "
            "Alternative: TMP-SMX DS PO BID for 3 days if local resistance <20%.  "
            "Avoid fluoroquinolones for uncomplicated UTI."
        ),
    },
    {
        "id": "abx-uti-2",
        "source": "Hospital Antibiotic Guideline v4.2",
        "text": (
            "Complicated UTI or pyelonephritis — Outpatient: Ciprofloxacin 500 mg PO "
            "BID for 7 days or TMP-SMX DS PO BID for 14 days.  Inpatient: Ceftriaxone "
            "1 g IV daily; step down to PO based on culture sensitivity."
        ),
    },
    {
        "id": "anticoag-af-1",
        "source": "Hospital Anticoagulation Protocol v3.1",
        "text": (
            "Atrial fibrillation — stroke prevention: Calculate CHA₂DS₂-VASc score.  "
            "Score ≥2 in men or ≥3 in women: recommend DOAC (apixaban 5 mg BID preferred).  "
            "Dose-reduce apixaban to 2.5 mg BID if ≥2 of: age ≥80, weight ≤60 kg, "
            "creatinine ≥1.5 mg/dL."
        ),
    },
    {
        "id": "anticoag-af-2",
        "source": "Hospital Anticoagulation Protocol v3.1",
        "text": (
            "Warfarin is reserved for patients with mechanical heart valves or "
            "moderate-severe mitral stenosis.  Target INR 2.0-3.0 (2.5-3.5 for "
            "mechanical mitral valves).  Bridge with LMWH when initiating warfarin."
        ),
    },
    {
        "id": "dm-mgmt-1",
        "source": "Diabetes Management Protocol v2.8",
        "text": (
            "Type 2 diabetes — initial therapy: Metformin 500 mg PO BID, titrate to "
            "2000 mg/day as tolerated.  Target HbA1c <7% for most adults.  If eGFR "
            "<30 mL/min, discontinue metformin.  Add SGLT2i if ASCVD, HF, or CKD present."
        ),
    },
    {
        "id": "dm-mgmt-2",
        "source": "Diabetes Management Protocol v2.8",
        "text": (
            "Insulin initiation in Type 2 diabetes: Start basal insulin (glargine or "
            "degludec) 10 units at bedtime if HbA1c >10% or symptomatic hyperglycemia.  "
            "Titrate by 2 units every 3 days targeting fasting glucose 80-130 mg/dL."
        ),
    },
    {
        "id": "pain-mgmt-1",
        "source": "Pain Management Guideline v5.0",
        "text": (
            "Acute pain — stepwise approach: Step 1: Acetaminophen 1 g PO Q6H (max 3 g/day "
            "if liver disease, 4 g/day otherwise).  Step 2: Add NSAID (ibuprofen 400-600 mg "
            "PO TID) if no contraindications.  Step 3: Short-acting opioid (oxycodone 5 mg "
            "PO Q4-6H PRN) only if steps 1-2 inadequate; limit to 3 days when possible."
        ),
    },
]

# Predefined test queries
TEST_QUERIES = [
    "What antibiotic should I prescribe for outpatient pneumonia in a patient with COPD?",
    "My patient has atrial fibrillation — when should I use apixaban vs warfarin?",
    "What is the first-line treatment for an uncomplicated UTI?",
    "How do I start insulin in a patient with uncontrolled Type 2 diabetes?",
    "What is the pain management protocol for acute post-operative pain?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_client():
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai")
        sys.exit(1)
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    try:
        client.models.list()
    except Exception as exc:
        print(f"Cannot reach Ollama: {exc}\nRun: ollama serve")
        sys.exit(1)
    return client


def pick_gen_model(client) -> str:
    available = {m.id for m in client.models.list().data}
    for candidate in [GEN_MODEL, "mistral", "phi3"]:
        if candidate in available:
            return candidate
    print(f"⚠  No generation model found.  Pull one:  ollama pull {GEN_MODEL}")
    sys.exit(1)


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------
class LocalRAG:
    """Fully local RAG: Ollama embeddings + ChromaDB + Ollama generation."""

    def __init__(self, client, gen_model: str):
        try:
            import chromadb
        except ImportError:
            print("ERROR: pip install chromadb")
            sys.exit(1)

        self.client = client
        self.gen_model = gen_model
        self.chroma = chromadb.Client()
        self.collection = self.chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._indexed = False

    def index_documents(self, documents: list[dict]) -> None:
        """Embed and store documents locally."""
        texts = [d["text"] for d in documents]
        ids = [d["id"] for d in documents]
        metadatas = [{"source": d.get("source", "")} for d in documents]

        print(f"  Embedding {len(texts)} chunks with {EMBED_MODEL}...")
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=texts)
        embeddings = [d.embedding for d in resp.data]

        self.collection.add(
            ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
        )
        self._indexed = True
        print(f"  ✓ Indexed {len(texts)} chunks into ChromaDB (in-memory).")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve the most relevant chunks for a query."""
        q_resp = self.client.embeddings.create(model=EMBED_MODEL, input=query)
        q_vec = q_resp.data[0].embedding

        results = self.collection.query(query_embeddings=[q_vec], n_results=top_k)
        retrieved = []
        for doc, doc_id, meta in zip(
            results["documents"][0], results["ids"][0], results["metadatas"][0]
        ):
            retrieved.append({"id": doc_id, "text": doc, "source": meta.get("source", "")})
        return retrieved

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        """Generate an answer grounded in retrieved context."""
        ctx_block = "\n\n".join(
            f"[{c['id']}] ({c['source']}): {c['text']}" for c in context_chunks
        )
        prompt = (
            "You are a clinical decision-support assistant at a hospital.  "
            "Answer the clinician's question using ONLY the provided guidelines.  "
            "Cite guideline IDs when possible.\n\n"
            f"--- Guidelines ---\n{ctx_block}\n\n"
            f"--- Question ---\n{query}\n\n"
            "Provide a concise, actionable answer."
        )
        resp = self.client.chat.completions.create(
            model=self.gen_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        return resp.choices[0].message.content or ""

    def query(self, question: str, top_k: int = 3, verbose: bool = True) -> str:
        """Full RAG pipeline: retrieve → generate."""
        if verbose:
            print(f"\n  Query: {question}")

        start = time.time()
        chunks = self.retrieve(question, top_k=top_k)
        ret_time = time.time() - start

        if verbose:
            print(f"  Retrieved {len(chunks)} chunks ({ret_time:.2f}s):")
            for c in chunks:
                print(f"    • [{c['id']}] {c['text'][:70]}...")

        start = time.time()
        answer = self.generate(question, chunks)
        gen_time = time.time() - start

        if verbose:
            print(f"\n  Answer ({gen_time:.1f}s):\n{textwrap.indent(answer, '    ')}")

        return answer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Exercise 4: Complete Local RAG System")
    print("=" * 60)
    print("Everything runs locally — zero data leaves this machine.\n")

    client = get_client()
    gen_model = pick_gen_model(client)

    # Check embedding model
    available = {m.id for m in client.models.list().data}
    if EMBED_MODEL not in available:
        print(f"⚠  Embedding model '{EMBED_MODEL}' not found.  Run: ollama pull {EMBED_MODEL}")
        return

    print(f"Generation model : {gen_model}")
    print(f"Embedding model  : {EMBED_MODEL}")

    # --- Build index ---
    print(f"\n[1] Indexing {len(GUIDELINES)} hospital guideline chunks...")
    rag = LocalRAG(client, gen_model)
    rag.index_documents(GUIDELINES)

    # --- Run test queries ---
    print(f"\n[2] Running {len(TEST_QUERIES)} test queries...\n")
    for i, q in enumerate(TEST_QUERIES, 1):
        print(f"\n{'='*50}")
        print(f"Test Query {i}/{len(TEST_QUERIES)}")
        print(f"{'='*50}")
        rag.query(q)

    # --- Interactive mode ---
    print(f"\n{'='*60}")
    print("Interactive Mode — Ask your own questions")
    print("Type 'quit' to exit.")
    print(f"{'='*60}")
    while True:
        question = input("\nYou: ").strip()
        if not question or question.lower() in ("quit", "exit", "q"):
            break
        rag.query(question)

    print("\n✓ Session complete.  No data was transmitted externally.")


if __name__ == "__main__":
    main()
