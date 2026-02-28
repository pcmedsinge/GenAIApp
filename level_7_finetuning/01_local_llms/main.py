"""
Level 7 – Project 01: Local LLMs with Ollama
=============================================
Run large language models entirely on your own machine.
No data leaves the network — ideal for healthcare / PHI workloads.

Demos
-----
1. Ollama Basics          – OpenAI-compatible API via Ollama
2. Model Comparison       – Same healthcare prompt across local models
3. Local Embeddings       – nomic-embed-text for local vector search
4. Local RAG System       – Full RAG pipeline with zero cloud calls

Prerequisites
-------------
* Ollama installed and running  →  https://ollama.com
* Models pulled:  ollama pull llama3 && ollama pull mistral && ollama pull phi3
* pip install openai chromadb
"""

import json
import subprocess
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Helper: create an Ollama-backed OpenAI client
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"  # Ollama ignores the key, but the SDK requires one


def get_ollama_client():
    """Return an OpenAI client pointed at a local Ollama instance."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed.  Run:  pip install openai")
        sys.exit(1)

    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

    # Quick connectivity check
    try:
        client.models.list()
    except Exception as exc:
        print(
            "\n⚠  Cannot reach Ollama at", OLLAMA_BASE_URL,
            "\n   Make sure Ollama is running:  ollama serve",
            "\n   Install:  curl -fsSL https://ollama.com/install.sh | sh",
            f"\n   Error: {exc}\n",
        )
        sys.exit(1)

    return client


def pull_model(model_name: str) -> None:
    """Pull a model via the ollama CLI (shows progress)."""
    print(f"\n--- Pulling model '{model_name}' (this may take a while) ---")
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"✓ Model '{model_name}' is ready.\n")
    except FileNotFoundError:
        print("ERROR: 'ollama' CLI not found.  Install Ollama first.")
    except subprocess.CalledProcessError as exc:
        print(f"ERROR pulling model: {exc}")


def list_local_models() -> list[str]:
    """Return names of locally-available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split("\n")[1:]  # skip header
        return [line.split()[0] for line in lines if line.strip()]
    except Exception:
        return []


# ============================================================
# DEMO 1: Ollama Basics
# ============================================================
def demo_ollama_basics():
    """Show the fundamentals of calling Ollama through the OpenAI SDK."""
    print("\n" + "=" * 60)
    print("DEMO 1: Ollama Basics — OpenAI-Compatible Local Inference")
    print("=" * 60)

    client = get_ollama_client()

    # --- List available models ---
    print("\n[1] Models available locally:")
    models = client.models.list()
    for m in models.data:
        print(f"    • {m.id}")

    # --- Simple chat completion ---
    print("\n[2] Chat completion (llama3):")
    prompt = "Explain the difference between Type 1 and Type 2 diabetes in 3 sentences."
    print(f"    Prompt: {prompt}\n")

    start = time.time()
    response = client.chat.completions.create(
        model="llama3",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=300,
    )
    elapsed = time.time() - start

    answer = response.choices[0].message.content
    print(f"    Response ({elapsed:.1f}s):\n    {answer}")

    # --- Show usage stats ---
    if response.usage:
        print(f"\n    Tokens — prompt: {response.usage.prompt_tokens}, "
              f"completion: {response.usage.completion_tokens}, "
              f"total: {response.usage.total_tokens}")

    # --- Pulling a model programmatically ---
    print("\n[3] Pulling a model via subprocess:")
    print("    (Skipping actual pull to save time — see pull_model() helper)")
    print("    Example:  pull_model('phi3')")


# ============================================================
# DEMO 2: Model Comparison
# ============================================================
def demo_model_comparison():
    """Compare several local models on the same healthcare query."""
    print("\n" + "=" * 60)
    print("DEMO 2: Model Comparison — Local Models Head-to-Head")
    print("=" * 60)

    client = get_ollama_client()
    available = {m.id for m in client.models.list().data}

    candidate_models = ["llama3", "mistral", "phi3"]
    models_to_test = [m for m in candidate_models if m in available]

    if not models_to_test:
        print("\n⚠  None of the candidate models are pulled locally.")
        print(f"   Candidates: {candidate_models}")
        print("   Pull at least one:  ollama pull llama3")
        return

    query = (
        "A 55-year-old male presents with crushing chest pain radiating to the left arm, "
        "diaphoresis, and shortness of breath. What are the most likely diagnoses and "
        "immediate workup steps?"
    )
    system_msg = (
        "You are an experienced emergency medicine physician. "
        "Provide a concise differential diagnosis and initial workup plan."
    )

    print(f"\nQuery:\n  {query}\n")
    print(f"Models to compare: {models_to_test}\n")

    results = []
    for model_name in models_to_test:
        print(f"--- {model_name} ---")
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
                max_tokens=400,
            )
            elapsed = time.time() - start
            text = resp.choices[0].message.content
            tokens = resp.usage.total_tokens if resp.usage else "N/A"
            results.append({
                "model": model_name,
                "time_s": round(elapsed, 2),
                "tokens": tokens,
                "response": text,
            })
            # Print first 300 chars for brevity
            preview = text[:300] + ("..." if len(text) > 300 else "")
            print(f"  Time: {elapsed:.2f}s | Tokens: {tokens}")
            print(f"  Response preview: {preview}\n")
        except Exception as exc:
            print(f"  ERROR: {exc}\n")

    # Summary table
    if results:
        print("\n--- Comparison Summary ---")
        print(f"{'Model':<12} {'Time (s)':<10} {'Tokens':<8}")
        print("-" * 30)
        for r in results:
            print(f"{r['model']:<12} {r['time_s']:<10} {r['tokens']:<8}")


# ============================================================
# DEMO 3: Local Embeddings
# ============================================================
def demo_local_embeddings():
    """Generate embeddings locally with Ollama's nomic-embed-text model."""
    print("\n" + "=" * 60)
    print("DEMO 3: Local Embeddings — nomic-embed-text via Ollama")
    print("=" * 60)

    client = get_ollama_client()

    # Check if embedding model is available
    available = {m.id for m in client.models.list().data}
    embed_model = "nomic-embed-text"
    if embed_model not in available:
        print(f"\n⚠  Embedding model '{embed_model}' not found locally.")
        print(f"   Pull it first:  ollama pull {embed_model}")
        return

    # --- Single embedding ---
    sample_text = "Patient presents with acute myocardial infarction."
    print(f"\n[1] Embedding a single sentence:")
    print(f"    Text: \"{sample_text}\"")

    start = time.time()
    resp = client.embeddings.create(model=embed_model, input=sample_text)
    elapsed = time.time() - start

    vec = resp.data[0].embedding
    print(f"    Dimensions: {len(vec)}")
    print(f"    First 5 values: {vec[:5]}")
    print(f"    Time: {elapsed:.3f}s")

    # --- Batch embeddings + similarity ---
    print("\n[2] Batch embeddings + cosine similarity:")
    sentences = [
        "Patient has chest pain and elevated troponin",
        "The patient complains of severe headache and photophobia",
        "ECG shows ST-elevation in leads II, III, aVF",
        "Lab results indicate elevated cardiac enzymes",
        "MRI reveals a 2cm mass in the left frontal lobe",
    ]

    resp_batch = client.embeddings.create(model=embed_model, input=sentences)
    embeddings = [d.embedding for d in resp_batch.data]

    # Simple cosine similarity
    def cosine_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    print("\n    Similarity matrix (selected pairs):")
    pairs = [(0, 2), (0, 3), (0, 4), (1, 4), (2, 3)]
    for i, j in pairs:
        sim = cosine_sim(embeddings[i], embeddings[j])
        print(f"    [{i}] vs [{j}]  sim={sim:.4f}")
        print(f"        \"{sentences[i][:50]}...\"")
        print(f"        \"{sentences[j][:50]}...\"")

    print("\n    Notice: cardiac-related sentences cluster together.")


# ============================================================
# DEMO 4: Local RAG System
# ============================================================
def demo_local_rag():
    """Complete RAG pipeline using only local models — zero data leaves the machine."""
    print("\n" + "=" * 60)
    print("DEMO 4: Local RAG System — Fully Offline Retrieval-Augmented Generation")
    print("=" * 60)

    try:
        import chromadb
    except ImportError:
        print("\n⚠  chromadb not installed.  Run:  pip install chromadb")
        return

    client = get_ollama_client()
    embed_model = "nomic-embed-text"

    # Check models
    available = {m.id for m in client.models.list().data}
    gen_model = next((m for m in ["llama3", "mistral", "phi3"] if m in available), None)
    if not gen_model:
        print("⚠  No generation model available. Pull one: ollama pull llama3")
        return
    if embed_model not in available:
        print(f"⚠  Embedding model '{embed_model}' not pulled. Run: ollama pull {embed_model}")
        return

    # --- Build local knowledge base ---
    documents = [
        {
            "id": "guideline-htn-1",
            "text": (
                "Stage 1 hypertension is defined as systolic BP 130-139 mmHg or "
                "diastolic BP 80-89 mmHg. Initial management includes lifestyle "
                "modifications: DASH diet, sodium restriction (<2300 mg/day), "
                "regular aerobic exercise (150 min/week), weight loss if overweight, "
                "and limiting alcohol intake."
            ),
        },
        {
            "id": "guideline-htn-2",
            "text": (
                "Stage 2 hypertension is systolic BP ≥140 mmHg or diastolic ≥90 mmHg. "
                "Pharmacotherapy is recommended in addition to lifestyle changes. "
                "First-line agents include ACE inhibitors, ARBs, calcium channel blockers, "
                "and thiazide diuretics. Combination therapy may be needed."
            ),
        },
        {
            "id": "guideline-dm-1",
            "text": (
                "For Type 2 diabetes, metformin remains the first-line pharmacologic "
                "therapy. Target HbA1c is generally <7% for most adults. If HbA1c "
                "remains above target after 3 months, consider adding a second agent "
                "such as an SGLT2 inhibitor, GLP-1 receptor agonist, or DPP-4 inhibitor."
            ),
        },
        {
            "id": "guideline-dm-2",
            "text": (
                "SGLT2 inhibitors (e.g., empagliflozin, dapagliflozin) have shown "
                "cardiovascular and renal protective benefits in patients with Type 2 "
                "diabetes. They are preferred add-on agents in patients with established "
                "atherosclerotic cardiovascular disease, heart failure, or CKD."
            ),
        },
        {
            "id": "guideline-chol-1",
            "text": (
                "High-intensity statin therapy (atorvastatin 40-80 mg or rosuvastatin "
                "20-40 mg) is recommended for patients with clinical ASCVD. Target LDL "
                "reduction is ≥50% from baseline. If LDL remains ≥70 mg/dL on maximum "
                "statin, consider adding ezetimibe or a PCSK9 inhibitor."
            ),
        },
    ]

    print(f"\n[1] Building local vector store with {len(documents)} guideline chunks...")

    # Embed all documents locally
    texts = [d["text"] for d in documents]
    ids = [d["id"] for d in documents]

    embed_resp = client.embeddings.create(model=embed_model, input=texts)
    embeddings = [d.embedding for d in embed_resp.data]

    # Store in ChromaDB (in-memory)
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="local_medical_guidelines",
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(ids=ids, documents=texts, embeddings=embeddings)
    print("    ✓ Vector store ready (ChromaDB in-memory)")

    # --- Query ---
    query = "What medication should I start for a patient with Type 2 diabetes and heart failure?"
    print(f"\n[2] Query: \"{query}\"")

    # Embed query locally
    q_resp = client.embeddings.create(model=embed_model, input=query)
    q_vec = q_resp.data[0].embedding

    results = collection.query(query_embeddings=[q_vec], n_results=3)

    print("\n[3] Retrieved context (top-3 chunks):")
    context_chunks = []
    for i, (doc, doc_id) in enumerate(zip(results["documents"][0], results["ids"][0])):
        print(f"    [{i+1}] {doc_id}: {doc[:80]}...")
        context_chunks.append(doc)

    # --- Generate answer locally ---
    context_block = "\n\n".join(f"[Guideline {i+1}]: {c}" for i, c in enumerate(context_chunks))
    rag_prompt = (
        f"Based ONLY on the following clinical guidelines, answer the question.\n\n"
        f"Guidelines:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Provide a concise, evidence-based answer."
    )

    print(f"\n[4] Generating answer with '{gen_model}' (fully local)...")
    start = time.time()
    gen_resp = client.chat.completions.create(
        model=gen_model,
        messages=[
            {"role": "system", "content": "You are a clinical decision-support assistant. Answer only from the provided guidelines."},
            {"role": "user", "content": rag_prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    elapsed = time.time() - start
    answer = gen_resp.choices[0].message.content

    print(f"\n    Answer ({elapsed:.1f}s):\n    {answer}")
    print("\n    ✓ Entire pipeline ran locally — zero data sent to the cloud.")


# ============================================================
# Main Menu
# ============================================================
def main():
    """Interactive menu for Level 7, Project 01 demos."""
    demos = {
        "1": ("Ollama Basics", demo_ollama_basics),
        "2": ("Model Comparison", demo_model_comparison),
        "3": ("Local Embeddings", demo_local_embeddings),
        "4": ("Local RAG System", demo_local_rag),
    }

    while True:
        print("\n" + "=" * 60)
        print("LEVEL 7 · PROJECT 01 — Local LLMs with Ollama")
        print("=" * 60)
        for key, (title, _) in demos.items():
            print(f"  {key}. {title}")
        print("  q. Quit")

        choice = input("\nSelect a demo (1-4, q): ").strip().lower()
        if choice == "q":
            print("Goodbye!")
            break
        if choice in demos:
            demos[choice][1]()
        else:
            print("Invalid choice — try again.")


if __name__ == "__main__":
    main()
