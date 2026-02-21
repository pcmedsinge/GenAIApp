"""
Exercise 2: Experiment with Chunk Sizes
Compare how different chunk sizes (50, 100, 200 words) affect retrieval and answer quality.

Skills practiced:
- Understanding how chunk size impacts retrieval precision
- Measuring retrieval quality quantitatively
- Recognizing the chunk size trade-off (precision vs context)
- Systematic experimentation with RAG parameters

Healthcare context:
  Clinical guidelines have varying granularity — a drug dose is a few words,
  but a treatment protocol is a paragraph. Chunk size determines whether
  the RAG system retrieves precise snippets or broad context.
  Getting this right is crucial for clinical accuracy.
"""

import os
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# Knowledge Base (2 documents for focused comparison)
# ============================================================

DOCUMENTS = [
    {
        "id": "diabetes_type2",
        "title": "Type 2 Diabetes Management",
        "content": """Type 2 Diabetes diagnosis criteria: Fasting glucose 126 mg/dL or higher,
HbA1c 6.5 percent or higher, or random glucose 200 mg/dL or higher with symptoms.
First-line therapy: Metformin 500mg starting dose, titrate to 2000mg/day as tolerated.
HbA1c target: less than 7 percent for most adults, less than 8 percent for elderly or those with comorbidities.
If HbA1c not at target after 3 months on metformin, add second agent.
GLP-1 receptor agonists (semaglutide, liraglutide) preferred if cardiovascular disease or obesity.
SGLT2 inhibitors (empagliflozin, dapagliflozin) preferred if heart failure or CKD.
DPP-4 inhibitors (sitagliptin) if cost is a concern.
Monitor HbA1c every 3 months until stable, then every 6 months.
Annual screening: eye exam, foot exam, kidney function, lipid panel.
Patient education on hypoglycemia recognition and self-monitoring of blood glucose is essential.""",
        "category": "endocrinology"
    },
    {
        "id": "hypertension_guidelines",
        "title": "Hypertension Management Guidelines",
        "content": """Hypertension is defined as blood pressure consistently at or above 130/80 mmHg.
Stage 1 hypertension: 130-139/80-89 mmHg. Stage 2: 140/90 mmHg or higher.
First-line treatments include lifestyle modifications: dietary changes (DASH diet),
regular exercise (150 min/week moderate intensity), weight management,
sodium restriction (less than 2300mg/day), and limiting alcohol intake.
Pharmacological therapy first-line agents: ACE inhibitors (lisinopril, enalapril),
ARBs (losartan, valsartan), calcium channel blockers (amlodipine),
thiazide diuretics (hydrochlorothiazide).
Target BP for most adults: less than 130/80 mmHg. For patients over 65: less than 130/80 if tolerated.
Monitor kidney function and electrolytes when starting ACE inhibitor or ARB therapy.
Combination therapy is often needed for Stage 2 or if target not met with monotherapy.
Black patients may respond better to calcium channel blockers or thiazides as initial therapy.""",
        "category": "cardiology"
    },
    {
        "id": "anticoagulation",
        "title": "Anticoagulation Therapy Guidelines",
        "content": """Indications for anticoagulation: Atrial fibrillation (CHA2DS2-VASc score 2 or higher in men,
3 or higher in women), VTE (DVT or PE) treatment and prevention,
mechanical heart valves, antiphospholipid syndrome.
DOACs preferred over warfarin for most non-valvular atrial fibrillation:
apixaban 5mg twice daily, rivaroxaban 20mg daily with food,
dabigatran 150mg twice daily, edoxaban 60mg daily.
Dose adjustments for renal impairment: Apixaban reduced to 2.5mg twice daily
if 2 of 3 criteria met (age 80 or older, weight 60kg or less, creatinine 1.5 or higher).
Dabigatran contraindicated if creatinine clearance below 30.
Warfarin: Target INR 2-3 for atrial fibrillation, 2.5-3.5 for mechanical valves.
Reversal agents: Idarucizumab for dabigatran, andexanet alfa for factor Xa inhibitors,
vitamin K plus fresh frozen plasma or PCC for warfarin.
Assess bleeding risk with HAS-BLED score before starting anticoagulation.""",
        "category": "hematology"
    }
]


# ============================================================
# RAG Infrastructure
# ============================================================

def simple_chunk(text, chunk_size=80, overlap=15):
    """Split text into overlapping word-based chunks"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding vector"""
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def cosine_similarity(vec_a, vec_b):
    """Cosine similarity between two vectors"""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_index(documents, chunk_size, overlap):
    """Build index with specified chunk size"""
    index = []
    for doc in documents:
        chunks = simple_chunk(doc["content"], chunk_size=chunk_size, overlap=overlap)
        for i, chunk_text in enumerate(chunks):
            embedding = get_embedding(chunk_text)
            index.append({
                "doc_id": doc["id"],
                "title": doc["title"],
                "chunk_index": i,
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
                "embedding": embedding
            })
    return index


def retrieve(query, index, top_k=3):
    """Find most relevant chunks"""
    query_embedding = get_embedding(query)
    scored = []
    for item in index:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append({**item, "similarity": score})
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


def generate_answer(query, retrieved_chunks):
    """LLM answers using retrieved context"""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(f"[Source {i+1}: {chunk['title']}]\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a medical assistant. Answer using ONLY the provided context.
If context is insufficient, say so. Cite sources as [Source X]. Educational purposes only."""},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ],
        max_tokens=300, temperature=0.2
    )
    return response.choices[0].message.content, response.usage


# ============================================================
# Demo 1: Chunk Size Comparison
# ============================================================

def demo_chunk_comparison():
    """Compare 3 chunk sizes on the same questions"""
    print("\n" + "=" * 70)
    print("DEMO 1: CHUNK SIZE COMPARISON (50 vs 100 vs 200 words)")
    print("=" * 70)

    chunk_configs = [
        {"size": 50,  "overlap": 10, "label": "SMALL (50 words)"},
        {"size": 100, "overlap": 20, "label": "MEDIUM (100 words)"},
        {"size": 200, "overlap": 40, "label": "LARGE (200 words)"},
    ]

    test_questions = [
        # Specific question — small chunks should be more precise
        "What is the starting dose of metformin?",
        # Broad question — large chunks may provide better context
        "How should you treat a patient with both diabetes and heart failure?",
        # Medium question
        "What are the first-line medications for hypertension?",
    ]

    # Build indices for each chunk size
    indices = {}
    for config in chunk_configs:
        print(f"\n🔨 Building index: {config['label']}...")
        index = build_index(DOCUMENTS, config["size"], config["overlap"])
        indices[config["label"]] = index
        print(f"   → {len(index)} chunks total")

    # Show chunk distribution
    print(f"\n{'─' * 70}")
    print("📊 CHUNK STATISTICS:")
    print(f"{'─' * 70}")
    for label, index in indices.items():
        word_counts = [item["word_count"] for item in index]
        print(f"   {label}: {len(index)} chunks, avg {np.mean(word_counts):.0f} words/chunk")

    # Test each question across all chunk sizes
    for q in test_questions:
        print(f"\n{'═' * 70}")
        print(f"❓ {q}")
        print(f"{'═' * 70}")

        for label, index in indices.items():
            retrieved = retrieve(q, index, top_k=2)
            top_sim = retrieved[0]["similarity"]
            total_context_words = sum(len(r["text"].split()) for r in retrieved)

            print(f"\n   📦 {label}")
            print(f"      Top similarity: {top_sim:.4f}")
            print(f"      Context words: {total_context_words}")
            print(f"      Top chunk: \"{retrieved[0]['text'][:120]}...\"")

            answer, usage = generate_answer(q, retrieved)
            print(f"      📋 Answer: {answer[:200]}...")
            print(f"      Tokens: {usage.total_tokens}")

    print(f"""
{'═' * 70}
💡 CHUNK SIZE TRADE-OFFS:

   📦 SMALL CHUNKS (50 words):
      ✅ More PRECISE retrieval (less noise)
      ✅ Faster embedding creation
      ❌ May miss surrounding context
      ❌ More chunks = more storage

   📦 MEDIUM CHUNKS (100 words):
      ✅ Good balance of precision and context
      ✅ Usually the sweet spot for clinical text
      ⚖️ Reasonable storage and cost

   📦 LARGE CHUNKS (200 words):
      ✅ More CONTEXT in each chunk
      ✅ Fewer chunks = less storage
      ❌ More noise in retrieval (irrelevant info mixed in)
      ❌ Uses more prompt tokens (higher cost)

   🎯 RECOMMENDATION for healthcare:
      80-120 words is usually optimal for clinical guidelines.
      Adjust based on your specific content structure.
""")


# ============================================================
# Demo 2: Overlap Impact
# ============================================================

def demo_overlap_impact():
    """Show how overlap affects chunking"""
    print("\n" + "=" * 70)
    print("DEMO 2: OVERLAP IMPACT ON CHUNKS")
    print("=" * 70)

    sample_text = DOCUMENTS[0]["content"]  # Diabetes guidelines

    overlap_configs = [
        {"overlap": 0,  "label": "No overlap"},
        {"overlap": 10, "label": "10-word overlap"},
        {"overlap": 25, "label": "25-word overlap"},
    ]

    chunk_size = 50

    for config in overlap_configs:
        chunks = simple_chunk(sample_text, chunk_size=chunk_size, overlap=config["overlap"])
        print(f"\n📦 {config['label']} (chunk size: {chunk_size} words)")
        print(f"   → {len(chunks)} chunks")

        # Show first 3 chunks and where they overlap
        for i, chunk in enumerate(chunks[:3]):
            words = chunk.split()
            print(f"\n   Chunk {i}: ({len(words)} words)")
            print(f"   \"{chunk[:100]}...\"")

            if i > 0 and config["overlap"] > 0:
                prev_words = chunks[i-1].split()
                curr_words = chunk.split()
                # Find overlap
                overlap_words = []
                for w in curr_words[:config["overlap"]]:
                    if w in prev_words[-config["overlap"]:]:
                        overlap_words.append(w)
                if overlap_words:
                    print(f"   🔗 Overlap with prev chunk: \"{' '.join(overlap_words[:8])}...\"")

    print(f"""
{'═' * 70}
💡 OVERLAP INSIGHTS:

   🔗 No Overlap:     Information at boundaries may be LOST
   🔗 Small Overlap:  Good safety net without too many chunks
   🔗 Large Overlap:  Maximum coverage but many more chunks (cost + storage)

   🎯 Rule of thumb: overlap = 15-25% of chunk size
""")


# ============================================================
# Demo 3: Detailed Side-by-Side
# ============================================================

def demo_side_by_side():
    """Show same question answered by small vs large chunks"""
    print("\n" + "=" * 70)
    print("DEMO 3: SIDE-BY-SIDE ANSWER QUALITY")
    print("=" * 70)

    question = "What second-line diabetes medications should I consider for a patient with obesity?"

    print(f"\n❓ {question}\n")

    for size, overlap, label in [(50, 10, "SMALL"), (200, 40, "LARGE")]:
        print(f"{'─' * 70}")
        print(f"📦 {label} CHUNKS ({size} words, overlap {overlap}):")
        index = build_index(DOCUMENTS, size, overlap)
        print(f"   Built: {len(index)} chunks\n")

        retrieved = retrieve(question, index, top_k=3)
        for i, r in enumerate(retrieved):
            print(f"   Chunk {i+1} (sim: {r['similarity']:.4f}, {r['word_count']} words):")
            print(f"      \"{r['text'][:150]}...\"\n")

        answer, usage = generate_answer(question, retrieved)
        print(f"   📋 ANSWER:\n   {answer}")
        print(f"\n   📊 Tokens used: {usage.total_tokens}\n")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n📏 Exercise 2: Chunk Size Experiments")
    print("=" * 70)
    print("How does chunk size affect retrieval and answer quality?\n")

    print("Choose a demo:")
    print("1. Chunk size comparison (50 vs 100 vs 200)")
    print("2. Overlap impact")
    print("3. Side-by-side answer quality")
    print("4. Run all demos")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        demo_chunk_comparison()
    elif choice == "2":
        demo_overlap_impact()
    elif choice == "3":
        demo_side_by_side()
    elif choice == "4":
        demo_chunk_comparison()
        demo_overlap_impact()
        demo_side_by_side()
    else:
        print("Invalid choice")

    print(f"""
{'═' * 70}
KEY LEARNINGS:
{'═' * 70}
1. Chunk size is one of the most important RAG hyperparameters
2. Small chunks → precise but may lack context
3. Large chunks → rich context but more noise and higher cost
4. Overlap prevents losing information at chunk boundaries
5. 80-120 words is a good starting point for clinical text
6. Always test with YOUR specific data and questions

🏥 CLINICAL INSIGHT:
   Drug doses → small chunks (need precision)
   Treatment protocols → medium chunks (need some context)
   Differential diagnosis → larger chunks (need full reasoning)

   There's no "one size fits all" — match chunk size to your use case!
""")


if __name__ == "__main__":
    main()
