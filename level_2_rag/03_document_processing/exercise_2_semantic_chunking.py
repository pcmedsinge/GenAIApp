"""
Exercise 2: Semantic Chunking (Split at Topic Boundaries)
Implement a chunking strategy that splits text where the TOPIC changes,
not at arbitrary word counts or sentence counts.

Skills practiced:
- Using embeddings to detect topic shifts within a document
- Comparing semantic chunking vs fixed-size chunking
- Understanding sliding-window similarity for boundary detection
- Building a smarter chunking pipeline

Healthcare context:
  A clinical guideline flows from diagnosis → treatment → monitoring.
  Fixed-size chunks might cut right in the middle of a medication list.
  Semantic chunking detects where the topic shifts (e.g., from medications
  to lifestyle changes) and cuts THERE — keeping each chunk topically coherent.
"""

import os
import re
import numpy as np
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
# Clinical Guideline (same as main.py)
# ============================================================

CLINICAL_GUIDELINE = """
CLINICAL PRACTICE GUIDELINE: Management of Heart Failure

Section 1: Definition and Classification

Heart failure (HF) is a clinical syndrome caused by structural or functional cardiac
abnormalities resulting in reduced cardiac output or elevated intracardiac pressures.
Classification by ejection fraction: HFrEF (EF 40% or less), HFmrEF (EF 41-49%),
and HFpEF (EF 50% or more). The New York Heart Association (NYHA) functional
classification ranges from Class I (no limitation) to Class IV (symptoms at rest).

Section 2: Diagnosis

Diagnosis requires a combination of clinical history, physical exam, and diagnostic
testing. Key symptoms: dyspnea, orthopnea, paroxysmal nocturnal dyspnea, fatigue,
and peripheral edema. Essential tests include BNP or NT-proBNP (BNP greater than
100 pg/mL or NT-proBNP greater than 300 pg/mL suggests HF), echocardiography
to assess ejection fraction and structural abnormalities, ECG to identify arrhythmias
or ischemia, and chest X-ray to evaluate cardiomegaly and pulmonary congestion.

Section 3: Pharmacological Treatment for HFrEF

Guideline-directed medical therapy (GDMT) for HFrEF includes four pillars:
1. ACE inhibitor or ARB or ARNI (sacubitril-valsartan preferred if tolerated)
2. Beta-blocker (carvedilol, metoprolol succinate, or bisoprolol)
3. Mineralocorticoid receptor antagonist (spironolactone or eplerenone)
4. SGLT2 inhibitor (dapagliflozin or empagliflozin)

Titrate medications to target doses over weeks to months. Monitor blood pressure,
heart rate, kidney function, and potassium levels. Start low and go slow, especially
in elderly patients. Add hydralazine-isosorbide dinitrate for African American patients
or those intolerant to ACE/ARB/ARNI.

Section 4: Device Therapy

Consider ICD for primary prevention if EF 35% or less on optimal GDMT for 3 months.
CRT recommended for EF 35% or less with LBBB and QRS 150ms or more.
CRT may be considered for QRS 120-149ms with LBBB. Left ventricular assist device
(LVAD) for advanced HF as bridge to transplant or destination therapy.

Section 5: Lifestyle and Self-Management

Sodium restriction to less than 2000mg daily. Fluid restriction 1.5-2L daily for
patients with congestion. Daily weight monitoring (report gain of 2+ pounds overnight
or 5+ pounds in a week). Regular exercise: cardiac rehabilitation recommended.
Influenza and pneumococcal vaccination. Avoid NSAIDs, thiazolidinediones, and
most calcium channel blockers (except amlodipine) in HFrEF.

Section 6: Monitoring and Follow-up

Follow-up within 7-14 days after hospital discharge. Monitor BNP/NT-proBNP trends.
Reassess NYHA class and adjust therapy accordingly. Check renal function and
electrolytes 1-2 weeks after medication changes. Annual echocardiography to assess
EF trends. Consider palliative care referral for NYHA Class IV refractory symptoms.
"""


# ============================================================
# Embedding helper
# ============================================================

def get_embedding(text):
    """Get embedding vector from OpenAI"""
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding


def cosine_similarity(a, b):
    """Cosine similarity between two vectors"""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ============================================================
# Fixed-size chunking (for comparison)
# ============================================================

def chunk_fixed_size(text, chunk_size=100, overlap=20):
    """Simple fixed-size word-count chunking"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ============================================================
# SEMANTIC CHUNKING — The Core Algorithm
# ============================================================

def chunk_semantically(text, similarity_threshold=0.75, min_chunk_sentences=2):
    """
    Split text at topic boundaries using embedding similarity.

    Algorithm:
    1. Split text into sentences
    2. Embed each sentence
    3. Compare each consecutive pair of sentences
    4. Where similarity drops below threshold → topic boundary → split here
    5. Group sentences between boundaries into chunks

    This is a simplified version of what production systems use.
    """
    # Step 1: Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    if len(sentences) <= 2:
        return [text.strip()]

    # Step 2: Embed each sentence
    print(f"   Embedding {len(sentences)} sentences...")
    embeddings = []
    for sent in sentences:
        embeddings.append(get_embedding(sent))

    # Step 3: Calculate similarity between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    # Step 4: Find topic boundaries (where similarity drops)
    boundaries = []
    for i, sim in enumerate(similarities):
        if sim < similarity_threshold:
            boundaries.append(i + 1)  # Split AFTER this sentence

    # Step 5: Group sentences into chunks
    chunks = []
    start = 0
    for boundary in boundaries:
        chunk_sentences = sentences[start:boundary]
        if len(chunk_sentences) >= min_chunk_sentences:
            chunks.append(" ".join(chunk_sentences))
            start = boundary
        # If too few sentences, merge with next chunk

    # Add remaining sentences
    if start < len(sentences):
        remaining = " ".join(sentences[start:])
        if chunks and len(sentences[start:]) < min_chunk_sentences:
            # Merge tiny tail with last chunk
            chunks[-1] += " " + remaining
        else:
            chunks.append(remaining)

    return chunks, similarities, sentences, boundaries


# ============================================================
# Demo 1: Visualize similarity between sentences
# ============================================================

def demo_similarity_heatmap():
    """Show similarity scores between consecutive sentences"""
    print("\n" + "=" * 70)
    print("DEMO 1: SENTENCE SIMILARITY SCORES")
    print("=" * 70)

    print("\n⏳ Analyzing sentence-by-sentence similarity...")
    result = chunk_semantically(CLINICAL_GUIDELINE, similarity_threshold=0.75)
    chunks, similarities, sentences, boundaries = result

    print(f"\n📊 Consecutive sentence similarities ({len(similarities)} pairs):\n")

    for i, sim in enumerate(similarities):
        # Visual bar
        bar_len = int(sim * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        is_boundary = (i + 1) in boundaries
        marker = " ✂️  TOPIC SHIFT" if is_boundary else ""

        sent_preview = sentences[i][:50].replace('\n', ' ')
        print(f"   {i:>2}→{i+1:>2} | {sim:.3f} |{bar}|{marker}")

    print(f"""
💡 HOW TO READ THIS:
   • High similarity (0.85+): same topic, consecutive thoughts
   • Low similarity (< 0.75): topic is shifting → good split point
   • ✂️ marks where semantic chunking would cut
   • Compare this to fixed-size which cuts every N words regardless
""")


# ============================================================
# Demo 2: Compare semantic vs fixed-size chunks
# ============================================================

def demo_compare_chunking():
    """Side by side: semantic vs fixed-size chunks"""
    print("\n" + "=" * 70)
    print("DEMO 2: SEMANTIC vs FIXED-SIZE CHUNKS")
    print("=" * 70)

    # Semantic chunks
    print("\n⏳ Running semantic chunking...")
    result = chunk_semantically(CLINICAL_GUIDELINE, similarity_threshold=0.75)
    semantic_chunks = result[0]

    # Fixed-size chunks (100 words)
    fixed_chunks = chunk_fixed_size(CLINICAL_GUIDELINE, chunk_size=100, overlap=20)

    print(f"\n📑 SEMANTIC CHUNKS ({len(semantic_chunks)} chunks):")
    for i, chunk in enumerate(semantic_chunks):
        words = len(chunk.split())
        first_line = chunk[:100].replace('\n', ' ')
        print(f"   {i}. ({words:>3} words) \"{first_line}...\"")

    print(f"\n📏 FIXED-SIZE CHUNKS ({len(fixed_chunks)} chunks, 100 words each):")
    for i, chunk in enumerate(fixed_chunks):
        words = len(chunk.split())
        first_line = chunk[:100].replace('\n', ' ')
        print(f"   {i}. ({words:>3} words) \"{first_line}...\"")

    print("""
💡 KEY DIFFERENCES:
   • Semantic chunks vary in size (follows natural topic boundaries)
   • Fixed chunks are uniform size (may cut mid-topic)
   • Semantic chunks keep related info together (e.g., all medications in one chunk)
   • Fixed chunks are simpler to implement and predict
""")


# ============================================================
# Demo 3: Retrieval comparison
# ============================================================

def demo_retrieval_comparison():
    """Test both chunking strategies with clinical questions"""
    print("\n" + "=" * 70)
    print("DEMO 3: RETRIEVAL QUALITY — SEMANTIC vs FIXED")
    print("=" * 70)

    questions = [
        "What are the four medication pillars for heart failure?",
        "When should a patient get an ICD?",
        "What dietary restrictions apply to heart failure?",
        "How is heart failure diagnosed?",
        "What is the follow-up schedule after discharge?",
    ]

    # Build semantic chunks
    print("\n⏳ Building semantic chunks...")
    result = chunk_semantically(CLINICAL_GUIDELINE, similarity_threshold=0.75)
    semantic_chunks = result[0]

    # Build fixed chunks
    fixed_chunks = chunk_fixed_size(CLINICAL_GUIDELINE, chunk_size=100, overlap=20)

    # Index both in ChromaDB
    chroma_client = chromadb.Client()

    sem_coll = chroma_client.create_collection(name="semantic", embedding_function=openai_ef)
    sem_coll.add(
        ids=[f"sem_{i}" for i in range(len(semantic_chunks))],
        documents=semantic_chunks,
        metadatas=[{"strategy": "semantic", "index": i} for i in range(len(semantic_chunks))]
    )

    fix_coll = chroma_client.create_collection(name="fixed", embedding_function=openai_ef)
    fix_coll.add(
        ids=[f"fix_{i}" for i in range(len(fixed_chunks))],
        documents=fixed_chunks,
        metadatas=[{"strategy": "fixed", "index": i} for i in range(len(fixed_chunks))]
    )

    print(f"   Semantic: {len(semantic_chunks)} chunks | Fixed: {len(fixed_chunks)} chunks\n")

    # Compare retrieval
    semantic_wins = 0
    fixed_wins = 0

    for question in questions:
        print(f"{'─' * 70}")
        print(f"❓ \"{question}\"\n")

        sem_results = sem_coll.query(query_texts=[question], n_results=1)
        fix_results = fix_coll.query(query_texts=[question], n_results=1)

        sem_dist = sem_results["distances"][0][0]
        fix_dist = fix_results["distances"][0][0]
        sem_text = sem_results["documents"][0][0]
        fix_text = fix_results["documents"][0][0]

        winner = "SEMANTIC" if sem_dist < fix_dist else "FIXED"
        if sem_dist < fix_dist:
            semantic_wins += 1
        else:
            fixed_wins += 1

        print(f"   📑 Semantic: dist={sem_dist:.4f} ({len(sem_text.split())} words)")
        print(f"      \"{sem_text[:100]}...\"")
        print(f"   📏 Fixed:    dist={fix_dist:.4f} ({len(fix_text.split())} words)")
        print(f"      \"{fix_text[:100]}...\"")
        print(f"   → Winner: {winner}")

    print(f"\n{'─' * 70}")
    print(f"\n📊 SCORECARD: Semantic {semantic_wins} — Fixed {fixed_wins}")

    print("""
💡 WHEN SEMANTIC CHUNKING WINS:
   • Questions about a complete topic (e.g., "all four medications")
   • Structured documents with clear section boundaries
   • When answers shouldn't be split across chunks

   WHEN FIXED CHUNKING WINS:
   • Very long sections (semantic may produce oversized chunks)
   • Unstructured text without clear topic shifts
   • When you need predictable chunk sizes for token budgets
""")


# ============================================================
# Demo 4: Tuning the similarity threshold
# ============================================================

def demo_threshold_tuning():
    """Show how the threshold parameter affects chunk boundaries"""
    print("\n" + "=" * 70)
    print("DEMO 4: TUNING THE SIMILARITY THRESHOLD")
    print("=" * 70)

    thresholds = [0.65, 0.75, 0.85]

    for threshold in thresholds:
        print(f"\n{'─' * 70}")
        print(f"📐 Threshold: {threshold}")
        print(f"   (Split where consecutive similarity < {threshold})\n")

        result = chunk_semantically(
            CLINICAL_GUIDELINE,
            similarity_threshold=threshold,
            min_chunk_sentences=2
        )
        chunks = result[0]

        print(f"   Produced: {len(chunks)} chunks")
        sizes = [len(c.split()) for c in chunks]
        print(f"   Sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)} words")

        for i, chunk in enumerate(chunks):
            words = len(chunk.split())
            preview = chunk[:80].replace('\n', ' ')
            print(f"   {i}. ({words:>3} words) \"{preview}...\"")

    print("""
💡 THRESHOLD GUIDELINES:
   • LOW threshold (0.65): fewer splits → larger, broader chunks
     - Good for: documents with gradual topic transitions
   • MEDIUM threshold (0.75): balanced — standard starting point
     - Good for: most clinical guidelines
   • HIGH threshold (0.85): more splits → smaller, focused chunks
     - Good for: documents that jump between topics frequently

   HOW TO CHOOSE:
   1. Start at 0.75
   2. If chunks are too big → increase threshold
   3. If chunks are too small → decrease threshold
   4. Always validate with retrieval quality tests (Demo 3)
""")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n🧠 Exercise 2: Semantic Chunking")
    print("=" * 70)
    print("Split at topic boundaries, not arbitrary points\n")

    print("Choose a demo:")
    print("1. Sentence similarity scores (see where topics shift)")
    print("2. Compare semantic vs fixed-size chunks")
    print("3. Retrieval quality comparison (5 questions)")
    print("4. Tune the similarity threshold")
    print("5. Run all demos in sequence")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_similarity_heatmap()
    elif choice == "2":
        demo_compare_chunking()
    elif choice == "3":
        demo_retrieval_comparison()
    elif choice == "4":
        demo_threshold_tuning()
    elif choice == "5":
        demo_similarity_heatmap()
        demo_compare_chunking()
        demo_retrieval_comparison()
        demo_threshold_tuning()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY LEARNINGS — Exercise 2
{'=' * 70}

1. SEMANTIC CHUNKING ALGORITHM:
   • Split text into sentences → embed each → compare consecutive pairs
   • Where similarity drops below threshold → that's a topic boundary
   • Group sentences between boundaries into chunks

2. TRADEOFF:
   • Semantic: topic-coherent, variable size, higher embedding cost
   • Fixed: predictable size, simple, may split topics

3. SIMILARITY THRESHOLD:
   • Lower (0.65): fewer, larger chunks
   • Higher (0.85): more, smaller chunks
   • 0.75 is a good starting point for clinical text

4. COST CONSIDERATION:
   • Semantic chunking requires embedding EVERY SENTENCE first
   • For a 100-page document: thousands of embedding API calls
   • Fixed chunking: zero API calls for the chunking step itself
   • Use semantic when quality justifies the cost
""")


if __name__ == "__main__":
    main()
