"""
Exercise 1: Chunk Size Comparison (50, 100, 200 Words)
Chunk the same clinical guideline with different sizes and compare how each
affects retrieval quality for the same set of clinical questions.

Skills practiced:
- Understanding chunk size as a tunable parameter
- Observing the too-small vs too-large tradeoff
- Comparing retrieval similarity scores across chunk sizes
- Using ChromaDB to test chunking decisions quickly

Healthcare context:
  A 50-word chunk about heart failure medication might capture only one drug.
  A 200-word chunk captures the full four-pillar therapy but also includes
  noise about titration and monitoring. The RIGHT size depends on the type
  of questions clinicians will ask. This exercise makes the tradeoff visible.
"""

import os
import re
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
# Clinical Guideline Document (from main.py)
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
# Chunking function (fixed-size with overlap)
# ============================================================

def chunk_fixed_size(text, chunk_size=100, overlap=20):
    """Split text into chunks of approximately chunk_size words with overlap"""
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
# Demo 1: Visualize chunks at different sizes
# ============================================================

def demo_visualize_chunk_sizes():
    """See how the document splits at 50, 100, and 200 words"""
    print("\n" + "=" * 70)
    print("DEMO 1: CHUNK SIZE VISUALIZATION")
    print("=" * 70)

    sizes = [
        (50, 10, "Small — granular, many chunks"),
        (100, 20, "Medium — balanced, standard choice"),
        (200, 40, "Large — broad context, fewer chunks"),
    ]

    for chunk_size, overlap, description in sizes:
        chunks = chunk_fixed_size(CLINICAL_GUIDELINE, chunk_size, overlap)
        print(f"\n📏 {chunk_size}-WORD CHUNKS (overlap={overlap}): {description}")
        print(f"   Produced: {len(chunks)} chunks\n")

        for i, chunk in enumerate(chunks[:3]):
            word_count = len(chunk.split())
            print(f"   Chunk {i} ({word_count} words): \"{chunk[:100]}...\"")
        if len(chunks) > 3:
            print(f"   ... ({len(chunks) - 3} more chunks)")

    print("""
💡 OBSERVATION:
   • 50-word: many small pieces — each covers a narrow topic
   • 100-word: moderate pieces — covers ~1 paragraph
   • 200-word: large pieces — may span multiple topics
   • More chunks = more API calls to embed (cost consideration)
""")


# ============================================================
# Demo 2: Retrieval comparison across sizes
# ============================================================

def demo_retrieval_comparison():
    """Same questions, three chunk sizes — compare retrieval quality"""
    print("\n" + "=" * 70)
    print("DEMO 2: RETRIEVAL QUALITY — 50 vs 100 vs 200 WORDS")
    print("=" * 70)

    test_questions = [
        "What are the four pillars of heart failure medication?",
        "When should a patient get an ICD?",
        "What dietary restrictions apply to heart failure patients?",
        "How is heart failure diagnosed?",
        "What is the NYHA classification?",
    ]

    sizes = [(50, 10), (100, 20), (200, 40)]

    # Build ChromaDB collections for each size
    chroma_client = chromadb.Client()
    collections = {}

    print("\n⏳ Building collections for each chunk size...")
    for chunk_size, overlap in sizes:
        chunks = chunk_fixed_size(CLINICAL_GUIDELINE, chunk_size, overlap)
        name = f"chunks_{chunk_size}"
        coll = chroma_client.create_collection(name=name, embedding_function=openai_ef)
        coll.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"chunk_size": chunk_size, "chunk_index": i} for i in range(len(chunks))]
        )
        collections[chunk_size] = coll
        print(f"   {chunk_size}-word: {len(chunks)} chunks indexed")

    # Test each question
    for question in test_questions:
        print(f"\n{'─' * 70}")
        print(f"❓ \"{question}\"\n")

        for chunk_size, _ in sizes:
            results = collections[chunk_size].query(
                query_texts=[question],
                n_results=1
            )
            dist = results["distances"][0][0]
            text = results["documents"][0][0]
            word_count = len(text.split())
            print(f"   📏 {chunk_size:>3}-word | dist: {dist:.4f} | ({word_count} words)")
            print(f"               \"{text[:120]}...\"")

    print("""
💡 WHAT TO NOTICE:
   • Distance scores show how "close" the best chunk is to the question
   • Lower distance = better match
   • 50-word chunks may miss part of the answer (too narrow)
   • 200-word chunks may include irrelevant text (too broad)
   • 100-word is often the sweet spot for clinical guidelines
""")


# ============================================================
# Demo 3: Top-K sensitivity to chunk size
# ============================================================

def demo_topk_sensitivity():
    """Show how chunk size affects what you need for top_k"""
    print("\n" + "=" * 70)
    print("DEMO 3: TOP-K SENSITIVITY TO CHUNK SIZE")
    print("=" * 70)

    question = "What medications are used to treat heart failure?"
    print(f"\n❓ Question: \"{question}\"")
    print("   (Answer spans medication names, doses, AND titration advice)\n")

    chroma_client = chromadb.Client()

    for chunk_size, overlap in [(50, 10), (100, 20), (200, 40)]:
        chunks = chunk_fixed_size(CLINICAL_GUIDELINE, chunk_size, overlap)
        coll = chroma_client.create_collection(
            name=f"topk_{chunk_size}",
            embedding_function=openai_ef
        )
        coll.add(
            ids=[f"c_{i}" for i in range(len(chunks))],
            documents=chunks
        )

        # Get top 3 results
        results = coll.query(query_texts=[question], n_results=3)

        print(f"📏 {chunk_size}-WORD CHUNKS (top 3):")
        total_words = 0
        for i in range(len(results["ids"][0])):
            text = results["documents"][0][i]
            wc = len(text.split())
            total_words += wc
            print(f"   {i+1}. dist: {results['distances'][0][i]:.4f} "
                  f"({wc} words): \"{text[:80]}...\"")
        print(f"   → Total context: {total_words} words from top-3\n")

    print("""
💡 INSIGHT:
   • 50-word × top-3 = ~150 words of context (compact but scattered)
   • 100-word × top-3 = ~300 words of context (good coverage)
   • 200-word × top-3 = ~600 words of context (broad, possibly noisy)
   
   The LLM has a context window limit — bigger chunks means fewer 
   can fit. Smaller chunks means you need higher top_k to cover 
   the same ground. This is a fundamental tradeoff.
""")


# ============================================================
# Demo 4: Guidelines for choosing chunk size
# ============================================================

def demo_guidelines():
    """Print recommendations for chunk size selection"""
    print("\n" + "=" * 70)
    print("DEMO 4: CHUNK SIZE SELECTION GUIDE")
    print("=" * 70)

    print("""
    ┌─────────────────────┬─────────┬─────────────────────────────────────┐
    │ Document Type        │ Size    │ Why                                 │
    ├─────────────────────┼─────────┼─────────────────────────────────────┤
    │ Clinical guidelines  │ 100-200 │ Each section covers one topic well   │
    │ Progress notes       │ 50-100  │ Short, dense; each sentence matters  │
    │ Research papers      │ 200-300 │ Longer arguments need more context   │
    │ Drug formulary       │ 30-50   │ Each entry is self-contained         │
    │ Discharge summaries  │ 100-150 │ Multiple sections, moderate detail   │
    │ Lab results          │ 20-40   │ Each result is atomic                │
    │ Policy documents     │ 150-250 │ Legal language needs full clauses    │
    └─────────────────────┴─────────┴─────────────────────────────────────┘

    OVERLAP RECOMMENDATIONS:
    ┌─────────────┬─────────┬──────────────────────────────────────┐
    │ Chunk Size   │ Overlap │ Reasoning                            │
    ├─────────────┼─────────┼──────────────────────────────────────┤
    │ 50 words     │ 10-15   │ 20-30% — small chunks need more     │
    │ 100 words    │ 15-25   │ 15-25% — standard range             │
    │ 200 words    │ 25-40   │ 12-20% — large chunks, less overlap │
    │ 300+ words   │ 30-50   │ 10-15% — minimal, just for safety   │
    └─────────────┴─────────┴──────────────────────────────────────┘

    THE GOLDEN RULE:
    There is no universal "best" chunk size. You MUST test with 
    representative questions from your actual use case and measure 
    retrieval quality. What works for drug formularies fails for 
    research papers.
""")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n📏 Exercise 1: Chunk Size Comparison")
    print("=" * 70)
    print("Compare 50, 100, and 200-word chunks on the same guideline\n")

    print("Choose a demo:")
    print("1. Visualize chunk sizes (see the splits)")
    print("2. Retrieval quality comparison (5 questions × 3 sizes)")
    print("3. Top-K sensitivity (how chunk size affects context volume)")
    print("4. Chunk size selection guide (recommendations)")
    print("5. Run all demos in sequence")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_visualize_chunk_sizes()
    elif choice == "2":
        demo_retrieval_comparison()
    elif choice == "3":
        demo_topk_sensitivity()
    elif choice == "4":
        demo_guidelines()
    elif choice == "5":
        demo_visualize_chunk_sizes()
        demo_retrieval_comparison()
        demo_topk_sensitivity()
        demo_guidelines()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY LEARNINGS — Exercise 1
{'=' * 70}

1. CHUNK SIZE TRADEOFF:
   • Too small (50): precise but often misses context
   • Too large (200): captures context but includes noise
   • Sweet spot depends on your document type and questions

2. OVERLAP IS ESSENTIAL:
   • 10-25% of chunk size prevents boundary information loss
   • Clinical text deserves MORE overlap (critical info at edges)

3. TOP-K AND CHUNK SIZE ARE LINKED:
   • Small chunks + high top_k ≈ Large chunks + low top_k
   • But small chunks give you more GRANULAR control

4. ALWAYS TEST WITH REAL QUESTIONS:
   • No formula replaces empirical testing
   • Build a test set of 10-20 representative questions
   • Measure which chunk size gives the best retrieval scores
""")


if __name__ == "__main__":
    main()
