"""
Exercise 4: Retrieval Accuracy Comparison (10 Test Questions)
Build a systematic evaluation: 10 clinical questions tested against all
chunking strategies, scored and ranked to find the best approach.

Skills practiced:
- Building a retrieval evaluation framework
- Creating ground-truth test datasets
- Measuring retrieval quality with distance scores
- Comparing strategies with aggregate statistics
- Understanding why systematic testing beats intuition

Healthcare context:
  Before deploying a clinical RAG system, you need evidence that it retrieves
  the RIGHT information. This exercise builds a mini evaluation suite — the
  same kind of quality assurance that production healthcare AI teams run before
  any clinical deployment.
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
# Clinical Guideline (from main.py)
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
# 10 Test Questions with Expected Section (Ground Truth)
# ============================================================

TEST_QUESTIONS = [
    {
        "question": "What are the four pillars of heart failure medication?",
        "expected_section": "Pharmacological Treatment",
        "difficulty": "direct"
    },
    {
        "question": "What is the NYHA classification system?",
        "expected_section": "Definition and Classification",
        "difficulty": "direct"
    },
    {
        "question": "What BNP level suggests heart failure?",
        "expected_section": "Diagnosis",
        "difficulty": "detail"
    },
    {
        "question": "When should an ICD be implanted?",
        "expected_section": "Device Therapy",
        "difficulty": "direct"
    },
    {
        "question": "How much sodium can a heart failure patient consume?",
        "expected_section": "Lifestyle",
        "difficulty": "detail"
    },
    {
        "question": "How soon should a patient be seen after hospital discharge?",
        "expected_section": "Monitoring and Follow-up",
        "difficulty": "detail"
    },
    {
        "question": "What medications should be avoided in HFrEF?",
        "expected_section": "Lifestyle",
        "difficulty": "inference"
    },
    {
        "question": "What ejection fraction defines HFrEF versus HFpEF?",
        "expected_section": "Definition and Classification",
        "difficulty": "detail"
    },
    {
        "question": "When should palliative care be considered for heart failure?",
        "expected_section": "Monitoring and Follow-up",
        "difficulty": "inference"
    },
    {
        "question": "What is sacubitril-valsartan and when is it used?",
        "expected_section": "Pharmacological Treatment",
        "difficulty": "detail"
    },
]


# ============================================================
# Chunking strategies
# ============================================================

def chunk_fixed_size(text, chunk_size=100, overlap=20):
    """Fixed-size word-count chunking"""
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


def chunk_by_sentences(text, sentences_per_chunk=3, overlap_sentences=1):
    """Sentence-based chunking"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    start = 0
    while start < len(sentences):
        end = min(start + sentences_per_chunk, len(sentences))
        chunk = " ".join(sentences[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += sentences_per_chunk - overlap_sentences
    return chunks


def chunk_by_sections(text, section_pattern=r'\nSection \d+:'):
    """Section-based chunking (split at headers)"""
    parts = re.split(section_pattern, text)
    headers = re.findall(section_pattern, text)
    chunks = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        header = headers[i - 1].strip() if i > 0 and i - 1 < len(headers) else ""
        chunk_text = f"{header}\n{part}" if header else part
        chunks.append(chunk_text.strip())
    return chunks


# ============================================================
# Demo 1: Run the full evaluation
# ============================================================

def demo_full_evaluation():
    """Test all strategies against 10 questions"""
    print("\n" + "=" * 70)
    print("DEMO 1: FULL EVALUATION — 10 Questions × 4 Strategies")
    print("=" * 70)

    strategies = {
        "Fixed-50":    chunk_fixed_size(CLINICAL_GUIDELINE, 50, 10),
        "Fixed-100":   chunk_fixed_size(CLINICAL_GUIDELINE, 100, 20),
        "Sentences-4": chunk_by_sentences(CLINICAL_GUIDELINE, 4, 1),
        "Sections":    chunk_by_sections(CLINICAL_GUIDELINE),
    }

    # Build ChromaDB collections
    chroma_client = chromadb.Client()
    collections = {}

    print("\n⏳ Indexing all strategies...")
    for name, chunks in strategies.items():
        coll = chroma_client.create_collection(name=name.lower().replace("-", "_"),
                                                embedding_function=openai_ef)
        coll.add(
            ids=[f"{name}_{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"strategy": name, "index": i} for i in range(len(chunks))]
        )
        collections[name] = coll
        print(f"   {name:15s}: {len(chunks)} chunks indexed")

    # Score tracking
    scores = {name: [] for name in strategies}
    relevance_hits = {name: 0 for name in strategies}

    print(f"\n{'=' * 70}")
    print(f"{'Question':<55} {'F-50':>6} {'F-100':>6} {'Sent':>6} {'Sect':>6}")
    print(f"{'─' * 70}")

    for tq in TEST_QUESTIONS:
        question = tq["question"]
        expected = tq["expected_section"].lower()
        short_q = question[:52] + "..." if len(question) > 55 else question

        row = f"{short_q:<55}"

        for name in strategies:
            results = collections[name].query(query_texts=[question], n_results=1)
            dist = results["distances"][0][0]
            text = results["documents"][0][0].lower()
            scores[name].append(dist)

            # Check if the retrieved chunk contains expected section content
            # Use a simple heuristic: does the retrieved text contain keywords from expected section?
            expected_keywords = expected.split()
            relevant = any(kw in text for kw in expected_keywords)
            if relevant:
                relevance_hits[name] += 1

            row += f" {dist:>5.3f}"

        print(row)

    # Summary statistics
    print(f"\n{'=' * 70}")
    print(f"{'SUMMARY':^70}")
    print(f"{'=' * 70}\n")

    print(f"   {'Strategy':<15} {'Avg Dist':>10} {'Min Dist':>10} {'Max Dist':>10} {'Relevance':>12}")
    print(f"   {'─' * 57}")

    best_avg = None
    best_name = None
    for name in strategies:
        avg_d = sum(scores[name]) / len(scores[name])
        min_d = min(scores[name])
        max_d = max(scores[name])
        rel = f"{relevance_hits[name]}/10"

        if best_avg is None or avg_d < best_avg:
            best_avg = avg_d
            best_name = name

        print(f"   {name:<15} {avg_d:>10.4f} {min_d:>10.4f} {max_d:>10.4f} {rel:>12}")

    print(f"\n   🏆 Best average distance: {best_name} ({best_avg:.4f})")

    print("""
💡 HOW TO READ:
   • Lower distance = better retrieval (closer match)
   • Relevance = how often the top-1 chunk contains expected section content
   • Best strategy depends on your document type and question patterns
""")


# ============================================================
# Demo 2: Per-question detailed analysis
# ============================================================

def demo_detailed_analysis():
    """Deep dive on each question — see exactly what each strategy retrieves"""
    print("\n" + "=" * 70)
    print("DEMO 2: DETAILED PER-QUESTION ANALYSIS")
    print("=" * 70)

    strategies = {
        "Fixed-100":   chunk_fixed_size(CLINICAL_GUIDELINE, 100, 20),
        "Sentences-4": chunk_by_sentences(CLINICAL_GUIDELINE, 4, 1),
        "Sections":    chunk_by_sections(CLINICAL_GUIDELINE),
    }

    chroma_client = chromadb.Client()
    collections = {}
    for name, chunks in strategies.items():
        coll = chroma_client.create_collection(
            name=f"detail_{name.lower().replace('-', '_')}",
            embedding_function=openai_ef
        )
        coll.add(
            ids=[f"{name}_{i}" for i in range(len(chunks))],
            documents=chunks
        )
        collections[name] = coll

    # Pick 3 interesting questions (different difficulty levels)
    selected = [
        TEST_QUESTIONS[0],  # direct: medication pillars
        TEST_QUESTIONS[2],  # detail: BNP level
        TEST_QUESTIONS[8],  # inference: palliative care
    ]

    for tq in selected:
        print(f"\n{'=' * 70}")
        print(f"❓ \"{tq['question']}\"")
        print(f"   Expected section: {tq['expected_section']} | Difficulty: {tq['difficulty']}")

        for name in strategies:
            results = collections[name].query(query_texts=[tq["question"]], n_results=1)
            dist = results["distances"][0][0]
            text = results["documents"][0][0]

            print(f"\n   📊 {name}:")
            print(f"      Distance: {dist:.4f}")
            print(f"      Retrieved: \"{text[:150]}...\"")

        # Identify best
        best_dist = 999
        best_strat = ""
        for name in strategies:
            results = collections[name].query(query_texts=[tq["question"]], n_results=1)
            if results["distances"][0][0] < best_dist:
                best_dist = results["distances"][0][0]
                best_strat = name
        print(f"\n   → Best: {best_strat} ({best_dist:.4f})")

    print("""
💡 PATTERNS TO NOTICE:
   • "Direct" questions: section-based often wins (full topic in one chunk)
   • "Detail" questions: smaller chunks may be closer to the specific fact
   • "Inference" questions: harder — correct answer may span multiple chunks
   • No single strategy wins ALL question types
""")


# ============================================================
# Demo 3: Question difficulty analysis
# ============================================================

def demo_difficulty_analysis():
    """How does question difficulty affect retrieval quality?"""
    print("\n" + "=" * 70)
    print("DEMO 3: DIFFICULTY ANALYSIS")
    print("=" * 70)

    strategies = {
        "Fixed-100": chunk_fixed_size(CLINICAL_GUIDELINE, 100, 20),
        "Sections":  chunk_by_sections(CLINICAL_GUIDELINE),
    }

    chroma_client = chromadb.Client()
    collections = {}
    for name, chunks in strategies.items():
        coll = chroma_client.create_collection(
            name=f"diff_{name.lower().replace('-', '_')}",
            embedding_function=openai_ef
        )
        coll.add(
            ids=[f"{name}_{i}" for i in range(len(chunks))],
            documents=chunks
        )
        collections[name] = coll

    # Group by difficulty
    difficulty_scores = {
        "direct": {name: [] for name in strategies},
        "detail": {name: [] for name in strategies},
        "inference": {name: [] for name in strategies},
    }

    for tq in TEST_QUESTIONS:
        for name in strategies:
            results = collections[name].query(query_texts=[tq["question"]], n_results=1)
            dist = results["distances"][0][0]
            difficulty_scores[tq["difficulty"]][name].append(dist)

    print(f"\n📊 Average distance by difficulty level:\n")
    print(f"   {'Difficulty':<12} {'Fixed-100':>12} {'Sections':>12} {'Winner':>12}")
    print(f"   {'─' * 48}")

    for diff in ["direct", "detail", "inference"]:
        avgs = {}
        for name in strategies:
            scores = difficulty_scores[diff][name]
            avgs[name] = sum(scores) / len(scores) if scores else 999

        winner = min(avgs, key=avgs.get)
        print(f"   {diff:<12}", end="")
        for name in strategies:
            print(f" {avgs[name]:>12.4f}", end="")
        print(f" {winner:>12}")

    print(f"""
💡 DIFFICULTY PATTERNS:
   • "direct" questions: ask about an obvious topic → easy to retrieve
   • "detail" questions: ask about a specific fact buried in text → moderate
   • "inference" questions: need reasoning across content → hardest
   
   INSIGHT: All chunking strategies struggle with inference questions.
   This is where re-ranking (Project 04) and multi-query retrieval help.
""")


# ============================================================
# Demo 4: Build your own evaluation set
# ============================================================

def demo_build_evaluation():
    """Guidance on building evaluation frameworks"""
    print("\n" + "=" * 70)
    print("DEMO 4: HOW TO BUILD YOUR OWN EVALUATION SET")
    print("=" * 70)

    print("""
    📋 EVALUATION FRAMEWORK STEPS:

    1. CREATE TEST QUESTIONS (10-50):
       ┌─────────────────────────────────────────────────┐
       │ Question:        "What beta-blocker for HF?"    │
       │ Expected answer:  carvedilol, metoprolol, etc.  │
       │ Expected section: Pharmacological Treatment     │
       │ Difficulty:       direct / detail / inference    │
       └─────────────────────────────────────────────────┘

    2. CHOOSE METRICS:
       • Distance score (lower = better match)
       • Relevance hit rate (did top-1 contain the answer?)
       • Top-K recall (is the answer in top-3? top-5?)
       • Citation accuracy (does metadata point to right section?)

    3. TEST MULTIPLE CONFIGURATIONS:
       • Chunk sizes: 50, 100, 200 words
       • Strategies: fixed, sentence, section, semantic
       • Overlap: 0%, 10%, 20%, 30%
       • top_k: 1, 3, 5

    4. ANALYZE RESULTS:
       • Which strategy has lowest average distance?
       • Which handles "hard" questions best?
       • Is there a strategy that's WORST at any question?

    5. DECIDE:
       • Pick the strategy with best OVERALL performance
       • Or use different strategies for different document types
       • Re-evaluate whenever documents or questions change
""")

    print("""
    🏥 HEALTHCARE-SPECIFIC TIPS:
    
    • Include questions from DIFFERENT user roles:
      - Physician: "What's the target dose of carvedilol?"
      - Nurse: "When should I call the doctor about weight gain?"
      - Pharmacist: "What drug interactions to check for?"
      
    • Include edge cases:
      - Negation: "What medications should NOT be used?"
      - Multi-step: "If metformin fails, what's next?"
      - Ambiguous: "What's the treatment?" (which condition?)
      
    • Test with REAL clinician questions (not made-up ones)
    • Track accuracy over time as your knowledge base grows
""")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n📊 Exercise 4: Retrieval Accuracy Comparison")
    print("=" * 70)
    print("10 test questions × 4 chunking strategies = systematic evaluation\n")

    print("Choose a demo:")
    print("1. Full evaluation (10 questions × 4 strategies, scored)")
    print("2. Detailed per-question analysis (3 selected)")
    print("3. Difficulty analysis (direct vs detail vs inference)")
    print("4. How to build your own evaluation set")
    print("5. Run all demos in sequence")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_full_evaluation()
    elif choice == "2":
        demo_detailed_analysis()
    elif choice == "3":
        demo_difficulty_analysis()
    elif choice == "4":
        demo_build_evaluation()
    elif choice == "5":
        demo_full_evaluation()
        demo_detailed_analysis()
        demo_difficulty_analysis()
        demo_build_evaluation()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY LEARNINGS — Exercise 4
{'=' * 70}

1. SYSTEMATIC EVALUATION:
   • Don't guess which strategy is best — MEASURE it
   • 10+ test questions with ground truth expected sections
   • Compare average distance, relevance hits, per-difficulty scores

2. NO STRATEGY WINS EVERYTHING:
   • Section-based excels for structured documents
   • Fixed-size is a reliable default
   • Sentence-based is best for unstructured text
   • The DOCUMENT TYPE determines the best strategy

3. QUESTION DIFFICULTY MATTERS:
   • "Direct" questions: all strategies do reasonably well
   • "Detail" questions: smaller chunks may be more precise
   • "Inference" questions: all strategies struggle (need re-ranking)

4. PRODUCTION PATTERN:
   • Build an evaluation set BEFORE choosing a strategy
   • Re-evaluate when documents or question patterns change
   • Track retrieval quality over time (regression testing)
   • This is the same approach used by clinical AI teams
""")


if __name__ == "__main__":
    main()
