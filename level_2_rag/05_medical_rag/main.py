"""
Project 5: Medical RAG Capstone — Complete Healthcare Knowledge Base
Combines ALL Level 2 skills into a production-pattern medical Q&A system.

What this builds:
- ChromaDB-backed medical knowledge base
- Smart chunking with metadata
- Multi-query retrieval for broad coverage
- Cited answers with confidence scoring
- Handles "I don't know" gracefully

This is what a real healthcare RAG system looks like.
"""

import os
import json
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
# Comprehensive Medical Knowledge Base
# ============================================================

MEDICAL_DOCUMENTS = [
    # === CARDIOLOGY ===
    {"id": "htn_def", "text": "Hypertension is defined as blood pressure consistently at or above 130/80 mmHg. Stage 1 is 130-139 systolic or 80-89 diastolic. Stage 2 is 140/90 or higher. Hypertensive crisis is above 180/120 requiring immediate intervention.", "metadata": {"specialty": "cardiology", "topic": "hypertension", "subtopic": "definition"}},
    {"id": "htn_meds", "text": "First-line antihypertensives: ACE inhibitors (lisinopril 10-40mg, enalapril 5-40mg), ARBs (losartan 50-100mg, valsartan 80-320mg), CCBs (amlodipine 2.5-10mg), thiazides (HCTZ 12.5-25mg). Start monotherapy. If not at target in 4-6 weeks, add second agent from different class or increase dose. Most Stage 2 patients need combination therapy from the start.", "metadata": {"specialty": "cardiology", "topic": "hypertension", "subtopic": "medications"}},
    {"id": "htn_special", "text": "Special populations in hypertension: Black patients may respond better to CCBs or thiazides as initial therapy. Patients with CKD or proteinuria should receive ACE/ARB. Patients with diabetes benefit from ACE/ARB. Pregnant patients should avoid ACE/ARB; use labetalol or nifedipine instead. Elderly patients start at lower doses.", "metadata": {"specialty": "cardiology", "topic": "hypertension", "subtopic": "special_populations"}},
    {"id": "hf_pillars", "text": "Heart failure with reduced EF treatment has four medication pillars that should ALL be started: (1) ARNI (sacubitril-valsartan) preferred over ACEi/ARB, (2) Beta-blocker (carvedilol, metoprolol succinate, or bisoprolol), (3) MRA (spironolactone or eplerenone), (4) SGLT2i (dapagliflozin or empagliflozin). Start low, titrate to target doses. All four improve survival.", "metadata": {"specialty": "cardiology", "topic": "heart_failure", "subtopic": "medications"}},
    {"id": "afib_anticoag", "text": "Atrial fibrillation anticoagulation: Calculate CHA2DS2-VASc score. Score 2+ in men or 3+ in women warrants anticoagulation. DOACs preferred: apixaban 5mg BID (reduce to 2.5mg if age 80+, weight 60kg or less, Cr 1.5+), rivaroxaban 20mg daily with food, dabigatran 150mg BID. Warfarin if mechanical valve (target INR 2-3).", "metadata": {"specialty": "cardiology", "topic": "atrial_fibrillation", "subtopic": "anticoagulation"}},

    # === ENDOCRINOLOGY ===
    {"id": "dm_dx", "text": "Type 2 Diabetes diagnosis: fasting glucose 126+ mg/dL on two occasions, HbA1c 6.5%+, 2-hour OGTT 200+ mg/dL, or random glucose 200+ with symptoms. Prediabetes: fasting glucose 100-125, HbA1c 5.7-6.4%, 2-hour OGTT 140-199. Screen adults 35-70 who are overweight.", "metadata": {"specialty": "endocrinology", "topic": "diabetes", "subtopic": "diagnosis"}},
    {"id": "dm_firstline", "text": "Metformin is first-line for Type 2 diabetes. Start 500mg once daily with meals, titrate to 2000mg daily. Contraindicated if eGFR below 30. Reduce dose if eGFR 30-45. GI side effects common initially; extended-release may help. Does not cause hypoglycemia. Monitor B12 levels annually with long-term use.", "metadata": {"specialty": "endocrinology", "topic": "diabetes", "subtopic": "metformin"}},
    {"id": "dm_addon", "text": "Add-on therapy if HbA1c not at target after 3 months of metformin: GLP-1 agonists (semaglutide, liraglutide) preferred if cardiovascular disease or obesity — provide weight loss and cardiovascular benefit. SGLT2 inhibitors (empagliflozin, dapagliflozin) preferred if heart failure or CKD. DPP-4 inhibitors (sitagliptin) if cost-sensitive. Insulin if HbA1c very high (10%+) or symptomatic.", "metadata": {"specialty": "endocrinology", "topic": "diabetes", "subtopic": "second_line"}},

    # === PULMONOLOGY ===
    {"id": "asthma_steps", "text": "Asthma stepwise management: Step 1 (intermittent) as-needed SABA or low-dose ICS-formoterol. Step 2 (mild persistent) low-dose ICS daily. Step 3 (moderate) low-dose ICS-LABA. Step 4 (severe) medium-high dose ICS-LABA. Step 5 (very severe) add tiotropium or biologic. Assess control every 1-3 months and step down if well-controlled for 3+ months.", "metadata": {"specialty": "pulmonology", "topic": "asthma", "subtopic": "stepwise_therapy"}},
    {"id": "copd_mgmt", "text": "COPD management based on GOLD classification. Group A: bronchodilator PRN. Group B: LAMA (tiotropium) or LABA. Group E (exacerbation history): LAMA+LABA, add ICS if eosinophils 300+. Smoking cessation is the ONLY intervention proven to slow FEV1 decline. Pulmonary rehabilitation improves quality of life. Annual flu vaccine and pneumococcal vaccine recommended.", "metadata": {"specialty": "pulmonology", "topic": "copd", "subtopic": "management"}},

    # === NEPHROLOGY ===
    {"id": "ckd_staging", "text": "CKD staged by GFR: Stage 1 (GFR 90+, kidney damage present), Stage 2 (60-89), Stage 3a (45-59), Stage 3b (30-44), Stage 4 (15-29), Stage 5 (below 15). Also classified by albuminuria: A1 (normal, below 30), A2 (moderate, 30-300), A3 (severe, above 300). Both GFR and albuminuria determine risk.", "metadata": {"specialty": "nephrology", "topic": "ckd", "subtopic": "staging"}},
    {"id": "ckd_mgmt", "text": "CKD management: BP target less than 130/80. ACE/ARB for proteinuria. Avoid nephrotoxins (NSAIDs, aminoglycosides, IV contrast without preparation). Adjust medication doses for GFR. Monitor potassium, phosphorus, calcium, PTH. Refer to nephrology at Stage 4 or rapid decline. Prepare for dialysis access when GFR below 20.", "metadata": {"specialty": "nephrology", "topic": "ckd", "subtopic": "management"}},

    # === PSYCHIATRY ===
    {"id": "dep_screen", "text": "PHQ-9 depression screening: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-19 moderately severe, 20-27 severe. Screen all adults annually. For positive screens, assess for suicidal ideation (PHQ-9 question 9), substance use, bipolar symptoms (mood disorders questionnaire), and medical causes of depression.", "metadata": {"specialty": "psychiatry", "topic": "depression", "subtopic": "screening"}},
    {"id": "dep_tx", "text": "Depression treatment: Mild — watchful waiting, psychotherapy, or lifestyle changes. Moderate — SSRI (sertraline, escitalopram, fluoxetine) plus psychotherapy (CBT). Severe — SSRI plus CBT; consider SNRI (venlafaxine, duloxetine) or mirtazapine if SSRI fails. Treatment-resistant (failed 2+ adequate trials) — augment with aripiprazole, lithium, or bupropion; refer to psychiatry.", "metadata": {"specialty": "psychiatry", "topic": "depression", "subtopic": "treatment"}},
]


# ============================================================
# RAG System Components
# ============================================================

def build_knowledge_base():
    """Set up ChromaDB collection with all medical documents"""
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_knowledge_base",
        embedding_function=openai_ef
    )
    collection.add(
        ids=[d["id"] for d in MEDICAL_DOCUMENTS],
        documents=[d["text"] for d in MEDICAL_DOCUMENTS],
        metadatas=[d["metadata"] for d in MEDICAL_DOCUMENTS]
    )
    print(f"📦 Knowledge base loaded: {collection.count()} documents "
          f"across {len(set(d['metadata']['specialty'] for d in MEDICAL_DOCUMENTS))} specialties")
    return collection


def multi_query_retrieve(question, collection, n_results=4):
    """Retrieve using multiple query phrasings for broader coverage"""
    # Generate alternative phrasings
    rephrase = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Generate 2 alternative medical phrasings of this question. Return ONLY a JSON array of strings."},
            {"role": "user", "content": question}
        ],
        max_tokens=150, temperature=0.5
    )

    try:
        alternatives = json.loads(rephrase.choices[0].message.content)
    except json.JSONDecodeError:
        alternatives = []

    all_queries = [question] + alternatives[:2]

    # Search with each query, deduplicate
    seen = {}
    for q in all_queries:
        results = collection.query(query_texts=[q], n_results=n_results)
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            if doc_id not in seen or results["distances"][0][i] < seen[doc_id]["distance"]:
                seen[doc_id] = {
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }

    # Return sorted by distance (best first)
    return sorted(seen.values(), key=lambda x: x["distance"])[:n_results]


def generate_cited_answer(question, sources):
    """Generate answer with citations and confidence"""
    context = "\n\n".join([
        f"[Source {i+1}: {s['metadata']['specialty']}/{s['metadata']['topic']}]\n{s['text']}"
        for i, s in enumerate(sources)
    ])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a medical knowledge assistant powered by a clinical guidelines database.

RULES:
1. Answer ONLY from the provided sources. Do not use outside knowledge.
2. Cite every clinical claim with [Source X].
3. If sources don't contain enough information, say "Based on available sources, I cannot fully answer this."
4. End with a confidence level: HIGH (directly answered by sources), MEDIUM (partially answered), LOW (sources tangentially related).
5. Add disclaimer: "For educational purposes only. Consult a healthcare provider for medical decisions."

Be specific with medication names, doses, and criteria."""
            },
            {
                "role": "user",
                "content": f"Sources:\n{context}\n\nQuestion: {question}\n\nProvide a cited answer with confidence level:"
            }
        ],
        max_tokens=600, temperature=0.2
    )

    return response.choices[0].message.content, response.usage


# ============================================================
# DEMO 1: Full Medical RAG System
# ============================================================

def demo_full_system():
    """Complete medical RAG with all advanced features"""
    print("\n" + "=" * 70)
    print("DEMO 1: FULL MEDICAL RAG SYSTEM")
    print("=" * 70)

    collection = build_knowledge_base()

    test_questions = [
        "A 55-year-old Black woman has Stage 2 hypertension. What medication should be started?",
        "How should Type 2 diabetes be managed in a patient who also has heart failure?",
        "What are the four medication pillars for heart failure treatment?",
    ]

    for question in test_questions:
        print(f"\n{'─' * 70}")
        print(f"❓ {question}\n")

        # Multi-query retrieve
        sources = multi_query_retrieve(question, collection, n_results=4)
        print(f"📎 Sources retrieved ({len(sources)}):")
        for i, s in enumerate(sources):
            print(f"   [{i+1}] {s['id']} ({s['metadata']['specialty']}/{s['metadata']['topic']}) "
                  f"distance={s['distance']:.4f}")

        # Generate answer
        answer, usage = generate_cited_answer(question, sources)
        print(f"\n📋 ANSWER:\n{answer}")
        print(f"\n   Tokens: {usage.total_tokens} | Cost: ~${usage.total_tokens * 0.00000015:.6f}")

    return collection


# ============================================================
# DEMO 2: Interactive Multi-Topic Q&A
# ============================================================

def demo_interactive_qa(collection):
    """Ask your own questions across all medical specialties"""
    print("\n\n" + "=" * 70)
    print("DEMO 2: INTERACTIVE MEDICAL Q&A")
    print("=" * 70)

    specialties = set(d["metadata"]["specialty"] for d in MEDICAL_DOCUMENTS)
    topics = set(d["metadata"]["topic"] for d in MEDICAL_DOCUMENTS)

    print(f"\n📚 Knowledge base covers:")
    print(f"   Specialties: {', '.join(sorted(specialties))}")
    print(f"   Topics: {', '.join(sorted(topics))}")
    print(f"\n💬 Ask anything! Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue

        sources = multi_query_retrieve(question, collection, n_results=4)

        # Show sources
        print(f"\n   📎 Sources:")
        for i, s in enumerate(sources):
            print(f"      [{i+1}] {s['metadata']['specialty']}/{s['metadata']['topic']} "
                  f"(distance: {s['distance']:.3f})")

        answer, usage = generate_cited_answer(question, sources)
        print(f"\n   📋 {answer}")
        print(f"\n   (Tokens: {usage.total_tokens})\n")


# ============================================================
# DEMO 3: Confidence Scoring
# ============================================================

def demo_confidence_scoring():
    """Show how retrieval distance indicates confidence"""
    print("\n\n" + "=" * 70)
    print("DEMO 3: CONFIDENCE-SCORED ANSWERS")
    print("=" * 70)
    print("""
💡 Not all answers are equally confident.
   Retrieval distance tells us how well the sources match the question.
""")

    collection = build_knowledge_base()

    questions = [
        ("What is first-line treatment for Type 2 diabetes?", "Direct match expected"),
        ("How do SGLT2 inhibitors benefit CKD patients?", "Cross-topic, moderate match"),
        ("What is the treatment for pancreatic cancer?", "NOT in knowledge base"),
    ]

    for question, expected in questions:
        print(f"\n{'─' * 70}")
        print(f"❓ {question}")
        print(f"   Expected: {expected}\n")

        sources = multi_query_retrieve(question, collection, n_results=3)

        # Calculate confidence from distances
        avg_distance = sum(s["distance"] for s in sources) / len(sources)
        best_distance = sources[0]["distance"]

        if best_distance < 0.8:
            confidence_icon = "🟢 HIGH"
        elif best_distance < 1.2:
            confidence_icon = "🟡 MEDIUM"
        else:
            confidence_icon = "🔴 LOW"

        print(f"   Retrieval confidence: {confidence_icon}")
        print(f"   Best match distance: {best_distance:.4f} (lower = better)")
        print(f"   Average distance: {avg_distance:.4f}")

        for i, s in enumerate(sources):
            print(f"   [{i+1}] {s['id']} distance={s['distance']:.4f}")

        answer, _ = generate_cited_answer(question, sources)
        print(f"\n   📋 {answer}")

    print("""
💡 CONFIDENCE SCORING:
   • Distance < 0.8  → HIGH confidence (sources directly relevant)
   • Distance 0.8-1.2 → MEDIUM confidence (sources partially relevant)
   • Distance > 1.2  → LOW confidence (sources may be irrelevant)
   
   Use this to:
   • Show confidence indicators to users
   • Trigger "I don't know" responses for low confidence
   • Route low-confidence queries to a human expert
""")


# ============================================================
# DEMO 4: RAG Limitations
# ============================================================

def demo_limitations():
    """Show what happens when RAG can't answer"""
    print("\n\n" + "=" * 70)
    print("DEMO 4: RAG LIMITATIONS — When It Can't Help")
    print("=" * 70)

    collection = build_knowledge_base()

    out_of_scope = [
        "What is the latest treatment for glioblastoma?",
        "How should I treat a broken femur?",
        "What are the side effects of chemotherapy?",
    ]

    print("\n🔍 Asking questions OUTSIDE the knowledge base:\n")

    for question in out_of_scope:
        print(f"❓ {question}")
        sources = multi_query_retrieve(question, collection, n_results=3)
        best_distance = sources[0]["distance"]

        print(f"   Best match distance: {best_distance:.4f} ({'⚠️ weak match' if best_distance > 1.0 else ''})")

        answer, _ = generate_cited_answer(question, sources)
        print(f"   📋 {answer}\n")

    print("""
💡 RAG LIMITATIONS:
   1. Can ONLY answer from documents in the knowledge base
   2. If no relevant document exists → should say "I don't know"
   3. Won't make up answers IF properly prompted (that's why system prompt matters!)
   4. Knowledge base quality = answer quality
   
🔑 HANDLING LIMITATIONS IN PRODUCTION:
   • Set a distance threshold (e.g., > 1.2 = "I don't have info on this")
   • Track unanswered questions → signals for knowledge base expansion
   • Hybrid approach: RAG for known topics + fallback disclaimer for unknown
   • Always include "consult healthcare provider" disclaimer
""")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n🏥 Level 2, Project 5: Medical RAG Capstone")
    print("=" * 70)
    print("Complete healthcare knowledge base — all Level 2 skills combined\n")

    print("Choose a demo:")
    print("1. Full medical RAG system (multi-query + citations)")
    print("2. Interactive Q&A (ask your own questions)")
    print("3. Confidence-scored answers")
    print("4. RAG limitations (out-of-scope questions)")
    print("5. Run ALL demos")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_full_system()
    elif choice == "2":
        collection = build_knowledge_base()
        demo_interactive_qa(collection)
    elif choice == "3":
        demo_confidence_scoring()
    elif choice == "4":
        demo_limitations()
    elif choice == "5":
        collection = demo_full_system()
        demo_interactive_qa(collection)
        demo_confidence_scoring()
        demo_limitations()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
🎉 LEVEL 2 COMPLETE — CONGRATULATIONS!
{'=' * 70}

You've mastered RAG — the most widely-used GenAI pattern in enterprise applications!

📊 WHAT YOU BUILT:
   • RAG from scratch (no frameworks)
   • ChromaDB vector database integration
   • Multiple chunking strategies
   • Multi-query retrieval + re-ranking
   • Cited medical answers with confidence scoring
   • Graceful "I don't know" handling

🏥 HEALTHCARE VALUE:
   You can now build systems that answer questions from YOUR organization's
   clinical guidelines, policies, and procedures — grounded in real documents
   with traceable citations.

🎯 READY FOR LEVEL 3: AI AGENTS!
   In Level 3, you'll build autonomous agents that can:
   • Think and plan (ReAct pattern)
   • Use tools (including YOUR RAG system!)
   • Remember context across interactions
   • Collaborate with other agents
   • Make clinical decisions with guardrails

   This is where it gets really exciting! 🚀
""")


if __name__ == "__main__":
    main()
