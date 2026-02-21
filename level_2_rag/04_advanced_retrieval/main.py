"""
Project 4: Advanced Retrieval Techniques
Go beyond basic "embed and search" to produce better, more reliable answers.

Techniques:
1. Multi-Query Retrieval — rephrase the question multiple ways for broader coverage
2. LLM Re-Ranking — use the LLM to judge which retrieved chunks are actually relevant
3. Cited Answers — every claim traced back to a specific source
4. Retrieval Evaluation — measure whether your retrieval is actually working

Builds on: Projects 01-03 (RAG pipeline, ChromaDB, chunking)
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

# Medical knowledge base
DOCUMENTS = [
    {"id": "htn_lifestyle", "text": "Hypertension lifestyle management includes DASH diet, regular exercise 150 minutes per week of moderate intensity, weight management with BMI target under 25, sodium restriction to less than 2300mg daily, limiting alcohol to 2 drinks per day for men and 1 for women, and stress management techniques.", "metadata": {"topic": "hypertension", "section": "lifestyle"}},
    {"id": "htn_meds", "text": "First-line antihypertensive medications include ACE inhibitors such as lisinopril and enalapril, ARBs such as losartan and valsartan, calcium channel blockers such as amlodipine, and thiazide diuretics such as hydrochlorothiazide. Start monotherapy and add second agent if target BP not achieved in 4-6 weeks.", "metadata": {"topic": "hypertension", "section": "pharmacology"}},
    {"id": "htn_targets", "text": "Blood pressure targets: less than 130/80 mmHg for most adults including those with diabetes or CKD. For patients over 65, target less than 130/80 if tolerated. For patients with high cardiovascular risk, more aggressive targets may be considered. Home BP monitoring recommended with target less than 130/80.", "metadata": {"topic": "hypertension", "section": "targets"}},
    {"id": "dm_diagnosis", "text": "Type 2 Diabetes diagnostic criteria: fasting plasma glucose 126 mg/dL or higher on two occasions, HbA1c 6.5% or higher, oral glucose tolerance test 2-hour value 200 mg/dL or higher, or random glucose 200 mg/dL or higher with classic symptoms of hyperglycemia.", "metadata": {"topic": "diabetes", "section": "diagnosis"}},
    {"id": "dm_firstline", "text": "Metformin is first-line therapy for Type 2 diabetes. Start at 500mg once daily with meals, titrate to 1000mg twice daily as tolerated. Contraindicated if eGFR below 30. Reduce dose if eGFR 30-45. Common side effects include GI upset which usually improves over time. Extended-release formulation may reduce GI side effects.", "metadata": {"topic": "diabetes", "section": "pharmacology"}},
    {"id": "dm_secondline", "text": "Second-line diabetes agents after metformin: GLP-1 receptor agonists (semaglutide, liraglutide) preferred for patients with cardiovascular disease or obesity due to cardioprotective and weight loss benefits. SGLT2 inhibitors (empagliflozin, dapagliflozin) preferred for heart failure or CKD due to renal and cardiac protective effects.", "metadata": {"topic": "diabetes", "section": "pharmacology"}},
    {"id": "hf_treatment", "text": "Heart failure with reduced ejection fraction treatment pillars: ARNI (sacubitril-valsartan) or ACEi/ARB, beta-blocker (carvedilol, metoprolol succinate, bisoprolol), mineralocorticoid receptor antagonist (spironolactone, eplerenone), and SGLT2 inhibitor (dapagliflozin, empagliflozin). All four classes should be initiated and titrated to target doses.", "metadata": {"topic": "heart_failure", "section": "pharmacology"}},
    {"id": "hf_devices", "text": "Device therapy in heart failure: ICD for primary prevention if EF 35% or less despite 3 months of optimal medical therapy. CRT for patients with EF 35% or less, LBBB pattern, and QRS duration 150ms or more. LVAD considered for advanced heart failure as bridge to transplant or destination therapy.", "metadata": {"topic": "heart_failure", "section": "devices"}},
    {"id": "ckd_management", "text": "CKD management priorities: blood pressure control target less than 130/80, diabetes management if applicable, ACE inhibitor or ARB for proteinuria regardless of blood pressure, avoid nephrotoxins especially NSAIDs and aminoglycosides, limit protein intake in stages 4-5, and monitor GFR every 3-6 months.", "metadata": {"topic": "ckd", "section": "management"}},
    {"id": "dep_treatment", "text": "Depression initial pharmacotherapy: SSRIs are first-line including sertraline 50-200mg, escitalopram 10-20mg, and fluoxetine 20-80mg. Allow 4-6 weeks for initial response. Combine medication with psychotherapy (CBT or IPT) for moderate to severe depression. If no response after 6-8 weeks at adequate dose, switch agent or augment.", "metadata": {"topic": "depression", "section": "pharmacology"}},
]


def setup_collection():
    """Create ChromaDB collection with medical documents"""
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_advanced",
        embedding_function=openai_ef
    )
    collection.add(
        ids=[d["id"] for d in DOCUMENTS],
        documents=[d["text"] for d in DOCUMENTS],
        metadatas=[d["metadata"] for d in DOCUMENTS]
    )
    return collection


# ============================================================
# DEMO 1: Multi-Query Retrieval
# ============================================================

def demo_multi_query():
    """Rephrase the user's question multiple ways for broader retrieval"""
    print("\n" + "=" * 70)
    print("DEMO 1: MULTI-QUERY RETRIEVAL")
    print("=" * 70)
    print("""
💡 The Problem: User might phrase their question differently than the documents.
   Example: "What helps with high blood pressure?" vs document says "antihypertensive"
   
   Solution: Rephrase the question multiple ways and search with ALL of them.
""")

    collection = setup_collection()

    original_query = "What medications help with high blood pressure?"
    print(f"Original question: \"{original_query}\"\n")

    # Step 1: Use LLM to generate alternative phrasings
    print("🔄 Step 1: Generating alternative phrasings...")
    rephrase_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Generate 3 alternative phrasings of the user's medical question. Return ONLY a JSON array of strings. Keep medical accuracy."
            },
            {"role": "user", "content": original_query}
        ],
        max_tokens=200, temperature=0.7
    )

    try:
        alternatives = json.loads(rephrase_response.choices[0].message.content)
    except json.JSONDecodeError:
        alternatives = [
            "What are first-line antihypertensive drugs?",
            "Pharmacological treatment options for hypertension",
            "Which drugs are prescribed for elevated blood pressure?"
        ]

    all_queries = [original_query] + alternatives
    print("   Queries to search:")
    for i, q in enumerate(all_queries):
        print(f"   {i+1}. \"{q}\"")

    # Step 2: Search with each query
    print(f"\n🔍 Step 2: Searching with each query...")
    all_results = {}  # doc_id → best score

    for query in all_queries:
        results = collection.query(query_texts=[query], n_results=3)
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            distance = results["distances"][0][i]
            if doc_id not in all_results or distance < all_results[doc_id]["distance"]:
                all_results[doc_id] = {
                    "distance": distance,
                    "text": results["documents"][0][i],
                    "query": query
                }

    # Compare: single query vs multi-query
    single_results = collection.query(query_texts=[original_query], n_results=3)
    single_ids = set(single_results["ids"][0])
    multi_ids = set(all_results.keys())

    print(f"\n📊 Results comparison:")
    print(f"   Single query found: {len(single_ids)} unique documents → {single_ids}")
    print(f"   Multi-query found:  {len(multi_ids)} unique documents → {multi_ids}")
    print(f"   Additional docs from multi-query: {multi_ids - single_ids}")

    print(f"\n   🏆 Multi-query top results:")
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["distance"])
    for doc_id, info in sorted_results[:5]:
        print(f"   • {doc_id} (distance: {info['distance']:.4f}, matched query: \"{info['query'][:50]}...\")")

    print("""
💡 MULTI-QUERY VALUE:
   • Catches documents that match DIFFERENT phrasings
   • Especially useful when users use informal language
   • Healthcare: "high blood pressure" vs "hypertension" vs "elevated BP"
   • Cost: ~1 extra LLM call to generate alternatives (very cheap)
""")


# ============================================================
# DEMO 2: LLM Re-Ranking
# ============================================================

def demo_reranking():
    """Use the LLM to judge which retrieved results are truly relevant"""
    print("\n" + "=" * 70)
    print("DEMO 2: LLM RE-RANKING")
    print("=" * 70)
    print("""
💡 The Problem: Embedding similarity isn't always the best relevance measure.
   A chunk about "diabetes medications" might rank higher than "diabetes diagnosis"
   for a diagnosis question, just because both mention "diabetes" a lot.
   
   Solution: Retrieve broadly, then use the LLM to re-rank by TRUE relevance.
""")

    collection = setup_collection()
    query = "What should I check before starting a patient on metformin?"

    # Step 1: Broad retrieval
    print(f"❓ Question: \"{query}\"\n")
    print("📥 Step 1: Broad retrieval (top 5 by embedding similarity)...")
    results = collection.query(query_texts=[query], n_results=5)

    initial_ranking = []
    for i in range(len(results["ids"][0])):
        initial_ranking.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i]
        })

    print("   Initial ranking (by embedding distance):")
    for i, r in enumerate(initial_ranking):
        print(f"   {i+1}. [{r['id']}] distance={r['distance']:.4f} — {r['text'][:80]}...")

    # Step 2: LLM re-ranking
    print(f"\n🤖 Step 2: LLM re-ranking (judging true relevance)...")

    docs_for_ranking = "\n".join([
        f"Document {i+1} (ID: {r['id']}): {r['text'][:200]}"
        for i, r in enumerate(initial_ranking)
    ])

    rerank_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a medical relevance judge. Given a question and documents,
rank the documents by relevance to answering the question.
Return ONLY a JSON array of objects: [{"rank": 1, "doc_id": "...", "relevance": "high/medium/low", "reason": "brief reason"}]"""
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nDocuments:\n{docs_for_ranking}"
            }
        ],
        max_tokens=500, temperature=0
    )

    print(f"\n   LLM Re-ranked results:")
    try:
        reranked = json.loads(rerank_response.choices[0].message.content)
        for item in reranked:
            print(f"   {item['rank']}. [{item['doc_id']}] Relevance: {item['relevance']} — {item['reason']}")
    except (json.JSONDecodeError, KeyError):
        print(f"   {rerank_response.choices[0].message.content}")

    print("""
💡 RE-RANKING VALUE:
   • Embedding search is fast but approximate
   • LLM re-ranking is slow but precise
   • Best approach: retrieve broadly (5-10), re-rank to keep best 2-3
   • Healthcare: ensures the most clinically relevant chunks are used
   • Pattern: "Retrieve broadly, rank precisely"
""")


# ============================================================
# DEMO 3: Cited Answers
# ============================================================

def demo_cited_answers():
    """Generate answers with proper source citations"""
    print("\n" + "=" * 70)
    print("DEMO 3: CITED ANSWERS — Traceable Medical Responses")
    print("=" * 70)
    print("""
💡 In healthcare, EVERY claim needs a source. Clinicians won't trust
   answers they can't verify. Citations make RAG outputs trustworthy.
""")

    collection = setup_collection()

    questions = [
        "How should Type 2 diabetes be treated in a patient with heart failure?",
        "What blood pressure target should be used for a 70-year-old with CKD?",
    ]

    for question in questions:
        print(f"\n{'─' * 70}")
        print(f"❓ Question: \"{question}\"\n")

        results = collection.query(query_texts=[question], n_results=4)

        # Build numbered source list
        sources = []
        for i in range(len(results["ids"][0])):
            sources.append({
                "number": i + 1,
                "id": results["ids"][0][i],
                "topic": results["metadatas"][0][i]["topic"],
                "text": results["documents"][0][i]
            })

        context = "\n\n".join([
            f"[Source {s['number']}: {s['topic']}]\n{s['text']}" for s in sources
        ])

        # Generate answer with citations
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a medical knowledge assistant. Answer questions using ONLY
the provided sources. For EVERY clinical claim, cite the source: [Source X].
If sources conflict or are insufficient, say so. Be specific and clinically useful.
This is for educational purposes only."""
                },
                {
                    "role": "user",
                    "content": f"Sources:\n{context}\n\nQuestion: {question}\n\nAnswer with inline citations:"
                }
            ],
            max_tokens=400, temperature=0.2
        )

        print(f"📋 ANSWER:\n{response.choices[0].message.content}")

        print(f"\n📎 Sources used:")
        for s in sources:
            print(f"   [{s['number']}] {s['id']} ({s['topic']})")

    print("""
💡 CITATION BEST PRACTICES:
   • Number each source clearly in the context
   • Instruct LLM to cite INLINE: "Metformin is first-line [Source 2]"
   • Include source IDs so users can look up the original document
   • If no source supports a claim, the LLM should say "not in provided sources"
   • In healthcare: citations are not optional — they're required for trust
""")


# ============================================================
# DEMO 4: Retrieval Evaluation
# ============================================================

def demo_retrieval_evaluation():
    """Measure whether your retrieval is actually working"""
    print("\n" + "=" * 70)
    print("DEMO 4: RETRIEVAL EVALUATION")
    print("=" * 70)
    print("""
💡 How do you know your RAG system is working well?
   You need test cases with KNOWN CORRECT answers.
""")

    collection = setup_collection()

    # Test cases: question + expected document IDs that should be retrieved
    test_cases = [
        {
            "question": "What medications treat hypertension?",
            "expected_docs": ["htn_meds", "htn_lifestyle"],
            "topic": "hypertension treatment"
        },
        {
            "question": "How is Type 2 diabetes diagnosed?",
            "expected_docs": ["dm_diagnosis"],
            "topic": "diabetes diagnosis"
        },
        {
            "question": "What drugs are used for heart failure?",
            "expected_docs": ["hf_treatment"],
            "topic": "heart failure treatment"
        },
        {
            "question": "When does a CKD patient need an ACE inhibitor?",
            "expected_docs": ["ckd_management"],
            "topic": "CKD management"
        },
        {
            "question": "What is first-line treatment for depression?",
            "expected_docs": ["dep_treatment"],
            "topic": "depression treatment"
        },
    ]

    total_precision = 0
    total_recall = 0
    results_table = []

    for test in test_cases:
        results = collection.query(query_texts=[test["question"]], n_results=3)
        retrieved_ids = set(results["ids"][0])
        expected_ids = set(test["expected_docs"])

        hits = retrieved_ids & expected_ids
        precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0
        recall = len(hits) / len(expected_ids) if expected_ids else 0

        total_precision += precision
        total_recall += recall

        status = "✅" if recall >= 1.0 else "⚠️" if recall > 0 else "❌"
        results_table.append({
            "status": status,
            "topic": test["topic"],
            "precision": precision,
            "recall": recall,
            "retrieved": retrieved_ids,
            "expected": expected_ids
        })

    # Display results
    print(f"\n📊 Evaluation Results (top-3 retrieval):\n")
    for r in results_table:
        print(f"   {r['status']} {r['topic']:30s} Recall: {r['recall']:.0%}  Precision: {r['precision']:.0%}")
        if r['recall'] < 1.0:
            missed = r['expected'] - r['retrieved']
            print(f"      Missed: {missed}")

    avg_precision = total_precision / len(test_cases)
    avg_recall = total_recall / len(test_cases)
    print(f"\n   {'─' * 50}")
    print(f"   Average Recall:    {avg_recall:.0%} (did we find the right docs?)")
    print(f"   Average Precision: {avg_precision:.0%} (are retrieved docs relevant?)")

    print(f"""
💡 EVALUATION TERMS:
   • Recall: Of the correct documents, how many did we retrieve? (higher = better)
   • Precision: Of retrieved documents, how many are correct? (higher = less noise)
   • Both should be high for a good RAG system

📋 IMPROVING LOW SCORES:
   Recall too low → retrieve more results (higher top_k), use multi-query
   Precision too low → better chunking, re-ranking, metadata filters
   Both low → fundamental issue with embeddings or documents

🏥 HEALTHCARE:
   • Recall is CRITICAL — can't miss relevant clinical information
   • Precision matters for efficiency — too much noise wastes clinician time
   • Build a test suite of 20-50 questions with known answers
   • Run evaluation after EVERY change to chunking/retrieval
""")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n🔬 Level 2, Project 4: Advanced Retrieval Techniques")
    print("=" * 70)
    print("Go beyond basic search for better, more reliable answers\n")

    print("Choose a demo:")
    print("1. Multi-query retrieval (rephrase for better coverage)")
    print("2. LLM re-ranking (judge true relevance)")
    print("3. Cited answers (traceable responses)")
    print("4. Retrieval evaluation (measure quality)")
    print("5. Run ALL demos")

    choice = input("\nEnter choice (1-5): ").strip()

    demos = {"1": demo_multi_query, "2": demo_reranking,
             "3": demo_cited_answers, "4": demo_retrieval_evaluation}

    if choice == "5":
        for demo in demos.values():
            demo()
    elif choice in demos:
        demos[choice]()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY TAKEAWAYS
{'=' * 70}

🔄 MULTI-QUERY: Rephrase questions for broader document coverage
   User says "high blood pressure" → also search "hypertension", "elevated BP"

🏆 RE-RANKING: Retrieve broadly (5-10), rank precisely (keep 2-3)
   Embedding similarity is fast but rough; LLM judgment is precise

📎 CITATIONS: Every clinical claim needs a traceable source
   [Source 1] → specific document that supports the claim

📊 EVALUATION: Build test cases, measure recall and precision
   Run after EVERY change to ensure you haven't broken retrieval

🎯 NEXT: Move to 05_medical_rag for the capstone —
   a complete medical knowledge base combining ALL techniques!
""")


if __name__ == "__main__":
    main()
