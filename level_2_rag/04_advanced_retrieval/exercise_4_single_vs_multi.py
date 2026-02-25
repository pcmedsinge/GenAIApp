"""
Exercise 4: Single Query vs Multi-Query Comparison (10 Test Questions)
Systematically compare basic single-query retrieval against multi-query
retrieval using 10 medical test questions with known expected documents.

Skills practiced:
- Building retrieval evaluation benchmarks
- Implementing multi-query retrieval from scratch
- Measuring recall, precision, and MRR
- Understanding when multi-query helps (and when it doesn't)

Healthcare context:
  Before deploying multi-query retrieval in a clinical system, you need
  EVIDENCE that it actually improves results. This exercise builds that
  evidence with a structured 10-question evaluation.
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
# Medical Knowledge Base (from main.py)
# ============================================================

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


# ============================================================
# 10 Test Questions with Ground Truth
# ============================================================

TEST_QUESTIONS = [
    {
        "question": "What medications are used to treat high blood pressure?",
        "expected_docs": ["htn_meds"],
        "category": "direct",
    },
    {
        "question": "What lifestyle changes help with hypertension?",
        "expected_docs": ["htn_lifestyle"],
        "category": "direct",
    },
    {
        "question": "How do you diagnose Type 2 diabetes?",
        "expected_docs": ["dm_diagnosis"],
        "category": "direct",
    },
    {
        "question": "What should a patient try before insulin for diabetes?",
        "expected_docs": ["dm_firstline", "dm_secondline"],
        "category": "inference",
    },
    {
        "question": "How do you treat a patient who has both heart failure and diabetes?",
        "expected_docs": ["hf_treatment", "dm_secondline"],
        "category": "multi-topic",
    },
    {
        "question": "When should a defibrillator be implanted?",
        "expected_docs": ["hf_devices"],
        "category": "rephrasing",
    },
    {
        "question": "What blood pressure target for a patient with kidney problems?",
        "expected_docs": ["htn_targets", "ckd_management"],
        "category": "multi-topic",
    },
    {
        "question": "What are SSRIs and what are they used for?",
        "expected_docs": ["dep_treatment"],
        "category": "direct",
    },
    {
        "question": "My patient can't tolerate metformin GI side effects — alternatives?",
        "expected_docs": ["dm_firstline", "dm_secondline"],
        "category": "inference",
    },
    {
        "question": "Which diabetes drugs also protect the heart and kidneys?",
        "expected_docs": ["dm_secondline", "hf_treatment"],
        "category": "multi-topic",
    },
]


def setup_collection():
    """Create ChromaDB collection"""
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_comparison",
        embedding_function=openai_ef
    )
    collection.add(
        ids=[d["id"] for d in DOCUMENTS],
        documents=[d["text"] for d in DOCUMENTS],
        metadatas=[d["metadata"] for d in DOCUMENTS]
    )
    return collection


# ============================================================
# Retrieval Methods
# ============================================================

def retrieve_single(collection, query, n_results=3):
    """Standard single-query retrieval"""
    results = collection.query(query_texts=[query], n_results=n_results)
    return {
        "ids": results["ids"][0],
        "distances": results["distances"][0],
        "documents": results["documents"][0],
    }


def generate_alternative_queries(query, n_alternatives=3):
    """Use LLM to generate alternative phrasings"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"Generate {n_alternatives} alternative phrasings of this medical question. "
                           "Use different medical terminology and framing. "
                           "Return ONLY a JSON array of strings."
            },
            {"role": "user", "content": query}
        ],
        max_tokens=200,
        temperature=0.7
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return []


def retrieve_multi(collection, query, n_results=3, n_alternatives=3):
    """Multi-query retrieval: search with original + LLM-generated alternatives"""
    alternatives = generate_alternative_queries(query, n_alternatives)
    all_queries = [query] + alternatives

    combined = {}  # doc_id → {distance, text, matched_query}
    for q in all_queries:
        results = collection.query(query_texts=[q], n_results=n_results)
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            dist = results["distances"][0][i]
            if doc_id not in combined or dist < combined[doc_id]["distance"]:
                combined[doc_id] = {
                    "distance": dist,
                    "text": results["documents"][0][i],
                    "matched_query": q
                }

    # Sort by distance and return top results
    sorted_results = sorted(combined.items(), key=lambda x: x[1]["distance"])

    return {
        "ids": [r[0] for r in sorted_results],
        "distances": [r[1]["distance"] for r in sorted_results],
        "documents": [r[1]["text"] for r in sorted_results],
        "queries_used": all_queries,
        "total_unique_docs": len(combined),
    }


# ============================================================
# Evaluation Metrics
# ============================================================

def compute_recall(retrieved_ids, expected_ids, k=None):
    """What fraction of expected docs were retrieved?"""
    if k:
        retrieved_ids = retrieved_ids[:k]
    hits = set(retrieved_ids) & set(expected_ids)
    return len(hits) / len(expected_ids) if expected_ids else 0


def compute_precision(retrieved_ids, expected_ids, k=None):
    """What fraction of retrieved docs are relevant?"""
    if k:
        retrieved_ids = retrieved_ids[:k]
    hits = set(retrieved_ids) & set(expected_ids)
    return len(hits) / len(retrieved_ids) if retrieved_ids else 0


def compute_mrr(retrieved_ids, expected_ids):
    """Mean Reciprocal Rank: how high is the first correct result?"""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in expected_ids:
            return 1.0 / (i + 1)
    return 0.0


# ============================================================
# Demo 1: Full Head-to-Head Comparison
# ============================================================

def demo_full_comparison():
    """Run all 10 questions through both methods and compare"""
    print("\n" + "=" * 70)
    print("DEMO 1: FULL HEAD-TO-HEAD — 10 Questions")
    print("=" * 70)

    collection = setup_collection()

    single_recalls = []
    multi_recalls = []
    single_mrrs = []
    multi_mrrs = []

    print(f"\n   {'#':<3} {'Question':<52} {'S-Recall':>8} {'M-Recall':>8} {'Winner':>8}")
    print(f"   {'─' * 79}")

    for i, tq in enumerate(TEST_QUESTIONS):
        # Single query
        single = retrieve_single(collection, tq["question"], n_results=5)
        s_recall = compute_recall(single["ids"], tq["expected_docs"], k=5)
        s_mrr = compute_mrr(single["ids"], tq["expected_docs"])
        single_recalls.append(s_recall)
        single_mrrs.append(s_mrr)

        # Multi query
        multi = retrieve_multi(collection, tq["question"], n_results=3, n_alternatives=3)
        m_recall = compute_recall(multi["ids"], tq["expected_docs"], k=5)
        m_mrr = compute_mrr(multi["ids"], tq["expected_docs"])
        multi_recalls.append(m_recall)
        multi_mrrs.append(m_mrr)

        # Winner
        if m_recall > s_recall:
            winner = "Multi ⬆️"
        elif s_recall > m_recall:
            winner = "Single"
        else:
            winner = "Tie"

        short_q = tq["question"][:49] + "..." if len(tq["question"]) > 52 else tq["question"]
        print(f"   {i+1:<3} {short_q:<52} {s_recall:>7.0%} {m_recall:>7.0%} {winner:>8}")

    # Summary
    avg_s_recall = sum(single_recalls) / len(single_recalls)
    avg_m_recall = sum(multi_recalls) / len(multi_recalls)
    avg_s_mrr = sum(single_mrrs) / len(single_mrrs)
    avg_m_mrr = sum(multi_mrrs) / len(multi_mrrs)

    multi_wins = sum(1 for s, m in zip(single_recalls, multi_recalls) if m > s)
    single_wins = sum(1 for s, m in zip(single_recalls, multi_recalls) if s > m)
    ties = len(TEST_QUESTIONS) - multi_wins - single_wins

    print(f"\n   {'═' * 79}")
    print(f"   SUMMARY:")
    print(f"   {'─' * 79}")
    print(f"   {'Metric':<30} {'Single':>12} {'Multi':>12} {'Difference':>12}")
    print(f"   {'─' * 79}")
    print(f"   {'Average Recall @5':<30} {avg_s_recall:>11.0%} {avg_m_recall:>11.0%} {avg_m_recall - avg_s_recall:>+11.0%}")
    print(f"   {'Average MRR':<30} {avg_s_mrr:>11.3f} {avg_m_mrr:>11.3f} {avg_m_mrr - avg_s_mrr:>+11.3f}")
    print(f"   {'─' * 79}")
    print(f"   Multi wins: {multi_wins}  |  Single wins: {single_wins}  |  Ties: {ties}")

    print("""
💡 HOW TO READ:
   • Recall @5: Of the expected docs, how many appeared in top 5?
   • MRR: How high is the FIRST correct result? (1.0 = first position)
   • Multi-query should help most on "inference" and "multi-topic" questions
""")


# ============================================================
# Demo 2: Category Breakdown
# ============================================================

def demo_category_breakdown():
    """Analyze which question categories benefit most from multi-query"""
    print("\n" + "=" * 70)
    print("DEMO 2: CATEGORY BREAKDOWN — Where Does Multi-Query Help?")
    print("=" * 70)

    collection = setup_collection()

    categories = {}  # category → {"single_recalls": [], "multi_recalls": []}

    for tq in TEST_QUESTIONS:
        cat = tq["category"]
        if cat not in categories:
            categories[cat] = {"single_recalls": [], "multi_recalls": []}

        single = retrieve_single(collection, tq["question"], n_results=5)
        multi = retrieve_multi(collection, tq["question"], n_results=3, n_alternatives=3)

        s_recall = compute_recall(single["ids"], tq["expected_docs"], k=5)
        m_recall = compute_recall(multi["ids"], tq["expected_docs"], k=5)

        categories[cat]["single_recalls"].append(s_recall)
        categories[cat]["multi_recalls"].append(m_recall)

    print(f"\n   {'Category':<15} {'Count':>6} {'Single Recall':>14} {'Multi Recall':>14} {'Improvement':>12}")
    print(f"   {'─' * 61}")

    for cat, data in sorted(categories.items()):
        count = len(data["single_recalls"])
        s_avg = sum(data["single_recalls"]) / count
        m_avg = sum(data["multi_recalls"]) / count
        diff = m_avg - s_avg
        arrow = "⬆️" if diff > 0.05 else "⬇️" if diff < -0.05 else "  "

        print(f"   {cat:<15} {count:>6} {s_avg:>13.0%} {m_avg:>13.0%} {arrow} {diff:>+10.0%}")

    print("""
💡 CATEGORY INSIGHTS:
   • "direct" questions: Single query usually does fine
   • "rephrasing" questions: Multi-query catches different terminology
   • "inference" questions: Multi-query find pieces needed to infer
   • "multi-topic" questions: Multi-query covers different topic aspects
   
   RULE: The more complex/ambiguous the question, the more multi-query helps
""")


# ============================================================
# Demo 3: Query Alternatives Visualization
# ============================================================

def demo_alternatives_visualization():
    """Show exactly what alternative queries are generated and their impact"""
    print("\n" + "=" * 70)
    print("DEMO 3: ALTERNATIVE QUERIES — WHAT THE LLM GENERATES")
    print("=" * 70)

    collection = setup_collection()

    # Pick 3 diverse questions
    selected = [
        TEST_QUESTIONS[3],   # inference: before insulin
        TEST_QUESTIONS[4],   # multi-topic: HF + DM
        TEST_QUESTIONS[5],   # rephrasing: defibrillator
    ]

    for tq in selected:
        print(f"\n{'═' * 70}")
        print(f"❓ \"{tq['question']}\"")
        print(f"   Category: {tq['category']} | Expected: {tq['expected_docs']}\n")

        # Generate alternatives
        alternatives = generate_alternative_queries(tq["question"])
        all_queries = [tq["question"]] + alternatives

        print(f"   Queries used:")
        for j, q in enumerate(all_queries):
            label = "Original" if j == 0 else f"Alt {j}"
            print(f"      {label:>8}: \"{q}\"")

        # Search each query individually
        print(f"\n   Results per query:")
        all_found = set()

        for j, q in enumerate(all_queries):
            results = collection.query(query_texts=[q], n_results=3)
            found_ids = results["ids"][0]
            distances = results["distances"][0]
            all_found.update(found_ids)

            label = "Original" if j == 0 else f"Alt {j}"
            hits = set(found_ids) & set(tq["expected_docs"])
            icon = "✅" if hits else "  "
            print(f"      {label:>8}: {found_ids} (distances: {[f'{d:.3f}' for d in distances]}) {icon}")

        # Summary for this question
        single_results = collection.query(query_texts=[tq["question"]], n_results=3)
        single_ids = set(single_results["ids"][0])
        new_docs = all_found - single_ids

        s_recall = compute_recall(list(single_ids), tq["expected_docs"])
        m_recall = compute_recall(list(all_found), tq["expected_docs"])

        print(f"\n   📊 Single recall: {s_recall:.0%} → Multi recall: {m_recall:.0%}")
        if new_docs:
            expected_new = new_docs & set(tq["expected_docs"])
            print(f"   ✨ New docs from alternatives: {new_docs}")
            if expected_new:
                print(f"   🎯 Of those, EXPECTED docs found: {expected_new}")

    print("""
💡 WHAT TO NOTICE:
   • Each alternative uses different medical terminology
   • Some alternatives find docs the original query missed
   • Not every alternative helps — some find the same docs
   • Multi-query's cost: 1 LLM call (~$0.001) + N extra embedding searches
""")


# ============================================================
# Demo 4: Cost-Benefit Analysis
# ============================================================

def demo_cost_benefit():
    """Analyze whether multi-query is worth the extra cost"""
    print("\n" + "=" * 70)
    print("DEMO 4: COST-BENEFIT ANALYSIS")
    print("=" * 70)

    collection = setup_collection()

    # Run evaluation
    single_recalls = []
    multi_recalls = []

    print(f"\n   ⏳ Running evaluation on all 10 questions...\n")

    for tq in TEST_QUESTIONS:
        single = retrieve_single(collection, tq["question"], n_results=5)
        multi = retrieve_multi(collection, tq["question"], n_results=3, n_alternatives=3)

        single_recalls.append(compute_recall(single["ids"], tq["expected_docs"], k=5))
        multi_recalls.append(compute_recall(multi["ids"], tq["expected_docs"], k=5))

    avg_s = sum(single_recalls) / len(single_recalls)
    avg_m = sum(multi_recalls) / len(multi_recalls)
    improvement = avg_m - avg_s

    # Cost estimation
    queries_per_day = 1000  # example
    cost_per_rephrase = 0.001  # ~$0.001 per gpt-4o-mini call
    cost_per_embed = 0.0001   # ~$0.0001 per embedding call
    n_alternatives = 3

    single_cost_daily = queries_per_day * cost_per_embed
    multi_cost_daily = queries_per_day * (cost_per_rephrase + (1 + n_alternatives) * cost_per_embed)
    extra_cost = multi_cost_daily - single_cost_daily

    print(f"   📊 PERFORMANCE:")
    print(f"      Single-query average recall:  {avg_s:.0%}")
    print(f"      Multi-query average recall:   {avg_m:.0%}")
    print(f"      Improvement:                 {improvement:>+.0%}")
    print(f"      Questions improved:           {sum(1 for s,m in zip(single_recalls, multi_recalls) if m > s)}/10")

    print(f"\n   💰 COST (at {queries_per_day} queries/day):")
    print(f"      Single-query daily cost:  ${single_cost_daily:.2f}")
    print(f"      Multi-query daily cost:   ${multi_cost_daily:.2f}")
    print(f"      Extra daily cost:         ${extra_cost:.2f}")
    print(f"      Extra monthly cost:       ${extra_cost * 30:.2f}")

    print(f"\n   ⚡ LATENCY:")
    print(f"      Single-query: 1 embedding call (~100ms)")
    print(f"      Multi-query:  1 LLM call (~500ms) + {1+n_alternatives} embedding calls (~{(1+n_alternatives)*100}ms)")
    print(f"      Extra latency: ~{500 + n_alternatives*100}ms per query")

    # Decision matrix
    print(f"""
   📋 DECISION MATRIX:

   ┌──────────────────────────────────────────────────────────────┐
   │ Scenario                         │ Recommendation           │
   ├──────────────────────────────────┼──────────────────────────┤
   │ Simple, direct queries           │ Single query (no gain)   │
   │ Complex, multi-topic queries     │ Multi-query (big gain)   │
   │ Latency-critical (<200ms)        │ Single query             │
   │ Accuracy-critical (healthcare)   │ Multi-query              │
   │ High volume (>10k queries/day)   │ Single + smart caching   │
   │ Low volume (<100 queries/day)    │ Multi-query (cost tiny)  │
   └──────────────────────────────────┴──────────────────────────┘

💡 BOTTOM LINE:
   • Multi-query costs ~$1/day extra at 1000 queries
   • The recall improvement justifies this in healthcare
   • Cache LLM-generated alternatives for repeated queries
   • Can also pre-compute alternatives for known common questions
""")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n🔬 Exercise 4: Single vs Multi-Query Comparison")
    print("=" * 70)
    print("10 test questions — which retrieval method wins?\n")

    print("Choose a demo:")
    print("1. Full head-to-head comparison (10 questions)")
    print("2. Category breakdown (direct vs inference vs multi-topic)")
    print("3. Alternative queries visualization (see what LLM generates)")
    print("4. Cost-benefit analysis (is multi-query worth it?)")
    print("5. Run all demos in sequence")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_full_comparison()
    elif choice == "2":
        demo_category_breakdown()
    elif choice == "3":
        demo_alternatives_visualization()
    elif choice == "4":
        demo_cost_benefit()
    elif choice == "5":
        demo_full_comparison()
        demo_category_breakdown()
        demo_alternatives_visualization()
        demo_cost_benefit()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY LEARNINGS — Exercise 4: Single vs Multi-Query
{'=' * 70}

1. MULTI-QUERY HELPS MOST FOR COMPLEX QUESTIONS:
   • Direct questions ("What treats HTN?"): single query is fine
   • Rephrasing questions ("defibrillator" vs "ICD"): multi-query catches synonyms
   • Multi-topic questions ("HF + DM"): multi-query covers both topics
   • Inference questions: multi-query finds supporting evidence pieces

2. EVALUATION IS ESSENTIAL:
   • Don't assume multi-query is better — MEASURE IT
   • 10+ test questions with known correct documents
   • Metrics: Recall@K, Precision@K, MRR (Mean Reciprocal Rank)
   • Re-evaluate when documents or query patterns change

3. COST-BENEFIT:
   • Multi-query adds ~500ms latency + ~$0.001/query
   • For healthcare (accuracy critical, low volume): ALWAYS worth it
   • For consumer apps (latency sensitive, high volume): use selectively
   • Cache alternatives for frequently asked questions

4. PRODUCTION RECOMMENDATIONS:
   • Start with single-query as baseline
   • Add multi-query for complex/clinical queries
   • Use query classification to decide which method to use per query
   • Monitor recall over time with automated test suites
   • This is the evaluation approach used by production RAG teams
""")


if __name__ == "__main__":
    main()
