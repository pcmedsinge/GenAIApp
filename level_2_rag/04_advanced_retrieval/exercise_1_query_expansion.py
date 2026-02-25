"""
Exercise 1: Query Expansion with Medical Synonyms
Implement a technique that automatically expands a user's query with
medical synonyms and related terms to improve retrieval coverage.

Skills practiced:
- Building a medical synonym dictionary
- Expanding queries programmatically and with LLM
- Measuring how expansion improves recall
- Understanding terminology mismatches in healthcare

Healthcare context:
  A patient might say "high blood sugar" while clinical documents say
  "hyperglycemia." A nurse might search for "water pill" while the
  guideline says "thiazide diuretic." Query expansion bridges this gap.
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


def setup_collection():
    """Create ChromaDB collection with medical documents"""
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_synonyms",
        embedding_function=openai_ef
    )
    collection.add(
        ids=[d["id"] for d in DOCUMENTS],
        documents=[d["text"] for d in DOCUMENTS],
        metadatas=[d["metadata"] for d in DOCUMENTS]
    )
    return collection


# ============================================================
# Medical Synonym Dictionary
# ============================================================

MEDICAL_SYNONYMS = {
    # Lay term → clinical/formal terms
    "high blood pressure":     ["hypertension", "elevated blood pressure", "HTN"],
    "high blood sugar":        ["hyperglycemia", "elevated glucose", "diabetes mellitus"],
    "low blood sugar":         ["hypoglycemia", "glucose below 70"],
    "heart attack":            ["myocardial infarction", "MI", "acute coronary syndrome"],
    "water pill":              ["diuretic", "thiazide", "hydrochlorothiazide", "HCTZ"],
    "blood thinner":           ["anticoagulant", "warfarin", "heparin", "DOAC"],
    "sugar diabetes":          ["type 2 diabetes", "diabetes mellitus", "DM"],
    "bad cholesterol":         ["LDL cholesterol", "low-density lipoprotein"],
    "kidney disease":          ["chronic kidney disease", "CKD", "renal insufficiency"],
    "heart failure":           ["congestive heart failure", "CHF", "HF", "HFrEF"],
    "feeling sad":             ["depression", "depressive disorder", "MDD"],
    "blood pressure medicine": ["antihypertensive", "ACE inhibitor", "ARB", "calcium channel blocker"],
    "diabetes medicine":       ["antidiabetic", "metformin", "oral hypoglycemic"],
    "pacemaker":               ["cardiac device", "ICD", "CRT", "implantable cardioverter"],
    "weak heart":              ["reduced ejection fraction", "HFrEF", "systolic dysfunction"],
}


# ============================================================
# Query Expansion Functions
# ============================================================

def expand_with_dictionary(query):
    """Expand a query using the static synonym dictionary"""
    query_lower = query.lower()
    expanded_terms = []

    for lay_term, clinical_terms in MEDICAL_SYNONYMS.items():
        if lay_term in query_lower:
            expanded_terms.extend(clinical_terms)

    return expanded_terms


def expand_with_llm(query):
    """Use the LLM to generate medical synonyms and related terms"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a medical terminology expert. Given a patient-facing query,
generate a JSON array of 3-5 alternative medical terms or phrasings that would appear
in clinical documents covering the same topic.
Return ONLY a JSON array of strings. Focus on clinical/formal terminology."""
            },
            {"role": "user", "content": query}
        ],
        max_tokens=150,
        temperature=0.3
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return []


def search_with_expansion(collection, query, expansions, n_results=3):
    """Search using the original query and all expansions, deduplicate results"""
    all_queries = [query] + expansions
    combined_results = {}

    for q in all_queries:
        results = collection.query(query_texts=[q], n_results=n_results)
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            distance = results["distances"][0][i]
            if doc_id not in combined_results or distance < combined_results[doc_id]["distance"]:
                combined_results[doc_id] = {
                    "distance": distance,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "matched_query": q
                }

    return dict(sorted(combined_results.items(), key=lambda x: x[1]["distance"]))


# ============================================================
# Demo 1: Dictionary-Based Expansion
# ============================================================

def demo_dictionary_expansion():
    """Show how a static synonym dictionary improves retrieval"""
    print("\n" + "=" * 70)
    print("DEMO 1: DICTIONARY-BASED QUERY EXPANSION")
    print("=" * 70)

    collection = setup_collection()

    test_queries = [
        "What water pill should I take for high blood pressure?",
        "My mom has sugar diabetes, what medicine should she start?",
        "What helps with a weak heart?",
        "I'm feeling sad all the time, what can help?",
    ]

    for query in test_queries:
        print(f"\n{'─' * 70}")
        print(f"❓ Patient says: \"{query}\"")

        # Expansion
        expansions = expand_with_dictionary(query)
        if expansions:
            print(f"   📖 Dictionary expansions: {expansions}")
        else:
            print(f"   📖 No dictionary matches found")

        # Compare: without vs with expansion
        baseline = collection.query(query_texts=[query], n_results=3)
        expanded = search_with_expansion(collection, query, expansions)

        baseline_ids = set(baseline["ids"][0])
        expanded_ids = set(expanded.keys())
        new_docs = expanded_ids - baseline_ids

        print(f"\n   Without expansion (top 3): {list(baseline_ids)}")
        print(f"   With expansion (all hits): {list(expanded_ids)}")
        if new_docs:
            print(f"   ✨ New docs found via expansion: {new_docs}")
        else:
            print(f"   (Same results — embedding already handles these synonyms)")

        # Show best match
        if expanded:
            best_id, best_info = next(iter(expanded.items()))
            print(f"   🏆 Best: [{best_id}] dist={best_info['distance']:.4f} via \"{best_info['matched_query'][:50]}\"")

    print("""
💡 DICTIONARY EXPANSION:
   ✅ Pros: Fast, no API calls, deterministic, no hallucination risk
   ❌ Cons: Need to maintain the dictionary, misses novel terms
   🏥 Best for: Known lay-term → clinical-term mappings (e.g., patient portal)
""")


# ============================================================
# Demo 2: LLM-Based Expansion
# ============================================================

def demo_llm_expansion():
    """Show how the LLM generates expansions dynamically"""
    print("\n" + "=" * 70)
    print("DEMO 2: LLM-BASED QUERY EXPANSION")
    print("=" * 70)

    collection = setup_collection()

    test_queries = [
        "What should I check before giving metformin to an older patient?",
        "How do I treat someone with both diabetes and a bad heart?",
        "When should I implant a device for heart failure?",
    ]

    for query in test_queries:
        print(f"\n{'─' * 70}")
        print(f"❓ Query: \"{query}\"")

        # LLM expansion
        print(f"   🤖 Asking LLM for medical synonyms...")
        llm_terms = expand_with_llm(query)
        print(f"      Generated: {llm_terms}")

        # Search without and with expansion
        baseline = collection.query(query_texts=[query], n_results=3)
        expanded = search_with_expansion(collection, query, llm_terms)

        baseline_ids = list(baseline["ids"][0])
        expanded_ids = list(expanded.keys())

        print(f"\n   Without expansion: {baseline_ids}")
        print(f"   With LLM expansion: {expanded_ids}")

        # Show distance improvement
        print(f"\n   Top results with expansion:")
        for i, (doc_id, info) in enumerate(list(expanded.items())[:3]):
            marker = "⬆️" if doc_id not in baseline_ids else "  "
            print(f"   {marker} {i+1}. [{doc_id}] dist={info['distance']:.4f} topic={info['metadata']['topic']}")

    print("""
💡 LLM EXPANSION:
   ✅ Pros: Handles ANY query, understands context, generates relevant clinical terms
   ❌ Cons: API call cost/latency, may hallucinate terms, non-deterministic
   🏥 Best for: Open-ended user queries where you can't predict the terminology
""")


# ============================================================
# Demo 3: Combined Expansion Strategy
# ============================================================

def demo_combined_expansion():
    """Combine dictionary + LLM expansion for maximum coverage"""
    print("\n" + "=" * 70)
    print("DEMO 3: COMBINED EXPANSION (Dictionary + LLM)")
    print("=" * 70)

    collection = setup_collection()

    query = "My dad has kidney disease and high blood pressure — what medicines work for both?"
    print(f"\n❓ Query: \"{query}\"\n")

    # Phase 1: Dictionary
    dict_terms = expand_with_dictionary(query)
    print(f"📖 Dictionary expansions: {dict_terms}")

    # Phase 2: LLM
    print(f"🤖 LLM expansions...")
    llm_terms = expand_with_llm(query)
    print(f"   {llm_terms}")

    # Deduplicate
    all_terms = list(set(dict_terms + llm_terms))
    print(f"\n🔗 Combined unique terms ({len(all_terms)}): {all_terms}")

    # Search strategies comparison
    print(f"\n{'─' * 70}")
    print(f"📊 RETRIEVAL COMPARISON:\n")

    # 1) No expansion
    baseline = collection.query(query_texts=[query], n_results=3)
    print(f"   Baseline (no expansion):")
    for i in range(len(baseline["ids"][0])):
        print(f"      {i+1}. [{baseline['ids'][0][i]}] dist={baseline['distances'][0][i]:.4f}")

    # 2) Dictionary only
    dict_results = search_with_expansion(collection, query, dict_terms)
    print(f"\n   Dictionary expansion:")
    for i, (doc_id, info) in enumerate(list(dict_results.items())[:5]):
        print(f"      {i+1}. [{doc_id}] dist={info['distance']:.4f}")

    # 3) LLM only
    llm_results = search_with_expansion(collection, query, llm_terms)
    print(f"\n   LLM expansion:")
    for i, (doc_id, info) in enumerate(list(llm_results.items())[:5]):
        print(f"      {i+1}. [{doc_id}] dist={info['distance']:.4f}")

    # 4) Combined
    combined_results = search_with_expansion(collection, query, all_terms)
    print(f"\n   Combined expansion:")
    for i, (doc_id, info) in enumerate(list(combined_results.items())[:5]):
        print(f"      {i+1}. [{doc_id}] dist={info['distance']:.4f} (via \"{info['matched_query'][:40]}\")")

    # Summary
    base_ids = set(baseline["ids"][0])
    combined_ids = set(combined_results.keys())
    print(f"\n   📈 Coverage: baseline found {len(base_ids)} docs, combined found {len(combined_ids)} docs")
    if combined_ids - base_ids:
        print(f"   ✨ New documents discovered: {combined_ids - base_ids}")

    print("""
💡 COMBINED STRATEGY:
   1. Dictionary first (fast, zero cost) → catches known lay terms
   2. LLM second (smart, small cost) → catches nuanced/contextual terms
   3. Deduplicate all expansions
   4. Search with ALL queries, keep best distance per document
   
   🏥 This mimics how a clinical librarian thinks:
      "The patient said X, which in medical terms is Y or Z..."
""")


# ============================================================
# Demo 4: Expansion Impact Measurement
# ============================================================

def demo_expansion_measurement():
    """Quantify how much expansion helps across multiple queries"""
    print("\n" + "=" * 70)
    print("DEMO 4: MEASURING EXPANSION IMPACT")
    print("=" * 70)

    collection = setup_collection()

    # Test set with expected relevant documents
    test_set = [
        {"query": "What water pill for high blood pressure?",
         "expected": ["htn_meds", "htn_lifestyle"]},
        {"query": "Sugar diabetes starting medicine",
         "expected": ["dm_firstline", "dm_diagnosis"]},
        {"query": "Weak heart device therapy",
         "expected": ["hf_devices", "hf_treatment"]},
        {"query": "Kidney disease blood pressure pills",
         "expected": ["ckd_management", "htn_meds"]},
        {"query": "Feeling sad need medicine",
         "expected": ["dep_treatment"]},
        {"query": "Bad cholesterol and high blood pressure together",
         "expected": ["htn_meds", "htn_lifestyle", "htn_targets"]},
    ]

    print(f"\n📊 Testing {len(test_set)} queries with and without expansion...\n")

    baseline_recalls = []
    expanded_recalls = []

    print(f"   {'Query':<45} {'Base Recall':>12} {'Exp. Recall':>12} {'Improvement':>12}")
    print(f"   {'─' * 81}")

    for test in test_set:
        q = test["query"]
        expected = set(test["expected"])
        short_q = q[:42] + "..." if len(q) > 45 else q

        # Baseline
        baseline = collection.query(query_texts=[q], n_results=3)
        base_ids = set(baseline["ids"][0])
        base_recall = len(base_ids & expected) / len(expected)
        baseline_recalls.append(base_recall)

        # With expansion (dict + LLM)
        dict_terms = expand_with_dictionary(q)
        llm_terms = expand_with_llm(q)
        all_terms = list(set(dict_terms + llm_terms))
        expanded = search_with_expansion(collection, q, all_terms)
        exp_ids = set(list(expanded.keys())[:5])  # top 5
        exp_recall = len(exp_ids & expected) / len(expected)
        expanded_recalls.append(exp_recall)

        improvement = exp_recall - base_recall
        arrow = "⬆️" if improvement > 0 else "  " if improvement == 0 else "⬇️"
        print(f"   {short_q:<45} {base_recall:>11.0%} {exp_recall:>11.0%} {arrow} {improvement:>+10.0%}")

    # Aggregate
    avg_base = sum(baseline_recalls) / len(baseline_recalls)
    avg_exp = sum(expanded_recalls) / len(expanded_recalls)
    improvement_pct = (avg_exp - avg_base) / avg_base * 100 if avg_base > 0 else 0

    print(f"\n   {'─' * 81}")
    print(f"   {'AVERAGE':<45} {avg_base:>11.0%} {avg_exp:>11.0%}    {improvement_pct:>+8.1f}%")

    queries_improved = sum(1 for b, e in zip(baseline_recalls, expanded_recalls) if e > b)
    queries_same = sum(1 for b, e in zip(baseline_recalls, expanded_recalls) if e == b)
    queries_worse = sum(1 for b, e in zip(baseline_recalls, expanded_recalls) if e < b)

    print(f"""
   📈 Summary:
      Queries improved: {queries_improved}/{len(test_set)}
      Queries unchanged: {queries_same}/{len(test_set)}
      Queries worse:    {queries_worse}/{len(test_set)}

💡 WHEN EXPANSION HELPS MOST:
   • Lay-term queries ("water pill") — high impact
   • Already clinical queries ("hydrochlorothiazide dosing") — low impact
   • Multi-topic queries ("kidney + blood pressure") — moderate impact
   
   RULE OF THUMB: Expansion helps most when your users are NOT clinicians.
   For clinician-facing tools, LLM re-ranking may matter more.
""")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n🔍 Exercise 1: Query Expansion with Medical Synonyms")
    print("=" * 70)
    print("Bridge the gap between patient language and clinical documents\n")

    print("Choose a demo:")
    print("1. Dictionary-based expansion (static synonym mapping)")
    print("2. LLM-based expansion (dynamic term generation)")
    print("3. Combined strategy (dictionary + LLM)")
    print("4. Measure expansion impact (recall comparison)")
    print("5. Run all demos in sequence")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_dictionary_expansion()
    elif choice == "2":
        demo_llm_expansion()
    elif choice == "3":
        demo_combined_expansion()
    elif choice == "4":
        demo_expansion_measurement()
    elif choice == "5":
        demo_dictionary_expansion()
        demo_llm_expansion()
        demo_combined_expansion()
        demo_expansion_measurement()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY LEARNINGS — Exercise 1: Query Expansion
{'=' * 70}

1. TERMINOLOGY MISMATCH IS REAL:
   • Patients say "water pill," documents say "thiazide diuretic"
   • Embeddings handle SOME synonyms, but not all
   • Systematic expansion catches what embeddings miss

2. TWO EXPANSION APPROACHES:
   • Dictionary: fast, free, deterministic — best for known mappings
   • LLM: flexible, smart, costs a small API call — best for open-ended

3. COMBINED IS BEST:
   • Use dictionary first (instant, zero cost)
   • Add LLM for terms the dictionary missed
   • Deduplicate, search all, keep best distances

4. MEASURE THE IMPACT:
   • Don't assume expansion helps — measure recall before and after
   • Expansion helps MOST for non-clinical users
   • For clinical users, re-ranking may be more valuable

5. HEALTHCARE APPLICATION:
   • Patient portal: heavy expansion (lay → clinical)
   • Clinician tool: light expansion (abbreviation → full term)
   • Always test with real user queries from your population
""")


if __name__ == "__main__":
    main()
