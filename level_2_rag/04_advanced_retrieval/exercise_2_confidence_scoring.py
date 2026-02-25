"""
Exercise 2: Confidence Scoring System
Build a system that assigns high/medium/low confidence to RAG answers
based on retrieval distances, result agreement, and source coverage.

Skills practiced:
- Designing multi-factor confidence scoring
- Setting distance thresholds from empirical data
- Combining signals (distance, agreement, coverage)
- Communicating uncertainty to clinicians

Healthcare context:
  A clinician asking a question needs to know: "How confident should I be
  in this answer?" A retrieval distance of 0.3 is meaningless to them —
  but "HIGH CONFIDENCE: multiple sources agree" is actionable.
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
        name="medical_confidence",
        embedding_function=openai_ef
    )
    collection.add(
        ids=[d["id"] for d in DOCUMENTS],
        documents=[d["text"] for d in DOCUMENTS],
        metadatas=[d["metadata"] for d in DOCUMENTS]
    )
    return collection


# ============================================================
# Confidence Scoring Engine
# ============================================================

# Thresholds (tuned empirically — adjust for your embeddings and domain)
DISTANCE_THRESHOLDS = {
    "high": 0.65,     # distance < 0.65 → very close match
    "medium": 1.00,   # distance < 1.00 → reasonable match
    # above 1.00 → weak match
}


def score_distance(distances):
    """Score based on top-1 retrieval distance (lower = better)"""
    top_dist = distances[0]
    if top_dist < DISTANCE_THRESHOLDS["high"]:
        return "high", top_dist
    elif top_dist < DISTANCE_THRESHOLDS["medium"]:
        return "medium", top_dist
    else:
        return "low", top_dist


def score_agreement(distances, n_agree=3):
    """Score based on how many of the top-N results are close together.
    If the top results are all close in distance, they likely agree."""
    top_n = distances[:n_agree]
    if len(top_n) < 2:
        return "low", 0

    spread = max(top_n) - min(top_n)
    avg_dist = sum(top_n) / len(top_n)

    if spread < 0.15 and avg_dist < DISTANCE_THRESHOLDS["medium"]:
        return "high", spread
    elif spread < 0.30:
        return "medium", spread
    else:
        return "low", spread


def score_topic_coverage(metadatas):
    """Score based on whether results come from the same topic (focused) or many
    topics (scattered → possibly confused query)."""
    topics = [m.get("topic", "unknown") for m in metadatas]
    unique_topics = set(topics)
    dominant_count = max(topics.count(t) for t in unique_topics) if topics else 0
    total = len(topics)

    if total == 0:
        return "low", 0
    focus_ratio = dominant_count / total

    if focus_ratio >= 0.67:
        return "high", focus_ratio  # most results from same topic
    elif focus_ratio >= 0.5:
        return "medium", focus_ratio
    else:
        return "low", focus_ratio  # results scattered across topics


def compute_confidence(distances, metadatas):
    """Combine all signals into a final confidence score"""
    dist_conf, dist_val = score_distance(distances)
    agree_conf, agree_val = score_agreement(distances)
    topic_conf, topic_val = score_topic_coverage(metadatas)

    # Weighted voting: distance matters most, then agreement, then topic coverage
    weight_map = {"high": 3, "medium": 2, "low": 1}
    weights = {"distance": 3, "agreement": 2, "topic": 1}

    weighted_score = (
        weight_map[dist_conf] * weights["distance"]
        + weight_map[agree_conf] * weights["agreement"]
        + weight_map[topic_conf] * weights["topic"]
    )
    max_score = 3 * sum(weights.values())  # all high

    ratio = weighted_score / max_score

    if ratio >= 0.78:
        final = "HIGH"
    elif ratio >= 0.55:
        final = "MEDIUM"
    else:
        final = "LOW"

    details = {
        "final_confidence": final,
        "confidence_ratio": ratio,
        "distance": {"level": dist_conf, "value": dist_val},
        "agreement": {"level": agree_conf, "value": agree_val},
        "topic_focus": {"level": topic_conf, "value": topic_val},
    }
    return final, details


# ============================================================
# Demo 1: Basic Confidence Scoring
# ============================================================

def demo_basic_confidence():
    """Show confidence scoring on varied queries"""
    print("\n" + "=" * 70)
    print("DEMO 1: BASIC CONFIDENCE SCORING")
    print("=" * 70)

    collection = setup_collection()

    queries = [
        ("What are first-line medications for hypertension?",  "Exact match expected → HIGH"),
        ("How is Type 2 diabetes diagnosed?",                   "Direct topic match → HIGH"),
        ("What should I do for a patient with CKD and HTN?",    "Multi-topic → MEDIUM"),
        ("What is the best treatment for insomnia?",            "Not in knowledge base → LOW"),
        ("Metformin dosing and contraindications",              "Specific drug info → HIGH"),
    ]

    for query, description in queries:
        results = collection.query(query_texts=[query], n_results=5)
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        confidence, details = compute_confidence(distances, metadatas)

        icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}[confidence]
        print(f"\n{'─' * 70}")
        print(f"❓ \"{query}\"")
        print(f"   Expected: {description}")
        print(f"   {icon} Confidence: {confidence} (ratio: {details['confidence_ratio']:.2f})")
        print(f"      Distance:    {details['distance']['level']:>6} (top-1 dist: {details['distance']['value']:.4f})")
        print(f"      Agreement:   {details['agreement']['level']:>6} (spread: {details['agreement']['value']:.4f})")
        print(f"      Topic focus: {details['topic_focus']['level']:>6} (ratio: {details['topic_focus']['value']:.2f})")

    print("""
💡 THREE SIGNALS COMBINED:
   1. Distance — Is the best match actually close?
   2. Agreement — Do top results cluster together or diverge?
   3. Topic focus — Do results come from a single topic?

   A HIGH confidence answer has: close match + clustered results + same topic
   A LOW confidence answer has: distant match + scattered results + mixed topics
""")


# ============================================================
# Demo 2: Confidence-Aware Answer Generation
# ============================================================

def demo_confident_answers():
    """Generate answers with confidence-appropriate framing"""
    print("\n" + "=" * 70)
    print("DEMO 2: CONFIDENCE-AWARE ANSWER GENERATION")
    print("=" * 70)

    collection = setup_collection()

    queries = [
        "What medications are used for heart failure?",
        "How do I treat a patient with both CKD and depression?",
        "What is the recommended treatment for fibromyalgia?",
    ]

    for query in queries:
        results = collection.query(query_texts=[query], n_results=5)
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]

        confidence, details = compute_confidence(distances, metadatas)
        icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}[confidence]

        # Adjust the LLM prompt based on confidence
        confidence_instructions = {
            "HIGH": "Answer confidently based on the provided sources. Use specific details.",
            "MEDIUM": "Answer with appropriate hedging. Note that sources may not fully cover the question. Mention what is covered and what might be missing.",
            "LOW": "The provided sources may NOT contain the answer. State clearly what the sources cover, what they DON'T cover, and recommend the user consult additional references.",
        }

        context = "\n\n".join([f"[Source {i+1}]: {doc}" for i, doc in enumerate(documents[:3])])

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a clinical knowledge assistant. {confidence_instructions[confidence]}
Always cite sources as [Source X]. This is for educational purposes only."""
                },
                {
                    "role": "user",
                    "content": f"Sources:\n{context}\n\nQuestion: {query}"
                }
            ],
            max_tokens=300,
            temperature=0.2
        )

        print(f"\n{'─' * 70}")
        print(f"❓ \"{query}\"")
        print(f"   {icon} Confidence: {confidence}")
        print(f"\n{response.choices[0].message.content}")

    print("""
💡 CONFIDENCE-AWARE RESPONSES:
   🟢 HIGH: "According to guidelines, the four pillars of HF treatment are..."
   🟡 MEDIUM: "Based on available sources, ... however, the question spans
       multiple conditions and the sources may not fully address..."
   🔴 LOW: "The available knowledge base does not appear to contain specific
       information about this topic. Please consult..."

   This prevents the LLM from confidently hallucinating when sources don't match!
""")


# ============================================================
# Demo 3: Threshold Calibration
# ============================================================

def demo_threshold_calibration():
    """Explore how different thresholds affect confidence distribution"""
    print("\n" + "=" * 70)
    print("DEMO 3: THRESHOLD CALIBRATION")
    print("=" * 70)

    collection = setup_collection()

    # A diverse test set
    test_queries = [
        "What medications treat hypertension?",
        "How is diabetes diagnosed?",
        "Heart failure device therapy options",
        "CKD blood pressure management",
        "Depression treatment with SSRIs",
        "Metformin starting dose and side effects",
        "Best treatment for migraine headaches",
        "How to manage fibromyalgia pain",
        "Lifestyle changes for high blood pressure",
        "When to refer heart failure to transplant",
    ]

    threshold_configs = [
        {"name": "Strict",     "high": 0.50, "medium": 0.80},
        {"name": "Default",    "high": 0.65, "medium": 1.00},
        {"name": "Permissive", "high": 0.80, "medium": 1.20},
    ]

    # Collect distances for all queries
    all_distances = []
    for query in test_queries:
        results = collection.query(query_texts=[query], n_results=5)
        all_distances.append({
            "query": query,
            "distances": results["distances"][0],
            "metadatas": results["metadatas"][0]
        })

    for config in threshold_configs:
        # Temporarily override thresholds
        original_high = DISTANCE_THRESHOLDS["high"]
        original_med = DISTANCE_THRESHOLDS["medium"]
        DISTANCE_THRESHOLDS["high"] = config["high"]
        DISTANCE_THRESHOLDS["medium"] = config["medium"]

        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for entry in all_distances:
            conf, _ = compute_confidence(entry["distances"], entry["metadatas"])
            counts[conf] += 1

        # Restore
        DISTANCE_THRESHOLDS["high"] = original_high
        DISTANCE_THRESHOLDS["medium"] = original_med

        total = len(test_queries)
        bar_h = "█" * (counts["HIGH"] * 3)
        bar_m = "█" * (counts["MEDIUM"] * 3)
        bar_l = "█" * (counts["LOW"] * 3)

        print(f"\n   {config['name']:12s} (high<{config['high']}, med<{config['medium']}):")
        print(f"      🟢 HIGH:   {counts['HIGH']:>2}/{total}  {bar_h}")
        print(f"      🟡 MEDIUM: {counts['MEDIUM']:>2}/{total}  {bar_m}")
        print(f"      🔴 LOW:    {counts['LOW']:>2}/{total}  {bar_l}")

    # Show actual top-1 distance distribution
    print(f"\n   {'─' * 50}")
    print(f"   Actual top-1 distances for reference:\n")
    sorted_entries = sorted(all_distances, key=lambda x: x["distances"][0])
    for entry in sorted_entries:
        d = entry["distances"][0]
        bar = "▓" * int(d * 30)
        short_q = entry["query"][:40]
        print(f"      {d:.4f} {bar} {short_q}")

    print("""
💡 CALIBRATION GUIDANCE:
   • STRICT thresholds → more LOW ratings → safer but less helpful
   • PERMISSIVE thresholds → more HIGH ratings → more helpful but riskier
   • HEALTHCARE: err on the side of STRICT — it's better to say "I'm not sure"
     than to confidently give wrong information
   
   How to calibrate:
   1. Run 50+ test queries with known answers
   2. Check: when system says HIGH, is it actually correct?
   3. Check: when system says LOW, was the answer actually in the KB?
   4. Adjust thresholds until false-HIGHs are near zero
""")


# ============================================================
# Demo 4: Confidence Dashboard
# ============================================================

def demo_confidence_dashboard():
    """Interactive — enter queries and see confidence scores"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE CONFIDENCE DASHBOARD")
    print("=" * 70)

    collection = setup_collection()

    sample_queries = [
        "What are the four pillars of heart failure treatment?",
        "How do SGLT2 inhibitors help in both diabetes and heart failure?",
        "What antibiotics treat pneumonia?",
    ]

    print("\n   Running sample queries first...\n")

    for query in sample_queries:
        results = collection.query(query_texts=[query], n_results=5)
        confidence, details = compute_confidence(results["distances"][0], results["metadatas"][0])

        icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}[confidence]
        topics = [m.get("topic") for m in results["metadatas"][0][:3]]

        print(f"   {icon} {confidence:6s} │ dist={details['distance']['value']:.3f} │ "
              f"agree={details['agreement']['level']:>6} │ topics={topics}")
        print(f"           │ \"{query}\"")
        print()

    # Interactive mode
    print(f"{'─' * 70}")
    print("Now try your own queries (type 'quit' to exit):\n")

    while True:
        user_query = input("❓ Your question: ").strip()
        if user_query.lower() in ("quit", "exit", "q", ""):
            break

        results = collection.query(query_texts=[user_query], n_results=5)
        confidence, details = compute_confidence(results["distances"][0], results["metadatas"][0])

        icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}[confidence]

        print(f"\n   {icon} Confidence: {confidence} (ratio: {details['confidence_ratio']:.2f})")
        print(f"      Distance:  {details['distance']['level']:>6} (value: {details['distance']['value']:.4f})")
        print(f"      Agreement: {details['agreement']['level']:>6} (spread: {details['agreement']['value']:.4f})")
        print(f"      Topic:     {details['topic_focus']['level']:>6} (focus: {details['topic_focus']['value']:.2f})")
        print(f"      Top results:")
        for i in range(min(3, len(results["ids"][0]))):
            print(f"         {i+1}. [{results['ids'][0][i]}] dist={results['distances'][0][i]:.4f}")
        print()

    print("   Dashboard closed.\n")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n📊 Exercise 2: Confidence Scoring System")
    print("=" * 70)
    print("Assign HIGH/MEDIUM/LOW confidence to RAG answers\n")

    print("Choose a demo:")
    print("1. Basic confidence scoring (5 varied queries)")
    print("2. Confidence-aware answer generation (LLM adapts tone)")
    print("3. Threshold calibration (strict vs permissive)")
    print("4. Interactive confidence dashboard")
    print("5. Run all demos in sequence")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_basic_confidence()
    elif choice == "2":
        demo_confident_answers()
    elif choice == "3":
        demo_threshold_calibration()
    elif choice == "4":
        demo_confidence_dashboard()
    elif choice == "5":
        demo_basic_confidence()
        demo_confident_answers()
        demo_threshold_calibration()
        demo_confidence_dashboard()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY LEARNINGS — Exercise 2: Confidence Scoring
{'=' * 70}

1. THREE SIGNALS FOR CONFIDENCE:
   • Distance — How close is the best match? (most important)
   • Agreement — Do top results cluster or scatter? (consistency check)
   • Topic focus — Do results come from one topic? (specificity check)

2. CONFIDENCE-AWARE GENERATION:
   • HIGH confidence → answer confidently with specific details
   • MEDIUM confidence → hedge, note limitations
   • LOW confidence → say "I don't have enough information"
   • This prevents the LLM from hallucinating on weak retrieval!

3. THRESHOLD CALIBRATION:
   • Healthcare: use STRICT thresholds (better safe than wrong)
   • Consumer apps: can be more permissive
   • Calibrate with labeled test queries + manual review
   • Re-calibrate when embedding model or documents change

4. PRODUCTION PATTERN:
   • Log confidence with every answer (track quality over time)
   • Alert on sustained LOW confidence (knowledge gap detected)
   • Use confidence to route: HIGH → auto-answer, LOW → escalate to human
   • In healthcare: LOW confidence = "consult your care team"
""")


if __name__ == "__main__":
    main()
