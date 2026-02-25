"""
Exercise 3: "I Don't Know" Detector
Build a system that detects when retrieved chunks aren't relevant enough
to answer the question, and responds honestly instead of hallucinating.

Skills practiced:
- Setting relevance thresholds for retrieval
- Using LLM as a relevance judge
- Designing graceful failure responses
- Balancing helpfulness vs safety

Healthcare context:
  The most dangerous thing a clinical AI can do is answer confidently
  when it doesn't actually have the right information. An "I don't know"
  response is infinitely safer than a hallucinated drug dosage.
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
        name="medical_idk",
        embedding_function=openai_ef
    )
    collection.add(
        ids=[d["id"] for d in DOCUMENTS],
        documents=[d["text"] for d in DOCUMENTS],
        metadatas=[d["metadata"] for d in DOCUMENTS]
    )
    return collection


# ============================================================
# "I Don't Know" Detection Methods
# ============================================================

def detect_idk_distance(distances, threshold=0.95):
    """Method 1: Simple distance threshold.
    If even the best match is far away, we probably don't have the answer."""
    top_distance = distances[0]
    should_decline = top_distance > threshold
    return {
        "method": "distance_threshold",
        "should_decline": should_decline,
        "top_distance": top_distance,
        "threshold": threshold,
        "reason": f"Top distance {top_distance:.4f} {'>' if should_decline else '<='} threshold {threshold}"
    }


def detect_idk_gap(distances, gap_threshold=0.3):
    """Method 2: Distance gap between top results.
    If there's a big gap between #1 and #2, the match might be an outlier."""
    if len(distances) < 2:
        return {"method": "distance_gap", "should_decline": True, "reason": "Not enough results"}

    gap = distances[1] - distances[0]
    top_dist = distances[0]

    # If top-1 is close AND there's a big gap, it's actually a strong unique match (good)
    # If top-1 is far AND there's a big gap, everything is bad (decline)
    should_decline = top_dist > 0.80 and gap > gap_threshold

    return {
        "method": "distance_gap",
        "should_decline": should_decline,
        "top_distance": top_dist,
        "gap": gap,
        "threshold": gap_threshold,
        "reason": f"Top dist={top_dist:.4f}, gap to #2={gap:.4f}"
    }


def detect_idk_llm(query, documents, distances):
    """Method 3: Ask the LLM to judge if the retrieved documents can answer the query."""
    context = "\n".join([f"Doc {i+1} (dist={distances[i]:.3f}): {doc[:150]}"
                         for i, doc in enumerate(documents[:3])])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a relevance judge for a medical knowledge system.
Given a question and the retrieved documents, determine if the documents contain
enough information to answer the question accurately.

Return ONLY a JSON object:
{
  "can_answer": true/false,
  "confidence": "high"/"medium"/"low",
  "reason": "brief explanation",
  "missing": "what information would be needed"
}"""
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nRetrieved documents:\n{context}"
            }
        ],
        max_tokens=200,
        temperature=0
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return {
            "method": "llm_judge",
            "should_decline": not result.get("can_answer", True),
            "confidence": result.get("confidence", "unknown"),
            "reason": result.get("reason", ""),
            "missing": result.get("missing", "")
        }
    except (json.JSONDecodeError, KeyError):
        return {
            "method": "llm_judge",
            "should_decline": False,
            "reason": "Could not parse LLM response"
        }


def detect_idk_combined(query, distances, documents, metadatas):
    """Combine all methods for a robust decision"""
    d1 = detect_idk_distance(distances)
    d2 = detect_idk_gap(distances)
    d3 = detect_idk_llm(query, documents, distances)

    # Voting: if 2+ methods say decline, we decline
    votes_to_decline = sum([d1["should_decline"], d2["should_decline"], d3["should_decline"]])

    should_decline = votes_to_decline >= 2

    return {
        "final_decision": "DECLINE" if should_decline else "ANSWER",
        "votes_to_decline": votes_to_decline,
        "distance_check": d1,
        "gap_check": d2,
        "llm_check": d3,
    }


# ============================================================
# Demo 1: Distance-Based Detection
# ============================================================

def demo_distance_detection():
    """Show how distance thresholds catch out-of-scope questions"""
    print("\n" + "=" * 70)
    print("DEMO 1: DISTANCE-BASED 'I DON'T KNOW' DETECTION")
    print("=" * 70)

    collection = setup_collection()

    # Mix of in-scope and out-of-scope questions
    queries = [
        ("What medications treat hypertension?",            True,  "In KB"),
        ("How is Type 2 diabetes diagnosed?",               True,  "In KB"),
        ("What antibiotics treat pneumonia?",               False, "NOT in KB"),
        ("What is the treatment for epilepsy?",             False, "NOT in KB"),
        ("Heart failure device therapy",                    True,  "In KB"),
        ("How to manage chronic pain with opioids?",        False, "NOT in KB"),
        ("What is the DASH diet for blood pressure?",       True,  "In KB (indirect)"),
        ("Pediatric dosing for amoxicillin?",               False, "NOT in KB"),
    ]

    print(f"\n   Threshold: 0.95 (distances above this → decline)\n")
    print(f"   {'Query':<50} {'Top Dist':>8} {'Decision':>10} {'Correct?':>10}")
    print(f"   {'─' * 78}")

    correct = 0
    total = len(queries)

    for query, in_scope, label in queries:
        results = collection.query(query_texts=[query], n_results=3)
        check = detect_idk_distance(results["distances"][0])
        decision = "DECLINE" if check["should_decline"] else "ANSWER"

        # Check if decision matches reality
        expected_decision = "ANSWER" if in_scope else "DECLINE"
        is_correct = decision == expected_decision
        if is_correct:
            correct += 1

        icon = "✅" if is_correct else "❌"
        short_q = query[:47] + "..." if len(query) > 50 else query
        print(f"   {short_q:<50} {check['top_distance']:>8.4f} {decision:>10} {icon:>10}")

    print(f"\n   Accuracy: {correct}/{total} ({correct/total:.0%})")

    print("""
💡 DISTANCE THRESHOLD:
   ✅ Simple, fast, no extra API calls
   ❌ One threshold doesn't fit all questions
   ❌ Some in-scope questions may have high distances (poor recall)
   ❌ Some out-of-scope questions may have low distances (false confidence)
   
   Best as a FIRST filter — combine with other methods for safety
""")


# ============================================================
# Demo 2: LLM-Judged Detection
# ============================================================

def demo_llm_detection():
    """Use the LLM to judge retrieved relevance"""
    print("\n" + "=" * 70)
    print("DEMO 2: LLM-JUDGED RELEVANCE DETECTION")
    print("=" * 70)

    collection = setup_collection()

    queries = [
        "What are the treatment options for hypertension?",
        "What is the recommended therapy for rheumatoid arthritis?",
        "How do you manage a patient with both diabetes and heart failure?",
        "What pediatric vaccines are required for school enrollment?",
    ]

    for query in queries:
        print(f"\n{'─' * 70}")
        print(f"❓ \"{query}\"")

        results = collection.query(query_texts=[query], n_results=3)
        distances = results["distances"][0]
        documents = results["documents"][0]

        print(f"   Top-3 distances: {[f'{d:.4f}' for d in distances]}")

        # LLM judgment
        print(f"   🤖 Asking LLM to judge relevance...")
        llm_result = detect_idk_llm(query, documents, distances)

        icon = "🔴 DECLINE" if llm_result["should_decline"] else "🟢 ANSWER"
        print(f"   {icon}")
        print(f"      Confidence: {llm_result.get('confidence', 'N/A')}")
        print(f"      Reason: {llm_result.get('reason', 'N/A')}")
        if llm_result.get("missing"):
            print(f"      Missing: {llm_result['missing']}")

    print("""
💡 LLM-JUDGED DETECTION:
   ✅ Understands nuance ("partially answerable", "related but not exact")
   ✅ Can explain WHY it can't answer
   ✅ Catches cases where distance is low but content is irrelevant
   ❌ Costs an extra API call per query
   ❌ LLM might be overconfident or underconfident
   
   Best as a SECOND layer of defense after distance checking
""")


# ============================================================
# Demo 3: Combined Detection with Graceful Responses
# ============================================================

def demo_combined_detection():
    """Full pipeline: detect → decide → respond appropriately"""
    print("\n" + "=" * 70)
    print("DEMO 3: COMBINED DETECTION + GRACEFUL RESPONSES")
    print("=" * 70)

    collection = setup_collection()

    queries = [
        "What are the four pillars of heart failure treatment?",
        "How should acute appendicitis be managed in the ER?",
        "What is the recommended statin dose for high cholesterol?",
        "What blood pressure medication should a CKD patient take?",
    ]

    for query in queries:
        print(f"\n{'═' * 70}")
        print(f"❓ \"{query}\"\n")

        results = collection.query(query_texts=[query], n_results=5)
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # Combined detection
        detection = detect_idk_combined(query, distances, documents, metadatas)

        # Show detection details
        d = detection
        print(f"   📋 Detection Results:")
        print(f"      Distance check:  {'DECLINE' if d['distance_check']['should_decline'] else 'ANSWER':>8}  (top dist: {d['distance_check']['top_distance']:.4f})")
        print(f"      Gap check:       {'DECLINE' if d['gap_check']['should_decline'] else 'ANSWER':>8}  ({d['gap_check']['reason']})")
        print(f"      LLM check:       {'DECLINE' if d['llm_check']['should_decline'] else 'ANSWER':>8}  ({d['llm_check'].get('reason', 'N/A')[:60]})")
        print(f"      Votes to decline: {d['votes_to_decline']}/3")

        if detection["final_decision"] == "DECLINE":
            # Generate a helpful decline response
            print(f"\n   🔴 DECISION: DECLINE — generating helpful 'I don't know' response...\n")

            decline_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a clinical knowledge assistant. The user asked a question
that our knowledge base cannot reliably answer. Generate a helpful decline response that:
1. Acknowledges the question
2. Explains what topics we CAN answer (hypertension, diabetes, heart failure, CKD, depression)
3. Suggests where to find the answer
4. Is polite and professional
Keep it to 3-4 sentences."""
                    },
                    {"role": "user", "content": f"Question I cannot answer: {query}"}
                ],
                max_tokens=200, temperature=0.3
            )
            print(f"   💬 {decline_response.choices[0].message.content}")

        else:
            # Normal answer with the retrieved context
            print(f"\n   🟢 DECISION: ANSWER — generating response...\n")

            context = "\n".join([f"[Source {i+1}]: {doc}" for i, doc in enumerate(documents[:3])])
            answer_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Answer the question using ONLY the provided sources. Cite sources as [Source X]. Be specific and clinically useful. Educational purposes only."
                    },
                    {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {query}"}
                ],
                max_tokens=250, temperature=0.2
            )
            print(f"   💬 {answer_response.choices[0].message.content}")

    print("""
💡 GRACEFUL DECLINE PATTERN:
   1. Don't just say "I don't know" — say WHAT you know and WHERE to look
   2. Multiple detection methods reduce false positives and false negatives
   3. A polite, helpful decline builds MORE trust than a wrong answer
   
   🏥 HEALTHCARE GOLD RULE:
   "It is better to say 'I don't have that information — please consult
   your care team' than to give a confident wrong answer about a medication."
""")


# ============================================================
# Demo 4: Tuning the Detection Sensitivity
# ============================================================

def demo_sensitivity_tuning():
    """Compare aggressive vs conservative detection"""
    print("\n" + "=" * 70)
    print("DEMO 4: DETECTION SENSITIVITY TUNING")
    print("=" * 70)

    collection = setup_collection()

    # Mixed queries — some in scope, some out
    queries = [
        {"q": "What medications treat hypertension?",           "in_scope": True},
        {"q": "How is diabetes diagnosed?",                     "in_scope": True},
        {"q": "Heart failure device therapy options",           "in_scope": True},
        {"q": "Depression treatment first-line",                "in_scope": True},
        {"q": "What antibiotics treat UTI?",                    "in_scope": False},
        {"q": "Pediatric asthma inhaler technique",             "in_scope": False},
        {"q": "How to treat acute stroke in the ER?",           "in_scope": False},
        {"q": "CKD management and blood pressure",             "in_scope": True},
        {"q": "What is the treatment for gout?",                "in_scope": False},
        {"q": "Metformin contraindications",                    "in_scope": True},
    ]

    thresholds = [
        {"name": "Conservative (safe)", "distance": 0.80},
        {"name": "Balanced",            "distance": 0.95},
        {"name": "Aggressive (helpful)", "distance": 1.15},
    ]

    for config in thresholds:
        true_pos = 0   # correctly answered
        true_neg = 0   # correctly declined
        false_pos = 0  # wrongly answered (danger!)
        false_neg = 0  # wrongly declined (missed opportunity)

        for entry in queries:
            results = collection.query(query_texts=[entry["q"]], n_results=3)
            check = detect_idk_distance(results["distances"][0], threshold=config["distance"])
            answered = not check["should_decline"]

            if entry["in_scope"] and answered:
                true_pos += 1
            elif not entry["in_scope"] and not answered:
                true_neg += 1
            elif not entry["in_scope"] and answered:
                false_pos += 1  # DANGEROUS in healthcare
            else:
                false_neg += 1  # Annoying but safe

        total = len(queries)
        print(f"\n   {config['name']} (threshold={config['distance']}):")
        print(f"      ✅ Correct answers (true pos):    {true_pos}")
        print(f"      ✅ Correct declines (true neg):   {true_neg}")
        print(f"      ⚠️  Missed answers (false neg):    {false_neg}")
        print(f"      🚨 Wrong answers (false pos):     {false_pos}  {'← DANGEROUS!' if false_pos > 0 else ''}")
        print(f"      Accuracy: {(true_pos + true_neg)/total:.0%}")

    print(f"""
💡 SENSITIVITY TRADEOFF:

   Conservative ←────────────────────→ Aggressive
   (says "IDK" a lot)                  (answers everything)
   
   Fewer false positives              More false positives
   More false negatives               Fewer false negatives
   SAFER but less helpful             More helpful but RISKIER

   🏥 HEALTHCARE RECOMMENDATION:
      Start CONSERVATIVE and loosen ONLY with evidence that false positives
      are near zero. A false positive (wrong answer) in healthcare can be
      life-threatening. A false negative (missed answer) is just inconvenient.

   METRICS TO TRACK:
   • False Positive Rate → should be < 5% for clinical systems
   • Answer Rate → % of queries you actually answer (aim for > 70%)
   • User satisfaction → are "I don't know" responses helpful enough?
""")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n🚫 Exercise 3: 'I Don't Know' Detector")
    print("=" * 70)
    print("Know when you DON'T know — the most important safety feature\n")

    print("Choose a demo:")
    print("1. Distance-based detection (simple threshold)")
    print("2. LLM-judged detection (smart relevance check)")
    print("3. Combined detection + graceful responses")
    print("4. Sensitivity tuning (conservative vs aggressive)")
    print("5. Run all demos in sequence")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_distance_detection()
    elif choice == "2":
        demo_llm_detection()
    elif choice == "3":
        demo_combined_detection()
    elif choice == "4":
        demo_sensitivity_tuning()
    elif choice == "5":
        demo_distance_detection()
        demo_llm_detection()
        demo_combined_detection()
        demo_sensitivity_tuning()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY LEARNINGS — Exercise 3: "I Don't Know" Detector
{'=' * 70}

1. THREE DETECTION METHODS:
   • Distance threshold — fast, simple, catches obvious misses
   • Gap analysis — spots isolated/outlier matches
   • LLM judge — understands nuance, explains why

2. COMBINE FOR SAFETY:
   • Voting (2-of-3) reduces both false positives and false negatives
   • Distance is the cheap first filter
   • LLM is the expensive but smart second filter

3. GRACEFUL DECLINE IS AN ART:
   • Don't just say "I don't know"
   • Say WHAT you know, WHAT you don't, and WHERE to look
   • A helpful decline builds more trust than a confident wrong answer

4. HEALTHCARE SAFETY:
   • False positives (wrong answers) are DANGEROUS
   • False negatives (missed answers) are ANNOYING but SAFE
   • Always start conservative and loosen with evidence
   • Target: <5% false positive rate for clinical systems

5. PRODUCTION PATTERN:
   • Log every decline with the reason
   • Track decline rate over time (rising = knowledge gaps)
   • Use declines to identify what to ADD to your knowledge base
   • Route declined questions to human experts
""")


if __name__ == "__main__":
    main()
