"""
Exercise 4: Confidence Scoring
Add a confidence score that shows how relevant the retrieved chunks are.

Skills practiced:
- Building a confidence scoring system for RAG
- Using similarity thresholds to gauge answer reliability
- Generating confidence-aware responses
- Detecting when RAG doesn't have enough information

Healthcare context:
  In clinical settings, a wrong answer with high confidence is DANGEROUS.
  A confidence score tells clinicians: "I'm 90% sure" vs "I'm guessing."
  This is critical for safe clinical decision support systems.
"""

import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# Knowledge Base
# ============================================================

MEDICAL_KNOWLEDGE_BASE = [
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
        "id": "asthma_protocol",
        "title": "Asthma Management Protocol",
        "content": """Asthma diagnosis confirmed by spirometry showing variable airflow obstruction
(FEV1 improvement of 12 percent or more after bronchodilator).
Step 1 Mild intermittent: As-needed low-dose ICS-formoterol or SABA.
Step 2 Mild persistent: Low-dose ICS daily (budesonide 200mcg or fluticasone 100mcg).
Step 3 Moderate persistent: Low-dose ICS-LABA combination (fluticasone-salmeterol or budesonide-formoterol).
Step 4 Severe: Medium to high-dose ICS-LABA combination.
Step 5 Very severe: Add-on therapy including tiotropium or biologics
(omalizumab for allergic asthma, mepolizumab for eosinophilic asthma).
All patients should have a written asthma action plan.
Assess control every 1-3 months. Step down therapy if well-controlled for 3 or more months.
Emergency management: Systemic corticosteroids (prednisone 40-60mg) for exacerbations.
Common triggers to avoid: allergens, smoke, cold air, exercise (use pre-treatment SABA).""",
        "category": "pulmonology"
    },
    {
        "id": "ckd_staging",
        "title": "Chronic Kidney Disease Staging and Management",
        "content": """CKD staging based on GFR: Stage 1 (GFR 90 or higher with kidney damage),
Stage 2 (GFR 60-89), Stage 3a (GFR 45-59), Stage 3b (GFR 30-44),
Stage 4 (GFR 15-29), Stage 5 (GFR less than 15, kidney failure).
Key management principles: Control blood pressure (target less than 130/80),
manage diabetes if present (HbA1c less than 7 percent),
prescribe ACE inhibitor or ARB for proteinuria even if blood pressure is normal.
Limit protein intake in Stage 4-5. Avoid nephrotoxic drugs: NSAIDs,
aminoglycoside antibiotics, IV contrast dye (prepare with hydration if needed).
Monitor: GFR every 3-6 months, urine albumin-to-creatinine ratio,
potassium, phosphorus, PTH levels.
Referral to nephrology recommended at Stage 4 or rapidly declining GFR.
Prepare for renal replacement therapy when GFR drops below 20.
Manage anemia with ESAs when hemoglobin falls below 10 g/dL.""",
        "category": "nephrology"
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
    },
    {
        "id": "depression_management",
        "title": "Depression Screening and Initial Management",
        "content": """Screen all adults for depression using PHQ-9 questionnaire.
Score interpretation: 0-4 Minimal or none, 5-9 Mild, 10-14 Moderate,
15-19 Moderately severe, 20-27 Severe depression.
Initial treatment for moderate depression: SSRIs are first-line pharmacotherapy.
Common SSRIs: sertraline 50mg daily, escitalopram 10mg daily, fluoxetine 20mg daily.
Allow 4-6 weeks for initial response. Full remission may take 8-12 weeks.
Combination of medication plus psychotherapy (CBT) is most effective for moderate-severe depression.
Monitor for suicidal ideation especially in first 4 weeks and in patients under 25 years.
If no response at 6 weeks: increase dose or switch medication class.
Augmentation strategies include adding bupropion, aripiprazole, or lithium.
Continue medication for at least 6-12 months after remission to prevent relapse.
Referral to psychiatry for treatment-resistant depression (failed 2 or more adequate medication trials).""",
        "category": "psychiatry"
    }
]


# ============================================================
# RAG Infrastructure
# ============================================================

def simple_chunk(text, chunk_size=80, overlap=15):
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
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_index(documents):
    index = []
    for doc in documents:
        chunks = simple_chunk(doc["content"])
        for i, chunk_text in enumerate(chunks):
            embedding = get_embedding(chunk_text)
            index.append({
                "doc_id": doc["id"],
                "title": doc["title"],
                "category": doc["category"],
                "chunk_index": i,
                "text": chunk_text,
                "embedding": embedding
            })
    return index


def retrieve(query, index, top_k=3):
    query_embedding = get_embedding(query)
    scored = []
    for item in index:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append({**item, "similarity": score})
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


# ============================================================
# Confidence Scoring System
# ============================================================

def calculate_confidence(retrieved_chunks):
    """
    Calculate a confidence score based on retrieval quality.

    Factors considered:
    1. Top similarity score — how well does the best chunk match?
    2. Average similarity — overall retrieval quality
    3. Similarity gap — is top chunk much better than the rest? (specificity)
    4. Source diversity — more sources can mean broader or noisier context

    Returns: confidence (0-100), level (HIGH/MEDIUM/LOW), explanation
    """
    if not retrieved_chunks:
        return 0, "NONE", "No chunks retrieved"

    similarities = [c["similarity"] for c in retrieved_chunks]
    top_sim = similarities[0]
    avg_sim = np.mean(similarities)
    min_sim = similarities[-1]

    # Factor 1: Top similarity (0-40 points)
    # Scale: 0.85+ = 40pts, 0.80 = 30pts, 0.75 = 20pts, 0.70 = 10pts, <0.65 = 0
    if top_sim >= 0.85:
        top_score = 40
    elif top_sim >= 0.80:
        top_score = 30
    elif top_sim >= 0.75:
        top_score = 20
    elif top_sim >= 0.70:
        top_score = 10
    else:
        top_score = max(0, int((top_sim - 0.55) * 100))

    # Factor 2: Average similarity (0-30 points)
    if avg_sim >= 0.80:
        avg_score = 30
    elif avg_sim >= 0.75:
        avg_score = 22
    elif avg_sim >= 0.70:
        avg_score = 15
    elif avg_sim >= 0.65:
        avg_score = 8
    else:
        avg_score = 0

    # Factor 3: Consistency — low variance means all chunks are relevant (0-20 points)
    spread = top_sim - min_sim
    if spread < 0.03:
        consistency_score = 20  # Very consistent
    elif spread < 0.06:
        consistency_score = 15
    elif spread < 0.10:
        consistency_score = 10
    else:
        consistency_score = 5  # Big spread — some chunks may be noisy

    # Factor 4: Source agreement — multiple docs agreeing is good for complex Q (0-10 points)
    unique_sources = len(set(c["title"] for c in retrieved_chunks))
    if unique_sources == 1:
        source_score = 10  # Single authoritative source
    elif unique_sources == 2:
        source_score = 8
    else:
        source_score = 5   # Many sources — broader but possibly diluted

    total = top_score + avg_score + consistency_score + source_score

    # Determine level
    if total >= 75:
        level = "HIGH"
    elif total >= 50:
        level = "MEDIUM"
    elif total >= 30:
        level = "LOW"
    else:
        level = "VERY LOW"

    explanation = (
        f"Top match: {top_sim:.4f} ({top_score}/40) | "
        f"Avg: {avg_sim:.4f} ({avg_score}/30) | "
        f"Consistency: {consistency_score}/20 | "
        f"Sources: {source_score}/10"
    )

    return total, level, explanation


def generate_with_confidence(query, retrieved_chunks, confidence, level):
    """Generate answer with confidence-aware prompt"""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(f"[Source {i+1}: {chunk['title']}] (relevance: {chunk['similarity']:.2f})\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    # Adjust system prompt based on confidence
    if level == "HIGH":
        confidence_instruction = "The retrieved context is highly relevant. Provide a clear, definitive answer."
    elif level == "MEDIUM":
        confidence_instruction = "The retrieved context is moderately relevant. Answer with appropriate caveats where the context is incomplete."
    elif level == "LOW":
        confidence_instruction = "The retrieved context may not fully address the question. Be transparent about gaps and state what you CAN answer."
    else:
        confidence_instruction = "The retrieved context has LOW relevance. State clearly that the knowledge base may not cover this topic. Only answer based on what the context actually says."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""You are a medical knowledge assistant with a confidence-aware system.
{confidence_instruction}
Answer ONLY from the provided context. Cite sources as [Source X].
If context is insufficient, explicitly say so.
Educational purposes only — not clinical advice."""},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer (confidence: {level}, score: {confidence}/100):"}
        ],
        max_tokens=400, temperature=0.2
    )
    return response.choices[0].message.content


# ============================================================
# Demo 1: Confidence Scoring in Action
# ============================================================

def demo_confidence_scoring(index):
    """Show confidence scores for different types of questions"""
    print("\n" + "=" * 70)
    print("DEMO 1: CONFIDENCE SCORING IN ACTION")
    print("=" * 70)

    questions = [
        # Should be HIGH confidence — directly in knowledge base
        ("What is the target HbA1c for most diabetic adults?", "HIGH expected"),
        # Should be MEDIUM — partially covered
        ("How should you manage hypertension in a diabetic patient with CKD?", "MEDIUM expected"),
        # Should be LOW — not in knowledge base
        ("What is the treatment for acute appendicitis?", "LOW expected"),
        # Should be HIGH — very specific, directly answerable
        ("What is the reversal agent for dabigatran?", "HIGH expected"),
    ]

    for question, expected in questions:
        print(f"\n{'─' * 70}")
        print(f"❓ {question}")
        print(f"   (Expected: {expected})")

        retrieved = retrieve(question, index, top_k=3)
        confidence, level, explanation = calculate_confidence(retrieved)

        # Confidence badge
        badge = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🟠", "VERY LOW": "🔴"}.get(level, "⚪")

        print(f"\n   {badge} CONFIDENCE: {level} ({confidence}/100)")
        print(f"   📊 {explanation}")
        print(f"\n   📎 Retrieved chunks:")
        for i, c in enumerate(retrieved):
            print(f"      {i+1}. [{c['title']}] sim: {c['similarity']:.4f}")

        answer = generate_with_confidence(question, retrieved, confidence, level)
        print(f"\n   📋 {answer}")

    return


# ============================================================
# Demo 2: Confidence Threshold Gateway
# ============================================================

def demo_confidence_gateway(index):
    """Only answer if confidence is above threshold"""
    print("\n" + "=" * 70)
    print("DEMO 2: CONFIDENCE THRESHOLD GATEWAY")
    print("=" * 70)
    print("""
🚦 This demo acts like a clinical safety gateway:
   🟢 HIGH confidence    → Full answer
   🟡 MEDIUM confidence  → Answer with caveats
   🔴 LOW confidence     → Refuse to answer, suggest alternatives
""")

    threshold = 50  # Minimum confidence to provide an answer

    questions = [
        "What is the first-line treatment for stage 1 hypertension?",
        "What are the dosing adjustments for apixaban in elderly patients?",
        "How do you treat a pulmonary embolism in pregnancy?",
        "What SSRI should I start for moderate depression?",
    ]

    for q in questions:
        print(f"\n❓ {q}")

        retrieved = retrieve(q, index, top_k=3)
        confidence, level, explanation = calculate_confidence(retrieved)

        badge = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🟠", "VERY LOW": "🔴"}.get(level, "⚪")
        print(f"   {badge} Confidence: {level} ({confidence}/100)")

        if confidence >= threshold:
            answer = generate_with_confidence(q, retrieved, confidence, level)
            print(f"   📋 {answer[:250]}...")
        else:
            print(f"   🚫 BLOCKED: Confidence ({confidence}) below threshold ({threshold})")
            print(f"   💡 This topic may not be in the knowledge base.")
            print(f"      Suggest: Consult clinical references or add relevant guidelines.")


# ============================================================
# Demo 3: Interactive Q&A with Confidence
# ============================================================

def demo_interactive(index):
    """Interactive Q&A showing confidence for every answer"""
    print("\n" + "=" * 70)
    print("DEMO 3: INTERACTIVE Q&A WITH CONFIDENCE SCORES")
    print("=" * 70)
    print("""
💬 Ask medical questions! Each answer shows confidence level.
   Topics covered: hypertension, diabetes, asthma, CKD, anticoagulation, depression
   Type 'quit' to exit.
""")

    while True:
        query = input("Your question: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue

        retrieved = retrieve(query, index, top_k=3)
        confidence, level, explanation = calculate_confidence(retrieved)

        badge = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🟠", "VERY LOW": "🔴"}.get(level, "⚪")

        print(f"\n   {badge} Confidence: {level} ({confidence}/100)")
        print(f"   📊 {explanation}")

        answer = generate_with_confidence(query, retrieved, confidence, level)
        print(f"\n   📋 {answer}\n")


# ============================================================
# Demo 4: Confidence Calibration
# ============================================================

def demo_calibration(index):
    """Test confidence scoring across a range of question types"""
    print("\n" + "=" * 70)
    print("DEMO 4: CONFIDENCE CALIBRATION TABLE")
    print("=" * 70)
    print("\n   Testing confidence across IN-DOMAIN and OUT-OF-DOMAIN questions\n")

    test_cases = [
        ("What is the starting dose of metformin?", True),
        ("What are the CKD stages based on GFR?", True),
        ("What biologics are used for severe asthma?", True),
        ("What is the HAS-BLED score used for?", True),
        ("How do you treat a fractured femur?", False),
        ("What is the treatment for acute pancreatitis?", False),
        ("How do you manage an opioid overdose?", False),
        ("What is the screening protocol for colon cancer?", False),
    ]

    print(f"   {'Question':<55} {'In KB':>5} {'Score':>6} {'Level':>10}")
    print(f"   {'─' * 55} {'─' * 5} {'─' * 6} {'─' * 10}")

    correct = 0
    total = len(test_cases)

    for question, in_domain in test_cases:
        retrieved = retrieve(question, index, top_k=3)
        confidence, level, _ = calculate_confidence(retrieved)

        badge = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🟠", "VERY LOW": "🔴"}.get(level, "⚪")

        # Check if confidence aligns with domain
        predicted_in_domain = confidence >= 50
        match = "✅" if predicted_in_domain == in_domain else "❌"
        if predicted_in_domain == in_domain:
            correct += 1

        in_label = "Yes" if in_domain else "No"
        print(f"   {question:<55} {in_label:>5} {confidence:>5} {badge} {level:<8} {match}")

    accuracy = correct / total * 100
    print(f"\n   📊 Calibration accuracy: {correct}/{total} ({accuracy:.0f}%)")
    print(f"      (Does confidence correctly predict in/out of domain?)")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n🎯 Exercise 4: Confidence Scoring")
    print("=" * 70)
    print("Know when your RAG system is confident vs guessing\n")

    print("🔨 Building index...")
    index = build_index(MEDICAL_KNOWLEDGE_BASE)
    print(f"   ✅ {len(index)} chunks indexed\n")

    print("Choose a demo:")
    print("1. Confidence scoring in action")
    print("2. Confidence threshold gateway")
    print("3. Interactive Q&A with confidence")
    print("4. Confidence calibration table")
    print("5. Run demos 1, 2, and 4")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_confidence_scoring(index)
    elif choice == "2":
        demo_confidence_gateway(index)
    elif choice == "3":
        demo_interactive(index)
    elif choice == "4":
        demo_calibration(index)
    elif choice == "5":
        demo_confidence_scoring(index)
        demo_confidence_gateway(index)
        demo_calibration(index)
    else:
        print("Invalid choice")

    print(f"""
{'═' * 70}
KEY LEARNINGS:
{'═' * 70}
1. Confidence scoring turns RAG from "always answers" to "knows its limits"
2. Key factors: top similarity, average similarity, consistency, source count
3. Confidence thresholds act as safety gates — critical for clinical use
4. OUT-OF-DOMAIN questions naturally get lower confidence scores
5. Calibration testing validates your scoring system

🏥 CLINICAL SAFETY:
   In healthcare AI, KNOWING WHEN YOU DON'T KNOW is as important
   as knowing the answer. A confident wrong answer is dangerous.
   Confidence scoring is a fundamental safety mechanism.

📊 SCORING BREAKDOWN:
   • Top similarity: 0-40 pts (how well does best chunk match?)
   • Average similarity: 0-30 pts (overall retrieval quality)
   • Consistency: 0-20 pts (are all chunks relevant?)
   • Source agreement: 0-10 pts (single authoritative vs scattered)

   🟢 75-100: HIGH    → Safe to present to clinicians
   🟡 50-74:  MEDIUM  → Present with caveats
   🟠 30-49:  LOW     → Flag for human review
   🔴 0-29:   VERY LOW → Don't answer, suggest alternatives
""")


if __name__ == "__main__":
    main()
