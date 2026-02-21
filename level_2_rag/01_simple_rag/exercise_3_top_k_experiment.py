"""
Exercise 3: Top-K Retrieval Experiments
Try different top_k values (1, 3, 5) — more context vs more noise?

Skills practiced:
- Understanding how top_k affects answer quality
- Measuring the trade-off between recall and precision
- Analyzing relevance scores to find the "sweet spot"
- Cost analysis for different k values

Healthcare context:
  Retrieving too few chunks may miss critical clinical details.
  Retrieving too many may dilute the answer with irrelevant info.
  For medications, you need precision. For complex diagnoses, you need breadth.
"""

import os
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# Knowledge Base (all 6 documents for variety)
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
    """Build index with default settings"""
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
    """Retrieve top_k chunks, returning ALL with similarity scores"""
    query_embedding = get_embedding(query)
    scored = []
    for item in index:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append({**item, "similarity": score})
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k], scored  # Return top_k AND full list for analysis


def generate_answer(query, retrieved_chunks):
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(f"[Source {i+1}: {chunk['title']}]\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a medical assistant. Answer using ONLY the provided context.
Cite sources as [Source X]. Be specific and concise. Educational purposes only."""},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ],
        max_tokens=400, temperature=0.2
    )
    return response.choices[0].message.content, response.usage


# ============================================================
# Demo 1: Top-K Comparison
# ============================================================

def demo_topk_comparison(index):
    """Compare k=1, k=3, k=5 on the same questions"""
    print("\n" + "=" * 70)
    print("DEMO 1: TOP-K COMPARISON (k=1 vs k=3 vs k=5)")
    print("=" * 70)

    questions = [
        # Focused — k=1 should be enough
        "What is the target INR for atrial fibrillation on warfarin?",
        # Multi-faceted — larger k needed
        "A patient has diabetes, CKD, and hypertension. What are the treatment priorities?",
        # Moderately broad
        "What are the side effects to monitor when starting an SSRI?",
    ]

    k_values = [1, 3, 5]

    for q in questions:
        print(f"\n{'═' * 70}")
        print(f"❓ {q}")
        print(f"{'═' * 70}")

        for k in k_values:
            top_k_chunks, all_scored = retrieve(q, index, top_k=k)
            answer, usage = generate_answer(q, top_k_chunks)

            # Analyze retrieved chunks
            sims = [c["similarity"] for c in top_k_chunks]
            avg_sim = np.mean(sims)
            min_sim = np.min(sims)
            sources = set(c["title"] for c in top_k_chunks)
            context_words = sum(len(c["text"].split()) for c in top_k_chunks)

            print(f"\n   📦 k={k}:")
            print(f"      Similarities: {' | '.join(f'{s:.4f}' for s in sims)}")
            print(f"      Avg similarity: {avg_sim:.4f} | Min: {min_sim:.4f}")
            print(f"      Sources: {', '.join(sources)}")
            print(f"      Context words: {context_words} | Tokens: {usage.total_tokens}")
            print(f"      📋 {answer[:200]}...")


# ============================================================
# Demo 2: Similarity Score Distribution
# ============================================================

def demo_score_distribution(index):
    """Analyze the full distribution of retrieval scores"""
    print("\n" + "=" * 70)
    print("DEMO 2: SIMILARITY SCORE DISTRIBUTION")
    print("=" * 70)
    print("\n   See ALL chunk scores to understand retrieval quality\n")

    questions = [
        "What is the starting dose of metformin?",
        "How do you manage a patient with multiple comorbidities?",
    ]

    for q in questions:
        _, all_scored = retrieve(q, index, top_k=1)
        all_sims = [c["similarity"] for c in all_scored]

        print(f"❓ {q}")
        print(f"   Total chunks: {len(all_sims)}")
        print(f"   Top 1:  {all_sims[0]:.4f}")
        print(f"   Top 3:  {np.mean(all_sims[:3]):.4f} avg")
        print(f"   Top 5:  {np.mean(all_sims[:5]):.4f} avg")
        print(f"   Bottom: {all_sims[-1]:.4f}")
        print(f"   Gap (1st - 2nd): {all_sims[0] - all_sims[1]:.4f}")
        print(f"   Gap (3rd - 4th): {all_sims[2] - all_sims[3]:.4f}")

        # Visual distribution
        print(f"\n   Score distribution (all {len(all_sims)} chunks):")
        buckets = {"0.80+": 0, "0.75-0.80": 0, "0.70-0.75": 0, "0.65-0.70": 0, "<0.65": 0}
        for s in all_sims:
            if s >= 0.80:
                buckets["0.80+"] += 1
            elif s >= 0.75:
                buckets["0.75-0.80"] += 1
            elif s >= 0.70:
                buckets["0.70-0.75"] += 1
            elif s >= 0.65:
                buckets["0.65-0.70"] += 1
            else:
                buckets["<0.65"] += 1

        for bucket, count in buckets.items():
            bar = "█" * count
            print(f"   {bucket:>10}: {bar} ({count})")

        print()


# ============================================================
# Demo 3: Cost vs Quality Analysis
# ============================================================

def demo_cost_analysis(index):
    """Quantify the cost of different k values"""
    print("\n" + "=" * 70)
    print("DEMO 3: COST vs QUALITY ANALYSIS")
    print("=" * 70)

    question = "What medications should be avoided in chronic kidney disease?"

    print(f"\n❓ {question}\n")

    results = []
    for k in [1, 2, 3, 5, 7]:
        top_k_chunks, _ = retrieve(question, index, top_k=k)
        answer, usage = generate_answer(question, top_k_chunks)

        sims = [c["similarity"] for c in top_k_chunks]
        cost = usage.total_tokens * 0.00000015

        results.append({
            "k": k,
            "tokens": usage.total_tokens,
            "cost": cost,
            "avg_sim": np.mean(sims),
            "min_sim": np.min(sims),
            "sources": len(set(c["title"] for c in top_k_chunks)),
            "answer_len": len(answer),
        })

        print(f"   k={k}: {usage.total_tokens} tokens | ${cost:.6f} | "
              f"avg sim: {np.mean(sims):.4f} | min sim: {np.min(sims):.4f} | "
              f"{len(set(c['title'] for c in top_k_chunks))} sources")

    print(f"""
{'─' * 70}
📊 COST SCALING:
   k=1 → k=5 token increase: {results[3]['tokens'] - results[0]['tokens']} tokens
   k=1 → k=5 cost increase: ${results[3]['cost'] - results[0]['cost']:.6f}

💡 DIMINISHING RETURNS:
   Adding chunks beyond the relevant ones INCREASES cost but may 
   DECREASE quality if irrelevant context dilutes the answer.
   
   Watch the min_similarity — when it drops sharply, you're adding noise!
""")


# ============================================================
# Demo 4: Smart K Selection
# ============================================================

def demo_smart_k(index):
    """Automatically choose k based on similarity scores"""
    print("\n" + "=" * 70)
    print("DEMO 4: SMART TOP-K SELECTION")
    print("=" * 70)
    print("\n   Instead of fixed k, choose k based on similarity thresholds\n")

    def smart_retrieve(query, index, min_similarity=0.72, max_k=5):
        """Retrieve chunks that meet a minimum similarity threshold"""
        query_embedding = get_embedding(query)
        scored = []
        for item in index:
            score = cosine_similarity(query_embedding, item["embedding"])
            scored.append({**item, "similarity": score})
        scored.sort(key=lambda x: x["similarity"], reverse=True)

        # Filter by threshold, up to max_k
        selected = [c for c in scored[:max_k] if c["similarity"] >= min_similarity]

        # Always return at least 1
        if not selected:
            selected = scored[:1]

        return selected

    questions = [
        "What is the target INR for warfarin in atrial fibrillation?",
        "How should you manage a patient with diabetes, depression, and CKD?",
        "What is the reversal agent for dabigatran?",
    ]

    for q in questions:
        print(f"❓ {q}")

        # Fixed k=3
        fixed_chunks, _ = retrieve(q, index, top_k=3)
        fixed_sims = [c["similarity"] for c in fixed_chunks]

        # Smart k
        smart_chunks = smart_retrieve(q, index, min_similarity=0.72, max_k=5)
        smart_sims = [c["similarity"] for c in smart_chunks]

        print(f"   Fixed k=3: sims = {[f'{s:.3f}' for s in fixed_sims]}")
        print(f"   Smart k:   sims = {[f'{s:.3f}' for s in smart_sims]} (k={len(smart_chunks)})")

        answer, usage = generate_answer(q, smart_chunks)
        print(f"   📋 {answer[:150]}...")
        print()

    print(f"""
💡 SMART K APPROACH:
   Instead of fixed top_k, use a similarity THRESHOLD:
   • Only include chunks above a minimum relevance score
   • Focused questions → fewer chunks (less noise)
   • Complex questions → more chunks (broader context)
   • This adapts automatically to each query!
""")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n🔢 Exercise 3: Top-K Retrieval Experiments")
    print("=" * 70)
    print("More context vs more noise — finding the sweet spot\n")

    print("🔨 Building index (one-time)...")
    index = build_index(MEDICAL_KNOWLEDGE_BASE)
    print(f"   ✅ {len(index)} chunks indexed\n")

    print("Choose a demo:")
    print("1. Top-K comparison (k=1 vs k=3 vs k=5)")
    print("2. Similarity score distribution")
    print("3. Cost vs quality analysis")
    print("4. Smart K selection (adaptive)")
    print("5. Run all demos")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_topk_comparison(index)
    elif choice == "2":
        demo_score_distribution(index)
    elif choice == "3":
        demo_cost_analysis(index)
    elif choice == "4":
        demo_smart_k(index)
    elif choice == "5":
        demo_topk_comparison(index)
        demo_score_distribution(index)
        demo_cost_analysis(index)
        demo_smart_k(index)
    else:
        print("Invalid choice")

    print(f"""
{'═' * 70}
KEY LEARNINGS:
{'═' * 70}
1. k=1 is precise but fragile — miss one chunk, miss the answer
2. k=3 is a good default — balances coverage and noise
3. k=5+ adds more context but may dilute with irrelevant chunks
4. Watch the similarity GAP between chunks — big drop = noise boundary
5. Smart/adaptive k using thresholds is better than fixed k
6. Cost scales linearly with k (more context = more prompt tokens)

🏥 CLINICAL GUIDELINES:
   • Drug lookup (specific): k=1-2 is sufficient
   • Treatment protocol (moderate): k=2-3 works well
   • Complex patient (multi-comorbidity): k=3-5 needed
   • Differential diagnosis: k=4-5 for broad coverage

   Match top_k to the COMPLEXITY of the clinical question!
""")


if __name__ == "__main__":
    main()
