"""
Exercise 4: Create a Golden Test Dataset for Medical Q&A Evaluation

Skills practiced:
  - Designing golden test datasets with ground truth answers
  - Storing and loading evaluation data as JSON
  - Running automated evaluation against a golden dataset
  - Reporting per-question and aggregate scores
  - Tracking evaluation results over time

Healthcare context:
  A golden test dataset is the foundation of any evaluation pipeline. It contains
  curated questions with verified ground truth answers, relevant document references,
  and expected quality criteria. In healthcare, these datasets must be reviewed by
  domain experts to ensure the ground truth is medically accurate.

  This exercise creates a golden dataset spanning 8 medical specialties, stores it
  as JSON, evaluates a RAG system against it, and produces a detailed scorecard.
"""

import os
import json
import time
import numpy as np
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

GOLDEN_DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_results.json")


# ============================================================
# Golden Dataset Definition
# ============================================================

GOLDEN_DATASET = {
    "metadata": {
        "name": "Medical RAG Golden Test Dataset v1.0",
        "description": "Curated Q&A pairs across 8 medical specialties for RAG evaluation",
        "created": "2026-02-28",
        "num_questions": 15,
        "specialties": [
            "Cardiology", "Endocrinology", "Pulmonology", "Nephrology",
            "Infectious Disease", "Hematology", "Neurology", "Emergency Medicine",
        ],
    },
    "documents": [
        {"id": "card_001", "specialty": "Cardiology",
         "content": "Acute STEMI management: Aspirin 325mg immediately. Activate cath lab for primary PCI with door-to-balloon <90 min. Dual antiplatelet therapy with P2Y12 inhibitor (ticagrelor 180mg loading or clopidogrel 600mg). Anticoagulation with heparin. Supplemental O2 if SpO2 <90%. Nitroglycerin for ongoing chest pain (contraindicated if systolic BP <90, RV infarction, or PDE5 inhibitor use within 24-48h). Morphine if pain uncontrolled. Fibrinolysis if PCI not available within 120 min."},
        {"id": "card_002", "specialty": "Cardiology",
         "content": "Heart failure with reduced EF (≤40%): Four pillars of GDMT — ACEi/ARB/ARNI (sacubitril/valsartan preferred), beta-blocker (carvedilol, metoprolol succinate, bisoprolol), MRA (spironolactone/eplerenone), SGLT2i (dapagliflozin/empagliflozin). Start low, titrate to target doses. Loop diuretics for congestion. ICD if EF ≤35% on 3+ months GDMT. CRT if LBBB with QRS ≥150ms and EF ≤35%."},
        {"id": "endo_001", "specialty": "Endocrinology",
         "content": "Type 2 diabetes initial management: Lifestyle + metformin first-line. Start 500mg daily, titrate to 2000mg. Second-line based on comorbid conditions: ASCVD → GLP-1 RA (semaglutide, liraglutide); HF → SGLT2i; CKD → SGLT2i + finerenone. HbA1c target generally <7%. Insulin if HbA1c >10% or symptomatic hyperglycemia. Monitor A1c q3mo until stable, then q6mo."},
        {"id": "endo_002", "specialty": "Endocrinology",
         "content": "Diabetic ketoacidosis (DKA) management: IV normal saline 1-1.5 L/hr initially. Regular insulin 0.1 units/kg bolus then 0.1 units/kg/hr infusion. Monitor glucose hourly. When glucose <200, switch to D5 half-normal saline and reduce insulin to 0.02-0.05 units/kg/hr. Replace potassium if K <5.3 (add 20-40 mEq/L to IV fluids). Bicarbonate only if pH <6.9. Resolution: glucose <200, bicarb ≥15, pH >7.3, AG normal."},
        {"id": "pulm_001", "specialty": "Pulmonology",
         "content": "Asthma stepwise therapy: Step 1 SABA PRN. Step 2 low-dose ICS (fluticasone 88-264mcg/day). Step 3 low-dose ICS+LABA or medium ICS. Step 4 medium ICS+LABA. Step 5 high ICS+LABA ± biologic (omalizumab, dupilumab, mepolizumab). Uncontrolled: symptoms >2d/wk, nighttime >2x/mo, SABA >2d/wk. Step up after 2-6 weeks if uncontrolled. Step down after 3 months well-controlled."},
        {"id": "pulm_002", "specialty": "Pulmonology",
         "content": "Pulmonary embolism diagnosis and treatment: Use Wells score or Geneva score. D-dimer to rule out if low probability. CT pulmonary angiography (CTPA) for diagnosis. Massive PE (with hemodynamic instability): systemic thrombolysis with alteplase 100mg IV over 2 hours. Submassive/low-risk: anticoagulation with DOAC (rivaroxaban or apixaban preferred). Duration 3 months for provoked, extended for unprovoked. IVC filter if anticoagulation contraindicated."},
        {"id": "neph_001", "specialty": "Nephrology",
         "content": "CKD management by stage: All stages — BP control <130/80, ACEi/ARB for proteinuria, SGLT2i if eGFR ≥20. Stage 3-5: dietary protein restriction 0.8g/kg/day, phosphate binders if phosphorus elevated, erythropoietin-stimulating agents for Hgb <10. Stage 4-5: nephrology referral, AV fistula planning 6 months before anticipated dialysis. Stage 5: renal replacement therapy (hemodialysis, peritoneal dialysis, transplant). Avoid NSAIDs and nephrotoxins at all stages."},
        {"id": "id_001", "specialty": "Infectious Disease",
         "content": "Community-acquired pneumonia (CAP): Outpatient no comorbidities — amoxicillin 1g TID or doxycycline 100mg BID x 5-7 days. Outpatient with comorbidities — amoxicillin-clavulanate + macrolide or respiratory fluoroquinolone. Inpatient non-ICU — beta-lactam + macrolide or fluoroquinolone alone. ICU — beta-lactam + macrolide or beta-lactam + fluoroquinolone. Use CURB-65 or PSI for severity. Blood cultures and sputum before antibiotics if inpatient."},
        {"id": "id_002", "specialty": "Infectious Disease",
         "content": "Sepsis recognition and management: qSOFA screening (AMS, SBP ≤100, RR ≥22). SEP-1 bundle within 1 hour: lactate, blood cultures, broad-spectrum antibiotics, 30mL/kg crystalloid if hypotension or lactate ≥4. Norepinephrine first-line vasopressor for MAP <65 after fluids. Add vasopressin as second agent. Hydrocortisone 200mg/day if vasopressor-dependent. Source control. De-escalate antibiotics with culture data. Reassess at 3 and 6 hours."},
        {"id": "hem_001", "specialty": "Hematology",
         "content": "Warfarin management: Target INR 2-3 for AF, VTE. Target 2.5-3.5 for mechanical mitral valve. Initiate 5mg daily, check INR day 3-4. Adjust by 5-15% of weekly dose. Interactions: amiodarone, fluconazole, metronidazole INCREASE INR. Rifampin, carbamazepine, phenytoin DECREASE INR. Vitamin K-rich foods affect levels. Supratherapeutic INR: hold warfarin, vitamin K 2.5mg PO if INR 5-9, IV vitamin K + 4-factor PCC if bleeding."},
        {"id": "hem_002", "specialty": "Hematology",
         "content": "Iron deficiency anemia: Diagnosis — low ferritin (<30), low iron, high TIBC, low transferrin saturation (<20%). Microcytic hypochromic on smear. Oral iron: ferrous sulfate 325mg (65mg elemental iron) daily on empty stomach with vitamin C. If intolerant or non-responsive, IV iron (ferric carboxymaltose 750mg x2, or iron sucrose 200mg x5). Investigate cause: GI bleeding (colonoscopy if >50 or alarm features), menorrhagia, celiac disease, H. pylori."},
        {"id": "neuro_001", "specialty": "Neurology",
         "content": "Acute ischemic stroke: Last known well time is critical. If within 4.5 hours: IV alteplase 0.9mg/kg (max 90mg), 10% bolus then 60min infusion. Door-to-needle <60 min. If within 24 hours with large vessel occlusion and favorable imaging: mechanical thrombectomy. Contraindications to tPA: active bleeding, recent surgery <14 days, platelet <100K, INR >1.7, glucose <50. Post-tPA: BP <180/105, no anticoagulation x24h, repeat CT at 24h."},
        {"id": "em_001", "specialty": "Emergency Medicine",
         "content": "Anaphylaxis management: Epinephrine IM 0.3-0.5mg (1:1000) into anterolateral thigh — repeat every 5-15 min if needed. Adjuncts: IV fluids 1-2L NS bolus, albuterol for bronchospasm, diphenhydramine 50mg IV, ranitidine 50mg IV, methylprednisolone 125mg IV. Refractory: epinephrine infusion 0.1-1 mcg/kg/min. Observe 4-6 hours minimum (biphasic reaction risk). Prescribe EpiPen and allergy referral at discharge."},
        {"id": "em_002", "specialty": "Emergency Medicine",
         "content": "Acute chest pain workup: ECG within 10 minutes. Troponin at 0 and 3 hours (or high-sensitivity at 0 and 1 hour). Chest X-ray. If STEMI — activate cath lab. If NSTEMI — ASA + P2Y12 + heparin, risk stratify with TIMI/GRACE. Differential: ACS, PE, aortic dissection, pneumothorax, pericarditis, esophageal rupture. D-dimer if PE suspected. CT angiography if dissection suspected. HEART score for low-risk stratification."},
    ],
    "test_cases": [
        {"id": "tc_01", "specialty": "Cardiology",
         "question": "What is the door-to-balloon time target for STEMI?",
         "ground_truth": "Door-to-balloon time target for STEMI is less than 90 minutes.",
         "relevant_docs": ["card_001"],
         "difficulty": "easy",
         "key_facts": ["<90 minutes", "primary PCI"]},
        {"id": "tc_02", "specialty": "Cardiology",
         "question": "What are the four pillars of heart failure treatment for HFrEF?",
         "ground_truth": "The four pillars are ACEi/ARB/ARNI, beta-blocker, MRA, and SGLT2 inhibitor.",
         "relevant_docs": ["card_002"],
         "difficulty": "medium",
         "key_facts": ["ACEi/ARB/ARNI", "beta-blocker", "MRA", "SGLT2i"]},
        {"id": "tc_03", "specialty": "Endocrinology",
         "question": "What is the first-line treatment for type 2 diabetes?",
         "ground_truth": "Lifestyle modifications plus metformin, starting at 500mg daily and titrating to 2000mg.",
         "relevant_docs": ["endo_001"],
         "difficulty": "easy",
         "key_facts": ["metformin", "lifestyle", "500mg"]},
        {"id": "tc_04", "specialty": "Endocrinology",
         "question": "How is DKA insulin therapy managed?",
         "ground_truth": "Regular insulin 0.1 units/kg bolus then 0.1 units/kg/hr infusion. When glucose <200, reduce to 0.02-0.05 units/kg/hr.",
         "relevant_docs": ["endo_002"],
         "difficulty": "hard",
         "key_facts": ["0.1 units/kg", "bolus then infusion", "glucose <200 switch"]},
        {"id": "tc_05", "specialty": "Pulmonology",
         "question": "What biologics are used in Step 5 asthma management?",
         "ground_truth": "Biologics include omalizumab, dupilumab, and mepolizumab.",
         "relevant_docs": ["pulm_001"],
         "difficulty": "medium",
         "key_facts": ["omalizumab", "dupilumab", "mepolizumab"]},
        {"id": "tc_06", "specialty": "Pulmonology",
         "question": "What is the treatment for massive pulmonary embolism?",
         "ground_truth": "Systemic thrombolysis with alteplase 100mg IV over 2 hours for hemodynamically unstable PE.",
         "relevant_docs": ["pulm_002"],
         "difficulty": "medium",
         "key_facts": ["thrombolysis", "alteplase", "100mg", "hemodynamic instability"]},
        {"id": "tc_07", "specialty": "Nephrology",
         "question": "When should AV fistula planning begin for CKD patients?",
         "ground_truth": "AV fistula planning should begin 6 months before anticipated dialysis, in CKD stage 4-5.",
         "relevant_docs": ["neph_001"],
         "difficulty": "medium",
         "key_facts": ["6 months", "stage 4-5", "before dialysis"]},
        {"id": "tc_08", "specialty": "Infectious Disease",
         "question": "What is the outpatient antibiotic for CAP without comorbidities?",
         "ground_truth": "Amoxicillin 1g TID or doxycycline 100mg BID for 5-7 days.",
         "relevant_docs": ["id_001"],
         "difficulty": "easy",
         "key_facts": ["amoxicillin", "doxycycline", "5-7 days"]},
        {"id": "tc_09", "specialty": "Infectious Disease",
         "question": "What is the first-line vasopressor for septic shock?",
         "ground_truth": "Norepinephrine is the first-line vasopressor, targeting MAP ≥65 mmHg.",
         "relevant_docs": ["id_002"],
         "difficulty": "easy",
         "key_facts": ["norepinephrine", "MAP ≥65"]},
        {"id": "tc_10", "specialty": "Hematology",
         "question": "How should supratherapeutic INR between 5-9 be managed?",
         "ground_truth": "Hold warfarin and give vitamin K 2.5mg PO.",
         "relevant_docs": ["hem_001"],
         "difficulty": "medium",
         "key_facts": ["hold warfarin", "vitamin K", "2.5mg PO"]},
        {"id": "tc_11", "specialty": "Hematology",
         "question": "What is the diagnostic criteria for iron deficiency anemia?",
         "ground_truth": "Low ferritin (<30), low iron, high TIBC, low transferrin saturation (<20%), microcytic hypochromic on smear.",
         "relevant_docs": ["hem_002"],
         "difficulty": "medium",
         "key_facts": ["low ferritin", "high TIBC", "low transferrin saturation", "microcytic"]},
        {"id": "tc_12", "specialty": "Neurology",
         "question": "What is the tPA dosing for acute ischemic stroke?",
         "ground_truth": "IV alteplase 0.9mg/kg (max 90mg), 10% as bolus then remainder over 60 minutes.",
         "relevant_docs": ["neuro_001"],
         "difficulty": "hard",
         "key_facts": ["0.9mg/kg", "max 90mg", "10% bolus", "60 minutes"]},
        {"id": "tc_13", "specialty": "Emergency Medicine",
         "question": "What is the epinephrine dose for anaphylaxis?",
         "ground_truth": "Epinephrine 0.3-0.5mg IM (1:1000) into anterolateral thigh, repeat every 5-15 minutes.",
         "relevant_docs": ["em_001"],
         "difficulty": "easy",
         "key_facts": ["0.3-0.5mg", "IM", "1:1000", "anterolateral thigh"]},
        {"id": "tc_14", "specialty": "Emergency Medicine",
         "question": "What scoring system is used for low-risk chest pain stratification?",
         "ground_truth": "HEART score is used for low-risk chest pain stratification.",
         "relevant_docs": ["em_002"],
         "difficulty": "easy",
         "key_facts": ["HEART score"]},
        {"id": "tc_15", "specialty": "Cardiology",
         "question": "When is fibrinolysis indicated for STEMI?",
         "ground_truth": "Fibrinolysis is indicated when PCI is not available within 120 minutes.",
         "relevant_docs": ["card_001"],
         "difficulty": "medium",
         "key_facts": ["PCI not available", "120 minutes"]},
    ],
}


# ============================================================
# Helper Functions
# ============================================================

def get_embeddings_batch(texts: list) -> list:
    """Get embeddings for a batch of texts."""
    all_embs = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embs.extend([item.embedding for item in response.data])
    return all_embs


def get_embedding(text: str) -> list:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_top_k(query, doc_embeddings, documents, k=3):
    q_emb = get_embedding(query)
    sims = [cosine_similarity(q_emb, de) for de in doc_embeddings]
    ranked = np.argsort(sims)[::-1][:k]
    return [{"document": documents[i], "score": sims[i]} for i in ranked]


def generate_answer(question, context_docs):
    context = "\n\n".join(f"[{d['id']}]: {d['content']}" for d in context_docs)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are a medical assistant. Answer using ONLY the provided context. "
                    "Be specific about dosages, medications, and clinical details. "
                    "Do not fabricate information."
                )},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error: {e}]"


def call_llm_json(system_prompt, user_message):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Evaluation Functions
# ============================================================

def score_key_facts(answer: str, key_facts: list) -> dict:
    """
    Check how many key facts from the ground truth appear in the answer.
    Returns dict with individual fact results and overall score.
    """
    answer_lower = answer.lower()
    results = []
    for fact in key_facts:
        found = fact.lower() in answer_lower
        results.append({"fact": fact, "found": found})

    found_count = sum(1 for r in results if r["found"])
    return {
        "facts": results,
        "found": found_count,
        "total": len(key_facts),
        "score": found_count / len(key_facts) if key_facts else 0.0,
    }


def score_correctness_llm(answer: str, ground_truth: str) -> float:
    """Use LLM to score answer correctness against ground truth."""
    result = call_llm_json(
        system_prompt=(
            "Compare the answer to the ground truth. Score correctness from 0.0 to 1.0. "
            "1.0 means all key facts match. 0.0 means completely wrong. "
            "Respond with JSON: {\"correctness\": <float>, \"explanation\": \"...\"}"
        ),
        user_message=f"Ground Truth: {ground_truth}\n\nAnswer: {answer}",
    )
    return float(result.get("correctness", 0.0))


def score_faithfulness_llm(answer: str, context: str) -> float:
    """Check if answer is faithful to context."""
    result = call_llm_json(
        system_prompt=(
            "Evaluate if the answer only uses information from the context (no hallucination). "
            "Score 0.0 to 1.0. Respond with JSON: {\"faithfulness\": <float>}"
        ),
        user_message=f"Context:\n{context}\n\nAnswer:\n{answer}",
    )
    return float(result.get("faithfulness", 0.0))


def retrieval_hit(retrieved_ids: list, relevant_ids: list, k: int = 3) -> bool:
    """Check if any relevant doc is in the top-k retrieved."""
    return bool(set(retrieved_ids[:k]) & set(relevant_ids))


# ============================================================
# Save / Load Golden Dataset
# ============================================================

def save_golden_dataset():
    """Save the golden dataset to a JSON file."""
    with open(GOLDEN_DATASET_PATH, "w") as f:
        json.dump(GOLDEN_DATASET, f, indent=2)
    print(f"  Golden dataset saved to: {GOLDEN_DATASET_PATH}")
    print(f"  Documents: {len(GOLDEN_DATASET['documents'])}")
    print(f"  Test cases: {len(GOLDEN_DATASET['test_cases'])}")


def load_golden_dataset() -> dict:
    """Load the golden dataset from JSON."""
    if os.path.exists(GOLDEN_DATASET_PATH):
        with open(GOLDEN_DATASET_PATH, "r") as f:
            return json.load(f)
    return GOLDEN_DATASET


# ============================================================
# Main Evaluation Pipeline
# ============================================================

def run_evaluation():
    """
    Run the full evaluation pipeline against the golden dataset.
    """
    print("=" * 80)
    print("  Golden Dataset Evaluation for Medical RAG")
    print("=" * 80)

    # Step 1: Save golden dataset
    print("\nStep 1: Saving golden dataset to JSON...")
    save_golden_dataset()

    # Step 2: Load it back (demonstrating the workflow)
    print("\nStep 2: Loading golden dataset...")
    dataset = load_golden_dataset()
    documents = dataset["documents"]
    test_cases = dataset["test_cases"]
    print(f"  Loaded {len(test_cases)} test cases across {len(dataset['metadata']['specialties'])} specialties.")

    # Step 3: Embed documents
    print("\nStep 3: Embedding documents...")
    doc_texts = [d["content"] for d in documents]
    doc_embeddings = get_embeddings_batch(doc_texts)
    print(f"  Embedded {len(documents)} documents.")

    # Step 4: Evaluate each test case
    print("\nStep 4: Running evaluation...\n")
    results = []

    for tc in test_cases:
        tc_id = tc["id"]
        question = tc["question"]
        ground_truth = tc["ground_truth"]
        relevant_doc_ids = tc["relevant_docs"]
        key_facts = tc["key_facts"]
        difficulty = tc["difficulty"]
        specialty = tc["specialty"]

        print(f"  [{tc_id}] {question[:60]}...")

        # Retrieve
        retrieved = retrieve_top_k(question, doc_embeddings, documents, k=3)
        retrieved_ids = [r["document"]["id"] for r in retrieved]
        context_docs = [r["document"] for r in retrieved]
        context_text = "\n".join(d["content"] for d in context_docs)

        # Generate
        answer = generate_answer(question, context_docs)

        # Score
        hit = retrieval_hit(retrieved_ids, relevant_doc_ids, k=3)
        fact_result = score_key_facts(answer, key_facts)
        correctness = score_correctness_llm(answer, ground_truth)
        faithfulness = score_faithfulness_llm(answer, context_text)

        row = {
            "id": tc_id,
            "specialty": specialty,
            "difficulty": difficulty,
            "question": question,
            "answer": answer[:200],
            "retrieval_hit": hit,
            "key_fact_score": fact_result["score"],
            "key_facts_found": fact_result["found"],
            "key_facts_total": fact_result["total"],
            "correctness": correctness,
            "faithfulness": faithfulness,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_doc_ids,
        }
        results.append(row)

        status = "✅" if hit and correctness >= 0.7 else "⚠️" if hit else "❌"
        print(f"    {status} Hit={hit}  Facts={fact_result['found']}/{fact_result['total']}  "
              f"Corr={correctness:.2f}  Faith={faithfulness:.2f}")

    # Step 5: Save results
    print("\nStep 5: Saving evaluation results...")
    eval_output = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "num_test_cases": len(results),
        "results": results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(eval_output, f, indent=2, default=str)
    print(f"  Results saved to: {RESULTS_PATH}")

    # ============================================================
    # Step 6: Per-Question Report
    # ============================================================
    print("\n\n" + "=" * 90)
    print("PER-QUESTION SCORECARD")
    print("=" * 90)

    print(f"\n{'ID':<7} {'Specialty':<18} {'Diff':<7} {'Hit':>4} {'Facts':>6} {'Corr':>6} {'Faith':>6} {'Status'}")
    print("-" * 90)

    for r in results:
        status = "PASS" if r["retrieval_hit"] and r["correctness"] >= 0.7 else "FAIL"
        facts_str = f"{r['key_facts_found']}/{r['key_facts_total']}"
        print(f"{r['id']:<7} {r['specialty']:<18} {r['difficulty']:<7} "
              f"{'✓' if r['retrieval_hit'] else '✗':>4} {facts_str:>6} "
              f"{r['correctness']:>6.2f} {r['faithfulness']:>6.2f} {status:>6}")

    print("-" * 90)

    # ============================================================
    # Step 7: Aggregate Scores
    # ============================================================
    print("\n" + "=" * 60)
    print("AGGREGATE SCORES")
    print("=" * 60)

    hit_rate = np.mean([r["retrieval_hit"] for r in results])
    avg_facts = np.mean([r["key_fact_score"] for r in results])
    avg_corr = np.mean([r["correctness"] for r in results])
    avg_faith = np.mean([r["faithfulness"] for r in results])

    print(f"\n  Retrieval Hit Rate:       {hit_rate:.3f}  ({sum(r['retrieval_hit'] for r in results)}/{len(results)})")
    print(f"  Key Fact Coverage:        {avg_facts:.3f}")
    print(f"  Answer Correctness:       {avg_corr:.3f}")
    print(f"  Faithfulness:             {avg_faith:.3f}")

    overall = 0.25 * hit_rate + 0.25 * avg_facts + 0.25 * avg_corr + 0.25 * avg_faith
    print(f"\n  Overall Score:            {overall:.3f}")

    # ============================================================
    # Step 8: Breakdown by Specialty
    # ============================================================
    print("\n" + "=" * 60)
    print("SCORES BY SPECIALTY")
    print("=" * 60)

    specialties = set(r["specialty"] for r in results)
    print(f"\n{'Specialty':<20} {'#Q':>3} {'Hit%':>6} {'Facts':>6} {'Corr':>6} {'Faith':>6}")
    print("-" * 60)
    for spec in sorted(specialties):
        spec_results = [r for r in results if r["specialty"] == spec]
        n = len(spec_results)
        s_hit = np.mean([r["retrieval_hit"] for r in spec_results])
        s_fact = np.mean([r["key_fact_score"] for r in spec_results])
        s_corr = np.mean([r["correctness"] for r in spec_results])
        s_faith = np.mean([r["faithfulness"] for r in spec_results])
        print(f"{spec:<20} {n:>3} {s_hit:>6.2f} {s_fact:>6.2f} {s_corr:>6.2f} {s_faith:>6.2f}")

    # ============================================================
    # Step 9: Breakdown by Difficulty
    # ============================================================
    print("\n" + "=" * 60)
    print("SCORES BY DIFFICULTY")
    print("=" * 60)

    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in results if r["difficulty"] == diff]
        if not diff_results:
            continue
        n = len(diff_results)
        d_hit = np.mean([r["retrieval_hit"] for r in diff_results])
        d_corr = np.mean([r["correctness"] for r in diff_results])
        d_faith = np.mean([r["faithfulness"] for r in diff_results])
        print(f"\n  {diff.upper()} ({n} questions):")
        print(f"    Hit Rate:     {d_hit:.2f}")
        print(f"    Correctness:  {d_corr:.2f}")
        print(f"    Faithfulness: {d_faith:.2f}")

    # ============================================================
    # Step 10: Failure Analysis
    # ============================================================
    failures = [r for r in results if not r["retrieval_hit"] or r["correctness"] < 0.5]
    if failures:
        print("\n" + "=" * 60)
        print("FAILURE ANALYSIS")
        print("=" * 60)
        for f in failures:
            print(f"\n  {f['id']}: {f['question'][:60]}")
            if not f["retrieval_hit"]:
                print(f"    Issue: Retrieval miss — relevant docs {f['relevant_ids']} not in {f['retrieved_ids'][:3]}")
            if f["correctness"] < 0.5:
                print(f"    Issue: Low correctness ({f['correctness']:.2f}) — answer may be wrong or incomplete")
            missing_facts = [kf for i, kf in enumerate(
                next((tc["key_facts"] for tc in test_cases if tc["id"] == f["id"]), [])
            ) if not any(kf.lower() in f.get("answer", "").lower() for _ in [None])]
            if missing_facts:
                print(f"    Missing key facts: {missing_facts}")

    # Final grade
    print("\n" + "=" * 60)
    if overall >= 0.85:
        print("🟢 GRADE: EXCELLENT — RAG system passes the golden test dataset.")
    elif overall >= 0.7:
        print("🟡 GRADE: GOOD — Some test cases need attention.")
    elif overall >= 0.5:
        print("🟠 GRADE: FAIR — Significant gaps in retrieval or generation.")
    else:
        print("🔴 GRADE: POOR — Major rework needed.")
    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():
    print("🏥 Exercise 4: Golden Test Dataset Evaluation\n")
    print("Creates a golden test dataset with 15 medical Q&A pairs across")
    print("8 specialties, evaluates a RAG system against it, and reports")
    print("per-question and aggregate scores.\n")
    run_evaluation()


if __name__ == "__main__":
    main()
