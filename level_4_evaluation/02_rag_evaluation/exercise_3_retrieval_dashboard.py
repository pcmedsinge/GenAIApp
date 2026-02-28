"""
Exercise 3: Build a Text-Based Retrieval Quality Dashboard

Skills practiced:
  - Computing precision/recall at multiple k values
  - Identifying failure cases in retrieval
  - Building formatted text-based reports and visualizations
  - Top-k sensitivity analysis

Healthcare context:
  When deploying a medical RAG system, you need ongoing monitoring of retrieval
  quality. If the system fails to retrieve the correct drug interaction guideline,
  the downstream generation will be wrong — no matter how good the LLM is.

  This exercise builds a text-based dashboard that shows:
    1. Precision/Recall curves across k=1 through k=8
    2. Per-query retrieval scores
    3. Failure case analysis (which queries fail and why)
    4. Top-k sensitivity analysis

  Run on 12 medical queries to get statistically meaningful results.
"""

import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"


# ============================================================
# Medical Knowledge Base (12+ documents)
# ============================================================

DOCUMENTS = [
    {"id": "d01", "title": "ACE Inhibitors",
     "content": "ACE inhibitors (lisinopril, enalapril, ramipril) reduce angiotensin II production. First-line for hypertension with diabetes or CKD. Contraindicated in pregnancy, bilateral renal artery stenosis, and history of angioedema. Common side effects: dry cough, hyperkalemia, dizziness. Monitor potassium and creatinine."},
    {"id": "d02", "title": "Beta Blockers",
     "content": "Beta blockers (metoprolol, carvedilol, atenolol, propranolol) reduce heart rate and blood pressure. Indicated for heart failure (carvedilol, metoprolol succinate, bisoprolol), post-MI, and rate control in atrial fibrillation. Avoid in acute decompensated HF, severe bradycardia, and uncontrolled asthma. Carvedilol has alpha-blocking properties."},
    {"id": "d03", "title": "Statins",
     "content": "Statins (atorvastatin, rosuvastatin, simvastatin, pravastatin) are HMG-CoA reductase inhibitors for hyperlipidemia. High-intensity: atorvastatin 40-80mg, rosuvastatin 20-40mg. Monitor LFTs at baseline. Side effects include myalgia, rarely rhabdomyolysis. Avoid simvastatin with strong CYP3A4 inhibitors. Check CK if muscle symptoms develop."},
    {"id": "d04", "title": "Metformin",
     "content": "Metformin is first-line for type 2 diabetes. Starting dose 500mg daily, titrate to 2000mg in divided doses. Contraindicated if eGFR <30. GI side effects common (nausea, diarrhea). Lactic acidosis is rare but serious. Hold before iodinated contrast if eGFR 30-60. B12 deficiency with long-term use — monitor levels periodically."},
    {"id": "d05", "title": "Insulin Therapy",
     "content": "Basal insulin (glargine, detemir, degludec) for type 2 diabetes when oral agents insufficient. Start 10 units or 0.1-0.2 units/kg at bedtime. Titrate by 2 units every 3 days targeting fasting glucose 80-130 mg/dL. Hypoglycemia is the main risk. Educate on injection technique, storage, and sick day rules. Prandial insulin added if postprandial glucose remains elevated."},
    {"id": "d06", "title": "Warfarin Management",
     "content": "Warfarin is a vitamin K antagonist for anticoagulation. Target INR 2-3 for most indications, 2.5-3.5 for mechanical heart valves. Drug interactions: amiodarone and fluconazole increase INR, rifampin decreases INR. Food interactions: vitamin K-rich foods (spinach, kale, broccoli). Monitor INR weekly initially, then monthly. Reversal: vitamin K, 4-factor PCC for emergencies."},
    {"id": "d07", "title": "Direct Oral Anticoagulants",
     "content": "DOACs (apixaban, rivaroxaban, dabigatran, edoxaban) for VTE and non-valvular AF. Apixaban 5mg BID (reduce to 2.5mg if age ≥80, weight ≤60kg, Cr ≥1.5). Rivaroxaban 20mg daily with food. No routine INR monitoring needed. Renal dose adjustments required. Reversal: idarucizumab for dabigatran, andexanet alfa for apixaban/rivaroxaban."},
    {"id": "d08", "title": "Asthma Stepwise Therapy",
     "content": "Asthma management: Step 1 SABA PRN. Step 2 low-dose ICS. Step 3 low-dose ICS+LABA or medium ICS. Step 4 medium ICS+LABA. Step 5 high ICS+LABA plus biologic or oral steroids. Assess control every 1-6 months. Step up if uncontrolled (symptoms >2 days/week, nocturnal >2x/month). Step down after 3 months well-controlled. Albuterol for rescue."},
    {"id": "d09", "title": "COPD Management",
     "content": "COPD management: GOLD groups based on symptoms and exacerbation history. Group A: bronchodilator PRN. Group B: LAMA (tiotropium) or LABA. Group E: LAMA+LABA ± ICS if eos ≥300. Pulmonary rehabilitation improves exercise capacity and quality of life. Smoking cessation is the single most effective intervention. Supplemental oxygen if PaO2 ≤55 or SpO2 ≤88%."},
    {"id": "d10", "title": "Sepsis Management",
     "content": "Sepsis 1-hour bundle: measure lactate, blood cultures before antibiotics, broad-spectrum antibiotics, 30 mL/kg crystalloid for hypotension or lactate ≥4 mmol/L, vasopressors for MAP <65 after fluids. First-line vasopressor: norepinephrine. Reassess volume status and tissue perfusion regularly. Source control within 6-12 hours. De-escalate antibiotics based on culture results."},
    {"id": "d11", "title": "Acute Kidney Injury",
     "content": "AKI defined by KDIGO criteria: increase in serum creatinine ≥0.3 mg/dL within 48 hours, or ≥1.5x baseline within 7 days, or urine output <0.5 mL/kg/hr for 6 hours. Prerenal (60%): dehydration, heart failure, sepsis. Intrinsic: ATN, glomerulonephritis, interstitial nephritis. Postrenal: obstruction. Management: treat underlying cause, optimize volume, avoid nephrotoxins, monitor electrolytes."},
    {"id": "d12", "title": "Heart Failure Pharmacotherapy",
     "content": "HFrEF (EF ≤40%) four pillars: ACEi/ARB/ARNI (sacubitril/valsartan preferred), evidence-based beta-blocker (carvedilol, metoprolol succinate, bisoprolol), MRA (spironolactone/eplerenone), SGLT2 inhibitor (dapagliflozin/empagliflozin). Diuretics for volume management. Hydralazine/isosorbide dinitrate if RAAS intolerant. ICD if EF ≤35% on 3 months GDMT. CRT if LBBB with QRS ≥150ms."},
    {"id": "d13", "title": "DVT Diagnosis and Treatment",
     "content": "DVT diagnosis: Wells score for pretest probability. Low probability + negative D-dimer rules out DVT. Confirm with compression ultrasonography. Treatment: apixaban 10mg BID x 7 days then 5mg BID, or rivaroxaban 15mg BID x 21 days then 20mg daily. Duration 3 months for provoked, extended for unprovoked. IVC filter if anticoagulation contraindicated."},
    {"id": "d14", "title": "Pneumonia Treatment",
     "content": "CAP outpatient: amoxicillin 1g TID or doxycycline 100mg BID (no comorbidities). With comorbidities: amoxicillin-clavulanate + macrolide, or respiratory fluoroquinolone. Inpatient: beta-lactam + macrolide or fluoroquinolone alone. ICU: beta-lactam + macrolide or beta-lactam + fluoroquinolone. CURB-65 for severity. Duration 5-7 days."},
]


# ============================================================
# Evaluation Queries (12 queries)
# ============================================================

QUERIES = [
    {"question": "What are the side effects of ACE inhibitors?", "relevant": ["d01"]},
    {"question": "Which beta blockers are used for heart failure?", "relevant": ["d02", "d12"]},
    {"question": "What are the high-intensity statin options?", "relevant": ["d03"]},
    {"question": "What are the contraindications for metformin?", "relevant": ["d04"]},
    {"question": "How is basal insulin initiated and titrated?", "relevant": ["d05"]},
    {"question": "What foods interact with warfarin?", "relevant": ["d06"]},
    {"question": "What is the reversal agent for dabigatran?", "relevant": ["d07"]},
    {"question": "How is asthma control assessed and stepped up?", "relevant": ["d08"]},
    {"question": "What is the most effective intervention for COPD?", "relevant": ["d09"]},
    {"question": "What is the first-line vasopressor for sepsis?", "relevant": ["d10"]},
    {"question": "What are the KDIGO criteria for acute kidney injury?", "relevant": ["d11"]},
    {"question": "What are the four pillars of HFrEF treatment?", "relevant": ["d02", "d12"]},
]


# ============================================================
# Helper Functions
# ============================================================

def get_embeddings_batch(texts: list) -> list:
    """Embed a batch of texts."""
    all_embs = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embs.extend([item.embedding for item in response.data])
    return all_embs


def get_embedding(text: str) -> list:
    """Embed a single text."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_ranked(query: str, doc_embeddings: list, documents: list, k: int) -> list:
    """Retrieve top-k documents, returning list of doc ids in ranked order."""
    q_emb = get_embedding(query)
    sims = [cosine_similarity(q_emb, de) for de in doc_embeddings]
    ranked = np.argsort(sims)[::-1][:k]
    return [{"id": documents[i]["id"], "title": documents[i]["title"],
             "score": sims[i], "rank": r + 1} for r, i in enumerate(ranked)]


# ============================================================
# Retrieval Metrics
# ============================================================

def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    return len(set(top_k) & set(relevant_ids)) / k if k > 0 else 0.0


def recall_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    return len(set(top_k) & set(relevant_ids)) / len(relevant_ids) if relevant_ids else 0.0


def mrr(retrieved_ids, relevant_ids):
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved_ids, relevant_ids):
    """Compute average precision for a single query."""
    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / len(relevant_ids) if relevant_ids else 0.0


def hit_at_k(retrieved_ids, relevant_ids, k):
    """1 if any relevant doc is in top-k, else 0."""
    return 1.0 if set(retrieved_ids[:k]) & set(relevant_ids) else 0.0


# ============================================================
# Dashboard Functions
# ============================================================

def print_header(title: str, width: int = 80):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_precision_recall_curves(query_results: list, max_k: int = 8):
    """Display precision and recall curves across k values."""
    print_header("PRECISION / RECALL CURVES (averaged across all queries)")

    k_values = list(range(1, max_k + 1))
    avg_precision = []
    avg_recall = []

    for k in k_values:
        p_vals = [precision_at_k(r["retrieved_ids"], r["relevant_ids"], k) for r in query_results]
        r_vals = [recall_at_k(r["retrieved_ids"], r["relevant_ids"], k) for r in query_results]
        avg_precision.append(np.mean(p_vals))
        avg_recall.append(np.mean(r_vals))

    # Text-based chart
    print(f"\n{'k':>3}  {'Precision':>10}  {'Recall':>10}  Precision (visual)          Recall (visual)")
    print("-" * 80)
    for i, k in enumerate(k_values):
        p = avg_precision[i]
        r = avg_recall[i]
        p_bar = "█" * int(p * 25) + "░" * (25 - int(p * 25))
        r_bar = "█" * int(r * 25) + "░" * (25 - int(r * 25))
        print(f"  {k:>1}  {p:>10.3f}  {r:>10.3f}  {p_bar}  {r_bar}")

    print(f"\nTrend: Precision {'decreases' if avg_precision[0] > avg_precision[-1] else 'varies'} "
          f"as k increases (more noise).")
    print(f"       Recall {'increases' if avg_recall[-1] > avg_recall[0] else 'varies'} "
          f"as k increases (more coverage).")


def print_per_query_table(query_results: list):
    """Display per-query retrieval scores."""
    print_header("PER-QUERY RETRIEVAL SCORES")

    print(f"\n{'#':>2}  {'Query':<50} {'P@3':>5} {'R@3':>5} {'MRR':>5} {'AP':>5} {'Hit@1':>5}")
    print("-" * 80)

    for i, r in enumerate(query_results, 1):
        q_short = r["question"][:48]
        ids = r["retrieved_ids"]
        rel = r["relevant_ids"]
        p3 = precision_at_k(ids, rel, 3)
        r3 = recall_at_k(ids, rel, 3)
        m = mrr(ids, rel)
        ap = average_precision(ids, rel)
        h1 = hit_at_k(ids, rel, 1)
        print(f"{i:>2}  {q_short:<50} {p3:>5.2f} {r3:>5.2f} {m:>5.2f} {ap:>5.2f} {h1:>5.0f}")

    print("-" * 80)

    # Averages
    avg_p = np.mean([precision_at_k(r["retrieved_ids"], r["relevant_ids"], 3) for r in query_results])
    avg_r = np.mean([recall_at_k(r["retrieved_ids"], r["relevant_ids"], 3) for r in query_results])
    avg_m = np.mean([mrr(r["retrieved_ids"], r["relevant_ids"]) for r in query_results])
    avg_ap = np.mean([average_precision(r["retrieved_ids"], r["relevant_ids"]) for r in query_results])
    avg_h = np.mean([hit_at_k(r["retrieved_ids"], r["relevant_ids"], 1) for r in query_results])
    print(f"{'':>2}  {'AVERAGE':<50} {avg_p:>5.2f} {avg_r:>5.2f} {avg_m:>5.2f} {avg_ap:>5.2f} {avg_h:>5.1f}")


def print_failure_analysis(query_results: list):
    """Identify and analyze queries where retrieval failed."""
    print_header("FAILURE CASE ANALYSIS")

    failures = []
    for r in query_results:
        h1 = hit_at_k(r["retrieved_ids"], r["relevant_ids"], 1)
        r3 = recall_at_k(r["retrieved_ids"], r["relevant_ids"], 3)
        if h1 == 0 or r3 < 1.0:
            failures.append(r)

    if not failures:
        print("\n  No failures detected! All relevant documents retrieved in top-3.")
        return

    print(f"\n  Found {len(failures)} queries with retrieval issues:\n")

    for i, r in enumerate(failures, 1):
        h1 = hit_at_k(r["retrieved_ids"], r["relevant_ids"], 1)
        r3 = recall_at_k(r["retrieved_ids"], r["relevant_ids"], 3)

        print(f"  Failure {i}: {r['question']}")
        print(f"    Expected: {r['relevant_ids']}")
        print(f"    Retrieved (top-5): {r['retrieved_ids'][:5]}")
        print(f"    Hit@1: {h1:.0f}  Recall@3: {r3:.2f}")

        # Analyze why it failed
        missing = set(r["relevant_ids"]) - set(r["retrieved_ids"][:3])
        if missing:
            print(f"    Missing from top-3: {list(missing)}")

        # Check where relevant docs actually ranked
        for rel_id in r["relevant_ids"]:
            if rel_id in r["retrieved_ids"]:
                rank = r["retrieved_ids"].index(rel_id) + 1
                score = r["scores"][r["retrieved_ids"].index(rel_id)]
                print(f"    → {rel_id} found at rank {rank} (score: {score:.4f})")
            else:
                print(f"    → {rel_id} NOT in top results")

        # Suggest improvement
        if len(r["relevant_ids"]) > 1:
            print(f"    💡 This query needs multiple documents — consider increasing k or query expansion.")
        else:
            print(f"    💡 The relevant document may need better representation or the query may be ambiguous.")
        print()


def print_topk_sensitivity(query_results: list):
    """Show how metrics change as k varies."""
    print_header("TOP-K SENSITIVITY ANALYSIS")

    k_values = [1, 2, 3, 5, 8]
    print(f"\n{'k':>3}  {'Hit Rate':>9}  {'Precision':>10}  {'Recall':>8}  {'MAP':>6}  Recommendation")
    print("-" * 75)

    for k in k_values:
        hit = np.mean([hit_at_k(r["retrieved_ids"], r["relevant_ids"], k) for r in query_results])
        p = np.mean([precision_at_k(r["retrieved_ids"], r["relevant_ids"], k) for r in query_results])
        r = np.mean([recall_at_k(r["retrieved_ids"], r["relevant_ids"], k) for r in query_results])
        ap = np.mean([average_precision(r["retrieved_ids"], r["relevant_ids"]) for r in query_results])

        rec = ""
        if k == 1:
            rec = "Strict — misses multi-doc queries"
        elif k == 3:
            rec = "← Good balance of precision/recall"
        elif k == 5:
            rec = "High recall, more noise"
        elif k == 8:
            rec = "Max recall, lowest precision"

        print(f"  {k:>1}  {hit:>9.3f}  {p:>10.3f}  {r:>8.3f}  {ap:>6.3f}  {rec}")

    print()
    # Find optimal k (best F1)
    best_k = 1
    best_f1 = 0.0
    for k in k_values:
        p = np.mean([precision_at_k(r["retrieved_ids"], r["relevant_ids"], k) for r in query_results])
        r = np.mean([recall_at_k(r["retrieved_ids"], r["relevant_ids"], k) for r in query_results])
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_k = k

    print(f"  Optimal k (by F1 score): k={best_k} (F1={best_f1:.3f})")


def print_summary_dashboard(query_results: list):
    """Print a high-level summary."""
    print_header("RETRIEVAL QUALITY DASHBOARD SUMMARY")

    total = len(query_results)
    hit1 = sum(1 for r in query_results if hit_at_k(r["retrieved_ids"], r["relevant_ids"], 1))
    hit3 = sum(1 for r in query_results if hit_at_k(r["retrieved_ids"], r["relevant_ids"], 3))
    avg_mrr = np.mean([mrr(r["retrieved_ids"], r["relevant_ids"]) for r in query_results])
    avg_map = np.mean([average_precision(r["retrieved_ids"], r["relevant_ids"]) for r in query_results])

    print(f"""
  Total queries evaluated:  {total}
  Hit@1 (correct at rank 1): {hit1}/{total} ({hit1/total*100:.0f}%)
  Hit@3 (correct in top 3):  {hit3}/{total} ({hit3/total*100:.0f}%)
  Mean Reciprocal Rank:      {avg_mrr:.3f}
  Mean Average Precision:    {avg_map:.3f}
""")

    # Overall grade
    if avg_mrr >= 0.9:
        grade = "🟢 EXCELLENT — Retrieval is highly accurate."
    elif avg_mrr >= 0.7:
        grade = "🟡 GOOD — Most queries retrieve the right documents."
    elif avg_mrr >= 0.5:
        grade = "🟠 FAIR — Retrieval needs improvement for production use."
    else:
        grade = "🔴 POOR — Significant retrieval issues detected."
    print(f"  Overall Grade: {grade}")


# ============================================================
# Main Dashboard Runner
# ============================================================

def run_dashboard():
    """Build and display the full retrieval quality dashboard."""
    print("=" * 80)
    print("  MEDICAL RAG RETRIEVAL QUALITY DASHBOARD")
    print("=" * 80)

    print("\nEmbedding document collection...")
    doc_texts = [d["content"] for d in DOCUMENTS]
    doc_embeddings = get_embeddings_batch(doc_texts)
    print(f"  Embedded {len(DOCUMENTS)} documents.\n")

    print("Running retrieval for all queries...")
    max_k = 8
    query_results = []

    for q in QUERIES:
        retrieved = retrieve_ranked(q["question"], doc_embeddings, DOCUMENTS, k=max_k)
        retrieved_ids = [r["id"] for r in retrieved]
        scores = [r["score"] for r in retrieved]

        query_results.append({
            "question": q["question"],
            "relevant_ids": q["relevant"],
            "retrieved_ids": retrieved_ids,
            "scores": scores,
            "retrieved_details": retrieved,
        })

    print(f"  Evaluated {len(query_results)} queries.\n")

    # Display all dashboard sections
    print_summary_dashboard(query_results)
    print_per_query_table(query_results)
    print_precision_recall_curves(query_results, max_k=max_k)
    print_topk_sensitivity(query_results)
    print_failure_analysis(query_results)

    print("\n" + "=" * 80)
    print("  Dashboard complete. Use these insights to improve your RAG retrieval.")
    print("=" * 80)


# ============================================================
# Main
# ============================================================

def main():
    print("🏥 Exercise 3: Retrieval Quality Dashboard\n")
    print("Builds a comprehensive text-based dashboard showing retrieval")
    print("quality metrics across 12 medical queries.\n")
    run_dashboard()


if __name__ == "__main__":
    main()
