"""
Project: RAG Evaluation — Measuring Retrieval and Generation Quality
Objective: Learn to rigorously evaluate RAG systems using retrieval and generation metrics
Concepts: Precision@k, Recall@k, MRR, Faithfulness, Answer Relevancy, End-to-End Evaluation

Healthcare Use Case: Evaluating a medical guidelines RAG system

A RAG system that retrieves the wrong guideline or hallucinates a dosage can cause patient
harm. This project teaches you how to measure every stage of the RAG pipeline — from
retrieval accuracy to generation faithfulness — so you can identify and fix problems
before they reach clinicians.

Key techniques:
  1. Retrieval Metrics: Precision@k, Recall@k, MRR to measure document retrieval quality
  2. Faithfulness Scoring: Detect hallucination by checking if answers are grounded in context
  3. Answer Relevancy: Verify the generated answer actually addresses the user's question
  4. End-to-End Evaluation: Combine all metrics into a single evaluation pipeline
"""

import os
import json
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


# ============================================================
# MEDICAL KNOWLEDGE BASE (inline documents)
# ============================================================

MEDICAL_GUIDELINES = [
    {
        "id": "guide_001",
        "title": "Hypertension Management — First-Line Therapy",
        "content": (
            "First-line antihypertensive agents include ACE inhibitors (e.g., lisinopril, "
            "enalapril), ARBs (e.g., losartan, valsartan), calcium channel blockers (e.g., "
            "amlodipine, nifedipine), and thiazide diuretics (e.g., hydrochlorothiazide, "
            "chlorthalidone). Choice depends on patient comorbidities. ACE inhibitors are "
            "preferred in patients with diabetes or chronic kidney disease. Target blood "
            "pressure is generally <130/80 mmHg for most adults. Lifestyle modifications "
            "including dietary sodium restriction, weight loss, and regular exercise should "
            "accompany pharmacotherapy."
        ),
        "topic": "hypertension",
    },
    {
        "id": "guide_002",
        "title": "Type 2 Diabetes — Metformin Guidelines",
        "content": (
            "Metformin is the preferred initial pharmacologic agent for type 2 diabetes. "
            "Starting dose is 500mg once or twice daily, titrated to a maximum of 2000-2550mg "
            "daily in divided doses. Contraindications include eGFR <30 mL/min/1.73m², "
            "acute or chronic metabolic acidosis, and hypersensitivity. Common side effects "
            "are gastrointestinal: nausea, diarrhea, abdominal discomfort. Lactic acidosis "
            "is a rare but serious adverse effect. Monitor renal function at baseline and "
            "at least annually. Hold metformin before iodinated contrast procedures if "
            "eGFR is 30-60."
        ),
        "topic": "diabetes",
    },
    {
        "id": "guide_003",
        "title": "Warfarin Therapy — Drug and Food Interactions",
        "content": (
            "Warfarin has numerous clinically significant interactions. Vitamin K-rich foods "
            "(green leafy vegetables such as spinach, kale, broccoli) can reduce warfarin "
            "efficacy. Cranberry juice and grapefruit may increase warfarin effect. Drug "
            "interactions include: NSAIDs increase bleeding risk, amiodarone increases INR, "
            "rifampin decreases INR, and many antibiotics (e.g., fluconazole, metronidazole) "
            "potentiate warfarin. Target INR for most indications is 2.0-3.0. For mechanical "
            "heart valves, target is 2.5-3.5. Monitor INR weekly when initiating therapy, "
            "then monthly when stable."
        ),
        "topic": "anticoagulation",
    },
    {
        "id": "guide_004",
        "title": "Acute Chest Pain — Emergency Evaluation",
        "content": (
            "Acute chest pain requires rapid evaluation. Key differential includes acute "
            "coronary syndrome (ACS), pulmonary embolism (PE), aortic dissection, tension "
            "pneumothorax, and pericarditis. Initial workup: 12-lead ECG within 10 minutes, "
            "troponin levels (serial at 0 and 3 hours), chest X-ray, and D-dimer if PE "
            "suspected. High-risk features include ST-segment elevation, hemodynamic instability, "
            "and positive troponin. STEMI requires emergent PCI within 90 minutes of first "
            "medical contact. For NSTEMI, antiplatelet therapy with aspirin and P2Y12 inhibitor, "
            "anticoagulation with heparin, and risk stratification using TIMI or GRACE scores."
        ),
        "topic": "emergency",
    },
    {
        "id": "guide_005",
        "title": "Asthma — Stepwise Management",
        "content": (
            "Asthma management follows a stepwise approach. Step 1: SABA as needed (albuterol). "
            "Step 2: Low-dose ICS (e.g., fluticasone 88-264mcg/day). Step 3: Low-dose ICS + "
            "LABA (e.g., fluticasone/salmeterol) or medium-dose ICS. Step 4: Medium-dose ICS + "
            "LABA. Step 5: High-dose ICS + LABA ± oral corticosteroids or biologic therapy. "
            "Assess control every 1-6 months. Step up if not well-controlled, step down if "
            "well-controlled for 3+ months. Key indicators of uncontrolled asthma: symptoms "
            ">2 days/week, nighttime awakenings >2x/month, SABA use >2 days/week, any "
            "activity limitation."
        ),
        "topic": "pulmonology",
    },
    {
        "id": "guide_006",
        "title": "Chronic Kidney Disease — Staging and Management",
        "content": (
            "CKD is staged by eGFR: Stage 1 (≥90, with kidney damage), Stage 2 (60-89), "
            "Stage 3a (45-59), Stage 3b (30-44), Stage 4 (15-29), Stage 5 (<15 or dialysis). "
            "Management includes blood pressure control (target <130/80), RAAS inhibition "
            "with ACE inhibitor or ARB, SGLT2 inhibitors for diabetic and non-diabetic CKD, "
            "and dietary protein restriction (0.8g/kg/day in stages 3-5). Avoid nephrotoxins: "
            "NSAIDs, aminoglycosides, iodinated contrast (with precautions). Monitor potassium, "
            "phosphorus, calcium, and hemoglobin. Refer to nephrology when eGFR <30 or rapidly "
            "declining. Prepare for renal replacement therapy when eGFR approaches 15-20."
        ),
        "topic": "nephrology",
    },
    {
        "id": "guide_007",
        "title": "Sepsis — Early Recognition and Bundle",
        "content": (
            "Sepsis is defined as life-threatening organ dysfunction caused by dysregulated "
            "host response to infection. Use qSOFA for screening: altered mental status, "
            "systolic BP ≤100 mmHg, respiratory rate ≥22. The 1-hour bundle includes: "
            "measure lactate, obtain blood cultures before antibiotics, administer broad-spectrum "
            "antibiotics, begin rapid IV crystalloid (30 mL/kg) for hypotension or lactate ≥4, "
            "apply vasopressors if hypotensive during or after fluid resuscitation to maintain "
            "MAP ≥65 mmHg. Reassess volume status and tissue perfusion. Norepinephrine is the "
            "first-line vasopressor. Source control should be achieved as soon as practical."
        ),
        "topic": "critical_care",
    },
    {
        "id": "guide_008",
        "title": "Heart Failure — Pharmacotherapy",
        "content": (
            "Guideline-directed medical therapy (GDMT) for HFrEF (EF ≤40%) includes four "
            "pillars: ACE inhibitor/ARB/ARNI (sacubitril/valsartan preferred), beta-blocker "
            "(carvedilol, metoprolol succinate, or bisoprolol), mineralocorticoid receptor "
            "antagonist (spironolactone or eplerenone), and SGLT2 inhibitor (dapagliflozin "
            "or empagliflozin). Initiate at low doses and titrate to target. Monitor potassium "
            "and renal function. Diuretics (furosemide, bumetanide) for volume management. "
            "Hydralazine/isosorbide dinitrate is an alternative to ACE/ARB in patients who "
            "cannot tolerate RAAS inhibitors. ICD for primary prevention if EF ≤35% despite "
            "3 months of optimal medical therapy."
        ),
        "topic": "cardiology",
    },
]


# ============================================================
# EVALUATION DATA — Queries with Ground Truth
# ============================================================

EVAL_QUERIES = [
    {
        "question": "What is the first-line treatment for hypertension?",
        "relevant_doc_ids": ["guide_001"],
        "ground_truth": (
            "First-line agents include ACE inhibitors, ARBs, calcium channel blockers, "
            "and thiazide diuretics. Target BP is generally <130/80 mmHg."
        ),
    },
    {
        "question": "What are the contraindications for metformin?",
        "relevant_doc_ids": ["guide_002"],
        "ground_truth": (
            "Contraindications include eGFR <30 mL/min/1.73m², acute or chronic "
            "metabolic acidosis, and hypersensitivity to metformin."
        ),
    },
    {
        "question": "What foods interact with warfarin?",
        "relevant_doc_ids": ["guide_003"],
        "ground_truth": (
            "Vitamin K-rich foods such as green leafy vegetables (spinach, kale, broccoli) "
            "reduce warfarin efficacy. Cranberry juice and grapefruit may increase effect."
        ),
    },
    {
        "question": "What is the initial workup for acute chest pain?",
        "relevant_doc_ids": ["guide_004"],
        "ground_truth": (
            "Initial workup includes 12-lead ECG within 10 minutes, serial troponin levels, "
            "chest X-ray, and D-dimer if pulmonary embolism is suspected."
        ),
    },
    {
        "question": "How is asthma managed in a stepwise approach?",
        "relevant_doc_ids": ["guide_005"],
        "ground_truth": (
            "Step 1: SABA as needed. Step 2: Low-dose ICS. Step 3: Low-dose ICS + LABA "
            "or medium-dose ICS. Step 4: Medium-dose ICS + LABA. Step 5: High-dose ICS + "
            "LABA with possible oral steroids or biologics."
        ),
    },
    {
        "question": "What medications are used for heart failure with reduced ejection fraction?",
        "relevant_doc_ids": ["guide_008"],
        "ground_truth": (
            "GDMT includes four pillars: ACE inhibitor/ARB/ARNI, beta-blocker, "
            "mineralocorticoid receptor antagonist, and SGLT2 inhibitor."
        ),
    },
    {
        "question": "How is sepsis initially managed?",
        "relevant_doc_ids": ["guide_007"],
        "ground_truth": (
            "The 1-hour bundle: measure lactate, blood cultures before antibiotics, "
            "broad-spectrum antibiotics, rapid IV crystalloid for hypotension, "
            "vasopressors to maintain MAP ≥65 mmHg."
        ),
    },
    {
        "question": "What are the stages of chronic kidney disease?",
        "relevant_doc_ids": ["guide_006"],
        "ground_truth": (
            "CKD stages: Stage 1 (eGFR ≥90 with damage), Stage 2 (60-89), "
            "Stage 3a (45-59), Stage 3b (30-44), Stage 4 (15-29), Stage 5 (<15)."
        ),
    },
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_embedding(text: str) -> list:
    """Get embedding vector for a text string."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list) -> list:
    """Get embeddings for a batch of texts."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


def cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_top_k(query: str, doc_embeddings: list, documents: list, k: int = 3) -> list:
    """
    Retrieve top-k most similar documents for a query.

    Returns:
        List of dicts with 'document', 'score', and 'rank'.
    """
    query_embedding = get_embedding(query)
    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in doc_embeddings
    ]
    ranked_indices = np.argsort(similarities)[::-1][:k]
    results = []
    for rank, idx in enumerate(ranked_indices, 1):
        results.append({
            "document": documents[idx],
            "score": similarities[idx],
            "rank": rank,
        })
    return results


def generate_answer(question: str, context_docs: list) -> str:
    """Generate an answer using retrieved context documents."""
    context_text = "\n\n".join(
        f"[{doc['title']}]: {doc['content']}" for doc in context_docs
    )
    system_prompt = (
        "You are a medical assistant. Answer the question using ONLY the provided context. "
        "If the context does not contain enough information, say so. Do not make up facts. "
        "Be specific and cite information from the context."
    )
    user_message = f"Context:\n{context_text}\n\nQuestion: {question}"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error: {e}]"


def call_llm(system_prompt: str, user_message: str, temperature: float = 0.0,
             max_tokens: int = 500) -> str:
    """Make a single LLM call and return the response text."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error: {e}]"


def print_separator(title: str):
    """Print a formatted section separator."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def precision_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    """
    Precision@k: Of the top-k retrieved documents, how many are relevant?

    precision@k = |relevant ∩ retrieved[:k]| / k
    """
    top_k = retrieved_ids[:k]
    relevant_retrieved = set(top_k) & set(relevant_ids)
    return len(relevant_retrieved) / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    """
    Recall@k: Of all relevant documents, how many appear in the top-k?

    recall@k = |relevant ∩ retrieved[:k]| / |relevant|
    """
    top_k = retrieved_ids[:k]
    relevant_retrieved = set(top_k) & set(relevant_ids)
    return len(relevant_retrieved) / len(relevant_ids) if relevant_ids else 0.0


def mean_reciprocal_rank(retrieved_ids: list, relevant_ids: list) -> float:
    """
    MRR: Reciprocal of the rank of the first relevant document.

    MRR = 1 / rank_of_first_relevant
    """
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    """
    NDCG@k: Normalized Discounted Cumulative Gain.
    Uses binary relevance (1 if relevant, 0 otherwise).
    """
    top_k = retrieved_ids[:k]
    dcg = sum(
        (1.0 if doc_id in relevant_ids else 0.0) / np.log2(rank + 1)
        for rank, doc_id in enumerate(top_k, 1)
    )
    # Ideal DCG: all relevant docs at the top
    ideal_relevant_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_relevant_count + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# DEMO 1: Retrieval Metrics
# ============================================================

def demo_retrieval_metrics():
    """
    Embed all medical guideline documents, retrieve for each query,
    and compute precision@k, recall@k, MRR, and NDCG@k.
    """
    print_separator("DEMO 1: Retrieval Metrics")

    print("Embedding medical guideline documents...")
    doc_texts = [doc["content"] for doc in MEDICAL_GUIDELINES]
    doc_embeddings = get_embeddings_batch(doc_texts)
    print(f"  Embedded {len(doc_embeddings)} documents.\n")

    k_values = [1, 3, 5]
    all_metrics = {k: {"precision": [], "recall": [], "mrr": [], "ndcg": []} for k in k_values}

    for query_data in EVAL_QUERIES:
        question = query_data["question"]
        relevant_ids = query_data["relevant_doc_ids"]

        print(f"Query: {question}")
        results = retrieve_top_k(question, doc_embeddings, MEDICAL_GUIDELINES, k=5)
        retrieved_ids = [r["document"]["id"] for r in results]

        print(f"  Retrieved: {retrieved_ids}")
        print(f"  Relevant:  {relevant_ids}")

        for k in k_values:
            p = precision_at_k(retrieved_ids, relevant_ids, k)
            r = recall_at_k(retrieved_ids, relevant_ids, k)
            mrr = mean_reciprocal_rank(retrieved_ids, relevant_ids)
            n = ndcg_at_k(retrieved_ids, relevant_ids, k)

            all_metrics[k]["precision"].append(p)
            all_metrics[k]["recall"].append(r)
            all_metrics[k]["mrr"].append(mrr)
            all_metrics[k]["ndcg"].append(n)

            if k == 3:
                print(f"  Precision@3={p:.2f}  Recall@3={r:.2f}  MRR={mrr:.2f}  NDCG@3={n:.2f}")
        print()

    # Aggregate results
    print("=" * 55)
    print(f"{'Metric':<15} {'k=1':>8} {'k=3':>8} {'k=5':>8}")
    print("-" * 55)
    for metric_name in ["precision", "recall", "mrr", "ndcg"]:
        row = f"{metric_name.upper():<15}"
        for k in k_values:
            avg = np.mean(all_metrics[k][metric_name])
            row += f" {avg:>7.3f}"
        print(row)
    print("=" * 55)


# ============================================================
# DEMO 2: Faithfulness Scoring
# ============================================================

def score_faithfulness(answer: str, context: str) -> dict:
    """
    Score whether an answer is faithful to the retrieved context.
    A faithful answer uses ONLY information present in the context.
    """
    system_prompt = """You are an expert evaluator for medical AI systems.
Your job is to assess whether an answer is FAITHFUL to the provided context.

Faithfulness means:
- Every claim in the answer can be traced back to the context
- No information is fabricated or hallucinated
- No external knowledge is added beyond what the context provides

Analyze the answer claim by claim. For each claim, determine if it is:
- SUPPORTED: Directly supported by the context
- NOT SUPPORTED: Not found in or contradicted by the context

Respond ONLY with valid JSON:
{
  "claims": [{"claim": "...", "verdict": "SUPPORTED" or "NOT SUPPORTED"}],
  "faithfulness_score": <float between 0.0 and 1.0>,
  "explanation": "brief explanation"
}"""

    user_message = f"Context:\n{context}\n\nAnswer to evaluate:\n{answer}"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)
    except Exception as e:
        return {"faithfulness_score": 0.0, "explanation": f"Error: {e}", "claims": []}


def demo_faithfulness_scoring():
    """
    Generate answers from retrieved context, then check if the answers
    are faithful (not hallucinated).
    """
    print_separator("DEMO 2: Faithfulness Scoring")

    print("Embedding documents for retrieval...")
    doc_texts = [doc["content"] for doc in MEDICAL_GUIDELINES]
    doc_embeddings = get_embeddings_batch(doc_texts)

    test_queries = EVAL_QUERIES[:4]  # First 4 for demo
    faithfulness_scores = []

    for query_data in test_queries:
        question = query_data["question"]
        print(f"Question: {question}")

        # Retrieve top-3 documents
        results = retrieve_top_k(question, doc_embeddings, MEDICAL_GUIDELINES, k=3)
        context_docs = [r["document"] for r in results]
        context_text = "\n".join(doc["content"] for doc in context_docs)

        # Generate answer
        answer = generate_answer(question, context_docs)
        print(f"  Answer: {answer[:150]}...")

        # Score faithfulness
        faith_result = score_faithfulness(answer, context_text)
        score = faith_result.get("faithfulness_score", 0.0)
        faithfulness_scores.append(score)

        print(f"  Faithfulness Score: {score:.2f}")

        # Show claim analysis
        claims = faith_result.get("claims", [])
        for claim in claims[:3]:  # Show up to 3 claims
            verdict = claim.get("verdict", "UNKNOWN")
            marker = "✅" if verdict == "SUPPORTED" else "❌"
            print(f"    {marker} {claim.get('claim', '')[:80]}")

        print()

    avg_faith = np.mean(faithfulness_scores) if faithfulness_scores else 0.0
    print(f"Average Faithfulness Score: {avg_faith:.2f}")
    if avg_faith >= 0.9:
        print("Excellent! Answers are highly grounded in retrieved context.")
    elif avg_faith >= 0.7:
        print("Good faithfulness, but some claims may need verification.")
    else:
        print("⚠️  Low faithfulness — the RAG system may be hallucinating.")


# ============================================================
# DEMO 3: Answer Relevancy
# ============================================================

def score_answer_relevancy(question: str, answer: str) -> dict:
    """
    Score whether an answer is relevant to the question asked.
    An answer is relevant if it directly and completely addresses the question.
    """
    system_prompt = """You are an expert evaluator for medical AI systems.
Assess whether the answer is RELEVANT to the question.

Relevancy means:
- The answer directly addresses the question asked
- Key aspects of the question are covered
- The answer is not off-topic or tangential

Score on these dimensions:
1. Directness: Does the answer directly address the question? (0.0-1.0)
2. Completeness: Does the answer cover the key aspects? (0.0-1.0)
3. Conciseness: Is the answer focused without unnecessary information? (0.0-1.0)

Respond ONLY with valid JSON:
{
  "directness": <float>,
  "completeness": <float>,
  "conciseness": <float>,
  "overall_relevancy": <float between 0.0 and 1.0>,
  "explanation": "brief explanation"
}"""

    user_message = f"Question: {question}\n\nAnswer: {answer}"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)
    except Exception as e:
        return {
            "directness": 0.0, "completeness": 0.0, "conciseness": 0.0,
            "overall_relevancy": 0.0, "explanation": f"Error: {e}",
        }


def demo_answer_relevancy():
    """
    Generate answers, then score whether they are relevant to the questions.
    """
    print_separator("DEMO 3: Answer Relevancy")

    print("Embedding documents for retrieval...")
    doc_texts = [doc["content"] for doc in MEDICAL_GUIDELINES]
    doc_embeddings = get_embeddings_batch(doc_texts)

    test_queries = EVAL_QUERIES[:4]
    relevancy_scores = []

    for query_data in test_queries:
        question = query_data["question"]
        print(f"Question: {question}")

        # Retrieve and generate
        results = retrieve_top_k(question, doc_embeddings, MEDICAL_GUIDELINES, k=3)
        context_docs = [r["document"] for r in results]
        answer = generate_answer(question, context_docs)
        print(f"  Answer: {answer[:150]}...")

        # Score relevancy
        rel_result = score_answer_relevancy(question, answer)
        overall = rel_result.get("overall_relevancy", 0.0)
        relevancy_scores.append(overall)

        print(f"  Directness:   {rel_result.get('directness', 0.0):.2f}")
        print(f"  Completeness: {rel_result.get('completeness', 0.0):.2f}")
        print(f"  Conciseness:  {rel_result.get('conciseness', 0.0):.2f}")
        print(f"  Overall:      {overall:.2f}")
        print()

    avg_rel = np.mean(relevancy_scores) if relevancy_scores else 0.0
    print(f"Average Answer Relevancy: {avg_rel:.2f}")
    if avg_rel >= 0.85:
        print("Excellent! Answers are highly relevant to the questions.")
    elif avg_rel >= 0.7:
        print("Good relevancy. Some answers could be more focused.")
    else:
        print("⚠️  Low relevancy — the system may not be answering the right questions.")


# ============================================================
# DEMO 4: End-to-End Evaluation
# ============================================================

def demo_end_to_end_evaluation():
    """
    Combine all metrics into a comprehensive evaluation run.
    """
    print_separator("DEMO 4: End-to-End RAG Evaluation")

    print("Building evaluation pipeline...")
    print("Step 1: Embedding documents...")
    doc_texts = [doc["content"] for doc in MEDICAL_GUIDELINES]
    doc_embeddings = get_embeddings_batch(doc_texts)

    results_table = []
    k = 3

    for query_data in EVAL_QUERIES:
        question = query_data["question"]
        relevant_ids = query_data["relevant_doc_ids"]

        print(f"\nEvaluating: {question[:60]}...")

        # Step 2: Retrieve
        retrieved = retrieve_top_k(question, doc_embeddings, MEDICAL_GUIDELINES, k=k)
        retrieved_ids = [r["document"]["id"] for r in retrieved]
        context_docs = [r["document"] for r in retrieved]
        context_text = "\n".join(doc["content"] for doc in context_docs)

        # Step 3: Generate
        answer = generate_answer(question, context_docs)

        # Step 4: Compute retrieval metrics
        p_at_k = precision_at_k(retrieved_ids, relevant_ids, k)
        r_at_k = recall_at_k(retrieved_ids, relevant_ids, k)
        mrr = mean_reciprocal_rank(retrieved_ids, relevant_ids)

        # Step 5: Compute generation metrics
        faith_result = score_faithfulness(answer, context_text)
        faith_score = faith_result.get("faithfulness_score", 0.0)

        rel_result = score_answer_relevancy(question, answer)
        rel_score = rel_result.get("overall_relevancy", 0.0)

        row = {
            "question": question[:50],
            "precision@k": p_at_k,
            "recall@k": r_at_k,
            "mrr": mrr,
            "faithfulness": faith_score,
            "relevancy": rel_score,
        }
        results_table.append(row)

        print(f"  P@{k}={p_at_k:.2f}  R@{k}={r_at_k:.2f}  MRR={mrr:.2f}  "
              f"Faith={faith_score:.2f}  Rel={rel_score:.2f}")

    # Summary report
    print("\n" + "=" * 80)
    print("END-TO-END EVALUATION REPORT")
    print("=" * 80)
    print(f"\n{'Question':<52} {'P@k':>5} {'R@k':>5} {'MRR':>5} {'Faith':>6} {'Rel':>5}")
    print("-" * 80)
    for row in results_table:
        print(f"{row['question']:<52} {row['precision@k']:>5.2f} {row['recall@k']:>5.2f} "
              f"{row['mrr']:>5.2f} {row['faithfulness']:>6.2f} {row['relevancy']:>5.2f}")
    print("-" * 80)

    # Averages
    avg = {
        metric: np.mean([row[metric] for row in results_table])
        for metric in ["precision@k", "recall@k", "mrr", "faithfulness", "relevancy"]
    }
    print(f"{'AVERAGE':<52} {avg['precision@k']:>5.2f} {avg['recall@k']:>5.2f} "
          f"{avg['mrr']:>5.2f} {avg['faithfulness']:>6.2f} {avg['relevancy']:>5.2f}")
    print("=" * 80)

    # Overall RAG quality score
    rag_score = (
        0.2 * avg["precision@k"]
        + 0.2 * avg["recall@k"]
        + 0.1 * avg["mrr"]
        + 0.25 * avg["faithfulness"]
        + 0.25 * avg["relevancy"]
    )
    print(f"\nOverall RAG Quality Score: {rag_score:.2f}")
    if rag_score >= 0.85:
        print("🟢 Excellent — RAG system is performing well across all dimensions.")
    elif rag_score >= 0.7:
        print("🟡 Good — Some areas need improvement. Check lowest-scoring queries.")
    else:
        print("🔴 Needs Work — Significant issues in retrieval or generation quality.")


# ============================================================
# MAIN MENU
# ============================================================

def main():
    """
    Run RAG evaluation demos with a menu interface.
    """
    print("🏥 Level 4.2: RAG Evaluation\n")
    print("This project demonstrates how to measure retrieval and generation")
    print("quality for healthcare RAG systems.\n")

    while True:
        print("\n" + "=" * 50)
        print("MAIN MENU")
        print("=" * 50)
        print("1. Retrieval Metrics (Precision, Recall, MRR)")
        print("2. Faithfulness Scoring (Hallucination Detection)")
        print("3. Answer Relevancy Scoring")
        print("4. End-to-End RAG Evaluation")
        print("5. Run All Demos (1-4)")
        print("q. Quit")
        print("=" * 50)

        choice = input("Select demo (1-5 or q): ").strip().lower()

        if choice == "1":
            demo_retrieval_metrics()
        elif choice == "2":
            demo_faithfulness_scoring()
        elif choice == "3":
            demo_answer_relevancy()
        elif choice == "4":
            demo_end_to_end_evaluation()
        elif choice == "5":
            demo_retrieval_metrics()
            demo_faithfulness_scoring()
            demo_answer_relevancy()
            demo_end_to_end_evaluation()
        elif choice in ("q", "quit", "exit"):
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-5 or q.")


if __name__ == "__main__":
    main()
