"""
Exercise 1: Implement RAGAS-Style Metrics from Scratch

Skills practiced:
  - Building faithfulness, answer relevancy, context precision, context recall
  - Decomposing answers into verifiable claims
  - Using LLM-as-judge to verify claims against context
  - Aggregating multi-dimensional scores into a coherent evaluation

Healthcare context:
  The RAGAS framework (Retrieval Augmented Generation Assessment) is widely used
  to evaluate RAG systems across four key dimensions:
    1. Faithfulness — Is the answer grounded in the retrieved context?
    2. Answer Relevancy — Does the answer address the question?
    3. Context Precision — Are the retrieved documents relevant?
    4. Context Recall — Does the context contain all the information needed?

  In healthcare, each dimension matters: unfaithful answers hallucinate treatments,
  irrelevant answers waste clinician time, imprecise retrieval introduces noise,
  and poor recall means critical information is missing.

  You will implement each metric from scratch and evaluate 5 medical Q&A pairs.
"""

import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


# ============================================================
# Medical Knowledge Base
# ============================================================

MEDICAL_DOCUMENTS = [
    {
        "id": "doc_001",
        "content": (
            "Acute myocardial infarction (AMI) management: Administer aspirin 325mg "
            "immediately. Obtain 12-lead ECG within 10 minutes. For STEMI, activate cath "
            "lab for primary PCI with door-to-balloon time <90 minutes. Administer heparin "
            "and a P2Y12 inhibitor (clopidogrel, ticagrelor, or prasugrel). Morphine for "
            "pain if needed, oxygen if SpO2 <90%, nitroglycerin for ongoing chest pain "
            "(avoid if systolic BP <90 or right ventricular infarction)."
        ),
    },
    {
        "id": "doc_002",
        "content": (
            "Type 2 diabetes management: Metformin is first-line therapy, starting at "
            "500mg daily and titrating to 2000mg. Second-line options include SGLT2 "
            "inhibitors (empagliflozin, dapagliflozin) which provide cardiovascular and "
            "renal benefits, GLP-1 receptor agonists (semaglutide, liraglutide) which "
            "promote weight loss, or DPP-4 inhibitors (sitagliptin). Target HbA1c is "
            "generally <7% but individualize based on patient factors. Monitor renal "
            "function, especially with metformin (hold if eGFR <30)."
        ),
    },
    {
        "id": "doc_003",
        "content": (
            "Pneumonia treatment in adults: For community-acquired pneumonia (CAP), "
            "outpatient treatment is amoxicillin 1g TID or doxycycline 100mg BID for "
            "healthy adults without comorbidities. For those with comorbidities, use "
            "amoxicillin-clavulanate plus a macrolide, or a respiratory fluoroquinolone. "
            "Inpatient non-ICU: beta-lactam plus macrolide or respiratory fluoroquinolone "
            "alone. ICU: beta-lactam plus macrolide or beta-lactam plus fluoroquinolone. "
            "Duration is typically 5-7 days. Use CURB-65 for severity assessment."
        ),
    },
    {
        "id": "doc_004",
        "content": (
            "Anticoagulation for atrial fibrillation: Use CHA2DS2-VASc score to assess "
            "stroke risk. Score ≥2 in men or ≥3 in women warrants oral anticoagulation. "
            "DOACs (apixaban, rivarelbane, edoxaban, dabigatran) are preferred over warfarin "
            "for non-valvular AF. Apixaban: 5mg BID (reduce to 2.5mg BID if ≥2 of: age ≥80, "
            "weight ≤60kg, creatinine ≥1.5). Assess bleeding risk with HAS-BLED score. "
            "Monitor renal function annually. Warfarin remains preferred for mechanical "
            "heart valves and moderate-to-severe mitral stenosis."
        ),
    },
    {
        "id": "doc_005",
        "content": (
            "Pediatric fever management: Fever is defined as temperature ≥38°C (100.4°F). "
            "For children ≥3 months: acetaminophen 10-15mg/kg every 4-6 hours or ibuprofen "
            "5-10mg/kg every 6-8 hours (ibuprofen only for age ≥6 months). Do not alternate "
            "antipyretics routinely. Encourage oral hydration. Red flags requiring urgent "
            "evaluation: age <3 months, toxic appearance, petechial rash, altered mental "
            "status, fever >40°C lasting >5 days. Febrile seizures are generally benign "
            "but evaluate for meningitis if clinical suspicion is high."
        ),
    },
    {
        "id": "doc_006",
        "content": (
            "Deep vein thrombosis (DVT) diagnosis and treatment: Suspect DVT with unilateral "
            "leg swelling, pain, warmth. Use Wells score to assess probability. D-dimer to "
            "rule out low-probability cases. Confirm with compression ultrasonography. "
            "Treatment: Anticoagulate with DOAC (rivaroxaban or apixaban as monotherapy, "
            "or dabigatran/edoxaban after 5 days heparin bridge). Duration: 3 months for "
            "provoked DVT, consider indefinite for unprovoked or recurrent. IVC filter only "
            "if anticoagulation contraindicated."
        ),
    },
]


# ============================================================
# Evaluation Q&A Pairs with Ground Truth
# ============================================================

EVAL_PAIRS = [
    {
        "question": "How should a STEMI be managed in the emergency department?",
        "ground_truth": (
            "Give aspirin 325mg immediately, obtain ECG within 10 minutes, activate cath lab "
            "for primary PCI with door-to-balloon time under 90 minutes, administer heparin "
            "and a P2Y12 inhibitor."
        ),
        "relevant_doc_ids": ["doc_001"],
    },
    {
        "question": "What is the first-line medication for type 2 diabetes and its dosing?",
        "ground_truth": (
            "Metformin is first-line, starting at 500mg daily and titrated to 2000mg. "
            "Monitor renal function and hold if eGFR <30."
        ),
        "relevant_doc_ids": ["doc_002"],
    },
    {
        "question": "What antibiotics are used for community-acquired pneumonia in outpatients?",
        "ground_truth": (
            "For healthy adults without comorbidities: amoxicillin 1g TID or doxycycline "
            "100mg BID. For those with comorbidities: amoxicillin-clavulanate plus macrolide "
            "or respiratory fluoroquinolone."
        ),
        "relevant_doc_ids": ["doc_003"],
    },
    {
        "question": "When should anticoagulation be started for atrial fibrillation?",
        "ground_truth": (
            "Use CHA2DS2-VASc score. Anticoagulation is indicated for score ≥2 in men or "
            "≥3 in women. DOACs are preferred over warfarin for non-valvular AF."
        ),
        "relevant_doc_ids": ["doc_004"],
    },
    {
        "question": "What are the red flags for pediatric fever requiring urgent evaluation?",
        "ground_truth": (
            "Red flags: age under 3 months, toxic appearance, petechial rash, altered "
            "mental status, fever above 40°C lasting more than 5 days."
        ),
        "relevant_doc_ids": ["doc_005"],
    },
]


# ============================================================
# Helper Functions
# ============================================================

def get_embedding(text: str) -> list:
    """Get embedding vector for a text string."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def get_embeddings_batch(texts: list) -> list:
    """Get embeddings for a batch of texts."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_top_k(query: str, doc_embeddings: list, documents: list, k: int = 3) -> list:
    """Retrieve top-k documents by cosine similarity."""
    query_emb = get_embedding(query)
    sims = [cosine_similarity(query_emb, de) for de in doc_embeddings]
    ranked = np.argsort(sims)[::-1][:k]
    return [{"document": documents[i], "score": sims[i]} for i in ranked]


def generate_answer(question: str, context_docs: list) -> str:
    """Generate an answer from retrieved context."""
    context = "\n\n".join(doc["content"] for doc in context_docs)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are a medical assistant. Answer using ONLY the provided context. "
                    "Do not add information not present in the context."
                )},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error: {e}]"


def call_llm_json(system_prompt: str, user_message: str) -> dict:
    """Call LLM and parse JSON response."""
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
        return {"error": str(e)}


# ============================================================
# RAGAS Metric 1: Faithfulness
# ============================================================

def compute_faithfulness(answer: str, context: str) -> float:
    """
    Faithfulness: What fraction of claims in the answer are supported by the context?

    Steps:
    1. Decompose the answer into individual claims
    2. For each claim, check if it is supported by the context
    3. Faithfulness = # supported claims / # total claims
    """
    # Step 1: Extract claims
    claims_result = call_llm_json(
        system_prompt=(
            "Extract all factual claims from the following answer. "
            "Return JSON: {\"claims\": [\"claim1\", \"claim2\", ...]}"
        ),
        user_message=f"Answer: {answer}",
    )
    claims = claims_result.get("claims", [])
    if not claims:
        return 0.0

    # Step 2: Verify each claim against context
    verification_result = call_llm_json(
        system_prompt=(
            "For each claim, determine if it is SUPPORTED or NOT_SUPPORTED by the context. "
            "A claim is SUPPORTED only if the context explicitly contains this information.\n\n"
            "Respond with JSON: {\"verdicts\": [{\"claim\": \"...\", \"verdict\": \"SUPPORTED\" "
            "or \"NOT_SUPPORTED\"}]}"
        ),
        user_message=f"Context:\n{context}\n\nClaims:\n" + "\n".join(
            f"- {c}" for c in claims
        ),
    )
    verdicts = verification_result.get("verdicts", [])

    # Step 3: Calculate score
    if not verdicts:
        return 0.0
    supported = sum(1 for v in verdicts if v.get("verdict") == "SUPPORTED")
    return supported / len(verdicts)


# ============================================================
# RAGAS Metric 2: Answer Relevancy
# ============================================================

def compute_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer Relevancy: How well does the answer address the question?

    Approach: Generate N hypothetical questions that the answer would address,
    then measure cosine similarity between original question and generated questions.
    High similarity = high relevancy.
    """
    # Generate 3 hypothetical questions the answer would address
    gen_result = call_llm_json(
        system_prompt=(
            "Given the following answer, generate exactly 3 questions that this answer "
            "would be a good response to. Return JSON: {\"questions\": [\"q1\", \"q2\", \"q3\"]}"
        ),
        user_message=f"Answer: {answer}",
    )
    generated_questions = gen_result.get("questions", [])
    if not generated_questions:
        return 0.0

    # Embed original question and generated questions
    all_texts = [question] + generated_questions
    embeddings = get_embeddings_batch(all_texts)

    original_emb = embeddings[0]
    gen_embs = embeddings[1:]

    # Average cosine similarity
    similarities = [cosine_similarity(original_emb, ge) for ge in gen_embs]
    return float(np.mean(similarities))


# ============================================================
# RAGAS Metric 3: Context Precision
# ============================================================

def compute_context_precision(question: str, retrieved_docs: list, relevant_doc_ids: list) -> float:
    """
    Context Precision: Are the retrieved documents actually relevant to the question?

    Uses the LLM to judge whether each retrieved document is relevant.
    Precision = # relevant retrieved / # total retrieved
    Weighted by rank (higher-ranked documents count more).
    """
    if not retrieved_docs:
        return 0.0

    doc_texts = "\n\n".join(
        f"[Document {i+1}]: {doc['content'][:300]}"
        for i, doc in enumerate(retrieved_docs)
    )

    result = call_llm_json(
        system_prompt=(
            "Evaluate whether each retrieved document is relevant to answering the question. "
            "A document is relevant if it contains information needed to answer the question.\n\n"
            "Respond with JSON: {\"relevance\": [true or false for each document in order]}"
        ),
        user_message=f"Question: {question}\n\nDocuments:\n{doc_texts}",
    )
    relevance = result.get("relevance", [])

    if not relevance:
        return 0.0

    # Weighted precision: higher ranks matter more
    weighted_sum = 0.0
    relevant_so_far = 0
    for i, is_relevant in enumerate(relevance):
        if is_relevant:
            relevant_so_far += 1
            precision_at_i = relevant_so_far / (i + 1)
            weighted_sum += precision_at_i

    total_relevant = sum(1 for r in relevance if r)
    return weighted_sum / total_relevant if total_relevant > 0 else 0.0


# ============================================================
# RAGAS Metric 4: Context Recall
# ============================================================

def compute_context_recall(ground_truth: str, context: str) -> float:
    """
    Context Recall: Does the retrieved context contain all information needed
    to construct the ground truth answer?

    Steps:
    1. Decompose ground truth into individual statements
    2. Check if each statement can be attributed to the context
    3. Recall = # attributable statements / # total statements
    """
    result = call_llm_json(
        system_prompt=(
            "Decompose the ground truth answer into individual factual statements. "
            "For each statement, determine if it can be attributed to the provided context.\n\n"
            "Respond with JSON: {\"statements\": [{\"statement\": \"...\", "
            "\"attributable\": true or false}]}"
        ),
        user_message=f"Context:\n{context}\n\nGround Truth Answer:\n{ground_truth}",
    )
    statements = result.get("statements", [])

    if not statements:
        return 0.0
    attributable = sum(1 for s in statements if s.get("attributable", False))
    return attributable / len(statements)


# ============================================================
# Run Full Evaluation
# ============================================================

def run_ragas_evaluation():
    """
    Run all four RAGAS metrics on the 5 medical Q&A pairs.
    """
    print("=" * 70)
    print("  RAGAS-Style Evaluation of Medical Q&A")
    print("=" * 70)

    # Embed all documents
    print("\nEmbedding medical knowledge base...")
    doc_texts = [doc["content"] for doc in MEDICAL_DOCUMENTS]
    doc_embeddings = get_embeddings_batch(doc_texts)
    print(f"  Embedded {len(doc_embeddings)} documents.\n")

    results = []

    for i, pair in enumerate(EVAL_PAIRS, 1):
        question = pair["question"]
        ground_truth = pair["ground_truth"]
        relevant_ids = pair["relevant_doc_ids"]

        print(f"{'─' * 70}")
        print(f"Q{i}: {question}")
        print(f"{'─' * 70}")

        # Retrieve top-3 docs
        retrieved = retrieve_top_k(question, doc_embeddings, MEDICAL_DOCUMENTS, k=3)
        context_docs = [r["document"] for r in retrieved]
        context_text = "\n\n".join(doc["content"] for doc in context_docs)
        retrieved_ids = [r["document"]["id"] for r in retrieved]

        print(f"  Retrieved: {retrieved_ids}")

        # Generate answer
        answer = generate_answer(question, context_docs)
        print(f"  Answer: {answer[:120]}...")

        # Compute all 4 metrics
        print("  Computing metrics...")

        faithfulness = compute_faithfulness(answer, context_text)
        print(f"    Faithfulness:      {faithfulness:.3f}")

        answer_relevancy = compute_answer_relevancy(question, answer)
        print(f"    Answer Relevancy:  {answer_relevancy:.3f}")

        context_precision = compute_context_precision(question, context_docs, relevant_ids)
        print(f"    Context Precision: {context_precision:.3f}")

        context_recall = compute_context_recall(ground_truth, context_text)
        print(f"    Context Recall:    {context_recall:.3f}")

        row = {
            "question": question,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        results.append(row)
        print()

    # Summary Report
    print("\n" + "=" * 78)
    print("RAGAS EVALUATION SUMMARY")
    print("=" * 78)
    print(f"\n{'#':<3} {'Question':<45} {'Faith':>6} {'Rel':>6} {'CPrec':>6} {'CRec':>6}")
    print("-" * 78)

    for i, row in enumerate(results, 1):
        q_short = row["question"][:43]
        print(f"{i:<3} {q_short:<45} {row['faithfulness']:>6.3f} "
              f"{row['answer_relevancy']:>6.3f} {row['context_precision']:>6.3f} "
              f"{row['context_recall']:>6.3f}")

    print("-" * 78)

    # Averages
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    avgs = {m: np.mean([r[m] for r in results]) for m in metrics}
    print(f"{'':>3} {'AVERAGE':<45} {avgs['faithfulness']:>6.3f} "
          f"{avgs['answer_relevancy']:>6.3f} {avgs['context_precision']:>6.3f} "
          f"{avgs['context_recall']:>6.3f}")
    print("=" * 78)

    # Interpretation
    print("\nInterpretation:")
    for metric, avg in avgs.items():
        label = metric.replace("_", " ").title()
        if avg >= 0.85:
            grade = "🟢 Excellent"
        elif avg >= 0.7:
            grade = "🟡 Good"
        elif avg >= 0.5:
            grade = "🟠 Fair"
        else:
            grade = "🔴 Poor"
        print(f"  {label:<25} {avg:.3f}  {grade}")

    # Identify weakest area
    weakest = min(avgs, key=avgs.get)
    print(f"\nWeakest dimension: {weakest.replace('_', ' ').title()}")
    if weakest == "faithfulness":
        print("  → Recommendation: Strengthen system prompt to stay grounded in context.")
    elif weakest == "answer_relevancy":
        print("  → Recommendation: Improve prompt to directly address the question asked.")
    elif weakest == "context_precision":
        print("  → Recommendation: Improve embeddings or add re-ranking to reduce noise.")
    elif weakest == "context_recall":
        print("  → Recommendation: Improve chunking strategy or increase retrieval k.")


# ============================================================
# Main
# ============================================================

def main():
    print("🏥 Exercise 1: RAGAS-Style Metrics for Medical RAG\n")
    print("This exercise implements the four core RAGAS metrics from scratch")
    print("and evaluates them on 5 medical Q&A pairs.\n")
    run_ragas_evaluation()


if __name__ == "__main__":
    main()
