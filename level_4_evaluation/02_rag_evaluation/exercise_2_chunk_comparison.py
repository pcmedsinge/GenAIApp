"""
Exercise 2: Compare Chunk Sizes Using Evaluation Metrics

Skills practiced:
  - Implementing different chunking strategies (128, 256, 512, 1024 tokens)
  - Running the same queries across different chunk configurations
  - Measuring retrieval and generation quality for each chunk size
  - Statistical comparison and visualization of results

Healthcare context:
  Chunk size profoundly affects RAG quality in medical systems. Too small and
  you lose clinical context (e.g., a drug dosage separated from its indication).
  Too large and you retrieve irrelevant noise alongside the answer. This exercise
  helps you find the optimal chunk size for a medical knowledge base by measuring
  precision, recall, faithfulness, and answer relevancy across four chunk sizes.
"""

import os
import json
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


# ============================================================
# Extended Medical Text (longer document to chunk differently)
# ============================================================

MEDICAL_TEXT_CORPUS = [
    {
        "source": "Hypertension Guidelines",
        "text": (
            "Hypertension is defined as systolic blood pressure ≥130 mmHg or diastolic "
            "blood pressure ≥80 mmHg. Elevated blood pressure (120-129/<80) should be "
            "managed with lifestyle modifications alone. Stage 1 hypertension (130-139/80-89) "
            "warrants pharmacotherapy if 10-year ASCVD risk ≥10% or if comorbidities are "
            "present. Stage 2 hypertension (≥140/≥90) requires pharmacotherapy in all cases. "
            "First-line antihypertensives include ACE inhibitors such as lisinopril (starting "
            "dose 10mg daily, target 40mg daily), ARBs such as losartan (starting 50mg daily, "
            "target 100mg daily), calcium channel blockers such as amlodipine (starting 5mg "
            "daily, target 10mg daily), and thiazide diuretics such as chlorthalidone (starting "
            "12.5mg daily, target 25mg daily). For patients with diabetes or CKD, ACE "
            "inhibitors or ARBs are preferred due to renal protective effects. "
            "Combination therapy with two first-line agents from different classes is "
            "recommended for stage 2 hypertension or when blood pressure is >20/10 mmHg "
            "above target. Avoid combining ACE inhibitors with ARBs due to increased risk "
            "of hyperkalemia and renal dysfunction. Monitor potassium and creatinine 2-4 "
            "weeks after initiating or titrating RAAS inhibitors. Resistant hypertension "
            "(uncontrolled on 3 agents including a diuretic) should prompt evaluation for "
            "secondary causes: renal artery stenosis, primary aldosteronism, pheochromocytoma, "
            "Cushing syndrome, obstructive sleep apnea, and thyroid disorders. Spironolactone "
            "25-50mg daily is effective add-on therapy for resistant hypertension. "
            "Hypertensive emergencies (BP >180/120 with end-organ damage) require IV "
            "antihypertensives with the goal of reducing MAP by no more than 25% in the "
            "first hour. Preferred IV agents include nicardipine, labetalol, and nitroprusside."
        ),
    },
    {
        "source": "Diabetes Management",
        "text": (
            "Type 2 diabetes mellitus should be managed with a patient-centered approach. "
            "Initial therapy includes lifestyle modification (150 minutes/week moderate "
            "exercise, medical nutrition therapy) plus metformin. Start metformin 500mg "
            "once daily with meals, titrate to 1000mg BID over 1-2 months. If HbA1c is "
            "≥1.5% above target at diagnosis, consider dual therapy from the start. "
            "Second-line agents should be chosen based on comorbidities. For patients with "
            "established atherosclerotic cardiovascular disease (ASCVD): add GLP-1 RA with "
            "proven CV benefit (liraglutide 1.8mg daily or semaglutide 1mg weekly) or SGLT2 "
            "inhibitor (empagliflozin 10-25mg daily or dapagliflozin 10mg daily). For "
            "patients with heart failure: add SGLT2 inhibitor regardless of HbA1c. For "
            "patients with CKD: add SGLT2 inhibitor if eGFR ≥20, or finerenone for "
            "additional renal protection. If cost is a major concern, consider sulfonylureas "
            "(glipizide 5-20mg daily) or thiazolidinediones (pioglitazone 15-45mg daily). "
            "Insulin therapy: basal insulin (glargine, detemir, degludec) should be added "
            "when oral agents are insufficient. Start 10 units or 0.1-0.2 units/kg at "
            "bedtime, titrate by 2 units every 3 days to fasting glucose 80-130. "
            "Monitoring: HbA1c every 3 months until at target, then every 6 months. "
            "Annual screening: dilated eye exam, urine albumin-to-creatinine ratio, "
            "lipid panel, comprehensive foot exam. Self-monitoring of blood glucose for "
            "patients on insulin or sulfonylureas. Hypoglycemia management: rule of 15 "
            "(15g fast-acting glucose, recheck in 15 minutes). Glucagon kit for severe "
            "hypoglycemia."
        ),
    },
    {
        "source": "Anticoagulation Therapy",
        "text": (
            "Anticoagulation therapy is used for venous thromboembolism (VTE), atrial "
            "fibrillation, and mechanical heart valves. For VTE treatment: DOACs are "
            "preferred over warfarin. Rivaroxaban 15mg BID for 21 days then 20mg daily, "
            "or apixaban 10mg BID for 7 days then 5mg BID. For dabigatran or edoxaban, "
            "start with 5 days of parenteral anticoagulation (enoxaparin 1mg/kg BID). "
            "Duration: 3 months for provoked VTE (surgery, immobilization), extended "
            "therapy for unprovoked VTE or cancer-associated VTE (reassess annually). "
            "For atrial fibrillation: calculate CHA2DS2-VASc score. Score 0 in men or "
            "1 in women: no anticoagulation. Score 1 in men or 2 in women: consider "
            "anticoagulation. Score ≥2 in men or ≥3 in women: anticoagulate. DOACs "
            "preferred over warfarin for non-valvular AF. Warfarin (target INR 2-3) "
            "remains indicated for mechanical heart valves and moderate-severe mitral "
            "stenosis. Drug interactions with warfarin: amiodarone, fluconazole, and "
            "metronidazole increase INR; rifampin and carbamazepine decrease INR. "
            "Dietary interactions: consistent intake of vitamin K-rich foods (spinach, "
            "kale, broccoli, Brussels sprouts). Periprocedural management: hold DOACs "
            "24-48 hours before procedures depending on renal function and bleeding risk. "
            "Warfarin bridging with heparin only for high-risk indications (mechanical "
            "valve, recent VTE <3 months). Reversal agents: idarucizumab for dabigatran, "
            "andexanet alfa for apixaban/rivaroxaban, vitamin K plus 4-factor PCC for "
            "warfarin."
        ),
    },
]


# ============================================================
# Evaluation Questions with Ground Truth
# ============================================================

EVAL_QUESTIONS = [
    {
        "question": "What is the starting dose of lisinopril for hypertension?",
        "ground_truth": "Lisinopril starting dose is 10mg daily with a target of 40mg daily.",
        "source": "Hypertension Guidelines",
    },
    {
        "question": "When should combination antihypertensive therapy be used?",
        "ground_truth": (
            "Combination therapy is recommended for stage 2 hypertension or when blood "
            "pressure is more than 20/10 mmHg above target."
        ),
        "source": "Hypertension Guidelines",
    },
    {
        "question": "What second-line diabetes agents are recommended for patients with heart failure?",
        "ground_truth": "SGLT2 inhibitor should be added regardless of HbA1c for patients with heart failure.",
        "source": "Diabetes Management",
    },
    {
        "question": "How should basal insulin be initiated in type 2 diabetes?",
        "ground_truth": (
            "Start 10 units or 0.1-0.2 units/kg at bedtime, titrate by 2 units every "
            "3 days to fasting glucose 80-130."
        ),
        "source": "Diabetes Management",
    },
    {
        "question": "What is the dosing regimen for rivaroxaban in VTE treatment?",
        "ground_truth": "Rivaroxaban 15mg BID for 21 days then 20mg daily.",
        "source": "Anticoagulation Therapy",
    },
    {
        "question": "What are the reversal agents for DOACs?",
        "ground_truth": (
            "Idarucizumab for dabigatran, andexanet alfa for apixaban and rivaroxaban."
        ),
        "source": "Anticoagulation Therapy",
    },
]


# ============================================================
# Chunking Strategies
# ============================================================

def chunk_by_words(text: str, chunk_size_words: int, overlap_words: int = 20) -> list:
    """
    Split text into chunks by approximate word count with overlap.
    (Approximating tokens ~ words * 1.3 for simplicity.)
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap_words
        if start < 0:
            start = end
    return chunks


def build_chunked_index(corpus: list, chunk_size_words: int) -> tuple:
    """
    Chunk all documents in the corpus and embed them.

    Returns:
        (chunks_list, embeddings_list) where each chunk is a dict with
        'text', 'source', and 'chunk_idx'.
    """
    all_chunks = []
    for doc in corpus:
        chunks = chunk_by_words(doc["text"], chunk_size_words)
        for idx, chunk_text in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text,
                "source": doc["source"],
                "chunk_idx": idx,
            })

    # Embed all chunks
    chunk_texts = [c["text"] for c in all_chunks]
    embeddings = get_embeddings_batch(chunk_texts)

    return all_chunks, embeddings


def get_embeddings_batch(texts: list) -> list:
    """Get embeddings for a batch of texts, handling batch size limits."""
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def get_embedding(text: str) -> list:
    """Get a single embedding."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_chunks(query: str, chunks: list, embeddings: list, k: int = 3) -> list:
    """Retrieve top-k chunks for a query."""
    query_emb = get_embedding(query)
    sims = [cosine_similarity(query_emb, e) for e in embeddings]
    ranked = np.argsort(sims)[::-1][:k]
    return [{"chunk": chunks[i], "score": sims[i]} for i in ranked]


def generate_answer(question: str, context_chunks: list) -> str:
    """Generate an answer from retrieved chunks."""
    context = "\n\n".join(c["text"] for c in context_chunks)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are a medical assistant. Answer using ONLY the provided context. "
                    "Be specific about dosages, medications, and clinical details."
                )},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error: {e}]"


# ============================================================
# Evaluation Metrics
# ============================================================

def score_faithfulness(answer: str, context: str) -> float:
    """Score whether answer is faithful to context (0-1)."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": (
                    "Evaluate if the answer is faithful to the context (uses only information "
                    "from the context, no hallucination). Respond with JSON: "
                    "{\"faithfulness\": <float 0.0-1.0>, \"explanation\": \"...\"}"
                )},
                {"role": "user", "content": (
                    f"Context:\n{context}\n\nAnswer:\n{answer}"
                )},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        result = json.loads(raw)
        return float(result.get("faithfulness", 0.0))
    except Exception:
        return 0.5


def score_relevancy(question: str, answer: str) -> float:
    """Score whether answer is relevant to the question (0-1)."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": (
                    "Evaluate if the answer is relevant to the question (directly addresses "
                    "what was asked). Respond with JSON: "
                    "{\"relevancy\": <float 0.0-1.0>, \"explanation\": \"...\"}"
                )},
                {"role": "user", "content": (
                    f"Question: {question}\n\nAnswer:\n{answer}"
                )},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        result = json.loads(raw)
        return float(result.get("relevancy", 0.0))
    except Exception:
        return 0.5


def score_answer_correctness(answer: str, ground_truth: str) -> float:
    """Score answer against ground truth for correctness (0-1)."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": (
                    "Compare the answer to the ground truth. Score how correct the answer "
                    "is on a 0.0-1.0 scale. A score of 1.0 means all key facts from ground "
                    "truth appear correctly in the answer. Respond with JSON: "
                    "{\"correctness\": <float 0.0-1.0>, \"explanation\": \"...\"}"
                )},
                {"role": "user", "content": (
                    f"Ground Truth: {ground_truth}\n\nAnswer: {answer}"
                )},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        result = json.loads(raw)
        return float(result.get("correctness", 0.0))
    except Exception:
        return 0.5


def source_match(retrieved_chunks: list, expected_source: str) -> float:
    """Check if the top-k retrieved chunks match the expected source document."""
    if not retrieved_chunks:
        return 0.0
    matches = sum(1 for c in retrieved_chunks if c["source"] == expected_source)
    return matches / len(retrieved_chunks)


# ============================================================
# Main Comparison
# ============================================================

def run_chunk_comparison():
    """
    Compare chunk sizes across all evaluation questions.
    """
    print("=" * 75)
    print("  Chunk Size Comparison for Medical RAG")
    print("=" * 75)

    # Approximate token-to-word conversion: 1 token ≈ 0.75 words
    # So chunk sizes in words: 96, 192, 384, 768
    chunk_configs = [
        {"label": "128 tokens (~96 words)", "words": 96},
        {"label": "256 tokens (~192 words)", "words": 192},
        {"label": "512 tokens (~384 words)", "words": 384},
        {"label": "1024 tokens (~768 words)", "words": 768},
    ]

    all_results = {}

    for config in chunk_configs:
        label = config["label"]
        words = config["words"]
        print(f"\n{'─' * 75}")
        print(f"Chunk Size: {label}")
        print(f"{'─' * 75}")

        # Build index
        print("  Building index...")
        chunks, embeddings = build_chunked_index(MEDICAL_TEXT_CORPUS, words)
        print(f"  Total chunks: {len(chunks)}")

        metrics = {
            "faithfulness": [],
            "relevancy": [],
            "correctness": [],
            "source_match": [],
        }

        for q_data in EVAL_QUESTIONS:
            question = q_data["question"]
            ground_truth = q_data["ground_truth"]
            expected_source = q_data["source"]

            # Retrieve top-3 chunks
            results = retrieve_chunks(question, chunks, embeddings, k=3)
            context_chunks = [r["chunk"] for r in results]
            context_text = "\n".join(c["text"] for c in context_chunks)

            # Generate answer
            answer = generate_answer(question, context_chunks)

            # Compute metrics
            faith = score_faithfulness(answer, context_text)
            rel = score_relevancy(question, answer)
            corr = score_answer_correctness(answer, ground_truth)
            src = source_match(context_chunks, expected_source)

            metrics["faithfulness"].append(faith)
            metrics["relevancy"].append(rel)
            metrics["correctness"].append(corr)
            metrics["source_match"].append(src)

            print(f"  Q: {question[:55]}…  F={faith:.2f} R={rel:.2f} C={corr:.2f} S={src:.2f}")

        # Store averages
        avgs = {m: float(np.mean(v)) for m, v in metrics.items()}
        avgs["num_chunks"] = len(chunks)
        all_results[label] = avgs
        print(f"  Averages — Faith: {avgs['faithfulness']:.3f}  "
              f"Rel: {avgs['relevancy']:.3f}  Corr: {avgs['correctness']:.3f}  "
              f"Src: {avgs['source_match']:.3f}")

    # ============================================================
    # Comparison Report
    # ============================================================
    print("\n\n" + "=" * 80)
    print("CHUNK SIZE COMPARISON REPORT")
    print("=" * 80)

    header = f"{'Chunk Size':<30} {'#Chunks':>7} {'Faith':>7} {'Rel':>7} {'Corr':>7} {'SrcM':>7} {'AVG':>7}"
    print(f"\n{header}")
    print("-" * 80)

    best_label = None
    best_avg = -1.0

    for label, avgs in all_results.items():
        overall = np.mean([avgs["faithfulness"], avgs["relevancy"],
                           avgs["correctness"], avgs["source_match"]])
        if overall > best_avg:
            best_avg = overall
            best_label = label

        print(f"{label:<30} {avgs['num_chunks']:>7} {avgs['faithfulness']:>7.3f} "
              f"{avgs['relevancy']:>7.3f} {avgs['correctness']:>7.3f} "
              f"{avgs['source_match']:>7.3f} {overall:>7.3f}")

    print("-" * 80)
    print(f"\n🏆 Best chunk size: {best_label} (overall avg: {best_avg:.3f})")

    # Visual comparison (text bar chart)
    print("\nVisual Comparison (Overall Score):")
    for label, avgs in all_results.items():
        overall = np.mean([avgs["faithfulness"], avgs["relevancy"],
                           avgs["correctness"], avgs["source_match"]])
        bar_len = int(overall * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        marker = " ← BEST" if label == best_label else ""
        print(f"  {label:<30} {bar} {overall:.3f}{marker}")

    # Insights
    print("\nInsights:")
    labels = list(all_results.keys())
    if len(labels) >= 2:
        smallest = labels[0]
        largest = labels[-1]
        s_faith = all_results[smallest]["faithfulness"]
        l_faith = all_results[largest]["faithfulness"]
        if s_faith > l_faith:
            print("  - Smaller chunks tend to produce more faithful answers (less noise).")
        else:
            print("  - Larger chunks maintain better faithfulness (more complete context).")

        s_corr = all_results[smallest]["correctness"]
        l_corr = all_results[largest]["correctness"]
        if l_corr > s_corr:
            print("  - Larger chunks improve answer correctness (more context available).")
        else:
            print("  - Smaller chunks are sufficient for answer correctness.")


# ============================================================
# Main
# ============================================================

def main():
    print("🏥 Exercise 2: Chunk Size Comparison for Medical RAG\n")
    print("Compares 128, 256, 512, and 1024 token chunk sizes")
    print("using retrieval and generation quality metrics.\n")
    run_chunk_comparison()


if __name__ == "__main__":
    main()
