"""
Exercise 1 — Model Benchmarks for Local LLMs
=============================================

Skills practiced
----------------
* Benchmarking inference speed (tokens/second) across Ollama models
* Measuring response quality with a simple rubric
* Collecting and reporting structured performance metrics
* Understanding trade-offs between model size and quality

Healthcare context
------------------
Hospitals evaluating on-premise LLM deployments need hard numbers:
latency targets for clinical decision-support, throughput for batch
processing of discharge summaries, and quality floors for patient-safety.
This exercise gives you a repeatable benchmarking harness.

Usage
-----
    python exercise_1_model_benchmarks.py

Prerequisites
-------------
    ollama pull llama3 && ollama pull mistral && ollama pull phi3
    pip install openai
"""

import json
import sys
import time
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"

CANDIDATE_MODELS = ["llama3", "mistral", "phi3"]

MEDICAL_QUESTIONS = [
    {
        "id": "q1",
        "question": "What are the ACC/AHA guideline-recommended first-line antihypertensives?",
        "keywords": ["ACE inhibitor", "ARB", "calcium channel blocker", "thiazide"],
    },
    {
        "id": "q2",
        "question": "Describe the initial management of diabetic ketoacidosis in an adult.",
        "keywords": ["insulin", "fluids", "potassium", "bicarbonate", "monitoring"],
    },
    {
        "id": "q3",
        "question": "List the Duke criteria for infective endocarditis.",
        "keywords": ["blood culture", "echocardiogram", "vegetation", "fever", "vascular"],
    },
    {
        "id": "q4",
        "question": "What is the CURB-65 score and how does it guide pneumonia management?",
        "keywords": ["confusion", "urea", "respiratory rate", "blood pressure", "age"],
    },
    {
        "id": "q5",
        "question": "Explain the pathophysiology of heart failure with preserved ejection fraction (HFpEF).",
        "keywords": ["diastolic", "stiffness", "filling pressure", "EF", "concentric"],
    },
]

SYSTEM_PROMPT = (
    "You are a board-certified internal medicine physician. "
    "Answer medical questions accurately and concisely."
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class QuestionResult:
    question_id: str
    model: str
    response_time_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    keyword_hits: int = 0
    keyword_total: int = 0
    quality_score: float = 0.0  # keyword_hits / keyword_total
    response_text: str = ""
    error: str = ""


@dataclass
class ModelSummary:
    model: str
    avg_time_s: float = 0.0
    avg_tokens: float = 0.0
    avg_quality: float = 0.0
    questions_answered: int = 0
    questions_failed: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_client():
    """Create an OpenAI client pointed at Ollama."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai")
        sys.exit(1)
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    try:
        client.models.list()
    except Exception as exc:
        print(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}: {exc}")
        print("Start Ollama with:  ollama serve")
        sys.exit(1)
    return client


def score_keywords(text: str, keywords: list[str]) -> tuple[int, int]:
    """Return (hits, total) for case-insensitive keyword matching."""
    lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return hits, len(keywords)


def run_benchmark(client, model: str, question: dict) -> QuestionResult:
    """Send one question to a model and measure the result."""
    result = QuestionResult(question_id=question["id"], model=model)
    result.keyword_total = len(question["keywords"])

    try:
        start = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question["question"]},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        result.response_time_s = round(time.time() - start, 3)
        result.response_text = resp.choices[0].message.content or ""

        if resp.usage:
            result.prompt_tokens = resp.usage.prompt_tokens
            result.completion_tokens = resp.usage.completion_tokens
            result.total_tokens = resp.usage.total_tokens

        hits, total = score_keywords(result.response_text, question["keywords"])
        result.keyword_hits = hits
        result.quality_score = round(hits / total, 2) if total else 0.0

    except Exception as exc:
        result.error = str(exc)

    return result


def summarise(results: list[QuestionResult], model: str) -> ModelSummary:
    """Aggregate per-model results into a summary."""
    ok = [r for r in results if not r.error]
    fail = [r for r in results if r.error]
    summary = ModelSummary(model=model, questions_answered=len(ok), questions_failed=len(fail))
    if ok:
        summary.avg_time_s = round(sum(r.response_time_s for r in ok) / len(ok), 3)
        summary.avg_tokens = round(sum(r.total_tokens for r in ok) / len(ok), 1)
        summary.avg_quality = round(sum(r.quality_score for r in ok) / len(ok), 2)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Exercise 1: Model Benchmarks for Local LLMs")
    print("=" * 60)

    client = get_client()

    # Detect which candidate models are actually available
    available = {m.id for m in client.models.list().data}
    models = [m for m in CANDIDATE_MODELS if m in available]
    if not models:
        print(f"\n⚠  No candidate models found locally. Pull one:")
        for m in CANDIDATE_MODELS:
            print(f"   ollama pull {m}")
        return

    print(f"\nModels to benchmark: {models}")
    print(f"Questions: {len(MEDICAL_QUESTIONS)}\n")

    all_results: dict[str, list[QuestionResult]] = {m: [] for m in models}

    for q in MEDICAL_QUESTIONS:
        print(f"\n--- {q['id']}: {q['question'][:60]}... ---")
        for model in models:
            result = run_benchmark(client, model, q)
            all_results[model].append(result)

            status = "OK" if not result.error else f"ERR: {result.error}"
            print(
                f"  {model:<10}  {result.response_time_s:>6.2f}s  "
                f"quality={result.quality_score:.0%}  tokens={result.total_tokens}  {status}"
            )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Model':<12} {'Avg Time':<10} {'Avg Tokens':<12} {'Avg Quality':<12} {'OK/Fail'}")
    print("-" * 58)
    for model in models:
        s = summarise(all_results[model], model)
        print(
            f"{s.model:<12} {s.avg_time_s:<10.2f} {s.avg_tokens:<12.1f} "
            f"{s.avg_quality:<12.0%} {s.questions_answered}/{s.questions_failed}"
        )

    # --- Detailed JSON dump ---
    print("\n--- Detailed results (JSON) ---")
    flat = []
    for model_results in all_results.values():
        for r in model_results:
            d = asdict(r)
            d.pop("response_text")  # keep output short
            flat.append(d)
    print(json.dumps(flat, indent=2))


if __name__ == "__main__":
    main()
