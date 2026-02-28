"""
Exercise 2 — Local Medical Q&A System
======================================

Skills practiced
----------------
* Building an interactive medical Q&A loop with a local LLM
* Crafting effective healthcare system prompts
* Comparing local model answers with GPT-4o (cloud) answers
* Measuring answer overlap and quality gaps

Healthcare context
------------------
A hospital's IT team is evaluating whether a locally-hosted LLM can
serve as the backbone for a clinical decision-support chatbot.  The key
question: "How close can a local 7–8 B parameter model get to GPT-4o
quality on realistic clinical queries?"  This exercise provides a
side-by-side comparison framework.

Usage
-----
    python exercise_2_local_medical_qa.py

Prerequisites
-------------
    ollama pull llama3
    pip install openai
    (Optional) set OPENAI_API_KEY for GPT-4o comparison
"""

import os
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"
LOCAL_MODEL = "llama3"

CLOUD_MODEL = "gpt-4o"  # used only when OPENAI_API_KEY is set

SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant deployed inside a "
    "hospital network.  Provide accurate, evidence-based answers to "
    "medical questions.  When uncertain, state your level of confidence "
    "and recommend consulting a specialist.  Never fabricate references."
)

SAMPLE_QUESTIONS = [
    "What is the recommended initial antibiotic regimen for community-acquired pneumonia in an immunocompetent adult?",
    "Describe the Wells score for pulmonary embolism and its clinical use.",
    "A patient on warfarin has an INR of 8.5 with minor gum bleeding. What is the management?",
    "What are the contraindications for thrombolytic therapy in acute ischemic stroke?",
    "Explain the difference between nephrotic and nephritic syndrome, including key lab findings.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_ollama_client():
    from openai import OpenAI
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    try:
        client.models.list()
    except Exception as exc:
        print(f"⚠  Cannot reach Ollama: {exc}")
        print("   Start it:  ollama serve")
        sys.exit(1)
    return client


def get_cloud_client() -> Optional[object]:
    """Return an OpenAI client for GPT-4o, or None if no key is set."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI
    return OpenAI()  # uses default base_url + env key


def ask_model(client, model: str, question: str) -> dict:
    """Send a question and return {answer, time_s, tokens}."""
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        elapsed = time.time() - start
        answer = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        return {"answer": answer, "time_s": round(elapsed, 2), "tokens": tokens, "error": ""}
    except Exception as exc:
        return {"answer": "", "time_s": round(time.time() - start, 2), "tokens": 0, "error": str(exc)}


def overlap_score(text_a: str, text_b: str) -> float:
    """Rough word-overlap (Jaccard) between two answers."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------
def run_comparison(ollama_client, cloud_client, questions: list[str]):
    """Run each question through local + cloud and print comparison."""
    print(f"\n{'='*60}")
    print("Side-by-Side Comparison: Local ({}) vs Cloud ({})".format(
        LOCAL_MODEL, CLOUD_MODEL if cloud_client else "N/A"
    ))
    print(f"{'='*60}")

    results = []

    for i, question in enumerate(questions, 1):
        print(f"\n--- Q{i}: {question[:70]}... ---\n")

        local = ask_model(ollama_client, LOCAL_MODEL, question)
        print(f"  [{LOCAL_MODEL}] ({local['time_s']}s, {local['tokens']} tok)")
        print(f"  {local['answer'][:300]}{'...' if len(local['answer'])>300 else ''}\n")

        cloud = {"answer": "", "time_s": 0, "tokens": 0, "error": "skipped"}
        if cloud_client:
            cloud = ask_model(cloud_client, CLOUD_MODEL, question)
            print(f"  [{CLOUD_MODEL}] ({cloud['time_s']}s, {cloud['tokens']} tok)")
            print(f"  {cloud['answer'][:300]}{'...' if len(cloud['answer'])>300 else ''}\n")

        if cloud_client and cloud["answer"]:
            sim = overlap_score(local["answer"], cloud["answer"])
            print(f"  Word-overlap (Jaccard): {sim:.2%}")
        else:
            sim = None

        results.append({
            "question": question,
            "local_time": local["time_s"],
            "cloud_time": cloud["time_s"],
            "overlap": sim,
        })

    # --- Summary ---
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Q#':<4} {'Local (s)':<10} {'Cloud (s)':<10} {'Overlap':<10}")
    print("-" * 34)
    for i, r in enumerate(results, 1):
        overlap_str = f"{r['overlap']:.0%}" if r["overlap"] is not None else "N/A"
        print(f"Q{i:<3} {r['local_time']:<10} {r['cloud_time']:<10} {overlap_str:<10}")

    if any(r["overlap"] is not None for r in results):
        avg = sum(r["overlap"] for r in results if r["overlap"] is not None) / sum(
            1 for r in results if r["overlap"] is not None
        )
        print(f"\nAverage word-overlap: {avg:.1%}")


# ---------------------------------------------------------------------------
# Interactive Q&A
# ---------------------------------------------------------------------------
def interactive_qa(ollama_client):
    """Free-form local Q&A loop."""
    print(f"\n{'='*60}")
    print("Interactive Local Medical Q&A")
    print(f"{'='*60}")
    print("Type your medical questions.  Type 'quit' to exit.\n")

    history = []
    while True:
        question = input("You: ").strip()
        if not question or question.lower() in ("quit", "exit", "q"):
            break

        history.append({"role": "user", "content": question})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-10:]

        start = time.time()
        try:
            resp = ollama_client.chat.completions.create(
                model=LOCAL_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=600,
            )
            elapsed = time.time() - start
            answer = resp.choices[0].message.content or ""
            history.append({"role": "assistant", "content": answer})
            print(f"\nAssistant ({elapsed:.1f}s):\n{answer}\n")
        except Exception as exc:
            print(f"\nError: {exc}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Exercise 2: Local Medical Q&A System")
    print("=" * 60)

    try:
        from openai import OpenAI  # noqa: F401
    except ImportError:
        print("ERROR: pip install openai")
        sys.exit(1)

    ollama_client = get_ollama_client()
    cloud_client = get_cloud_client()

    if cloud_client:
        print(f"✓ Cloud client available ({CLOUD_MODEL}) — will run comparison.")
    else:
        print("ℹ  No OPENAI_API_KEY set — skipping cloud comparison (local-only mode).")

    print("\nChoose a mode:")
    print("  1. Run benchmark comparison (5 sample questions)")
    print("  2. Interactive Q&A (free-form)")
    print("  3. Both")

    choice = input("\nChoice (1/2/3): ").strip()

    if choice in ("1", "3"):
        run_comparison(ollama_client, cloud_client, SAMPLE_QUESTIONS)
    if choice in ("2", "3"):
        interactive_qa(ollama_client)
    if choice not in ("1", "2", "3"):
        print("Running benchmark comparison by default...")
        run_comparison(ollama_client, cloud_client, SAMPLE_QUESTIONS)

    print("\nDone.")


if __name__ == "__main__":
    main()
