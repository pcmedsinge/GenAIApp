"""
Exercise 2: Model Comparison Study
====================================
Compare approaches: prompted GPT-4o vs fine-tuned local model vs
few-shot local model. Score accuracy, cost, latency, and privacy.

Learning Objectives:
- Design fair model comparison experiments
- Evaluate trade-offs: accuracy vs cost vs latency vs privacy
- Understand when to use cloud vs local models
- Build comparison frameworks for medical AI

Run:
    python exercise_2_model_comparison.py
"""

from openai import OpenAI
import json
import os
import time
import random
from collections import Counter, defaultdict


# --- Test dataset ---
TEST_CASES = [
    {"note": "67-year-old male with acute substernal chest pain radiating to left arm, diaphoresis. ECG: ST elevation V1-V4. Troponin 3.2 ng/mL.",
     "true_code": "I21.0", "category": "cardiology"},
    {"note": "78-year-old female with progressive DOE, bilateral LE edema, orthopnea × 2 weeks. BNP 1450. Echo EF 25%.",
     "true_code": "I50.9", "category": "cardiology"},
    {"note": "62-year-old male, palpitations and irregular pulse. ECG: irregularly irregular, absent P waves, rate 142.",
     "true_code": "I48.0", "category": "cardiology"},
    {"note": "55-year-old obese female with polyuria, polydipsia, blurred vision × 3 weeks. FBG 245, HbA1c 9.1%.",
     "true_code": "E11.65", "category": "endocrinology"},
    {"note": "42-year-old female, fatigue, 20 lb weight gain, cold intolerance, constipation. TSH 18.2, fT4 0.4.",
     "true_code": "E03.9", "category": "endocrinology"},
    {"note": "73-year-old female with productive cough, T 101.8F, RLL crackles. CXR: RLL consolidation. WBC 14,500.",
     "true_code": "J18.9", "category": "pulmonology"},
    {"note": "68-year-old male, known COPD, worsening dyspnea × 3 days, increased sputum, wheezing. SpO2 88%.",
     "true_code": "J44.1", "category": "pulmonology"},
    {"note": "29-year-old male with acute RLQ pain, nausea, anorexia, T 100.9F. McBurney's tenderness. WBC 15,800.",
     "true_code": "K35.80", "category": "gastroenterology"},
    {"note": "74-year-old female, sudden right hemiparesis, facial droop, aphasia. Onset 90 min. CT neg. NIHSS 16.",
     "true_code": "I63.9", "category": "neurology"},
    {"note": "24-year-old female with dysuria, frequency, urgency, suprapubic pain. UA: positive nitrites and LE.",
     "true_code": "N39.0", "category": "urology"},
    {"note": "82-year-old confused, T 103.1F, HR 112, BP 85/52, cloudy urine, positive blood cultures, lactate 3.8.",
     "true_code": "A41.9", "category": "infectious disease"},
    {"note": "51-year-old male with severe epigastric pain radiating to back, nausea, vomiting. Heavy EtOH hx. Lipase 1850.",
     "true_code": "K85.9", "category": "gastroenterology"},
]

# --- Few-shot examples for local model ---
FEW_SHOT_EXAMPLES = [
    {"note": "65-year-old male, acute chest pain, ST changes, elevated troponin.", "code": "I21.0"},
    {"note": "80-year-old female, dyspnea, edema, low EF on echo.", "code": "I50.9"},
    {"note": "50-year-old with new-onset diabetes, glucose >200, HbA1c elevated.", "code": "E11.65"},
    {"note": "40-year-old female, hypothyroid symptoms, high TSH, low T4.", "code": "E03.9"},
    {"note": "70-year-old with cough, fever, consolidation on CXR.", "code": "J18.9"},
]


def evaluate_gpt4o_zero_shot(client: OpenAI, test_cases: list) -> dict:
    """Evaluate GPT-4o with zero-shot prompting."""

    print("\n--- Approach 1: GPT-4o Zero-Shot ---")
    results = []
    total_tokens = 0
    start = time.time()

    for tc in test_cases:
        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output ONLY: ICD-10: [CODE] - [DESCRIPTION]"},
                    {"role": "user", "content": tc["note"]}
                ],
                temperature=0.1,
                max_tokens=80,
            )
            output = response.choices[0].message.content.strip()
            total_tokens += response.usage.total_tokens
            latency = time.time() - t0

            if "ICD-10:" in output:
                pred = output.split("ICD-10:")[1].strip().split(" - ")[0].strip().split(" ")[0]
            else:
                pred = output[:10]

            results.append({
                "true": tc["true_code"],
                "pred": pred,
                "correct": pred == tc["true_code"],
                "latency": latency,
            })
        except Exception as e:
            results.append({"true": tc["true_code"], "pred": "ERR", "correct": False, "latency": 0})

    elapsed = time.time() - start
    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)

    print(f"  Accuracy:     {accuracy:.1%}")
    print(f"  Avg latency:  {avg_latency:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Est. cost:    ${total_tokens * 0.005 / 1000:.4f}")

    return {
        "model": "GPT-4o (zero-shot)",
        "accuracy": accuracy,
        "avg_latency": avg_latency,
        "total_tokens": total_tokens,
        "est_cost_per_query": total_tokens / len(results) * 0.005 / 1000,
        "privacy": "Cloud — PHI sent to OpenAI",
        "results": results,
    }


def evaluate_gpt4o_few_shot(client: OpenAI, test_cases: list) -> dict:
    """Evaluate GPT-4o with few-shot examples."""

    print("\n--- Approach 2: GPT-4o Few-Shot ---")
    results = []
    total_tokens = 0
    start = time.time()

    # Build few-shot prompt
    few_shot_text = "Here are examples of clinical note to ICD-10 coding:\n\n"
    for ex in FEW_SHOT_EXAMPLES:
        few_shot_text += f"Note: {ex['note']}\nICD-10: {ex['code']}\n\n"
    few_shot_text += "Now code the following note in the same format:"

    for tc in test_cases:
        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical coding assistant."},
                    {"role": "user", "content": f"{few_shot_text}\n\nNote: {tc['note']}"}
                ],
                temperature=0.1,
                max_tokens=80,
            )
            output = response.choices[0].message.content.strip()
            total_tokens += response.usage.total_tokens
            latency = time.time() - t0

            if "ICD-10:" in output:
                pred = output.split("ICD-10:")[1].strip().split(" - ")[0].strip().split(" ")[0]
            elif ":" in output:
                pred = output.split(":")[1].strip().split(" ")[0].split("-")[0].strip()
            else:
                import re
                match = re.search(r'[A-Z]\d{2}\.?\d*', output)
                pred = match.group(0) if match else output[:10]

            results.append({
                "true": tc["true_code"],
                "pred": pred,
                "correct": pred == tc["true_code"],
                "latency": latency,
            })
        except Exception as e:
            results.append({"true": tc["true_code"], "pred": "ERR", "correct": False, "latency": 0})

    elapsed = time.time() - start
    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)

    print(f"  Accuracy:     {accuracy:.1%}")
    print(f"  Avg latency:  {avg_latency:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Est. cost:    ${total_tokens * 0.005 / 1000:.4f}")

    return {
        "model": "GPT-4o (few-shot)",
        "accuracy": accuracy,
        "avg_latency": avg_latency,
        "total_tokens": total_tokens,
        "est_cost_per_query": total_tokens / len(results) * 0.005 / 1000,
        "privacy": "Cloud — PHI sent to OpenAI",
        "results": results,
    }


def simulate_finetuned_local(test_cases: list) -> dict:
    """Simulate a fine-tuned local model evaluation."""

    print("\n--- Approach 3: Fine-Tuned Local Model (Simulated) ---")

    random.seed(42)
    results = []

    for tc in test_cases:
        # Simulate ~75-85% accuracy
        if random.random() < 0.80:
            pred = tc["true_code"]
        else:
            pred = random.choice([t["true_code"] for t in test_cases])

        latency = random.uniform(0.2, 0.8)  # local inference
        results.append({
            "true": tc["true_code"],
            "pred": pred,
            "correct": pred == tc["true_code"],
            "latency": latency,
        })

    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)

    print(f"  Accuracy:     {accuracy:.1%} (simulated)")
    print(f"  Avg latency:  {avg_latency:.2f}s (simulated local GPU)")
    print(f"  Cost:         $0.00 (runs locally)")

    return {
        "model": "Fine-Tuned Local (LoRA)",
        "accuracy": accuracy,
        "avg_latency": avg_latency,
        "total_tokens": 0,
        "est_cost_per_query": 0,
        "privacy": "Local — HIPAA compliant, no data leaves",
        "results": results,
    }


def simulate_fewshot_local(test_cases: list) -> dict:
    """Simulate few-shot prompting with a local model (no fine-tuning)."""

    print("\n--- Approach 4: Few-Shot Local Model (Simulated) ---")

    random.seed(123)
    results = []

    for tc in test_cases:
        # Simulate ~55-65% accuracy (local model without fine-tuning)
        if random.random() < 0.58:
            pred = tc["true_code"]
        else:
            pred = random.choice([t["true_code"] for t in test_cases])

        latency = random.uniform(0.5, 1.5)
        results.append({
            "true": tc["true_code"],
            "pred": pred,
            "correct": pred == tc["true_code"],
            "latency": latency,
        })

    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)

    print(f"  Accuracy:     {accuracy:.1%} (simulated)")
    print(f"  Avg latency:  {avg_latency:.2f}s (simulated local GPU)")
    print(f"  Cost:         $0.00 (runs locally)")

    return {
        "model": "Few-Shot Local (no FT)",
        "accuracy": accuracy,
        "avg_latency": avg_latency,
        "total_tokens": 0,
        "est_cost_per_query": 0,
        "privacy": "Local — HIPAA compliant, no data leaves",
        "results": results,
    }


def comprehensive_comparison(all_approaches: list):
    """Build comprehensive comparison table."""

    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 80)

    # --- Accuracy comparison ---
    print(f"\n  {'Model':30s} {'Accuracy':>9s} {'Latency':>9s} {'Cost/Query':>11s}")
    print("  " + "-" * 65)
    for a in all_approaches:
        cost_str = f"${a['est_cost_per_query']:.4f}" if a['est_cost_per_query'] > 0 else "$0.0000"
        print(f"  {a['model']:30s} {a['accuracy']:9.1%} {a['avg_latency']:8.2f}s {cost_str:>11s}")

    # --- Visual accuracy bars ---
    print(f"\n  Accuracy:")
    max_acc = max(a["accuracy"] for a in all_approaches)
    for a in all_approaches:
        bar_len = int(a["accuracy"] / max_acc * 35)
        bar = "█" * bar_len
        print(f"    {a['model']:30s} {bar} {a['accuracy']:.1%}")

    # --- Visual latency bars ---
    print(f"\n  Latency (lower is better):")
    max_lat = max(a["avg_latency"] for a in all_approaches)
    for a in all_approaches:
        bar_len = int(a["avg_latency"] / max_lat * 35)
        bar = "▓" * bar_len
        print(f"    {a['model']:30s} {bar} {a['avg_latency']:.2f}s")

    # --- Privacy comparison ---
    print(f"\n  Privacy:")
    for a in all_approaches:
        print(f"    {a['model']:30s} {a['privacy']}")

    # --- Cost projection ---
    print(f"\n  Cost Projection (1000 queries/day × 30 days):")
    for a in all_approaches:
        monthly = a['est_cost_per_query'] * 1000 * 30
        if monthly > 0:
            print(f"    {a['model']:30s} ${monthly:>8.2f}/month")
        else:
            # Estimate local hardware cost
            print(f"    {a['model']:30s} $0 API + ~$50/month electricity")

    # --- Recommendation matrix ---
    print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │                    RECOMMENDATION MATRIX                        │
  ├──────────────────────┬──────────────────┬───────────────────────┤
  │ Priority             │ Best Choice      │ Reasoning             │
  ├──────────────────────┼──────────────────┼───────────────────────┤
  │ Maximum accuracy     │ GPT-4o few-shot  │ Highest raw accuracy  │
  │ HIPAA compliance     │ Fine-tuned local │ No data leaves system │
  │ Lowest cost          │ Local models     │ Zero API costs        │
  │ Lowest latency       │ Fine-tuned local │ Optimized for task    │
  │ Easiest to maintain  │ GPT-4o zero-shot │ No training needed    │
  │ Best accuracy/privacy│ Fine-tuned local │ Competitive + private │
  └──────────────────────┴──────────────────┴───────────────────────┘
""")


def main():
    """Compare multiple model approaches for ICD-10 coding."""

    print("=" * 60)
    print("Exercise 2: Model Comparison Study")
    print("=" * 60)

    all_approaches = []

    # Approach 1 & 2: GPT-4o
    try:
        client = OpenAI()

        result1 = evaluate_gpt4o_zero_shot(client, TEST_CASES)
        all_approaches.append(result1)

        result2 = evaluate_gpt4o_few_shot(client, TEST_CASES)
        all_approaches.append(result2)

    except Exception as e:
        print(f"\n  GPT-4o evaluation failed: {e}")
        print("  Using simulated results...")

        random.seed(99)
        for name, acc in [("GPT-4o (zero-shot)", 0.83), ("GPT-4o (few-shot)", 0.88)]:
            results = [{"true": tc["true_code"],
                        "pred": tc["true_code"] if random.random() < acc else random.choice([t["true_code"] for t in TEST_CASES]),
                        "correct": random.random() < acc,
                        "latency": random.uniform(0.5, 1.5)}
                       for tc in TEST_CASES]
            all_approaches.append({
                "model": name + " (simulated)",
                "accuracy": sum(r["correct"] for r in results) / len(results),
                "avg_latency": sum(r["latency"] for r in results) / len(results),
                "total_tokens": 0,
                "est_cost_per_query": 0.003,
                "privacy": "Cloud — PHI sent to OpenAI",
                "results": results,
            })

    # Approach 3: Fine-tuned local
    result3 = simulate_finetuned_local(TEST_CASES)
    all_approaches.append(result3)

    # Approach 4: Few-shot local
    result4 = simulate_fewshot_local(TEST_CASES)
    all_approaches.append(result4)

    # Comprehensive comparison
    comprehensive_comparison(all_approaches)

    print("✓ Model comparison complete!")


if __name__ == "__main__":
    main()
