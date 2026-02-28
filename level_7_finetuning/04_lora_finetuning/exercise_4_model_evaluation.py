"""
Exercise 4: Model Evaluation
=============================
Evaluate fine-tuned model: accuracy, F1 score, confusion matrix,
per-class performance. Compare base model vs fine-tuned vs GPT-4o.

Learning Objectives:
- Calculate classification metrics (accuracy, precision, recall, F1)
- Build confusion matrices
- Analyze per-class performance
- Compare multiple model approaches

Run:
    python exercise_4_model_evaluation.py
"""

from openai import OpenAI
import json
import os
import random
from collections import Counter, defaultdict


# --- ICD-10 test cases ---
TEST_CASES = [
    {"note": "67-year-old male with acute substernal chest pain radiating to left arm, ST elevation in V1-V4, troponin 3.2.",
     "true_code": "I21.0", "true_desc": "Acute STEMI anterior wall"},
    {"note": "78-year-old female with progressive dyspnea, bilateral edema, orthopnea, BNP 1450, EF 25%.",
     "true_code": "I50.9", "true_desc": "Heart failure, unspecified"},
    {"note": "62-year-old male with irregular pulse, palpitations, ECG shows absent P waves, rate 142.",
     "true_code": "I48.0", "true_desc": "Paroxysmal atrial fibrillation"},
    {"note": "55-year-old obese female with polyuria, polydipsia, fasting glucose 245, HbA1c 9.1%.",
     "true_code": "E11.65", "true_desc": "Type 2 diabetes with hyperglycemia"},
    {"note": "42-year-old female with fatigue, weight gain 20lbs, cold intolerance, TSH 18.2, free T4 0.4.",
     "true_code": "E03.9", "true_desc": "Hypothyroidism, unspecified"},
    {"note": "73-year-old female with productive cough, fever 101.8F, RLL crackles, CXR consolidation, WBC 14500.",
     "true_code": "J18.9", "true_desc": "Pneumonia, unspecified organism"},
    {"note": "68-year-old male with known COPD, worsening dyspnea 3 days, increased sputum, O2 sat 88%.",
     "true_code": "J44.1", "true_desc": "COPD with acute exacerbation"},
    {"note": "8-year-old with acute wheezing, dyspnea, chest tightness after playing, O2 92%, bilateral wheezes.",
     "true_code": "J45.41", "true_desc": "Moderate persistent asthma with exacerbation"},
    {"note": "29-year-old male with acute RLQ pain, nausea, fever 100.9F, McBurney's tenderness, WBC 15800.",
     "true_code": "K35.80", "true_desc": "Acute appendicitis"},
    {"note": "51-year-old male with epigastric pain radiating to back, vomiting, alcohol history, lipase 1850.",
     "true_code": "K85.9", "true_desc": "Acute pancreatitis"},
    {"note": "74-year-old female with sudden right hemiparesis, facial droop, aphasia, CT negative, NIHSS 16.",
     "true_code": "I63.9", "true_desc": "Cerebral infarction"},
    {"note": "28-year-old female with recurrent severe unilateral headache, photophobia, nausea, 12-24 hours.",
     "true_code": "G43.909", "true_desc": "Migraine, unspecified"},
    {"note": "24-year-old female with dysuria, frequency, urgency, suprapubic pain, positive nitrites and LE.",
     "true_code": "N39.0", "true_desc": "UTI"},
    {"note": "45-year-old with acute low back pain after lifting, radiates to buttock, negative SLR.",
     "true_code": "M54.5", "true_desc": "Low back pain"},
    {"note": "82-year-old confused, T 103.1, HR 112, BP 85/52, cloudy urine, lactate 3.8.",
     "true_code": "A41.9", "true_desc": "Sepsis, unspecified"},
]


def extract_icd10_code(text: str) -> str:
    """Extract ICD-10 code from model output."""
    text = text.strip()

    # Try "ICD-10: X99.9" format
    if "ICD-10:" in text:
        after = text.split("ICD-10:")[1].strip()
        code = after.split(" ")[0].split("-")[0].strip().rstrip(".-")
        return code

    # Try just a code pattern
    import re
    match = re.search(r'[A-Z]\d{2}\.?\d{0,4}', text)
    if match:
        return match.group(0)

    return text[:10]


def simulate_model_predictions(test_cases: list, model_type: str,
                                accuracy: float = 0.8) -> list:
    """Simulate predictions from a model (for demo without actual models)."""

    random.seed(hash(model_type))
    all_codes = list(set(tc["true_code"] for tc in test_cases))
    predictions = []

    for tc in test_cases:
        if random.random() < accuracy:
            pred_code = tc["true_code"]
        else:
            # Pick a wrong code, sometimes from same category
            if random.random() < 0.6:
                same_cat = [c for c in all_codes if c[0] == tc["true_code"][0] and c != tc["true_code"]]
                pred_code = random.choice(same_cat) if same_cat else random.choice(all_codes)
            else:
                pred_code = random.choice(all_codes)

        predictions.append({
            "note": tc["note"],
            "true_code": tc["true_code"],
            "pred_code": pred_code,
            "correct": pred_code == tc["true_code"],
        })

    return predictions


def evaluate_with_gpt4o(client: OpenAI, test_cases: list) -> list:
    """Get ICD-10 predictions from GPT-4o."""

    predictions = []

    for tc in test_cases:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output ONLY the ICD-10 code in format: ICD-10: [CODE] - [DESCRIPTION]. Nothing else."},
                    {"role": "user", "content": tc["note"]}
                ],
                temperature=0.1,
                max_tokens=100,
            )

            output = response.choices[0].message.content.strip()
            pred_code = extract_icd10_code(output)

            predictions.append({
                "note": tc["note"],
                "true_code": tc["true_code"],
                "pred_code": pred_code,
                "raw_output": output,
                "correct": pred_code == tc["true_code"],
            })

        except Exception as e:
            predictions.append({
                "note": tc["note"],
                "true_code": tc["true_code"],
                "pred_code": "ERROR",
                "raw_output": str(e),
                "correct": False,
            })

    return predictions


def calculate_metrics(predictions: list) -> dict:
    """Calculate classification metrics."""

    all_codes = sorted(set(p["true_code"] for p in predictions) |
                       set(p["pred_code"] for p in predictions))

    # Overall accuracy
    correct = sum(1 for p in predictions if p["correct"])
    accuracy = correct / len(predictions) if predictions else 0

    # Per-code metrics
    per_code = {}
    for code in all_codes:
        tp = sum(1 for p in predictions if p["true_code"] == code and p["pred_code"] == code)
        fp = sum(1 for p in predictions if p["true_code"] != code and p["pred_code"] == code)
        fn = sum(1 for p in predictions if p["true_code"] == code and p["pred_code"] != code)
        tn = sum(1 for p in predictions if p["true_code"] != code and p["pred_code"] != code)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = tp + fn

        per_code[code] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": support, "tp": tp, "fp": fp, "fn": fn,
        }

    # Macro averages
    f1_values = [m["f1"] for m in per_code.values() if m["support"] > 0]
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0

    # Weighted averages
    total = sum(m["support"] for m in per_code.values())
    weighted_f1 = sum(m["f1"] * m["support"] for m in per_code.values()) / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_code": per_code,
        "correct": correct,
        "total": len(predictions),
    }


def print_classification_report(metrics: dict, title: str = ""):
    """Print a sklearn-style classification report."""

    print(f"\n  {'=' * 70}")
    if title:
        print(f"  {title}")
    print(f"  {'=' * 70}")
    print(f"\n  {'Code':12s} {'Precision':>10s} {'Recall':>8s} {'F1':>6s} {'Support':>8s}")
    print("  " + "-" * 48)

    for code, m in sorted(metrics["per_code"].items()):
        if m["support"] > 0:
            print(f"  {code:12s} {m['precision']:10.2f} {m['recall']:8.2f} {m['f1']:6.2f} {m['support']:8d}")

    print("  " + "-" * 48)
    print(f"  {'accuracy':12s} {'':10s} {'':8s} {metrics['accuracy']:6.2f} {metrics['total']:8d}")
    print(f"  {'macro avg':12s} {'':10s} {'':8s} {metrics['macro_f1']:6.2f} {metrics['total']:8d}")
    print(f"  {'weighted avg':12s} {'':10s} {'':8s} {metrics['weighted_f1']:6.2f} {metrics['total']:8d}")


def print_confusion_matrix(predictions: list):
    """Print a text-based confusion matrix."""

    codes = sorted(set(p["true_code"] for p in predictions))
    code_to_idx = {c: i for i, c in enumerate(codes)}

    matrix = [[0] * len(codes) for _ in range(len(codes))]
    for p in predictions:
        if p["true_code"] in code_to_idx and p["pred_code"] in code_to_idx:
            i = code_to_idx[p["true_code"]]
            j = code_to_idx[p["pred_code"]]
            matrix[i][j] += 1

    # Compact code labels
    short_codes = [c[:7] for c in codes]

    print(f"\n  Confusion Matrix:")
    print(f"  {'':8s}", end="")
    for sc in short_codes:
        print(f" {sc:>7s}", end="")
    print("  (predicted)")

    for i, (code, row) in enumerate(zip(short_codes, matrix)):
        print(f"  {code:8s}", end="")
        for j, val in enumerate(row):
            if val > 0:
                marker = f" {val:7d}" if i != j else f" [{val:5d}]"
                print(marker, end="")
            else:
                print(f" {'·':>7s}", end="")
        print()


def compare_models(all_results: dict):
    """Compare performance across different models."""

    print(f"\n{'=' * 70}")
    print("Model Comparison")
    print(f"{'=' * 70}\n")

    print(f"  {'Model':25s} {'Accuracy':>9s} {'Macro F1':>9s} {'Weighted F1':>11s}")
    print("  " + "-" * 58)

    for model_name, metrics in all_results.items():
        print(f"  {model_name:25s} {metrics['accuracy']:9.2%} "
              f"{metrics['macro_f1']:9.4f} {metrics['weighted_f1']:11.4f}")

    # Visual comparison
    print(f"\n  Accuracy comparison:")
    max_acc = max(m["accuracy"] for m in all_results.values())
    for name, metrics in all_results.items():
        bar_len = int(metrics["accuracy"] / max_acc * 30)
        bar = "█" * bar_len
        print(f"    {name:25s} {bar} {metrics['accuracy']:.1%}")


def main():
    """Evaluate and compare models for ICD-10 coding."""

    print("=" * 60)
    print("Exercise 4: Model Evaluation & Comparison")
    print("=" * 60)

    all_results = {}

    # --- 1. Simulate base model (no fine-tuning) ---
    print("\n--- Evaluating: Base Model (simulated, ~40% accuracy) ---")
    base_preds = simulate_model_predictions(TEST_CASES, "base_model", accuracy=0.40)
    base_metrics = calculate_metrics(base_preds)
    print_classification_report(base_metrics, "Base Model (no fine-tuning)")
    all_results["Base Model (no FT)"] = base_metrics

    # --- 2. Simulate fine-tuned model ---
    print("\n--- Evaluating: Fine-Tuned Model (simulated, ~78% accuracy) ---")
    ft_preds = simulate_model_predictions(TEST_CASES, "finetuned", accuracy=0.78)
    ft_metrics = calculate_metrics(ft_preds)
    print_classification_report(ft_metrics, "Fine-Tuned LoRA Model")
    print_confusion_matrix(ft_preds)
    all_results["Fine-Tuned LoRA"] = ft_metrics

    # --- 3. Evaluate GPT-4o ---
    print("\n--- Evaluating: GPT-4o ---")
    try:
        client = OpenAI()
        gpt4_preds = evaluate_with_gpt4o(client, TEST_CASES)
        gpt4_metrics = calculate_metrics(gpt4_preds)
        print_classification_report(gpt4_metrics, "GPT-4o")

        # Show GPT-4o predictions
        print(f"\n  GPT-4o Predictions:")
        for p in gpt4_preds:
            status = "✓" if p["correct"] else "✗"
            print(f"  {status} True: {p['true_code']:8s} Pred: {p['pred_code']:8s}  "
                  f"{p['note'][:60]}...")

        all_results["GPT-4o"] = gpt4_metrics
    except Exception as e:
        print(f"  Could not evaluate GPT-4o: {e}")
        # Use simulated results
        gpt4_preds = simulate_model_predictions(TEST_CASES, "gpt4o", accuracy=0.87)
        gpt4_metrics = calculate_metrics(gpt4_preds)
        print_classification_report(gpt4_metrics, "GPT-4o (simulated)")
        all_results["GPT-4o (simulated)"] = gpt4_metrics

    # --- 4. Compare all models ---
    compare_models(all_results)

    # --- 5. Error analysis ---
    print(f"\n{'=' * 60}")
    print("Error Analysis (Fine-Tuned Model)")
    print(f"{'=' * 60}")

    errors = [p for p in ft_preds if not p["correct"]]
    print(f"\n  Total errors: {len(errors)}/{len(ft_preds)}")

    for err in errors:
        true_cat = err["true_code"].split(".")[0]
        pred_cat = err["pred_code"].split(".")[0]
        error_type = "same category" if true_cat == pred_cat else "wrong category"
        print(f"\n  True: {err['true_code']} → Pred: {err['pred_code']} ({error_type})")
        print(f"  Note: {err['note'][:80]}...")

    print(f"\n{'=' * 60}")
    print("✓ Model evaluation complete!")
    print("  Key insight: Fine-tuning bridges the gap between base model")
    print("  and GPT-4o while keeping inference local and HIPAA-safe.")


if __name__ == "__main__":
    main()
