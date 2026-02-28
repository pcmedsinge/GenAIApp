"""
Exercise 3: Error Analysis
===========================
Analyze model errors. Categorize failures: wrong code, wrong category,
hallucinated code, ambiguous case. Suggest improvements for each type.

Learning Objectives:
- Systematically categorize model errors
- Identify patterns in failure modes
- Use LLM to analyze why errors occurred
- Generate actionable improvement suggestions

Run:
    python exercise_3_error_analysis.py
"""

from openai import OpenAI
import json
import os
import random
from collections import Counter, defaultdict


# --- ICD-10 code reference ---
VALID_CODES = {
    "I21.0": ("Acute STEMI anterior wall", "Diseases of the circulatory system"),
    "I21.19": ("STEMI inferior wall", "Diseases of the circulatory system"),
    "I50.9": ("Heart failure, unspecified", "Diseases of the circulatory system"),
    "I48.0": ("Paroxysmal atrial fibrillation", "Diseases of the circulatory system"),
    "I63.9": ("Cerebral infarction", "Diseases of the circulatory system"),
    "E11.9": ("T2DM without complications", "Endocrine diseases"),
    "E11.65": ("T2DM with hyperglycemia", "Endocrine diseases"),
    "E03.9": ("Hypothyroidism", "Endocrine diseases"),
    "J18.9": ("Pneumonia", "Diseases of the respiratory system"),
    "J44.1": ("COPD with exacerbation", "Diseases of the respiratory system"),
    "J45.41": ("Asthma with exacerbation", "Diseases of the respiratory system"),
    "K35.80": ("Acute appendicitis", "Diseases of the digestive system"),
    "K85.9": ("Acute pancreatitis", "Diseases of the digestive system"),
    "N39.0": ("UTI", "Diseases of the genitourinary system"),
    "A41.9": ("Sepsis", "Infectious diseases"),
    "M54.5": ("Low back pain", "Diseases of the musculoskeletal system"),
    "G43.909": ("Migraine", "Diseases of the nervous system"),
}


def generate_simulated_errors() -> list:
    """Generate a realistic set of model predictions including errors."""

    random.seed(42)

    test_set = [
        # Correct predictions
        {"note": "67M, chest pain, ST elevation V1-V4, troponin 3.2.",
         "true": "I21.0", "pred": "I21.0"},
        {"note": "55F, polyuria, polydipsia, FBG 245, A1c 9.1%.",
         "true": "E11.65", "pred": "E11.65"},
        {"note": "73F, productive cough, fever, RLL consolidation, WBC 14k.",
         "true": "J18.9", "pred": "J18.9"},
        {"note": "29M, RLQ pain, McBurney's tenderness, WBC 15.8k.",
         "true": "K35.80", "pred": "K35.80"},
        {"note": "42F, fatigue, weight gain, TSH 18.2, fT4 0.4.",
         "true": "E03.9", "pred": "E03.9"},
        {"note": "24F, dysuria, frequency, positive nitrites.",
         "true": "N39.0", "pred": "N39.0"},

        # Wrong specific code (same category)
        {"note": "70M, chest pain radiating to back, inferior ST changes, troponin elevated.",
         "true": "I21.19", "pred": "I21.0",
         "error_type": "wrong_specific"},
        {"note": "58F, new diabetes, glucose 180, A1c 7.5%, no complications noted.",
         "true": "E11.9", "pred": "E11.65",
         "error_type": "wrong_specific"},
        {"note": "78F, dyspnea, edema, BNP 900, EF 35%.",
         "true": "I50.9", "pred": "I48.0",
         "error_type": "wrong_specific"},

        # Wrong category entirely
        {"note": "82F, confused, T 103, HR 112, BP 85/52, positive blood cultures.",
         "true": "A41.9", "pred": "N39.0",
         "error_type": "wrong_category"},
        {"note": "45M, acute back pain radiating to leg, positive SLR.",
         "true": "M54.5", "pred": "G43.909",
         "error_type": "wrong_category"},

        # Hallucinated code
        {"note": "35F, intermittent palpitations, normal ECG, negative workup.",
         "true": "R00.2", "pred": "Z99.99",
         "error_type": "hallucinated"},
        {"note": "60M, chronic cough, no fever, CXR clear.",
         "true": "R05.9", "pred": "J99.999",
         "error_type": "hallucinated"},

        # Ambiguous cases
        {"note": "68M, COPD exacerbation with infiltrate on CXR. Possible pneumonia.",
         "true": "J44.1", "pred": "J18.9",
         "error_type": "ambiguous"},
        {"note": "75F, new atrial fibrillation with decompensated heart failure.",
         "true": "I48.0", "pred": "I50.9",
         "error_type": "ambiguous"},
    ]

    return test_set


def categorize_error(prediction: dict) -> str:
    """Categorize the type of error."""

    true_code = prediction["true"]
    pred_code = prediction["pred"]

    if true_code == pred_code:
        return "correct"

    # Check for pre-labeled error types
    if "error_type" in prediction:
        return prediction["error_type"]

    # Auto-categorize
    if pred_code not in VALID_CODES:
        return "hallucinated"

    true_category = VALID_CODES.get(true_code, ("", "Unknown"))[1]
    pred_category = VALID_CODES.get(pred_code, ("", "Unknown"))[1]

    if true_category == pred_category:
        return "wrong_specific"
    else:
        return "wrong_category"


def analyze_errors_with_llm(client: OpenAI, errors: list) -> list:
    """Use GPT-4o to analyze why each error occurred."""

    analyzed = []

    for err in errors[:8]:  # Limit to save API calls
        prompt = f"""Analyze this medical coding error:

Clinical note: "{err['note']}"
True ICD-10 code: {err['true']} ({VALID_CODES.get(err['true'], ('Unknown',))[0]})
Predicted code: {err['pred']} ({VALID_CODES.get(err['pred'], ('Unknown/hallucinated',))[0]})
Error type: {err.get('error_type', 'unknown')}

Provide:
1. WHY the model likely made this error (1-2 sentences)
2. What TRAINING DATA would help prevent this error
3. A DIFFICULTY rating (1-5, where 5 = very ambiguous)

Return ONLY a JSON object with "why", "training_suggestion", and "difficulty" fields. No markdown."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical AI error analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )

            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            analysis = json.loads(raw)
            err["analysis"] = analysis
        except Exception as e:
            err["analysis"] = {"why": f"Analysis failed: {e}", "training_suggestion": "", "difficulty": 0}

        analyzed.append(err)

    return analyzed


def print_error_report(predictions: list, analyzed_errors: list):
    """Print a comprehensive error analysis report."""

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS REPORT")
    print("=" * 70)

    # --- Overall stats ---
    correct = sum(1 for p in predictions if p["true"] == p["pred"])
    total = len(predictions)
    errors = [p for p in predictions if p["true"] != p["pred"]]

    print(f"\n  Overall: {correct}/{total} correct ({correct/total:.1%} accuracy)")
    print(f"  Errors:  {len(errors)}")

    # --- Error type distribution ---
    print(f"\n--- Error Type Distribution ---")
    error_types = Counter(categorize_error(e) for e in errors)

    type_descriptions = {
        "wrong_specific": "Wrong specific code (same ICD-10 category)",
        "wrong_category": "Wrong ICD-10 category entirely",
        "hallucinated": "Hallucinated/invalid ICD-10 code",
        "ambiguous": "Ambiguous case (multiple valid codes)",
    }

    max_count = max(error_types.values()) if error_types else 1
    for etype, count in error_types.most_common():
        desc = type_descriptions.get(etype, etype)
        bar = "█" * int(count / max_count * 25)
        pct = count / len(errors) * 100
        print(f"  {desc:50s} {count:3d} ({pct:4.0f}%)  {bar}")

    # --- Detailed error listing ---
    print(f"\n--- Detailed Errors ---")
    for i, err in enumerate(errors):
        etype = categorize_error(err)
        true_desc = VALID_CODES.get(err["true"], ("Unknown",))[0]
        pred_desc = VALID_CODES.get(err["pred"], ("Unknown/hallucinated",))[0]

        print(f"\n  Error {i + 1} [{etype.upper()}]:")
        print(f"    Note:  {err['note'][:80]}...")
        print(f"    True:  {err['true']} ({true_desc})")
        print(f"    Pred:  {err['pred']} ({pred_desc})")

    # --- LLM analysis results ---
    if analyzed_errors:
        print(f"\n--- LLM Error Analysis ---")
        for i, err in enumerate(analyzed_errors):
            analysis = err.get("analysis", {})
            print(f"\n  Error {i + 1}: {err['true']} → {err['pred']}")
            print(f"    Why:        {analysis.get('why', 'N/A')}")
            print(f"    Fix:        {analysis.get('training_suggestion', 'N/A')}")
            print(f"    Difficulty: {analysis.get('difficulty', '?')}/5")

    # --- Improvement recommendations ---
    print(f"\n--- Improvement Recommendations ---")
    print(f"""
  Based on error analysis:

  1. WRONG SPECIFIC CODE ({error_types.get('wrong_specific', 0)} errors):
     → Add more training examples distinguishing similar codes
     → Include notes that contrast subtle differences (e.g., I21.0 vs I21.19)
     → Add specificity hints in system prompt

  2. WRONG CATEGORY ({error_types.get('wrong_category', 0)} errors):
     → Review training data balance across categories
     → Add hard negatives: cases that look like one category but are another
     → Consider multi-label training for comorbid conditions

  3. HALLUCINATED CODES ({error_types.get('hallucinated', 0)} errors):
     → Add code validation as a post-processing step
     → Include negative examples with correction in training data
     → Constrain generation to valid ICD-10 code list

  4. AMBIGUOUS CASES ({error_types.get('ambiguous', 0)} errors):
     → These may not be errors — multiple codes can be correct
     → Consider multi-label output format
     → Add training examples that explicitly address ambiguity
     → Include "primary diagnosis" and "secondary diagnosis" in output format
""")


def main():
    """Analyze model errors for ICD-10 coding."""

    print("=" * 60)
    print("Exercise 3: Error Analysis")
    print("=" * 60)

    # Generate simulated predictions
    print("\n--- Generating Simulated Predictions ---")
    predictions = generate_simulated_errors()
    print(f"  Total predictions: {len(predictions)}")

    errors = [p for p in predictions if p["true"] != p["pred"]]
    print(f"  Errors to analyze: {len(errors)}")

    # LLM-powered analysis
    print("\n--- Running LLM Error Analysis ---")
    try:
        client = OpenAI()
        analyzed = analyze_errors_with_llm(client, errors)
    except Exception as e:
        print(f"  LLM analysis unavailable: {e}")
        analyzed = []

    # Print report
    print_error_report(predictions, analyzed)

    # Save report
    report_path = os.path.join(os.path.dirname(__file__) or ".", "error_analysis_report.json")
    report_data = {
        "total_predictions": len(predictions),
        "total_errors": len(errors),
        "error_types": dict(Counter(categorize_error(e) for e in errors)),
        "errors": [
            {
                "note": e["note"],
                "true_code": e["true"],
                "pred_code": e["pred"],
                "error_type": categorize_error(e),
                "analysis": e.get("analysis", {}),
            }
            for e in errors
        ],
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n  Report saved to: {report_path}")
    print(f"\n  ✓ Error analysis complete!")


if __name__ == "__main__":
    main()
