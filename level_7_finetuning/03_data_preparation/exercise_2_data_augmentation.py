"""
Exercise 2: Data Augmentation via LLM Paraphrasing
===================================================
Augment training data using LLM paraphrasing. Take 10 base examples,
generate 3 variations each, and validate augmented data quality.

Learning Objectives:
- Use LLMs for data augmentation
- Preserve clinical accuracy while varying expression
- Validate augmented data quality
- Expand small datasets efficiently

Run:
    python exercise_2_data_augmentation.py
"""

from openai import OpenAI
import json
import os
import random
from collections import Counter


# --- Base training examples to augment ---
BASE_EXAMPLES = [
    {
        "clinical_note": "67-year-old male presenting with substernal chest pain radiating to left arm, diaphoresis, and shortness of breath. ECG shows ST elevation in leads II, III, aVF. Troponin elevated at 2.4 ng/mL.",
        "icd10_code": "I21.19",
        "icd10_description": "ST elevation myocardial infarction involving other coronary artery of inferior wall",
    },
    {
        "clinical_note": "78-year-old female with progressive dyspnea on exertion, bilateral lower extremity edema, and orthopnea over the past 2 weeks. BNP 1450 pg/mL. Echo shows EF 25%.",
        "icd10_code": "I50.9",
        "icd10_description": "Heart failure, unspecified",
    },
    {
        "clinical_note": "55-year-old obese female with polyuria, polydipsia, and blurred vision. Fasting glucose 245 mg/dL, HbA1c 9.1%. No prior diabetes diagnosis.",
        "icd10_code": "E11.65",
        "icd10_description": "Type 2 diabetes mellitus with hyperglycemia",
    },
    {
        "clinical_note": "42-year-old female with fatigue, weight gain of 20 lbs, cold intolerance, and constipation. TSH 18.2 mIU/L, free T4 0.4 ng/dL.",
        "icd10_code": "E03.9",
        "icd10_description": "Hypothyroidism, unspecified",
    },
    {
        "clinical_note": "73-year-old female with productive cough, fever 101.8F, and right lower lobe crackles. CXR shows RLL consolidation. WBC 14,500.",
        "icd10_code": "J18.9",
        "icd10_description": "Pneumonia, unspecified organism",
    },
    {
        "clinical_note": "68-year-old male with known COPD, worsening dyspnea over 3 days, increased sputum production, and wheezing. O2 sat 88% on room air.",
        "icd10_code": "J44.1",
        "icd10_description": "COPD with acute exacerbation",
    },
    {
        "clinical_note": "29-year-old male with acute onset RLQ pain, nausea, anorexia, fever 100.9F. Positive McBurney's point tenderness with rebound. WBC 15,800.",
        "icd10_code": "K35.80",
        "icd10_description": "Unspecified acute appendicitis without abscess",
    },
    {
        "clinical_note": "74-year-old female with sudden onset right-sided hemiparesis, facial droop, and expressive aphasia. Symptoms began 90 minutes ago. CT head negative for hemorrhage.",
        "icd10_code": "I63.9",
        "icd10_description": "Cerebral infarction, unspecified",
    },
    {
        "clinical_note": "51-year-old male with severe epigastric pain radiating to back, nausea, persistent vomiting. Heavy alcohol history. Lipase 1,850 U/L.",
        "icd10_code": "K85.9",
        "icd10_description": "Acute pancreatitis, unspecified",
    },
    {
        "clinical_note": "24-year-old female with dysuria, urinary frequency, urgency, and suprapubic discomfort. Urine dipstick positive for nitrites and leukocyte esterase.",
        "icd10_code": "N39.0",
        "icd10_description": "Urinary tract infection, site not specified",
    },
]


def augment_with_paraphrasing(client: OpenAI, example: dict, num_variations: int = 3) -> list:
    """Generate paraphrased variations of a clinical note using GPT-4o."""

    prompt = f"""Paraphrase the following clinical note {num_variations} different ways.
Each paraphrase must:
1. Preserve ALL clinical facts (symptoms, vitals, lab values, demographics)
2. Change the sentence structure and word choice
3. Remain clinically accurate and realistic
4. Be a different length than the original (shorter or longer)
5. Use different medical terminology where appropriate (e.g., "dyspnea" ↔ "shortness of breath")

Original clinical note:
"{example['clinical_note']}"

Correct ICD-10 code: {example['icd10_code']} - {example['icd10_description']}

Return ONLY a JSON array of objects with "clinical_note" field. No markdown fences."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a medical text augmentation assistant. Generate clinically accurate paraphrases that preserve all medical facts while varying the expression."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        variations = json.loads(raw)
    except json.JSONDecodeError:
        print(f"    Warning: Could not parse response for {example['icd10_code']}")
        variations = []

    return variations


def validate_augmented_data(client: OpenAI, original: dict, augmented_note: str) -> dict:
    """Use GPT-4o to validate that augmented data preserves clinical accuracy."""

    prompt = f"""Compare the original and paraphrased clinical notes below.
Score the paraphrase on these criteria (1-5 each):

1. ACCURACY: Are all clinical facts preserved? (symptoms, vitals, labs, demographics)
2. DIVERSITY: How different is the wording from the original?
3. REALISM: Does it read like a real clinical note?
4. COMPLETENESS: Are any important details missing?

Original: "{original['clinical_note']}"
Paraphrase: "{augmented_note}"
Expected ICD-10: {original['icd10_code']}

Return ONLY a JSON object with "accuracy", "diversity", "realism", "completeness" (integers 1-5), and "issues" (string, empty if none). No markdown."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a medical data quality reviewer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        scores = {"accuracy": 0, "diversity": 0, "realism": 0, "completeness": 0, "issues": "parse error"}

    return scores


def format_training_sample(note: str, code: str, description: str) -> dict:
    """Format a note/code pair into instruction-tuning format."""
    return {
        "messages": [
            {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
            {"role": "user", "content": note},
            {"role": "assistant", "content": f"ICD-10: {code} - {description}"}
        ]
    }


def main():
    """Augment training data using LLM paraphrasing."""

    print("=" * 60)
    print("Exercise 2: Data Augmentation via LLM Paraphrasing")
    print("=" * 60)

    client = OpenAI()

    # --- Step 1: Show base dataset ---
    print(f"\n--- Base Dataset: {len(BASE_EXAMPLES)} examples ---")
    for i, ex in enumerate(BASE_EXAMPLES):
        print(f"  {i + 1}. [{ex['icd10_code']}] {ex['clinical_note'][:80]}...")

    # --- Step 2: Augment each example ---
    print(f"\n--- Augmenting with {3} paraphrases each ---")
    all_augmented = []
    augmentation_log = []

    for i, example in enumerate(BASE_EXAMPLES):
        print(f"\n  Example {i + 1}/{len(BASE_EXAMPLES)}: {example['icd10_code']}")
        print(f"    Original: {example['clinical_note'][:70]}...")

        try:
            variations = augment_with_paraphrasing(client, example, num_variations=3)

            for j, var in enumerate(variations):
                note = var.get("clinical_note", "")
                print(f"    Variation {j + 1}: {note[:70]}...")
                all_augmented.append({
                    "note": note,
                    "code": example["icd10_code"],
                    "description": example["icd10_description"],
                    "source": "augmented",
                    "original_index": i,
                })
                augmentation_log.append({
                    "original_code": example["icd10_code"],
                    "original_len": len(example["clinical_note"]),
                    "augmented_len": len(note),
                })

        except Exception as e:
            print(f"    Error: {e}")

    print(f"\n--- Augmentation Results ---")
    print(f"  Base examples:      {len(BASE_EXAMPLES)}")
    print(f"  Augmented examples: {len(all_augmented)}")
    print(f"  Total dataset size: {len(BASE_EXAMPLES) + len(all_augmented)}")

    # --- Step 3: Validate a sample of augmented data ---
    print(f"\n--- Validating Augmented Data Quality ---")
    validation_sample = random.sample(all_augmented, min(5, len(all_augmented)))

    quality_scores = []
    for aug in validation_sample:
        original = BASE_EXAMPLES[aug["original_index"]]
        print(f"\n  Validating: [{aug['code']}] {aug['note'][:60]}...")

        try:
            scores = validate_augmented_data(client, original, aug["note"])
            quality_scores.append(scores)
            print(f"    Accuracy: {scores.get('accuracy', '?')}/5")
            print(f"    Diversity: {scores.get('diversity', '?')}/5")
            print(f"    Realism: {scores.get('realism', '?')}/5")
            print(f"    Completeness: {scores.get('completeness', '?')}/5")
            if scores.get("issues"):
                print(f"    Issues: {scores['issues']}")
        except Exception as e:
            print(f"    Validation error: {e}")

    # --- Step 4: Quality summary ---
    if quality_scores:
        print(f"\n--- Quality Summary ({len(quality_scores)} samples validated) ---")
        for metric in ["accuracy", "diversity", "realism", "completeness"]:
            values = [s.get(metric, 0) for s in quality_scores]
            avg = sum(values) / len(values) if values else 0
            print(f"  {metric:15s}: {avg:.1f}/5")

    # --- Step 5: Length distribution ---
    print(f"\n--- Length Distribution ---")
    orig_lengths = [len(e["clinical_note"]) for e in BASE_EXAMPLES]
    aug_lengths = [a["augmented_len"] for a in augmentation_log]
    print(f"  Original avg length:  {sum(orig_lengths)/len(orig_lengths):.0f} chars")
    if aug_lengths:
        print(f"  Augmented avg length: {sum(aug_lengths)/len(aug_lengths):.0f} chars")

    # --- Step 6: Save combined dataset ---
    print(f"\n--- Saving Augmented Dataset ---")
    combined = []

    # Add originals
    for ex in BASE_EXAMPLES:
        combined.append(format_training_sample(
            ex["clinical_note"], ex["icd10_code"], ex["icd10_description"]
        ))

    # Add augmented
    for aug in all_augmented:
        combined.append(format_training_sample(
            aug["note"], aug["code"], aug["description"]
        ))

    output_path = os.path.join(os.path.dirname(__file__) or ".", "icd10_augmented.jsonl")
    with open(output_path, "w") as f:
        for sample in combined:
            f.write(json.dumps(sample) + "\n")

    print(f"  Saved {len(combined)} samples to: {output_path}")
    print(f"  Augmentation factor: {len(combined)/len(BASE_EXAMPLES):.1f}x")
    print(f"\n  ✓ Augmented dataset ready!")


if __name__ == "__main__":
    main()
