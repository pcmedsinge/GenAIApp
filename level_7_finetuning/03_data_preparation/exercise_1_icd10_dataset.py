"""
Exercise 1: ICD-10 Training Dataset Creation
=============================================
Create training data for ICD-10 coding. Generate clinical note → ICD-10 code
pairs covering 20+ examples across multiple medical specialties.

Learning Objectives:
- Understand instruction-tuning data format
- Generate diverse medical training examples
- Use GPT-4o for synthetic data creation
- Validate and save data in JSONL format

Run:
    python exercise_1_icd10_dataset.py
"""

from openai import OpenAI
import json
import os
import random
from collections import Counter


# --- ICD-10 codes by specialty ---
ICD10_BY_SPECIALTY = {
    "cardiology": {
        "I21.0": "Acute ST elevation myocardial infarction of anterior wall",
        "I50.9": "Heart failure, unspecified",
        "I48.0": "Paroxysmal atrial fibrillation",
        "I10": "Essential (primary) hypertension",
        "I25.10": "Atherosclerotic heart disease of native coronary artery",
    },
    "pulmonology": {
        "J18.9": "Pneumonia, unspecified organism",
        "J44.1": "COPD with acute exacerbation",
        "J45.41": "Moderate persistent asthma with acute exacerbation",
        "J96.00": "Acute respiratory failure, unspecified",
        "J06.9": "Acute upper respiratory infection, unspecified",
    },
    "endocrinology": {
        "E11.9": "Type 2 diabetes mellitus without complications",
        "E11.65": "Type 2 diabetes mellitus with hyperglycemia",
        "E03.9": "Hypothyroidism, unspecified",
        "E05.90": "Thyrotoxicosis, unspecified without thyrotoxic crisis",
        "E66.01": "Morbid (severe) obesity due to excess calories",
    },
    "gastroenterology": {
        "K21.0": "Gastro-esophageal reflux disease with esophagitis",
        "K35.80": "Unspecified acute appendicitis without abscess",
        "K85.9": "Acute pancreatitis, unspecified",
        "K80.20": "Calculus of gallbladder without obstruction",
        "K50.90": "Crohn's disease, unspecified, without complications",
    },
    "neurology": {
        "I63.9": "Cerebral infarction, unspecified",
        "G43.909": "Migraine, unspecified, not intractable",
        "G40.909": "Epilepsy, unspecified, not intractable",
        "G20": "Parkinson's disease",
        "G35": "Multiple sclerosis",
    },
}


def generate_training_data_with_llm(client: OpenAI, num_per_code: int = 2) -> list:
    """Use GPT-4o to generate clinical note → ICD-10 pairs."""

    all_samples = []

    for specialty, codes in ICD10_BY_SPECIALTY.items():
        print(f"\n--- Generating {specialty} samples ---")

        for code, description in codes.items():
            prompt = f"""Generate {num_per_code} realistic synthetic clinical note snippets for:
ICD-10: {code} - {description}
Specialty: {specialty}

Each note must:
1. Be 2-5 sentences long
2. Include patient age and sex
3. Include relevant symptoms, vitals, or lab values
4. Be clinically realistic but entirely fictional
5. Be distinct from each other

Return ONLY a JSON array of objects with a "clinical_note" field. No markdown fences."""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a medical data generation expert. Create realistic synthetic clinical notes for AI training. All data is fictional."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.9,
                )

                raw = response.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

                notes = json.loads(raw)

                for note_obj in notes:
                    sample = {
                        "messages": [
                            {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
                            {"role": "user", "content": note_obj["clinical_note"]},
                            {"role": "assistant", "content": f"ICD-10: {code} - {description}"}
                        ],
                        "metadata": {
                            "specialty": specialty,
                            "icd10_code": code,
                        }
                    }
                    all_samples.append(sample)

                print(f"  ✓ {code} ({description[:40]}): {len(notes)} samples")

            except Exception as e:
                print(f"  ✗ {code}: {e}")

    return all_samples


def generate_manual_samples() -> list:
    """Create hand-crafted training samples as a baseline."""

    samples = [
        # Cardiology
        ("67-year-old male presenting with substernal chest pain radiating to left arm, diaphoresis, and dyspnea. ECG shows ST elevation in leads V1-V4. Troponin I elevated at 3.2 ng/mL.",
         "I21.0", "Acute ST elevation myocardial infarction of anterior wall"),
        ("78-year-old female with progressive dyspnea on exertion, bilateral lower extremity edema, and orthopnea. BNP elevated at 1450 pg/mL. Ejection fraction 25% on echocardiogram.",
         "I50.9", "Heart failure, unspecified"),
        ("62-year-old male presents with palpitations and irregular pulse. ECG reveals irregularly irregular rhythm with absent P waves and ventricular rate of 142 bpm.",
         "I48.0", "Paroxysmal atrial fibrillation"),

        # Pulmonology
        ("73-year-old female with productive cough, fever 101.8F, and right lower lobe crackles on auscultation. Chest X-ray shows right lower lobe consolidation. WBC 14,500.",
         "J18.9", "Pneumonia, unspecified organism"),
        ("68-year-old male with known COPD presenting with worsening dyspnea over 3 days, increased sputum production, and wheezing. O2 sat 88% on room air. Using accessory muscles.",
         "J44.1", "COPD with acute exacerbation"),

        # Endocrinology
        ("55-year-old obese female with polyuria, polydipsia, and blurred vision. Fasting glucose 245 mg/dL, HbA1c 9.1%. No prior diabetes diagnosis.",
         "E11.65", "Type 2 diabetes mellitus with hyperglycemia"),
        ("42-year-old female with fatigue, weight gain of 20 lbs, cold intolerance, and constipation. TSH 18.2 mIU/L, free T4 0.4 ng/dL.",
         "E03.9", "Hypothyroidism, unspecified"),

        # Gastroenterology
        ("29-year-old male with acute onset right lower quadrant pain, nausea, anorexia, and fever 100.9F. Positive McBurney's point tenderness, rebound guarding. WBC 15,800.",
         "K35.80", "Unspecified acute appendicitis without abscess"),
        ("51-year-old male with severe epigastric pain radiating to back, nausea, persistent vomiting. Heavy alcohol history. Lipase 1,850 U/L, amylase 920 U/L.",
         "K85.9", "Acute pancreatitis, unspecified"),

        # Neurology
        ("74-year-old female with sudden onset right-sided hemiparesis, facial droop, and expressive aphasia. Symptoms began 90 minutes ago. CT head negative for hemorrhage. NIHSS 16.",
         "I63.9", "Cerebral infarction, unspecified"),
        ("28-year-old female with recurrent severe unilateral throbbing headache, photophobia, phonophobia, and nausea. Lasting 12-24 hours. Normal neurological exam.",
         "G43.909", "Migraine, unspecified, not intractable"),

        # Additional variety
        ("3-year-old male brought by mother with high-pitched barking cough, inspiratory stridor, and low-grade fever. Symptoms worsened overnight. Neck X-ray shows steeple sign.",
         "J05.0", "Acute obstructive laryngitis [croup]"),
        ("82-year-old female found confused with T 103.1F, HR 112, BP 85/52, RR 24. Urine cloudy with positive nitrites and leukocyte esterase. Lactate 3.8 mmol/L.",
         "A41.9", "Sepsis, unspecified organism"),
        ("45-year-old construction worker with acute low back pain after lifting heavy object. Pain radiates to right buttock. No neurological deficits. Straight leg raise negative.",
         "M54.5", "Low back pain"),
        ("19-year-old male soccer player with acute right knee pain and swelling after twisting injury. Positive Lachman test, positive anterior drawer. MRI pending.",
         "S83.511A", "Sprain of anterior cruciate ligament of right knee, initial encounter"),
        ("58-year-old female with burning epigastric pain worse after meals, acid regurgitation, and chronic cough. Symptoms present for 6 months. On OTC antacids with partial relief.",
         "K21.0", "Gastro-esophageal reflux disease with esophagitis"),
        ("36-year-old female with recurrent bloody diarrhea, abdominal cramping, weight loss of 12 lbs over 2 months. Colonoscopy shows transmural inflammation with skip lesions.",
         "K50.90", "Crohn's disease, unspecified, without complications"),
        ("70-year-old male with resting tremor of right hand, bradykinesia, cogwheel rigidity, and shuffling gait. Symptoms progressive over 2 years.",
         "G20", "Parkinson's disease"),
        ("31-year-old female with episodes of blurred vision, right arm numbness, and fatigue over past year. MRI brain shows multiple periventricular white matter lesions. CSF shows oligoclonal bands.",
         "G35", "Multiple sclerosis"),
        ("89-year-old male with acute oliguria, BUN 68, creatinine 4.2 (baseline 1.1). Urine output 150 mL in 12 hours. Recent course of NSAIDs for back pain.",
         "N17.9", "Acute kidney failure, unspecified"),
        ("24-year-old female with dysuria, urinary frequency, urgency, and suprapubic discomfort. Urine dipstick positive for nitrites and leukocyte esterase. No fever.",
         "N39.0", "Urinary tract infection, site not specified"),
    ]

    formatted = []
    for note, code, description in samples:
        formatted.append({
            "messages": [
                {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
                {"role": "user", "content": note},
                {"role": "assistant", "content": f"ICD-10: {code} - {description}"}
            ],
            "metadata": {"icd10_code": code, "source": "manual"}
        })

    return formatted


def validate_dataset(samples: list) -> dict:
    """Validate that samples are well-formed."""

    stats = {"total": len(samples), "valid": 0, "issues": []}

    for i, sample in enumerate(samples):
        msgs = sample.get("messages", [])
        roles = [m["role"] for m in msgs]
        if set(roles) != {"system", "user", "assistant"}:
            stats["issues"].append(f"Sample {i}: missing role(s) — has {roles}")
            continue
        user_msg = next(m for m in msgs if m["role"] == "user")
        asst_msg = next(m for m in msgs if m["role"] == "assistant")
        if len(user_msg["content"]) < 20:
            stats["issues"].append(f"Sample {i}: user content too short ({len(user_msg['content'])} chars)")
            continue
        if "ICD-10:" not in asst_msg["content"]:
            stats["issues"].append(f"Sample {i}: assistant missing ICD-10 code")
            continue
        stats["valid"] += 1

    return stats


def save_dataset(samples: list, filename: str) -> str:
    """Save dataset as JSONL."""
    output_path = os.path.join(os.path.dirname(__file__) or ".", filename)
    with open(output_path, "w") as f:
        for sample in samples:
            # Remove metadata before saving training data
            train_sample = {"messages": sample["messages"]}
            f.write(json.dumps(train_sample) + "\n")
    return output_path


def main():
    """Create a comprehensive ICD-10 coding training dataset."""

    print("=" * 60)
    print("Exercise 1: ICD-10 Training Dataset Creation")
    print("=" * 60)

    # --- Step 1: Generate manual baseline samples ---
    print("\n--- Step 1: Creating Manual Baseline Samples ---")
    manual_samples = generate_manual_samples()
    print(f"  Created {len(manual_samples)} hand-crafted samples")

    # Validate
    stats = validate_dataset(manual_samples)
    print(f"  Valid: {stats['valid']}/{stats['total']}")
    for issue in stats["issues"][:5]:
        print(f"    Issue: {issue}")

    # --- Step 2: Generate synthetic data with GPT-4o ---
    print("\n--- Step 2: Generating Synthetic Data with GPT-4o ---")
    try:
        client = OpenAI()
        llm_samples = generate_training_data_with_llm(client, num_per_code=2)
        print(f"\n  Generated {len(llm_samples)} synthetic samples")

        stats2 = validate_dataset(llm_samples)
        print(f"  Valid: {stats2['valid']}/{stats2['total']}")
    except Exception as e:
        print(f"  Could not generate synthetic data: {e}")
        llm_samples = []

    # --- Step 3: Combine datasets ---
    print("\n--- Step 3: Combining Datasets ---")
    combined = manual_samples + llm_samples
    print(f"  Manual:    {len(manual_samples)} samples")
    print(f"  Synthetic: {len(llm_samples)} samples")
    print(f"  Combined:  {len(combined)} samples")

    # --- Step 4: Analyze distribution ---
    print("\n--- Step 4: Dataset Distribution ---")
    codes = []
    for s in combined:
        asst = next(m for m in s["messages"] if m["role"] == "assistant")
        code = asst["content"].split(":")[1].strip().split(" - ")[0].strip()
        codes.append(code)

    code_dist = Counter(codes)
    print(f"  Unique ICD-10 codes: {len(code_dist)}")
    print(f"\n  {'Code':12s} {'Count':>6s}")
    print("  " + "-" * 20)
    for code, count in code_dist.most_common():
        bar = "█" * count
        print(f"  {code:12s} {count:6d}  {bar}")

    # --- Step 5: Save dataset ---
    print("\n--- Step 5: Saving Dataset ---")
    path = save_dataset(combined, "icd10_training_full.jsonl")
    size = os.path.getsize(path)
    print(f"  Saved to: {path}")
    print(f"  File size: {size:,} bytes")
    print(f"  Total samples: {len(combined)}")
    print(f"\n  ✓ Dataset ready for fine-tuning!")


if __name__ == "__main__":
    main()
