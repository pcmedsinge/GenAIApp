"""
Data Preparation for Fine-Tuning — Main Demo
=============================================
Demonstrates training data formats, quality filtering, synthetic data
generation, and dataset analysis for medical LLM fine-tuning.

Run:
    python main.py
"""

from openai import OpenAI
import json
import random
import os
from collections import Counter, defaultdict


# ============================================================
# DEMO 1: Instruction Format
# ============================================================
def demo_instruction_format():
    """Show system/user/assistant format for instruction tuning.
    Create example training samples for medical tasks."""

    print("\n" + "=" * 60)
    print("DEMO 1: Instruction Format for Medical Fine-Tuning")
    print("=" * 60)

    # --- The standard instruction-tuning format ---
    print("\n--- Standard Instruction Format (OpenAI Chat) ---")
    print("Each training example is a list of messages with roles:\n")

    example_1 = {
        "messages": [
            {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
            {"role": "user", "content": "Patient is a 67-year-old male presenting with substernal chest pain radiating to left arm, diaphoresis, and shortness of breath. ECG shows ST elevation in leads II, III, aVF. Troponin elevated at 2.4 ng/mL."},
            {"role": "assistant", "content": "ICD-10: I21.19 - ST elevation (STEMI) myocardial infarction involving other coronary artery of inferior wall"}
        ]
    }

    print(json.dumps(example_1, indent=2))

    # --- Create diverse medical training samples ---
    print("\n--- Generating Medical Training Samples ---")

    training_samples = [
        {
            "messages": [
                {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
                {"role": "user", "content": "45-year-old female with progressive fatigue, weight gain of 15 lbs over 3 months, cold intolerance, and constipation. TSH elevated at 12.5 mIU/L, free T4 low at 0.6 ng/dL."},
                {"role": "assistant", "content": "ICD-10: E03.9 - Hypothyroidism, unspecified"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
                {"role": "user", "content": "8-year-old child brought in with acute onset wheezing, dyspnea, and chest tightness after playing outside. O2 sat 93%. Bilateral expiratory wheezes on auscultation. History of similar episodes."},
                {"role": "assistant", "content": "ICD-10: J45.31 - Mild persistent asthma with acute exacerbation"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
                {"role": "user", "content": "72-year-old male with sudden onset right-sided weakness, facial droop, and slurred speech. Symptoms began 2 hours ago. CT head negative for hemorrhage. NIHSS score 14."},
                {"role": "assistant", "content": "ICD-10: I63.9 - Cerebral infarction, unspecified"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a triage nurse assistant. Classify the urgency of the patient presentation."},
                {"role": "user", "content": "32-year-old female with severe headache, stiff neck, fever of 103.2F, and photophobia. Onset 6 hours ago. No recent trauma."},
                {"role": "assistant", "content": "URGENCY: Emergency (ESI Level 1)\nREASON: Presentation highly suspicious for meningitis — severe headache with nuchal rigidity, high fever, and photophobia constitute a medical emergency requiring immediate evaluation, lumbar puncture, and empiric antibiotics."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a clinical decision support assistant. Suggest relevant differential diagnoses."},
                {"role": "user", "content": "55-year-old male with epigastric pain radiating to the back, nausea, vomiting. History of heavy alcohol use. Lipase elevated at 1200 U/L."},
                {"role": "assistant", "content": "Primary diagnosis: Acute pancreatitis (K85.0 - Idiopathic acute pancreatitis, likely alcohol-induced K85.20)\n\nDifferential diagnoses:\n1. Peptic ulcer perforation (K25.5)\n2. Acute cholecystitis (K81.0)\n3. Mesenteric ischemia (K55.0)\n4. Aortic dissection (I71.0) — less likely but critical to exclude"}
            ]
        },
    ]

    print(f"Created {len(training_samples)} training samples\n")
    for i, sample in enumerate(training_samples):
        task = sample["messages"][0]["content"][:60]
        user_preview = sample["messages"][1]["content"][:60]
        print(f"  Sample {i + 1}: [{task}...]")
        print(f"            Input: {user_preview}...")
        print()

    # --- Show JSONL format ---
    print("--- JSONL Output Format (one JSON per line) ---")
    print("This is what training files look like:\n")
    for sample in training_samples[:3]:
        print(json.dumps(sample))
    print("...")

    print(f"\nTotal samples shown: {len(training_samples)}")
    print("In practice, you need 500–5000+ examples for good fine-tuning.")


# ============================================================
# DEMO 2: Data Quality
# ============================================================
def demo_data_quality():
    """Filter and clean training data: remove duplicates, fix formatting,
    validate completeness, check for bias."""

    print("\n" + "=" * 60)
    print("DEMO 2: Data Quality Filtering Pipeline")
    print("=" * 60)

    # --- Sample noisy dataset ---
    raw_data = [
        {"messages": [{"role": "system", "content": "Medical coder."}, {"role": "user", "content": "Chest pain"}, {"role": "assistant", "content": "I20.0"}]},
        {"messages": [{"role": "system", "content": "Medical coder."}, {"role": "user", "content": "Chest pain"}, {"role": "assistant", "content": "I20.0"}]},  # Duplicate
        {"messages": [{"role": "user", "content": "Patient has diabetes"}, {"role": "assistant", "content": "E11.9"}]},  # Missing system
        {"messages": [{"role": "system", "content": "Medical coder."}, {"role": "user", "content": ""}, {"role": "assistant", "content": "J45.0"}]},  # Empty input
        {"messages": [{"role": "system", "content": "Medical coder."}, {"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]},  # Too short
        {"messages": [{"role": "system", "content": "You are a medical coding assistant."}, {"role": "user", "content": "67-year-old female with type 2 diabetes mellitus, HbA1c 8.2%, on metformin 1000mg BID. Presenting for routine follow-up. Fasting glucose 156 mg/dL."}, {"role": "assistant", "content": "ICD-10: E11.65 - Type 2 diabetes mellitus with hyperglycemia"}]},
        {"messages": [{"role": "system", "content": "You are a medical coding assistant."}, {"role": "user", "content": "55-year-old male with acute onset right lower quadrant pain, fever 101.3F, positive McBurney's point tenderness, elevated WBC at 14,200. CT shows appendiceal inflammation."}, {"role": "assistant", "content": "ICD-10: K35.80 - Unspecified acute appendicitis without abscess"}]},
        {"messages": [{"role": "system", "content": "You are a medical coding assistant."}, {"role": "user", "content": "x" * 5000}, {"role": "assistant", "content": "ICD-10: Z99.9"}]},  # Too long / garbage
        {"messages": [{"role": "system", "content": "You are a medical coding assistant."}, {"role": "user", "content": "34-year-old female with recurrent urinary tract infections, dysuria, frequency, and suprapubic tenderness. Urine culture positive for E. coli > 100,000 CFU/mL."}, {"role": "assistant", "content": "ICD-10: N39.0 - Urinary tract infection, site not specified"}]},
    ]

    print(f"\nRaw dataset: {len(raw_data)} samples")

    # --- Filter 1: Remove exact duplicates ---
    print("\n--- Filter 1: Remove Exact Duplicates ---")
    seen_hashes = set()
    deduped = []
    duplicates_removed = 0
    for sample in raw_data:
        h = json.dumps(sample, sort_keys=True)
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(sample)
        else:
            duplicates_removed += 1
    print(f"  Removed {duplicates_removed} duplicates → {len(deduped)} remaining")

    # --- Filter 2: Validate completeness ---
    print("\n--- Filter 2: Validate Completeness ---")
    complete = []
    incomplete = 0
    for sample in deduped:
        msgs = sample.get("messages", [])
        roles = [m["role"] for m in msgs]
        if "system" in roles and "user" in roles and "assistant" in roles:
            complete.append(sample)
        else:
            incomplete += 1
            print(f"  REMOVED (missing role): roles={roles}")
    print(f"  Removed {incomplete} incomplete → {len(complete)} remaining")

    # --- Filter 3: Check content length ---
    print("\n--- Filter 3: Check Content Length ---")
    MIN_USER_CHARS = 20
    MAX_USER_CHARS = 2000
    MIN_ASST_CHARS = 5
    length_ok = []
    length_bad = 0
    for sample in complete:
        user_msg = next(m for m in sample["messages"] if m["role"] == "user")
        asst_msg = next(m for m in sample["messages"] if m["role"] == "assistant")
        user_len = len(user_msg["content"])
        asst_len = len(asst_msg["content"])
        if MIN_USER_CHARS <= user_len <= MAX_USER_CHARS and asst_len >= MIN_ASST_CHARS:
            length_ok.append(sample)
        else:
            length_bad += 1
            print(f"  REMOVED: user_len={user_len}, asst_len={asst_len}")
    print(f"  Removed {length_bad} bad-length → {len(length_ok)} remaining")

    # --- Filter 4: Check for empty/whitespace content ---
    print("\n--- Filter 4: Check for Empty Content ---")
    content_ok = []
    empty_count = 0
    for sample in length_ok:
        all_filled = all(m["content"].strip() for m in sample["messages"])
        if all_filled:
            content_ok.append(sample)
        else:
            empty_count += 1
    print(f"  Removed {empty_count} empty → {len(content_ok)} remaining")

    # --- Summary ---
    print("\n--- Quality Filter Summary ---")
    print(f"  Raw:        {len(raw_data)}")
    print(f"  After dedup: {len(deduped)}")
    print(f"  After completeness: {len(complete)}")
    print(f"  After length check: {len(length_ok)}")
    print(f"  After empty check:  {len(content_ok)}")
    print(f"  Final clean dataset: {len(content_ok)} samples ({len(content_ok)/len(raw_data)*100:.0f}% kept)")

    # --- Label distribution check ---
    print("\n--- Label Distribution Check ---")
    codes = []
    for sample in content_ok:
        asst = next(m for m in sample["messages"] if m["role"] == "assistant")
        text = asst["content"]
        if "ICD-10:" in text:
            code = text.split("ICD-10:")[1].strip().split(" ")[0].rstrip("-")
            codes.append(code)
    dist = Counter(codes)
    print(f"  Unique ICD-10 codes: {len(dist)}")
    for code, count in dist.most_common(10):
        print(f"    {code}: {count}")


# ============================================================
# DEMO 3: Synthetic Data Generation
# ============================================================
def demo_synthetic_data():
    """Use GPT-4o to generate synthetic training data for medical tasks."""

    print("\n" + "=" * 60)
    print("DEMO 3: Synthetic Data Generation with GPT-4o")
    print("=" * 60)

    client = OpenAI()

    # --- Prompt for generating training data ---
    generation_prompt = """Generate 5 realistic clinical note snippets paired with ICD-10 codes.

Each example must:
1. Be a different medical specialty (cardiology, pulmonology, endocrinology, neurology, gastroenterology)
2. Include patient age, sex, symptoms, vitals/labs where relevant
3. Be 2-4 sentences long
4. Map to a specific ICD-10 code (not unspecified when possible)

Output as JSON array with objects having "clinical_note" and "icd10_code" and "icd10_description" fields.
Return ONLY the JSON array, no markdown fences."""

    print("\n--- Generating synthetic training data ---")
    print(f"Prompt (truncated): {generation_prompt[:100]}...\n")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a medical data generation assistant. Generate realistic but synthetic clinical data for training AI models. All data must be fictional."},
            {"role": "user", "content": generation_prompt}
        ],
        temperature=0.8,
    )

    raw_output = response.choices[0].message.content.strip()
    if raw_output.startswith("```"):
        raw_output = raw_output.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        synthetic_data = json.loads(raw_output)
    except json.JSONDecodeError:
        print("  Warning: Could not parse JSON response. Raw output:")
        print(raw_output[:500])
        synthetic_data = []

    print(f"Generated {len(synthetic_data)} synthetic examples:\n")
    for i, item in enumerate(synthetic_data):
        note = item.get("clinical_note", "N/A")
        code = item.get("icd10_code", "N/A")
        desc = item.get("icd10_description", "N/A")
        print(f"  Example {i + 1}:")
        print(f"    Note: {note[:120]}...")
        print(f"    Code: {code} — {desc}")
        print()

    # --- Convert to training format ---
    print("--- Converting to Training Format ---")
    training_samples = []
    for item in synthetic_data:
        sample = {
            "messages": [
                {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
                {"role": "user", "content": item.get("clinical_note", "")},
                {"role": "assistant", "content": f"ICD-10: {item.get('icd10_code', '')} - {item.get('icd10_description', '')}"}
            ]
        }
        training_samples.append(sample)

    print(f"  Converted {len(training_samples)} samples to instruction format")
    print(f"\n  Sample JSONL line:")
    if training_samples:
        print(f"  {json.dumps(training_samples[0])}")

    # --- Quality validation of generated data ---
    print("\n--- Validating Generated Data ---")
    valid = 0
    issues = []
    for i, sample in enumerate(training_samples):
        user_msg = next((m for m in sample["messages"] if m["role"] == "user"), None)
        asst_msg = next((m for m in sample["messages"] if m["role"] == "assistant"), None)
        if not user_msg or len(user_msg["content"]) < 20:
            issues.append(f"  Sample {i + 1}: user content too short")
        elif not asst_msg or "ICD-10:" not in asst_msg["content"]:
            issues.append(f"  Sample {i + 1}: missing ICD-10 code in response")
        else:
            valid += 1

    print(f"  Valid: {valid}/{len(training_samples)}")
    for issue in issues:
        print(issue)

    print(f"\nTokens used: {response.usage.total_tokens}")


# ============================================================
# DEMO 4: Dataset Statistics
# ============================================================
def demo_dataset_statistics():
    """Analyze a training dataset: distribution of labels, token lengths,
    quality scores."""

    print("\n" + "=" * 60)
    print("DEMO 4: Dataset Statistics & Analysis")
    print("=" * 60)

    # --- Build a sample dataset for analysis ---
    random.seed(42)

    specialties = {
        "cardiology": ["I20.0", "I21.0", "I25.10", "I48.0", "I50.9"],
        "endocrinology": ["E11.9", "E03.9", "E05.90", "E66.01", "E78.5"],
        "pulmonology": ["J45.20", "J18.9", "J44.1", "J96.00", "J06.9"],
        "neurology": ["G43.909", "I63.9", "G40.909", "G20", "G35"],
        "gastroenterology": ["K21.0", "K50.90", "K80.20", "K25.9", "K85.9"],
        "orthopedics": ["M54.5", "M17.11", "S72.001A", "M79.3", "M25.511"],
    }

    dataset = []
    for specialty, codes in specialties.items():
        n = random.randint(15, 50)
        for _ in range(n):
            code = random.choice(codes)
            note_len = random.randint(80, 400)
            note = f"[Simulated {specialty} note, {note_len} chars]" + "x" * max(0, note_len - 40)
            dataset.append({
                "specialty": specialty,
                "icd10_code": code,
                "note_length": note_len,
                "messages": [
                    {"role": "system", "content": "You are a medical coding assistant."},
                    {"role": "user", "content": note},
                    {"role": "assistant", "content": f"ICD-10: {code}"}
                ]
            })

    print(f"\nDataset size: {len(dataset)} samples\n")

    # --- Specialty distribution ---
    print("--- Specialty Distribution ---")
    spec_counts = Counter(d["specialty"] for d in dataset)
    max_count = max(spec_counts.values())
    for spec, count in spec_counts.most_common():
        bar = "█" * int(count / max_count * 30)
        print(f"  {spec:20s} {count:3d}  {bar}")

    # --- ICD-10 code distribution ---
    print("\n--- ICD-10 Code Distribution (top 15) ---")
    code_counts = Counter(d["icd10_code"] for d in dataset)
    max_c = max(code_counts.values())
    for code, count in code_counts.most_common(15):
        bar = "█" * int(count / max_c * 25)
        print(f"  {code:12s} {count:3d}  {bar}")
    print(f"  ... {len(code_counts)} unique codes total")

    # --- Note length distribution ---
    print("\n--- Note Length Distribution ---")
    lengths = [d["note_length"] for d in dataset]
    min_l, max_l = min(lengths), max(lengths)
    avg_l = sum(lengths) / len(lengths)
    median_l = sorted(lengths)[len(lengths) // 2]
    print(f"  Min: {min_l}  Max: {max_l}  Mean: {avg_l:.0f}  Median: {median_l}")

    # Histogram
    buckets = [0, 100, 150, 200, 250, 300, 350, 400, 500]
    print("\n  Length histogram:")
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        cnt = sum(1 for l in lengths if lo <= l < hi)
        bar = "█" * int(cnt / len(lengths) * 40)
        print(f"    {lo:3d}-{hi:3d}  {cnt:3d}  {bar}")

    # --- Class balance analysis ---
    print("\n--- Class Balance Analysis ---")
    code_list = list(code_counts.values())
    min_class = min(code_list)
    max_class = max(code_list)
    imbalance_ratio = max_class / min_class if min_class > 0 else float("inf")
    print(f"  Smallest class: {min_class} samples")
    print(f"  Largest class:  {max_class} samples")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}x")
    if imbalance_ratio > 5:
        print("  ⚠ High imbalance — consider oversampling rare classes")
    elif imbalance_ratio > 2:
        print("  ⚠ Moderate imbalance — monitor per-class metrics")
    else:
        print("  ✓ Reasonably balanced dataset")

    # --- Estimated token counts ---
    print("\n--- Estimated Token Counts ---")
    total_chars = sum(
        sum(len(m["content"]) for m in d["messages"])
        for d in dataset
    )
    est_tokens = total_chars / 4  # rough char-to-token ratio
    print(f"  Total characters: {total_chars:,}")
    print(f"  Estimated tokens: {est_tokens:,.0f}")
    print(f"  Avg tokens/sample: {est_tokens / len(dataset):.0f}")

    # --- Data quality summary ---
    print("\n--- Dataset Summary ---")
    print(f"  Samples:      {len(dataset)}")
    print(f"  Specialties:  {len(spec_counts)}")
    print(f"  Unique codes: {len(code_counts)}")
    print(f"  Est. tokens:  {est_tokens:,.0f}")
    print(f"  Ready for fine-tuning: {'Yes' if len(dataset) >= 100 else 'Need more data'}")


# ============================================================
# Main Menu
# ============================================================
def main():
    """Interactive demo menu for Data Preparation."""
    demos = {
        "1": ("Instruction Format", demo_instruction_format),
        "2": ("Data Quality Filtering", demo_data_quality),
        "3": ("Synthetic Data Generation (requires API key)", demo_synthetic_data),
        "4": ("Dataset Statistics & Analysis", demo_dataset_statistics),
    }

    while True:
        print("\n" + "=" * 60)
        print("DATA PREPARATION FOR FINE-TUNING")
        print("=" * 60)
        for key, (name, _) in demos.items():
            print(f"  {key}. {name}")
        print("  q. Quit")

        choice = input("\nSelect demo: ").strip().lower()
        if choice == "q":
            print("Goodbye!")
            break
        elif choice in demos:
            demos[choice][1]()
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
