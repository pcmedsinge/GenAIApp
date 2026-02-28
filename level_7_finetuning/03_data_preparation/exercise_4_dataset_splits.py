"""
Exercise 4: Dataset Splitting and JSONL Export
==============================================
Split dataset into train/validation/test sets. Ensure balanced distribution
across splits. Analyze each split. Save as JSONL format.

Learning Objectives:
- Properly split medical datasets
- Ensure stratified/balanced splits
- Analyze and compare split distributions
- Export in JSONL format for training

Run:
    python exercise_4_dataset_splits.py
"""

import json
import os
import random
from collections import Counter, defaultdict


# --- Sample ICD-10 dataset ---
def create_sample_dataset(size: int = 200) -> list:
    """Create a sample ICD-10 dataset with realistic distribution."""

    random.seed(42)

    codes = {
        "I21.0": ("Acute STEMI anterior wall", "cardiology", 0.08),
        "I50.9": ("Heart failure, unspecified", "cardiology", 0.10),
        "I48.0": ("Paroxysmal atrial fibrillation", "cardiology", 0.06),
        "E11.9": ("Type 2 diabetes without complications", "endocrinology", 0.12),
        "E11.65": ("Type 2 diabetes with hyperglycemia", "endocrinology", 0.07),
        "E03.9": ("Hypothyroidism, unspecified", "endocrinology", 0.05),
        "J18.9": ("Pneumonia, unspecified organism", "pulmonology", 0.09),
        "J44.1": ("COPD with acute exacerbation", "pulmonology", 0.08),
        "J45.41": ("Moderate persistent asthma, exacerbation", "pulmonology", 0.05),
        "K35.80": ("Acute appendicitis", "gastroenterology", 0.04),
        "K85.9": ("Acute pancreatitis", "gastroenterology", 0.04),
        "I63.9": ("Cerebral infarction", "neurology", 0.06),
        "G43.909": ("Migraine, unspecified", "neurology", 0.05),
        "N39.0": ("Urinary tract infection", "urology", 0.07),
        "M54.5": ("Low back pain", "orthopedics", 0.04),
    }

    # Generate samples based on distribution
    dataset = []
    for code, (description, specialty, freq) in codes.items():
        n = max(2, int(size * freq))
        for i in range(n):
            age = random.randint(18, 92)
            sex = random.choice(["male", "female"])
            note_len = random.randint(100, 350)
            note = f"{age}-year-old {sex} presenting with symptoms consistent with {description.lower()}. " + \
                   f"[Simulated clinical details for {specialty}]" + " " * max(0, note_len - 100)

            sample = {
                "messages": [
                    {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
                    {"role": "user", "content": note.strip()},
                    {"role": "assistant", "content": f"ICD-10: {code} - {description}"}
                ],
                "_metadata": {
                    "icd10_code": code,
                    "specialty": specialty,
                    "description": description,
                }
            }
            dataset.append(sample)

    random.shuffle(dataset)
    return dataset


def extract_code(sample: dict) -> str:
    """Extract ICD-10 code from a sample."""
    asst = next(m for m in sample["messages"] if m["role"] == "assistant")
    text = asst["content"]
    if "ICD-10:" in text:
        return text.split("ICD-10:")[1].strip().split(" - ")[0].strip()
    return "UNKNOWN"


def extract_specialty(sample: dict) -> str:
    """Extract specialty from metadata."""
    return sample.get("_metadata", {}).get("specialty", "unknown")


def stratified_split(data: list, train_ratio: float = 0.7,
                     val_ratio: float = 0.15, test_ratio: float = 0.15,
                     seed: int = 42) -> tuple:
    """Split data with stratification by ICD-10 code."""

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
        "Ratios must sum to 1.0"

    random.seed(seed)

    # Group by ICD-10 code
    by_code = defaultdict(list)
    for sample in data:
        code = extract_code(sample)
        by_code[code].append(sample)

    train, val, test = [], [], []

    for code, samples in by_code.items():
        random.shuffle(samples)
        n = len(samples)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        # Ensure at least 1 sample in each split if possible
        if n >= 3:
            n_test = n - n_train - n_val
            if n_test < 1:
                n_val = max(1, n_val - 1)
                n_test = n - n_train - n_val
        else:
            # For very small classes, put everything in train
            n_train = n
            n_val = 0
            n_test = 0

        train.extend(samples[:n_train])
        val.extend(samples[n_train:n_train + n_val])
        test.extend(samples[n_train + n_val:])

    # Shuffle each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def analyze_split(name: str, data: list):
    """Print detailed analysis of a dataset split."""

    print(f"\n  === {name.upper()} Split: {len(data)} samples ===")

    if not data:
        print("    (empty)")
        return

    # Code distribution
    code_counts = Counter(extract_code(s) for s in data)
    print(f"    Unique codes: {len(code_counts)}")
    print(f"\n    {'Code':12s} {'Count':>6s} {'Pct':>6s}  Distribution")
    print("    " + "-" * 50)
    for code, count in code_counts.most_common():
        pct = count / len(data) * 100
        bar = "█" * int(pct)
        print(f"    {code:12s} {count:6d} {pct:5.1f}%  {bar}")

    # Specialty distribution
    spec_counts = Counter(extract_specialty(s) for s in data)
    print(f"\n    {'Specialty':18s} {'Count':>6s} {'Pct':>6s}")
    print("    " + "-" * 35)
    for spec, count in spec_counts.most_common():
        pct = count / len(data) * 100
        print(f"    {spec:18s} {count:6d} {pct:5.1f}%")

    # Content length stats
    user_lengths = []
    for s in data:
        user_msg = next(m for m in s["messages"] if m["role"] == "user")
        user_lengths.append(len(user_msg["content"]))
    avg_len = sum(user_lengths) / len(user_lengths)
    min_len = min(user_lengths)
    max_len = max(user_lengths)
    print(f"\n    Note length: min={min_len}, max={max_len}, avg={avg_len:.0f}")


def compare_distributions(train: list, val: list, test: list):
    """Compare label distributions across splits to check balance."""

    print("\n--- Distribution Comparison Across Splits ---")

    train_codes = Counter(extract_code(s) for s in train)
    val_codes = Counter(extract_code(s) for s in val)
    test_codes = Counter(extract_code(s) for s in test)

    all_codes = sorted(set(train_codes.keys()) | set(val_codes.keys()) | set(test_codes.keys()))

    print(f"\n  {'Code':12s}  {'Train':>6s} {'Train%':>7s}  {'Val':>5s} {'Val%':>6s}  {'Test':>5s} {'Test%':>6s}  Drift")
    print("  " + "-" * 75)

    max_drift = 0
    for code in all_codes:
        tc = train_codes.get(code, 0)
        vc = val_codes.get(code, 0)
        tsc = test_codes.get(code, 0)

        tp = tc / len(train) * 100 if train else 0
        vp = vc / len(val) * 100 if val else 0
        tsp = tsc / len(test) * 100 if test else 0

        # Drift = max deviation between any two splits
        drift = max(abs(tp - vp), abs(tp - tsp), abs(vp - tsp))
        max_drift = max(max_drift, drift)
        drift_flag = " ⚠" if drift > 5 else ""

        print(f"  {code:12s}  {tc:6d} {tp:6.1f}%  {vc:5d} {vp:5.1f}%  {tsc:5d} {tsp:5.1f}%  {drift:4.1f}%{drift_flag}")

    print(f"\n  Max distribution drift: {max_drift:.1f}%")
    if max_drift > 5:
        print("  ⚠ Some codes have significant drift — consider re-stratifying")
    else:
        print("  ✓ Distributions well-balanced across splits")


def save_split(data: list, filename: str) -> str:
    """Save a split as JSONL, removing metadata."""
    output_path = os.path.join(os.path.dirname(__file__) or ".", filename)
    with open(output_path, "w") as f:
        for sample in data:
            # Remove internal metadata before saving
            clean = {"messages": sample["messages"]}
            f.write(json.dumps(clean) + "\n")
    return output_path


def main():
    """Split dataset and analyze distributions."""

    print("=" * 60)
    print("Exercise 4: Dataset Splitting & JSONL Export")
    print("=" * 60)

    # --- Step 1: Create or load dataset ---
    print("\n--- Step 1: Creating Sample Dataset ---")
    dataset = create_sample_dataset(size=200)
    print(f"  Created {len(dataset)} samples")

    # Show overall distribution
    code_counts = Counter(extract_code(s) for s in dataset)
    print(f"  Unique codes: {len(code_counts)}")
    print(f"  Most common: {code_counts.most_common(3)}")

    # --- Step 2: Stratified split ---
    print("\n--- Step 2: Stratified Split (70/15/15) ---")
    train, val, test = stratified_split(
        dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )
    print(f"  Train: {len(train)} samples")
    print(f"  Val:   {len(val)} samples")
    print(f"  Test:  {len(test)} samples")
    print(f"  Total: {len(train) + len(val) + len(test)} samples")

    # --- Step 3: Analyze each split ---
    print("\n--- Step 3: Split Analysis ---")
    analyze_split("Train", train)
    analyze_split("Validation", val)
    analyze_split("Test", test)

    # --- Step 4: Compare distributions ---
    compare_distributions(train, val, test)

    # --- Step 5: Save splits as JSONL ---
    print("\n--- Step 5: Saving Splits ---")
    files = {
        "icd10_train.jsonl": train,
        "icd10_val.jsonl": val,
        "icd10_test.jsonl": test,
    }

    for filename, data in files.items():
        path = save_split(data, filename)
        size = os.path.getsize(path)
        print(f"  {filename:25s} {len(data):4d} samples  {size:>8,} bytes")

    # --- Summary ---
    print("\n--- Split Summary ---")
    print(f"  Total samples:  {len(dataset)}")
    print(f"  Train:          {len(train)} ({len(train)/len(dataset)*100:.0f}%)")
    print(f"  Validation:     {len(val)} ({len(val)/len(dataset)*100:.0f}%)")
    print(f"  Test:           {len(test)} ({len(test)/len(dataset)*100:.0f}%)")
    print(f"  Unique codes:   {len(code_counts)}")
    print(f"\n  Files saved:")
    print(f"    icd10_train.jsonl — for model training")
    print(f"    icd10_val.jsonl   — for monitoring during training")
    print(f"    icd10_test.jsonl  — for final evaluation (never peek!)")
    print(f"\n  ✓ Dataset splits ready for fine-tuning!")


if __name__ == "__main__":
    main()
