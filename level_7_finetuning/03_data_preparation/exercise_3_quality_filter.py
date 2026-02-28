"""
Exercise 3: Data Quality Pipeline
==================================
Build a comprehensive data quality pipeline: check token length, validate
format, detect duplicates, score quality with LLM, filter below threshold.

Learning Objectives:
- Build a multi-stage quality pipeline
- Detect near-duplicate content
- Use LLM-as-judge for quality scoring
- Filter and report on data quality metrics

Run:
    python exercise_3_quality_filter.py
"""

from openai import OpenAI
import json
import os
import hashlib
import random
from collections import Counter


def create_noisy_dataset() -> list:
    """Create a dataset with intentional quality issues for filtering."""

    good_samples = [
        {"messages": [
            {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
            {"role": "user", "content": "67-year-old male presenting with substernal chest pain radiating to left arm, diaphoresis, and dyspnea. ECG shows ST elevation in leads V1-V4. Troponin I elevated at 3.2 ng/mL."},
            {"role": "assistant", "content": "ICD-10: I21.0 - Acute ST elevation myocardial infarction of anterior wall"}
        ]},
        {"messages": [
            {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
            {"role": "user", "content": "78-year-old female with progressive dyspnea on exertion, bilateral lower extremity edema, and orthopnea. BNP elevated at 1450 pg/mL. Echocardiogram shows ejection fraction 25%."},
            {"role": "assistant", "content": "ICD-10: I50.9 - Heart failure, unspecified"}
        ]},
        {"messages": [
            {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
            {"role": "user", "content": "55-year-old obese female with polyuria, polydipsia, and blurred vision over 3 weeks. Fasting glucose 245 mg/dL, HbA1c 9.1%. BMI 38.2."},
            {"role": "assistant", "content": "ICD-10: E11.65 - Type 2 diabetes mellitus with hyperglycemia"}
        ]},
        {"messages": [
            {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
            {"role": "user", "content": "42-year-old female with progressive fatigue, weight gain of 20 lbs over 4 months, cold intolerance, constipation, and dry skin. TSH 18.2 mIU/L, free T4 0.4 ng/dL."},
            {"role": "assistant", "content": "ICD-10: E03.9 - Hypothyroidism, unspecified"}
        ]},
        {"messages": [
            {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
            {"role": "user", "content": "73-year-old female with productive cough of green sputum, fever 101.8F, chills, and pleuritic chest pain. Right lower lobe crackles on auscultation. CXR shows RLL consolidation. WBC 14,500 with left shift."},
            {"role": "assistant", "content": "ICD-10: J18.9 - Pneumonia, unspecified organism"}
        ]},
        {"messages": [
            {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
            {"role": "user", "content": "29-year-old male with acute onset right lower quadrant pain, nausea, anorexia, and low-grade fever 100.9F. Positive McBurney's point tenderness, positive Rovsing sign, and rebound guarding. WBC 15,800."},
            {"role": "assistant", "content": "ICD-10: K35.80 - Unspecified acute appendicitis without abscess"}
        ]},
        {"messages": [
            {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
            {"role": "user", "content": "74-year-old female with sudden onset right-sided hemiparesis, right facial droop, and expressive aphasia beginning 90 minutes ago. CT head negative for hemorrhage. NIHSS score 16."},
            {"role": "assistant", "content": "ICD-10: I63.9 - Cerebral infarction, unspecified"}
        ]},
        {"messages": [
            {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
            {"role": "user", "content": "8-year-old male with acute onset wheezing, dyspnea, and chest tightness after playing outdoors. O2 sat 92% on room air. Bilateral expiratory wheezes. History of recurrent similar episodes."},
            {"role": "assistant", "content": "ICD-10: J45.41 - Moderate persistent asthma with acute exacerbation"}
        ]},
    ]

    # Add bad samples
    bad_samples = [
        # Exact duplicate
        good_samples[0].copy(),
        # Missing system message
        {"messages": [
            {"role": "user", "content": "Patient has diabetes."},
            {"role": "assistant", "content": "E11.9"}
        ]},
        # Empty user content
        {"messages": [
            {"role": "system", "content": "You are a medical coding assistant."},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "ICD-10: J45.0"}
        ]},
        # Too short (low signal)
        {"messages": [
            {"role": "system", "content": "Medical coder."},
            {"role": "user", "content": "Headache"},
            {"role": "assistant", "content": "R51"}
        ]},
        # Garbage / too long
        {"messages": [
            {"role": "system", "content": "You are a medical coding assistant."},
            {"role": "user", "content": "test " * 2000},
            {"role": "assistant", "content": "ICD-10: Z99.9 - test"}
        ]},
        # Wrong format (no ICD-10 prefix)
        {"messages": [
            {"role": "system", "content": "You are a medical coding assistant."},
            {"role": "user", "content": "Patient presents with acute bronchitis, cough for 5 days, low-grade fever. No infiltrate on CXR."},
            {"role": "assistant", "content": "The diagnosis is bronchitis."}
        ]},
        # Near-duplicate (slightly different wording)
        {"messages": [
            {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
            {"role": "user", "content": "67-year-old male with substernal chest pain radiating to the left arm, diaphoresis, and dyspnea. ECG reveals ST elevation in leads V1-V4. Troponin I is elevated at 3.2 ng/mL."},
            {"role": "assistant", "content": "ICD-10: I21.0 - Acute ST elevation myocardial infarction of anterior wall"}
        ]},
    ]

    random.seed(42)
    dataset = good_samples + bad_samples
    random.shuffle(dataset)
    return dataset


class QualityPipeline:
    """Multi-stage data quality filtering pipeline."""

    def __init__(self, client: OpenAI = None):
        self.client = client
        self.stages = []
        self.log = []

    def _log(self, stage: str, action: str, sample_idx: int, detail: str = ""):
        self.log.append({"stage": stage, "action": action, "index": sample_idx, "detail": detail})

    def filter_completeness(self, data: list) -> list:
        """Remove samples with missing required fields/roles."""
        stage = "completeness"
        passed = []
        for i, sample in enumerate(data):
            msgs = sample.get("messages", [])
            roles = set(m.get("role", "") for m in msgs)
            required = {"system", "user", "assistant"}
            if required.issubset(roles):
                passed.append(sample)
            else:
                missing = required - roles
                self._log(stage, "REMOVED", i, f"missing roles: {missing}")
        self.stages.append((stage, len(data), len(passed)))
        return passed

    def filter_empty_content(self, data: list) -> list:
        """Remove samples with empty or whitespace-only content."""
        stage = "empty_content"
        passed = []
        for i, sample in enumerate(data):
            all_filled = all(
                m.get("content", "").strip()
                for m in sample["messages"]
            )
            if all_filled:
                passed.append(sample)
            else:
                self._log(stage, "REMOVED", i, "empty content field")
        self.stages.append((stage, len(data), len(passed)))
        return passed

    def filter_length(self, data: list, min_user: int = 30, max_user: int = 3000,
                      min_asst: int = 5) -> list:
        """Remove samples where content is too short or too long."""
        stage = "length"
        passed = []
        for i, sample in enumerate(data):
            user_msg = next((m for m in sample["messages"] if m["role"] == "user"), None)
            asst_msg = next((m for m in sample["messages"] if m["role"] == "assistant"), None)
            if not user_msg or not asst_msg:
                self._log(stage, "REMOVED", i, "missing user or assistant")
                continue
            u_len = len(user_msg["content"])
            a_len = len(asst_msg["content"])
            if min_user <= u_len <= max_user and a_len >= min_asst:
                passed.append(sample)
            else:
                self._log(stage, "REMOVED", i, f"user_len={u_len}, asst_len={a_len}")
        self.stages.append((stage, len(data), len(passed)))
        return passed

    def filter_format(self, data: list, required_prefix: str = "ICD-10:") -> list:
        """Check that the assistant response follows expected format."""
        stage = "format"
        passed = []
        for i, sample in enumerate(data):
            asst_msg = next((m for m in sample["messages"] if m["role"] == "assistant"), None)
            if asst_msg and required_prefix in asst_msg["content"]:
                passed.append(sample)
            else:
                self._log(stage, "REMOVED", i, f"missing '{required_prefix}' prefix")
        self.stages.append((stage, len(data), len(passed)))
        return passed

    def filter_duplicates(self, data: list) -> list:
        """Remove exact duplicates based on content hash."""
        stage = "exact_duplicates"
        seen = set()
        passed = []
        for i, sample in enumerate(data):
            content_str = json.dumps(sample["messages"], sort_keys=True)
            h = hashlib.md5(content_str.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                passed.append(sample)
            else:
                self._log(stage, "REMOVED", i, "exact duplicate")
        self.stages.append((stage, len(data), len(passed)))
        return passed

    def filter_near_duplicates(self, data: list, threshold: float = 0.9) -> list:
        """Remove near-duplicates based on character-level similarity."""
        stage = "near_duplicates"

        def char_similarity(a: str, b: str) -> float:
            """Simple character set overlap similarity."""
            if not a or not b:
                return 0.0
            set_a = set(a.lower().split())
            set_b = set(b.lower().split())
            if not set_a or not set_b:
                return 0.0
            intersection = set_a & set_b
            union = set_a | set_b
            return len(intersection) / len(union)

        passed = []
        user_contents = []
        for i, sample in enumerate(data):
            user_msg = next((m for m in sample["messages"] if m["role"] == "user"), None)
            if not user_msg:
                passed.append(sample)
                continue
            content = user_msg["content"]
            is_near_dup = False
            for existing in user_contents:
                sim = char_similarity(content, existing)
                if sim >= threshold:
                    is_near_dup = True
                    self._log(stage, "REMOVED", i, f"near-duplicate (sim={sim:.2f})")
                    break
            if not is_near_dup:
                passed.append(sample)
                user_contents.append(content)

        self.stages.append((stage, len(data), len(passed)))
        return passed

    def score_quality_with_llm(self, data: list, sample_size: int = 5,
                                min_score: float = 3.0) -> list:
        """Score a sample of data using LLM-as-judge and flag low quality."""
        stage = "llm_quality"
        if not self.client:
            print("  (Skipping LLM quality scoring — no client)")
            self.stages.append((stage, len(data), len(data)))
            return data

        indices = random.sample(range(len(data)), min(sample_size, len(data)))
        scores = {}

        for idx in indices:
            sample = data[idx]
            user_msg = next(m for m in sample["messages"] if m["role"] == "user")
            asst_msg = next(m for m in sample["messages"] if m["role"] == "assistant")

            prompt = f"""Rate the quality of this medical training example on a scale of 1-5:

Clinical note: "{user_msg['content']}"
Label: "{asst_msg['content']}"

Score on:
1. Clinical realism (does it read like a real clinical note?)
2. Label accuracy (is the ICD-10 code appropriate for the note?)
3. Completeness (are demographics, symptoms, and findings present?)

Return ONLY a JSON object with "average_score" (float) and "reasoning" (string). No markdown."""

            try:
                response = self.client.chat.completions.create(
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
                result = json.loads(raw)
                score = float(result.get("average_score", 5))
                scores[idx] = score
                if score < min_score:
                    self._log(stage, "FLAGGED", idx,
                              f"score={score:.1f}, reason={result.get('reasoning', '')[:80]}")
            except Exception as e:
                scores[idx] = 5  # default pass on error
                self._log(stage, "ERROR", idx, str(e))

        # Remove low-scoring samples
        passed = []
        for i, sample in enumerate(data):
            if i in scores and scores[i] < min_score:
                self._log(stage, "REMOVED", i, f"score={scores[i]:.1f} < {min_score}")
            else:
                passed.append(sample)

        self.stages.append((stage, len(data), len(passed)))
        return passed

    def run(self, data: list) -> list:
        """Run the complete quality pipeline."""
        print(f"\n--- Running Quality Pipeline ---")
        print(f"  Input: {len(data)} samples\n")

        data = self.filter_completeness(data)
        print(f"  After completeness check:    {len(data)}")

        data = self.filter_empty_content(data)
        print(f"  After empty content check:   {len(data)}")

        data = self.filter_length(data)
        print(f"  After length check:          {len(data)}")

        data = self.filter_format(data)
        print(f"  After format check:          {len(data)}")

        data = self.filter_duplicates(data)
        print(f"  After exact dedup:           {len(data)}")

        data = self.filter_near_duplicates(data)
        print(f"  After near-dedup:            {len(data)}")

        data = self.score_quality_with_llm(data)
        print(f"  After LLM quality check:     {len(data)}")

        return data

    def report(self, original_size: int):
        """Print pipeline report."""
        print(f"\n--- Pipeline Report ---")
        print(f"  {'Stage':25s} {'Input':>6s} {'Output':>7s} {'Removed':>8s}")
        print("  " + "-" * 50)
        for stage, inp, out in self.stages:
            removed = inp - out
            print(f"  {stage:25s} {inp:6d} {out:7d} {removed:8d}")

        total_removed = original_size - (self.stages[-1][2] if self.stages else original_size)
        pct_kept = (1 - total_removed / original_size) * 100 if original_size > 0 else 0
        print(f"\n  Original: {original_size} → Final: {original_size - total_removed}")
        print(f"  Kept: {pct_kept:.0f}%")

        if self.log:
            print(f"\n--- Removal Log ({len(self.log)} events) ---")
            for entry in self.log[:15]:
                print(f"  [{entry['stage']}] {entry['action']} idx={entry['index']}: {entry['detail'][:60]}")
            if len(self.log) > 15:
                print(f"  ... and {len(self.log) - 15} more")


def main():
    """Build and run a data quality pipeline."""

    print("=" * 60)
    print("Exercise 3: Data Quality Pipeline")
    print("=" * 60)

    # --- Create noisy dataset ---
    print("\n--- Creating Noisy Dataset ---")
    dataset = create_noisy_dataset()
    print(f"  Created dataset with {len(dataset)} samples (includes intentional quality issues)")

    # --- Initialize and run pipeline ---
    try:
        client = OpenAI()
    except Exception:
        client = None
        print("  (No OpenAI client — LLM quality scoring will be skipped)")

    pipeline = QualityPipeline(client=client)
    clean_data = pipeline.run(dataset)

    # --- Report ---
    pipeline.report(len(dataset))

    # --- Save clean dataset ---
    print(f"\n--- Saving Clean Dataset ---")
    output_path = os.path.join(os.path.dirname(__file__) or ".", "icd10_clean.jsonl")
    with open(output_path, "w") as f:
        for sample in clean_data:
            f.write(json.dumps(sample) + "\n")
    print(f"  Saved {len(clean_data)} clean samples to: {output_path}")
    print(f"\n  ✓ Quality pipeline complete!")


if __name__ == "__main__":
    main()
