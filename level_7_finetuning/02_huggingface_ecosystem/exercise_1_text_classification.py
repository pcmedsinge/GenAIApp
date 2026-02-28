"""
Exercise 1 — Clinical Text Classification
==========================================

Skills practiced
----------------
* Using HuggingFace zero-shot classification pipeline
* Designing label sets for medical specialty routing
* Evaluating classification confidence and thresholds
* Handling multi-label clinical notes

Healthcare context
------------------
Hospital EHR systems receive thousands of clinical notes daily.  Automatic
specialty classification helps route notes to the right department,
trigger relevant clinical-decision-support alerts, and power analytics
dashboards.  This exercise builds a specialty classifier using a
pre-trained zero-shot model — no fine-tuning required.

Usage
-----
    python exercise_1_text_classification.py

Prerequisites
-------------
    pip install transformers torch
"""

import sys
import time
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Check dependencies
# ---------------------------------------------------------------------------
try:
    from transformers import pipeline
except ImportError:
    print("ERROR: pip install transformers torch")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ZSC_MODEL = "facebook/bart-large-mnli"

SPECIALTIES = [
    "cardiology",
    "endocrinology",
    "pulmonology",
    "neurology",
    "gastroenterology",
    "nephrology",
    "oncology",
    "infectious disease",
    "orthopedics",
    "psychiatry",
]

CLINICAL_NOTES = [
    {
        "id": "note-1",
        "text": (
            "72-year-old male with a history of myocardial infarction presents with "
            "exertional dyspnea and bilateral lower extremity edema. Echocardiogram "
            "shows ejection fraction of 30%. BNP elevated at 1200 pg/mL. "
            "Started on furosemide, carvedilol, and sacubitril-valsartan."
        ),
        "expected": "cardiology",
    },
    {
        "id": "note-2",
        "text": (
            "45-year-old female with poorly controlled type 2 diabetes, HbA1c 9.8%. "
            "Currently on metformin 2000mg daily and glipizide 10mg BID. "
            "Fasting glucose consistently above 200 mg/dL. Considering initiation "
            "of basal insulin. Also has diabetic retinopathy and microalbuminuria."
        ),
        "expected": "endocrinology",
    },
    {
        "id": "note-3",
        "text": (
            "58-year-old male smoker with progressive dyspnea over 6 months. "
            "Spirometry shows FEV1/FVC ratio of 0.62 and FEV1 45% predicted. "
            "Chest CT reveals emphysematous changes. Diagnosed with severe COPD. "
            "Started on tiotropium and budesonide-formoterol inhaler."
        ),
        "expected": "pulmonology",
    },
    {
        "id": "note-4",
        "text": (
            "34-year-old female presents with recurrent episodes of unilateral "
            "throbbing headache with visual aura, nausea, and photophobia lasting "
            "4-72 hours. Episodes occur 3-4 times per month. MRI brain is normal. "
            "Started on topiramate for migraine prophylaxis."
        ),
        "expected": "neurology",
    },
    {
        "id": "note-5",
        "text": (
            "55-year-old male with 3-month history of progressive dysphagia to "
            "solids, 15-pound weight loss, and iron deficiency anemia. "
            "Upper endoscopy reveals a 4 cm ulcerated mass in the distal esophagus. "
            "Biopsy confirms adenocarcinoma. CT staging pending."
        ),
        "expected": "oncology",
    },
    {
        "id": "note-6",
        "text": (
            "28-year-old male presents with acute onset right lower quadrant "
            "abdominal pain, low-grade fever, and leukocytosis. CT abdomen shows "
            "dilated appendix with periappendiceal fat stranding. Surgical "
            "consultation obtained for appendectomy."
        ),
        "expected": "gastroenterology",
    },
    {
        "id": "note-7",
        "text": (
            "62-year-old female with progressive bilateral knee pain worse with "
            "weight-bearing and morning stiffness lasting 20 minutes. X-rays show "
            "joint space narrowing and osteophyte formation. BMI 34. Started on "
            "physical therapy, weight loss program, and acetaminophen."
        ),
        "expected": "orthopedics",
    },
    {
        "id": "note-8",
        "text": (
            "40-year-old male with HIV on antiretroviral therapy presents with "
            "fever, cough, and dyspnea. CD4 count 85 cells/µL. Chest X-ray shows "
            "bilateral diffuse infiltrates. LDH elevated. Induced sputum positive "
            "for Pneumocystis jirovecii. Started on TMP-SMX."
        ),
        "expected": "infectious disease",
    },
]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
@dataclass
class ClassificationResult:
    note_id: str
    predicted: str
    expected: str
    correct: bool
    confidence: float
    all_scores: dict  # specialty → score


def classify_notes(notes: list[dict], specialties: list[str]) -> list[ClassificationResult]:
    """Classify each note using zero-shot classification."""
    print(f"\nLoading model: {ZSC_MODEL} ...")
    zsc = pipeline("zero-shot-classification", model=ZSC_MODEL)
    print("✓ Model loaded.\n")

    results = []
    for note in notes:
        start = time.time()
        out = zsc(note["text"], candidate_labels=specialties)
        elapsed = time.time() - start

        predicted = out["labels"][0]
        confidence = out["scores"][0]
        all_scores = dict(zip(out["labels"], out["scores"]))
        correct = predicted == note["expected"]

        result = ClassificationResult(
            note_id=note["id"],
            predicted=predicted,
            expected=note["expected"],
            correct=correct,
            confidence=confidence,
            all_scores=all_scores,
        )
        results.append(result)

        mark = "✓" if correct else "✗"
        print(f"  {mark} {note['id']}: predicted={predicted} (conf={confidence:.3f}) "
              f"expected={note['expected']}  [{elapsed:.1f}s]")

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyse_results(results: list[ClassificationResult]):
    """Print accuracy, confusion cases, and confidence analysis."""
    print(f"\n{'='*60}")
    print("CLASSIFICATION RESULTS")
    print(f"{'='*60}")

    correct = sum(1 for r in results if r.correct)
    total = len(results)
    print(f"\nAccuracy: {correct}/{total} ({correct/total:.0%})")

    # Confusion cases
    wrong = [r for r in results if not r.correct]
    if wrong:
        print(f"\nMisclassified ({len(wrong)}):")
        for r in wrong:
            print(f"  {r.note_id}: predicted={r.predicted} (conf={r.confidence:.3f}), "
                  f"expected={r.expected}")
            exp_score = r.all_scores.get(r.expected, 0.0)
            print(f"    Score for correct label: {exp_score:.3f}")

    # Confidence histogram
    print(f"\nConfidence distribution:")
    for bucket_low in [0.0, 0.3, 0.5, 0.7, 0.9]:
        bucket_high = bucket_low + (0.3 if bucket_low < 0.9 else 0.1)
        count = sum(1 for r in results if bucket_low <= r.confidence < bucket_high)
        bar = "█" * count
        print(f"  {bucket_low:.1f}-{bucket_high:.1f}: {bar} ({count})")

    # Multi-label analysis: notes where >1 specialty scores above 0.3
    print(f"\nMulti-specialty notes (top-2 scores both > 0.15):")
    for r in results:
        sorted_scores = sorted(r.all_scores.items(), key=lambda x: -x[1])
        if len(sorted_scores) >= 2 and sorted_scores[1][1] > 0.15:
            s1, s2 = sorted_scores[0], sorted_scores[1]
            print(f"  {r.note_id}: {s1[0]}={s1[1]:.3f}, {s2[0]}={s2[1]:.3f}")

    # Threshold recommendation
    avg_conf = sum(r.confidence for r in results) / len(results)
    print(f"\nAverage confidence: {avg_conf:.3f}")
    print("Recommendation: set a routing confidence threshold of ~0.40.")
    print("Notes below threshold → manual review queue.")


# ---------------------------------------------------------------------------
# Interactive classifier
# ---------------------------------------------------------------------------
def interactive_classify():
    """Let the user type a clinical note and see the classification."""
    print(f"\n{'='*60}")
    print("Interactive Specialty Classifier")
    print("Type a clinical note.  Type 'quit' to exit.")
    print(f"{'='*60}\n")

    zsc = pipeline("zero-shot-classification", model=ZSC_MODEL)

    while True:
        text = input("Note: ").strip()
        if not text or text.lower() in ("quit", "exit", "q"):
            break

        out = zsc(text, candidate_labels=SPECIALTIES)
        print("\n  Predicted specialties:")
        for label, score in zip(out["labels"][:5], out["scores"][:5]):
            bar = "█" * int(score * 30)
            print(f"    {label:<22} {score:.3f} {bar}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Exercise 1: Clinical Text Classification")
    print("=" * 60)

    print("\n1. Run benchmark (8 sample notes)")
    print("2. Interactive classifier")
    print("3. Both")
    choice = input("\nChoice (1/2/3): ").strip()

    if choice in ("1", "3"):
        results = classify_notes(CLINICAL_NOTES, SPECIALTIES)
        analyse_results(results)
    if choice in ("2", "3"):
        interactive_classify()
    if choice not in ("1", "2", "3"):
        print("Running benchmark by default...")
        results = classify_notes(CLINICAL_NOTES, SPECIALTIES)
        analyse_results(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
