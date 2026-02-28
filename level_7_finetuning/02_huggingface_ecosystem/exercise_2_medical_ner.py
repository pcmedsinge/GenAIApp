"""
Exercise 2 — Medical Named Entity Recognition
==============================================

Skills practiced
----------------
* Using biomedical NER models from HuggingFace
* Extracting diseases, medications, dosages, and procedures from text
* Post-processing NER output (merging sub-word tokens, deduplication)
* Evaluating entity extraction quality against a gold standard

Healthcare context
------------------
Structured data extraction from free-text clinical notes is one of the
highest-value NLP tasks in healthcare.  Automatically identifying
medications, diagnoses, and procedures enables: medication reconciliation,
automated ICD coding, adverse-event detection, and clinical trial matching.

Usage
-----
    python exercise_2_medical_ner.py

Prerequisites
-------------
    pip install transformers torch
"""

import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from transformers import pipeline
except ImportError:
    print("ERROR: pip install transformers torch")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NER_MODELS = [
    "d4data/biomedical-ner-all",       # broad biomedical NER
    "dslim/bert-base-NER",             # general fallback
]

# ---------------------------------------------------------------------------
# Clinical notes with gold-standard annotations
# ---------------------------------------------------------------------------
ANNOTATED_NOTES = [
    {
        "id": "note-1",
        "text": (
            "Assessment: 72-year-old male with congestive heart failure and "
            "type 2 diabetes mellitus. Currently on metformin 1000mg BID, "
            "lisinopril 20mg daily, and carvedilol 12.5mg BID. "
            "Plan: Increase furosemide to 40mg PO daily. Order echocardiogram. "
            "Check HbA1c and basic metabolic panel."
        ),
        "expected_entities": {
            "Disease": ["congestive heart failure", "type 2 diabetes mellitus"],
            "Medication": ["metformin", "lisinopril", "carvedilol", "furosemide"],
            "Procedure": ["echocardiogram"],
            "Lab": ["HbA1c", "basic metabolic panel"],
        },
    },
    {
        "id": "note-2",
        "text": (
            "Patient is a 55-year-old female diagnosed with stage IIIA non-small "
            "cell lung cancer. She completed 4 cycles of carboplatin and paclitaxel "
            "chemotherapy followed by concurrent radiation therapy (60 Gy in 30 "
            "fractions). PET-CT shows partial response. Immunotherapy with "
            "pembrolizumab to be initiated."
        ),
        "expected_entities": {
            "Disease": ["non-small cell lung cancer"],
            "Medication": ["carboplatin", "paclitaxel", "pembrolizumab"],
            "Procedure": ["radiation therapy", "PET-CT", "chemotherapy"],
        },
    },
    {
        "id": "note-3",
        "text": (
            "Chief complaint: acute onset chest pain. History: The patient, a "
            "48-year-old male, experienced sudden crushing substernal chest pain "
            "radiating to the left arm while at rest. He has a history of "
            "hyperlipidemia and is on atorvastatin 40mg nightly. ECG shows "
            "ST-elevation in leads V1-V4. Troponin I elevated at 2.5 ng/mL. "
            "Assessment: ST-elevation myocardial infarction (STEMI). "
            "Plan: Emergent cardiac catheterization. Load aspirin 325mg and "
            "ticagrelor 180mg. Start heparin drip per protocol."
        ),
        "expected_entities": {
            "Disease": ["hyperlipidemia", "ST-elevation myocardial infarction", "STEMI"],
            "Medication": ["atorvastatin", "aspirin", "ticagrelor", "heparin"],
            "Procedure": ["ECG", "cardiac catheterization"],
            "Lab": ["Troponin I"],
        },
    },
    {
        "id": "note-4",
        "text": (
            "Patient presents with a 3-day history of dysuria, urinary frequency, "
            "and suprapubic tenderness. Urinalysis positive for leukocyte esterase "
            "and nitrites. Urine culture pending. Assessment: uncomplicated urinary "
            "tract infection. Started nitrofurantoin 100mg PO BID for 5 days. "
            "Advised to increase fluid intake. Follow-up if symptoms persist."
        ),
        "expected_entities": {
            "Disease": ["urinary tract infection"],
            "Medication": ["nitrofurantoin"],
            "Procedure": ["Urinalysis", "Urine culture"],
        },
    },
]


# ---------------------------------------------------------------------------
# NER Extraction
# ---------------------------------------------------------------------------
@dataclass
class ExtractedEntity:
    text: str
    label: str
    score: float
    start: int = 0
    end: int = 0


def load_ner_pipeline():
    """Try loading NER models in order of preference."""
    for model_name in NER_MODELS:
        try:
            print(f"  Loading NER model: {model_name} ...")
            ner = pipeline("ner", model=model_name, aggregation_strategy="simple")
            print(f"  ✓ Loaded: {model_name}")
            return ner, model_name
        except Exception as exc:
            print(f"  ✗ {model_name}: {exc}")
    print("⚠  No NER model could be loaded.")
    sys.exit(1)


def extract_entities(ner_pipe, text: str) -> list[ExtractedEntity]:
    """Run NER and return structured entities."""
    raw = ner_pipe(text)
    entities = []
    for ent in raw:
        entities.append(ExtractedEntity(
            text=ent["word"].strip(),
            label=ent["entity_group"],
            score=round(ent["score"], 4),
            start=ent.get("start", 0),
            end=ent.get("end", 0),
        ))
    return entities


def group_entities(entities: list[ExtractedEntity]) -> dict[str, list[str]]:
    """Group entities by label, deduplicating."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for ent in entities:
        normed = ent.text.lower().strip()
        if normed and normed not in [x.lower() for x in grouped[ent.label]]:
            grouped[ent.label].append(ent.text)
    return dict(grouped)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_extraction(extracted: dict[str, list[str]],
                        expected: dict[str, list[str]]) -> dict:
    """Compute precision/recall/F1 per entity type (fuzzy match)."""
    results = {}

    # Flatten all expected entities across types for a global match
    all_expected = []
    for etype, items in expected.items():
        for item in items:
            all_expected.append((etype, item.lower()))

    all_extracted = []
    for etype, items in extracted.items():
        for item in items:
            all_extracted.append((etype, item.lower()))

    # Simple fuzzy: extracted entity matches if it's a substring of expected (or vice versa)
    true_pos = 0
    for _, ext_text in all_extracted:
        for _, exp_text in all_expected:
            if ext_text in exp_text or exp_text in ext_text:
                true_pos += 1
                break

    precision = true_pos / len(all_extracted) if all_extracted else 0.0
    recall = true_pos / len(all_expected) if all_expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "true_pos": true_pos,
        "extracted_count": len(all_extracted),
        "expected_count": len(all_expected),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Exercise 2: Medical Named Entity Recognition")
    print("=" * 60)

    ner_pipe, model_name = load_ner_pipeline()

    overall_metrics = []

    for note in ANNOTATED_NOTES:
        print(f"\n{'='*50}")
        print(f"Note: {note['id']}")
        print(f"{'='*50}")
        print(f"  Text: \"{note['text'][:100]}...\"\n")

        start = time.time()
        entities = extract_entities(ner_pipe, note["text"])
        elapsed = time.time() - start

        # Display raw entities
        print(f"  Extracted entities ({len(entities)}, {elapsed:.2f}s):")
        for ent in entities:
            print(f"    • \"{ent.text}\"  [{ent.label}]  score={ent.score:.3f}")

        # Grouped view
        grouped = group_entities(entities)
        print(f"\n  Grouped by type:")
        for label, items in sorted(grouped.items()):
            print(f"    {label}: {items}")

        # Expected vs extracted
        expected = note.get("expected_entities", {})
        if expected:
            print(f"\n  Expected entities:")
            for label, items in sorted(expected.items()):
                print(f"    {label}: {items}")

            metrics = evaluate_extraction(grouped, expected)
            overall_metrics.append(metrics)
            print(f"\n  Evaluation:")
            print(f"    Precision: {metrics['precision']:.1%}  "
                  f"Recall: {metrics['recall']:.1%}  "
                  f"F1: {metrics['f1']:.1%}")

    # --- Overall summary ---
    if overall_metrics:
        print(f"\n{'='*60}")
        print("OVERALL NER EVALUATION")
        print(f"{'='*60}")
        avg_p = sum(m["precision"] for m in overall_metrics) / len(overall_metrics)
        avg_r = sum(m["recall"] for m in overall_metrics) / len(overall_metrics)
        avg_f1 = sum(m["f1"] for m in overall_metrics) / len(overall_metrics)
        print(f"  Model     : {model_name}")
        print(f"  Notes     : {len(overall_metrics)}")
        print(f"  Avg Prec  : {avg_p:.1%}")
        print(f"  Avg Recall: {avg_r:.1%}")
        print(f"  Avg F1    : {avg_f1:.1%}")
        print()
        print("  Tips for improvement:")
        print("  • Use a domain-specific model (Bio_ClinicalBERT-based NER)")
        print("  • Fine-tune on your institution's annotated notes")
        print("  • Post-process with medical ontologies (SNOMED, RxNorm)")

    # --- Interactive mode ---
    print(f"\n{'='*60}")
    print("Interactive NER — paste a clinical note")
    print("Type 'quit' to exit.")
    print(f"{'='*60}\n")
    while True:
        text = input("Note: ").strip()
        if not text or text.lower() in ("quit", "exit", "q"):
            break
        entities = extract_entities(ner_pipe, text)
        grouped = group_entities(entities)
        for label, items in sorted(grouped.items()):
            print(f"  {label}: {items}")
        print()


if __name__ == "__main__":
    main()
