"""
Exercise 4 — Custom HuggingFace Pipeline
=========================================

Skills practiced
----------------
* Chaining multiple HuggingFace models into a multi-step pipeline
* NER → Classification → Summarisation workflow
* Structured output design for clinical decision-support
* Error handling and graceful degradation when a pipeline step fails

Healthcare context
------------------
A clinical note arrives as unstructured free text.  The pipeline:
  1. **NER** — extract entities (diseases, medications, procedures)
  2. **Classification** — route the note to a medical specialty
  3. **Summary** — generate a concise structured abstract

This mirrors real-world clinical NLP workflows: intake → extract → route → summarise.

Usage
-----
    python exercise_4_custom_pipeline.py

Prerequisites
-------------
    pip install transformers torch
"""

import json
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    print("ERROR: pip install transformers torch")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NER_MODEL = "dslim/bert-base-NER"                     # general NER (fallback-safe)
CLF_MODEL = "facebook/bart-large-mnli"                 # zero-shot classification
SUM_MODEL = "sshleifer/distilbart-cnn-12-6"            # lightweight summarisation

SPECIALTIES = [
    "cardiology", "endocrinology", "pulmonology", "neurology",
    "gastroenterology", "nephrology", "oncology",
    "infectious disease", "orthopedics", "psychiatry",
    "general surgery", "emergency medicine",
]

# ---------------------------------------------------------------------------
# Clinical notes
# ---------------------------------------------------------------------------
CLINICAL_NOTES = [
    {
        "id": "note-A",
        "text": (
            "History of Present Illness: The patient is a 68-year-old male with a "
            "past medical history of hypertension, type 2 diabetes, and coronary "
            "artery disease status post CABG in 2019. He presents to the emergency "
            "department with acute onset substernal chest pain radiating to the left "
            "arm, associated with diaphoresis and nausea. Pain started 2 hours ago "
            "while at rest. ECG on arrival shows ST-elevation in leads V2-V5. "
            "Troponin I is elevated at 3.2 ng/mL. The patient was loaded with "
            "aspirin 325 mg, ticagrelor 180 mg, and started on a heparin drip. "
            "Cardiology was consulted for emergent cardiac catheterization."
        ),
    },
    {
        "id": "note-B",
        "text": (
            "Discharge Summary: 45-year-old female admitted for management of "
            "newly diagnosed stage IIIB non-small cell lung cancer. During this "
            "admission, the patient underwent CT-guided biopsy of the right upper "
            "lobe mass, which confirmed adenocarcinoma with PD-L1 expression of 80%. "
            "EGFR and ALK testing were negative. Oncology recommended initiation of "
            "pembrolizumab 200 mg IV every 3 weeks as first-line immunotherapy. "
            "The patient tolerated the first infusion well. She was discharged in "
            "stable condition with follow-up in 3 weeks for cycle 2. Pain managed "
            "with acetaminophen and oxycodone 5 mg PRN."
        ),
    },
    {
        "id": "note-C",
        "text": (
            "Progress Note: 73-year-old female, hospital day 4, admitted for acute "
            "exacerbation of COPD. The patient has a 50-pack-year smoking history "
            "and is on home oxygen at 2 L/min. On admission, she was tachypneic "
            "with SpO2 82% on room air. Treated with nebulized albuterol and "
            "ipratropium, IV methylprednisolone 125 mg daily, and azithromycin "
            "500 mg daily. Chest X-ray shows hyperinflation without consolidation. "
            "Today, the patient feels improved. SpO2 94% on 3 L nasal cannula. "
            "Plan to transition to oral prednisone taper and discharge tomorrow "
            "with pulmonology follow-up."
        ),
    },
    {
        "id": "note-D",
        "text": (
            "Consultation Note: 52-year-old male referred by primary care for "
            "evaluation of chronic epigastric pain, dyspepsia, and a 20-pound "
            "unintentional weight loss over 4 months. He reports early satiety, "
            "occasional nausea, and dark stools. Lab work shows iron deficiency "
            "anemia with hemoglobin 9.2 g/dL and ferritin 12 ng/mL. "
            "Upper endoscopy revealed a 3 cm ulcerated mass at the gastric antrum. "
            "Biopsies obtained and sent for pathology. CT abdomen/pelvis ordered "
            "for staging. Suspect gastric adenocarcinoma; awaiting pathology "
            "results before finalizing treatment plan."
        ),
    },
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class PipelineOutput:
    note_id: str
    # NER
    entities: dict = field(default_factory=dict)  # label → [text, ...]
    entity_count: int = 0
    # Classification
    specialty: str = ""
    specialty_score: float = 0.0
    top_specialties: list = field(default_factory=list)  # [{label, score}, ...]
    # Summary
    summary: str = ""
    # Timing
    ner_time_s: float = 0.0
    clf_time_s: float = 0.0
    sum_time_s: float = 0.0
    total_time_s: float = 0.0
    # Errors
    errors: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
class ClinicalPipeline:
    """NER → Classification → Summarisation pipeline."""

    def __init__(self):
        self.ner_pipe = None
        self.clf_pipe = None
        self.sum_pipe = None

    def load(self):
        """Load all models. Gracefully skip any that fail."""
        print("\nLoading pipeline models...")

        try:
            self.ner_pipe = hf_pipeline("ner", model=NER_MODEL,
                                         aggregation_strategy="simple")
            print(f"  ✓ NER: {NER_MODEL}")
        except Exception as exc:
            print(f"  ✗ NER: {exc}")

        try:
            self.clf_pipe = hf_pipeline("zero-shot-classification", model=CLF_MODEL)
            print(f"  ✓ Classification: {CLF_MODEL}")
        except Exception as exc:
            print(f"  ✗ Classification: {exc}")

        try:
            self.sum_pipe = hf_pipeline("summarization", model=SUM_MODEL)
            print(f"  ✓ Summarisation: {SUM_MODEL}")
        except Exception as exc:
            print(f"  ✗ Summarisation: {exc}")

        print()

    def _step_ner(self, text: str, output: PipelineOutput):
        """Step 1: Extract entities."""
        if self.ner_pipe is None:
            output.errors.append("NER model not loaded")
            return

        start = time.time()
        raw = self.ner_pipe(text)
        output.ner_time_s = round(time.time() - start, 3)

        grouped: dict[str, list[str]] = defaultdict(list)
        for ent in raw:
            word = ent["word"].strip()
            label = ent["entity_group"]
            if word.lower() not in [w.lower() for w in grouped[label]]:
                grouped[label].append(word)

        output.entities = dict(grouped)
        output.entity_count = sum(len(v) for v in grouped.values())

    def _step_classify(self, text: str, output: PipelineOutput):
        """Step 2: Classify specialty."""
        if self.clf_pipe is None:
            output.errors.append("Classification model not loaded")
            return

        start = time.time()
        result = self.clf_pipe(text, candidate_labels=SPECIALTIES)
        output.clf_time_s = round(time.time() - start, 3)

        output.specialty = result["labels"][0]
        output.specialty_score = round(result["scores"][0], 4)
        output.top_specialties = [
            {"label": l, "score": round(s, 4)}
            for l, s in zip(result["labels"][:3], result["scores"][:3])
        ]

    def _step_summarise(self, text: str, output: PipelineOutput):
        """Step 3: Summarise."""
        if self.sum_pipe is None:
            output.errors.append("Summarisation model not loaded")
            return

        start = time.time()
        # Truncate input to avoid exceeding model's max length
        max_chars = 1024
        truncated = text[:max_chars]
        try:
            result = self.sum_pipe(truncated, max_length=120, min_length=30,
                                    do_sample=False)
            output.summary = result[0]["summary_text"]
        except Exception as exc:
            output.errors.append(f"Summarisation error: {exc}")
        output.sum_time_s = round(time.time() - start, 3)

    def process(self, note_id: str, text: str) -> PipelineOutput:
        """Run the full 3-step pipeline on a clinical note."""
        output = PipelineOutput(note_id=note_id)
        total_start = time.time()

        self._step_ner(text, output)
        self._step_classify(text, output)
        self._step_summarise(text, output)

        output.total_time_s = round(time.time() - total_start, 3)
        return output


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def print_output(out: PipelineOutput):
    """Pretty-print a pipeline output."""
    print(f"\n{'─'*50}")
    print(f"Note: {out.note_id}")
    print(f"{'─'*50}")

    # NER
    print(f"\n  [Step 1] Named Entities ({out.entity_count} found, {out.ner_time_s}s):")
    if out.entities:
        for label, items in sorted(out.entities.items()):
            print(f"    {label}: {', '.join(items)}")
    else:
        print("    (none)")

    # Classification
    print(f"\n  [Step 2] Specialty Classification ({out.clf_time_s}s):")
    if out.specialty:
        print(f"    Predicted: {out.specialty} (confidence: {out.specialty_score:.3f})")
        if out.top_specialties:
            print("    Top 3:")
            for s in out.top_specialties:
                bar = "█" * int(s["score"] * 30)
                print(f"      {s['label']:<22} {s['score']:.3f} {bar}")
    else:
        print("    (unavailable)")

    # Summary
    print(f"\n  [Step 3] Summary ({out.sum_time_s}s):")
    print(f"    {out.summary if out.summary else '(unavailable)'}")

    # Timing
    print(f"\n  Total pipeline time: {out.total_time_s}s")

    if out.errors:
        print(f"\n  Warnings/Errors:")
        for e in out.errors:
            print(f"    ⚠ {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Exercise 4: Custom HuggingFace Pipeline")
    print("  NER → Specialty Classification → Summarisation")
    print("=" * 60)

    pipe = ClinicalPipeline()
    pipe.load()

    all_outputs = []

    for note in CLINICAL_NOTES:
        print(f"\nProcessing {note['id']}...")
        output = pipe.process(note["id"], note["text"])
        print_output(output)
        all_outputs.append(output)

    # --- Summary table ---
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Note':<10} {'Entities':<10} {'Specialty':<22} {'Conf':<8} {'Time(s)'}")
    print("  " + "-" * 62)
    for o in all_outputs:
        print(f"  {o.note_id:<10} {o.entity_count:<10} {o.specialty:<22} "
              f"{o.specialty_score:<8.3f} {o.total_time_s}")

    # --- JSON export ---
    print(f"\n--- Structured output (JSON) ---")
    export = []
    for o in all_outputs:
        d = asdict(o)
        export.append(d)
    print(json.dumps(export, indent=2, default=str))

    # --- Interactive mode ---
    print(f"\n{'='*60}")
    print("Interactive Mode — paste a clinical note")
    print("Type 'quit' to exit.")
    print(f"{'='*60}\n")

    while True:
        text = input("Note: ").strip()
        if not text or text.lower() in ("quit", "exit", "q"):
            break
        output = pipe.process("interactive", text)
        print_output(output)

    print("\nDone.")


if __name__ == "__main__":
    main()
