"""
Exercise 4 — Complete Transcription Pipeline
==============================================
Full pipeline: audio → transcription → speaker separation → medical
spell-check → SOAP note → structured data extraction.

Objectives
----------
* Chain all previous components into a single pipeline
* Handle real audio files (Whisper) or simulated text gracefully
* Track latency and token usage per stage
* Produce a final comprehensive clinical document
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Simulated raw transcript
# ---------------------------------------------------------------------------

SIMULATED_ENCOUNTER = """\
Good morning Ms. Thompson. I'm Dr. Rivera. What brings you in today? \
I've been feeling really tired for the past month and I've lost about \
ten pounds without trying. I'm also really thirsty all the time and \
going to the bathroom a lot. \
That's concerning. Any blurry vision or numbness in your hands or feet? \
My vision has been a little blurry especially at night. No numbness though. \
Any family history of diabetes? \
Yes my mother has type 2 diabetes and my sister was just diagnosed last year. \
What medications are you currently taking? \
I take levothyroxine 75 micrograms for my thyroid and sertraline 50 \
milligrams for anxiety. \
Any allergies? \
I had a reaction to amoxicillin once, got hives. \
Let me check you over. Blood pressure 138 over 86. Heart rate 78. \
Temperature 98.4. Weight 185 pounds, down from 195 last visit. BMI 30.8. \
Heart exam normal. Lungs clear. No edema. Skin appears dry. \
I checked a finger stick glucose and it's 287. We also have your labs from \
this morning — fasting glucose 256, hemoglobin A1c 10.2, creatinine 0.9, \
BUN 18, sodium 141, potassium 4.2. Urinalysis shows 3 plus glucose and \
trace ketones. \
Based on your symptoms, family history, and labs, this is new-onset \
type 2 diabetes mellitus. Your A1c of 10.2 is quite high so I'd like to \
start Metformin 500mg twice daily with meals and titrate up as tolerated. \
I'm also starting you on a glucose monitor. Check your blood sugar fasting \
and two hours after dinner. \
I'm referring you to diabetes education and a registered dietitian. \
We need to schedule an eye exam for diabetic retinopathy screening. \
Come back in four weeks. If you develop severe nausea, vomiting, or \
abdominal pain, go to the ER as that could indicate diabetic ketoacidosis. \
Is metformin going to upset my stomach? \
It can cause some GI side effects initially. Take it with food and we'll \
start with a lower dose and increase gradually. If it's intolerable, \
there are alternatives. \
Okay thank you doctor.
"""


# ---------------------------------------------------------------------------
# Pipeline stage tracking
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    name: str = ""
    output: str = ""
    data: dict = field(default_factory=dict)
    tokens: int = 0
    latency_s: float = 0.0


@dataclass
class PipelineResult:
    stages: list = field(default_factory=list)
    total_tokens: int = 0
    total_latency_s: float = 0.0
    final_document: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage 1: Transcription (Whisper or simulated)
# ---------------------------------------------------------------------------

def stage_transcribe(audio_path: str = None) -> StageResult:
    """Transcribe audio or use simulated text."""
    t0 = time.time()
    result = StageResult(name="transcribe")

    if audio_path and os.path.exists(audio_path):
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en",
                prompt=(
                    "Metformin, Levothyroxine, Sertraline, hemoglobin A1c, "
                    "diabetic ketoacidosis, retinopathy, creatinine"
                ),
                response_format="text",
            )
        result.output = transcription if isinstance(transcription, str) else transcription.text
    else:
        result.output = SIMULATED_ENCOUNTER

    result.latency_s = round(time.time() - t0, 2)
    return result


# ---------------------------------------------------------------------------
# Stage 2: Speaker separation
# ---------------------------------------------------------------------------

SPEAKER_PROMPT = """\
Add speaker labels (PROVIDER: or PATIENT:) to each utterance in this
clinical transcript.  Determine the speaker from context.  Put each
labeled utterance on its own line.  Return ONLY the labeled text.
"""


def stage_speaker_separation(raw_text: str) -> StageResult:
    """Separate speakers using GPT."""
    t0 = time.time()
    result = StageResult(name="speaker_separation")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SPEAKER_PROMPT},
            {"role": "user", "content": raw_text},
        ],
        max_tokens=1200,
        temperature=0,
    )

    result.output = response.choices[0].message.content.strip()
    result.tokens = response.usage.total_tokens
    result.latency_s = round(time.time() - t0, 2)

    # Parse into structured dialogue
    dialogue = []
    for line in result.output.split("\n"):
        line = line.strip()
        if line.startswith("PROVIDER:"):
            dialogue.append({"speaker": "PROVIDER", "text": line[9:].strip()})
        elif line.startswith("PATIENT:"):
            dialogue.append({"speaker": "PATIENT", "text": line[8:].strip()})
    result.data = {"dialogue": dialogue, "turns": len(dialogue)}

    return result


# ---------------------------------------------------------------------------
# Stage 3: Medical spell-check
# ---------------------------------------------------------------------------

SPELLCHECK_PROMPT = """\
You are a medical transcription editor.  Fix any misspelled medical
terminology in this text.  Leave non-medical words unchanged.
Return ONLY the corrected text.
"""


def stage_spellcheck(text: str) -> StageResult:
    """Fix medical spelling errors using GPT."""
    t0 = time.time()
    result = StageResult(name="spellcheck")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SPELLCHECK_PROMPT},
            {"role": "user", "content": text},
        ],
        max_tokens=1200,
        temperature=0,
    )

    result.output = response.choices[0].message.content.strip()
    result.tokens = response.usage.total_tokens
    result.latency_s = round(time.time() - t0, 2)
    return result


# ---------------------------------------------------------------------------
# Stage 4: SOAP note generation
# ---------------------------------------------------------------------------

SOAP_PROMPT = """\
Generate a complete SOAP note from this clinical encounter transcript.
Use professional medical documentation format with clear sections:
SUBJECTIVE, OBJECTIVE, ASSESSMENT, PLAN.
Include ICD-10 codes where determinable.
"""


def stage_soap_generation(text: str) -> StageResult:
    """Generate a SOAP note from the cleaned transcript."""
    t0 = time.time()
    result = StageResult(name="soap_generation")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SOAP_PROMPT},
            {"role": "user", "content": text},
        ],
        max_tokens=1200,
        temperature=0.2,
    )

    result.output = response.choices[0].message.content.strip()
    result.tokens = response.usage.total_tokens
    result.latency_s = round(time.time() - t0, 2)
    return result


# ---------------------------------------------------------------------------
# Stage 5: Structured data extraction
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
Convert this SOAP note into structured JSON with this schema:

{
  "patient": {"name": "", "age": null},
  "encounter_date": null,
  "subjective": {
    "chief_complaint": "",
    "hpi_summary": "",
    "symptoms": [],
    "medications": [{"name": "", "dose": "", "indication": ""}],
    "allergies": [{"allergen": "", "reaction": ""}],
    "family_history": []
  },
  "objective": {
    "vitals": {"bp": "", "hr": "", "temp": "", "weight": "", "bmi": ""},
    "exam_findings": [],
    "lab_results": [{"test": "", "value": "", "unit": "", "flag": ""}]
  },
  "assessment": {
    "diagnoses": [{"name": "", "icd10": "", "status": ""}],
    "clinical_reasoning": ""
  },
  "plan": {
    "new_medications": [{"name": "", "dose": "", "route": "", "frequency": "", "instructions": ""}],
    "orders": [],
    "referrals": [],
    "follow_up": "",
    "patient_education": [],
    "return_precautions": []
  }
}

Return ONLY valid JSON, no markdown fences.
"""


def stage_structured_extraction(soap_text: str) -> StageResult:
    """Extract structured data from the SOAP note."""
    t0 = time.time()
    result = StageResult(name="structured_extraction")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": soap_text},
        ],
        max_tokens=1200,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        result.data = json.loads(raw)
    except json.JSONDecodeError:
        result.data = {"_raw": raw, "_error": "JSON parse failed"}

    result.output = raw
    result.tokens = response.usage.total_tokens
    result.latency_s = round(time.time() - t0, 2)
    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(audio_path: str = None) -> PipelineResult:
    """Execute the complete transcription-to-documentation pipeline.

    Stages:
    1. Transcribe (Whisper or simulated)
    2. Speaker separation
    3. Medical spell-check
    4. SOAP note generation
    5. Structured data extraction
    """
    pipeline = PipelineResult()

    # Stage 1
    print("  [1/5] Transcribing …")
    s1 = stage_transcribe(audio_path)
    pipeline.stages.append(s1)

    # Stage 2
    print("  [2/5] Separating speakers …")
    s2 = stage_speaker_separation(s1.output)
    pipeline.stages.append(s2)

    # Stage 3
    print("  [3/5] Spell-checking medical terms …")
    s3 = stage_spellcheck(s2.output)
    pipeline.stages.append(s3)

    # Stage 4
    print("  [4/5] Generating SOAP note …")
    s4 = stage_soap_generation(s3.output)
    pipeline.stages.append(s4)

    # Stage 5
    print("  [5/5] Extracting structured data …")
    s5 = stage_structured_extraction(s4.output)
    pipeline.stages.append(s5)

    # Aggregate
    pipeline.total_tokens = sum(s.tokens for s in pipeline.stages)
    pipeline.total_latency_s = round(sum(s.latency_s for s in pipeline.stages), 2)
    pipeline.final_document = s5.data

    return pipeline


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_pipeline_report(pipeline: PipelineResult) -> None:
    """Print a comprehensive pipeline execution report."""
    print("\n" + "=" * 70)
    print("  TRANSCRIPTION PIPELINE REPORT")
    print("=" * 70)

    print(f"\n  Total tokens:  {pipeline.total_tokens}")
    print(f"  Total latency: {pipeline.total_latency_s}s\n")

    print("  Stage Breakdown:")
    print(f"  {'Stage':<25} {'Tokens':>8} {'Latency':>10}")
    print("  " + "-" * 45)
    for s in pipeline.stages:
        print(f"  {s.name:<25} {s.tokens:>8} {s.latency_s:>9.2f}s")

    # Print final document excerpt
    print("\n" + "=" * 70)
    print("  FINAL STRUCTURED DOCUMENT")
    print("=" * 70)
    print(json.dumps(pipeline.final_document, indent=2)[:1200])

    # Quality check
    doc = pipeline.final_document
    if isinstance(doc, dict) and "_error" not in doc:
        sections = ["subjective", "objective", "assessment", "plan"]
        present = [s for s in sections if s in doc]
        print(f"\n  SOAP sections present: {len(present)}/{len(sections)}")

        # Count medications
        new_meds = doc.get("plan", {}).get("new_medications", [])
        curr_meds = doc.get("subjective", {}).get("medications", [])
        print(f"  Current medications:   {len(curr_meds)}")
        print(f"  New medications:       {len(new_meds)}")

        diagnoses = doc.get("assessment", {}).get("diagnoses", [])
        print(f"  Diagnoses:             {len(diagnoses)}")

        labs = doc.get("objective", {}).get("lab_results", [])
        print(f"  Lab results:           {len(labs)}")
    print()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_transcription_pipeline():
    print("=" * 70)
    print("  Exercise 4 — Complete Transcription Pipeline")
    print("=" * 70)
    print()
    print("This pipeline chains 5 stages:")
    print("  1. Audio transcription  (Whisper or simulated)")
    print("  2. Speaker separation   (GPT)")
    print("  3. Medical spell-check  (GPT)")
    print("  4. SOAP note generation (GPT)")
    print("  5. Structured extraction (GPT)")
    print()

    audio_file = input("Enter audio file path (or Enter for simulated) → ").strip()
    audio_path = audio_file if audio_file else None

    print("\nRunning pipeline …\n")
    result = run_pipeline(audio_path)
    print_pipeline_report(result)

    # Optionally save full output
    save = input("Save full report to JSON? (y/n) → ").strip().lower()
    if save == "y":
        output = {
            "total_tokens": result.total_tokens,
            "total_latency_s": result.total_latency_s,
            "stages": [
                {
                    "name": s.name,
                    "tokens": s.tokens,
                    "latency_s": s.latency_s,
                    "output_preview": s.output[:500],
                }
                for s in result.stages
            ],
            "final_document": result.final_document,
        }
        path = "pipeline_output.json"
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_transcription_pipeline()
