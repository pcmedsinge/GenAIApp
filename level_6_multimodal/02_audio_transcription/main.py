"""









































































and regulatory reporting pipelines.enabling reliable integration with EHR systems, clinical decision support,Structured outputs eliminate parsing errors in clinical data extraction,## Healthcare Relevance```result = completion.choices[0].message.parsed  # typed MySchema object)    response_format=MySchema,    messages=[...],    model="gpt-4o",completion = client.beta.chat.completions.parse(    value: float    name: strclass MySchema(BaseModel):from openai import OpenAIfrom pydantic import BaseModel```python## Key API Pattern```python exercise_1_clinical_extraction.py # Run exercise directlypython main.py                           # Interactive demo menu```bash## Running```pip install openai pydantic python-dotenv```bash## Prerequisites4. `exercise_4_discharge_summary.py` — Structured discharge summary generation3. `exercise_3_lab_reports.py` — Structured lab report generation2. `exercise_2_fhir_resources.py` — Generate FHIR-compatible resources1. `exercise_1_clinical_extraction.py` — Extract structured patient data from free text## Exercises4. **Schema Evolution** — handle v1 → v2 schema changes gracefully3. **Enum Constraints** — Literal types to constrain field values2. **Nested Schemas** — complex nested objects (Patient + Medications + Dosing)1. **JSON Schema Basics** — response_format with Pydantic model parsing## Demos in main.py- Bridge unstructured clinical text → structured interoperable data- Generate FHIR Patient, Condition, MedicationRequest resources- Fast Healthcare Interoperability Resources (FHIR) standard### FHIR-Compatible Data- Lab reports with multiple test results and reference ranges- Patient → Medications → Dosing instructions- Complex objects with lists, nested models, optional fields### Nested Schemas- Validation happens both at the API level and in your code- `client.beta.chat.completions.parse()` returns typed Pydantic objects- Define data models in Pydantic, convert to JSON schema automatically### Pydantic → JSON Schema- Uses constrained decoding, not prompt-based tricks- Guarantees output matches your schema exactly — no parsing errors- OpenAI's `response_format` parameter with `json_schema` type### JSON Schema Enforcement## Key Conceptsmachine-readable and conform to standards like FHIR.conform to exact JSON schemas. This is critical in healthcare where data must beLearn to use OpenAI's structured outputs feature to guarantee that model responses## OverviewLevel 6.2 — Audio Transcription: Whisper API & Clinical SOAP Notes
====================================================================
Demonstrates the OpenAI Whisper API for medical audio transcription,
post-processing of transcripts, and SOAP note generation.

Demos
-----
1. Whisper Basics             — API parameters, response formats
2. Medical Terminology        — prompt priming for medical vocab
3. Transcription Post-Processing — cleanup, abbreviations, speakers
4. SOAP Note Generation       — transcription → structured SOAP note
"""

import json
import os
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def print_banner(title: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")


def print_separator() -> None:
    print("-" * 70)


# ---------------------------------------------------------------------------
# Simulated transcription data (used when no audio file is available)
# ---------------------------------------------------------------------------

SIMULATED_RAW_TRANSCRIPTION = """\
Doctor: Good morning Mrs. Johnson. How are you feeling today?
Patient: Well, I've been having these headaches for the past two weeks. \
They're mostly on the right side and sometimes I feel nauseous.
Doctor: I see. On a scale of one to ten, how would you rate the pain?
Patient: About a seven or eight when it's really bad. The ibuprofen \
doesn't seem to help much anymore.
Doctor: Have you noticed any visual changes, like seeing spots or \
flashing lights?
Patient: Actually yes, sometimes I see these zigzag lines before \
the headache starts.
Doctor: That sounds like it could be a migraine with aura. Let me \
check your blood pressure and do a neurological exam. Your BP is \
one forty two over ninety which is a bit elevated. Neurological exam \
is otherwise normal. I'd like to start you on sumatriptan twenty five \
milligrams as needed for acute episodes and we should discuss \
preventive options. I'm also ordering an MRI of the brain just to \
rule out any structural causes. Do you have any allergies to medications?
Patient: I'm allergic to sulfa drugs.
Doctor: Noted. And your current medications?
Patient: I take lisinopril ten milligrams for blood pressure and \
metformin five hundred milligrams twice daily for my diabetes.
Doctor: Okay. Let's also check your hemoglobin A1c while we're at it. \
I'd like you to keep a headache diary tracking frequency, duration, \
triggers and severity. Follow up in four weeks or sooner if the \
headaches worsen.
Patient: Thank you doctor."""

SIMULATED_WHISPER_OUTPUT = {
    "text": SIMULATED_RAW_TRANSCRIPTION,
    "language": "en",
    "duration": 127.5,
    "segments": [
        {"start": 0.0, "end": 4.2, "text": "Good morning Mrs. Johnson. How are you feeling today?"},
        {"start": 4.5, "end": 14.8, "text": "Well, I've been having these headaches for the past two weeks."},
    ],
}


# ===================================================================
#  DEMO 1 — Whisper Basics
# ===================================================================
def demo_whisper_basics():
    """Demonstrate the Whisper transcription API and its parameters.

    API: ``client.audio.transcriptions.create()``

    Parameters
    ----------
    model : str
        Currently ``"whisper-1"`` is the only model.
    file : file-like
        Audio file (mp3, wav, m4a, mp4, mpeg, mpga, webm).  Max 25 MB.
    language : str, optional
        ISO-639-1 code (e.g. ``"en"``).  Helps accuracy when known.
    prompt : str, optional
        Text to prime the model — improves recognition of names,
        medical terms, abbreviations.
    response_format : str
        ``"text"`` — plain string
        ``"json"`` — ``{"text": "..."}``
        ``"verbose_json"`` — includes segments, language, duration
        ``"srt"`` / ``"vtt"`` — subtitle formats
    temperature : float
        0.0–1.0.  Lower = more deterministic.
    """
    print_banner("DEMO 1 — Whisper Basics")

    # --- Show the API pattern ---
    print("=== Whisper Transcription API Pattern ===\n")
    print("""\
# Basic transcription call:
transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=open("patient_encounter.mp3", "rb"),
    language="en",
    response_format="verbose_json",
    temperature=0.0,
)

# Access results:
print(transcription.text)           # full text
print(transcription.language)       # detected language
print(transcription.duration)       # audio duration in seconds
print(transcription.segments)       # timestamped segments
""")

    # --- Try real file if available ---
    audio_path = "sample_audio.mp3"
    if os.path.exists(audio_path):
        print(f"Found {audio_path} — transcribing …\n")
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en",
                response_format="verbose_json",
                temperature=0.0,
            )
        print(f"Language: {result.language}")
        print(f"Duration: {result.duration}s")
        print(f"\nTranscription:\n{result.text[:500]}")
    else:
        print(f"No audio file found at '{audio_path}' — using simulated output.\n")
        print(f"Language: {SIMULATED_WHISPER_OUTPUT['language']}")
        print(f"Duration: {SIMULATED_WHISPER_OUTPUT['duration']}s")
        print(f"\n=== Simulated Transcription ===\n")
        print(SIMULATED_WHISPER_OUTPUT["text"][:600])

    # --- Response format comparison ---
    print_separator()
    print("\n=== Response Formats ===\n")
    formats = {
        "text": "Plain text string — simplest, most common",
        "json": '{"text": "..."} — minimal JSON wrapper',
        "verbose_json": "Includes segments[], language, duration — richest",
        "srt": "SubRip subtitle format — for video captioning",
        "vtt": "WebVTT subtitle format — for web video players",
    }
    for fmt, desc in formats.items():
        print(f"  {fmt:<15} {desc}")


# ===================================================================
#  DEMO 2 — Medical Terminology
# ===================================================================

MEDICAL_PROMPT_TERMS = (
    "Sumatriptan, Lisinopril, Metformin, hemoglobin A1c, "
    "echocardiogram, MRI, CT scan, CBC, BMP, migraine with aura, "
    "hypertension, type 2 diabetes mellitus, sulfonamide allergy, "
    "neurological examination, blood pressure"
)


def demo_medical_terminology():
    """Show how the ``prompt`` parameter improves medical term recognition.

    The ``prompt`` field lets you provide example text that primes
    Whisper's vocabulary.  For medical transcription, include drug
    names, procedures, abbreviations, and diagnoses you expect to hear.
    """
    print_banner("DEMO 2 — Medical Terminology Handling")

    print("=== The Prompt Parameter ===\n")
    print("When transcribing medical audio, common errors include:\n")
    print("  • 'some a trip tan' instead of 'sumatriptan'")
    print("  • 'lie sin oh pril' instead of 'lisinopril'")
    print("  • 'hemoglobin A one C' instead of 'hemoglobin A1c'\n")
    print("The `prompt` parameter primes the model with expected vocabulary:\n")
    print(f"  prompt = \"{MEDICAL_PROMPT_TERMS}\"\n")

    # --- Show API call with prompt ---
    print("=== API Call with Medical Prompt ===\n")
    print("""\
transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=open("encounter.mp3", "rb"),
    language="en",
    prompt="Sumatriptan, Lisinopril, Metformin, hemoglobin A1c, "
           "echocardiogram, MRI, migraine with aura, hypertension",
    response_format="verbose_json",
)
""")

    # --- Simulated comparison ---
    print("=== Simulated Before/After Comparison ===\n")

    without_prompt = (
        "The patient is on lie sin oh pril ten milligrams and "
        "met for men five hundred milligrams. Her hemoglobin A one C "
        "was seven point two. I'm prescribing some a trip tan for "
        "the my grains."
    )
    with_prompt = (
        "The patient is on Lisinopril 10 milligrams and "
        "Metformin 500 milligrams. Her hemoglobin A1c "
        "was 7.2. I'm prescribing Sumatriptan for "
        "the migraines."
    )

    print("WITHOUT prompt priming:")
    print(f"  {without_prompt}\n")
    print("WITH prompt priming:")
    print(f"  {with_prompt}\n")

    # --- Building a medical prompt dictionary ---
    print_separator()
    print("\n=== Building a Medical Prompt Dictionary ===\n")

    specialties = {
        "cardiology": "echocardiogram, EKG, troponin, ejection fraction, "
                      "atrial fibrillation, warfarin, metoprolol, stent",
        "endocrinology": "hemoglobin A1c, Metformin, insulin glargine, "
                         "thyroid stimulating hormone, levothyroxine",
        "neurology": "sumatriptan, MRI, EEG, migraine with aura, "
                     "gabapentin, carbamazepine, lumbar puncture",
        "pulmonology": "spirometry, albuterol, FEV1, bronchoscopy, "
                       "CPAP, pulmonary function test, prednisone",
    }

    for specialty, terms in specialties.items():
        print(f"  {specialty.upper()}: {terms}")

    print("\nTip: Load the prompt from a per-specialty config file at runtime.")


# ===================================================================
#  DEMO 3 — Transcription Post-Processing
# ===================================================================

MEDICAL_ABBREVIATIONS = {
    "BP": "blood pressure",
    "HR": "heart rate",
    "RR": "respiratory rate",
    "SpO2": "oxygen saturation",
    "CBC": "complete blood count",
    "BMP": "basic metabolic panel",
    "CMP": "comprehensive metabolic panel",
    "MRI": "magnetic resonance imaging",
    "CT": "computed tomography",
    "EKG": "electrocardiogram",
    "ECG": "electrocardiogram",
    "PRN": "as needed",
    "BID": "twice daily",
    "TID": "three times daily",
    "QD": "once daily",
    "PO": "by mouth",
    "IV": "intravenous",
    "IM": "intramuscular",
}

SPEAKER_MARKERS = {
    "Doctor:": "PROVIDER",
    "Patient:": "PATIENT",
    "Nurse:": "NURSE",
    "doctor:": "PROVIDER",
    "patient:": "PATIENT",
    "nurse:": "NURSE",
}


def post_process_transcription(raw_text: str) -> dict:
    """Clean up a raw transcription.

    Steps:
    1. Identify speakers from text cues
    2. Expand medical abbreviations
    3. Normalize number expressions
    4. Return structured dialogue
    """
    lines = raw_text.strip().split("\n")
    dialogue = []
    current_speaker = "UNKNOWN"

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect speaker changes
        for marker, speaker in SPEAKER_MARKERS.items():
            if line.startswith(marker):
                current_speaker = speaker
                line = line[len(marker):].strip()
                break

        dialogue.append({
            "speaker": current_speaker,
            "text": line,
        })

    return {
        "turns": len(dialogue),
        "speakers": list(set(d["speaker"] for d in dialogue)),
        "dialogue": dialogue,
    }


def expand_abbreviations(text: str) -> str:
    """Expand medical abbreviations in transcription text."""
    result = text
    for abbr, full in MEDICAL_ABBREVIATIONS.items():
        # Only expand standalone abbreviations (word boundaries)
        import re
        result = re.sub(rf'\b{re.escape(abbr)}\b', f"{abbr} ({full})", result, count=1)
    return result


def demo_post_processing():
    """Post-process a transcription: speakers, abbreviations, formatting."""
    print_banner("DEMO 3 — Transcription Post-Processing")

    raw = SIMULATED_RAW_TRANSCRIPTION
    print("=== Raw Transcription (first 300 chars) ===\n")
    print(raw[:300] + " …\n")

    # --- Speaker identification ---
    print("=== Speaker Identification ===\n")
    result = post_process_transcription(raw)
    print(f"Turns detected: {result['turns']}")
    print(f"Speakers: {', '.join(result['speakers'])}\n")

    print("First 5 turns:")
    for turn in result["dialogue"][:5]:
        print(f"  [{turn['speaker']}] {turn['text'][:80]}")

    # --- Abbreviation expansion ---
    print_separator()
    print("\n=== Abbreviation Expansion ===\n")
    sample = "BP is 142/90, HR 78, SpO2 98%. Order CBC and BMP. Sumatriptan 25mg PO PRN."
    expanded = expand_abbreviations(sample)
    print(f"Before: {sample}")
    print(f"After:  {expanded}")

    # --- GPT-based cleanup ---
    print_separator()
    print("\n=== GPT-Based Transcription Cleanup ===\n")

    cleanup_prompt = """\
Clean up this raw medical transcription.  Fix:
- Spell out numbers as digits (e.g., "one forty two over ninety" → "142/90")
- Capitalize drug names properly
- Fix medical term spelling
- Keep speaker labels (Doctor: / Patient:)
- Add paragraph breaks between speaker changes

Return only the cleaned transcription text.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": cleanup_prompt},
            {"role": "user", "content": raw},
        ],
        max_tokens=800,
        temperature=0,
    )
    cleaned = response.choices[0].message.content.strip()
    print(cleaned[:600])


# ===================================================================
#  DEMO 4 — SOAP Note Generation
# ===================================================================
def demo_soap_generation():
    """Take a transcription and generate a structured SOAP note.

    SOAP = Subjective, Objective, Assessment, Plan
    """
    print_banner("DEMO 4 — SOAP Note Generation")

    print("Generating SOAP note from clinical encounter transcription …\n")

    soap_prompt = """\
You are an expert medical scribe.  Given the following clinical encounter
transcription, generate a professional SOAP note.

SOAP NOTE FORMAT:

**SUBJECTIVE:**
- Chief complaint
- History of present illness (HPI)
- Review of systems (ROS)
- Past medical/surgical history mentioned
- Current medications
- Allergies

**OBJECTIVE:**
- Vital signs mentioned
- Physical examination findings
- Diagnostic results mentioned

**ASSESSMENT:**
- Primary diagnosis with ICD-10 code if determinable
- Differential diagnoses if applicable
- Clinical reasoning

**PLAN:**
- Medications prescribed (with dose, route, frequency)
- Diagnostic orders (labs, imaging)
- Follow-up instructions
- Patient education

Use professional medical documentation style.  Be thorough but concise.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": soap_prompt},
            {"role": "user", "content": SIMULATED_RAW_TRANSCRIPTION},
        ],
        max_tokens=1000,
        temperature=0.2,
    )

    soap_note = response.choices[0].message.content.strip()
    print("=== Generated SOAP Note ===\n")
    print(soap_note)

    # --- Structured JSON extraction ---
    print_separator()
    print("\n=== Structured JSON SOAP ===\n")

    json_prompt = """\
Convert this SOAP note into structured JSON with these keys:
{
  "subjective": {
    "chief_complaint": "",
    "hpi": "",
    "medications": [],
    "allergies": []
  },
  "objective": {
    "vitals": {},
    "exam_findings": []
  },
  "assessment": {
    "primary_diagnosis": "",
    "icd10_code": "",
    "differential": []
  },
  "plan": {
    "medications": [],
    "orders": [],
    "follow_up": "",
    "education": []
  }
}
Return ONLY valid JSON."""

    json_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": json_prompt},
            {"role": "user", "content": soap_note},
        ],
        max_tokens=800,
        temperature=0,
    )

    raw_json = json_response.choices[0].message.content.strip()
    if raw_json.startswith("```"):
        raw_json = raw_json.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        structured = json.loads(raw_json)
        print(json.dumps(structured, indent=2))
    except json.JSONDecodeError:
        print(raw_json)

    usage = json_response.usage
    print(f"\nTokens — prompt: {usage.prompt_tokens}  "
          f"completion: {usage.completion_tokens}  "
          f"total: {usage.total_tokens}")


# ===================================================================
#  Main menu
# ===================================================================
DEMOS = {
    "1": ("Whisper Basics", demo_whisper_basics),
    "2": ("Medical Terminology", demo_medical_terminology),
    "3": ("Transcription Post-Processing", demo_post_processing),
    "4": ("SOAP Note Generation", demo_soap_generation),
}


def main():
    print_banner("Level 6.2 — Audio Transcription: Whisper API & SOAP Notes")
    print("This module demonstrates the Whisper API for medical transcription")
    print("and GPT-4o for generating structured clinical notes.\n")
    print("Note: Demos that require audio files will use simulated text when")
    print("no .mp3/.wav file is available.\n")

    while True:
        print("\nAvailable demos:")
        for key, (title, _) in DEMOS.items():
            print(f"  [{key}] {title}")
        print("  [a] Run ALL demos")
        print("  [q] Quit\n")

        choice = input("Select demo → ").strip().lower()

        if choice == "q":
            print("Goodbye!")
            break
        elif choice == "a":
            for _, (_, func) in DEMOS.items():
                func()
                print_separator()
        elif choice in DEMOS:
            DEMOS[choice][1]()
        else:
            print("Invalid selection. Try again.")


if __name__ == "__main__":
    main()
