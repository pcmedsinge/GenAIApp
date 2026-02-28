"""






































































patient narratives, lab data, and medical history to make decisions.mirrors actual clinical practice where providers synthesize visual findings,Real clinical workflows rarely involve a single data type. This capstone## Healthcare Relevance```[Labs/Data] ──┘[Text]      ──┤      Reasoning]         Clinical Data][Audio]     ──┼──→  [Multimodal    ──→  [Structured[Image]     ──┐─────────────         ──────────           ──────Input Sources          Processing           Output```## Architecture```python exercise_1_triage_assistant.py   # Run exercise directlypython main.py                          # Interactive demo menu```bash## Running```pip install openai pydantic python-dotenv```bash## Prerequisites4. `exercise_4_multimodal_pipeline.py` — End-to-end pipeline with evaluation3. `exercise_3_case_analyzer.py` — Multi-source case analysis with reasoning2. `exercise_2_clinical_scribe.py` — Audio → structured clinical documentation1. `exercise_1_triage_assistant.py` — Multimodal triage with image + text## Exercises4. **Full Multimodal Pipeline** — all modalities in single clinical workflow3. **Reasoning + Multiple Inputs** — complex case via reasoning model2. **Audio + Structured Output** — clinical audio → structured SOAP note1. **Image + Text Analysis** — medical image with patient history context## Demos in main.py- Maintain audit trails for all AI-assisted decisions- Never replace clinical judgment — augment it- Flag cases requiring human review- Always include confidence levels and limitations### Healthcare Safety- Error handling across modality boundaries- Confidence scoring for clinical decision support- Measure end-to-end accuracy and consistency- Evaluate each pipeline stage independently### Quality and Evaluation- Full pipeline: all modalities → assessment → structured output- Case analysis: history + labs + imaging → differential diagnosis- Scribe: audio dictation → structured clinical notes- Triage: image + symptoms → urgency classification### Clinical Workflow Pipelines- Use reasoning models for complex cases with multiple data sources- Chain audio transcription → structured output for clinical documentation- Combine image analysis + text context for richer clinical assessments### Multimodal Integration## Key Conceptsmultiple input modalities simultaneously.clinical workflows. Build real-world healthcare AI systems that processtranscription, reasoning models, and structured outputs — into integratedThis capstone project combines all multimodal capabilities — vision, audio## OverviewExercise 2 — SOAP Note Generator
==================================
Take raw clinical transcription text and generate a structured SOAP
note (Subjective, Objective, Assessment, Plan) using the LLM.

Objectives
----------
* Design effective prompts for SOAP note extraction
* Parse four SOAP sections from free-text clinical conversation
* Output both prose and structured JSON formats
* Calculate quality metrics (section completeness, coverage)
"""

import json
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Sample transcriptions
# ---------------------------------------------------------------------------

TRANSCRIPTION_CARDIAC = """\
Doctor: Good morning Mr. Davis. How are you doing today?
Patient: Not great, doc. I've been having this chest tightness for about a week.
Doctor: Tell me more about it. When does it happen?
Patient: Mostly when I climb stairs or walk fast. It feels like someone is
sitting on my chest. It goes away after I rest for a few minutes.
Doctor: Any pain going to your arm, jaw, or back?
Patient: Sometimes to my left arm.
Doctor: Any shortness of breath, sweating, or nausea with it?
Patient: A little short of breath, but no sweating or nausea.
Doctor: What's your medical history?
Patient: I have high blood pressure and high cholesterol. My dad had bypass
surgery at sixty. I'm also a former smoker — quit five years ago.
Doctor: Current medications?
Patient: Lisinopril 20mg, Atorvastatin 40mg, and baby aspirin.
Doctor: Allergies?
Patient: No known drug allergies.
Doctor: Let me examine you. Blood pressure is 148/92, heart rate 82, regular.
Oxygen saturation 97%. Heart sounds normal, S1 S2, no murmurs, gallops, or rubs.
Lungs clear bilaterally. No peripheral edema. EKG shows normal sinus rhythm
with no ST changes.
Doctor: Based on your symptoms, risk factors, and exam, I'm concerned about
possible stable angina. I'd like to order a troponin level, a lipid panel,
and schedule a stress test. Continue your current medications. I'm adding
nitroglycerin 0.4mg sublingual as needed for chest pain. If you have chest
pain lasting more than 15 minutes or at rest, go to the emergency room.
Let's follow up in one week after the stress test.
Patient: Okay, thank you doctor.
"""

TRANSCRIPTION_DIABETES = """\
Doctor: Hi Sarah, welcome back. How has your blood sugar been?
Patient: It's been all over the place. My fasting levels are usually around
160 to 180 and after meals it goes up to 250 sometimes.
Doctor: Are you taking your Metformin regularly?
Patient: Yes, 1000mg twice a day. But I think I need something else.
Doctor: How about your diet? Have you been following the meal plan?
Patient: I try but it's hard. I've been eating more carbs lately with the
holidays.
Doctor: Any episodes of low blood sugar? Shakiness, sweating, confusion?
Patient: No, nothing like that.
Doctor: Numbness or tingling in your feet?
Patient: Actually yes, I've noticed some tingling in both feet the past month.
Doctor: That could be early diabetic neuropathy. Let me check. Your A1c came
back at 8.9, which is up from 7.8 three months ago. Fasting glucose today
is 172. Foot exam shows decreased sensation to monofilament in both feet.
Blood pressure 134/82. BMI is 32.
Doctor: Your diabetes is not well controlled. I'd like to add Glipizide 5mg
before breakfast. Let's also start Gabapentin 300mg at bedtime for the
neuropathy symptoms. I'm ordering a comprehensive metabolic panel, urine
microalbumin, and referring you to a dietitian and a podiatrist. Please
check your blood sugar four times a day — fasting and two hours after each
meal. Follow up in four weeks with your glucose log.
Patient: Will the new medicine cause low blood sugar?
Doctor: Glipizide can cause hypoglycemia. Eat regular meals, carry glucose
tablets, and if you feel shaky or dizzy, check your sugar immediately.
Patient: Got it. Thank you.
"""

SAMPLE_TRANSCRIPTIONS = {
    "cardiac": TRANSCRIPTION_CARDIAC,
    "diabetes": TRANSCRIPTION_DIABETES,
}


# ---------------------------------------------------------------------------
# SOAP generation prompt
# ---------------------------------------------------------------------------

SOAP_SYSTEM_PROMPT = """\
You are an expert medical scribe.  Given a clinical encounter transcription,
generate a complete SOAP note following standard medical documentation format.

## FORMAT

**SUBJECTIVE:**
- Chief Complaint (CC)
- History of Present Illness (HPI): onset, location, duration, character,
  aggravating/alleviating factors, radiation, timing, severity
- Review of Systems (pertinent positives and negatives)
- Past Medical History (PMH)
- Past Surgical History (PSH) if mentioned
- Family History (FHx) if mentioned
- Social History (SHx) if mentioned
- Medications
- Allergies

**OBJECTIVE:**
- Vital Signs
- Physical Examination findings
- Relevant lab / test results mentioned

**ASSESSMENT:**
- Primary diagnosis with ICD-10 code
- Secondary diagnoses if applicable
- Clinical reasoning (1-2 sentences)

**PLAN:**
- Medications (new and continued) with dose, route, frequency
- Orders (labs, imaging, referrals)
- Follow-up timing
- Patient education / instructions
- Return precautions

Use concise professional medical language.  Do not fabricate information
not mentioned in the transcription.
"""

SOAP_JSON_PROMPT = """\
Convert the following SOAP note into structured JSON.  Use this exact schema:

{
  "subjective": {
    "chief_complaint": "<string>",
    "hpi": "<string>",
    "ros_pertinent_positives": ["<string>"],
    "ros_pertinent_negatives": ["<string>"],
    "past_medical_history": ["<string>"],
    "family_history": "<string or null>",
    "social_history": "<string or null>",
    "medications": [{"name": "<string>", "dose": "<string>", "frequency": "<string>"}],
    "allergies": ["<string>"]
  },
  "objective": {
    "vitals": {
      "bp": "<string>",
      "hr": "<string>",
      "spo2": "<string>",
      "temp": "<string or null>",
      "rr": "<string or null>"
    },
    "exam_findings": ["<string>"],
    "lab_results": ["<string>"]
  },
  "assessment": {
    "primary_diagnosis": "<string>",
    "icd10_code": "<string>",
    "secondary_diagnoses": ["<string>"],
    "clinical_reasoning": "<string>"
  },
  "plan": {
    "new_medications": [{"name": "<string>", "dose": "<string>", "route": "<string>", "frequency": "<string>"}],
    "continued_medications": ["<string>"],
    "orders": ["<string>"],
    "referrals": ["<string>"],
    "follow_up": "<string>",
    "patient_education": ["<string>"],
    "return_precautions": ["<string>"]
  }
}

Return ONLY valid JSON with no markdown fences.
"""


# ---------------------------------------------------------------------------
# SOAP dataclass
# ---------------------------------------------------------------------------

@dataclass
class SOAPNote:
    prose: str = ""
    structured: dict = field(default_factory=dict)
    tokens_prose: int = 0
    tokens_json: int = 0
    source_transcription: str = ""


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------

def generate_soap_prose(transcription: str) -> tuple[str, int]:
    """Generate a SOAP note in prose format.

    Returns
    -------
    (soap_text, total_tokens)
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SOAP_SYSTEM_PROMPT},
            {"role": "user", "content": transcription},
        ],
        max_tokens=1200,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip(), response.usage.total_tokens


def soap_prose_to_json(soap_text: str) -> tuple[dict, int]:
    """Convert a SOAP prose note into structured JSON.

    Returns
    -------
    (parsed_dict, total_tokens)
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SOAP_JSON_PROMPT},
            {"role": "user", "content": soap_text},
        ],
        max_tokens=1000,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"_raw": raw, "_error": "JSON parse failed"}
    return data, response.usage.total_tokens


def generate_soap(transcription: str) -> SOAPNote:
    """Full SOAP generation pipeline: transcription → prose → JSON."""
    note = SOAPNote(source_transcription=transcription)

    # Step 1: Prose SOAP
    note.prose, note.tokens_prose = generate_soap_prose(transcription)

    # Step 2: Structured JSON
    note.structured, note.tokens_json = soap_prose_to_json(note.prose)

    return note


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = ["subjective", "objective", "assessment", "plan"]


def compute_quality(note: SOAPNote) -> dict:
    """Compute quality metrics for a generated SOAP note."""
    sections_present = [s for s in REQUIRED_SECTIONS if s in note.structured]
    completeness = len(sections_present) / len(REQUIRED_SECTIONS) * 100

    # Check sub-field coverage
    total_fields = 0
    filled_fields = 0
    for section in REQUIRED_SECTIONS:
        sec_data = note.structured.get(section, {})
        if isinstance(sec_data, dict):
            for k, v in sec_data.items():
                total_fields += 1
                if v is not None and v != "" and v != [] and v != {}:
                    filled_fields += 1

    field_coverage = (filled_fields / total_fields * 100) if total_fields else 0

    return {
        "sections_present": sections_present,
        "section_completeness_pct": round(completeness, 1),
        "total_fields": total_fields,
        "filled_fields": filled_fields,
        "field_coverage_pct": round(field_coverage, 1),
        "total_tokens": note.tokens_prose + note.tokens_json,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_soap_generator():
    print("=" * 70)
    print("  Exercise 2 — SOAP Note Generator")
    print("=" * 70)

    for name, transcript in SAMPLE_TRANSCRIPTIONS.items():
        print(f"\n{'='*70}")
        print(f"  Generating SOAP for: {name.upper()} encounter")
        print(f"{'='*70}\n")

        note = generate_soap(transcript)

        # Prose output
        print("=== SOAP Note (Prose) ===\n")
        print(note.prose[:800])
        if len(note.prose) > 800:
            print("  … [truncated]")

        # JSON output
        print("\n=== SOAP Note (JSON) ===\n")
        print(json.dumps(note.structured, indent=2)[:600])

        # Quality
        quality = compute_quality(note)
        print(f"\n=== Quality Metrics ===")
        print(f"  Section completeness: {quality['section_completeness_pct']}%")
        print(f"  Field coverage:       {quality['field_coverage_pct']}%")
        print(f"  Total tokens used:    {quality['total_tokens']}")
        print("-" * 70)


if __name__ == "__main__":
    demo_soap_generator()
