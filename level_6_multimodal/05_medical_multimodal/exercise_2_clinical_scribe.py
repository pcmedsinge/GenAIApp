"""
Exercise 2: Clinical Scribe — Audio to Structured Documentation
================================================================
Combines audio transcription with structured output generation to create
a clinical scribe that converts dictated notes into formatted clinical
documentation (SOAP notes, H&P sections).

Learning Objectives:
  - Chain audio transcription → structured extraction pipeline
  - Generate structured SOAP notes from conversational input
  - Handle multiple documentation formats (SOAP, H&P, Progress Note)
  - Implement quality checks on generated documentation

Usage:
  python exercise_2_clinical_scribe.py
"""

import json
import os
import time
from typing import List, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI()
MODEL = "gpt-4o"


# ============================================================================
# DOCUMENTATION SCHEMAS
# ============================================================================

class VitalSigns(BaseModel):
    """Vital signs extracted from encounter."""
    blood_pressure: Optional[str] = None
    heart_rate: Optional[int] = None
    temperature: Optional[float] = None
    respiratory_rate: Optional[int] = None
    oxygen_saturation: Optional[int] = None
    weight: Optional[str] = None
    height: Optional[str] = None
    bmi: Optional[float] = None


class ReviewOfSystems(BaseModel):
    """Review of Systems section."""
    constitutional: Optional[str] = None
    cardiovascular: Optional[str] = None
    respiratory: Optional[str] = None
    gastrointestinal: Optional[str] = None
    musculoskeletal: Optional[str] = None
    neurological: Optional[str] = None
    psychiatric: Optional[str] = None
    skin: Optional[str] = None
    other: Optional[str] = None


class PhysicalExam(BaseModel):
    """Physical examination findings."""
    general: Optional[str] = None
    heent: Optional[str] = None
    cardiovascular: Optional[str] = None
    respiratory: Optional[str] = None
    abdomen: Optional[str] = None
    musculoskeletal: Optional[str] = None
    neurological: Optional[str] = None
    skin: Optional[str] = None


class AssessmentPlan(BaseModel):
    """Single assessment/plan item."""
    diagnosis: str = Field(..., description="Diagnosis or problem")
    icd10: Optional[str] = Field(None, description="ICD-10 code")
    assessment: str = Field(..., description="Assessment of current status")
    plan_items: List[str] = Field(..., description="Plan items for this problem")


class SOAPNote(BaseModel):
    """Structured SOAP note."""
    patient_name: Optional[str] = None
    visit_date: Optional[str] = None
    visit_type: Optional[str] = None
    provider: Optional[str] = None
    subjective: str = Field(..., description="Subjective section — HPI, symptoms, concerns")
    review_of_systems: Optional[ReviewOfSystems] = None
    objective: str = Field(..., description="Objective section — exam findings, vitals")
    vitals: Optional[VitalSigns] = None
    physical_exam: Optional[PhysicalExam] = None
    assessment: str = Field(..., description="Assessment — clinical impression")
    assessment_plans: List[AssessmentPlan] = Field(
        default_factory=list, description="Problem-based assessment and plan"
    )
    plan_summary: List[str] = Field(
        default_factory=list, description="Overall plan items"
    )
    follow_up: Optional[str] = None
    time_spent: Optional[str] = None


class DocumentationQuality(BaseModel):
    """Quality assessment of generated documentation."""
    completeness_score: int = Field(..., ge=1, le=5, description="1-5 completeness")
    organization_score: int = Field(..., ge=1, le=5, description="1-5 organization")
    medical_accuracy_score: int = Field(..., ge=1, le=5, description="1-5 accuracy")
    coding_support_score: int = Field(..., ge=1, le=5, description="1-5 coding support")
    missing_elements: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


# ============================================================================
# CLINICAL ENCOUNTER SCENARIOS
# ============================================================================

ENCOUNTER_SCENARIOS = [
    {
        "title": "Primary Care — Diabetes Follow-up",
        "audio_scenario": (
            "A primary care physician seeing a 62-year-old female patient, "
            "Mrs. Patricia Chen, for diabetes follow-up. She reports her blood "
            "sugars have been running 160-200 fasting, she's been compliant with "
            "metformin but forgot her evening dose a few times. She has tingling "
            "in her feet that's gotten worse. She's also here for her annual flu "
            "shot. No chest pain, shortness of breath, or visual changes. "
            "Vitals: BP 138/84, HR 78, Temp 98.2, Weight 192 lbs, BMI 32.1. "
            "Physical exam: well-appearing, heart regular rate rhythm, lungs clear, "
            "feet show decreased monofilament sensation bilaterally, no ulcers. "
            "Labs from last week: HbA1c 8.6% (was 7.8%), fasting glucose 195, "
            "creatinine 1.1, eGFR 62. Assessment: diabetes not at goal, "
            "worsening neuropathy, need to optimize therapy."
        ),
    },
    {
        "title": "Urgent Care — Acute Back Pain",
        "audio_scenario": (
            "Urgent care visit for David Miller, 38-year-old male, presenting "
            "with acute low back pain for 3 days after helping a friend move. "
            "Pain is in the lower lumbar area, 7 out of 10, worse with bending "
            "and sitting, better when lying flat. No leg numbness, weakness, "
            "no bowel or bladder changes, no fever. He took ibuprofen 400 with "
            "mild relief. No history of back problems or surgery. Works as an "
            "accountant, mostly desk work. Review of systems otherwise negative. "
            "Vitals are all normal: BP 124/76, HR 72, Temp 98.4. "
            "Exam: paravertebral muscle spasm bilateral L4-L5 region, tenderness "
            "to palpation, limited forward flexion due to pain, straight leg raise "
            "negative bilaterally, normal strength and sensation in lower extremities, "
            "normal reflexes. Assessment: acute mechanical low back pain, muscular strain."
        ),
    },
    {
        "title": "Cardiology Consult — New Heart Failure",
        "audio_scenario": (
            "Cardiology outpatient consultation for James Washington, 71-year-old "
            "male, referred by PCP for new diagnosis of heart failure. Patient "
            "reports 2-3 months of progressive exertional dyspnea, can now only "
            "walk about half a block before needing to stop, which is a significant "
            "decline from his baseline. He sleeps on 3 pillows, has noticed ankle "
            "swelling for about 6 weeks, and has gained about 12 pounds. He also "
            "reports paroxysmal nocturnal dyspnea waking him 2-3 times per week. "
            "History: hypertension 20 years, type 2 diabetes, prior MI in 2020 "
            "with stent to RCA. Currently on metoprolol 25 BID, lisinopril 10 daily, "
            "aspirin, atorvastatin, metformin. Vitals: BP 108/68, HR 92 irregular, "
            "RR 20, SpO2 93%, Weight 215 lbs. Exam: JVP elevated at 12 cm, heart "
            "shows irregular rhythm, S3 gallop present, bibasilar crackles in lower "
            "third of lung fields, hepatomegaly 3 cm below costal margin, "
            "bilateral lower extremity 2+ pitting edema to mid-calf. "
            "Recent echo shows EF 25%, global hypokinesis with severe anterior "
            "wall dysfunction. BNP was 1,850. Assessment: newly diagnosed heart "
            "failure with reduced ejection fraction, NYHA class III, likely "
            "ischemic cardiomyopathy, volume overloaded."
        ),
    },
    {
        "title": "Pediatrics — Well-Child Visit (2-year-old)",
        "audio_scenario": (
            "Well-child visit for Maya Rodriguez, 2-year-old female, brought by "
            "mother. Growth and development review: weight 27 pounds (60th percentile), "
            "height 34 inches (55th percentile), head circumference 19 inches. "
            "Mother reports she is saying about 50 words, putting 2-word sentences "
            "together, running, climbing stairs with help, eating well with "
            "varied diet, sleeping through the night 10-11 hours. Drinks whole "
            "milk about 16 ounces a day. Using a cup, starting to use a spoon. "
            "No concerns about hearing or vision. No recent illnesses. "
            "Immunizations are up to date, due for Hep A second dose today. "
            "Review of systems: no fevers, eating well, regular bowel movements, "
            "no rashes. Physical exam: well-appearing, alert, playful toddler, "
            "anterior fontanelle closed, TMs clear bilaterally, heart and lungs "
            "normal, abdomen soft, skin clear, walks with normal gait. "
            "Development: meeting all milestones for age."
        ),
    },
]


# ============================================================================
# SCRIBE FUNCTIONS
# ============================================================================

def simulate_transcription(scenario: str) -> str:
    """Simulate audio transcription from scenario description."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate a realistic verbatim transcription of a clinical "
                    "encounter as if recorded by a microphone in the exam room. "
                    "Include doctor-patient dialogue, the doctor's dictated "
                    "findings, and natural speech patterns. Make it sound like "
                    "a real clinical encounter, not a summary."
                ),
            },
            {
                "role": "user",
                "content": f"Generate transcription for:\n{scenario}",
            },
        ],
        max_tokens=1200,
    )
    return response.choices[0].message.content


def generate_soap_note(transcript: str) -> SOAPNote:
    """Convert a clinical transcript into a structured SOAP note."""
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical documentation AI (scribe) that converts "
                    "clinical encounter transcriptions into structured SOAP notes.\n\n"
                    "Rules:\n"
                    "1. Use proper medical terminology\n"
                    "2. Organize by SOAP sections\n"
                    "3. Include all mentioned vitals, exam findings, and labs\n"
                    "4. Create problem-based assessment/plan entries\n"
                    "5. Include ICD-10 codes when diagnoses are clear\n"
                    "6. Be concise but complete — don't add information not in transcript\n"
                    "7. Use standard abbreviations appropriately"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Convert this transcription to a structured SOAP note:\n\n"
                    f"{transcript}"
                ),
            },
        ],
        response_format=SOAPNote,
    )
    return completion.choices[0].message.parsed


def evaluate_documentation(soap: SOAPNote) -> DocumentationQuality:
    """Evaluate the quality of generated documentation."""
    soap_json = json.dumps(soap.model_dump(exclude_none=True), indent=2)

    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical documentation quality reviewer. Evaluate "
                    "the SOAP note for completeness, organization, accuracy, and "
                    "coding support (ICD-10 codes present and appropriate). "
                    "Identify missing elements and provide suggestions."
                ),
            },
            {
                "role": "user",
                "content": f"Evaluate this SOAP note:\n\n{soap_json}",
            },
        ],
        response_format=DocumentationQuality,
    )
    return completion.choices[0].message.parsed


def display_soap_note(soap: SOAPNote, title: str = ""):
    """Display formatted SOAP note."""
    if title:
        print(f"\n  {'═' * 55}")
        print(f"  SOAP NOTE — {title}")
        print(f"  {'═' * 55}")

    if soap.patient_name:
        print(f"\n  Patient: {soap.patient_name}")
    if soap.visit_date:
        print(f"  Date:    {soap.visit_date}")
    if soap.visit_type:
        print(f"  Type:    {soap.visit_type}")
    if soap.provider:
        print(f"  Provider: {soap.provider}")

    print(f"\n  ┌── SUBJECTIVE ──────────────────────────────────────────")
    print(f"  │ {soap.subjective[:300]}")
    if len(soap.subjective) > 300:
        print(f"  │ ...")

    if soap.vitals:
        v = soap.vitals
        print(f"\n  ┌── VITALS ─────────────────────────────────────────────")
        vitals_parts = []
        if v.blood_pressure:
            vitals_parts.append(f"BP {v.blood_pressure}")
        if v.heart_rate:
            vitals_parts.append(f"HR {v.heart_rate}")
        if v.temperature:
            vitals_parts.append(f"T {v.temperature}")
        if v.respiratory_rate:
            vitals_parts.append(f"RR {v.respiratory_rate}")
        if v.oxygen_saturation:
            vitals_parts.append(f"SpO2 {v.oxygen_saturation}%")
        if v.weight:
            vitals_parts.append(f"Wt {v.weight}")
        print(f"  │ {' | '.join(vitals_parts)}")

    print(f"\n  ┌── OBJECTIVE ──────────────────────────────────────────")
    print(f"  │ {soap.objective[:300]}")
    if len(soap.objective) > 300:
        print(f"  │ ...")

    print(f"\n  ┌── ASSESSMENT ─────────────────────────────────────────")
    print(f"  │ {soap.assessment[:200]}")

    if soap.assessment_plans:
        print(f"\n  ┌── ASSESSMENT & PLAN (Problem-Based) ─────────────────")
        for i, ap in enumerate(soap.assessment_plans, 1):
            icd = f" ({ap.icd10})" if ap.icd10 else ""
            print(f"  │ {i}. {ap.diagnosis}{icd}")
            print(f"  │    Assessment: {ap.assessment}")
            for item in ap.plan_items:
                print(f"  │    → {item}")

    if soap.plan_summary:
        print(f"\n  ┌── PLAN SUMMARY ───────────────────────────────────────")
        for item in soap.plan_summary:
            print(f"  │ • {item}")

    if soap.follow_up:
        print(f"\n  Follow-up: {soap.follow_up}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run clinical scribe pipeline across encounter types."""
    print("=" * 60)
    print("EXERCISE 2: Clinical Scribe — Audio to Structured Documentation")
    print("=" * 60)

    for scenario in ENCOUNTER_SCENARIOS:
        print(f"\n\n{'#' * 60}")
        print(f"  {scenario['title']}")
        print(f"{'#' * 60}")

        # Step 1: Transcribe
        print(f"\n  Step 1: Generating transcription...")
        start = time.time()
        transcript = simulate_transcription(scenario["audio_scenario"])
        t1 = time.time() - start
        print(f"  ✓ Transcript ({len(transcript)} chars, {t1:.2f}s)")
        print(f"  Preview: {transcript[:150]}...\n")

        # Step 2: Generate SOAP note
        print(f"  Step 2: Generating structured SOAP note...")
        start = time.time()
        soap = generate_soap_note(transcript)
        t2 = time.time() - start
        print(f"  ✓ SOAP note generated ({t2:.2f}s)")

        # Display
        display_soap_note(soap, title=scenario["title"])

        # Step 3: Quality check
        print(f"\n  Step 3: Quality evaluation...")
        start = time.time()
        quality = evaluate_documentation(soap)
        t3 = time.time() - start

        print(f"\n  ┌── QUALITY ASSESSMENT ──────────────────────────────────")
        for dim, score in [
            ("Completeness", quality.completeness_score),
            ("Organization", quality.organization_score),
            ("Medical Accuracy", quality.medical_accuracy_score),
            ("Coding Support", quality.coding_support_score),
        ]:
            bar = "█" * score + "░" * (5 - score)
            print(f"  │ {dim:<20} {bar} {score}/5")

        if quality.missing_elements:
            print(f"  │\n  │ Missing elements:")
            for elem in quality.missing_elements:
                print(f"  │   ⚠ {elem}")

        if quality.suggestions:
            print(f"  │\n  │ Suggestions:")
            for sug in quality.suggestions:
                print(f"  │   💡 {sug}")

        total_time = t1 + t2 + t3
        print(f"\n  Pipeline timing: Transcribe {t1:.1f}s → SOAP {t2:.1f}s → "
              f"QA {t3:.1f}s = {total_time:.1f}s total")

    # Summary
    print(f"\n\n{'=' * 60}")
    print("Exercise complete! Key takeaways:")
    print("  • Audio → transcription → structured output is a powerful pipeline")
    print("  • Problem-based A&P format supports clinical decision-making")
    print("  • Quality evaluation catches missing documentation elements")
    print("  • Clinical scribes save providers 1-2 hours of documentation daily")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
