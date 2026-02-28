"""
Exercise 1: Clinical Data Extraction with Structured Outputs
=============================================================
Extract structured clinical data from free-text clinical notes using
OpenAI's structured outputs. The schema includes nested patient data:
Patient(name, dob, gender, medications: List[Medication], diagnoses: List[Diagnosis])

Learning Objectives:
  - Use client.beta.chat.completions.parse() for type-safe extraction
  - Define nested Pydantic models for complex medical data
  - Handle missing/ambiguous data gracefully
  - Validate extracted structured data

Usage:
  python exercise_1_clinical_extraction.py
"""

import json
import os
import time
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI()
MODEL = "gpt-4o"


# ============================================================================
# SCHEMAS
# ============================================================================

class Medication(BaseModel):
    """A medication with dosing information."""
    name: str = Field(..., description="Medication name")
    dose: Optional[str] = Field(None, description="Dose, e.g. '500 mg'")
    frequency: Optional[str] = Field(None, description="Frequency, e.g. 'twice daily'")
    route: Optional[str] = Field(None, description="Route, e.g. 'oral'")
    indication: Optional[str] = Field(None, description="Reason for medication")
    start_date: Optional[str] = Field(None, description="When medication was started")


class Diagnosis(BaseModel):
    """A clinical diagnosis."""
    name: str = Field(..., description="Diagnosis name")
    icd10_code: Optional[str] = Field(None, description="ICD-10 code if available")
    status: Optional[str] = Field(None, description="active, resolved, or chronic")
    date_diagnosed: Optional[str] = Field(None, description="When diagnosed")
    notes: Optional[str] = Field(None, description="Additional clinical notes")


class Allergy(BaseModel):
    """A drug or environmental allergy."""
    allergen: str = Field(..., description="The allergen")
    reaction: Optional[str] = Field(None, description="Type of reaction")
    severity: Optional[str] = Field(None, description="mild, moderate, or severe")


class Patient(BaseModel):
    """Complete structured patient record extracted from clinical notes."""
    name: str = Field(..., description="Patient full name")
    date_of_birth: Optional[str] = Field(None, description="Date of birth")
    age: Optional[int] = Field(None, description="Age in years")
    gender: Optional[str] = Field(None, description="Patient gender")
    medical_record_number: Optional[str] = Field(None, description="MRN if available")
    chief_complaint: Optional[str] = Field(None, description="Primary reason for visit")
    medications: List[Medication] = Field(
        default_factory=list, description="Current medications"
    )
    diagnoses: List[Diagnosis] = Field(
        default_factory=list, description="Clinical diagnoses"
    )
    allergies: List[Allergy] = Field(
        default_factory=list, description="Known allergies"
    )
    vitals_summary: Optional[str] = Field(None, description="Summary of vital signs")


# ============================================================================
# SAMPLE CLINICAL NOTES
# ============================================================================

SAMPLE_NOTES = [
    {
        "title": "Primary Care Visit",
        "text": """
        PATIENT: Johnson, Elizabeth A.
        DOB: 03/15/1958   MRN: 4478291   Gender: Female   Age: 67

        CHIEF COMPLAINT: Follow-up for hypertension and type 2 diabetes.

        VITAL SIGNS: BP 142/88, HR 76, Temp 98.4°F, Wt 187 lbs, BMI 31.2

        ALLERGIES: Penicillin (rash), Sulfa drugs (anaphylaxis - severe)

        CURRENT MEDICATIONS:
        1. Metformin 1000 mg PO BID — diabetes, started 2019
        2. Lisinopril 20 mg PO daily — hypertension, started 2018
        3. Atorvastatin 40 mg PO QHS — hyperlipidemia, started 2020
        4. Aspirin 81 mg PO daily — cardiovascular prevention
        5. Metoprolol succinate 50 mg PO daily — rate control, started 2021

        ASSESSMENT/DIAGNOSES:
        1. Type 2 diabetes mellitus (E11.65) — chronic, HbA1c 7.4%, above goal
        2. Essential hypertension (I10) — chronic, not at goal today
        3. Hyperlipidemia (E78.5) — chronic, lipids well controlled
        4. Obesity (E66.01) — active, counseled on diet and exercise
        5. History of atrial fibrillation (I48.91) — currently rate-controlled

        PLAN:
        - Increase metformin to 1500 mg BID if tolerated
        - Add amlodipine 5 mg daily for BP control
        - Continue statin, aspirin, metoprolol
        - Recheck labs in 3 months (HbA1c, CMP, lipid panel)
        - Dietary counseling referral
        """,
    },
    {
        "title": "Emergency Department Note",
        "text": """
        Patient Name: Marcus Rivera
        Age: 34    Sex: Male    DOB: 07/22/1991

        Chief Complaint: Severe chest pain x 2 hours

        HPI: 34 y/o male with no significant PMH presents with acute onset
        substernal chest pain radiating to left arm, associated with
        diaphoresis and shortness of breath. Pain started while at rest.
        Denies recent illness, trauma, or drug use. No prior episodes.

        Vitals: BP 158/94, HR 110, RR 22, Temp 98.9°F, SpO2 97% on RA

        Allergies: NKDA (No Known Drug Allergies)

        Medications: None

        ECG: ST elevation in leads II, III, aVF (inferior STEMI)

        Assessment:
        1. Acute ST-elevation myocardial infarction, inferior wall (I21.19) — emergent
        2. Hypertension, undiagnosed (I10) — new finding

        ED Medications Administered:
        1. Aspirin 325 mg PO x1 — ACS protocol
        2. Heparin bolus 60 units/kg IV then 12 units/kg/hr — anticoagulation
        3. Nitroglycerin 0.4 mg SL x1 — chest pain
        4. Morphine 4 mg IV — pain management

        Plan: Emergent cardiac catheterization. Cardiology consulted.
        """,
    },
    {
        "title": "Psychiatry Outpatient Note",
        "text": """
        Patient: Sarah Kim, 29F, DOB 11/03/1996

        CC: Follow-up for depression and anxiety management.

        Current Medications:
        - Sertraline 100 mg daily (started 6 months ago for MDD)
        - Buspirone 10 mg BID (added 3 months ago for GAD)
        - Oral contraceptive (Yaz) daily

        Allergies: Latex (contact dermatitis - mild)

        Psychiatric Diagnoses:
        1. Major depressive disorder, recurrent, moderate (F33.1) — active,
           improving on current regimen. PHQ-9 score improved from 18 to 9.
        2. Generalized anxiety disorder (F41.1) — active, GAD-7 improved
           from 15 to 8 since adding buspirone.
        3. Insomnia disorder (G47.00) — resolved with sleep hygiene measures.

        Vitals: BP 118/72, HR 68, Wt 135 lbs

        Plan: Continue current medications. Follow up in 8 weeks.
        Consider dose increase of sertraline to 150 mg if plateau.
        """,
    },
]


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_patient_data(clinical_note: str) -> Patient:
    """
    Extract structured patient data from a free-text clinical note.
    Uses OpenAI's structured outputs for guaranteed schema compliance.
    """
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical data extraction system. Extract ALL structured "
                    "patient information from the clinical note. Rules:\n"
                    "1. Only include information explicitly stated in the note\n"
                    "2. For medications, capture dose, frequency, route, and indication\n"
                    "3. For diagnoses, include ICD-10 codes if provided\n"
                    "4. Mark diagnosis status as active, chronic, or resolved\n"
                    "5. Include all allergies with reaction type and severity\n"
                    "6. If information is not mentioned, leave the field as null"
                ),
            },
            {
                "role": "user",
                "content": f"Extract all patient data from this note:\n\n{clinical_note}",
            },
        ],
        response_format=Patient,
    )
    return completion.choices[0].message.parsed


def display_patient(patient: Patient, title: str = ""):
    """Display extracted patient data in a formatted view."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    print(f"\n  PATIENT INFORMATION")
    print(f"  {'─' * 40}")
    print(f"  Name:   {patient.name}")
    print(f"  DOB:    {patient.date_of_birth or '—'}")
    print(f"  Age:    {patient.age or '—'}")
    print(f"  Gender: {patient.gender or '—'}")
    print(f"  MRN:    {patient.medical_record_number or '—'}")
    if patient.chief_complaint:
        print(f"  CC:     {patient.chief_complaint}")
    if patient.vitals_summary:
        print(f"  Vitals: {patient.vitals_summary}")

    if patient.allergies:
        print(f"\n  ALLERGIES ({len(patient.allergies)})")
        print(f"  {'─' * 40}")
        for a in patient.allergies:
            severity = f" [{a.severity}]" if a.severity else ""
            reaction = f" — {a.reaction}" if a.reaction else ""
            print(f"  ⚠ {a.allergen}{reaction}{severity}")

    if patient.medications:
        print(f"\n  MEDICATIONS ({len(patient.medications)})")
        print(f"  {'─' * 40}")
        for i, med in enumerate(patient.medications, 1):
            dose_info = f" {med.dose}" if med.dose else ""
            freq_info = f" {med.frequency}" if med.frequency else ""
            route_info = f" {med.route}" if med.route else ""
            print(f"  {i}. {med.name}{dose_info}{freq_info}{route_info}")
            if med.indication:
                print(f"     Indication: {med.indication}")

    if patient.diagnoses:
        print(f"\n  DIAGNOSES ({len(patient.diagnoses)})")
        print(f"  {'─' * 40}")
        for i, dx in enumerate(patient.diagnoses, 1):
            icd = f" ({dx.icd10_code})" if dx.icd10_code else ""
            status = f" [{dx.status}]" if dx.status else ""
            print(f"  {i}. {dx.name}{icd}{status}")
            if dx.notes:
                print(f"     Notes: {dx.notes}")


def validate_extraction(patient: Patient) -> dict:
    """Validate the quality of extracted data."""
    issues = []
    scores = {}

    # Check required fields
    if not patient.name:
        issues.append("Missing patient name")
    scores["name"] = 1.0 if patient.name else 0.0

    # Check medications have key fields
    med_scores = []
    for med in patient.medications:
        fields_present = sum([
            bool(med.name), bool(med.dose), bool(med.frequency),
        ])
        med_scores.append(fields_present / 3.0)
    scores["medication_completeness"] = (
        sum(med_scores) / len(med_scores) if med_scores else 0.0
    )

    # Check diagnoses have codes
    dx_with_codes = sum(1 for dx in patient.diagnoses if dx.icd10_code)
    scores["diagnosis_coding"] = (
        dx_with_codes / len(patient.diagnoses) if patient.diagnoses else 0.0
    )

    # Overall score
    scores["overall"] = sum(scores.values()) / len(scores)

    return {"scores": scores, "issues": issues}


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Extract structured data from multiple clinical notes."""
    print("=" * 60)
    print("EXERCISE 1: Clinical Data Extraction with Structured Outputs")
    print("=" * 60)

    for note_info in SAMPLE_NOTES:
        print(f"\n\n{'#' * 60}")
        print(f"  Processing: {note_info['title']}")
        print(f"{'#' * 60}")

        print(f"\n--- Raw Note (first 200 chars) ---")
        print(f"  {note_info['text'].strip()[:200]}...")

        # Extract
        start = time.time()
        patient = extract_patient_data(note_info["text"])
        elapsed = time.time() - start

        # Display
        display_patient(patient, title=f"Extracted Data — {note_info['title']}")

        # Validate
        validation = validate_extraction(patient)
        print(f"\n  VALIDATION")
        print(f"  {'─' * 40}")
        for metric, score in validation["scores"].items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"  {metric:<30} {bar} {score:.0%}")

        if validation["issues"]:
            print(f"\n  Issues found:")
            for issue in validation["issues"]:
                print(f"    ⚠ {issue}")

        print(f"\n  [Extraction time: {elapsed:.2f}s]")

        # Show JSON
        print(f"\n--- JSON Output ---")
        print(json.dumps(patient.model_dump(), indent=2, default=str)[:500])
        if len(json.dumps(patient.model_dump())) > 500:
            print("  ... (truncated)")

    print(f"\n\n{'=' * 60}")
    print("Exercise complete! Key takeaways:")
    print("  • Structured outputs guarantee valid JSON every time")
    print("  • Pydantic models provide type-safe extraction")
    print("  • Nested schemas handle complex medical data structures")
    print("  • Validation catches completeness issues post-extraction")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
