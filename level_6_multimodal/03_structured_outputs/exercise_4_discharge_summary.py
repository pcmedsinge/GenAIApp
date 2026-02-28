"""
Exercise 4: Structured Discharge Summary Generation
=====================================================
Generate structured discharge summaries from clinical notes. Includes
patient demographics, admission/discharge dates, diagnoses, procedures,
medications at discharge, and follow-up instructions.

Schema: DischargeSummary(patient, admission_date, discharge_date, diagnoses,
        procedures, medications_at_discharge, follow_up_instructions)

Learning Objectives:
  - Model complex discharge summary schemas
  - Extract temporal information (dates, durations)
  - Handle medication reconciliation (admission vs discharge)
  - Generate actionable follow-up instructions

Usage:
  python exercise_4_discharge_summary.py
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
# SCHEMAS
# ============================================================================

class PatientDemographics(BaseModel):
    """Patient demographic information."""
    name: str = Field(..., description="Full name")
    age: Optional[int] = Field(None, description="Age at discharge")
    gender: Optional[str] = Field(None, description="Gender")
    date_of_birth: Optional[str] = Field(None, description="DOB")
    medical_record_number: Optional[str] = Field(None, description="MRN")


class DischargeDiagnosis(BaseModel):
    """A discharge diagnosis with coding."""
    name: str = Field(..., description="Diagnosis name")
    icd10_code: Optional[str] = Field(None, description="ICD-10 code")
    is_primary: bool = Field(False, description="Whether this is the principal diagnosis")
    present_on_admission: Optional[bool] = Field(
        None, description="Whether the condition was present on admission"
    )


class Procedure(BaseModel):
    """A procedure performed during the hospitalization."""
    name: str = Field(..., description="Procedure name")
    date: Optional[str] = Field(None, description="Procedure date")
    cpt_code: Optional[str] = Field(None, description="CPT code if available")
    provider: Optional[str] = Field(None, description="Performing provider")
    notes: Optional[str] = Field(None, description="Procedure notes")


class DischargeMedication(BaseModel):
    """A medication prescribed at discharge."""
    name: str = Field(..., description="Medication name")
    dose: str = Field(..., description="Dose, e.g. '500 mg'")
    frequency: str = Field(..., description="Frequency, e.g. 'twice daily'")
    route: str = Field(..., description="Route, e.g. 'oral'")
    duration: Optional[str] = Field(None, description="Duration if time-limited")
    is_new: bool = Field(False, description="Whether this is a newly started medication")
    is_changed: bool = Field(False, description="Whether dose/frequency was changed")
    special_instructions: Optional[str] = Field(None, description="Special instructions")


class FollowUpInstruction(BaseModel):
    """A follow-up instruction for after discharge."""
    type: Literal[
        "appointment", "lab_work", "imaging", "activity_restriction",
        "diet", "wound_care", "medication_monitoring", "other"
    ] = Field(..., description="Type of follow-up")
    description: str = Field(..., description="Detailed instruction")
    timeframe: Optional[str] = Field(None, description="When to complete")
    provider: Optional[str] = Field(None, description="Provider/specialist for appointment")
    urgency: Optional[Literal["routine", "urgent", "as_needed"]] = Field(
        None, description="Urgency level"
    )


class HospitalCourse(BaseModel):
    """Summary of the hospital course."""
    brief_summary: str = Field(..., description="1-2 sentence summary of hospitalization")
    key_events: List[str] = Field(
        default_factory=list, description="Key events during hospitalization"
    )
    consultations: List[str] = Field(
        default_factory=list, description="Specialist consultations"
    )


class DischargeSummary(BaseModel):
    """Complete structured discharge summary."""
    patient: PatientDemographics = Field(..., description="Patient demographics")
    admission_date: str = Field(..., description="Date of admission")
    discharge_date: str = Field(..., description="Date of discharge")
    length_of_stay_days: Optional[int] = Field(None, description="Length of stay in days")
    admitting_diagnosis: Optional[str] = Field(None, description="Admitting diagnosis")
    discharge_disposition: Optional[Literal[
        "home", "home_with_services", "skilled_nursing", "rehab",
        "long_term_care", "hospice", "against_medical_advice", "expired"
    ]] = Field(None, description="Discharge disposition")
    hospital_course: HospitalCourse = Field(..., description="Hospital course summary")
    diagnoses: List[DischargeDiagnosis] = Field(
        ..., description="All discharge diagnoses"
    )
    procedures: List[Procedure] = Field(
        default_factory=list, description="Procedures performed"
    )
    medications_at_discharge: List[DischargeMedication] = Field(
        ..., description="Medication list at discharge"
    )
    follow_up_instructions: List[FollowUpInstruction] = Field(
        ..., description="Post-discharge instructions"
    )
    condition_at_discharge: Optional[Literal[
        "stable", "improved", "guarded", "critical"
    ]] = Field(None, description="Patient condition at discharge")


# ============================================================================
# SAMPLE DISCHARGE NOTES
# ============================================================================

SAMPLE_DISCHARGE_NOTES = [
    {
        "title": "Cardiac Admission",
        "text": """
        DISCHARGE SUMMARY

        Patient: Thomas Wright    MRN: DS-44201    DOB: 08/25/1958
        Age: 67    Gender: Male
        Admission Date: 11/10/2025    Discharge Date: 11/15/2025
        Admitting Diagnosis: Acute STEMI
        Attending: Dr. Angela Martinez, Cardiology

        HOSPITAL COURSE:
        Mr. Wright presented to the ED on 11/10/2025 with acute substernal
        chest pain and diaphoresis. ECG showed ST elevation in V1-V4 consistent
        with anterior STEMI. Troponin peaked at 18.5 ng/mL.

        Emergent cardiac catheterization on 11/10/2025 by Dr. Martinez revealed
        99% occlusion of the LAD. Successful PCI with drug-eluting stent placement.
        Post-procedure, patient was monitored in CCU for 48 hours.

        Echocardiogram on 11/12/2025 showed EF 40% with anterior wall hypokinesis.
        Started on guideline-directed medical therapy for HFrEF.

        Cardiology and cardiac rehab were consulted. Patient was mobilized
        progressively and tolerated well. Discharge on 11/15/2025 in stable condition.

        DISCHARGE DIAGNOSES:
        1. Acute ST-elevation myocardial infarction, anterior wall (I21.09) — PRIMARY
        2. Coronary artery disease, native vessel (I25.10)
        3. Heart failure with reduced ejection fraction (I50.22) — new diagnosis
        4. Hypertension (I10) — present on admission
        5. Type 2 diabetes mellitus (E11.65) — present on admission

        PROCEDURES:
        1. Left heart catheterization — 11/10/2025, Dr. Martinez
        2. PCI with DES to LAD — 11/10/2025, Dr. Martinez
        3. Transthoracic echocardiogram — 11/12/2025

        MEDICATIONS AT DISCHARGE:
        1. Aspirin 81 mg daily (continued)
        2. Ticagrelor 90 mg twice daily — NEW (do not stop without calling cardiologist)
        3. Metoprolol succinate 25 mg daily — NEW
        4. Lisinopril 5 mg daily — NEW (was on 10 mg, reduced due to low BP)
        5. Atorvastatin 80 mg at bedtime — NEW (high-intensity statin)
        6. Metformin 500 mg twice daily (continued, dose reduced from 1000 mg)
        7. Nitroglycerin 0.4 mg SL PRN chest pain — NEW

        FOLLOW-UP:
        1. Cardiology — Dr. Martinez in 1 week (urgent)
        2. Primary care — Dr. Chen in 2 weeks
        3. Cardiac rehabilitation — schedule within 2 weeks
        4. Labs: BMP, CBC, lipid panel in 1 week
        5. No driving for 1 week
        6. No heavy lifting (>10 lbs) for 4 weeks
        7. Heart-healthy diet, low sodium
        8. Call 911 if chest pain not relieved by 1 nitroglycerin tablet in 5 minutes

        Discharged home in stable condition.
        """,
    },
    {
        "title": "COPD Exacerbation",
        "text": """
        DISCHARGE SUMMARY

        Patient: Dorothy Baker    MRN: DS-55302    DOB: 01/12/1948
        Age: 77    Gender: Female
        Admission Date: 11/13/2025    Discharge Date: 11/17/2025
        Admitting Diagnosis: Acute COPD exacerbation
        Attending: Dr. James Park, Pulmonology

        HOSPITAL COURSE:
        Mrs. Baker presented with 3-day history of worsening dyspnea,
        increased sputum production (purulent), and fever to 101.2°F.
        CXR showed right lower lobe infiltrate. Admitted for acute COPD
        exacerbation with community-acquired pneumonia.

        Started on IV azithromycin and ceftriaxone, systemic corticosteroids
        (methylprednisolone 40 mg IV Q8H), and nebulized bronchodilators.
        Oxygen requirement peaked at 4L NC on day 2, weaned to room air by day 3.

        Infectious disease was consulted — sputum culture grew Streptococcus
        pneumoniae sensitive to current antibiotics. Transitioned to oral
        antibiotics on day 3. Corticosteroids tapered.

        Pulmonology was consulted for optimization of COPD regimen. Added
        roflumilast given frequent exacerbations (3rd in 12 months).

        Patient ambulating independently, SpO2 94% on room air at discharge.

        DISCHARGE DIAGNOSES:
        1. Acute exacerbation of COPD (J44.1) — PRIMARY
        2. Community-acquired pneumonia (J18.9)
        3. COPD, severe (J44.1) — chronic
        4. Oxygen-dependent at baseline (Z99.81) — resolved this admission
        5. Osteoporosis (M81.0) — chronic
        6. GERD (K21.0) — chronic

        PROCEDURES:
        1. Bronchoscopy with BAL — 11/14/2025, Dr. Park

        MEDICATIONS AT DISCHARGE:
        1. Amoxicillin/clavulanate 875/125 mg twice daily for 5 more days — NEW
        2. Prednisone 40 mg daily x 3 days, then 20 mg x 2 days, then stop — NEW
        3. Tiotropium 18 mcg inhaled daily (continued)
        4. Fluticasone/salmeterol 250/50 inhaled twice daily (continued)
        5. Albuterol MDI 2 puffs Q4-6H PRN (continued)
        6. Roflumilast 500 mcg daily — NEW
        7. Alendronate 70 mg weekly (continued)
        8. Omeprazole 20 mg daily (continued)
        9. Calcium/Vitamin D 600/400 mg daily (continued)

        FOLLOW-UP:
        1. Pulmonology — Dr. Park in 2 weeks (urgent)
        2. Primary care — 1 month
        3. Chest X-ray — repeat in 6 weeks to confirm infiltrate resolution
        4. Smoking cessation resources provided (patient reports quitting 2 years ago)
        5. Pneumococcal and influenza vaccines — discuss at PCP visit
        6. Activity: walk daily, increase gradually
        7. Call or go to ER if fever >101°F, worsening shortness of breath,
           or coughing up blood

        Discharged home with home health services. Condition: improved.
        """,
    },
]


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_discharge_summary(note_text: str) -> DischargeSummary:
    """Extract a structured discharge summary from clinical text."""
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical documentation system that generates structured "
                    "discharge summaries. Extract ALL information from the note.\n\n"
                    "Rules:\n"
                    "1. Identify the primary (principal) diagnosis\n"
                    "2. Note which conditions were present on admission vs. new\n"
                    "3. For medications, flag which are new and which were changed\n"
                    "4. Categorize all follow-up instructions by type\n"
                    "5. Calculate length of stay from admission/discharge dates\n"
                    "6. Include all procedures with dates and providers\n"
                    "7. Determine discharge disposition from the note"
                ),
            },
            {
                "role": "user",
                "content": f"Generate structured discharge summary:\n\n{note_text}",
            },
        ],
        response_format=DischargeSummary,
    )
    return completion.choices[0].message.parsed


def display_discharge_summary(ds: DischargeSummary):
    """Display a discharge summary in formatted view."""
    print(f"\n  ╔{'═' * 58}╗")
    print(f"  ║  DISCHARGE SUMMARY                                       ║")
    print(f"  ╚{'═' * 58}╝")

    # Patient info
    p = ds.patient
    print(f"\n  Patient:    {p.name} ({p.gender or '—'}, age {p.age or '—'})")
    print(f"  MRN:        {p.medical_record_number or '—'}")
    print(f"  Admitted:   {ds.admission_date}")
    print(f"  Discharged: {ds.discharge_date} (LOS: {ds.length_of_stay_days or '—'} days)")
    print(f"  Disposition:{ds.discharge_disposition or '—'}")
    print(f"  Condition:  {ds.condition_at_discharge or '—'}")

    # Hospital course
    print(f"\n  HOSPITAL COURSE")
    print(f"  {'─' * 50}")
    print(f"  {ds.hospital_course.brief_summary}")
    if ds.hospital_course.key_events:
        print(f"\n  Key Events:")
        for event in ds.hospital_course.key_events:
            print(f"    • {event}")
    if ds.hospital_course.consultations:
        print(f"\n  Consultations: {', '.join(ds.hospital_course.consultations)}")

    # Diagnoses
    print(f"\n  DIAGNOSES ({len(ds.diagnoses)})")
    print(f"  {'─' * 50}")
    for i, dx in enumerate(ds.diagnoses, 1):
        primary = " ★ PRIMARY" if dx.is_primary else ""
        poa = " [POA]" if dx.present_on_admission else ""
        code = f" ({dx.icd10_code})" if dx.icd10_code else ""
        print(f"  {i}. {dx.name}{code}{primary}{poa}")

    # Procedures
    if ds.procedures:
        print(f"\n  PROCEDURES ({len(ds.procedures)})")
        print(f"  {'─' * 50}")
        for proc in ds.procedures:
            date = f" — {proc.date}" if proc.date else ""
            provider = f" by {proc.provider}" if proc.provider else ""
            print(f"  • {proc.name}{date}{provider}")

    # Medications
    print(f"\n  MEDICATIONS AT DISCHARGE ({len(ds.medications_at_discharge)})")
    print(f"  {'─' * 50}")
    for med in ds.medications_at_discharge:
        flags = []
        if med.is_new:
            flags.append("NEW")
        if med.is_changed:
            flags.append("CHANGED")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        duration = f" x {med.duration}" if med.duration else ""
        print(f"  💊 {med.name} {med.dose} {med.route} {med.frequency}{duration}{flag_str}")
        if med.special_instructions:
            print(f"     ⚠ {med.special_instructions}")

    # Follow-up
    print(f"\n  FOLLOW-UP INSTRUCTIONS ({len(ds.follow_up_instructions)})")
    print(f"  {'─' * 50}")
    type_icons = {
        "appointment": "📅", "lab_work": "🔬", "imaging": "📷",
        "activity_restriction": "🚫", "diet": "🥗", "wound_care": "🩹",
        "medication_monitoring": "💊", "other": "📋",
    }
    for fu in ds.follow_up_instructions:
        icon = type_icons.get(fu.type, "📋")
        timeframe = f" — {fu.timeframe}" if fu.timeframe else ""
        urgency = f" [{fu.urgency}]" if fu.urgency else ""
        print(f"  {icon} {fu.description}{timeframe}{urgency}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Process discharge summaries."""
    print("=" * 60)
    print("EXERCISE 4: Structured Discharge Summary Generation")
    print("=" * 60)

    for note_info in SAMPLE_DISCHARGE_NOTES:
        print(f"\n\n{'#' * 60}")
        print(f"  Processing: {note_info['title']}")
        print(f"{'#' * 60}")

        # Extract
        start = time.time()
        summary = extract_discharge_summary(note_info["text"])
        elapsed = time.time() - start

        # Display
        display_discharge_summary(summary)

        # Medication reconciliation summary
        new_meds = [m for m in summary.medications_at_discharge if m.is_new]
        changed_meds = [m for m in summary.medications_at_discharge if m.is_changed]
        continued_meds = [
            m for m in summary.medications_at_discharge
            if not m.is_new and not m.is_changed
        ]
        print(f"\n  MEDICATION RECONCILIATION")
        print(f"  {'─' * 50}")
        print(f"    New medications:       {len(new_meds)}")
        print(f"    Changed medications:   {len(changed_meds)}")
        print(f"    Continued medications: {len(continued_meds)}")
        print(f"    Total at discharge:    {len(summary.medications_at_discharge)}")

        # JSON output snippet
        print(f"\n  [Extraction time: {elapsed:.2f}s]")
        ds_json = summary.model_dump(exclude_none=True)
        print(f"  [JSON size: {len(json.dumps(ds_json))} bytes]")

    # Summary
    print(f"\n\n{'=' * 60}")
    print("Exercise complete! Key takeaways:")
    print("  • Discharge summaries require deeply nested schemas")
    print("  • Medication reconciliation (new/changed/continued) is critical")
    print("  • Follow-up categorization aids care coordination")
    print("  • Structured outputs enable automated transition-of-care workflows")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
