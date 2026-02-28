"""
Exercise 1: Clinical Data Extraction with Pydantic Models

Skills practiced:
  - Designing Pydantic models for healthcare data (Patient, Medication, Diagnosis, Vitals)
  - Prompting LLMs to produce schema-compliant JSON
  - Parsing and validating LLM output against strict models
  - Handling missing or ambiguous clinical data gracefully

Healthcare context:
  Clinical notes are rich but unstructured. A physician writes "pt is 67yo F with
  DM2 on metformin 1000 BID, A1c 7.2, BP 142/88." A downstream system needs this
  as structured data: name, age, sex, medications (with dose and frequency), lab
  results, and vitals — all validated.

  This exercise builds the full extraction pipeline: define models, prompt the LLM,
  parse and validate, then display clean structured data. You'll handle edge cases
  like missing vitals, ambiguous dosages, and abbreviation expansion.
"""

import os
import json
from typing import Optional
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"


# ============================================================
# Pydantic Models for Clinical Data
# ============================================================

class Medication(BaseModel):
    """A single medication with dosage information."""
    name: str = Field(..., description="Generic medication name")
    dose: str = Field(..., description="Dose with units, e.g. '500mg'")
    frequency: str = Field(..., description="Frequency, e.g. 'BID', 'daily', 'Q8H'")
    route: str = Field("oral", description="Route of administration")
    indication: Optional[str] = Field(None, description="What this med is prescribed for")

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Medication name cannot be empty")
        return v.strip()


class VitalSigns(BaseModel):
    """Patient vital signs at time of encounter."""
    systolic_bp: Optional[int] = Field(None, ge=50, le=300, description="Systolic BP in mmHg")
    diastolic_bp: Optional[int] = Field(None, ge=20, le=200, description="Diastolic BP in mmHg")
    heart_rate: Optional[int] = Field(None, ge=20, le=250, description="Heart rate in bpm")
    respiratory_rate: Optional[int] = Field(None, ge=4, le=60, description="Respiratory rate")
    temperature_f: Optional[float] = Field(None, ge=90.0, le=110.0, description="Temp in °F")
    spo2_percent: Optional[int] = Field(None, ge=50, le=100, description="SpO2 percentage")
    weight_kg: Optional[float] = Field(None, ge=1.0, le=500.0, description="Weight in kg")


class Diagnosis(BaseModel):
    """A clinical diagnosis or problem list entry."""
    condition: str = Field(..., description="Diagnosis or condition name")
    icd10_code: Optional[str] = Field(None, description="ICD-10 code if identifiable")
    status: str = Field("active", description="active, resolved, or suspected")
    onset: Optional[str] = Field(None, description="When the condition began")

    @field_validator("status")
    @classmethod
    def valid_status(cls, v: str) -> str:
        v_lower = v.strip().lower()
        if v_lower not in ("active", "resolved", "suspected"):
            raise ValueError(f"status must be active, resolved, or suspected; got '{v}'")
        return v_lower


class LabResult(BaseModel):
    """A laboratory test result."""
    test_name: str = Field(..., description="Name of the lab test")
    value: str = Field(..., description="Result value with units")
    reference_range: Optional[str] = Field(None, description="Normal reference range")
    is_abnormal: bool = Field(False, description="Whether result is outside normal range")


class PatientRecord(BaseModel):
    """Complete structured patient record extracted from a clinical note."""
    patient_name: str = Field(..., description="Full patient name")
    age: int = Field(..., ge=0, le=150, description="Patient age in years")
    sex: str = Field(..., description="male or female")
    date_of_birth: Optional[str] = Field(None, description="DOB in YYYY-MM-DD format")
    mrn: Optional[str] = Field(None, description="Medical record number")
    chief_complaint: str = Field(..., description="Primary reason for visit")
    medications: list[Medication] = Field(default_factory=list)
    vitals: Optional[VitalSigns] = None
    diagnoses: list[Diagnosis] = Field(default_factory=list)
    labs: list[LabResult] = Field(default_factory=list)
    plan: list[str] = Field(default_factory=list, description="Treatment plan items")
    allergies: list[str] = Field(default_factory=list, description="Known allergies")

    @field_validator("sex")
    @classmethod
    def normalize_sex(cls, v: str) -> str:
        v_lower = v.strip().lower()
        if v_lower not in ("male", "female"):
            raise ValueError(f"sex must be 'male' or 'female', got '{v}'")
        return v_lower


# ============================================================
# Sample Clinical Notes
# ============================================================

CLINICAL_NOTES = [
    {
        "id": "note_A",
        "text": (
            "Patient: Robert Chen, 72-year-old male. DOB: 1953-04-18. MRN: 5509132. "
            "Allergies: Penicillin (rash), Sulfa drugs. "
            "Chief Complaint: Worsening dyspnea on exertion for the past 3 weeks. "
            "PMH: COPD (Gold Stage III), HTN, atrial fibrillation. "
            "Medications: Tiotropium 18mcg inhaled daily, Albuterol 2 puffs Q4H PRN, "
            "Lisinopril 10mg daily, Warfarin 5mg daily, Prednisone 10mg daily (started "
            "1 week ago for COPD exacerbation). "
            "Vitals: BP 136/84 mmHg, HR 92 irregular, RR 24, Temp 99.1°F, SpO2 89% on RA. "
            "Labs: WBC 11.2, CRP 4.8 mg/L (elevated), INR 2.3, BNP 280 pg/mL (mildly elevated). "
            "Assessment: Acute COPD exacerbation, possible pneumonia. A-fib with adequate rate "
            "control. "
            "Plan: Chest X-ray, sputum culture, increase prednisone to 40mg daily for 5 days, "
            "add azithromycin 500mg day 1 then 250mg days 2-5, continue bronchodilators, "
            "monitor INR closely due to azithromycin-warfarin interaction, follow-up in 3 days."
        ),
    },
    {
        "id": "note_B",
        "text": (
            "Pt: Sarah Johnson, 38yo F, DOB 1987-12-05, MRN 8821456. "
            "CC: Severe headache x 3 days, photophobia, nausea. No known allergies. "
            "Hx: Migraine with aura diagnosed age 22, depression. "
            "Meds: Sumatriptan 100mg PRN, Escitalopram 20mg daily. "
            "VS: BP 122/78, HR 76, RR 16, T 98.4F, O2 99%. Wt: 68 kg. "
            "Neuro exam: no focal deficits, photophobia present, no nuchal rigidity. "
            "Assessment: Intractable migraine, failed outpatient management. "
            "Plan: Ketorolac 30mg IV, Metoclopramide 10mg IV, IV fluids NS 1L, "
            "dark quiet room, reassess in 2 hours, consider DHE protocol if no relief, "
            "neurology referral for prophylaxis evaluation."
        ),
    },
    {
        "id": "note_C",
        "text": (
            "Patient: David Okafor, 61 y/o male, DOB: 1964-06-30, MRN: 2204817. "
            "Allergies: NKDA. "
            "CC: Routine follow-up for diabetes management. "
            "PMH: Type 2 DM x 12 years, Obesity (BMI 34.2), Hyperlipidemia, GERD. "
            "Current Meds: Metformin 1000mg BID, Glipizide 10mg daily, Atorvastatin 80mg "
            "daily, Omeprazole 20mg daily, Semaglutide 0.5mg SubQ weekly. "
            "Vitals: BP 128/82, HR 78, RR 14, Temp 98.6°F, SpO2 98%, Wt 104 kg. "
            "Labs: HbA1c 7.8% (target <7%), fasting glucose 156 mg/dL, LDL 92 mg/dL, "
            "eGFR 72 mL/min, microalbumin/creatinine ratio 45 mg/g (mildly elevated). "
            "Assessment: DM2 suboptimally controlled. Early diabetic nephropathy suspected. "
            "CKD Stage 2. Hyperlipidemia at goal. "
            "Plan: Increase semaglutide to 1.0mg weekly, add lisinopril 5mg daily for "
            "renoprotection, repeat microalbumin in 3 months, dietary counseling, "
            "continue statin, follow-up 3 months."
        ),
    },
]


# ============================================================
# Extraction Functions
# ============================================================

def build_extraction_prompt() -> str:
    """Build the system prompt with the full JSON schema."""
    schema = PatientRecord.model_json_schema()
    return f"""You are a clinical data extraction system. Extract ALL structured data
from the clinical note and return valid JSON matching this schema:

{json.dumps(schema, indent=2)}

Rules:
1. Extract every medication mentioned with name, dose, frequency, and route.
2. Parse vitals into individual numeric fields (systolic_bp and diastolic_bp as integers).
3. List all diagnoses with status (active, resolved, or suspected).
4. Extract lab results with values, reference ranges if mentioned, and abnormal flag.
5. List each plan item separately.
6. sex must be exactly "male" or "female".
7. date_of_birth in YYYY-MM-DD format.
8. If a field value is not found in the note, use null.
9. Expand medical abbreviations in condition names (e.g., HTN → Hypertension).

Return ONLY valid JSON. No explanation or markdown."""


def extract_patient_record(note_text: str, max_retries: int = 3) -> Optional[PatientRecord]:
    """
    Extract a structured PatientRecord from a clinical note.
    Retries with error feedback on validation failure.
    """
    system_prompt = build_extraction_prompt()
    last_error = None

    for attempt in range(1, max_retries + 1):
        user_msg = note_text
        if last_error:
            user_msg += (
                f"\n\n[VALIDATION FAILED — ATTEMPT {attempt}]\n"
                f"Errors: {last_error}\n"
                f"Fix these issues and return corrected JSON."
            )

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [Attempt {attempt}] API error: {e}")
            last_error = str(e)
            continue

        # Parse and validate
        try:
            record = PatientRecord.model_validate_json(raw)
            if attempt > 1:
                print(f"  ✓ Succeeded on attempt {attempt}")
            return record
        except ValidationError as e:
            last_error = "; ".join(
                f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
                for err in e.errors()
            )
            print(f"  [Attempt {attempt}] Validation errors: {last_error}")

    print(f"  ✗ Failed after {max_retries} attempts")
    return None


def display_record(record: PatientRecord) -> None:
    """Pretty-print a PatientRecord."""
    print(f"\n  Patient:  {record.patient_name}")
    print(f"  Age/Sex:  {record.age} / {record.sex}")
    print(f"  DOB:      {record.date_of_birth or 'N/A'}")
    print(f"  MRN:      {record.mrn or 'N/A'}")
    print(f"  CC:       {record.chief_complaint}")
    print(f"  Allergies:{', '.join(record.allergies) if record.allergies else ' None'}")

    if record.vitals:
        v = record.vitals
        print(f"  Vitals:   BP {v.systolic_bp}/{v.diastolic_bp}, HR {v.heart_rate}, "
              f"RR {v.respiratory_rate}, T {v.temperature_f}°F, SpO2 {v.spo2_percent}%"
              + (f", Wt {v.weight_kg}kg" if v.weight_kg else ""))

    print(f"  Medications ({len(record.medications)}):")
    for m in record.medications:
        indication = f" [{m.indication}]" if m.indication else ""
        print(f"    - {m.name} {m.dose} {m.frequency} ({m.route}){indication}")

    print(f"  Diagnoses ({len(record.diagnoses)}):")
    for d in record.diagnoses:
        code = f" [{d.icd10_code}]" if d.icd10_code else ""
        onset = f", onset: {d.onset}" if d.onset else ""
        print(f"    - {d.condition} ({d.status}){code}{onset}")

    print(f"  Labs ({len(record.labs)}):")
    for lab in record.labs:
        flag = " ⚠ ABNORMAL" if lab.is_abnormal else ""
        ref = f" (ref: {lab.reference_range})" if lab.reference_range else ""
        print(f"    - {lab.test_name}: {lab.value}{ref}{flag}")

    print(f"  Plan ({len(record.plan)} items):")
    for item in record.plan:
        print(f"    - {item}")


# ============================================================
# Main
# ============================================================

def main():
    """Extract structured patient records from all sample clinical notes."""
    print("=" * 65)
    print("Exercise 1: Clinical Data Extraction with Pydantic Models")
    print("=" * 65)

    results = []
    for note in CLINICAL_NOTES:
        print(f"\n--- Processing {note['id']} ---")
        record = extract_patient_record(note["text"])
        if record:
            display_record(record)
            results.append(record)
        else:
            print(f"  Could not extract structured data from {note['id']}")

    # Summary
    print("\n" + "=" * 65)
    print("EXTRACTION SUMMARY")
    print("=" * 65)
    print(f"Notes processed:    {len(CLINICAL_NOTES)}")
    print(f"Successful:         {len(results)}")
    print(f"Failed:             {len(CLINICAL_NOTES) - len(results)}")

    total_meds = sum(len(r.medications) for r in results)
    total_dx = sum(len(r.diagnoses) for r in results)
    total_labs = sum(len(r.labs) for r in results)
    print(f"Total medications:  {total_meds}")
    print(f"Total diagnoses:    {total_dx}")
    print(f"Total lab results:  {total_labs}")

    # Export to JSON
    output = [r.model_dump() for r in results]
    print(f"\nJSON export preview (first record):")
    print(json.dumps(output[0], indent=2, default=str) if output else "No records")


if __name__ == "__main__":
    main()
