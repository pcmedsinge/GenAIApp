"""
Project: Structured Outputs and Validation
Objective: Ensure LLM outputs match expected schemas using JSON mode, Pydantic, and retry logic
Concepts: JSON mode, Pydantic validation, OpenAI Structured Outputs, retry with feedback

Healthcare Use Case: Extracting structured patient data from clinical notes

When LLM outputs feed into downstream systems — EHRs, clinical decision support,
billing, or lab integrations — they MUST conform to a precise schema. A missing
medication dosage, a malformed date, or an unexpected field name can break pipelines
or, worse, silently corrupt patient records.

This project demonstrates four progressively more robust strategies for ensuring
LLM outputs are structured, validated, and production-ready:

  1. JSON Mode          — Syntactically valid JSON (but schema not guaranteed)
  2. Pydantic Validation — Parse into typed Python models with constraints
  3. Structured Outputs  — API-level schema enforcement (guaranteed structure)
  4. Retry with Feedback — Self-healing loop when validation fails
"""

import os
import json
from datetime import date
from typing import Optional
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"


# ============================================================
# SAMPLE CLINICAL NOTES (shared test data)
# ============================================================

CLINICAL_NOTE_1 = """
Patient: Maria Santos
DOB: 03/15/1958
Visit Date: 01/22/2026

Chief Complaint: Follow-up for diabetes management.

HPI: 67-year-old female presenting for 3-month follow-up of Type 2 Diabetes Mellitus.
Patient reports improved dietary adherence. She has been walking 20 minutes daily.
Occasional episodes of dizziness when standing quickly.

Medications:
  - Metformin 1000mg PO BID
  - Lisinopril 20mg PO daily
  - Atorvastatin 40mg PO QHS

Vitals:
  BP: 138/82 mmHg
  HR: 76 bpm
  Temp: 98.6°F
  SpO2: 97%
  Weight: 172 lbs

Assessment:
  1. Type 2 Diabetes Mellitus — HbA1c improved to 7.2% from 7.8%
  2. Essential Hypertension — controlled
  3. Hyperlipidemia — stable on statin therapy

Plan:
  - Continue current medications
  - Recheck HbA1c in 3 months
  - Orthostatic vitals at next visit
  - Dietary counseling referral
"""

CLINICAL_NOTE_2 = """
Patient: James O'Brien
DOB: 11/02/1975
Visit Date: 01/25/2026

Chief Complaint: Acute onset chest pain.

HPI: 50-year-old male presenting to the ED with crushing substernal chest pain
radiating to the left arm and jaw, onset 3 hours ago while shoveling snow.
Pain is 8/10, associated with diaphoresis and nausea. History of hypertension
and smoking (1 PPD x 20 years). Family history of MI (father at age 52).

Medications:
  - Amlodipine 10mg PO daily
  - Aspirin 81mg PO daily

Vitals:
  BP: 162/98 mmHg
  HR: 102 bpm
  Temp: 98.4°F
  SpO2: 95%
  RR: 22 breaths/min

Assessment:
  1. Acute coronary syndrome — STEMI, troponin I elevated at 2.4 ng/mL
  2. Hypertension, uncontrolled
  3. Tobacco use disorder

Plan:
  - Emergent cardiac catheterization
  - Heparin drip, loading dose given
  - Clopidogrel 600mg loading dose
  - Morphine 4mg IV for pain
  - Cardiology stat consult
  - Admit to CCU
"""

CLINICAL_NOTE_3 = """
Patient: Aisha Patel
DOB: 07/20/2017
Visit Date: 01/28/2026

Chief Complaint: Routine well-child visit and asthma follow-up.

HPI: 8-year-old female here for annual well-child exam and asthma management.
Mother reports 1-2 nighttime awakenings per week with cough. Uses albuterol
inhaler about 3 times weekly. No ER visits since last appointment. Doing well
in school with no absences related to asthma in the past month.

Medications:
  - Fluticasone 44mcg 2 puffs BID
  - Albuterol 90mcg/actuation PRN

Vitals:
  BP: 98/62 mmHg
  HR: 88 bpm
  Temp: 98.2°F
  SpO2: 99%
  Weight: 56 lbs
  Height: 50 inches

Assessment:
  1. Moderate persistent asthma — not optimally controlled
  2. Well-child exam — growth at 50th percentile, development appropriate

Plan:
  - Step up Fluticasone to 110mcg 2 puffs BID
  - Continue Albuterol PRN
  - Asthma action plan reviewed with mother
  - Return in 4 weeks to assess response
  - Annual vaccinations administered today
"""


# ============================================================
# PYDANTIC MODELS
# ============================================================

class Medication(BaseModel):
    """A single prescribed medication."""
    name: str = Field(..., description="Medication name (generic or brand)")
    dosage: str = Field(..., description="Dosage amount and unit, e.g. '1000mg'")
    route: str = Field(default="PO", description="Route of administration, e.g. PO, IV, SQ")
    frequency: str = Field(..., description="Frequency, e.g. BID, daily, PRN")


class VitalSigns(BaseModel):
    """Patient vital signs."""
    blood_pressure: str = Field(..., description="Blood pressure as 'systolic/diastolic mmHg'")
    heart_rate: int = Field(..., ge=20, le=300, description="Heart rate in bpm")
    temperature: float = Field(..., ge=90.0, le=110.0, description="Temperature in °F")
    spo2: int = Field(..., ge=50, le=100, description="Oxygen saturation percentage")


class Diagnosis(BaseModel):
    """A clinical diagnosis."""
    description: str = Field(..., description="Diagnosis description")
    status: str = Field(default="active", description="Status: active, resolved, chronic")


class PatientRecord(BaseModel):
    """Structured patient record extracted from a clinical note."""
    patient_name: str = Field(..., description="Full patient name")
    date_of_birth: str = Field(..., description="Date of birth as YYYY-MM-DD")
    visit_date: str = Field(..., description="Visit date as YYYY-MM-DD")
    chief_complaint: str = Field(..., description="Reason for visit")
    medications: list[Medication] = Field(..., description="List of current medications")
    vitals: VitalSigns = Field(..., description="Vital signs")
    diagnoses: list[Diagnosis] = Field(..., description="List of diagnoses")
    plan: list[str] = Field(..., description="List of plan items")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def call_llm(system_prompt: str, user_message: str, temperature: float = 0.2,
             max_tokens: int = 1500, json_mode: bool = False) -> str:
    """Make a single LLM call and return the response text."""
    kwargs = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f'{{"error": "{e}"}}'


def print_separator(title: str):
    """Print a formatted section separator."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_json(data, indent: int = 2):
    """Pretty-print a dict or Pydantic model as JSON."""
    if isinstance(data, BaseModel):
        print(json.dumps(data.model_dump(), indent=indent, default=str))
    elif isinstance(data, dict):
        print(json.dumps(data, indent=indent, default=str))
    else:
        print(data)


# ============================================================
# DEMO 1: Basic JSON Mode
# ============================================================

def demo_json_mode():
    """
    Use response_format={"type": "json_object"} to guarantee syntactically
    valid JSON from the LLM. Compare text-mode vs JSON-mode output.
    """
    print_separator("DEMO 1: Basic JSON Mode")

    clinical_note = CLINICAL_NOTE_1

    # --- Part A: Without JSON mode (text output) ---
    print("--- Part A: Standard text output (no JSON mode) ---\n")

    text_prompt = (
        "Extract the patient's name, date of birth, medications, diagnoses, "
        "and vital signs from the following clinical note. Format as JSON."
    )
    text_output = call_llm(text_prompt, clinical_note, json_mode=False)
    print(f"Raw output (first 300 chars):\n{text_output[:300]}\n")

    # Try parsing — may fail if LLM wraps in markdown code blocks
    try:
        parsed = json.loads(text_output)
        print("✅ Happened to be valid JSON\n")
    except json.JSONDecodeError as e:
        print(f"❌ Not valid JSON: {e}")
        print("   (Common issue: LLM may wrap JSON in ```json blocks)\n")

    # --- Part B: With JSON mode ---
    print("--- Part B: JSON mode enabled ---\n")

    json_prompt = (
        "Extract the patient's name, date of birth, medications, diagnoses, "
        "and vital signs from the following clinical note. "
        "Return a JSON object with keys: patient_name, date_of_birth, "
        "medications (array of objects with name, dosage, frequency), "
        "diagnoses (array of strings), vitals (object with bp, hr, temp, spo2)."
    )
    json_output = call_llm(json_prompt, clinical_note, json_mode=True)

    try:
        parsed = json.loads(json_output)
        print("✅ Valid JSON extracted:\n")
        print_json(parsed)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        print(f"Raw: {json_output[:200]}")

    print("\n--- Key takeaway ---")
    print("JSON mode guarantees syntactically valid JSON, but does NOT guarantee")
    print("that specific keys are present or values match expected types.")
    print("For that, you need Pydantic validation (Demo 2) or Structured Outputs (Demo 3).")


# ============================================================
# DEMO 2: Pydantic Validation
# ============================================================

def demo_pydantic_validation():
    """
    Define Pydantic models and attempt to parse LLM output into them.
    Demonstrates both successful parsing and handling validation errors.
    """
    print_separator("DEMO 2: Pydantic Validation")

    extraction_prompt = """You are a clinical data extraction system. Extract structured
patient data from the provided clinical note. Return a JSON object with this exact structure:

{
  "patient_name": "string",
  "date_of_birth": "YYYY-MM-DD",
  "visit_date": "YYYY-MM-DD",
  "chief_complaint": "string",
  "medications": [
    {"name": "string", "dosage": "string", "route": "string", "frequency": "string"}
  ],
  "vitals": {
    "blood_pressure": "string like 138/82 mmHg",
    "heart_rate": integer,
    "temperature": float,
    "spo2": integer
  },
  "diagnoses": [
    {"description": "string", "status": "string"}
  ],
  "plan": ["string", "string"]
}

Return ONLY the JSON object. No additional text or explanation."""

    # --- Extract from Clinical Note 1 ---
    print("Extracting from Clinical Note 1 (Maria Santos)...\n")

    raw_output = call_llm(extraction_prompt, CLINICAL_NOTE_1, json_mode=True)

    try:
        data = json.loads(raw_output)
        record = PatientRecord(**data)

        print("✅ Pydantic validation PASSED\n")
        print(f"  Patient: {record.patient_name}")
        print(f"  DOB:     {record.date_of_birth}")
        print(f"  Visit:   {record.visit_date}")
        print(f"  CC:      {record.chief_complaint}")
        print(f"\n  Medications ({len(record.medications)}):")
        for med in record.medications:
            print(f"    - {med.name} {med.dosage} {med.route} {med.frequency}")
        print(f"\n  Vitals:")
        print(f"    BP:   {record.vitals.blood_pressure}")
        print(f"    HR:   {record.vitals.heart_rate} bpm")
        print(f"    Temp: {record.vitals.temperature}°F")
        print(f"    SpO2: {record.vitals.spo2}%")
        print(f"\n  Diagnoses ({len(record.diagnoses)}):")
        for dx in record.diagnoses:
            print(f"    - {dx.description} [{dx.status}]")
        print(f"\n  Plan ({len(record.plan)} items):")
        for item in record.plan:
            print(f"    - {item}")

    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
    except ValidationError as e:
        print(f"❌ Pydantic validation error:\n{e}")

    # --- Demonstrate validation failure ---
    print("\n\n--- Demonstrating validation failure ---\n")

    bad_data = {
        "patient_name": "Test Patient",
        "date_of_birth": "1990-01-01",
        "visit_date": "2026-01-22",
        "chief_complaint": "Testing",
        "medications": [
            {"name": "Aspirin", "dosage": "81mg", "frequency": "daily"}
        ],
        "vitals": {
            "blood_pressure": "120/80",
            "heart_rate": 5,       # Too low — min is 20
            "temperature": 200.0,  # Too high — max is 110
            "spo2": 150            # Too high — max is 100
        },
        "diagnoses": [{"description": "Test"}],
        "plan": ["Follow up"]
    }

    print("Attempting to validate intentionally bad data:")
    print(f"  heart_rate: 5 (min allowed: 20)")
    print(f"  temperature: 200.0 (max allowed: 110)")
    print(f"  spo2: 150 (max allowed: 100)\n")

    try:
        record = PatientRecord(**bad_data)
        print("✅ Passed (unexpected!)")
    except ValidationError as e:
        print(f"❌ Validation caught {e.error_count()} error(s):\n")
        for error in e.errors():
            loc = " → ".join(str(l) for l in error["loc"])
            print(f"  Field: {loc}")
            print(f"  Error: {error['msg']}")
            print(f"  Value: {error['input']}")
            print()

    print("--- Key takeaway ---")
    print("Pydantic catches type errors, missing fields, and constraint violations.")
    print("This is essential for healthcare data where out-of-range values signal errors.")


# ============================================================
# DEMO 3: Structured Outputs (JSON Schema)
# ============================================================

def demo_structured_outputs():
    """
    Use OpenAI's structured outputs feature to provide a JSON schema.
    The API guarantees conformant output — no parsing or retry needed.
    """
    print_separator("DEMO 3: Structured Outputs (JSON Schema)")

    # Define the schema for structured outputs
    patient_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "patient_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "Full patient name"
                    },
                    "date_of_birth": {
                        "type": "string",
                        "description": "Date of birth in YYYY-MM-DD format"
                    },
                    "age_years": {
                        "type": "integer",
                        "description": "Patient age in years"
                    },
                    "chief_complaint": {
                        "type": "string",
                        "description": "Primary reason for visit"
                    },
                    "medications": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "dosage": {"type": "string"},
                                "frequency": {"type": "string"}
                            },
                            "required": ["name", "dosage", "frequency"],
                            "additionalProperties": False
                        },
                        "description": "List of current medications"
                    },
                    "diagnoses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of diagnoses"
                    },
                    "vitals": {
                        "type": "object",
                        "properties": {
                            "blood_pressure": {"type": "string"},
                            "heart_rate": {"type": "integer"},
                            "temperature_f": {"type": "number"},
                            "spo2_percent": {"type": "integer"}
                        },
                        "required": ["blood_pressure", "heart_rate", "temperature_f", "spo2_percent"],
                        "additionalProperties": False
                    }
                },
                "required": [
                    "patient_name", "date_of_birth", "age_years",
                    "chief_complaint", "medications", "diagnoses", "vitals"
                ],
                "additionalProperties": False
            }
        }
    }

    print("Schema provided to the API:")
    print(f"  Required fields: patient_name, date_of_birth, age_years,")
    print(f"                  chief_complaint, medications, diagnoses, vitals")
    print(f"  Medications: array of objects with name, dosage, frequency")
    print(f"  Vitals: object with blood_pressure, heart_rate, temperature_f, spo2_percent")
    print(f"  strict: True (API guarantees conformance)\n")

    # Test with each clinical note
    notes = [
        ("Clinical Note 1 (Maria Santos)", CLINICAL_NOTE_1),
        ("Clinical Note 2 (James O'Brien)", CLINICAL_NOTE_2),
        ("Clinical Note 3 (Aisha Patel)", CLINICAL_NOTE_3),
    ]

    for note_name, note_text in notes:
        print(f"--- {note_name} ---\n")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a clinical data extraction system. Extract structured "
                            "patient data from the provided clinical note."
                        )
                    },
                    {"role": "user", "content": note_text}
                ],
                response_format=patient_schema,
                temperature=0.0,
                max_tokens=1000
            )

            raw = response.choices[0].message.content
            data = json.loads(raw)

            print(f"  Patient: {data['patient_name']}")
            print(f"  DOB:     {data['date_of_birth']}")
            print(f"  Age:     {data['age_years']}")
            print(f"  CC:      {data['chief_complaint']}")
            print(f"  Meds:    {len(data['medications'])} medications")
            for med in data["medications"]:
                print(f"           - {med['name']} {med['dosage']} {med['frequency']}")
            print(f"  Dx:      {', '.join(data['diagnoses'])}")
            v = data["vitals"]
            print(f"  Vitals:  BP {v['blood_pressure']}, HR {v['heart_rate']}, "
                  f"Temp {v['temperature_f']}°F, SpO2 {v['spo2_percent']}%")
            print(f"\n  ✅ Structured output conforms to schema")

        except Exception as e:
            print(f"  ❌ Error: {e}")

        print()

    print("--- Key takeaway ---")
    print("Structured Outputs guarantee schema conformance at the API level.")
    print("Every field specified in the schema is present with the correct type.")
    print("This eliminates the need for post-hoc parsing and retry loops.")


# ============================================================
# DEMO 4: Retry Logic with Error Feedback
# ============================================================

def demo_retry_logic():
    """
    When LLM output fails validation, retry with the error message appended.
    The LLM uses the error feedback to correct its output.
    """
    print_separator("DEMO 4: Retry Logic with Error Feedback")

    # Use a deliberately tricky note with ambiguous data
    tricky_note = """
    Pt: R. Thompson (Bob), 82 y/o M
    Seen today for SOB and bilateral LE edema, worsening x 2 weeks.
    PMHx: CHF (EF 30%), AFib, CKD stage III, COPD
    Meds: Lasix 40 twice a day, Eliquis 5 mg bid, Lisinopril ten mg qd,
           Metoprolol succinate 50mg daily, home O2 2L NC
    VS: 148/92, P 88 irreg, T 97.9, O2 91% on 2L
    Wt: 198 (up 8 lbs from 2 weeks ago)
    A/P: Acute on chronic systolic HF exacerbation. Increase Lasix to 80mg BID.
         Daily weights. Fluid restrict 1.5L. Renal panel in AM. Consider
         cardiology re-eval if no improvement in 48h.
    """

    extraction_prompt = """Extract structured data from this clinical note. Return JSON matching this exact schema:
{
  "patient_name": "string (full name)",
  "age": integer,
  "sex": "M" or "F",
  "chief_complaint": "string",
  "medications": [
    {"name": "string", "dosage": "string (numeric with unit)", "route": "PO", "frequency": "string (use standard: BID, daily, etc.)"}
  ],
  "vitals": {
    "blood_pressure": "string (systolic/diastolic)",
    "heart_rate": integer,
    "temperature": number (in °F),
    "spo2": integer (percentage, 0-100)
  },
  "diagnoses": [{"description": "string (full name, not abbreviation)", "status": "active"}],
  "plan": ["string"]
}

Return ONLY the JSON object."""

    max_retries = 3
    messages = [
        {"role": "system", "content": extraction_prompt},
        {"role": "user", "content": tricky_note}
    ]

    print(f"Extracting from tricky note with abbreviated data...")
    print(f"Max retries: {max_retries}\n")

    for attempt in range(1, max_retries + 1):
        print(f"--- Attempt {attempt}/{max_retries} ---\n")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1000
            )
            raw_output = response.choices[0].message.content.strip()

            # Try to parse JSON
            data = json.loads(raw_output)

            # Try to validate with Pydantic
            record = PatientRecord(
                patient_name=data.get("patient_name", ""),
                date_of_birth=data.get("date_of_birth", "1900-01-01"),
                visit_date=data.get("visit_date", "2026-01-01"),
                chief_complaint=data.get("chief_complaint", ""),
                medications=[Medication(**m) for m in data.get("medications", [])],
                vitals=VitalSigns(**data.get("vitals", {})),
                diagnoses=[Diagnosis(**d) if isinstance(d, dict) else Diagnosis(description=d)
                           for d in data.get("diagnoses", [])],
                plan=data.get("plan", [])
            )

            print(f"✅ Validation PASSED on attempt {attempt}!\n")
            print(f"  Patient: {record.patient_name}")
            print(f"  CC:      {record.chief_complaint}")
            print(f"  Meds:    {len(record.medications)} medications")
            for med in record.medications:
                print(f"           - {med.name} {med.dosage} {med.route} {med.frequency}")
            print(f"  Vitals:  BP {record.vitals.blood_pressure}, HR {record.vitals.heart_rate}, "
                  f"Temp {record.vitals.temperature}°F, SpO2 {record.vitals.spo2}%")
            print(f"  Dx:      {len(record.diagnoses)} diagnoses")
            for dx in record.diagnoses:
                print(f"           - {dx.description}")
            print(f"  Plan:    {len(record.plan)} items")
            break

        except json.JSONDecodeError as e:
            error_msg = f"JSON parse error: {e}"
            print(f"  ❌ {error_msg}")
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({
                "role": "user",
                "content": f"Your output was not valid JSON. Error: {error_msg}\n"
                           f"Please fix the JSON and try again. Return ONLY valid JSON."
            })

        except ValidationError as e:
            error_details = []
            for error in e.errors():
                loc = " → ".join(str(l) for l in error["loc"])
                error_details.append(f"  Field '{loc}': {error['msg']} (got: {error['input']})")
            error_msg = "\n".join(error_details)
            print(f"  ❌ Validation errors:\n{error_msg}\n")

            messages.append({"role": "assistant", "content": raw_output})
            messages.append({
                "role": "user",
                "content": (
                    f"Your JSON output had validation errors:\n{error_msg}\n\n"
                    f"Please fix these issues and return corrected JSON. Remember:\n"
                    f"- heart_rate must be an integer between 20 and 300\n"
                    f"- temperature must be a float between 90.0 and 110.0\n"
                    f"- spo2 must be an integer between 50 and 100\n"
                    f"- All medication fields (name, dosage, frequency) are required"
                )
            })

        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            break

    else:
        print(f"\n⚠️  Failed after {max_retries} attempts.")
        print("In production, this would be logged for human review.")

    print("\n--- Key takeaway ---")
    print("Retry with error feedback lets the LLM self-correct. Each attempt")
    print("gets more context about what went wrong, improving success rates.")
    print("Always set a max retry count and have a fallback strategy.")


# ============================================================
# MAIN MENU
# ============================================================

def main():
    """
    Run output validation demos with a menu interface.
    """
    print("🏥 Level 4.3: Structured Outputs and Validation\n")
    print("This project demonstrates techniques for ensuring LLM outputs")
    print("match expected schemas — critical for healthcare data pipelines.\n")

    while True:
        print("\n" + "=" * 50)
        print("MAIN MENU")
        print("=" * 50)
        print("1. Basic JSON Mode")
        print("2. Pydantic Validation")
        print("3. Structured Outputs (JSON Schema)")
        print("4. Retry Logic with Error Feedback")
        print("5. Run All Demos (1-4)")
        print("q. Quit")
        print("=" * 50)

        choice = input("Select demo (1-5 or q): ").strip().lower()

        if choice == "1":
            demo_json_mode()
        elif choice == "2":
            demo_pydantic_validation()
        elif choice == "3":
            demo_structured_outputs()
        elif choice == "4":
            demo_retry_logic()
        elif choice == "5":
            demo_json_mode()
            demo_pydantic_validation()
            demo_structured_outputs()
            demo_retry_logic()
        elif choice in ("q", "quit", "exit"):
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-5 or q.")


if __name__ == "__main__":
    main()
