"""
Exercise 2: Schema Complexity — Simple → Medium → Complex

Skills practiced:
  - Designing Pydantic schemas at increasing levels of complexity
  - Understanding how schema complexity affects LLM extraction accuracy
  - Measuring field-level accuracy across complexity tiers
  - Tuning prompts to improve compliance with complex schemas

Healthcare context:
  Clinical data ranges from simple (patient name + age) to deeply nested
  (a full encounter record with medications, vitals, diagnoses, procedures,
  insurance, and social history). More complex schemas are harder for LLMs to
  fill correctly — fields get omitted, types get confused, and nested objects
  may be flattened.

  This exercise builds three schema tiers (simple, medium, complex), runs the
  same clinical note through each, and measures extraction accuracy to show
  exactly where complexity causes failures.
"""

import os
import json
import time
from typing import Optional
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"


# ============================================================
# TIER 1: Simple Schema (4 fields)
# ============================================================

class SimplePatient(BaseModel):
    """Minimal patient extraction — name, age, sex, chief complaint."""
    patient_name: str = Field(..., description="Full patient name")
    age: int = Field(..., ge=0, le=150)
    sex: str = Field(..., description="male or female")
    chief_complaint: str = Field(..., description="Primary reason for visit")

    @field_validator("sex")
    @classmethod
    def normalize_sex(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("male", "female"):
            raise ValueError(f"sex must be 'male' or 'female', got '{v}'")
        return v


# ============================================================
# TIER 2: Medium Schema (nested medications + vitals)
# ============================================================

class MediumMedication(BaseModel):
    name: str = Field(..., description="Medication name")
    dose: str = Field(..., description="Dose with units")
    frequency: str = Field(..., description="Frequency (e.g., BID, daily)")

class MediumVitals(BaseModel):
    systolic_bp: Optional[int] = Field(None, ge=50, le=300)
    diastolic_bp: Optional[int] = Field(None, ge=20, le=200)
    heart_rate: Optional[int] = Field(None, ge=20, le=250)
    spo2_percent: Optional[int] = Field(None, ge=50, le=100)

class MediumPatient(BaseModel):
    """Mid-complexity: demographics + medications + vitals + assessment."""
    patient_name: str
    age: int = Field(..., ge=0, le=150)
    sex: str
    chief_complaint: str
    medications: list[MediumMedication] = Field(default_factory=list)
    vitals: Optional[MediumVitals] = None
    assessment: str = Field(..., description="Clinical assessment/diagnosis")
    plan: list[str] = Field(default_factory=list)

    @field_validator("sex")
    @classmethod
    def normalize_sex(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("male", "female"):
            raise ValueError(f"sex must be 'male' or 'female', got '{v}'")
        return v


# ============================================================
# TIER 3: Complex Schema (deeply nested, many fields)
# ============================================================

class ComplexMedication(BaseModel):
    name: str
    dose: str
    frequency: str
    route: str = Field("oral", description="oral, IV, inhaled, SubQ, etc.")
    indication: Optional[str] = None
    start_date: Optional[str] = None
    is_new: bool = Field(False, description="True if started during this encounter")

class ComplexVitals(BaseModel):
    systolic_bp: Optional[int] = Field(None, ge=50, le=300)
    diastolic_bp: Optional[int] = Field(None, ge=20, le=200)
    heart_rate: Optional[int] = Field(None, ge=20, le=250)
    respiratory_rate: Optional[int] = Field(None, ge=4, le=60)
    temperature_f: Optional[float] = Field(None, ge=90.0, le=110.0)
    spo2_percent: Optional[int] = Field(None, ge=50, le=100)
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None

class ComplexDiagnosis(BaseModel):
    condition: str
    icd10_code: Optional[str] = None
    status: str = Field("active")
    severity: Optional[str] = Field(None, description="mild, moderate, severe")
    onset: Optional[str] = None

    @field_validator("status")
    @classmethod
    def valid_status(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("active", "resolved", "suspected", "chronic"):
            raise ValueError(f"status must be active/resolved/suspected/chronic, got '{v}'")
        return v

class ComplexLabResult(BaseModel):
    test_name: str
    value: str
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    is_abnormal: bool = False

class SocialHistory(BaseModel):
    smoking_status: Optional[str] = None
    alcohol_use: Optional[str] = None
    exercise: Optional[str] = None
    occupation: Optional[str] = None

class ComplexPatient(BaseModel):
    """Full-complexity: deeply nested encounter record."""
    patient_name: str
    age: int = Field(..., ge=0, le=150)
    sex: str
    date_of_birth: Optional[str] = None
    mrn: Optional[str] = None
    chief_complaint: str
    history_of_present_illness: Optional[str] = None
    past_medical_history: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    medications: list[ComplexMedication] = Field(default_factory=list)
    vitals: Optional[ComplexVitals] = None
    diagnoses: list[ComplexDiagnosis] = Field(default_factory=list)
    labs: list[ComplexLabResult] = Field(default_factory=list)
    social_history: Optional[SocialHistory] = None
    plan: list[str] = Field(default_factory=list)
    follow_up: Optional[str] = None
    referrals: list[str] = Field(default_factory=list)

    @field_validator("sex")
    @classmethod
    def normalize_sex(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("male", "female"):
            raise ValueError(f"sex must be 'male' or 'female', got '{v}'")
        return v


# ============================================================
# Test Clinical Note
# ============================================================

TEST_NOTE = (
    "Patient: Margaret Okonkwo, 58-year-old female. DOB: 1967-08-14. MRN: 6634201. "
    "Allergies: Codeine (nausea), Latex. "
    "CC: Chest tightness and palpitations for 5 days. "
    "HPI: Ms. Okonkwo reports intermittent chest tightness, non-exertional, associated "
    "with palpitations. Denies syncope or dyspnea. She reports increased work stress and "
    "poor sleep. She is a non-smoker, drinks wine socially (1-2 glasses/week), walks 30 "
    "minutes daily. She works as a school teacher. "
    "PMH: Hypertension x 8 years, Anxiety disorder, GERD. "
    "Medications: Amlodipine 5mg daily (HTN), Sertraline 50mg daily (anxiety), "
    "Omeprazole 20mg daily (GERD). "
    "Vitals: BP 144/90 mmHg, HR 96 irregular, RR 16, Temp 98.4°F, SpO2 98%, Wt 72 kg, "
    "BMI 27.1. "
    "Labs: TSH 2.1 mIU/L (normal 0.4-4.0), potassium 4.1 mEq/L (normal 3.5-5.0), "
    "magnesium 1.6 mg/dL (low, normal 1.7-2.2), BNP 45 pg/mL (normal <100). "
    "ECG: Premature atrial contractions, otherwise normal sinus rhythm. "
    "Assessment: 1. Premature atrial contractions — likely related to hypomagnesemia and "
    "stress. 2. Hypertension — suboptimally controlled. 3. Hypomagnesemia — mild. "
    "Plan: Start magnesium oxide 400mg daily, increase amlodipine to 10mg daily, "
    "Holter monitor for 48 hours, sleep hygiene counseling, stress management referral, "
    "recheck magnesium and BP in 2 weeks. Cardiology referral if PACs persist."
)


# ============================================================
# Extraction Engine
# ============================================================

def extract_with_schema(schema_class: type[BaseModel], note: str,
                        tier_name: str, max_retries: int = 2) -> dict:
    """
    Extract data using a given Pydantic schema. Returns a result dict
    with success status, attempts, record, and timing.
    """
    schema_json = json.dumps(schema_class.model_json_schema(), indent=2)

    system_prompt = f"""Extract structured clinical data from the note.
Return valid JSON conforming exactly to this schema:

{schema_json}

Rules:
- Fill every field you can find evidence for in the note.
- Use null for fields with no evidence.
- sex must be "male" or "female".
- Expand abbreviations (HTN → Hypertension, GERD → Gastroesophageal reflux disease).
- Return ONLY valid JSON."""

    start_time = time.time()
    last_error = None

    for attempt in range(1, max_retries + 1):
        user_msg = note
        if last_error:
            user_msg += f"\n\n[VALIDATION ERROR]\n{last_error}\nPlease fix and retry."

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
            record = schema_class.model_validate_json(raw)
            elapsed = time.time() - start_time
            return {
                "tier": tier_name,
                "success": True,
                "attempts": attempt,
                "elapsed_s": round(elapsed, 2),
                "record": record,
                "fields_total": count_fields(record),
                "fields_filled": count_filled_fields(record),
            }
        except ValidationError as e:
            last_error = "; ".join(
                f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
                for err in e.errors()
            )
        except Exception as e:
            last_error = str(e)

    elapsed = time.time() - start_time
    return {
        "tier": tier_name,
        "success": False,
        "attempts": max_retries,
        "elapsed_s": round(elapsed, 2),
        "record": None,
        "error": last_error,
        "fields_total": 0,
        "fields_filled": 0,
    }


def count_fields(model: BaseModel) -> int:
    """Count the total number of leaf fields in a Pydantic model (recursive)."""
    count = 0
    for field_name in model.model_fields:
        value = getattr(model, field_name)
        if isinstance(value, BaseModel):
            count += count_fields(value)
        elif isinstance(value, list):
            if value and isinstance(value[0], BaseModel):
                for item in value:
                    count += count_fields(item)
            else:
                count += 1
        else:
            count += 1
    return count


def count_filled_fields(model: BaseModel) -> int:
    """Count the number of non-None, non-empty leaf fields."""
    count = 0
    for field_name in model.model_fields:
        value = getattr(model, field_name)
        if isinstance(value, BaseModel):
            count += count_filled_fields(value)
        elif isinstance(value, list):
            if value and isinstance(value[0], BaseModel):
                for item in value:
                    count += count_filled_fields(item)
            elif value:  # non-empty list
                count += 1
        elif value is not None:
            count += 1
    return count


# ============================================================
# Main
# ============================================================

def main():
    """Run extraction at each complexity tier and compare results."""
    print("=" * 65)
    print("Exercise 2: Schema Complexity — Simple → Medium → Complex")
    print("=" * 65)

    tiers = [
        ("SIMPLE (4 fields)", SimplePatient),
        ("MEDIUM (~8 fields + nested)", MediumPatient),
        ("COMPLEX (~20+ fields, deeply nested)", ComplexPatient),
    ]

    results = []
    for tier_name, schema_class in tiers:
        print(f"\n{'─' * 60}")
        print(f"TIER: {tier_name}")
        print(f"Schema fields: {len(schema_class.model_fields)}")
        print(f"{'─' * 60}")

        result = extract_with_schema(schema_class, TEST_NOTE, tier_name)
        results.append(result)

        if result["success"]:
            record = result["record"]
            print(f"  ✓ Extraction succeeded (attempt {result['attempts']}, "
                  f"{result['elapsed_s']}s)")
            print(f"  Fields: {result['fields_filled']}/{result['fields_total']} filled")
            print(f"  Patient: {record.patient_name}, {record.age}yo {record.sex}")
            print(f"  CC: {record.chief_complaint}")

            if hasattr(record, "medications"):
                print(f"  Medications: {len(record.medications)}")
                for m in record.medications:
                    print(f"    - {m.name} {m.dose} {m.frequency}")

            if hasattr(record, "diagnoses") and record.diagnoses:
                print(f"  Diagnoses: {len(record.diagnoses)}")
                for d in record.diagnoses:
                    sev = f" ({d.severity})" if hasattr(d, "severity") and d.severity else ""
                    print(f"    - {d.condition}{sev}")

            if hasattr(record, "labs") and record.labs:
                print(f"  Labs: {len(record.labs)}")
                for lab in record.labs:
                    flag = " ⚠" if lab.is_abnormal else ""
                    print(f"    - {lab.test_name}: {lab.value}{flag}")

            if hasattr(record, "social_history") and record.social_history:
                sh = record.social_history
                print(f"  Social: smoking={sh.smoking_status}, alcohol={sh.alcohol_use}")
        else:
            print(f"  ✗ Extraction failed after {result['attempts']} attempts")
            print(f"  Error: {result.get('error', 'unknown')}")

    # Comparison summary
    print("\n" + "=" * 65)
    print("COMPLEXITY COMPARISON SUMMARY")
    print("=" * 65)
    print(f"{'Tier':<40} {'Success':>8} {'Attempts':>9} {'Time':>7} {'Fields':>10}")
    print("─" * 74)
    for r in results:
        status = "✓" if r["success"] else "✗"
        fields = f"{r['fields_filled']}/{r['fields_total']}" if r["success"] else "N/A"
        print(f"{r['tier']:<40} {status:>8} {r['attempts']:>9} "
              f"{r['elapsed_s']:>6}s {fields:>10}")

    print("\n💡 Key insights:")
    print("  - Simple schemas almost always validate on the first attempt")
    print("  - Complex nested schemas may need retries or prompt tuning")
    print("  - More fields = more chances for validation error on any single field")
    print("  - Optional fields help: the LLM can use null for missing data")


if __name__ == "__main__":
    main()
