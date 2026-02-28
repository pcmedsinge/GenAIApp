"""
Exercise 3: Validation Errors — Edge Cases, Retries, and Fallbacks

Skills practiced:
  - Identifying common LLM validation failure modes
  - Implementing retry-with-feedback for automatic self-correction
  - Designing fallback schemas for graceful degradation
  - Logging validation failures for debugging and improvement

Healthcare context:
  When an LLM extracts medication data from a note and returns "dose": "two tabs"
  instead of "dose": "500mg", validation fails. In a live clinical system you can't
  just crash — you need a strategy. This exercise explores three strategies:
    1. Retry: re-prompt the LLM with the error so it can self-correct
    2. Fallback schema: accept a simpler model when the complex one fails
    3. Failure logging: capture every failure for later analysis and prompt tuning

  You'll test with adversarial clinical notes designed to trigger edge cases:
  ambiguous dosages, missing fields, unusual formats, conflicting data.
"""

import os
import json
import time
from typing import Optional
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"


# ============================================================
# Strict Schema (many constraints — likely to fail on edge cases)
# ============================================================

class StrictMedication(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    dose_mg: float = Field(..., gt=0, le=10000, description="Dose in milligrams (numeric)")
    frequency_per_day: int = Field(..., ge=1, le=12, description="Times per day (integer)")
    route: str = Field(..., description="oral, IV, IM, SubQ, inhaled, topical, rectal")

    @field_validator("route")
    @classmethod
    def valid_route(cls, v: str) -> str:
        allowed = {"oral", "iv", "im", "subq", "inhaled", "topical", "rectal", "sublingual"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(f"route must be one of {allowed}, got '{v}'")
        return v_lower


class StrictVitals(BaseModel):
    systolic_bp: int = Field(..., ge=60, le=280)
    diastolic_bp: int = Field(..., ge=30, le=180)
    heart_rate: int = Field(..., ge=25, le=220)
    temperature_f: float = Field(..., ge=92.0, le=108.0)
    spo2_percent: int = Field(..., ge=50, le=100)


class StrictPatient(BaseModel):
    patient_name: str = Field(..., min_length=3, max_length=100)
    age: int = Field(..., ge=0, le=120)
    sex: str
    chief_complaint: str = Field(..., min_length=5)
    medications: list[StrictMedication] = Field(..., min_length=1)
    vitals: StrictVitals
    assessment: str = Field(..., min_length=10)

    @field_validator("sex")
    @classmethod
    def normalize_sex(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("male", "female"):
            raise ValueError(f"sex must be 'male' or 'female', got '{v}'")
        return v


# ============================================================
# Fallback Schema (relaxed — always parsable)
# ============================================================

class FallbackPatient(BaseModel):
    """Relaxed schema that accepts partial data gracefully."""
    patient_name: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    chief_complaint: Optional[str] = None
    medications_raw: Optional[list[str]] = Field(
        default=None,
        description="Medications as simple strings if structured parsing fails"
    )
    vitals_raw: Optional[str] = Field(
        default=None,
        description="Vitals as a single string if structured parsing fails"
    )
    assessment: Optional[str] = None
    raw_json: Optional[str] = Field(
        default=None,
        description="The original LLM JSON if all parsing failed"
    )


# ============================================================
# Adversarial Clinical Notes (designed to trigger edge cases)
# ============================================================

EDGE_CASE_NOTES = [
    {
        "id": "edge_1_ambiguous_dose",
        "description": "Ambiguous dosage format (tabs instead of mg)",
        "text": (
            "Patient: Amy Liu, 45-year-old female. "
            "CC: Lower back pain x 1 week. "
            "Meds: Ibuprofen 2 tabs TID, Cyclobenzaprine half tab at bedtime, "
            "Acetaminophen 500-1000mg Q6H PRN. "
            "Vitals: BP 118/74, HR 72, RR 14, Temp 98.6°F, SpO2 99%. "
            "Assessment: Acute lumbar strain. "
            "Plan: Continue current meds, physical therapy referral."
        ),
    },
    {
        "id": "edge_2_missing_vitals",
        "description": "Vitals partially missing (no temp, no SpO2)",
        "text": (
            "Pt: John Martinez, 82yo M. "
            "CC: Confusion and falls x 2 days. "
            "Meds: Donepezil 10mg daily, Memantine 10mg BID, Quetiapine 25mg QHS. "
            "Vitals: BP 132/78, HR 64. (RR, temp, O2 sat not documented). "
            "Assessment: Altered mental status, rule out UTI vs delirium. "
            "Plan: UA, CBC, BMP, CT head, hold quetiapine, sitter at bedside."
        ),
    },
    {
        "id": "edge_3_conflicting_data",
        "description": "Conflicting information in the note",
        "text": (
            "Patient: Carlos Rivera, 55 year old male (DOB indicates age 57). "
            "CC: Follow-up after MI. "
            "Meds: Aspirin 81mg daily, Atorvastatin 80mg daily, Metoprolol 50mg BID, "
            "Clopidogrel 75mg daily, Lisinopril 20mg daily. "
            "Vitals: BP 128/82, HR 68, RR 16, Temp 98.2°F, SpO2 97%. "
            "Assessment: STEMI (resolved), on appropriate secondary prevention. "
            "Plan: Continue all meds, cardiac rehab, follow-up in 1 month."
        ),
    },
    {
        "id": "edge_4_unusual_meds",
        "description": "IV and non-standard medication routes",
        "text": (
            "Patient: Priya Sharma, 30-year-old female. "
            "CC: Severe migraine, vomiting. "
            "Meds given in ED: Ketorolac 30mg IV x1, Metoclopramide 10mg IV x1, "
            "Normal saline 1L IV, Sumatriptan 6mg SubQ x1, Ondansetron 4mg IV PRN. "
            "Home meds: Topiramate 50mg BID (oral), Rizatriptan 10mg PRN (oral). "
            "Vitals: BP 140/88, HR 96, RR 18, Temp 98.4°F, SpO2 99%. "
            "Assessment: Status migrainosus. "
            "Plan: Admit for DHE protocol, neurology consult."
        ),
    },
]


# ============================================================
# Failure Logger
# ============================================================

class ValidationFailureLog:
    """Captures validation failures for analysis."""

    def __init__(self):
        self.failures: list[dict] = []

    def log(self, note_id: str, attempt: int, schema_name: str,
            errors: list[dict], raw_json: str) -> None:
        self.failures.append({
            "timestamp": datetime.now().isoformat(),
            "note_id": note_id,
            "attempt": attempt,
            "schema": schema_name,
            "error_count": len(errors),
            "errors": errors,
            "raw_json_preview": raw_json[:300],
        })

    def summary(self) -> None:
        print(f"\n{'─' * 60}")
        print(f"FAILURE LOG: {len(self.failures)} total failures")
        print(f"{'─' * 60}")

        # Aggregate by error type
        error_types: dict[str, int] = {}
        for failure in self.failures:
            for err in failure["errors"]:
                err_type = err.get("type", "unknown")
                error_types[err_type] = error_types.get(err_type, 0) + 1

        if error_types:
            print("\nFailure types:")
            for t, count in sorted(error_types.items(), key=lambda x: -x[1]):
                print(f"  {t}: {count}")

        # Per-note failures
        notes = {}
        for f in self.failures:
            nid = f["note_id"]
            notes[nid] = notes.get(nid, 0) + 1
        print("\nFailures per note:")
        for nid, count in notes.items():
            print(f"  {nid}: {count}")


failure_log = ValidationFailureLog()


# ============================================================
# Extraction with Retry + Fallback
# ============================================================

def extract_with_retry_and_fallback(
    note_id: str,
    note_text: str,
    max_retries: int = 3,
) -> dict:
    """
    Try strict schema first with retries, then fall back to relaxed schema.

    Returns dict with: strategy used, record, and success status.
    """
    strict_schema_json = json.dumps(StrictPatient.model_json_schema(), indent=2)

    system_prompt = f"""Extract structured clinical data from this note.
Return valid JSON matching this schema:

{strict_schema_json}

Critical rules:
- dose_mg MUST be a number in milligrams. Convert "500mg" → 500, "2 tabs" → estimate mg.
- frequency_per_day MUST be an integer: daily=1, BID=2, TID=3, QID=4, Q6H=4, Q8H=3, Q4H=6.
- route: oral, IV, IM, SubQ, inhaled, topical, rectal, sublingual.
- All vitals fields are REQUIRED. If not documented, estimate or use 0.
- sex: exactly "male" or "female".
Return ONLY valid JSON."""

    # --- Strategy 1: Retry with feedback ---
    last_error = None
    last_raw = ""

    for attempt in range(1, max_retries + 1):
        user_msg = note_text
        if last_error:
            user_msg += (
                f"\n\n[VALIDATION FAILED — ATTEMPT {attempt}]\n"
                f"Errors: {last_error}\n"
                f"Fix these specific issues and return corrected JSON."
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
                max_tokens=1500,
            )
            last_raw = response.choices[0].message.content.strip()
        except Exception as e:
            last_error = str(e)
            continue

        try:
            record = StrictPatient.model_validate_json(last_raw)
            return {
                "note_id": note_id,
                "strategy": f"strict (attempt {attempt})",
                "success": True,
                "record": record,
            }
        except ValidationError as e:
            failure_log.log(
                note_id=note_id,
                attempt=attempt,
                schema_name="StrictPatient",
                errors=[err for err in e.errors()],
                raw_json=last_raw,
            )
            last_error = "; ".join(
                f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
                for err in e.errors()
            )

    # --- Strategy 2: Fallback to relaxed schema ---
    print(f"    Strict schema failed after {max_retries} retries. Trying fallback...")

    fallback_prompt = """Extract whatever patient data you can from this clinical note.
Return JSON with these optional fields:
- patient_name, age, sex, chief_complaint, assessment
- medications_raw: list of medication strings (e.g., ["Aspirin 81mg daily"])
- vitals_raw: vitals as a single string (e.g., "BP 120/80, HR 72")
All fields are optional. Return ONLY valid JSON."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": fallback_prompt},
                {"role": "user", "content": note_text},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1000,
        )
        raw = response.choices[0].message.content.strip()
        # Also store the raw JSON in case we need it
        data = json.loads(raw)
        data["raw_json"] = last_raw  # keep the original failed attempt
        record = FallbackPatient.model_validate(data)
        return {
            "note_id": note_id,
            "strategy": "fallback",
            "success": True,
            "record": record,
        }
    except Exception as e:
        return {
            "note_id": note_id,
            "strategy": "fallback (failed)",
            "success": False,
            "error": str(e),
        }


# ============================================================
# Main
# ============================================================

def main():
    """Test edge-case notes with retry and fallback strategies."""
    print("=" * 65)
    print("Exercise 3: Validation Errors — Edge Cases, Retries, Fallbacks")
    print("=" * 65)

    results = []
    for note in EDGE_CASE_NOTES:
        print(f"\n{'─' * 60}")
        print(f"NOTE: {note['id']}")
        print(f"Edge case: {note['description']}")
        print(f"{'─' * 60}")

        result = extract_with_retry_and_fallback(note["id"], note["text"])
        results.append(result)

        if result["success"]:
            rec = result["record"]
            print(f"  ✓ Strategy: {result['strategy']}")
            if isinstance(rec, StrictPatient):
                print(f"  Patient: {rec.patient_name}, {rec.age}yo {rec.sex}")
                print(f"  Medications ({len(rec.medications)}):")
                for m in rec.medications:
                    print(f"    - {m.name} {m.dose_mg}mg x{m.frequency_per_day}/day ({m.route})")
                print(f"  Vitals: BP {rec.vitals.systolic_bp}/{rec.vitals.diastolic_bp}, "
                      f"HR {rec.vitals.heart_rate}")
            elif isinstance(rec, FallbackPatient):
                print(f"  [FALLBACK] Patient: {rec.patient_name}, {rec.age}")
                if rec.medications_raw:
                    print(f"  Medications (raw): {rec.medications_raw}")
                if rec.vitals_raw:
                    print(f"  Vitals (raw): {rec.vitals_raw}")
        else:
            print(f"  ✗ All strategies failed: {result.get('error', 'unknown')}")

    # Summary
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    strict_ok = sum(1 for r in results if r["success"] and "strict" in r["strategy"])
    fallback_ok = sum(1 for r in results if r["success"] and "fallback" in r["strategy"])
    failed = sum(1 for r in results if not r["success"])

    print(f"  Strict schema succeeded:  {strict_ok}/{len(results)}")
    print(f"  Fell back to relaxed:     {fallback_ok}/{len(results)}")
    print(f"  Total failures:           {failed}/{len(results)}")

    # Failure log analysis
    failure_log.summary()

    print("\n💡 Key insights:")
    print("  - Ambiguous doses (e.g., '2 tabs') are the hardest for strict numeric schemas")
    print("  - Missing required fields need fallback schemas, not just retries")
    print("  - Conflicting data (age mismatch) can confuse the LLM")
    print("  - Logging failures reveals patterns you can fix with prompt engineering")


if __name__ == "__main__":
    main()
