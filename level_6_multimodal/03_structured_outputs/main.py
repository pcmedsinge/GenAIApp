"""
Project 03: Structured Outputs — Main Demo
==========================================
Demonstrates OpenAI's structured outputs feature to guarantee JSON responses
that conform to exact schemas. Uses Pydantic models for type-safe extraction.

Demos:
  1. response_format with JSON schema (beta parse)
  2. Nested schemas — Patient → Medications → Dosing
  3. Enum / Literal constraints on output values
  4. Schema evolution — v1 → v2 with backward compatibility
"""

import json
import os
import time
from enum import Enum
from typing import List, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI()
MODEL = "gpt-4o"


# ============================================================================
# SHARED SCHEMAS
# ============================================================================

class VitalSigns(BaseModel):
    """Basic vital signs."""
    heart_rate: Optional[int] = Field(None, description="Heart rate in bpm")
    blood_pressure: Optional[str] = Field(None, description="e.g. 120/80 mmHg")
    temperature: Optional[float] = Field(None, description="Temperature in °F")
    respiratory_rate: Optional[int] = Field(None, description="Breaths per minute")
    oxygen_saturation: Optional[int] = Field(None, description="SpO2 percentage")


class Dosing(BaseModel):
    """Dosing schedule for a medication."""
    amount: str = Field(..., description="Dose amount, e.g. '500 mg'")
    frequency: str = Field(..., description="How often, e.g. 'twice daily'")
    route: str = Field(..., description="Administration route, e.g. 'oral'")
    duration: Optional[str] = Field(None, description="Duration, e.g. '7 days'")


class Medication(BaseModel):
    """A single medication entry."""
    name: str = Field(..., description="Medication name")
    dosing: Dosing = Field(..., description="Dosing information")
    indication: Optional[str] = Field(None, description="Reason for prescribing")


class SimplePatient(BaseModel):
    """Simple patient extraction schema."""
    name: str = Field(..., description="Patient full name")
    age: Optional[int] = Field(None, description="Patient age in years")
    gender: Optional[str] = Field(None, description="Patient gender")
    chief_complaint: str = Field(..., description="Primary reason for visit")
    assessment: str = Field(..., description="Clinical assessment")


# ============================================================================
# DEMO 1: response_format WITH JSON SCHEMA
# ============================================================================

def demo_1_json_schema_basics():
    """
    Use OpenAI's structured outputs beta to guarantee JSON matching a schema.
    client.beta.chat.completions.parse() returns a typed Pydantic object.
    """
    print("\n" + "=" * 70)
    print("DEMO 1: response_format with JSON Schema")
    print("=" * 70)

    clinical_note = """
    Patient: Maria Garcia, 58-year-old female
    Chief Complaint: Persistent cough for 3 weeks with occasional blood-tinged sputum.
    Assessment: Chronic cough with hemoptysis, concerning for possible malignancy
    vs bronchiectasis vs infection. CT chest recommended. Current smoker, 30 pack-years.
    """

    print(f"\n--- Input Clinical Note ---\n{clinical_note.strip()}")
    print("\n--- Extracting with schema enforcement ---")

    start = time.time()
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical data extraction system. Extract structured "
                    "patient information from the clinical note. Be precise and only "
                    "include information explicitly stated in the note."
                ),
            },
            {"role": "user", "content": f"Extract patient data:\n\n{clinical_note}"},
        ],
        response_format=SimplePatient,
    )
    elapsed = time.time() - start

    patient = completion.choices[0].message.parsed
    print(f"\n--- Structured Output (SimplePatient) ---")
    print(f"  Name:            {patient.name}")
    print(f"  Age:             {patient.age}")
    print(f"  Gender:          {patient.gender}")
    print(f"  Chief Complaint: {patient.chief_complaint}")
    print(f"  Assessment:      {patient.assessment}")
    print(f"\n  [Tokens: {completion.usage.total_tokens} | Time: {elapsed:.2f}s]")

    # Show the raw JSON as well
    raw_json = completion.choices[0].message.content
    print(f"\n--- Raw JSON ---")
    print(json.dumps(json.loads(raw_json), indent=2))


# ============================================================================
# DEMO 2: NESTED SCHEMAS
# ============================================================================

class NestedPatient(BaseModel):
    """Patient with nested medication list and vitals."""
    name: str = Field(..., description="Patient full name")
    age: Optional[int] = Field(None, description="Age in years")
    gender: Optional[str] = Field(None, description="Gender")
    chief_complaint: str = Field(..., description="Reason for visit")
    vitals: Optional[VitalSigns] = Field(None, description="Vital signs if available")
    medications: List[Medication] = Field(
        default_factory=list, description="Current medications"
    )
    assessment: str = Field(..., description="Clinical assessment")
    plan: List[str] = Field(default_factory=list, description="Plan items")


def demo_2_nested_schemas():
    """
    Complex nested objects: Patient with list of Medications, each with dosing.
    Demonstrates that structured outputs handle arbitrarily nested schemas.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Nested Schemas — Patient → Medications → Dosing")
    print("=" * 70)

    clinical_note = """
    Patient: Robert Thompson, 72-year-old male
    Vitals: BP 148/92, HR 88, Temp 98.6°F, RR 18, SpO2 96%
    Chief Complaint: Follow-up for hypertension and type 2 diabetes management.

    Current Medications:
    - Metformin 1000 mg twice daily by mouth for diabetes
    - Lisinopril 20 mg once daily by mouth for hypertension
    - Atorvastatin 40 mg once daily by mouth at bedtime for hyperlipidemia
    - Aspirin 81 mg once daily by mouth for cardiovascular prevention

    Assessment: Blood pressure remains above target despite current regimen.
    HbA1c improved to 7.2% from 8.1%. Lipids well controlled.

    Plan:
    1. Increase lisinopril to 40 mg daily
    2. Continue metformin, monitor renal function
    3. Continue atorvastatin
    4. Recheck BP in 2 weeks
    5. HbA1c recheck in 3 months
    """

    print(f"\n--- Input Clinical Note ---\n{clinical_note.strip()}")
    print("\n--- Extracting nested structure ---")

    start = time.time()
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract structured patient data including vitals, all medications "
                    "with complete dosing details, assessment, and plan items. "
                    "Be precise and thorough."
                ),
            },
            {"role": "user", "content": f"Extract:\n\n{clinical_note}"},
        ],
        response_format=NestedPatient,
    )
    elapsed = time.time() - start

    patient = completion.choices[0].message.parsed
    print(f"\n--- Structured Output (NestedPatient) ---")
    print(f"  Name: {patient.name}  |  Age: {patient.age}  |  Gender: {patient.gender}")
    print(f"  Chief Complaint: {patient.chief_complaint}")

    if patient.vitals:
        v = patient.vitals
        print(f"\n  Vitals:")
        print(f"    BP: {v.blood_pressure}  HR: {v.heart_rate}  Temp: {v.temperature}")
        print(f"    RR: {v.respiratory_rate}  SpO2: {v.oxygen_saturation}%")

    print(f"\n  Medications ({len(patient.medications)}):")
    for i, med in enumerate(patient.medications, 1):
        print(f"    {i}. {med.name}")
        print(f"       Dose: {med.dosing.amount} {med.dosing.frequency} ({med.dosing.route})")
        if med.indication:
            print(f"       Indication: {med.indication}")

    print(f"\n  Assessment: {patient.assessment}")
    print(f"\n  Plan:")
    for item in patient.plan:
        print(f"    • {item}")

    print(f"\n  [Tokens: {completion.usage.total_tokens} | Time: {elapsed:.2f}s]")


# ============================================================================
# DEMO 3: ENUM CONSTRAINTS
# ============================================================================

class ConditionStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    RESOLVED = "resolved"


class ClinicalCondition(BaseModel):
    """A clinical condition with constrained status values."""
    name: str = Field(..., description="Condition name")
    status: Literal["active", "inactive", "resolved"] = Field(
        ..., description="Current status of the condition"
    )
    onset_date: Optional[str] = Field(None, description="When the condition started")
    notes: Optional[str] = Field(None, description="Additional notes")


class ProblemList(BaseModel):
    """Patient problem list with constrained statuses."""
    patient_name: str = Field(..., description="Patient name")
    conditions: List[ClinicalCondition] = Field(
        ..., description="List of clinical conditions"
    )
    last_updated: Optional[str] = Field(None, description="Date of last update")


def demo_3_enum_constraints():
    """
    Use Literal types and enums to constrain output values.
    Status must be one of: active, inactive, resolved.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Enum Constraints — Literal Types for Value Restriction")
    print("=" * 70)

    clinical_history = """
    Patient: Sandra Lee

    Medical History:
    - Type 2 diabetes diagnosed 2019, currently managed with medication, HbA1c 7.0%
    - Hypertension since 2017, controlled on current regimen
    - Appendicitis in 2005, resolved after appendectomy
    - Seasonal allergies, currently symptomatic
    - Breast cancer diagnosed 2020, completed treatment 2021, no recurrence — in remission
    - Hypothyroidism since 2015, well controlled on levothyroxine
    - Ankle fracture in 2022, fully healed
    """

    print(f"\n--- Input Clinical History ---\n{clinical_history.strip()}")
    print("\n--- Extracting with enum constraints (active/inactive/resolved) ---")

    start = time.time()
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract the patient's problem list. For each condition, determine "
                    "its current status:\n"
                    "- 'active' = currently being treated/symptomatic\n"
                    "- 'inactive' = diagnosed but currently in remission or not symptomatic\n"
                    "- 'resolved' = condition is gone / fully healed\n"
                    "Include onset date if mentioned."
                ),
            },
            {"role": "user", "content": f"Extract problem list:\n\n{clinical_history}"},
        ],
        response_format=ProblemList,
    )
    elapsed = time.time() - start

    problem_list = completion.choices[0].message.parsed
    print(f"\n--- Problem List for {problem_list.patient_name} ---")
    print(f"{'Condition':<30} {'Status':<12} {'Onset':<12}")
    print("-" * 54)
    for cond in problem_list.conditions:
        status_icon = {"active": "🔴", "inactive": "🟡", "resolved": "🟢"}.get(
            cond.status, "⚪"
        )
        onset = cond.onset_date or "—"
        print(f"  {status_icon} {cond.name:<27} {cond.status:<12} {onset:<12}")
        if cond.notes:
            print(f"      Notes: {cond.notes}")

    active_count = sum(1 for c in problem_list.conditions if c.status == "active")
    inactive_count = sum(1 for c in problem_list.conditions if c.status == "inactive")
    resolved_count = sum(1 for c in problem_list.conditions if c.status == "resolved")
    print(f"\n  Summary: {active_count} active, {inactive_count} inactive, {resolved_count} resolved")
    print(f"  [Tokens: {completion.usage.total_tokens} | Time: {elapsed:.2f}s]")


# ============================================================================
# DEMO 4: SCHEMA EVOLUTION
# ============================================================================

class PatientV1(BaseModel):
    """Version 1 schema — basic patient info."""
    name: str
    age: Optional[int] = None
    chief_complaint: str
    assessment: str


class PatientV2(BaseModel):
    """Version 2 schema — extended with new fields, backward compatible."""
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None  # NEW in v2
    chief_complaint: str
    assessment: str
    icd10_codes: List[str] = Field(default_factory=list)  # NEW in v2
    severity: Optional[Literal["mild", "moderate", "severe"]] = None  # NEW in v2
    schema_version: Literal["2.0"] = "2.0"  # NEW in v2


def demo_4_schema_evolution():
    """
    Handle schema changes gracefully: v1 → v2 with backward compatibility.
    Shows how to evolve schemas without breaking existing data.
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Schema Evolution — v1 → v2 with Backward Compatibility")
    print("=" * 70)

    clinical_note = """
    Patient: James Chen, 45-year-old male
    Chief Complaint: Severe headache for 2 days with photophobia and nausea.
    Assessment: Migraine without aura. History of episodic migraines.
    ICD-10: G43.009 (Migraine, unspecified, not intractable, without status migrainosus)
    """

    print(f"\n--- Input Note ---\n{clinical_note.strip()}")

    # Extract with V1 schema
    print("\n--- Extracting with Schema V1 ---")
    start = time.time()
    completion_v1 = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Extract basic patient information from the note.",
            },
            {"role": "user", "content": clinical_note},
        ],
        response_format=PatientV1,
    )
    elapsed_v1 = time.time() - start
    patient_v1 = completion_v1.choices[0].message.parsed

    print(f"  V1 Output:")
    print(f"    Name:            {patient_v1.name}")
    print(f"    Age:             {patient_v1.age}")
    print(f"    Chief Complaint: {patient_v1.chief_complaint}")
    print(f"    Assessment:      {patient_v1.assessment}")
    print(f"    [Tokens: {completion_v1.usage.total_tokens} | Time: {elapsed_v1:.2f}s]")

    # Extract with V2 schema (new fields)
    print("\n--- Extracting with Schema V2 (new fields: gender, ICD-10, severity) ---")
    start = time.time()
    completion_v2 = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract patient information with extended details. Include ICD-10 "
                    "codes if mentioned, assess severity (mild/moderate/severe), and "
                    "include gender."
                ),
            },
            {"role": "user", "content": clinical_note},
        ],
        response_format=PatientV2,
    )
    elapsed_v2 = time.time() - start
    patient_v2 = completion_v2.choices[0].message.parsed

    print(f"  V2 Output:")
    print(f"    Name:            {patient_v2.name}")
    print(f"    Age:             {patient_v2.age}")
    print(f"    Gender:          {patient_v2.gender}")
    print(f"    Chief Complaint: {patient_v2.chief_complaint}")
    print(f"    Assessment:      {patient_v2.assessment}")
    print(f"    ICD-10 Codes:    {patient_v2.icd10_codes}")
    print(f"    Severity:        {patient_v2.severity}")
    print(f"    Schema Version:  {patient_v2.schema_version}")
    print(f"    [Tokens: {completion_v2.usage.total_tokens} | Time: {elapsed_v2:.2f}s]")

    # Show backward compatibility
    print("\n--- Backward Compatibility Check ---")
    v1_dict = patient_v1.model_dump()
    v2_dict = patient_v2.model_dump()
    common_keys = set(v1_dict.keys()) & set(v2_dict.keys())
    new_keys = set(v2_dict.keys()) - set(v1_dict.keys())
    print(f"  Common fields (V1 ∩ V2): {sorted(common_keys)}")
    print(f"  New fields in V2:        {sorted(new_keys)}")
    print(f"  V2 is backward compatible: all V1 fields present in V2 ✓")


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    """Interactive menu for structured output demos."""
    demos = {
        "1": ("JSON Schema Basics (response_format + parse)", demo_1_json_schema_basics),
        "2": ("Nested Schemas (Patient → Medications → Dosing)", demo_2_nested_schemas),
        "3": ("Enum Constraints (Literal types)", demo_3_enum_constraints),
        "4": ("Schema Evolution (v1 → v2)", demo_4_schema_evolution),
    }

    while True:
        print("\n" + "=" * 70)
        print("PROJECT 03: STRUCTURED OUTPUTS — DEMO MENU")
        print("=" * 70)
        for key, (desc, _) in demos.items():
            print(f"  [{key}] {desc}")
        print(f"  [a] Run all demos")
        print(f"  [q] Quit")

        choice = input("\nSelect demo: ").strip().lower()

        if choice == "q":
            print("Goodbye!")
            break
        elif choice == "a":
            for key in sorted(demos.keys()):
                demos[key][1]()
        elif choice in demos:
            demos[choice][1]()
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
