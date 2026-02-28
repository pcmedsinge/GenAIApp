"""
Exercise 4: FHIR-Compatible Data Extraction

Skills practiced:
  - Modeling FHIR resources (Patient, Condition, MedicationStatement) in Pydantic
  - Extracting FHIR-compatible data from free-text clinical notes
  - Generating valid FHIR JSON bundles from LLM output
  - Validating FHIR resource structure and required fields

Healthcare context:
  FHIR (Fast Healthcare Interoperability Resources) is the dominant standard for
  exchanging healthcare data between systems. If an LLM extracts patient data from
  a note and an EHR needs to ingest it, the data must conform to FHIR resource
  schemas — not arbitrary JSON.

  This exercise builds Pydantic models that mirror simplified FHIR resources:
    - Patient: demographics, identifiers, contact info
    - Condition: diagnoses with clinical status and codes
    - MedicationStatement: current medications with dosage instructions

  You'll extract all three from clinical notes and assemble them into a FHIR Bundle
  that could be sent to any FHIR-compliant system.
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
# FHIR-Compatible Pydantic Models
# ============================================================

# --- FHIR Patient Resource ---

class FHIRIdentifier(BaseModel):
    """FHIR Identifier datatype."""
    system: str = Field("urn:oid:2.16.840.1.113883.19.5", description="Identifier system")
    value: str = Field(..., description="Identifier value (e.g., MRN)")


class FHIRHumanName(BaseModel):
    """FHIR HumanName datatype."""
    family: str = Field(..., description="Family/last name")
    given: list[str] = Field(default_factory=list, description="Given/first names")
    text: Optional[str] = Field(None, description="Full name as a single string")


class FHIRPatient(BaseModel):
    """Simplified FHIR Patient resource."""
    resourceType: str = Field("Patient", description="Always 'Patient'")
    identifier: list[FHIRIdentifier] = Field(default_factory=list)
    name: list[FHIRHumanName] = Field(..., min_length=1)
    gender: str = Field(..., description="male | female | other | unknown")
    birthDate: Optional[str] = Field(None, description="YYYY-MM-DD format")

    @field_validator("resourceType")
    @classmethod
    def must_be_patient(cls, v: str) -> str:
        if v != "Patient":
            raise ValueError("resourceType must be 'Patient'")
        return v

    @field_validator("gender")
    @classmethod
    def valid_gender(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("male", "female", "other", "unknown"):
            raise ValueError(f"gender must be male/female/other/unknown, got '{v}'")
        return v


# --- FHIR Condition Resource ---

class FHIRCoding(BaseModel):
    """FHIR Coding datatype."""
    system: str = Field("http://hl7.org/fhir/sid/icd-10-cm", description="Code system")
    code: Optional[str] = Field(None, description="Code value (e.g., ICD-10)")
    display: str = Field(..., description="Human-readable name")


class FHIRCodeableConcept(BaseModel):
    """FHIR CodeableConcept datatype."""
    coding: list[FHIRCoding] = Field(default_factory=list)
    text: str = Field(..., description="Plain text representation")


class FHIRCondition(BaseModel):
    """Simplified FHIR Condition resource."""
    resourceType: str = Field("Condition", description="Always 'Condition'")
    clinicalStatus: str = Field(
        "active",
        description="active | recurrence | relapse | inactive | remission | resolved"
    )
    code: FHIRCodeableConcept = Field(..., description="Condition code and name")
    onsetString: Optional[str] = Field(None, description="When the condition started")

    @field_validator("resourceType")
    @classmethod
    def must_be_condition(cls, v: str) -> str:
        if v != "Condition":
            raise ValueError("resourceType must be 'Condition'")
        return v

    @field_validator("clinicalStatus")
    @classmethod
    def valid_status(cls, v: str) -> str:
        v = v.strip().lower()
        allowed = {"active", "recurrence", "relapse", "inactive", "remission", "resolved"}
        if v not in allowed:
            raise ValueError(f"clinicalStatus must be one of {allowed}, got '{v}'")
        return v


# --- FHIR MedicationStatement Resource ---

class FHIRDosage(BaseModel):
    """FHIR Dosage datatype (simplified)."""
    text: str = Field(..., description="Full dosage as text, e.g. '500mg twice daily'")
    route: Optional[str] = Field(None, description="Route: oral, IV, SubQ, etc.")
    frequency: Optional[str] = Field(None, description="e.g. BID, daily, Q8H")


class FHIRMedicationStatement(BaseModel):
    """Simplified FHIR MedicationStatement resource."""
    resourceType: str = Field("MedicationStatement", description="Always 'MedicationStatement'")
    status: str = Field("active", description="active | completed | stopped | on-hold")
    medicationCodeableConcept: FHIRCodeableConcept = Field(
        ..., description="Medication name and optional code"
    )
    dosage: list[FHIRDosage] = Field(default_factory=list)

    @field_validator("resourceType")
    @classmethod
    def must_be_med_statement(cls, v: str) -> str:
        if v != "MedicationStatement":
            raise ValueError("resourceType must be 'MedicationStatement'")
        return v

    @field_validator("status")
    @classmethod
    def valid_status(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("active", "completed", "stopped", "on-hold"):
            raise ValueError(f"status must be active/completed/stopped/on-hold, got '{v}'")
        return v


# --- FHIR Bundle ---

class FHIRBundleEntry(BaseModel):
    """A single entry in a FHIR Bundle."""
    resource: dict = Field(..., description="The FHIR resource as a dict")


class FHIRBundle(BaseModel):
    """Simplified FHIR Bundle to hold extracted resources."""
    resourceType: str = Field("Bundle")
    type: str = Field("collection")
    entry: list[FHIRBundleEntry] = Field(default_factory=list)

    def add_resource(self, resource: BaseModel) -> None:
        self.entry.append(FHIRBundleEntry(resource=resource.model_dump()))


# ============================================================
# Clinical Notes
# ============================================================

FHIR_NOTES = [
    {
        "id": "fhir_note_1",
        "text": (
            "Patient: Elena Vasquez, female, DOB: 1975-03-21. MRN: 9934021. "
            "Chief Complaint: Worsening joint pain and stiffness. "
            "History: Rheumatoid arthritis diagnosed 2018, Hypertension diagnosed 2020, "
            "Osteoporosis diagnosed 2023. "
            "Medications: Methotrexate 15mg weekly (oral), Folic acid 1mg daily, "
            "Amlodipine 5mg daily, Alendronate 70mg weekly (oral), Prednisone 5mg daily. "
            "Assessment: RA with moderate disease activity, HTN controlled, osteoporosis "
            "stable on therapy. "
            "Plan: Check CRP, ESR, CBC. Consider adding hydroxychloroquine. DXA scan in "
            "6 months. Continue current medications."
        ),
    },
    {
        "id": "fhir_note_2",
        "text": (
            "Pt: Michael Thompson, male, born 1960-11-09. MRN 5578103. "
            "CC: Follow-up after CABG surgery (6 weeks post-op). "
            "PMH: Coronary artery disease (3-vessel, CABG 2026-01-15), Type 2 diabetes "
            "mellitus, Hyperlipidemia, former smoker (quit 2025). "
            "Meds: Aspirin 81mg daily, Clopidogrel 75mg daily, Atorvastatin 80mg daily, "
            "Metoprolol succinate 50mg daily, Lisinopril 10mg daily, Metformin 1000mg BID, "
            "Insulin glargine 20 units SubQ at bedtime. "
            "Assessment: Post-CABG recovery on track. DM2 control improving — last A1c 7.1%. "
            "CAD stable on dual antiplatelet therapy. "
            "Plan: Continue all meds. Cardiac rehab phase 2. Check A1c in 3 months. Lipid "
            "panel in 6 weeks. Follow-up cardiology in 3 months."
        ),
    },
]


# ============================================================
# FHIR Extraction Functions
# ============================================================

def extract_fhir_patient(note_text: str) -> Optional[FHIRPatient]:
    """Extract a FHIR Patient resource from a clinical note."""
    schema = json.dumps(FHIRPatient.model_json_schema(), indent=2)
    system_prompt = f"""Extract a FHIR Patient resource from this clinical note.
Return valid JSON matching this schema:

{schema}

Rules:
- resourceType must be "Patient"
- Split name into family (last) and given (first, middle) names
- gender: "male" or "female"
- birthDate: YYYY-MM-DD format
- identifier: use MRN as the value
Return ONLY valid JSON."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": note_text},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()
        return FHIRPatient.model_validate_json(raw)
    except (ValidationError, Exception) as e:
        print(f"  ✗ Patient extraction failed: {e}")
        return None


def extract_fhir_conditions(note_text: str) -> list[FHIRCondition]:
    """Extract FHIR Condition resources from a clinical note."""
    schema = json.dumps(FHIRCondition.model_json_schema(), indent=2)
    system_prompt = f"""Extract ALL diagnoses/conditions from this clinical note as FHIR Condition resources.
Return a JSON object with key "conditions" containing an array of Condition resources.

Each condition must match this schema:
{schema}

Rules:
- resourceType must be "Condition"
- clinicalStatus: "active" for current problems, "resolved" for resolved ones
- code.text: full condition name (expand abbreviations: HTN → Hypertension)
- code.coding: include ICD-10 code if you can identify it
- onsetString: when the condition started, if mentioned
Return ONLY valid JSON."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": note_text},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1500,
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        conditions = []
        for item in data.get("conditions", []):
            try:
                conditions.append(FHIRCondition.model_validate(item))
            except ValidationError as e:
                print(f"  ⚠ Skipping invalid condition: {e.errors()[0]['msg']}")
        return conditions
    except Exception as e:
        print(f"  ✗ Condition extraction failed: {e}")
        return []


def extract_fhir_medications(note_text: str) -> list[FHIRMedicationStatement]:
    """Extract FHIR MedicationStatement resources from a clinical note."""
    schema = json.dumps(FHIRMedicationStatement.model_json_schema(), indent=2)
    system_prompt = f"""Extract ALL medications from this clinical note as FHIR MedicationStatement resources.
Return a JSON object with key "medications" containing an array of MedicationStatement resources.

Each medication must match this schema:
{schema}

Rules:
- resourceType must be "MedicationStatement"
- status: "active" for current meds
- medicationCodeableConcept.text: medication name
- dosage.text: full dosage string (e.g., "500mg twice daily")
- dosage.route: oral, IV, SubQ, etc.
- dosage.frequency: BID, daily, weekly, etc.
Return ONLY valid JSON."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": note_text},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        meds = []
        for item in data.get("medications", []):
            try:
                meds.append(FHIRMedicationStatement.model_validate(item))
            except ValidationError as e:
                print(f"  ⚠ Skipping invalid medication: {e.errors()[0]['msg']}")
        return meds
    except Exception as e:
        print(f"  ✗ Medication extraction failed: {e}")
        return []


def build_fhir_bundle(
    patient: Optional[FHIRPatient],
    conditions: list[FHIRCondition],
    medications: list[FHIRMedicationStatement],
) -> FHIRBundle:
    """Assemble extracted resources into a FHIR Bundle."""
    bundle = FHIRBundle()
    if patient:
        bundle.add_resource(patient)
    for condition in conditions:
        bundle.add_resource(condition)
    for med in medications:
        bundle.add_resource(med)
    return bundle


# ============================================================
# Main
# ============================================================

def main():
    """Extract FHIR-compatible data from clinical notes."""
    print("=" * 65)
    print("Exercise 4: FHIR-Compatible Data Extraction")
    print("=" * 65)

    all_bundles = []

    for note in FHIR_NOTES:
        print(f"\n{'─' * 60}")
        print(f"Processing: {note['id']}")
        print(f"{'─' * 60}")

        # Extract each resource type
        print("\n  Extracting FHIR Patient...")
        patient = extract_fhir_patient(note["text"])
        if patient:
            name = patient.name[0]
            print(f"  ✓ Patient: {name.text or f'{name.given} {name.family}'}")
            print(f"    Gender: {patient.gender}, DOB: {patient.birthDate}")
            if patient.identifier:
                print(f"    MRN: {patient.identifier[0].value}")

        print("\n  Extracting FHIR Conditions...")
        conditions = extract_fhir_conditions(note["text"])
        print(f"  ✓ Found {len(conditions)} conditions:")
        for c in conditions:
            code_str = ""
            if c.code.coding and c.code.coding[0].code:
                code_str = f" [{c.code.coding[0].code}]"
            onset = f", onset: {c.onsetString}" if c.onsetString else ""
            print(f"    - {c.code.text}{code_str} ({c.clinicalStatus}{onset})")

        print("\n  Extracting FHIR MedicationStatements...")
        medications = extract_fhir_medications(note["text"])
        print(f"  ✓ Found {len(medications)} medications:")
        for m in medications:
            dosage_str = m.dosage[0].text if m.dosage else "no dosage"
            route = m.dosage[0].route if m.dosage and m.dosage[0].route else "N/A"
            print(f"    - {m.medicationCodeableConcept.text}: {dosage_str} ({route})")

        # Build and display the FHIR Bundle
        bundle = build_fhir_bundle(patient, conditions, medications)
        all_bundles.append(bundle)

        print(f"\n  FHIR Bundle: {len(bundle.entry)} entries")

    # Summary
    print("\n" + "=" * 65)
    print("FHIR EXTRACTION SUMMARY")
    print("=" * 65)
    for i, (note, bundle) in enumerate(zip(FHIR_NOTES, all_bundles)):
        resource_types = []
        for entry in bundle.entry:
            rt = entry.resource.get("resourceType", "Unknown")
            resource_types.append(rt)

        type_counts = {}
        for rt in resource_types:
            type_counts[rt] = type_counts.get(rt, 0) + 1

        print(f"\n  {note['id']}:")
        print(f"    Total resources: {len(bundle.entry)}")
        for rt, count in type_counts.items():
            print(f"    - {rt}: {count}")

    # Export first bundle as sample
    if all_bundles:
        print("\n" + "─" * 60)
        print("Sample FHIR Bundle JSON (first note):")
        print("─" * 60)
        bundle_json = all_bundles[0].model_dump()
        print(json.dumps(bundle_json, indent=2, default=str)[:2000])
        if len(json.dumps(bundle_json)) > 2000:
            print("  ... (truncated)")

    print("\n💡 Key takeaways:")
    print("  - FHIR resources have specific required fields and value sets")
    print("  - Pydantic models mirror FHIR structure for validation")
    print("  - LLMs can extract FHIR-compatible data with good prompting")
    print("  - Real FHIR validation would use a FHIR validator (e.g., HAPI FHIR)")


if __name__ == "__main__":
    main()
