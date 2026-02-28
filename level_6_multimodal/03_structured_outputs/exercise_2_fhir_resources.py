"""
Exercise 2: FHIR-Compatible Resource Generation
=================================================
Generate FHIR (Fast Healthcare Interoperability Resources) compatible resources
from clinical text. Defines Pydantic models for FHIR Patient, Condition, and
MedicationRequest resources with proper FHIR structure.

Learning Objectives:
  - Model FHIR resource structures in Pydantic
  - Extract FHIR-compatible data from unstructured clinical notes
  - Understand FHIR resource relationships and references
  - Validate resource structure against FHIR conventions

Usage:
  python exercise_2_fhir_resources.py
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
# FHIR RESOURCE SCHEMAS
# ============================================================================

class FHIRCoding(BaseModel):
    """FHIR Coding element — a code from a code system."""
    system: str = Field(..., description="Code system URI, e.g. http://hl7.org/fhir/sid/icd-10-cm")
    code: str = Field(..., description="Code value")
    display: str = Field(..., description="Human-readable display text")


class FHIRCodeableConcept(BaseModel):
    """FHIR CodeableConcept — a concept with codings and text."""
    coding: List[FHIRCoding] = Field(default_factory=list, description="Code(s)")
    text: str = Field(..., description="Plain text representation")


class FHIRReference(BaseModel):
    """FHIR Reference to another resource."""
    reference: str = Field(..., description="Resource reference, e.g. Patient/123")
    display: Optional[str] = Field(None, description="Display text")


class FHIRHumanName(BaseModel):
    """FHIR HumanName element."""
    use: Literal["official", "usual", "nickname", "old"] = Field("official")
    family: str = Field(..., description="Family/last name")
    given: List[str] = Field(default_factory=list, description="Given/first names")
    prefix: Optional[List[str]] = Field(None, description="Name prefix, e.g. Mr., Dr.")


class FHIRPatient(BaseModel):
    """FHIR Patient resource (simplified)."""
    resourceType: Literal["Patient"] = "Patient"
    id: Optional[str] = Field(None, description="Resource ID")
    name: List[FHIRHumanName] = Field(..., description="Patient names")
    gender: Optional[Literal["male", "female", "other", "unknown"]] = Field(None)
    birthDate: Optional[str] = Field(None, description="Date of birth (YYYY-MM-DD)")
    active: bool = Field(True, description="Whether record is active")


class FHIRCondition(BaseModel):
    """FHIR Condition resource (simplified)."""
    resourceType: Literal["Condition"] = "Condition"
    id: Optional[str] = Field(None, description="Resource ID")
    clinicalStatus: FHIRCodeableConcept = Field(
        ..., description="active | recurrence | relapse | inactive | remission | resolved"
    )
    verificationStatus: Optional[FHIRCodeableConcept] = Field(None)
    code: FHIRCodeableConcept = Field(..., description="Condition code (ICD-10)")
    subject: FHIRReference = Field(..., description="Reference to Patient")
    onsetDateTime: Optional[str] = Field(None, description="When condition started")
    note: Optional[List[dict]] = Field(None, description="Additional notes")


class FHIRDosage(BaseModel):
    """FHIR Dosage element."""
    text: str = Field(..., description="Free text dosage instructions")
    route: Optional[FHIRCodeableConcept] = Field(None, description="Route of administration")


class FHIRMedicationRequest(BaseModel):
    """FHIR MedicationRequest resource (simplified)."""
    resourceType: Literal["MedicationRequest"] = "MedicationRequest"
    id: Optional[str] = Field(None, description="Resource ID")
    status: Literal["active", "on-hold", "cancelled", "completed", "stopped"] = Field(
        "active", description="Order status"
    )
    intent: Literal["order", "plan", "proposal"] = Field("order")
    medicationCodeableConcept: FHIRCodeableConcept = Field(
        ..., description="Medication code and name"
    )
    subject: FHIRReference = Field(..., description="Reference to Patient")
    dosageInstruction: List[FHIRDosage] = Field(
        default_factory=list, description="Dosing instructions"
    )
    reasonCode: Optional[List[FHIRCodeableConcept]] = Field(
        None, description="Reason for medication"
    )


class FHIRBundle(BaseModel):
    """A collection of FHIR resources extracted from a clinical note."""
    patient: FHIRPatient = Field(..., description="Patient resource")
    conditions: List[FHIRCondition] = Field(
        default_factory=list, description="Condition resources"
    )
    medication_requests: List[FHIRMedicationRequest] = Field(
        default_factory=list, description="MedicationRequest resources"
    )


# ============================================================================
# SAMPLE CLINICAL NOTES
# ============================================================================

SAMPLE_NOTES = [
    {
        "title": "Cardiology Follow-up",
        "text": """
        Patient: William Foster, Male, DOB: 1955-06-12
        MRN: CF-90012

        Visit Date: 2025-11-15

        Active Problems:
        1. Coronary artery disease (I25.10) — diagnosed 2020, s/p PCI with
           drug-eluting stent to LAD. Currently stable angina class I.
        2. Heart failure with reduced ejection fraction (I50.22) — EF 35%,
           NYHA class II. Diagnosed 2021.
        3. Atrial fibrillation (I48.91) — persistent, rate-controlled.
           Diagnosed 2022.
        4. Type 2 diabetes (E11.65) — chronic, HbA1c 7.1%.

        Current Medications:
        1. Metoprolol succinate 100 mg oral once daily — heart rate control
        2. Lisinopril 10 mg oral once daily — heart failure / BP
        3. Apixaban 5 mg oral twice daily — anticoagulation for AFib
        4. Atorvastatin 80 mg oral at bedtime — CAD secondary prevention
        5. Metformin 1000 mg oral twice daily — diabetes
        6. Sacubitril/valsartan 49/51 mg oral twice daily — heart failure
        7. Furosemide 20 mg oral once daily — volume management
        """,
    },
    {
        "title": "Pediatric Well-Child Visit",
        "text": """
        Patient: Emma Rodriguez, Female, DOB: 2020-03-22

        Visit Date: 2025-09-22 (age 5 years)

        Active Problems:
        1. Asthma, mild persistent (J45.30) — diagnosed age 3, well-controlled
           on current regimen
        2. Allergic rhinitis (J30.1) — seasonal, active in spring/fall

        Resolved:
        3. Otitis media (H66.90) — resolved, last episode Feb 2025

        Medications:
        1. Fluticasone nasal spray 50 mcg, 1 spray each nostril daily — allergies
        2. Montelukast 4 mg chewable tablet oral once daily at bedtime — asthma
        3. Albuterol MDI 90 mcg, 2 puffs inhaled as needed — rescue inhaler
        """,
    },
]


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_fhir_bundle(clinical_note: str) -> FHIRBundle:
    """
    Extract FHIR-compatible resources from a clinical note.
    Returns a bundle with Patient, Condition, and MedicationRequest resources.
    """
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a FHIR resource extraction system. Extract structured "
                    "FHIR-compatible resources from clinical notes.\n\n"
                    "Rules:\n"
                    "1. Generate proper FHIR Patient, Condition, and MedicationRequest resources\n"
                    "2. Use ICD-10-CM codes with system 'http://hl7.org/fhir/sid/icd-10-cm'\n"
                    "3. Use RxNorm for medication codes (system 'http://www.nlm.nih.gov/research/umls/rxnorm')\n"
                    "4. Set clinicalStatus using FHIR clinical status codes (active, resolved, etc.)\n"
                    "5. Reference the patient in each Condition and MedicationRequest\n"
                    "6. Use YYYY-MM-DD date format\n"
                    "7. For medication codes, use the medication name as the code if RxNorm code is unknown"
                ),
            },
            {
                "role": "user",
                "content": f"Extract FHIR resources from:\n\n{clinical_note}",
            },
        ],
        response_format=FHIRBundle,
    )
    return completion.choices[0].message.parsed


def display_fhir_bundle(bundle: FHIRBundle, title: str = ""):
    """Display FHIR bundle resources."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  FHIR Bundle — {title}")
        print(f"{'=' * 60}")

    # Patient
    p = bundle.patient
    name = p.name[0] if p.name else None
    name_str = f"{', '.join(name.given)} {name.family}" if name else "Unknown"
    print(f"\n  📋 FHIR Patient")
    print(f"  {'─' * 45}")
    print(f"    Name:      {name_str}")
    print(f"    Gender:    {p.gender or '—'}")
    print(f"    BirthDate: {p.birthDate or '—'}")
    print(f"    Active:    {p.active}")

    # Conditions
    if bundle.conditions:
        print(f"\n  🏥 FHIR Conditions ({len(bundle.conditions)})")
        print(f"  {'─' * 45}")
        for i, cond in enumerate(bundle.conditions, 1):
            code_display = cond.code.text
            coding_info = ""
            if cond.code.coding:
                c = cond.code.coding[0]
                coding_info = f" [{c.code}]"
            status = cond.clinicalStatus.text
            print(f"    {i}. {code_display}{coding_info}")
            print(f"       Status: {status}")
            if cond.onsetDateTime:
                print(f"       Onset:  {cond.onsetDateTime}")

    # MedicationRequests
    if bundle.medication_requests:
        print(f"\n  💊 FHIR MedicationRequests ({len(bundle.medication_requests)})")
        print(f"  {'─' * 45}")
        for i, med in enumerate(bundle.medication_requests, 1):
            med_name = med.medicationCodeableConcept.text
            dosage = med.dosageInstruction[0].text if med.dosageInstruction else "—"
            print(f"    {i}. {med_name}")
            print(f"       Dosage: {dosage}")
            print(f"       Status: {med.status} | Intent: {med.intent}")
            if med.reasonCode:
                reasons = [r.text for r in med.reasonCode]
                print(f"       Reason: {', '.join(reasons)}")


def validate_fhir_bundle(bundle: FHIRBundle) -> dict:
    """Validate FHIR bundle for completeness and correctness."""
    results = {"valid": True, "warnings": [], "errors": []}

    # Check patient has a name
    if not bundle.patient.name:
        results["errors"].append("Patient missing name")
        results["valid"] = False

    # Check conditions reference patient
    for i, cond in enumerate(bundle.conditions):
        if not cond.subject.reference:
            results["errors"].append(f"Condition {i+1} missing patient reference")
            results["valid"] = False
        if not cond.code.coding:
            results["warnings"].append(f"Condition {i+1} ({cond.code.text}) missing coding")

    # Check medication requests
    for i, med in enumerate(bundle.medication_requests):
        if not med.subject.reference:
            results["errors"].append(f"MedicationRequest {i+1} missing patient reference")
            results["valid"] = False
        if not med.dosageInstruction:
            results["warnings"].append(
                f"MedicationRequest {i+1} ({med.medicationCodeableConcept.text}) missing dosage"
            )

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Generate FHIR resources from clinical notes."""
    print("=" * 60)
    print("EXERCISE 2: FHIR-Compatible Resource Generation")
    print("=" * 60)

    for note_info in SAMPLE_NOTES:
        print(f"\n\n{'#' * 60}")
        print(f"  Processing: {note_info['title']}")
        print(f"{'#' * 60}")

        # Extract
        start = time.time()
        bundle = extract_fhir_bundle(note_info["text"])
        elapsed = time.time() - start

        # Display
        display_fhir_bundle(bundle, title=note_info["title"])

        # Validate
        validation = validate_fhir_bundle(bundle)
        print(f"\n  ✅ FHIR VALIDATION")
        print(f"  {'─' * 45}")
        print(f"    Overall: {'VALID ✓' if validation['valid'] else 'INVALID ✗'}")
        if validation["errors"]:
            for err in validation["errors"]:
                print(f"    ❌ Error: {err}")
        if validation["warnings"]:
            for warn in validation["warnings"]:
                print(f"    ⚠  Warning: {warn}")
        if not validation["errors"] and not validation["warnings"]:
            print(f"    No issues found")

        # Show FHIR JSON for patient resource
        print(f"\n  📄 FHIR Patient JSON:")
        patient_json = bundle.patient.model_dump(exclude_none=True)
        print(json.dumps(patient_json, indent=4))

        print(f"\n  [Extraction time: {elapsed:.2f}s]")

    # Summary
    print(f"\n\n{'=' * 60}")
    print("Exercise complete! Key takeaways:")
    print("  • FHIR resources can be modeled with Pydantic for type safety")
    print("  • Structured outputs guarantee FHIR-compatible JSON")
    print("  • ICD-10 and RxNorm coding enables interoperability")
    print("  • Validation catches structural issues before integration")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
