"""
Exercise 3: Structured Lab Report Generation
=============================================
Generate structured lab reports from clinical text. Each report includes
patient identifiers, ordered tests with results, reference ranges,
and interpretations (normal, high, low, critical).

Schema: LabReport(patient_id, tests: List[LabTest(name, value, unit,
        reference_range, interpretation)])

Learning Objectives:
  - Model lab report schemas with proper units and ranges
  - Use enums/literals to constrain interpretation values
  - Handle panels (CBC, CMP, lipid) as grouped tests
  - Flag critical values automatically

Usage:
  python exercise_3_lab_reports.py
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

class LabTest(BaseModel):
    """A single laboratory test result."""
    name: str = Field(..., description="Test name, e.g. 'Hemoglobin'")
    value: str = Field(..., description="Result value as string (handles numeric and text)")
    numeric_value: Optional[float] = Field(None, description="Numeric value if applicable")
    unit: Optional[str] = Field(None, description="Unit of measurement, e.g. 'g/dL'")
    reference_range: Optional[str] = Field(None, description="Normal range, e.g. '12.0-16.0'")
    interpretation: Literal["normal", "high", "low", "critical_high", "critical_low", "abnormal"] = Field(
        ..., description="Interpretation of the result"
    )
    is_critical: bool = Field(False, description="Whether this is a critical value requiring immediate action")
    notes: Optional[str] = Field(None, description="Additional notes or context")


class LabPanel(BaseModel):
    """A group of related lab tests."""
    panel_name: str = Field(..., description="Panel name, e.g. 'Complete Blood Count'")
    tests: List[LabTest] = Field(..., description="Individual tests in the panel")


class LabReport(BaseModel):
    """Complete structured lab report."""
    patient_id: str = Field(..., description="Patient identifier")
    patient_name: Optional[str] = Field(None, description="Patient name")
    collection_date: Optional[str] = Field(None, description="Specimen collection date")
    report_date: Optional[str] = Field(None, description="Report date")
    ordering_provider: Optional[str] = Field(None, description="Ordering physician")
    panels: List[LabPanel] = Field(default_factory=list, description="Lab panels")
    summary: str = Field(..., description="Brief summary of significant findings")
    critical_values_present: bool = Field(
        False, description="Whether any critical values were found"
    )
    follow_up_recommended: Optional[str] = Field(
        None, description="Recommended follow-up based on results"
    )


# ============================================================================
# SAMPLE LAB DATA
# ============================================================================

SAMPLE_LAB_TEXTS = [
    {
        "title": "Comprehensive Metabolic Panel + CBC",
        "text": """
        LABORATORY REPORT
        Patient: Anderson, Robert J.    MRN: LAB-20251234
        DOB: 04/18/1960    Collection Date: 2025-11-20
        Ordering Provider: Dr. Sarah Chen

        COMPLETE BLOOD COUNT (CBC):
        WBC: 12.8 K/uL (Ref: 4.5-11.0) — HIGH
        RBC: 4.2 M/uL (Ref: 4.5-5.5) — LOW
        Hemoglobin: 11.2 g/dL (Ref: 13.5-17.5) — LOW
        Hematocrit: 33.8% (Ref: 38.0-50.0) — LOW
        MCV: 80.5 fL (Ref: 80-100) — NORMAL
        Platelets: 245 K/uL (Ref: 150-400) — NORMAL
        Neutrophils: 78% (Ref: 40-70) — HIGH

        COMPREHENSIVE METABOLIC PANEL (CMP):
        Glucose: 242 mg/dL (Ref: 70-100) — HIGH (CRITICAL >400)
        BUN: 32 mg/dL (Ref: 7-20) — HIGH
        Creatinine: 1.8 mg/dL (Ref: 0.7-1.3) — HIGH
        eGFR: 38 mL/min (Ref: >60) — LOW
        Sodium: 141 mEq/L (Ref: 136-145) — NORMAL
        Potassium: 5.6 mEq/L (Ref: 3.5-5.0) — HIGH (CRITICAL >6.0)
        Chloride: 104 mEq/L (Ref: 98-106) — NORMAL
        CO2: 20 mEq/L (Ref: 23-29) — LOW
        Calcium: 9.1 mg/dL (Ref: 8.5-10.5) — NORMAL
        Total Protein: 6.8 g/dL (Ref: 6.0-8.3) — NORMAL
        Albumin: 3.2 g/dL (Ref: 3.5-5.0) — LOW
        AST: 45 U/L (Ref: 10-40) — HIGH
        ALT: 52 U/L (Ref: 7-56) — NORMAL
        Alk Phos: 110 U/L (Ref: 44-147) — NORMAL
        Bilirubin Total: 1.0 mg/dL (Ref: 0.1-1.2) — NORMAL

        HbA1c: 9.8% (Ref: <5.7 normal, 5.7-6.4 prediabetes, >6.5 diabetes) — HIGH
        """,
    },
    {
        "title": "Lipid Panel + Thyroid + Iron Studies",
        "text": """
        LABORATORY REPORT
        Patient: Martinez, Isabella    MRN: LAB-20259876
        DOB: 09/30/1985    Collection Date: 2025-11-18
        Ordering Provider: Dr. Michael Torres

        LIPID PANEL (fasting):
        Total Cholesterol: 268 mg/dL (Ref: <200 desirable) — HIGH
        LDL Cholesterol: 178 mg/dL (Ref: <100 optimal) — HIGH
        HDL Cholesterol: 42 mg/dL (Ref: >60 desirable) — LOW
        Triglycerides: 240 mg/dL (Ref: <150) — HIGH
        VLDL: 48 mg/dL (Ref: <30) — HIGH
        Total/HDL Ratio: 6.4 (Ref: <5.0) — HIGH

        THYROID PANEL:
        TSH: 8.2 mIU/L (Ref: 0.4-4.0) — HIGH
        Free T4: 0.7 ng/dL (Ref: 0.8-1.8) — LOW
        Free T3: 2.0 pg/mL (Ref: 2.3-4.2) — LOW

        IRON STUDIES:
        Serum Iron: 35 mcg/dL (Ref: 60-170) — LOW
        TIBC: 420 mcg/dL (Ref: 250-370) — HIGH
        Transferrin Saturation: 8% (Ref: 20-50) — LOW
        Ferritin: 8 ng/mL (Ref: 12-150) — LOW (CRITICAL <10)
        """,
    },
    {
        "title": "Cardiac Markers (Emergency)",
        "text": """
        LABORATORY REPORT — STAT
        Patient: Williams, David    MRN: LAB-20250001
        DOB: 02/14/1970    Collection Date: 2025-11-21 03:45 AM
        Ordering Provider: Dr. Lisa Park (Emergency Medicine)

        CARDIAC MARKERS:
        Troponin I: 2.4 ng/mL (Ref: <0.04) — CRITICAL HIGH
        CK-MB: 45 U/L (Ref: <25) — HIGH
        CK Total: 580 U/L (Ref: 30-200) — HIGH
        BNP: 1250 pg/mL (Ref: <100) — CRITICAL HIGH
        Myoglobin: 320 ng/mL (Ref: <90) — HIGH

        D-DIMER: 0.3 mg/L FEU (Ref: <0.5) — NORMAL

        BASIC METABOLIC PANEL (STAT):
        Glucose: 186 mg/dL (Ref: 70-100) — HIGH
        Potassium: 4.2 mEq/L (Ref: 3.5-5.0) — NORMAL
        Creatinine: 1.1 mg/dL (Ref: 0.7-1.3) — NORMAL
        Magnesium: 1.8 mg/dL (Ref: 1.7-2.2) — NORMAL
        """,
    },
]


# ============================================================================
# EXTRACTION AND ANALYSIS
# ============================================================================

def extract_lab_report(lab_text: str) -> LabReport:
    """Extract a structured lab report from text."""
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical laboratory information system. Extract structured "
                    "lab results from the report text.\n\n"
                    "Rules:\n"
                    "1. Group tests into their panels (CBC, CMP, Lipid, etc.)\n"
                    "2. Set interpretation: normal, high, low, critical_high, critical_low, abnormal\n"
                    "3. Mark is_critical=true for values flagged as critical or dangerously abnormal\n"
                    "4. Include numeric_value when the result is a number\n"
                    "5. Include reference ranges as stated\n"
                    "6. Write a brief summary highlighting significant findings\n"
                    "7. Set critical_values_present=true if any critical values exist\n"
                    "8. Recommend follow-up based on abnormal results"
                ),
            },
            {
                "role": "user",
                "content": f"Extract structured lab report:\n\n{lab_text}",
            },
        ],
        response_format=LabReport,
    )
    return completion.choices[0].message.parsed


def display_lab_report(report: LabReport):
    """Display structured lab report with visual indicators."""
    print(f"\n  STRUCTURED LAB REPORT")
    print(f"  {'═' * 55}")
    print(f"  Patient:   {report.patient_name or '—'} ({report.patient_id})")
    print(f"  Collected: {report.collection_date or '—'}")
    print(f"  Provider:  {report.ordering_provider or '—'}")

    if report.critical_values_present:
        print(f"\n  🚨 CRITICAL VALUES PRESENT — IMMEDIATE ACTION REQUIRED 🚨")

    for panel in report.panels:
        print(f"\n  ┌─ {panel.panel_name} ─{'─' * (45 - len(panel.panel_name))}")
        print(f"  │ {'Test':<25} {'Result':<15} {'Ref Range':<15} {'Status':<12}")
        print(f"  │ {'─'*25} {'─'*15} {'─'*15} {'─'*12}")

        for test in panel.tests:
            # Status indicator
            status_icons = {
                "normal": "  ✓",
                "high": " ⬆ H",
                "low": " ⬇ L",
                "critical_high": "🔴⬆ CH",
                "critical_low": "🔴⬇ CL",
                "abnormal": " ⚠ A",
            }
            icon = status_icons.get(test.interpretation, "  ?")

            value_str = f"{test.value} {test.unit or ''}"
            ref_str = test.reference_range or "—"
            print(f"  │ {test.name:<25} {value_str:<15} {ref_str:<15} {icon}")

    print(f"\n  Summary: {report.summary}")

    if report.follow_up_recommended:
        print(f"\n  📋 Follow-up: {report.follow_up_recommended}")


def analyze_abnormals(report: LabReport) -> dict:
    """Analyze abnormal and critical values in the report."""
    total_tests = sum(len(p.tests) for p in report.panels)
    abnormal = []
    critical = []

    for panel in report.panels:
        for test in panel.tests:
            if test.interpretation != "normal":
                abnormal.append(test)
            if test.is_critical:
                critical.append(test)

    return {
        "total_tests": total_tests,
        "normal_count": total_tests - len(abnormal),
        "abnormal_count": len(abnormal),
        "critical_count": len(critical),
        "abnormal_tests": abnormal,
        "critical_tests": critical,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Process multiple lab reports."""
    print("=" * 60)
    print("EXERCISE 3: Structured Lab Report Generation")
    print("=" * 60)

    for lab_info in SAMPLE_LAB_TEXTS:
        print(f"\n\n{'#' * 60}")
        print(f"  Processing: {lab_info['title']}")
        print(f"{'#' * 60}")

        # Extract
        start = time.time()
        report = extract_lab_report(lab_info["text"])
        elapsed = time.time() - start

        # Display
        display_lab_report(report)

        # Analyze
        analysis = analyze_abnormals(report)
        print(f"\n  📊 ANALYSIS")
        print(f"  {'─' * 45}")
        print(f"    Total tests:    {analysis['total_tests']}")
        print(f"    Normal:         {analysis['normal_count']} ✓")
        print(f"    Abnormal:       {analysis['abnormal_count']} ⚠")
        print(f"    Critical:       {analysis['critical_count']} 🔴")

        if analysis["critical_tests"]:
            print(f"\n    🚨 CRITICAL VALUES:")
            for t in analysis["critical_tests"]:
                print(f"      • {t.name}: {t.value} {t.unit or ''} "
                      f"(ref: {t.reference_range or '—'})")

        print(f"\n  [Extraction time: {elapsed:.2f}s]")

    # Summary
    print(f"\n\n{'=' * 60}")
    print("Exercise complete! Key takeaways:")
    print("  • Lab reports benefit greatly from structured output schemas")
    print("  • Literal types constrain interpretations to valid values")
    print("  • Critical value flags enable automated alerting workflows")
    print("  • Grouped panels maintain logical test organization")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
