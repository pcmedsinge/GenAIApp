"""
Exercise 3: Complex Case Analyzer
===================================
Analyze complex clinical cases combining patient history (text), lab results
(structured data), and imaging descriptions using a reasoning model for
comprehensive assessment.

Learning Objectives:
  - Synthesize multiple data sources for clinical reasoning
  - Use reasoning models for complex multi-factor analysis
  - Generate structured assessments from heterogeneous inputs
  - Implement confidence scoring for clinical decisions

Usage:
  python exercise_3_case_analyzer.py
"""

import json
import os
import time
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI()
MODEL = "gpt-4o"
REASONING_MODEL = "o1-mini"


# ============================================================================
# SCHEMAS
# ============================================================================

class LabResult(BaseModel):
    """A single lab result with interpretation."""
    test: str
    value: str
    unit: str
    reference_range: str
    is_abnormal: bool
    clinical_significance: Optional[str] = None


class ImagingFinding(BaseModel):
    """A finding from medical imaging."""
    modality: str = Field(..., description="CT, MRI, X-ray, Echo, etc.")
    body_region: str
    finding: str
    clinical_significance: Literal["normal", "incidental", "significant", "critical"]


class DataSourceSummary(BaseModel):
    """Summary of each input data source."""
    source_type: str = Field(..., description="history, labs, imaging")
    key_findings: List[str]
    abnormalities: List[str]
    missing_data: List[str]


class ClinicalAssessment(BaseModel):
    """Comprehensive clinical assessment from multiple data sources."""
    patient_summary: str = Field(..., description="Brief patient summary")
    data_sources_analyzed: List[DataSourceSummary] = Field(
        ..., description="Summary of each data source"
    )
    primary_diagnosis: str = Field(..., description="Most likely diagnosis")
    primary_diagnosis_confidence: float = Field(
        ..., ge=0, le=1, description="Confidence 0-1"
    )
    supporting_evidence: List[str] = Field(
        ..., description="Evidence supporting primary diagnosis"
    )
    differential_diagnoses: List[str] = Field(
        ..., description="Alternative diagnoses ranked by likelihood"
    )
    critical_findings: List[str] = Field(
        default_factory=list, description="Findings requiring immediate attention"
    )
    data_gaps: List[str] = Field(
        default_factory=list, description="Missing information that would aid diagnosis"
    )
    recommended_actions: List[str] = Field(
        ..., description="Recommended next steps"
    )
    urgency: Literal["emergent", "urgent", "semi-urgent", "routine"] = Field(
        ..., description="Overall urgency"
    )
    reasoning_summary: str = Field(
        ..., description="Summary of clinical reasoning process"
    )


# ============================================================================
# COMPLEX CASES
# ============================================================================

COMPLEX_CASES = [
    {
        "title": "Unexplained Dyspnea with Multi-System Findings",
        "history": """
        PATIENT HISTORY:
        - 54-year-old female, former smoker (15 pack-years, quit 10 years ago)
        - Presenting complaint: 4 months of progressive exertional dyspnea
        - Associated: dry cough, fatigue, 10-lb weight loss
        - PMH: Sjogren's syndrome (diagnosed 2018), Raynaud's phenomenon
        - Medications: Pilocarpine 5mg TID, Hydroxychloroquine 200mg BID
        - Family history: Father died of pulmonary fibrosis at age 68
        - Occupational: Worked in textile factory for 15 years
        - 6-minute walk test: 280 meters (predicted >450), SpO2 dropped to 84%
        """,
        "labs": {
            "CBC": {"WBC": "5.8 K/uL (N)", "Hgb": "13.1 g/dL (N)", "Plt": "195 K/uL (N)"},
            "BMP": {"Na": "139 (N)", "K": "4.1 (N)", "Cr": "0.9 (N)", "BUN": "14 (N)"},
            "Inflammatory": {"ESR": "42 mm/hr (H)", "CRP": "1.8 mg/dL (H)"},
            "Autoimmune": {
                "ANA": "1:320 speckled (positive)",
                "Anti-SSA": "Positive",
                "Anti-SSB": "Positive",
                "RF": "45 IU/mL (H)",
                "Anti-CCP": "Negative",
                "Anti-Scl70": "Negative",
            },
            "Pulmonary": {
                "ABG_pH": "7.44 (N)",
                "ABG_pO2": "68 mmHg (L)",
                "ABG_pCO2": "34 mmHg (L-N)",
                "KL-6": "1850 U/mL (H, ref <500)",
            },
            "PFTs": {
                "FVC": "62% predicted (L)",
                "FEV1": "68% predicted (L)",
                "FEV1/FVC": "85% (H — restrictive)",
                "DLCO": "42% predicted (severely reduced)",
            },
        },
        "imaging": [
            {
                "modality": "HRCT Chest",
                "findings": (
                    "Bilateral subpleural reticular opacities predominantly in lower "
                    "lobes. Ground-glass opacities with traction bronchiectasis. "
                    "Honeycombing pattern in bilateral bases. No pleural effusions. "
                    "Mediastinal lymphadenopathy (1.5 cm subcarinal node). "
                    "Pattern is consistent with UIP (usual interstitial pneumonia) "
                    "vs NSIP (nonspecific interstitial pneumonia)."
                ),
            },
            {
                "modality": "Echocardiogram",
                "findings": (
                    "Normal LV function, EF 60%. RV mildly dilated. Estimated RVSP "
                    "48 mmHg (elevated, suggesting pulmonary hypertension). "
                    "Mild tricuspid regurgitation. No pericardial effusion."
                ),
            },
        ],
    },
    {
        "title": "Acute Liver Injury with Systemic Symptoms",
        "history": """
        PATIENT HISTORY:
        - 38-year-old male, previously healthy except seasonal allergies
        - Presenting: 10 days of fatigue, nausea, dark urine, RUQ pain
        - Noticed yellowing of eyes 5 days ago
        - Fever to 101°F intermittently for 1 week
        - Joint pain (hands, knees) preceding the jaundice by 2 weeks
        - No alcohol use. No recreational drugs.
        - Traveled to Southeast Asia (Thailand) 6 weeks ago — ate street food
        - Sexual history: new partner 3 months ago, inconsistent condom use
        - Started a new herbal supplement (kava kava) 3 weeks ago for anxiety
        - Takes cetirizine daily and acetaminophen PRN (~2g/day for joint pain)
        - No family history of liver disease
        - No blood transfusions, no tattoos
        """,
        "labs": {
            "LFTs": {
                "AST": "1,820 U/L (H, ref 10-40)",
                "ALT": "2,340 U/L (H, ref 7-56)",
                "Alk_Phos": "185 U/L (H, ref 44-147)",
                "Total_Bilirubin": "12.4 mg/dL (H, ref 0.1-1.2)",
                "Direct_Bilirubin": "8.8 mg/dL (H)",
                "Albumin": "3.2 g/dL (L, ref 3.5-5.0)",
                "GGT": "220 U/L (H)",
            },
            "Coagulation": {
                "INR": "1.8 (H, ref 0.8-1.1)",
                "PT": "22 seconds (H)",
            },
            "Viral_Hepatitis": {
                "Hep_A_IgM": "PENDING",
                "HBsAg": "PENDING",
                "HBc_IgM": "PENDING",
                "Hep_C_Ab": "PENDING",
                "Hep_E_IgM": "PENDING",
            },
            "Other": {
                "Acetaminophen_level": "<10 mcg/mL (N)",
                "Ceruloplasmin": "PENDING",
                "ANA": "PENDING",
                "Anti_smooth_muscle": "PENDING",
                "IgG_levels": "PENDING",
                "Ferritin": "980 ng/mL (H)",
                "Iron": "185 mcg/dL (H)",
            },
            "CBC": {
                "WBC": "11.2 K/uL (H)",
                "Hgb": "13.5 g/dL (N)",
                "Plt": "165 K/uL (N)",
                "Eosinophils": "8% (H)",
            },
        },
        "imaging": [
            {
                "modality": "RUQ Ultrasound",
                "findings": (
                    "Liver appears diffusely echogenic with mild hepatomegaly (17 cm span). "
                    "No focal lesions. No intrahepatic or extrahepatic biliary dilatation. "
                    "Gallbladder wall mildly thickened (5mm) but no stones. "
                    "Patent portal and hepatic veins. Mild perihepatic and perisplenic "
                    "free fluid. Mild splenomegaly (14 cm)."
                ),
            },
        ],
    },
    {
        "title": "New-Onset Seizure with Incidental Findings",
        "history": """
        PATIENT HISTORY:
        - 42-year-old male, brought by spouse after witnessed generalized
          tonic-clonic seizure lasting ~2 minutes with post-ictal confusion
        - No prior seizure history
        - For past 3-4 months: progressive headaches (worse in morning),
          personality changes per spouse (irritability, poor judgment),
          mild word-finding difficulty
        - Lost 15 lbs unintentionally over 2 months
        - Recent: nausea, occasional vomiting (morning)
        - PMH: Melanoma excised from back 2 years ago (Breslow 2.8 mm,
          Clark level IV, no sentinel node biopsy at patient's request)
        - Medications: None
        - No alcohol or drug use
        - Family history: mother — breast cancer, father — healthy
        """,
        "labs": {
            "CBC": {"WBC": "9.8 K/uL (N)", "Hgb": "14.2 g/dL (N)", "Plt": "285 K/uL (N)"},
            "BMP": {
                "Na": "132 mEq/L (L)",
                "K": "4.0 (N)",
                "Cr": "0.9 (N)",
                "Glucose": "95 (N)",
                "Calcium": "9.0 (N)",
            },
            "LFTs": {
                "AST": "28 (N)",
                "ALT": "32 (N)",
                "Alk_Phos": "225 U/L (H)",
                "LDH": "420 U/L (H)",
            },
            "Other": {
                "Prolactin": "42 ng/mL (H, ref <20) — post-seizure",
                "S100B": "0.4 mcg/L (H, ref <0.1)",
                "Lactate": "2.1 (N — post-seizure likely normalized)",
            },
        },
        "imaging": [
            {
                "modality": "CT Head without contrast (ED)",
                "findings": (
                    "Hyperdense lesion in the right frontal lobe, approximately 3.5 cm, "
                    "with surrounding vasogenic edema and 4 mm midline shift to the left. "
                    "A second smaller hyperdense lesion (1.2 cm) in the left parietal "
                    "region. No acute hemorrhage outside the lesions. Mild hydrocephalus."
                ),
            },
            {
                "modality": "MRI Brain with contrast",
                "findings": (
                    "Right frontal lobe: 3.5 x 3.2 cm enhancing mass with central "
                    "necrosis, ring enhancement pattern, significant surrounding edema. "
                    "Left parietal: 1.2 cm enhancing nodule with minimal surrounding edema. "
                    "Both lesions at gray-white junction. Leptomeningeal enhancement "
                    "not seen. No diffusion restriction to suggest abscess."
                ),
            },
            {
                "modality": "CT Chest/Abdomen/Pelvis",
                "findings": (
                    "3 small pulmonary nodules (all <1 cm). No mediastinal "
                    "lymphadenopathy. Liver: 2 hypodense lesions (1.5 cm and 0.8 cm) "
                    "in right lobe, too small to characterize. Adrenals normal. "
                    "No other suspicious findings."
                ),
            },
        ],
    },
]


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def format_case_for_analysis(case: dict) -> str:
    """Format all case data into a comprehensive prompt."""
    sections = [case["history"]]

    # Format labs
    lab_text = "\nLABORATORY RESULTS:\n"
    for panel, tests in case["labs"].items():
        lab_text += f"\n  {panel}:\n"
        for test, value in tests.items():
            lab_text += f"    {test}: {value}\n"
    sections.append(lab_text)

    # Format imaging
    imaging_text = "\nIMAGING STUDIES:\n"
    for img in case["imaging"]:
        imaging_text += f"\n  {img['modality']}:\n    {img['findings']}\n"
    sections.append(imaging_text)

    return "\n".join(sections)


def analyze_with_reasoning(case_text: str) -> dict:
    """Analyze case using reasoning model."""
    start = time.time()
    prompt = (
        "You are a senior physician and expert diagnostician. Analyze the "
        "following case with data from multiple sources (history, labs, imaging). "
        "Use systematic clinical reasoning. Consider the significance of each "
        "finding and how they relate to each other.\n\n"
        f"{case_text}\n\n"
        "Provide:\n"
        "1. Primary diagnosis with confidence level\n"
        "2. Supporting evidence from each data source\n"
        "3. Differential diagnoses\n"
        "4. Critical findings requiring immediate action\n"
        "5. Data gaps and recommended additional workup\n"
        "6. Urgency assessment\n"
        "7. Summary of your reasoning process"
    )
    response = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=5000,
    )
    elapsed = time.time() - start
    return {
        "content": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
        "elapsed": elapsed,
    }


def generate_structured_assessment(case_text: str) -> ClinicalAssessment:
    """Generate structured clinical assessment."""
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical decision support system. Analyze ALL data "
                    "sources and generate a comprehensive structured assessment. "
                    "Identify which findings came from which source (history, labs, imaging). "
                    "Be thorough and systematic."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate structured clinical assessment:\n\n{case_text}"
                ),
            },
        ],
        response_format=ClinicalAssessment,
    )
    return completion.choices[0].message.parsed


def display_assessment(assessment: ClinicalAssessment, title: str = ""):
    """Display structured clinical assessment."""
    if title:
        print(f"\n  {'═' * 55}")
        print(f"  CLINICAL ASSESSMENT — {title}")
        print(f"  {'═' * 55}")

    print(f"\n  Patient: {assessment.patient_summary}")

    # Data sources
    print(f"\n  DATA SOURCES ANALYZED ({len(assessment.data_sources_analyzed)}):")
    for ds in assessment.data_sources_analyzed:
        print(f"\n    [{ds.source_type.upper()}]")
        for finding in ds.key_findings[:3]:
            print(f"      • {finding}")
        if ds.abnormalities:
            print(f"      ⚠ Abnormalities: {', '.join(ds.abnormalities[:3])}")
        if ds.missing_data:
            print(f"      ? Missing: {', '.join(ds.missing_data[:2])}")

    # Primary diagnosis
    conf_bar = "█" * int(assessment.primary_diagnosis_confidence * 10)
    conf_bar += "░" * (10 - len(conf_bar))
    print(f"\n  PRIMARY DIAGNOSIS: {assessment.primary_diagnosis}")
    print(f"  Confidence: {conf_bar} {assessment.primary_diagnosis_confidence:.0%}")

    print(f"\n  Supporting Evidence:")
    for ev in assessment.supporting_evidence:
        print(f"    ✓ {ev}")

    # Critical findings
    if assessment.critical_findings:
        print(f"\n  🚨 CRITICAL FINDINGS:")
        for cf in assessment.critical_findings:
            print(f"    🔴 {cf}")

    # Differential
    print(f"\n  Differential Diagnoses:")
    for i, dx in enumerate(assessment.differential_diagnoses, 1):
        print(f"    {i}. {dx}")

    # Data gaps
    if assessment.data_gaps:
        print(f"\n  Data Gaps:")
        for gap in assessment.data_gaps:
            print(f"    ? {gap}")

    # Actions
    print(f"\n  Recommended Actions:")
    for action in assessment.recommended_actions:
        print(f"    → {action}")

    urgency_icons = {
        "emergent": "🔴", "urgent": "🟠",
        "semi-urgent": "🟡", "routine": "🟢",
    }
    icon = urgency_icons.get(assessment.urgency, "⚪")
    print(f"\n  Urgency: {icon} {assessment.urgency.upper()}")

    print(f"\n  Reasoning Summary:")
    print(f"  {assessment.reasoning_summary[:300]}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complex case analysis across scenarios."""
    print("=" * 60)
    print("EXERCISE 3: Complex Case Analyzer — Multi-Source Reasoning")
    print("=" * 60)

    for case in COMPLEX_CASES:
        print(f"\n\n{'#' * 60}")
        print(f"  CASE: {case['title']}")
        print(f"{'#' * 60}")

        # Format case data
        case_text = format_case_for_analysis(case)

        # Reasoning model analysis (free text)
        print(f"\n--- Reasoning Model Analysis ---")
        reasoning_result = analyze_with_reasoning(case_text)
        print(f"\n{reasoning_result['content'][:600]}...")
        print(f"\n  [Tokens: {reasoning_result['tokens']} | "
              f"Time: {reasoning_result['elapsed']:.2f}s]")

        # Structured assessment
        print(f"\n--- Structured Assessment ---")
        start = time.time()
        assessment = generate_structured_assessment(case_text)
        elapsed = time.time() - start

        display_assessment(assessment, title=case["title"])
        print(f"\n  [Structured output time: {elapsed:.2f}s]")

    # Summary
    print(f"\n\n{'=' * 60}")
    print("Exercise complete! Key takeaways:")
    print("  • Multi-source analysis catches patterns single sources miss")
    print("  • Reasoning models excel at synthesizing complex clinical data")
    print("  • Structured outputs make findings actionable for downstream systems")
    print("  • Data gap identification improves clinical workflow completeness")
    print("  • Always combine AI analysis with expert clinical judgment")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
