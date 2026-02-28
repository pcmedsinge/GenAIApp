"""
Exercise 1: Multimodal Triage Assistant
========================================
Accept image (wound/rash description) + text description, then classify
urgency and recommend action. Combines vision capabilities with clinical
reasoning for triage decisions.

Learning Objectives:
  - Combine image and text inputs for clinical assessment
  - Use structured outputs for triage classification
  - Implement urgency scoring with confidence levels
  - Handle real-world triage decision-making

Usage:
  python exercise_1_triage_assistant.py
"""

import base64
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

class VisualFinding(BaseModel):
    """A finding from image analysis."""
    description: str = Field(..., description="Description of finding")
    location: Optional[str] = Field(None, description="Body location")
    severity_indicator: Literal["benign", "concerning", "alarming"] = Field(
        ..., description="Visual severity indicator"
    )


class TriageAssessment(BaseModel):
    """Complete triage assessment from multimodal input."""
    urgency: Literal["emergent", "urgent", "semi-urgent", "non-urgent"] = Field(
        ..., description="Urgency classification (ESI-like scale)"
    )
    urgency_score: int = Field(
        ..., ge=1, le=5, description="1=emergent, 5=non-urgent"
    )
    visual_findings: List[VisualFinding] = Field(
        default_factory=list, description="Findings from image analysis"
    )
    history_findings: List[str] = Field(
        default_factory=list, description="Key findings from patient history"
    )
    red_flags: List[str] = Field(
        default_factory=list, description="Red flag symptoms/signs identified"
    )
    primary_impression: str = Field(
        ..., description="Most likely clinical impression"
    )
    differential: List[str] = Field(
        default_factory=list, description="Differential diagnoses to consider"
    )
    recommended_action: str = Field(
        ..., description="Recommended immediate action"
    )
    recommended_workup: List[str] = Field(
        default_factory=list, description="Recommended tests/evaluations"
    )
    disposition: Literal[
        "ED_immediate", "ED_urgent", "urgent_care", "PCP_24h",
        "PCP_routine", "self_care"
    ] = Field(..., description="Recommended care setting")
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence in triage decision"
    )
    limitations: List[str] = Field(
        default_factory=list, description="Limitations of this assessment"
    )


# ============================================================================
# TRIAGE SCENARIOS
# ============================================================================

TRIAGE_SCENARIOS = [
    {
        "title": "Acute Wound Assessment",
        "image_description": """
        [SIMULATED IMAGE — Right forearm wound]
        A 3 cm laceration on the right ventral forearm, appears approximately
        1 cm deep. Wound edges are irregular and gaping. Active bleeding
        that is dark red (venous). Surrounding skin shows mild erythema
        extending 0.5 cm from wound edges. No visible foreign body.
        No exposed tendon or bone. Wound appears clean without gross
        contamination. Adjacent skin is normal with good capillary refill
        in the hand and fingers.
        """,
        "patient_description": """
        Patient: 28-year-old male
        Mechanism: Cut forearm on broken glass while washing dishes, 45 minutes ago
        Symptoms: Pain at wound site, continuous bleeding
        Medical History: None significant
        Medications: None
        Allergies: NKDA
        Tetanus: Up to date (booster 3 years ago)
        Self-treatment: Direct pressure with clean towel, bleeding slowed but persists
        Additional: Full range of motion in wrist/fingers, sensation intact distally
        """,
    },
    {
        "title": "Skin Rash Evaluation",
        "image_description": """
        [SIMULATED IMAGE — Trunk and bilateral arms]
        Diffuse, erythematous, maculopapular rash covering the trunk and
        proximal bilateral upper extremities. Lesions are 3-8 mm in diameter,
        blanching with pressure. Some lesions show central clearing
        (target-like appearance). A few lesions on palms noted.
        Bilateral lip swelling visible. No vesicles or bullae.
        No mucosal erosions visible in the images. Skin between lesions
        appears normal.
        """,
        "patient_description": """
        Patient: 35-year-old female
        Onset: Rash appeared 6 hours ago, rapidly spreading
        Symptoms: Itching, mild burning sensation, lip swelling, mild throat tightness
        Recent history: Started allopurinol 5 days ago for gout
        Medical History: Gout, hypertension
        Medications: Allopurinol 100 mg daily (started 5 days ago), Losartan 50 mg daily
        Allergies: Sulfa drugs (rash)
        Vitals (home): Felt warm but no thermometer
        Additional: No difficulty breathing, no blisters, can eat and drink
        """,
    },
    {
        "title": "Diabetic Foot Ulcer",
        "image_description": """
        [SIMULATED IMAGE — Left foot, plantar surface]
        Chronic-appearing ulcer on the plantar surface of the left foot
        at the first metatarsal head. Approximately 2 x 2.5 cm. Wound bed
        shows 60% yellow-gray slough tissue and 40% pale pink granulation.
        Wound margins are calloused and macerated. Surrounding skin shows
        erythema extending 2.5 cm from wound edge with warmth. Mild
        malodorous drainage (greenish). Toes appear dusky with thickened,
        dystrophic nails. Pitting edema to mid-calf. No visible bone
        or tendon in wound bed.
        """,
        "patient_description": """
        Patient: 68-year-old male
        Duration: Ulcer present for 6 weeks, worsening over past week
        Symptoms: Increased drainage, new foul odor, noticed red streaking
          up the ankle yesterday, fever 101.4°F this morning, chills
        Medical History:
          - Type 2 diabetes x 20 years (HbA1c 10.2% last month)
          - Peripheral neuropathy (cannot feel monofilament)
          - Peripheral arterial disease (ABI 0.6 bilaterally)
          - CKD stage 3 (eGFR 42)
          - Previous right foot amputation (3rd toe, 2023)
        Medications: Insulin, statin, aspirin, gabapentin, lisinopril
        Last seen by podiatry: 2 months ago
        """,
    },
    {
        "title": "Pediatric Rash",
        "image_description": """
        [SIMULATED IMAGE — Toddler arm and torso]
        Scattered, round, well-circumscribed red papules and small vesicles
        on an erythematous base, 2-5 mm in size. Lesions are in various
        stages: some fresh vesicles, some crusted. Distribution: trunk,
        face, and proximal extremities. A few lesions on scalp visible
        through thin hair. No purpura or petechiae. Surrounding skin
        appears normal. Child appears alert and active in the photo.
        """,
        "patient_description": """
        Patient: 3-year-old female
        Onset: Rash started 2 days ago with low-grade fever (100.8°F)
        Symptoms: Itching, mild fussiness, eating and drinking okay
        History: Rash started on trunk, now spreading to face and arms.
          New vesicles appearing. Some older ones have crusted over.
        Medical History: Healthy, immunizations up to date EXCEPT varicella
          vaccine (parent declined)
        Exposure: Classmate at daycare diagnosed with chickenpox 2 weeks ago
        Medications: Children's Tylenol PRN
        No difficulty breathing, no high fever (max 101°F), no lethargy
        """,
    },
    {
        "title": "Chest Wall Emergency",
        "image_description": """
        [SIMULATED IMAGE — Left chest/axilla]
        Large area of erythema and swelling over left lateral chest wall
        and axilla. Skin is tense, warm, and erythematous with indistinct
        borders. An area of fluctuance approximately 6 cm in diameter.
        Skin has a dusky/violaceous hue centrally suggesting tissue
        compromise. Surrounding skin shows spreading erythema with
        irregular margins. Previously marked border (pen marks visible)
        shows erythema has expanded approximately 3 cm beyond markings.
        Subcutaneous crepitus noted in description.
        """,
        "patient_description": """
        Patient: 52-year-old male
        Onset: Swelling started 3 days ago after a small cut in left axilla
        Symptoms: Severe pain (9/10), fever 103.8°F, rigors, feeling
          "extremely unwell", spreading redness that keeps expanding
        Medical History: Type 2 diabetes (poorly controlled), obesity,
          IV drug use history (clean 1 year)
        Medications: Metformin (not taking regularly)
        Vitals (EMS): BP 88/54, HR 128, RR 26, Temp 104.1°F, SpO2 94%
        Additional: Confused, unable to raise left arm, crackling feel
          under skin around the swelling
        """,
    },
]


# ============================================================================
# TRIAGE FUNCTIONS
# ============================================================================

def perform_triage(image_desc: str, patient_desc: str) -> TriageAssessment:
    """
    Perform multimodal triage assessment.

    In production, the image would be sent as base64-encoded data.
    Here we use a text description of the image findings.
    """
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an emergency triage system that combines visual findings "
                    "with patient history to make triage decisions. Follow ESI "
                    "(Emergency Severity Index) principles:\n"
                    "  1 = Emergent: life-threatening, immediate intervention needed\n"
                    "  2 = Urgent: high risk, confused/lethargic, severe pain\n"
                    "  3 = Semi-urgent: stable but needs 2+ resources\n"
                    "  4 = Non-urgent: needs 1 resource\n"
                    "  5 = Non-urgent: no resources needed\n\n"
                    "Rules:\n"
                    "- Always identify red flags\n"
                    "- Consider worst-case scenario in differential\n"
                    "- Include confidence level and limitations\n"
                    "- Never replace clinical judgment"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"IMAGE FINDINGS:\n{image_desc}\n\n"
                    f"PATIENT INFORMATION:\n{patient_desc}\n\n"
                    "Perform triage assessment combining visual and clinical findings."
                ),
            },
        ],
        response_format=TriageAssessment,
    )
    return completion.choices[0].message.parsed


def display_triage(assessment: TriageAssessment, title: str = ""):
    """Display triage assessment results."""

    urgency_colors = {
        "emergent": "🔴 EMERGENT",
        "urgent": "🟠 URGENT",
        "semi-urgent": "🟡 SEMI-URGENT",
        "non-urgent": "🟢 NON-URGENT",
    }

    disposition_labels = {
        "ED_immediate": "Emergency Department — Immediate",
        "ED_urgent": "Emergency Department — Urgent",
        "urgent_care": "Urgent Care Center",
        "PCP_24h": "Primary Care within 24 hours",
        "PCP_routine": "Primary Care — Routine",
        "self_care": "Self-Care with Instructions",
    }

    if title:
        print(f"\n  {'═' * 55}")
        print(f"  TRIAGE ASSESSMENT — {title}")
        print(f"  {'═' * 55}")

    urgency_label = urgency_colors.get(assessment.urgency, assessment.urgency)
    print(f"\n  ┌────────────────────────────────────────────────────────┐")
    print(f"  │  URGENCY: {urgency_label:<44} │")
    print(f"  │  ESI Score: {assessment.urgency_score}/5{'':<41} │")
    print(f"  │  Confidence: {assessment.confidence:.0%}{'':<41} │")
    print(f"  └────────────────────────────────────────────────────────┘")

    if assessment.red_flags:
        print(f"\n  🚨 RED FLAGS:")
        for flag in assessment.red_flags:
            print(f"    ⚠ {flag}")

    print(f"\n  Primary Impression: {assessment.primary_impression}")

    if assessment.visual_findings:
        print(f"\n  Visual Findings:")
        for vf in assessment.visual_findings:
            icon = {"benign": "🟢", "concerning": "🟡", "alarming": "🔴"}.get(
                vf.severity_indicator, "⚪"
            )
            print(f"    {icon} {vf.description}")

    if assessment.history_findings:
        print(f"\n  Key History Findings:")
        for hf in assessment.history_findings:
            print(f"    • {hf}")

    if assessment.differential:
        print(f"\n  Differential Diagnoses:")
        for i, dx in enumerate(assessment.differential, 1):
            print(f"    {i}. {dx}")

    disp_label = disposition_labels.get(assessment.disposition, assessment.disposition)
    print(f"\n  📍 Disposition: {disp_label}")
    print(f"  💡 Action: {assessment.recommended_action}")

    if assessment.recommended_workup:
        print(f"\n  Recommended Workup:")
        for test in assessment.recommended_workup:
            print(f"    → {test}")

    if assessment.limitations:
        print(f"\n  ⚠ Assessment Limitations:")
        for lim in assessment.limitations:
            print(f"    - {lim}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run multimodal triage across scenarios."""
    print("=" * 60)
    print("EXERCISE 1: Multimodal Triage Assistant")
    print("=" * 60)
    print("\nThis exercise simulates multimodal triage combining image")
    print("analysis with patient history for urgency classification.\n")

    results = []
    for scenario in TRIAGE_SCENARIOS:
        print(f"\n{'#' * 60}")
        print(f"  CASE: {scenario['title']}")
        print(f"{'#' * 60}")

        # Perform triage
        start = time.time()
        assessment = perform_triage(
            scenario["image_description"],
            scenario["patient_description"],
        )
        elapsed = time.time() - start

        # Display
        display_triage(assessment, title=scenario["title"])
        print(f"\n  [Processing time: {elapsed:.2f}s]")

        results.append({
            "title": scenario["title"],
            "urgency": assessment.urgency,
            "score": assessment.urgency_score,
            "confidence": assessment.confidence,
            "disposition": assessment.disposition,
            "elapsed": elapsed,
        })

    # Summary table
    print(f"\n\n{'=' * 60}")
    print("TRIAGE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Case':<30} {'Urgency':<15} {'ESI':<5} {'Conf':<8} {'Disposition':<15}")
    print(f"  {'─'*30} {'─'*15} {'─'*5} {'─'*8} {'─'*15}")
    for r in results:
        print(f"  {r['title']:<30} {r['urgency']:<15} {r['score']:<5} "
              f"{r['confidence']:.0%}{'':<4} {r['disposition']:<15}")

    print(f"\nKey takeaways:")
    print("  • Multimodal input (image + text) enables better triage decisions")
    print("  • Confidence scores help identify cases needing human review")
    print("  • Red flag identification is critical for patient safety")
    print("  • AI triage is a decision support tool — not a replacement for clinicians")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
