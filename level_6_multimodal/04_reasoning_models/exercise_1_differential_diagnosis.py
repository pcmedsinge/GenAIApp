"""
Exercise 1: Differential Diagnosis with Reasoning Models
=========================================================
Use a reasoning model (o1-mini) for complex differential diagnosis.
Present symptoms, labs, and history, then get a ranked differential.
Compare quality of reasoning between GPT-4o and o1-mini.

Learning Objectives:
  - Use reasoning models for complex clinical analysis
  - Understand API differences (no system message, max_completion_tokens)
  - Compare reasoning quality between standard and reasoning models
  - Build structured differential diagnosis outputs

Usage:
  python exercise_1_differential_diagnosis.py
"""

import json
import os
import time
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI()
STANDARD_MODEL = "gpt-4o"
REASONING_MODEL = "o1-mini"


# ============================================================================
# SCHEMAS
# ============================================================================

class DifferentialEntry(BaseModel):
    """A single entry in the differential diagnosis."""
    diagnosis: str = Field(..., description="Diagnosis name")
    likelihood: str = Field(..., description="Estimated likelihood: high, moderate, low")
    confidence_percent: Optional[int] = Field(None, description="Confidence 0-100%")
    supporting_evidence: List[str] = Field(
        default_factory=list, description="Evidence supporting this diagnosis"
    )
    against_evidence: List[str] = Field(
        default_factory=list, description="Evidence against this diagnosis"
    )
    key_differentiating_test: Optional[str] = Field(
        None, description="Test that would confirm/exclude this diagnosis"
    )


class DifferentialDiagnosis(BaseModel):
    """Complete differential diagnosis response."""
    patient_summary: str = Field(..., description="Brief patient summary")
    primary_diagnosis: DifferentialEntry = Field(
        ..., description="Most likely diagnosis"
    )
    differential: List[DifferentialEntry] = Field(
        ..., description="Ranked list of differential diagnoses"
    )
    critical_actions: List[str] = Field(
        default_factory=list, description="Immediate actions needed"
    )
    recommended_workup: List[str] = Field(
        default_factory=list, description="Recommended additional tests"
    )


# ============================================================================
# CLINICAL CASES
# ============================================================================

CLINICAL_CASES = [
    {
        "title": "Young Woman with Fatigue and Joint Pain",
        "presentation": """
        CLINICAL PRESENTATION:

        Patient: 32-year-old female, previously healthy
        Chief Complaint: 6 weeks of progressive fatigue, joint pain, and rash

        History of Present Illness:
        - Progressive fatigue limiting daily activities for 6 weeks
        - Symmetric joint pain and stiffness in hands, wrists, and knees
          (worse in mornings, lasting >1 hour)
        - Facial rash that worsens with sun exposure
        - Intermittent low-grade fevers (100-101°F)
        - Recent hair loss (diffuse thinning)
        - Oral sores (painless) noted in past 2 weeks
        - Raynaud's phenomenon in cold weather (new over past 3 months)
        - Unintentional 8-lb weight loss

        Past Medical History: Unremarkable
        Family History: Mother with hypothyroidism, sister with celiac disease
        Social History: Non-smoker, occasional alcohol, works as a teacher

        Physical Exam:
        - Vitals: T 100.4°F, BP 128/82, HR 92, RR 18
        - Erythematous macular rash across cheeks and nose bridge
        - Diffuse hair thinning
        - 2 painless oral ulcers on hard palate
        - Tender, swollen MCP and PIP joints bilaterally
        - Bilateral knee effusions
        - No organomegaly
        - Lower extremity edema, trace bilateral

        Laboratory Results:
        - CBC: WBC 3.4 K/uL (L), Hgb 10.8 g/dL (L), Plt 132 K/uL (L-normal)
        - ESR: 58 mm/hr (H), CRP: 2.4 mg/dL (H)
        - BMP: Creatinine 1.2 mg/dL (borderline), BUN 18
        - Urinalysis: Protein 2+, RBC 12/hpf, RBC casts seen
        - ANA: Positive, 1:640, homogeneous pattern
        - Anti-dsDNA: 85 IU/mL (positive)
        - C3: 52 mg/dL (low, ref 90-180), C4: 6 mg/dL (low, ref 10-40)
        - Anti-Smith: Positive
        - Anti-SSA/Ro: Positive
        """,
    },
    {
        "title": "Middle-Aged Man with Shortness of Breath",
        "presentation": """
        CLINICAL PRESENTATION:

        Patient: 58-year-old male
        Chief Complaint: Progressive shortness of breath over 3 months

        History of Present Illness:
        - Gradually worsening dyspnea on exertion (now with walking 1 block)
        - Dry, nonproductive cough for 2 months
        - 12-lb weight loss over 3 months (unintentional)
        - Night sweats (drenching, 2-3x per week for 1 month)
        - Mild chest discomfort, dull, non-pleuritic
        - No hemoptysis, no leg swelling
        - Former smoker: 40 pack-years, quit 5 years ago

        Past Medical History:
        - COPD (moderate, FEV1 65% predicted)
        - Hypertension
        - Former asbestos exposure (shipyard worker, 1990-2010)

        Physical Exam:
        - Vitals: T 99.8°F, BP 134/78, HR 88, RR 22, SpO2 91% on RA
        - Decreased breath sounds right lower lobe
        - Dullness to percussion right base
        - No palpable lymphadenopathy (cervical, supraclavicular)
        - Mild digital clubbing noted
        - No hepatosplenomegaly

        Laboratory/Imaging:
        - CBC: WBC 11.2 (H), Hgb 12.8, Plt 380 (H)
        - LDH: 320 U/L (H)
        - Chest X-ray: Large right pleural effusion, right hilar prominence
        - Pleural fluid: Exudative (protein 4.8 g/dL, LDH 260 U/L)
        - Pleural fluid cytology: Pending
        - Calcium: 11.2 mg/dL (H)
        """,
    },
    {
        "title": "Elderly Patient with Acute Confusion",
        "presentation": """
        CLINICAL PRESENTATION:

        Patient: 78-year-old female, brought by family
        Chief Complaint: Acute confusion and behavioral changes over 48 hours

        History of Present Illness:
        - Normally independent, alert, and oriented
        - Over past 48 hours: increasingly confused, disoriented
        - Fluctuating level of consciousness (lucid periods alternating with confusion)
        - Visual hallucinations (seeing people who aren't there)
        - Decreased oral intake for 3 days
        - New urinary incontinence (not typical)
        - Low-grade fever yesterday per family

        Past Medical History:
        - Type 2 diabetes (on metformin, glipizide)
        - Hypertension
        - Mild cognitive impairment (baseline MMSE 24/30)
        - Osteoarthritis
        - UTI treated 3 weeks ago with ciprofloxacin
        - s/p right hip replacement 2022

        Medications: Metformin 500 BID, Glipizide 5 daily, Lisinopril 10 daily,
        Acetaminophen PRN, Zolpidem 5 mg QHS (for insomnia)

        Physical Exam:
        - Vitals: T 100.8°F, BP 98/62, HR 102, RR 20, SpO2 94% on RA
        - Inattentive, oriented to self only, CAM positive
        - Suprapubic tenderness on palpation
        - Dry mucous membranes
        - No focal neurological deficits
        - No meningeal signs

        Labs:
        - CBC: WBC 15.4 (H), bands 12% (H), Hgb 11.2, Plt 210
        - BMP: Na 148 (H), K 3.2 (L), BUN 42 (H), Cr 1.8 (H, baseline 1.0),
          Glucose 62 (L)
        - Urinalysis: Positive nitrites, positive LE, WBC >50/hpf, bacteria 3+
        - Lactate: 2.8 mmol/L (H)
        - Blood cultures: Pending
        """,
    },
]


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def get_standard_differential(case_text: str) -> dict:
    """Get differential diagnosis from standard model (GPT-4o)."""
    start = time.time()

    # Standard model supports system messages
    response = client.chat.completions.create(
        model=STANDARD_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert diagnostician. Provide a ranked differential "
                    "diagnosis with supporting/refuting evidence for each. Include "
                    "confidence percentages. Be thorough and systematic."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{case_text}\n\n"
                    "Provide a ranked differential diagnosis with:\n"
                    "1. Most likely diagnosis with evidence\n"
                    "2. Alternative diagnoses ranked by likelihood\n"
                    "3. Key differentiating tests\n"
                    "4. Critical immediate actions"
                ),
            },
        ],
        max_tokens=2000,
        temperature=0.2,
    )
    elapsed = time.time() - start

    return {
        "model": STANDARD_MODEL,
        "content": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "elapsed": elapsed,
    }


def get_reasoning_differential(case_text: str) -> dict:
    """Get differential diagnosis from reasoning model (o1-mini)."""
    start = time.time()

    # Reasoning model: NO system message, use max_completion_tokens
    prompt = (
        "You are an expert diagnostician. Provide a ranked differential "
        "diagnosis with supporting/refuting evidence for each. Include "
        "confidence percentages. Be thorough and systematic.\n\n"
        f"{case_text}\n\n"
        "Provide a ranked differential diagnosis with:\n"
        "1. Most likely diagnosis with evidence\n"
        "2. Alternative diagnoses ranked by likelihood\n"
        "3. Key differentiating tests\n"
        "4. Critical immediate actions"
    )

    response = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=4000,
    )
    elapsed = time.time() - start

    result = {
        "model": REASONING_MODEL,
        "content": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "elapsed": elapsed,
    }

    if hasattr(response.usage, "completion_tokens_details"):
        details = response.usage.completion_tokens_details
        if hasattr(details, "reasoning_tokens") and details.reasoning_tokens:
            result["reasoning_tokens"] = details.reasoning_tokens

    return result


def compare_models(standard: dict, reasoning: dict):
    """Compare outputs from standard and reasoning models."""
    print(f"\n  {'─' * 60}")
    print(f"  MODEL COMPARISON")
    print(f"  {'─' * 60}")
    print(f"  {'Metric':<25} {'GPT-4o':<20} {'o1-mini':<20}")
    print(f"  {'─'*25} {'─'*20} {'─'*20}")
    print(f"  {'Latency':<25} {standard['elapsed']:.2f}s{'':<15} "
          f"{reasoning['elapsed']:.2f}s")
    print(f"  {'Total tokens':<25} {standard['tokens']:<20} {reasoning['tokens']:<20}")
    print(f"  {'Prompt tokens':<25} {standard['prompt_tokens']:<20} "
          f"{reasoning['prompt_tokens']:<20}")
    print(f"  {'Completion tokens':<25} {standard['completion_tokens']:<20} "
          f"{reasoning['completion_tokens']:<20}")
    if "reasoning_tokens" in reasoning:
        print(f"  {'Reasoning tokens':<25} {'N/A':<20} {reasoning['reasoning_tokens']:<20}")
    print(f"  {'Response length':<25} {len(standard['content'])} chars{'':<6} "
          f"{len(reasoning['content'])} chars")

    speedup = reasoning["elapsed"] / standard["elapsed"] if standard["elapsed"] > 0 else 0
    print(f"\n  Reasoning model was {speedup:.1f}x {'slower' if speedup > 1 else 'faster'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run differential diagnosis comparison across clinical cases."""
    print("=" * 60)
    print("EXERCISE 1: Differential Diagnosis with Reasoning Models")
    print("=" * 60)

    for case in CLINICAL_CASES:
        print(f"\n\n{'#' * 60}")
        print(f"  CASE: {case['title']}")
        print(f"{'#' * 60}")

        # Show brief case summary
        lines = case["presentation"].strip().split("\n")
        preview = "\n".join(lines[:8])
        print(f"\n{preview}\n  ...")

        # Standard model
        print(f"\n--- Standard Model ({STANDARD_MODEL}) ---")
        standard = get_standard_differential(case["presentation"])
        # Show first 500 chars
        print(f"\n{standard['content'][:500]}...")
        print(f"\n  [Tokens: {standard['tokens']} | Time: {standard['elapsed']:.2f}s]")

        # Reasoning model
        print(f"\n--- Reasoning Model ({REASONING_MODEL}) ---")
        reasoning = get_reasoning_differential(case["presentation"])
        print(f"\n{reasoning['content'][:500]}...")
        print(f"\n  [Tokens: {reasoning['tokens']} | Time: {reasoning['elapsed']:.2f}s]")

        # Compare
        compare_models(standard, reasoning)

    # Summary
    print(f"\n\n{'=' * 60}")
    print("Exercise complete! Key takeaways:")
    print("  • Reasoning models excel at systematic differential diagnosis")
    print("  • o1-mini considers evidence more thoroughly (for/against each dx)")
    print("  • Standard models are faster but may miss subtle connections")
    print("  • No system message for o1 — combine instructions in user message")
    print("  • Use max_completion_tokens (not max_tokens) for reasoning models")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
