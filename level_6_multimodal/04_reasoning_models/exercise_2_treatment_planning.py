"""
Exercise 2: Complex Treatment Planning with Reasoning Models
=============================================================
Use reasoning models for complex treatment planning with multiple
comorbidities. The reasoning model considers drug interactions,
contraindications, and patient-specific factors.

Learning Objectives:
  - Apply reasoning models to complex drug interaction analysis
  - Handle multiple comorbidities in treatment planning
  - Compare treatment plan quality between standard and reasoning models
  - Understand when reasoning models provide superior clinical analysis

Usage:
  python exercise_2_treatment_planning.py
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
STANDARD_MODEL = "gpt-4o"
REASONING_MODEL = "o1-mini"


# ============================================================================
# SCHEMAS
# ============================================================================

class DrugInteraction(BaseModel):
    """A potential drug interaction to consider."""
    drug_a: str
    drug_b: str
    severity: Literal["major", "moderate", "minor"]
    description: str
    recommendation: str


class TreatmentRecommendation(BaseModel):
    """A single treatment recommendation."""
    medication: str = Field(..., description="Recommended medication")
    dose: str = Field(..., description="Recommended dose")
    rationale: str = Field(..., description="Why this medication was chosen")
    monitoring: List[str] = Field(
        default_factory=list, description="Monitoring parameters"
    )
    cautions: List[str] = Field(
        default_factory=list, description="Specific cautions for this patient"
    )


class ContraindicatedOption(BaseModel):
    """A treatment option that is contraindicated."""
    medication: str
    reason: str
    which_comorbidity: str = Field(..., description="Which condition makes it contraindicated")


class TreatmentPlan(BaseModel):
    """Complete treatment plan response."""
    patient_summary: str
    treatment_goals: List[str]
    recommended_medications: List[TreatmentRecommendation]
    contraindicated_options: List[ContraindicatedOption]
    drug_interactions: List[DrugInteraction]
    non_pharmacologic: List[str] = Field(
        default_factory=list, description="Non-drug interventions"
    )
    overall_risk_assessment: str
    follow_up_plan: str


# ============================================================================
# COMPLEX CLINICAL SCENARIOS
# ============================================================================

SCENARIOS = [
    {
        "title": "Pain Management in Multi-Organ Disease",
        "description": """
        PATIENT PROFILE:
        58-year-old male with:
        1. Severe osteoarthritis (bilateral knees, lumbar spine) — VAS pain 8/10
        2. Chronic kidney disease stage 4 (eGFR 22 mL/min)
        3. Decompensated cirrhosis (Child-Pugh C, MELD 22)
        4. History of GI bleeding (variceal, 6 months ago)
        5. Heart failure with reduced EF (EF 30%)
        6. History of opioid use disorder (in recovery 3 years, on buprenorphine)

        Current Medications:
        - Buprenorphine/naloxone 8/2 mg SL daily
        - Furosemide 80 mg daily
        - Spironolactone 50 mg daily
        - Carvedilol 12.5 mg BID
        - Lactulose 30 mL TID
        - Rifaximin 550 mg BID
        - Pantoprazole 40 mg daily

        QUESTION: Design an optimal pain management strategy for this patient's
        severe osteoarthritis pain, considering ALL comorbidities and current
        medications. Address:
        1. Why common pain medications are contraindicated
        2. What can safely be used and at what doses
        3. Drug interactions with current medications
        4. Required monitoring
        5. Non-pharmacologic approaches
        """,
    },
    {
        "title": "Anticoagulation in Complex Medical History",
        "description": """
        PATIENT PROFILE:
        72-year-old female with:
        1. New diagnosis of atrial fibrillation (CHA2DS2-VASc = 5)
        2. History of hemorrhagic stroke (2 years ago, left basal ganglia)
        3. Mechanical mitral valve replacement (On-X valve, 2019)
        4. Chronic kidney disease stage 3b (eGFR 35)
        5. History of HIT (heparin-induced thrombocytopenia) — 2020
        6. Active peptic ulcer disease (recently healed, 3 months ago)
        7. Thrombocytopenia (platelets 85,000)

        Current Medications:
        - Warfarin 5 mg daily (INR target 2.5-3.5 for mechanical valve)
        - Metoprolol 50 mg BID
        - Pantoprazole 40 mg daily
        - Amlodipine 5 mg daily
        - Atorvastatin 40 mg daily

        QUESTION: Optimize the anticoagulation strategy for this patient
        considering the competing risks of:
        - Stroke from atrial fibrillation
        - Valve thrombosis from mechanical valve
        - Recurrent hemorrhagic stroke
        - GI bleeding risk
        - HIT history
        - Renal impairment
        - Thrombocytopenia

        Explain the reasoning for each decision.
        """,
    },
    {
        "title": "Immunosuppression Management with Infections",
        "description": """
        PATIENT PROFILE:
        45-year-old female with:
        1. Severe systemic lupus erythematosus with class IV nephritis
           (active flare with rising creatinine, active urine sediment)
        2. Recent positive QuantiFERON (latent TB) — not yet treated
        3. Chronic hepatitis B (HBsAg+, HBeAg-, viral load 2,100 IU/mL)
        4. History of progressive multifocal leukoencephalopathy (PML)
           scare with prior rituximab use
        5. Diabetes (HbA1c 8.2%)
        6. Osteoporosis (T-score -3.2 at lumbar spine)

        Current Medications:
        - Prednisone 10 mg daily (flare, was 5 mg)
        - Mycophenolate 1000 mg BID
        - Hydroxychloroquine 200 mg BID
        - Entecavir 0.5 mg daily (Hep B prophylaxis)
        - Insulin glargine 20 units QHS
        - Calcium/Vitamin D

        QUESTION: The patient's lupus nephritis is flaring (rising creatinine
        from 1.0 to 1.8, active urine sediment with RBC casts). Design a
        treatment intensification plan that addresses:
        1. Lupus nephritis flare management
        2. TB prophylaxis timing and interaction with immunosuppression
        3. Hepatitis B reactivation risk with increased immunosuppression
        4. Avoiding PML risk (no rituximab)
        5. Bone protection with increased corticosteroids
        6. Diabetes management with steroid intensification
        """,
    },
]


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def get_treatment_plan_standard(scenario: str) -> dict:
    """Get treatment plan from standard model."""
    start = time.time()
    response = client.chat.completions.create(
        model=STANDARD_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical pharmacology and internal medicine expert. "
                    "Provide a comprehensive, evidence-based treatment plan. Consider "
                    "ALL drug interactions, contraindications, and patient-specific factors. "
                    "Be systematic and thorough."
                ),
            },
            {"role": "user", "content": scenario},
        ],
        max_tokens=2500,
        temperature=0.1,
    )
    elapsed = time.time() - start
    return {
        "model": STANDARD_MODEL,
        "content": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
        "elapsed": elapsed,
    }


def get_treatment_plan_reasoning(scenario: str) -> dict:
    """Get treatment plan from reasoning model."""
    start = time.time()
    prompt = (
        "You are a clinical pharmacology and internal medicine expert. "
        "Provide a comprehensive, evidence-based treatment plan. Consider "
        "ALL drug interactions, contraindications, and patient-specific factors. "
        "Be systematic and thorough.\n\n"
        f"{scenario}"
    )
    response = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=5000,
    )
    elapsed = time.time() - start
    result = {
        "model": REASONING_MODEL,
        "content": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
        "elapsed": elapsed,
    }
    if hasattr(response.usage, "completion_tokens_details"):
        details = response.usage.completion_tokens_details
        if hasattr(details, "reasoning_tokens") and details.reasoning_tokens:
            result["reasoning_tokens"] = details.reasoning_tokens
    return result


def evaluate_plan_quality(plan_text: str) -> dict:
    """Use GPT-4o to evaluate the quality of a treatment plan."""
    response = client.chat.completions.create(
        model=STANDARD_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical quality reviewer. Evaluate the treatment plan "
                    "on these criteria (score 1-5 each):\n"
                    "1. Completeness — addresses all comorbidities\n"
                    "2. Safety — identifies contraindications and interactions\n"
                    "3. Evidence basis — cites guidelines or evidence\n"
                    "4. Monitoring plan — includes appropriate follow-up\n"
                    "5. Practicality — realistic and implementable\n\n"
                    "Respond in JSON: {\"completeness\": N, \"safety\": N, "
                    "\"evidence\": N, \"monitoring\": N, \"practicality\": N, "
                    "\"total\": N, \"comments\": \"...\"}"
                ),
            },
            {
                "role": "user",
                "content": f"Evaluate this treatment plan:\n\n{plan_text[:3000]}",
            },
        ],
        max_tokens=500,
        temperature=0.1,
    )
    try:
        content = response.choices[0].message.content
        # Extract JSON from response
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            return json.loads(content[start_idx:end_idx])
    except (json.JSONDecodeError, ValueError):
        pass
    return {"total": 0, "comments": "Could not parse evaluation"}


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complex treatment planning scenarios."""
    print("=" * 60)
    print("EXERCISE 2: Complex Treatment Planning with Reasoning Models")
    print("=" * 60)

    all_scores = {"standard": [], "reasoning": []}

    for scenario in SCENARIOS:
        print(f"\n\n{'#' * 60}")
        print(f"  SCENARIO: {scenario['title']}")
        print(f"{'#' * 60}")

        # Brief preview
        lines = scenario["description"].strip().split("\n")
        print("\n".join(lines[:10]))
        print("  ...")

        # Standard model
        print(f"\n--- Standard Model ({STANDARD_MODEL}) ---")
        standard = get_treatment_plan_standard(scenario["description"])
        print(f"\n{standard['content'][:600]}...")
        print(f"\n  [Tokens: {standard['tokens']} | Time: {standard['elapsed']:.2f}s]")

        # Reasoning model
        print(f"\n--- Reasoning Model ({REASONING_MODEL}) ---")
        reasoning = get_treatment_plan_reasoning(scenario["description"])
        print(f"\n{reasoning['content'][:600]}...")
        print(f"\n  [Tokens: {reasoning['tokens']} | Time: {reasoning['elapsed']:.2f}s]")

        # Evaluate both
        print(f"\n--- Quality Evaluation ---")
        std_eval = evaluate_plan_quality(standard["content"])
        rsn_eval = evaluate_plan_quality(reasoning["content"])

        print(f"  {'Criterion':<20} {'GPT-4o':<12} {'o1-mini':<12}")
        print(f"  {'─'*20} {'─'*12} {'─'*12}")

        for key in ["completeness", "safety", "evidence", "monitoring", "practicality"]:
            s_score = std_eval.get(key, "?")
            r_score = rsn_eval.get(key, "?")
            print(f"  {key:<20} {s_score:<12} {r_score:<12}")

        s_total = std_eval.get("total", 0)
        r_total = rsn_eval.get("total", 0)
        print(f"  {'TOTAL':<20} {s_total:<12} {r_total:<12}")

        all_scores["standard"].append(s_total)
        all_scores["reasoning"].append(r_total)

        # Comparison
        print(f"\n  Time comparison: {standard['elapsed']:.2f}s vs {reasoning['elapsed']:.2f}s")
        if "reasoning_tokens" in reasoning:
            print(f"  Reasoning tokens used: {reasoning['reasoning_tokens']}")

    # Overall summary
    print(f"\n\n{'=' * 60}")
    print("OVERALL COMPARISON")
    print(f"{'=' * 60}")
    avg_std = sum(all_scores["standard"]) / len(all_scores["standard"]) if all_scores["standard"] else 0
    avg_rsn = sum(all_scores["reasoning"]) / len(all_scores["reasoning"]) if all_scores["reasoning"] else 0
    print(f"  Average quality score (GPT-4o):  {avg_std:.1f}/25")
    print(f"  Average quality score (o1-mini): {avg_rsn:.1f}/25")
    print(f"\nKey takeaways:")
    print("  • Reasoning models excel at multi-factor treatment decisions")
    print("  • They better identify drug interactions across comorbidities")
    print("  • The extra latency is justified for safety-critical planning")
    print("  • Always validate AI treatment plans with clinical pharmacists")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
