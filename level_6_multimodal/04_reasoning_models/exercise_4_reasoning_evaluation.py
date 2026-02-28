"""
Exercise 4: Reasoning Model Performance Evaluation
====================================================
Evaluate reasoning model performance on 5 complex clinical scenarios.
Score each on: accuracy, reasoning quality, cost, and latency.
Compare across model tiers.

Learning Objectives:
  - Design evaluation frameworks for clinical AI
  - Benchmark reasoning vs standard models systematically
  - Measure multiple dimensions: accuracy, quality, cost, speed
  - Understand performance tradeoffs for model selection

Usage:
  python exercise_4_reasoning_evaluation.py
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


# ============================================================================
# EVALUATION SCHEMAS
# ============================================================================

class EvaluationScore(BaseModel):
    """Score for a single evaluation dimension."""
    dimension: str
    score: int = Field(..., ge=1, le=5, description="Score 1-5")
    justification: str


class ModelEvaluation(BaseModel):
    """Complete evaluation of a model's response."""
    accuracy: EvaluationScore
    reasoning_quality: EvaluationScore
    completeness: EvaluationScore
    clinical_safety: EvaluationScore
    overall_score: float = Field(..., description="Weighted average 1-5")
    strengths: List[str]
    weaknesses: List[str]


# ============================================================================
# TEST SCENARIOS WITH KNOWN CORRECT ANSWERS
# ============================================================================

EVALUATION_SCENARIOS = [
    {
        "id": 1,
        "title": "Drug-Induced QT Prolongation Risk",
        "clinical_question": """
        A psychiatrist wants to start a 48-year-old female on quetiapine for
        treatment-resistant depression. She is currently on:
        - Escitalopram 20 mg daily (SSRI)
        - Methadone 80 mg daily (opioid use disorder)
        - Ondansetron 8 mg PRN (nausea)
        - Levofloxacin (just started for UTI)

        Her baseline QTc is 445 ms. She has hypokalemia (K+ 3.2).

        What are the QT prolongation risks and how should this be managed?
        """,
        "key_points": [
            "Methadone is a major QT prolonger",
            "Escitalopram prolongs QT (dose-dependent)",
            "Ondansetron prolongs QT",
            "Levofloxacin prolongs QT (fluoroquinolone class effect)",
            "Adding quetiapine creates dangerous cumulative QT risk",
            "Hypokalemia worsens QT prolongation risk",
            "Baseline QTc 445 is already borderline prolonged (>440 in women)",
            "Must correct potassium first",
            "Consider ECG monitoring",
            "May need to avoid quetiapine or substitute less QT-prolonging agents",
        ],
    },
    {
        "id": 2,
        "title": "Serotonin Syndrome vs NMS Differentiation",
        "clinical_question": """
        A 55-year-old male was recently started on linezolid for a resistant
        wound infection. He is also on sertraline 150 mg, tramadol PRN for
        pain, and was given metoclopramide for nausea. He presents with:
        - Temperature 103.2°F
        - Heart rate 120
        - Blood pressure 168/94
        - Tremor, myoclonus, hyperreflexia
        - Agitation and confusion
        - Diaphoresis
        - Dilated pupils

        Is this serotonin syndrome or neuroleptic malignant syndrome? What is
        the cause and management?
        """,
        "key_points": [
            "This is serotonin syndrome (not NMS)",
            "Linezolid is an MAO inhibitor (often overlooked)",
            "Sertraline + linezolid = dangerous serotonergic combination",
            "Tramadol has serotonergic activity (additional risk)",
            "Metoclopramide has weak serotonergic properties",
            "Key differentiators: hyperreflexia, myoclonus, diaphoresis favor SS",
            "NMS would show lead-pipe rigidity, bradyreflexia",
            "Management: stop all serotonergic agents",
            "Cyproheptadine is the specific antidote for SS",
            "Benzodiazepines for agitation",
            "Supportive care, cooling if needed",
        ],
    },
    {
        "id": 3,
        "title": "Complex Acid-Base Analysis",
        "clinical_question": """
        A 62-year-old with COPD and chronic kidney disease presents with:
        ABG: pH 7.22, pCO2 55, HCO3 22, PaO2 58
        BMP: Na 140, K 5.8, Cl 112, HCO3 22, BUN 68, Cr 4.2, Glucose 105
        Lactate: 4.2 mmol/L
        Urine drug screen: Negative
        History: 3 days of worsening dyspnea, productive cough

        Identify ALL acid-base disorders present. Show your work with the
        expected values and calculations.
        """,
        "key_points": [
            "Primary respiratory acidosis (elevated pCO2 with low pH)",
            "Expected HCO3 compensation for acute resp acidosis: HCO3 should rise ~1 for every 10 pCO2 rise",
            "Expected HCO3 for chronic: rise ~3.5 per 10 pCO2 increase",
            "Anion gap = 140 - 112 - 22 = 6 (normal)",
            "Non-anion gap metabolic acidosis also present (if HCO3 lower than expected)",
            "Delta-delta ratio should be calculated",
            "Elevated lactate (4.2) suggests lactic acidosis component",
            "CKD contributing to metabolic acidosis (uremic)",
            "This is a mixed acid-base disorder",
            "COPD exacerbation causing acute-on-chronic respiratory acidosis",
        ],
    },
    {
        "id": 4,
        "title": "Warfarin Reversal Decision",
        "clinical_question": """
        An 80-year-old on warfarin (for mechanical aortic valve, target INR 2.0-3.0)
        presents with:
        - INR: 8.2
        - No active bleeding
        - But needs emergent surgery for acute appendicitis within 6 hours
        - Cr 2.1 (CKD stage 3)
        - History of HIT (can't use heparin)
        - Weight: 55 kg

        What is the optimal reversal strategy? What bridging anticoagulation
        after surgery? Consider the mechanical valve, surgery urgency, and HIT history.
        """,
        "key_points": [
            "4-factor PCC (Kcentra) is preferred over FFP for urgent reversal",
            "IV Vitamin K 10 mg slow infusion (but takes hours for full effect)",
            "PCC provides immediate INR correction",
            "Cannot use heparin (HIT history) — need alternative bridging",
            "Bivalirudin or argatroban for post-op bridging (direct thrombin inhibitors)",
            "Mechanical valve requires uninterrupted anticoagulation plan",
            "Recheck INR after PCC, target <1.5 for surgery",
            "Resume warfarin post-op when safe (with bridging until therapeutic)",
            "CKD affects PCC dosing considerations",
            "FFP has volume concerns in elderly patients",
        ],
    },
    {
        "id": 5,
        "title": "Pediatric Diabetic Ketoacidosis Management",
        "clinical_question": """
        A 7-year-old (25 kg) presents with new-onset Type 1 diabetes in DKA:
        - pH 7.08, pCO2 15, HCO3 5
        - Glucose 520 mg/dL
        - Na 128, K 5.8, BUN 32, Cr 1.1
        - Appears moderately dehydrated (8-10%)
        - Kussmaul breathing, lethargic but responsive

        Provide the complete DKA management protocol including:
        1. Fluid resuscitation plan with specific rates
        2. Insulin protocol
        3. Potassium management strategy
        4. When to switch to D5 fluids
        5. Monitoring frequency
        6. Cerebral edema prevention measures
        """,
        "key_points": [
            "This is severe DKA (pH <7.1)",
            "Initial bolus: 10-20 mL/kg NS = 250-500 mL over 1 hour",
            "Maintenance + deficit replacement over 24-48 hours",
            "Insulin: 0.05-0.1 units/kg/hr continuous infusion",
            "Do NOT bolus insulin in pediatric DKA (cerebral edema risk)",
            "K+ 5.8 but will drop with insulin — check before starting insulin",
            "Add K+ to fluids when K+ <5.5 and patient has adequate urine output",
            "Switch to D5 when glucose <250-300 mg/dL",
            "Monitor glucose hourly, BMP q2-4h, neuro checks q1h",
            "Cerebral edema prevention: gradual correction, avoid rapid Na shifts",
            "Corrected sodium = 128 + 1.6*(520-100)/100 = ~135 — relatively normal",
            "Avoid bicarbonate unless pH <6.9",
        ],
    },
]


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def get_model_response(model: str, question: str) -> dict:
    """Get response from any model with appropriate parameters."""
    start = time.time()

    if model in ("gpt-4o", "gpt-4o-mini"):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical expert. Provide thorough, evidence-based "
                        "analysis. Show your reasoning step by step."
                    ),
                },
                {"role": "user", "content": question},
            ],
            max_tokens=2000,
            temperature=0.1,
        )
    else:
        # Reasoning model
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": (
                    "You are a clinical expert. Provide thorough, evidence-based "
                    "analysis. Show your reasoning step by step.\n\n"
                    + question
                ),
            }],
            max_completion_tokens=4000,
        )

    elapsed = time.time() - start
    usage = response.usage

    result = {
        "model": model,
        "content": response.choices[0].message.content,
        "tokens": usage.total_tokens,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "elapsed": elapsed,
    }
    return result


def score_response(response_text: str, key_points: list) -> dict:
    """Score a response against known key points."""
    # Use GPT-4o to evaluate
    points_text = "\n".join(f"- {p}" for p in key_points)

    eval_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical education evaluator. Score the response against "
                    "the key points. For each key point, determine if it was:\n"
                    "- COVERED: explicitly addressed\n"
                    "- PARTIALLY: mentioned but incomplete\n"
                    "- MISSED: not addressed\n\n"
                    "Also rate overall:\n"
                    "- accuracy (1-5): correctness of medical content\n"
                    "- reasoning_quality (1-5): clarity and depth of reasoning\n"
                    "- completeness (1-5): how many key points covered\n"
                    "- clinical_safety (1-5): no dangerous omissions or errors\n\n"
                    "Respond in JSON: {\"points_covered\": N, \"points_partial\": N, "
                    "\"points_missed\": N, \"total_points\": N, \"accuracy\": N, "
                    "\"reasoning_quality\": N, \"completeness\": N, \"clinical_safety\": N, "
                    "\"key_strengths\": [...], \"key_weaknesses\": [...]}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"KEY POINTS TO EVALUATE AGAINST:\n{points_text}\n\n"
                    f"RESPONSE TO EVALUATE:\n{response_text[:3000]}"
                ),
            },
        ],
        max_tokens=800,
        temperature=0.1,
    )

    try:
        content = eval_response.choices[0].message.content
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            return json.loads(content[start_idx:end_idx])
    except (json.JSONDecodeError, ValueError):
        pass

    return {
        "accuracy": 0, "reasoning_quality": 0,
        "completeness": 0, "clinical_safety": 0,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run evaluation across models and scenarios."""
    print("=" * 70)
    print("EXERCISE 4: Reasoning Model Performance Evaluation")
    print("=" * 70)

    models_to_test = ["gpt-4o", "o1-mini"]
    all_results = {model: [] for model in models_to_test}

    for scenario in EVALUATION_SCENARIOS:
        print(f"\n\n{'#' * 70}")
        print(f"  SCENARIO {scenario['id']}: {scenario['title']}")
        print(f"{'#' * 70}")
        print(f"  ({len(scenario['key_points'])} key points to evaluate)")

        for model in models_to_test:
            print(f"\n  --- {model} ---")

            # Get response
            result = get_model_response(model, scenario["clinical_question"])
            print(f"  Response: {result['content'][:200]}...")
            print(f"  Tokens: {result['tokens']} | Time: {result['elapsed']:.2f}s")

            # Score response
            scores = score_response(result["content"], scenario["key_points"])
            result["scores"] = scores
            all_results[model].append(result)

            # Display scores
            for dim in ["accuracy", "reasoning_quality", "completeness", "clinical_safety"]:
                score = scores.get(dim, 0)
                bar = "█" * score + "░" * (5 - score)
                print(f"    {dim:<20} {bar} {score}/5")

            covered = scores.get("points_covered", 0)
            total = scores.get("total_points", len(scenario["key_points"]))
            print(f"    Key points covered: {covered}/{total}")

    # ============================================================================
    # AGGREGATE RESULTS
    # ============================================================================
    print(f"\n\n{'=' * 70}")
    print("AGGREGATE EVALUATION RESULTS")
    print(f"{'=' * 70}")

    dimensions = ["accuracy", "reasoning_quality", "completeness", "clinical_safety"]

    print(f"\n  {'Metric':<25}", end="")
    for model in models_to_test:
        print(f" {model:<15}", end="")
    print()
    print(f"  {'─'*25}", end="")
    for _ in models_to_test:
        print(f" {'─'*15}", end="")
    print()

    model_totals = {m: {d: 0 for d in dimensions} for m in models_to_test}

    for dim in dimensions:
        print(f"  {dim:<25}", end="")
        for model in models_to_test:
            scores = [
                r["scores"].get(dim, 0)
                for r in all_results[model]
                if "scores" in r
            ]
            avg = sum(scores) / len(scores) if scores else 0
            model_totals[model][dim] = avg
            print(f" {avg:.2f}/5{'':<8}", end="")
        print()

    # Overall scores
    print(f"\n  {'OVERALL':<25}", end="")
    for model in models_to_test:
        avg_all = sum(model_totals[model].values()) / len(dimensions)
        print(f" {avg_all:.2f}/5{'':<8}", end="")
    print()

    # Cost and latency
    print(f"\n  {'Avg latency':<25}", end="")
    for model in models_to_test:
        avg_time = sum(r["elapsed"] for r in all_results[model]) / len(all_results[model])
        print(f" {avg_time:.2f}s{'':<10}", end="")
    print()

    print(f"  {'Avg tokens':<25}", end="")
    for model in models_to_test:
        avg_tok = sum(r["tokens"] for r in all_results[model]) / len(all_results[model])
        print(f" {avg_tok:.0f}{'':<10}", end="")
    print()

    # Recommendations
    print(f"\n\n{'=' * 70}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 70}")
    print("  Based on evaluation results:")
    print("  • Reasoning models (o1-mini) typically score higher on complex cases")
    print("  • Standard models (GPT-4o) are 3-5x faster and cheaper")
    print("  • For safety-critical decisions, the accuracy improvement justifies cost")
    print("  • Consider hybrid: route simple queries to GPT-4o, complex to o1-mini")
    print("  • Always validate AI outputs with clinical expertise")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
