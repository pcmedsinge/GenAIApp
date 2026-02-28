"""
Exercise 1: Build a Custom Scoring Rubric for Clinical Notes

Skills practiced:
  - Defining multi-dimensional evaluation criteria
  - Using LLM-as-judge for automated scoring
  - Aggregating and comparing scores across scenarios
  - Weighting criteria by clinical importance

Healthcare context:
  Clinical notes must be accurate, complete, safe, and readable. In this exercise
  you build a rubric that evaluates prompts on four dimensions:
    1. Medical Accuracy — Are diagnoses, medications, and values correct?
    2. Completeness — Are all SOAP sections present and populated?
    3. Patient Safety — Are warnings, allergies, and contraindications mentioned?
    4. Readability — Is the note easy for another clinician to scan quickly?

  You then test multiple system prompts against several clinical scenarios and
  produce a ranked report.
"""

import os
import json
import statistics
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"


# ============================================================
# Scoring Rubric Definition
# ============================================================

RUBRIC = {
    "medical_accuracy": {
        "description": (
            "Are medical facts correct? Medications, dosages, lab values, and "
            "diagnoses should be clinically appropriate for the scenario."
        ),
        "weight": 2.0,  # Double-weighted — accuracy is paramount
    },
    "completeness": {
        "description": (
            "Does the note include all expected sections? Chief Complaint, HPI, "
            "Review of Systems, Physical Exam, Assessment, and Plan should all "
            "be present and substantive."
        ),
        "weight": 1.5,
    },
    "patient_safety": {
        "description": (
            "Does the note mention relevant safety considerations? Allergies, drug "
            "interactions, contraindications, follow-up instructions, and red flags "
            "should be addressed where applicable."
        ),
        "weight": 2.0,  # Double-weighted — safety is critical
    },
    "readability": {
        "description": (
            "Is the note well-organized, easy to scan, and uses appropriate medical "
            "abbreviations? Another clinician should be able to quickly find key info."
        ),
        "weight": 1.0,
    },
}


# ============================================================
# Clinical Test Scenarios
# ============================================================

CLINICAL_SCENARIOS = [
    {
        "name": "Post-surgical follow-up — knee replacement",
        "input": (
            "Generate a clinical note for a 67-year-old female, 2 weeks post right "
            "total knee arthroplasty. On Enoxaparin 40mg SC daily for DVT prophylaxis. "
            "Wound is clean, dry, intact. ROM 0-90 degrees. Pain 4/10, managed with "
            "Acetaminophen 1000mg TID. Allergic to Sulfonamides. BMI 28."
        ),
    },
    {
        "name": "New diabetes diagnosis",
        "input": (
            "Generate a clinical note for a 52-year-old male newly diagnosed with "
            "Type 2 diabetes. Fasting glucose 186 mg/dL, HbA1c 8.4%. BMI 33.1. "
            "Hypertension controlled on Amlodipine 5mg daily. No known drug allergies. "
            "Starting Metformin 500mg BID with plan to titrate. Needs referral to "
            "diabetes educator and ophthalmology for baseline eye exam."
        ),
    },
    {
        "name": "Pediatric ear infection",
        "input": (
            "Generate a clinical note for a 4-year-old male with 3 days of right ear "
            "pain, fever 101.2°F, and irritability. Otoscopic exam shows erythematous, "
            "bulging TM on the right. Left ear normal. No penicillin allergy. Plan to "
            "start Amoxicillin 90mg/kg/day divided BID for 10 days. Weight 18 kg."
        ),
    },
    {
        "name": "Elderly fall assessment",
        "input": (
            "Generate a clinical note for an 81-year-old female presenting after a "
            "mechanical fall at home. Tripped over a rug. No LOC. Mild right hip pain, "
            "weight-bearing. On Warfarin for atrial fibrillation (INR 2.8 last week). "
            "Also on Donepezil 10mg for mild Alzheimer's. CT head negative. Hip X-ray "
            "shows no fracture. Allergic to Codeine."
        ),
    },
]


# ============================================================
# System Prompts to Evaluate
# ============================================================

SYSTEM_PROMPTS = {
    "minimal": (
        "You are a clinical documentation assistant. Write a brief SOAP note."
    ),
    "structured": (
        "You are a clinical documentation assistant. Write a comprehensive clinical "
        "note with these sections: Chief Complaint, History of Present Illness, "
        "Review of Systems, Physical Exam, Assessment, and Plan. Include all relevant "
        "medications, allergies, and safety considerations."
    ),
    "safety_focused": (
        "You are a clinical documentation assistant specializing in patient safety. "
        "Write a thorough clinical note. ALWAYS highlight: drug allergies, potential "
        "drug interactions, contraindications, fall risk, and follow-up requirements. "
        "Use SOAP format. Flag any safety concerns prominently."
    ),
    "teaching_style": (
        "You are a clinical documentation assistant at a teaching hospital. Write "
        "detailed clinical notes suitable for medical education. Explain clinical "
        "reasoning behind assessment and plan decisions. Include differential diagnoses "
        "where appropriate. Note all medications with dosages and rationale."
    ),
}


# ============================================================
# Core Evaluation Functions
# ============================================================

def call_llm(system_prompt: str, user_message: str, temperature: float = 0.3) -> str:
    """Generate a clinical note using the given system prompt."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error: {e}]"


def score_with_rubric(output_text: str, scenario_name: str) -> dict:
    """
    Score a clinical note using the defined rubric via LLM-as-judge.

    Returns dict of {criterion: score} where score is 1-5.
    """
    criteria_desc = "\n".join(
        f"- {name}: {info['description']}"
        for name, info in RUBRIC.items()
    )

    judge_prompt = f"""You are an expert clinical documentation reviewer.
Score the following clinical note on each criterion (1-5 scale):
  1 = Very poor / Missing
  2 = Poor / Major gaps
  3 = Acceptable / Minor gaps
  4 = Good / Meets expectations
  5 = Excellent / Exceeds expectations

Criteria:
{criteria_desc}

Context: This note is for the scenario "{scenario_name}".

Respond ONLY with valid JSON: {json.dumps({name: 0 for name in RUBRIC})}
No explanation — just the JSON with integer scores."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": f"Clinical note to evaluate:\n\n{output_text}"}
            ],
            temperature=0.0,
            max_tokens=150
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        scores = json.loads(raw)
        for key in RUBRIC:
            scores[key] = max(1, min(5, int(scores.get(key, 3))))
        return scores
    except Exception as e:
        print(f"    [Judge error: {e}]")
        return {name: 3 for name in RUBRIC}


def weighted_average(scores: dict) -> float:
    """Compute a weighted average score using rubric weights."""
    total_weight = sum(info["weight"] for info in RUBRIC.values())
    weighted_sum = sum(
        scores[name] * RUBRIC[name]["weight"]
        for name in RUBRIC
    )
    return weighted_sum / total_weight


# ============================================================
# Evaluation Runner
# ============================================================

def evaluate_all_prompts():
    """
    Run every system prompt against every scenario, score each output,
    and produce a ranked comparison report.
    """
    print("=" * 65)
    print("  CLINICAL NOTE SCORING RUBRIC — FULL EVALUATION")
    print("=" * 65)
    print(f"\nPrompts under test: {len(SYSTEM_PROMPTS)}")
    print(f"Scenarios: {len(CLINICAL_SCENARIOS)}")
    print(f"Total evaluations: {len(SYSTEM_PROMPTS) * len(CLINICAL_SCENARIOS)}\n")

    # Print rubric
    print("Rubric:")
    for name, info in RUBRIC.items():
        print(f"  {name:<20} (weight {info['weight']:.1f}x): {info['description'][:70]}...")
    print()

    # Store all results: {prompt_name: [weighted_avg_per_scenario]}
    all_results = {name: [] for name in SYSTEM_PROMPTS}
    detailed_results = []

    for scenario in CLINICAL_SCENARIOS:
        print(f"--- Scenario: {scenario['name']} ---")

        for prompt_name, system_prompt in SYSTEM_PROMPTS.items():
            print(f"  Evaluating prompt: '{prompt_name}'...", end=" ", flush=True)

            output = call_llm(system_prompt, scenario["input"])
            scores = score_with_rubric(output, scenario["name"])
            w_avg = weighted_average(scores)
            all_results[prompt_name].append(w_avg)

            detailed_results.append({
                "prompt": prompt_name,
                "scenario": scenario["name"],
                "scores": scores,
                "weighted_avg": w_avg,
            })

            score_str = " | ".join(f"{k}: {v}" for k, v in scores.items())
            print(f"[{score_str}]  weighted avg: {w_avg:.2f}")

        print()

    # ============================================================
    # Summary Report
    # ============================================================
    print("=" * 65)
    print("  SUMMARY REPORT")
    print("=" * 65)

    prompt_averages = {}
    for prompt_name, scores_list in all_results.items():
        avg = statistics.mean(scores_list)
        prompt_averages[prompt_name] = avg

    # Sort by average score descending
    ranked = sorted(prompt_averages.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<6} {'Prompt':<20} {'Avg Weighted Score':<20}")
    print("-" * 46)
    for rank, (name, avg) in enumerate(ranked, 1):
        bar = "█" * int(avg) + "░" * (5 - int(avg))
        print(f"  {rank}    {name:<20} {bar}  {avg:.2f}/5.00")

    best = ranked[0]
    worst = ranked[-1]
    print(f"\n✅ Best prompt:  '{best[0]}' (avg {best[1]:.2f})")
    print(f"⚠️  Worst prompt: '{worst[0]}' (avg {worst[1]:.2f})")

    # Per-criterion breakdown for each prompt
    print(f"\nPer-Criterion Averages:")
    for prompt_name in SYSTEM_PROMPTS:
        details = [d for d in detailed_results if d["prompt"] == prompt_name]
        print(f"\n  Prompt: '{prompt_name}'")
        for criterion in RUBRIC:
            avg_c = statistics.mean(d["scores"][criterion] for d in details)
            bar = "█" * int(round(avg_c)) + "░" * (5 - int(round(avg_c)))
            print(f"    {criterion:<20} {bar}  {avg_c:.2f}")

    print(f"\n{'='*65}")
    print("Evaluation complete.")


# ============================================================
# Main
# ============================================================

def main():
    print("🏥 Exercise 1: Custom Scoring Rubric for Clinical Notes\n")
    print("This exercise evaluates multiple system prompts across clinical")
    print("scenarios using a weighted scoring rubric.\n")
    evaluate_all_prompts()


if __name__ == "__main__":
    main()
