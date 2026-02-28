"""
Exercise 4: Automated Evaluation Pipeline

Skills practiced:
  - Building end-to-end evaluation pipelines
  - Running all combinations of prompts × test cases
  - Scoring and aggregating results programmatically
  - Generating summary reports with best/worst performers

Healthcare context:
  Before deploying any clinical AI prompt to production, it must pass through
  an automated evaluation pipeline. This exercise builds a pipeline that:
    1. Takes a set of candidate prompts and test scenarios
    2. Runs every prompt against every scenario
    3. Scores each output on multiple criteria via LLM-as-judge
    4. Aggregates results into a leaderboard
    5. Produces an actionable report identifying the best prompt for deployment
"""

import os
import json
import time
import statistics
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"


# ============================================================
# Candidate Prompts
# ============================================================

CANDIDATE_PROMPTS = {
    "baseline": (
        "You are a medical assistant. Answer clinical questions."
    ),
    "structured": (
        "You are a clinical Q&A assistant. Structure answers with: "
        "1) Direct answer, 2) Key considerations, 3) Safety warnings, "
        "4) When to refer to a specialist. Be evidence-based and concise."
    ),
    "safety_first": (
        "You are a clinical safety advisor. When answering medical questions, "
        "ALWAYS lead with safety warnings and contraindications. Then provide "
        "the clinical answer with evidence. Highlight drug interactions and "
        "red flags prominently. Always recommend consulting specialists for "
        "complex cases."
    ),
    "teaching": (
        "You are a clinical educator. When answering medical questions, explain "
        "the underlying pathophysiology briefly, then give the clinical answer. "
        "Include relevant differential diagnoses. Cite current guidelines when "
        "possible. Use clear medical terminology with brief explanations."
    ),
    "concise": (
        "You are a clinical Q&A system. Provide brief, factual answers to medical "
        "questions. Use bullet points. No unnecessary explanation. Focus on "
        "actionable clinical information only."
    ),
}


# ============================================================
# Test Scenarios
# ============================================================

TEST_SCENARIOS = [
    {
        "id": "SC-001",
        "name": "Hypertensive emergency",
        "input": "A patient presents with BP 220/130 and headache. What is the management?",
        "expected_keywords": ["IV", "nitroprusside", "labetalol", "ICU", "gradual"],
        "category": "Emergency",
    },
    {
        "id": "SC-002",
        "name": "Insulin adjustment for illness",
        "input": "How should insulin be adjusted during acute illness in a Type 1 diabetic?",
        "expected_keywords": ["sick day", "ketone", "hydration", "glucose monitoring"],
        "category": "Endocrinology",
    },
    {
        "id": "SC-003",
        "name": "Anticoagulation bridging",
        "input": "When should bridging anticoagulation be used for a patient on Warfarin undergoing surgery?",
        "expected_keywords": ["risk", "thromboembolic", "heparin", "INR"],
        "category": "Hematology",
    },
    {
        "id": "SC-004",
        "name": "Neonatal jaundice workup",
        "input": "What is the initial workup for neonatal jaundice in a 3-day-old newborn?",
        "expected_keywords": ["bilirubin", "direct", "indirect", "phototherapy"],
        "category": "Pediatrics",
    },
    {
        "id": "SC-005",
        "name": "COPD exacerbation treatment",
        "input": "What is the treatment protocol for an acute COPD exacerbation?",
        "expected_keywords": ["bronchodilator", "steroid", "antibiotic", "oxygen"],
        "category": "Pulmonology",
    },
    {
        "id": "SC-006",
        "name": "Depression screening in primary care",
        "input": "What validated tools should be used to screen for depression in primary care?",
        "expected_keywords": ["PHQ", "screening", "questionnaire"],
        "category": "Psychiatry",
    },
    {
        "id": "SC-007",
        "name": "Post-MI medications",
        "input": "What medications should be started after a myocardial infarction?",
        "expected_keywords": ["aspirin", "statin", "beta blocker", "ACE inhibitor"],
        "category": "Cardiology",
    },
    {
        "id": "SC-008",
        "name": "Chronic kidney disease staging",
        "input": "How is chronic kidney disease staged and what are the key management steps at each stage?",
        "expected_keywords": ["GFR", "stage", "proteinuria", "referral"],
        "category": "Nephrology",
    },
]


# ============================================================
# Scoring Criteria
# ============================================================

SCORING_CRITERIA = {
    "accuracy": "Medical accuracy — correct facts, dosages, and clinical reasoning",
    "completeness": "Covers all important aspects of the clinical question",
    "safety": "Mentions appropriate warnings, contraindications, and caveats",
    "actionability": "Provides clear, actionable guidance for the clinician",
}


# ============================================================
# Core Pipeline Functions
# ============================================================

def call_llm(system_prompt: str, user_message: str) -> str:
    """Generate an LLM response."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error: {e}]"


def score_output(output_text: str, scenario_name: str) -> dict:
    """Score an output using LLM-as-judge."""
    criteria_text = "\n".join(f"- {k}: {v}" for k, v in SCORING_CRITERIA.items())

    judge_prompt = f"""You are an expert clinical AI evaluator.
Score this text on each criterion (1 to 5):
  1 = Very poor, 2 = Poor, 3 = Acceptable, 4 = Good, 5 = Excellent

Criteria:
{criteria_text}

Context: Answer to "{scenario_name}".
Respond ONLY with JSON: {json.dumps({k: 0 for k in SCORING_CRITERIA})}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": f"Text:\n\n{output_text}"}
            ],
            temperature=0.0,
            max_tokens=100
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        scores = json.loads(raw)
        return {k: max(1, min(5, int(scores.get(k, 3)))) for k in SCORING_CRITERIA}
    except Exception:
        return {k: 3 for k in SCORING_CRITERIA}


def check_keywords(output: str, expected_keywords: list) -> dict:
    """Check which expected keywords appear in the output."""
    output_lower = output.lower()
    found = [kw for kw in expected_keywords if kw.lower() in output_lower]
    missing = [kw for kw in expected_keywords if kw.lower() not in output_lower]
    return {
        "found": found,
        "missing": missing,
        "pass": len(found) > 0,
    }


# ============================================================
# The Pipeline
# ============================================================

def run_evaluation_pipeline(
    prompts: dict = None,
    scenarios: list = None,
    verbose: bool = True,
) -> dict:
    """
    Run the full evaluation pipeline: all prompts × all scenarios.

    Args:
        prompts: Dict of {name: system_prompt_text}
        scenarios: List of scenario dicts
        verbose: Whether to print progress

    Returns:
        Full results dict with per-prompt and per-scenario breakdowns.
    """
    prompts = prompts or CANDIDATE_PROMPTS
    scenarios = scenarios or TEST_SCENARIOS

    total_evals = len(prompts) * len(scenarios)

    if verbose:
        print("=" * 70)
        print("  AUTOMATED EVALUATION PIPELINE")
        print("=" * 70)
        print(f"\n  Prompts:    {len(prompts)}")
        print(f"  Scenarios:  {len(scenarios)}")
        print(f"  Total evals: {total_evals}")
        print(f"  Criteria:   {', '.join(SCORING_CRITERIA.keys())}")
        print()

    start_time = time.time()

    # Results storage
    all_results = []  # List of individual result dicts
    prompt_scores = {name: [] for name in prompts}  # Overall scores per prompt
    prompt_passes = {name: 0 for name in prompts}   # Keyword pass counts

    eval_count = 0

    for prompt_name, system_prompt in prompts.items():
        if verbose:
            print(f"  Evaluating prompt: '{prompt_name}'")

        for scenario in scenarios:
            eval_count += 1
            if verbose:
                print(f"    [{eval_count:>3}/{total_evals}] {scenario['id']}: {scenario['name'][:40]}...",
                      end=" ", flush=True)

            # Generate output
            output = call_llm(system_prompt, scenario["input"])

            # Score output
            scores = score_output(output, scenario["name"])
            avg_score = statistics.mean(scores.values())

            # Keyword check
            kw_result = check_keywords(output, scenario["expected_keywords"])

            result = {
                "prompt": prompt_name,
                "scenario_id": scenario["id"],
                "scenario_name": scenario["name"],
                "category": scenario["category"],
                "scores": scores,
                "avg_score": avg_score,
                "keyword_pass": kw_result["pass"],
                "keywords_found": kw_result["found"],
                "keywords_missing": kw_result["missing"],
                "output_preview": output[:150],
            }
            all_results.append(result)
            prompt_scores[prompt_name].append(avg_score)
            if kw_result["pass"]:
                prompt_passes[prompt_name] += 1

            if verbose:
                status = "✅" if kw_result["pass"] else "❌"
                print(f"{status} avg={avg_score:.2f}")

        if verbose:
            print()

    elapsed = time.time() - start_time

    # ============================================================
    # Build Leaderboard
    # ============================================================
    leaderboard = []
    for prompt_name in prompts:
        scores_list = prompt_scores[prompt_name]
        passes = prompt_passes[prompt_name]
        total = len(scenarios)

        entry = {
            "prompt": prompt_name,
            "avg_score": statistics.mean(scores_list),
            "min_score": min(scores_list),
            "max_score": max(scores_list),
            "std_dev": statistics.stdev(scores_list) if len(scores_list) > 1 else 0.0,
            "keyword_pass_rate": passes / total,
            "passes": passes,
            "total": total,
        }

        # Per-criterion averages
        prompt_results = [r for r in all_results if r["prompt"] == prompt_name]
        for criterion in SCORING_CRITERIA:
            entry[f"avg_{criterion}"] = statistics.mean(
                r["scores"][criterion] for r in prompt_results
            )

        leaderboard.append(entry)

    # Sort by average score descending
    leaderboard.sort(key=lambda x: x["avg_score"], reverse=True)

    return {
        "leaderboard": leaderboard,
        "all_results": all_results,
        "elapsed_seconds": elapsed,
        "total_evaluations": total_evals,
    }


# ============================================================
# Report Generator
# ============================================================

def print_report(pipeline_results: dict):
    """Print a comprehensive evaluation report."""
    leaderboard = pipeline_results["leaderboard"]
    all_results = pipeline_results["all_results"]
    elapsed = pipeline_results["elapsed_seconds"]
    total = pipeline_results["total_evaluations"]

    print("\n" + "=" * 70)
    print("  EVALUATION REPORT")
    print("=" * 70)

    # Leaderboard
    print(f"\n  {'Rank':<6} {'Prompt':<18} {'Avg Score':<11} {'Pass Rate':<11} {'Min':<6} {'Max':<6} {'StdDev'}")
    print(f"  {'-'*64}")

    for rank, entry in enumerate(leaderboard, 1):
        print(f"  {rank:<6} {entry['prompt']:<18} {entry['avg_score']:<11.2f} "
              f"{entry['keyword_pass_rate']*100:<10.0f}% {entry['min_score']:<6.2f} "
              f"{entry['max_score']:<6.2f} {entry['std_dev']:.2f}")

    # Best and worst
    best = leaderboard[0]
    worst = leaderboard[-1]

    print(f"\n  🏆 BEST PERFORMER:  '{best['prompt']}' (avg {best['avg_score']:.2f}, "
          f"pass rate {best['keyword_pass_rate']*100:.0f}%)")
    print(f"  ⚠️  WORST PERFORMER: '{worst['prompt']}' (avg {worst['avg_score']:.2f}, "
          f"pass rate {worst['keyword_pass_rate']*100:.0f}%)")

    # Per-criterion breakdown for best prompt
    print(f"\n  Per-Criterion Breakdown (Best Prompt: '{best['prompt']}'):")
    for criterion in SCORING_CRITERIA:
        score = best[f"avg_{criterion}"]
        bar = "█" * int(round(score)) + "░" * (5 - int(round(score)))
        print(f"    {criterion:<16} {bar}  {score:.2f}/5")

    # Category analysis
    print(f"\n  Performance by Category (Best Prompt: '{best['prompt']}'):")
    best_results = [r for r in all_results if r["prompt"] == best["prompt"]]
    categories = sorted(set(r["category"] for r in best_results))
    for cat in categories:
        cat_results = [r for r in best_results if r["category"] == cat]
        cat_avg = statistics.mean(r["avg_score"] for r in cat_results)
        cat_passes = sum(1 for r in cat_results if r["keyword_pass"])
        status = "✅" if cat_passes == len(cat_results) else "⚠️"
        print(f"    {status} {cat:<20} avg={cat_avg:.2f}  keywords={cat_passes}/{len(cat_results)}")

    # Weakest scenarios across all prompts
    print(f"\n  Weakest Scenarios (lowest avg across all prompts):")
    scenario_avgs = {}
    for scenario in TEST_SCENARIOS:
        s_results = [r for r in all_results if r["scenario_id"] == scenario["id"]]
        scenario_avgs[scenario["id"]] = {
            "name": scenario["name"],
            "avg": statistics.mean(r["avg_score"] for r in s_results),
        }
    weakest = sorted(scenario_avgs.items(), key=lambda x: x[1]["avg"])[:3]
    for sid, info in weakest:
        print(f"    {sid}: {info['name']} (avg {info['avg']:.2f})")

    # Metadata
    print(f"\n  Pipeline Stats:")
    print(f"    Total evaluations: {total}")
    print(f"    Duration: {elapsed:.1f}s")
    print(f"    Avg time per eval: {elapsed/total:.1f}s")

    # Deployment recommendation
    print(f"\n{'='*70}")
    if best["keyword_pass_rate"] >= 0.8 and best["avg_score"] >= 3.5:
        print(f"  RECOMMENDATION: Deploy '{best['prompt']}' — meets quality thresholds")
    elif best["keyword_pass_rate"] >= 0.6:
        print(f"  RECOMMENDATION: '{best['prompt']}' is best but needs improvement")
        print(f"  Review weak scenarios before deployment.")
    else:
        print(f"  RECOMMENDATION: No prompt meets deployment thresholds. Revise all candidates.")
    print(f"{'='*70}\n")


# ============================================================
# Custom Pipeline Mode
# ============================================================

def custom_pipeline():
    """
    Let users add their own prompts and run them through the pipeline.
    """
    print("\n" + "=" * 70)
    print("  CUSTOM EVALUATION PIPELINE")
    print("=" * 70)
    print("\nEnter custom prompts to evaluate alongside the defaults.")
    print("Type 'done' when finished adding prompts.\n")

    custom_prompts = dict(CANDIDATE_PROMPTS)

    while True:
        name = input("Prompt name (or 'done'): ").strip()
        if name.lower() == "done":
            break
        if not name:
            continue
        prompt_text = input("System prompt text: ").strip()
        if prompt_text:
            custom_prompts[name] = prompt_text
            print(f"  → Added '{name}'\n")

    print(f"\nRunning pipeline with {len(custom_prompts)} prompts...\n")
    results = run_evaluation_pipeline(prompts=custom_prompts, verbose=True)
    print_report(results)


# ============================================================
# Main
# ============================================================

def main():
    print("🏥 Exercise 4: Automated Evaluation Pipeline\n")
    print("Choose mode:")
    print("  1. Run full pipeline with default prompts")
    print("  2. Add custom prompts and evaluate")
    print()

    choice = input("Choice (1 or 2): ").strip()

    if choice == "2":
        custom_pipeline()
    else:
        results = run_evaluation_pipeline(verbose=True)
        print_report(results)


if __name__ == "__main__":
    main()
