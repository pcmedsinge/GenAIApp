"""
Project: Systematic Prompt Evaluation
Objective: Learn to rigorously test, compare, and validate LLM prompts
Concepts: Scoring rubrics, regression testing, A/B testing, LLM-as-judge

Healthcare Use Case: Ensuring prompt quality and safety for clinical AI systems

In healthcare AI, prompt quality is not just about style — it's about patient safety.
A prompt that omits drug interactions or gives ambiguous dosage instructions can cause
real harm. This project teaches you how to systematically evaluate prompts so that
every change is validated before reaching production.

Key techniques:
  1. LLM-as-Judge: Use a second LLM call to score outputs on defined criteria
  2. Regression Testing: Ensure prompt changes don't break expected behaviors
  3. A/B Testing: Statistically compare prompt variants across multiple trials
  4. Interactive Lab: Rapidly iterate on prompt design with instant feedback
"""

import os
import json
import time
import statistics
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def call_llm(system_prompt: str, user_message: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
    """
    Make a single LLM call and return the response text.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error: {e}]"


def llm_judge_score(output_text: str, criteria: dict, context: str = "") -> dict:
    """
    Use an LLM to score an output on multiple criteria.

    Args:
        output_text: The text to evaluate
        criteria: Dict of {criterion_name: description}
        context: Optional context about what the output should contain

    Returns:
        Dict of {criterion_name: score (1-5)}
    """
    criteria_text = "\n".join(
        f"- {name}: {desc}" for name, desc in criteria.items()
    )

    judge_prompt = f"""You are an expert evaluator for healthcare AI outputs.
Score the following text on each criterion using a scale of 1 to 5:
  1 = Very poor
  2 = Poor
  3 = Acceptable
  4 = Good
  5 = Excellent

Criteria:
{criteria_text}

{f'Context: {context}' if context else ''}

Respond ONLY with valid JSON in this exact format:
{json.dumps({name: 0 for name in criteria})}

Do not include any explanation, just the JSON object with integer scores."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": f"Text to evaluate:\n\n{output_text}"}
            ],
            temperature=0.0,
            max_tokens=200
        )
        raw = response.choices[0].message.content.strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        scores = json.loads(raw)
        # Validate scores are in range
        for key in criteria:
            if key in scores:
                scores[key] = max(1, min(5, int(scores[key])))
            else:
                scores[key] = 3  # Default if missing
        return scores
    except Exception as e:
        print(f"  [Judge error: {e}]")
        return {name: 3 for name in criteria}


def print_separator(title: str):
    """Print a formatted section separator."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_scores(scores: dict, indent: int = 2):
    """Print scores in a formatted table."""
    prefix = " " * indent
    for criterion, score in scores.items():
        bar = "█" * score + "░" * (5 - score)
        print(f"{prefix}{criterion:<20} {bar}  {score}/5")


# ============================================================
# CLINICAL SCENARIOS (shared test data)
# ============================================================

CLINICAL_SCENARIOS = [
    {
        "name": "Diabetic patient follow-up",
        "input": "Generate a clinical note for a 58-year-old male patient with Type 2 diabetes "
                 "presenting for a 3-month follow-up. HbA1c is 7.8% (previously 8.2%). Current "
                 "medications: Metformin 1000mg BID, Lisinopril 10mg daily. BMI 31.2. Patient "
                 "reports improved dietary adherence but ongoing difficulty with exercise.",
    },
    {
        "name": "Chest pain evaluation",
        "input": "Generate a clinical note for a 45-year-old female presenting to the ED with "
                 "acute onset chest pain radiating to the left arm, onset 2 hours ago. Pain is "
                 "7/10, crushing quality. History of hypertension and hyperlipidemia. Troponin "
                 "pending, ECG shows ST-segment changes in leads V1-V4.",
    },
    {
        "name": "Pediatric asthma visit",
        "input": "Generate a clinical note for a 9-year-old patient with moderate persistent "
                 "asthma presenting for routine follow-up. Currently on Fluticasone 110mcg 2 "
                 "puffs BID and Albuterol PRN. Parent reports 2 nighttime awakenings per week "
                 "and albuterol use 3 times weekly. Spirometry shows FEV1 78% predicted.",
    },
]

SCORING_CRITERIA = {
    "clarity": "How clear and well-organized is the clinical note? Easy to read and follow?",
    "completeness": "Does the note include all relevant clinical information: history, exam, assessment, plan?",
    "accuracy": "Is the medical content accurate and appropriate for the clinical scenario?",
}


# ============================================================
# DEMO 1: Basic Prompt Comparison
# ============================================================

def demo_basic_prompt_comparison():
    """
    Compare two system prompts for clinical note generation.
    An LLM judge scores each prompt's output on clarity, completeness, and accuracy.
    """
    print_separator("DEMO 1: Basic Prompt Comparison")

    prompts = {
        "Concise Clinical Assistant": (
            "You are a concise clinical documentation assistant. Write brief, "
            "focused clinical notes using standard medical abbreviations. Keep notes "
            "short and to the point. Use SOAP format. Avoid unnecessary detail."
        ),
        "Detailed Clinical Narrator": (
            "You are a thorough clinical documentation assistant. Write comprehensive "
            "clinical notes that capture every relevant detail. Include full medical "
            "terminology with explanations. Structure the note with clear sections: "
            "Chief Complaint, History of Present Illness, Review of Systems, Physical "
            "Exam, Assessment, and Plan. Ensure nothing clinically significant is omitted."
        ),
    }

    scenario = CLINICAL_SCENARIOS[0]  # Diabetic follow-up
    print(f"Scenario: {scenario['name']}")
    print(f"Input: {scenario['input'][:100]}...\n")

    results = {}

    for prompt_name, system_prompt in prompts.items():
        print(f"--- Testing: {prompt_name} ---")

        # Generate clinical note
        output = call_llm(system_prompt, scenario["input"], temperature=0.3)
        print(f"\nGenerated Note (first 200 chars):\n  {output[:200]}...\n")

        # Score with LLM judge
        scores = llm_judge_score(
            output,
            SCORING_CRITERIA,
            context=f"This is a clinical note for: {scenario['name']}"
        )
        print_scores(scores)

        avg = statistics.mean(scores.values())
        print(f"  {'Average':<20} {'':>6} {avg:.2f}/5")
        results[prompt_name] = {"scores": scores, "average": avg, "output": output}
        print()

    # Determine winner
    winner = max(results, key=lambda k: results[k]["average"])
    loser = min(results, key=lambda k: results[k]["average"])
    print(f"Winner: {winner} ({results[winner]['average']:.2f} vs {results[loser]['average']:.2f})")


# ============================================================
# DEMO 2: Regression Test Suite
# ============================================================

def demo_regression_test_suite():
    """
    Define test cases with expected behaviors, run a prompt against them,
    and check if each test passes or fails.
    """
    print_separator("DEMO 2: Regression Test Suite")

    system_prompt = (
        "You are a clinical Q&A assistant for healthcare professionals. "
        "Provide accurate, evidence-based answers to medical questions. "
        "Always mention relevant drug interactions and contraindications. "
        "Include appropriate caveats about consulting specialists when needed."
    )

    test_cases = [
        {
            "name": "Metformin contraindications",
            "input": "What are the main contraindications for Metformin?",
            "expected_keywords": ["renal", "kidney", "eGFR", "lactic acidosis"],
            "forbidden_content": ["always safe", "no side effects"],
        },
        {
            "name": "Hypertension first-line treatment",
            "input": "What is the first-line treatment for essential hypertension?",
            "expected_keywords": ["ACE inhibitor", "ARB", "calcium channel", "thiazide"],
            "forbidden_content": ["beta blocker as first line"],
        },
        {
            "name": "Warfarin interactions",
            "input": "What foods interact with Warfarin?",
            "expected_keywords": ["vitamin K", "green leafy", "cranberry"],
            "forbidden_content": ["no food interactions"],
        },
        {
            "name": "Chest pain differential",
            "input": "What is the differential diagnosis for acute chest pain?",
            "expected_keywords": ["myocardial infarction", "pulmonary embolism", "pneumothorax"],
            "forbidden_content": ["always benign", "nothing to worry"],
        },
        {
            "name": "Specialist referral advice",
            "input": "When should a primary care physician refer a patient with persistent headaches?",
            "expected_keywords": ["neurolog", "red flag", "imaging"],
            "forbidden_content": ["never refer", "always self-limiting"],
        },
    ]

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        output = call_llm(system_prompt, test["input"], temperature=0.2)
        output_lower = output.lower()

        # Check expected keywords (at least one must be present)
        found_keywords = [
            kw for kw in test["expected_keywords"]
            if kw.lower() in output_lower
        ]
        keywords_pass = len(found_keywords) > 0

        # Check forbidden content
        found_forbidden = [
            fb for fb in test["forbidden_content"]
            if fb.lower() in output_lower
        ]
        forbidden_pass = len(found_forbidden) == 0

        test_passed = keywords_pass and forbidden_pass

        if test_passed:
            print(f"  ✅ PASS")
            passed += 1
        else:
            print(f"  ❌ FAIL")
            failed += 1
            if not keywords_pass:
                print(f"     Missing keywords: expected one of {test['expected_keywords']}")
                print(f"     Found: {found_keywords if found_keywords else 'none'}")
            if not forbidden_pass:
                print(f"     Forbidden content found: {found_forbidden}")

        print(f"     Keywords matched: {found_keywords}")
        print()

    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    pct = (passed / len(test_cases)) * 100
    print(f"Pass rate: {pct:.0f}%")
    if pct == 100:
        print("All tests passed! Prompt is ready for deployment.")
    elif pct >= 80:
        print("Most tests passed. Review failures before deployment.")
    else:
        print("⚠️  Significant failures detected. Prompt needs revision.")


# ============================================================
# DEMO 3: Statistical A/B Testing
# ============================================================

def demo_ab_testing():
    """
    Run multiple comparisons between two prompt variants and compute win rates.
    Uses the LLM judge to determine which prompt produces better output on each trial.
    """
    print_separator("DEMO 3: Statistical A/B Testing")

    prompt_a = {
        "name": "Prompt A (Structured)",
        "system": (
            "You are a clinical assistant. When answering medical questions, always "
            "structure your response with: 1) Direct answer, 2) Key considerations, "
            "3) When to seek specialist care. Be concise and evidence-based."
        ),
    }

    prompt_b = {
        "name": "Prompt B (Conversational)",
        "system": (
            "You are a friendly clinical assistant. Answer medical questions in a "
            "conversational, approachable tone while maintaining medical accuracy. "
            "Include practical advice patients and providers can act on."
        ),
    }

    test_questions = [
        "What are the warning signs of sepsis?",
        "How should insulin dosing be adjusted for a patient starting steroids?",
        "What is the appropriate workup for new-onset atrial fibrillation?",
        "Describe the management of acute anaphylaxis.",
        "What are the indications for urgent dialysis?",
    ]

    num_trials = len(test_questions)
    scores_a = []
    scores_b = []

    print(f"Running {num_trials} head-to-head comparisons...\n")

    for i, question in enumerate(test_questions, 1):
        print(f"Trial {i}/{num_trials}: {question[:60]}...")

        # Generate outputs from both prompts
        output_a = call_llm(prompt_a["system"], question, temperature=0.3)
        output_b = call_llm(prompt_b["system"], question, temperature=0.3)

        # Score both outputs
        criteria = {
            "accuracy": "Medical accuracy and correctness of information",
            "usefulness": "How actionable and useful the information is for a clinician",
            "safety": "Appropriate mention of safety considerations, warnings, and caveats",
        }

        score_a = llm_judge_score(output_a, criteria, context=f"Answer to: {question}")
        score_b = llm_judge_score(output_b, criteria, context=f"Answer to: {question}")

        avg_a = statistics.mean(score_a.values())
        avg_b = statistics.mean(score_b.values())
        scores_a.append(avg_a)
        scores_b.append(avg_b)

        winner = "A" if avg_a > avg_b else ("B" if avg_b > avg_a else "Tie")
        print(f"  Prompt A: {avg_a:.2f}  |  Prompt B: {avg_b:.2f}  |  Winner: {winner}")

    # Compute statistics
    print(f"\n{'='*50}")
    print("A/B TEST RESULTS")
    print(f"{'='*50}")

    wins_a = sum(1 for a, b in zip(scores_a, scores_b) if a > b)
    wins_b = sum(1 for a, b in zip(scores_a, scores_b) if b > a)
    ties = sum(1 for a, b in zip(scores_a, scores_b) if a == b)

    print(f"\n{prompt_a['name']}:")
    print(f"  Wins: {wins_a}/{num_trials} ({wins_a/num_trials*100:.0f}%)")
    print(f"  Average score: {statistics.mean(scores_a):.2f}")
    if len(scores_a) > 1:
        print(f"  Std deviation:  {statistics.stdev(scores_a):.2f}")

    print(f"\n{prompt_b['name']}:")
    print(f"  Wins: {wins_b}/{num_trials} ({wins_b/num_trials*100:.0f}%)")
    print(f"  Average score: {statistics.mean(scores_b):.2f}")
    if len(scores_b) > 1:
        print(f"  Std deviation:  {statistics.stdev(scores_b):.2f}")

    print(f"\n  Ties: {ties}/{num_trials}")

    # Recommendation
    if wins_a > wins_b:
        print(f"\n→ Recommendation: Use {prompt_a['name']} (higher win rate)")
    elif wins_b > wins_a:
        print(f"\n→ Recommendation: Use {prompt_b['name']} (higher win rate)")
    else:
        print(f"\n→ Recommendation: No clear winner. Consider additional trials or criteria.")


# ============================================================
# DEMO 4: Interactive Prompt Lab
# ============================================================

def demo_interactive_prompt_lab():
    """
    Let the user enter a system prompt and test it against clinical scenarios.
    The LLM judge scores the output in real time.
    """
    print_separator("DEMO 4: Interactive Prompt Lab")

    print("Welcome to the Prompt Lab!")
    print("Enter a system prompt and test it against clinical scenarios.")
    print("Type 'quit' to exit.\n")

    scenarios_menu = {str(i+1): s for i, s in enumerate(CLINICAL_SCENARIOS)}

    while True:
        print("-" * 50)
        print("Enter your system prompt (or 'quit' to exit):")
        system_prompt = input("System prompt: ").strip()

        if system_prompt.lower() in ("quit", "exit", "q"):
            print("Exiting Prompt Lab.")
            break

        if not system_prompt:
            print("Empty prompt. Try again.\n")
            continue

        print("\nChoose a clinical scenario:")
        for key, scenario in scenarios_menu.items():
            print(f"  {key}. {scenario['name']}")
        print(f"  a. Run ALL scenarios")

        choice = input("Choice: ").strip().lower()

        if choice == "a":
            selected = list(scenarios_menu.values())
        elif choice in scenarios_menu:
            selected = [scenarios_menu[choice]]
        else:
            print("Invalid choice.\n")
            continue

        for scenario in selected:
            print(f"\n--- Scenario: {scenario['name']} ---")
            output = call_llm(system_prompt, scenario["input"], temperature=0.3)
            print(f"\nOutput:\n{output}\n")

            scores = llm_judge_score(
                output,
                SCORING_CRITERIA,
                context=f"Clinical note for: {scenario['name']}"
            )
            print("Scores:")
            print_scores(scores, indent=4)
            avg = statistics.mean(scores.values())
            print(f"    {'Average':<20} {'':>6} {avg:.2f}/5")

        print()


# ============================================================
# MAIN MENU
# ============================================================

def main():
    """
    Run prompt evaluation demos with a menu interface.
    """
    print("🏥 Level 4.1: Systematic Prompt Evaluation\n")
    print("This project demonstrates how to rigorously test and compare")
    print("LLM prompts for healthcare AI applications.\n")

    while True:
        print("\n" + "=" * 50)
        print("MAIN MENU")
        print("=" * 50)
        print("1. Basic Prompt Comparison (LLM-as-Judge)")
        print("2. Regression Test Suite")
        print("3. Statistical A/B Testing")
        print("4. Interactive Prompt Lab")
        print("5. Run All Demos (1-3)")
        print("q. Quit")
        print("=" * 50)

        choice = input("Select demo (1-5 or q): ").strip().lower()

        if choice == "1":
            demo_basic_prompt_comparison()
        elif choice == "2":
            demo_regression_test_suite()
        elif choice == "3":
            demo_ab_testing()
        elif choice == "4":
            demo_interactive_prompt_lab()
        elif choice == "5":
            demo_basic_prompt_comparison()
            demo_regression_test_suite()
            demo_ab_testing()
        elif choice in ("q", "quit", "exit"):
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-5 or q.")


if __name__ == "__main__":
    main()
