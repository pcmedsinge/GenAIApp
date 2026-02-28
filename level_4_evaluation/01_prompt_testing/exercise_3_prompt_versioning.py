"""
Exercise 3: Prompt Version Tracking and Regression Detection

Skills practiced:
  - Storing prompt versions with timestamps and metadata
  - Running evaluations on each version programmatically
  - Comparing metrics across versions to detect improvements and regressions
  - Building a version history with audit trail

Healthcare context:
  In regulated healthcare environments, every change to a clinical AI system
  must be tracked and auditable. Prompt versioning ensures that:
    - You can always roll back to a known-good prompt version
    - Regressions are detected automatically before deployment
    - Compliance teams can review the full change history
    - Performance trends are visible over time
"""

import os
import json
import time
import statistics
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"


# ============================================================
# Prompt Version Store
# ============================================================

class PromptVersionStore:
    """
    In-memory version store for prompt tracking.
    In production, this would be backed by a database.
    """

    def __init__(self):
        self.versions = []
        self.evaluation_history = []

    def add_version(self, prompt_text: str, author: str, change_description: str,
                    timestamp: datetime = None) -> dict:
        """Register a new prompt version."""
        version_num = len(self.versions) + 1
        version = {
            "version": version_num,
            "prompt": prompt_text,
            "author": author,
            "change_description": change_description,
            "timestamp": (timestamp or datetime.now()).isoformat(),
            "status": "draft",  # draft, testing, approved, deployed, retired
        }
        self.versions.append(version)
        return version

    def get_version(self, version_num: int) -> dict:
        """Retrieve a specific version."""
        for v in self.versions:
            if v["version"] == version_num:
                return v
        return None

    def get_latest(self) -> dict:
        """Get the most recent version."""
        return self.versions[-1] if self.versions else None

    def get_all_versions(self) -> list:
        """Return all versions."""
        return list(self.versions)

    def update_status(self, version_num: int, status: str):
        """Update the status of a version."""
        for v in self.versions:
            if v["version"] == version_num:
                v["status"] = status
                break

    def record_evaluation(self, version_num: int, scores: dict, test_pass_rate: float):
        """Record evaluation results for a version."""
        record = {
            "version": version_num,
            "scores": scores,
            "test_pass_rate": test_pass_rate,
            "evaluated_at": datetime.now().isoformat(),
        }
        self.evaluation_history.append(record)
        return record

    def get_evaluations(self, version_num: int = None) -> list:
        """Get evaluation history, optionally filtered by version."""
        if version_num is not None:
            return [e for e in self.evaluation_history if e["version"] == version_num]
        return list(self.evaluation_history)


# ============================================================
# Evaluation Infrastructure
# ============================================================

EVAL_CRITERIA = {
    "accuracy": "Medical accuracy and correctness of clinical information",
    "completeness": "Thoroughness — are all relevant clinical details included?",
    "safety": "Appropriate mention of warnings, contraindications, and safety caveats",
}

EVAL_SCENARIOS = [
    {
        "name": "Diabetes management question",
        "input": "What are the current guidelines for initial management of Type 2 diabetes?",
        "expected_keywords": ["metformin", "lifestyle", "HbA1c", "diet"],
    },
    {
        "name": "Drug interaction query",
        "input": "What should I consider when prescribing SSRIs to a patient already on MAOIs?",
        "expected_keywords": ["serotonin syndrome", "contraindicated", "dangerous"],
    },
    {
        "name": "Pediatric dosing question",
        "input": "How should Amoxicillin be dosed for acute otitis media in children?",
        "expected_keywords": ["mg/kg", "weight", "divided", "days"],
    },
    {
        "name": "Emergency protocol",
        "input": "What is the acute management protocol for status epilepticus?",
        "expected_keywords": ["benzodiazepine", "lorazepam", "airway", "IV"],
    },
]


def call_llm(system_prompt: str, user_message: str) -> str:
    """Make an LLM call and return text."""
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
    """Score an output using LLM-as-judge on the evaluation criteria."""
    criteria_text = "\n".join(f"- {k}: {v}" for k, v in EVAL_CRITERIA.items())

    judge_prompt = f"""You are an expert clinical AI evaluator.
Score this output on each criterion (1-5 scale):
{criteria_text}

Context: Answer for "{scenario_name}".
Respond ONLY with JSON: {json.dumps({k: 0 for k in EVAL_CRITERIA})}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": f"Output to evaluate:\n\n{output_text}"}
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
        return {k: max(1, min(5, int(scores.get(k, 3)))) for k in EVAL_CRITERIA}
    except Exception:
        return {k: 3 for k in EVAL_CRITERIA}


def evaluate_version(store: PromptVersionStore, version_num: int) -> dict:
    """
    Run full evaluation for a prompt version: score outputs + keyword tests.
    Returns summary dict with scores and pass rate.
    """
    version = store.get_version(version_num)
    if not version:
        print(f"  Version {version_num} not found.")
        return {}

    prompt = version["prompt"]
    all_scores = {k: [] for k in EVAL_CRITERIA}
    passes = 0
    total = len(EVAL_SCENARIOS)

    for scenario in EVAL_SCENARIOS:
        output = call_llm(prompt, scenario["input"])
        output_lower = output.lower()

        # Keyword check
        found = any(kw.lower() in output_lower for kw in scenario["expected_keywords"])
        if found:
            passes += 1

        # Score check
        scores = score_output(output, scenario["name"])
        for k in EVAL_CRITERIA:
            all_scores[k].append(scores[k])

    avg_scores = {k: statistics.mean(v) for k, v in all_scores.items()}
    pass_rate = passes / total

    # Record results
    store.record_evaluation(version_num, avg_scores, pass_rate)

    return {
        "version": version_num,
        "avg_scores": avg_scores,
        "pass_rate": pass_rate,
        "overall": statistics.mean(avg_scores.values()),
    }


# ============================================================
# Demonstration: Version History Workflow
# ============================================================

def demo_version_workflow():
    """
    Simulate a realistic prompt versioning workflow:
    1. Create initial prompt version
    2. Evaluate it
    3. Make improvements, create new versions
    4. Evaluate each version
    5. Compare and detect regressions
    """
    store = PromptVersionStore()

    print("=" * 65)
    print("  PROMPT VERSION TRACKING — CLINICAL Q&A SYSTEM")
    print("=" * 65)

    # --- Version 1: Basic prompt ---
    base_time = datetime.now() - timedelta(days=30)
    store.add_version(
        prompt_text=(
            "You are a medical assistant. Answer clinical questions."
        ),
        author="Dr. Smith",
        change_description="Initial prompt — minimal instructions",
        timestamp=base_time,
    )

    # --- Version 2: Added safety instructions ---
    store.add_version(
        prompt_text=(
            "You are a clinical Q&A assistant. Provide accurate, evidence-based "
            "answers. Always mention safety warnings and drug interactions."
        ),
        author="Dr. Smith",
        change_description="Added safety and evidence-based instructions",
        timestamp=base_time + timedelta(days=7),
    )

    # --- Version 3: Full detailed prompt ---
    store.add_version(
        prompt_text=(
            "You are a clinical Q&A assistant for healthcare professionals. "
            "Provide accurate, evidence-based answers. Always mention relevant "
            "contraindications, drug interactions, and safety warnings. Recommend "
            "specialist consultation when appropriate. Never make definitive diagnoses. "
            "Cite current guidelines where possible. Be concise but thorough."
        ),
        author="Dr. Patel",
        change_description="Comprehensive rewrite with specialist referral and guideline citation",
        timestamp=base_time + timedelta(days=14),
    )

    # --- Version 4: Regression — accidentally simplified ---
    store.add_version(
        prompt_text=(
            "You are a helpful assistant. Answer questions about health topics "
            "in a friendly, conversational tone."
        ),
        author="Intern",
        change_description="Simplified for friendlier tone (experimental)",
        timestamp=base_time + timedelta(days=21),
    )

    # Print version history
    print("\n📋 Version History:\n")
    for v in store.get_all_versions():
        print(f"  v{v['version']}  [{v['timestamp'][:10]}]  by {v['author']}")
        print(f"      {v['change_description']}")
        print(f"      Prompt: \"{v['prompt'][:70]}...\"")
        print()

    # Evaluate all versions
    print("=" * 65)
    print("  EVALUATING ALL VERSIONS")
    print("=" * 65)

    eval_results = []
    for v in store.get_all_versions():
        print(f"\n  Evaluating v{v['version']}...", flush=True)
        result = evaluate_version(store, v["version"])
        eval_results.append(result)

        print(f"    Pass rate: {result['pass_rate']*100:.0f}%")
        for criterion, score in result["avg_scores"].items():
            bar = "█" * int(round(score)) + "░" * (5 - int(round(score)))
            print(f"    {criterion:<15} {bar}  {score:.2f}")
        print(f"    Overall: {result['overall']:.2f}/5.00")

    # ============================================================
    # Regression Detection
    # ============================================================
    print(f"\n{'='*65}")
    print("  REGRESSION ANALYSIS")
    print(f"{'='*65}")

    for i in range(1, len(eval_results)):
        prev = eval_results[i - 1]
        curr = eval_results[i]
        diff = curr["overall"] - prev["overall"]
        pass_diff = curr["pass_rate"] - prev["pass_rate"]

        v_prev = prev["version"]
        v_curr = curr["version"]

        if diff < -0.3 or pass_diff < -0.1:
            print(f"\n  ⚠️  REGRESSION DETECTED: v{v_prev} → v{v_curr}")
            print(f"     Overall score: {prev['overall']:.2f} → {curr['overall']:.2f} ({diff:+.2f})")
            print(f"     Pass rate:     {prev['pass_rate']*100:.0f}% → {curr['pass_rate']*100:.0f}% ({pass_diff*100:+.0f}%)")
            print(f"     Recommendation: Roll back to v{v_prev}")
        elif diff > 0.2:
            print(f"\n  ✅ IMPROVEMENT: v{v_prev} → v{v_curr}")
            print(f"     Overall score: {prev['overall']:.2f} → {curr['overall']:.2f} ({diff:+.2f})")
            print(f"     Pass rate:     {prev['pass_rate']*100:.0f}% → {curr['pass_rate']*100:.0f}% ({pass_diff*100:+.0f}%)")
        else:
            print(f"\n  ── No significant change: v{v_prev} → v{v_curr} ({diff:+.2f})")

    # Summary table
    print(f"\n{'='*65}")
    print("  VERSION SUMMARY TABLE")
    print(f"{'='*65}")
    print(f"\n  {'Version':<10} {'Overall':<10} {'Pass Rate':<12} {'Status'}")
    print(f"  {'-'*42}")

    best_version = max(eval_results, key=lambda r: r["overall"])
    for result in eval_results:
        v = result["version"]
        marker = " ← BEST" if result["version"] == best_version["version"] else ""
        print(f"  v{v:<9} {result['overall']:.2f}      {result['pass_rate']*100:.0f}%         "
              f"{store.get_version(v)['status']}{marker}")

    print(f"\n  Recommended version for deployment: v{best_version['version']}")
    print(f"  (Overall score: {best_version['overall']:.2f}, Pass rate: {best_version['pass_rate']*100:.0f}%)")
    print()


# ============================================================
# Interactive Mode: Add and evaluate your own versions
# ============================================================

def interactive_versioning():
    """Let the user add prompt versions and evaluate them."""
    store = PromptVersionStore()

    # Seed with a baseline version
    store.add_version(
        prompt_text=(
            "You are a clinical Q&A assistant. Provide accurate, evidence-based answers. "
            "Mention safety warnings and drug interactions."
        ),
        author="Baseline",
        change_description="Default baseline prompt",
    )

    print("\n" + "=" * 65)
    print("  INTERACTIVE PROMPT VERSIONING")
    print("=" * 65)
    print("\nCommands:")
    print("  add     — Add a new prompt version")
    print("  eval    — Evaluate a version")
    print("  list    — Show all versions")
    print("  compare — Compare two versions")
    print("  quit    — Exit\n")

    while True:
        cmd = input("Command: ").strip().lower()

        if cmd == "quit":
            break

        elif cmd == "add":
            prompt = input("  Enter system prompt: ").strip()
            author = input("  Author name: ").strip() or "User"
            desc = input("  Change description: ").strip() or "No description"
            v = store.add_version(prompt, author, desc)
            print(f"  → Created v{v['version']}\n")

        elif cmd == "eval":
            v_num = input("  Version number to evaluate: ").strip()
            try:
                v_num = int(v_num)
                print(f"  Evaluating v{v_num}...")
                result = evaluate_version(store, v_num)
                if result:
                    print(f"  Overall: {result['overall']:.2f}, Pass rate: {result['pass_rate']*100:.0f}%")
                    for k, v in result["avg_scores"].items():
                        print(f"    {k}: {v:.2f}")
            except ValueError:
                print("  Invalid version number.")
            print()

        elif cmd == "list":
            for v in store.get_all_versions():
                evals = store.get_evaluations(v["version"])
                eval_info = ""
                if evals:
                    latest = evals[-1]
                    eval_info = f" | score: {statistics.mean(latest['scores'].values()):.2f}, pass: {latest['test_pass_rate']*100:.0f}%"
                print(f"  v{v['version']} [{v['timestamp'][:10]}] by {v['author']}: {v['change_description']}{eval_info}")
            print()

        elif cmd == "compare":
            try:
                v1 = int(input("  First version: ").strip())
                v2 = int(input("  Second version: ").strip())
                e1 = store.get_evaluations(v1)
                e2 = store.get_evaluations(v2)
                if not e1 or not e2:
                    print("  Both versions must be evaluated first. Use 'eval' command.")
                else:
                    s1 = statistics.mean(e1[-1]["scores"].values())
                    s2 = statistics.mean(e2[-1]["scores"].values())
                    print(f"  v{v1}: {s1:.2f}  vs  v{v2}: {s2:.2f}")
                    diff = s2 - s1
                    if diff > 0.3:
                        print(f"  → v{v2} is significantly better ({diff:+.2f})")
                    elif diff < -0.3:
                        print(f"  → v{v1} is significantly better ({-diff:+.2f})")
                    else:
                        print(f"  → No significant difference ({diff:+.2f})")
            except ValueError:
                print("  Invalid version numbers.")
            print()

        else:
            print("  Unknown command. Try: add, eval, list, compare, quit\n")


# ============================================================
# Main
# ============================================================

def main():
    print("🏥 Exercise 3: Prompt Version Tracking\n")
    print("Choose mode:")
    print("  1. Automated demo (simulated version history)")
    print("  2. Interactive mode (add and evaluate your own)")
    print()

    choice = input("Choice (1 or 2): ").strip()

    if choice == "2":
        interactive_versioning()
    else:
        demo_version_workflow()


if __name__ == "__main__":
    main()
