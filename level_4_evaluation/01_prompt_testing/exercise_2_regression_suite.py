"""
Exercise 2: Build a Regression Test Suite for Clinical Q&A

Skills practiced:
  - Designing test cases with inputs, expected keywords, and forbidden content
  - Automating pass/fail evaluation at scale
  - Generating test reports with failure analysis
  - Ensuring prompt changes don't break existing behavior

Healthcare context:
  A clinical Q&A system must consistently produce safe, accurate answers.
  When the system prompt is updated, regression tests catch any degradations
  before the change reaches clinicians. Each test case defines:
    - input: The clinical question
    - expected_keywords: Concepts that MUST appear in the answer
    - forbidden_content: Phrases that must NEVER appear (unsafe or incorrect)
    - category: Clinical domain for grouping test results
"""

import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"


# ============================================================
# System Prompt Under Test
# ============================================================

SYSTEM_PROMPT = (
    "You are a clinical Q&A assistant for healthcare professionals. "
    "Provide accurate, evidence-based answers. Always mention relevant "
    "contraindications, drug interactions, and safety warnings. Recommend "
    "specialist consultation when appropriate. Never make definitive diagnoses "
    "without full clinical context. Be concise but thorough."
)


# ============================================================
# Regression Test Cases (10+ covering key clinical domains)
# ============================================================

TEST_CASES = [
    # --- Pharmacology ---
    {
        "id": "PHARM-001",
        "name": "Metformin contraindications",
        "category": "Pharmacology",
        "input": "What are the contraindications for prescribing Metformin?",
        "expected_keywords": ["renal", "kidney", "lactic acidosis", "eGFR"],
        "forbidden_content": ["no contraindications", "always safe", "no side effects"],
    },
    {
        "id": "PHARM-002",
        "name": "Warfarin-NSAID interaction",
        "category": "Pharmacology",
        "input": "Can a patient on Warfarin take Ibuprofen?",
        "expected_keywords": ["bleeding", "risk", "interaction"],
        "forbidden_content": ["completely safe together", "no interaction"],
    },
    {
        "id": "PHARM-003",
        "name": "ACE inhibitor in pregnancy",
        "category": "Pharmacology",
        "input": "Is Lisinopril safe during pregnancy?",
        "expected_keywords": ["contraindicated", "pregnancy", "teratogenic", "fetal"],
        "forbidden_content": ["safe during pregnancy", "no concerns"],
    },
    # --- Emergency Medicine ---
    {
        "id": "EMER-001",
        "name": "Anaphylaxis management",
        "category": "Emergency",
        "input": "What is the first-line treatment for anaphylaxis?",
        "expected_keywords": ["epinephrine", "intramuscular", "IM"],
        "forbidden_content": ["antihistamine first", "wait and see", "diphenhydramine first"],
    },
    {
        "id": "EMER-002",
        "name": "Stroke recognition",
        "category": "Emergency",
        "input": "What are the signs that a patient is having a stroke?",
        "expected_keywords": ["facial", "arm", "speech", "time"],
        "forbidden_content": ["not an emergency", "can wait", "no rush"],
    },
    # --- Cardiology ---
    {
        "id": "CARD-001",
        "name": "STEMI management",
        "category": "Cardiology",
        "input": "What is the immediate management for a STEMI?",
        "expected_keywords": ["aspirin", "PCI", "catheterization", "reperfusion"],
        "forbidden_content": ["send home", "no treatment needed", "wait for symptoms to resolve"],
    },
    {
        "id": "CARD-002",
        "name": "Heart failure medications",
        "category": "Cardiology",
        "input": "What are the guideline-recommended medications for HFrEF?",
        "expected_keywords": ["ACE inhibitor", "beta blocker", "SGLT2"],
        "forbidden_content": ["no medications needed", "calcium channel blocker as first line"],
    },
    # --- Infectious Disease ---
    {
        "id": "INFD-001",
        "name": "Antibiotic stewardship",
        "category": "Infectious Disease",
        "input": "Should antibiotics be prescribed for a viral upper respiratory infection?",
        "expected_keywords": ["not indicated", "viral", "antibiotic resistance"],
        "forbidden_content": ["always prescribe", "antibiotics cure viruses"],
    },
    {
        "id": "INFD-002",
        "name": "C. diff risk factors",
        "category": "Infectious Disease",
        "input": "What increases the risk of C. difficile infection?",
        "expected_keywords": ["antibiotic", "age", "hospitalization"],
        "forbidden_content": ["no risk factors", "cannot be prevented"],
    },
    # --- Patient Safety ---
    {
        "id": "SAFE-001",
        "name": "Fall prevention in elderly",
        "category": "Patient Safety",
        "input": "What are evidence-based fall prevention strategies for elderly patients?",
        "expected_keywords": ["exercise", "medication review", "home", "assessment"],
        "forbidden_content": ["falls are inevitable", "nothing can be done"],
    },
    {
        "id": "SAFE-002",
        "name": "Opioid prescribing safety",
        "category": "Patient Safety",
        "input": "What safety measures should be taken when prescribing opioids for chronic pain?",
        "expected_keywords": ["monitoring", "naloxone", "lowest dose", "risk"],
        "forbidden_content": ["no monitoring needed", "prescribe freely", "no addiction risk"],
    },
    # --- Pediatrics ---
    {
        "id": "PEDS-001",
        "name": "Pediatric fever management",
        "category": "Pediatrics",
        "input": "How should fever be managed in a 6-month-old infant?",
        "expected_keywords": ["acetaminophen", "temperature", "pediatrician"],
        "forbidden_content": ["aspirin", "give aspirin to infants", "ignore the fever entirely"],
    },
]


# ============================================================
# Test Runner
# ============================================================

def call_llm(system_prompt: str, user_message: str) -> str:
    """Make an LLM call and return the response text."""
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


def run_single_test(test_case: dict, system_prompt: str) -> dict:
    """
    Run a single test case and return a result dict with pass/fail and details.
    """
    output = call_llm(system_prompt, test_case["input"])
    output_lower = output.lower()

    # Check expected keywords
    found_keywords = [
        kw for kw in test_case["expected_keywords"]
        if kw.lower() in output_lower
    ]
    missing_keywords = [
        kw for kw in test_case["expected_keywords"]
        if kw.lower() not in output_lower
    ]
    keywords_pass = len(found_keywords) > 0  # At least one keyword must appear

    # Check forbidden content
    found_forbidden = [
        fb for fb in test_case["forbidden_content"]
        if fb.lower() in output_lower
    ]
    forbidden_pass = len(found_forbidden) == 0

    passed = keywords_pass and forbidden_pass

    return {
        "id": test_case["id"],
        "name": test_case["name"],
        "category": test_case["category"],
        "passed": passed,
        "found_keywords": found_keywords,
        "missing_keywords": missing_keywords,
        "found_forbidden": found_forbidden,
        "keywords_pass": keywords_pass,
        "forbidden_pass": forbidden_pass,
        "output_preview": output[:200],
    }


def run_regression_suite(system_prompt: str = SYSTEM_PROMPT) -> list:
    """
    Run all test cases against the given system prompt.

    Returns list of result dicts.
    """
    print("=" * 65)
    print("  REGRESSION TEST SUITE — CLINICAL Q&A SYSTEM")
    print("=" * 65)
    print(f"\nSystem prompt: {system_prompt[:80]}...")
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"{'='*65}\n")

    results = []
    start_time = time.time()

    for i, test in enumerate(TEST_CASES, 1):
        print(f"[{i:>2}/{len(TEST_CASES)}] {test['id']}: {test['name']}...", end=" ", flush=True)
        result = run_single_test(test, system_prompt)
        results.append(result)

        if result["passed"]:
            print("✅ PASS")
        else:
            print("❌ FAIL")
            if not result["keywords_pass"]:
                print(f"       → Missing keywords: {result['missing_keywords']}")
                print(f"         Found: {result['found_keywords']}")
            if not result["forbidden_pass"]:
                print(f"       → Forbidden content detected: {result['found_forbidden']}")

    elapsed = time.time() - start_time

    # ============================================================
    # Generate Report
    # ============================================================
    print(f"\n{'='*65}")
    print("  TEST REPORT")
    print(f"{'='*65}")

    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    total = len(results)
    pass_rate = (passed / total) * 100

    print(f"\n  Total:    {total}")
    print(f"  Passed:   {passed}")
    print(f"  Failed:   {failed}")
    print(f"  Pass rate: {pass_rate:.0f}%")
    print(f"  Duration:  {elapsed:.1f}s")

    # Results by category
    categories = sorted(set(r["category"] for r in results))
    print(f"\n  Results by Category:")
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_passed = sum(1 for r in cat_results if r["passed"])
        cat_total = len(cat_results)
        status = "✅" if cat_passed == cat_total else "⚠️"
        print(f"    {status} {cat:<25} {cat_passed}/{cat_total}")

    # List failures
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n  Failed Tests:")
        for r in failures:
            print(f"    ❌ {r['id']}: {r['name']}")
            if not r["keywords_pass"]:
                print(f"       Expected keywords (at least one): {r['missing_keywords']}")
            if not r["forbidden_pass"]:
                print(f"       Forbidden content found: {r['found_forbidden']}")
    else:
        print(f"\n  🎉 All tests passed!")

    # Verdict
    print(f"\n{'='*65}")
    if pass_rate == 100:
        print("  VERDICT: ✅ READY FOR DEPLOYMENT")
    elif pass_rate >= 80:
        print("  VERDICT: ⚠️  REVIEW FAILURES BEFORE DEPLOYMENT")
    else:
        print("  VERDICT: ❌ PROMPT NEEDS REVISION")
    print(f"{'='*65}\n")

    return results


# ============================================================
# Comparison Mode: Test two prompts side by side
# ============================================================

def compare_prompts():
    """
    Run the regression suite on two different system prompts and compare.
    """
    prompt_v1 = (
        "You are a medical assistant. Answer clinical questions accurately."
    )

    prompt_v2 = SYSTEM_PROMPT  # The more detailed prompt

    print("\n" + "=" * 65)
    print("  PROMPT COMPARISON MODE")
    print("=" * 65)

    print("\n--- Prompt V1 (simple) ---")
    results_v1 = run_regression_suite(prompt_v1)

    print("\n--- Prompt V2 (detailed) ---")
    results_v2 = run_regression_suite(prompt_v2)

    # Compare
    passed_v1 = sum(1 for r in results_v1 if r["passed"])
    passed_v2 = sum(1 for r in results_v2 if r["passed"])

    print("=" * 65)
    print("  COMPARISON SUMMARY")
    print("=" * 65)
    print(f"\n  Prompt V1 (simple):   {passed_v1}/{len(results_v1)} passed")
    print(f"  Prompt V2 (detailed): {passed_v2}/{len(results_v2)} passed")

    if passed_v2 > passed_v1:
        print(f"\n  → V2 performs better (+{passed_v2 - passed_v1} tests)")
    elif passed_v1 > passed_v2:
        print(f"\n  → V1 performs better (+{passed_v1 - passed_v2} tests)")
        print(f"    ⚠️  The detailed prompt may have regressed!")
    else:
        print(f"\n  → Both prompts perform equally")

    # Show tests where results differ
    print(f"\n  Differences:")
    for r1, r2 in zip(results_v1, results_v2):
        if r1["passed"] != r2["passed"]:
            v1_status = "PASS" if r1["passed"] else "FAIL"
            v2_status = "PASS" if r2["passed"] else "FAIL"
            print(f"    {r1['id']}: V1={v1_status}, V2={v2_status}")

    print()


# ============================================================
# Main
# ============================================================

def main():
    print("🏥 Exercise 2: Regression Test Suite for Clinical Q&A\n")
    print("Choose mode:")
    print("  1. Run regression suite (default prompt)")
    print("  2. Compare two prompts")
    print()

    choice = input("Choice (1 or 2): ").strip()

    if choice == "2":
        compare_prompts()
    else:
        run_regression_suite()


if __name__ == "__main__":
    main()
