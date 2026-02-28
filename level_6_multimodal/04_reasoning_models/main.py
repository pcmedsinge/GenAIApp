"""
Project 04: Reasoning Models — Main Demo
=========================================
Explores OpenAI's reasoning models (o1-mini, o3-mini) for complex clinical
reasoning tasks. Compares with standard models on accuracy, cost, and latency.

Demos:
  1. Reasoning vs Standard — GPT-4o vs o1-mini comparison
  2. Extended Thinking — multi-step differential diagnosis
  3. Cost/Latency Analysis — token cost and timing comparison
  4. When to Use Reasoning — decision framework with examples
"""

import json
import os
import time
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# Model configuration
STANDARD_MODEL = "gpt-4o"
REASONING_MODEL = "o1-mini"  # Use o1-mini for cost-effective reasoning


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def call_standard_model(messages: list, model: str = STANDARD_MODEL,
                        max_tokens: int = 1500) -> dict:
    """Call a standard model and return response with metadata."""
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
    )
    elapsed = time.time() - start
    return {
        "model": model,
        "content": response.choices[0].message.content,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "elapsed_seconds": elapsed,
    }


def call_reasoning_model(user_message: str, model: str = REASONING_MODEL,
                         max_completion_tokens: int = 4000,
                         reasoning_effort: str = "medium") -> dict:
    """
    Call a reasoning model.

    Key differences from standard models:
    - No system message (use 'user' or 'developer' role instead)
    - Use max_completion_tokens instead of max_tokens
    - reasoning_effort parameter: low, medium, high
    - No temperature parameter
    """
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            # NOTE: o1 models do NOT support system messages.
            # Use 'user' role for instructions + content combined.
            {"role": "user", "content": user_message},
        ],
        max_completion_tokens=max_completion_tokens,
        # reasoning_effort=reasoning_effort,  # Uncomment if supported by your API version
    )
    elapsed = time.time() - start

    usage = response.usage
    result = {
        "model": model,
        "content": response.choices[0].message.content,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "elapsed_seconds": elapsed,
    }
    # Reasoning tokens may be reported separately
    if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
        details = usage.completion_tokens_details
        if hasattr(details, "reasoning_tokens"):
            result["reasoning_tokens"] = details.reasoning_tokens
    return result


def print_result(result: dict, label: str = ""):
    """Pretty-print a model result."""
    if label:
        print(f"\n--- {label} ---")
    print(f"  Model: {result['model']}")
    print(f"  Time:  {result['elapsed_seconds']:.2f}s")
    print(f"  Tokens: {result['total_tokens']} "
          f"(prompt={result['prompt_tokens']}, completion={result['completion_tokens']})")
    if "reasoning_tokens" in result:
        print(f"  Reasoning tokens: {result['reasoning_tokens']}")
    print(f"\n  Response:\n")
    for line in result["content"].split("\n"):
        print(f"    {line}")


# ============================================================================
# DEMO 1: REASONING VS STANDARD
# ============================================================================

def demo_1_reasoning_vs_standard():
    """
    Compare GPT-4o vs o1-mini on a complex medical question.
    Shows different prompting: no system message for o1, different parameters.
    """
    print("\n" + "=" * 70)
    print("DEMO 1: Reasoning vs Standard — GPT-4o vs o1-mini")
    print("=" * 70)

    complex_question = """
    A 67-year-old male with a history of type 2 diabetes, chronic kidney disease
    (stage 3b, eGFR 38), heart failure with reduced ejection fraction (EF 30%),
    and atrial fibrillation presents with an acute gout flare in his left first
    metatarsophalangeal joint.

    Current medications: Metformin 500mg BID, Carvedilol 25mg BID, Lisinopril 10mg daily,
    Apixaban 5mg BID, Furosemide 40mg daily.

    What is the optimal treatment approach for the acute gout flare, considering
    all comorbidities and current medications? Explain your reasoning step by step,
    including which common treatments are contraindicated and why.
    """

    print(f"\n--- Complex Clinical Question ---\n{complex_question.strip()}")

    # Standard model (with system message)
    print("\n" + "-" * 50)
    print("Calling STANDARD model (GPT-4o)...")
    standard_messages = [
        {
            "role": "system",
            "content": (
                "You are a clinical pharmacology expert. Provide detailed, "
                "evidence-based treatment recommendations. Consider all drug "
                "interactions and contraindications."
            ),
        },
        {"role": "user", "content": complex_question},
    ]
    standard_result = call_standard_model(standard_messages)
    print_result(standard_result, "Standard Model (GPT-4o)")

    # Reasoning model (no system message, different params)
    print("\n" + "-" * 50)
    print("Calling REASONING model (o1-mini)...")
    # For o1, we combine the instruction + question into a single user message
    reasoning_prompt = (
        "You are a clinical pharmacology expert. Provide detailed, evidence-based "
        "treatment recommendations. Consider all drug interactions and contraindications.\n\n"
        f"{complex_question}"
    )
    reasoning_result = call_reasoning_model(reasoning_prompt)
    print_result(reasoning_result, "Reasoning Model (o1-mini)")

    # Comparison summary
    print("\n" + "-" * 50)
    print("--- Comparison Summary ---")
    print(f"  {'Metric':<20} {'GPT-4o':<20} {'o1-mini':<20}")
    print(f"  {'-'*20} {'-'*20} {'-'*20}")
    print(f"  {'Time':<20} {standard_result['elapsed_seconds']:.2f}s{'':<14} "
          f"{reasoning_result['elapsed_seconds']:.2f}s")
    print(f"  {'Total tokens':<20} {standard_result['total_tokens']:<20} "
          f"{reasoning_result['total_tokens']:<20}")
    print(f"  {'Response length':<20} {len(standard_result['content'])} chars{'':<6} "
          f"{len(reasoning_result['content'])} chars")


# ============================================================================
# DEMO 2: EXTENDED THINKING
# ============================================================================

def demo_2_extended_thinking():
    """
    Demonstrate how reasoning models work through multi-step problems.
    Shows a complex differential diagnosis case.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Extended Thinking — Multi-Step Differential Diagnosis")
    print("=" * 70)

    case_presentation = """
    CLINICAL CASE FOR DIFFERENTIAL DIAGNOSIS:

    A 34-year-old woman presents with a 3-month history of:
    - Progressive fatigue and weakness
    - Unintentional weight loss of 15 lbs
    - Intermittent low-grade fevers (99.5-100.4°F)
    - Joint pain in hands and wrists (bilateral, symmetric)
    - New butterfly-shaped rash across cheeks and nose bridge
    - Oral ulcers (painless)
    - Hair thinning
    - Raynaud's phenomenon (fingers turning white/blue in cold)

    Lab Results:
    - CBC: WBC 3.2 (low), Hgb 10.8 (low), Platelets 130 (low-normal)
    - ESR: 55 mm/hr (elevated)
    - CRP: 2.8 mg/dL (elevated)
    - ANA: Positive (1:640, homogeneous pattern)
    - Anti-dsDNA: Positive
    - C3: 65 mg/dL (low), C4: 8 mg/dL (low)
    - Urinalysis: Protein 2+, RBC 10-15/hpf
    - Serum creatinine: 1.3 mg/dL (slightly elevated)
    - Anti-Smith antibody: Positive

    Family History: Mother has rheumatoid arthritis, maternal aunt has Hashimoto's.

    Please provide:
    1. A ranked differential diagnosis with confidence percentages
    2. Step-by-step reasoning for each diagnosis considered
    3. The most critical next diagnostic steps
    4. Urgency assessment
    """

    print(f"\n--- Case Presentation ---\n{case_presentation.strip()}")

    print("\n--- Reasoning Model Working Through the Case ---")
    result = call_reasoning_model(
        user_message=case_presentation,
        max_completion_tokens=5000,
        reasoning_effort="high",
    )
    print_result(result, "Reasoning Model Analysis")

    if "reasoning_tokens" in result:
        print(f"\n  💡 The model used {result['reasoning_tokens']} reasoning tokens")
        print(f"     (internal chain-of-thought, not visible in output)")


# ============================================================================
# DEMO 3: COST/LATENCY ANALYSIS
# ============================================================================

def demo_3_cost_latency_analysis():
    """
    Compare token costs and latency between reasoning and standard models.
    Uses multiple queries of varying complexity.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Cost/Latency Analysis — Standard vs Reasoning")
    print("=" * 70)

    test_queries = [
        {
            "name": "Simple Lookup",
            "query": "What is the standard adult dose of amoxicillin for strep throat?",
        },
        {
            "name": "Moderate Complexity",
            "query": (
                "A patient on warfarin needs an antibiotic for a UTI. "
                "What are the safest options and what monitoring is needed?"
            ),
        },
        {
            "name": "High Complexity",
            "query": (
                "A 55-year-old with CKD stage 4, heart failure, liver cirrhosis "
                "(Child-Pugh B), and a history of GI bleeding needs pain management "
                "for severe osteoarthritis. NSAIDs, acetaminophen high-dose, and "
                "opioids each have significant risks. What is the optimal approach?"
            ),
        },
    ]

    # Approximate costs (per 1M tokens) — update with current pricing
    COST_PER_1M = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "o1-mini": {"input": 3.00, "output": 12.00},
    }

    results = []
    for query_info in test_queries:
        print(f"\n--- Testing: {query_info['name']} ---")
        print(f"  Query: {query_info['query'][:80]}...")

        # Standard model
        standard_msgs = [
            {"role": "system", "content": "You are a clinical expert. Be concise."},
            {"role": "user", "content": query_info["query"]},
        ]
        std_result = call_standard_model(standard_msgs, max_tokens=800)

        # Reasoning model
        reasoning_prompt = f"You are a clinical expert. Be concise.\n\n{query_info['query']}"
        rsn_result = call_reasoning_model(reasoning_prompt, max_completion_tokens=2000)

        results.append({
            "name": query_info["name"],
            "standard": std_result,
            "reasoning": rsn_result,
        })

        # Estimate costs
        std_cost = (
            std_result["prompt_tokens"] * COST_PER_1M[STANDARD_MODEL]["input"] / 1_000_000
            + std_result["completion_tokens"] * COST_PER_1M[STANDARD_MODEL]["output"] / 1_000_000
        )
        rsn_cost = (
            rsn_result["prompt_tokens"] * COST_PER_1M[REASONING_MODEL]["input"] / 1_000_000
            + rsn_result["completion_tokens"] * COST_PER_1M[REASONING_MODEL]["output"] / 1_000_000
        )

        print(f"  {'Metric':<18} {'GPT-4o':<18} {'o1-mini':<18}")
        print(f"  {'-'*18} {'-'*18} {'-'*18}")
        print(f"  {'Latency':<18} {std_result['elapsed_seconds']:.2f}s{'':<13} "
              f"{rsn_result['elapsed_seconds']:.2f}s")
        print(f"  {'Total tokens':<18} {std_result['total_tokens']:<18} "
              f"{rsn_result['total_tokens']:<18}")
        print(f"  {'Est. cost':<18} ${std_cost:.6f}{'':<11} ${rsn_cost:.6f}")

    # Summary
    print("\n" + "-" * 60)
    print("--- Overall Summary ---")
    total_std_time = sum(r["standard"]["elapsed_seconds"] for r in results)
    total_rsn_time = sum(r["reasoning"]["elapsed_seconds"] for r in results)
    total_std_tokens = sum(r["standard"]["total_tokens"] for r in results)
    total_rsn_tokens = sum(r["reasoning"]["total_tokens"] for r in results)
    print(f"  Total Standard Time:   {total_std_time:.2f}s  |  Tokens: {total_std_tokens}")
    print(f"  Total Reasoning Time:  {total_rsn_time:.2f}s  |  Tokens: {total_rsn_tokens}")
    if total_std_time > 0:
        print(f"  Reasoning overhead:    {total_rsn_time / total_std_time:.1f}x slower")
    print(f"\n  💡 Reasoning models are best reserved for complex, multi-step problems")
    print(f"     where accuracy is more important than speed/cost.")


# ============================================================================
# DEMO 4: WHEN TO USE REASONING
# ============================================================================

def demo_4_when_to_use_reasoning():
    """
    Decision framework: simple lookup → standard, complex logic → reasoning.
    Classifies queries and routes to appropriate model.
    """
    print("\n" + "=" * 70)
    print("DEMO 4: When to Use Reasoning — Decision Framework")
    print("=" * 70)

    # Decision framework
    framework = """
    MODEL SELECTION FRAMEWORK:

    ┌─────────────────────────────────────────────────────────────────┐
    │  Query Type              │  Recommended Model  │  Reason       │
    ├─────────────────────────────────────────────────────────────────┤
    │  Simple lookup/fact      │  gpt-4o-mini        │  Fast, cheap  │
    │  Formatting/summarize    │  gpt-4o-mini        │  Routine task │
    │  Classification          │  gpt-4o             │  Good balance │
    │  Multi-step reasoning    │  o1-mini            │  Needs CoT    │
    │  Complex interactions    │  o1-mini / o1       │  Safety-crit  │
    │  Differential diagnosis  │  o1-mini            │  Multi-factor │
    └─────────────────────────────────────────────────────────────────┘
    """
    print(framework)

    # Test queries across the spectrum
    test_cases = [
        {
            "query": "What is the generic name for Lipitor?",
            "expected_model": "gpt-4o-mini",
            "category": "Simple Lookup",
        },
        {
            "query": "Summarize the mechanism of action of metformin in 2 sentences.",
            "expected_model": "gpt-4o-mini",
            "category": "Summarization",
        },
        {
            "query": "Classify the following symptom as neurological, cardiac, or musculoskeletal: sudden onset chest pain radiating to the left arm with diaphoresis.",
            "expected_model": "gpt-4o",
            "category": "Classification",
        },
        {
            "query": (
                "A patient on methotrexate, prednisone, and a TNF-inhibitor develops "
                "a new fever and cough. Consider immunosuppression levels, infection risks, "
                "and the differential between opportunistic infection, drug reaction, and "
                "disease flare. What is your assessment?"
            ),
            "expected_model": "o1-mini",
            "category": "Complex Reasoning",
        },
    ]

    for case in test_cases:
        print(f"\n{'─' * 60}")
        print(f"  Category: {case['category']}")
        print(f"  Query:    {case['query'][:80]}...")
        print(f"  Optimal Model: {case['expected_model']}")

        # Route to the appropriate model
        model = case["expected_model"]
        print(f"  Routing to: {model}")

        if model in ("gpt-4o-mini", "gpt-4o"):
            messages = [
                {"role": "system", "content": "You are a clinical expert. Be concise."},
                {"role": "user", "content": case["query"]},
            ]
            result = call_standard_model(messages, model=model, max_tokens=500)
        else:
            prompt = f"You are a clinical expert. Be concise.\n\n{case['query']}"
            result = call_reasoning_model(prompt, max_completion_tokens=2000)

        print(f"  Time: {result['elapsed_seconds']:.2f}s | Tokens: {result['total_tokens']}")
        # Show first 200 chars of response
        preview = result["content"][:200].replace("\n", " ")
        print(f"  Preview: {preview}...")

    print(f"\n{'─' * 60}")
    print("  KEY TAKEAWAY: Match model capability to query complexity.")
    print("  Don't use o1 for simple lookups — don't use gpt-4o-mini for")
    print("  complex multi-step reasoning with safety implications.")


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    """Interactive menu for reasoning model demos."""
    demos = {
        "1": ("Reasoning vs Standard (GPT-4o vs o1-mini)", demo_1_reasoning_vs_standard),
        "2": ("Extended Thinking (Differential Diagnosis)", demo_2_extended_thinking),
        "3": ("Cost/Latency Analysis", demo_3_cost_latency_analysis),
        "4": ("When to Use Reasoning (Decision Framework)", demo_4_when_to_use_reasoning),
    }

    while True:
        print("\n" + "=" * 70)
        print("PROJECT 04: REASONING MODELS — DEMO MENU")
        print("=" * 70)
        for key, (desc, _) in demos.items():
            print(f"  [{key}] {desc}")
        print(f"  [a] Run all demos")
        print(f"  [q] Quit")

        choice = input("\nSelect demo: ").strip().lower()

        if choice == "q":
            print("Goodbye!")
            break
        elif choice == "a":
            for key in sorted(demos.keys()):
                demos[key][1]()
        elif choice in demos:
            demos[choice][1]()
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
