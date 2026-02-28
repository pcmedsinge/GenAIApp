"""
Exercise 3: Hallucination Checker
===================================

Skills practiced:
- Claim extraction from generated text
- Fact verification against source documents
- Faithfulness scoring (0-1 scale)
- Sentence-level hallucination flagging

Healthcare context:
Hallucinations in healthcare AI are dangerous. A fabricated drug interaction,
an invented dosage, or a fictitious clinical guideline can lead to patient harm.
This exercise builds a hallucination checker that:
1. Extracts individual claims from an LLM-generated answer
2. Verifies each claim against the source context
3. Computes a faithfulness score
4. Flags specific hallucinated sentences with severity levels

In a production RAG system, this would run on every response before it reaches
the clinician, with low-faithfulness answers triggering a fallback response.

Usage:
    python exercise_3_hallucination_checker.py
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

def extract_claims(text: str) -> list[dict]:
    """Extract individual factual claims from a text passage."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Extract all factual claims from the given text. 
Each claim should be a single, verifiable statement. Return JSON:
{
    "claims": [
        {
            "claim_id": 1,
            "text": "the factual claim",
            "type": "factual" | "numerical" | "causal" | "recommendation"
        }
    ]
}

Rules:
- Split compound sentences into individual claims
- Include numerical values (doses, percentages, counts) as separate claims
- Include causal statements ("X causes Y") as claims
- Include recommendations ("should be treated with X") as claims
- Exclude opinions, hedging language, and disclaimers"""
            },
            {"role": "user", "content": f"Extract claims from:\n\n{text}"}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    result = json.loads(response.choices[0].message.content)
    return result.get("claims", [])


# ---------------------------------------------------------------------------
# Claim verification
# ---------------------------------------------------------------------------

def verify_claim(claim: str, source_context: str) -> dict:
    """Verify a single claim against source context."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a fact-checker. Determine if the given claim is 
supported by the source context. Return JSON:
{
    "verdict": "supported" | "not_supported" | "contradicted",
    "confidence": 0.0-1.0,
    "evidence": "quote or paraphrase from source that supports/contradicts, or 'no evidence found'",
    "explanation": "brief reasoning"
}

Rules:
- "supported": The claim is directly stated or clearly implied by the source
- "not_supported": The claim cannot be verified from the source (may be true, but not in source)
- "contradicted": The source explicitly states something different
- Be STRICT: if a specific number, name, or date is in the claim but not in the source, it's "not_supported"
- Paraphrasing is OK — the meaning must match, not exact words"""
            },
            {
                "role": "user",
                "content": f"CLAIM: {claim}\n\nSOURCE CONTEXT:\n{source_context}"
            }
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Full hallucination check
# ---------------------------------------------------------------------------

def check_hallucinations(source_context: str, generated_answer: str) -> dict:
    """
    Full hallucination check: extract claims, verify each, compute score.
    Returns detailed analysis with faithfulness score.
    """
    # Step 1: Extract claims
    claims = extract_claims(generated_answer)
    if not claims:
        return {
            "faithfulness_score": 1.0,
            "total_claims": 0,
            "supported": 0,
            "not_supported": 0,
            "contradicted": 0,
            "details": [],
            "verdict": "no_claims_found",
        }

    # Step 2: Verify each claim
    details = []
    supported_count = 0
    not_supported_count = 0
    contradicted_count = 0

    for claim in claims:
        verification = verify_claim(claim["text"], source_context)
        verdict = verification.get("verdict", "not_supported")

        if verdict == "supported":
            supported_count += 1
            severity = "none"
        elif verdict == "contradicted":
            contradicted_count += 1
            severity = "critical"
        else:
            not_supported_count += 1
            # Determine severity based on claim type
            claim_type = claim.get("type", "factual")
            if claim_type == "numerical":
                severity = "critical"  # Wrong numbers in healthcare are dangerous
            elif claim_type == "recommendation":
                severity = "major"
            else:
                severity = "minor"

        details.append({
            "claim_id": claim.get("claim_id"),
            "claim": claim["text"],
            "claim_type": claim.get("type", "factual"),
            "verdict": verdict,
            "severity": severity,
            "confidence": verification.get("confidence", 0),
            "evidence": verification.get("evidence", ""),
            "explanation": verification.get("explanation", ""),
        })

    # Step 3: Compute faithfulness score
    total = len(claims)
    # Contradicted claims count double against faithfulness
    faithfulness = max(0.0, (supported_count - contradicted_count) / total) if total > 0 else 1.0
    faithfulness = min(1.0, faithfulness)

    # Overall verdict
    if faithfulness >= 0.9:
        overall_verdict = "faithful"
    elif faithfulness >= 0.5:
        overall_verdict = "partially_faithful"
    else:
        overall_verdict = "hallucinated"

    return {
        "faithfulness_score": round(faithfulness, 3),
        "total_claims": total,
        "supported": supported_count,
        "not_supported": not_supported_count,
        "contradicted": contradicted_count,
        "details": details,
        "verdict": overall_verdict,
    }


def print_hallucination_report(result: dict, label: str = ""):
    """Print a formatted hallucination check report."""
    print(f"\n{'─' * 60}")
    if label:
        print(f"  {label}")
        print(f"{'─' * 60}")

    score = result["faithfulness_score"]
    bar_length = int(score * 20)
    bar = "█" * bar_length + "░" * (20 - bar_length)

    print(f"  Faithfulness: [{bar}] {score:.1%}")
    print(f"  Verdict: {result['verdict'].upper()}")
    print(f"  Claims: {result['total_claims']} total — "
          f"{result['supported']} supported, "
          f"{result['not_supported']} unsupported, "
          f"{result['contradicted']} contradicted")

    if result["details"]:
        print(f"\n  Claim-level analysis:")
        for d in result["details"]:
            icon = {"supported": "✅", "not_supported": "⚠️ ", "contradicted": "🔴"}.get(
                d["verdict"], "?")
            print(f"    {icon} [{d['verdict'].upper()}] (severity={d['severity']})")
            print(f"       Claim: \"{d['claim'][:80]}{'...' if len(d['claim']) > 80 else ''}\"")
            if d["verdict"] != "supported":
                print(f"       Evidence: {d['evidence'][:80]}{'...' if len(d.get('evidence', '')) > 80 else ''}")
                print(f"       Explanation: {d['explanation'][:80]}")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "label": "Faithful answer about metformin",
        "source": (
            "Metformin hydrochloride is a biguanide antidiabetic agent. It is indicated for "
            "the treatment of type 2 diabetes mellitus. Metformin decreases hepatic glucose "
            "production, decreases intestinal absorption of glucose, and improves insulin "
            "sensitivity. The recommended starting dose is 500 mg orally twice daily or "
            "850 mg once daily with meals. Maximum recommended daily dose is 2550 mg. "
            "Common adverse reactions include diarrhea, nausea, vomiting, flatulence, "
            "and abdominal discomfort. Metformin is contraindicated in patients with "
            "severe renal impairment (eGFR below 30 mL/min/1.73 m²)."
        ),
        "answer": (
            "Metformin is a biguanide medication for type 2 diabetes. It works by reducing "
            "liver glucose production and improving how the body responds to insulin. The "
            "typical starting dose is 500 mg twice a day with meals. Side effects commonly "
            "include nausea, diarrhea, and stomach discomfort. It should not be used in "
            "patients with severe kidney problems."
        ),
    },
    {
        "label": "Hallucinated answer about metformin",
        "source": (
            "Metformin hydrochloride is a biguanide antidiabetic agent. It is indicated for "
            "the treatment of type 2 diabetes mellitus. The recommended starting dose is "
            "500 mg orally twice daily. Common adverse reactions include diarrhea and nausea."
        ),
        "answer": (
            "Metformin is used for type 2 diabetes. The starting dose is 500 mg twice daily. "
            "Clinical trials have shown it reduces cardiovascular mortality by 39% compared to "
            "sulfonylureas. It is also FDA-approved for prediabetes prevention and PCOS. "
            "The drug achieves peak plasma concentration in 2.5 hours and has a half-life "
            "of 6.2 hours. Side effects include diarrhea, nausea, and vitamin B12 deficiency "
            "in 30% of long-term users."
        ),
    },
    {
        "label": "Mixed — warfarin information",
        "source": (
            "Warfarin is an anticoagulant used to prevent blood clots. It works by inhibiting "
            "vitamin K-dependent clotting factors (II, VII, IX, and X). The dose is adjusted "
            "based on INR monitoring, with a target INR of 2.0-3.0 for most indications. "
            "Major drug interactions include aspirin, NSAIDs, and many antibiotics. Warfarin "
            "is contraindicated in pregnancy (FDA Category X). Common side effects include "
            "bleeding and bruising."
        ),
        "answer": (
            "Warfarin prevents blood clots by blocking vitamin K-dependent clotting factors "
            "II, VII, IX, and X. INR should be maintained between 2.0 and 3.0. It interacts "
            "with aspirin and NSAIDs. Warfarin is unsafe during pregnancy. Side effects include "
            "bleeding (occurs in about 8% of patients annually). Genetic testing for CYP2C9 "
            "and VKORC1 polymorphisms is recommended before starting warfarin to determine "
            "optimal dosing. The drug has a half-life of 40 hours."
        ),
    },
]


def main():
    """Run the hallucination checker on test cases."""
    print("=" * 70)
    print("  Exercise 3: Hallucination Checker")
    print("  Claim extraction + verification + faithfulness scoring")
    print("=" * 70)

    all_results = []

    for case in TEST_CASES:
        print(f"\n{'=' * 60}")
        print(f"  Case: {case['label']}")
        print(f"{'=' * 60}")
        print(f"\n  Source: {case['source'][:100]}...")
        print(f"  Answer: {case['answer'][:100]}...")

        result = check_hallucinations(case["source"], case["answer"])
        print_hallucination_report(result, case["label"])
        all_results.append(result)

    # Summary
    print(f"\n{'=' * 70}")
    print("Overall Summary")
    print(f"{'=' * 70}")
    for case, result in zip(TEST_CASES, all_results):
        score = result["faithfulness_score"]
        verdict = result["verdict"]
        print(f"  {case['label']}: {score:.1%} ({verdict})")

    avg_score = sum(r["faithfulness_score"] for r in all_results) / len(all_results)
    print(f"\n  Average faithfulness: {avg_score:.1%}")
    print(f"\n  Production threshold recommendation: ≥ 0.85 for clinical use")


if __name__ == "__main__":
    main()
