"""
Exercise 3 — Quantization Comparison
=====================================

Skills practiced
----------------
* Understanding model quantization (Q4_0, Q4_K_M, Q5_K_M, Q8_0, FP16)
* Measuring quality vs. speed vs. model-size trade-offs
* Running the same medical prompts at different quantization levels
* Interpreting perplexity-like quality metrics without a formal eval set

Healthcare context
------------------
When deploying an LLM on hospital hardware (often a single GPU or even
CPU-only), quantization determines whether the model fits in memory at
all.  A 70 B model at Q4 may fit in 40 GB VRAM while the FP16 version
needs 140 GB.  This exercise helps you decide which quantization level
gives "good enough" quality for clinical use.

Usage
-----
    python exercise_3_quantization_comparison.py

Prerequisites
-------------
    # Pull multiple quantization variants of the same model family.
    # Ollama names variants like:  llama3:8b-instruct-q4_0
    ollama pull llama3
    pip install openai
"""

import json
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"

# Quantization variants to compare.  Adjust to what you have pulled.
# If only the default tag is available, we'll use that as a single data point.
QUANT_VARIANTS = [
    "llama3:8b-instruct-q4_0",
    "llama3:8b-instruct-q4_K_M",
    "llama3:8b-instruct-q5_K_M",
    "llama3:8b-instruct-q8_0",
    "llama3",  # default tag — usually Q4_K_M or similar
]

MEDICAL_PROMPTS = [
    {
        "id": "p1",
        "prompt": (
            "A 62-year-old woman with a history of atrial fibrillation on warfarin "
            "presents with INR 9.2 and hematuria.  Outline step-by-step management."
        ),
        "keywords": ["hold warfarin", "vitamin K", "FFP", "prothrombin complex",
                      "monitor", "INR", "hematuria"],
    },
    {
        "id": "p2",
        "prompt": (
            "Explain the mechanism of action of SGLT2 inhibitors and their benefits "
            "in heart failure with reduced ejection fraction."
        ),
        "keywords": ["sodium-glucose", "kidney", "glucose", "natriuresis",
                      "preload", "afterload", "HFrEF", "empagliflozin"],
    },
    {
        "id": "p3",
        "prompt": (
            "Describe the diagnostic workup for suspected pulmonary embolism, "
            "including when to use the Wells score, D-dimer, and CT pulmonary angiography."
        ),
        "keywords": ["Wells", "D-dimer", "CTPA", "probability", "anticoagulation",
                      "heparin", "V/Q scan"],
    },
]

SYSTEM_PROMPT = (
    "You are a senior internal medicine physician. "
    "Provide accurate, concise, evidence-based answers."
)


# ---------------------------------------------------------------------------
# Quantization explainer
# ---------------------------------------------------------------------------
QUANT_INFO = {
    "q4_0":   {"bits": 4, "desc": "4-bit uniform quantization — smallest, fastest, lowest quality"},
    "q4_K_M": {"bits": 4, "desc": "4-bit K-quant (medium) — good balance of size and quality"},
    "q5_K_M": {"bits": 5, "desc": "5-bit K-quant (medium) — slightly larger, better quality"},
    "q8_0":   {"bits": 8, "desc": "8-bit quantization — near-FP16 quality, ~2x the size of Q4"},
    "fp16":   {"bits": 16, "desc": "Full half-precision — baseline quality, largest model file"},
}


def print_quantization_primer():
    """Print a short explanation of quantization levels."""
    print("\n--- Quantization Primer ---")
    print("Quantization reduces the precision of model weights to shrink the file")
    print("and speed up inference, at the cost of some quality degradation.\n")
    print(f"{'Level':<10} {'Bits':<6} {'Description'}")
    print("-" * 65)
    for level, info in QUANT_INFO.items():
        print(f"{level:<10} {info['bits']:<6} {info['desc']}")
    print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_client():
    from openai import OpenAI
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    try:
        client.models.list()
    except Exception as exc:
        print(f"Cannot reach Ollama: {exc}")
        sys.exit(1)
    return client


def detect_available_variants(client) -> list[str]:
    """Return the subset of QUANT_VARIANTS that are pulled locally."""
    available = {m.id for m in client.models.list().data}
    found = [v for v in QUANT_VARIANTS if v in available]
    if not found:
        # Fall back: check if any llama3 variant exists
        fallback = [m for m in available if "llama3" in m.lower()]
        if fallback:
            found = fallback[:3]
    return found


def score_keywords(text: str, keywords: list[str]) -> float:
    lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return hits / len(keywords) if keywords else 0.0


def estimate_model_size_gb(variant: str) -> str:
    """Rough estimate of model size based on quantization level."""
    # These are approximate for a 7-8B parameter model
    if "q4_0" in variant:
        return "~4.0 GB"
    elif "q4_k" in variant.lower():
        return "~4.4 GB"
    elif "q5_k" in variant.lower():
        return "~5.1 GB"
    elif "q8" in variant:
        return "~7.7 GB"
    elif "fp16" in variant or "f16" in variant:
        return "~14.5 GB"
    return "~4-5 GB"


def run_prompt(client, model: str, prompt_data: dict) -> dict:
    """Run a single prompt and return metrics."""
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_data["prompt"]},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        elapsed = time.time() - start
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        quality = score_keywords(text, prompt_data["keywords"])
        return {
            "time_s": round(elapsed, 2),
            "tokens": tokens,
            "quality": round(quality, 2),
            "response": text,
            "error": "",
        }
    except Exception as exc:
        return {"time_s": round(time.time() - start, 2), "tokens": 0,
                "quality": 0.0, "response": "", "error": str(exc)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Exercise 3: Quantization Comparison")
    print("=" * 60)

    print_quantization_primer()

    client = get_client()
    variants = detect_available_variants(client)

    if not variants:
        print("⚠  No llama3 variants found.  Pull at least one:")
        for v in QUANT_VARIANTS:
            print(f"   ollama pull {v}")
        return

    print(f"Found {len(variants)} variant(s): {variants}\n")

    if len(variants) == 1:
        print("TIP: Pull multiple quantization variants for a meaningful comparison:")
        print("   ollama pull llama3:8b-instruct-q4_0")
        print("   ollama pull llama3:8b-instruct-q8_0")
        print("   (Continuing with the single available variant...)\n")

    # --- Run benchmarks ---
    all_results: dict[str, list[dict]] = {v: [] for v in variants}

    for p in MEDICAL_PROMPTS:
        print(f"\n--- {p['id']}: {p['prompt'][:65]}... ---")
        for variant in variants:
            result = run_prompt(client, variant, p)
            all_results[variant].append(result)
            status = "OK" if not result["error"] else f"ERR: {result['error']}"
            print(
                f"  {variant:<35} {result['time_s']:>5.1f}s  "
                f"quality={result['quality']:.0%}  tokens={result['tokens']}  {status}"
            )

    # --- Summary table ---
    print(f"\n{'='*60}")
    print("QUANTIZATION COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':<35} {'Est. Size':<12} {'Avg Time':<10} {'Avg Quality':<12}")
    print("-" * 69)

    for variant in variants:
        results = [r for r in all_results[variant] if not r["error"]]
        if results:
            avg_time = sum(r["time_s"] for r in results) / len(results)
            avg_qual = sum(r["quality"] for r in results) / len(results)
        else:
            avg_time = avg_qual = 0.0
        size = estimate_model_size_gb(variant)
        print(f"{variant:<35} {size:<12} {avg_time:<10.2f} {avg_qual:<12.0%}")

    # --- Analysis ---
    print(f"\n--- Analysis ---")
    print("Key trade-offs to observe:")
    print("  • Q4 variants are fastest and smallest but may miss nuances.")
    print("  • Q8 variants are ~2x the size but noticeably closer to FP16 quality.")
    print("  • For clinical decision-support, Q5_K_M often hits the sweet spot.")
    print("  • Always validate quality on YOUR specific use-case before deploying.")

    # --- Optional: detailed dump ---
    print("\n--- Detailed Results (JSON) ---")
    summary = []
    for variant in variants:
        for i, r in enumerate(all_results[variant]):
            summary.append({
                "variant": variant,
                "prompt_id": MEDICAL_PROMPTS[i]["id"],
                "time_s": r["time_s"],
                "quality": r["quality"],
                "tokens": r["tokens"],
                "error": r["error"],
            })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
