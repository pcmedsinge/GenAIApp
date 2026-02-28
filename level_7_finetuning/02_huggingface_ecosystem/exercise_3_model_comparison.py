"""
Exercise 3 — HuggingFace Model Comparison
==========================================

Skills practiced
----------------
* Loading and running multiple HuggingFace models
* Measuring inference speed, memory footprint, and output quality
* Building a structured comparison table
* Choosing the right model for a given resource budget

Healthcare context
------------------
When selecting a model for clinical NLP, engineers must balance:
  - **Accuracy** on domain-specific tasks
  - **Latency** (real-time clinical alerts need <1 s)
  - **Memory** (edge devices in clinics may have <8 GB RAM)
  - **Licensing** (some models restrict clinical/commercial use)

This exercise benchmarks 2-3 small, CPU-friendly models on the same set
of clinical text tasks, then produces a head-to-head comparison.

Usage
-----
    python exercise_3_model_comparison.py

Prerequisites
-------------
    pip install transformers torch psutil
"""

import os
import sys
import time
import warnings
from dataclasses import dataclass, field

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from transformers import pipeline, AutoTokenizer
    import torch
except ImportError:
    print("ERROR: pip install transformers torch")
    sys.exit(1)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Models to compare — small enough for CPU
MODELS_TEXT_GEN = [
    {"name": "distilgpt2", "task": "text-generation"},
    {"name": "sshleifer/tiny-gpt2", "task": "text-generation"},
]

MODELS_CLASSIFICATION = [
    {"name": "distilbert-base-uncased-finetuned-sst-2-english", "task": "text-classification"},
    {"name": "textattack/bert-base-uncased-SST-2", "task": "text-classification"},
]

MODELS_NER = [
    {"name": "dslim/bert-base-NER", "task": "ner"},
    {"name": "elastic/distilbert-base-uncased-finetuned-conll03-english", "task": "ner"},
]

# Clinical test inputs
CLINICAL_TEXTS = {
    "text-generation": [
        "The patient was diagnosed with",
        "Treatment plan includes",
        "Lab results indicate elevated",
    ],
    "text-classification": [
        "The patient is recovering well and vital signs are stable.",
        "Prognosis is guarded with significant risk of complications.",
        "Routine follow-up appointment, no acute concerns.",
        "Critical condition, requires immediate ICU transfer.",
    ],
    "ner": [
        "Dr. Smith at Massachusetts General Hospital prescribed Metformin 500mg for diabetes.",
        "Patient John Doe from Boston was admitted to the cardiology unit at Mayo Clinic.",
        "Nurse Johnson administered Heparin 5000 units subcutaneously at Johns Hopkins.",
    ],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    model_name: str
    task: str
    load_time_s: float = 0.0
    avg_inference_s: float = 0.0
    memory_mb: float = 0.0
    num_params_m: float = 0.0
    sample_output: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_memory_mb() -> float:
    """Current process RSS in MB."""
    if HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    return 0.0


def count_params(model) -> float:
    """Count parameters in millions."""
    try:
        return sum(p.numel() for p in model.parameters()) / 1e6
    except Exception:
        return 0.0


def benchmark_model(model_cfg: dict, texts: list[str]) -> BenchmarkResult:
    """Load a model, run it on sample texts, and measure performance."""
    result = BenchmarkResult(model_name=model_cfg["name"], task=model_cfg["task"])

    mem_before = get_memory_mb()

    # Load
    try:
        start = time.time()
        extra_kwargs = {}
        if model_cfg["task"] == "ner":
            extra_kwargs["aggregation_strategy"] = "simple"
        if model_cfg["task"] == "text-generation":
            extra_kwargs["max_new_tokens"] = 30

        pipe = pipeline(model_cfg["task"], model=model_cfg["name"], **extra_kwargs)
        result.load_time_s = round(time.time() - start, 2)
    except Exception as exc:
        result.error = str(exc)
        return result

    mem_after = get_memory_mb()
    result.memory_mb = round(mem_after - mem_before, 1)

    # Count params
    if hasattr(pipe, "model"):
        result.num_params_m = round(count_params(pipe.model), 1)

    # Inference
    times = []
    outputs = []
    for text in texts:
        try:
            start = time.time()
            out = pipe(text)
            times.append(time.time() - start)
            outputs.append(str(out)[:200])
        except Exception as exc:
            result.error = str(exc)

    if times:
        result.avg_inference_s = round(sum(times) / len(times), 4)
    if outputs:
        result.sample_output = outputs[0][:150]

    # Cleanup to free memory
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------
def compare_models(model_group: list[dict], group_name: str):
    """Benchmark a group of models and print comparison."""
    print(f"\n{'='*60}")
    print(f"Comparing: {group_name}")
    print(f"{'='*60}")

    task = model_group[0]["task"]
    texts = CLINICAL_TEXTS.get(task, ["Test input."])
    results = []

    for cfg in model_group:
        print(f"\n  Benchmarking: {cfg['name']} ...")
        r = benchmark_model(cfg, texts)
        results.append(r)

        if r.error:
            print(f"    ERROR: {r.error}")
        else:
            print(f"    Load time     : {r.load_time_s:.2f}s")
            print(f"    Avg inference  : {r.avg_inference_s:.4f}s")
            print(f"    Memory delta   : {r.memory_mb:.1f} MB")
            print(f"    Parameters     : {r.num_params_m:.1f}M")
            print(f"    Sample output  : {r.sample_output[:100]}...")

    # --- Comparison table ---
    print(f"\n  {'Model':<55} {'Load(s)':<9} {'Infer(s)':<10} {'Mem(MB)':<10} {'Params(M)'}")
    print("  " + "-" * 94)
    for r in results:
        if not r.error:
            print(f"  {r.model_name:<55} {r.load_time_s:<9.2f} {r.avg_inference_s:<10.4f} "
                  f"{r.memory_mb:<10.1f} {r.num_params_m:.1f}")
        else:
            print(f"  {r.model_name:<55} ERROR: {r.error[:40]}")

    # Winner analysis
    ok = [r for r in results if not r.error]
    if len(ok) >= 2:
        fastest = min(ok, key=lambda r: r.avg_inference_s)
        smallest = min(ok, key=lambda r: r.memory_mb)
        print(f"\n  Fastest inference : {fastest.model_name} ({fastest.avg_inference_s:.4f}s)")
        print(f"  Smallest memory   : {smallest.model_name} ({smallest.memory_mb:.1f} MB)")

    return results


# ---------------------------------------------------------------------------
# Overall analysis
# ---------------------------------------------------------------------------
def overall_analysis(all_results: list[BenchmarkResult]):
    """Print a cross-task summary and recommendations."""
    print(f"\n{'='*60}")
    print("OVERALL COMPARISON SUMMARY")
    print(f"{'='*60}")

    ok = [r for r in all_results if not r.error]
    if not ok:
        print("  No successful benchmarks to summarise.")
        return

    print(f"\n  {'Model':<55} {'Task':<20} {'Infer(s)':<10} {'Mem(MB)'}")
    print("  " + "-" * 95)
    for r in sorted(ok, key=lambda x: x.avg_inference_s):
        print(f"  {r.model_name:<55} {r.task:<20} {r.avg_inference_s:<10.4f} {r.memory_mb:.1f}")

    print("\n  Recommendations for clinical deployments:")
    print("  • For real-time alerts (< 200ms): choose the fastest model that meets quality bar.")
    print("  • For batch processing (overnight reports): choose the highest-quality model.")
    print("  • For edge / embedded devices: prioritize memory footprint.")
    print("  • Always validate on YOUR clinical data — general benchmarks may not transfer.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Exercise 3: HuggingFace Model Comparison")
    print("=" * 60)

    if not HAS_PSUTIL:
        print("\nTIP: pip install psutil  for memory measurement\n")

    all_results = []

    # 1. Text generation models
    results = compare_models(MODELS_TEXT_GEN, "Text Generation Models")
    all_results.extend(results)

    # 2. Classification models
    results = compare_models(MODELS_CLASSIFICATION, "Text Classification Models")
    all_results.extend(results)

    # 3. NER models
    results = compare_models(MODELS_NER, "Named Entity Recognition Models")
    all_results.extend(results)

    # Overall
    overall_analysis(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
