"""
Exercise 1: LoRA Hyperparameter Experiments
===========================================
Experiment with LoRA hyperparameters: rank (4, 8, 16, 32),
alpha (16, 32, 64), dropout (0.0, 0.05, 0.1). Show configuration
combinations and expected behavior.

Learning Objectives:
- Understand how each LoRA hyperparameter affects training
- Calculate parameter counts for different configurations
- Choose configurations based on dataset size and task complexity
- Build a hyperparameter search grid

Run:
    python exercise_1_lora_config.py
"""

import json
import math
import itertools
from collections import defaultdict

try:
    from peft import LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# --- Model architecture reference ---
MODEL_ARCHITECTURES = {
    "phi-2": {
        "hidden_size": 2560,
        "num_layers": 32,
        "total_params": 2_700_000_000,
        "target_modules": ["q_proj", "v_proj", "k_proj", "dense"],
        "modules_per_layer": 4,
    },
    "mistral-7b": {
        "hidden_size": 4096,
        "num_layers": 32,
        "total_params": 7_241_000_000,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "modules_per_layer": 7,
    },
    "llama-3-8b": {
        "hidden_size": 4096,
        "num_layers": 32,
        "total_params": 8_030_000_000,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "modules_per_layer": 7,
    },
}


def calculate_lora_params(hidden_size: int, rank: int, num_modules: int, num_layers: int) -> int:
    """Calculate total trainable parameters for a LoRA configuration."""
    params_per_module = hidden_size * rank + rank * hidden_size  # A + B matrices
    return params_per_module * num_modules * num_layers


def analyze_rank_effect():
    """Show how rank affects parameter count and capacity."""

    print("\n--- Effect of Rank on Parameter Count ---")
    print(f"\n  Rank controls the capacity of the LoRA adapter.")
    print(f"  Higher rank = more trainable params = more capacity.\n")

    ranks = [4, 8, 16, 32, 64]

    for model_name, arch in MODEL_ARCHITECTURES.items():
        d = arch["hidden_size"]
        n_mod = arch["modules_per_layer"]
        n_lay = arch["num_layers"]
        total = arch["total_params"]

        print(f"\n  Model: {model_name} ({total/1e9:.1f}B params, d={d})")
        print(f"  {'Rank':>6s}  {'LoRA Params':>14s}  {'% of Model':>11s}  {'Capacity'}")
        print("  " + "-" * 55)

        for r in ranks:
            lora_p = calculate_lora_params(d, r, n_mod, n_lay)
            pct = lora_p / total * 100
            capacity = "low" if r <= 4 else "medium" if r <= 16 else "high" if r <= 32 else "very high"
            bar = "█" * int(pct * 100)
            print(f"  {r:6d}  {lora_p:14,}  {pct:10.4f}%  {capacity:10s} {bar}")


def analyze_alpha_effect():
    """Show how alpha/rank ratio affects the learning signal."""

    print("\n--- Effect of Alpha on LoRA Scaling ---")
    print(f"\n  The effective scaling factor is alpha/rank.")
    print(f"  Higher alpha = stronger LoRA signal relative to base model.\n")

    ranks = [8, 16, 32]
    alphas = [8, 16, 32, 64, 128]

    print(f"  {'Rank':>6s}  {'Alpha':>6s}  {'Scale':>8s}  {'Effect'}")
    print("  " + "-" * 55)

    for r in ranks:
        for a in alphas:
            scale = a / r
            if scale < 0.5:
                effect = "very weak — barely modifies base behavior"
            elif scale < 1.0:
                effect = "weak — subtle adjustments"
            elif scale == 1.0:
                effect = "balanced — standard starting point"
            elif scale <= 2.0:
                effect = "standard — commonly used"
            elif scale <= 4.0:
                effect = "strong — significant adaptation"
            else:
                effect = "very strong — risk of instability"
            print(f"  {r:6d}  {a:6d}  {scale:8.2f}  {effect}")
        print()


def analyze_dropout_effect():
    """Show how dropout affects regularization."""

    print("\n--- Effect of Dropout on Regularization ---")
    print(f"\n  LoRA dropout is applied to the low-rank matrices during training.")
    print(f"  Higher dropout = more regularization = less overfitting.\n")

    dropouts = [0.0, 0.01, 0.05, 0.1, 0.2]
    dataset_sizes = [100, 500, 1000, 5000]

    print(f"  {'Dropout':>8s}  {'Best for Dataset Size':>25s}  {'Notes'}")
    print("  " + "-" * 70)

    for d in dropouts:
        if d == 0.0:
            best = ">5000 samples"
            notes = "No regularization — only for large datasets"
        elif d <= 0.01:
            best = "2000-5000 samples"
            notes = "Minimal regularization"
        elif d <= 0.05:
            best = "500-2000 samples"
            notes = "Standard default — good starting point"
        elif d <= 0.1:
            best = "100-500 samples"
            notes = "Moderate regularization for small datasets"
        else:
            best = "<100 samples"
            notes = "Heavy regularization — may underfit"
        print(f"  {d:8.2f}  {best:>25s}  {notes}")


def generate_search_grid():
    """Generate a hyperparameter search grid with recommendations."""

    print("\n--- Hyperparameter Search Grid ---")

    ranks = [4, 8, 16, 32]
    alphas = [16, 32, 64]
    dropouts = [0.0, 0.05, 0.1]
    target_sets = [
        (["q_proj", "v_proj"], "minimal"),
        (["q_proj", "v_proj", "k_proj", "o_proj"], "attention"),
        (["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], "all_linear"),
    ]

    total_combos = len(ranks) * len(alphas) * len(dropouts) * len(target_sets)
    print(f"\n  Total possible combinations: {total_combos}")
    print(f"  Showing recommended configurations:\n")

    configs = []

    # --- Small dataset config ---
    configs.append({
        "name": "Small Dataset (100-500 samples)",
        "rank": 4, "alpha": 16, "dropout": 0.1,
        "targets": "q_proj, v_proj",
        "reasoning": "Low rank prevents overfitting. High dropout for regularization. Minimal targets.",
    })

    # --- Medium dataset config ---
    configs.append({
        "name": "Medium Dataset (500-2000 samples)",
        "rank": 8, "alpha": 16, "dropout": 0.05,
        "targets": "q_proj, v_proj, k_proj, o_proj",
        "reasoning": "Standard rank. Balanced alpha. Moderate dropout. Attention modules.",
    })

    # --- Large dataset config ---
    configs.append({
        "name": "Large Dataset (2000-10000 samples)",
        "rank": 16, "alpha": 32, "dropout": 0.05,
        "targets": "q_proj, v_proj, k_proj, o_proj",
        "reasoning": "Higher rank for more capacity. Proportional alpha. Standard dropout.",
    })

    # --- Complex task config ---
    configs.append({
        "name": "Complex Task (medical coding, many classes)",
        "rank": 32, "alpha": 64, "dropout": 0.05,
        "targets": "all linear layers",
        "reasoning": "High rank for complex decision boundaries. All linear layers for maximum adaptation.",
    })

    # --- Resource-constrained config ---
    configs.append({
        "name": "Resource Constrained (limited GPU memory)",
        "rank": 4, "alpha": 8, "dropout": 0.05,
        "targets": "q_proj, v_proj",
        "reasoning": "Minimal parameter overhead. Works with QLoRA on 6GB VRAM.",
    })

    for config in configs:
        print(f"  ┌ {config['name']}")
        print(f"  │  rank={config['rank']}, alpha={config['alpha']}, dropout={config['dropout']}")
        print(f"  │  targets: {config['targets']}")
        print(f"  │  {config['reasoning']}")

        # Calculate params for Mistral-7B
        arch = MODEL_ARCHITECTURES["mistral-7b"]
        n_modules = 2 if "q_proj, v_proj" == config["targets"] else \
                    4 if "attention" in config["targets"] else 7
        lora_p = calculate_lora_params(arch["hidden_size"], config["rank"], n_modules, arch["num_layers"])
        print(f"  │  Est. trainable params (Mistral-7B): {lora_p:,}")
        print(f"  └")
        print()


def create_peft_configs():
    """Create actual PEFT LoraConfig objects for each recommendation."""

    print("\n--- Creating PEFT LoraConfig Objects ---")

    if PEFT_AVAILABLE:
        configs = [
            ("Conservative (small data)", LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=4, lora_alpha=16, lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],
                bias="none",
            )),
            ("Standard (medium data)", LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8, lora_alpha=16, lora_dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none",
            )),
            ("Aggressive (large data)", LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none",
            )),
            ("Maximum (complex task)", LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=32, lora_alpha=64, lora_dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                bias="none",
            )),
        ]

        for name, config in configs:
            print(f"\n  {name}:")
            print(f"    r={config.r}, alpha={config.lora_alpha}, "
                  f"dropout={config.lora_dropout}")
            print(f"    targets: {config.target_modules}")
            print(f"    effective scale: {config.lora_alpha / config.r:.1f}")

    else:
        print("\n  peft not installed. Install with: pip install peft")
        print("  Showing config dictionaries instead:\n")

        config_dicts = [
            {"name": "Conservative", "r": 4, "alpha": 16, "dropout": 0.1, "targets": ["q_proj", "v_proj"]},
            {"name": "Standard", "r": 8, "alpha": 16, "dropout": 0.05, "targets": ["q_proj", "v_proj", "k_proj", "o_proj"]},
            {"name": "Aggressive", "r": 16, "alpha": 32, "dropout": 0.05, "targets": ["q_proj", "v_proj", "k_proj", "o_proj"]},
            {"name": "Maximum", "r": 32, "alpha": 64, "dropout": 0.05, "targets": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
        ]
        for cfg in config_dicts:
            print(f"  {cfg['name']}: r={cfg['r']}, alpha={cfg['alpha']}, "
                  f"dropout={cfg['dropout']}, targets={cfg['targets']}")


def simulate_training_comparison():
    """Simulate training curves for different LoRA configs."""

    print("\n--- Simulated Training Curves by Configuration ---")
    print(f"  (Simulated loss over 200 steps for different rank values)\n")

    import random as rng
    rng.seed(42)

    configs_to_sim = [
        ("r=4,  α=16", 4, 16, 0.1),
        ("r=8,  α=16", 8, 16, 0.05),
        ("r=16, α=32", 16, 32, 0.05),
        ("r=32, α=64", 32, 64, 0.05),
    ]

    steps = 200
    print(f"  {'Step':>6s}", end="")
    for name, _, _, _ in configs_to_sim:
        print(f"  {name:>12s}", end="")
    print()
    print("  " + "-" * (6 + 14 * len(configs_to_sim)))

    for step in [0, 10, 25, 50, 75, 100, 125, 150, 175, 199]:
        print(f"  {step:6d}", end="")
        for name, rank, alpha, dropout in configs_to_sim:
            progress = step / steps
            # Higher rank → faster convergence, lower final loss
            convergence_speed = 1.0 + (rank / 32) * 0.5
            final_loss = 0.3 + (1.0 / rank) * 0.8
            base_loss = 3.0 * (1 - progress) ** convergence_speed + final_loss
            noise = rng.gauss(0, 0.05 * (1 - progress))
            loss = max(0.1, base_loss + noise)
            print(f"  {loss:12.4f}", end="")
        print()

    print(f"\n  Key observations:")
    print(f"    - Higher rank → faster convergence and lower final loss")
    print(f"    - But also higher risk of overfitting on small datasets")
    print(f"    - rank=8 or 16 is the sweet spot for most medical tasks")


def main():
    """Experiment with LoRA hyperparameters."""

    print("=" * 60)
    print("Exercise 1: LoRA Hyperparameter Experiments")
    print("=" * 60)

    # Step 1: Rank analysis
    analyze_rank_effect()

    # Step 2: Alpha analysis
    analyze_alpha_effect()

    # Step 3: Dropout analysis
    analyze_dropout_effect()

    # Step 4: Search grid
    generate_search_grid()

    # Step 5: Create configs
    create_peft_configs()

    # Step 6: Simulated comparison
    simulate_training_comparison()

    print(f"\n{'=' * 60}")
    print(f"✓ Hyperparameter experiment complete!")
    print(f"  Recommendation for ICD-10 medical coding:")
    print(f"  → Start with rank=8, alpha=16, dropout=0.05")
    print(f"  → Target modules: q_proj, v_proj, k_proj, o_proj")
    print(f"  → Scale up rank if validation loss plateaus")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
