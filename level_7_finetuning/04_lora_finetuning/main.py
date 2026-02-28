"""
LoRA Fine-Tuning — Main Demo
=============================
Demonstrates LoRA configuration, training setup, training loop,
and model merging for medical language model fine-tuning.

Run:
    python main.py

Note: Heavy ML libraries (transformers, peft, etc.) are imported
with try/except — demos show patterns even if not installed.
"""

import json
import os
import math
import random
from collections import defaultdict

# --- Attempt heavy imports ---
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        PeftModel,
        TaskType,
        prepare_model_for_kbit_training,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from datasets import Dataset, load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False


def check_libraries():
    """Print library availability status."""
    print("\n--- Library Status ---")
    libs = [
        ("torch", TORCH_AVAILABLE),
        ("transformers", TRANSFORMERS_AVAILABLE),
        ("peft", PEFT_AVAILABLE),
        ("datasets", DATASETS_AVAILABLE),
        ("bitsandbytes", BNB_AVAILABLE),
    ]
    for name, available in libs:
        status = "✓ installed" if available else "✗ not installed"
        print(f"  {name:20s} {status}")
    print()
    any_missing = not all(a for _, a in libs)
    if any_missing:
        print("  Some libraries are missing. Demos will show code patterns")
        print("  and explanations even without running actual training.\n")
        print("  Install with: pip install transformers peft datasets accelerate bitsandbytes torch\n")
    return not any_missing


# ============================================================
# DEMO 1: LoRA Explained
# ============================================================
def demo_lora_explained():
    """Explain LoRA with code: low-rank adaptation concept, rank,
    alpha, dropout. Show LoraConfig from peft."""

    print("\n" + "=" * 60)
    print("DEMO 1: LoRA (Low-Rank Adaptation) Explained")
    print("=" * 60)

    # --- Concept ---
    print("""
=== What is LoRA? ===

Standard fine-tuning updates ALL model weights:
  W_new = W_old + ΔW       (ΔW is full-rank, same size as W)

LoRA decomposes the update into two small matrices:
  W_new = W_old + (α/r) · B × A

  Where:
    W_old: original weight matrix (d × d), e.g., 4096 × 4096
    A: low-rank matrix (d × r), e.g., 4096 × 8
    B: low-rank matrix (r × d), e.g., 8 × 4096
    r: rank (typically 4–64, usually 8 or 16)
    α: scaling factor (typically 2× rank)

Parameter savings:
  Full update:  4096 × 4096 = 16,777,216 parameters
  LoRA (r=8):   4096 × 8 + 8 × 4096 = 65,536 parameters
  Reduction:    99.6% fewer trainable parameters!
""")

    # --- Show with actual numbers ---
    print("--- Parameter Comparison ---")
    d = 4096  # typical hidden dimension
    for r in [4, 8, 16, 32, 64]:
        full_params = d * d
        lora_params = d * r + r * d
        reduction = (1 - lora_params / full_params) * 100
        print(f"  Rank {r:2d}: {lora_params:>10,} params ({reduction:.2f}% reduction)")

    # --- LoRA Config ---
    print("\n--- LoRA Configuration ---")
    if PEFT_AVAILABLE:
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
        )
        print(f"  LoraConfig created successfully:")
        print(f"    task_type:      {config.task_type}")
        print(f"    r:              {config.r}")
        print(f"    lora_alpha:     {config.lora_alpha}")
        print(f"    lora_dropout:   {config.lora_dropout}")
        print(f"    target_modules: {config.target_modules}")
        print(f"    bias:           {config.bias}")
    else:
        print("  peft not installed — showing configuration pattern:\n")
        print("""  from peft import LoraConfig, TaskType

  config = LoraConfig(
      task_type=TaskType.CAUSAL_LM,   # Causal language modeling
      r=8,                              # Rank of the update matrices
      lora_alpha=16,                    # Scaling factor (alpha/r applied)
      lora_dropout=0.05,                # Dropout on LoRA layers
      target_modules=[                  # Which layers to adapt
          "q_proj", "v_proj",           # Attention Q and V projections
          "k_proj", "o_proj",           # Attention K and O projections
      ],
      bias="none",                      # Don't train bias terms
  )""")

    # --- Hyperparameter guidance ---
    print("\n--- Hyperparameter Guide ---")
    print("""
  ┌──────────┬───────────┬──────────────────────────────────┐
  │ Param    │ Range     │ Guidance                         │
  ├──────────┼───────────┼──────────────────────────────────┤
  │ r (rank) │ 4–64      │ 8 for small datasets, 16–32 for │
  │          │           │ complex tasks, 64 if underfitting│
  ├──────────┼───────────┼──────────────────────────────────┤
  │ alpha    │ 1–128     │ Usually 2× rank. Higher = more   │
  │          │           │ influence from LoRA weights.      │
  ├──────────┼───────────┼──────────────────────────────────┤
  │ dropout  │ 0.0–0.1   │ 0.05 default. Increase if        │
  │          │           │ overfitting on small datasets.    │
  ├──────────┼───────────┼──────────────────────────────────┤
  │ modules  │ varies    │ q_proj + v_proj minimum.          │
  │          │           │ Add k_proj, o_proj for more.      │
  └──────────┴───────────┴──────────────────────────────────┘
""")

    # --- Simulate LoRA matrix math ---
    print("--- Simulated LoRA Forward Pass ---")
    if TORCH_AVAILABLE:
        d_model = 256
        r = 8
        W = torch.randn(d_model, d_model) * 0.02  # pretrained weights
        A = torch.randn(d_model, r) * 0.02         # LoRA down-projection
        B = torch.zeros(r, d_model)                 # LoRA up-projection (init to 0)
        alpha = 16

        x = torch.randn(1, d_model)  # sample input

        # Standard forward
        out_base = x @ W

        # LoRA forward: base + scaled low-rank update
        out_lora = x @ W + (alpha / r) * (x @ A @ B)

        # Initially B=0, so outputs are identical
        diff = (out_lora - out_base).abs().max().item()
        print(f"  d_model={d_model}, rank={r}, alpha={alpha}")
        print(f"  Base params:  {d_model * d_model:,}")
        print(f"  LoRA params:  {d_model * r + r * d_model:,}")
        print(f"  Max diff (B=0 init): {diff:.6f}  (should be ~0)")
        print("  → LoRA starts as identity (no change), then learns updates")
    else:
        print("  (torch not available — skipping matrix simulation)")


# ============================================================
# DEMO 2: Training Setup
# ============================================================
def demo_training_setup():
    """Show complete training configuration: model loading, LoRA config,
    training arguments, data loading."""

    print("\n" + "=" * 60)
    print("DEMO 2: Complete Training Setup")
    print("=" * 60)

    all_available = TRANSFORMERS_AVAILABLE and PEFT_AVAILABLE and DATASETS_AVAILABLE and TORCH_AVAILABLE

    if all_available:
        print("\n--- Loading Model and Tokenizer ---")
        model_name = "microsoft/phi-2"  # small enough for demo
        print(f"  Model: {model_name}")
        print("  Note: Using a small model for demonstration.\n")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )

            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Model loaded: {total_params:,} parameters")

            # Apply LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                bias="none",
            )

            model = get_peft_model(model, lora_config)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in model.parameters())
            print(f"\n  After LoRA:")
            print(f"    Trainable params: {trainable_params:,}")
            print(f"    Total params:     {all_params:,}")
            print(f"    Trainable %:      {100 * trainable_params / all_params:.4f}%")

            model.print_trainable_parameters()

        except Exception as e:
            print(f"  Could not load model: {e}")
            print("  This is expected if the model hasn't been downloaded.")
    else:
        print("\n  Required libraries not fully installed.")
        print("  Showing the complete training setup pattern:\n")

    # Always show the code pattern
    print("\n--- Complete Training Setup Code ---")
    print("""
  from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
  from peft import LoraConfig, get_peft_model, TaskType
  from datasets import Dataset
  import torch

  # 1. Load model and tokenizer
  model_name = "mistralai/Mistral-7B-v0.1"  # or any base model
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  tokenizer.pad_token = tokenizer.eos_token

  # 2. Load in 4-bit for QLoRA (optional, saves VRAM)
  from transformers import BitsAndBytesConfig
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_use_double_quant=True,
  )
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config=bnb_config,
      device_map="auto",
  )

  # 3. Configure LoRA
  lora_config = LoraConfig(
      task_type=TaskType.CAUSAL_LM,
      r=16,
      lora_alpha=32,
      lora_dropout=0.05,
      target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
  )
  model = get_peft_model(model, lora_config)
  model.print_trainable_parameters()

  # 4. Training arguments
  training_args = TrainingArguments(
      output_dir="./medical-lora-output",
      num_train_epochs=3,
      per_device_train_batch_size=4,
      gradient_accumulation_steps=4,
      learning_rate=2e-4,
      warmup_steps=100,
      logging_steps=10,
      save_steps=200,
      evaluation_strategy="steps",
      eval_steps=200,
      fp16=True,
      report_to="none",
  )

  # 5. Create Trainer
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
  )

  # 6. Train!
  trainer.train()
  model.save_pretrained("./medical-lora-adapter")
""")


# ============================================================
# DEMO 3: Training Loop
# ============================================================
def demo_training_loop():
    """Show the training process: forward pass, loss, backward pass,
    optimizer step. Simulate a training run."""

    print("\n" + "=" * 60)
    print("DEMO 3: Training Loop Walkthrough")
    print("=" * 60)

    # --- Explain the training loop ---
    print("""
=== Training Loop Steps ===

For each batch:
  1. Forward pass:  logits = model(input_ids)
  2. Compute loss:  loss = cross_entropy(logits, labels)
  3. Backward pass:  loss.backward()  → compute gradients
  4. Optimizer step: optimizer.step()  → update LoRA weights only
  5. Zero gradients: optimizer.zero_grad()
  6. Log metrics:    record loss, learning rate, etc.

With Hugging Face Trainer, all of this is handled automatically.
""")

    # --- Simulate a training run ---
    print("--- Simulated Training Run ---")
    print("  (Simulating loss curve for 500 steps)\n")

    random.seed(42)
    num_steps = 500
    initial_loss = 3.2
    final_loss = 0.45
    warmup_steps = 50

    losses = []
    for step in range(num_steps):
        progress = step / num_steps
        base_loss = initial_loss * (1 - progress) ** 1.5 + final_loss
        noise = random.gauss(0, 0.08 * (1 - progress))
        if step < warmup_steps:
            lr_mult = step / warmup_steps
            base_loss = initial_loss - (initial_loss - base_loss) * lr_mult
        loss = max(0.1, base_loss + noise)
        losses.append(loss)

    # Print loss at key intervals
    print(f"  {'Step':>6s}  {'Loss':>8s}  {'LR':>10s}  {'Visual'}")
    print("  " + "-" * 55)
    for step in [0, 10, 25, 50, 100, 150, 200, 300, 400, 499]:
        loss = losses[step]
        # Cosine LR with warmup
        if step < warmup_steps:
            lr = 2e-4 * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (num_steps - warmup_steps)
            lr = 2e-4 * 0.5 * (1 + math.cos(math.pi * progress))
        bar_len = int(loss / initial_loss * 30)
        bar = "█" * bar_len
        print(f"  {step:6d}  {loss:8.4f}  {lr:10.2e}  {bar}")

    print(f"\n  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Best loss:    {min(losses):.4f} (step {losses.index(min(losses))})")

    # --- Show eval metrics ---
    print("\n--- Simulated Evaluation Metrics ---")
    eval_steps = list(range(0, num_steps, 100))
    print(f"  {'Step':>6s}  {'Train Loss':>11s}  {'Val Loss':>9s}  {'Accuracy':>9s}")
    print("  " + "-" * 42)
    for step in eval_steps:
        train_loss = losses[step]
        val_loss = train_loss + random.uniform(0.05, 0.2)
        accuracy = max(0, min(1, 1 - val_loss / 4 + random.uniform(-0.02, 0.02)))
        print(f"  {step:6d}  {train_loss:11.4f}  {val_loss:9.4f}  {accuracy:9.2%}")

    # --- Checkpoint saving ---
    print("\n--- Checkpoint Strategy ---")
    print("""
  Recommended checkpointing:
    save_steps=200            # Save every 200 steps
    save_total_limit=3        # Keep only 3 most recent
    load_best_model_at_end=True  # Load best checkpoint when done
    metric_for_best_model="eval_loss"

  Checkpoint contents:
    adapter_config.json       # LoRA configuration
    adapter_model.bin         # LoRA weights only (~30MB for r=16)
    optimizer.pt              # Optimizer state
    scheduler.pt              # LR scheduler state
    training_args.bin         # Training arguments
""")


# ============================================================
# DEMO 4: Model Merging
# ============================================================
def demo_model_merging():
    """After training, merge LoRA weights back into base model.
    Save and load the merged model."""

    print("\n" + "=" * 60)
    print("DEMO 4: Model Merging & Export")
    print("=" * 60)

    # --- Explain merging ---
    print("""
=== Why Merge? ===

During training, LoRA keeps adapter weights separate from base weights:
  output = base_model(x) + lora_adapter(x)

For deployment, we merge them into a single model:
  merged_weights = base_weights + (alpha/r) * B @ A

Benefits of merging:
  ✓ No adapter overhead at inference time
  ✓ Standard model format — works with any inference engine
  ✓ Can convert to GGUF for llama.cpp / Ollama
  ✓ Simpler deployment pipeline
""")

    # --- Show merging code ---
    print("--- Merging Code Pattern ---")
    print("""
  from peft import PeftModel
  from transformers import AutoModelForCausalLM, AutoTokenizer

  # 1. Load base model
  base_model = AutoModelForCausalLM.from_pretrained(
      "mistralai/Mistral-7B-v0.1",
      torch_dtype=torch.float16,
      device_map="auto",
  )

  # 2. Load LoRA adapter
  model = PeftModel.from_pretrained(
      base_model,
      "./medical-lora-adapter",  # path to saved adapter
  )

  # 3. Merge weights
  merged_model = model.merge_and_unload()

  # 4. Save merged model
  merged_model.save_pretrained("./medical-model-merged")
  tokenizer.save_pretrained("./medical-model-merged")
""")

    # --- Show size comparison ---
    print("--- Size Comparison ---")
    print("""
  Model sizes (Mistral 7B example):
  ┌──────────────────┬───────────┬──────────────────────┐
  │ Component        │ Size      │ Notes                │
  ├──────────────────┼───────────┼──────────────────────┤
  │ Base model (FP16)│ ~14 GB    │ Full model weights   │
  │ LoRA adapter     │ ~30 MB    │ Just the ΔW matrices │
  │ Merged model     │ ~14 GB    │ Base + ΔW combined   │
  │ GGUF Q4_K_M      │ ~4.4 GB   │ 4-bit quantized      │
  │ GGUF Q8_0        │ ~7.7 GB   │ 8-bit quantized      │
  └──────────────────┴───────────┴──────────────────────┘
""")

    # --- Show GGUF conversion ---
    print("--- Converting to GGUF for Ollama ---")
    print("""
  # Install llama.cpp conversion tools
  git clone https://github.com/ggerganov/llama.cpp
  cd llama.cpp
  pip install -r requirements.txt

  # Convert merged model to GGUF
  python convert_hf_to_gguf.py \\
      ../medical-model-merged \\
      --outfile medical-model.gguf \\
      --outtype q4_K_M    # 4-bit quantization

  # Create Ollama Modelfile
  cat > Modelfile << 'EOF'
  FROM ./medical-model.gguf
  SYSTEM "You are a medical coding assistant specialized in ICD-10 coding."
  PARAMETER temperature 0.1
  PARAMETER num_ctx 4096
  EOF

  # Import into Ollama
  ollama create medical-coder -f Modelfile

  # Test
  ollama run medical-coder "Patient has type 2 diabetes with neuropathy"
""")

    # --- Simulate merging math ---
    print("--- Simulated Merge Operation ---")
    if TORCH_AVAILABLE:
        d = 128
        r = 8
        alpha = 16

        W_base = torch.randn(d, d) * 0.02
        A = torch.randn(d, r) * 0.01
        B = torch.randn(r, d) * 0.01

        # Merge: W_merged = W_base + (alpha/r) * A @ B.T... actually B @ A
        # In LoRA: h = W_base @ x + (alpha/r) * B @ A @ x
        # So: W_merged = W_base + (alpha/r) * (A @ B)
        delta_W = (alpha / r) * (A @ B)
        W_merged = W_base + delta_W

        print(f"  Base weight norm:   {W_base.norm():.4f}")
        print(f"  Delta weight norm:  {delta_W.norm():.4f}")
        print(f"  Merged weight norm: {W_merged.norm():.4f}")
        print(f"  Delta / Base ratio: {delta_W.norm() / W_base.norm():.4f}")
        print(f"  → LoRA makes small, targeted adjustments to base weights")
    else:
        print("  (torch not available — skipping merge simulation)")


# ============================================================
# Main Menu
# ============================================================
def main():
    """Interactive demo menu for LoRA Fine-Tuning."""
    check_libraries()

    demos = {
        "1": ("LoRA Explained", demo_lora_explained),
        "2": ("Training Setup", demo_training_setup),
        "3": ("Training Loop", demo_training_loop),
        "4": ("Model Merging & Export", demo_model_merging),
    }

    while True:
        print("\n" + "=" * 60)
        print("LoRA FINE-TUNING")
        print("=" * 60)
        for key, (name, _) in demos.items():
            print(f"  {key}. {name}")
        print("  q. Quit")

        choice = input("\nSelect demo: ").strip().lower()
        if choice == "q":
            print("Goodbye!")
            break
        elif choice in demos:
            demos[choice][1]()
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
