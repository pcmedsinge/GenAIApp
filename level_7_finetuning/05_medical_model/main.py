"""
Medical Model Capstone — Main Demo
===================================
End-to-end medical model pipeline: data preparation, fine-tuning,
evaluation, and local deployment for ICD-10 coding.

Run:
    python main.py
"""

from openai import OpenAI
import json
import os
import random
import math
import time
from collections import Counter, defaultdict

# --- Attempt heavy imports ---
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


# --- ICD-10 Code Reference ---
ICD10_CODES = {
    "I21.0": "Acute ST elevation myocardial infarction of anterior wall",
    "I21.19": "ST elevation myocardial infarction involving other coronary artery of inferior wall",
    "I50.9": "Heart failure, unspecified",
    "I48.0": "Paroxysmal atrial fibrillation",
    "J18.9": "Pneumonia, unspecified organism",
    "J44.1": "Chronic obstructive pulmonary disease with acute exacerbation",
    "J45.41": "Moderate persistent asthma with acute exacerbation",
    "E11.9": "Type 2 diabetes mellitus without complications",
    "E11.65": "Type 2 diabetes mellitus with hyperglycemia",
    "E03.9": "Hypothyroidism, unspecified",
    "N39.0": "Urinary tract infection, site not specified",
    "K35.80": "Unspecified acute appendicitis without abscess",
    "K85.9": "Acute pancreatitis, unspecified",
    "I63.9": "Cerebral infarction, unspecified",
    "G43.909": "Migraine, unspecified, not intractable, without status migrainosus",
    "M54.5": "Low back pain",
    "S72.001A": "Fracture of unspecified part of neck of right femur, initial encounter",
    "A41.9": "Sepsis, unspecified organism",
    "N17.9": "Acute kidney failure, unspecified",
    "R50.9": "Fever, unspecified",
}


# ============================================================
# DEMO 1: ICD-10 Dataset Preparation
# ============================================================
def demo_dataset_preparation():
    """Prepare training data for ICD-10 coding from clinical notes."""

    print("\n" + "=" * 60)
    print("DEMO 1: ICD-10 Dataset Preparation")
    print("=" * 60)

    client = OpenAI()

    # --- Generate synthetic clinical notes ---
    print("\n--- Generating Synthetic Clinical Notes with GPT-4o ---")

    codes_to_generate = list(ICD10_CODES.items())[:10]  # first 10 codes

    all_samples = []

    for code, description in codes_to_generate:
        prompt = f"""Generate 3 realistic but synthetic clinical note snippets that would be coded as:
ICD-10: {code} - {description}

Each note should:
- Be 2-4 sentences
- Include patient demographics (age, sex)
- Include relevant symptoms, vitals, or lab values
- Be distinct from each other (different presentations)

Return as JSON array with objects having "clinical_note" field only.
Return ONLY the JSON array, no markdown."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Generate realistic synthetic clinical data. All patients are fictional."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,
            )

            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            notes = json.loads(raw)
            for note_obj in notes:
                sample = {
                    "messages": [
                        {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
                        {"role": "user", "content": note_obj["clinical_note"]},
                        {"role": "assistant", "content": f"ICD-10: {code} - {description}"}
                    ]
                }
                all_samples.append(sample)

            print(f"  ✓ {code}: {len(notes)} notes generated")
        except Exception as e:
            print(f"  ✗ {code}: Error — {e}")

    print(f"\n--- Dataset Summary ---")
    print(f"  Total samples: {len(all_samples)}")

    # Show distribution
    code_counts = Counter()
    for s in all_samples:
        asst = next(m for m in s["messages"] if m["role"] == "assistant")
        c = asst["content"].split(":")[1].strip().split(" - ")[0].strip()
        code_counts[c] += 1

    print(f"  Unique codes: {len(code_counts)}")
    for c, n in code_counts.most_common():
        print(f"    {c}: {n} samples")

    # Show sample
    if all_samples:
        print(f"\n--- Sample Training Example ---")
        sample = all_samples[0]
        for msg in sample["messages"]:
            role = msg["role"].upper()
            content = msg["content"][:150]
            print(f"  [{role}]: {content}{'...' if len(msg['content']) > 150 else ''}")

    # Save to file
    output_path = os.path.join(os.path.dirname(__file__), "icd10_training_data.jsonl")
    with open(output_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    print(f"\n  Saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path):,} bytes")


# ============================================================
# DEMO 2: Fine-Tuning Pipeline
# ============================================================
def demo_finetuning_pipeline():
    """Complete LoRA fine-tuning pipeline."""

    print("\n" + "=" * 60)
    print("DEMO 2: Complete Fine-Tuning Pipeline")
    print("=" * 60)

    all_available = TRANSFORMERS_AVAILABLE and PEFT_AVAILABLE and DATASETS_AVAILABLE and TORCH_AVAILABLE

    if all_available:
        print("\n--- Attempting Live Fine-Tuning Pipeline ---")

        # Check for training data
        data_path = os.path.join(os.path.dirname(__file__), "icd10_training_data.jsonl")
        if not os.path.exists(data_path):
            print(f"  Training data not found at {data_path}")
            print("  Run Demo 1 first to generate training data.")
            print("  Falling back to code pattern display...\n")
            all_available = False

    if all_available:
        try:
            # Load training data
            print("  Loading training data...")
            samples = []
            with open(data_path) as f:
                for line in f:
                    samples.append(json.loads(line))
            print(f"  Loaded {len(samples)} samples")

            # Split
            random.shuffle(samples)
            split = int(len(samples) * 0.8)
            train_data = samples[:split]
            eval_data = samples[split:]
            print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")

            # Load model
            model_name = "microsoft/phi-2"
            print(f"\n  Loading model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )

            # Apply LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            print("\n  Pipeline ready — full training would proceed from here.")
            print("  (Skipping actual training to save time/compute)")

        except Exception as e:
            print(f"\n  Pipeline setup error: {e}")
            print("  Showing code pattern instead...")
            all_available = False

    # Always show the complete pattern
    print("\n--- Complete Fine-Tuning Pipeline Code ---")
    print("""
  # === STEP 1: Load and prepare data ===
  import json
  from datasets import Dataset

  samples = []
  with open("icd10_training_data.jsonl") as f:
      for line in f:
          samples.append(json.loads(line))

  # Format for training
  def format_for_training(sample):
      text = ""
      for msg in sample["messages"]:
          text += f"<|{msg['role']}|>\\n{msg['content']}\\n"
      return {"text": text}

  dataset = Dataset.from_list([format_for_training(s) for s in samples])
  dataset = dataset.train_test_split(test_size=0.2, seed=42)

  # === STEP 2: Load model with QLoRA ===
  from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
  import torch

  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
  )
  model = AutoModelForCausalLM.from_pretrained(
      "mistralai/Mistral-7B-v0.1",
      quantization_config=bnb_config,
      device_map="auto",
  )
  tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
  tokenizer.pad_token = tokenizer.eos_token

  # === STEP 3: Configure LoRA ===
  from peft import LoraConfig, get_peft_model, TaskType

  lora_config = LoraConfig(
      task_type=TaskType.CAUSAL_LM,
      r=16, lora_alpha=32, lora_dropout=0.05,
      target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
  )
  model = get_peft_model(model, lora_config)

  # === STEP 4: Tokenize ===
  def tokenize(example):
      return tokenizer(example["text"], truncation=True, max_length=512, padding="max_length")

  tokenized = dataset.map(tokenize, batched=True)

  # === STEP 5: Train ===
  from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

  trainer = Trainer(
      model=model,
      args=TrainingArguments(
          output_dir="./icd10-lora",
          num_train_epochs=3,
          per_device_train_batch_size=4,
          gradient_accumulation_steps=4,
          learning_rate=2e-4,
          warmup_steps=50,
          logging_steps=10,
          save_steps=100,
          eval_strategy="steps",
          eval_steps=100,
          fp16=True,
          report_to="none",
      ),
      train_dataset=tokenized["train"],
      eval_dataset=tokenized["test"],
      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
  )
  trainer.train()

  # === STEP 6: Save adapter ===
  model.save_pretrained("./icd10-lora-adapter")
  tokenizer.save_pretrained("./icd10-lora-adapter")
""")


# ============================================================
# DEMO 3: Model Evaluation
# ============================================================
def demo_model_evaluation():
    """Comprehensive evaluation on test set."""

    print("\n" + "=" * 60)
    print("DEMO 3: Comprehensive Model Evaluation")
    print("=" * 60)

    # --- Simulate evaluation results ---
    print("\n--- Simulated Evaluation Results ---")
    print("(In production, run actual inference on held-out test set)\n")

    random.seed(42)

    test_codes = list(ICD10_CODES.keys())[:12]
    test_size = 120  # 10 per code

    # Simulate predictions
    results = []
    for i in range(test_size):
        true_code = test_codes[i % len(test_codes)]
        # Simulate model accuracy ~78%
        if random.random() < 0.78:
            pred_code = true_code
        else:
            pred_code = random.choice(test_codes)
        results.append({"true": true_code, "pred": pred_code})

    # --- Overall Accuracy ---
    correct = sum(1 for r in results if r["true"] == r["pred"])
    accuracy = correct / len(results)
    print(f"  Overall Accuracy: {accuracy:.1%} ({correct}/{len(results)})")

    # --- Per-code performance ---
    print(f"\n--- Per-Code Performance ---")
    print(f"  {'Code':12s} {'Correct':>8s} {'Total':>6s} {'Acc':>6s} {'Description'}")
    print("  " + "-" * 80)

    code_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0})
    for r in results:
        code_metrics[r["true"]]["total"] += 1
        if r["true"] == r["pred"]:
            code_metrics[r["true"]]["tp"] += 1
        else:
            code_metrics[r["true"]]["fn"] += 1
            code_metrics[r["pred"]]["fp"] += 1

    for code in test_codes:
        m = code_metrics[code]
        acc = m["tp"] / m["total"] if m["total"] > 0 else 0
        desc = ICD10_CODES.get(code, "Unknown")[:40]
        print(f"  {code:12s} {m['tp']:8d} {m['total']:6d} {acc:6.0%} {desc}")

    # --- F1 Scores ---
    print(f"\n--- F1 Scores (Macro & Weighted) ---")
    f1_scores = []
    for code in test_codes:
        m = code_metrics[code]
        precision = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0
        recall = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    weighted_f1 = sum(
        f1 * code_metrics[code]["total"]
        for f1, code in zip(f1_scores, test_codes)
    ) / len(results)

    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")

    # --- Confusion matrix (simplified) ---
    print(f"\n--- Confusion Matrix (top 6 codes) ---")
    top_codes = test_codes[:6]
    short = [c.split(".")[0] + "." + c.split(".")[1][:2] if "." in c else c[:6] for c in top_codes]

    print(f"  {'':8s}", end="")
    for s in short:
        print(f" {s:>7s}", end="")
    print("  (predicted)")

    confusion = defaultdict(lambda: defaultdict(int))
    for r in results:
        if r["true"] in top_codes:
            confusion[r["true"]][r["pred"]] += 1

    for true_code, s in zip(top_codes, short):
        print(f"  {s:8s}", end="")
        for pred_code in top_codes:
            count = confusion[true_code].get(pred_code, 0)
            if count > 0:
                print(f" {count:7d}", end="")
            else:
                print(f" {'·':>7s}", end="")
        print()

    # --- Failure analysis ---
    print(f"\n--- Failure Analysis ---")
    failures = [r for r in results if r["true"] != r["pred"]]
    print(f"  Total failures: {len(failures)}/{len(results)}")

    failure_types = Counter()
    for f in failures:
        true_cat = f["true"].split(".")[0]
        pred_cat = f["pred"].split(".")[0]
        if true_cat == pred_cat:
            failure_types["Same category, wrong specific code"] += 1
        else:
            failure_types["Wrong ICD-10 category entirely"] += 1

    for ftype, count in failure_types.most_common():
        print(f"    {ftype}: {count} ({count/len(failures)*100:.0f}%)")


# ============================================================
# DEMO 4: Local Deployment
# ============================================================
def demo_local_deployment():
    """Deploy fine-tuned model via Ollama for HIPAA-safe inference."""

    print("\n" + "=" * 60)
    print("DEMO 4: Local Deployment via Ollama")
    print("=" * 60)

    print("""
=== Deployment Pipeline ===

  Fine-tuned Model (HF format)
        │
        ▼
  Merge LoRA → Full Model
        │
        ▼
  Convert to GGUF (quantize)
        │
        ▼
  Create Ollama Modelfile
        │
        ▼
  Import into Ollama
        │
        ▼
  Serve via REST API
        │
        ▼
  HIPAA-Safe Local Inference
""")

    # --- Step 1: Merge and convert ---
    print("--- Step 1: Merge LoRA Weights ---")
    print("""
  from peft import PeftModel
  from transformers import AutoModelForCausalLM

  base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
  model = PeftModel.from_pretrained(base, "./icd10-lora-adapter")
  merged = model.merge_and_unload()
  merged.save_pretrained("./icd10-model-merged")
""")

    # --- Step 2: GGUF conversion ---
    print("--- Step 2: Convert to GGUF ---")
    print("""
  # Clone llama.cpp for conversion tools
  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp

  # Convert to GGUF with 4-bit quantization
  python convert_hf_to_gguf.py ../icd10-model-merged \\
      --outfile ../icd10-coder.Q4_K_M.gguf \\
      --outtype q4_K_M

  # File sizes:
  #   Original FP16:  ~14 GB
  #   Q4_K_M:         ~4.4 GB
  #   Q8_0:           ~7.7 GB
""")

    # --- Step 3: Ollama Modelfile ---
    print("--- Step 3: Create Ollama Modelfile ---")
    modelfile_content = '''FROM ./icd10-coder.Q4_K_M.gguf

SYSTEM """You are a medical coding assistant specialized in ICD-10 coding.
Given a clinical note, identify the most appropriate ICD-10 diagnosis code.
Always output in the format: ICD-10: [CODE] - [DESCRIPTION]
Be specific — avoid unspecified codes when clinical detail supports a specific code."""

PARAMETER temperature 0.1
PARAMETER num_ctx 4096
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

TEMPLATE """{{ if .System }}{{ .System }}{{ end }}
{{ if .Prompt }}Clinical Note: {{ .Prompt }}{{ end }}
{{ .Response }}"""'''

    print(f"  Modelfile contents:\n")
    for line in modelfile_content.split("\n"):
        print(f"    {line}")

    # --- Step 4: Import and test ---
    print("\n--- Step 4: Import into Ollama ---")
    print("""
  # Create the model
  ollama create icd10-coder -f Modelfile

  # Verify
  ollama list | grep icd10-coder

  # Test with a clinical note
  ollama run icd10-coder "67-year-old male presents with acute substernal \\
  chest pain radiating to left arm. ECG shows ST elevation in leads \\
  II, III, aVF. Troponin I elevated at 2.4 ng/mL."
""")

    # --- Step 5: API usage ---
    print("--- Step 5: REST API Integration ---")
    print("""
  import requests

  def code_clinical_note(note: str) -> str:
      response = requests.post(
          "http://localhost:11434/api/generate",
          json={
              "model": "icd10-coder",
              "prompt": note,
              "stream": False,
              "options": {"temperature": 0.1}
          }
      )
      return response.json()["response"]

  # Usage
  result = code_clinical_note(
      "45-year-old female with fatigue, weight gain, cold intolerance. "
      "TSH elevated at 12.5, free T4 low at 0.6."
  )
  print(result)
  # → ICD-10: E03.9 - Hypothyroidism, unspecified
""")

    # --- Benchmark ---
    print("--- Step 6: Performance Benchmark ---")
    print("""
  Typical benchmark results (Mistral 7B Q4_K_M on consumer hardware):

  ┌─────────────────┬──────────────────┬────────────────┐
  │ Metric          │ GPU (RTX 3080)   │ CPU (i7-12700) │
  ├─────────────────┼──────────────────┼────────────────┤
  │ Tokens/sec      │ ~40-60 t/s       │ ~8-12 t/s      │
  │ Time to first   │ ~200ms           │ ~800ms         │
  │ Memory          │ ~5 GB VRAM       │ ~6 GB RAM      │
  │ Latency (avg)   │ ~1.5 sec         │ ~5 sec         │
  └─────────────────┴──────────────────┴────────────────┘

  HIPAA Compliance Notes:
  ✓ All data stays on local machine
  ✓ No API calls to external services
  ✓ No patient data leaves the network
  ✓ Model weights stored locally
  ✓ Audit logging via application layer
""")


# ============================================================
# Main Menu
# ============================================================
def main():
    """Interactive demo menu for Medical Model Capstone."""
    demos = {
        "1": ("ICD-10 Dataset Preparation (requires API key)", demo_dataset_preparation),
        "2": ("Fine-Tuning Pipeline", demo_finetuning_pipeline),
        "3": ("Model Evaluation", demo_model_evaluation),
        "4": ("Local Deployment via Ollama", demo_local_deployment),
    }

    while True:
        print("\n" + "=" * 60)
        print("MEDICAL MODEL CAPSTONE — ICD-10 Coding")
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
