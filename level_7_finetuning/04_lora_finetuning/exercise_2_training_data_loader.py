"""
Exercise 2: Custom Training Data Loader
========================================
Build a custom data loader for fine-tuning. Load JSONL data, tokenize
with model tokenizer, create DataLoader with padding and truncation.

Learning Objectives:
- Load and parse JSONL training data
- Tokenize text for causal language modeling
- Handle padding, truncation, and attention masks
- Create efficient DataLoader with batching

Run:
    python exercise_2_training_data_loader.py
"""

import json
import os
import random
from collections import Counter

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# --- Sample training data ---
SAMPLE_DATA = [
    {"messages": [
        {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
        {"role": "user", "content": "67-year-old male presenting with substernal chest pain radiating to left arm, diaphoresis, and dyspnea. ECG shows ST elevation in leads V1-V4. Troponin I elevated at 3.2 ng/mL."},
        {"role": "assistant", "content": "ICD-10: I21.0 - Acute ST elevation myocardial infarction of anterior wall"}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
        {"role": "user", "content": "78-year-old female with progressive dyspnea on exertion, bilateral lower extremity edema, and orthopnea. BNP 1450 pg/mL. Echo EF 25%."},
        {"role": "assistant", "content": "ICD-10: I50.9 - Heart failure, unspecified"}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
        {"role": "user", "content": "55-year-old obese female with polyuria, polydipsia, blurred vision. Fasting glucose 245 mg/dL, HbA1c 9.1%. No prior diabetes diagnosis."},
        {"role": "assistant", "content": "ICD-10: E11.65 - Type 2 diabetes mellitus with hyperglycemia"}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
        {"role": "user", "content": "42-year-old female with fatigue, weight gain of 20 lbs, cold intolerance, constipation. TSH 18.2 mIU/L, free T4 0.4 ng/dL."},
        {"role": "assistant", "content": "ICD-10: E03.9 - Hypothyroidism, unspecified"}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
        {"role": "user", "content": "73-year-old female with productive cough, fever 101.8F, right lower lobe crackles. CXR shows RLL consolidation. WBC 14,500."},
        {"role": "assistant", "content": "ICD-10: J18.9 - Pneumonia, unspecified organism"}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
        {"role": "user", "content": "68-year-old male with known COPD, worsening dyspnea, increased sputum, wheezing. O2 sat 88% on room air. Using accessory muscles."},
        {"role": "assistant", "content": "ICD-10: J44.1 - COPD with acute exacerbation"}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
        {"role": "user", "content": "29-year-old male with acute RLQ pain, nausea, anorexia, fever 100.9F. Positive McBurney's tenderness, rebound guarding. WBC 15,800."},
        {"role": "assistant", "content": "ICD-10: K35.80 - Unspecified acute appendicitis without abscess"}
    ]},
    {"messages": [
        {"role": "system", "content": "You are a medical coding assistant. Given a clinical note, output the most appropriate ICD-10 code and description."},
        {"role": "user", "content": "74-year-old female with sudden right hemiparesis, facial droop, expressive aphasia. Onset 90 min ago. CT head negative. NIHSS 16."},
        {"role": "assistant", "content": "ICD-10: I63.9 - Cerebral infarction, unspecified"}
    ]},
]


def format_messages_to_text(messages: list, format_style: str = "chatml") -> str:
    """Convert message list to a single text string for tokenization.

    Supports multiple format styles:
    - chatml: <|system|>\n...\n<|user|>\n...\n<|assistant|>\n...
    - alpaca: ### System:\n...\n### Input:\n...\n### Response:\n...
    - simple: System: ...\nUser: ...\nAssistant: ...
    """

    if format_style == "chatml":
        parts = []
        for msg in messages:
            parts.append(f"<|{msg['role']}|>\n{msg['content']}")
        return "\n".join(parts) + "\n<|end|>"

    elif format_style == "alpaca":
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        assistant = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        return f"### System:\n{system}\n\n### Input:\n{user}\n\n### Response:\n{assistant}"

    elif format_style == "simple":
        parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            parts.append(f"{role}: {msg['content']}")
        return "\n\n".join(parts)

    else:
        raise ValueError(f"Unknown format style: {format_style}")


class MedicalCodingDataset:
    """Custom dataset for medical ICD-10 coding fine-tuning.

    Works with or without PyTorch. When torch is available,
    inherits from torch.utils.data.Dataset.
    """

    def __init__(self, data: list, tokenizer=None, max_length: int = 512,
                 format_style: str = "chatml"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_style = format_style

        # Pre-format all texts
        self.texts = [
            format_messages_to_text(sample["messages"], format_style)
            for sample in data
        ]

        # Pre-tokenize if tokenizer available
        self.encodings = None
        if tokenizer is not None:
            self._tokenize()

    def _tokenize(self):
        """Tokenize all texts."""
        self.encodings = []
        for text in self.texts:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt" if TORCH_AVAILABLE else None,
            )
            self.encodings.append(encoding)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.encodings is not None:
            encoding = self.encodings[idx]
            if TORCH_AVAILABLE:
                item = {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                }
                # For causal LM, labels = input_ids (shifted internally by model)
                item["labels"] = item["input_ids"].clone()
                # Mask padding tokens in labels (set to -100)
                item["labels"][item["attention_mask"] == 0] = -100
                return item
            else:
                return encoding
        return {"text": self.texts[idx]}

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        stats = {
            "num_samples": len(self.data),
            "format_style": self.format_style,
            "max_length": self.max_length,
        }

        # Text lengths
        text_lengths = [len(t) for t in self.texts]
        stats["text_length_min"] = min(text_lengths)
        stats["text_length_max"] = max(text_lengths)
        stats["text_length_avg"] = sum(text_lengths) / len(text_lengths)

        # Token lengths (if tokenized)
        if self.encodings and self.tokenizer:
            token_lengths = []
            for enc in self.encodings:
                mask = enc["attention_mask"]
                if TORCH_AVAILABLE:
                    n_tokens = mask.sum().item()
                else:
                    n_tokens = sum(mask)
                token_lengths.append(n_tokens)
            stats["token_length_min"] = min(token_lengths)
            stats["token_length_max"] = max(token_lengths)
            stats["token_length_avg"] = sum(token_lengths) / len(token_lengths)
            stats["truncated"] = sum(1 for t in token_lengths if t >= self.max_length)

        return stats


def load_jsonl(filepath: str) -> list:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: line {line_num} is not valid JSON: {e}")
    return data


def demo_format_styles():
    """Show all supported text formatting styles."""

    print("\n--- Text Format Styles ---")
    sample = SAMPLE_DATA[0]

    for style in ["chatml", "alpaca", "simple"]:
        text = format_messages_to_text(sample["messages"], style)
        print(f"\n  [{style.upper()}]:")
        for line in text.split("\n"):
            print(f"    {line[:100]}{'...' if len(line) > 100 else ''}")


def demo_tokenization():
    """Show the tokenization process."""

    print("\n--- Tokenization Demo ---")

    if not TRANSFORMERS_AVAILABLE:
        print("  transformers not installed. Showing process description:")
        print("""
  1. Format messages into a single text string
  2. Tokenize with model-specific tokenizer
  3. Truncate to max_length (e.g., 512 tokens)
  4. Pad shorter sequences to max_length
  5. Create attention mask (1 for real tokens, 0 for padding)
  6. Set labels = input_ids, with padding masked as -100

  Install: pip install transformers""")
        return

    # Load a small tokenizer
    print("  Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"  Tokenizer: gpt2 (vocab_size={tokenizer.vocab_size})")
    except Exception as e:
        print(f"  Could not load tokenizer: {e}")
        return

    # Tokenize a sample
    sample_text = format_messages_to_text(SAMPLE_DATA[0]["messages"], "chatml")
    print(f"\n  Input text ({len(sample_text)} chars):")
    print(f"    {sample_text[:200]}...")

    encoding = tokenizer(sample_text, truncation=True, max_length=256, padding="max_length")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    real_tokens = sum(attention_mask)

    print(f"\n  Tokenized:")
    print(f"    Total tokens: {len(input_ids)} (max_length=256)")
    print(f"    Real tokens:  {real_tokens}")
    print(f"    Padding:      {len(input_ids) - real_tokens}")
    print(f"    First 20 token IDs: {input_ids[:20]}")
    print(f"    Decoded first 20: '{tokenizer.decode(input_ids[:20])}'")

    # Show attention mask
    print(f"\n  Attention mask (first 30): {attention_mask[:30]}")
    print(f"  Attention mask (last 30):  {attention_mask[-30:]}")


def demo_dataloader():
    """Create and iterate a DataLoader."""

    print("\n--- DataLoader Demo ---")

    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        print("  torch or transformers not installed. Showing pattern:")
        print("""
  from torch.utils.data import DataLoader

  dataset = MedicalCodingDataset(data, tokenizer, max_length=512)
  loader = DataLoader(dataset, batch_size=4, shuffle=True)

  for batch in loader:
      input_ids = batch["input_ids"]      # [4, 512]
      attention_mask = batch["attention_mask"]  # [4, 512]
      labels = batch["labels"]            # [4, 512]
      # Forward pass: outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
      break""")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"  Could not load tokenizer: {e}")
        return

    dataset = MedicalCodingDataset(SAMPLE_DATA, tokenizer, max_length=256)

    # Print stats
    stats = dataset.get_stats()
    print(f"\n  Dataset stats:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.1f}")
        else:
            print(f"    {k}: {v}")

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    print(f"\n  DataLoader: {len(loader)} batches (batch_size=2)")

    # Iterate one batch
    for batch in loader:
        print(f"\n  First batch:")
        print(f"    input_ids shape:      {batch['input_ids'].shape}")
        print(f"    attention_mask shape:  {batch['attention_mask'].shape}")
        print(f"    labels shape:          {batch['labels'].shape}")
        print(f"    Non-padding tokens:    {(batch['attention_mask'] == 1).sum(dim=1).tolist()}")
        print(f"    Label -100 count:      {(batch['labels'] == -100).sum(dim=1).tolist()}")
        break


def main():
    """Build a custom data loader for fine-tuning."""

    print("=" * 60)
    print("Exercise 2: Custom Training Data Loader")
    print("=" * 60)

    # Step 1: Format styles
    print("\n[Step 1] Format Styles")
    demo_format_styles()

    # Step 2: Tokenization
    print(f"\n{'=' * 60}")
    print("[Step 2] Tokenization")
    demo_tokenization()

    # Step 3: Dataset creation
    print(f"\n{'=' * 60}")
    print("[Step 3] Dataset & DataLoader")
    demo_dataloader()

    # Step 4: Loading from JSONL
    print(f"\n{'=' * 60}")
    print("[Step 4] Loading from JSONL Files")

    # Check for existing JSONL files
    data_dir = os.path.dirname(__file__) or "."
    parent_dir = os.path.join(data_dir, "..", "03_data_preparation")
    jsonl_files = []
    for d in [data_dir, parent_dir]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(".jsonl"):
                    jsonl_files.append(os.path.join(d, f))

    if jsonl_files:
        print(f"\n  Found JSONL files:")
        for f in jsonl_files:
            data = load_jsonl(f)
            print(f"    {os.path.basename(f)}: {len(data)} samples")
    else:
        print("\n  No JSONL files found. Use the data preparation exercises")
        print("  (Level 7 Project 03) to generate training data first.")
        print(f"\n  Saving sample data for demonstration...")
        sample_path = os.path.join(data_dir, "sample_training_data.jsonl")
        with open(sample_path, "w") as f:
            for sample in SAMPLE_DATA:
                f.write(json.dumps(sample) + "\n")
        print(f"  Saved {len(SAMPLE_DATA)} samples to: {sample_path}")

    print(f"\n{'=' * 60}")
    print("✓ Data loader exercise complete!")
    print("  Key takeaways:")
    print("  - Format messages into text before tokenization")
    print("  - Use max_length=512 to balance context and memory")
    print("  - Mask padding tokens in labels with -100")
    print("  - Shuffle training data for better convergence")


if __name__ == "__main__":
    main()
