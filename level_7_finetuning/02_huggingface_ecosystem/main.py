"""
Level 7 – Project 02: HuggingFace Ecosystem
============================================
Work with the Transformers library — pipelines, tokenizers, the Model Hub,
and biomedical NER — all running locally on CPU.

Demos
-----
1. Transformers Pipeline  – text generation, classification, NER via pipeline()
2. Tokenization           – encode/decode, special tokens, tokenizer comparison
3. Model Hub              – discover and inspect models programmatically
4. Medical NER            – extract clinical entities from narrative text

Prerequisites
-------------
    pip install transformers torch sentencepiece protobuf
"""

import json
import sys
import time
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
def _check_deps():
    missing = []
    for pkg in ("transformers", "torch"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"ERROR: missing packages: {missing}")
        print("Install:  pip install transformers torch sentencepiece protobuf")
        sys.exit(1)

_check_deps()

from transformers import (       # noqa: E402
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)


# ============================================================
# DEMO 1: Transformers Pipeline
# ============================================================
def demo_transformers_pipeline():
    """Show how pipeline() provides one-line inference for common NLP tasks."""
    print("\n" + "=" * 60)
    print("DEMO 1: Transformers Pipeline — One-Line NLP Inference")
    print("=" * 60)

    # --- Text Generation ---
    print("\n[1] Text Generation Pipeline")
    print("    Model: distilgpt2 (small, fast, CPU-friendly)")
    try:
        gen = pipeline("text-generation", model="distilgpt2", max_new_tokens=60)
        prompt = "The patient was admitted with"
        start = time.time()
        result = gen(prompt, num_return_sequences=1, do_sample=True, temperature=0.7)
        elapsed = time.time() - start
        print(f"    Prompt: \"{prompt}\"")
        print(f"    Output: \"{result[0]['generated_text']}\"")
        print(f"    Time: {elapsed:.2f}s")
    except Exception as exc:
        print(f"    Skipped — {exc}")

    # --- Sentiment / Text Classification ---
    print("\n[2] Text Classification Pipeline")
    print("    Model: distilbert-base-uncased-finetuned-sst-2-english")
    try:
        clf = pipeline("text-classification",
                        model="distilbert-base-uncased-finetuned-sst-2-english")
        samples = [
            "The patient is recovering well and vital signs are stable.",
            "The prognosis is poor with significant risk of complications.",
            "Lab results are within normal limits.",
        ]
        for text in samples:
            res = clf(text)[0]
            print(f"    \"{text[:60]}...\"  → {res['label']} ({res['score']:.3f})")
    except Exception as exc:
        print(f"    Skipped — {exc}")

    # --- Named Entity Recognition ---
    print("\n[3] Named Entity Recognition Pipeline")
    print("    Model: dslim/bert-base-NER")
    try:
        ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
        text = "Dr. Smith at Johns Hopkins prescribed Metformin for the patient from Baltimore."
        entities = ner(text)
        print(f"    Text: \"{text}\"")
        print("    Entities:")
        for ent in entities:
            print(f"      • {ent['word']:<20} {ent['entity_group']:<8} score={ent['score']:.3f}")
    except Exception as exc:
        print(f"    Skipped — {exc}")

    # --- Zero-Shot Classification ---
    print("\n[4] Zero-Shot Classification Pipeline")
    print("    Model: facebook/bart-large-mnli")
    try:
        zsc = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        text = "Patient presents with polyuria, polydipsia, and unexplained weight loss."
        labels = ["endocrinology", "cardiology", "pulmonology", "neurology", "gastroenterology"]
        result = zsc(text, candidate_labels=labels)
        print(f"    Text: \"{text}\"")
        print("    Predicted specialties:")
        for label, score in zip(result["labels"], result["scores"]):
            bar = "█" * int(score * 30)
            print(f"      {label:<20} {score:.3f} {bar}")
    except Exception as exc:
        print(f"    Skipped — {exc}")


# ============================================================
# DEMO 2: Tokenization
# ============================================================
def demo_tokenization():
    """Show how tokenizers work: encode, decode, special tokens, comparison."""
    print("\n" + "=" * 60)
    print("DEMO 2: Tokenization — How Models See Text")
    print("=" * 60)

    sample = "The patient was prescribed Metformin 500mg twice daily for Type 2 diabetes mellitus."

    # --- Basic encode/decode ---
    print(f"\n[1] Sample text:\n    \"{sample}\"\n")

    tokenizer_names = [
        "bert-base-uncased",
        "gpt2",
        "google/flan-t5-small",
    ]

    for name in tokenizer_names:
        print(f"  --- Tokenizer: {name} ---")
        try:
            tok = AutoTokenizer.from_pretrained(name)
            encoded = tok.encode(sample)
            tokens = tok.convert_ids_to_tokens(encoded)
            decoded = tok.decode(encoded)

            print(f"    Token count : {len(encoded)}")
            print(f"    Token IDs   : {encoded[:15]}{'...' if len(encoded)>15 else ''}")
            print(f"    Tokens      : {tokens[:15]}{'...' if len(tokens)>15 else ''}")
            print(f"    Decoded     : \"{decoded[:80]}...\"")
            print(f"    Vocab size  : {tok.vocab_size:,}")
            print()
        except Exception as exc:
            print(f"    Skipped — {exc}\n")

    # --- Special tokens ---
    print("[2] Special tokens (bert-base-uncased):")
    try:
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        print(f"    [CLS] = {tok.cls_token_id}, [SEP] = {tok.sep_token_id}, "
              f"[PAD] = {tok.pad_token_id}, [MASK] = {tok.mask_token_id}")
        enc = tok("Hello world", return_tensors="pt")
        print(f"    Encoded 'Hello world': {enc['input_ids'].tolist()}")
    except Exception as exc:
        print(f"    Skipped — {exc}")

    # --- Medical term tokenization ---
    print("\n[3] How medical terms get tokenized (gpt2):")
    try:
        tok = AutoTokenizer.from_pretrained("gpt2")
        terms = ["acetaminophen", "thrombocytopenia", "electrocardiogram", "HbA1c", "SGLT2"]
        for term in terms:
            ids = tok.encode(term)
            subtokens = tok.convert_ids_to_tokens(ids)
            print(f"    {term:<25} → {subtokens}")
    except Exception as exc:
        print(f"    Skipped — {exc}")


# ============================================================
# DEMO 3: Model Hub
# ============================================================
def demo_model_hub():
    """Show how to discover and inspect models on the HuggingFace Hub."""
    print("\n" + "=" * 60)
    print("DEMO 3: Model Hub — Discovering Healthcare Models")
    print("=" * 60)

    try:
        from huggingface_hub import HfApi, ModelFilter
        api = HfApi()
        hub_available = True
    except ImportError:
        print("\n    (huggingface_hub not installed — showing curated list instead)")
        print("    Install:  pip install huggingface_hub")
        hub_available = False

    # --- Curated list of healthcare-relevant models ---
    print("\n[1] Healthcare-Relevant Models on HuggingFace:")
    curated = [
        ("emilyalsentzer/Bio_ClinicalBERT", "Clinical NLP — trained on MIMIC-III notes"),
        ("allenai/scibert_scivocab_uncased", "Scientific literature NLP"),
        ("dmis-lab/biobert-base-cased-v1.2", "Biomedical text mining (PubMed + PMC)"),
        ("d4data/biomedical-ner-all", "Biomedical NER — diseases, chemicals, genes"),
        ("samrawal/bert-base-uncased_clinical-ner", "Clinical NER — problems, treatments, tests"),
        ("facebook/bart-large-mnli", "Zero-shot classification (useful for triage)"),
        ("google/flan-t5-small", "Instruction-tuned T5 — lightweight, versatile"),
    ]
    for model_id, desc in curated:
        print(f"    • {model_id}")
        print(f"      {desc}")

    # --- Search the Hub (if available) ---
    if hub_available:
        print("\n[2] Live Hub search — 'biomedical NER' models:")
        try:
            models = api.list_models(
                search="biomedical ner",
                sort="downloads",
                direction=-1,
                limit=5,
            )
            for m in models:
                downloads = getattr(m, "downloads", "N/A")
                print(f"    • {m.id}  (downloads: {downloads})")
        except Exception as exc:
            print(f"    Search failed: {exc}")

        print("\n[3] Model card snippet — d4data/biomedical-ner-all:")
        try:
            info = api.model_info("d4data/biomedical-ner-all")
            print(f"    Pipeline tag : {info.pipeline_tag}")
            print(f"    Library      : {info.library_name}")
            print(f"    Downloads    : {getattr(info, 'downloads', 'N/A')}")
            tags = getattr(info, "tags", [])
            print(f"    Tags         : {tags[:8]}")
        except Exception as exc:
            print(f"    Lookup failed: {exc}")
    else:
        print("\n    Install huggingface_hub to enable live Hub search.")

    print("\n[4] Tip: Always check the model card for:")
    print("    • Training data (was it trained on clinical text?)")
    print("    • Intended use / limitations")
    print("    • Evaluation metrics")
    print("    • License (some medical models restrict commercial use)")


# ============================================================
# DEMO 4: Medical NER
# ============================================================
def demo_medical_ner():
    """Extract clinical entities from narrative text using a biomedical NER model."""
    print("\n" + "=" * 60)
    print("DEMO 4: Medical NER — Extracting Clinical Entities")
    print("=" * 60)

    # We try a biomedical NER model; fall back to general NER if unavailable
    ner_models = [
        "d4data/biomedical-ner-all",
        "dslim/bert-base-NER",
    ]

    ner_pipe = None
    model_used = None
    for model_name in ner_models:
        try:
            print(f"\n    Loading model: {model_name} ...")
            ner_pipe = pipeline("ner", model=model_name, aggregation_strategy="simple")
            model_used = model_name
            print(f"    ✓ Loaded successfully.")
            break
        except Exception as exc:
            print(f"    ✗ Could not load {model_name}: {exc}")

    if ner_pipe is None:
        print("\n⚠  No NER model could be loaded. Install transformers + torch.")
        return

    clinical_notes = [
        (
            "Patient is a 67-year-old male with a history of hypertension, "
            "type 2 diabetes mellitus, and chronic kidney disease stage 3. "
            "He was admitted for acute decompensated heart failure. "
            "Started on furosemide 40mg IV BID and lisinopril 10mg daily."
        ),
        (
            "MRI of the brain shows a 2.3 cm enhancing mass in the left "
            "temporal lobe concerning for glioblastoma multiforme. "
            "Neurosurgery consultation requested for biopsy."
        ),
        (
            "The patient underwent laparoscopic cholecystectomy for acute "
            "cholecystitis. Post-operative course was uncomplicated. "
            "Discharged on ciprofloxacin 500mg PO BID for 7 days."
        ),
    ]

    for i, note in enumerate(clinical_notes, 1):
        print(f"\n--- Clinical Note {i} ---")
        print(f"    \"{note[:100]}...\"\n")

        entities = ner_pipe(note)
        if entities:
            print(f"    {'Entity':<30} {'Type':<18} {'Score'}")
            print("    " + "-" * 60)
            for ent in entities:
                word = ent["word"][:28]
                group = ent["entity_group"]
                score = ent["score"]
                print(f"    {word:<30} {group:<18} {score:.3f}")
        else:
            print("    (no entities detected)")

    print(f"\n    Model used: {model_used}")
    print("    Note: Biomedical NER models distinguish disease, chemical,")
    print("    gene, species entities — more granular than general NER.")


# ============================================================
# Main Menu
# ============================================================
def main():
    """Interactive menu for Level 7, Project 02 demos."""
    demos = {
        "1": ("Transformers Pipeline", demo_transformers_pipeline),
        "2": ("Tokenization", demo_tokenization),
        "3": ("Model Hub", demo_model_hub),
        "4": ("Medical NER", demo_medical_ner),
    }

    while True:
        print("\n" + "=" * 60)
        print("LEVEL 7 · PROJECT 02 — HuggingFace Ecosystem")
        print("=" * 60)
        for key, (title, _) in demos.items():
            print(f"  {key}. {title}")
        print("  q. Quit")

        choice = input("\nSelect a demo (1-4, q): ").strip().lower()
        if choice == "q":
            print("Goodbye!")
            break
        if choice in demos:
            demos[choice][1]()
        else:
            print("Invalid choice — try again.")


if __name__ == "__main__":
    main()
