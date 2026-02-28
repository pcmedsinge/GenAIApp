# Level 7, Project 02: HuggingFace Ecosystem

## Overview
Explore the HuggingFace **Transformers** library — the open-source standard
for working with pre-trained language models.  Learn pipelines, tokenizers,
the Model Hub, and apply biomedical NER models to clinical text.

## Why HuggingFace for Healthcare
- **Model variety**: thousands of biomedical / clinical NLP models available.
- **Local inference**: run models on your own hardware — no data leaves the network.
- **Fine-tuning ready**: the same library used for inference also supports training.
- **Community**: active research community sharing medical NLP models and datasets.

## Prerequisites
| Requirement | Install command |
|---|---|
| Transformers | `pip install transformers` |
| PyTorch (CPU) | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Datasets (optional) | `pip install datasets` |
| SentencePiece | `pip install sentencepiece protobuf` |

> **Tip**: All demos use lightweight models that run on CPU.  No GPU required.

## Demos (main.py)
| # | Demo | What you learn |
|---|---|---|
| 1 | Transformers Pipeline | `pipeline()` for generation, classification, NER |
| 2 | Tokenization | How tokenizers encode/decode text; compare tokenizers |
| 3 | Model Hub | Search models, read model cards, discover medical models |
| 4 | Medical NER | Extract clinical entities from narrative text |

## Exercises
| # | File | Skills |
|---|---|---|
| 1 | exercise_1_text_classification.py | Classify clinical notes by specialty |
| 2 | exercise_2_medical_ner.py | Extract diseases, meds, dosages from notes |
| 3 | exercise_3_model_comparison.py | Compare small models on speed, quality, memory |
| 4 | exercise_4_custom_pipeline.py | Chain NER → classification → summary |

## Quick Start
```bash
# 1. Install dependencies
pip install transformers torch sentencepiece protobuf

# 2. Run the demos
python main.py

# 3. Try an exercise
python exercise_1_text_classification.py
```

## Key Concept — Pipelines
A HuggingFace **pipeline** wraps tokenization, inference, and post-processing
into a single callable:

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="...")
result = classifier("Patient has acute chest pain.")
```

Pipelines auto-download models from the Hub on first use and cache them locally.
