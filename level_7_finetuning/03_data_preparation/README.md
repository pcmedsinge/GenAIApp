# Project 03: Data Preparation for Fine-Tuning

## Overview
Training data quality is the single most important factor in fine-tuning success.
This project covers how to create, clean, validate, and format datasets for
instruction-tuning medical language models.

## Key Concepts

### Training Data Formats
- **Instruction Format**: system/user/assistant message triples
- **Completion Format**: prompt → completion pairs
- **Chat Format**: multi-turn conversation logs
- **JSONL**: one JSON object per line — the standard for training data

### Quality Filtering
- Remove duplicates (exact and near-duplicate)
- Validate completeness (all required fields present)
- Check token lengths (too short = low signal, too long = truncated)
- Score quality with LLM-as-judge
- Detect and mitigate bias in labels

### Synthetic Data Generation
- Use GPT-4o to generate realistic clinical scenarios
- Prompt engineering for diversity and accuracy
- Validate generated data against medical standards
- Augment small datasets with paraphrased variants

## What You'll Build
| Demo/Exercise | Description |
|---|---|
| Demo 1 | Instruction format for medical tasks |
| Demo 2 | Data quality filtering pipeline |
| Demo 3 | Synthetic data generation with GPT-4o |
| Demo 4 | Dataset statistics and analysis |
| Exercise 1 | ICD-10 coding dataset creation |
| Exercise 2 | Data augmentation via LLM paraphrasing |
| Exercise 3 | Quality filter pipeline |
| Exercise 4 | Dataset splitting and JSONL export |

## Prerequisites
```bash
pip install openai
```

## Running
```bash
python main.py              # Interactive demo menu
python exercise_1_icd10_dataset.py
python exercise_2_data_augmentation.py
python exercise_3_quality_filter.py
python exercise_4_dataset_splits.py
```

## Data Format Example
```json
{"messages": [
  {"role": "system", "content": "You are a medical coding assistant."},
  {"role": "user", "content": "Patient presents with acute chest pain..."},
  {"role": "assistant", "content": "ICD-10: I20.0 - Unstable angina"}
]}
```

## Tips
- Aim for 500–5000 high-quality examples for instruction tuning
- Diverse inputs prevent overfitting to narrow patterns
- Always hold out a test set the model never sees during training
