# Project 04: LoRA Fine-Tuning

## Overview
LoRA (Low-Rank Adaptation) enables fine-tuning large language models on consumer
hardware by training only a small number of additional parameters. This project
covers LoRA configuration, training, and model merging.

## Key Concepts

### LoRA (Low-Rank Adaptation)
Instead of updating all model weights, LoRA injects trainable low-rank matrices
into each transformer layer. This reduces trainable parameters by 99%+ while
preserving model quality.

### QLoRA
Combines LoRA with 4-bit quantization — fine-tune a 7B model on a single GPU
with 6GB VRAM. Uses `bitsandbytes` for NF4 quantization.

### PEFT Library
Hugging Face's Parameter-Efficient Fine-Tuning library provides:
- `LoraConfig` — configure rank, alpha, dropout, target modules
- `get_peft_model()` — wrap any model with LoRA adapters
- `PeftModel.from_pretrained()` — load trained adapters

### Training Configuration
- **Rank (r)**: 4–64, controls adapter capacity (8–16 typical)
- **Alpha**: scaling factor, usually 2× rank
- **Dropout**: 0.0–0.1 for regularization
- **Target modules**: which layers to adapt (q_proj, v_proj, etc.)

## What You'll Build
| Demo/Exercise | Description |
|---|---|
| Demo 1 | LoRA concept explained with code |
| Demo 2 | Complete training setup |
| Demo 3 | Training loop walkthrough |
| Demo 4 | Model merging and export |
| Exercise 1 | LoRA hyperparameter experiments |
| Exercise 2 | Custom training data loader |
| Exercise 3 | Training monitor with metrics |
| Exercise 4 | Model evaluation and comparison |

## Prerequisites
```bash
pip install transformers peft datasets accelerate bitsandbytes
pip install openai  # for comparison baselines
```

## Running
```bash
python main.py
python exercise_1_lora_config.py
python exercise_2_training_data_loader.py
python exercise_3_training_monitor.py
python exercise_4_model_evaluation.py
```

## Hardware Requirements
- **QLoRA 7B model**: ~6GB VRAM (single GPU)
- **LoRA 7B model**: ~14GB VRAM
- **CPU-only**: works but very slow — demos show patterns regardless

## Tips
- Start with rank=8, alpha=16 — increase only if quality is low
- Monitor validation loss to detect overfitting early
- Merge weights before deployment to eliminate adapter overhead
