# Level 7: Fine-Tuning & Open-Source Models
**Customize models for your domain and run locally for HIPAA compliance**

## Overview

Sometimes you need models customized for your domain, or running on-premise for
HIPAA compliance. This level covers local model deployment with Ollama, the
HuggingFace ecosystem, training data preparation, and LoRA/QLoRA fine-tuning.

### When to Fine-Tune vs Prompt

```
Use Prompting When:          Use Fine-Tuning When:
- Task is well-defined       - Need consistent formatting
- Few-shot examples work     - Prompt is getting too long
- Quick iteration needed     - Domain-specific jargon
- Low volume                 - High volume / cost matters
                             - On-premise deployment needed
```

### Local Models for Healthcare
Running models locally (Ollama) means:
- **No data leaves your network** → HIPAA-friendly
- **No API costs** → unlimited inference
- **Full control** → customize, quantize, fine-tune
- **Tradeoff**: Smaller models, need your own hardware

## Prerequisites
- **Levels 1-6 Complete**: Understanding of all AI patterns
- **Ollama installed**: `curl -fsSL https://ollama.com/install.sh | sh`
- **8GB+ RAM**: For running local models
- **GPU optional**: Speeds up inference and is needed for fine-tuning

## Projects

### 01_local_llms — Running Models Locally with Ollama
- Install and configure Ollama
- Pull and run models (Llama 3, Mistral, medical models)
- OpenAI-compatible API (same code works!)
- **Healthcare Example**: Local medical Q&A with no data leaving your network

### 02_huggingface_ecosystem — The Open-Source AI Hub
- Transformers library: loading models and tokenizers
- Model Hub: finding and evaluating models
- Inference pipelines: text generation, classification, NER
- **Healthcare Example**: Medical NER for clinical entity extraction

### 03_data_preparation — Building Training Datasets
- Instruction tuning dataset format (system/user/assistant)
- Data quality filtering and deduplication
- Synthetic data generation with LLMs
- **Healthcare Example**: Create ICD-10 coding training data

### 04_lora_finetuning — Efficient Model Customization
- LoRA and QLoRA: parameter-efficient fine-tuning
- Training with HuggingFace PEFT + Accelerate
- Overfitting prevention: validation, early stopping
- **Healthcare Example**: Fine-tune for clinical note formatting

### 05_medical_model — Capstone: Custom Healthcare Model
- Fine-tune for ICD-10 code prediction
- Evaluation: accuracy, F1, confusion matrix
- Compare: fine-tuned vs prompted GPT-4o vs local model
- **Healthcare Example**: End-to-end medical coding assistant

## Learning Objectives

After completing Level 7, you will:
- ✅ Run local models with Ollama (HIPAA-compliant)
- ✅ Navigate the HuggingFace ecosystem
- ✅ Prepare high-quality training datasets
- ✅ Fine-tune models with LoRA/QLoRA
- ✅ Evaluate and compare model performance
- ✅ Know when to fine-tune vs when to prompt

## Time Estimate
15-20 hours total (3-4 hours per project)

## How This Connects to Level 8
Level 8 takes everything to production — FastAPI deployments, caching, monitoring,
security. You'll deploy both cloud API and local model endpoints.
