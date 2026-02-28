# Project 05: Medical Model Capstone

## Overview
This capstone project ties together everything from Level 7: local LLMs,
Hugging Face ecosystem, data preparation, and LoRA fine-tuning into a complete
end-to-end medical model pipeline.

## The Mission
Build and deploy a local ICD-10 coding model that:
1. Takes clinical notes as input
2. Predicts the correct ICD-10 diagnostic code
3. Runs entirely on local hardware (HIPAA-safe)
4. Achieves competitive accuracy vs GPT-4o

## Pipeline Stages

### Stage 1: Data Preparation
- Generate synthetic clinical note → ICD-10 code pairs
- Clean, validate, and format training data
- Split into train / validation / test sets

### Stage 2: Fine-Tuning
- Load base model (e.g., Llama 3, Mistral, Phi-3)
- Configure LoRA adapters for medical coding
- Train with Hugging Face Trainer
- Save adapter weights and merged model

### Stage 3: Evaluation
- Test on held-out clinical notes
- Measure accuracy, F1, per-code performance
- Analyze failure modes and edge cases
- Compare against GPT-4o baseline

### Stage 4: Deployment
- Convert to Ollama format for easy serving
- Create Modelfile with medical system prompt
- Benchmark inference speed and memory usage
- Build API wrapper for integration

## What You'll Build
| Demo/Exercise | Description |
|---|---|
| Demo 1 | ICD-10 dataset preparation |
| Demo 2 | Complete fine-tuning pipeline |
| Demo 3 | Comprehensive model evaluation |
| Demo 4 | Local deployment via Ollama |
| Exercise 1 | End-to-end pipeline |
| Exercise 2 | Model comparison study |
| Exercise 3 | Error analysis |
| Exercise 4 | Deployment guide |

## Running
```bash
python main.py
python exercise_1_end_to_end.py
python exercise_2_model_comparison.py
python exercise_3_error_analysis.py
python exercise_4_deployment_guide.py
```

## This Is the Summit
Completing this project means you can build, train, evaluate, and deploy
a custom medical AI model from scratch — entirely on local infrastructure.
