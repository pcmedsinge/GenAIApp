# Project 04: Guardrails

## Overview

Safety layers are essential for any production AI system — especially in healthcare where
incorrect or harmful outputs can have real consequences. This project teaches you to build
**guardrails**: automated checks that wrap around your LLM to filter dangerous inputs,
catch unsafe outputs, restrict off-topic conversations, and detect hallucinations.

You'll learn to build a defense-in-depth pipeline: validate inputs before they reach the
model, validate outputs before they reach the user, and verify factual accuracy against
source documents.

## Key Concepts

- **Input Guardrails** — Reject harmful, off-topic, or adversarial prompts before they reach the LLM
- **Output Guardrails** — Filter model responses for safety (disclaimers, no PHI leakage, no diagnoses)
- **Topic Restriction** — Constrain the AI to only answer questions within its designated domain
- **Hallucination Detection** — Compare model outputs against source context to identify fabricated claims
- **Safety Scoring** — Quantify the safety and faithfulness of every interaction

## Healthcare Context

In healthcare AI, guardrails are not optional:
- AI must never provide definitive diagnoses without disclaimers
- Protected Health Information (PHI) must never leak into responses
- Off-topic queries (legal advice, financial planning) must be redirected
- Hallucinated drug interactions or dosages can cause patient harm
- Every unsafe interaction must be logged for compliance auditing

## Demos (main.py)

1. **Input Guardrails** — Detect off-topic, harmful, or inappropriate inputs before sending to LLM
2. **Output Guardrails** — Filter LLM responses for safety (disclaimers, no PHI leakage)
3. **Topic Restriction** — Constrain agent to only answer healthcare-related questions
4. **Hallucination Detection** — Compare LLM answer against source context to detect fabrication

## Exercises

1. **PII Detector** (`exercise_1_pii_detector.py`) — Build a PII detector that identifies names, dates, SSNs, MRNs in clinical text using regex + LLM. Mask detected PHI.
2. **Topic Filter** (`exercise_2_topic_filter.py`) — Build a topic classifier that blocks non-medical questions and allows clinical, medication, and lab queries.
3. **Hallucination Checker** (`exercise_3_hallucination_checker.py`) — Given a source document and an LLM-generated answer, identify hallucinated claims and score faithfulness.
4. **Safety Pipeline** (`exercise_4_safety_pipeline.py`) — Build a complete end-to-end safety pipeline: input check → LLM generation → output check → hallucination check.

## Running

```bash
# Run demos
python main.py

# Run exercises
python exercise_1_pii_detector.py
python exercise_2_topic_filter.py
python exercise_3_hallucination_checker.py
python exercise_4_safety_pipeline.py
```

## Prerequisites

- OpenAI API key configured in `.env`
- Python packages: `openai`, `python-dotenv`
- Completion of Level 4 Projects 01-03 recommended
