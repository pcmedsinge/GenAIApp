# Level 4: Evaluation, Safety & Guardrails
**Ensure your AI systems are correct, safe, and compliant**

## Overview

You've built RAG systems and agents — but how do you know they work correctly?
In healthcare, wrong answers can be dangerous. This level teaches systematic
evaluation, output validation, and safety guardrails.

### Why Evaluation Before More Features?

```
Without evaluation:  Build → Deploy → Hope it works → Find bugs in production
With evaluation:     Build → Evaluate → Validate → Guard → Deploy with confidence
```

### The Healthcare Stakes
- A wrong medication recommendation could harm a patient
- Hallucinated lab values could lead to wrong treatment
- Missing safety guardrails could expose PHI (Protected Health Information)

## Prerequisites
- **Levels 1-3 Complete**: You'll evaluate systems you've already built
- **OpenAI API Key**: Configured in .env
- **Additional packages**: `pip install ragas pydantic` (in requirements.txt)

## Projects

### 01_prompt_testing — Systematic Prompt Evaluation
- A/B testing prompts with scoring rubrics
- Regression testing (ensure changes don't break existing behavior)
- Automated test suites for prompt quality
- **Healthcare Example**: Test clinical note generation prompts

### 02_rag_evaluation — Measuring RAG Quality
- RAGAS metrics: faithfulness, answer relevancy, context precision
- Retrieval quality scoring (precision@k, recall@k, MRR)
- End-to-end RAG evaluation pipelines
- **Healthcare Example**: Evaluate medical guidelines RAG from Level 2

### 03_output_validation — Structured Outputs & Validation
- OpenAI structured outputs with JSON schema
- Pydantic models for LLM output validation
- Retry strategies for malformed outputs
- **Healthcare Example**: Validate clinical data extraction

### 04_guardrails — Safety Layers for AI
- Content filtering and topic restriction
- Input/output guardrails (PII detection, off-topic blocking)
- Hallucination detection techniques
- **Healthcare Example**: Guard a medication recommendation agent

### 05_healthcare_compliance — Capstone: Clinical AI Safety
- HIPAA considerations for LLM usage
- PHI de-identification in clinical text
- Audit trails for AI-assisted decisions
- Clinical validation workflows
- **Healthcare Example**: Compliant clinical note summarizer

## Learning Objectives

After completing Level 4, you will:
- ✅ Systematically test and evaluate prompts
- ✅ Measure RAG quality with industry-standard metrics
- ✅ Enforce structured LLM outputs with JSON schema + Pydantic
- ✅ Implement safety guardrails for production AI
- ✅ Handle HIPAA considerations for healthcare AI
- ✅ Detect and mitigate hallucinations

## Time Estimate
10-12 hours total (2-2.5 hours per project)

## How This Connects to Level 5
MCP (Level 5) connects your agents to external tools and data. Evaluation and
guardrails from this level ensure those connections are safe and reliable.
