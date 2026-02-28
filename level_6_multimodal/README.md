# Level 6: Multimodal & Reasoning Models
**Go beyond text — vision, audio, structured output, and extended reasoning**

## Overview

Modern AI goes far beyond text. Vision models analyze medical images, audio models
transcribe clinical encounters, and reasoning models (o1/o3) solve complex diagnostic
problems through extended thinking.

### The Multimodal Landscape

```
Text:      GPT-4o, Claude 3.5 — chat, analysis, generation
Vision:    GPT-4o, Claude 3.5 — image understanding, document extraction
Audio:     Whisper — transcription, medical terminology
Reasoning: o1, o3, DeepSeek R1 — extended thinking for complex problems
Structured: JSON schema enforcement — guaranteed output format
```

### Why Multimodal Matters for Healthcare
- Medical imaging (X-rays, dermatology photos, lab report images)
- Clinical encounter transcription (doctor-patient conversations)
- Complex differential diagnosis (requires extended reasoning)
- Structured data extraction (FHIR, HL7 from free text)

## Prerequisites
- **Levels 1-3 Complete**: Core API and agent skills
- **Level 4 Recommended**: Output validation skills
- **OpenAI API Key**: Configured in .env (needs access to GPT-4o, Whisper)

## Projects

### 01_vision_models — Image Understanding
- GPT-4o and Claude vision capabilities
- Image analysis: description, classification, comparison
- Document/chart/form extraction from images
- **Healthcare Example**: Analyze medical forms and extract structured data

### 02_audio_transcription — Speech-to-Text AI
- OpenAI Whisper for audio transcription
- Medical terminology handling and accuracy
- Post-processing transcriptions
- **Healthcare Example**: Transcribe clinical encounters, generate SOAP notes

### 03_structured_outputs — Schema-Enforced Generation
- OpenAI structured outputs with JSON schema
- Pydantic model → JSON schema → validated output
- Complex nested schemas for clinical data
- **Healthcare Example**: Extract FHIR-compatible patient data

### 04_reasoning_models — Extended Thinking
- Reasoning models: o1, o3, DeepSeek R1
- When to use reasoning vs standard models
- Cost/latency tradeoffs
- **Healthcare Example**: Differential diagnosis with reasoning

### 05_medical_multimodal — Capstone: Multimodal Clinical Assistant
- Accept images (skin lesion, lab report, medical form)
- Accept audio descriptions (patient symptoms)
- Use reasoning for complex analysis
- **Healthcare Example**: Multimodal triage assistant

## Learning Objectives

After completing Level 6, you will:
- ✅ Analyze images with GPT-4o and Claude vision APIs
- ✅ Transcribe audio with Whisper
- ✅ Enforce structured outputs with JSON schema
- ✅ Use reasoning models for complex problems
- ✅ Build multimodal AI pipelines
- ✅ Know when to use which model type

## Time Estimate
12-15 hours total (2.5-3 hours per project)

## How This Connects to Level 7
Level 7 teaches you to run models locally (Ollama, HuggingFace) and fine-tune them.
The multimodal skills from this level help you understand what capabilities you gain
from cloud APIs vs what you can run on-premise.
