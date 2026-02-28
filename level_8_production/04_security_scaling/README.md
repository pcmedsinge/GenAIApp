# Project 04: Security & Scaling

## Overview
LLM applications face unique security threats — prompt injection, data
exfiltration, and jailbreaks — alongside traditional scaling challenges.
This project covers both offensive (red-teaming) and defensive approaches.

## Why LLM Security Matters
- **Prompt Injection**: Attackers manipulate model behavior by embedding
  instructions in user input
- **Data Leakage**: Models may reveal system prompts, training data, or PII
- **Jailbreaking**: Bypassing safety guidelines for harmful outputs
- **Scaling**: High-load scenarios introduce latency, cost, and reliability issues

## Key Concepts

### Prompt Injection Attacks
- Direct injection: "Ignore previous instructions and..."
- Indirect injection: Hidden instructions in retrieved documents
- Delimiter attacks: Breaking out of user-input boundaries
- Encoding tricks: Base64, ROT13, Unicode to evade filters

### Defense Strategies
- Input sanitization: Detect and neutralize injection patterns
- Output validation: Check responses before returning to users
- System prompt reinforcement: Repeated safety instructions
- Instruction hierarchy: Prioritize system over user messages

### Input/Output Filtering
- PII detection and redaction in inputs
- Harmful content blocking in outputs
- Off-topic response filtering
- Medical disclaimer enforcement

### Scaling Patterns
- Concurrent request handling with rate limiting
- Load testing to identify bottlenecks
- Graceful degradation under high load
- Cost control at scale

## Demos in main.py
1. **Prompt Injection Attacks** — Common attack patterns demonstrated
2. **Injection Defense** — Input sanitization and system reinforcement
3. **Input/Output Filtering** — PII, harmful content, off-topic filtering
4. **Security Scorecard** — Battery of attack tests with scoring

## Exercises
1. `exercise_1_red_team.py` — Red-team a medical chatbot with 15+ attacks
2. `exercise_2_input_sanitizer.py` — Robust input sanitization pipeline
3. `exercise_3_output_filter.py` — Output filtering for medical safety
4. `exercise_4_load_testing.py` — Load testing with concurrency analysis

## Healthcare Relevance
- Medical chatbots must not provide unsafe advice under any manipulation
- HIPAA requires PII protection in all AI interactions
- Clinical decision support cannot be jailbroken to skip safety disclaimers
- Hospital systems must handle peak loads during emergencies

## Running
```bash
python main.py                      # Interactive demo menu
python exercise_1_red_team.py       # Red-team exercise
python exercise_2_input_sanitizer.py # Input sanitizer exercise
python exercise_3_output_filter.py   # Output filter exercise
python exercise_4_load_testing.py    # Load testing exercise
```
