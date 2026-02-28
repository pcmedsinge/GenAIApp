# Project 4: Reasoning Models — Extended Thinking

## What You'll Learn
- Use reasoning models (o1, o3, DeepSeek R1) for complex problems
- Understand the difference between reasoning and standard models
- Know when to use reasoning vs standard (cost/latency tradeoffs)
- Build model selection logic for different query types

## Reasoning vs Standard Models
```
Standard (GPT-4o):    Fast, cheap, great for most tasks
Reasoning (o1/o3):    Slower, expensive, but THINKS through complex multi-step problems
```

Reasoning models spend "thinking tokens" before answering — like a doctor
taking time to consider a differential diagnosis rather than giving a snap answer.

## Running the Code
```bash
cd level_6_multimodal/04_reasoning_models
python main.py
```

## Demos
1. **Reasoning vs Standard** — Compare GPT-4o vs o1-mini on complex medical questions
2. **Extended Thinking** — Watch how reasoning models work through differential diagnosis
3. **Cost/Latency Analysis** — Compare token costs and response times
4. **When to Use Reasoning** — Decision framework for model selection

## Exercises
1. Differential diagnosis with reasoning model vs standard
2. Complex treatment planning with multiple comorbidities
3. Build an intelligent model router (simple → cheap, complex → reasoning)
4. Evaluate reasoning model performance on clinical scenarios

## Key Concepts
- **Thinking tokens**: Internal reasoning before the answer
- **No system messages**: o1 models don't support system prompts
- **max_completion_tokens**: Use instead of max_tokens for reasoning models
- **reasoning_effort**: Control how much thinking the model does
