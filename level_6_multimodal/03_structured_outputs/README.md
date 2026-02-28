# Project 3: Structured Outputs — Schema-Enforced Generation

## What You'll Learn
- Use OpenAI structured outputs to guarantee JSON matching a schema
- Build Pydantic models that become JSON schemas for the LLM
- Handle complex nested schemas for clinical data
- Evolve schemas gracefully over time

## Why Structured Outputs?
LLMs return free text by default. For production systems, you need GUARANTEED structure:
```
Free text:   "The patient's BMI is 28.5, which is overweight..."
Structured:  {"bmi": 28.5, "category": "overweight", "risk_level": "moderate"}
```

## Running the Code
```bash
cd level_6_multimodal/03_structured_outputs
python main.py
```

## Demos
1. **JSON Schema Enforcement** — `response_format` with strict JSON schema
2. **Nested Schemas** — Patient with medications list, each with dosing
3. **Enum Constraints** — Literal types to constrain LLM output values
4. **Schema Evolution** — Handle v1 → v2 schema changes gracefully

## Exercises
1. Extract structured clinical data (Patient, Medication, Diagnosis) from notes
2. Generate FHIR-compatible resources from clinical text
3. Build structured lab reports with reference ranges and interpretations
4. Generate structured discharge summaries

## Key Concepts
- **response_format**: OpenAI parameter for schema-enforced output
- **Pydantic → JSON Schema**: Automatic schema generation from Python models
- **Nested models**: Lists, optional fields, enums within schemas
- **Retry logic**: Handle rare schema violations gracefully
