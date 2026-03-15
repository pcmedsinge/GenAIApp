# Project 4: Prompt Engineering

## What You'll Learn
- Zero-shot, few-shot, and chain-of-thought prompting
- System prompts and role-based instructions
- Prompt templates with variables
- Output formatting (JSON, structured data)
- Avoiding common prompt pitfalls

## Key Concepts

### Prompting Techniques

#### 1. Zero-Shot
Ask directly — no examples provided:
```
"What ICD-10 code would you assign to acute bronchitis?"
```

#### 2. Few-Shot
Provide examples first, then ask:
```
"Here are examples of clinical notes and their ICD-10 codes:
 Note: 'Patient has type 2 diabetes' → E11.9
 Note: 'Acute upper respiratory infection' → J06.9
 Now classify: 'Patient presents with acute bronchitis'"
```

#### 3. Chain-of-Thought (CoT)
Ask the model to think step by step:
```
"A patient is on warfarin (INR 3.5) and needs dental extraction.
Think step by step about the risks and management plan."
```

### Output Formatting
- Plain text responses
- JSON structured output
- Markdown tables
- Specific formats (SOAP notes, discharge summaries)

## Running the Code

```bash
python main.py
```

## Exercises

### Exercise 1: Clinical Note Structurer
Convert free-text clinical notes into structured SOAP format using prompt engineering.

### Exercise 2: Few-Shot ICD-10 Coder
Build a few-shot prompt that accurately assigns ICD-10 codes to clinical descriptions.

### Exercise 3: Chain-of-Thought Diagnosis
Use CoT prompting for step-by-step differential diagnosis reasoning.

### Exercise 4: Prompt Template Library
Build a library of reusable healthcare prompt templates with variable substitution.

### Exercise 5: Jinja2 Prompt Registry + Versioning
Build a versioned prompt registry with Jinja2 rendering, active-version promotion,
and rollback to prior prompt versions.

## Healthcare Applications
- Clinical documentation structuring
- Medical coding assistance
- Differential diagnosis reasoning
- Patient education content generation
- Discharge summary generation

## Next Steps
After mastering prompt engineering (including template versioning), move to
**05_streaming** to learn real-time response streaming!
