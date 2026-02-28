# Project 1: Systematic Prompt Evaluation

## What You'll Learn
- How to systematically compare prompts using scoring rubrics
- Building regression test suites for prompt quality assurance
- Running statistical A/B tests on prompt variants
- Automating prompt evaluation pipelines
- Tracking prompt versions and detecting regressions
- Using LLM-as-judge for scalable evaluation

## Why This Matters in Healthcare

Prompts are the "source code" of LLM applications. In healthcare:
- A poorly worded prompt could omit critical safety information
- Prompt changes can silently degrade output quality
- Regulatory compliance requires consistent, auditable outputs
- Patient safety depends on reliable, tested prompt behavior

Systematic prompt evaluation ensures that every change to a prompt is validated against known expectations before it reaches production.

## Key Concepts

### Scoring Rubrics
- Define measurable criteria (clarity, accuracy, completeness, safety)
- Score each criterion on a consistent scale (1-5)
- Use LLM-as-judge to automate scoring at scale
- Aggregate scores for overall prompt quality

### Regression Testing
- Define test cases with inputs, expected behaviors, and forbidden content
- Run prompts against the full test suite after every change
- Detect regressions before they reach users
- Track pass/fail rates over time

### A/B Testing
- Compare two prompt variants head-to-head
- Run multiple trials for statistical confidence
- Compute win rates and performance differences
- Make data-driven decisions about prompt changes

### Prompt Versioning
- Track every prompt change with timestamps and metadata
- Compare metrics across prompt versions
- Roll back to previous versions when regressions occur
- Maintain an audit trail for compliance

## Running the Code

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Run the main demos
python level_4_evaluation/01_prompt_testing/main.py

# Run individual exercises
python level_4_evaluation/01_prompt_testing/exercise_1_scoring_rubric.py
python level_4_evaluation/01_prompt_testing/exercise_2_regression_suite.py
python level_4_evaluation/01_prompt_testing/exercise_3_prompt_versioning.py
python level_4_evaluation/01_prompt_testing/exercise_4_automated_eval.py
```

## Demos (main.py)

### Demo 1: Basic Prompt Comparison
Compare two system prompts for clinical note generation. An LLM judge scores each on clarity, completeness, and accuracy (1-5 scale). See which prompt produces better clinical notes.

### Demo 2: Regression Test Suite
Define test cases with expected keywords and forbidden content. Run a clinical Q&A prompt against all test cases and see pass/fail results with detailed failure reasons.

### Demo 3: Statistical A/B Testing
Run multiple head-to-head comparisons between prompt variants. Compute win rates, average scores, and statistical summaries to make confident prompt decisions.

### Demo 4: Interactive Prompt Lab
Enter your own system prompt and test it against clinical scenarios. See scored results in real time and iterate on prompt design interactively.

## Exercises

### Exercise 1: Custom Scoring Rubric (`exercise_1_scoring_rubric.py`)
Build a multi-dimensional scoring rubric for clinical notes. Evaluate prompts on medical accuracy, completeness, patient safety mentions, and readability. Test across multiple clinical scenarios.

### Exercise 2: Regression Test Suite (`exercise_2_regression_suite.py`)
Build a comprehensive regression test suite with 10+ test cases for a clinical Q&A system. Each test specifies input, expected keywords, and forbidden content. Run all tests and generate a pass/fail report.

### Exercise 3: Prompt Versioning (`exercise_3_prompt_versioning.py`)
Implement a prompt version tracking system. Store prompt versions with timestamps, run evaluations on each version, compare metrics across versions, and identify regressions automatically.

### Exercise 4: Automated Evaluation Pipeline (`exercise_4_automated_eval.py`)
Build a full automated evaluation pipeline. Given a set of prompts and test cases, run all combinations, score each, and generate a summary report identifying the best and worst performers.

## Healthcare Best Practices

⚠️ **IMPORTANT**:
- Never use real patient data in prompt testing
- Include safety checks in all clinical prompt evaluations
- Document prompt changes and evaluation results for audit trails
- Ensure prompts include appropriate medical disclaimers
- Test for harmful or misleading outputs explicitly
- Validate that prompts handle edge cases (rare conditions, ambiguous symptoms)

## Expected Output

```
=== DEMO 1: Basic Prompt Comparison ===
Testing prompt: "Concise Clinical Assistant"
  Clarity:      4/5
  Completeness: 3/5
  Accuracy:     4/5
  Average:      3.67

Testing prompt: "Detailed Clinical Narrator"
  Clarity:      3/5
  Completeness: 5/5
  Accuracy:     4/5
  Average:      4.00

Winner: Detailed Clinical Narrator (4.00 vs 3.67)
```
