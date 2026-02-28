# Project 3: Structured Outputs and Validation

## What You'll Learn
- Forcing LLMs to produce valid JSON with `response_format={"type": "json_object"}`
- Defining Pydantic models to validate and parse LLM outputs
- Using OpenAI Structured Outputs with JSON schema for guaranteed structure
- Building retry strategies that feed validation errors back to the LLM
- Extracting structured clinical data from free-text notes
- Working with FHIR-compatible data structures

## Why This Matters in Healthcare

LLM outputs are unpredictable by nature. In healthcare, unstructured or malformed
output can break downstream systems, corrupt patient records, or silently drop
critical information. Structured output validation ensures:
- Medications always include dosage, route, and frequency
- Diagnoses map to valid ICD-10 codes
- Vital signs are numeric and within physiologically possible ranges
- Extracted data conforms to interoperability standards like FHIR
- Downstream EHR integrations receive well-formed records every time

## Key Concepts

### JSON Mode
- Use `response_format={"type": "json_object"}` to guarantee valid JSON
- The LLM is constrained to produce only JSON output
- You still need to validate the schema — JSON mode only ensures syntactic validity
- Always include "JSON" in your prompt when using JSON mode

### Pydantic Validation
- Define Python data models with type hints and constraints
- Parse raw LLM output into validated objects
- Get precise error messages when output doesn't match expectations
- Use `Field()` for constraints: min/max values, regex patterns, enums

### OpenAI Structured Outputs
- Provide a JSON schema and the API guarantees conformant output
- Eliminates the need for post-hoc parsing and retry loops
- Supports nested objects, arrays, enums, and optional fields
- Ideal for production systems where reliability is critical

### Retry with Error Feedback
- When validation fails, feed the error message back to the LLM
- The LLM learns from its mistake and corrects the output
- Set a maximum retry count to avoid infinite loops
- Log failures for monitoring and prompt improvement

## Running the Code

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Run the main demos
python level_4_evaluation/03_output_validation/main.py

# Run individual exercises
python level_4_evaluation/03_output_validation/exercise_1_clinical_extraction.py
python level_4_evaluation/03_output_validation/exercise_2_schema_complexity.py
python level_4_evaluation/03_output_validation/exercise_3_validation_errors.py
python level_4_evaluation/03_output_validation/exercise_4_fhir_extraction.py
```

## Demos (main.py)

### Demo 1: Basic JSON Mode
Use `response_format={"type": "json_object"}` to extract patient data from a clinical
note as valid JSON. Compare the raw text output to the structured JSON output.

### Demo 2: Pydantic Validation
Define Pydantic models for patient data (name, DOB, medications, diagnoses, vitals).
Parse LLM JSON output into typed Python objects. Handle validation errors gracefully.

### Demo 3: Structured Outputs with JSON Schema
Use OpenAI's structured outputs feature to provide a JSON schema and receive
guaranteed conformant output. No parsing or retry needed.

### Demo 4: Retry Logic with Error Feedback
When the LLM produces output that fails validation, automatically retry with the
error message appended. Watch the LLM correct itself in real time.

## Exercises

### Exercise 1: Clinical Data Extraction (`exercise_1_clinical_extraction.py`)
Extract structured clinical data from free-text notes using Pydantic models. Define
models for Patient, Medication, Diagnosis, and Vital Signs. Parse 3-4 sample clinical
notes and validate every extracted field.

### Exercise 2: Schema Complexity Ladder (`exercise_2_schema_complexity.py`)
Build increasingly complex schemas — simple (name/age), medium (patient with medications
list), complex (full encounter with nested objects). Test LLM extraction reliability at
each complexity level and measure success rates.

### Exercise 3: Validation Error Handling (`exercise_3_validation_errors.py`)
Deliberately test edge cases that cause validation failures: missing required fields,
wrong types, out-of-range values. Implement graceful error handling with retry,
fallback to simpler schemas, and failure logging.

### Exercise 4: FHIR-Compatible Extraction (`exercise_4_fhir_extraction.py`)
Extract data matching FHIR (Fast Healthcare Interoperability Resources) structures from
clinical notes. Define Pydantic models for FHIR Patient, Condition, and
MedicationStatement resources. Validate and output conformant FHIR JSON.

## Healthcare Best Practices

⚠️ **IMPORTANT**:
- Never use real patient data for testing — always use synthetic examples
- Validate all extracted clinical data before writing to any system of record
- Log validation failures for audit and continuous improvement
- Include human review steps for safety-critical extractions
- Test with diverse clinical note styles (ED, surgery, primary care, etc.)
- Monitor extraction accuracy in production with automated metrics

## Expected Output

```
=== DEMO 1: Basic JSON Mode ===
Extracting patient data from clinical note...

Raw JSON output:
{
  "patient_name": "Maria Santos",
  "date_of_birth": "1958-03-15",
  "medications": ["Metformin 1000mg BID", "Lisinopril 20mg daily"],
  "diagnoses": ["Type 2 Diabetes Mellitus", "Essential Hypertension"],
  "vitals": {"bp": "138/82", "hr": 76, "temp": 98.6}
}

✅ Valid JSON extracted successfully

=== DEMO 2: Pydantic Validation ===
Validating extracted data against PatientRecord model...
✅ All fields validated
  Name: Maria Santos (str ✓)
  DOB: 1958-03-15 (date ✓)
  Medications: 2 items (list[Medication] ✓)
  Vitals: BP 138/82, HR 76 (VitalSigns ✓)
```
