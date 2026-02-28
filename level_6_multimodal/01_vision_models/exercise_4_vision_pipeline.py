"""








































































```result = completion.choices[0].message.parsed  # typed MySchema instance)    response_format=MySchema,    messages=[...],    model="gpt-4o",completion = client.beta.chat.completions.parse(    value: float    name: strclass MySchema(BaseModel):from pydantic import BaseModel```python## Key API Pattern```python exercise_1_clinical_extraction.py# Individual exercisespython main.py# Interactive menu```bash## Running```pip install openai pydantic python-dotenv```bash## Prerequisites| 4 | `exercise_4_discharge_summary.py` | Complete structured discharge summaries || 3 | `exercise_3_lab_reports.py` | Structured lab reports with reference ranges || 2 | `exercise_2_fhir_resources.py` | Generate FHIR-compatible Patient, Condition, MedicationRequest || 1 | `exercise_1_clinical_extraction.py` | Extract structured clinical data from free text ||---|------|------|| # | File | Goal |## Exercises| 4 | Schema evolution — v1 → v2 with backward compatibility || 3 | Enum / Literal constraints on output values || 2 | Nested schemas — Patient → Medications → Dosing || 1 | `response_format` with JSON Schema via `client.beta.chat.completions.parse()` ||------|-------|| Demo | Topic |## Demos in main.pyhealth records. Structured outputs can generate FHIR-compatible JSON directly.Fast Healthcare Interoperability Resources (FHIR) is the standard for electronic### FHIR-Compatible Datawith medication lists, lab panels, or FHIR resources.literal types. This lets you model real healthcare data structures like patientsSchemas can include nested objects, arrays of objects, enums, optionals, and### Nested & Complex Schemasautomatically. This gives you type safety on both the request and response side.Define your data models in Pydantic, then let the SDK convert them to JSON Schema### Pydantic → Schemamatches your schema exactly.is constrained at the token-sampling level so every response is valid JSON thatOpenAI's `response_format` parameter accepts a JSON Schema definition. The model### JSON Schema Enforcement## Key Conceptskeys — the model's output is **always** valid against your schema.a specific schema. This eliminates fragile parsing, retry loops, and hallucinatedStructured outputs let you guarantee that OpenAI models return JSON conforming to## OverviewExercise 4 — Complete Vision Pipeline
======================================
End-to-end pipeline: receive image → classify type → extract relevant
data → generate a summary report.

Objectives
----------
* Chain multiple Vision API calls into a coherent pipeline
* Route extraction logic based on document-type classification
* Aggregate outputs into a final structured report
* Measure latency and token usage across the pipeline
"""

import json
import time
import os
import base64
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

VISION_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_image_message(text: str, image_source: str, detail: str = "high") -> dict:
    if image_source.startswith(("http://", "https://", "data:")):
        url = image_source
    else:
        ext = os.path.splitext(image_source)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        with open(image_source, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        url = f"data:{mime};base64,{b64}"
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": url, "detail": detail}},
        ],
    }


def call_vision(system: str, user_text: str, image_source: str,
                max_tokens: int = 600) -> tuple[str, dict]:
    """Make a Vision API call and return (content, usage_dict)."""
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": system},
            build_image_message(user_text, image_source),
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    content = response.choices[0].message.content.strip()
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return content, usage


def parse_json_response(raw: str) -> dict:
    """Attempt to parse a JSON response, stripping markdown fences."""
    cleaned = raw
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"_raw": raw, "_error": "JSON parse failed"}


# ---------------------------------------------------------------------------
# Pipeline stage dataclass
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    stage: str = ""
    data: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)
    latency_s: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Stage 1: Classification
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = """\
You are a medical image classifier.  Classify this image into ONE of:
  medical_form, lab_report, prescription, radiology_image,
  clinical_photo, ecg_tracing, anatomical_diagram, unknown

Respond with JSON only:
{"document_type": "<label>", "confidence": <0-100>}
"""


def stage_classify(image_source: str) -> StageResult:
    """Stage 1 — Classify the image type."""
    t0 = time.time()
    raw, usage = call_vision(CLASSIFY_SYSTEM, "Classify this image.", image_source, 150)
    data = parse_json_response(raw)
    return StageResult(
        stage="classify",
        data=data,
        tokens=usage,
        latency_s=round(time.time() - t0, 2),
    )


# ---------------------------------------------------------------------------
# Stage 2: Extraction (routed by document type)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPTS = {
    "medical_form": """\
Extract structured data from this medical form as JSON:
{"patient_name", "dob", "insurance_id", "chief_complaint", "medications", "provider"}
Use null for fields you cannot read.""",

    "lab_report": """\
Extract lab results from this report as JSON:
{"patient_name", "test_date", "tests": [{"name", "value", "unit", "reference_range", "flag"}]}
Use null for unreadable fields.""",

    "ecg_tracing": """\
Analyze this ECG tracing and return JSON:
{"rhythm", "rate_bpm", "intervals": {"pr_ms", "qrs_ms", "qt_ms"}, "abnormalities": [], "interpretation"}
Use null for indeterminate values.""",

    "radiology_image": """\
Analyze this radiology image and return JSON:
{"modality", "body_region", "findings": [], "impression", "follow_up_recommended": bool}""",

    "prescription": """\
Extract prescription details as JSON:
{"patient_name", "medications": [{"name", "dose", "route", "frequency"}], "prescriber", "date"}""",
}

DEFAULT_EXTRACTION = """\
Describe this medical image and extract any structured information as JSON:
{"description", "key_findings": [], "additional_notes"}"""


def stage_extract(image_source: str, doc_type: str) -> StageResult:
    """Stage 2 — Extract structured data based on document type."""
    prompt = EXTRACTION_PROMPTS.get(doc_type, DEFAULT_EXTRACTION)
    system = (
        "You are an expert medical document processor. "
        "Return ONLY valid JSON, no markdown."
    )
    t0 = time.time()
    raw, usage = call_vision(system, prompt, image_source, 600)
    data = parse_json_response(raw)
    return StageResult(
        stage="extract",
        data=data,
        tokens=usage,
        latency_s=round(time.time() - t0, 2),
    )


# ---------------------------------------------------------------------------
# Stage 3: Summary generation (text-only, no image needed)
# ---------------------------------------------------------------------------

def stage_summarize(classification: dict, extraction: dict) -> StageResult:
    """Stage 3 — Generate a human-readable summary from extracted data."""
    t0 = time.time()

    prompt = f"""\
You are a medical documentation specialist.  Given the classification
and extracted data below, write a concise 3-5 sentence summary report
suitable for a clinician's review.

Classification: {json.dumps(classification)}
Extracted Data: {json.dumps(extraction)}

Return JSON: {{"summary": "<text>", "action_items": ["<list>"]}}
"""
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3,
    )
    raw = response.choices[0].message.content.strip()
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    data = parse_json_response(raw)
    return StageResult(
        stage="summarize",
        data=data,
        tokens=usage,
        latency_s=round(time.time() - t0, 2),
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineReport:
    image_source: str = ""
    stages: list = field(default_factory=list)
    total_tokens: int = 0
    total_latency_s: float = 0.0
    final_summary: str = ""
    action_items: list = field(default_factory=list)


def run_pipeline(image_source: str) -> PipelineReport:
    """Execute the full vision pipeline on a single image.

    Steps: classify → extract → summarize
    """
    report = PipelineReport(image_source=image_source)

    # Stage 1
    print("  [1/3] Classifying image …")
    s1 = stage_classify(image_source)
    report.stages.append(s1)
    doc_type = s1.data.get("document_type", "unknown")
    print(f"        → {doc_type} ({s1.data.get('confidence', '?')}%)")

    # Stage 2
    print("  [2/3] Extracting data …")
    s2 = stage_extract(image_source, doc_type)
    report.stages.append(s2)

    # Stage 3
    print("  [3/3] Generating summary …")
    s3 = stage_summarize(s1.data, s2.data)
    report.stages.append(s3)

    # Aggregate
    report.total_tokens = sum(s.tokens.get("total_tokens", 0) for s in report.stages)
    report.total_latency_s = round(sum(s.latency_s for s in report.stages), 2)
    report.final_summary = s3.data.get("summary", "")
    report.action_items = s3.data.get("action_items", [])

    return report


def print_pipeline_report(report: PipelineReport) -> None:
    """Pretty-print a pipeline execution report."""
    print("\n" + "=" * 70)
    print("  PIPELINE REPORT")
    print("=" * 70)
    print(f"\n  Image:    {report.image_source[:80]}")
    print(f"  Tokens:   {report.total_tokens}")
    print(f"  Latency:  {report.total_latency_s}s\n")

    for s in report.stages:
        print(f"  --- Stage: {s.stage} ({s.latency_s}s, {s.tokens.get('total_tokens', 0)} tokens) ---")
        print(f"  {json.dumps(s.data, indent=2)[:500]}\n")

    print("  === Final Summary ===")
    print(f"  {report.final_summary}\n")
    if report.action_items:
        print("  Action Items:")
        for item in report.action_items:
            print(f"    • {item}")
    print()


# ---------------------------------------------------------------------------
# Sample images
# ---------------------------------------------------------------------------

SAMPLE_IMAGES = {
    "CMS-1500 Form": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "8/8e/CMS-1500_claim_form.jpg/400px-CMS-1500_claim_form.jpg"
    ),
    "ECG Tracing": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "1/1b/Sinus_rhythm_labels.svg/600px-Sinus_rhythm_labels.svg.png"
    ),
    "Anatomy Diagram": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "d/d5/Anatomical_chart%2C_Cyclopaedia%2C_1728%2C_Volume_1.jpg/"
        "440px-Anatomical_chart%2C_Cyclopaedia%2C_1728%2C_Volume_1.jpg"
    ),
}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_vision_pipeline():
    print("=" * 70)
    print("  Exercise 4 — Complete Vision Pipeline")
    print("=" * 70)

    print("\nAvailable sample images:")
    for i, name in enumerate(SAMPLE_IMAGES, 1):
        print(f"  [{i}] {name}")
    print(f"  [{len(SAMPLE_IMAGES) + 1}] Enter custom URL")
    print()

    choice = input("Select image → ").strip()
    names = list(SAMPLE_IMAGES.keys())

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(names):
            url = SAMPLE_IMAGES[names[idx]]
            print(f"\nRunning pipeline on: {names[idx]}\n")
        else:
            url = input("Enter image URL → ").strip()
            print()
    except ValueError:
        url = input("Enter image URL → ").strip()
        print()

    if not url:
        url = list(SAMPLE_IMAGES.values())[0]
        print(f"Using default: CMS-1500 Form\n")

    report = run_pipeline(url)
    print_pipeline_report(report)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_vision_pipeline()
