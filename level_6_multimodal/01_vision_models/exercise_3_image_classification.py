"""
Exercise 3 — Multi-Category Medical Image Classifier
=====================================================
Build a robust image classifier that assigns **multiple** labels to a
medical image:

1. **Document type** — form, lab report, prescription, radiology, photo, diagram
2. **Normal vs. Abnormal** — clinical assessment of content
3. **Urgency** — routine, urgent, emergent

Objectives
----------
* Combine multiple classification axes in one prompt
* Return structured JSON with confidence scores per axis
* Aggregate results across a batch of images
* Display a formatted classification report
"""

import json
import os
import base64
from dataclasses import dataclass, field, asdict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

VISION_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_image_message(text: str, image_source: str, detail: str = "auto") -> dict:
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


# ---------------------------------------------------------------------------
# Classification categories
# ---------------------------------------------------------------------------

DOCUMENT_TYPES = [
    "medical_form", "lab_report", "prescription",
    "radiology_image", "clinical_photo", "anatomical_diagram",
    "ecg_tracing", "other",
]

CLINICAL_STATUS = ["normal", "abnormal", "indeterminate"]

URGENCY_LEVELS = ["routine", "urgent", "emergent"]


# ---------------------------------------------------------------------------
# Classification dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    image_label: str = ""
    document_type: str = ""
    document_type_confidence: int = 0
    clinical_status: str = ""
    clinical_status_confidence: int = 0
    urgency: str = ""
    urgency_confidence: int = 0
    reasoning: str = ""
    raw_response: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

CLASSIFICATION_SYSTEM = f"""\
You are a medical image triage classifier.  Given a medical image, assign
THREE classification labels with confidence scores (0-100).

DOCUMENT TYPE (pick one):
{json.dumps(DOCUMENT_TYPES)}

CLINICAL STATUS (pick one):
{json.dumps(CLINICAL_STATUS)}

URGENCY (pick one):
{json.dumps(URGENCY_LEVELS)}

Respond ONLY with valid JSON (no markdown fences):

{{
  "document_type": "<label>",
  "document_type_confidence": <0-100>,
  "clinical_status": "<label>",
  "clinical_status_confidence": <0-100>,
  "urgency": "<label>",
  "urgency_confidence": <0-100>,
  "reasoning": "<1-2 sentence explanation>"
}}
"""


# ---------------------------------------------------------------------------
# Core classification function
# ---------------------------------------------------------------------------

def classify_image(image_source: str, label: str = "") -> ClassificationResult:
    """Classify a medical image along three axes.

    Parameters
    ----------
    image_source : str
        URL or local file path.
    label : str, optional
        Human-readable label for the image.

    Returns
    -------
    ClassificationResult
    """
    user_message = build_image_message(
        "Classify this medical image.", image_source,
    )

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": CLASSIFICATION_SYSTEM},
            user_message,
        ],
        max_tokens=300,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    result = ClassificationResult(image_label=label, raw_response=raw)

    try:
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(cleaned)
        result.document_type = data.get("document_type", "")
        result.document_type_confidence = data.get("document_type_confidence", 0)
        result.clinical_status = data.get("clinical_status", "")
        result.clinical_status_confidence = data.get("clinical_status_confidence", 0)
        result.urgency = data.get("urgency", "")
        result.urgency_confidence = data.get("urgency_confidence", 0)
        result.reasoning = data.get("reasoning", "")
    except json.JSONDecodeError as exc:
        result.error = str(exc)

    return result


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------

def classify_batch(images: list[tuple[str, str]]) -> list[ClassificationResult]:
    """Classify a batch of images.

    Parameters
    ----------
    images : list of (label, url_or_path) tuples.

    Returns
    -------
    list[ClassificationResult]
    """
    results = []
    for label, source in images:
        print(f"  Classifying: {label} …")
        results.append(classify_image(source, label))
    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def print_classification_report(results: list[ClassificationResult]) -> None:
    """Pretty-print a classification report table."""
    header = (
        f"{'Image':<25} {'Doc Type':<20} {'Status':<14} "
        f"{'Urgency':<12} {'Reasoning'}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        doc = f"{r.document_type} ({r.document_type_confidence}%)"
        sta = f"{r.clinical_status} ({r.clinical_status_confidence}%)"
        urg = f"{r.urgency} ({r.urgency_confidence}%)"
        print(f"{r.image_label:<25} {doc:<20} {sta:<14} {urg:<12} {r.reasoning[:50]}")


def compute_statistics(results: list[ClassificationResult]) -> dict:
    """Compute aggregate statistics over a batch of results."""
    from collections import Counter
    doc_counts = Counter(r.document_type for r in results if r.document_type)
    status_counts = Counter(r.clinical_status for r in results if r.clinical_status)
    urgency_counts = Counter(r.urgency for r in results if r.urgency)

    avg_doc_conf = (sum(r.document_type_confidence for r in results) / len(results)
                    if results else 0)

    return {
        "total_images": len(results),
        "document_type_distribution": dict(doc_counts),
        "clinical_status_distribution": dict(status_counts),
        "urgency_distribution": dict(urgency_counts),
        "avg_document_type_confidence": round(avg_doc_conf, 1),
    }


# ---------------------------------------------------------------------------
# Sample images
# ---------------------------------------------------------------------------

SAMPLE_IMAGES = [
    ("Anatomical Diagram",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/"
     "d/d5/Anatomical_chart%2C_Cyclopaedia%2C_1728%2C_Volume_1.jpg/"
     "440px-Anatomical_chart%2C_Cyclopaedia%2C_1728%2C_Volume_1.jpg"),
    ("ECG Tracing",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/"
     "1/1b/Sinus_rhythm_labels.svg/"
     "600px-Sinus_rhythm_labels.svg.png"),
    ("CMS-1500 Claim Form",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/"
     "8/8e/CMS-1500_claim_form.jpg/"
     "400px-CMS-1500_claim_form.jpg"),
]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_image_classification():
    print("=" * 70)
    print("  Exercise 3 — Multi-Category Medical Image Classifier")
    print("=" * 70)

    # --- Classify batch ---
    print("\nClassifying sample images …\n")
    results = classify_batch(SAMPLE_IMAGES)

    # --- Report ---
    print("\n=== Classification Report ===\n")
    print_classification_report(results)

    # --- Statistics ---
    stats = compute_statistics(results)
    print("\n=== Batch Statistics ===\n")
    print(json.dumps(stats, indent=2))

    # --- Interactive ---
    print("\n" + "-" * 70)
    user_url = input("\nEnter an image URL to classify (or Enter to skip) → ").strip()
    if user_url:
        r = classify_image(user_url, "User Image")
        print(f"\n  Document type : {r.document_type} ({r.document_type_confidence}%)")
        print(f"  Clinical status: {r.clinical_status} ({r.clinical_status_confidence}%)")
        print(f"  Urgency       : {r.urgency} ({r.urgency_confidence}%)")
        print(f"  Reasoning     : {r.reasoning}")


if __name__ == "__main__":
    demo_image_classification()
