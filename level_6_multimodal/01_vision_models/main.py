"""
Level 6.1 — Vision Models: Image Understanding with GPT-4o
===========================================================
Demonstrates how to send images to GPT-4o / GPT-4o-mini and extract
descriptions, classifications, structured data, and comparisons.

Demos
-----
1. Image Analysis Basics     — send an image URL, get a description
2. Image Classification      — classify images into medical categories
3. Document Extraction       — extract structured data from a form image
4. Image Comparison          — compare two images, identify differences
"""

import json
import base64
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()   # reads OPENAI_API_KEY from environment

VISION_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def print_banner(title: str) -> None:
    """Print a section banner."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")


def print_separator() -> None:
    print("-" * 70)


def build_image_message(text: str, image_url: str, detail: str = "auto") -> dict:
    """Build a user message that contains text + an image URL."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": detail,      # "low", "high", or "auto"
                },
            },
        ],
    }


def build_two_image_message(text: str, url_1: str, url_2: str) -> dict:
    """Build a user message containing text + two image URLs."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": url_1, "detail": "auto"}},
            {"type": "image_url", "image_url": {"url": url_2, "detail": "auto"}},
        ],
    }


def encode_local_image(path: str) -> str:
    """Return a base64 data-URI for a local image file (PNG/JPEG)."""
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


# ---------------------------------------------------------------------------
# Sample image URLs (public, royalty-free medical diagrams)
# ---------------------------------------------------------------------------

SAMPLE_ANATOMY_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "d/d5/Anatomical_chart%2C_Cyclopaedia%2C_1728%2C_Volume_1.jpg/"
    "440px-Anatomical_chart%2C_Cyclopaedia%2C_1728%2C_Volume_1.jpg"
)

SAMPLE_CHART_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "1/1b/Sinus_rhythm_labels.svg/"
    "600px-Sinus_rhythm_labels.svg.png"
)

SAMPLE_FORM_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "8/8e/CMS-1500_claim_form.jpg/"
    "400px-CMS-1500_claim_form.jpg"
)


# ===================================================================
#  DEMO 1 — Image Analysis Basics
# ===================================================================
def demo_image_analysis_basics():
    """Send an image URL to GPT-4o-mini with vision and get a description.

    The Vision API accepts images inline inside the message array.  Each
    image is a content block of type ``image_url``.  You can mix text and
    images freely in a single message.

    Supported image sources
    -----------------------
    * **URL** — any publicly-accessible HTTPS URL
    * **Base64** — a data URI  ``data:image/<type>;base64,<data>``

    The ``detail`` parameter controls resolution:
    * ``low``  — 512 × 512 fixed (fast, cheaper)
    * ``high`` — full resolution up to 2048px (slower, more tokens)
    * ``auto`` — model decides
    """
    print_banner("DEMO 1 — Image Analysis Basics")

    print("Sending a public anatomical diagram to GPT-4o-mini for analysis …\n")
    print(f"Image URL: {SAMPLE_ANATOMY_URL}\n")

    message = build_image_message(
        text=(
            "You are a medical education assistant. "
            "Describe this anatomical image in detail. "
            "Identify the body systems shown and any labels visible."
        ),
        image_url=SAMPLE_ANATOMY_URL,
    )

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[message],
        max_tokens=600,
    )

    description = response.choices[0].message.content
    print("=== GPT-4o-mini Image Description ===\n")
    print(description)

    # --- token usage ---
    usage = response.usage
    print(f"\nTokens — prompt: {usage.prompt_tokens}  "
          f"completion: {usage.completion_tokens}  "
          f"total: {usage.total_tokens}")

    # --- demonstrate base64 pattern (without a real file) ---
    print_separator()
    print("\n--- Base64 Upload Pattern (code only, no real file) ---\n")
    print("""\
# To send a local file instead of a URL:
#
# data_uri = encode_local_image("xray.png")
# message = build_image_message(
#     text="Describe this X-ray.",
#     image_url=data_uri,
# )
""")


# ===================================================================
#  DEMO 2 — Image Classification
# ===================================================================

CLASSIFICATION_CATEGORIES = [
    "medical_form",
    "lab_report",
    "prescription",
    "clinical_photo",
    "radiology_image",
    "anatomical_diagram",
    "ecg_tracing",
    "unknown",
]


def demo_image_classification():
    """Classify images into predefined medical-document categories.

    Strategy: ask the model to return JSON with a ``category`` field
    (one of our labels) and a ``confidence`` score from 0-100.
    """
    print_banner("DEMO 2 — Image Classification")

    images_to_classify = [
        ("Anatomical Diagram", SAMPLE_ANATOMY_URL),
        ("ECG / Sinus Rhythm", SAMPLE_CHART_URL),
        ("CMS-1500 Claim Form", SAMPLE_FORM_URL),
    ]

    system_prompt = (
        "You are a medical image classifier. Given an image, classify it into "
        "exactly ONE of the following categories:\n\n"
        + "\n".join(f"  - {c}" for c in CLASSIFICATION_CATEGORIES)
        + "\n\nRespond ONLY with valid JSON: "
        '{{"category": "<label>", "confidence": <0-100>, "reasoning": "<brief>"}}'
    )

    for label, url in images_to_classify:
        print(f"Classifying: {label}")
        print(f"  URL: {url[:80]}…\n")

        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                build_image_message("Classify this medical image.", url),
            ],
            max_tokens=200,
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()
        # Try to parse JSON
        try:
            cleaned = raw
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(cleaned)
            print(f"  Category  : {result.get('category', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 'N/A')}%")
            print(f"  Reasoning : {result.get('reasoning', 'N/A')}")
        except json.JSONDecodeError:
            print(f"  Raw response: {raw}")
        print_separator()


# ===================================================================
#  DEMO 3 — Document Extraction
# ===================================================================
def demo_document_extraction():
    """Extract structured data from a medical form image.

    We ask GPT-4o-mini to read the CMS-1500 claim form and return
    structured JSON with fields like patient name, insurer, dates, etc.
    """
    print_banner("DEMO 3 — Document Extraction")

    print("Sending CMS-1500 claim form image for structured extraction …\n")
    print(f"Image URL: {SAMPLE_FORM_URL}\n")

    extraction_prompt = """\
You are an expert medical document processor.  Examine this medical claim
form image carefully and extract as much structured data as you can.

Return your answer as a JSON object with these fields (use null for any
field you cannot read):

{
  "form_type": "<string>",
  "patient_name": "<string>",
  "patient_dob": "<string>",
  "insured_name": "<string>",
  "insurance_id": "<string>",
  "diagnosis_codes": ["<string>", ...],
  "service_date": "<string>",
  "provider_name": "<string>",
  "total_charge": "<string>",
  "additional_notes": "<string>"
}

Return ONLY valid JSON. No markdown fences.
"""

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[build_image_message(extraction_prompt, SAMPLE_FORM_URL, detail="high")],
        max_tokens=600,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    try:
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
        print("=== Extracted Fields ===\n")
        print(json.dumps(data, indent=2))
    except json.JSONDecodeError:
        print("=== Raw Response (not valid JSON) ===\n")
        print(raw)

    print_separator()
    print("\nTip: Use detail='high' for form images to improve OCR accuracy.")
    print("Tip: Specify exact field names in the prompt to guide extraction.")


# ===================================================================
#  DEMO 4 — Image Comparison
# ===================================================================
def demo_image_comparison():
    """Compare two images and identify differences.

    GPT-4o can receive multiple images in one message.  This is useful
    for before/after comparisons in clinical settings (e.g., treatment
    progression, wound healing, radiographic changes).
    """
    print_banner("DEMO 4 — Image Comparison")

    print("Comparing two public medical images …\n")
    print(f"Image 1 (Anatomy) : {SAMPLE_ANATOMY_URL[:70]}…")
    print(f"Image 2 (ECG)     : {SAMPLE_CHART_URL[:70]}…\n")

    comparison_prompt = """\
You are a medical imaging specialist.  You have been given TWO images.

For each image:
1. Describe what the image shows.
2. Identify the type of medical content.

Then provide:
3. A comparison — how do these images differ in content, style, and purpose?
4. In a clinical setting, when would each image be useful?

Structure your answer with clear numbered sections.
"""

    message = build_two_image_message(
        comparison_prompt, SAMPLE_ANATOMY_URL, SAMPLE_CHART_URL,
    )

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[message],
        max_tokens=800,
    )

    print("=== Image Comparison Results ===\n")
    print(response.choices[0].message.content)

    # --- Demonstrate programmatic comparison pattern ---
    print_separator()
    print("\n--- Before/After Comparison Pattern ---\n")
    print("""\
# In a real clinical workflow you might compare before/after images:
#
# message = build_two_image_message(
#     text="Compare the wound in Image 1 (Day 0) with Image 2 (Day 14). "
#          "Has the wound improved? Describe changes in size, color, edges.",
#     url_1="https://example.com/wound_day0.jpg",
#     url_2="https://example.com/wound_day14.jpg",
# )
#
# This pattern works for radiology follow-ups, dermatology progression,
# post-surgical healing, and treatment response monitoring.
""")


# ===================================================================
#  Main menu
# ===================================================================
DEMOS = {
    "1": ("Image Analysis Basics", demo_image_analysis_basics),
    "2": ("Image Classification", demo_image_classification),
    "3": ("Document Extraction", demo_document_extraction),
    "4": ("Image Comparison", demo_image_comparison),
}


def main():
    print_banner("Level 6.1 — Vision Models: Image Understanding with GPT-4o")
    print("This module demonstrates GPT-4o / GPT-4o-mini vision capabilities.")
    print("Each demo sends one or more images to the API and processes results.\n")

    while True:
        print("\nAvailable demos:")
        for key, (title, _) in DEMOS.items():
            print(f"  [{key}] {title}")
        print("  [a] Run ALL demos")
        print("  [q] Quit\n")

        choice = input("Select demo → ").strip().lower()

        if choice == "q":
            print("Goodbye!")
            break
        elif choice == "a":
            for _, (_, func) in DEMOS.items():
                func()
                print_separator()
        elif choice in DEMOS:
            DEMOS[choice][1]()
        else:
            print("Invalid selection. Try again.")


if __name__ == "__main__":
    main()
