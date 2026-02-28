"""
Exercise 1 — Medical Form Reader
=================================
Read medical form images and extract structured patient data as JSON.

Objectives
----------
* Send a form image to GPT-4o-mini via the Vision API
* Use a detailed extraction prompt to pull specific fields
* Parse the response into validated JSON
* Handle missing / illegible fields gracefully

Fields to extract:
  patient_name, date_of_birth, insurance_provider, insurance_id,
  chief_complaint, current_medications, allergies, provider_name
"""

import json
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

VISION_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Helper: build image message
# ---------------------------------------------------------------------------

def build_image_message(text: str, image_source: str, detail: str = "high") -> dict:
    """Create a user message with text + image.

    ``image_source`` can be an HTTPS URL *or* a local file path.
    Local files are base64-encoded automatically.
    """
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
# Required extraction fields & their descriptions
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "patient_name": "Full legal name of the patient",
    "date_of_birth": "Patient date of birth (MM/DD/YYYY)",
    "insurance_provider": "Name of health insurance company",
    "insurance_id": "Insurance member / policy ID number",
    "chief_complaint": "Primary reason for the visit",
    "current_medications": "List of current medications and dosages",
    "allergies": "Known allergies (medications, food, environmental)",
    "provider_name": "Name of the treating physician / provider",
}


def build_extraction_prompt() -> str:
    """Build a detailed extraction system prompt."""
    fields_block = "\n".join(
        f'  "{k}": "<{v}>"' for k, v in REQUIRED_FIELDS.items()
    )
    return f"""\
You are an expert medical document processor specializing in reading
handwritten and printed medical intake forms.

TASK: Examine the provided medical form image and extract the following
fields into a JSON object.  If a field is not visible or illegible,
set its value to null.  For list fields (medications, allergies),
return a JSON array of strings.

REQUIRED JSON STRUCTURE:
{{
{fields_block}
}}

RULES:
- Return ONLY valid JSON.  No markdown fences, no commentary.
- Dates should be normalized to MM/DD/YYYY.
- Medication entries should include dosage when visible (e.g., "Lisinopril 10mg").
- If the form is blank or does not contain medical data, return all nulls.
"""


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def extract_form_data(image_source: str) -> dict:
    """Send a medical form image to the Vision API and return structured data.

    Parameters
    ----------
    image_source : str
        URL or local path to the form image.

    Returns
    -------
    dict  — parsed JSON with the required fields.
    """
    system_prompt = build_extraction_prompt()
    user_message = build_image_message(
        "Extract all patient information from this medical form.",
        image_source,
        detail="high",
    )

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            user_message,
        ],
        max_tokens=600,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"_raw_response": raw, "_error": "Failed to parse JSON"}

    return data


def validate_extraction(data: dict) -> dict:
    """Validate extraction results and compute a completeness score.

    Returns a report dict with:
    - fields_found: int
    - fields_missing: list[str]
    - completeness_pct: float
    - data: the original extracted dict
    """
    found = 0
    missing = []
    for field in REQUIRED_FIELDS:
        value = data.get(field)
        if value is not None and value != "" and value != []:
            found += 1
        else:
            missing.append(field)

    total = len(REQUIRED_FIELDS)
    return {
        "fields_found": found,
        "fields_missing": missing,
        "completeness_pct": round(found / total * 100, 1),
        "data": data,
    }


# ---------------------------------------------------------------------------
# Demo with a public sample form
# ---------------------------------------------------------------------------

SAMPLE_FORM_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "8/8e/CMS-1500_claim_form.jpg/"
    "400px-CMS-1500_claim_form.jpg"
)


def demo_form_reader():
    """Run a full form-reading demo against a public CMS-1500 image."""
    print("=" * 70)
    print("  Exercise 1 — Medical Form Reader")
    print("=" * 70)
    print()
    print(f"Image: {SAMPLE_FORM_URL}\n")

    print("Extracting data from form image …\n")
    data = extract_form_data(SAMPLE_FORM_URL)

    print("=== Extracted Data ===\n")
    print(json.dumps(data, indent=2))

    report = validate_extraction(data)
    print(f"\n=== Validation Report ===")
    print(f"  Fields found   : {report['fields_found']} / {len(REQUIRED_FIELDS)}")
    print(f"  Completeness   : {report['completeness_pct']}%")
    if report["fields_missing"]:
        print(f"  Missing fields : {', '.join(report['fields_missing'])}")
    print()

    # --- Interactive mode: user can supply their own URL / path ---
    print("-" * 70)
    print("\nYou can also supply your own image URL or local path.\n")
    user_input = input("Enter image URL/path (or press Enter to skip) → ").strip()
    if user_input:
        print("\nProcessing your image …\n")
        result = extract_form_data(user_input)
        print(json.dumps(result, indent=2))
        rpt = validate_extraction(result)
        print(f"\nCompleteness: {rpt['completeness_pct']}%")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_form_reader()
