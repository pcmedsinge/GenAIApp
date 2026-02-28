"""
Exercise 1: PII Detector for Clinical Text
============================================

Skills practiced:
- Regular expression pattern matching for structured identifiers
- LLM-based entity recognition for unstructured PII
- Combining regex + LLM for comprehensive PHI detection
- Text masking and de-identification

Healthcare context:
Protected Health Information (PHI) must be identified and masked before clinical
text can be shared, stored in logs, or used for training. HIPAA defines 18 types
of identifiers that constitute PHI. This exercise builds a detector for the most
common ones: patient names, dates of birth, Social Security Numbers, Medical
Record Numbers, phone numbers, and email addresses.

A robust PII detector uses both:
- Regex: Catches structured patterns (SSNs, phone numbers, MRNs)
- LLM: Catches unstructured PII (names, addresses, contextual dates)

Usage:
    python exercise_1_pii_detector.py
"""

import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Regex-based PII detection
# ---------------------------------------------------------------------------

PII_PATTERNS = {
    "SSN": {
        "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
        "description": "Social Security Number (XXX-XX-XXXX)",
    },
    "MRN": {
        "pattern": r"\b(?:MRN|mrn)[:\s#]*(\d{6,10})\b",
        "description": "Medical Record Number",
    },
    "PHONE": {
        "pattern": r"\b(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b",
        "description": "Phone number",
    },
    "EMAIL": {
        "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "description": "Email address",
    },
    "DOB": {
        "pattern": r"\b(?:DOB|Date of Birth|D\.O\.B\.)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        "description": "Date of birth",
    },
    "DATE": {
        "pattern": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        "description": "Date (may be DOB, admission, etc.)",
    },
    "ZIP": {
        "pattern": r"\b\d{5}(?:-\d{4})?\b",
        "description": "ZIP code",
    },
}


def detect_pii_regex(text: str) -> list[dict]:
    """Detect PII using regular expression patterns."""
    findings = []
    for pii_type, config in PII_PATTERNS.items():
        for match in re.finditer(config["pattern"], text, re.IGNORECASE):
            findings.append({
                "type": pii_type,
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
                "method": "regex",
                "description": config["description"],
            })
    return findings


# ---------------------------------------------------------------------------
# LLM-based PII detection
# ---------------------------------------------------------------------------

def detect_pii_llm(text: str) -> list[dict]:
    """Use LLM to detect PII that regex might miss (names, addresses, context)."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a PHI (Protected Health Information) detector.
Analyze the clinical text and identify ALL instances of PHI. Return JSON:
{
    "findings": [
        {
            "type": "PATIENT_NAME" | "PROVIDER_NAME" | "ADDRESS" | "DOB" | "AGE" | "PHONE" | "EMAIL" | "SSN" | "MRN" | "FACILITY_NAME" | "DATE" | "OTHER_PHI",
            "value": "exact text found",
            "description": "what this identifier is"
        }
    ]
}

Look for ALL 18 HIPAA identifiers:
1. Names  2. Dates (except year)  3. Phone numbers  4. Fax numbers
5. Email addresses  6. SSN  7. MRN  8. Health plan numbers
9. Account numbers  10. Certificate/license numbers  11. Vehicle IDs
12. Device IDs  13. URLs  14. IP addresses  15. Biometric IDs
16. Full-face photos  17. Any other unique identifying number
18. Geographic data smaller than state"""
            },
            {"role": "user", "content": f"Find all PHI in this clinical text:\n\n{text}"}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    result = json.loads(response.choices[0].message.content)
    findings = []
    for item in result.get("findings", []):
        findings.append({
            "type": item.get("type", "UNKNOWN"),
            "value": item.get("value", ""),
            "method": "llm",
            "description": item.get("description", ""),
        })
    return findings


# ---------------------------------------------------------------------------
# Combined detection + masking
# ---------------------------------------------------------------------------

def detect_pii_combined(text: str) -> list[dict]:
    """Combine regex and LLM detection for comprehensive PII finding."""
    regex_findings = detect_pii_regex(text)
    llm_findings = detect_pii_llm(text)

    # Deduplicate based on value
    seen_values = set()
    combined = []
    for f in regex_findings:
        key = f["value"].strip().lower()
        if key not in seen_values:
            seen_values.add(key)
            combined.append(f)
    for f in llm_findings:
        key = f["value"].strip().lower()
        if key not in seen_values:
            seen_values.add(key)
            combined.append(f)

    return combined


def mask_pii(text: str, findings: list[dict]) -> str:
    """Replace detected PII with masked placeholders."""
    masked_text = text
    # Sort by value length descending to handle overlapping matches
    sorted_findings = sorted(findings, key=lambda x: len(x["value"]), reverse=True)
    for finding in sorted_findings:
        value = finding["value"]
        pii_type = finding["type"]
        placeholder = f"[{pii_type}_REDACTED]"
        masked_text = masked_text.replace(value, placeholder)
    return masked_text


def generate_pii_report(findings: list[dict]) -> str:
    """Generate a summary report of PII findings."""
    if not findings:
        return "No PII detected."

    report_lines = [
        f"PII Detection Report",
        f"{'=' * 50}",
        f"Total findings: {len(findings)}",
        f"",
        f"{'Type':<20} {'Value':<30} {'Method':<8}",
        f"{'-' * 20} {'-' * 30} {'-' * 8}",
    ]
    for f in findings:
        display_value = f["value"][:28] + ".." if len(f["value"]) > 30 else f["value"]
        report_lines.append(f"{f['type']:<20} {display_value:<30} {f['method']:<8}")

    # Summary by type
    type_counts = {}
    for f in findings:
        type_counts[f["type"]] = type_counts.get(f["type"], 0) + 1
    report_lines.append(f"\nSummary by type:")
    for pii_type, count in sorted(type_counts.items()):
        report_lines.append(f"  {pii_type}: {count}")

    return "\n".join(report_lines)


# ---------------------------------------------------------------------------
# Test clinical notes
# ---------------------------------------------------------------------------

SAMPLE_NOTES = [
    {
        "title": "Discharge Summary",
        "text": (
            "Patient: Maria Garcia (DOB: 04/22/1978, MRN: 10034521)\n"
            "SSN: 421-55-6789\n"
            "Address: 1234 Oak Street, Springfield, IL 62704\n"
            "Phone: (555) 234-5678 | Email: mgarcia@email.com\n\n"
            "Attending Physician: Dr. James Wilson, Internal Medicine\n"
            "Discharge Date: 01/15/2026\n\n"
            "Ms. Garcia was admitted on 01/10/2026 for community-acquired pneumonia. "
            "She was treated with IV levofloxacin 750mg daily and transitioned to oral "
            "antibiotics on day 3. Chest X-ray on 01/13/2026 showed improvement. "
            "She is discharged in stable condition with follow-up in 2 weeks."
        ),
    },
    {
        "title": "Progress Note",
        "text": (
            "Date: 02/05/2026\n"
            "Patient: Robert Chen, 67-year-old male (MRN#8876543)\n"
            "Chief Complaint: Follow-up for type 2 diabetes management.\n\n"
            "Mr. Chen reports compliance with metformin 1000mg BID. His wife, "
            "Linda Chen, confirms medication adherence. Home glucose readings "
            "range 120-180 mg/dL fasting. A1c was 7.8% on 01/20/2026.\n\n"
            "Plan: Continue current regimen. Referral to Dr. Sarah Patel at "
            "Springfield Endocrine Associates, phone 555-987-6543.\n"
            "Next visit: 05/05/2026."
        ),
    },
    {
        "title": "Lab Report (minimal PHI expected)",
        "text": (
            "Lab Results Summary\n"
            "CBC with differential showing WBC 8.2, RBC 4.5, Hgb 14.2, Hct 42%, "
            "Platelets 250. BMP within normal limits: Na 140, K 4.1, Cl 102, "
            "CO2 24, BUN 15, Creatinine 0.9, Glucose 105. All values within "
            "reference ranges. No critical values to report."
        ),
    },
]


def main():
    """Run the PII detector on sample clinical notes."""
    print("=" * 70)
    print("  Exercise 1: PII Detector for Clinical Text")
    print("  Regex + LLM combined detection")
    print("=" * 70)

    for note in SAMPLE_NOTES:
        print(f"\n{'─' * 70}")
        print(f"📄 {note['title']}")
        print(f"{'─' * 70}")
        print(f"\nOriginal text:\n{note['text']}\n")

        # Step 1: Detect PII
        print("Scanning for PII...")
        findings = detect_pii_combined(note["text"])

        # Step 2: Report
        report = generate_pii_report(findings)
        print(f"\n{report}")

        # Step 3: Mask
        if findings:
            masked = mask_pii(note["text"], findings)
            print(f"\nMasked text:\n{masked}")
        else:
            print("\nNo PII to mask — text is clean.")

    # Summary stats
    print(f"\n{'=' * 70}")
    print("Detection complete. In production, masked text would be used for:")
    print("  - AI model training data")
    print("  - Sharing with third parties")
    print("  - Research datasets")
    print("  - Audit log storage")


if __name__ == "__main__":
    main()
