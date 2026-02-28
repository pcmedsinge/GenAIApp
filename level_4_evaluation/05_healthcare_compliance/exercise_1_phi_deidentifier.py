"""
Exercise 1: Comprehensive PHI De-identification System
=======================================================

Skills practiced:
- Regex-based detection of structured PHI (SSN, MRN, phone, email, dates)
- LLM-based detection of unstructured PHI (names, addresses, facility names)
- Safe-Harbor method de-identification (HIPAA §164.514(b))
- Preserving clinical meaning after PHI removal
- Testing across multiple clinical note types

Healthcare context:
HIPAA's Safe Harbor method requires removal or generalization of 18 specific
identifier types before clinical text can be considered de-identified. A robust
de-identification system must handle:
- Free-text clinical notes with embedded PHI
- Structured fields (demographics, contact info)
- Contextual identifiers (relationships, facilities, providers)

This exercise builds a comprehensive de-identification pipeline and tests it
against three different clinical note formats.

Usage:
    python exercise_1_phi_deidentifier.py
"""

import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Regex patterns for structured PHI
# ---------------------------------------------------------------------------

PHI_REGEX_PATTERNS = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "MRN": r"\b(?:MRN|mrn)[:\s#]*\d{5,10}\b",
    "PHONE": r"\b(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b",
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "DATE": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    "DOB_LABELED": r"(?:DOB|Date of Birth|D\.O\.B\.)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
    "ZIP": r"\b\d{5}(?:-\d{4})?\b",
}


def detect_regex_phi(text: str) -> list[dict]:
    """Find structured PHI using regex patterns."""
    findings = []
    for phi_type, pattern in PHI_REGEX_PATTERNS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            findings.append({
                "type": phi_type,
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
                "method": "regex",
            })
    return findings


# ---------------------------------------------------------------------------
# LLM-based PHI detection
# ---------------------------------------------------------------------------

def detect_llm_phi(text: str) -> list[dict]:
    """Use LLM to detect unstructured PHI that regex misses."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a HIPAA PHI detector. Find ALL Protected Health Information
in the clinical text. Focus on identifiers that regex cannot easily catch:
- Patient names (first, last, nicknames)
- Provider/physician names
- Family member names
- Facility/hospital names
- Street addresses (number + street name)
- City names (when combined with other location data)
- Ages over 89
- Any unique identifying codes

Return JSON:
{
    "findings": [
        {"type": "PATIENT_NAME" | "PROVIDER_NAME" | "FAMILY_NAME" | "FACILITY" | "ADDRESS" | "CITY" | "AGE_OVER_89" | "OTHER",
         "value": "exact text",
         "context": "brief surrounding context"}
    ]
}"""
            },
            {"role": "user", "content": f"Find all PHI:\n\n{text}"}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    result = json.loads(response.choices[0].message.content)
    return [
        {"type": f["type"], "value": f["value"], "method": "llm",
         "context": f.get("context", "")}
        for f in result.get("findings", [])
    ]


# ---------------------------------------------------------------------------
# Combined detection
# ---------------------------------------------------------------------------

def detect_all_phi(text: str) -> list[dict]:
    """Combine regex and LLM detection, deduplicate results."""
    regex_results = detect_regex_phi(text)
    llm_results = detect_llm_phi(text)

    seen = set()
    combined = []
    for finding in regex_results + llm_results:
        key = finding["value"].strip().lower()
        if key not in seen and len(key) > 1:
            seen.add(key)
            combined.append(finding)
    return combined


# ---------------------------------------------------------------------------
# De-identification (masking)
# ---------------------------------------------------------------------------

REPLACEMENT_MAP = {
    "SSN": "[SSN_REDACTED]",
    "MRN": "[MRN_REDACTED]",
    "PHONE": "[PHONE_REDACTED]",
    "EMAIL": "[EMAIL_REDACTED]",
    "DATE": "[DATE_REDACTED]",
    "DOB_LABELED": "[DOB_REDACTED]",
    "ZIP": "[ZIP_REDACTED]",
    "PATIENT_NAME": "[PATIENT]",
    "PROVIDER_NAME": "[PROVIDER]",
    "FAMILY_NAME": "[FAMILY_MEMBER]",
    "FACILITY": "[FACILITY]",
    "ADDRESS": "[ADDRESS_REDACTED]",
    "CITY": "[CITY_REDACTED]",
    "AGE_OVER_89": "[AGE_REDACTED]",
    "OTHER": "[REDACTED]",
}


def deidentify(text: str, findings: list[dict]) -> tuple[str, list[dict]]:
    """
    Replace detected PHI with safe placeholders.
    Returns (deidentified_text, replacement_log).
    """
    masked = text
    log = []
    # Sort by length descending to handle substrings correctly
    sorted_findings = sorted(findings, key=lambda x: len(x["value"]), reverse=True)

    for finding in sorted_findings:
        original = finding["value"]
        phi_type = finding["type"]
        replacement = REPLACEMENT_MAP.get(phi_type, "[REDACTED]")

        if original in masked:
            masked = masked.replace(original, replacement)
            log.append({
                "original": original,
                "replacement": replacement,
                "type": phi_type,
                "method": finding.get("method", "unknown"),
            })

    return masked, log


# ---------------------------------------------------------------------------
# Verification: check if any PHI remains
# ---------------------------------------------------------------------------

def verify_deidentification(original: str, deidentified: str) -> dict:
    """Use LLM to verify that de-identification is complete."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Check if the de-identified text still contains any PHI.
Return JSON:
{
    "is_clean": true/false,
    "remaining_phi": [
        {"value": "PHI text found", "type": "identifier type"}
    ],
    "clinical_meaning_preserved": true/false,
    "notes": "any observations"
}"""
            },
            {
                "role": "user",
                "content": (f"ORIGINAL:\n{original}\n\n"
                            f"DE-IDENTIFIED:\n{deidentified}")
            }
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Test clinical notes
# ---------------------------------------------------------------------------

CLINICAL_NOTES = [
    {
        "title": "Emergency Department Note",
        "text": (
            "ED Note — 02/14/2026, 03:45 AM\n"
            "Patient: James Rodriguez, 54-year-old male\n"
            "DOB: 08/19/1971 | MRN: 55128903 | SSN: 478-55-1234\n"
            "Address: 2201 Maple Drive, Austin, TX 78701\n"
            "Phone: (512) 555-3847 | Email: jrodriguez@gmail.com\n\n"
            "CC: Severe chest pain radiating to left arm, onset 2 hours ago.\n"
            "HPI: Mr. Rodriguez called 911 after developing crushing substernal "
            "chest pain while watching TV. His wife, Elena Rodriguez, reports he "
            "was clutching his chest and appeared diaphoretic. EMS administered "
            "aspirin 325mg and nitroglycerin SL x2 en route.\n\n"
            "PMH: HTN, hyperlipidemia, T2DM. Current meds: metformin 1000mg BID, "
            "atorvastatin 40mg daily, lisinopril 20mg daily.\n"
            "Attending: Dr. Patricia Owens, Emergency Medicine\n"
            "Facility: St. David's Medical Center, Austin TX"
        ),
    },
    {
        "title": "Discharge Summary",
        "text": (
            "Discharge Summary\n"
            "Patient: Aisha Patel, 38F (MRN#7729104)\n"
            "Admission: 01/28/2026 | Discharge: 02/01/2026\n"
            "Attending: Dr. Michael Chang, Gastroenterology\n\n"
            "Diagnosis: Acute pancreatitis secondary to gallstones.\n"
            "Hospital Course: Ms. Patel presented with severe epigastric pain, "
            "nausea, and vomiting. Lipase was 1,847 U/L. CT abdomen showed "
            "peripancreatic inflammation with cholelithiasis. She was managed "
            "conservatively with NPO, IV fluids, and pain control. Surgical "
            "consultation by Dr. Lisa Wong recommended cholecystectomy after "
            "resolution of acute inflammation.\n\n"
            "Discharge Meds: Pantoprazole 40mg daily, acetaminophen 500mg PRN.\n"
            "Follow-up: Dr. Chang on 02/15/2026 at Springfield GI Associates, "
            "phone (555) 892-4301. Surgery consult with Dr. Wong on 02/22/2026.\n"
            "Emergency contact: Raj Patel (husband) — (555) 892-4400"
        ),
    },
    {
        "title": "Progress Note (Outpatient)",
        "text": (
            "Progress Note — 02/20/2026\n"
            "Patient: William O'Brien, 72M\n"
            "DOB: 05/30/1953 | MRN: 33019287\n"
            "PCP: Dr. Susan Nakamura, Family Medicine\n\n"
            "Reason for Visit: Routine diabetes follow-up.\n"
            "Mr. O'Brien reports good compliance with medications. His daughter, "
            "Kathleen O'Brien, accompanies him and confirms he checks glucose "
            "twice daily. Fasting readings range 130-160 mg/dL.\n\n"
            "Labs (02/15/2026): A1c 7.6%, Creatinine 1.2, eGFR 62.\n"
            "Meds: Metformin 1000mg BID, glipizide 5mg daily, empagliflozin 10mg daily.\n"
            "Plan: Continue current regimen. Repeat A1c in 3 months. Referral to "
            "Dr. Ahmad al-Rashid, Endocrinology, at University Health Center.\n"
            "Address on file: 445 Birch Lane, Apt 3B, Portland, OR 97201\n"
            "Phone: 503-555-7722 | Email: wob1953@yahoo.com"
        ),
    },
]


def process_note(note: dict):
    """Process a single clinical note through the full de-identification pipeline."""
    print(f"\n{'━' * 70}")
    print(f"📄 {note['title']}")
    print(f"{'━' * 70}")

    # Step 1: Detect
    print("\n  Step 1: Detecting PHI...")
    findings = detect_all_phi(note["text"])
    print(f"  Found {len(findings)} PHI instances:")
    for f in findings:
        print(f"    [{f['type']}] \"{f['value']}\" ({f['method']})")

    # Step 2: De-identify
    print(f"\n  Step 2: De-identifying...")
    deidentified, replacement_log = deidentify(note["text"], findings)
    print(f"  Made {len(replacement_log)} replacements")

    print(f"\n  De-identified text:")
    for line in deidentified.split("\n"):
        print(f"    {line}")

    # Step 3: Verify
    print(f"\n  Step 3: Verifying completeness...")
    verification = verify_deidentification(note["text"], deidentified)
    is_clean = verification.get("is_clean", False)
    clinical_ok = verification.get("clinical_meaning_preserved", False)
    print(f"  Clean (no remaining PHI): {'✅ Yes' if is_clean else '❌ No'}")
    print(f"  Clinical meaning preserved: {'✅ Yes' if clinical_ok else '⚠️  Partial'}")

    remaining = verification.get("remaining_phi", [])
    if remaining:
        print(f"  ⚠️  Remaining PHI detected:")
        for r in remaining:
            print(f"    - [{r.get('type', '?')}] \"{r.get('value', '')}\"")

    return {
        "title": note["title"],
        "phi_count": len(findings),
        "replacements": len(replacement_log),
        "is_clean": is_clean,
        "clinical_preserved": clinical_ok,
    }


def main():
    """Run the PHI de-identification system on sample clinical notes."""
    print("=" * 70)
    print("  Exercise 1: Comprehensive PHI De-identification")
    print("  Regex + LLM detection → Masking → Verification")
    print("=" * 70)

    results = []
    for note in CLINICAL_NOTES:
        result = process_note(note)
        results.append(result)

    # Overall summary
    print(f"\n{'=' * 70}")
    print("De-identification Summary")
    print(f"{'=' * 70}")
    total_phi = sum(r["phi_count"] for r in results)
    total_replacements = sum(r["replacements"] for r in results)
    clean_count = sum(1 for r in results if r["is_clean"])

    print(f"  Notes processed:      {len(results)}")
    print(f"  Total PHI found:      {total_phi}")
    print(f"  Total replacements:   {total_replacements}")
    print(f"  Fully de-identified:  {clean_count}/{len(results)}")
    print(f"\n  Per-note breakdown:")
    for r in results:
        status = "✅" if r["is_clean"] else "⚠️ "
        print(f"    {status} {r['title']}: {r['phi_count']} PHI → "
              f"{r['replacements']} replacements | "
              f"Clinical preserved: {r['clinical_preserved']}")


if __name__ == "__main__":
    main()
