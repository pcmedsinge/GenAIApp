"""
Level 4 - Project 05: Healthcare Compliance
=============================================

HIPAA-aware AI workflows for healthcare. This module demonstrates:
- PHI detection (identifying the 18 HIPAA identifiers in clinical text)
- De-identification (replacing PHI with safe placeholders)
- Audit trails (logging every AI interaction for compliance)
- Clinical validation (physician-in-the-loop review workflows)

This is the capstone project for Level 4, integrating evaluation, validation,
guardrails, and compliance into production-ready healthcare AI patterns.

Usage:
    python main.py
"""

import os
import re
import json
import hashlib
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# DEMO 1: PHI Detection — Identify Protected Health Information
# ============================================================================

HIPAA_IDENTIFIERS = [
    "Names", "Geographic data (smaller than state)", "Dates (except year)",
    "Phone numbers", "Fax numbers", "Email addresses", "SSN",
    "Medical record numbers", "Health plan beneficiary numbers",
    "Account numbers", "Certificate/license numbers",
    "Vehicle identifiers", "Device identifiers", "Web URLs",
    "IP addresses", "Biometric identifiers", "Full-face photographs",
    "Any other unique identifying number/code",
]


def detect_phi(clinical_text: str) -> dict:
    """
    Use LLM to detect all 18 HIPAA identifier types in clinical text.
    Returns structured findings with identifier type, value, and location.
    """
    identifiers_list = "\n".join(f"  {i+1}. {ident}" for i, ident in enumerate(HIPAA_IDENTIFIERS))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""You are a HIPAA PHI detection system. Analyze clinical text and 
identify ALL instances of Protected Health Information.

The 18 HIPAA identifiers:
{identifiers_list}

Return JSON:
{{
    "phi_found": true/false,
    "total_count": number,
    "findings": [
        {{
            "identifier_type": "which of the 18 HIPAA types",
            "value": "the exact PHI text found",
            "hipaa_number": 1-18,
            "risk_level": "low" | "medium" | "high"
        }}
    ],
    "risk_summary": "overall risk assessment"
}}

Be thorough — find EVERY instance, including implied identifiers (e.g., 
"the patient's wife, Mary" contains a name)."""
            },
            {"role": "user", "content": f"Detect all PHI in:\n\n{clinical_text}"}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def demo_phi_detection():
    """DEMO 1: Identify Protected Health Information in clinical text."""
    print("\n" + "=" * 70)
    print("DEMO 1: PHI Detection (18 HIPAA Identifiers)")
    print("=" * 70)

    clinical_note = (
        "Patient: Sarah Johnson, DOB: 06/15/1982\n"
        "MRN: 00452189 | SSN: 312-45-6789\n"
        "Address: 456 Elm Street, Chicago, IL 60601\n"
        "Phone: (312) 555-0147 | Email: sjohnson@email.com\n"
        "Emergency Contact: Michael Johnson (husband) — (312) 555-0199\n\n"
        "Chief Complaint: Patient presents with 3-day history of worsening cough.\n"
        "History: Ms. Johnson was seen at Northwestern Memorial Hospital on 02/10/2026.\n"
        "She reports taking lisinopril 10mg daily prescribed by Dr. Robert Kim.\n"
        "Vitals: BP 128/82, HR 88, Temp 101.2°F, SpO2 96%.\n"
        "Assessment: Community-acquired pneumonia. Started on azithromycin 500mg.\n"
        "Follow-up scheduled for 02/17/2026 with Dr. Kim."
    )

    print(f"\nClinical Note:\n{clinical_note}\n")
    print("Scanning for PHI...\n")

    result = detect_phi(clinical_note)
    print(f"PHI Found: {result.get('phi_found', False)}")
    print(f"Total PHI instances: {result.get('total_count', 0)}")
    print(f"Risk Summary: {result.get('risk_summary', 'N/A')}\n")

    for finding in result.get("findings", []):
        risk_icon = {"low": "🟡", "medium": "🟠", "high": "🔴"}.get(
            finding.get("risk_level", ""), "⚪")
        print(f"  {risk_icon} [{finding.get('identifier_type', '?')}] "
              f"\"{finding.get('value', '')}\" (HIPAA #{finding.get('hipaa_number', '?')})")


# ============================================================================
# DEMO 2: De-identification — Replace PHI with Safe Placeholders
# ============================================================================

def deidentify_text(clinical_text: str) -> dict:
    """
    De-identify clinical text by replacing PHI with safe placeholders
    while preserving clinical meaning.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a clinical text de-identification system.
Replace all PHI with safe placeholders while preserving clinical meaning.

Replacement rules:
- Patient names → [PATIENT_NAME], [FAMILY_MEMBER_NAME]
- Provider names → [PROVIDER_NAME]  
- Dates → [DATE] (preserve temporal relationships like "3 days ago")
- SSN → [SSN_REDACTED]
- MRN → [MRN_REDACTED]
- Phone → [PHONE_REDACTED]
- Email → [EMAIL_REDACTED]
- Address → [ADDRESS_REDACTED]
- Facility names → [FACILITY_NAME]
- Ages → keep ages (ages are not PHI under Safe Harbor if <90)
- ZIP codes → keep first 3 digits only: "60601" → "606XX"

Return JSON:
{
    "deidentified_text": "the cleaned text",
    "replacements_made": number,
    "replacement_log": [
        {"original": "exact original text", "replacement": "what it was replaced with", "type": "PHI type"}
    ],
    "clinical_meaning_preserved": true/false
}"""
            },
            {"role": "user", "content": f"De-identify this clinical text:\n\n{clinical_text}"}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def demo_deidentification():
    """DEMO 2: Replace PHI with safe placeholders preserving clinical meaning."""
    print("\n" + "=" * 70)
    print("DEMO 2: De-identification")
    print("=" * 70)

    original_note = (
        "Patient Maria Santos (DOB: 11/03/1975, MRN: 99287341) presented to "
        "Springfield General Hospital on 01/22/2026 with acute chest pain. "
        "She was evaluated by Dr. Angela Reyes in the ED. ECG showed ST elevation "
        "in leads V1-V4. Troponin was 2.8 ng/mL. Patient underwent emergent PCI "
        "by Dr. Thomas Chen at 14:30. Drug-eluting stent placed in LAD. "
        "Post-procedure, Ms. Santos was started on dual antiplatelet therapy: "
        "aspirin 81mg daily and clopidogrel 75mg daily. Contact: (555) 234-8901. "
        "Home address: 789 Pine Ave, Springfield, IL 62704. "
        "Follow-up with Dr. Reyes on 02/05/2026."
    )

    print(f"\nOriginal note:\n{original_note}\n")
    print("De-identifying...\n")

    result = deidentify_text(original_note)

    print(f"De-identified text:\n{result.get('deidentified_text', 'ERROR')}\n")
    print(f"Replacements made: {result.get('replacements_made', 0)}")
    print(f"Clinical meaning preserved: {result.get('clinical_meaning_preserved', 'N/A')}")
    print(f"\nReplacement log:")
    for r in result.get("replacement_log", []):
        print(f"  {r.get('type', '?')}: \"{r.get('original', '')}\" → \"{r.get('replacement', '')}\"")


# ============================================================================
# DEMO 3: Audit Trail — Log Every AI Interaction
# ============================================================================

class AuditLogger:
    """Simple in-memory audit logger for AI interactions."""

    def __init__(self):
        self.logs = []

    def log_interaction(self, user_id: str, query: str, response: str,
                        model: str, guardrail_results: dict = None,
                        metadata: dict = None) -> dict:
        """Log a single AI interaction with full audit details."""
        entry = {
            "log_id": f"LOG-{len(self.logs) + 1:04d}",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "input_hash": hashlib.sha256(query.encode()).hexdigest()[:16],
            "output_hash": hashlib.sha256(response.encode()).hexdigest()[:16],
            "model": model,
            "query_length": len(query),
            "response_length": len(response),
            "guardrail_results": guardrail_results or {},
            "metadata": metadata or {},
        }
        self.logs.append(entry)
        return entry

    def get_logs(self, limit: int = 10) -> list:
        """Get recent audit log entries."""
        return self.logs[-limit:]

    def get_summary(self) -> dict:
        """Get summary statistics for the audit trail."""
        if not self.logs:
            return {"total_interactions": 0}

        models_used = {}
        guardrail_triggers = 0
        for log in self.logs:
            m = log.get("model", "unknown")
            models_used[m] = models_used.get(m, 0) + 1
            gr = log.get("guardrail_results", {})
            if gr and not gr.get("all_passed", True):
                guardrail_triggers += 1

        return {
            "total_interactions": len(self.logs),
            "unique_users": len(set(log["user_id"] for log in self.logs)),
            "models_used": models_used,
            "guardrail_triggers": guardrail_triggers,
            "date_range": {
                "first": self.logs[0]["timestamp"],
                "last": self.logs[-1]["timestamp"],
            },
        }


# Global audit logger instance
audit_logger = AuditLogger()


def demo_audit_trail():
    """DEMO 3: Log every AI interaction with timestamp, input hash, model, guardrail results."""
    print("\n" + "=" * 70)
    print("DEMO 3: Audit Trail")
    print("=" * 70)

    # Simulate several AI interactions
    interactions = [
        {
            "user": "DR-SMITH-001",
            "query": "What are the guidelines for managing diabetic ketoacidosis?",
            "guardrails": {"input_safe": True, "output_safe": True, "faithfulness": 0.95,
                           "all_passed": True},
        },
        {
            "user": "DR-SMITH-001",
            "query": "What is the lethal dose of potassium chloride IV?",
            "guardrails": {"input_safe": False, "output_safe": True, "faithfulness": None,
                           "all_passed": False, "blocked_reason": "potentially harmful query"},
        },
        {
            "user": "NP-JONES-042",
            "query": "What antibiotics cover MRSA?",
            "guardrails": {"input_safe": True, "output_safe": True, "faithfulness": 0.88,
                           "all_passed": True},
        },
        {
            "user": "DR-PATEL-017",
            "query": "Recommend insulin dosing for pediatric DKA",
            "guardrails": {"input_safe": True, "output_safe": True, "faithfulness": 0.72,
                           "all_passed": False, "blocked_reason": "low faithfulness score"},
        },
    ]

    print("\nSimulating AI interactions and logging...\n")

    for ix in interactions:
        # Simulate generating a response
        gen_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Provide a brief healthcare answer."},
                {"role": "user", "content": ix["query"]}
            ],
            max_tokens=100,
            temperature=0.3,
        )
        response_text = gen_response.choices[0].message.content

        # Log the interaction
        log_entry = audit_logger.log_interaction(
            user_id=ix["user"],
            query=ix["query"],
            response=response_text,
            model="gpt-4o-mini",
            guardrail_results=ix["guardrails"],
            metadata={"source": "demo", "session": "demo-session-001"},
        )
        print(f"  Logged: {log_entry['log_id']} | User: {ix['user']} | "
              f"Input hash: {log_entry['input_hash']}... | "
              f"Guardrails: {'PASS' if ix['guardrails']['all_passed'] else 'FAIL'}")

    # Show audit summary
    summary = audit_logger.get_summary()
    print(f"\nAudit Trail Summary:")
    print(f"  Total interactions: {summary['total_interactions']}")
    print(f"  Unique users: {summary['unique_users']}")
    print(f"  Models used: {summary['models_used']}")
    print(f"  Guardrail triggers: {summary['guardrail_triggers']}")

    # Show recent logs
    print(f"\nRecent logs:")
    for log in audit_logger.get_logs(3):
        gr = log.get("guardrail_results", {})
        status = "PASS" if gr.get("all_passed", True) else "FAIL"
        print(f"  [{log['log_id']}] {log['timestamp'][:19]} | {log['user_id']} | "
              f"{status} | hash={log['input_hash']}")


# ============================================================================
# DEMO 4: Clinical Validation — Physician-in-the-Loop Workflow
# ============================================================================

def generate_clinical_content(query: str, context: str = "") -> dict:
    """Generate clinical content that requires human validation."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are an AI clinical assistant. Generate a clinical response.
Mark your confidence level for each section. Format your response as:

CLINICAL ASSESSMENT:
[Your assessment]

RECOMMENDED ACTIONS:
[Your recommendations]

CONFIDENCE: [HIGH/MEDIUM/LOW]
EVIDENCE BASIS: [Guidelines referenced or "Clinical reasoning"]

IMPORTANT: This is AI-generated content requiring physician review."""
            },
            {"role": "user", "content": f"Context: {context}\n\nQuery: {query}" if context else query}
        ],
        temperature=0.3,
        max_tokens=400
    )
    content = response.choices[0].message.content

    return {
        "ai_generated_content": content,
        "model": "gpt-4o-mini",
        "timestamp": datetime.now().isoformat(),
        "requires_review": True,
        "review_status": "pending",
    }


def simulate_physician_review(content_package: dict) -> dict:
    """Simulate a physician reviewing AI-generated content."""
    # In production, this would be a real UI where a physician reviews and approves
    print("\n  📋 AI-Generated Content for Review:")
    print("  " + "─" * 55)
    for line in content_package["ai_generated_content"].split("\n"):
        print(f"    {line}")
    print("  " + "─" * 55)

    # Simulate physician decision (in production, this comes from UI)
    review_decision = {
        "reviewer_id": "DR-WILLIAMS-003",
        "review_timestamp": datetime.now().isoformat(),
        "decision": "approved_with_modifications",
        "modifications": "Added patient-specific dosing context",
        "clinical_accuracy": "confirmed",
        "ready_for_patient": True,
    }

    content_package["review_status"] = review_decision["decision"]
    content_package["review_details"] = review_decision
    return content_package


def demo_clinical_validation():
    """DEMO 4: Physician-in-the-loop workflow — AI generates, human reviews."""
    print("\n" + "=" * 70)
    print("DEMO 4: Clinical Validation (Physician-in-the-Loop)")
    print("=" * 70)

    query = "Patient with newly diagnosed atrial fibrillation. CHA2DS2-VASc score is 3. Recommend anticoagulation strategy."
    context = "72-year-old female, history of hypertension and diabetes. No prior bleeding events. Creatinine clearance 55 mL/min."

    print(f"\n  Query: {query}")
    print(f"  Context: {context}")

    # Step 1: AI generates
    print("\n  Step 1: AI generating clinical content...")
    content = generate_clinical_content(query, context)
    print(f"  ✅ Content generated (status: {content['review_status']})")

    # Step 2: Physician reviews
    print("\n  Step 2: Physician review (simulated)...")
    reviewed = simulate_physician_review(content)
    print(f"\n  Review decision: {reviewed['review_status']}")
    print(f"  Reviewer: {reviewed['review_details']['reviewer_id']}")
    print(f"  Clinical accuracy: {reviewed['review_details']['clinical_accuracy']}")
    print(f"  Ready for patient: {reviewed['review_details']['ready_for_patient']}")

    # Step 3: Audit the entire workflow
    audit_logger.log_interaction(
        user_id=reviewed["review_details"]["reviewer_id"],
        query=query,
        response=content["ai_generated_content"],
        model=content["model"],
        guardrail_results={"review_status": reviewed["review_status"],
                           "reviewer": reviewed["review_details"]["reviewer_id"]},
        metadata={"workflow": "clinical_validation", "context": context},
    )
    print(f"\n  ✅ Full workflow audited")


# ============================================================================
# Main Menu
# ============================================================================

def main():
    """Run healthcare compliance demos interactively."""
    print("=" * 70)
    print("  Level 4 - Project 05: Healthcare Compliance")
    print("  HIPAA-aware AI workflows — PHI, audit trails, clinical validation")
    print("=" * 70)

    demos = {
        "1": ("PHI Detection", demo_phi_detection),
        "2": ("De-identification", demo_deidentification),
        "3": ("Audit Trail", demo_audit_trail),
        "4": ("Clinical Validation", demo_clinical_validation),
    }

    while True:
        print("\nAvailable Demos:")
        for key, (name, _) in demos.items():
            print(f"  {key}. {name}")
        print("  A. Run all demos")
        print("  Q. Quit")

        choice = input("\nSelect demo (1-4, A, Q): ").strip().upper()

        if choice == "Q":
            print("\nGoodbye!")
            break
        elif choice == "A":
            for key in sorted(demos.keys()):
                demos[key][1]()
        elif choice in demos:
            demos[choice][1]()
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
