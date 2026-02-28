"""
Exercise 2: Audit Logger for AI-Assisted Clinical Decisions
=============================================================

Skills practiced:
- Structured audit logging with JSON storage
- Input/output hashing for integrity verification
- Guardrail result tracking per interaction
- Querying audit trails by user, date, and status
- Compliance-ready log format

Healthcare context:
Every AI-assisted clinical decision must be auditable. Regulators, risk managers,
and quality teams need to answer questions like:
- Who used the AI system and when?
- What was the input and what did the AI respond?
- Did the guardrails flag anything?
- Was the output reviewed by a clinician?

This exercise builds a complete audit logging system that stores interactions
as JSON, supports querying by multiple criteria, and generates audit summaries.

Usage:
    python exercise_2_audit_logger.py
"""

import os
import json
import hashlib
import time
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Audit Log Entry
# ---------------------------------------------------------------------------

class AuditEntry:
    """A single audit log entry for an AI interaction."""

    def __init__(self, user_id: str, query: str, response: str,
                 model: str, guardrail_results: dict = None,
                 confidence_score: float = None, metadata: dict = None):
        self.log_id = f"AUDIT-{int(time.time() * 1000)}-{hash(query) % 10000:04d}"
        self.timestamp = datetime.now().isoformat()
        self.user_id = user_id
        self.query = query
        self.response = response
        self.input_hash = hashlib.sha256(query.encode()).hexdigest()
        self.output_hash = hashlib.sha256(response.encode()).hexdigest()
        self.model = model
        self.guardrail_results = guardrail_results or {}
        self.confidence_score = confidence_score
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "log_id": self.log_id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "query": self.query,
            "response_preview": self.response[:200] + "..." if len(self.response) > 200 else self.response,
            "response_length": len(self.response),
            "model": self.model,
            "guardrail_results": self.guardrail_results,
            "confidence_score": self.confidence_score,
            "metadata": self.metadata,
        }

    def __repr__(self):
        return f"AuditEntry(log_id={self.log_id}, user={self.user_id}, model={self.model})"


# ---------------------------------------------------------------------------
# Audit Logger with JSON storage
# ---------------------------------------------------------------------------

class AuditLogger:
    """
    Audit logging system for AI-assisted clinical decisions.
    Stores logs in memory and can export to JSON file.
    """

    def __init__(self, log_file: str = None):
        self.entries: list[AuditEntry] = []
        self.log_file = log_file

    def log(self, user_id: str, query: str, response: str, model: str,
            guardrail_results: dict = None, confidence_score: float = None,
            metadata: dict = None) -> AuditEntry:
        """Log a new AI interaction."""
        entry = AuditEntry(
            user_id=user_id,
            query=query,
            response=response,
            model=model,
            guardrail_results=guardrail_results,
            confidence_score=confidence_score,
            metadata=metadata,
        )
        self.entries.append(entry)
        return entry

    def query_by_user(self, user_id: str) -> list[AuditEntry]:
        """Get all entries for a specific user."""
        return [e for e in self.entries if e.user_id == user_id]

    def query_by_guardrail_status(self, passed: bool) -> list[AuditEntry]:
        """Get entries where guardrails passed or failed."""
        results = []
        for e in self.entries:
            gr = e.guardrail_results
            entry_passed = gr.get("all_passed", True)
            if entry_passed == passed:
                results.append(e)
        return results

    def query_by_model(self, model: str) -> list[AuditEntry]:
        """Get all entries using a specific model."""
        return [e for e in self.entries if e.model == model]

    def query_low_confidence(self, threshold: float = 0.7) -> list[AuditEntry]:
        """Get entries with confidence below threshold."""
        return [e for e in self.entries
                if e.confidence_score is not None and e.confidence_score < threshold]

    def get_summary(self) -> dict:
        """Generate audit trail summary statistics."""
        if not self.entries:
            return {"total_entries": 0}

        users = set(e.user_id for e in self.entries)
        models = {}
        guardrail_failures = 0
        confidence_scores = []

        for e in self.entries:
            models[e.model] = models.get(e.model, 0) + 1
            if not e.guardrail_results.get("all_passed", True):
                guardrail_failures += 1
            if e.confidence_score is not None:
                confidence_scores.append(e.confidence_score)

        avg_confidence = (sum(confidence_scores) / len(confidence_scores)
                          if confidence_scores else None)

        return {
            "total_entries": len(self.entries),
            "unique_users": len(users),
            "users": sorted(users),
            "models_used": models,
            "guardrail_failures": guardrail_failures,
            "guardrail_failure_rate": guardrail_failures / len(self.entries) if self.entries else 0,
            "avg_confidence": round(avg_confidence, 3) if avg_confidence else None,
            "low_confidence_count": len(self.query_low_confidence()),
            "date_range": {
                "first": self.entries[0].timestamp,
                "last": self.entries[-1].timestamp,
            },
        }

    def export_json(self, filepath: str = None) -> str:
        """Export audit trail as JSON string (and optionally to file)."""
        data = {
            "audit_trail": [e.to_dict() for e in self.entries],
            "summary": self.get_summary(),
            "exported_at": datetime.now().isoformat(),
        }
        json_str = json.dumps(data, indent=2)
        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)
        return json_str

    def verify_integrity(self, entry: AuditEntry, original_query: str,
                         original_response: str) -> bool:
        """Verify that stored hashes match the original content."""
        query_hash = hashlib.sha256(original_query.encode()).hexdigest()
        response_hash = hashlib.sha256(original_response.encode()).hexdigest()
        return (entry.input_hash == query_hash and
                entry.output_hash == response_hash)


# ---------------------------------------------------------------------------
# Simulate AI interactions for logging
# ---------------------------------------------------------------------------

def simulate_interaction(query: str) -> tuple[str, float]:
    """Simulate an AI interaction, return (response, confidence_score)."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": ("You are a healthcare AI assistant. Provide a brief, accurate "
                            "response. End with a confidence indicator: [CONFIDENCE: HIGH/MEDIUM/LOW]")
            },
            {"role": "user", "content": query}
        ],
        temperature=0.3,
        max_tokens=200
    )
    text = response.choices[0].message.content

    # Extract confidence from response
    if "[CONFIDENCE: HIGH]" in text.upper():
        conf = 0.95
    elif "[CONFIDENCE: MEDIUM]" in text.upper():
        conf = 0.75
    elif "[CONFIDENCE: LOW]" in text.upper():
        conf = 0.45
    else:
        conf = 0.70

    return text, conf


SIMULATED_INTERACTIONS = [
    {
        "user": "DR-CHEN-012",
        "query": "What is the recommended first-line treatment for community-acquired pneumonia?",
        "guardrails": {"input_safe": True, "output_safe": True, "faithfulness": 0.92, "all_passed": True},
    },
    {
        "user": "DR-CHEN-012",
        "query": "What is the maximum dose of acetaminophen for a patient with liver cirrhosis?",
        "guardrails": {"input_safe": True, "output_safe": True, "faithfulness": 0.88, "all_passed": True},
    },
    {
        "user": "NP-ADAMS-033",
        "query": "How do I interpret a positive ANA test?",
        "guardrails": {"input_safe": True, "output_safe": True, "faithfulness": 0.91, "all_passed": True},
    },
    {
        "user": "NP-ADAMS-033",
        "query": "Can you recommend the best malpractice insurance?",
        "guardrails": {"input_safe": False, "output_safe": True, "faithfulness": None,
                       "all_passed": False, "blocked_reason": "off-topic (legal/financial)"},
    },
    {
        "user": "DR-WILSON-005",
        "query": "Recommend anticoagulation strategy for a patient with HIT?",
        "guardrails": {"input_safe": True, "output_safe": True, "faithfulness": 0.65,
                       "all_passed": False, "blocked_reason": "low faithfulness"},
    },
    {
        "user": "DR-WILSON-005",
        "query": "What are the diagnostic criteria for sepsis?",
        "guardrails": {"input_safe": True, "output_safe": True, "faithfulness": 0.94, "all_passed": True},
    },
    {
        "user": "PA-MARTINEZ-019",
        "query": "How should I adjust vancomycin dosing for renal impairment?",
        "guardrails": {"input_safe": True, "output_safe": True, "faithfulness": 0.78, "all_passed": True},
    },
    {
        "user": "DR-CHEN-012",
        "query": "What's the differential diagnosis for acute pancreatitis?",
        "guardrails": {"input_safe": True, "output_safe": True, "faithfulness": 0.90, "all_passed": True},
    },
]


def main():
    """Build and demonstrate the audit logging system."""
    print("=" * 70)
    print("  Exercise 2: Audit Logger for AI-Assisted Clinical Decisions")
    print("  Logging, querying, and exporting AI interaction audit trails")
    print("=" * 70)

    logger = AuditLogger()

    # Step 1: Log simulated interactions
    print(f"\n--- Step 1: Logging {len(SIMULATED_INTERACTIONS)} AI interactions ---\n")

    for ix in SIMULATED_INTERACTIONS:
        response_text, confidence = simulate_interaction(ix["query"])
        entry = logger.log(
            user_id=ix["user"],
            query=ix["query"],
            response=response_text,
            model="gpt-4o-mini",
            guardrail_results=ix["guardrails"],
            confidence_score=confidence,
            metadata={"source": "exercise_demo"},
        )
        gr_status = "PASS" if ix["guardrails"]["all_passed"] else "FAIL"
        print(f"  Logged: {entry.log_id[:30]}... | {ix['user']} | "
              f"Guardrails: {gr_status} | Conf: {confidence:.2f}")

    # Step 2: Query audit trail
    print(f"\n--- Step 2: Querying Audit Trail ---\n")

    # Query by user
    user_entries = logger.query_by_user("DR-CHEN-012")
    print(f"  DR-CHEN-012 interactions: {len(user_entries)}")
    for e in user_entries:
        print(f"    - {e.query[:60]}...")

    # Query guardrail failures
    failures = logger.query_by_guardrail_status(passed=False)
    print(f"\n  Guardrail failures: {len(failures)}")
    for e in failures:
        reason = e.guardrail_results.get("blocked_reason", "unknown")
        print(f"    - [{e.user_id}] {e.query[:50]}... → {reason}")

    # Query low confidence
    low_conf = logger.query_low_confidence(threshold=0.7)
    print(f"\n  Low confidence (<0.7): {len(low_conf)}")
    for e in low_conf:
        print(f"    - [{e.user_id}] Conf={e.confidence_score:.2f} | {e.query[:50]}...")

    # Step 3: Audit summary
    print(f"\n--- Step 3: Audit Summary ---\n")
    summary = logger.get_summary()
    print(f"  Total interactions:     {summary['total_entries']}")
    print(f"  Unique users:           {summary['unique_users']}")
    print(f"  Users:                  {', '.join(summary['users'])}")
    print(f"  Models used:            {summary['models_used']}")
    print(f"  Guardrail failures:     {summary['guardrail_failures']} "
          f"({summary['guardrail_failure_rate']:.1%})")
    print(f"  Avg confidence:         {summary['avg_confidence']}")
    print(f"  Low confidence count:   {summary['low_confidence_count']}")

    # Step 4: Integrity verification
    print(f"\n--- Step 4: Integrity Verification ---\n")
    if logger.entries:
        test_entry = logger.entries[0]
        is_valid = logger.verify_integrity(
            test_entry,
            SIMULATED_INTERACTIONS[0]["query"],
            test_entry.response
        )
        print(f"  Entry {test_entry.log_id[:25]}...")
        print(f"  Input hash:  {test_entry.input_hash[:32]}...")
        print(f"  Output hash: {test_entry.output_hash[:32]}...")
        print(f"  Integrity:   {'✅ Valid' if is_valid else '❌ Tampered'}")

    # Step 5: Export
    print(f"\n--- Step 5: Export Audit Trail ---\n")
    json_export = logger.export_json()
    print(f"  Export size: {len(json_export)} characters")
    print(f"  Preview (first 500 chars):")
    for line in json_export[:500].split("\n"):
        print(f"    {line}")
    print(f"    ...")

    print(f"\n  In production, this would be written to:")
    print(f"    - HIPAA-compliant database")
    print(f"    - Immutable audit log store")
    print(f"    - SIEM system for security monitoring")


if __name__ == "__main__":
    main()
