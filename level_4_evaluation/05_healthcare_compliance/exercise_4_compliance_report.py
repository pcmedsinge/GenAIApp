"""
Exercise 4: Compliance Report Generator
=========================================

Skills practiced:
- Aggregating compliance metrics across AI interactions
- Computing PHI incident rates, hallucination rates, guardrail trigger rates
- Generating formatted compliance reports
- Assessing audit trail completeness
- Building compliance dashboards for oversight

Healthcare context:
Healthcare organizations deploying AI must regularly report on system safety and
compliance. A compliance report answers key questions for administrators:
- How many AI interactions occurred?
- Were any PHI incidents detected?
- How often did guardrails trigger?
- What was the hallucination rate?
- Is the audit trail complete?
- Are there any high-risk patterns?

This exercise builds a compliance report generator that takes simulated system
data and produces a comprehensive, formatted report.

Usage:
    python exercise_4_compliance_report.py
"""

import os
import json
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Simulated compliance data
# ---------------------------------------------------------------------------

def generate_simulated_data() -> dict:
    """
    Generate simulated compliance data representing a month of AI system usage.
    In production, this data would come from real audit logs, guardrail systems,
    and monitoring tools.
    """
    return {
        "report_period": {
            "start": "2026-01-01",
            "end": "2026-01-31",
            "organization": "Springfield Health System",
            "system_name": "ClinAssist AI v2.1",
        },
        "usage_metrics": {
            "total_queries": 12847,
            "unique_users": 89,
            "user_breakdown": {
                "physicians": 42,
                "nurse_practitioners": 23,
                "physician_assistants": 15,
                "registered_nurses": 9,
            },
            "queries_per_day_avg": 414.4,
            "peak_day": {"date": "2026-01-15", "count": 687},
            "models_used": {
                "gpt-4o-mini": 10534,
                "gpt-4o": 2313,
            },
        },
        "phi_incidents": {
            "total_detected": 23,
            "by_type": {
                "patient_name_in_query": 8,
                "mrn_in_query": 6,
                "dob_in_query": 4,
                "ssn_in_query": 1,
                "phi_in_response": 3,
                "address_in_query": 1,
            },
            "blocked": 20,
            "passed_through": 3,
            "false_positives": 2,
        },
        "guardrail_metrics": {
            "input_guardrail": {
                "total_checked": 12847,
                "blocked": 342,
                "block_reasons": {
                    "off_topic": 189,
                    "harmful_content": 12,
                    "prompt_injection": 8,
                    "inappropriate": 5,
                    "ambiguous_blocked": 128,
                },
            },
            "output_guardrail": {
                "total_checked": 12505,
                "flagged": 876,
                "flag_reasons": {
                    "missing_disclaimer": 534,
                    "definitive_diagnosis": 198,
                    "dosage_without_context": 89,
                    "dangerous_advice": 32,
                    "phi_leakage": 23,
                },
                "auto_remediated": 821,
                "blocked": 55,
            },
            "topic_restriction": {
                "total_checked": 12847,
                "allowed": 12316,
                "redirected": 189,
                "blocked": 342,
            },
        },
        "hallucination_metrics": {
            "total_checked": 8500,
            "avg_faithfulness": 0.89,
            "faithfulness_distribution": {
                "0.95-1.0 (high)": 5100,
                "0.85-0.94 (acceptable)": 2125,
                "0.70-0.84 (warning)": 935,
                "below 0.70 (fail)": 340,
            },
            "blocked_for_hallucination": 340,
            "false_alarm_rate": 0.03,
        },
        "audit_trail": {
            "total_logged": 12847,
            "with_complete_fields": 12801,
            "missing_fields": 46,
            "missing_field_types": {
                "guardrail_result": 23,
                "confidence_score": 18,
                "user_id": 5,
            },
            "integrity_checks_passed": 12839,
            "integrity_checks_failed": 8,
        },
        "clinical_validation": {
            "total_generated": 3200,
            "physician_reviewed": 2890,
            "approved_as_is": 1734,
            "approved_with_modifications": 987,
            "rejected": 169,
            "pending_review": 310,
            "avg_review_time_minutes": 4.2,
        },
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def compute_compliance_scores(data: dict) -> dict:
    """Compute compliance scores from the raw data."""
    usage = data["usage_metrics"]
    phi = data["phi_incidents"]
    guardrails = data["guardrail_metrics"]
    hallucination = data["hallucination_metrics"]
    audit = data["audit_trail"]
    validation = data["clinical_validation"]

    # PHI safety score
    phi_incident_rate = phi["total_detected"] / usage["total_queries"]
    phi_block_rate = phi["blocked"] / phi["total_detected"] if phi["total_detected"] > 0 else 1.0
    phi_score = max(0, 1.0 - (phi["passed_through"] / usage["total_queries"]) * 100)

    # Guardrail effectiveness
    output_flag_rate = guardrails["output_guardrail"]["flagged"] / guardrails["output_guardrail"]["total_checked"]
    guardrail_remediation_rate = (guardrails["output_guardrail"]["auto_remediated"] /
                                  guardrails["output_guardrail"]["flagged"]
                                  if guardrails["output_guardrail"]["flagged"] > 0 else 1.0)

    # Hallucination score
    hallucination_pass_rate = (
        (hallucination["faithfulness_distribution"].get("0.95-1.0 (high)", 0) +
         hallucination["faithfulness_distribution"].get("0.85-0.94 (acceptable)", 0)) /
        hallucination["total_checked"]
    )

    # Audit completeness
    audit_completeness = audit["with_complete_fields"] / audit["total_logged"]
    audit_integrity = audit["integrity_checks_passed"] / audit["total_logged"]

    # Clinical validation coverage
    review_rate = validation["physician_reviewed"] / validation["total_generated"]
    rejection_rate = validation["rejected"] / validation["physician_reviewed"] if validation["physician_reviewed"] > 0 else 0

    # Overall compliance score (weighted)
    overall = (
        phi_score * 0.25 +
        guardrail_remediation_rate * 0.20 +
        hallucination_pass_rate * 0.25 +
        audit_completeness * 0.15 +
        review_rate * 0.15
    )

    return {
        "phi_safety": {
            "score": round(phi_score, 3),
            "incident_rate": round(phi_incident_rate, 5),
            "block_rate": round(phi_block_rate, 3),
        },
        "guardrail_effectiveness": {
            "output_flag_rate": round(output_flag_rate, 3),
            "remediation_rate": round(guardrail_remediation_rate, 3),
        },
        "hallucination_safety": {
            "pass_rate": round(hallucination_pass_rate, 3),
            "avg_faithfulness": hallucination["avg_faithfulness"],
        },
        "audit_compliance": {
            "completeness": round(audit_completeness, 4),
            "integrity": round(audit_integrity, 4),
        },
        "clinical_validation": {
            "review_rate": round(review_rate, 3),
            "rejection_rate": round(rejection_rate, 3),
        },
        "overall_compliance_score": round(overall, 3),
    }


def generate_executive_summary(data: dict, scores: dict) -> str:
    """Use LLM to generate a plain-English executive summary."""
    context = json.dumps({"data": data["report_period"], "scores": scores}, indent=2)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a healthcare AI compliance officer.
Write a brief executive summary (3-4 paragraphs) for a compliance report.
Highlight: overall compliance posture, key risks, areas of strength,
and recommended actions. Be specific with numbers. Be professional."""
            },
            {
                "role": "user",
                "content": f"Generate executive summary for this compliance data:\n\n{context}"
            }
        ],
        temperature=0.3,
        max_tokens=400
    )
    return response.choices[0].message.content


def format_compliance_report(data: dict, scores: dict, executive_summary: str) -> str:
    """Format the complete compliance report."""
    period = data["report_period"]
    usage = data["usage_metrics"]
    phi = data["phi_incidents"]
    guardrails = data["guardrail_metrics"]
    hallucination = data["hallucination_metrics"]
    audit = data["audit_trail"]
    validation = data["clinical_validation"]

    def bar(score, width=20):
        filled = int(score * width)
        return "█" * filled + "░" * (width - filled)

    report = []
    report.append("=" * 70)
    report.append(f"  HEALTHCARE AI COMPLIANCE REPORT")
    report.append(f"  {period['organization']} — {period['system_name']}")
    report.append(f"  Period: {period['start']} to {period['end']}")
    report.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("=" * 70)

    # Overall score
    overall = scores["overall_compliance_score"]
    status = "COMPLIANT" if overall >= 0.85 else "NEEDS ATTENTION" if overall >= 0.70 else "NON-COMPLIANT"
    report.append(f"\n  OVERALL COMPLIANCE: [{bar(overall)}] {overall:.1%} — {status}")

    # Executive Summary
    report.append(f"\n{'─' * 70}")
    report.append("  EXECUTIVE SUMMARY")
    report.append(f"{'─' * 70}")
    for line in executive_summary.split("\n"):
        report.append(f"  {line}")

    # Section 1: Usage Overview
    report.append(f"\n{'─' * 70}")
    report.append("  1. USAGE OVERVIEW")
    report.append(f"{'─' * 70}")
    report.append(f"  Total queries:        {usage['total_queries']:,}")
    report.append(f"  Unique users:         {usage['unique_users']}")
    report.append(f"  Avg queries/day:      {usage['queries_per_day_avg']:.1f}")
    report.append(f"  Peak day:             {usage['peak_day']['date']} ({usage['peak_day']['count']} queries)")
    report.append(f"  User breakdown:")
    for role, count in usage["user_breakdown"].items():
        report.append(f"    {role.replace('_', ' ').title()}: {count}")

    # Section 2: PHI Safety
    report.append(f"\n{'─' * 70}")
    report.append("  2. PHI SAFETY")
    report.append(f"{'─' * 70}")
    phi_score = scores["phi_safety"]["score"]
    report.append(f"  Score: [{bar(phi_score)}] {phi_score:.1%}")
    report.append(f"  Total PHI incidents:  {phi['total_detected']}")
    report.append(f"  Blocked:              {phi['blocked']} ({scores['phi_safety']['block_rate']:.1%})")
    report.append(f"  Passed through:       {phi['passed_through']} ⚠️")
    report.append(f"  False positives:      {phi['false_positives']}")
    report.append(f"  By type:")
    for phi_type, count in phi["by_type"].items():
        report.append(f"    {phi_type.replace('_', ' ').title()}: {count}")

    # Section 3: Guardrail Effectiveness
    report.append(f"\n{'─' * 70}")
    report.append("  3. GUARDRAIL EFFECTIVENESS")
    report.append(f"{'─' * 70}")
    ig = guardrails["input_guardrail"]
    og = guardrails["output_guardrail"]
    report.append(f"  Input Guardrail:")
    report.append(f"    Checked: {ig['total_checked']:,} | Blocked: {ig['blocked']}")
    report.append(f"    Block reasons:")
    for reason, count in ig["block_reasons"].items():
        report.append(f"      {reason.replace('_', ' ').title()}: {count}")
    report.append(f"  Output Guardrail:")
    report.append(f"    Checked: {og['total_checked']:,} | Flagged: {og['flagged']}")
    report.append(f"    Auto-remediated: {og['auto_remediated']} | Blocked: {og['blocked']}")
    remediation_rate = scores["guardrail_effectiveness"]["remediation_rate"]
    report.append(f"    Remediation rate: [{bar(remediation_rate)}] {remediation_rate:.1%}")

    # Section 4: Hallucination Safety
    report.append(f"\n{'─' * 70}")
    report.append("  4. HALLUCINATION SAFETY")
    report.append(f"{'─' * 70}")
    hs = scores["hallucination_safety"]
    report.append(f"  Pass rate:           [{bar(hs['pass_rate'])}] {hs['pass_rate']:.1%}")
    report.append(f"  Avg faithfulness:    {hs['avg_faithfulness']:.2f}")
    report.append(f"  Distribution:")
    for bucket, count in hallucination["faithfulness_distribution"].items():
        pct = count / hallucination["total_checked"]
        report.append(f"    {bucket}: {count:,} ({pct:.1%})")
    report.append(f"  Blocked for hallucination: {hallucination['blocked_for_hallucination']}")

    # Section 5: Audit Compliance
    report.append(f"\n{'─' * 70}")
    report.append("  5. AUDIT TRAIL COMPLIANCE")
    report.append(f"{'─' * 70}")
    ac = scores["audit_compliance"]
    report.append(f"  Completeness:  [{bar(ac['completeness'])}] {ac['completeness']:.2%}")
    report.append(f"  Integrity:     [{bar(ac['integrity'])}] {ac['integrity']:.2%}")
    report.append(f"  Missing fields: {audit['missing_fields']}")
    for field, count in audit["missing_field_types"].items():
        report.append(f"    {field.replace('_', ' ').title()}: {count}")
    report.append(f"  Integrity failures: {audit['integrity_checks_failed']}")

    # Section 6: Clinical Validation
    report.append(f"\n{'─' * 70}")
    report.append("  6. CLINICAL VALIDATION")
    report.append(f"{'─' * 70}")
    cv = scores["clinical_validation"]
    report.append(f"  Review rate:    [{bar(cv['review_rate'])}] {cv['review_rate']:.1%}")
    report.append(f"  Total generated:          {validation['total_generated']:,}")
    report.append(f"  Physician reviewed:       {validation['physician_reviewed']:,}")
    report.append(f"  Approved as-is:           {validation['approved_as_is']:,}")
    report.append(f"  Approved w/ modifications: {validation['approved_with_modifications']:,}")
    report.append(f"  Rejected:                 {validation['rejected']} ({cv['rejection_rate']:.1%})")
    report.append(f"  Pending review:           {validation['pending_review']}")
    report.append(f"  Avg review time:          {validation['avg_review_time_minutes']} min")

    # Recommendations
    report.append(f"\n{'─' * 70}")
    report.append("  7. RECOMMENDATIONS")
    report.append(f"{'─' * 70}")

    if phi["passed_through"] > 0:
        report.append(f"  🔴 CRITICAL: {phi['passed_through']} PHI instances passed through guardrails.")
        report.append(f"     Action: Review and strengthen PHI detection rules.")

    if hallucination["blocked_for_hallucination"] > 300:
        report.append(f"  🟠 WARNING: {hallucination['blocked_for_hallucination']} responses blocked for hallucination.")
        report.append(f"     Action: Review source document quality and retrieval accuracy.")

    if validation["pending_review"] > 200:
        report.append(f"  🟡 NOTICE: {validation['pending_review']} AI-generated documents pending physician review.")
        report.append(f"     Action: Establish review SLA and escalation process.")

    if audit["integrity_checks_failed"] > 0:
        report.append(f"  🔴 CRITICAL: {audit['integrity_checks_failed']} audit log integrity failures detected.")
        report.append(f"     Action: Investigate potential log tampering or system errors.")

    if overall >= 0.85:
        report.append(f"\n  ✅ System is within compliance thresholds.")
    else:
        report.append(f"\n  ⚠️  System requires attention to meet compliance thresholds.")

    report.append(f"\n{'=' * 70}")
    report.append(f"  END OF COMPLIANCE REPORT")
    report.append(f"{'=' * 70}")

    return "\n".join(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Generate a comprehensive compliance report."""
    print("=" * 70)
    print("  Exercise 4: Compliance Report Generator")
    print("  Aggregating metrics and generating compliance reports")
    print("=" * 70)

    # Step 1: Gather data
    print("\n  Step 1: Gathering compliance data...")
    data = generate_simulated_data()
    print(f"  Loaded data for period: {data['report_period']['start']} to {data['report_period']['end']}")

    # Step 2: Compute scores
    print("  Step 2: Computing compliance scores...")
    scores = compute_compliance_scores(data)
    print(f"  Overall compliance score: {scores['overall_compliance_score']:.1%}")

    # Step 3: Generate executive summary
    print("  Step 3: Generating executive summary (AI-assisted)...")
    executive_summary = generate_executive_summary(data, scores)

    # Step 4: Format report
    print("  Step 4: Formatting report...")
    report = format_compliance_report(data, scores, executive_summary)

    # Display
    print(f"\n{report}")

    # Show raw scores
    print(f"\n\nDetailed Compliance Scores (JSON):")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
