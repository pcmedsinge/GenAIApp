"""
Exercise 2: Human-in-the-Loop for High-Risk Recommendations

Skills practiced:
- Implementing human review gates in agent workflows
- Risk stratification to determine when human review is needed
- Building approval/rejection/modification workflows
- Understanding regulatory requirements for clinical AI

Key insight: Healthcare AI must NEVER make high-risk decisions
  autonomously. This exercise builds a workflow where low-risk
  recommendations are auto-approved, but high-risk ones PAUSE
  and require human clinician review before proceeding. This
  is a core FDA/regulatory pattern for clinical decision support.
"""

import os
import json
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# Risk Classification
# ============================================================

HIGH_RISK_INDICATORS = [
    "anticoagulation", "warfarin", "heparin",
    "insulin", "chemotherapy", "opioid",
    "intubation", "surgery", "transfusion",
    "pregnancy", "pediatric", "neonatal",
    "renal failure", "liver failure",
    "allergy", "anaphylaxis",
    "suicid", "overdose",
]

MODERATE_RISK_INDICATORS = [
    "new medication", "dose change", "antibiotic",
    "steroid", "benzodiazepine", "controlled substance",
    "elderly", "polypharmacy", "renal impairment",
]


def classify_risk(text: str) -> str:
    """Classify clinical risk based on keywords"""
    text_lower = text.lower()
    if any(kw in text_lower for kw in HIGH_RISK_INDICATORS):
        return "high"
    if any(kw in text_lower for kw in MODERATE_RISK_INDICATORS):
        return "moderate"
    return "low"


# ============================================================
# State
# ============================================================

class HILState(TypedDict):
    patient_case: str
    clinical_analysis: str
    recommendation: str
    risk_level: str               # low / moderate / high
    review_required: bool
    reviewer_decision: str        # approved / modified / rejected
    reviewer_notes: str
    final_recommendation: str
    audit_entry: str              # Who approved, when, why


# ============================================================
# Workflow Nodes
# ============================================================

def analyze_case(state: HILState) -> dict:
    """Step 1: Clinical analysis"""
    response = llm.invoke(
        f"Analyze this clinical case. Identify key problems, risk factors, "
        f"and clinical considerations:\n\n{state['patient_case']}\n\n"
        f"Educational purposes only."
    )
    return {"clinical_analysis": response.content}


def generate_recommendation(state: HILState) -> dict:
    """Step 2: Generate recommendation"""
    response = llm.invoke(
        f"Based on this analysis, provide specific clinical recommendations "
        f"including medications with doses and monitoring plan:\n\n"
        f"Case: {state['patient_case']}\n"
        f"Analysis: {state['clinical_analysis']}\n\n"
        f"Be specific. Educational purposes only."
    )
    return {"recommendation": response.content}


def risk_assessment(state: HILState) -> dict:
    """Step 3: Assess risk of the recommendation"""
    # Combine case + recommendation for risk classification
    combined = f"{state['patient_case']} {state['recommendation']}"
    risk = classify_risk(combined)

    # Also ask the LLM for a risk assessment
    response = llm.invoke(
        f"Rate the RISK LEVEL of this recommendation as LOW, MODERATE, or HIGH.\n\n"
        f"Patient: {state['patient_case'][:200]}\n"
        f"Recommendation: {state['recommendation'][:300]}\n\n"
        f"Consider: medication risk, patient vulnerability, potential for harm.\n"
        f"Reply with just the risk level and a one-line reason."
    )

    # Use keyword-based risk if LLM says high
    llm_risk = response.content.lower()
    if "high" in llm_risk:
        risk = "high"
    elif "moderate" in llm_risk and risk == "low":
        risk = "moderate"

    review_required = risk in ("high", "moderate")

    return {
        "risk_level": risk,
        "review_required": review_required,
    }


def human_review(state: HILState) -> dict:
    """Step 4: Human-in-the-loop review (simulated for demo)"""
    print(f"\n{'=' * 60}")
    print("  HUMAN REVIEW REQUIRED")
    print(f"{'=' * 60}")
    print(f"\n  Risk Level: {state['risk_level'].upper()}")
    print(f"\n  Patient: {state['patient_case'][:200]}")
    print(f"\n  AI Recommendation:")
    print(f"  {state['recommendation'][:400]}")
    print(f"\n  Options:")
    print(f"    1 → APPROVE (recommendation is safe)")
    print(f"    2 → MODIFY  (needs changes)")
    print(f"    3 → REJECT  (recommendation is unsafe)")

    choice = input("\n  Your decision (1-3): ").strip()

    if choice == "2":
        notes = input("  Modifications: ").strip()
        return {
            "reviewer_decision": "modified",
            "reviewer_notes": notes,
            "audit_entry": f"MODIFIED by clinician: {notes}",
        }
    elif choice == "3":
        notes = input("  Rejection reason: ").strip()
        return {
            "reviewer_decision": "rejected",
            "reviewer_notes": notes,
            "audit_entry": f"REJECTED by clinician: {notes}",
        }
    else:
        return {
            "reviewer_decision": "approved",
            "reviewer_notes": "Clinician review: approved as written",
            "audit_entry": "APPROVED by clinician review",
        }


def simulated_human_review(state: HILState) -> dict:
    """Simulated human review for automated demos"""
    # Simulate clinician behavior based on risk
    if state["risk_level"] == "high":
        if "pregnancy" in state["patient_case"].lower():
            return {
                "reviewer_decision": "modified",
                "reviewer_notes": "Remove ACEi — contraindicated in pregnancy. Switch to labetalol.",
                "audit_entry": "MODIFIED by Dr. Sim: Remove ACEi, switch to labetalol",
            }
        return {
            "reviewer_decision": "approved",
            "reviewer_notes": "High-risk recommendation reviewed and approved",
            "audit_entry": "APPROVED by Dr. Sim after high-risk review",
        }
    else:
        return {
            "reviewer_decision": "approved",
            "reviewer_notes": "Moderate-risk recommendation reviewed — acceptable",
            "audit_entry": "APPROVED by Dr. Sim (moderate-risk review)",
        }


def auto_approve(state: HILState) -> dict:
    """Step 4 alt: Auto-approve low-risk recommendations"""
    return {
        "reviewer_decision": "auto-approved",
        "reviewer_notes": "Low-risk: auto-approved by system",
        "audit_entry": "AUTO-APPROVED (low-risk, no human review needed)",
    }


def apply_modifications(state: HILState) -> dict:
    """Step 5: Apply reviewer modifications"""
    if state["reviewer_decision"] == "modified":
        response = llm.invoke(
            f"Revise this clinical recommendation based on clinician feedback:\n\n"
            f"Original: {state['recommendation']}\n"
            f"Clinician modification: {state['reviewer_notes']}\n\n"
            f"Produce the revised recommendation. Educational only."
        )
        return {"final_recommendation": response.content}
    elif state["reviewer_decision"] == "rejected":
        return {"final_recommendation": f"REJECTED — {state['reviewer_notes']}. No recommendation issued."}
    else:
        return {"final_recommendation": state["recommendation"]}


def format_output(state: HILState) -> dict:
    """Step 6: Format final output with audit trail"""
    audit = state.get("audit_entry", "No audit entry")
    return {
        "audit_entry": f"AUDIT: Risk={state['risk_level']}, Decision={state['reviewer_decision']}, {audit}"
    }


# ============================================================
# Route Functions
# ============================================================

def route_by_risk(state: HILState) -> str:
    return "needs_review" if state["review_required"] else "auto_approve"


# ============================================================
# Build Workflow
# ============================================================

def build_hil_workflow(interactive: bool = False):
    graph = StateGraph(HILState)

    graph.add_node("analyze", analyze_case)
    graph.add_node("recommend", generate_recommendation)
    graph.add_node("risk_assess", risk_assessment)
    graph.add_node("human_review", human_review if interactive else simulated_human_review)
    graph.add_node("auto_approve", auto_approve)
    graph.add_node("apply_mods", apply_modifications)
    graph.add_node("format", format_output)

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "recommend")
    graph.add_edge("recommend", "risk_assess")
    graph.add_conditional_edges(
        "risk_assess",
        route_by_risk,
        {"needs_review": "human_review", "auto_approve": "auto_approve"}
    )
    graph.add_edge("human_review", "apply_mods")
    graph.add_edge("auto_approve", "apply_mods")
    graph.add_edge("apply_mods", "format")
    graph.add_edge("format", END)

    return graph.compile()


# ============================================================
# DEMO 1: Risk-Based Routing
# ============================================================

def demo_risk_routing():
    """Show how different risk levels route differently"""
    print("\n" + "=" * 70)
    print("DEMO 1: RISK-BASED ROUTING")
    print("=" * 70)
    print("""
    LOW risk    → auto-approved (no human review)
    MODERATE    → human review required
    HIGH risk   → human review required + flagged
    """)

    app = build_hil_workflow(interactive=False)

    cases = [
        {"label": "LOW RISK — routine HTN follow-up",
         "patient_case": "55-year-old male, well-controlled HTN on amlodipine 5mg. BP 128/78. Routine follow-up. No complaints."},
        {"label": "MODERATE RISK — new medication in elderly",
         "patient_case": "78-year-old female with new-onset diabetes. HbA1c 7.2%. No other medications. Needs new medication recommendation."},
        {"label": "HIGH RISK — anticoagulation decision",
         "patient_case": "68-year-old male with new AFib, CKD Stage 3b (GFR 32), recent GI bleed 3 months ago. Needs anticoagulation decision."},
    ]

    for case in cases:
        print(f"\n{'─' * 60}")
        print(f"  CASE: {case['label']}")
        result = app.invoke({"patient_case": case["patient_case"]})
        print(f"  Risk Level:   {result.get('risk_level', '?').upper()}")
        print(f"  Review:       {result.get('reviewer_decision', '?')}")
        print(f"  Audit:        {result.get('audit_entry', 'N/A')}")
        print(f"  Final (excerpt): {result.get('final_recommendation', 'N/A')[:200]}...")


# ============================================================
# DEMO 2: Modification Workflow
# ============================================================

def demo_modification():
    """Show how human modifications change the recommendation"""
    print("\n" + "=" * 70)
    print("DEMO 2: HUMAN MODIFICATION WORKFLOW")
    print("=" * 70)
    print("""
    When a clinician MODIFIES a recommendation, the system:
      1. Takes the original AI recommendation
      2. Applies the clinician's corrections
      3. Produces a revised recommendation
      4. Logs everything for audit
    """)

    app = build_hil_workflow(interactive=False)

    # Pregnancy case — the simulated reviewer will catch the ACEi
    result = app.invoke({
        "patient_case": "32-year-old woman, 12 weeks pregnant, BP 150/95 on two readings. "
                        "History of chronic HTN, currently on lisinopril 10mg daily. "
                        "No proteinuria. Pregnancy confirmed."
    })

    print(f"\n  Risk Level: {result.get('risk_level', '?').upper()}")
    print(f"  Review Decision: {result.get('reviewer_decision', '?')}")
    print(f"  Reviewer Notes: {result.get('reviewer_notes', 'N/A')}")
    print(f"\n  FINAL RECOMMENDATION:")
    print(f"  {result.get('final_recommendation', 'N/A')[:400]}")
    print(f"\n  AUDIT TRAIL: {result.get('audit_entry', 'N/A')}")

    print("""
    THE SYSTEM CAUGHT: Lisinopril (ACEi) is contraindicated in pregnancy.
    The simulated reviewer modified the recommendation to use labetalol.
    The audit trail records who changed what and why.
    """)


# ============================================================
# DEMO 3: Audit Trail
# ============================================================

def demo_audit_trail():
    """Show the audit trail for multiple cases"""
    print("\n" + "=" * 70)
    print("DEMO 3: AUDIT TRAIL FOR COMPLIANCE")
    print("=" * 70)
    print("""
    Every recommendation has a complete audit trail:
      Who made it, who reviewed it, what was the decision.
    """)

    app = build_hil_workflow(interactive=False)

    cases = [
        "45-year-old with seasonal allergies needing antihistamine recommendation.",
        "70-year-old on warfarin (INR 3.8) with new knee pain needing pain management.",
        "28-year-old pregnant woman with depression, currently on sertraline 100mg.",
        "85-year-old with polypharmacy (8 medications) needing medication review.",
    ]

    audit_log = []
    for i, case in enumerate(cases, 1):
        result = app.invoke({"patient_case": case})
        entry = {
            "case_id": f"CASE-{i:03d}",
            "risk": result.get("risk_level", "?"),
            "review": result.get("reviewer_decision", "?"),
            "audit": result.get("audit_entry", "N/A"),
        }
        audit_log.append(entry)
        print(f"\n  {entry['case_id']}: Risk={entry['risk'].upper():10s} Decision={entry['review']:15s}")
        print(f"    {entry['audit']}")

    print(f"\n{'─' * 60}")
    print(f"  AUDIT SUMMARY:")
    print(f"    Total cases:    {len(audit_log)}")
    print(f"    Auto-approved:  {sum(1 for e in audit_log if e['review'] == 'auto-approved')}")
    print(f"    Human-reviewed: {sum(1 for e in audit_log if e['review'] in ('approved', 'modified', 'rejected'))}")
    print(f"    Modified:       {sum(1 for e in audit_log if e['review'] == 'modified')}")
    print(f"    Rejected:       {sum(1 for e in audit_log if e['review'] == 'rejected')}")


# ============================================================
# DEMO 4: Interactive Human-in-the-Loop
# ============================================================

def demo_interactive():
    """Interactive: YOU are the reviewing clinician"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE HUMAN-IN-THE-LOOP")
    print("=" * 70)
    print("  You will review AI recommendations as the clinician.")
    print("  Type 'quit' to exit.\n")

    app = build_hil_workflow(interactive=True)

    while True:
        case = input("  Patient case: ").strip()
        if case.lower() in ['quit', 'exit', 'q']:
            break
        if not case:
            continue

        print("\n  Processing...\n")
        result = app.invoke({"patient_case": case})

        print(f"\n  FINAL OUTCOME:")
        print(f"  Decision: {result.get('reviewer_decision', '?')}")
        print(f"  Recommendation: {result.get('final_recommendation', 'N/A')[:300]}")
        print(f"  Audit: {result.get('audit_entry', 'N/A')}\n")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 2: HUMAN-IN-THE-LOOP")
    print("=" * 70)
    print("""
    High-risk recommendations PAUSE for human clinician review.
    Low-risk recommendations are auto-approved.

    Workflow: Analyze → Recommend → Risk Assess → [Human Review] → Finalize

    Choose a demo:
      1 → Risk-based routing (low/moderate/high)
      2 → Modification workflow (clinician corrects AI)
      3 → Audit trail for compliance
      4 → Interactive (YOU review recommendations)
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1": demo_risk_routing()
    elif choice == "2": demo_modification()
    elif choice == "3": demo_audit_trail()
    elif choice == "4": demo_interactive()
    elif choice == "5":
        demo_risk_routing()
        demo_modification()
        demo_audit_trail()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. RISK-BASED ROUTING: Not every recommendation needs human review.
   Low-risk items auto-approve (efficiency). High-risk items pause
   for human review (safety). This balances speed and safety.

2. THREE REVIEW OUTCOMES: Approve (proceed), Modify (change then
   proceed), Reject (stop). The system handles all three and
   incorporates feedback automatically.

3. REGULATORY REQUIREMENT: FDA guidance for Clinical Decision
   Support software often requires human-in-the-loop for high-risk
   recommendations. This pattern satisfies that requirement.

4. AUDIT TRAIL: Every decision is logged — who made it, who
   reviewed it, what was changed. This is essential for:
   - Regulatory compliance (FDA, HIPAA)
   - Quality improvement
   - Liability protection
   - Trust building

5. PRODUCTION ENHANCEMENT: Real systems would:
   - Integrate with EHR for clinician identity (who approved)
   - Time-stamp all decisions
   - Escalate if no review within time limit
   - Track reviewer override patterns for system improvement
"""

if __name__ == "__main__":
    main()
