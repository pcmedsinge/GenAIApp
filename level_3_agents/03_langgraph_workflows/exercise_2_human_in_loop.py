"""
Exercise 2: Human-in-the-Loop Review Node

Skills practiced:
- Implementing a human review checkpoint in a LangGraph workflow
- Conditional routing based on risk level (auto-approve vs review)
- Pausing workflow for human input and resuming
- Understanding when human oversight is essential in healthcare AI

Key insight: AI should NOT make high-risk clinical decisions autonomously.
  This exercise adds a "review" node that pauses the workflow for human
  approval when the risk is high — a critical production pattern.
"""

import os
from typing import TypedDict, Literal
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# State with Human Review
# ============================================================

class ReviewState(TypedDict):
    patient_info: str
    symptoms: str
    extracted_data: str
    risk_level: str            # low, medium, high, critical
    assessment: str
    proposed_plan: str
    review_status: str         # pending, approved, modified, rejected
    reviewer_notes: str
    final_plan: str


# ============================================================
# Workflow Nodes
# ============================================================

def extract_data(state: ReviewState) -> dict:
    response = llm.invoke(
        f"Extract key clinical data. Include age, conditions, meds, labs.\n\n"
        f"Patient: {state['patient_info']}\nSymptoms: {state['symptoms']}"
    )
    return {"extracted_data": response.content}


def assess_risk(state: ReviewState) -> dict:
    """Classify risk as low/medium/high/critical"""
    response = llm.invoke(
        f"""Classify the clinical risk level for this patient.
Consider: acute symptoms, vital sign abnormalities, high-risk medications,
organ failure risk, and potential for rapid deterioration.

Patient data: {state['extracted_data']}
Symptoms: {state['symptoms']}

Respond with ONLY one word: low, medium, high, or critical.
""")
    risk = response.content.strip().lower()
    if "critical" in risk:
        risk = "critical"
    elif "high" in risk:
        risk = "high"
    elif "medium" in risk or "moderate" in risk:
        risk = "medium"
    else:
        risk = "low"
    return {"risk_level": risk}


def generate_assessment(state: ReviewState) -> dict:
    response = llm.invoke(
        f"Clinical assessment with top 3 differential diagnoses:\n"
        f"Data: {state['extracted_data']}\nRisk: {state['risk_level']}\n"
        f"Educational only."
    )
    return {"assessment": response.content}


def propose_plan(state: ReviewState) -> dict:
    response = llm.invoke(
        f"Propose a management plan with specific actions and medications:\n"
        f"Assessment: {state['assessment']}\nRisk: {state['risk_level']}\n"
        f"Educational only."
    )
    return {"proposed_plan": response.content}


def human_review(state: ReviewState) -> dict:
    """
    HUMAN-IN-THE-LOOP: Pauses the workflow for human review.
    In production, this would integrate with a review queue/UI.
    For this exercise, we use console input.
    """
    print(f"\n{'═' * 60}")
    print(f"  HUMAN REVIEW REQUIRED")
    print(f"  Risk Level: {state['risk_level'].upper()}")
    print(f"{'═' * 60}")
    print(f"\n  Patient Data:\n  {state['extracted_data'][:300]}...")
    print(f"\n  Assessment:\n  {state['assessment'][:300]}...")
    print(f"\n  Proposed Plan:\n  {state['proposed_plan'][:400]}...")

    print(f"\n  Options:")
    print(f"    [A] APPROVE — plan is acceptable")
    print(f"    [M] MODIFY — approve with changes (add notes)")
    print(f"    [R] REJECT — plan needs rework")

    choice = input(f"\n  Decision (A/M/R): ").strip().upper()

    if choice == "A":
        return {"review_status": "approved", "reviewer_notes": "Approved without changes."}
    elif choice == "M":
        notes = input("  Modifications: ").strip()
        return {"review_status": "modified", "reviewer_notes": notes or "Minor adjustments requested."}
    elif choice == "R":
        notes = input("  Rejection reason: ").strip()
        return {"review_status": "rejected", "reviewer_notes": notes or "Plan rejected — needs rework."}
    else:
        return {"review_status": "approved", "reviewer_notes": "Auto-approved (invalid input)."}


def auto_approve(state: ReviewState) -> dict:
    """Low-risk cases are auto-approved"""
    return {
        "review_status": "auto_approved",
        "reviewer_notes": f"Auto-approved: {state['risk_level']} risk — no human review required."
    }


def finalize_plan(state: ReviewState) -> dict:
    """Finalize the plan incorporating review feedback"""
    if state["review_status"] == "rejected":
        return {"final_plan": f"PLAN REJECTED. Reason: {state['reviewer_notes']}. Needs physician rework."}

    context = ""
    if state["review_status"] == "modified":
        context = f"Reviewer modifications: {state['reviewer_notes']}\n"

    response = llm.invoke(
        f"Finalize this plan incorporating review feedback.\n\n"
        f"Proposed plan: {state['proposed_plan']}\n"
        f"Review status: {state['review_status']}\n"
        f"{context}"
        f"Educational only."
    )
    return {"final_plan": response.content}


# ============================================================
# Routing Logic
# ============================================================

def route_by_risk(state: ReviewState) -> str:
    """High/critical risk → human review. Low/medium → auto-approve."""
    if state["risk_level"] in ("high", "critical"):
        return "needs_review"
    return "auto_approve"


def build_review_workflow():
    """Build the graph with human-in-the-loop"""
    graph = StateGraph(ReviewState)

    graph.add_node("extract", extract_data)
    graph.add_node("risk", assess_risk)
    graph.add_node("assess", generate_assessment)
    graph.add_node("propose", propose_plan)
    graph.add_node("human_review", human_review)
    graph.add_node("auto_approve", auto_approve)
    graph.add_node("finalize", finalize_plan)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "risk")
    graph.add_edge("risk", "assess")
    graph.add_edge("assess", "propose")

    # Conditional: high risk → human review, low risk → auto
    graph.add_conditional_edges(
        "propose",
        route_by_risk,
        {"needs_review": "human_review", "auto_approve": "auto_approve"}
    )

    graph.add_edge("human_review", "finalize")
    graph.add_edge("auto_approve", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# ============================================================
# DEMO 1: High-Risk Case (requires review)
# ============================================================

def demo_high_risk():
    """High-risk case that triggers human review"""
    print("\n" + "=" * 70)
    print("DEMO 1: HIGH-RISK CASE — HUMAN REVIEW REQUIRED")
    print("=" * 70)
    print("""
    Workflow: Extract → Risk → Assess → Propose → [HUMAN REVIEW] → Finalize
    High-risk cases PAUSE for human approval.
    """)

    app = build_review_workflow()

    result = app.invoke({
        "patient_info": "78-year-old male on warfarin and aspirin, history of GI bleed 3 months ago",
        "symptoms": "Black tarry stools for 2 days, dizziness, heart rate 110, BP 90/60"
    })

    print(f"\n  Risk Level: {result.get('risk_level', '?').upper()}")
    print(f"  Review: {result.get('review_status', '?')}")
    print(f"  Reviewer Notes: {result.get('reviewer_notes', 'N/A')[:200]}")
    print(f"\n  Final Plan:\n  {result.get('final_plan', 'N/A')[:500]}")


# ============================================================
# DEMO 2: Low-Risk Case (auto-approved)
# ============================================================

def demo_low_risk():
    """Low-risk case that is auto-approved"""
    print("\n" + "=" * 70)
    print("DEMO 2: LOW-RISK CASE — AUTO-APPROVED")
    print("=" * 70)
    print("""
    Workflow: Extract → Risk → Assess → Propose → [AUTO-APPROVE] → Finalize
    Low-risk cases skip human review.
    """)

    app = build_review_workflow()

    result = app.invoke({
        "patient_info": "32-year-old healthy female, no medications, no allergies",
        "symptoms": "Mild sore throat for 2 days, no fever, no difficulty swallowing"
    })

    print(f"\n  Risk Level: {result.get('risk_level', '?').upper()}")
    print(f"  Review: {result.get('review_status', '?')}")
    print(f"  Notes: {result.get('reviewer_notes', 'N/A')[:200]}")
    print(f"\n  Final Plan:\n  {result.get('final_plan', 'N/A')[:400]}")


# ============================================================
# DEMO 3: Simulated Review Queue
# ============================================================

def demo_review_queue():
    """Multiple cases — show which need review and which don't"""
    print("\n" + "=" * 70)
    print("DEMO 3: SIMULATED REVIEW QUEUE")
    print("=" * 70)
    print("""
    Run multiple cases and track which require human review.
    In production, these would go to a review queue/dashboard.
    """)

    # Build a version that auto-approves everything (no interactive input)
    graph = StateGraph(ReviewState)
    graph.add_node("extract", extract_data)
    graph.add_node("risk", assess_risk)
    graph.add_node("assess", generate_assessment)
    graph.add_node("propose", propose_plan)
    graph.add_node("auto_approve", auto_approve)
    graph.add_node("finalize", finalize_plan)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "risk")
    graph.add_edge("risk", "assess")
    graph.add_edge("assess", "propose")
    graph.add_edge("propose", "auto_approve")  # Skip human review for demo
    graph.add_edge("auto_approve", "finalize")
    graph.add_edge("finalize", END)
    auto_app = graph.compile()

    cases = [
        {"patient_info": "25-year-old healthy, no meds", "symptoms": "Runny nose, mild cough, 2 days"},
        {"patient_info": "70-year-old, CHF, on 5 medications", "symptoms": "Worsening shortness of breath, 10lb weight gain in a week"},
        {"patient_info": "40-year-old, well-controlled HTN on lisinopril", "symptoms": "Annual checkup, feeling fine"},
        {"patient_info": "85-year-old, CKD stage 4, diabetes, on insulin and lisinopril", "symptoms": "Confusion, lethargy, decreased urine output"},
        {"patient_info": "35-year-old, mild anxiety on sertraline", "symptoms": "Wants to discuss medication adjustment, doing well"},
    ]

    print(f"\n  {'#':<3} {'Patient':<45} {'Risk':<10} {'Review Needed':<15}")
    print(f"  {'─'*3} {'─'*45} {'─'*10} {'─'*15}")

    for i, case in enumerate(cases, 1):
        result = auto_app.invoke(case)
        risk = result.get("risk_level", "?")
        needs_review = "YES" if risk in ("high", "critical") else "No"
        marker = " <<<" if needs_review == "YES" else ""
        print(f"  {i:<3} {case['patient_info'][:45]:<45} {risk:<10} {needs_review:<15}{marker}")

    print("""
    PRODUCTION PATTERN:
      • Low/Medium risk: Auto-approved → immediate action
      • High/Critical risk: Queued for physician review
      • Review queue shows patient, risk level, proposed plan
      • Reviewer can approve, modify, or reject
      • Rejected plans return to the agent for rework
    """)


# ============================================================
# DEMO 4: Interactive
# ============================================================

def demo_interactive():
    """Interactive with human review"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE WITH HUMAN REVIEW")
    print("=" * 70)
    print("  High-risk cases will pause for your review.")
    print("  Type 'quit' to exit.\n")

    app = build_review_workflow()

    while True:
        info = input("  Patient info: ").strip()
        if info.lower() in ['quit', 'exit', 'q']:
            break
        if not info:
            continue
        symptoms = input("  Symptoms: ").strip()
        if not symptoms:
            continue

        print("\n  Running workflow...\n")
        result = app.invoke({"patient_info": info, "symptoms": symptoms})

        print(f"\n  Risk: {result.get('risk_level', '?').upper()}")
        print(f"  Review: {result.get('review_status', '?')}")
        print(f"  Final Plan: {result.get('final_plan', 'N/A')[:500]}\n")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 2: HUMAN-IN-THE-LOOP REVIEW NODE")
    print("=" * 70)
    print("""
    Adds human review for high-risk recommendations.
    Low-risk cases are auto-approved; high-risk cases pause.

    Choose a demo:
      1 → High-risk case (requires review — interactive)
      2 → Low-risk case (auto-approved)
      3 → Review queue simulation (multiple cases)
      4 → Interactive
      5 → Run demos 2-3 (non-interactive demos)
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1": demo_high_risk()
    elif choice == "2": demo_low_risk()
    elif choice == "3": demo_review_queue()
    elif choice == "4": demo_interactive()
    elif choice == "5":
        demo_low_risk()
        demo_review_queue()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. CONDITIONAL REVIEW: route_by_risk() determines whether a case
   needs human review. High/critical → human, Low/medium → auto.
   This is the core pattern for human-in-the-loop AI systems.

2. REVIEW OPTIONS: Approve / Modify / Reject gives the reviewer
   three clear actions. Modified plans get adjusted; rejected plans
   are flagged for rework.

3. PRODUCTION PATTERN: In real systems, the "pause" would be a
   database write (not console input). A dashboard shows pending
   reviews. The workflow resumes when the human responds.

4. REGULATORY REQUIREMENT: Many healthcare AI regulations require
   human oversight for clinical decisions. This pattern satisfies
   that requirement while still automating the low-risk workflow.

5. RISK CALIBRATION: The threshold for human review is configurable.
   Start conservative (review everything), then gradually auto-approve
   lower-risk categories as confidence in the system grows.
"""

if __name__ == "__main__":
    main()
