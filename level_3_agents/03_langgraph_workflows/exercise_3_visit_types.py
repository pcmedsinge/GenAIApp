"""
Exercise 3: Dual-Path Graph — New Patient Intake AND Follow-Up Visits

Skills practiced:
- Building a graph with multiple entry pathways
- Conditional routing based on visit type
- Different node sequences for different scenarios
- Shared vs unique nodes across pathways

Key insight: Real clinical workflows handle different visit TYPES differently.
  A new patient needs full intake (history, allergies, medications).
  A follow-up needs to reference the PREVIOUS visit and check progress.
  This graph handles both through a single workflow with branching paths.
"""

import os
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# State for Dual-Path Workflow
# ============================================================

class VisitState(TypedDict):
    patient_info: str
    visit_type: str            # "new" or "followup"
    chief_complaint: str
    previous_visit: str        # For follow-ups: summary of last visit
    intake_data: str           # New patients: full intake
    progress_note: str         # Follow-ups: progress since last visit
    assessment: str
    plan: str
    visit_summary: str         # Final summary for the chart


# ============================================================
# Router Node
# ============================================================

def classify_visit(state: VisitState) -> dict:
    """Determine visit type if not already specified"""
    if state.get("visit_type") in ("new", "followup"):
        return {}  # Already classified

    response = llm.invoke(
        f"Is this a NEW patient visit or a FOLLOW-UP visit?\n"
        f"Patient: {state['patient_info']}\n"
        f"Previous visit info: {state.get('previous_visit', 'None')}\n\n"
        f"Respond ONLY: new OR followup"
    )
    vtype = "followup" if "followup" in response.content.lower() or "follow" in response.content.lower() else "new"
    return {"visit_type": vtype}


def route_visit_type(state: VisitState) -> str:
    return state["visit_type"]


# ============================================================
# New Patient Pathway
# ============================================================

def new_patient_intake(state: VisitState) -> dict:
    """Full intake for new patients: demographics, history, meds, allergies"""
    response = llm.invoke(
        f"You are taking a new patient intake. Based on the patient information, "
        f"create a structured intake note covering:\n"
        f"1. Demographics and chief complaint\n"
        f"2. Past medical history\n"
        f"3. Current medications\n"
        f"4. Allergies\n"
        f"5. Family history (infer if not provided)\n"
        f"6. Social history (infer if not provided)\n\n"
        f"Patient: {state['patient_info']}\n"
        f"Chief complaint: {state['chief_complaint']}\n\n"
        f"Create a structured intake note."
    )
    return {"intake_data": response.content}


def new_patient_assessment(state: VisitState) -> dict:
    """Assessment for new patient based on full intake"""
    response = llm.invoke(
        f"Generate a clinical assessment for this NEW patient:\n\n"
        f"Intake: {state['intake_data']}\n"
        f"Chief complaint: {state['chief_complaint']}\n\n"
        f"Include differential diagnosis (top 3) and initial workup. "
        f"Educational only."
    )
    return {"assessment": response.content}


def new_patient_plan(state: VisitState) -> dict:
    """Initial treatment plan for new patients"""
    response = llm.invoke(
        f"Create an initial management plan for this new patient:\n\n"
        f"Assessment: {state['assessment']}\n"
        f"Intake: {state['intake_data']}\n\n"
        f"Include: diagnostic tests to order, initial medications (if any), "
        f"lifestyle recommendations, and follow-up timeline. Educational only."
    )
    return {"plan": response.content}


# ============================================================
# Follow-Up Pathway
# ============================================================

def followup_progress_check(state: VisitState) -> dict:
    """Review progress since last visit"""
    response = llm.invoke(
        f"You are reviewing a follow-up patient. Compare current status to last visit.\n\n"
        f"Previous visit summary: {state.get('previous_visit', 'No previous data')}\n"
        f"Current patient info: {state['patient_info']}\n"
        f"Current complaint: {state['chief_complaint']}\n\n"
        f"Note: what has improved, what is the same, what has worsened."
    )
    return {"progress_note": response.content}


def followup_assessment(state: VisitState) -> dict:
    """Assessment focused on progress and changes"""
    response = llm.invoke(
        f"Generate a follow-up clinical assessment:\n\n"
        f"Progress note: {state['progress_note']}\n"
        f"Current complaint: {state['chief_complaint']}\n\n"
        f"Focus on: treatment response, any new issues, and whether current "
        f"management is adequate. Educational only."
    )
    return {"assessment": response.content}


def followup_plan(state: VisitState) -> dict:
    """Adjusted plan based on follow-up assessment"""
    response = llm.invoke(
        f"Adjust the management plan based on this follow-up:\n\n"
        f"Progress: {state['progress_note']}\n"
        f"Assessment: {state['assessment']}\n\n"
        f"Include: medication adjustments (increase/decrease/continue/stop), "
        f"new labs to order, and next follow-up interval. Educational only."
    )
    return {"plan": response.content}


# ============================================================
# Shared Node (both pathways converge here)
# ============================================================

def generate_visit_summary(state: VisitState) -> dict:
    """Generate a complete visit summary for the chart"""
    visit_label = "NEW PATIENT" if state["visit_type"] == "new" else "FOLLOW-UP"

    response = llm.invoke(
        f"Generate a concise {visit_label} visit summary for the medical chart.\n\n"
        f"Patient: {state['patient_info']}\n"
        f"Chief complaint: {state['chief_complaint']}\n"
        f"Assessment: {state['assessment']}\n"
        f"Plan: {state['plan']}\n"
        f"{'Intake: ' + state.get('intake_data', '')[:200] if state['visit_type'] == 'new' else 'Progress: ' + state.get('progress_note', '')[:200]}\n\n"
        f"Format as a structured clinical note."
    )
    return {"visit_summary": response.content}


# ============================================================
# Build the Graph
# ============================================================

def build_visit_workflow():
    graph = StateGraph(VisitState)

    # Shared router
    graph.add_node("classify", classify_visit)

    # New patient pathway
    graph.add_node("intake", new_patient_intake)
    graph.add_node("new_assess", new_patient_assessment)
    graph.add_node("new_plan", new_patient_plan)

    # Follow-up pathway
    graph.add_node("progress", followup_progress_check)
    graph.add_node("fu_assess", followup_assessment)
    graph.add_node("fu_plan", followup_plan)

    # Shared
    graph.add_node("summary", generate_visit_summary)

    # Entry
    graph.set_entry_point("classify")

    # Route by visit type
    graph.add_conditional_edges(
        "classify",
        route_visit_type,
        {"new": "intake", "followup": "progress"}
    )

    # New patient path: intake → assess → plan → summary
    graph.add_edge("intake", "new_assess")
    graph.add_edge("new_assess", "new_plan")
    graph.add_edge("new_plan", "summary")

    # Follow-up path: progress → assess → plan → summary
    graph.add_edge("progress", "fu_assess")
    graph.add_edge("fu_assess", "fu_plan")
    graph.add_edge("fu_plan", "summary")

    # Both converge at summary → END
    graph.add_edge("summary", END)

    return graph.compile()


# ============================================================
# DEMO 1: New Patient Visit
# ============================================================

def demo_new_patient():
    """Walk through a new patient encounter"""
    print("\n" + "=" * 70)
    print("DEMO 1: NEW PATIENT VISIT")
    print("=" * 70)
    print("""
    Path: classify → intake → new_assess → new_plan → summary
    Full intake with history, allergies, initial assessment.
    """)

    app = build_visit_workflow()

    result = app.invoke({
        "patient_info": "54-year-old male, referred by PCP. History of obesity (BMI 34), "
                        "family history of diabetes. Current meds: none. NKDA. "
                        "Non-smoker, social drinker.",
        "visit_type": "new",
        "chief_complaint": "Increased thirst, frequent urination, and fatigue for 6 weeks. "
                          "Lost 10 pounds without trying.",
        "previous_visit": "",
    })

    print(f"\n  Visit Type: {result.get('visit_type', '?').upper()}")
    print(f"\n  INTAKE:\n  {result.get('intake_data', 'N/A')[:500]}")
    print(f"\n  ASSESSMENT:\n  {result.get('assessment', 'N/A')[:500]}")
    print(f"\n  PLAN:\n  {result.get('plan', 'N/A')[:500]}")
    print(f"\n  VISIT SUMMARY:\n  {result.get('visit_summary', 'N/A')[:500]}")


# ============================================================
# DEMO 2: Follow-Up Visit
# ============================================================

def demo_followup():
    """Walk through a follow-up visit"""
    print("\n" + "=" * 70)
    print("DEMO 2: FOLLOW-UP VISIT")
    print("=" * 70)
    print("""
    Path: classify → progress → fu_assess → fu_plan → summary
    References previous visit, checks progress.
    """)

    app = build_visit_workflow()

    result = app.invoke({
        "patient_info": "54-year-old male, diagnosed Type 2 Diabetes 3 months ago. "
                        "Started metformin 500mg BID. Current weight: 198lbs (was 206). "
                        "Labs: HbA1c 7.2% (was 9.1 at diagnosis).",
        "visit_type": "followup",
        "chief_complaint": "3-month diabetes follow-up. Feeling better but occasional GI upset.",
        "previous_visit": "Dx: Type 2 Diabetes. HbA1c 9.1%. Started metformin 500mg BID. "
                          "Gave diet/exercise counseling. Weight 206lbs. Follow-up 3 months.",
    })

    print(f"\n  Visit Type: {result.get('visit_type', '?').upper()}")
    print(f"\n  PROGRESS NOTE:\n  {result.get('progress_note', 'N/A')[:500]}")
    print(f"\n  ASSESSMENT:\n  {result.get('assessment', 'N/A')[:500]}")
    print(f"\n  PLAN:\n  {result.get('plan', 'N/A')[:500]}")
    print(f"\n  VISIT SUMMARY:\n  {result.get('visit_summary', 'N/A')[:500]}")


# ============================================================
# DEMO 3: Side-by-Side — Same Patient, Both Visit Types
# ============================================================

def demo_side_by_side():
    """Show how the same patient gets different workflows"""
    print("\n" + "=" * 70)
    print("DEMO 3: SAME PATIENT — NEW vs FOLLOW-UP")
    print("=" * 70)

    app = build_visit_workflow()

    # New visit
    new_result = app.invoke({
        "patient_info": "65-year-old female, new to practice, history of HTN and CKD",
        "visit_type": "new",
        "chief_complaint": "Establishing care, managing hypertension and kidney disease",
        "previous_visit": "",
    })

    # Follow-up
    fu_result = app.invoke({
        "patient_info": "65-year-old female, HTN and CKD, on lisinopril 20mg. Labs: GFR 45, K+ 4.8, BP 142/88",
        "visit_type": "followup",
        "chief_complaint": "3-month follow-up for HTN/CKD. Blood pressure still elevated.",
        "previous_visit": "Started lisinopril 20mg for HTN. GFR 48. BP 150/92. Follow-up 3 months.",
    })

    print(f"\n  NEW PATIENT PATH:")
    print(f"  Intake: {new_result.get('intake_data', 'N/A')[:200]}...")
    print(f"  Plan: {new_result.get('plan', 'N/A')[:200]}...")

    print(f"\n  FOLLOW-UP PATH:")
    print(f"  Progress: {fu_result.get('progress_note', 'N/A')[:200]}...")
    print(f"  Plan: {fu_result.get('plan', 'N/A')[:200]}...")

    print("""
    KEY DIFFERENCE:
      • New visit: Full intake, initial workup, baseline plan
      • Follow-up: Progress comparison, treatment adjustment, trending
      • Same patient, different workflows — both converge at summary
    """)


# ============================================================
# DEMO 4: Interactive
# ============================================================

def demo_interactive():
    """Interactive visit workflow"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE VISIT WORKFLOW")
    print("=" * 70)
    print("  Type 'quit' to exit.\n")

    app = build_visit_workflow()

    while True:
        vtype = input("  Visit type (new/followup): ").strip().lower()
        if vtype in ['quit', 'exit', 'q']:
            break
        if vtype not in ('new', 'followup'):
            print("  Please enter 'new' or 'followup'")
            continue

        info = input("  Patient info: ").strip()
        complaint = input("  Chief complaint: ").strip()
        prev = ""
        if vtype == "followup":
            prev = input("  Previous visit summary: ").strip()

        if not info or not complaint:
            continue

        print("\n  Running workflow...\n")
        result = app.invoke({
            "patient_info": info,
            "visit_type": vtype,
            "chief_complaint": complaint,
            "previous_visit": prev,
        })

        print(f"  Visit Type: {result.get('visit_type', '?').upper()}")
        print(f"\n  Visit Summary:\n  {result.get('visit_summary', 'N/A')[:600]}\n")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 3: DUAL-PATH — NEW PATIENT + FOLLOW-UP VISITS")
    print("=" * 70)
    print("""
    One graph handles both new patient intake and follow-up visits
    with different node sequences converging at a shared summary.

    Choose a demo:
      1 → New patient visit
      2 → Follow-up visit
      3 → Side-by-side comparison
      4 → Interactive
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1": demo_new_patient()
    elif choice == "2": demo_followup()
    elif choice == "3": demo_side_by_side()
    elif choice == "4": demo_interactive()
    elif choice == "5":
        demo_new_patient()
        demo_followup()
        demo_side_by_side()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. DUAL-PATH GRAPH: One workflow handles multiple scenarios through
   conditional routing. The router (classify_visit) determines
   which path to take, and each path has its own node sequence.

2. SHARED NODES: The 'summary' node is shared — both paths converge.
   This avoids duplicating logic while allowing path-specific handling.

3. VISIT TYPE MATTERS: New patients need full intake; follow-ups need
   progress comparison. The STATE carries different data depending
   on which path was taken (intake_data vs progress_note).

4. CONTINUITY OF CARE: Follow-up visits reference previous_visit,
   showing how state can bridge across encounters. Production systems
   would load this from an EHR database.

5. EXTENSIBILITY: Want to add "urgent visit" or "telehealth"?
   Just add new nodes and another conditional edge. The graph
   pattern scales to many visit types without spaghetti code.
"""

if __name__ == "__main__":
    main()
