"""
Exercise 1: Pharmacy Check Node — Medication Dosing Validation

Skills practiced:
- Adding a new node to an existing LangGraph workflow
- Inserting a node BETWEEN existing nodes (not just at the end)
- Building a medication safety validation step
- Understanding how state accumulates across nodes

Key insight: In healthcare, a "pharmacy check" happens BEFORE the final
  recommendation reaches the patient. This exercise adds that safety
  step — a node that validates medications, checks dosing, and flags
  issues before the plan is finalized.
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
# Medication Database for Pharmacy Check
# ============================================================

MEDICATION_RULES = {
    "metformin": {
        "max_dose": 2000,  # mg/day
        "renal_cutoff_gfr": 30,
        "dose_reduce_gfr": 45,
        "contraindications": ["severe renal impairment", "metabolic acidosis"],
        "monitoring": ["renal function", "B12 levels", "HbA1c"],
    },
    "lisinopril": {
        "max_dose": 40,
        "renal_cutoff_gfr": None,
        "contraindications": ["pregnancy", "bilateral renal artery stenosis", "angioedema history"],
        "monitoring": ["potassium", "creatinine", "blood pressure"],
        "potassium_max": 5.5,
    },
    "apixaban": {
        "standard_dose": 5,  # mg BID
        "reduced_dose": 2.5,  # mg BID — if age≥80 OR weight≤60kg OR Cr≥1.5
        "contraindications": ["active bleeding", "mechanical heart valve"],
        "monitoring": ["signs of bleeding", "renal function"],
    },
    "amlodipine": {
        "max_dose": 10,
        "contraindications": ["severe aortic stenosis"],
        "monitoring": ["blood pressure", "heart rate", "edema"],
    },
    "sertraline": {
        "max_dose": 200,
        "contraindications": ["MAOi within 14 days"],
        "interactions_avoid": ["tramadol", "MAOi"],
        "monitoring": ["mood", "suicidal ideation if age < 25"],
    },
}


# ============================================================
# Clinical Workflow State (extended with pharmacy check)
# ============================================================

class PharmacyState(TypedDict):
    patient_info: str
    symptoms: str
    extracted_data: str
    risk_factors: str
    urgency: str
    clinical_assessment: str
    proposed_plan: str       # Plan BEFORE pharmacy check
    pharmacy_review: str     # Pharmacy validation results
    final_plan: str          # Plan AFTER pharmacy check


# ============================================================
# Workflow Nodes
# ============================================================

def extract_data(state: PharmacyState) -> dict:
    response = llm.invoke(
        f"Extract key clinical data from this patient. Include age, sex, "
        f"conditions, current medications with doses, and relevant labs.\n\n"
        f"Patient: {state['patient_info']}\nSymptoms: {state['symptoms']}"
    )
    return {"extracted_data": response.content}


def assess_risk(state: PharmacyState) -> dict:
    response = llm.invoke(
        f"Identify risk factors and red flags:\n{state['extracted_data']}"
    )
    return {"risk_factors": response.content}


def classify_urgency(state: PharmacyState) -> dict:
    response = llm.invoke(
        f"Classify urgency (emergency/urgent/routine):\n"
        f"Data: {state['extracted_data']}\nRisk: {state['risk_factors']}\n"
        f"Respond with ONLY: emergency, urgent, or routine"
    )
    urgency = response.content.strip().lower()
    if "emergency" in urgency:
        return {"urgency": "emergency"}
    elif "urgent" in urgency:
        return {"urgency": "urgent"}
    return {"urgency": "routine"}


def generate_assessment(state: PharmacyState) -> dict:
    response = llm.invoke(
        f"Clinical assessment with differential diagnosis:\n"
        f"Data: {state['extracted_data']}\nRisk: {state['risk_factors']}\n"
        f"Urgency: {state['urgency']}\nEducational only."
    )
    return {"clinical_assessment": response.content}


def propose_plan(state: PharmacyState) -> dict:
    """Generate a proposed plan (BEFORE pharmacy check)"""
    response = llm.invoke(
        f"Propose a management plan including specific medications with doses:\n"
        f"Assessment: {state['clinical_assessment']}\n"
        f"Current patient data: {state['extracted_data']}\n"
        f"Include specific medication names and doses. Educational only."
    )
    return {"proposed_plan": response.content}


def pharmacy_check(state: PharmacyState) -> dict:
    """
    NEW NODE: Pharmacy validation.
    Checks proposed medications against the medication database
    for dosing, contraindications, and monitoring requirements.
    """
    # Ask LLM to extract medication names from the proposed plan
    extract_response = llm.invoke(
        f"Extract ONLY the medication names from this plan. "
        f"Return as a comma-separated list, lowercase:\n\n{state['proposed_plan']}"
    )

    med_names = [m.strip().lower() for m in extract_response.content.split(",")]

    review_items = []
    for med_name in med_names:
        rules = MEDICATION_RULES.get(med_name)
        if rules:
            review_items.append(f"\n  {med_name.upper()}:")
            review_items.append(f"    Max dose: {rules.get('max_dose', 'N/A')}mg/day")
            review_items.append(f"    Contraindications: {', '.join(rules.get('contraindications', []))}")
            review_items.append(f"    Required monitoring: {', '.join(rules.get('monitoring', []))}")
            if rules.get('renal_cutoff_gfr'):
                review_items.append(f"    RENAL WARNING: Contraindicated if GFR < {rules['renal_cutoff_gfr']}")
            if rules.get('dose_reduce_gfr'):
                review_items.append(f"    Dose reduction needed if GFR < {rules['dose_reduce_gfr']}")
        else:
            review_items.append(f"\n  {med_name.upper()}: Not in pharmacy database — manual review needed")

    # LLM-based cross-check
    cross_check = llm.invoke(
        f"You are a clinical pharmacist. Review this proposed plan against patient data.\n\n"
        f"PROPOSED PLAN:\n{state['proposed_plan']}\n\n"
        f"PATIENT DATA:\n{state['extracted_data']}\n\n"
        f"MEDICATION RULES:\n{''.join(review_items)}\n\n"
        f"Check for:\n"
        f"1. Dose appropriateness (within max limits)\n"
        f"2. Contraindications based on patient conditions/labs\n"
        f"3. Drug interactions with current medications\n"
        f"4. Required monitoring that should be ordered\n\n"
        f"Flag any issues. If all clear, state 'PHARMACY CHECK: APPROVED'."
    )

    return {"pharmacy_review": cross_check.content}


def finalize_plan(state: PharmacyState) -> dict:
    """Finalize plan incorporating pharmacy review"""
    response = llm.invoke(
        f"Finalize this management plan incorporating the pharmacy review.\n\n"
        f"ORIGINAL PLAN:\n{state['proposed_plan']}\n\n"
        f"PHARMACY REVIEW:\n{state['pharmacy_review']}\n\n"
        f"If the pharmacy review flagged issues, adjust the plan accordingly. "
        f"If approved, confirm the plan. Include monitoring schedule. Educational only."
    )
    return {"final_plan": response.content}


# ============================================================
# Build the Graph
# ============================================================

def build_pharmacy_workflow():
    graph = StateGraph(PharmacyState)

    # Add nodes
    graph.add_node("extract", extract_data)
    graph.add_node("risk", assess_risk)
    graph.add_node("classify", classify_urgency)
    graph.add_node("assess", generate_assessment)
    graph.add_node("propose", propose_plan)
    graph.add_node("pharmacy", pharmacy_check)   # NEW NODE
    graph.add_node("finalize", finalize_plan)

    # Edges: extract → risk → classify → assess → propose → PHARMACY → finalize
    graph.set_entry_point("extract")
    graph.add_edge("extract", "risk")
    graph.add_edge("risk", "classify")
    graph.add_edge("classify", "assess")
    graph.add_edge("assess", "propose")
    graph.add_edge("propose", "pharmacy")    # Plan goes through pharmacy
    graph.add_edge("pharmacy", "finalize")   # Pharmacy review feeds into final
    graph.add_edge("finalize", END)

    return graph.compile()


# ============================================================
# DEMO 1: Pharmacy Check in Action
# ============================================================

def demo_pharmacy_check():
    """Show the pharmacy node catching an issue"""
    print("\n" + "=" * 70)
    print("DEMO 1: PHARMACY CHECK IN ACTION")
    print("=" * 70)
    print("""
    Workflow: Extract → Risk → Classify → Assess → Propose → PHARMACY → Finalize

    The pharmacy check node validates the proposed plan BEFORE it's finalized.
    Like a pharmacist reviewing a prescription before dispensing.
    """)

    app = build_pharmacy_workflow()

    # Case that should trigger pharmacy concerns
    result = app.invoke({
        "patient_info": "72-year-old male, CKD Stage 3b (GFR 38), Type 2 Diabetes, HTN. "
                        "Current meds: metformin 2000mg daily, lisinopril 20mg daily. "
                        "Labs: HbA1c 8.1%, K+ 5.3, Cr 1.8",
        "symptoms": "Uncontrolled blood sugar despite max dose metformin. Feeling fatigued."
    })

    print(f"\n  URGENCY: {result.get('urgency', 'N/A').upper()}")
    print(f"\n  PROPOSED PLAN (before pharmacy):")
    print(f"  {result.get('proposed_plan', 'N/A')[:500]}")
    print(f"\n  PHARMACY REVIEW:")
    print(f"  {result.get('pharmacy_review', 'N/A')[:500]}")
    print(f"\n  FINAL PLAN (after pharmacy):")
    print(f"  {result.get('final_plan', 'N/A')[:500]}")


# ============================================================
# DEMO 2: Comparing With vs Without Pharmacy Check
# ============================================================

def demo_comparison():
    """Compare workflow with and without pharmacy check"""
    print("\n" + "=" * 70)
    print("DEMO 2: WITH vs WITHOUT PHARMACY CHECK")
    print("=" * 70)

    # Build workflow WITHOUT pharmacy check
    graph_no_pharm = StateGraph(PharmacyState)
    graph_no_pharm.add_node("extract", extract_data)
    graph_no_pharm.add_node("risk", assess_risk)
    graph_no_pharm.add_node("classify", classify_urgency)
    graph_no_pharm.add_node("assess", generate_assessment)
    graph_no_pharm.add_node("propose", propose_plan)
    graph_no_pharm.set_entry_point("extract")
    graph_no_pharm.add_edge("extract", "risk")
    graph_no_pharm.add_edge("risk", "classify")
    graph_no_pharm.add_edge("classify", "assess")
    graph_no_pharm.add_edge("assess", "propose")
    graph_no_pharm.add_edge("propose", END)
    no_pharm = graph_no_pharm.compile()

    # Build WITH pharmacy
    with_pharm = build_pharmacy_workflow()

    case = {
        "patient_info": "80-year-old female, 55kg, Cr 1.6, AFib, hypertension. "
                        "Current: apixaban 5mg BID, amlodipine 10mg daily.",
        "symptoms": "Bruising easily, gum bleeding, fatigued."
    }

    print(f"\n  Patient: {case['patient_info'][:80]}...")
    print(f"  Symptoms: {case['symptoms']}")

    print(f"\n{'─' * 60}")
    print("  WITHOUT pharmacy check:")
    result_no = no_pharm.invoke(case)
    print(f"  Plan: {result_no.get('proposed_plan', 'N/A')[:400]}")

    print(f"\n{'─' * 60}")
    print("  WITH pharmacy check:")
    result_with = with_pharm.invoke(case)
    print(f"  Pharmacy: {result_with.get('pharmacy_review', 'N/A')[:300]}")
    print(f"  Final: {result_with.get('final_plan', 'N/A')[:400]}")

    print("""
    KEY DIFFERENCE: The pharmacy check may catch that the apixaban dose
    should be REDUCED (2.5mg BID) because the patient meets criteria:
    age ≥80, weight ≤60kg, Cr ≥1.5. Without the check, this could be missed.
    """)


# ============================================================
# DEMO 3: Multiple Cases Through Pharmacy
# ============================================================

def demo_multiple_cases():
    """Run several cases to see pharmacy check patterns"""
    print("\n" + "=" * 70)
    print("DEMO 3: MULTIPLE CASES THROUGH PHARMACY CHECK")
    print("=" * 70)

    app = build_pharmacy_workflow()

    cases = [
        {
            "patient_info": "45-year-old healthy female, no medications, normal labs",
            "symptoms": "Mild tension headache for 2 days, manageable with rest"
        },
        {
            "patient_info": "65-year-old male, GFR 28, diabetes, on metformin 2000mg, K+ 5.6, on lisinopril 40mg",
            "symptoms": "Worsening fatigue, nausea, decreased appetite over 1 week"
        },
        {
            "patient_info": "55-year-old female, depression well controlled on sertraline 150mg, chronic back pain",
            "symptoms": "Severe back pain, asking about adding tramadol for pain management"
        },
    ]

    for i, case in enumerate(cases, 1):
        print(f"\n{'═' * 60}")
        print(f"  CASE {i}")
        print(f"  Patient: {case['patient_info'][:70]}...")
        print(f"  Symptoms: {case['symptoms'][:70]}...")

        result = app.invoke(case)

        print(f"  Urgency: {result.get('urgency', '?').upper()}")
        print(f"  Pharmacy: {result.get('pharmacy_review', 'N/A')[:250]}...")
        print(f"  Final Plan: {result.get('final_plan', 'N/A')[:250]}...")


# ============================================================
# DEMO 4: Interactive
# ============================================================

def demo_interactive():
    """Interactive pharmacy-checked triage"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE WITH PHARMACY CHECK")
    print("=" * 70)
    print("  Enter patient cases. Type 'quit' to exit.\n")

    app = build_pharmacy_workflow()

    while True:
        info = input("  Patient info: ").strip()
        if info.lower() in ['quit', 'exit', 'q']:
            break
        if not info:
            continue
        symptoms = input("  Symptoms: ").strip()
        if not symptoms:
            continue

        print("\n  Running workflow with pharmacy check...\n")
        result = app.invoke({"patient_info": info, "symptoms": symptoms})

        print(f"  Urgency: {result.get('urgency', '?').upper()}")
        print(f"\n  Pharmacy Review:\n  {result.get('pharmacy_review', 'N/A')[:500]}")
        print(f"\n  Final Plan:\n  {result.get('final_plan', 'N/A')[:500]}\n")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 1: PHARMACY CHECK NODE")
    print("=" * 70)
    print("""
    Adds a "pharmacy check" node that validates medication dosing
    before the final recommendation. Like a pharmacist reviewing
    prescriptions before dispensing.

    Workflow: Extract → Risk → Classify → Assess → Propose → PHARMACY → Finalize

    Choose a demo:
      1 → Pharmacy check in action (CKD patient)
      2 → With vs without pharmacy check
      3 → Multiple cases
      4 → Interactive
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1": demo_pharmacy_check()
    elif choice == "2": demo_comparison()
    elif choice == "3": demo_multiple_cases()
    elif choice == "4": demo_interactive()
    elif choice == "5":
        demo_pharmacy_check()
        demo_comparison()
        demo_multiple_cases()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. NODE INSERTION: Adding a node between 'propose' and 'finalize'
   creates a safety checkpoint. The graph structure makes this clean:
   just add the node and update the edges.

2. PHARMACY AS A GUARDRAIL: This node checks dosing, contraindications,
   and interactions BEFORE the plan reaches the patient. This is a
   pattern you'll see again in Project 05 (healthcare_agent guardrails).

3. STATE ACCUMULATION: Each node adds to the state. By the time we
   reach pharmacy_check, the state contains: extracted data, risk
   factors, urgency, assessment, AND proposed plan. The pharmacy
   node has ALL the context it needs.

4. REAL-WORLD PARALLEL: In hospitals, a pharmacist reviews every
   medication order. This node automates that safety check. Production
   systems would connect to a real drug database (e.g., First Databank).

5. GRAPH FLEXIBILITY: Adding/removing nodes is easy in LangGraph.
   Want to add a billing node? Just add it and connect the edges.
   The graph structure keeps workflows modular and maintainable.
"""

if __name__ == "__main__":
    main()
