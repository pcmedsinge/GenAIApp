"""
Exercise 1: Pharmacist Agent — Medication Safety Validator

Skills practiced:
- Adding a new specialist agent to a multi-agent pipeline
- Implementing medication safety validation as a dedicated agent
- Understanding agent specialization (each agent has ONE job)
- Inserting an agent between existing pipeline stages

Key insight: In hospitals, a pharmacist reviews EVERY medication order
  before it reaches the patient. This exercise adds a "pharmacist agent"
  to the pipeline that validates medication safety — dosing, interactions,
  contraindications — before the final report.
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
# State with Pharmacist Agent
# ============================================================

class PharmPipelineState(TypedDict):
    patient_case: str
    triage_output: str
    diagnosis_output: str
    treatment_output: str
    pharmacist_output: str     # NEW: Pharmacist review
    safety_output: str
    final_report: str


# ============================================================
# Medication Database for Pharmacist
# ============================================================

MEDICATION_DB = {
    "metformin": {"max_dose_mg": 2000, "renal_limit_gfr": 30, "dose_reduce_gfr": 45, "interactions": ["contrast_dye"], "monitor": ["HbA1c", "B12", "renal function"]},
    "lisinopril": {"max_dose_mg": 40, "interactions": ["potassium_supplements", "NSAIDs"], "monitor": ["K+", "Cr", "BP"], "contraindicated": ["pregnancy", "angioedema_history"]},
    "apixaban": {"standard_dose_mg": 5, "reduced_dose_mg": 2.5, "reduce_criteria": "age≥80 OR weight≤60kg OR Cr≥1.5", "interactions": ["aspirin", "dual_antiplatelet"], "monitor": ["bleeding_signs"]},
    "amlodipine": {"max_dose_mg": 10, "interactions": ["simvastatin_high_dose"], "monitor": ["BP", "edema"]},
    "sertraline": {"max_dose_mg": 200, "interactions": ["tramadol", "MAOi"], "monitor": ["mood", "suicidal_ideation"]},
    "furosemide": {"max_dose_mg": 600, "interactions": ["aminoglycosides", "lithium"], "monitor": ["K+", "Na+", "renal function", "weight"]},
    "warfarin": {"interactions": ["aspirin", "NSAIDs", "many_herbals"], "monitor": ["INR_weekly", "signs_of_bleeding"]},
    "insulin": {"interactions": ["beta_blockers_mask_hypoglycemia"], "monitor": ["blood_glucose", "HbA1c", "hypoglycemia_symptoms"]},
}


# ============================================================
# Pipeline Agents
# ============================================================

def triage_agent(state: PharmPipelineState) -> dict:
    """Agent 1: Triage Specialist"""
    response = llm.invoke(
        f"You are a TRIAGE SPECIALIST. Classify urgency and identify key concerns.\n\n"
        f"Patient: {state['patient_case']}\n\n"
        f"Provide: urgency level (emergency/urgent/routine), key vital signs concerns, "
        f"and initial clinical impression. Be concise."
    )
    return {"triage_output": response.content}


def diagnosis_agent(state: PharmPipelineState) -> dict:
    """Agent 2: Diagnostic Specialist"""
    response = llm.invoke(
        f"You are a DIAGNOSTIC SPECIALIST. Based on triage, provide differential diagnosis.\n\n"
        f"Patient: {state['patient_case']}\n"
        f"Triage: {state['triage_output']}\n\n"
        f"Provide: top 3 differential diagnoses with reasoning, and recommended workup."
    )
    return {"diagnosis_output": response.content}


def treatment_agent(state: PharmPipelineState) -> dict:
    """Agent 3: Treatment Specialist"""
    response = llm.invoke(
        f"You are a TREATMENT SPECIALIST. Propose specific treatments with medications and doses.\n\n"
        f"Patient: {state['patient_case']}\n"
        f"Diagnosis: {state['diagnosis_output']}\n\n"
        f"Provide: specific medications with doses, non-pharmacological treatments, "
        f"and monitoring plan. Be specific about drug names and doses. Educational only."
    )
    return {"treatment_output": response.content}


def pharmacist_agent(state: PharmPipelineState) -> dict:
    """Agent 4 (NEW): Clinical Pharmacist — Validates medication safety"""

    # Build medication reference context
    med_context = []
    for med_name, info in MEDICATION_DB.items():
        med_context.append(f"  {med_name}: max={info.get('max_dose_mg', 'varies')}mg, "
                          f"interactions={info.get('interactions', [])}, "
                          f"monitor={info.get('monitor', [])}")

    response = llm.invoke(
        f"""You are a CLINICAL PHARMACIST. Review the proposed treatment plan
for medication safety BEFORE it reaches the patient.

PATIENT CASE:
{state['patient_case']}

PROPOSED TREATMENT:
{state['treatment_output']}

MEDICATION DATABASE:
{chr(10).join(med_context)}

Check for:
1. DOSING: Are proposed doses within safe limits?
2. INTERACTIONS: Any drug-drug interactions between proposed AND current meds?
3. CONTRAINDICATIONS: Any meds contraindicated given patient conditions/labs?
4. MONITORING: What labs/vitals must be monitored?
5. RENAL DOSING: Any dose adjustments needed for renal function?

Format your review as:
- APPROVED medications (safe as proposed)
- FLAGGED medications (concerns identified)
- REQUIRED MONITORING (what to order)
- OVERALL STATUS: APPROVED / APPROVED WITH MODIFICATIONS / HOLD FOR REVIEW

Educational purposes only."""
    )
    return {"pharmacist_output": response.content}


def safety_agent(state: PharmPipelineState) -> dict:
    """Agent 5: Safety Reviewer (now reviews pharmacist output too)"""
    response = llm.invoke(
        f"You are a PATIENT SAFETY officer. Final safety review.\n\n"
        f"Patient: {state['patient_case']}\n"
        f"Treatment: {state['treatment_output']}\n"
        f"Pharmacist review: {state['pharmacist_output']}\n\n"
        f"Identify any remaining safety concerns. If the pharmacist flagged issues, "
        f"ensure they are addressed. Confirm or escalate. Educational only."
    )
    return {"safety_output": response.content}


def report_agent(state: PharmPipelineState) -> dict:
    """Agent 6: Report Generator"""
    response = llm.invoke(
        f"Generate a final clinical report integrating all specialist inputs.\n\n"
        f"Triage: {state['triage_output']}\n"
        f"Diagnosis: {state['diagnosis_output']}\n"
        f"Treatment: {state['treatment_output']}\n"
        f"Pharmacist: {state['pharmacist_output']}\n"
        f"Safety: {state['safety_output']}\n\n"
        f"Create a structured report with sections. Note any pharmacist modifications. "
        f"Educational only."
    )
    return {"final_report": response.content}


# ============================================================
# Build Pipeline
# ============================================================

def build_pharm_pipeline():
    graph = StateGraph(PharmPipelineState)

    graph.add_node("triage", triage_agent)
    graph.add_node("diagnosis", diagnosis_agent)
    graph.add_node("treatment", treatment_agent)
    graph.add_node("pharmacist", pharmacist_agent)   # NEW
    graph.add_node("safety", safety_agent)
    graph.add_node("report", report_agent)

    # Pipeline: triage → diagnosis → treatment → PHARMACIST → safety → report
    graph.set_entry_point("triage")
    graph.add_edge("triage", "diagnosis")
    graph.add_edge("diagnosis", "treatment")
    graph.add_edge("treatment", "pharmacist")   # Treatment goes through pharmacist
    graph.add_edge("pharmacist", "safety")      # Pharmacist feeds into safety
    graph.add_edge("safety", "report")
    graph.add_edge("report", END)

    return graph.compile()


# ============================================================
# DEMO 1: Full Pipeline with Pharmacist
# ============================================================

def demo_full_pipeline():
    """Run the full 6-agent pipeline"""
    print("\n" + "=" * 70)
    print("DEMO 1: FULL PIPELINE WITH PHARMACIST AGENT")
    print("=" * 70)
    print("""
    Pipeline: Triage → Diagnosis → Treatment → PHARMACIST → Safety → Report

    The pharmacist reviews EVERY medication before it reaches the patient.
    """)

    app = build_pharm_pipeline()

    result = app.invoke({
        "patient_case": "68-year-old male with CKD Stage 3 (GFR 35), Type 2 Diabetes, "
                        "HTN, and atrial fibrillation. Current meds: metformin 2000mg daily, "
                        "lisinopril 20mg, apixaban 5mg BID. Presenting with worsening "
                        "shortness of breath and bilateral edema. K+ 5.4, Cr 2.1, HbA1c 8.2%."
    })

    agents = [
        ("TRIAGE", "triage_output"),
        ("DIAGNOSIS", "diagnosis_output"),
        ("TREATMENT", "treatment_output"),
        ("PHARMACIST", "pharmacist_output"),
        ("SAFETY", "safety_output"),
    ]

    for label, key in agents:
        print(f"\n{'─' * 60}")
        print(f"  {label} AGENT:")
        print(f"  {result.get(key, 'N/A')[:400]}")

    print(f"\n{'═' * 60}")
    print(f"  FINAL REPORT:")
    print(f"  {result.get('final_report', 'N/A')[:600]}")


# ============================================================
# DEMO 2: Cases That Trigger Pharmacist Flags
# ============================================================

def demo_flagged_cases():
    """Cases designed to trigger pharmacist safety flags"""
    print("\n" + "=" * 70)
    print("DEMO 2: CASES THAT TRIGGER PHARMACIST FLAGS")
    print("=" * 70)

    app = build_pharm_pipeline()

    cases = [
        {
            "label": "Renal Dosing Issue",
            "patient_case": "82-year-old female, 52kg, CKD Stage 4 (GFR 22), AFib on apixaban 5mg BID, "
                           "diabetes on metformin 1000mg BID. Cr 2.8. Presenting for routine follow-up."
        },
        {
            "label": "Drug Interaction Risk",
            "patient_case": "45-year-old female, depression on sertraline 150mg, chronic pain. "
                           "Requesting tramadol for worsening back pain. No renal issues."
        },
        {
            "label": "High Bleeding Risk",
            "patient_case": "75-year-old male on warfarin for AFib (INR 3.2), recent GI bleed 2 months ago. "
                           "Now has new knee pain, requesting NSAID."
        },
    ]

    for case in cases:
        print(f"\n{'═' * 60}")
        print(f"  CASE: {case['label']}")
        print(f"  Patient: {case['patient_case'][:100]}...\n")

        result = app.invoke({"patient_case": case["patient_case"]})

        print(f"  PHARMACIST REVIEW:")
        print(f"  {result.get('pharmacist_output', 'N/A')[:500]}")
        print()


# ============================================================
# DEMO 3: With vs Without Pharmacist
# ============================================================

def demo_comparison():
    """Compare pipeline output with and without pharmacist"""
    print("\n" + "=" * 70)
    print("DEMO 3: WITH vs WITHOUT PHARMACIST")
    print("=" * 70)

    # WITHOUT pharmacist
    graph_no_pharm = StateGraph(PharmPipelineState)
    graph_no_pharm.add_node("triage", triage_agent)
    graph_no_pharm.add_node("diagnosis", diagnosis_agent)
    graph_no_pharm.add_node("treatment", treatment_agent)
    graph_no_pharm.add_node("report", report_agent)
    graph_no_pharm.set_entry_point("triage")
    graph_no_pharm.add_edge("triage", "diagnosis")
    graph_no_pharm.add_edge("diagnosis", "treatment")
    graph_no_pharm.add_edge("treatment", "report")
    graph_no_pharm.add_edge("report", END)
    no_pharm = graph_no_pharm.compile()

    # WITH pharmacist
    with_pharm = build_pharm_pipeline()

    case = "78-year-old, GFR 28, on metformin 2000mg and lisinopril 40mg, K+ 5.6. New symptoms: fatigue and nausea."

    print(f"\n  Case: {case}\n")

    result_no = no_pharm.invoke({"patient_case": case})
    result_with = with_pharm.invoke({"patient_case": case})

    print(f"  WITHOUT PHARMACIST:")
    print(f"  Treatment: {result_no.get('treatment_output', 'N/A')[:300]}...")

    print(f"\n  WITH PHARMACIST:")
    print(f"  Treatment: {result_with.get('treatment_output', 'N/A')[:200]}...")
    print(f"  Pharmacist flags: {result_with.get('pharmacist_output', 'N/A')[:300]}...")

    print("""
    The pharmacist should flag:
    - Metformin 2000mg with GFR 28 → CONTRAINDICATED (GFR<30)
    - K+ 5.6 on lisinopril → hyperkalemia risk
    """)


# ============================================================
# DEMO 4: Interactive
# ============================================================

def demo_interactive():
    """Interactive pipeline with pharmacist"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE PIPELINE WITH PHARMACIST")
    print("=" * 70)
    print("  Type 'quit' to exit.\n")

    app = build_pharm_pipeline()

    while True:
        case = input("  Patient case: ").strip()
        if case.lower() in ['quit', 'exit', 'q']:
            break
        if not case:
            continue

        print("\n  Running 6-agent pipeline...\n")
        result = app.invoke({"patient_case": case})

        print(f"  TRIAGE: {result.get('triage_output', 'N/A')[:200]}...")
        print(f"\n  PHARMACIST: {result.get('pharmacist_output', 'N/A')[:400]}...")
        print(f"\n  REPORT: {result.get('final_report', 'N/A')[:400]}...\n")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 1: PHARMACIST AGENT — MEDICATION SAFETY")
    print("=" * 70)
    print("""
    Adds a pharmacist agent to the multi-agent pipeline.
    Reviews every medication for dosing, interactions, and safety.

    Pipeline: Triage → Diagnosis → Treatment → PHARMACIST → Safety → Report

    Choose a demo:
      1 → Full pipeline (complex case)
      2 → Cases that trigger flags
      3 → With vs without pharmacist
      4 → Interactive
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1": demo_full_pipeline()
    elif choice == "2": demo_flagged_cases()
    elif choice == "3": demo_comparison()
    elif choice == "4": demo_interactive()
    elif choice == "5":
        demo_full_pipeline()
        demo_flagged_cases()
        demo_comparison()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. AGENT SPECIALIZATION: The pharmacist agent has ONE job — medication
   safety. It doesn't diagnose or treat. This focused role makes it
   more reliable than asking a general agent to "also check meds."

2. PIPELINE POSITION MATTERS: The pharmacist sits AFTER treatment
   (to review proposed meds) and BEFORE safety (to feed its flags
   into the final safety review). Order matters in pipelines.

3. MEDICATION DATABASE: The pharmacist uses a structured database
   (max doses, interactions, monitoring) as ground truth. This is
   more reliable than asking the LLM to recall drug facts from memory.

4. FLAG-AND-CONTINUE: The pharmacist doesn't block the pipeline.
   It flags concerns and passes them forward. The safety agent and
   report generator incorporate those flags. This is a production pattern.

5. REAL-WORLD PARALLEL: Every hospital has a pharmacy that reviews
   medication orders. This agent automates that safety check — a
   critical compliance requirement in healthcare AI.
"""

if __name__ == "__main__":
    main()
