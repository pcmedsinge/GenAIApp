"""
Exercise 4: 4-Agent Encounter Pipeline

Skills practiced:
- Building an end-to-end multi-agent pipeline for a complete patient encounter
- Coordinating intake, assessment, treatment, and discharge agents
- Structured handoffs between pipeline stages
- Understanding how real clinical workflows map to agent orchestration

Key insight: A real patient encounter follows a predictable flow:
  intake → assessment → treatment → discharge. Each stage is a
  specialized agent that receives context from the previous stage
  and passes enriched context forward — like a relay race with
  clinical data as the baton.
"""

import os
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# Encounter State
# ============================================================

class EncounterState(TypedDict):
    patient_info: str            # Raw patient presentation
    # Stage 1: Intake
    intake_summary: str          # Demographics, vitals, chief complaint
    # Stage 2: Assessment
    assessment: str              # Clinical assessment, differentials
    risk_level: str              # low / moderate / high / critical
    # Stage 3: Treatment
    treatment_plan: str          # Medications, procedures, orders
    # Stage 4: Discharge
    discharge_summary: str       # Instructions, follow-up, education
    # Meta
    encounter_log: str           # Running log of agent handoffs


# ============================================================
# Agent 1: Intake Agent
# ============================================================

def intake_agent(state: EncounterState) -> dict:
    """Processes patient arrival — demographics, vitals, chief complaint"""
    response = llm.invoke(
        f"""You are an INTAKE NURSE agent. Process this patient presentation
and create a structured intake summary.

Patient presentation: {state['patient_info']}

Extract and organize:
1. DEMOGRAPHICS: Age, sex, weight if available
2. CHIEF COMPLAINT: Why they're here (one line)
3. VITAL SIGNS: Any vitals mentioned (BP, HR, temp, etc.)
4. CURRENT MEDICATIONS: List all medications
5. ALLERGIES: Any known allergies
6. RELEVANT HISTORY: Past medical history
7. INTAKE PRIORITY: low / moderate / high / critical

Format as a structured intake form. If info is missing, note it.
Educational purposes only."""
    )

    log = f"[INTAKE] Processed patient arrival.\n"
    return {
        "intake_summary": response.content,
        "encounter_log": state.get("encounter_log", "") + log,
    }


# ============================================================
# Agent 2: Assessment Agent
# ============================================================

def assessment_agent(state: EncounterState) -> dict:
    """Clinical assessment — differential diagnosis, risk stratification"""
    response = llm.invoke(
        f"""You are a CLINICAL ASSESSMENT agent (physician). Based on the
intake summary, perform a clinical assessment.

INTAKE SUMMARY:
{state['intake_summary']}

Provide:
1. CLINICAL IMPRESSION: Most likely diagnosis
2. DIFFERENTIAL DIAGNOSIS: Top 3 alternatives
3. RISK STRATIFICATION: low / moderate / high / critical with reasoning
4. RECOMMENDED WORKUP: Labs, imaging, consults needed
5. RED FLAGS: Any urgent concerns

Be specific and evidence-based. Educational purposes only."""
    )

    content = response.content.lower()
    if "critical" in content:
        risk = "critical"
    elif "high" in content:
        risk = "high"
    elif "moderate" in content:
        risk = "moderate"
    else:
        risk = "low"

    log = f"[ASSESSMENT] Risk level: {risk}. Assessment complete.\n"
    return {
        "assessment": response.content,
        "risk_level": risk,
        "encounter_log": state.get("encounter_log", "") + log,
    }


# ============================================================
# Agent 3: Treatment Agent
# ============================================================

def treatment_agent(state: EncounterState) -> dict:
    """Creates treatment plan based on assessment"""
    response = llm.invoke(
        f"""You are a TREATMENT PLANNING agent. Based on the clinical
assessment, create a treatment plan.

CLINICAL ASSESSMENT:
{state['assessment']}

RISK LEVEL: {state['risk_level']}

Provide:
1. IMMEDIATE ORDERS: What to do now (within 1 hour)
2. MEDICATIONS:
   - Name, dose, route, frequency
   - Duration
   - Monitoring parameters
3. PROCEDURES: Any procedures ordered
4. MONITORING: Vitals frequency, labs to repeat
5. CONSULTS: Specialist referrals needed
6. PRECAUTIONS: Safety alerts for nursing staff

Adjust aggressiveness based on risk level.
Educational purposes only."""
    )

    log = f"[TREATMENT] Plan created for {state['risk_level']} risk patient.\n"
    return {
        "treatment_plan": response.content,
        "encounter_log": state.get("encounter_log", "") + log,
    }


# ============================================================
# Agent 4: Discharge Agent
# ============================================================

def discharge_agent(state: EncounterState) -> dict:
    """Creates discharge summary and patient instructions"""
    response = llm.invoke(
        f"""You are a DISCHARGE PLANNING agent. Create a discharge summary
and patient instructions.

INTAKE: {state['intake_summary'][:200]}
ASSESSMENT: {state['assessment'][:200]}
TREATMENT PLAN: {state['treatment_plan'][:200]}
RISK LEVEL: {state['risk_level']}

Provide:
1. DISCHARGE SUMMARY: Brief narrative (who, what, when)
2. DIAGNOSIS: Final working diagnosis
3. MEDICATIONS AT DISCHARGE:
   - New medications with instructions
   - Changed medications
   - Stopped medications
4. PATIENT INSTRUCTIONS (plain language):
   - Diet restrictions
   - Activity level
   - Warning signs to watch for
   - When to return to ED
5. FOLLOW-UP:
   - Who to see and when
   - Labs to get before follow-up
6. PATIENT EDUCATION: Key points about their condition

Write patient instructions at a 6th-grade reading level.
Educational purposes only."""
    )

    log = f"[DISCHARGE] Summary and instructions complete.\n"
    return {
        "discharge_summary": response.content,
        "encounter_log": state.get("encounter_log", "") + log,
    }


# ============================================================
# Build Pipeline
# ============================================================

def build_encounter_pipeline():
    graph = StateGraph(EncounterState)

    graph.add_node("intake", intake_agent)
    graph.add_node("assessment", assessment_agent)
    graph.add_node("treatment", treatment_agent)
    graph.add_node("discharge", discharge_agent)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "assessment")
    graph.add_edge("assessment", "treatment")
    graph.add_edge("treatment", "discharge")
    graph.add_edge("discharge", END)

    return graph.compile()


# ============================================================
# DEMO 1: Complete Encounter
# ============================================================

def demo_complete_encounter():
    """Run a full 4-agent encounter pipeline"""
    print("\n" + "=" * 70)
    print("DEMO 1: COMPLETE PATIENT ENCOUNTER")
    print("=" * 70)
    print("""
    Pipeline: Intake → Assessment → Treatment → Discharge
    Each agent receives context from the previous agent.
    """)

    app = build_encounter_pipeline()

    result = app.invoke({
        "patient_info": "72-year-old male presents to ED with chest pain radiating "
                        "to left arm × 2 hours. PMH: HTN, T2DM, hyperlipidemia. "
                        "Meds: metoprolol 50mg BID, metformin 1000mg BID, "
                        "atorvastatin 40mg daily. BP 165/95, HR 92, O2 sat 96%. "
                        "ECG shows ST depression in leads V4-V6. No allergies.",
        "encounter_log": ""
    })

    print(f"\n{'─' * 60}")
    print("  [AGENT 1] INTAKE SUMMARY:")
    print(f"{'─' * 60}")
    print(f"  {result.get('intake_summary', 'N/A')[:400]}")

    print(f"\n{'─' * 60}")
    print(f"  [AGENT 2] CLINICAL ASSESSMENT (Risk: {result.get('risk_level', '?').upper()}):")
    print(f"{'─' * 60}")
    print(f"  {result.get('assessment', 'N/A')[:400]}")

    print(f"\n{'─' * 60}")
    print("  [AGENT 3] TREATMENT PLAN:")
    print(f"{'─' * 60}")
    print(f"  {result.get('treatment_plan', 'N/A')[:400]}")

    print(f"\n{'─' * 60}")
    print("  [AGENT 4] DISCHARGE SUMMARY:")
    print(f"{'─' * 60}")
    print(f"  {result.get('discharge_summary', 'N/A')[:400]}")

    print(f"\n{'─' * 60}")
    print("  ENCOUNTER LOG:")
    print(f"{'─' * 60}")
    print(f"  {result.get('encounter_log', '')}")


# ============================================================
# DEMO 2: Compare Risk Levels
# ============================================================

def demo_compare_risk_levels():
    """Show how the pipeline adapts to different risk levels"""
    print("\n" + "=" * 70)
    print("DEMO 2: RISK LEVEL COMPARISON")
    print("=" * 70)

    app = build_encounter_pipeline()

    cases = [
        {
            "label": "LOW RISK — Ankle sprain",
            "patient_info": "28-year-old male twisted ankle playing basketball. "
                            "Pain 4/10, mild swelling, can bear weight. "
                            "BP 120/75, HR 72. No meds, no allergies."
        },
        {
            "label": "HIGH RISK — DKA",
            "patient_info": "45-year-old female T1DM found confused by family. "
                            "Blood glucose 520, fruity breath, rapid breathing. "
                            "BP 90/60, HR 110, temp 99.5. Meds: insulin glargine "
                            "30u daily, insulin lispro sliding scale. Allergic to sulfa."
        },
    ]

    for case in cases:
        print(f"\n{'=' * 60}")
        print(f"  CASE: {case['label']}")
        print(f"{'=' * 60}")

        result = app.invoke({"patient_info": case["patient_info"], "encounter_log": ""})

        print(f"\n  RISK LEVEL: {result.get('risk_level', '?').upper()}")
        print(f"\n  TREATMENT (excerpt): {result.get('treatment_plan', 'N/A')[:300]}...")
        print(f"\n  DISCHARGE (excerpt): {result.get('discharge_summary', 'N/A')[:300]}...")


# ============================================================
# DEMO 3: Encounter Log & Agent Handoffs
# ============================================================

def demo_agent_handoffs():
    """Show how context flows between agents"""
    print("\n" + "=" * 70)
    print("DEMO 3: AGENT HANDOFFS & CONTEXT FLOW")
    print("=" * 70)
    print("""
    Each agent enriches the state with its output.
    The next agent builds on ALL previous outputs.

    Intake provides → demographics, vitals, chief complaint
    Assessment uses intake → adds diagnosis, risk level
    Treatment uses assessment → adds medications, orders
    Discharge uses ALL → creates patient-friendly summary
    """)

    app = build_encounter_pipeline()

    result = app.invoke({
        "patient_info": "55-year-old female, 90kg, presents with 3-day history of "
                        "worsening dyspnea, orthopnea, bilateral leg edema. "
                        "PMH: CHF (EF 30%), CKD Stage 3, AFib. "
                        "Meds: furosemide 40mg daily, carvedilol 12.5mg BID, "
                        "apixaban 5mg BID, lisinopril 10mg daily. "
                        "BP 145/90, HR 105 irregular, O2 sat 89% on RA, "
                        "crackles bilateral bases. BNP 1850, Cr 2.1, K 5.2.",
        "encounter_log": ""
    })

    print(f"\n  ENCOUNTER LOG (shows agent sequence):")
    print(f"  {result.get('encounter_log', '')}")

    # Show state contents at each stage
    print(f"\n  CONTEXT GROWTH:")
    print(f"    After Intake:     intake_summary ({len(result.get('intake_summary', ''))} chars)")
    print(f"    After Assessment: + assessment ({len(result.get('assessment', ''))} chars)")
    print(f"    After Treatment:  + treatment_plan ({len(result.get('treatment_plan', ''))} chars)")
    print(f"    After Discharge:  + discharge_summary ({len(result.get('discharge_summary', ''))} chars)")

    total = sum(len(result.get(k, '')) for k in [
        'intake_summary', 'assessment', 'treatment_plan', 'discharge_summary'
    ])
    print(f"    TOTAL accumulated context: {total} chars")

    print(f"""
    WHY THIS MATTERS:
      • Each agent focuses on ONE task (separation of concerns)
      • Context accumulates — discharge agent sees everything
      • Easy to add/remove/reorder agents
      • Each agent can be independently tested and improved
    """)


# ============================================================
# DEMO 4: Interactive
# ============================================================

def demo_interactive():
    """Interactive patient encounter"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE ENCOUNTER")
    print("=" * 70)
    print("  Describe a patient. Type 'quit' to exit.\n")

    app = build_encounter_pipeline()

    while True:
        case = input("  Patient presentation: ").strip()
        if case.lower() in ['quit', 'exit', 'q']:
            break
        if not case:
            continue

        print("\n  Running 4-agent encounter pipeline...\n")
        result = app.invoke({"patient_info": case, "encounter_log": ""})

        print(f"  RISK: {result.get('risk_level', '?').upper()}")
        print(f"\n  INTAKE:\n  {result.get('intake_summary', 'N/A')[:300]}...")
        print(f"\n  ASSESSMENT:\n  {result.get('assessment', 'N/A')[:300]}...")
        print(f"\n  TREATMENT:\n  {result.get('treatment_plan', 'N/A')[:300]}...")
        print(f"\n  DISCHARGE:\n  {result.get('discharge_summary', 'N/A')[:300]}...")
        print(f"\n  LOG: {result.get('encounter_log', '')}\n")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 4: 4-AGENT ENCOUNTER PIPELINE")
    print("=" * 70)
    print("""
    A complete patient encounter as a multi-agent pipeline:
      Intake → Assessment → Treatment → Discharge

    Each agent specializes in one phase and passes context forward.

    Choose a demo:
      1 → Complete encounter (chest pain case)
      2 → Compare risk levels (low vs high)
      3 → Agent handoffs & context flow
      4 → Interactive
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1": demo_complete_encounter()
    elif choice == "2": demo_compare_risk_levels()
    elif choice == "3": demo_agent_handoffs()
    elif choice == "4": demo_interactive()
    elif choice == "5":
        demo_complete_encounter()
        demo_compare_risk_levels()
        demo_agent_handoffs()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. ENCOUNTER AS PIPELINE: A patient encounter naturally maps to a
   sequential pipeline. Each stage (intake → assessment → treatment
   → discharge) is a distinct agent with specialized expertise.

2. CONTEXT ACCUMULATION: The state grows as it passes through agents.
   The discharge agent has access to ALL previous stage outputs —
   just like a real clinician reviewing the chart before discharge.

3. SEPARATION OF CONCERNS: Each agent focuses on ONE task. The intake
   agent doesn't prescribe; the treatment agent doesn't do intake.
   This mirrors real clinical workflows with specialized roles.

4. RISK-ADAPTIVE BEHAVIOR: The treatment and discharge agents adjust
   their output based on risk level from the assessment agent.
   Higher risk → more aggressive treatment, stricter follow-up.

5. PRODUCTION PATTERNS: In a real system, you'd add:
   - Parallel workup (labs + imaging running concurrently)
   - Specialist consultation as a conditional branch
   - Real-time vital sign monitoring between stages
   - Patient consent gates before treatment
   - Billing/coding agent as a final step
"""

if __name__ == "__main__":
    main()
