"""
Exercise 2: Specialty Router — Route to Specialist Agents

Skills practiced:
- Building a router/supervisor that classifies tasks
- Creating specialist agents for different medical domains
- Conditional routing to the right specialist
- Understanding the router pattern for multi-agent systems

Key insight: A general agent trying to handle cardiology, endocrinology,
  and psychiatry will be mediocre at all three. A router that sends each
  case to the RIGHT specialist produces much better results. This is
  the "supervisor + specialists" pattern used in production AI systems.
"""

import os
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# State for Specialty Router
# ============================================================

class RouterState(TypedDict):
    patient_case: str
    detected_specialty: str
    specialist_output: str
    final_response: str


# ============================================================
# Router Agent
# ============================================================

def router_agent(state: RouterState) -> dict:
    """Supervisor agent that classifies the case to the right specialty"""
    response = llm.invoke(
        f"""You are a MEDICAL ROUTING SUPERVISOR. Classify this patient case
to the appropriate specialty.

Patient case: {state['patient_case']}

Choose EXACTLY ONE specialty:
- cardiology (heart/vascular issues, chest pain, arrhythmia, heart failure, hypertension)
- endocrinology (diabetes, thyroid, hormonal issues, metabolic disorders)
- psychiatry (depression, anxiety, bipolar, psychosis, substance abuse)
- nephrology (kidney disease, dialysis, electrolyte imbalances, CKD)
- pulmonology (lung/breathing issues, COPD, asthma, pneumonia)
- general (doesn't fit a specialty, routine care, minor issues)

Respond with ONLY the specialty name, lowercase."""
    )

    specialty = response.content.strip().lower()
    valid = ["cardiology", "endocrinology", "psychiatry", "nephrology", "pulmonology", "general"]
    if specialty not in valid:
        # Find closest match
        for v in valid:
            if v in specialty:
                specialty = v
                break
        else:
            specialty = "general"

    return {"detected_specialty": specialty}


# ============================================================
# Specialist Agents
# ============================================================

def cardiology_specialist(state: RouterState) -> dict:
    response = llm.invoke(
        f"""You are a CARDIOLOGIST. Provide a specialized cardiac assessment.

Patient: {state['patient_case']}

Include:
1. Cardiac-specific differential diagnosis (top 3)
2. Recommended cardiac workup (ECG, echo, troponin, BNP, etc.)
3. Cardiac-specific treatment recommendations
4. Risk stratification (HEART score, TIMI, etc. as applicable)
5. Follow-up recommendations

Educational only."""
    )
    return {"specialist_output": f"[CARDIOLOGY]\n{response.content}"}


def endocrinology_specialist(state: RouterState) -> dict:
    response = llm.invoke(
        f"""You are an ENDOCRINOLOGIST. Provide a specialized metabolic assessment.

Patient: {state['patient_case']}

Include:
1. Endocrine differential diagnosis (top 3)
2. Recommended labs (HbA1c, TSH, cortisol, insulin levels, etc.)
3. Medication management (insulin, metformin, thyroid meds, etc.)
4. Target goals (HbA1c, glucose ranges, thyroid levels)
5. Lifestyle/dietary recommendations

Educational only."""
    )
    return {"specialist_output": f"[ENDOCRINOLOGY]\n{response.content}"}


def psychiatry_specialist(state: RouterState) -> dict:
    response = llm.invoke(
        f"""You are a PSYCHIATRIST. Provide a specialized mental health assessment.

Patient: {state['patient_case']}

Include:
1. Psychiatric differential diagnosis (top 3)
2. Validated screening tools (PHQ-9, GAD-7, AUDIT, etc.)
3. Medication recommendations (SSRIs, SNRIs, mood stabilizers, etc.)
4. Therapy recommendations (CBT, DBT, etc.)
5. Safety assessment (suicidal ideation screening)

Educational only."""
    )
    return {"specialist_output": f"[PSYCHIATRY]\n{response.content}"}


def nephrology_specialist(state: RouterState) -> dict:
    response = llm.invoke(
        f"""You are a NEPHROLOGIST. Provide a specialized renal assessment.

Patient: {state['patient_case']}

Include:
1. Renal differential diagnosis
2. CKD staging and progression assessment
3. Renal-specific labs (GFR, BMP, urine albumin, urine protein)
4. Medication dose adjustments for renal impairment
5. Nephroprotective strategies and dialysis considerations

Educational only."""
    )
    return {"specialist_output": f"[NEPHROLOGY]\n{response.content}"}


def pulmonology_specialist(state: RouterState) -> dict:
    response = llm.invoke(
        f"""You are a PULMONOLOGIST. Provide a specialized respiratory assessment.

Patient: {state['patient_case']}

Include:
1. Pulmonary differential diagnosis (top 3)
2. Recommended pulmonary workup (PFTs, chest imaging, ABG, etc.)
3. Respiratory-specific treatments (bronchodilators, steroids, oxygen, etc.)
4. Severity classification (GOLD for COPD, step therapy for asthma, etc.)
5. Pulmonary rehabilitation considerations

Educational only."""
    )
    return {"specialist_output": f"[PULMONOLOGY]\n{response.content}"}


def general_specialist(state: RouterState) -> dict:
    response = llm.invoke(
        f"""You are a GENERAL INTERNIST. Provide a comprehensive assessment.

Patient: {state['patient_case']}

Include a general assessment, recommended workup, and management plan.
If a specialty referral seems warranted, recommend it.

Educational only."""
    )
    return {"specialist_output": f"[GENERAL MEDICINE]\n{response.content}"}


def synthesize_response(state: RouterState) -> dict:
    """Synthesize the specialist output into a final response"""
    response = llm.invoke(
        f"Create a patient-friendly summary of this specialist assessment.\n\n"
        f"Specialty: {state['detected_specialty']}\n"
        f"Assessment: {state['specialist_output']}\n\n"
        f"Include: key findings, recommended next steps, and when to seek "
        f"immediate care. Educational only."
    )
    return {"final_response": response.content}


# ============================================================
# Build Router
# ============================================================

def route_to_specialist(state: RouterState) -> str:
    return state["detected_specialty"]


def build_specialty_router():
    graph = StateGraph(RouterState)

    graph.add_node("router", router_agent)
    graph.add_node("cardiology", cardiology_specialist)
    graph.add_node("endocrinology", endocrinology_specialist)
    graph.add_node("psychiatry", psychiatry_specialist)
    graph.add_node("nephrology", nephrology_specialist)
    graph.add_node("pulmonology", pulmonology_specialist)
    graph.add_node("general", general_specialist)
    graph.add_node("synthesize", synthesize_response)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_to_specialist,
        {
            "cardiology": "cardiology",
            "endocrinology": "endocrinology",
            "psychiatry": "psychiatry",
            "nephrology": "nephrology",
            "pulmonology": "pulmonology",
            "general": "general",
        }
    )

    # All specialists converge at synthesize
    for specialty in ["cardiology", "endocrinology", "psychiatry", "nephrology", "pulmonology", "general"]:
        graph.add_edge(specialty, "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


# ============================================================
# DEMO 1: Routing in Action
# ============================================================

def demo_routing():
    """Show cases being routed to different specialties"""
    print("\n" + "=" * 70)
    print("DEMO 1: SPECIALTY ROUTING IN ACTION")
    print("=" * 70)
    print("""
    The router classifies each case and sends it to the right specialist.
    """)

    app = build_specialty_router()

    cases = [
        "58-year-old male with chest pressure, sweating, left arm pain for 30 minutes",
        "45-year-old female, HbA1c 9.2%, polyuria and polydipsia for 2 months",
        "32-year-old male, 3 months of depressed mood, insomnia, loss of interest",
        "70-year-old female, GFR 22, creatinine 3.1, bilateral edema, on metformin",
        "55-year-old male, productive cough 3 weeks, wheeze, 40-pack-year smoking history",
    ]

    for case in cases:
        print(f"\n{'─' * 60}")
        print(f"  Case: {case[:70]}...")
        result = app.invoke({"patient_case": case})
        specialty = result.get("detected_specialty", "?")
        print(f"  Routed to: {specialty.upper()}")
        print(f"  Assessment: {result.get('specialist_output', 'N/A')[:250]}...")


# ============================================================
# DEMO 2: Specialist Depth Comparison
# ============================================================

def demo_depth_comparison():
    """Compare specialist vs general response for the same case"""
    print("\n" + "=" * 70)
    print("DEMO 2: SPECIALIST vs GENERAL DEPTH")
    print("=" * 70)

    case = "62-year-old male with new-onset heart failure symptoms: dyspnea on exertion, bilateral edema, elevated BNP"

    # Run through router (should go to cardiology)
    app = build_specialty_router()
    routed_result = app.invoke({"patient_case": case})

    # Force through general
    general_state = RouterState(
        patient_case=case,
        detected_specialty="general",
        specialist_output="",
        final_response=""
    )
    general_out = general_specialist(general_state)

    print(f"\n  Case: {case}\n")
    print(f"  ROUTED TO: {routed_result.get('detected_specialty', '?').upper()}")
    print(f"\n  SPECIALIST RESPONSE:")
    print(f"  {routed_result.get('specialist_output', 'N/A')[:500]}")
    print(f"\n  GENERAL RESPONSE (same case):")
    print(f"  {general_out.get('specialist_output', 'N/A')[:500]}")

    print("""
    OBSERVATION: The cardiologist provides cardiac-specific workup
    (echo, BNP trending, troponin) and risk stratification.
    General medicine gives a broader but shallower assessment.
    Routing to the right specialist improves clinical depth.
    """)


# ============================================================
# DEMO 3: Ambiguous Cases
# ============================================================

def demo_ambiguous():
    """Cases that could go to multiple specialties"""
    print("\n" + "=" * 70)
    print("DEMO 3: AMBIGUOUS CASES — WHERE DOES IT ROUTE?")
    print("=" * 70)

    app = build_specialty_router()

    ambiguous = [
        "55-year-old diabetic with chest pain — cardiology or endocrinology?",
        "40-year-old with anxiety and palpitations — psychiatry or cardiology?",
        "65-year-old with CKD and uncontrolled diabetes — nephrology or endocrinology?",
        "50-year-old COPD patient with depression and insomnia — pulmonology or psychiatry?",
    ]

    for case in ambiguous:
        result = app.invoke({"patient_case": case})
        print(f"\n  Case: {case}")
        print(f"  Routed: {result.get('detected_specialty', '?').upper()}")

    print("""
    OBSERVATION: The router picks the PRIMARY specialty based on the
    chief complaint. In a production system, you might:
    - Route to primary specialty AND cc the secondary
    - Have the specialist recommend additional referrals
    - Use a multi-routing approach (parallel specialists)
    """)


# ============================================================
# DEMO 4: Interactive
# ============================================================

def demo_interactive():
    """Interactive specialty router"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE SPECIALTY ROUTER")
    print("=" * 70)
    print("  Specialties: cardiology, endocrinology, psychiatry, nephrology, pulmonology")
    print("  Type 'quit' to exit.\n")

    app = build_specialty_router()

    while True:
        case = input("  Patient case: ").strip()
        if case.lower() in ['quit', 'exit', 'q']:
            break
        if not case:
            continue

        print("\n  Routing...\n")
        result = app.invoke({"patient_case": case})

        print(f"  Routed to: {result.get('detected_specialty', '?').upper()}")
        print(f"\n  Assessment: {result.get('specialist_output', 'N/A')[:500]}")
        print(f"\n  Summary: {result.get('final_response', 'N/A')[:300]}\n")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 2: SPECIALTY ROUTER — ROUTE TO SPECIALISTS")
    print("=" * 70)
    print("""
    Router classifies cases and sends them to the right specialist:
    Cardiology, Endocrinology, Psychiatry, Nephrology, Pulmonology, General

    Choose a demo:
      1 → Routing in action (5 cases, 5 specialties)
      2 → Specialist vs general depth comparison
      3 → Ambiguous cases
      4 → Interactive
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1": demo_routing()
    elif choice == "2": demo_depth_comparison()
    elif choice == "3": demo_ambiguous()
    elif choice == "4": demo_interactive()
    elif choice == "5":
        demo_routing()
        demo_depth_comparison()
        demo_ambiguous()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. ROUTER PATTERN: A supervisor agent classifies → routes → specialist
   handles → synthesize. This is a core multi-agent architecture.

2. SPECIALIZATION > GENERALIZATION: A cardiology specialist gives
   cardiac-specific workup, risk scores, and treatments. A general
   agent gives broader but shallower advice. Route to the specialist.

3. SYSTEM PROMPT = SPECIALIZATION: Each specialist has a detailed
   system prompt specifying their domain, what to include, and how
   to format. The LLM is the same — the prompt creates the specialty.

4. AMBIGUOUS CASES: Some cases could go to multiple specialties.
   Production systems handle this with primary/secondary routing,
   multi-specialist consultation, or explicit referral recommendations.

5. EXTENSIBILITY: Adding a new specialty = add one node + one
   conditional edge. The router just needs to know the new option.
   This scales well as you add more departments.
"""

if __name__ == "__main__":
    main()
