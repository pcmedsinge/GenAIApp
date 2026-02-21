"""
Project 4: Multi-Agent Systems
Multiple specialized agents working together on complex tasks.

Key Patterns:
1. Sequential: Agent A → Agent B → Agent C (pipeline)
2. Router: Supervisor sends tasks to specialist agents
3. Debate: Two agents discuss/critique to improve output

Why multi-agent?
  Single agent handling everything → inconsistent, unfocused
  Multiple specialists → each does its job well, combined output is better

Builds on: Project 01 (ReAct), 02 (LangChain), 03 (LangGraph)
"""

import os
import json
from typing import TypedDict, Literal
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# DEMO 1: Sequential Pipeline (Agent Handoff)
# ============================================================

class PipelineState(TypedDict):
    patient_case: str
    triage_output: str
    diagnosis_output: str
    treatment_output: str
    safety_output: str
    final_report: str


def triage_agent(state: PipelineState) -> dict:
    """Agent 1: Triage Specialist"""
    response = llm.invoke(
        f"""You are a TRIAGE SPECIALIST nurse. Your ONLY job is to classify urgency.

Patient case: {state['patient_case']}

Classify as: EMERGENCY (life-threatening) | URGENT (same-day) | ROUTINE (scheduled)
Provide: urgency level, key concerning findings, recommended setting (ED/clinic/telehealth).
Be concise (3-4 sentences). You are ONLY doing triage, not diagnosis."""
    )
    return {"triage_output": response.content}


def diagnosis_agent(state: PipelineState) -> dict:
    """Agent 2: Diagnostic Specialist"""
    response = llm.invoke(
        f"""You are a DIAGNOSTIC SPECIALIST physician. Your ONLY job is differential diagnosis.

Patient case: {state['patient_case']}
Triage assessment: {state['triage_output']}

Provide: Top 3 differential diagnoses ranked by likelihood, key findings supporting each,
and recommended diagnostic workup (labs, imaging). You are ONLY doing diagnosis, not treatment."""
    )
    return {"diagnosis_output": response.content}


def treatment_agent(state: PipelineState) -> dict:
    """Agent 3: Treatment Specialist"""
    response = llm.invoke(
        f"""You are a TREATMENT SPECIALIST physician. Your ONLY job is treatment planning.

Patient case: {state['patient_case']}
Triage: {state['triage_output']}
Diagnoses: {state['diagnosis_output']}

Provide: Treatment plan for the most likely diagnosis, specific medications with doses,
non-pharmacological interventions, and follow-up timeline. Educational purposes only."""
    )
    return {"treatment_output": response.content}


def safety_agent(state: PipelineState) -> dict:
    """Agent 4: Safety Review Specialist"""
    response = llm.invoke(
        f"""You are a PATIENT SAFETY specialist. Your ONLY job is safety review.

Patient case: {state['patient_case']}
Proposed treatment: {state['treatment_output']}

Review for:
1. Drug interactions or contraindications
2. Missing safety considerations (allergies, age, renal/hepatic function)
3. Red flags that need immediate attention
4. Recommended monitoring

Flag any concerns. If everything looks safe, confirm clearance."""
    )
    return {"safety_output": response.content}


def compile_report(state: PipelineState) -> dict:
    """Final node: Compile all agent outputs"""
    report = f"""
═══════════════════════════════════════════════════════
CLINICAL DECISION SUPPORT REPORT
═══════════════════════════════════════════════════════

PATIENT CASE:
{state['patient_case']}

───────────────────────────────────────────────────────
🏥 TRIAGE (Agent 1):
{state['triage_output']}

───────────────────────────────────────────────────────
🔬 DIFFERENTIAL DIAGNOSIS (Agent 2):
{state['diagnosis_output']}

───────────────────────────────────────────────────────
💊 TREATMENT PLAN (Agent 3):
{state['treatment_output']}

───────────────────────────────────────────────────────
🛡️ SAFETY REVIEW (Agent 4):
{state['safety_output']}

═══════════════════════════════════════════════════════
⚕️ For educational purposes only. Consult a healthcare provider.
═══════════════════════════════════════════════════════"""
    return {"final_report": report}


def demo_sequential_pipeline():
    """Run the 4-agent pipeline"""
    print("\n" + "=" * 70)
    print("DEMO 1: SEQUENTIAL MULTI-AGENT PIPELINE")
    print("=" * 70)
    print("""
💡 Four specialized agents in sequence:
   [Triage] → [Diagnosis] → [Treatment] → [Safety] → [Report]
   Each agent focuses on ONE job and does it well.
""")

    graph = StateGraph(PipelineState)
    graph.add_node("triage", triage_agent)
    graph.add_node("diagnosis", diagnosis_agent)
    graph.add_node("treatment", treatment_agent)
    graph.add_node("safety", safety_agent)
    graph.add_node("report", compile_report)

    graph.set_entry_point("triage")
    graph.add_edge("triage", "diagnosis")
    graph.add_edge("diagnosis", "treatment")
    graph.add_edge("treatment", "safety")
    graph.add_edge("safety", "report")
    graph.add_edge("report", END)

    app = graph.compile()

    case = ("68-year-old female presenting with 3 days of worsening shortness of breath, "
            "bilateral lower extremity edema, and 10-pound weight gain over 2 weeks. "
            "History of hypertension, type 2 diabetes, and coronary artery disease. "
            "Current medications: metformin 1000mg BID, lisinopril 20mg daily, "
            "aspirin 81mg daily, atorvastatin 40mg daily. "
            "Vitals: BP 158/94, HR 98, RR 24, SpO2 92% on room air.")

    print(f"📝 Running agents on case...\n")
    print(f"   Case: {case[:120]}...\n")

    # Run with step tracking
    for i, (agent_name, label) in enumerate([
        ("triage", "🏥 Triage Agent"),
        ("diagnosis", "🔬 Diagnosis Agent"),
        ("treatment", "💊 Treatment Agent"),
        ("safety", "🛡️ Safety Agent"),
    ], 1):
        print(f"   Step {i}: {label} processing...")

    result = app.invoke({"patient_case": case})
    print(result["final_report"])

    return app


# ============================================================
# DEMO 2: Router Pattern (Supervisor → Specialists)
# ============================================================

class RouterState(TypedDict):
    query: str
    specialty: str
    response: str


def router_agent(state: RouterState) -> dict:
    """Supervisor agent that routes to the right specialist"""
    response = llm.invoke(
        f"""You are a medical routing supervisor.
Classify this query into ONE specialty: cardiology, endocrinology, psychiatry, pulmonology, nephrology, general.
Respond with ONLY the specialty name, nothing else.

Query: {state['query']}"""
    )
    specialty = response.content.strip().lower()
    valid = ["cardiology", "endocrinology", "psychiatry", "pulmonology", "nephrology"]
    if not any(s in specialty for s in valid):
        specialty = "general"
    for s in valid:
        if s in specialty:
            specialty = s
            break
    return {"specialty": specialty}


def cardiology_specialist(state: RouterState) -> dict:
    response = llm.invoke(f"You are a CARDIOLOGY specialist. Answer this clinical question concisely:\n{state['query']}")
    return {"response": f"[CARDIOLOGY] {response.content}"}


def endocrinology_specialist(state: RouterState) -> dict:
    response = llm.invoke(f"You are an ENDOCRINOLOGY specialist. Answer this clinical question concisely:\n{state['query']}")
    return {"response": f"[ENDOCRINOLOGY] {response.content}"}


def psychiatry_specialist(state: RouterState) -> dict:
    response = llm.invoke(f"You are a PSYCHIATRY specialist. Answer this clinical question concisely:\n{state['query']}")
    return {"response": f"[PSYCHIATRY] {response.content}"}


def general_specialist(state: RouterState) -> dict:
    response = llm.invoke(f"You are a GENERAL medicine specialist. Answer this clinical question concisely:\n{state['query']}")
    return {"response": f"[GENERAL MEDICINE] {response.content}"}


def route_to_specialist(state: RouterState) -> str:
    """Conditional edge function"""
    routing = {
        "cardiology": "cardiology",
        "endocrinology": "endocrinology",
        "psychiatry": "psychiatry",
    }
    return routing.get(state["specialty"], "general")


def demo_router_pattern():
    """Supervisor routes queries to specialist agents"""
    print("\n" + "=" * 70)
    print("DEMO 2: ROUTER PATTERN (Supervisor → Specialists)")
    print("=" * 70)
    print("""
💡 One ROUTER agent classifies the query,
   then sends it to the RIGHT specialist agent.
""")

    graph = StateGraph(RouterState)
    graph.add_node("router", router_agent)
    graph.add_node("cardiology", cardiology_specialist)
    graph.add_node("endocrinology", endocrinology_specialist)
    graph.add_node("psychiatry", psychiatry_specialist)
    graph.add_node("general", general_specialist)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_to_specialist, {
        "cardiology": "cardiology",
        "endocrinology": "endocrinology",
        "psychiatry": "psychiatry",
        "general": "general",
    })
    for node in ["cardiology", "endocrinology", "psychiatry", "general"]:
        graph.add_edge(node, END)

    app = graph.compile()

    queries = [
        "What blood pressure target for a diabetic patient on ACE inhibitor?",
        "How long should sertraline be continued after depression remission?",
        "When should insulin be started in Type 2 diabetes?",
        "How should chronic lower back pain be managed?",
    ]

    for query in queries:
        print(f"\n{'─' * 70}")
        print(f"❓ {query}")
        result = app.invoke({"query": query})
        print(f"   📍 Routed to: {result['specialty'].upper()}")
        print(f"   📋 {result['response'][:300]}...")


# ============================================================
# DEMO 3: Agent Debate (Two Perspectives)
# ============================================================

class DebateState(TypedDict):
    clinical_question: str
    position_a: str
    position_b: str
    rebuttal_a: str
    rebuttal_b: str
    consensus: str


def advocate_agent(state: DebateState) -> dict:
    """Agent A: Argues FOR a particular approach"""
    response = llm.invoke(
        f"You are a physician who favors aggressive early treatment. "
        f"Argue FOR early intervention for this clinical scenario in 3-4 sentences:\n"
        f"{state['clinical_question']}"
    )
    return {"position_a": response.content}


def conservative_agent(state: DebateState) -> dict:
    """Agent B: Argues for conservative approach"""
    response = llm.invoke(
        f"You are a physician who favors conservative, watchful management. "
        f"Argue FOR conservative approach for this clinical scenario in 3-4 sentences:\n"
        f"{state['clinical_question']}"
    )
    return {"position_b": response.content}


def rebuttal_a(state: DebateState) -> dict:
    """Agent A responds to Agent B"""
    response = llm.invoke(
        f"You favor early treatment. Respond to this conservative argument with 2-3 sentences:\n"
        f"Conservative view: {state['position_b']}"
    )
    return {"rebuttal_a": response.content}


def rebuttal_b(state: DebateState) -> dict:
    """Agent B responds to Agent A"""
    response = llm.invoke(
        f"You favor conservative management. Respond to this aggressive treatment argument with 2-3 sentences:\n"
        f"Aggressive view: {state['position_a']}"
    )
    return {"rebuttal_b": response.content}


def consensus_agent(state: DebateState) -> dict:
    """Mediator agent synthesizes both views"""
    response = llm.invoke(
        f"You are a senior physician mediating a clinical discussion.\n\n"
        f"Question: {state['clinical_question']}\n"
        f"Pro-treatment view: {state['position_a']}\n"
        f"Conservative view: {state['position_b']}\n"
        f"Pro-treatment rebuttal: {state['rebuttal_a']}\n"
        f"Conservative rebuttal: {state['rebuttal_b']}\n\n"
        f"Synthesize both perspectives into a balanced recommendation. "
        f"Acknowledge merit in both views. Provide a clear recommendation. Educational purposes only."
    )
    return {"consensus": response.content}


def demo_agent_debate():
    """Two agents debate, mediator synthesizes"""
    print("\n" + "=" * 70)
    print("DEMO 3: AGENT DEBATE (Two Perspectives → Consensus)")
    print("=" * 70)
    print("""
💡 Two agents argue different clinical approaches.
   A mediator synthesizes into a balanced recommendation.
   This produces BETTER answers than a single agent.
""")

    graph = StateGraph(DebateState)
    graph.add_node("advocate", advocate_agent)
    graph.add_node("conservative", conservative_agent)
    graph.add_node("rebuttal_a", rebuttal_a)
    graph.add_node("rebuttal_b", rebuttal_b)
    graph.add_node("consensus", consensus_agent)

    graph.set_entry_point("advocate")
    graph.add_edge("advocate", "conservative")
    graph.add_edge("conservative", "rebuttal_a")
    graph.add_edge("rebuttal_a", "rebuttal_b")
    graph.add_edge("rebuttal_b", "consensus")
    graph.add_edge("consensus", END)

    app = graph.compile()

    question = ("A 45-year-old with mildly elevated blood pressure (138/88) and no other risk factors. "
                "Should we start antihypertensive medication immediately or try lifestyle changes first?")

    print(f"\n❓ Clinical Question:\n   {question}\n")

    result = app.invoke({"clinical_question": question})

    print(f"{'─' * 70}")
    print(f"💊 PRO-TREATMENT AGENT:\n{result['position_a']}\n")
    print(f"🧘 CONSERVATIVE AGENT:\n{result['position_b']}\n")
    print(f"💊 PRO-TREATMENT REBUTTAL:\n{result['rebuttal_a']}\n")
    print(f"🧘 CONSERVATIVE REBUTTAL:\n{result['rebuttal_b']}\n")
    print(f"{'─' * 70}")
    print(f"⚖️ CONSENSUS (Mediator):\n{result['consensus']}")


# ============================================================
# DEMO 4: Interactive Pipeline
# ============================================================

def demo_interactive():
    """Run your own cases through the multi-agent pipeline"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE MULTI-AGENT PIPELINE")
    print("=" * 70)

    # Build the pipeline
    graph = StateGraph(PipelineState)
    graph.add_node("triage", triage_agent)
    graph.add_node("diagnosis", diagnosis_agent)
    graph.add_node("treatment", treatment_agent)
    graph.add_node("safety", safety_agent)
    graph.add_node("report", compile_report)
    graph.set_entry_point("triage")
    graph.add_edge("triage", "diagnosis")
    graph.add_edge("diagnosis", "treatment")
    graph.add_edge("treatment", "safety")
    graph.add_edge("safety", "report")
    graph.add_edge("report", END)
    app = graph.compile()

    print("\n💬 Enter patient cases for the 4-agent pipeline. Type 'quit' to exit.\n")

    while True:
        case = input("Patient case: ").strip()
        if case.lower() in ['quit', 'exit', 'q']:
            break
        if not case:
            continue

        print("\n⏳ Running 4-agent pipeline (Triage → Diagnosis → Treatment → Safety)...\n")
        result = app.invoke({"patient_case": case})
        print(result["final_report"])
        print()


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n👥 Level 3, Project 4: Multi-Agent Systems")
    print("=" * 70)
    print("Multiple specialized agents collaborating\n")

    print("Choose a demo:")
    print("1. Sequential pipeline (Triage → Diagnosis → Treatment → Safety)")
    print("2. Router pattern (Supervisor routes to specialists)")
    print("3. Agent debate (two perspectives → consensus)")
    print("4. Interactive pipeline (your own cases)")
    print("5. Run demos 1-3")

    choice = input("\nEnter choice (1-5): ").strip()

    demos = {"1": demo_sequential_pipeline, "2": demo_router_pattern,
             "3": demo_agent_debate, "4": demo_interactive}

    if choice == "5":
        demo_sequential_pipeline()
        demo_router_pattern()
        demo_agent_debate()
    elif choice in demos:
        demos[choice]()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY TAKEAWAYS
{'=' * 70}

👥 MULTI-AGENT PATTERNS:

   1. SEQUENTIAL PIPELINE:
      [Agent A] → [Agent B] → [Agent C] → [Report]
      Each agent specializes in one task
      Best for: Structured workflows (triage → diagnosis → treatment)

   2. ROUTER (SUPERVISOR):
      [Router] → routes to → [Specialist A] or [Specialist B] or [Specialist C]
      One agent classifies, specialists handle
      Best for: Multi-specialty, diverse queries

   3. DEBATE:
      [Agent A] vs [Agent B] → [Mediator] → Consensus
      Multiple perspectives improve answer quality
      Best for: Complex decisions with tradeoffs

🔑 WHY MULTI-AGENT BEATS SINGLE-AGENT:
   • Each agent has a focused system prompt → better at its job
   • Separation of concerns → easier to debug and improve
   • Specialist knowledge → more accurate domain handling
   • Safety checks → separate agent catches errors
   • Scalable → add new specialists without rewriting

🎯 NEXT: Move to 05_healthcare_agent for the capstone —
   a complete clinical decision support system combining
   RAG + Agents + Safety guardrails!
""")


if __name__ == "__main__":
    main()
