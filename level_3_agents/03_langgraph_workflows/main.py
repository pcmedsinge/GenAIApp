"""
Project 3: LangGraph Stateful Workflows
Design agent workflows as GRAPHS — nodes (steps) connected by edges (flow).

Why LangGraph over plain LangChain agents?
  LangChain Agent: LLM decides the entire flow (flexible but unpredictable)
  LangGraph:       YOU define the workflow structure, LLM reasons at each node

  Think of it like a flowchart:
    [Intake] → [Classify Urgency] → [Emergency? YES → Fast Track / NO → Standard] → [Respond]

  This is HOW production healthcare AI systems are built.

Builds on: Project 01 (ReAct pattern) + Project 02 (LangChain tools)
"""

import os
import json
from typing import TypedDict, Literal, Annotated
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# DEMO 1: Simple Graph — Understanding Nodes and Edges
# ============================================================

# State: what data flows through the graph
class SimpleState(TypedDict):
    patient_complaint: str
    assessment: str
    recommendation: str


def assess_node(state: SimpleState) -> dict:
    """Node 1: LLM assesses the patient complaint"""
    response = llm.invoke(
        f"You are a triage nurse. Briefly assess this patient complaint in 2-3 sentences: {state['patient_complaint']}"
    )
    return {"assessment": response.content}


def recommend_node(state: SimpleState) -> dict:
    """Node 2: LLM provides recommendation based on assessment"""
    response = llm.invoke(
        f"Based on this assessment: {state['assessment']}\n\n"
        f"Original complaint: {state['patient_complaint']}\n\n"
        "Provide a brief clinical recommendation (2-3 sentences). Educational purposes only."
    )
    return {"recommendation": response.content}


def demo_simple_graph():
    """Build and run a simple 2-node graph"""
    print("\n" + "=" * 70)
    print("DEMO 1: SIMPLE GRAPH (Nodes + Edges)")
    print("=" * 70)
    print("""
💡 LangGraph basics:
   • STATE: Data that flows through the graph (TypedDict)
   • NODE:  A function that processes and updates state
   • EDGE:  Connection between nodes (defines flow)
""")

    # Build the graph
    graph = StateGraph(SimpleState)

    # Add nodes
    graph.add_node("assess", assess_node)
    graph.add_node("recommend", recommend_node)

    # Add edges (flow)
    graph.set_entry_point("assess")           # Start here
    graph.add_edge("assess", "recommend")     # assess → recommend
    graph.add_edge("recommend", END)          # recommend → done

    # Compile
    app = graph.compile()

    # Run
    complaint = "58-year-old male with chest pressure radiating to left arm, started 30 minutes ago, sweating"
    print(f"\n📝 Patient complaint: \"{complaint}\"\n")
    print("Running graph: [assess] → [recommend] → END\n")

    result = app.invoke({"patient_complaint": complaint})

    print(f"🏥 Assessment:\n   {result['assessment']}\n")
    print(f"📋 Recommendation:\n   {result['recommendation']}")

    print(f"""
{'─' * 70}
💡 WHAT HAPPENED:
   1. Graph started at 'assess' node
   2. assess_node() called LLM to assess the complaint → updated state
   3. Edge moved to 'recommend' node
   4. recommend_node() called LLM using the assessment → updated state
   5. Graph ended

   The STATE flowed through: patient_complaint → assessment → recommendation
   Each node READ from state and WROTE back to state.
""")


# ============================================================
# DEMO 2: Conditional Routing
# ============================================================

class TriageState(TypedDict):
    patient_complaint: str
    urgency: str          # emergency, urgent, routine
    assessment: str
    recommendation: str
    pathway: str          # which path was taken


def classify_urgency(state: TriageState) -> dict:
    """Classify the urgency of the complaint"""
    response = llm.invoke(
        f"""Classify this patient complaint's urgency level.
Respond with ONLY one word: emergency, urgent, or routine.

Complaint: {state['patient_complaint']}

Classification:"""
    )
    urgency = response.content.strip().lower()
    # Normalize
    if "emergency" in urgency:
        urgency = "emergency"
    elif "urgent" in urgency:
        urgency = "urgent"
    else:
        urgency = "routine"
    return {"urgency": urgency}


def emergency_path(state: TriageState) -> dict:
    """Handle emergency cases"""
    response = llm.invoke(
        f"EMERGENCY TRIAGE for: {state['patient_complaint']}\n"
        "Provide immediate actions needed. Be specific and urgent. 3-4 sentences."
    )
    return {
        "assessment": f"🚨 EMERGENCY: {response.content}",
        "pathway": "emergency_fast_track"
    }


def urgent_path(state: TriageState) -> dict:
    """Handle urgent cases"""
    response = llm.invoke(
        f"URGENT assessment for: {state['patient_complaint']}\n"
        "Provide assessment and recommended timeline. 3-4 sentences."
    )
    return {
        "assessment": f"⚠️ URGENT: {response.content}",
        "pathway": "urgent_same_day"
    }


def routine_path(state: TriageState) -> dict:
    """Handle routine cases"""
    response = llm.invoke(
        f"ROUTINE assessment for: {state['patient_complaint']}\n"
        "Provide assessment and self-care advice. 3-4 sentences."
    )
    return {
        "assessment": f"📋 ROUTINE: {response.content}",
        "pathway": "routine_scheduled"
    }


def final_recommendation(state: TriageState) -> dict:
    """Generate final recommendation based on pathway"""
    response = llm.invoke(
        f"Patient complaint: {state['patient_complaint']}\n"
        f"Urgency: {state['urgency']}\n"
        f"Assessment: {state['assessment']}\n"
        f"Pathway: {state['pathway']}\n\n"
        "Provide a final summary recommendation with next steps. Educational purposes only."
    )
    return {"recommendation": response.content}


def route_by_urgency(state: TriageState) -> str:
    """Conditional edge: route to different paths based on urgency"""
    return state["urgency"]


def demo_conditional_routing():
    """Graph with conditional routing based on urgency"""
    print("\n" + "=" * 70)
    print("DEMO 2: CONDITIONAL ROUTING")
    print("=" * 70)
    print("""
💡 Conditional edges let you create DIFFERENT paths through the graph.
   Like a flowchart: if emergency → fast track; if routine → scheduled
""")

    # Build the graph
    graph = StateGraph(TriageState)

    # Nodes
    graph.add_node("classify", classify_urgency)
    graph.add_node("emergency", emergency_path)
    graph.add_node("urgent", urgent_path)
    graph.add_node("routine", routine_path)
    graph.add_node("recommend", final_recommendation)

    # Entry
    graph.set_entry_point("classify")

    # Conditional routing after classification
    graph.add_conditional_edges(
        "classify",
        route_by_urgency,
        {
            "emergency": "emergency",
            "urgent": "urgent",
            "routine": "routine",
        }
    )

    # All paths converge to final recommendation
    graph.add_edge("emergency", "recommend")
    graph.add_edge("urgent", "recommend")
    graph.add_edge("routine", "recommend")
    graph.add_edge("recommend", END)

    app = graph.compile()

    # Test with different urgency levels
    test_cases = [
        "Sudden severe chest pain, difficulty breathing, arm numbness",
        "Persistent headache for 3 days with mild fever and neck stiffness",
        "Mild knee pain after jogging, no swelling, started yesterday",
    ]

    for complaint in test_cases:
        print(f"\n{'─' * 70}")
        print(f"📝 Complaint: \"{complaint}\"\n")

        result = app.invoke({"patient_complaint": complaint})

        print(f"   Urgency:  {result['urgency'].upper()}")
        print(f"   Pathway:  {result['pathway']}")
        print(f"   Assessment: {result['assessment'][:200]}...")
        print(f"   Recommendation: {result['recommendation'][:200]}...")

    print(f"""
{'─' * 70}
💡 CONDITIONAL ROUTING:
   • classify node determines urgency
   • add_conditional_edges() routes to the right path
   • Each path handles the case differently
   • All paths converge at the final recommendation

   This is how real clinical triage systems work:
   Classify → Route → Handle → Respond
""")


# ============================================================
# DEMO 3: Clinical Triage Workflow (Full Pipeline)
# ============================================================

class ClinicalState(TypedDict):
    patient_info: str
    symptoms: str
    extracted_data: str
    risk_factors: str
    urgency: str
    clinical_assessment: str
    plan: str


def extract_clinical_data(state: ClinicalState) -> dict:
    """Extract structured data from free-text patient info"""
    response = llm.invoke(
        f"Extract key clinical data from this patient presentation. "
        f"Include: age, sex, vital signs (if mentioned), chief complaint, duration, "
        f"relevant history. Be structured.\n\n"
        f"Patient info: {state['patient_info']}\n"
        f"Symptoms: {state['symptoms']}"
    )
    return {"extracted_data": response.content}


def assess_risk_factors(state: ClinicalState) -> dict:
    """Identify risk factors"""
    response = llm.invoke(
        f"Based on this clinical data, identify key risk factors and red flags:\n"
        f"{state['extracted_data']}\n\n"
        f"List risk factors with severity. Be concise."
    )
    return {"risk_factors": response.content}


def classify_clinical_urgency(state: ClinicalState) -> dict:
    """Classify urgency based on full clinical picture"""
    response = llm.invoke(
        f"Classify urgency (emergency/urgent/routine) based on:\n"
        f"Clinical data: {state['extracted_data']}\n"
        f"Risk factors: {state['risk_factors']}\n\n"
        f"Respond ONLY: emergency, urgent, or routine"
    )
    urgency = response.content.strip().lower()
    if "emergency" in urgency:
        return {"urgency": "emergency"}
    elif "urgent" in urgency:
        return {"urgency": "urgent"}
    return {"urgency": "routine"}


def generate_clinical_assessment(state: ClinicalState) -> dict:
    """Generate structured clinical assessment"""
    response = llm.invoke(
        f"Generate a clinical assessment for:\n"
        f"Data: {state['extracted_data']}\n"
        f"Risk factors: {state['risk_factors']}\n"
        f"Urgency: {state['urgency']}\n\n"
        f"Include: differential diagnosis (top 3), recommended workup, "
        f"and immediate actions if any. Educational purposes only."
    )
    return {"clinical_assessment": response.content}


def generate_plan(state: ClinicalState) -> dict:
    """Generate treatment plan"""
    response = llm.invoke(
        f"Generate a brief management plan based on:\n"
        f"Assessment: {state['clinical_assessment']}\n"
        f"Urgency: {state['urgency']}\n\n"
        f"Include: immediate actions, medications if needed, follow-up timeline, "
        f"patient education points. Educational purposes only."
    )
    return {"plan": response.content}


def demo_clinical_workflow():
    """Full clinical triage workflow"""
    print("\n" + "=" * 70)
    print("DEMO 3: CLINICAL TRIAGE WORKFLOW")
    print("=" * 70)
    print("""
💡 A multi-step clinical workflow:
   [Extract Data] → [Assess Risk] → [Classify Urgency] → [Assessment] → [Plan]
""")

    graph = StateGraph(ClinicalState)

    graph.add_node("extract", extract_clinical_data)
    graph.add_node("risk", assess_risk_factors)
    graph.add_node("classify", classify_clinical_urgency)
    graph.add_node("assess", generate_clinical_assessment)
    graph.add_node("plan", generate_plan)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "risk")
    graph.add_edge("risk", "classify")
    graph.add_edge("classify", "assess")
    graph.add_edge("assess", "plan")
    graph.add_edge("plan", END)

    app = graph.compile()

    # Test case
    result = app.invoke({
        "patient_info": "62-year-old male with history of hypertension and type 2 diabetes. BMI 32. Current meds: metformin 1000mg BID, lisinopril 20mg daily.",
        "symptoms": "Progressive shortness of breath over 2 weeks, worse with exertion. New bilateral ankle swelling. Wakes up at night needing extra pillows to breathe. Weight gain of 8 pounds in 2 weeks."
    })

    print(f"\n📊 RESULTS:\n")
    print(f"📋 Extracted Data:\n{result['extracted_data']}\n")
    print(f"⚠️ Risk Factors:\n{result['risk_factors']}\n")
    print(f"🚦 Urgency: {result['urgency'].upper()}\n")
    print(f"🩺 Assessment:\n{result['clinical_assessment']}\n")
    print(f"📝 Plan:\n{result['plan']}")


# ============================================================
# DEMO 4: Interactive Triage
# ============================================================

def demo_interactive():
    """Interactive triage with your own cases"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE TRIAGE")
    print("=" * 70)

    graph = StateGraph(ClinicalState)
    graph.add_node("extract", extract_clinical_data)
    graph.add_node("risk", assess_risk_factors)
    graph.add_node("classify", classify_clinical_urgency)
    graph.add_node("assess", generate_clinical_assessment)
    graph.add_node("plan", generate_plan)
    graph.set_entry_point("extract")
    graph.add_edge("extract", "risk")
    graph.add_edge("risk", "classify")
    graph.add_edge("classify", "assess")
    graph.add_edge("assess", "plan")
    graph.add_edge("plan", END)
    app = graph.compile()

    print("\n💬 Enter patient cases for triage. Type 'quit' to exit.\n")

    while True:
        info = input("Patient info (age, history, meds): ").strip()
        if info.lower() in ['quit', 'exit', 'q']:
            break
        if not info:
            continue
        symptoms = input("Current symptoms: ").strip()
        if not symptoms:
            continue

        print("\n⏳ Running triage workflow...\n")
        result = app.invoke({"patient_info": info, "symptoms": symptoms})

        print(f"🚦 Urgency: {result['urgency'].upper()}")
        print(f"\n🩺 Assessment:\n{result['clinical_assessment'][:500]}")
        print(f"\n📝 Plan:\n{result['plan'][:500]}\n")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n📊 Level 3, Project 3: LangGraph Workflows")
    print("=" * 70)
    print("Design agent workflows as graphs — the production standard\n")

    print("Choose a demo:")
    print("1. Simple graph (nodes + edges basics)")
    print("2. Conditional routing (different paths by urgency)")
    print("3. Clinical triage workflow (full 5-step pipeline)")
    print("4. Interactive triage (enter your own cases)")
    print("5. Run demos 1-3")

    choice = input("\nEnter choice (1-5): ").strip()

    demos = {"1": demo_simple_graph, "2": demo_conditional_routing,
             "3": demo_clinical_workflow, "4": demo_interactive}

    if choice == "5":
        demo_simple_graph()
        demo_conditional_routing()
        demo_clinical_workflow()
    elif choice in demos:
        demos[choice]()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY TAKEAWAYS
{'=' * 70}

📊 LANGGRAPH CONCEPTS:
   • STATE (TypedDict): Data flowing through the graph
   • NODE (function):   Processes state, returns updates
   • EDGE:              Connects nodes (defines flow)
   • CONDITIONAL EDGE:  Routes based on state values

🏥 WHY LANGGRAPH FOR HEALTHCARE:
   • Predictable workflows (not random LLM decisions)
   • State tracking (patient data accumulates across steps)
   • Conditional routing (emergency vs routine paths)
   • Auditable (you can trace exactly what happened at each step)

📊 LANGGRAPH vs LANGCHAIN AGENTS:
   LangChain Agent: LLM decides the entire flow → flexible, less predictable
   LangGraph:       YOU design the flow → structured, reliable, production-ready

🔑 WHEN TO USE WHAT:
   Simple tool use → LangChain Agent (Project 02)
   Complex workflows → LangGraph (this project)
   Multi-agent → Next project (04_multi_agent)!

🎯 NEXT: Move to 04_multi_agent for systems where
   MULTIPLE agents collaborate on complex tasks!
""")


if __name__ == "__main__":
    main()
