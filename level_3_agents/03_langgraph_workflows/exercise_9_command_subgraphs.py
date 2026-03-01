"""
Exercise 9: LangGraph Command + Subgraphs

Skills practiced:
- Using Command() for in-node graph navigation and state updates
- Building subgraphs (nested StateGraphs) for modular workflows
- Composing subgraphs into parent graphs
- Using Command(goto=..., update=...) for flexible control flow
- Healthcare use case: modular clinical pathways

Why this matters:
  Command() lets a NODE control where the graph goes next AND update
  state in a single return. Instead of relying solely on conditional edges,
  the node itself decides routing. This is powerful for dynamic workflows.

  Subgraphs let you compose independent workflows like modules:
  build them separately, test them separately, combine them.

Architecture:

  Command (in-node navigation):
  ┌────────────────────────┐
  │  node_function():      │
  │    if condition_a:     │
  │      return Command(   │
  │        goto="node_b",  │  ← Node decides routing
  │        update={...}    │  ← AND updates state
  │      )                 │
  │    else:               │
  │      return Command(   │
  │        goto="node_c",  │
  │        update={...}    │
  │      )                 │
  └────────────────────────┘

  Subgraphs (nested graphs):
  ┌─── Parent Graph ──────────────────────────┐
  │                                            │
  │  ┌──────────┐    ┌──────────────────────┐ │
  │  │  triage   │──▶│  treatment_subgraph   │ │
  │  └──────────┘    │  ┌──────┐  ┌──────┐  │ │
  │                   │  │ eval │─▶│ plan │  │ │
  │                   │  └──────┘  └──────┘  │ │
  │                   └──────────────────────┘ │
  │                            │               │
  │                   ┌──────────────────────┐ │
  │                   │  followup_subgraph    │ │
  │                   │  ┌──────┐  ┌──────┐  │ │
  │                   │  │sched │─▶│notify│  │ │
  │                   │  └──────┘  └──────┘  │ │
  │                   └──────────────────────┘ │
  └────────────────────────────────────────────┘

Healthcare parallel:
  Clinical pathways are modular: ED triage → cardiac pathway OR
  sepsis pathway OR trauma pathway. Each is a self-contained
  subgraph that can be developed and tested independently.
"""

import os
import json
from typing import Annotated, TypedDict
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# DEMO 1: Basic Command — In-Node Routing
# ============================================================

def demo_basic_command():
    """Show Command() for node-level routing decisions."""
    print("\n" + "=" * 70)
    print("  DEMO 1: BASIC COMMAND — IN-NODE ROUTING")
    print("=" * 70)
    print("""
  Command() lets a node return BOTH a state update AND a routing decision.

  Without Command: Node returns state → conditional_edges decides routing
  With Command:    Node returns Command(goto="next", update={state})
                   The node itself decides where to go.

  This is simpler when the routing logic is tightly coupled to the node.
  """)

    class PatientState(TypedDict):
        symptoms: str
        severity: str
        pathway: str
        assessment: str

    def triage(state: PatientState) -> Command:
        """Triage node that decides the pathway using Command."""
        symptoms = state["symptoms"].lower()

        # Determine severity and route
        if any(w in symptoms for w in ["chest pain", "stroke", "unresponsive"]):
            print("    🚨 Triage: CRITICAL → emergency pathway")
            return Command(
                goto="emergency",
                update={"severity": "CRITICAL", "pathway": "emergency"},
            )
        elif any(w in symptoms for w in ["fever", "pain", "swelling"]):
            print("    ⚠️  Triage: MODERATE → urgent care pathway")
            return Command(
                goto="urgent_care",
                update={"severity": "MODERATE", "pathway": "urgent_care"},
            )
        else:
            print("    ✅ Triage: LOW → routine care pathway")
            return Command(
                goto="routine",
                update={"severity": "LOW", "pathway": "routine"},
            )

    def emergency(state: PatientState) -> Command:
        """Emergency pathway."""
        assessment = (
            f"EMERGENCY: {state['symptoms']} — Immediate intervention required. "
            f"Activate code team. Continuous monitoring."
        )
        print(f"    🏥 Emergency: {assessment[:60]}...")
        return Command(goto=END, update={"assessment": assessment})

    def urgent_care(state: PatientState) -> Command:
        """Urgent care pathway."""
        assessment = (
            f"URGENT CARE: {state['symptoms']} — Priority evaluation within 1hr. "
            f"Labs and imaging ordered. Pain management initiated."
        )
        print(f"    🏥 Urgent: {assessment[:60]}...")
        return Command(goto=END, update={"assessment": assessment})

    def routine(state: PatientState) -> Command:
        """Routine care pathway."""
        assessment = (
            f"ROUTINE: {state['symptoms']} — Standard evaluation. "
            f"Scheduled for next available appointment."
        )
        print(f"    🏥 Routine: {assessment[:60]}...")
        return Command(goto=END, update={"assessment": assessment})

    # Build graph — no conditional_edges needed!
    graph = StateGraph(PatientState)
    graph.add_node("triage", triage)
    graph.add_node("emergency", emergency)
    graph.add_node("urgent_care", urgent_care)
    graph.add_node("routine", routine)

    graph.set_entry_point("triage")
    # No add_conditional_edges — Command handles routing!
    # But we do need to tell LangGraph what edges are possible:
    graph.add_edge("triage", "emergency")
    graph.add_edge("triage", "urgent_care")
    graph.add_edge("triage", "routine")

    app = graph.compile()

    # Test different patients
    patients = [
        "Severe chest pain radiating to left arm",
        "Fever of 102F with swelling in right knee",
        "Mild headache for 2 days, no other symptoms",
    ]

    for symptoms in patients:
        print(f"\n  ─── Patient: {symptoms} ───")
        result = app.invoke({
            "symptoms": symptoms,
            "severity": "",
            "pathway": "",
            "assessment": "",
        })
        print(f"  Severity: {result['severity']}, Pathway: {result['pathway']}")

    print(f"\n  KEY INSIGHT: Command(goto=..., update=...) combines")
    print(f"  routing + state update in one return. No conditional_edges needed.")


# ============================================================
# DEMO 2: Command with Multi-Step Routing
# ============================================================

def demo_command_multistep():
    """Show Command for dynamic multi-step workflow."""
    print("\n" + "=" * 70)
    print("  DEMO 2: COMMAND — DYNAMIC MULTI-STEP WORKFLOW")
    print("=" * 70)
    print("""
  Command enables dynamic routing through a sequence of steps.
  Each node decides the NEXT step based on current state.

  This is like a clinical decision tree where each finding
  determines the next test or action.
  """)

    class WorkupState(TypedDict):
        patient: str
        current_step: str
        findings: list[str]
        final_diagnosis: str

    def initial_assessment(state: WorkupState) -> Command:
        """First node decides what to evaluate next."""
        patient = state["patient"].lower()
        findings = [f"Initial: {state['patient']}"]

        if "chest pain" in patient:
            print("    🔍 Initial → ordering cardiac workup")
            return Command(
                goto="cardiac_workup",
                update={"findings": findings, "current_step": "cardiac_workup"},
            )
        elif "cough" in patient or "dyspnea" in patient:
            print("    🔍 Initial → ordering pulmonary workup")
            return Command(
                goto="pulmonary_workup",
                update={"findings": findings, "current_step": "pulmonary_workup"},
            )
        else:
            print("    🔍 Initial → general workup")
            return Command(
                goto="general_workup",
                update={"findings": findings, "current_step": "general_workup"},
            )

    def cardiac_workup(state: WorkupState) -> Command:
        """Cardiac workup — decides next step based on 'results'."""
        findings = state["findings"] + ["Cardiac: ECG obtained, troponin sent"]
        patient = state["patient"].lower()

        if "elevated troponin" in patient or "st elevation" in patient:
            print("    🫀 Cardiac → troponin positive → cath lab")
            return Command(
                goto="diagnose",
                update={
                    "findings": findings + ["Positive troponin → Acute MI"],
                    "current_step": "diagnose",
                },
            )
        else:
            print("    🫀 Cardiac → troponin negative → observation")
            return Command(
                goto="diagnose",
                update={
                    "findings": findings + ["Troponin negative → Low risk ACS"],
                    "current_step": "diagnose",
                },
            )

    def pulmonary_workup(state: WorkupState) -> Command:
        """Pulmonary workup."""
        findings = state["findings"] + ["Pulmonary: CXR obtained, O2 sat checked"]
        print("    🫁 Pulmonary workup complete → diagnosing")
        return Command(
            goto="diagnose",
            update={
                "findings": findings + ["CXR: possible infiltrate"],
                "current_step": "diagnose",
            },
        )

    def general_workup(state: WorkupState) -> Command:
        """General workup."""
        findings = state["findings"] + ["General: CBC, BMP obtained"]
        print("    🩺 General workup complete → diagnosing")
        return Command(
            goto="diagnose",
            update={
                "findings": findings + ["Labs: within normal limits"],
                "current_step": "diagnose",
            },
        )

    def diagnose(state: WorkupState) -> Command:
        """Synthesize findings into diagnosis."""
        all_findings = "\n  ".join(state["findings"])
        diagnosis = f"Based on workup:\n  {all_findings}\n  → Assessment complete."
        print(f"    📋 Diagnosis formulated")
        return Command(
            goto=END,
            update={"final_diagnosis": diagnosis},
        )

    # Build graph
    graph = StateGraph(WorkupState)
    graph.add_node("initial_assessment", initial_assessment)
    graph.add_node("cardiac_workup", cardiac_workup)
    graph.add_node("pulmonary_workup", pulmonary_workup)
    graph.add_node("general_workup", general_workup)
    graph.add_node("diagnose", diagnose)

    graph.set_entry_point("initial_assessment")
    # Declare all possible edges (Command handles actual routing)
    for src in ["initial_assessment"]:
        for dst in ["cardiac_workup", "pulmonary_workup", "general_workup"]:
            graph.add_edge(src, dst)
    for src in ["cardiac_workup", "pulmonary_workup", "general_workup"]:
        graph.add_edge(src, "diagnose")

    app = graph.compile()

    # Test
    patients = [
        "65M with chest pain and elevated troponin",
        "45F with cough and dyspnea for 3 days",
        "30M with headache, no other complaints",
    ]

    for patient in patients:
        print(f"\n  ─── {patient} ───")
        result = app.invoke({
            "patient": patient,
            "current_step": "initial",
            "findings": [],
            "final_diagnosis": "",
        })
        print(f"  Diagnosis:\n    {result['final_diagnosis'][:200]}")

    print(f"\n  KEY INSIGHT: Command lets each node dynamically choose the next step.")
    print(f"  No need for separate conditional_edges — routing is inline.")


# ============================================================
# DEMO 3: Subgraphs — Modular Clinical Pathways
# ============================================================

def demo_subgraphs():
    """Show subgraphs for modular, composable workflows."""
    print("\n" + "=" * 70)
    print("  DEMO 3: SUBGRAPHS — MODULAR CLINICAL PATHWAYS")
    print("=" * 70)
    print("""
  Subgraphs = nested StateGraphs inside a parent graph.
  Each subgraph is:
  - Independently developed and tested
  - Compiled into a node of the parent graph
  - Has its own internal state and logic

  Like clinical pathways: each is a self-contained protocol
  that the main workflow can invoke.
  """)

    # === Cardiac Subgraph ===
    class CardiacState(TypedDict):
        patient_info: str
        cardiac_assessment: str

    def cardiac_eval(state: CardiacState) -> dict:
        """Evaluate cardiac symptoms."""
        print("    🫀 [Cardiac sub] Evaluating...")
        return {"cardiac_assessment": f"ECG: normal sinus. Troponin: pending."}

    def cardiac_plan(state: CardiacState) -> dict:
        """Create cardiac plan."""
        print("    🫀 [Cardiac sub] Planning...")
        assessment = state["cardiac_assessment"]
        plan = f"Cardiac plan — {assessment} → Serial troponins q6h, monitor telemetry."
        return {"cardiac_assessment": plan}

    cardiac_graph = StateGraph(CardiacState)
    cardiac_graph.add_node("eval", cardiac_eval)
    cardiac_graph.add_node("plan", cardiac_plan)
    cardiac_graph.set_entry_point("eval")
    cardiac_graph.add_edge("eval", "plan")
    cardiac_graph.add_edge("plan", END)
    cardiac_subgraph = cardiac_graph.compile()

    # === Sepsis Subgraph ===
    class SepsisState(TypedDict):
        patient_info: str
        sepsis_assessment: str

    def sepsis_screen(state: SepsisState) -> dict:
        """Screen for sepsis criteria."""
        print("    🦠 [Sepsis sub] Screening...")
        return {"sepsis_assessment": "SIRS criteria: 3/4 met. Lactate: 3.2"}

    def sepsis_bundle(state: SepsisState) -> dict:
        """Execute sepsis bundle."""
        print("    🦠 [Sepsis sub] Executing bundle...")
        assessment = state["sepsis_assessment"]
        bundle = f"Sepsis bundle — {assessment} → Blood cultures ×2, broad-spectrum abx, 30mL/kg fluids."
        return {"sepsis_assessment": bundle}

    sepsis_graph = StateGraph(SepsisState)
    sepsis_graph.add_node("screen", sepsis_screen)
    sepsis_graph.add_node("bundle", sepsis_bundle)
    sepsis_graph.set_entry_point("screen")
    sepsis_graph.add_edge("screen", "bundle")
    sepsis_graph.add_edge("bundle", END)
    sepsis_subgraph = sepsis_graph.compile()

    # === Parent Graph ===
    class EDState(TypedDict):
        patient_info: str
        pathway: str
        cardiac_assessment: str
        sepsis_assessment: str
        disposition: str

    def ed_triage(state: EDState) -> dict:
        """ED triage determines pathway."""
        info = state["patient_info"].lower()
        if "chest pain" in info:
            pathway = "cardiac"
        elif "fever" in info or "infection" in info:
            pathway = "sepsis"
        else:
            pathway = "general"
        print(f"    🏥 ED Triage → {pathway} pathway")
        return {"pathway": pathway}

    def route_pathway(state: EDState) -> str:
        """Route to appropriate subgraph."""
        if state["pathway"] == "cardiac":
            return "cardiac_pathway"
        elif state["pathway"] == "sepsis":
            return "sepsis_pathway"
        return "general_assessment"

    def cardiac_pathway(state: EDState) -> dict:
        """Run cardiac subgraph."""
        result = cardiac_subgraph.invoke({
            "patient_info": state["patient_info"],
            "cardiac_assessment": "",
        })
        return {"cardiac_assessment": result["cardiac_assessment"]}

    def sepsis_pathway(state: EDState) -> dict:
        """Run sepsis subgraph."""
        result = sepsis_subgraph.invoke({
            "patient_info": state["patient_info"],
            "sepsis_assessment": "",
        })
        return {"sepsis_assessment": result["sepsis_assessment"]}

    def general_assessment(state: EDState) -> dict:
        """General assessment (no subgraph)."""
        print("    🩺 General assessment")
        return {"disposition": "General eval complete. Routine workup."}

    def disposition(state: EDState) -> dict:
        """Final disposition decision."""
        parts = []
        if state.get("cardiac_assessment"):
            parts.append(f"Cardiac: {state['cardiac_assessment']}")
        if state.get("sepsis_assessment"):
            parts.append(f"Sepsis: {state['sepsis_assessment']}")
        if not parts:
            parts.append("General evaluation completed")

        disp = " | ".join(parts) + " → Admit for monitoring."
        return {"disposition": disp}

    # Build parent graph
    parent = StateGraph(EDState)
    parent.add_node("ed_triage", ed_triage)
    parent.add_node("cardiac_pathway", cardiac_pathway)
    parent.add_node("sepsis_pathway", sepsis_pathway)
    parent.add_node("general_assessment", general_assessment)
    parent.add_node("disposition", disposition)

    parent.set_entry_point("ed_triage")
    parent.add_conditional_edges("ed_triage", route_pathway, {
        "cardiac_pathway": "cardiac_pathway",
        "sepsis_pathway": "sepsis_pathway",
        "general_assessment": "general_assessment",
    })
    parent.add_edge("cardiac_pathway", "disposition")
    parent.add_edge("sepsis_pathway", "disposition")
    parent.add_edge("general_assessment", "disposition")
    parent.add_edge("disposition", END)

    app = parent.compile()

    # Test
    patients = [
        "54M with acute chest pain, diaphoresis",
        "68F with fever 103F, suspected UTI, confusion",
        "25M with ankle sprain",
    ]

    for patient in patients:
        print(f"\n  ─── {patient} ───")
        result = app.invoke({
            "patient_info": patient,
            "pathway": "",
            "cardiac_assessment": "",
            "sepsis_assessment": "",
            "disposition": "",
        })
        print(f"  Pathway: {result['pathway']}")
        print(f"  Disposition: {result['disposition'][:150]}")

    print(f"\n  KEY INSIGHT: Subgraphs are independently compiled StateGraphs")
    print(f"  invoked as nodes. Build, test, and compose them like modules.")


# ============================================================
# DEMO 4: Command + Subgraph — Dynamic Pathway Selection
# ============================================================

def demo_command_plus_subgraph():
    """Combine Command routing with subgraph execution."""
    print("\n" + "=" * 70)
    print("  DEMO 4: COMMAND + SUBGRAPH — DYNAMIC PATHWAY WITH LLM")
    print("=" * 70)
    print("""
  Combine Command() for dynamic routing with subgraphs for modularity.
  An LLM-powered triage node uses Command to route to the right subgraph.
  """)

    # Simple assessment subgraph builder
    def make_assessment_subgraph(specialty: str, system_prompt: str):
        class AssessmentState(TypedDict):
            input_text: str
            assessment: str

        def evaluate(state: AssessmentState) -> dict:
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user",
                 "content": f"Assess this patient in 2-3 sentences: {state['input_text']}"},
            ])
            return {"assessment": f"[{specialty.upper()}] {response.content}"}

        g = StateGraph(AssessmentState)
        g.add_node("evaluate", evaluate)
        g.set_entry_point("evaluate")
        g.add_edge("evaluate", END)
        return g.compile()

    # Build specialty subgraphs
    cardio_sub = make_assessment_subgraph(
        "Cardiology",
        "You are a cardiologist. Give a brief cardiac assessment."
    )
    neuro_sub = make_assessment_subgraph(
        "Neurology",
        "You are a neurologist. Give a brief neurological assessment."
    )
    general_sub = make_assessment_subgraph(
        "General Medicine",
        "You are an internist. Give a brief general assessment."
    )

    # Parent graph with Command routing
    class ClinicState(TypedDict):
        complaint: str
        specialty: str
        assessment: str

    def smart_triage(state: ClinicState) -> Command:
        """LLM-powered triage using Command for routing."""
        response = llm.invoke([
            {"role": "system",
             "content": "Classify this complaint into exactly one category: "
                        "'cardiology', 'neurology', or 'general'. "
                        "Respond with ONLY the category name."},
            {"role": "user", "content": state["complaint"]},
        ])
        specialty = response.content.strip().lower()

        # Map to node
        node_map = {
            "cardiology": "cardio_eval",
            "neurology": "neuro_eval",
            "general": "general_eval",
        }
        target = node_map.get(specialty, "general_eval")
        print(f"    🤖 LLM triage → {specialty} → {target}")

        return Command(
            goto=target,
            update={"specialty": specialty},
        )

    def cardio_eval(state: ClinicState) -> dict:
        result = cardio_sub.invoke({"input_text": state["complaint"], "assessment": ""})
        return {"assessment": result["assessment"]}

    def neuro_eval(state: ClinicState) -> dict:
        result = neuro_sub.invoke({"input_text": state["complaint"], "assessment": ""})
        return {"assessment": result["assessment"]}

    def general_eval(state: ClinicState) -> dict:
        result = general_sub.invoke({"input_text": state["complaint"], "assessment": ""})
        return {"assessment": result["assessment"]}

    graph = StateGraph(ClinicState)
    graph.add_node("smart_triage", smart_triage)
    graph.add_node("cardio_eval", cardio_eval)
    graph.add_node("neuro_eval", neuro_eval)
    graph.add_node("general_eval", general_eval)

    graph.set_entry_point("smart_triage")
    # Declare possible edges (Command handles actual routing)
    for node in ["cardio_eval", "neuro_eval", "general_eval"]:
        graph.add_edge("smart_triage", node)
        graph.add_edge(node, END)

    app = graph.compile()

    complaints = [
        "Crushing chest pain radiating to jaw, onset 20 minutes ago",
        "Sudden severe headache, worst of life, neck stiffness",
        "Fatigue and weight loss over the past 3 months",
    ]

    for complaint in complaints:
        print(f"\n  ─── {complaint[:55]}... ───")
        result = app.invoke({
            "complaint": complaint,
            "specialty": "",
            "assessment": "",
        })
        print(f"  Specialty: {result['specialty']}")
        print(f"  Assessment: {result['assessment'][:200]}...")

    print(f"\n  KEY INSIGHT: Command + Subgraphs = LLM decides the route,")
    print(f"  subgraphs encapsulate the specialty logic. Best of both worlds.")


# ============================================================
# DEMO 5: Subgraph Reuse — Same Subgraph, Different Contexts
# ============================================================

def demo_subgraph_reuse():
    """Show reusing the same subgraph in different contexts."""
    print("\n" + "=" * 70)
    print("  DEMO 5: SUBGRAPH REUSE — ONE SUBGRAPH, MULTIPLE USES")
    print("=" * 70)
    print("""
  A key benefit of subgraphs: write once, use in multiple parent graphs.

  Here we build a "documentation" subgraph and reuse it in:
  1. An admission workflow
  2. A discharge workflow
  Both use the SAME documentation subgraph but with different context.
  """)

    # Reusable documentation subgraph
    class DocState(TypedDict):
        context: str
        note_type: str
        note: str

    def generate_note(state: DocState) -> dict:
        response = llm.invoke([
            {"role": "system",
             "content": f"Generate a brief {state['note_type']} note (3-4 lines)."},
            {"role": "user", "content": f"Context: {state['context']}"},
        ])
        return {"note": response.content}

    def validate_note(state: DocState) -> dict:
        note = state["note"]
        issues = []
        if len(note) < 20:
            issues.append("too short")
        if not any(w in note.lower() for w in ["patient", "assessment", "plan", "history", "diagnosis"]):
            issues.append("missing key sections")
        validation = "PASSED" if not issues else f"ISSUES: {', '.join(issues)}"
        return {"note": f"{note}\n[Validation: {validation}]"}

    doc_graph = StateGraph(DocState)
    doc_graph.add_node("generate", generate_note)
    doc_graph.add_node("validate", validate_note)
    doc_graph.set_entry_point("generate")
    doc_graph.add_edge("generate", "validate")
    doc_graph.add_edge("validate", END)
    doc_subgraph = doc_graph.compile()

    # === Admission workflow (uses doc_subgraph) ===
    class AdmissionState(TypedDict):
        patient: str
        admission_note: str

    def admit_patient(state: AdmissionState) -> dict:
        print(f"    📥 Admitting patient...")
        result = doc_subgraph.invoke({
            "context": f"Admission for: {state['patient']}",
            "note_type": "admission",
            "note": "",
        })
        return {"admission_note": result["note"]}

    admission_graph = StateGraph(AdmissionState)
    admission_graph.add_node("admit", admit_patient)
    admission_graph.set_entry_point("admit")
    admission_graph.add_edge("admit", END)
    admission_app = admission_graph.compile()

    # === Discharge workflow (uses SAME doc_subgraph) ===
    class DischargeState(TypedDict):
        patient: str
        discharge_note: str

    def discharge_patient(state: DischargeState) -> dict:
        print(f"    📤 Discharging patient...")
        result = doc_subgraph.invoke({
            "context": f"Discharge for: {state['patient']}",
            "note_type": "discharge",
            "note": "",
        })
        return {"discharge_note": result["note"]}

    discharge_graph = StateGraph(DischargeState)
    discharge_graph.add_node("discharge", discharge_patient)
    discharge_graph.set_entry_point("discharge")
    discharge_graph.add_edge("discharge", END)
    discharge_app = discharge_graph.compile()

    # Test both workflows with same documentation subgraph
    patient = "72M with pneumonia, improving on antibiotics"

    print(f"\n  Patient: {patient}")

    print(f"\n  ─── Admission Workflow ───")
    admit_result = admission_app.invoke({"patient": patient, "admission_note": ""})
    print(f"  Note:\n    {admit_result['admission_note'][:300]}")

    print(f"\n  ─── Discharge Workflow ───")
    dc_result = discharge_app.invoke({"patient": patient, "discharge_note": ""})
    print(f"  Note:\n    {dc_result['discharge_note'][:300]}")

    print(f"\n  KEY INSIGHT: The SAME doc_subgraph is reused in both admission")
    print(f"  and discharge workflows. Write once, compose anywhere.")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 9: LANGGRAPH COMMAND + SUBGRAPHS")
    print("=" * 70)
    print("""
    Two powerful LangGraph features for complex workflows:

    Command() — In-node routing + state updates
    Subgraphs — Nested, modular, reusable StateGraphs

    Choose a demo:
      1 → Basic Command (in-node routing)
      2 → Command multi-step workflow (clinical decision tree)
      3 → Subgraphs (modular clinical pathways)
      4 → Command + Subgraph (LLM triage → specialist subgraph)
      5 → Subgraph reuse (same subgraph, multiple workflows)
      6 → Run all demos
    """)

    choice = input("  Enter choice (1-6): ").strip()

    demos = {
        "1": demo_basic_command,
        "2": demo_command_multistep,
        "3": demo_subgraphs,
        "4": demo_command_plus_subgraph,
        "5": demo_subgraph_reuse,
    }

    if choice == "6":
        for d in demos.values():
            d()
    elif choice in demos:
        demos[choice]()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. Command() = IN-NODE ROUTING + STATE UPDATE
   Instead of: return {"state": ...}  (then conditional_edges routes)
   Use:        return Command(goto="next_node", update={"state": ...})
   The node itself decides where to go. Cleaner when routing is node-specific.

2. Command REPLACES conditional_edges for dynamic routing:
   - No separate routing function needed
   - Node logic and routing are co-located
   - Easier to read and maintain

3. SUBGRAPHS = MODULAR WORKFLOWS:
   - Each subgraph is an independent StateGraph, compiled separately
   - Invoked as a function: subgraph.invoke({state})
   - Can be tested independently
   - Can be reused in multiple parent graphs

4. SUBGRAPH COMPOSITION:
   - Build specialty workflows as subgraphs (cardiac, sepsis, neuro)
   - Parent graph handles routing/orchestration
   - Each subgraph owns its internal logic
   - Like microservices for workflows

5. COMMAND + SUBGRAPH PATTERN:
   - Command handles dynamic routing to the right subgraph
   - Subgraph encapsulates the domain logic
   - LLM can power the routing decision
   - Best for: multi-pathway clinical workflows

6. WHEN TO USE:
   - Command: When the node knows where to go (routing + state in one place)
   - Subgraphs: When workflows are modular and reusable
   - Both: Complex dynamic workflows with reusable components
   - Neither: Simple linear pipelines (just use add_edge)
"""

if __name__ == "__main__":
    main()
