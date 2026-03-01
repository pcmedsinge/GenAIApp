"""
Exercise 6: LangGraph Native Checkpointing — Time-Travel, Interrupt & Resume

Skills practiced:
- Using LangGraph's built-in Checkpointer (MemorySaver, SqliteSaver)
- Thread-based state management (thread_id config pattern)
- interrupt_before / interrupt_after for human-in-the-loop
- Resuming workflows from any checkpoint
- Time-travel debugging: replay from past checkpoints
- Updating state mid-workflow (state injection)
- Comparing MemorySaver (dev) vs SqliteSaver (production)
- Checkpoint diffing: seeing how state changed between steps

Why this matters:
  Exercise 4 built CUSTOM JSON persistence. That works, but LangGraph
  has a NATIVE checkpointing system that is:
  - Automatic (every node saves a checkpoint — no manual save calls)
  - Thread-aware (multiple users/sessions handled by thread_id)
  - Time-travel capable (replay from any past checkpoint)
  - Interrupt-aware (pause before/after any node, resume later)
  - Pluggable (swap MemorySaver → SqliteSaver → PostgresSaver)

  This is the PRODUCTION way to handle persistence in LangGraph.

Architecture:

  ┌─────────────────────────────────────────────────────┐
  │                  LangGraph Runtime                   │
  │                                                      │
  │  invoke(input, config={"thread_id": "patient-123"}) │
  │       │                                              │
  │       ▼                                              │
  │  ┌─────────┐  auto   ┌─────────┐  auto   ┌───────┐ │
  │  │ Node A  │──save──▶│ Node B  │──save──▶│ Node C│ │
  │  └─────────┘  ckpt   └─────────┘  ckpt   └───────┘ │
  │       │                    │                  │      │
  │       ▼                    ▼                  ▼      │
  │   ┌──────────────────────────────────────────────┐  │
  │   │           Checkpointer Backend               │  │
  │   │  MemorySaver │ SqliteSaver │ PostgresSaver   │  │
  │   └──────────────────────────────────────────────┘  │
  │       │                                              │
  │       ▼                                              │
  │   thread_id="patient-123"                            │
  │     ├─ checkpoint_0 (initial state)                  │
  │     ├─ checkpoint_1 (after Node A)                   │
  │     ├─ checkpoint_2 (after Node B)                   │
  │     └─ checkpoint_3 (after Node C) ← current        │
  │                                                      │
  │   get_state(config)          → current state         │
  │   get_state_history(config)  → all checkpoints       │
  │   update_state(config, new)  → inject state          │
  │   invoke(None, old_config)   → time-travel replay    │
  └─────────────────────────────────────────────────────┘

  Key concept: Every node execution automatically creates a checkpoint.
  You NEVER call "save" manually. The checkpointer does it for you.

Healthcare parallel:
  Think of checkpoints like the medical record at each stage of care:
  - Triage checkpoint: initial assessment saved
  - Lab checkpoint: lab results added to the record
  - Physician checkpoint: assessment and plan added
  If you need to audit "what did we know at triage?" — you time-travel
  to that checkpoint. If a physician overrides a lab order, that's a
  state update. Everything is versioned automatically.
"""

import os
import json
import time
import sqlite3
from datetime import datetime
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

oai_client = OpenAI()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# Clinical Workflow State
# ============================================================

class ClinicalState(TypedDict):
    """State for a clinical assessment workflow with checkpoint tracking."""
    patient_info: str
    triage_result: str
    lab_results: str
    assessment: str
    treatment_plan: str
    safety_check: str
    messages: Annotated[list[str], operator.add]  # Append-only log
    current_node: str
    workflow_status: str


# ============================================================
# Clinical Workflow Nodes
# ============================================================

def triage_node(state: ClinicalState) -> dict:
    """Triage: classify urgency and identify immediate needs."""
    response = llm.invoke([
        {"role": "system", "content": (
            "You are an ED triage nurse. Classify patient acuity (ESI 1-5) "
            "and identify immediate needs. Be concise (3-4 lines)."
        )},
        {"role": "user", "content": f"Triage this patient:\n{state['patient_info']}"},
    ])
    result = response.content
    return {
        "triage_result": result,
        "messages": [f"[Triage] {result[:100]}..."],
        "current_node": "triage",
        "workflow_status": "triage_complete",
    }


def lab_review_node(state: ClinicalState) -> dict:
    """Lab review: interpret results and flag critical values."""
    response = llm.invoke([
        {"role": "system", "content": (
            "You are a clinical lab specialist. Interpret labs and flag "
            "critical values. Be concise (3-4 lines)."
        )},
        {"role": "user", "content": (
            f"Patient: {state['patient_info']}\n"
            f"Triage: {state['triage_result']}\n"
            f"Interpret the labs."
        )},
    ])
    result = response.content
    return {
        "lab_results": result,
        "messages": [f"[Lab Review] {result[:100]}..."],
        "current_node": "lab_review",
        "workflow_status": "labs_complete",
    }


def assessment_node(state: ClinicalState) -> dict:
    """Physician assessment: differential diagnosis and risk level."""
    response = llm.invoke([
        {"role": "system", "content": (
            "You are a senior physician. Provide differential diagnosis, "
            "risk stratification, and next steps. Be concise (4-5 lines)."
        )},
        {"role": "user", "content": (
            f"Patient: {state['patient_info']}\n"
            f"Triage: {state['triage_result']}\n"
            f"Labs: {state['lab_results']}\n"
            f"Provide your assessment."
        )},
    ])
    result = response.content
    return {
        "assessment": result,
        "messages": [f"[Assessment] {result[:100]}..."],
        "current_node": "assessment",
        "workflow_status": "assessment_complete",
    }


def treatment_node(state: ClinicalState) -> dict:
    """Treatment plan: specific orders and interventions."""
    response = llm.invoke([
        {"role": "system", "content": (
            "You are a treating physician. Create specific treatment orders "
            "including meds, monitoring, and disposition. Be concise (4-5 lines)."
        )},
        {"role": "user", "content": (
            f"Patient: {state['patient_info']}\n"
            f"Assessment: {state['assessment']}\n"
            f"Labs: {state['lab_results']}\n"
            f"Create treatment plan."
        )},
    ])
    result = response.content
    return {
        "treatment_plan": result,
        "messages": [f"[Treatment] {result[:100]}..."],
        "current_node": "treatment",
        "workflow_status": "treatment_complete",
    }


def safety_node(state: ClinicalState) -> dict:
    """Safety check: review for drug interactions, contraindications, errors."""
    response = llm.invoke([
        {"role": "system", "content": (
            "You are a patient safety officer. Check the treatment plan for "
            "drug interactions, contraindications, and potential errors. "
            "Be concise (3-4 lines). Flag any concerns."
        )},
        {"role": "user", "content": (
            f"Patient: {state['patient_info']}\n"
            f"Treatment: {state['treatment_plan']}\n"
            f"Assessment: {state['assessment']}\n"
            f"Safety review."
        )},
    ])
    result = response.content
    return {
        "safety_check": result,
        "messages": [f"[Safety] {result[:100]}..."],
        "current_node": "safety",
        "workflow_status": "complete",
    }


# ============================================================
# Build the Clinical Workflow Graph
# ============================================================

def build_clinical_graph() -> StateGraph:
    """Build the 5-node clinical assessment workflow."""
    graph = StateGraph(ClinicalState)

    graph.add_node("triage", triage_node)
    graph.add_node("lab_review", lab_review_node)
    graph.add_node("assessment", assessment_node)
    graph.add_node("treatment", treatment_node)
    graph.add_node("safety", safety_node)

    graph.set_entry_point("triage")
    graph.add_edge("triage", "lab_review")
    graph.add_edge("lab_review", "assessment")
    graph.add_edge("assessment", "treatment")
    graph.add_edge("treatment", "safety")
    graph.add_edge("safety", END)

    return graph


# ============================================================
# Sample Patient
# ============================================================

SAMPLE_PATIENT = (
    "55-year-old male presenting with chest pain for 2 hours, "
    "radiating to left arm. History: Type 2 DM, HTN, hyperlipidemia. "
    "Meds: Metformin 1000mg BID, Lisinopril 20mg, Atorvastatin 40mg. "
    "Vitals: BP 158/92, HR 98, SpO2 96%. "
    "Labs: Troponin 0.45 (critical), glucose 210, creatinine 1.3. "
    "ECG: ST depression V3-V6."
)


# ============================================================
# Helper: Print Checkpoint History
# ============================================================

def print_checkpoint_history(app, config, title: str = "Checkpoint History"):
    """Display all checkpoints for a thread."""
    states = list(app.get_state_history(config))
    print(f"\n  {title} ({len(states)} checkpoints)")
    print(f"  {'#':<4} {'Checkpoint ID':<16} {'Node':<15} {'Status':<25} {'Messages'}")
    print(f"  {'─' * 90}")

    for i, state in enumerate(states):
        ckpt_id = state.config["configurable"]["checkpoint_id"][:14]
        values = state.values if state.values else {}
        node = values.get("current_node", "(initial)")
        status = values.get("workflow_status", "pending")
        msg_count = len(values.get("messages", []))
        print(f"  {i:<4} {ckpt_id:<16} {node:<15} {status:<25} {msg_count} msg(s)")


# ============================================================
# DEMO 1: MemorySaver — Automatic Checkpoints
# ============================================================

def demo_memory_saver():
    """Show how MemorySaver automatically checkpoints every node."""
    print("\n" + "=" * 70)
    print("  DEMO 1: MEMORYSAVER — AUTOMATIC CHECKPOINTS")
    print("=" * 70)
    print("""
  MemorySaver stores checkpoints in memory (for dev/testing).
  Every node execution automatically creates a checkpoint.
  No manual "save" calls needed.

  We'll run a 5-node clinical workflow and see all the checkpoints.
  """)

    graph = build_clinical_graph()
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "patient-12345"}}

    print(f"  Running workflow with thread_id='patient-12345'...")
    start = time.time()
    result = app.invoke(
        {
            "patient_info": SAMPLE_PATIENT,
            "triage_result": "",
            "lab_results": "",
            "assessment": "",
            "treatment_plan": "",
            "safety_check": "",
            "messages": [],
            "current_node": "",
            "workflow_status": "pending",
        },
        config,
    )
    elapsed = round(time.time() - start, 2)

    print(f"  Workflow completed in {elapsed}s")
    print(f"  Final status: {result['workflow_status']}")
    print(f"  Message log: {len(result['messages'])} entries")

    # Show the automatically created checkpoints
    print_checkpoint_history(app, config)

    # Get current state
    current = app.get_state(config)
    print(f"\n  Current state (get_state):")
    print(f"    Node: {current.values.get('current_node')}")
    print(f"    Status: {current.values.get('workflow_status')}")
    print(f"    Next: {current.next}")  # Empty = workflow finished

    print(f"\n  KEY INSIGHT: {len(list(app.get_state_history(config)))} checkpoints "
          f"were created AUTOMATICALLY — one per node + initial state.")


# ============================================================
# DEMO 2: SqliteSaver — Persistent Checkpoints
# ============================================================

def demo_sqlite_saver():
    """Show SqliteSaver for durable, persistent checkpoints."""
    print("\n" + "=" * 70)
    print("  DEMO 2: SQLITESAVER — PERSISTENT CHECKPOINTS")
    print("=" * 70)
    print("""
  SqliteSaver stores checkpoints in a SQLite database.
  Checkpoints survive process restarts — true production persistence.

  We'll run a workflow, close it, then reload the checkpoints.
  """)

    db_path = os.path.join(os.path.dirname(__file__) or ".", "checkpoints.db")

    # Clean up from previous runs
    if os.path.exists(db_path):
        os.remove(db_path)

    # Phase 1: Run workflow with SqliteSaver
    print(f"\n  ═══ Phase 1: Run workflow and save to SQLite ═══")
    print(f"  DB path: {db_path}")

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_clinical_graph()
        app = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "patient-67890"}}

        result = app.invoke(
            {
                "patient_info": SAMPLE_PATIENT,
                "triage_result": "",
                "lab_results": "",
                "assessment": "",
                "treatment_plan": "",
                "safety_check": "",
                "messages": [],
                "current_node": "",
                "workflow_status": "pending",
            },
            config,
        )

        print(f"  Workflow completed. Status: {result['workflow_status']}")
        ckpt_count = len(list(app.get_state_history(config)))
        print(f"  Checkpoints saved: {ckpt_count}")

    # Show database file exists
    db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    print(f"\n  Database file: {db_path}")
    print(f"  Size: {db_size:,} bytes")

    # Phase 2: Reload from SQLite (simulate restart)
    print(f"\n  ═══ Phase 2: Reload from SQLite (simulating restart) ═══")

    with SqliteSaver.from_conn_string(db_path) as checkpointer2:
        graph2 = build_clinical_graph()
        app2 = graph2.compile(checkpointer=checkpointer2)

        config2 = {"configurable": {"thread_id": "patient-67890"}}

        current = app2.get_state(config2)
        print(f"  Loaded state for thread 'patient-67890':")
        print(f"    Node: {current.values.get('current_node')}")
        print(f"    Status: {current.values.get('workflow_status')}")
        print(f"    Triage: {current.values.get('triage_result', '')[:80]}...")

        print_checkpoint_history(app2, config2, "Reloaded Checkpoint History")

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)

    print(f"\n  KEY INSIGHT: SqliteSaver checkpoints survive process restarts.")
    print(f"  Swap to PostgresSaver for production multi-server deployments.")


# ============================================================
# DEMO 3: Interrupt & Resume — Human-in-the-Loop
# ============================================================

def demo_interrupt_resume():
    """Show interrupt_before to pause workflow for human review."""
    print("\n" + "=" * 70)
    print("  DEMO 3: INTERRUPT & RESUME — HUMAN-IN-THE-LOOP")
    print("=" * 70)
    print("""
  interrupt_before=['treatment'] pauses the workflow BEFORE the treatment
  node runs. A human can review the assessment, then resume.

  Flow:
    Triage ✅ → Lab Review ✅ → Assessment ✅ → [PAUSE] → Treatment → Safety

  This is the PROPER way to do human-in-the-loop with LangGraph.
  The checkpoint saves the exact state so the workflow can resume
  hours or days later.
  """)

    graph = build_clinical_graph()
    checkpointer = MemorySaver()
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["treatment"],  # Pause before treatment
    )

    config = {"configurable": {"thread_id": "patient-hitl-1"}}

    # Phase 1: Run until interrupt
    print(f"  ═══ Phase 1: Run until interrupt ═══")
    result = app.invoke(
        {
            "patient_info": SAMPLE_PATIENT,
            "triage_result": "",
            "lab_results": "",
            "assessment": "",
            "treatment_plan": "",
            "safety_check": "",
            "messages": [],
            "current_node": "",
            "workflow_status": "pending",
        },
        config,
    )

    state = app.get_state(config)
    print(f"  Workflow paused!")
    print(f"    Last completed node: {state.values.get('current_node')}")
    print(f"    Status: {state.values.get('workflow_status')}")
    print(f"    Next nodes: {state.next}")
    print(f"    Assessment: {state.values.get('assessment', '')[:120]}...")

    print(f"\n  ── Human reviews the assessment... ──")
    print(f"  (In production, a clinician would review here)")
    print(f"  (Hours or days could pass — the checkpoint persists)")

    # Phase 2: Resume
    print(f"\n  ═══ Phase 2: Resume from checkpoint ═══")
    final_result = app.invoke(None, config)  # None = resume from checkpoint

    print(f"  Workflow resumed and completed!")
    print(f"    Status: {final_result['workflow_status']}")
    print(f"    Treatment: {final_result['treatment_plan'][:120]}...")
    print(f"    Safety: {final_result['safety_check'][:120]}...")

    print_checkpoint_history(app, config)

    print(f"\n  KEY INSIGHT: invoke(None, config) resumes from the checkpoint.")
    print(f"  The workflow paused at 'treatment', resumed, and completed.")


# ============================================================
# DEMO 4: Time-Travel — Replay from Past Checkpoints
# ============================================================

def demo_time_travel():
    """Show time-travel: replay workflow from any past checkpoint."""
    print("\n" + "=" * 70)
    print("  DEMO 4: TIME-TRAVEL — REPLAY FROM PAST CHECKPOINTS")
    print("=" * 70)
    print("""
  Every checkpoint has a unique checkpoint_id.
  You can replay the workflow from ANY past checkpoint.

  Use case: "What if we had a different assessment?"
  → Go back to the post-lab-review checkpoint
  → Inject a new assessment
  → Let the workflow re-run treatment + safety with the new assessment

  This is like git: you can check out any commit and branch from there.
  """)

    graph = build_clinical_graph()
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "patient-tt-1"}}

    # Run full workflow
    print(f"  Running full workflow...")
    result = app.invoke(
        {
            "patient_info": SAMPLE_PATIENT,
            "triage_result": "",
            "lab_results": "",
            "assessment": "",
            "treatment_plan": "",
            "safety_check": "",
            "messages": [],
            "current_node": "",
            "workflow_status": "pending",
        },
        config,
    )

    # Get all checkpoints
    states = list(app.get_state_history(config))
    print(f"\n  Full history: {len(states)} checkpoints")
    for i, s in enumerate(states):
        node = s.values.get("current_node", "(initial)") if s.values else "(empty)"
        print(f"    [{i}] Node: {node}, "
              f"ckpt: {s.config['configurable']['checkpoint_id'][:14]}...")

    # Find the post-lab-review checkpoint (node = "lab_review")
    lab_checkpoint = None
    for s in states:
        if s.values and s.values.get("current_node") == "lab_review":
            lab_checkpoint = s
            break

    if lab_checkpoint:
        print(f"\n  ═══ Time-Travel: Replaying from post-lab-review checkpoint ═══")
        ckpt_id = lab_checkpoint.config['configurable']['checkpoint_id'][:14]
        print(f"  Checkpoint: {ckpt_id}...")
        print(f"  State at this point:")
        print(f"    Triage: {lab_checkpoint.values.get('triage_result', '')[:80]}...")
        print(f"    Labs: {lab_checkpoint.values.get('lab_results', '')[:80]}...")
        print(f"    Assessment: (not yet run)")

        # Replay from this checkpoint
        replayed = app.invoke(None, lab_checkpoint.config)
        print(f"\n  Replayed workflow completed!")
        print(f"    Status: {replayed['workflow_status']}")
        print(f"    Messages from replay: {len(replayed['messages'])}")

    print(f"\n  KEY INSIGHT: Time-travel lets you 'what-if' any point in the")
    print(f"  workflow. Replay from any checkpoint and the workflow continues")
    print(f"  from there — like checking out a past git commit.")


# ============================================================
# DEMO 5: State Update — Injecting State Mid-Workflow
# ============================================================

def demo_state_update():
    """Show update_state to inject/modify state at a checkpoint."""
    print("\n" + "=" * 70)
    print("  DEMO 5: STATE UPDATE — INJECTING STATE MID-WORKFLOW")
    print("=" * 70)
    print("""
  update_state() lets you modify the checkpoint state.
  The workflow then continues from the MODIFIED state.

  Use case: Physician overrides the automated assessment:
  1. Workflow runs triage → labs → assessment (automated)
  2. Physician reviews, disagrees with assessment
  3. Physician injects their own assessment via update_state()
  4. Workflow resumes treatment + safety using the PHYSICIAN'S assessment
  """)

    graph = build_clinical_graph()
    checkpointer = MemorySaver()
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["treatment"],
    )

    config = {"configurable": {"thread_id": "patient-override-1"}}

    # Run until interrupt (before treatment)
    print(f"  Running until assessment complete (interrupt before treatment)...")
    app.invoke(
        {
            "patient_info": SAMPLE_PATIENT,
            "triage_result": "",
            "lab_results": "",
            "assessment": "",
            "treatment_plan": "",
            "safety_check": "",
            "messages": [],
            "current_node": "",
            "workflow_status": "pending",
        },
        config,
    )

    state = app.get_state(config)
    auto_assessment = state.values.get("assessment", "")
    print(f"\n  Automated assessment: {auto_assessment[:150]}...")

    # Physician overrides
    physician_override = (
        "PHYSICIAN OVERRIDE: Based on clinical presentation and rising troponin "
        "(0.45 ng/mL), this is a NSTEMI. TIMI score 5 (high risk). "
        "Recommend urgent cardiac catheterization within 24 hours. "
        "Start dual antiplatelet therapy. Consult cardiology STAT."
    )

    print(f"\n  Physician injects override via update_state():")
    print(f"    {physician_override[:120]}...")

    app.update_state(config, {
        "assessment": physician_override,
        "messages": ["[PHYSICIAN OVERRIDE] Assessment replaced by attending"],
    })

    # Verify the state was updated
    updated = app.get_state(config)
    print(f"\n  State after override:")
    print(f"    Assessment: {updated.values.get('assessment', '')[:120]}...")
    print(f"    Next: {updated.next}")

    # Resume from the overridden state
    print(f"\n  Resuming workflow with physician's assessment...")
    final = app.invoke(None, config)

    print(f"\n  Treatment plan (based on override):")
    print(f"    {final['treatment_plan'][:200]}...")
    print(f"\n  Safety check:")
    print(f"    {final['safety_check'][:200]}...")

    # Show the audit trail
    print_checkpoint_history(app, config, "Full Audit Trail (including override)")

    print(f"\n  KEY INSIGHT: update_state() creates a NEW checkpoint with the")
    print(f"  modified state. The original automated assessment is still in the")
    print(f"  history — nothing is lost. Full audit trail preserved.")


# ============================================================
# DEMO 6: Multi-Thread — Parallel Patient Workflows
# ============================================================

def demo_multi_thread():
    """Show multiple concurrent workflows using thread_id."""
    print("\n" + "=" * 70)
    print("  DEMO 6: MULTI-THREAD — PARALLEL PATIENT WORKFLOWS")
    print("=" * 70)
    print("""
  thread_id isolates state between different workflows.
  Each patient gets their own checkpoint history.

  In production: one server handles many patients simultaneously,
  each with their own thread_id, each checkpointed independently.
  """)

    graph = build_clinical_graph()
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    patients = {
        "patient-A": (
            "45-year-old female with severe headache and neck stiffness. "
            "History: migraine without aura. Vitals: BP 140/88, HR 90. "
            "Labs: WBC 14.2 (elevated). No meds. Temp 38.9°C."
        ),
        "patient-B": (
            "72-year-old male with shortness of breath and bilateral leg edema. "
            "History: CHF (EF 35%), CKD stage 3, atrial fibrillation. "
            "Meds: Furosemide 40mg, Carvedilol 12.5mg, Warfarin 5mg. "
            "Vitals: BP 102/68, HR 110 irreg, SpO2 88%. BNP 1500."
        ),
    }

    initial_state = {
        "patient_info": "",
        "triage_result": "",
        "lab_results": "",
        "assessment": "",
        "treatment_plan": "",
        "safety_check": "",
        "messages": [],
        "current_node": "",
        "workflow_status": "pending",
    }

    # Run both patients
    for thread_id, info in patients.items():
        config = {"configurable": {"thread_id": thread_id}}
        state = {**initial_state, "patient_info": info}

        print(f"\n  Running workflow for {thread_id}...")
        result = app.invoke(state, config)
        print(f"    Status: {result['workflow_status']}")
        print(f"    Assessment: {result['assessment'][:100]}...")

    # Show that each thread has its own isolated history
    print(f"\n  ═══ Thread Isolation ═══")
    for thread_id in patients:
        config = {"configurable": {"thread_id": thread_id}}
        states = list(app.get_state_history(config))
        current = app.get_state(config)
        patient_preview = current.values.get("patient_info", "")[:50]
        print(f"\n  Thread '{thread_id}': {len(states)} checkpoints")
        print(f"    Patient: {patient_preview}...")
        print(f"    Status: {current.values.get('workflow_status')}")

    print(f"\n  KEY INSIGHT: thread_id provides complete isolation.")
    print(f"  Each patient's workflow has its own checkpoint history.")
    print(f"  No data leaks between threads.")


# ============================================================
# DEMO 7: Checkpoint Diffing — Compare State Between Steps
# ============================================================

def demo_checkpoint_diff():
    """Show how to diff state between checkpoints."""
    print("\n" + "=" * 70)
    print("  DEMO 7: CHECKPOINT DIFFING — WHAT CHANGED AT EACH STEP?")
    print("=" * 70)
    print("""
  By comparing consecutive checkpoints, you can see exactly what changed
  at each workflow step. This is invaluable for:
  - Debugging: "Why did the treatment include drug X?"
  - Auditing: "What info was available when the decision was made?"
  - Quality: "Did the safety check catch the interaction?"
  """)

    graph = build_clinical_graph()
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "patient-diff-1"}}
    app.invoke(
        {
            "patient_info": SAMPLE_PATIENT,
            "triage_result": "",
            "lab_results": "",
            "assessment": "",
            "treatment_plan": "",
            "safety_check": "",
            "messages": [],
            "current_node": "",
            "workflow_status": "pending",
        },
        config,
    )

    states = list(app.get_state_history(config))
    states.reverse()  # Chronological order

    tracked_fields = [
        "triage_result", "lab_results", "assessment",
        "treatment_plan", "safety_check", "current_node", "workflow_status",
    ]

    print(f"\n  State changes at each checkpoint:")
    print(f"  {'─' * 70}")

    prev_values = {}
    for i, state in enumerate(states):
        values = state.values if state.values else {}
        node = values.get("current_node", "(initial)")

        changed = []
        for field in tracked_fields:
            curr_val = values.get(field, "")
            prev_val = prev_values.get(field, "")
            if curr_val != prev_val and curr_val:
                changed.append(field)

        if changed:
            print(f"\n  Checkpoint {i}: after '{node}'")
            print(f"    Fields changed: {', '.join(changed)}")
            for field in changed:
                val = values.get(field, "")[:80].replace("\n", " | ")
                print(f"      {field}: {val}...")

        prev_values = dict(values)

    print(f"\n  KEY INSIGHT: Checkpoint diffs show exactly what each node")
    print(f"  contributed. Critical for debugging and audit trails.")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 6: LANGGRAPH NATIVE CHECKPOINTING")
    print("  Time-Travel, Interrupt & Resume, State Updates")
    print("=" * 70)
    print("""
    LangGraph's built-in checkpointing system provides:
    - Automatic checkpoint at every node
    - Thread-based state isolation
    - Interrupt/resume for human-in-the-loop
    - Time-travel debugging
    - Pluggable backends (Memory, SQLite, Postgres)

    Choose a demo:
      1 → MemorySaver: automatic checkpoints
      2 → SqliteSaver: persistent checkpoints (survives restart)
      3 → Interrupt & Resume: human-in-the-loop
      4 → Time-Travel: replay from past checkpoints
      5 → State Update: physician override mid-workflow
      6 → Multi-Thread: parallel patient workflows
      7 → Checkpoint Diffing: what changed at each step?
      8 → Run all demos (1-7)
    """)

    choice = input("  Enter choice (1-8): ").strip()

    demos = {
        "1": demo_memory_saver,
        "2": demo_sqlite_saver,
        "3": demo_interrupt_resume,
        "4": demo_time_travel,
        "5": demo_state_update,
        "6": demo_multi_thread,
        "7": demo_checkpoint_diff,
    }

    if choice == "8":
        for demo in [demo_memory_saver, demo_sqlite_saver,
                      demo_interrupt_resume, demo_time_travel,
                      demo_state_update, demo_multi_thread,
                      demo_checkpoint_diff]:
            demo()
    elif choice in demos:
        demos[choice]()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. CHECKPOINTER = AUTOMATIC VERSION CONTROL FOR WORKFLOW STATE
   Every node execution creates a checkpoint. You never call "save" manually.
   It's like git auto-committing after every function runs.

2. MEMORYSAVER vs SQLITESAVER:
   - MemorySaver: In-memory, fast, great for dev/testing. Gone when process dies.
   - SqliteSaver: Persistent to disk. Survives restarts. Good for single-server.
   - PostgresSaver: Production. Multi-server. Concurrent access. (install separately)
   
   Switching is one line: just change the checkpointer in compile().

3. THREAD_ID = SESSION ISOLATION:
   config = {"configurable": {"thread_id": "patient-123"}}
   Each thread_id gets its own checkpoint history. No data leaks.
   In production: thread_id = user_id, session_id, or patient_id.

4. INTERRUPT_BEFORE / INTERRUPT_AFTER = HUMAN-IN-THE-LOOP:
   app = graph.compile(checkpointer=mem, interrupt_before=["treatment"])
   The workflow pauses at this node. Resume with: app.invoke(None, config)
   The checkpoint persists — resume hours or days later.

5. TIME-TRAVEL = REPLAY FROM ANY CHECKPOINT:
   states = list(app.get_state_history(config))
   app.invoke(None, states[3].config)  # Replay from checkpoint 3
   Like git checkout <commit-hash>. The old state is never lost.

6. UPDATE_STATE = STATE INJECTION:
   app.update_state(config, {"assessment": "physician override"})
   Creates a new checkpoint with the modified state. Original preserved.
   Critical for physician overrides, corrections, manual adjustments.

7. CHECKPOINT DIFFING = AUDIT TRAIL:
   Compare consecutive checkpoints to see what each node contributed.
   Essential for healthcare compliance: "what was known at decision time?"

8. EXERCISE 4 vs EXERCISE 6:
   Exercise 4: Custom JSON persistence (educational, manual save/load)
   Exercise 6: LangGraph native checkpointing (production, automatic)
   Use Exercise 6's approach in real applications.
"""

if __name__ == "__main__":
    main()
