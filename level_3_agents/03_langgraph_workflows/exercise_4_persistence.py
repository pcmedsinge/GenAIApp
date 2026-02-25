"""
Exercise 4: Workflow Persistence — Save and Resume Mid-Process

Skills practiced:
- Serializing LangGraph state to JSON
- Saving workflow state at any checkpoint
- Resuming a workflow from a saved checkpoint
- Simulating interruptions and recovery in production systems

Key insight: Production workflows get interrupted — systems crash,
  reviewers step away, patients leave and return. Persistence lets
  you save state at any point and resume later without losing work.
  This is essential for long-running clinical workflows.
"""

import os
import json
from datetime import datetime
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# State with Checkpoint Tracking
# ============================================================

class PersistentState(TypedDict):
    patient_info: str
    symptoms: str
    extracted_data: str
    risk_factors: str
    urgency: str
    assessment: str
    plan: str
    current_step: str          # Track where we are in the workflow
    completed_steps: str       # JSON list of completed step names
    workflow_id: str


# ============================================================
# Persistence Manager
# ============================================================

class WorkflowPersistence:
    """Save and load workflow state to/from JSON files"""

    def __init__(self, storage_dir=None):
        if storage_dir is None:
            storage_dir = os.path.join(os.path.dirname(__file__), "workflow_checkpoints")
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def save(self, workflow_id: str, state: dict, step_name: str):
        """Save workflow state at a checkpoint"""
        checkpoint = {
            "workflow_id": workflow_id,
            "saved_at": datetime.now().isoformat(),
            "current_step": step_name,
            "state": dict(state),  # Convert TypedDict to regular dict
        }

        path = os.path.join(self.storage_dir, f"{workflow_id}.json")
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def load(self, workflow_id: str) -> dict:
        """Load workflow state from a checkpoint"""
        path = os.path.join(self.storage_dir, f"{workflow_id}.json")
        if not os.path.exists(path):
            return None

        with open(path, "r") as f:
            return json.load(f)

    def list_checkpoints(self):
        """List all saved checkpoints"""
        checkpoints = []
        for f in os.listdir(self.storage_dir):
            if f.endswith(".json"):
                path = os.path.join(self.storage_dir, f)
                with open(path, "r") as fp:
                    data = json.load(fp)
                checkpoints.append({
                    "workflow_id": data["workflow_id"],
                    "saved_at": data["saved_at"],
                    "current_step": data["current_step"],
                })
        return sorted(checkpoints, key=lambda c: c["saved_at"], reverse=True)

    def delete(self, workflow_id: str):
        path = os.path.join(self.storage_dir, f"{workflow_id}.json")
        if os.path.exists(path):
            os.remove(path)

    def clear_all(self):
        for f in os.listdir(self.storage_dir):
            if f.endswith(".json"):
                os.remove(os.path.join(self.storage_dir, f))


# Global persistence instance
persistence = WorkflowPersistence()


# ============================================================
# Checkpointed Nodes
# ============================================================

def extract_with_checkpoint(state: PersistentState) -> dict:
    """Extract data and save checkpoint"""
    response = llm.invoke(
        f"Extract key clinical data from:\n"
        f"Patient: {state['patient_info']}\nSymptoms: {state['symptoms']}"
    )

    completed = json.loads(state.get("completed_steps", "[]"))
    completed.append("extract")

    updates = {
        "extracted_data": response.content,
        "current_step": "extract",
        "completed_steps": json.dumps(completed),
    }

    # Save checkpoint
    merged = {**state, **updates}
    persistence.save(state["workflow_id"], merged, "extract")
    print(f"    [Checkpoint saved: extract]")

    return updates


def risk_with_checkpoint(state: PersistentState) -> dict:
    response = llm.invoke(
        f"Identify risk factors:\n{state['extracted_data']}"
    )

    completed = json.loads(state.get("completed_steps", "[]"))
    completed.append("risk")

    updates = {
        "risk_factors": response.content,
        "current_step": "risk",
        "completed_steps": json.dumps(completed),
    }

    merged = {**state, **updates}
    persistence.save(state["workflow_id"], merged, "risk")
    print(f"    [Checkpoint saved: risk]")

    return updates


def classify_with_checkpoint(state: PersistentState) -> dict:
    response = llm.invoke(
        f"Classify urgency (emergency/urgent/routine):\n"
        f"Data: {state['extracted_data']}\nRisk: {state['risk_factors']}\n"
        f"Respond ONLY: emergency, urgent, or routine"
    )
    urgency = response.content.strip().lower()
    if "emergency" in urgency: urgency = "emergency"
    elif "urgent" in urgency: urgency = "urgent"
    else: urgency = "routine"

    completed = json.loads(state.get("completed_steps", "[]"))
    completed.append("classify")

    updates = {
        "urgency": urgency,
        "current_step": "classify",
        "completed_steps": json.dumps(completed),
    }

    merged = {**state, **updates}
    persistence.save(state["workflow_id"], merged, "classify")
    print(f"    [Checkpoint saved: classify]")

    return updates


def assess_with_checkpoint(state: PersistentState) -> dict:
    response = llm.invoke(
        f"Clinical assessment with differential diagnosis:\n"
        f"Data: {state['extracted_data']}\nRisk: {state['risk_factors']}\n"
        f"Urgency: {state['urgency']}\nEducational only."
    )

    completed = json.loads(state.get("completed_steps", "[]"))
    completed.append("assess")

    updates = {
        "assessment": response.content,
        "current_step": "assess",
        "completed_steps": json.dumps(completed),
    }

    merged = {**state, **updates}
    persistence.save(state["workflow_id"], merged, "assess")
    print(f"    [Checkpoint saved: assess]")

    return updates


def plan_with_checkpoint(state: PersistentState) -> dict:
    response = llm.invoke(
        f"Generate management plan:\n"
        f"Assessment: {state['assessment']}\nUrgency: {state['urgency']}\n"
        f"Educational only."
    )

    completed = json.loads(state.get("completed_steps", "[]"))
    completed.append("plan")

    updates = {
        "plan": response.content,
        "current_step": "plan",
        "completed_steps": json.dumps(completed),
    }

    merged = {**state, **updates}
    persistence.save(state["workflow_id"], merged, "plan")
    print(f"    [Checkpoint saved: plan — WORKFLOW COMPLETE]")

    return updates


# ============================================================
# Build Workflows: Full and Partial (for resume)
# ============================================================

def build_full_workflow():
    """Full workflow from start"""
    graph = StateGraph(PersistentState)
    graph.add_node("extract", extract_with_checkpoint)
    graph.add_node("risk", risk_with_checkpoint)
    graph.add_node("classify", classify_with_checkpoint)
    graph.add_node("assess", assess_with_checkpoint)
    graph.add_node("plan", plan_with_checkpoint)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "risk")
    graph.add_edge("risk", "classify")
    graph.add_edge("classify", "assess")
    graph.add_edge("assess", "plan")
    graph.add_edge("plan", END)

    return graph.compile()


def build_resume_workflow(start_from: str):
    """Build a workflow that starts from a specific node"""
    steps = ["extract", "risk", "classify", "assess", "plan"]
    nodes = {
        "extract": extract_with_checkpoint,
        "risk": risk_with_checkpoint,
        "classify": classify_with_checkpoint,
        "assess": assess_with_checkpoint,
        "plan": plan_with_checkpoint,
    }

    if start_from not in steps:
        raise ValueError(f"Unknown step: {start_from}")

    start_idx = steps.index(start_from)
    remaining = steps[start_idx:]

    graph = StateGraph(PersistentState)

    for step_name in remaining:
        graph.add_node(step_name, nodes[step_name])

    graph.set_entry_point(remaining[0])

    for i in range(len(remaining) - 1):
        graph.add_edge(remaining[i], remaining[i + 1])

    graph.add_edge(remaining[-1], END)

    return graph.compile()


# ============================================================
# DEMO 1: Full Run with Checkpoints
# ============================================================

def demo_full_run():
    """Run full workflow showing checkpoints at each step"""
    print("\n" + "=" * 70)
    print("DEMO 1: FULL WORKFLOW WITH CHECKPOINTS")
    print("=" * 70)
    print("""
    Every node saves state to disk. If the system crashes mid-workflow,
    we can resume from the last checkpoint.
    """)

    wf_id = f"demo_{datetime.now().strftime('%H%M%S')}"
    app = build_full_workflow()

    print(f"  Workflow ID: {wf_id}")
    print(f"  Running: extract → risk → classify → assess → plan\n")

    result = app.invoke({
        "patient_info": "62-year-old male, HTN and T2DM, on metformin and lisinopril",
        "symptoms": "New shortness of breath on exertion, ankle swelling, 3-week duration",
        "workflow_id": wf_id,
        "completed_steps": "[]",
    })

    print(f"\n  COMPLETED WORKFLOW:")
    print(f"  Urgency: {result.get('urgency', '?').upper()}")
    print(f"  Steps completed: {result.get('completed_steps', '[]')}")
    print(f"  Plan: {result.get('plan', 'N/A')[:400]}")

    # Show saved checkpoint
    checkpoint = persistence.load(wf_id)
    if checkpoint:
        print(f"\n  SAVED CHECKPOINT:")
        print(f"    Workflow: {checkpoint['workflow_id']}")
        print(f"    Step: {checkpoint['current_step']}")
        print(f"    Saved at: {checkpoint['saved_at']}")

    persistence.delete(wf_id)


# ============================================================
# DEMO 2: Simulate Interruption and Resume
# ============================================================

def demo_interrupt_resume():
    """Simulate a crash mid-workflow, then resume"""
    print("\n" + "=" * 70)
    print("DEMO 2: INTERRUPT & RESUME")
    print("=" * 70)
    print("""
    Simulates: Run first 3 steps → "crash" → resume from step 4.
    The saved state lets us continue without re-running completed steps.
    """)

    wf_id = f"interrupted_{datetime.now().strftime('%H%M%S')}"

    # PHASE 1: Run first 3 steps, then "crash"
    print(f"\n  PHASE 1: Running first 3 steps...")
    graph_partial = StateGraph(PersistentState)
    graph_partial.add_node("extract", extract_with_checkpoint)
    graph_partial.add_node("risk", risk_with_checkpoint)
    graph_partial.add_node("classify", classify_with_checkpoint)
    graph_partial.set_entry_point("extract")
    graph_partial.add_edge("extract", "risk")
    graph_partial.add_edge("risk", "classify")
    graph_partial.add_edge("classify", END)
    partial_app = graph_partial.compile()

    partial_result = partial_app.invoke({
        "patient_info": "55-year-old female, depression on sertraline, new chest pain",
        "symptoms": "Sharp chest pain worse with deep breathing, no exertion component",
        "workflow_id": wf_id,
        "completed_steps": "[]",
    })

    print(f"\n  ⚡ SIMULATED CRASH after step: {partial_result.get('current_step')}")
    print(f"  Steps completed: {partial_result.get('completed_steps')}")

    # PHASE 2: Resume from checkpoint
    print(f"\n  PHASE 2: Loading checkpoint and resuming...")
    checkpoint = persistence.load(wf_id)
    if checkpoint:
        saved_state = checkpoint["state"]
        last_step = checkpoint["current_step"]

        steps = ["extract", "risk", "classify", "assess", "plan"]
        next_idx = steps.index(last_step) + 1

        if next_idx < len(steps):
            next_step = steps[next_idx]
            print(f"  Resuming from: {next_step} (after {last_step})")

            resume_app = build_resume_workflow(next_step)
            final_result = resume_app.invoke(saved_state)

            print(f"\n  RESUMED WORKFLOW COMPLETE:")
            print(f"  All steps: {final_result.get('completed_steps')}")
            print(f"  Urgency: {final_result.get('urgency', '?').upper()}")
            print(f"  Plan: {final_result.get('plan', 'N/A')[:400]}")
        else:
            print(f"  Workflow was already complete!")

    persistence.delete(wf_id)


# ============================================================
# DEMO 3: Checkpoint Browser
# ============================================================

def demo_checkpoint_browser():
    """Save multiple workflows and browse checkpoints"""
    print("\n" + "=" * 70)
    print("DEMO 3: CHECKPOINT BROWSER")
    print("=" * 70)

    app = build_full_workflow()

    cases = [
        {"info": "30yo healthy female", "symptoms": "Mild ankle sprain"},
        {"info": "70yo male, CHF, CKD, DM", "symptoms": "Worsening dyspnea"},
        {"info": "45yo male, anxiety", "symptoms": "Palpitations and mild dizziness"},
    ]

    for i, case in enumerate(cases, 1):
        wf_id = f"case_{i}_{datetime.now().strftime('%H%M%S')}"
        print(f"\n  Running case {i}: {case['info'][:40]}...")
        app.invoke({
            "patient_info": case["info"],
            "symptoms": case["symptoms"],
            "workflow_id": wf_id,
            "completed_steps": "[]",
        })

    # Browse checkpoints
    print(f"\n{'─' * 60}")
    print("  SAVED CHECKPOINTS:")
    print(f"{'─' * 60}")

    for cp in persistence.list_checkpoints():
        print(f"  ID: {cp['workflow_id']:<35} Step: {cp['current_step']:<10} Saved: {cp['saved_at'][:19]}")

    # Load and inspect one
    checkpoints = persistence.list_checkpoints()
    if checkpoints:
        cp = persistence.load(checkpoints[0]["workflow_id"])
        print(f"\n  INSPECTING: {checkpoints[0]['workflow_id']}")
        state = cp["state"]
        print(f"    Patient: {state.get('patient_info', '?')[:60]}")
        print(f"    Urgency: {state.get('urgency', '?')}")
        print(f"    Steps: {state.get('completed_steps', '[]')}")

    persistence.clear_all()
    print(f"\n  [Cleaned up all checkpoints]")


# ============================================================
# DEMO 4: Interactive
# ============================================================

def demo_interactive():
    """Interactive with save/resume"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE WITH PERSISTENCE")
    print("=" * 70)
    print("  Commands: 'run' — new case, 'list' — show checkpoints")
    print("  'resume' — resume a saved workflow, 'quit' — exit\n")

    while True:
        cmd = input("  Command (run/list/resume/quit): ").strip().lower()

        if cmd in ['quit', 'exit', 'q']:
            break
        elif cmd == 'run':
            info = input("  Patient info: ").strip()
            symptoms = input("  Symptoms: ").strip()
            if not info or not symptoms:
                continue
            wf_id = f"interactive_{datetime.now().strftime('%H%M%S')}"
            app = build_full_workflow()
            result = app.invoke({
                "patient_info": info, "symptoms": symptoms,
                "workflow_id": wf_id, "completed_steps": "[]",
            })
            print(f"\n  Done! Urgency: {result.get('urgency', '?').upper()}")
            print(f"  Plan: {result.get('plan', 'N/A')[:400]}\n")
        elif cmd == 'list':
            for cp in persistence.list_checkpoints():
                print(f"    {cp['workflow_id']} (step: {cp['current_step']}, saved: {cp['saved_at'][:19]})")
            if not persistence.list_checkpoints():
                print("    No checkpoints saved.")
            print()
        elif cmd == 'resume':
            checkpoints = persistence.list_checkpoints()
            if not checkpoints:
                print("  No checkpoints to resume.\n")
                continue
            for i, cp in enumerate(checkpoints):
                print(f"    {i+1}. {cp['workflow_id']} (step: {cp['current_step']})")
            idx = input("  Resume which? ").strip()
            try:
                cp = checkpoints[int(idx) - 1]
                saved = persistence.load(cp["workflow_id"])
                last = saved["current_step"]
                steps = ["extract", "risk", "classify", "assess", "plan"]
                nxt = steps.index(last) + 1
                if nxt < len(steps):
                    app = build_resume_workflow(steps[nxt])
                    result = app.invoke(saved["state"])
                    print(f"\n  Resumed! Plan: {result.get('plan', 'N/A')[:400]}\n")
                else:
                    print("  Workflow already complete.\n")
            except (ValueError, IndexError):
                print("  Invalid selection.\n")

    persistence.clear_all()


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 4: WORKFLOW PERSISTENCE — SAVE & RESUME")
    print("=" * 70)
    print("""
    Save workflow state at every step and resume from any checkpoint.
    Essential for long-running clinical workflows.

    Choose a demo:
      1 → Full run with checkpoints
      2 → Simulate interrupt & resume
      3 → Checkpoint browser
      4 → Interactive with save/resume
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1": demo_full_run()
    elif choice == "2": demo_interrupt_resume()
    elif choice == "3": demo_checkpoint_browser()
    elif choice == "4": demo_interactive()
    elif choice == "5":
        demo_full_run()
        demo_interrupt_resume()
        demo_checkpoint_browser()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. CHECKPOINTING: Save state after every node. If the system crashes,
   you only lose the current step — not the entire workflow.

2. RESUME FROM ANY POINT: build_resume_workflow() creates a graph
   starting from any step. Combined with the saved state, it
   continues exactly where it left off.

3. PRODUCTION PATTERNS: Real systems use databases (not JSON files)
   and message queues for checkpoint storage. The pattern is the same:
   serialize state → store → load → resume.

4. LONG-RUNNING WORKFLOWS: Clinical workflows can span hours/days
   (e.g., waiting for lab results, human review). Persistence is
   essential — you can't keep processes running that long.

5. AUDIT TRAIL BONUS: Checkpoints also serve as an audit trail.
   You can inspect the state at any step to understand what the
   system knew and decided at each point in the workflow.

6. LANGGRAPH NATIVE PERSISTENCE: LangGraph actually has built-in
   checkpointing (SqliteSaver, MemorySaver). This exercise teaches
   the concept from scratch; production code would use the built-in.
"""

if __name__ == "__main__":
    main()
