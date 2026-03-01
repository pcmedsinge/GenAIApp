"""
Exercise 8: Orchestration vs Choreography — Two Ways to Coordinate Agents

Skills practiced:
- Orchestration: a CENTRAL CONTROLLER tells agents what to do and when
- Choreography: agents react to EVENTS independently, no central controller
- Understanding the tradeoffs (control vs flexibility, coupling vs autonomy)
- Implementing both patterns for the same clinical workflow
- Knowing when each pattern is the right fit

Key insight: These are the TWO fundamental ways to coordinate multi-agent systems.
  Every pattern you've learned (pipeline, router, hierarchy, P2P, DAG) falls
  into one of these categories:

  ORCHESTRATION (central brain):
    Pipeline, Router, Hierarchy, DAG, Plan-and-Execute
    → One controller decides what happens, in what order

  CHOREOGRAPHY (distributed):
    P2P, Event-driven, Pub/Sub, Blackboard
    → Agents react to events independently, no single point of control

Architecture Comparison:

  ORCHESTRATION                          CHOREOGRAPHY
  ┌──────────────┐                       ┌──────────┐
  │ ORCHESTRATOR │                       │ Event Bus│
  │ (central     │                       │ (no brain│
  │  brain)      │                       │  — just  │
  └──┬──┬──┬─────┘                       │  routes) │
     │  │  │                             └─┬──┬──┬──┘
     │  │  │ tells each                    │  │  │  events flow
     │  │  │ what to do                    │  │  │  to whoever
     ▼  ▼  ▼                               ▼  ▼  ▼  listens
   ┌─┐┌─┐┌─┐                            ┌─┐┌─┐┌─┐
   │A││B││C│                             │A││B││C│
   └─┘└─┘└─┘                            └─┘└─┘└─┘
                                          each agent decides
   Orchestrator knows                     for itself when and
   the full workflow                      how to react

  Pros: Easy to understand,             Pros: Loosely coupled,
        debug, and monitor                    scalable, resilient
  Cons: Single point of failure,        Cons: Hard to debug,
        tight coupling                        hard to track flow

Healthcare parallel:
  Orchestration = Hospital protocol: "step 1: triage, step 2: labs, step 3: doc"
  Choreography = When a critical lab fires → pharmacy auto-adjusts dose,
    nursing auto-increases monitoring, attending gets auto-paged — no one
    "tells" them to do it, they each react to the same event.
"""

import os
import json
import time
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

client = OpenAI()


# ============================================================
# Shared Clinical Scenario
# ============================================================

SCENARIO_ACS = {
    "patient_id": "P-12345",
    "name": "John Smith",
    "age": 55,
    "chief_complaint": "Chest pain for 2 hours, radiating to left arm",
    "history": ["Type 2 diabetes", "Hypertension", "Hyperlipidemia"],
    "medications": ["Metformin 1000mg BID", "Lisinopril 20mg", "Atorvastatin 40mg"],
    "vitals": {"BP": "158/92", "HR": 98, "SpO2": 96, "Temp": 37.1},
    "labs": {"Troponin": 0.45, "Glucose": 210, "Creatinine": 1.3, "K": 4.8},
    "ecg": "ST depression V3-V6",
}


def scenario_to_text(scenario: dict) -> str:
    """Convert scenario dict to readable text."""
    return (
        f"Patient: {scenario['name']}, {scenario['age']}yo\n"
        f"CC: {scenario['chief_complaint']}\n"
        f"Hx: {', '.join(scenario['history'])}\n"
        f"Meds: {', '.join(scenario['medications'])}\n"
        f"Vitals: {json.dumps(scenario['vitals'])}\n"
        f"Labs: {json.dumps(scenario['labs'])}\n"
        f"ECG: {scenario['ecg']}"
    )


# ============================================================
# PATTERN 1: ORCHESTRATION (Central Controller)
# ============================================================

class ClinicalOrchestrator:
    """
    A central controller that manages the entire clinical workflow.
    It knows ALL the steps, decides the order, and passes data between agents.

    The orchestrator is the BRAIN — agents are just workers.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.workflow_log = []
        self.step_results = {}

    def _call_agent(self, agent_name: str, system_prompt: str, task: str,
                    context: str = "") -> str:
        """The orchestrator calls each agent — agents don't call each other."""
        start = time.time()

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{task}\n\nContext:\n{context}" if context else task},
            ],
            temperature=0,
        )

        result = response.choices[0].message.content
        elapsed = round(time.time() - start, 2)

        self.workflow_log.append({
            "step": len(self.workflow_log) + 1,
            "agent": agent_name,
            "task": task[:80],
            "elapsed": elapsed,
            "result_preview": result[:100],
        })

        return result

    def run(self, scenario: dict, verbose: bool = True) -> dict:
        """
        The orchestrator runs the FULL workflow.
        It knows every step, decides what data to pass, handles errors.
        """
        scenario_text = scenario_to_text(scenario)

        if verbose:
            print(f"\n  ╔══════════════════════════════════════╗")
            print(f"  ║  ORCHESTRATION PATTERN               ║")
            print(f"  ║  Central controller manages workflow  ║")
            print(f"  ╚══════════════════════════════════════╝")

        # Step 1: Triage
        if verbose:
            print(f"\n  Step 1: TRIAGE (orchestrator → triage agent)")
        triage = self._call_agent(
            "Triage Agent",
            "You are an ED triage nurse. Classify this patient's acuity (ESI 1-5) "
            "and identify immediate needs. Be concise.",
            f"Triage this patient:\n{scenario_text}",
        )
        self.step_results["triage"] = triage
        if verbose:
            print(f"    Result: {triage[:120].replace(chr(10), ' | ')}...")

        # Step 2: Lab interpretation
        if verbose:
            print(f"\n  Step 2: LAB REVIEW (orchestrator → lab agent)")
        labs = self._call_agent(
            "Lab Agent",
            "You are a clinical lab specialist. Interpret the labs and flag critical values.",
            f"Interpret labs for this patient:\n{scenario_text}",
            context=f"Triage assessment: {triage[:200]}",
        )
        self.step_results["labs"] = labs
        if verbose:
            print(f"    Result: {labs[:120].replace(chr(10), ' | ')}...")

        # Step 3: Clinical assessment (orchestrator decides to include triage + labs)
        if verbose:
            print(f"\n  Step 3: ASSESSMENT (orchestrator → clinical agent, passes triage + labs)")
        assessment = self._call_agent(
            "Clinical Agent",
            "You are a senior physician. Provide differential diagnosis and risk level.",
            f"Assess this patient:\n{scenario_text}",
            context=f"Triage: {triage[:200]}\nLabs: {labs[:200]}",
        )
        self.step_results["assessment"] = assessment
        if verbose:
            print(f"    Result: {assessment[:120].replace(chr(10), ' | ')}...")

        # Step 4: Treatment plan
        if verbose:
            print(f"\n  Step 4: TREATMENT (orchestrator → treatment agent, passes assessment)")
        treatment = self._call_agent(
            "Treatment Agent",
            "You are a treating physician. Create specific treatment orders.",
            f"Create treatment plan:\n{scenario_text}",
            context=f"Assessment: {assessment[:300]}",
        )
        self.step_results["treatment"] = treatment
        if verbose:
            print(f"    Result: {treatment[:120].replace(chr(10), ' | ')}...")

        # Step 5: Safety check (orchestrator adds ALL previous results)
        if verbose:
            print(f"\n  Step 5: SAFETY CHECK (orchestrator → safety agent, passes everything)")
        safety = self._call_agent(
            "Safety Agent",
            "You are a patient safety officer. Check for drug interactions, "
            "contraindications, and potential errors. Flag concerns.",
            f"Safety review:\n{scenario_text}",
            context=f"Treatment: {treatment[:300]}\nAssessment: {assessment[:200]}",
        )
        self.step_results["safety"] = safety
        if verbose:
            print(f"    Result: {safety[:120].replace(chr(10), ' | ')}...")

        if verbose:
            print(f"\n  ── Orchestrator completed {len(self.workflow_log)} steps ──")
            total_time = sum(s["elapsed"] for s in self.workflow_log)
            print(f"  Total time: {total_time:.2f}s (sequential — each step waits)")

        return {
            "pattern": "orchestration",
            "steps": len(self.workflow_log),
            "results": self.step_results,
            "log": self.workflow_log,
        }


# ============================================================
# PATTERN 2: CHOREOGRAPHY (Event-Driven)
# ============================================================

class Event:
    """An event that flows through the system."""

    def __init__(self, event_type: str, source: str, data: dict):
        self.event_type = event_type
        self.source = source
        self.data = data
        self.timestamp = datetime.now().isoformat()
        self.id = f"evt_{int(time.time() * 1000)}"

    def __repr__(self):
        return f"Event({self.event_type}, from={self.source})"


class EventBus:
    """
    A simple event bus. Routes events to interested subscribers.
    The bus has NO logic — it doesn't decide what to do with events.
    It just delivers them.

    This is the "choreography" equivalent of a message queue
    (like RabbitMQ, Kafka, or AWS SNS).
    """

    def __init__(self):
        self.subscribers = defaultdict(list)  # event_type → [callback]
        self.event_log = []
        self.results = {}  # agent_name → result

    def subscribe(self, event_type: str, callback):
        """An agent subscribes to events it cares about."""
        self.subscribers[event_type].append(callback)

    def publish(self, event: Event):
        """Publish an event. All subscribers to this event type get notified."""
        self.event_log.append({
            "event": event.event_type,
            "source": event.source,
            "timestamp": event.timestamp,
            "subscribers": len(self.subscribers.get(event.event_type, [])),
        })

        for callback in self.subscribers.get(event.event_type, []):
            callback(event)

    def store_result(self, agent_name: str, result: str):
        """Agents store their results for later retrieval."""
        self.results[agent_name] = result


class ChoreographyAgent:
    """
    An agent in a choreography-based system.
    It subscribes to events, processes them, and publishes new events.
    No one tells it what to do — it decides based on the events it receives.
    """

    def __init__(self, name: str, system_prompt: str, listens_to: list[str],
                 publishes: str, bus: EventBus, model: str = "gpt-4o-mini"):
        self.name = name
        self.system_prompt = system_prompt
        self.publishes = publishes
        self.bus = bus
        self.model = model
        self.processed = False

        # Self-subscribe to relevant events
        for event_type in listens_to:
            bus.subscribe(event_type, self.on_event)

    def on_event(self, event: Event):
        """React to an event — this is the agent's autonomous behavior."""
        if self.processed:
            return  # Only process once (in a real system, handle dedup differently)

        self.processed = True
        scenario_text = event.data.get("scenario", "")
        context_parts = [scenario_text]

        # Gather any available context from the bus results
        for agent_name, result in self.bus.results.items():
            context_parts.append(f"\n{agent_name}: {result[:300]}")

        context = "\n".join(context_parts)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Process this case:\n{context}"},
            ],
            temperature=0,
        )

        result = response.choices[0].message.content
        self.bus.store_result(self.name, result)

        # Publish completion event — other agents may react to this
        self.bus.publish(Event(
            event_type=self.publishes,
            source=self.name,
            data={"result": result, **event.data},
        ))


def build_choreography_system() -> tuple[EventBus, list[ChoreographyAgent]]:
    """
    Build a choreography-based clinical workflow.
    Each agent subscribes to events and publishes its own events.
    The flow emerges from the event subscriptions — not from a controller.

    Event flow:
      patient_arrived →
        triage_agent reacts → publishes "triage_complete" →
          lab_agent reacts → publishes "labs_complete" →
            clinical_agent reacts → publishes "assessment_complete" →
              treatment_agent reacts → publishes "treatment_complete" →
                safety_agent reacts → publishes "safety_complete"
    """
    bus = EventBus()

    agents = [
        ChoreographyAgent(
            name="Triage",
            system_prompt=(
                "You are an ED triage nurse. Classify acuity (ESI 1-5) and "
                "identify immediate needs. Be concise."
            ),
            listens_to=["patient_arrived"],
            publishes="triage_complete",
            bus=bus,
        ),
        ChoreographyAgent(
            name="Lab Interpreter",
            system_prompt=(
                "You are a clinical lab specialist. Interpret labs, flag critical values."
            ),
            listens_to=["triage_complete"],
            publishes="labs_complete",
            bus=bus,
        ),
        ChoreographyAgent(
            name="Clinical Assessor",
            system_prompt=(
                "You are a senior physician. Provide differential diagnosis and risk level."
            ),
            listens_to=["labs_complete"],
            publishes="assessment_complete",
            bus=bus,
        ),
        ChoreographyAgent(
            name="Treatment Planner",
            system_prompt=(
                "You are a treating physician. Create specific treatment orders."
            ),
            listens_to=["assessment_complete"],
            publishes="treatment_complete",
            bus=bus,
        ),
        ChoreographyAgent(
            name="Safety Checker",
            system_prompt=(
                "You are a safety officer. Check for drug interactions, "
                "contraindications, errors. Flag concerns."
            ),
            listens_to=["treatment_complete"],
            publishes="safety_complete",
            bus=bus,
        ),
    ]

    return bus, agents


def run_choreography(scenario: dict, verbose: bool = True) -> dict:
    """Run the choreography-based workflow."""
    if verbose:
        print(f"\n  ╔══════════════════════════════════════╗")
        print(f"  ║  CHOREOGRAPHY PATTERN                ║")
        print(f"  ║  Agents react to events, no controller║")
        print(f"  ╚══════════════════════════════════════╝")

    bus, agents = build_choreography_system()
    scenario_text = scenario_to_text(scenario)

    if verbose:
        print(f"\n  Event subscriptions:")
        for agent in agents:
            print(f"    {agent.name}: listens → publishes '{agent.publishes}'")

    # Trigger the cascade by publishing the initial event
    if verbose:
        print(f"\n  Publishing 'patient_arrived' event...")
        print(f"  (Watch the cascade — each event triggers the next agent)\n")

    start = time.time()
    bus.publish(Event(
        event_type="patient_arrived",
        source="ED_System",
        data={"scenario": scenario_text, "patient_id": scenario["patient_id"]},
    ))
    total_time = round(time.time() - start, 2)

    if verbose:
        print(f"  Event cascade:")
        for entry in bus.event_log:
            subs = entry["subscribers"]
            print(f"    📨 {entry['event']} (from {entry['source']}) → {subs} subscriber(s)")

        print(f"\n  Agent results:")
        for name, result in bus.results.items():
            preview = result[:120].replace("\n", " | ")
            print(f"    [{name}]: {preview}...")

        print(f"\n  ── Choreography completed in {total_time}s ──")
        print(f"  Events published: {len(bus.event_log)}")

    return {
        "pattern": "choreography",
        "events": len(bus.event_log),
        "results": bus.results,
        "event_log": bus.event_log,
        "elapsed": total_time,
    }


# ============================================================
# Demo Functions
# ============================================================

def demo_orchestration():
    """Show orchestration pattern."""
    print("\n" + "=" * 70)
    print("  DEMO 1: ORCHESTRATION — CENTRAL CONTROLLER")
    print("=" * 70)
    print("""
  The orchestrator (central brain) knows the full workflow:
    Step 1: Call triage agent
    Step 2: Call lab agent (pass triage results)
    Step 3: Call clinical agent (pass triage + lab results)
    Step 4: Call treatment agent (pass assessment)
    Step 5: Call safety agent (pass everything)

  The orchestrator decides: what runs, in what order, with what data.
  Agents are passive — they only work when the orchestrator calls them.
  """)

    orch = ClinicalOrchestrator()
    result = orch.run(SCENARIO_ACS)

    print(f"\n  Workflow log:")
    for step in result["log"]:
        print(f"    Step {step['step']}: {step['agent']} ({step['elapsed']}s)")


def demo_choreography():
    """Show choreography pattern."""
    print("\n" + "=" * 70)
    print("  DEMO 2: CHOREOGRAPHY — EVENT-DRIVEN, NO CONTROLLER")
    print("=" * 70)
    print("""
  No central controller. Instead:
  1. System publishes "patient_arrived" event
  2. Triage agent REACTS → publishes "triage_complete"
  3. Lab agent REACTS to triage_complete → publishes "labs_complete"
  4. Clinical agent REACTS to labs_complete → publishes "assessment_complete"
  5. Treatment agent REACTS → publishes "treatment_complete"
  6. Safety agent REACTS → publishes "safety_complete"

  The FLOW emerges from event subscriptions, not from a controller's logic.
  """)

    result = run_choreography(SCENARIO_ACS)


def demo_comparison():
    """Side-by-side comparison of both patterns."""
    print("\n" + "=" * 70)
    print("  DEMO 3: ORCHESTRATION vs CHOREOGRAPHY — COMPARISON")
    print("=" * 70)
    print("""
  Same clinical workflow, both patterns. Compare:
  - Control flow (who decides what happens?)
  - Coupling (how dependent are agents on each other?)
  - Failure handling (what breaks when something fails?)
  - Visibility (can you tell what's happening?)
  """)

    # Orchestration
    print(f"\n  ═══ ORCHESTRATION ═══")
    orch = ClinicalOrchestrator()
    orch_start = time.time()
    orch_result = orch.run(SCENARIO_ACS, verbose=False)
    orch_time = round(time.time() - orch_start, 2)
    print(f"  Steps: {orch_result['steps']}")
    print(f"  Time: {orch_time}s")

    # Choreography
    print(f"\n  ═══ CHOREOGRAPHY ═══")
    choreo_start = time.time()
    choreo_result = run_choreography(SCENARIO_ACS, verbose=False)
    choreo_time = round(time.time() - choreo_start, 2)
    print(f"  Events: {choreo_result['events']}")
    print(f"  Time: {choreo_time}s")

    # Comparison table
    print(f"\n\n  ═══ COMPARISON TABLE ═══")
    print(f"  {'Dimension':<30} {'Orchestration':>15} {'Choreography':>15}")
    print(f"  {'─' * 60}")
    print(f"  {'Control':<30} {'Centralized':>15} {'Distributed':>15}")
    print(f"  {'Coupling':<30} {'Tight':>15} {'Loose':>15}")
    print(f"  {'Visibility':<30} {'Easy (1 place)':>15} {'Hard (events)':>15}")
    print(f"  {'Single point of failure':<30} {'Yes (orch.)':>15} {'No':>15}")
    print(f"  {'Adding new agents':<30} {'Modify orch.':>15} {'Just subscribe':>15}")
    print(f"  {'Debugging':<30} {'Easy':>15} {'Hard':>15}")
    print(f"  {'Scalability':<30} {'Limited':>15} {'High':>15}")
    print(f"  {'Execution time':<30} {orch_time:>14}s {choreo_time:>14}s")

    print(f"\n  WHEN TO USE ORCHESTRATION:")
    print(f"    ✓ Well-defined, sequential workflows")
    print(f"    ✓ Need strict ordering guarantees")
    print(f"    ✓ Want easy monitoring and debugging")
    print(f"    ✓ Small number of agents (< 10)")

    print(f"\n  WHEN TO USE CHOREOGRAPHY:")
    print(f"    ✓ Many independent agents that shouldn't know about each other")
    print(f"    ✓ Need to add/remove agents without changing existing code")
    print(f"    ✓ Want resilience (no single point of failure)")
    print(f"    ✓ Event-driven domains (alerts, monitoring, notifications)")


def demo_hybrid():
    """Show a hybrid approach combining both patterns."""
    print("\n" + "=" * 70)
    print("  DEMO 4: HYBRID — ORCHESTRATION + CHOREOGRAPHY")
    print("=" * 70)
    print("""
  Real systems often COMBINE both patterns:
  - ORCHESTRATION for the critical path (must happen in order)
  - CHOREOGRAPHY for side effects (notifications, logging, alerts)

  Example: The treatment workflow is orchestrated (strict order).
  But "critical lab result" events trigger independent reactions
  from pharmacy, nursing, and attending — that's choreography.
  """)

    scenario_text = scenario_to_text(SCENARIO_ACS)

    # Orchestrated critical path
    print(f"\n  Step 1: ORCHESTRATED CRITICAL PATH")
    print(f"  (Must happen in order: Triage → Assessment → Treatment)")

    orch = ClinicalOrchestrator()

    triage = orch._call_agent(
        "Triage", "Triage nurse. Classify ESI 1-5, flag critical labs.",
        f"Triage:\n{scenario_text}"
    )
    print(f"    ✓ Triage: {triage[:80].replace(chr(10), ' | ')}...")

    assessment = orch._call_agent(
        "Assessor", "Senior physician. Differential + risk.",
        f"Assess:\n{scenario_text}", context=triage[:200]
    )
    print(f"    ✓ Assessment: {assessment[:80].replace(chr(10), ' | ')}...")

    treatment = orch._call_agent(
        "Treatment", "Create orders.", f"Treat:\n{scenario_text}",
        context=assessment[:200]
    )
    print(f"    ✓ Treatment: {treatment[:80].replace(chr(10), ' | ')}...")

    # Choreographed side effects
    print(f"\n  Step 2: CHOREOGRAPHED SIDE EFFECTS")
    print(f"  (Independent reactions to 'critical_lab_result' event)")

    bus = EventBus()

    # Multiple independent agents react to the same event
    reactions = {}

    def pharmacy_reaction(event):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Pharmacist: check if any meds need urgent adjustment."},
                {"role": "user", "content": f"Critical lab: Troponin 0.45. Meds: {', '.join(SCENARIO_ACS['medications'])}"},
            ],
            temperature=0,
        )
        reactions["Pharmacy"] = resp.choices[0].message.content

    def nursing_reaction(event):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Charge nurse: update monitoring protocol for critical lab."},
                {"role": "user", "content": f"Critical lab: Troponin 0.45, patient with chest pain. Current vitals: {json.dumps(SCENARIO_ACS['vitals'])}"},
            ],
            temperature=0,
        )
        reactions["Nursing"] = resp.choices[0].message.content

    def alert_reaction(event):
        reactions["Alert System"] = (
            f"ALERT: Critical troponin ({SCENARIO_ACS['labs']['Troponin']}) "
            f"for patient {SCENARIO_ACS['patient_id']}. "
            f"Attending paged. Cath lab notified."
        )

    bus.subscribe("critical_lab_result", pharmacy_reaction)
    bus.subscribe("critical_lab_result", nursing_reaction)
    bus.subscribe("critical_lab_result", alert_reaction)

    # Fire the event — all 3 react independently
    bus.publish(Event("critical_lab_result", "Lab System",
                      {"lab": "Troponin", "value": 0.45, "critical": True}))

    for name, reaction in reactions.items():
        preview = reaction[:100].replace("\n", " | ")
        print(f"    📨 {name}: {preview}...")

    print(f"\n  The critical path (orchestrated) ensures correct treatment.")
    print(f"  The side effects (choreographed) happen independently — adding")
    print(f"  a new reactor (e.g., billing, quality tracking) requires ZERO")
    print(f"  changes to the orchestrated workflow.")


def demo_interactive():
    """Interactive mode."""
    print("\n" + "=" * 70)
    print("  DEMO 5: INTERACTIVE — TRY BOTH PATTERNS")
    print("=" * 70)
    print("  Enter 'orch' for orchestration, 'choreo' for choreography,")
    print("  'compare' for side-by-side, or 'quit' to exit.\n")

    while True:
        choice = input("  > ").strip().lower()

        if choice in ['quit', 'exit', 'q']:
            break
        elif choice == 'orch':
            orch = ClinicalOrchestrator()
            orch.run(SCENARIO_ACS)
        elif choice == 'choreo':
            run_choreography(SCENARIO_ACS)
        elif choice == 'compare':
            demo_comparison()
        else:
            print("  Enter 'orch', 'choreo', 'compare', or 'quit'.")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 8: ORCHESTRATION vs CHOREOGRAPHY")
    print("=" * 70)
    print("""
    The two fundamental ways to coordinate agents:
    - Orchestration: central controller drives the workflow
    - Choreography: agents react to events independently

    Choose a demo:
      1 → Orchestration (central controller)
      2 → Choreography (event-driven, no controller)
      3 → Side-by-side comparison
      4 → Hybrid (both combined)
      5 → Interactive
      6 → Run demos 1-4
    """)

    choice = input("  Enter choice (1-6): ").strip()

    demos = {
        "1": demo_orchestration,
        "2": demo_choreography,
        "3": demo_comparison,
        "4": demo_hybrid,
        "5": demo_interactive,
    }

    if choice == "6":
        for demo in [demo_orchestration, demo_choreography,
                      demo_comparison, demo_hybrid]:
            demo()
    elif choice in demos:
        demos[choice]()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. ORCHESTRATION = ONE BRAIN, MANY HANDS
   The orchestrator knows the full workflow. It calls agents in order,
   passes data between them, and handles errors centrally.
   Like a conductor leading an orchestra — every musician follows the conductor.

2. CHOREOGRAPHY = MANY BRAINS, NO CONDUCTOR
   Each agent reacts to events independently. No one knows the full workflow.
   The flow EMERGES from event subscriptions.
   Like a jazz ensemble — each musician reacts to what they hear.

3. ALL YOUR PREVIOUS PATTERNS ARE ORCHESTRATION:
   - Pipeline (orchestrator calls A → B → C)
   - Router (orchestrator decides which agent)
   - Hierarchy (supervisor orchestrates sub-agents)
   - DAG (engine orchestrates task execution)
   - Plan-and-Execute (planner orchestrates execution)

4. CHOREOGRAPHY PATTERNS ARE RARER IN LLM SYSTEMS:
   - P2P/Blackboard (agents react to blackboard state)
   - Event-driven (agents subscribe to events)
   - Pub/Sub (publish results, interested parties consume)
   Why? Because LLM calls are expensive and slow. Choreography
   can trigger unnecessary work if agents react to events they
   shouldn't. Orchestration gives you more control over token spend.

5. HYBRID IS THE REAL-WORLD ANSWER:
   - Orchestrate the critical path (treatment workflow)
   - Choreograph the side effects (alerts, monitoring, billing)
   This gives you: ordering guarantees WHERE needed,
   loose coupling WHERE flexibility matters.

6. KEY TRADEOFFS:
   Orchestration: Easy to understand, easy to debug, single point of failure
   Choreography: Hard to debug, no single point of failure, easy to extend

7. PRODUCTION CHOICE DEPENDS ON:
   - Team size: Small team → orchestration (easier to maintain)
   - Scale: 100+ agents → choreography (can't have one controller)
   - Domain: Sequential process → orchestration. Reactive/event-driven → choreography
   - Compliance: Auditable workflow needed → orchestration (clear trail)
"""

if __name__ == "__main__":
    main()
