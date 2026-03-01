"""
Exercise 10: Event-Driven / Pub-Sub Agent Systems

Skills practiced:
- Implementing a publish-subscribe (Pub/Sub) message system for agents
- Designing event-driven agent architectures
- Understanding async reactions vs synchronous workflows
- Building reactive clinical monitoring systems
- Implementing event filtering, prioritization, and dead-letter queues

What is Event-Driven Architecture?
  Agents don't call each other directly. Instead:
  1. Agents PUBLISH events to topics/channels
  2. Other agents SUBSCRIBE to topics they care about
  3. When an event arrives, subscribers REACT independently
  4. No agent needs to know who else is listening

  This is pure CHOREOGRAPHY — no central controller.

Architecture:

  ┌──────────────────────────────────────────────┐
  │              EVENT BUS / BROKER               │
  │   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐   │
  │   │vitals│ │ labs │ │orders│ │ alerts   │   │
  │   │topic │ │topic │ │topic │ │ topic    │   │
  │   └──┬───┘ └──┬───┘ └──┬───┘ └────┬─────┘   │
  └──────┼────────┼────────┼──────────┼──────────┘
         │        │        │          │
    ┌────▼──┐ ┌───▼──┐ ┌──▼───┐ ┌───▼────┐
    │Monitor│ │Pharm │ │Safety│ │Escalate│
    │Agent  │ │Agent │ │Agent │ │Agent   │
    └───────┘ └──────┘ └──────┘ └────────┘

  Key properties:
  - LOOSE COUPLING: Publisher doesn't know subscribers
  - ASYNC: Subscribers react at their own pace
  - SCALABLE: Add new subscribers without changing publishers
  - RESILIENT: One subscriber failing doesn't affect others

Healthcare parallel:
  - Lab system publishes "critical result" event
  - Independently: pharmacy adjusts meds, nursing increases monitoring,
    attending gets paged, quality dashboard updates
  - The lab system doesn't know or care who's listening
  - Adding a new subscriber (e.g., research database) requires
    ZERO changes to the lab system
"""

import os
import json
import time
import threading
from datetime import datetime
from collections import defaultdict
from queue import Queue, Empty
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

client = OpenAI()


# ============================================================
# Event System
# ============================================================

class EventPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class ClinicalEvent:
    """A clinical event flowing through the system."""
    topic: str
    event_type: str
    source: str
    data: dict
    priority: EventPriority = EventPriority.NORMAL
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    event_id: str = field(default_factory=lambda: f"evt-{int(time.time() * 1000)}")
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return f"Event(topic={self.topic}, type={self.event_type}, priority={self.priority.name})"


class EventFilter:
    """Filter events based on criteria."""

    def __init__(self, event_types: list[str] = None,
                 min_priority: EventPriority = None,
                 custom_filter: callable = None):
        self.event_types = event_types
        self.min_priority = min_priority
        self.custom_filter = custom_filter

    def matches(self, event: ClinicalEvent) -> bool:
        if self.event_types and event.event_type not in self.event_types:
            return False
        if self.min_priority and event.priority.value < self.min_priority.value:
            return False
        if self.custom_filter and not self.custom_filter(event):
            return False
        return True


class EventBroker:
    """
    Central event broker (message bus).
    Routes events to topic subscribers. Supports:
    - Topic-based routing
    - Event filtering
    - Dead-letter queue (for failed deliveries)
    - Event history
    """

    def __init__(self):
        self.subscribers = defaultdict(list)  # topic → [(callback, filter)]
        self.event_history = []
        self.dead_letter_queue = []
        self.stats = defaultdict(int)

    def subscribe(self, topic: str, callback: callable,
                  event_filter: EventFilter = None, subscriber_name: str = ""):
        """Subscribe a callback to a topic with optional filtering."""
        self.subscribers[topic].append({
            "callback": callback,
            "filter": event_filter,
            "name": subscriber_name,
        })

    def publish(self, event: ClinicalEvent, verbose: bool = False):
        """Publish an event to all subscribers of its topic."""
        self.event_history.append(event)
        self.stats["published"] += 1
        self.stats[f"topic:{event.topic}"] += 1

        subscribers = self.subscribers.get(event.topic, [])

        if verbose:
            print(f"    📨 Published: {event.event_type} on '{event.topic}' "
                  f"→ {len(subscribers)} subscriber(s)")

        for sub in subscribers:
            # Apply filter
            if sub["filter"] and not sub["filter"].matches(event):
                self.stats["filtered_out"] += 1
                if verbose:
                    print(f"      ⊘ Filtered out for {sub['name']}")
                continue

            try:
                sub["callback"](event)
                self.stats["delivered"] += 1
            except Exception as e:
                self.dead_letter_queue.append({
                    "event": event,
                    "subscriber": sub["name"],
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
                self.stats["failed"] += 1
                if verbose:
                    print(f"      ❌ Delivery failed for {sub['name']}: {e}")

    def get_stats(self) -> dict:
        return dict(self.stats)


# ============================================================
# Reactive Clinical Agents
# ============================================================

class ReactiveAgent:
    """
    An agent that subscribes to events and reacts to them.
    Each agent is independent — it doesn't know about other agents.
    """

    def __init__(self, name: str, role: str, system_prompt: str,
                 broker: EventBroker, subscriptions: list[dict],
                 model: str = "gpt-4o-mini"):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.broker = broker
        self.model = model
        self.reactions = []

        # Subscribe to topics
        for sub in subscriptions:
            event_filter = None
            if "filter" in sub:
                event_filter = sub["filter"]

            broker.subscribe(
                topic=sub["topic"],
                callback=self._on_event,
                event_filter=event_filter,
                subscriber_name=self.name,
            )

    def _on_event(self, event: ClinicalEvent):
        """React to an event."""
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": (
                    f"React to this clinical event:\n"
                    f"Event: {event.event_type}\n"
                    f"Source: {event.source}\n"
                    f"Priority: {event.priority.name}\n"
                    f"Data: {json.dumps(event.data, indent=2)}"
                )},
            ],
            temperature=0,
        )

        result = response.choices[0].message.content
        reaction = {
            "agent": self.name,
            "event": event.event_type,
            "response": result,
            "timestamp": datetime.now().isoformat(),
        }
        self.reactions.append(reaction)

        # Some agents publish follow-up events
        if hasattr(self, 'publish_on_reaction'):
            self.publish_on_reaction(event, result)

        return result


class EscalatingAgent(ReactiveAgent):
    """An agent that can publish escalation events based on its analysis."""

    def __init__(self, *args, escalation_topic: str = "alerts", **kwargs):
        super().__init__(*args, **kwargs)
        self.escalation_topic = escalation_topic

    def publish_on_reaction(self, trigger_event: ClinicalEvent, result: str):
        """If the analysis indicates a critical situation, escalate."""
        result_lower = result.lower()
        is_critical = any(word in result_lower for word in
                          ["critical", "urgent", "emergency", "immediate", "stat"])

        if is_critical:
            self.broker.publish(ClinicalEvent(
                topic=self.escalation_topic,
                event_type="escalation",
                source=self.name,
                data={
                    "trigger_event": trigger_event.event_type,
                    "assessment": result[:300],
                    "patient_data": trigger_event.data,
                },
                priority=EventPriority.CRITICAL,
            ))


# ============================================================
# Pre-built Clinical Monitoring System
# ============================================================

def build_clinical_monitoring_system() -> tuple[EventBroker, list[ReactiveAgent]]:
    """
    Build a full event-driven clinical monitoring system.

    Topics:
    - vitals: vital sign readings
    - labs: lab results
    - orders: medication and procedure orders
    - alerts: clinical alerts and escalations
    - nursing: nursing assessments and notes
    """
    broker = EventBroker()

    agents = []

    # 1. Vital Signs Monitor — listens to vitals, escalates abnormals
    vitals_monitor = EscalatingAgent(
        name="Vitals Monitor",
        role="Vital signs monitoring agent",
        system_prompt=(
            "You are a clinical monitoring system for vital signs. "
            "Analyze vital sign changes and determine if they indicate "
            "deterioration. Be concise (2-3 lines). Flag critical changes. "
            "If any vital is critical, say 'CRITICAL' in your response."
        ),
        broker=broker,
        subscriptions=[{"topic": "vitals"}],
        escalation_topic="alerts",
    )
    agents.append(vitals_monitor)

    # 2. Lab Monitor — listens to labs, escalates critical values
    lab_monitor = EscalatingAgent(
        name="Lab Monitor",
        role="Lab result monitoring agent",
        system_prompt=(
            "You are a clinical lab monitoring system. Flag critical "
            "lab values immediately. Be concise (2-3 lines). "
            "If any value is outside critical range, say 'CRITICAL'."
        ),
        broker=broker,
        subscriptions=[{"topic": "labs"}],
        escalation_topic="alerts",
    )
    agents.append(lab_monitor)

    # 3. Pharmacy Agent — listens to orders, checks interactions
    pharmacy_agent = ReactiveAgent(
        name="Pharmacy",
        role="Pharmacy drug interaction checker",
        system_prompt=(
            "You are a pharmacist monitoring medication orders. "
            "Check for drug interactions and dosing appropriateness. "
            "Be concise (2-3 lines)."
        ),
        broker=broker,
        subscriptions=[{"topic": "orders", "filter": EventFilter(
            event_types=["medication_order", "medication_change"]
        )}],
    )
    agents.append(pharmacy_agent)

    # 4. Nursing Alert Agent — listens to alerts, creates care actions
    nursing_agent = ReactiveAgent(
        name="Nursing",
        role="Nursing response coordinator",
        system_prompt=(
            "You are a charge nurse. When you receive an alert, "
            "determine the appropriate nursing response and interventions. "
            "Be concise (2-3 lines)."
        ),
        broker=broker,
        subscriptions=[{"topic": "alerts"}],
    )
    agents.append(nursing_agent)

    # 5. Escalation Agent — listens to alerts, pages physicians for critical
    escalation_agent = ReactiveAgent(
        name="Escalation",
        role="Clinical escalation coordinator",
        system_prompt=(
            "You are a clinical escalation coordinator. When you receive "
            "critical alerts, determine who to page and what level of "
            "urgency. Be concise (2-3 lines)."
        ),
        broker=broker,
        subscriptions=[{"topic": "alerts", "filter": EventFilter(
            min_priority=EventPriority.HIGH
        )}],
    )
    agents.append(escalation_agent)

    # 6. Quality/Safety Agent — listens to everything, tracks patterns
    quality_agent = ReactiveAgent(
        name="Quality Tracker",
        role="Quality and safety tracker",
        system_prompt=(
            "You are a quality/safety officer. Track clinical events "
            "for patterns that might indicate systemic issues. "
            "Be concise (1-2 lines). Note any quality concerns."
        ),
        broker=broker,
        subscriptions=[
            {"topic": "vitals"},
            {"topic": "labs"},
            {"topic": "alerts"},
        ],
    )
    agents.append(quality_agent)

    return broker, agents


# ============================================================
# Demo Functions
# ============================================================

def demo_basic_pubsub():
    """Show basic publish-subscribe with clinical events."""
    print("\n" + "=" * 70)
    print("  DEMO 1: BASIC PUB/SUB — CLINICAL EVENT ROUTING")
    print("=" * 70)
    print("""
  Events are published to TOPICS. Agents subscribe to topics they care about.
  The event broker routes events — it doesn't process them.
  """)

    broker, agents = build_clinical_monitoring_system()

    print(f"\n  Agents and their subscriptions:")
    for agent in agents:
        print(f"    {agent.name} ({agent.role})")

    # Publish a vital signs event
    print(f"\n  ═══ Publishing vitals event ═══")
    broker.publish(
        ClinicalEvent(
            topic="vitals",
            event_type="vital_sign_change",
            source="Bedside Monitor",
            data={
                "patient_id": "P-12345",
                "BP": "158/92",
                "HR": 98,
                "SpO2": 96,
                "change": "HR increased from 82 to 98 in 15 minutes",
            },
            priority=EventPriority.HIGH,
        ),
        verbose=True,
    )

    # Show reactions
    print(f"\n  Agent reactions:")
    for agent in agents:
        for reaction in agent.reactions:
            preview = reaction["response"][:120].replace("\n", " | ")
            print(f"    [{agent.name}]: {preview}")

    stats = broker.get_stats()
    print(f"\n  Event stats: {dict(stats)}")


def demo_event_cascade():
    """Show how one event can trigger a cascade of reactions."""
    print("\n" + "=" * 70)
    print("  DEMO 2: EVENT CASCADE — ONE EVENT TRIGGERS MANY REACTIONS")
    print("=" * 70)
    print("""
  A critical lab result triggers:
  1. Lab Monitor reacts → publishes escalation to "alerts" topic
  2. Nursing reacts to the alert → plans interventions
  3. Escalation Agent reacts to critical alert → pages physician
  4. Quality Tracker logs the event

  All from ONE initial lab event — no one "orchestrates" this.
  The cascade EMERGES from subscriptions.
  """)

    broker, agents = build_clinical_monitoring_system()

    print(f"\n  Publishing critical lab result...")
    broker.publish(
        ClinicalEvent(
            topic="labs",
            event_type="critical_lab_result",
            source="Lab System",
            data={
                "patient_id": "P-12345",
                "test": "Troponin I",
                "value": 2.5,
                "reference_range": "0.00-0.04 ng/mL",
                "flag": "CRITICAL HIGH",
                "previous_value": 0.45,
                "trend": "Rising",
            },
            priority=EventPriority.CRITICAL,
        ),
        verbose=True,
    )

    # Show the cascade
    print(f"\n  ═══ EVENT CASCADE ═══")
    print(f"  Events published: {len(broker.event_history)}")
    for evt in broker.event_history:
        print(f"    {evt.event_type} (topic={evt.topic}, from={evt.source}, "
              f"priority={evt.priority.name})")

    print(f"\n  Agent reactions (in order):")
    all_reactions = []
    for agent in agents:
        for r in agent.reactions:
            all_reactions.append((agent.name, r))

    for agent_name, reaction in all_reactions:
        preview = reaction["response"][:120].replace("\n", " | ")
        print(f"    [{agent_name}] (reacted to {reaction['event']})")
        print(f"      → {preview}")


def demo_event_filtering():
    """Show how event filtering works."""
    print("\n" + "=" * 70)
    print("  DEMO 3: EVENT FILTERING — SELECTIVE REACTIONS")
    print("=" * 70)
    print("""
  Not every subscriber reacts to every event.
  Filters allow agents to only process events they care about:
  - Pharmacy: only reacts to 'medication_order' and 'medication_change'
  - Escalation: only reacts to HIGH+ priority alerts
  - Vitals Monitor: reacts to ALL vitals events

  Publishing 3 events with different types and priorities...
  """)

    broker, agents = build_clinical_monitoring_system()

    events = [
        ClinicalEvent(
            topic="orders",
            event_type="medication_order",
            source="Physician",
            data={"medication": "Heparin", "dose": "5000 units IV bolus"},
            priority=EventPriority.NORMAL,
        ),
        ClinicalEvent(
            topic="orders",
            event_type="diet_order",
            source="Physician",
            data={"diet": "NPO"},
            priority=EventPriority.LOW,
        ),
        ClinicalEvent(
            topic="alerts",
            event_type="info_alert",
            source="System",
            data={"message": "Patient due for scheduled vitals"},
            priority=EventPriority.LOW,
        ),
    ]

    for event in events:
        print(f"\n  Publishing: {event.event_type} (priority={event.priority.name})")
        broker.publish(event, verbose=True)

    # Show who reacted
    print(f"\n  ═══ REACTIONS ═══")
    for agent in agents:
        if agent.reactions:
            for r in agent.reactions:
                preview = r["response"][:100].replace("\n", " | ")
                print(f"    [{agent.name}] reacted to {r['event']}: {preview}")
        else:
            print(f"    [{agent.name}] — no reactions (filtered out)")

    stats = broker.get_stats()
    print(f"\n  Stats: published={stats.get('published', 0)}, "
          f"delivered={stats.get('delivered', 0)}, "
          f"filtered_out={stats.get('filtered_out', 0)}")


def demo_dead_letter():
    """Show dead-letter queue handling."""
    print("\n" + "=" * 70)
    print("  DEMO 4: DEAD-LETTER QUEUE — HANDLING DELIVERY FAILURES")
    print("=" * 70)
    print("""
  What happens when a subscriber fails to process an event?
  In production: the event goes to a DEAD-LETTER QUEUE for retry or review.
  The event is NOT lost, and other subscribers are NOT affected.
  """)

    broker = EventBroker()

    # Subscriber that always fails
    def failing_subscriber(event):
        raise Exception("SUBSCRIBER DOWN: Agent service unavailable for maintenance")

    # Subscriber that works
    results = []
    def working_subscriber(event):
        results.append(f"Processed {event.event_type} successfully")

    broker.subscribe("labs", failing_subscriber, subscriber_name="Down Agent")
    broker.subscribe("labs", working_subscriber, subscriber_name="Working Agent")

    print(f"\n  Publishing event to topic with 2 subscribers:")
    print(f"    - 'Down Agent' (will fail)")
    print(f"    - 'Working Agent' (will succeed)")

    broker.publish(
        ClinicalEvent(
            topic="labs",
            event_type="lab_result",
            source="Lab",
            data={"test": "CBC", "WBC": 8.5},
        ),
        verbose=True,
    )

    print(f"\n  Working Agent results: {results}")
    print(f"  Dead-letter queue: {len(broker.dead_letter_queue)} item(s)")

    for dlq in broker.dead_letter_queue:
        print(f"    Event: {dlq['event'].event_type}")
        print(f"    Subscriber: {dlq['subscriber']}")
        print(f"    Error: {dlq['error']}")

    print(f"\n  In production, dead-letter items would be:")
    print(f"    1. Retried after a delay")
    print(f"    2. Sent to a different endpoint")
    print(f"    3. Flagged for manual review")
    print(f"    4. Logged for monitoring dashboards")


def demo_full_clinical_scenario():
    """Run a full clinical scenario with multiple event cascades."""
    print("\n" + "=" * 70)
    print("  DEMO 5: FULL CLINICAL SCENARIO — ACS EVENT STREAM")
    print("=" * 70)
    print("""
  Simulate a sequence of clinical events for an ACS patient:
  1. Patient arrives → vitals recorded
  2. Labs result → troponin critical
  3. Medication ordered → heparin drip
  4. Follow-up vitals → deterioration

  Watch how the event-driven system handles each event independently.
  """)

    broker, agents = build_clinical_monitoring_system()

    clinical_events = [
        ClinicalEvent(
            topic="vitals",
            event_type="initial_vitals",
            source="ED Triage",
            data={
                "patient_id": "P-12345",
                "BP": "158/92", "HR": 98, "SpO2": 96,
                "notes": "Patient with chest pain, diaphoretic",
            },
            priority=EventPriority.HIGH,
        ),
        ClinicalEvent(
            topic="labs",
            event_type="critical_lab_result",
            source="Lab System",
            data={
                "patient_id": "P-12345",
                "test": "Troponin I",
                "value": 0.45,
                "flag": "CRITICAL HIGH",
                "reference": "0.00-0.04",
            },
            priority=EventPriority.CRITICAL,
        ),
        ClinicalEvent(
            topic="orders",
            event_type="medication_order",
            source="ED Physician",
            data={
                "patient_id": "P-12345",
                "medication": "Heparin",
                "dose": "60 units/kg IV bolus then 12 units/kg/hr",
                "indication": "ACS - NSTEMI",
            },
            priority=EventPriority.HIGH,
        ),
        ClinicalEvent(
            topic="vitals",
            event_type="vital_sign_change",
            source="Bedside Monitor",
            data={
                "patient_id": "P-12345",
                "BP": "90/58", "HR": 115, "SpO2": 91,
                "change": "Acute deterioration — hypotension and tachycardia",
            },
            priority=EventPriority.CRITICAL,
        ),
    ]

    for i, event in enumerate(clinical_events):
        print(f"\n  ━━━ Event {i + 1}: {event.event_type} (priority={event.priority.name}) ━━━")
        broker.publish(event, verbose=True)

        # Show immediate reactions
        new_reactions = []
        for agent in agents:
            for r in agent.reactions:
                if r not in [nr[1] for nr in new_reactions]:
                    new_reactions.append((agent.name, r))

        # Clear reactions to track only new ones per event
        time.sleep(0.1)  # Brief pause for readability

    # Summary
    print(f"\n\n  ═══ SCENARIO SUMMARY ═══")
    print(f"  Total events published: {len(broker.event_history)}")
    print(f"  Events by topic:")
    for key, count in broker.get_stats().items():
        if key.startswith("topic:"):
            print(f"    {key}: {count}")

    print(f"\n  Agent activity:")
    for agent in agents:
        print(f"    {agent.name}: {len(agent.reactions)} reaction(s)")
        for r in agent.reactions:
            preview = r["response"][:80].replace("\n", " | ")
            print(f"      → [{r['event']}] {preview}")


def demo_interactive():
    """Interactive mode to publish events."""
    print("\n" + "=" * 70)
    print("  DEMO 6: INTERACTIVE — PUBLISH YOUR OWN EVENTS")
    print("=" * 70)
    print("  Publish events and watch agents react.\n")

    broker, agents = build_clinical_monitoring_system()

    print("  Available topics: vitals, labs, orders, alerts")
    print("  Commands:")
    print("    vitals <BP> <HR> <SpO2>  — e.g., 'vitals 90/60 120 88'")
    print("    lab <test> <value>       — e.g., 'lab troponin 2.5'")
    print("    order <medication>       — e.g., 'order heparin'")
    print("    stats                    — show event statistics")
    print("    quit                     — exit\n")

    while True:
        user_input = input("  > ").strip()
        if user_input in ["quit", "exit", "q"]:
            break

        parts = user_input.split()
        if not parts:
            continue

        cmd = parts[0].lower()

        if cmd == "vitals" and len(parts) >= 4:
            broker.publish(ClinicalEvent(
                topic="vitals",
                event_type="vital_sign_change",
                source="User Input",
                data={"BP": parts[1], "HR": int(parts[2]), "SpO2": int(parts[3])},
                priority=EventPriority.HIGH if int(parts[3]) < 92 else EventPriority.NORMAL,
            ), verbose=True)

        elif cmd == "lab" and len(parts) >= 3:
            broker.publish(ClinicalEvent(
                topic="labs",
                event_type="critical_lab_result" if float(parts[2]) > 1.0 else "lab_result",
                source="User Input",
                data={"test": parts[1], "value": float(parts[2])},
                priority=EventPriority.CRITICAL if float(parts[2]) > 1.0 else EventPriority.NORMAL,
            ), verbose=True)

        elif cmd == "order" and len(parts) >= 2:
            broker.publish(ClinicalEvent(
                topic="orders",
                event_type="medication_order",
                source="User Input",
                data={"medication": " ".join(parts[1:])},
                priority=EventPriority.NORMAL,
            ), verbose=True)

        elif cmd == "stats":
            print(f"  Stats: {broker.get_stats()}")
            print(f"  Dead-letter: {len(broker.dead_letter_queue)} items")

        else:
            print("  Try: 'vitals 90/60 120 88' or 'lab troponin 2.5'")

        # Show new reactions
        for agent in agents:
            if agent.reactions:
                latest = agent.reactions[-1]
                preview = latest["response"][:100].replace("\n", " | ")
                print(f"    [{agent.name}]: {preview}")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 10: EVENT-DRIVEN / PUB-SUB AGENT SYSTEMS")
    print("=" * 70)
    print("""
    Event-driven architecture: agents publish events, other agents subscribe
    and react. No central controller. The workflow EMERGES from subscriptions.

    Choose a demo:
      1 → Basic Pub/Sub (clinical event routing)
      2 → Event cascade (one event triggers many reactions)
      3 → Event filtering (selective reactions)
      4 → Dead-letter queue (handling failures)
      5 → Full clinical scenario (ACS event stream)
      6 → Interactive (publish your own events)
      7 → Run demos 1-5
    """)

    choice = input("  Enter choice (1-7): ").strip()

    demos = {
        "1": demo_basic_pubsub,
        "2": demo_event_cascade,
        "3": demo_event_filtering,
        "4": demo_dead_letter,
        "5": demo_full_clinical_scenario,
        "6": demo_interactive,
    }

    if choice == "7":
        for demo in [demo_basic_pubsub, demo_event_cascade,
                      demo_event_filtering, demo_dead_letter,
                      demo_full_clinical_scenario]:
            demo()
    elif choice in demos:
        demos[choice]()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. PUB/SUB = ULTIMATE LOOSE COUPLING
   Publishers don't know subscribers. Subscribers don't know publishers.
   The broker just routes messages. This means you can add, remove,
   or replace agents WITHOUT changing any other code.

2. EVENTS vs COMMANDS:
   - COMMAND (orchestration): "Lab Agent, process this sample"
     → Directed at a specific agent, expects a response
   - EVENT (choreography): "Sample processed, result is X"
     → Published to a topic, zero or many agents may react

3. EVENT CASCADES ARE POWERFUL BUT DANGEROUS:
   One event can trigger reactions that publish more events that trigger
   more reactions. Without guards, you get infinite loops.
   Always design with circuit-breakers and max-depth limits.

4. DEAD-LETTER QUEUES PREVENT DATA LOSS:
   When a subscriber fails, the event goes to a DLQ instead of being lost.
   This is critical in healthcare — a missed critical lab alert could be fatal.

5. EVENT FILTERING REDUCES NOISE:
   Not every subscriber needs every event. Filters keep agents focused:
   - Type filter: pharmacy only reacts to medication events
   - Priority filter: escalation only reacts to HIGH+ priority
   - Custom filter: complex conditions (e.g., "only if troponin > 0.04")

6. TRADEOFF: DEBUGGING IS HARD
   In orchestration, you can trace the exact path: A → B → C.
   In event-driven systems, you need distributed tracing (correlation IDs,
   event chains, log aggregation) to understand what happened.

7. WHEN TO USE EVENT-DRIVEN:
   ✓ Many independent systems need to react to the same information
   ✓ You want to add new subscribers without modifying publishers
   ✓ Reactions are independent (one failing shouldn't block others)
   ✓ Real-time monitoring and alerting
   ✗ Strict ordering is required (use orchestration instead)
   ✗ You need request-response semantics (use direct calls)

8. PRODUCTION IMPLEMENTATIONS:
   Real Pub/Sub systems: Apache Kafka, RabbitMQ, AWS SNS/SQS,
   Google Cloud Pub/Sub, Redis Streams.
   Our in-memory EventBroker teaches the pattern;
   production would use a durable message broker.
"""

if __name__ == "__main__":
    main()
