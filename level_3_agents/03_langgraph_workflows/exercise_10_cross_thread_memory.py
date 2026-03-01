"""
Exercise 10: LangGraph InMemoryStore — Cross-Thread Shared Memory

Skills practiced:
- Using InMemoryStore for persistent memory across threads
- store.put() / store.get() / store.search() for key-value storage
- Sharing knowledge between different conversation threads
- Namespace-based organization of stored data
- Healthcare use case: shared patient knowledge base across sessions

Why this matters:
  Checkpointers save PER-THREAD state (one conversation's history).
  But what if multiple conversations need to SHARE information?

  InMemoryStore provides a key-value store that lives OUTSIDE threads.
  Any thread can read/write to it. This enables:
  - Patient records shared across clinician sessions
  - Accumulated knowledge from multiple consultations
  - Cross-session learning and context

Architecture:

  Checkpointer (per-thread):     InMemoryStore (cross-thread):
  ┌──────────┐  ┌──────────┐    ┌────────────────────────────┐
  │ Thread A  │  │ Thread B │    │      Shared Store          │
  │ messages: │  │ messages: │    │                            │
  │  [m1,m2]  │  │  [m3,m4] │    │  patient/P001 → {...}      │
  │           │  │          │    │  allergy/P001 → {...}      │
  └──────────┘  └──────────┘    │  lab/P001/trp → {...}      │
       ↑              ↑          └────────┬───────────────────┘
       │              │                   │
       └──────────────┴───── all threads read/write ──┘

  Checkpointer = conversation memory (what was said)
  InMemoryStore = knowledge memory (what was learned)
"""

import os
import json
import uuid
from datetime import datetime
from typing import Annotated, TypedDict
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# DEMO 1: Basic InMemoryStore — Put, Get, Search
# ============================================================

def demo_basic_store():
    """Show basic InMemoryStore operations."""
    print("\n" + "=" * 70)
    print("  DEMO 1: BASIC InMemoryStore — PUT, GET, SEARCH")
    print("=" * 70)
    print("""
  InMemoryStore is a namespace-based key-value store.
  - Namespace: like a folder path (tuple of strings)
  - Key: item ID within the namespace
  - Value: any dict

  store.put(namespace, key, value)
  store.get(namespace, key) → Item
  store.search(namespace) → list[Item]
  """)

    store = InMemoryStore()

    # Put: store patient data
    print("  ─── Storing patient data ───")
    store.put(
        namespace=("patients", "demographics"),
        key="P001",
        value={
            "name": "John Smith",
            "age": 65,
            "mrn": "MRN-001",
            "primary_dx": "Type 2 Diabetes",
        },
    )
    print("  ✅ Stored: patients/demographics/P001")

    store.put(
        namespace=("patients", "demographics"),
        key="P002",
        value={
            "name": "Jane Doe",
            "age": 45,
            "mrn": "MRN-002",
            "primary_dx": "Hypertension",
        },
    )
    print("  ✅ Stored: patients/demographics/P002")

    store.put(
        namespace=("patients", "allergies"),
        key="P001",
        value={
            "allergies": ["penicillin", "sulfa"],
            "severity": {"penicillin": "anaphylaxis", "sulfa": "rash"},
        },
    )
    print("  ✅ Stored: patients/allergies/P001")

    store.put(
        namespace=("patients", "allergies"),
        key="P002",
        value={"allergies": ["none known"], "severity": {}},
    )
    print("  ✅ Stored: patients/allergies/P002")

    # Get: retrieve specific item
    print("\n  ─── Retrieving data ───")
    item = store.get(("patients", "demographics"), "P001")
    print(f"  GET patients/demographics/P001:")
    print(f"    Value: {item.value}")
    print(f"    Key: {item.key}")

    # Search: list items in a namespace
    print("\n  ─── Searching namespace ───")
    results = store.search(("patients", "demographics"))
    print(f"  SEARCH patients/demographics: {len(results)} items")
    for r in results:
        print(f"    {r.key}: {r.value.get('name', 'N/A')} — {r.value.get('primary_dx', 'N/A')}")

    # Search allergies
    allergy_results = store.search(("patients", "allergies"))
    print(f"\n  SEARCH patients/allergies: {len(allergy_results)} items")
    for r in allergy_results:
        print(f"    {r.key}: {r.value.get('allergies', [])}")

    print(f"\n  KEY INSIGHT: InMemoryStore is a namespace+key store.")
    print(f"  Think of it as folders (namespaces) containing files (keys).")


# ============================================================
# DEMO 2: Cross-Thread Memory — Shared Patient Context
# ============================================================

def demo_cross_thread_memory():
    """Show how multiple threads share data through InMemoryStore."""
    print("\n" + "=" * 70)
    print("  DEMO 2: CROSS-THREAD MEMORY — SHARED PATIENT CONTEXT")
    print("=" * 70)
    print("""
  Two different clinician sessions (threads) sharing patient data:
  - Thread 1 (Morning nurse): Records vitals and allergies
  - Thread 2 (Afternoon doctor): Reads what the nurse stored

  The store persists across threads. The checkpointer is per-thread.
  """)

    store = InMemoryStore()
    checkpointer = MemorySaver()

    class ClinicalState(TypedDict):
        messages: Annotated[list, lambda x, y: x + y]
        patient_id: str

    def clinician_node(state: ClinicalState, store: InMemoryStore) -> dict:
        """Node that reads from and writes to the shared store."""
        patient_id = state.get("patient_id", "P001")
        last_msg = state["messages"][-1] if state["messages"] else None

        if not last_msg:
            return {"messages": [AIMessage(content="No message to process.")]}

        user_text = last_msg.content if isinstance(last_msg, HumanMessage) else ""

        # Check if this is a STORE command
        if user_text.lower().startswith("store:"):
            # Parse: "store: key=value"
            data_str = user_text[6:].strip()
            try:
                key, val = data_str.split("=", 1)
                key = key.strip()
                val = val.strip()

                store.put(
                    namespace=("patient_data", patient_id),
                    key=key,
                    value={"data": val, "recorded_at": datetime.now().isoformat()},
                )
                reply = f"Stored '{key}' for patient {patient_id}: {val}"
            except ValueError:
                reply = "Format: 'store: key=value'"

        elif user_text.lower().startswith("recall"):
            # Retrieve all stored data for this patient
            items = store.search(("patient_data", patient_id))
            if items:
                data_lines = []
                for item in items:
                    data_lines.append(
                        f"  {item.key}: {item.value['data']} "
                        f"(at {item.value.get('recorded_at', 'N/A')[:19]})"
                    )
                reply = f"Patient {patient_id} records:\n" + "\n".join(data_lines)
            else:
                reply = f"No stored data for patient {patient_id}."

        else:
            # Regular query — include stored context
            items = store.search(("patient_data", patient_id))
            context = ""
            if items:
                context = "Known patient data:\n" + "\n".join(
                    f"- {item.key}: {item.value['data']}" for item in items
                )

            response = llm.invoke([
                {"role": "system",
                 "content": f"You are a clinical assistant. {context}\n"
                            f"Answer the clinician's question concisely."},
                {"role": "user", "content": user_text},
            ])
            reply = response.content

        return {"messages": [AIMessage(content=reply)]}

    # Build graph
    graph = StateGraph(ClinicalState)
    graph.add_node("clinician", lambda state: clinician_node(state, store))
    graph.set_entry_point("clinician")
    graph.add_edge("clinician", END)

    app = graph.compile(checkpointer=checkpointer)

    # === Thread 1: Morning Nurse ===
    nurse_config = {"configurable": {"thread_id": "nurse-morning"}}
    print(f"\n  ═══ Thread 1: Morning Nurse ═══")

    nurse_msgs = [
        "store: vitals=BP 145/92, HR 88, T 98.6, SpO2 97%",
        "store: allergies=Penicillin (anaphylaxis), Sulfa (rash)",
        "store: medications=Metformin 1000mg BID, Lisinopril 20mg daily",
        "recall",
    ]

    for msg in nurse_msgs:
        result = app.invoke(
            {"messages": [HumanMessage(content=msg)], "patient_id": "P001"},
            nurse_config,
        )
        reply = result["messages"][-1].content
        print(f"  Nurse: {msg}")
        print(f"  System: {reply}\n")

    # === Thread 2: Afternoon Doctor ===
    doctor_config = {"configurable": {"thread_id": "doctor-afternoon"}}
    print(f"  ═══ Thread 2: Afternoon Doctor (DIFFERENT thread) ═══")

    doctor_msgs = [
        "recall",
        "Given the patient data, can I prescribe penicillin?",
        "store: assessment=Hypertension uncontrolled, increase lisinopril to 40mg",
    ]

    for msg in doctor_msgs:
        result = app.invoke(
            {"messages": [HumanMessage(content=msg)], "patient_id": "P001"},
            doctor_config,
        )
        reply = result["messages"][-1].content
        print(f"  Doctor: {msg}")
        print(f"  System: {reply}\n")

    # === Thread 3: Evening nurse sees EVERYTHING ===
    print(f"  ═══ Thread 3: Evening Nurse (sees ALL data) ═══")
    evening_config = {"configurable": {"thread_id": "nurse-evening"}}
    result = app.invoke(
        {"messages": [HumanMessage(content="recall")], "patient_id": "P001"},
        evening_config,
    )
    print(f"  Evening nurse recall:")
    print(f"  {result['messages'][-1].content}")

    print(f"\n  KEY INSIGHT: InMemoryStore persists across ALL threads.")
    print(f"  Nurse stored data → Doctor read it → Evening nurse sees everything.")
    print(f"  Checkpointer = per-thread history. Store = shared knowledge.")


# ============================================================
# DEMO 3: Namespaced Knowledge — Multiple Patients
# ============================================================

def demo_namespaced_knowledge():
    """Show namespace organization for multi-patient scenarios."""
    print("\n" + "=" * 70)
    print("  DEMO 3: NAMESPACED KNOWLEDGE — MULTI-PATIENT STORE")
    print("=" * 70)
    print("""
  Namespaces organize data hierarchically:
  ("patient", "P001", "labs")     → Patient P001's lab results
  ("patient", "P001", "meds")     → Patient P001's medications
  ("patient", "P002", "labs")     → Patient P002's lab results
  ("clinical", "guidelines")      → Shared clinical guidelines

  Each namespace is independent. You can search within a namespace
  or list items at any level.
  """)

    store = InMemoryStore()

    # Store data for multiple patients
    patients_data = {
        "P001": {
            "labs": [
                ("troponin-1", {"test": "troponin", "value": 0.45, "unit": "ng/mL", "time": "08:00"}),
                ("troponin-2", {"test": "troponin", "value": 0.22, "unit": "ng/mL", "time": "14:00"}),
                ("glucose-1", {"test": "glucose", "value": 185, "unit": "mg/dL", "time": "08:00"}),
            ],
            "meds": [
                ("heparin", {"drug": "heparin", "dose": "1000 units/hr", "route": "IV"}),
                ("aspirin", {"drug": "aspirin", "dose": "325mg", "route": "PO"}),
            ],
        },
        "P002": {
            "labs": [
                ("wbc-1", {"test": "WBC", "value": 18.5, "unit": "K/uL", "time": "09:00"}),
                ("lactate-1", {"test": "lactate", "value": 3.2, "unit": "mmol/L", "time": "09:00"}),
            ],
            "meds": [
                ("vancomycin", {"drug": "vancomycin", "dose": "1g", "route": "IV"}),
                ("zosyn", {"drug": "piperacillin-tazobactam", "dose": "4.5g", "route": "IV"}),
            ],
        },
    }

    # Store everything
    print("  ─── Storing multi-patient data ───")
    for patient_id, categories in patients_data.items():
        for category, items in categories.items():
            for key, value in items:
                store.put(
                    namespace=("patient", patient_id, category),
                    key=key,
                    value=value,
                )
    print("  ✅ Stored data for 2 patients (labs + meds)")

    # Query by patient and category
    print(f"\n  ─── Patient P001 Labs ───")
    p001_labs = store.search(("patient", "P001", "labs"))
    for item in p001_labs:
        v = item.value
        print(f"    {v['test']}: {v['value']} {v['unit']} at {v['time']}")

    print(f"\n  ─── Patient P001 Meds ───")
    p001_meds = store.search(("patient", "P001", "meds"))
    for item in p001_meds:
        v = item.value
        print(f"    {v['drug']}: {v['dose']} {v['route']}")

    print(f"\n  ─── Patient P002 Labs ───")
    p002_labs = store.search(("patient", "P002", "labs"))
    for item in p002_labs:
        v = item.value
        print(f"    {v['test']}: {v['value']} {v['unit']} at {v['time']}")

    print(f"\n  ─── Patient P002 Meds ───")
    p002_meds = store.search(("patient", "P002", "meds"))
    for item in p002_meds:
        v = item.value
        print(f"    {v['drug']}: {v['dose']} {v['route']}")

    # Store shared guidelines
    store.put(
        namespace=("clinical", "guidelines"),
        key="sepsis",
        value={"protocol": "Surviving Sepsis Campaign", "version": "2021"},
    )
    print(f"\n  ─── Shared guidelines ───")
    guidelines = store.search(("clinical", "guidelines"))
    for g in guidelines:
        print(f"    {g.key}: {g.value}")

    print(f"\n  KEY INSIGHT: Namespaces organize data hierarchically.")
    print(f"  ('patient', 'P001', 'labs') keeps each patient's data separate.")


# ============================================================
# DEMO 4: Store-Powered Agent — Knowledge Accumulation
# ============================================================

def demo_knowledge_agent():
    """Agent that accumulates knowledge across conversations."""
    print("\n" + "=" * 70)
    print("  DEMO 4: KNOWLEDGE AGENT — ACCUMULATES ACROSS SESSIONS")
    print("=" * 70)
    print("""
  An agent that gets SMARTER over time because it stores and retrieves
  knowledge from InMemoryStore. Each conversation adds to its knowledge.

  Session 1: Learns about a patient's cardiac history
  Session 2: Uses that knowledge when discussing medications
  Session 3: Has full context from both sessions
  """)

    store = InMemoryStore()
    checkpointer = MemorySaver()

    def knowledge_agent(state: MessagesState, *, store: InMemoryStore) -> dict:
        """Agent that reads/writes to shared store."""
        if not state["messages"]:
            return {"messages": [AIMessage(content="Ready.")]}

        last_msg = state["messages"][-1]
        user_text = last_msg.content if isinstance(last_msg, HumanMessage) else ""

        # Retrieve all stored knowledge
        knowledge_items = store.search(("knowledge", "clinical"))
        knowledge_context = ""
        if knowledge_items:
            knowledge_lines = [
                f"- [{item.key}] {json.dumps(item.value)}" for item in knowledge_items
            ]
            knowledge_context = (
                "Your accumulated knowledge:\n" + "\n".join(knowledge_lines) + "\n\n"
            )

        # Ask LLM
        response = llm.invoke([
            {"role": "system",
             "content": (
                 "You are a clinical knowledge agent. You accumulate knowledge across "
                 "conversations. When you learn something new, mention it clearly. "
                 "Use your accumulated knowledge to give better answers.\n\n"
                 f"{knowledge_context}"
                 "Be concise (2-3 sentences)."
             )},
            {"role": "user", "content": user_text},
        ])

        # Extract and store any new knowledge
        if any(kw in user_text.lower() for kw in [
            "patient", "diagnosed", "allergic", "started", "history", "takes",
            "prescribed", "lab", "result"
        ]):
            knowledge_key = f"fact-{len(knowledge_items) + 1}"
            store.put(
                namespace=("knowledge", "clinical"),
                key=knowledge_key,
                value={"fact": user_text, "learned_at": datetime.now().isoformat()[:19]},
            )
            stored_note = f"\n[Stored as {knowledge_key}]"
        else:
            stored_note = ""

        return {"messages": [AIMessage(content=response.content + stored_note)]}

    # Build graph
    graph = StateGraph(MessagesState)
    graph.add_node("agent", lambda state: knowledge_agent(state, store=store))
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)

    app = graph.compile(checkpointer=checkpointer)

    # Session 1: Learn about patient
    print(f"\n  ═══ Session 1: Initial Consult (thread: consult-1) ═══")
    session1_msgs = [
        "Patient John, 65M, was diagnosed with acute MI yesterday. Troponin peaked at 2.1.",
        "He has a history of type 2 diabetes and hypertension.",
        "He is allergic to penicillin — causes anaphylaxis.",
    ]
    config1 = {"configurable": {"thread_id": "consult-1"}}
    for msg in session1_msgs:
        result = app.invoke(
            {"messages": [HumanMessage(content=msg)]}, config1,
        )
        print(f"  📝 Session 1: {msg[:60]}...")
        print(f"  🤖 {result['messages'][-1].content[:120]}...\n")

    # Session 2: Different thread, uses accumulated knowledge
    print(f"  ═══ Session 2: Medication Review (thread: consult-2) ═══")
    session2_msgs = [
        "What do you know about this patient?",
        "Can we prescribe amoxicillin for a dental infection?",
        "Patient was started on clopidogrel 75mg daily and atorvastatin 80mg.",
    ]
    config2 = {"configurable": {"thread_id": "consult-2"}}
    for msg in session2_msgs:
        result = app.invoke(
            {"messages": [HumanMessage(content=msg)]}, config2,
        )
        print(f"  📝 Session 2: {msg[:60]}...")
        print(f"  🤖 {result['messages'][-1].content[:150]}...\n")

    # Session 3: Has ALL knowledge
    print(f"  ═══ Session 3: Follow-up (thread: consult-3) ═══")
    config3 = {"configurable": {"thread_id": "consult-3"}}
    result = app.invoke(
        {"messages": [HumanMessage(content="Summarize everything we know about this patient.")]},
        config3,
    )
    print(f"  📝 Session 3: Summarize everything we know")
    print(f"  🤖 {result['messages'][-1].content}")

    # Show what's in the store
    print(f"\n  ─── Store contents ───")
    all_knowledge = store.search(("knowledge", "clinical"))
    for item in all_knowledge:
        print(f"    {item.key}: {item.value['fact'][:60]}...")

    print(f"\n  KEY INSIGHT: The agent accumulates knowledge across threads.")
    print(f"  Session 1 stored facts → Session 2 used them → Session 3 has all.")


# ============================================================
# DEMO 5: Store vs Checkpointer — Understanding the Difference
# ============================================================

def demo_store_vs_checkpointer():
    """Clarify the difference between InMemoryStore and Checkpointer."""
    print("\n" + "=" * 70)
    print("  DEMO 5: STORE vs CHECKPOINTER — WHEN TO USE EACH")
    print("=" * 70)
    print("""
  ┌────────────────────────────────────────────────────────────┐
  │ Feature          │ Checkpointer       │ InMemoryStore      │
  ├──────────────────┼────────────────────┼────────────────────┤
  │ Scope            │ Per-thread          │ Cross-thread       │
  │ What it stores   │ Graph state/history │ Key-value data     │
  │ Access pattern   │ Automatic (built-in)│ Manual (put/get)   │
  │ Use case         │ Conversation memory │ Knowledge base     │
  │ Time travel      │ Yes (replay states) │ No (latest only)*  │
  │ Thread isolation  │ Yes                │ No (shared)        │
  └────────────────────────────────────────────────────────────┘

  * InMemoryStore overwrites on same key. Use versioned keys for history.

  Rule of thumb:
  - "What was said in THIS conversation?" → Checkpointer
  - "What do we KNOW across ALL conversations?" → InMemoryStore
  """)

    store = InMemoryStore()
    checkpointer = MemorySaver()

    def demo_node(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content="Noted.")]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", demo_node)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)
    app = graph.compile(checkpointer=checkpointer)

    # Checkpointer: per-thread
    print(f"\n  ─── Checkpointer (per-thread) ───")
    cfg_a = {"configurable": {"thread_id": "thread-A"}}
    cfg_b = {"configurable": {"thread_id": "thread-B"}}

    app.invoke({"messages": [HumanMessage(content="Hello from A")]}, cfg_a)
    app.invoke({"messages": [HumanMessage(content="Hello from B")]}, cfg_b)

    state_a = app.get_state(cfg_a)
    state_b = app.get_state(cfg_b)
    print(f"  Thread A messages: {len(state_a.values['messages'])} (isolated)")
    print(f"  Thread B messages: {len(state_b.values['messages'])} (isolated)")

    # InMemoryStore: shared
    print(f"\n  ─── InMemoryStore (cross-thread) ───")
    store.put(("shared",), "greeting", {"from": "thread-A", "msg": "Hello"})
    print(f"  Thread A stored: greeting = Hello")

    item = store.get(("shared",), "greeting")
    print(f"  Thread B reads: greeting = {item.value['msg']} (shared!)")

    print(f"\n  Checkpointer: Thread A can't see Thread B's messages.")
    print(f"  Store: Thread B CAN see what Thread A stored.")

    print(f"""
  WHEN TO USE EACH:
  ─────────────────
  Checkpointer ONLY:
    Simple chatbot, single-user conversations

  InMemoryStore ONLY:
    Shared configuration, reference data

  BOTH (most powerful):
    Multi-session clinical workflows where:
    - Each session has its own conversation history (checkpointer)
    - Patient knowledge accumulates across sessions (store)
  """)


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 10: LANGGRAPH InMemoryStore — CROSS-THREAD MEMORY")
    print("=" * 70)
    print("""
    InMemoryStore: shared key-value memory across threads.
    - store.put(namespace, key, value) — write
    - store.get(namespace, key) — read one
    - store.search(namespace) — list all in namespace

    Choose a demo:
      1 → Basic InMemoryStore (put, get, search)
      2 → Cross-thread memory (nurse stores → doctor reads)
      3 → Namespaced knowledge (multi-patient organization)
      4 → Knowledge agent (accumulates across sessions)
      5 → Store vs Checkpointer (when to use each)
      6 → Run all demos
    """)

    choice = input("  Enter choice (1-6): ").strip()

    demos = {
        "1": demo_basic_store,
        "2": demo_cross_thread_memory,
        "3": demo_namespaced_knowledge,
        "4": demo_knowledge_agent,
        "5": demo_store_vs_checkpointer,
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

1. InMemoryStore = CROSS-THREAD SHARED MEMORY
   Unlike checkpointers (per-thread), InMemoryStore is shared.
   Thread A stores data → Thread B reads it. All threads share one store.

2. NAMESPACE-BASED ORGANIZATION:
   Namespaces are tuples: ("patient", "P001", "labs")
   Think of them as directory paths for organizing data.
   Search within any namespace level.

3. store.put() / store.get() / store.search():
   - put(namespace, key, value): Store a dict
   - get(namespace, key): Retrieve one item
   - search(namespace): List all items in namespace

4. CHECKPOINTER vs STORE:
   - Checkpointer: "What happened in THIS conversation?" (per-thread)
   - Store: "What do we KNOW across ALL conversations?" (shared)
   - Use BOTH for the most powerful pattern.

5. KNOWLEDGE ACCUMULATION PATTERN:
   Each conversation extracts and stores facts.
   Future conversations retrieve stored facts for context.
   The agent gets smarter with each interaction.

6. HEALTHCARE USE CASES:
   - Patient records shared across clinician sessions
   - Accumulated clinical findings across visits
   - Shared allergy/medication data across departments
   - Cross-shift handoff information
   - Clinical decision support with growing knowledge base
"""

if __name__ == "__main__":
    main()
