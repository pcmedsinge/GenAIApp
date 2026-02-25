"""
Exercise 2: Persistent Conversation Memory

Skills practiced:
- Saving conversation history to disk (JSON file)
- Loading and resuming conversations across program restarts
- Managing multiple conversation sessions
- Understanding memory patterns for healthcare continuity

Key insight: In healthcare, patient conversations span days/weeks.
  Persisting memory lets the agent recall "last visit the potassium
  was 5.4 on lisinopril" — critical for continuity of care.
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent as create_langchain_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# Tools (from main.py)
# ============================================================

@tool
def lookup_medication(medication_name: str) -> str:
    """Look up medication information. Available: metformin, lisinopril, amlodipine, apixaban, sertraline, omeprazole."""
    medications = {
        "metformin": "Class: Biguanide | Indication: Type 2 Diabetes | Dose: 500-2000mg daily | Contraindicated: eGFR<30",
        "lisinopril": "Class: ACE Inhibitor | Indication: HTN, HF | Dose: 10-40mg daily | Side Effects: Dry cough, hyperkalemia",
        "amlodipine": "Class: CCB | Indication: HTN, Angina | Dose: 2.5-10mg daily",
        "apixaban": "Class: DOAC | Indication: AFib, VTE | Dose: 5mg BID",
        "sertraline": "Class: SSRI | Indication: Depression, Anxiety | Dose: 50-200mg daily",
        "omeprazole": "Class: PPI | Indication: GERD | Dose: 20-40mg daily",
    }
    result = medications.get(medication_name.lower())
    return result if result else f"Not found. Available: {', '.join(medications.keys())}"


@tool
def check_lab_value(test_name: str, value: float) -> str:
    """Interpret a lab value. Available: hba1c, gfr, potassium, creatinine, hemoglobin."""
    tests = {
        "hba1c": lambda v: f"HbA1c {v}%: {'Normal' if v < 5.7 else 'Prediabetes' if v < 6.5 else 'Diabetes'}",
        "gfr": lambda v: f"GFR {v}: {'Normal' if v >= 90 else 'Mild CKD' if v >= 60 else 'Mod CKD' if v >= 30 else 'Severe CKD'}",
        "potassium": lambda v: f"K+ {v}: {'LOW' if v < 3.5 else 'Normal' if v <= 5.0 else 'HIGH'}",
        "creatinine": lambda v: f"Cr {v}: {'Normal' if 0.7 <= v <= 1.3 else 'Abnormal'}",
        "hemoglobin": lambda v: f"Hb {v}: {'Low' if v < 12 else 'Normal' if v <= 17 else 'Elevated'}",
    }
    fn = tests.get(test_name.lower())
    return fn(value) if fn else f"Not found. Available: {', '.join(tests.keys())}"


@tool
def check_drug_interaction(drug1: str, drug2: str) -> str:
    """Check for interactions between two medications."""
    interactions = {
        ("lisinopril", "potassium"): "MAJOR: Both increase K+ → hyperkalemia risk.",
        ("sertraline", "tramadol"): "MAJOR: Serotonin syndrome risk.",
        ("apixaban", "aspirin"): "MAJOR: Increased bleeding risk.",
        ("metformin", "contrast_dye"): "MODERATE: Hold metformin 48h around contrast.",
    }
    key1, key2 = (drug1.lower(), drug2.lower()), (drug2.lower(), drug1.lower())
    result = interactions.get(key1) or interactions.get(key2)
    return result if result else f"No known interaction between {drug1} and {drug2}."


healthcare_tools = [lookup_medication, check_lab_value, check_drug_interaction]


# ============================================================
# Persistent Memory Manager
# ============================================================

class PersistentMemory:
    """
    Saves and loads conversation history to/from JSON files.

    Healthcare parallel: Electronic Health Records persist patient data
    across visits. This memory does the same for agent conversations.
    """

    def __init__(self, storage_dir=None):
        if storage_dir is None:
            storage_dir = os.path.join(os.path.dirname(__file__), "conversation_history")
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def _filepath(self, session_id):
        return os.path.join(self.storage_dir, f"{session_id}.json")

    def save(self, session_id, chat_history, metadata=None):
        """Save conversation to disk"""
        data = {
            "session_id": session_id,
            "updated_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "messages": []
        }
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                data["messages"].append({"role": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                data["messages"].append({"role": "ai", "content": msg.content})

        with open(self._filepath(session_id), "w") as f:
            json.dump(data, f, indent=2)

    def load(self, session_id):
        """Load conversation from disk"""
        path = self._filepath(session_id)
        if not os.path.exists(path):
            return [], {}

        with open(path, "r") as f:
            data = json.load(f)

        chat_history = []
        for msg in data["messages"]:
            if msg["role"] == "human":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                chat_history.append(AIMessage(content=msg["content"]))

        return chat_history, data.get("metadata", {})

    def list_sessions(self):
        """List all saved sessions"""
        sessions = []
        for f in os.listdir(self.storage_dir):
            if f.endswith(".json"):
                path = os.path.join(self.storage_dir, f)
                with open(path, "r") as fp:
                    data = json.load(fp)
                sessions.append({
                    "session_id": data["session_id"],
                    "updated_at": data["updated_at"],
                    "message_count": len(data["messages"]),
                    "metadata": data.get("metadata", {}),
                })
        return sorted(sessions, key=lambda s: s["updated_at"], reverse=True)

    def delete(self, session_id):
        """Delete a saved session"""
        path = self._filepath(session_id)
        if os.path.exists(path):
            os.remove(path)

    def clear_all(self):
        """Delete all sessions"""
        for f in os.listdir(self.storage_dir):
            if f.endswith(".json"):
                os.remove(os.path.join(self.storage_dir, f))


def create_agent():
    return create_langchain_agent(
        llm,
        tools=healthcare_tools,
        system_prompt="""You are a clinical decision support agent with memory.
You remember the FULL conversation history including previous sessions.
Reference previous findings when relevant. Use tools when needed.
Educational purposes only."""
    )


# ============================================================
# DEMO 1: Save & Load Conversation
# ============================================================

def demo_save_load():
    """Show saving and loading a conversation"""
    print("\n" + "=" * 70)
    print("DEMO 1: SAVE & LOAD CONVERSATION")
    print("=" * 70)
    print("""
    Simulate two separate "sessions" with the same patient.
    Session 1: Check labs and medication.
    Session 2: Resume and ask follow-up questions.
    """)

    memory = PersistentMemory()
    agent = create_agent()
    session_id = "patient_jones_001"

    # ----- Session 1 -----
    print(f"\n{'═' * 60}")
    print("  SESSION 1 — Initial Assessment")
    print(f"{'═' * 60}")

    chat_history = []
    session1_questions = [
        "Check potassium level: 5.4 mEq/L",
        "This patient is on lisinopril 20mg. Is the potassium a concern?",
    ]

    for q in session1_questions:
        print(f"\n  Doctor: {q}")
        chat_history.append({"role": "user", "content": q})
        result = agent.invoke({"messages": chat_history})
        answer = result["messages"][-1].content
        print(f"  Agent: {answer[:300]}")
        chat_history = result["messages"]

    # Save — convert messages to HumanMessage/AIMessage for persistence
    save_history = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            save_history.append(msg)
        elif isinstance(msg, AIMessage) and msg.content:
            save_history.append(msg)
    memory.save(session_id, save_history, metadata={"patient": "Jones", "date": "2025-02-22"})
    print(f"\n  [Saved {len(save_history)} messages to disk]")

    # ----- Session 2 (resume) -----
    print(f"\n{'═' * 60}")
    print("  SESSION 2 — Follow-up (loaded from disk)")
    print(f"{'═' * 60}")

    loaded_history, meta = memory.load(session_id)
    print(f"  [Loaded {len(loaded_history)} messages, patient: {meta.get('patient', '?')}]")

    # Convert loaded HumanMessage/AIMessage back to dicts for the agent
    messages = []
    for msg in loaded_history:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})

    session2_questions = [
        "What was the potassium level we checked last time? And what did you recommend?",
        "The patient's repeat potassium came back at 4.8. How does that compare?",
    ]

    for q in session2_questions:
        print(f"\n  Doctor: {q}")
        messages.append({"role": "user", "content": q})
        result = agent.invoke({"messages": messages})
        answer = result["messages"][-1].content
        print(f"  Agent: {answer[:300]}")
        messages = [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                     for m in result["messages"] if isinstance(m, (HumanMessage, AIMessage)) and m.content]

    # Save updated — convert back to message objects
    updated_history = []
    for m in messages:
        if m["role"] == "user":
            updated_history.append(HumanMessage(content=m["content"]))
        else:
            updated_history.append(AIMessage(content=m["content"]))
    memory.save(session_id, updated_history, metadata=meta)
    print(f"\n  [Updated session: now {len(updated_history)} messages]")

    # Clean up
    memory.delete(session_id)
    print("  [Cleaned up test data]")


# ============================================================
# DEMO 2: Multiple Sessions
# ============================================================

def demo_multiple_sessions():
    """Manage multiple patient sessions"""
    print("\n" + "=" * 70)
    print("DEMO 2: MULTIPLE PATIENT SESSIONS")
    print("=" * 70)

    memory = PersistentMemory()
    agent = create_agent()

    patients = {
        "patient_smith": {
            "metadata": {"patient": "Smith", "age": 65, "conditions": ["HTN", "CKD"]},
            "questions": [
                "Look up lisinopril for this hypertension patient.",
                "Check GFR of 42 — what stage CKD?",
            ]
        },
        "patient_garcia": {
            "metadata": {"patient": "Garcia", "age": 52, "conditions": ["T2DM"]},
            "questions": [
                "HbA1c of 7.8%, what interpretation?",
                "Look up metformin for diabetes management.",
            ]
        },
        "patient_chen": {
            "metadata": {"patient": "Chen", "age": 78, "conditions": ["AFib"]},
            "questions": [
                "Look up apixaban for an atrial fibrillation patient.",
            ]
        },
    }

    # Create sessions
    for session_id, patient_data in patients.items():
        chat_history = []
        messages = []
        print(f"\n  --- {patient_data['metadata']['patient']} ---")
        for q in patient_data["questions"]:
            messages.append({"role": "user", "content": q})
            result = agent.invoke({"messages": messages})
            answer = result["messages"][-1].content
            messages = [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                         for m in result["messages"] if isinstance(m, (HumanMessage, AIMessage)) and m.content]
            chat_history.append(HumanMessage(content=q))
            chat_history.append(AIMessage(content=answer))
            print(f"    Q: {q[:50]}... → Answered")
        memory.save(session_id, chat_history, metadata=patient_data["metadata"])

    # List all sessions
    print(f"\n{'─' * 60}")
    print("  SAVED SESSIONS:")
    print(f"{'─' * 60}")
    for s in memory.list_sessions():
        print(f"  ID: {s['session_id']:<20} Messages: {s['message_count']:<4} "
              f"Patient: {s['metadata'].get('patient', '?')}")

    # Clean up
    memory.clear_all()
    print(f"\n  [Cleaned up all test sessions]")


# ============================================================
# DEMO 3: Memory Impact
# ============================================================

def demo_memory_impact():
    """Show agent answers differently with vs without memory"""
    print("\n" + "=" * 70)
    print("DEMO 3: MEMORY IMPACT — WITH vs WITHOUT")
    print("=" * 70)
    print("""
    Same follow-up question asked with and without memory.
    Shows how memory enables clinical continuity.
    """)

    agent = create_agent()

    # Build history
    setup_questions = [
        "Check potassium level of 5.6",
        "Patient is on lisinopril 20mg daily",
    ]
    messages = []
    for q in setup_questions:
        messages.append({"role": "user", "content": q})
        result = agent.invoke({"messages": messages})
        messages = [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                     for m in result["messages"] if isinstance(m, (HumanMessage, AIMessage)) and m.content]
        print(f"  Setup: {q}")

    follow_up = "Given what we discussed, should we adjust the medication?"

    # WITH memory
    print(f"\n{'─' * 60}")
    print(f"  Follow-up: \"{follow_up}\"")
    print(f"{'─' * 60}")

    messages_with = messages + [{"role": "user", "content": follow_up}]
    result_with = agent.invoke({"messages": messages_with})
    print(f"\n  WITH MEMORY:\n  {result_with['messages'][-1].content[:400]}")

    # WITHOUT memory
    result_without = agent.invoke({"messages": [{"role": "user", "content": follow_up}]})
    print(f"\n  WITHOUT MEMORY:\n  {result_without['messages'][-1].content[:400]}")

    print("""
    OBSERVATION:
      • WITH memory: Agent references K+ 5.6 and lisinopril specifically.
      • WITHOUT memory: Agent gives a generic answer — it doesn't know
        what medication or what lab was discussed!
      • This is why persistence matters in clinical workflows.
    """)


# ============================================================
# DEMO 4: Interactive with Persistent Memory
# ============================================================

def demo_interactive():
    """Interactive session with save/load"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE WITH PERSISTENT MEMORY")
    print("=" * 70)
    print("  Commands: 'save' — save session, 'load' — load session")
    print("  'sessions' — list saved, 'new' — start fresh, 'quit' — exit\n")

    memory = PersistentMemory()
    agent = create_agent()
    messages = []
    current_session = f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    while True:
        inp = input("  You: ").strip()
        if not inp:
            continue

        if inp.lower() in ['quit', 'exit', 'q']:
            break
        elif inp.lower() == 'save':
            # Convert current messages to HumanMessage/AIMessage for storage
            save_hist = []
            for m in messages:
                if isinstance(m, dict):
                    if m["role"] == "user":
                        save_hist.append(HumanMessage(content=m["content"]))
                    else:
                        save_hist.append(AIMessage(content=m["content"]))
                elif isinstance(m, HumanMessage):
                    save_hist.append(m)
                elif isinstance(m, AIMessage) and m.content:
                    save_hist.append(m)
            memory.save(current_session, save_hist)
            print(f"  [Saved as '{current_session}' ({len(save_hist)} msgs)]")
            continue
        elif inp.lower() == 'load':
            sessions = memory.list_sessions()
            if not sessions:
                print("  No saved sessions.")
                continue
            for i, s in enumerate(sessions):
                print(f"    {i+1}. {s['session_id']} ({s['message_count']} msgs)")
            idx = input("  Load which? ").strip()
            try:
                s = sessions[int(idx) - 1]
                loaded_history, _ = memory.load(s["session_id"])
                current_session = s["session_id"]
                # Convert loaded messages to dict format for agent
                messages = []
                for msg in loaded_history:
                    if isinstance(msg, HumanMessage):
                        messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        messages.append({"role": "assistant", "content": msg.content})
                print(f"  [Loaded '{current_session}' ({len(messages)} msgs)]")
            except (ValueError, IndexError):
                print("  Invalid selection.")
            continue
        elif inp.lower() == 'sessions':
            for s in memory.list_sessions():
                print(f"    {s['session_id']} ({s['message_count']} msgs, {s['updated_at'][:10]})")
            continue
        elif inp.lower() == 'new':
            messages = []
            current_session = f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"  [New session: {current_session}]")
            continue

        messages.append({"role": "user", "content": inp})
        result = agent.invoke({"messages": messages})
        answer = result["messages"][-1].content
        print(f"\n  Agent: {answer}\n")
        messages = [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                     for m in result["messages"] if isinstance(m, (HumanMessage, AIMessage)) and m.content]


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 2: PERSISTENT CONVERSATION MEMORY")
    print("=" * 70)
    print("""
    Save and load conversations across program restarts.
    Like patient chart notes that persist between visits.

    Choose a demo:
      1 → Save & Load (two sessions)
      2 → Multiple patient sessions
      3 → Memory impact (with vs without)
      4 → Interactive with save/load
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1": demo_save_load()
    elif choice == "2": demo_multiple_sessions()
    elif choice == "3": demo_memory_impact()
    elif choice == "4": demo_interactive()
    elif choice == "5":
        demo_save_load()
        demo_multiple_sessions()
        demo_memory_impact()
    else:
        print("  Invalid choice.")

    # Final cleanup
    storage_dir = os.path.join(os.path.dirname(__file__), "conversation_history")
    if os.path.exists(storage_dir):
        import shutil
        shutil.rmtree(storage_dir)


"""
KEY LEARNINGS:
=============

1. SERIALIZE MESSAGES: Convert HumanMessage/AIMessage to JSON dicts
   for storage, then reconstruct on load. The pattern:
   save: message → {"role": "human", "content": "..."}
   load: {"role": "human", ...} → HumanMessage(content="...")

2. SESSION MANAGEMENT: Each patient/conversation gets a session ID.
   Multiple sessions can be saved and listed — like patient records.

3. MEMORY ENABLES CONTINUITY: Without memory, the agent can't reference
   previous findings. With persistent memory, conversations span
   sessions — critical for multi-visit patient care.

4. PRODUCTION CONSIDERATIONS:
   - JSON files work for learning, but production uses databases
   - Consider encryption for PHI (Protected Health Information)
   - Add session expiry and access controls
   - LangChain also has built-in memory backends (Redis, SQL, etc.)
"""

if __name__ == "__main__":
    main()
