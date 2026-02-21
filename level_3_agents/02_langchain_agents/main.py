"""
Project 2: LangChain Agents
Use the LangChain framework to build agents with custom tools and memory.

LangChain wraps the ReAct pattern (Project 01) with:
- @tool decorator for easy tool creation
- AgentExecutor for managing the agent loop
- Memory for multi-turn conversations
- Error handling and retries built-in

Builds on: Project 01 (ReAct from scratch)
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================
# Initialize LLM
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# Custom Healthcare Tools (using @tool decorator)
# ============================================================

@tool
def lookup_medication(medication_name: str) -> str:
    """Look up detailed information about a medication including class, dosing, side effects, and contraindications.
    Available medications: metformin, lisinopril, amlodipine, apixaban, sertraline, omeprazole."""
    medications = {
        "metformin": "Class: Biguanide | Indication: Type 2 Diabetes | Dose: 500-2000mg daily | Side Effects: GI upset | Contraindicated: eGFR<30 | Monitor: HbA1c, B12, renal function",
        "lisinopril": "Class: ACE Inhibitor | Indication: HTN, HF, Diabetic Nephropathy | Dose: 10-40mg daily | Side Effects: Dry cough, hyperkalemia | Contraindicated: Pregnancy | Monitor: BP, K+, Cr",
        "amlodipine": "Class: CCB | Indication: HTN, Angina | Dose: 2.5-10mg daily | Side Effects: Edema, dizziness | Monitor: BP, heart rate",
        "apixaban": "Class: DOAC | Indication: AFib, VTE | Dose: 5mg BID (2.5mg if age 80+/weight ≤60kg/Cr≥1.5) | Side Effects: Bleeding | Contraindicated: Active bleeding, mechanical valve",
        "sertraline": "Class: SSRI | Indication: Depression, Anxiety, PTSD | Dose: 50-200mg daily | Side Effects: Nausea, insomnia | Monitor: Mood, suicidal ideation (age<25)",
        "omeprazole": "Class: PPI | Indication: GERD, Peptic ulcer | Dose: 20-40mg daily | Side Effects: C. diff risk, B12 def, fracture risk long-term | Monitor: Mg2+ if long-term"
    }
    result = medications.get(medication_name.lower())
    if result:
        return result
    return f"Medication '{medication_name}' not found. Available: {', '.join(medications.keys())}"


@tool
def check_lab_value(test_name: str, value: float) -> str:
    """Interpret a lab value and determine clinical significance.
    Available tests: hba1c (%), gfr (mL/min), potassium (mEq/L), creatinine (mg/dL), inr, hemoglobin (g/dL)."""
    interpretations = {
        "hba1c": lambda v: f"HbA1c {v}%: {'Normal' if v < 5.7 else 'Prediabetes' if v < 6.5 else 'Diabetes'} (target <7% for most diabetics)",
        "gfr": lambda v: f"GFR {v}: {'Normal' if v >= 90 else 'Mild decrease' if v >= 60 else 'Stage 3 CKD' if v >= 30 else 'Stage 4 CKD' if v >= 15 else 'Stage 5 - Kidney failure'}",
        "potassium": lambda v: f"K+ {v}: {'LOW - risk of arrhythmia' if v < 3.5 else 'Normal' if v <= 5.0 else 'HIGH - risk of arrhythmia' if v <= 6.0 else 'CRITICAL HIGH'}",
        "creatinine": lambda v: f"Creatinine {v}: {'Normal' if 0.7 <= v <= 1.3 else 'Abnormal - evaluate renal function'}",
        "inr": lambda v: f"INR {v}: {'Subtherapeutic' if v < 2.0 else 'Therapeutic (2-3)' if v <= 3.0 else 'Supratherapeutic - bleeding risk' if v <= 4.0 else 'CRITICAL - high bleeding risk'}",
        "hemoglobin": lambda v: f"Hb {v}: {'Low - Anemia' if v < 12 else 'Normal' if v <= 17 else 'Elevated - evaluate'}",
    }
    fn = interpretations.get(test_name.lower())
    if fn:
        return fn(value)
    return f"Test '{test_name}' not recognized. Available: {', '.join(interpretations.keys())}"


@tool
def check_drug_interaction(drug1: str, drug2: str) -> str:
    """Check for interactions between two medications."""
    interactions = {
        ("lisinopril", "potassium"): "MAJOR: Both increase potassium levels → hyperkalemia risk. Monitor K+ closely.",
        ("lisinopril", "nsaids"): "MODERATE: NSAIDs reduce ACEi effectiveness and increase renal risk.",
        ("apixaban", "aspirin"): "MAJOR: Increased bleeding risk. Only combine if strong indication (e.g., recent ACS).",
        ("sertraline", "tramadol"): "MAJOR: Serotonin syndrome risk. Avoid combination.",
        ("sertraline", "nsaids"): "MODERATE: Increased GI bleeding risk. Consider PPI.",
        ("metformin", "contrast_dye"): "MODERATE: Hold metformin 48h before and after IV contrast. Risk of lactic acidosis.",
        ("warfarin", "omeprazole"): "MINOR: Omeprazole may slightly increase INR. Monitor.",
    }

    key1 = (drug1.lower(), drug2.lower())
    key2 = (drug2.lower(), drug1.lower())

    result = interactions.get(key1) or interactions.get(key2)
    if result:
        return result
    return f"No known significant interaction between {drug1} and {drug2} in database."


@tool
def calculate_bmi(weight_kg: float, height_cm: float) -> str:
    """Calculate BMI and interpret the result. Weight in kilograms, height in centimeters."""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    elif bmi < 35:
        category = "Obesity Class I"
    elif bmi < 40:
        category = "Obesity Class II"
    else:
        category = "Obesity Class III (Severe)"

    return f"BMI: {bmi:.1f} ({category}). Weight: {weight_kg}kg, Height: {height_cm}cm"


# All tools available to agent
healthcare_tools = [lookup_medication, check_lab_value, check_drug_interaction, calculate_bmi]


# ============================================================
# DEMO 1: Custom Tools Showcase
# ============================================================

def demo_custom_tools():
    """Show how @tool decorator works"""
    print("\n" + "=" * 70)
    print("DEMO 1: CUSTOM TOOLS WITH @tool DECORATOR")
    print("=" * 70)

    print("""
💡 LangChain's @tool decorator:
   - Wraps any Python function into a tool the agent can call
   - Docstring becomes the tool DESCRIPTION (how agent knows when to use it)
   - Type hints become the PARAMETERS (how agent knows what to send)
   - Same concept as OpenAI function definitions in Level 1, just easier syntax!
""")

    print("📋 Registered tools:")
    for t in healthcare_tools:
        print(f"\n   🔧 {t.name}")
        print(f"      Description: {t.description[:100]}...")

    print(f"\n\n🧪 Testing tools directly:")
    print(f"   lookup_medication('metformin'):")
    print(f"   → {lookup_medication.invoke('metformin')}")
    print(f"\n   check_lab_value(test_name='gfr', value=42.0):")
    print(f"   → {check_lab_value.invoke({'test_name': 'gfr', 'value': 42.0})}")
    print(f"\n   check_drug_interaction(drug1='lisinopril', drug2='nsaids'):")
    print(f"   → {check_drug_interaction.invoke({'drug1': 'lisinopril', 'drug2': 'nsaids'})}")


# ============================================================
# DEMO 2: Agent with Tools
# ============================================================

def demo_agent_with_tools():
    """Create and run a LangChain agent"""
    print("\n" + "=" * 70)
    print("DEMO 2: LangChain AGENT WITH TOOLS")
    print("=" * 70)

    # Create agent prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a clinical decision support agent with access to medical tools.
Use your tools to look up specific medication info, interpret lab values,
check drug interactions, and calculate BMI when needed.
Always explain your clinical reasoning. For educational purposes only."""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create the agent
    agent = create_tool_calling_agent(llm, healthcare_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=healthcare_tools, verbose=True)

    questions = [
        "What monitoring is needed for a patient starting lisinopril?",
        "A patient on sertraline needs pain management. Can they take tramadol?",
        "Patient is 85kg, 170cm tall with HbA1c of 7.8% — assess and recommend.",
    ]

    for q in questions:
        print(f"\n{'─' * 70}")
        print(f"❓ {q}\n")
        result = agent_executor.invoke({"input": q})
        print(f"\n📋 FINAL: {result['output']}")


# ============================================================
# DEMO 3: Agent with Memory
# ============================================================

def demo_agent_with_memory():
    """Agent that remembers previous interactions"""
    print("\n" + "=" * 70)
    print("DEMO 3: AGENT WITH MEMORY")
    print("=" * 70)
    print("""
💡 Memory allows multi-turn conversations:
   Turn 1: "Check if potassium 5.4 is normal"
   Turn 2: "The patient is also on lisinopril — any concern?"
   Agent remembers the potassium result from Turn 1!
""")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a clinical decision support agent with memory.
You remember the full conversation history. Use your tools when needed.
Reference previous findings in your answers. Educational purposes only."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, healthcare_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=healthcare_tools, verbose=True)

    # Simulate multi-turn conversation
    chat_history = []

    conversation = [
        "Check the potassium level of 5.4 for a 70-year-old patient.",
        "The same patient is on lisinopril. Is this a concern with the potassium level?",
        "What should we do? Should we stop the lisinopril?",
    ]

    for turn, question in enumerate(conversation, 1):
        print(f"\n{'═' * 70}")
        print(f"Turn {turn}: \"{question}\"")
        print(f"{'═' * 70}")

        result = agent_executor.invoke({
            "input": question,
            "chat_history": chat_history
        })

        print(f"\n📋 Agent: {result['output']}")

        # Add to history for next turn
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=result["output"]))

    print(f"""
💡 MEMORY VALUE:
   • Turn 2 referenced Turn 1's potassium finding
   • Turn 3 built on both Turn 1 (potassium) and Turn 2 (lisinopril)
   • Without memory, each turn would be isolated — no continuity
""")


# ============================================================
# DEMO 4: Interactive Agent
# ============================================================

def demo_interactive():
    """Chat with the agent yourself"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE MEDICAL AGENT")
    print("=" * 70)
    print("\n💬 Chat with the agent! It has memory across turns.")
    print("   Tools: medication lookup, lab interpretation, drug interactions, BMI")
    print("   Type 'quit' to exit\n")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful clinical decision support agent. Use tools when needed. Educational purposes only."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, healthcare_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=healthcare_tools, verbose=False)
    chat_history = []

    while True:
        question = input("You: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue

        result = agent_executor.invoke({
            "input": question,
            "chat_history": chat_history
        })

        print(f"\nAgent: {result['output']}\n")
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=result["output"]))


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n🔗 Level 3, Project 2: LangChain Agents")
    print("=" * 70)
    print("Build agents with the most popular GenAI framework\n")

    print("Choose a demo:")
    print("1. Custom tools (@tool decorator)")
    print("2. Agent with tools (watch LangChain agent work)")
    print("3. Agent with memory (multi-turn conversation)")
    print("4. Interactive agent (chat yourself)")
    print("5. Run demos 1-3")

    choice = input("\nEnter choice (1-5): ").strip()

    demos = {"1": demo_custom_tools, "2": demo_agent_with_tools,
             "3": demo_agent_with_memory, "4": demo_interactive}

    if choice == "5":
        demo_custom_tools()
        demo_agent_with_tools()
        demo_agent_with_memory()
    elif choice in demos:
        demos[choice]()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY TAKEAWAYS
{'=' * 70}

🔧 @tool DECORATOR:
   • Wraps Python functions into agent tools
   • Docstring = description (agent reads this to decide when to use)
   • Type hints = parameter schema
   • Same as OpenAI function definitions, but easier syntax

🤖 LANGCHAIN AGENT:
   • create_tool_calling_agent() — creates the agent logic
   • AgentExecutor — manages the ReAct loop (same pattern as Project 01!)
   • verbose=True shows the agent's reasoning process

💾 MEMORY:
   • Pass chat_history to agent_executor.invoke()
   • Agent references previous turns in reasoning
   • Critical for clinical workflows (patient context carries forward)

📊 LangChain vs FROM-SCRATCH:
   From scratch (Project 01):  More control, less code for simple agents
   LangChain (Project 02):     More features, better for complex agents

🎯 NEXT: Move to 03_langgraph_workflows for production-grade
   stateful agents with conditional routing!
""")


if __name__ == "__main__":
    main()
