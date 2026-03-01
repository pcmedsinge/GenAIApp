"""
Exercise 7: LangGraph Prebuilt — create_react_agent, ToolNode, tools_condition

Skills practiced:
- Using create_react_agent() to build a tool-using agent in ONE line
- Understanding ToolNode (automatic tool execution node)
- Using tools_condition for automatic routing (call tool or end)
- Building custom graphs with ToolNode for more control
- Comparing prebuilt vs hand-built approaches

Why this matters:
  In earlier exercises, we built graphs node-by-node with StateGraph.
  That's great for learning, but LangGraph provides PREBUILT components
  that handle the most common pattern: an LLM that decides which tools
  to call, calls them, and loops until done.

  create_react_agent() = the standard way to build tool-using agents.
  ToolNode + tools_condition = the building blocks if you need more control.

Architecture:

  create_react_agent (prebuilt — does ALL of this for you):
  ┌─────────────────────────────────────────────────┐
  │                                                  │
  │   ┌──────────┐    tools_condition    ┌────────┐ │
  │   │  agent   │──── has tool call? ──▶│  tools │ │
  │   │  (LLM)   │◀───────────────────── │(ToolNode│ │
  │   └──────────┘    tool results       └────────┘ │
  │        │                                         │
  │        │ no tool call                            │
  │        ▼                                         │
  │      [END]                                       │
  │                                                  │
  └─────────────────────────────────────────────────┘

  The loop:
  1. LLM decides: call a tool, or respond to user
  2. If tool call → ToolNode executes it → result goes back to LLM
  3. If no tool call → output to user (END)
  4. Repeat until LLM is satisfied

Healthcare parallel:
  A physician (LLM) who can order labs, check guidelines, look up drugs.
  They keep ordering tests and looking things up until they have enough
  info to give a diagnosis. That's the ReAct loop.
"""

import os
import json
from typing import Annotated
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# Clinical Tools (using LangChain @tool decorator)
# ============================================================

@tool
def check_lab_value(test_name: str, value: float) -> str:
    """Check if a lab value is within normal range and flag abnormals.

    Args:
        test_name: Name of the lab test (e.g., 'troponin', 'glucose', 'potassium')
        value: The numeric lab value to check
    """
    ranges = {
        "troponin": {"low": 0.0, "high": 0.04, "unit": "ng/mL", "critical_high": 0.4},
        "glucose": {"low": 70, "high": 100, "unit": "mg/dL", "critical_high": 400},
        "potassium": {"low": 3.5, "high": 5.0, "unit": "mEq/L", "critical_high": 6.0},
        "sodium": {"low": 136, "high": 145, "unit": "mEq/L", "critical_high": 155},
        "creatinine": {"low": 0.7, "high": 1.3, "unit": "mg/dL", "critical_high": 4.0},
        "hemoglobin": {"low": 12.0, "high": 17.5, "unit": "g/dL", "critical_high": 20.0},
        "wbc": {"low": 4.5, "high": 11.0, "unit": "K/uL", "critical_high": 30.0},
        "bnp": {"low": 0, "high": 100, "unit": "pg/mL", "critical_high": 900},
    }

    test = test_name.lower().strip()
    if test not in ranges:
        return f"Unknown test: {test_name}. Available: {', '.join(ranges.keys())}"

    ref = ranges[test]
    status = "NORMAL"
    if value < ref["low"]:
        status = "LOW"
    elif value > ref.get("critical_high", ref["high"]):
        status = "CRITICAL HIGH"
    elif value > ref["high"]:
        status = "HIGH"

    return (
        f"{test_name}: {value} {ref['unit']} — {status} "
        f"(reference: {ref['low']}-{ref['high']} {ref['unit']})"
    )


@tool
def lookup_drug_info(drug_name: str) -> str:
    """Look up drug information including class, common doses, and key interactions.

    Args:
        drug_name: Name of the medication to look up
    """
    drugs = {
        "aspirin": {
            "class": "Antiplatelet / NSAID",
            "common_dose": "81-325mg daily",
            "interactions": ["warfarin (bleeding risk)", "ibuprofen (reduced antiplatelet effect)"],
            "contraindications": ["active GI bleeding", "aspirin allergy"],
        },
        "heparin": {
            "class": "Anticoagulant",
            "common_dose": "60 units/kg bolus, 12 units/kg/hr infusion",
            "interactions": ["aspirin (bleeding risk)", "NSAIDs (bleeding risk)"],
            "contraindications": ["HIT history", "active bleeding"],
        },
        "metformin": {
            "class": "Biguanide (antidiabetic)",
            "common_dose": "500-1000mg BID",
            "interactions": ["contrast dye (lactic acidosis)", "alcohol"],
            "contraindications": ["eGFR < 30", "metabolic acidosis"],
        },
        "lisinopril": {
            "class": "ACE Inhibitor",
            "common_dose": "10-40mg daily",
            "interactions": ["potassium supplements (hyperkalemia)", "NSAIDs (reduced effect)"],
            "contraindications": ["angioedema history", "pregnancy", "bilateral renal artery stenosis"],
        },
        "atorvastatin": {
            "class": "HMG-CoA Reductase Inhibitor (Statin)",
            "common_dose": "10-80mg daily",
            "interactions": ["gemfibrozil (myopathy risk)", "cyclosporine"],
            "contraindications": ["active liver disease", "pregnancy"],
        },
        "clopidogrel": {
            "class": "Antiplatelet (P2Y12 inhibitor)",
            "common_dose": "75mg daily (300-600mg loading dose)",
            "interactions": ["omeprazole (reduced effect)", "warfarin (bleeding)"],
            "contraindications": ["active bleeding", "severe liver disease"],
        },
    }

    drug = drug_name.lower().strip()
    if drug not in drugs:
        return f"Drug not found: {drug_name}. Available: {', '.join(drugs.keys())}"

    info = drugs[drug]
    return (
        f"Drug: {drug_name}\n"
        f"Class: {info['class']}\n"
        f"Dose: {info['common_dose']}\n"
        f"Interactions: {', '.join(info['interactions'])}\n"
        f"Contraindications: {', '.join(info['contraindications'])}"
    )


@tool
def get_clinical_guideline(condition: str) -> str:
    """Get evidence-based clinical guidelines for a condition.

    Args:
        condition: The clinical condition (e.g., 'ACS', 'CHF', 'sepsis')
    """
    guidelines = {
        "acs": (
            "ACS (Acute Coronary Syndrome) — ACC/AHA Guidelines:\n"
            "1. Aspirin 325mg immediately (then 81mg daily)\n"
            "2. P2Y12 inhibitor (clopidogrel 300mg or ticagrelor 180mg load)\n"
            "3. Anticoagulation (heparin or enoxaparin)\n"
            "4. Nitroglycerin for ongoing pain\n"
            "5. Beta-blocker within 24h if no contraindication\n"
            "6. Statin (high-intensity, e.g., atorvastatin 80mg)\n"
            "7. Cardiac cath within 24h for NSTEMI, emergent for STEMI"
        ),
        "chf": (
            "CHF (Heart Failure) — ACC/AHA Guidelines:\n"
            "1. Loop diuretic (furosemide) for volume overload\n"
            "2. ACE-I or ARB (or ARNI for HFrEF)\n"
            "3. Beta-blocker (carvedilol, metoprolol succinate, bisoprolol)\n"
            "4. Aldosterone antagonist if EF ≤ 35%\n"
            "5. SGLT2 inhibitor (empagliflozin/dapagliflozin)\n"
            "6. Daily weights, fluid restriction 1.5-2L\n"
            "7. Device therapy (ICD/CRT) if EF ≤ 35% on optimal therapy"
        ),
        "sepsis": (
            "Sepsis — Surviving Sepsis Campaign:\n"
            "1. Blood cultures before antibiotics\n"
            "2. Broad-spectrum antibiotics within 1 hour\n"
            "3. 30 mL/kg crystalloid for hypotension or lactate ≥ 4\n"
            "4. Vasopressors (norepinephrine first-line) if MAP < 65\n"
            "5. Measure lactate, repeat if elevated\n"
            "6. Reassess volume status and tissue perfusion\n"
            "7. Source control (drain abscess, remove infected device)"
        ),
    }

    key = condition.lower().strip()
    if key not in guidelines:
        return f"No guideline for '{condition}'. Available: {', '.join(guidelines.keys())}"

    return guidelines[key]


@tool
def calculate_risk_score(score_type: str, parameters: str) -> str:
    """Calculate a clinical risk score.

    Args:
        score_type: Type of score ('timi', 'heart', 'wells')
        parameters: JSON string of score parameters
    """
    try:
        params = json.loads(parameters)
    except json.JSONDecodeError:
        return "Error: parameters must be valid JSON"

    if score_type.lower() == "timi":
        score = 0
        if params.get("age_over_65", False): score += 1
        if params.get("three_or_more_risk_factors", False): score += 1
        if params.get("known_cad", False): score += 1
        if params.get("aspirin_use_7days", False): score += 1
        if params.get("severe_angina", False): score += 1
        if params.get("st_deviation", False): score += 1
        if params.get("elevated_biomarker", False): score += 1

        risk = "Low" if score <= 2 else "Intermediate" if score <= 4 else "High"
        return f"TIMI Score: {score}/7 — Risk: {risk}. (0-2: Low 4.7%, 3-4: Intermediate 13.2%, 5-7: High 40.9%)"

    return f"Unknown score type: {score_type}. Available: timi, heart, wells"


# All tools together
clinical_tools = [check_lab_value, lookup_drug_info, get_clinical_guideline, calculate_risk_score]


# ============================================================
# DEMO 1: create_react_agent — One-Line Agent
# ============================================================

def demo_prebuilt_react_agent():
    """Show create_react_agent: build a full tool-using agent in one call."""
    print("\n" + "=" * 70)
    print("  DEMO 1: create_react_agent — ONE-LINE TOOL-USING AGENT")
    print("=" * 70)
    print("""
  create_react_agent() builds a complete ReAct agent:
  - LLM node that decides what tools to call
  - ToolNode that executes tool calls
  - Automatic routing loop (call tool → get result → decide again)
  - Built-in message state management

  One function call replaces 20+ lines of manual graph construction.
  """)

    # One line to create a full agent
    agent = create_react_agent(
        model=llm,
        tools=clinical_tools,
        prompt="You are a clinical decision support agent. Use the available "
               "tools to look up lab values, drug information, clinical guidelines, "
               "and risk scores. Always check relevant data before giving advice. "
               "Be thorough but concise.",
    )

    # Query the agent
    queries = [
        "Patient has troponin of 0.45 and is on metformin. Check the troponin level and look up metformin info.",
        "What are the ACS guidelines? Also calculate TIMI score for a 70yo with ST deviation and elevated biomarker.",
    ]

    for i, query in enumerate(queries):
        print(f"\n  ─── Query {i + 1}: {query[:70]}... ───")
        result = agent.invoke({"messages": [HumanMessage(content=query)]})

        # Show the message flow
        for msg in result["messages"]:
            if isinstance(msg, HumanMessage):
                print(f"    👤 Human: {msg.content[:80]}...")
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"    🤖 Tool call: {tc['name']}({json.dumps(tc['args'])[:60]}...)")
                else:
                    print(f"    🤖 AI: {msg.content[:120]}...")
            elif isinstance(msg, ToolMessage):
                print(f"    🔧 Tool result: {msg.content[:100]}...")

    print(f"\n  KEY INSIGHT: create_react_agent() gives you a full ReAct agent")
    print(f"  with tools in ONE call. No StateGraph, no manual edges.")


# ============================================================
# DEMO 2: ToolNode + tools_condition — Building Blocks
# ============================================================

def demo_toolnode_manual():
    """Show ToolNode and tools_condition as manual building blocks."""
    print("\n" + "=" * 70)
    print("  DEMO 2: TOOLNODE + TOOLS_CONDITION — MANUAL BUILDING BLOCKS")
    print("=" * 70)
    print("""
  create_react_agent uses ToolNode and tools_condition internally.
  When you need MORE CONTROL, use them directly:

  - ToolNode: A node that automatically executes tool calls from the LLM
  - tools_condition: A routing function that checks "did the LLM call a tool?"

  This lets you add custom nodes BETWEEN the LLM and tool execution.
  """)

    # Build the graph manually with ToolNode
    tool_node = ToolNode(clinical_tools)

    # Bind tools to LLM (tells the LLM what tools are available)
    llm_with_tools = llm.bind_tools(clinical_tools)

    def agent_node(state: MessagesState) -> dict:
        """The LLM node — decides whether to call tools or respond."""
        # Add system message
        system_msg = {
            "role": "system",
            "content": (
                "You are a clinical decision support agent. "
                "Use tools to look up information before answering. "
                "Be concise."
            ),
        }
        response = llm_with_tools.invoke([system_msg] + state["messages"])
        return {"messages": [response]}

    # Build graph
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")

    # tools_condition automatically routes:
    # - If LLM made tool calls → go to "tools" node
    # - If no tool calls → END
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")  # After tool execution, go back to LLM

    app = graph.compile()

    # Test it
    print(f"\n  Graph structure: agent → (tools_condition) → tools → agent → ... → END")

    query = "Check if potassium 5.8 is normal, and look up heparin interactions."
    print(f"\n  Query: {query}")

    result = app.invoke({"messages": [HumanMessage(content=query)]})

    for msg in result["messages"]:
        if isinstance(msg, HumanMessage):
            print(f"\n    👤 {msg.content[:80]}...")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"    🤖 Tool call: {tc['name']}({json.dumps(tc['args'])[:60]})")
            else:
                print(f"    🤖 Response: {msg.content[:150]}...")
        elif isinstance(msg, ToolMessage):
            print(f"    🔧 {msg.name}: {msg.content[:100]}...")

    print(f"\n  KEY INSIGHT: ToolNode + tools_condition are the building blocks.")
    print(f"  Use them when you need custom pre/post-processing around tools.")


# ============================================================
# DEMO 3: Custom Pre-Processing with ToolNode
# ============================================================

def demo_custom_preprocessing():
    """Show adding custom logic between LLM and tool execution."""
    print("\n" + "=" * 70)
    print("  DEMO 3: CUSTOM PRE-PROCESSING — SAFETY GATE BEFORE TOOLS")
    print("=" * 70)
    print("""
  Why use ToolNode manually instead of create_react_agent?
  Because you can INSERT CUSTOM NODES between the LLM and tools.

  Example: A safety gate that validates tool calls before execution.
  The LLM wants to call a tool → safety gate checks it → then ToolNode runs.
  """)

    tool_node = ToolNode(clinical_tools)
    llm_with_tools = llm.bind_tools(clinical_tools)

    safety_log = []

    def agent_node(state: MessagesState) -> dict:
        system_msg = {
            "role": "system",
            "content": "You are a clinical agent. Use tools to look up information.",
        }
        response = llm_with_tools.invoke([system_msg] + state["messages"])
        return {"messages": [response]}

    def safety_gate(state: MessagesState) -> dict:
        """Validate tool calls before execution — custom pre-processing."""
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                safety_log.append({
                    "tool": tc["name"],
                    "args": tc["args"],
                    "status": "approved",
                })
                print(f"    ✅ Safety gate approved: {tc['name']}({json.dumps(tc['args'])[:50]})")
        return {}  # Pass through — don't modify state

    def route_after_agent(state: MessagesState) -> str:
        """Route to safety gate if tool calls, else END."""
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "safety_gate"
        return END

    # Build graph with safety gate
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("safety_gate", safety_gate)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent,
                                 {"safety_gate": "safety_gate", END: END})
    graph.add_edge("safety_gate", "tools")
    graph.add_edge("tools", "agent")

    app = graph.compile()

    query = "Look up aspirin and clopidogrel — I need to check for interactions."
    print(f"\n  Query: {query}")
    print(f"  Flow: agent → safety_gate → tools → agent → ...\n")

    result = app.invoke({"messages": [HumanMessage(content=query)]})

    # Final response
    final = result["messages"][-1]
    if isinstance(final, AIMessage):
        print(f"\n    🤖 Final: {final.content[:200]}...")

    print(f"\n  Safety gate log: {len(safety_log)} tool calls validated")
    for entry in safety_log:
        print(f"    {entry['status'].upper()}: {entry['tool']}({json.dumps(entry['args'])[:50]})")

    print(f"\n  KEY INSIGHT: By building with ToolNode manually, you can insert")
    print(f"  custom nodes (safety gate, logging, approval) in the tool loop.")


# ============================================================
# DEMO 4: Prebuilt Agent with Checkpointing
# ============================================================

def demo_react_with_checkpointing():
    """Show create_react_agent with a checkpointer for persistence."""
    print("\n" + "=" * 70)
    print("  DEMO 4: PREBUILT AGENT + CHECKPOINTING")
    print("=" * 70)
    print("""
  create_react_agent works seamlessly with checkpointers.
  Add checkpointer= and you get:
  - Conversation persistence
  - Multi-turn tool usage with memory
  - Thread isolation
  """)

    checkpointer = MemorySaver()
    agent = create_react_agent(
        model=llm,
        tools=clinical_tools,
        prompt="You are a clinical decision support agent. Use tools to help. "
               "Remember previous conversation context.",
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "patient-consult-1"}}

    # Turn 1
    print(f"\n  ─── Turn 1 ───")
    query1 = "Check troponin of 0.45 and look up ACS guidelines."
    print(f"  👤 {query1}")
    result1 = agent.invoke({"messages": [HumanMessage(content=query1)]}, config)
    final1 = result1["messages"][-1]
    print(f"  🤖 {final1.content[:200]}...")

    # Turn 2 — references previous context
    print(f"\n  ─── Turn 2 (with memory) ───")
    query2 = "Based on those results, what's the recommended antiplatelet therapy?"
    print(f"  👤 {query2}")
    result2 = agent.invoke({"messages": [HumanMessage(content=query2)]}, config)
    final2 = result2["messages"][-1]
    print(f"  🤖 {final2.content[:200]}...")

    # Show message count growing
    print(f"\n  Total messages in thread: {len(result2['messages'])}")
    print(f"  (Includes all tool calls and results from both turns)")

    print(f"\n  KEY INSIGHT: create_react_agent + checkpointer = multi-turn")
    print(f"  tool-using agent with zero manual state management.")


# ============================================================
# DEMO 5: Comparing Prebuilt vs Hand-Built
# ============================================================

def demo_comparison():
    """Compare the prebuilt approach vs hand-built."""
    print("\n" + "=" * 70)
    print("  DEMO 5: PREBUILT vs HAND-BUILT — WHEN TO USE EACH")
    print("=" * 70)
    print("""
  ┌──────────────────────────────────────────────────────────────┐
  │ Approach              │ Lines │ When to Use                  │
  ├───────────────────────┼───────┼──────────────────────────────┤
  │ create_react_agent()  │  1-3  │ Standard tool-using agent    │
  │                       │       │ No custom logic needed       │
  │                       │       │ Quick prototyping            │
  ├───────────────────────┼───────┼──────────────────────────────┤
  │ ToolNode + manual     │ 15-25 │ Need custom pre/post logic   │
  │ StateGraph            │       │ Safety gates, logging        │
  │                       │       │ Custom routing beyond tools  │
  ├───────────────────────┼───────┼──────────────────────────────┤
  │ Full manual graph     │ 30+   │ Complex multi-step workflows │
  │ (no ToolNode)         │       │ Non-tool nodes in the loop   │
  │                       │       │ Custom state beyond messages │
  └──────────────────────────────────────────────────────────────┘

  create_react_agent is the RIGHT DEFAULT for most agent tasks.
  Only drop to manual when you need control it doesn't provide.
  """)

    # Prebuilt: 3 lines
    print(f"  ═══ Prebuilt (3 lines) ═══")
    print(f"    agent = create_react_agent(model=llm, tools=tools, prompt='...')")
    print(f"    result = agent.invoke({{'messages': [HumanMessage('...')]}}))")

    agent = create_react_agent(model=llm, tools=clinical_tools,
        prompt="Clinical agent. Be concise.")
    r1 = agent.invoke({"messages": [HumanMessage(content="Check glucose of 210")]})
    final1 = r1["messages"][-1].content[:100]
    print(f"    Result: {final1}...")

    # Manual: 15+ lines
    print(f"\n  ═══ Manual with ToolNode (15+ lines) ═══")
    print(f"    graph = StateGraph(MessagesState)")
    print(f"    graph.add_node('agent', agent_fn)")
    print(f"    graph.add_node('tools', ToolNode(tools))")
    print(f"    graph.add_conditional_edges('agent', tools_condition)")
    print(f"    graph.add_edge('tools', 'agent')")
    print(f"    app = graph.compile()")

    tool_node = ToolNode(clinical_tools)
    llm_t = llm.bind_tools(clinical_tools)
    def agent_fn(state: MessagesState):
        return {"messages": [llm_t.invoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent_fn)
    g.add_node("tools", tool_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    app = g.compile()

    r2 = app.invoke({"messages": [HumanMessage(content="Check glucose of 210")]})
    final2 = r2["messages"][-1].content[:100]
    print(f"    Result: {final2}...")

    print(f"\n  Same result. Prebuilt is simpler. Manual gives more control.")


# ============================================================
# DEMO 6: Interactive Agent
# ============================================================

def demo_interactive():
    """Interactive clinical agent."""
    print("\n" + "=" * 70)
    print("  DEMO 6: INTERACTIVE — CHAT WITH THE CLINICAL AGENT")
    print("=" * 70)
    print("  Tools: check_lab_value, lookup_drug_info, get_clinical_guideline, calculate_risk_score")
    print("  Try: 'Check troponin 0.45 and look up heparin'")
    print("  Type 'quit' to exit.\n")

    checkpointer = MemorySaver()
    agent = create_react_agent(
        model=llm,
        tools=clinical_tools,
        prompt="You are a clinical decision support agent. Use the available "
               "tools to look up lab values, drug info, guidelines, and risk scores. "
               "Be thorough and concise. Always use tools before giving advice.",
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "interactive-session"}}

    while True:
        user_input = input("  You: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config,
        )

        # Show tool usage
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  🔧 Using: {tc['name']}({json.dumps(tc['args'])[:60]})")

        final = result["messages"][-1]
        print(f"  Agent: {final.content}\n")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 7: LANGGRAPH PREBUILT — create_react_agent + ToolNode")
    print("=" * 70)
    print("""
    LangGraph's prebuilt components for tool-using agents:
    - create_react_agent(): Full agent in one call
    - ToolNode: Automatic tool execution node
    - tools_condition: Automatic "tool or end?" routing

    Choose a demo:
      1 → create_react_agent (one-line agent)
      2 → ToolNode + tools_condition (manual building blocks)
      3 → Custom pre-processing (safety gate before tools)
      4 → Prebuilt agent + checkpointing (multi-turn)
      5 → Prebuilt vs hand-built comparison
      6 → Interactive clinical agent
      7 → Run demos 1-5
    """)

    choice = input("  Enter choice (1-7): ").strip()

    demos = {
        "1": demo_prebuilt_react_agent,
        "2": demo_toolnode_manual,
        "3": demo_custom_preprocessing,
        "4": demo_react_with_checkpointing,
        "5": demo_comparison,
        "6": demo_interactive,
    }

    if choice == "7":
        for d in [demo_prebuilt_react_agent, demo_toolnode_manual,
                   demo_custom_preprocessing, demo_react_with_checkpointing,
                   demo_comparison]:
            d()
    elif choice in demos:
        demos[choice]()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. create_react_agent() = THE DEFAULT CHOICE
   For any "LLM that uses tools" task, start here. It handles:
   - Tool binding, execution, result routing, looping — all automatic
   - Works with any LangChain tool (@tool decorator)
   - Supports checkpointing, streaming, etc.

2. ToolNode = AUTOMATIC TOOL EXECUTION
   Takes tool calls from the LLM message and executes them.
   Returns ToolMessage results. No manual parsing needed.
   It handles parallel tool calls automatically.

3. tools_condition = SMART ROUTING
   Checks the last AI message: if it has tool_calls → route to tools.
   If no tool_calls → route to END. This is the loop condition.

4. WHEN TO USE EACH:
   - create_react_agent: Standard tool use, quick prototyping, most cases
   - ToolNode + manual graph: Need custom nodes in the loop (safety, logging)
   - Full manual (no ToolNode): Complex workflows, custom state, non-tool patterns

5. PREBUILT + CHECKPOINTER = PRODUCTION AGENT:
   create_react_agent(model, tools, checkpointer=MemorySaver())
   gives you a persistent, multi-turn, tool-using agent with zero boilerplate.

6. @tool DECORATOR IS THE BRIDGE:
   LangChain's @tool decorator creates tools that work everywhere:
   - create_react_agent (prebuilt)
   - ToolNode (manual graph)
   - LangChain AgentExecutor (legacy)
   Write tools once, use anywhere.
"""

if __name__ == "__main__":
    main()
