"""
Level 5 - Project 04: MCP with Agents
=======================================

Connecting MCP servers to LangChain/LangGraph agents — dynamic tool discovery,
multi-server orchestration, and agent workflows powered by MCP.

Builds on: Projects 01-03 (MCP protocol, tools, resources) and Level 3 (agents).

This module demonstrates how to bridge MCP tool definitions into LangChain's
tool system, enabling agents to discover and use MCP tools at runtime. We
simulate MCP server connections by converting tool schemas into LangChain
@tool functions, and show multi-server orchestration with LangGraph.

Since full MCP client→server transport requires async I/O and a running
server process, we simulate the MCP layer and focus on the agent integration
patterns that work identically with real MCP connections.

Usage:
    python main.py
"""

import os
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Note: langchain packages not installed.")
    print("      Install with: pip install langchain-openai langchain-core")
    print("      Demos will use simulated agent behavior.\n")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ============================================================================
# Simulated MCP Server Tool Definitions
# ============================================================================

# These mirror what a real MCP server would return from tools/list

MEDICATION_SERVER_TOOLS = {
    "server_name": "medication_server",
    "tools": [
        {
            "name": "lookup_medication",
            "description": "Look up medication information: class, dosing, side effects, contraindications.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "medication_name": {"type": "string", "description": "Name of the medication"}
                },
                "required": ["medication_name"],
            },
        },
        {
            "name": "check_interaction",
            "description": "Check for drug-drug interactions between two medications.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "drug_a": {"type": "string", "description": "First medication"},
                    "drug_b": {"type": "string", "description": "Second medication"},
                },
                "required": ["drug_a", "drug_b"],
            },
        },
    ],
}

LAB_SERVER_TOOLS = {
    "server_name": "lab_server",
    "tools": [
        {
            "name": "get_lab_results",
            "description": "Get recent lab results for a patient.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string", "description": "Patient identifier"},
                    "test_name": {"type": "string", "description": "Specific test (optional)"},
                },
                "required": ["patient_id"],
            },
        },
        {
            "name": "interpret_lab_value",
            "description": "Interpret a lab test result and return normal/abnormal status.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "test_name": {"type": "string", "description": "Lab test name"},
                    "value": {"type": "number", "description": "Numeric result"},
                    "unit": {"type": "string", "description": "Unit of measure"},
                },
                "required": ["test_name", "value", "unit"],
            },
        },
    ],
}

VITALS_SERVER_TOOLS = {
    "server_name": "vitals_server",
    "tools": [
        {
            "name": "get_vitals",
            "description": "Get the latest vital signs for a patient.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string", "description": "Patient identifier"},
                },
                "required": ["patient_id"],
            },
        },
        {
            "name": "calculate_bmi",
            "description": "Calculate BMI from weight (kg) and height (m).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "weight_kg": {"type": "number", "description": "Weight in kilograms"},
                    "height_m": {"type": "number", "description": "Height in meters"},
                },
                "required": ["weight_kg", "height_m"],
            },
        },
    ],
}

ALL_SERVERS = [MEDICATION_SERVER_TOOLS, LAB_SERVER_TOOLS, VITALS_SERVER_TOOLS]


# ============================================================================
# Mock tool implementations (simulate real MCP server responses)
# ============================================================================

MEDICATION_DB = {
    "metformin": {"class": "Biguanide", "indication": "T2DM", "dose": "500-2000mg daily",
                  "side_effects": ["nausea", "diarrhea"]},
    "lisinopril": {"class": "ACE Inhibitor", "indication": "HTN, HF", "dose": "10-40mg daily",
                   "side_effects": ["dry cough", "hyperkalemia"]},
    "atorvastatin": {"class": "Statin", "indication": "Hyperlipidemia", "dose": "10-80mg daily",
                     "side_effects": ["myalgia", "elevated LFTs"]},
}

PATIENT_LABS = {
    "P001": [
        {"test": "hemoglobin", "value": 14.2, "unit": "g/dL", "flag": "normal"},
        {"test": "glucose", "value": 118, "unit": "mg/dL", "flag": "high"},
        {"test": "creatinine", "value": 1.1, "unit": "mg/dL", "flag": "normal"},
        {"test": "hba1c", "value": 7.8, "unit": "%", "flag": "high"},
    ],
}

PATIENT_VITALS = {
    "P001": {"bp": "138/85", "hr": 78, "temp": 98.4, "spo2": 97, "weight_kg": 88.5, "height_m": 1.78},
}

LAB_RANGES = {
    "hemoglobin": {"low": 12.0, "high": 17.5, "unit": "g/dL"},
    "glucose": {"low": 70, "high": 100, "unit": "mg/dL"},
    "creatinine": {"low": 0.7, "high": 1.3, "unit": "mg/dL"},
    "hba1c": {"low": 4.0, "high": 5.6, "unit": "%"},
}


def mock_tool_call(tool_name: str, arguments: dict) -> dict:
    """Execute a mock MCP tool call and return results."""
    if tool_name == "lookup_medication":
        name = arguments["medication_name"].lower()
        if name in MEDICATION_DB:
            return {"found": True, **MEDICATION_DB[name]}
        return {"found": False, "error": f"Medication '{name}' not in database"}

    elif tool_name == "check_interaction":
        d1, d2 = arguments["drug_a"].lower(), arguments["drug_b"].lower()
        if d1 == "metformin" and d2 == "lisinopril":
            return {"severity": "minor", "description": "ACE-I may enhance hypoglycemic effect"}
        return {"severity": "none", "description": "No significant interaction documented"}

    elif tool_name == "get_lab_results":
        pid = arguments["patient_id"]
        if pid in PATIENT_LABS:
            results = PATIENT_LABS[pid]
            if "test_name" in arguments and arguments["test_name"]:
                results = [r for r in results if r["test"] == arguments["test_name"]]
            return {"patient_id": pid, "results": results}
        return {"error": f"Patient {pid} not found"}

    elif tool_name == "interpret_lab_value":
        test = arguments["test_name"].lower()
        value = arguments["value"]
        if test in LAB_RANGES:
            rng = LAB_RANGES[test]
            if value < rng["low"]:
                flag = "LOW"
            elif value > rng["high"]:
                flag = "HIGH"
            else:
                flag = "NORMAL"
            return {"test": test, "value": value, "flag": flag,
                    "range": f"{rng['low']}-{rng['high']} {rng['unit']}"}
        return {"error": f"Unknown test '{test}'"}

    elif tool_name == "get_vitals":
        pid = arguments["patient_id"]
        if pid in PATIENT_VITALS:
            return {"patient_id": pid, **PATIENT_VITALS[pid]}
        return {"error": f"Patient {pid} not found"}

    elif tool_name == "calculate_bmi":
        w, h = arguments["weight_kg"], arguments["height_m"]
        bmi = round(w / (h ** 2), 1)
        cat = "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
        return {"bmi": bmi, "category": cat}

    return {"error": f"Unknown tool '{tool_name}'"}


# ============================================================================
# Helper functions
# ============================================================================

def print_json(label: str, obj, indent: int = 2):
    """Pretty-print a JSON-serializable object."""
    print(f"\n  {label}:")
    text = json.dumps(obj, indent=indent) if isinstance(obj, (dict, list)) else str(obj)
    for line in text.split("\n"):
        print(f"    {line}")


def print_banner(title: str):
    """Print a section banner."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ============================================================================
# DEMO 1: MCP Tools → LangChain
# ============================================================================

def demo_mcp_to_langchain():
    """DEMO 1: Convert MCP tool definitions into LangChain-compatible tools."""
    print_banner("DEMO 1: MCP Tools → LangChain")

    print("""
  MCP tools are defined with JSON Schema input specifications.
  LangChain tools use Python @tool decorators with type hints.

  The bridge converts:
    MCP inputSchema  →  LangChain tool parameters
    MCP description  →  LangChain docstring
    MCP tool name    →  LangChain function name
    """)

    # Show the MCP schema
    mcp_tool = MEDICATION_SERVER_TOOLS["tools"][0]
    print("  MCP Tool Definition:")
    print_json("tools/list response", mcp_tool)

    # Show the equivalent LangChain tool
    print("\n  Equivalent LangChain @tool function:")
    print("""
    @tool
    def lookup_medication(medication_name: str) -> str:
        \\"\\"\\"Look up medication information: class, dosing, side effects,
        contraindications.\\"\\"\\"
        # Call MCP server via client
        result = mcp_client.call_tool("lookup_medication",
                                       {"medication_name": medication_name})
        return json.dumps(result)
    """)

    # Demonstrate the conversion for all tools across all servers
    print("\n  Converting all MCP tools to LangChain format:")
    print("  " + "─" * 55)
    for server in ALL_SERVERS:
        print(f"\n  Server: {server['server_name']}")
        for t in server["tools"]:
            params = t["inputSchema"]["properties"]
            param_str = ", ".join(
                f"{name}: {p.get('type', 'str')}" for name, p in params.items()
            )
            print(f"    MCP:  {t['name']}({param_str})")
            print(f"    → LC: @tool {t['name']}({param_str}) -> str")

    # Test the mock implementations
    print("\n  Testing converted tools (mock execution):")
    print("  " + "─" * 55)
    test_calls = [
        ("lookup_medication", {"medication_name": "metformin"}),
        ("check_interaction", {"drug_a": "metformin", "drug_b": "lisinopril"}),
        ("get_lab_results", {"patient_id": "P001"}),
        ("get_vitals", {"patient_id": "P001"}),
        ("calculate_bmi", {"weight_kg": 88.5, "height_m": 1.78}),
    ]
    for tool_name, args in test_calls:
        result = mock_tool_call(tool_name, args)
        print(f"\n    {tool_name}({json.dumps(args)})")
        print(f"    → {json.dumps(result)[:100]}{'...' if len(json.dumps(result)) > 100 else ''}")


# ============================================================================
# DEMO 2: Dynamic Tool Discovery
# ============================================================================

def demo_dynamic_discovery():
    """DEMO 2: Agent discovers available tools from MCP server at runtime."""
    print_banner("DEMO 2: Dynamic Tool Discovery")

    print("""
  With MCP, agents don't need hard-coded tool lists. They discover
  available tools at runtime via tools/list, then decide which to use
  based on the user's query.

  Flow:
    1. Agent connects to MCP server
    2. Agent calls tools/list to discover available tools
    3. Agent receives tool schemas with descriptions
    4. For each user query, agent selects the appropriate tool
    5. Agent calls tools/call with the correct arguments
    """)

    # Simulate tool discovery from the medication server
    print("  --- Simulating tool discovery from medication_server ---")
    discovered_tools = MEDICATION_SERVER_TOOLS["tools"]
    print(f"\n  tools/list returned {len(discovered_tools)} tools:")
    for t in discovered_tools:
        print(f"    • {t['name']}: {t['description'][:60]}...")

    # Simulate agent reasoning about which tool to use
    queries = [
        "What are the side effects of metformin?",
        "Can I take lisinopril and atorvastatin together?",
        "What is the dosing for lisinopril?",
    ]

    print("\n  --- Agent tool selection based on query ---")
    for query in queries:
        # Simple keyword-based routing (in practice, the LLM does this)
        if "interaction" in query.lower() or "together" in query.lower():
            selected = "check_interaction"
        else:
            selected = "lookup_medication"
        print(f"\n    Query: \"{query}\"")
        print(f"    Selected tool: {selected}")

        # Execute
        if selected == "lookup_medication":
            # Extract medication name (simplified)
            for med in MEDICATION_DB:
                if med in query.lower():
                    result = mock_tool_call(selected, {"medication_name": med})
                    print(f"    Result: {json.dumps(result)[:80]}...")
                    break
        elif selected == "check_interaction":
            result = mock_tool_call(selected, {"drug_a": "lisinopril", "drug_b": "atorvastatin"})
            print(f"    Result: {json.dumps(result)[:80]}...")

    # Show what happens when new tools are added
    print("\n  --- Dynamic extensibility ---")
    print("  If the MCP server adds new tools, the agent discovers them")
    print("  automatically on the next tools/list call — no code changes needed.")
    new_tool = {
        "name": "check_formulary_status",
        "description": "Check if a medication is on the hospital formulary.",
        "inputSchema": {
            "type": "object",
            "properties": {"medication_name": {"type": "string"}},
            "required": ["medication_name"],
        },
    }
    print(f"\n  New tool added to server: {new_tool['name']}")
    print(f"  Agent discovers it via tools/list → ready to use immediately")


# ============================================================================
# DEMO 3: Multi-Server Agent
# ============================================================================

def demo_multi_server_agent():
    """DEMO 3: Agent connects to multiple MCP servers and uses tools from all."""
    print_banner("DEMO 3: Multi-Server Agent")

    print("""
  A clinical agent needs data from multiple systems. With MCP, each
  system is a separate server, and the agent has tools from all of them:

    ┌────────────────────────────────────────────────┐
    │              Clinical Agent                      │
    │  Tools: lookup_med, check_interaction,           │
    │         get_labs, interpret_lab,                  │
    │         get_vitals, calculate_bmi                │
    └──────┬───────────┬───────────┬──────────────────┘
           │           │           │
    ┌──────▼──┐  ┌─────▼────┐  ┌──▼───────┐
    │ Med     │  │ Lab      │  │ Vitals   │
    │ Server  │  │ Server   │  │ Server   │
    └─────────┘  └──────────┘  └──────────┘
    """)

    # Collect all tools from all servers
    all_tools = []
    for server in ALL_SERVERS:
        for t in server["tools"]:
            all_tools.append({**t, "_server": server["server_name"]})
    print(f"  Agent has {len(all_tools)} tools from {len(ALL_SERVERS)} servers:")
    for t in all_tools:
        print(f"    [{t['_server']:<20}] {t['name']}")

    # Simulate a clinical query that requires multiple servers
    print("\n  --- Clinical query: 'Review patient P001 — vitals, labs, and current meds' ---")
    print()

    steps = [
        ("get_vitals", {"patient_id": "P001"}, "vitals_server"),
        ("get_lab_results", {"patient_id": "P001"}, "lab_server"),
        ("lookup_medication", {"medication_name": "metformin"}, "medication_server"),
        ("interpret_lab_value", {"test_name": "hba1c", "value": 7.8, "unit": "%"}, "lab_server"),
        ("calculate_bmi", {"weight_kg": 88.5, "height_m": 1.78}, "vitals_server"),
    ]

    for tool_name, args, server in steps:
        result = mock_tool_call(tool_name, args)
        result_summary = json.dumps(result)[:80]
        print(f"  Step: {tool_name} (via {server})")
        print(f"    Args:   {json.dumps(args)}")
        print(f"    Result: {result_summary}{'...' if len(json.dumps(result)) > 80 else ''}")
        print()

    # Show the synthesized clinical summary
    print("  --- Agent synthesized summary ---")
    print("""
    Patient P001 Review:
    • Vitals: BP 138/85 (borderline high), HR 78, SpO2 97%
    • BMI: 27.9 (Overweight)
    • Labs: Glucose 118 mg/dL (HIGH), HbA1c 7.8% (HIGH),
            Hemoglobin 14.2 (normal), Creatinine 1.1 (normal)
    • Current med: Metformin (Biguanide) for T2DM

    Assessment: Suboptimal glycemic control (HbA1c 7.8% > 7.0% target).
    Consider intensifying diabetes therapy.
    """)


# ============================================================================
# DEMO 4: Agent Workflow with MCP (LangGraph-style)
# ============================================================================

def demo_agent_workflow():
    """DEMO 4: LangGraph-style workflow where nodes use different MCP servers."""
    print_banner("DEMO 4: Agent Workflow with MCP")

    print("""
  A LangGraph workflow can route to different MCP servers at each node.
  This creates a structured clinical pipeline:

    [Intake] → [Vitals Check] → [Lab Review] → [Med Review] → [Assessment]
                    │                  │              │
              vitals_server      lab_server     med_server
    """)

    # Define workflow nodes
    class WorkflowState:
        def __init__(self, patient_id: str, query: str):
            self.patient_id = patient_id
            self.query = query
            self.vitals = None
            self.labs = None
            self.medications = []
            self.assessment = ""
            self.steps_completed = []

    def intake_node(state: WorkflowState) -> WorkflowState:
        """Node 1: Parse the clinical query."""
        state.steps_completed.append("intake")
        print("  Node: INTAKE")
        print(f"    Patient: {state.patient_id}")
        print(f"    Query:   {state.query}")
        return state

    def vitals_node(state: WorkflowState) -> WorkflowState:
        """Node 2: Get vitals from vitals_server."""
        state.steps_completed.append("vitals")
        result = mock_tool_call("get_vitals", {"patient_id": state.patient_id})
        state.vitals = result
        print("  Node: VITALS CHECK (via vitals_server)")
        print(f"    BP: {result.get('bp')}, HR: {result.get('hr')}, SpO2: {result.get('spo2')}%")
        return state

    def lab_node(state: WorkflowState) -> WorkflowState:
        """Node 3: Get and interpret labs from lab_server."""
        state.steps_completed.append("labs")
        results = mock_tool_call("get_lab_results", {"patient_id": state.patient_id})
        state.labs = results.get("results", [])
        print("  Node: LAB REVIEW (via lab_server)")
        for lab in state.labs:
            flag_marker = "✓" if lab["flag"] == "normal" else "⚠"
            print(f"    {flag_marker} {lab['test']}: {lab['value']} {lab['unit']} [{lab['flag']}]")
        return state

    def med_node(state: WorkflowState) -> WorkflowState:
        """Node 4: Review medications from medication_server."""
        state.steps_completed.append("medications")
        # Look up a medication associated with the patient
        result = mock_tool_call("lookup_medication", {"medication_name": "metformin"})
        state.medications.append(result)
        print("  Node: MEDICATION REVIEW (via medication_server)")
        print(f"    Current: metformin ({result.get('class')}) for {result.get('indication')}")
        return state

    def assessment_node(state: WorkflowState) -> WorkflowState:
        """Node 5: Generate clinical assessment from gathered data."""
        state.steps_completed.append("assessment")
        # Determine assessment based on collected data
        issues = []
        if state.labs:
            for lab in state.labs:
                if lab["flag"] != "normal":
                    issues.append(f"{lab['test']} {lab['flag']} ({lab['value']} {lab['unit']})")
        if state.vitals:
            bp = state.vitals.get("bp", "")
            if bp:
                systolic = int(bp.split("/")[0])
                if systolic >= 130:
                    issues.append(f"Elevated BP: {bp}")

        state.assessment = (
            f"Patient {state.patient_id}: {len(issues)} issue(s) identified. "
            + "; ".join(issues) if issues else f"Patient {state.patient_id}: No issues."
        )
        print("  Node: ASSESSMENT")
        print(f"    {state.assessment}")
        return state

    # Execute the workflow
    print("\n  --- Executing Clinical Workflow ---\n")
    state = WorkflowState(patient_id="P001", query="Comprehensive patient review")
    for node_fn in [intake_node, vitals_node, lab_node, med_node, assessment_node]:
        state = node_fn(state)
        print()

    print(f"  Workflow complete. Steps: {' → '.join(state.steps_completed)}")
    print(f"  MCP servers used: vitals_server, lab_server, medication_server")

    # Show the LangGraph-equivalent structure
    print("\n  LangGraph equivalent:")
    print("""
    from langgraph.graph import StateGraph

    workflow = StateGraph(ClinicalState)
    workflow.add_node("intake", intake_node)
    workflow.add_node("vitals", vitals_node)       # uses vitals MCP server
    workflow.add_node("labs", lab_node)             # uses lab MCP server
    workflow.add_node("meds", med_node)             # uses medication MCP server
    workflow.add_node("assessment", assessment_node)

    workflow.add_edge("intake", "vitals")
    workflow.add_edge("vitals", "labs")
    workflow.add_edge("labs", "meds")
    workflow.add_edge("meds", "assessment")
    workflow.set_entry_point("intake")

    app = workflow.compile()
    """)


# ============================================================================
# Main Menu
# ============================================================================

def main():
    """Run MCP with Agents demos interactively."""
    print("=" * 70)
    print("  Level 5 - Project 04: MCP with Agents")
    print("  Connecting MCP servers to LangChain/LangGraph agents")
    print("=" * 70)

    if MCP_AVAILABLE:
        print("  ✓ MCP SDK available")
    else:
        print("  ⚠ MCP SDK not installed (demos use simulated MCP)")

    if LANGCHAIN_AVAILABLE:
        print("  ✓ LangChain available")
    else:
        print("  ⚠ LangChain not installed (demos use simulated agents)")

    demos = {
        "1": ("MCP Tools → LangChain", demo_mcp_to_langchain),
        "2": ("Dynamic Tool Discovery", demo_dynamic_discovery),
        "3": ("Multi-Server Agent", demo_multi_server_agent),
        "4": ("Agent Workflow with MCP", demo_agent_workflow),
    }

    while True:
        print("\nAvailable Demos:")
        for key, (name, _) in demos.items():
            print(f"  {key}. {name}")
        print("  A. Run all demos")
        print("  Q. Quit")

        choice = input("\nSelect demo (1-4, A, Q): ").strip().upper()

        if choice == "Q":
            print("\nGoodbye!")
            break
        elif choice == "A":
            for key in sorted(demos.keys()):
                demos[key][1]()
        elif choice in demos:
            demos[choice][1]()
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
