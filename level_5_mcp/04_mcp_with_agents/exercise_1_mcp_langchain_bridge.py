"""
Exercise 1: MCP → LangChain Bridge
=====================================

Skills practiced:
- Converting MCP tool schemas (JSON Schema) to LangChain @tool functions
- Mapping MCP inputSchema properties to Python function parameters
- Creating a reusable bridge that works with any MCP tool definition
- Testing converted tools with mock MCP server backends

Healthcare context:
When integrating MCP servers with LangChain/LangGraph agents, you need a
bridge that converts MCP tool definitions into LangChain-compatible tools.
This bridge must handle the JSON Schema → Python type mapping, preserve
tool descriptions for agent reasoning, and route calls back to the MCP
server for execution.

This exercise builds a systematic bridge and tests it with healthcare tools.

Usage:
    python exercise_1_mcp_langchain_bridge.py
"""

import os
import json
from typing import Any, Callable
from dotenv import load_dotenv

load_dotenv()

try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

try:
    from langchain_core.tools import tool, StructuredTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# ============================================================================
# MCP Tool Definitions (from server's tools/list response)
# ============================================================================

MCP_TOOL_DEFINITIONS = [
    {
        "name": "lookup_medication",
        "description": "Look up detailed medication information including class, dosing, "
                       "side effects, and contraindications.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "medication_name": {
                    "type": "string",
                    "description": "Name of the medication to look up"
                }
            },
            "required": ["medication_name"],
        },
    },
    {
        "name": "check_drug_interaction",
        "description": "Check for drug-drug interactions between two medications. "
                       "Returns severity and clinical recommendation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "drug_a": {"type": "string", "description": "First medication name"},
                "drug_b": {"type": "string", "description": "Second medication name"},
            },
            "required": ["drug_a", "drug_b"],
        },
    },
    {
        "name": "interpret_lab_value",
        "description": "Interpret a laboratory test result. Returns whether the value "
                       "is normal, low, or high with clinical significance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "test_name": {"type": "string", "description": "Lab test name (e.g., hemoglobin)"},
                "value": {"type": "number", "description": "Numeric result value"},
                "unit": {"type": "string", "description": "Unit (e.g., g/dL, mg/dL)"},
            },
            "required": ["test_name", "value", "unit"],
        },
    },
    {
        "name": "calculate_bmi",
        "description": "Calculate Body Mass Index from weight and height. "
                       "Returns BMI value and WHO classification.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "weight_kg": {"type": "number", "description": "Patient weight in kilograms"},
                "height_m": {"type": "number", "description": "Patient height in meters"},
            },
            "required": ["weight_kg", "height_m"],
        },
    },
    {
        "name": "get_patient_vitals",
        "description": "Get the latest vital signs for a patient by ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "Patient identifier"},
            },
            "required": ["patient_id"],
        },
    },
]


# ============================================================================
# Mock MCP Server Backend
# ============================================================================

MEDICATION_DB = {
    "metformin": {"class": "Biguanide", "indication": "T2DM", "dose": "500-2000mg daily",
                  "side_effects": ["nausea", "diarrhea", "lactic acidosis (rare)"]},
    "lisinopril": {"class": "ACE Inhibitor", "indication": "HTN, HF", "dose": "10-40mg daily",
                   "side_effects": ["dry cough", "hyperkalemia", "angioedema (rare)"]},
    "atorvastatin": {"class": "Statin", "indication": "Hyperlipidemia", "dose": "10-80mg daily",
                     "side_effects": ["myalgia", "elevated LFTs"]},
}

LAB_RANGES = {
    "hemoglobin": {"low": 12.0, "high": 17.5, "unit": "g/dL"},
    "glucose": {"low": 70, "high": 100, "unit": "mg/dL"},
    "creatinine": {"low": 0.7, "high": 1.3, "unit": "mg/dL"},
    "hba1c": {"low": 4.0, "high": 5.6, "unit": "%"},
}

PATIENT_VITALS = {
    "P001": {"bp": "138/85", "hr": 78, "temp": 98.4, "spo2": 97, "weight_kg": 88.5},
    "P002": {"bp": "125/78", "hr": 72, "temp": 98.2, "spo2": 99, "weight_kg": 65.0},
}


def mock_mcp_call(tool_name: str, arguments: dict) -> str:
    """Simulate calling an MCP server tool. Returns JSON string."""
    if tool_name == "lookup_medication":
        name = arguments["medication_name"].lower()
        if name in MEDICATION_DB:
            return json.dumps({"found": True, **MEDICATION_DB[name]})
        return json.dumps({"found": False, "error": f"'{name}' not found"})

    elif tool_name == "check_drug_interaction":
        d1, d2 = arguments["drug_a"].lower(), arguments["drug_b"].lower()
        known_interactions = {
            ("metformin", "lisinopril"): ("minor", "ACE-I may enhance hypoglycemic effect"),
            ("atorvastatin", "amlodipine"): ("moderate", "Limit atorvastatin to 20mg"),
        }
        key1, key2 = (d1, d2), (d2, d1)
        if key1 in known_interactions:
            sev, desc = known_interactions[key1]
        elif key2 in known_interactions:
            sev, desc = known_interactions[key2]
        else:
            sev, desc = "none", "No significant interaction documented"
        return json.dumps({"drug_a": d1, "drug_b": d2, "severity": sev, "description": desc})

    elif tool_name == "interpret_lab_value":
        test = arguments["test_name"].lower()
        value = arguments["value"]
        if test in LAB_RANGES:
            rng = LAB_RANGES[test]
            flag = "LOW" if value < rng["low"] else "HIGH" if value > rng["high"] else "NORMAL"
            return json.dumps({"test": test, "value": value, "flag": flag,
                               "range": f"{rng['low']}-{rng['high']}"})
        return json.dumps({"error": f"Unknown test: {test}"})

    elif tool_name == "calculate_bmi":
        w, h = arguments["weight_kg"], arguments["height_m"]
        bmi = round(w / (h ** 2), 1)
        cat = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else \
              "Overweight" if bmi < 30 else "Obese"
        return json.dumps({"bmi": bmi, "category": cat})

    elif tool_name == "get_patient_vitals":
        pid = arguments["patient_id"]
        if pid in PATIENT_VITALS:
            return json.dumps({"patient_id": pid, **PATIENT_VITALS[pid]})
        return json.dumps({"error": f"Patient {pid} not found"})

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ============================================================================
# MCP → LangChain Bridge
# ============================================================================

JSON_TYPE_MAP = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def mcp_to_langchain_tool(mcp_def: dict,
                           executor: Callable = mock_mcp_call) -> Callable:
    """
    Convert an MCP tool definition to a LangChain-compatible callable.

    Args:
        mcp_def: MCP tool definition with name, description, inputSchema
        executor: Function that takes (tool_name, arguments) and returns result

    Returns:
        A callable function that can be used as a LangChain tool.
    """
    tool_name = mcp_def["name"]
    description = mcp_def["description"]
    schema = mcp_def.get("inputSchema", {})
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    def tool_fn(**kwargs) -> str:
        """Wrapper that calls the MCP server."""
        return executor(tool_name, kwargs)

    tool_fn.__name__ = tool_name
    tool_fn.__doc__ = description

    # Attach metadata for inspection
    tool_fn._mcp_schema = mcp_def
    tool_fn._parameters = properties
    tool_fn._required = required

    return tool_fn


def convert_all_tools(mcp_definitions: list,
                       executor: Callable = mock_mcp_call) -> dict:
    """
    Convert a list of MCP tool definitions to LangChain-compatible tools.

    Returns:
        Dict mapping tool_name → callable
    """
    tools = {}
    for mcp_def in mcp_definitions:
        fn = mcp_to_langchain_tool(mcp_def, executor)
        tools[mcp_def["name"]] = fn
    return tools


def describe_conversion(mcp_def: dict) -> str:
    """Return a human-readable description of the MCP→LangChain mapping."""
    name = mcp_def["name"]
    desc = mcp_def["description"][:60]
    props = mcp_def.get("inputSchema", {}).get("properties", {})
    params = []
    for pname, pinfo in props.items():
        ptype = JSON_TYPE_MAP.get(pinfo.get("type", "string"), str).__name__
        params.append(f"{pname}: {ptype}")
    signature = f"{name}({', '.join(params)}) -> str"
    return f"@tool\ndef {signature}:\n    \"\"\"{desc}...\"\"\""


# ============================================================================
# Demo / Test
# ============================================================================

def main():
    """Demonstrate the MCP → LangChain bridge."""
    print("=" * 70)
    print("  Exercise 1: MCP → LangChain Bridge")
    print("  Converting MCP tool schemas to LangChain @tool functions")
    print("=" * 70)

    # 1. Show all MCP tool definitions
    print("\n--- MCP Tool Definitions ---")
    for mcp_def in MCP_TOOL_DEFINITIONS:
        props = mcp_def["inputSchema"]["properties"]
        param_list = ", ".join(props.keys())
        print(f"  • {mcp_def['name']}({param_list})")
        print(f"    {mcp_def['description'][:70]}")

    # 2. Convert all tools
    print("\n--- Conversion: MCP → LangChain ---")
    tools = convert_all_tools(MCP_TOOL_DEFINITIONS)
    print(f"  Converted {len(tools)} tools:\n")
    for mcp_def in MCP_TOOL_DEFINITIONS:
        code = describe_conversion(mcp_def)
        for line in code.split("\n"):
            print(f"    {line}")
        print()

    # 3. Test each converted tool
    print("--- Testing Converted Tools ---")
    test_cases = [
        ("lookup_medication", {"medication_name": "metformin"}),
        ("lookup_medication", {"medication_name": "lisinopril"}),
        ("check_drug_interaction", {"drug_a": "metformin", "drug_b": "lisinopril"}),
        ("interpret_lab_value", {"test_name": "glucose", "value": 118, "unit": "mg/dL"}),
        ("interpret_lab_value", {"test_name": "hemoglobin", "value": 14.2, "unit": "g/dL"}),
        ("calculate_bmi", {"weight_kg": 88.5, "height_m": 1.78}),
        ("get_patient_vitals", {"patient_id": "P001"}),
    ]

    for tool_name, args in test_cases:
        fn = tools[tool_name]
        result = fn(**args)
        parsed = json.loads(result)
        status = "✓" if "error" not in parsed else "✗"
        print(f"  {status} {tool_name}({json.dumps(args)})")
        print(f"    → {result[:80]}{'...' if len(result) > 80 else ''}")

    # 4. Show schema metadata
    print("\n--- Schema Metadata ---")
    for tool_name, fn in tools.items():
        schema = fn._mcp_schema
        params = fn._parameters
        required = fn._required
        print(f"\n  {tool_name}:")
        for pname, pinfo in params.items():
            req_marker = "*" if pname in required else " "
            print(f"    {req_marker}{pname}: {pinfo.get('type', '?')} — {pinfo.get('description', '')}")

    # 5. Error handling test
    print("\n--- Error Handling ---")
    result = tools["lookup_medication"](medication_name="unknown_drug")
    print(f"  Unknown medication: {result}")
    result = tools["get_patient_vitals"](patient_id="P999")
    print(f"  Unknown patient:    {result}")
    result = tools["interpret_lab_value"](test_name="unknown_test", value=5.0, unit="?")
    print(f"  Unknown lab test:   {result}")

    print(f"\n  ✓ Bridge successfully converted {len(tools)} MCP tools to LangChain format")


if __name__ == "__main__":
    main()
