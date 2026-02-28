"""
Exercise 2: Multi-Server Agent
=================================

Skills practiced:
- Building an agent that uses tools from multiple simulated MCP servers
- Routing queries to the correct server's tools based on content
- Combining results from multiple servers into a unified response
- Implementing a tool registry that organizes tools by server origin

Healthcare context:
A clinical AI assistant needs simultaneous access to multiple data systems:
medications, lab results, and vital signs. Each system is exposed as a
separate MCP server. The agent must route each query to the correct server
and combine results when a query spans multiple domains.

Servers: medication_server, lab_server, vitals_server

Usage:
    python exercise_2_multi_server_agent.py
"""

import os
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


# ============================================================================
# Simulated MCP Server Backends
# ============================================================================

# --- Medication Server ---
MEDICATION_DB = {
    "metformin": {
        "class": "Biguanide", "indication": "Type 2 Diabetes",
        "dose": "500-2000mg daily", "side_effects": ["nausea", "diarrhea"],
        "contraindications": ["eGFR < 30", "metabolic acidosis"],
    },
    "lisinopril": {
        "class": "ACE Inhibitor", "indication": "Hypertension, Heart Failure",
        "dose": "10-40mg daily", "side_effects": ["dry cough", "hyperkalemia"],
        "contraindications": ["bilateral RAS", "pregnancy"],
    },
    "atorvastatin": {
        "class": "Statin", "indication": "Hyperlipidemia",
        "dose": "10-80mg daily", "side_effects": ["myalgia", "elevated LFTs"],
        "contraindications": ["active liver disease", "pregnancy"],
    },
    "amlodipine": {
        "class": "CCB", "indication": "Hypertension, Angina",
        "dose": "5-10mg daily", "side_effects": ["edema", "dizziness"],
        "contraindications": ["severe aortic stenosis"],
    },
    "semaglutide": {
        "class": "GLP-1 RA", "indication": "Type 2 Diabetes, Obesity",
        "dose": "0.25-2mg weekly injection", "side_effects": ["nausea", "vomiting"],
        "contraindications": ["MEN2", "medullary thyroid carcinoma"],
    },
}

DRUG_INTERACTIONS = {
    ("metformin", "lisinopril"): {"severity": "minor", "note": "ACE-I may enhance hypoglycemic effect"},
    ("atorvastatin", "amlodipine"): {"severity": "moderate", "note": "Limit atorvastatin to 20mg"},
    ("metformin", "semaglutide"): {"severity": "minor", "note": "Common combo, monitor GI effects"},
}

# --- Lab Server ---
PATIENT_LABS = {
    "P001": {
        "name": "John Smith", "date": "2026-02-20",
        "results": [
            {"test": "hemoglobin", "value": 14.2, "unit": "g/dL", "flag": "normal"},
            {"test": "glucose", "value": 118, "unit": "mg/dL", "flag": "high"},
            {"test": "creatinine", "value": 1.1, "unit": "mg/dL", "flag": "normal"},
            {"test": "hba1c", "value": 7.8, "unit": "%", "flag": "high"},
            {"test": "potassium", "value": 4.2, "unit": "mEq/L", "flag": "normal"},
        ],
    },
    "P002": {
        "name": "Maria Garcia", "date": "2026-02-22",
        "results": [
            {"test": "hemoglobin", "value": 11.5, "unit": "g/dL", "flag": "low"},
            {"test": "glucose", "value": 92, "unit": "mg/dL", "flag": "normal"},
            {"test": "creatinine", "value": 0.8, "unit": "mg/dL", "flag": "normal"},
            {"test": "tsh", "value": 6.2, "unit": "mIU/L", "flag": "high"},
        ],
    },
}

LAB_RANGES = {
    "hemoglobin": {"low": 12.0, "high": 17.5}, "glucose": {"low": 70, "high": 100},
    "creatinine": {"low": 0.7, "high": 1.3}, "hba1c": {"low": 4.0, "high": 5.6},
    "potassium": {"low": 3.5, "high": 5.0}, "tsh": {"low": 0.4, "high": 4.0},
}

# --- Vitals Server ---
PATIENT_VITALS = {
    "P001": {
        "name": "John Smith",
        "readings": [
            {"date": "2026-02-20", "bp": "138/85", "hr": 78, "temp": 98.4, "spo2": 97, "weight_kg": 88.5},
            {"date": "2026-02-15", "bp": "142/88", "hr": 82, "temp": 98.6, "spo2": 96, "weight_kg": 89.0},
        ],
    },
    "P002": {
        "name": "Maria Garcia",
        "readings": [
            {"date": "2026-02-22", "bp": "125/78", "hr": 72, "temp": 98.2, "spo2": 99, "weight_kg": 65.0},
        ],
    },
}


# ============================================================================
# Server Tool Implementations
# ============================================================================

def medication_server_call(tool_name: str, args: dict) -> dict:
    """Route calls to the medication server."""
    if tool_name == "lookup_medication":
        name = args["medication_name"].lower()
        if name in MEDICATION_DB:
            return {"found": True, "medication": name, **MEDICATION_DB[name]}
        return {"found": False, "error": f"'{name}' not in formulary"}

    elif tool_name == "check_interaction":
        d1, d2 = args["drug_a"].lower(), args["drug_b"].lower()
        key1, key2 = (d1, d2), (d2, d1)
        if key1 in DRUG_INTERACTIONS:
            return {"drugs": [d1, d2], **DRUG_INTERACTIONS[key1]}
        elif key2 in DRUG_INTERACTIONS:
            return {"drugs": [d1, d2], **DRUG_INTERACTIONS[key2]}
        return {"drugs": [d1, d2], "severity": "none", "note": "No interaction documented"}

    elif tool_name == "list_medications":
        return {"medications": list(MEDICATION_DB.keys()), "total": len(MEDICATION_DB)}

    return {"error": f"Unknown tool: {tool_name}"}


def lab_server_call(tool_name: str, args: dict) -> dict:
    """Route calls to the lab server."""
    if tool_name == "get_lab_results":
        pid = args["patient_id"]
        if pid in PATIENT_LABS:
            data = PATIENT_LABS[pid]
            results = data["results"]
            if "test_name" in args and args["test_name"]:
                results = [r for r in results if r["test"] == args["test_name"].lower()]
            return {"patient_id": pid, "name": data["name"], "date": data["date"], "results": results}
        return {"error": f"Patient {pid} not found"}

    elif tool_name == "interpret_lab":
        test = args["test_name"].lower()
        value = args["value"]
        if test in LAB_RANGES:
            rng = LAB_RANGES[test]
            flag = "LOW" if value < rng["low"] else "HIGH" if value > rng["high"] else "NORMAL"
            return {"test": test, "value": value, "flag": flag, "range": f"{rng['low']}-{rng['high']}"}
        return {"error": f"Unknown test: {test}"}

    return {"error": f"Unknown tool: {tool_name}"}


def vitals_server_call(tool_name: str, args: dict) -> dict:
    """Route calls to the vitals server."""
    if tool_name == "get_vitals":
        pid = args["patient_id"]
        if pid in PATIENT_VITALS:
            data = PATIENT_VITALS[pid]
            latest = data["readings"][0]
            return {"patient_id": pid, "name": data["name"], "latest": latest}
        return {"error": f"Patient {pid} not found"}

    elif tool_name == "get_vitals_trend":
        pid = args["patient_id"]
        if pid in PATIENT_VITALS:
            data = PATIENT_VITALS[pid]
            return {"patient_id": pid, "name": data["name"], "readings": data["readings"]}
        return {"error": f"Patient {pid} not found"}

    elif tool_name == "calculate_bmi":
        w, h = args["weight_kg"], args["height_m"]
        bmi = round(w / (h ** 2), 1)
        cat = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else \
              "Overweight" if bmi < 30 else "Obese"
        return {"bmi": bmi, "category": cat}

    return {"error": f"Unknown tool: {tool_name}"}


# ============================================================================
# Tool Registry (multi-server)
# ============================================================================

class ToolRegistry:
    """Registry that tracks tools from multiple MCP servers."""

    def __init__(self):
        self.tools = {}
        self.server_map = {}  # tool_name → server_name

    def register_server(self, server_name: str, tools: list, executor: callable):
        """Register a server's tools."""
        for t in tools:
            self.tools[t["name"]] = {
                "definition": t,
                "server": server_name,
                "executor": executor,
            }
            self.server_map[t["name"]] = server_name

    def list_tools(self, server: str = None) -> list:
        """List all tools, optionally filtered by server."""
        result = []
        for name, info in self.tools.items():
            if server is None or info["server"] == server:
                result.append({
                    "name": name,
                    "server": info["server"],
                    "description": info["definition"]["description"],
                })
        return result

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool by name, routing to the correct server."""
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found",
                    "available": list(self.tools.keys())}
        info = self.tools[tool_name]
        return info["executor"](tool_name, arguments)

    def route_query(self, query: str) -> str:
        """Determine which server should handle a query (keyword-based)."""
        query_lower = query.lower()
        med_keywords = ["medication", "drug", "dose", "interaction", "prescrib",
                        "formulary", "side effect", "contraindication"]
        lab_keywords = ["lab", "result", "test", "hemoglobin", "glucose", "creatinine",
                        "hba1c", "tsh", "potassium", "interpret"]
        vitals_keywords = ["vital", "blood pressure", "bp", "heart rate", "temperature",
                           "oxygen", "spo2", "weight", "bmi"]

        scores = {"medication_server": 0, "lab_server": 0, "vitals_server": 0}
        for kw in med_keywords:
            if kw in query_lower:
                scores["medication_server"] += 1
        for kw in lab_keywords:
            if kw in query_lower:
                scores["lab_server"] += 1
        for kw in vitals_keywords:
            if kw in query_lower:
                scores["vitals_server"] += 1

        return max(scores, key=scores.get)


# ============================================================================
# Build the Registry
# ============================================================================

def build_registry() -> ToolRegistry:
    """Create and populate the tool registry with all three servers."""
    registry = ToolRegistry()

    med_tools = [
        {"name": "lookup_medication",
         "description": "Look up medication details: class, dosing, side effects",
         "inputSchema": {"type": "object",
                         "properties": {"medication_name": {"type": "string"}},
                         "required": ["medication_name"]}},
        {"name": "check_interaction",
         "description": "Check drug-drug interactions between two medications",
         "inputSchema": {"type": "object",
                         "properties": {"drug_a": {"type": "string"}, "drug_b": {"type": "string"}},
                         "required": ["drug_a", "drug_b"]}},
        {"name": "list_medications",
         "description": "List all available medications in the formulary",
         "inputSchema": {"type": "object", "properties": {}, "required": []}},
    ]
    registry.register_server("medication_server", med_tools, medication_server_call)

    lab_tools = [
        {"name": "get_lab_results",
         "description": "Get lab results for a patient",
         "inputSchema": {"type": "object",
                         "properties": {"patient_id": {"type": "string"},
                                        "test_name": {"type": "string"}},
                         "required": ["patient_id"]}},
        {"name": "interpret_lab",
         "description": "Interpret a lab value as normal/low/high",
         "inputSchema": {"type": "object",
                         "properties": {"test_name": {"type": "string"},
                                        "value": {"type": "number"}},
                         "required": ["test_name", "value"]}},
    ]
    registry.register_server("lab_server", lab_tools, lab_server_call)

    vitals_tools = [
        {"name": "get_vitals",
         "description": "Get latest vital signs for a patient",
         "inputSchema": {"type": "object",
                         "properties": {"patient_id": {"type": "string"}},
                         "required": ["patient_id"]}},
        {"name": "get_vitals_trend",
         "description": "Get vital signs trend over time",
         "inputSchema": {"type": "object",
                         "properties": {"patient_id": {"type": "string"}},
                         "required": ["patient_id"]}},
        {"name": "calculate_bmi",
         "description": "Calculate BMI from weight and height",
         "inputSchema": {"type": "object",
                         "properties": {"weight_kg": {"type": "number"},
                                        "height_m": {"type": "number"}},
                         "required": ["weight_kg", "height_m"]}},
    ]
    registry.register_server("vitals_server", vitals_tools, vitals_server_call)

    return registry


# ============================================================================
# Demo / Test
# ============================================================================

def main():
    """Demonstrate the multi-server agent."""
    print("=" * 70)
    print("  Exercise 2: Multi-Server Agent")
    print("  Using tools from 3 simulated MCP servers")
    print("=" * 70)

    registry = build_registry()

    # 1. List all available tools
    print("\n--- All Available Tools ---")
    all_tools = registry.list_tools()
    for t in all_tools:
        print(f"  [{t['server']:<20}] {t['name']:<25} — {t['description'][:45]}")
    print(f"\n  Total: {len(all_tools)} tools across 3 servers")

    # 2. Query routing
    print("\n--- Query Routing ---")
    test_queries = [
        "What are the side effects of metformin?",
        "Show me the latest lab results for patient P001",
        "What is the patient's blood pressure?",
        "Can I take lisinopril and atorvastatin together?",
        "What is the patient's BMI?",
        "Interpret HbA1c of 7.8%",
    ]
    for q in test_queries:
        server = registry.route_query(q)
        print(f"  \"{q[:55]}{'...' if len(q) > 55 else ''}\"")
        print(f"    → {server}")

    # 3. Execute tools across servers
    print("\n--- Cross-Server Tool Execution ---")
    calls = [
        ("lookup_medication", {"medication_name": "metformin"}),
        ("get_lab_results", {"patient_id": "P001"}),
        ("get_vitals", {"patient_id": "P001"}),
        ("check_interaction", {"drug_a": "atorvastatin", "drug_b": "amlodipine"}),
        ("calculate_bmi", {"weight_kg": 88.5, "height_m": 1.78}),
        ("interpret_lab", {"test_name": "hba1c", "value": 7.8}),
    ]
    for tool_name, args in calls:
        server = registry.server_map.get(tool_name, "?")
        result = registry.call_tool(tool_name, args)
        result_str = json.dumps(result)
        print(f"\n  {tool_name} (via {server})")
        print(f"    Args:   {json.dumps(args)}")
        print(f"    Result: {result_str[:80]}{'...' if len(result_str) > 80 else ''}")

    # 4. Comprehensive patient review (multi-server)
    print("\n" + "─" * 60)
    print("  Comprehensive Patient Review: P001")
    print("─" * 60)
    vitals = registry.call_tool("get_vitals", {"patient_id": "P001"})
    labs = registry.call_tool("get_lab_results", {"patient_id": "P001"})
    med = registry.call_tool("lookup_medication", {"medication_name": "metformin"})
    bmi = registry.call_tool("calculate_bmi", {"weight_kg": 88.5, "height_m": 1.78})

    print(f"\n  Patient: {vitals.get('name', 'P001')}")
    v = vitals.get("latest", {})
    print(f"  Vitals:  BP {v.get('bp')}, HR {v.get('hr')}, SpO2 {v.get('spo2')}%")
    print(f"  BMI:     {bmi.get('bmi')} ({bmi.get('category')})")
    print(f"  Labs:")
    for lab in labs.get("results", []):
        flag = "✓" if lab["flag"] == "normal" else "⚠"
        print(f"    {flag} {lab['test']}: {lab['value']} {lab['unit']} [{lab['flag']}]")
    print(f"  Current Med: {med.get('medication', '?')} ({med.get('class', '?')})")

    # 5. Error handling
    print("\n--- Error Handling ---")
    result = registry.call_tool("nonexistent_tool", {})
    print(f"  Unknown tool: {result}")
    result = registry.call_tool("get_vitals", {"patient_id": "P999"})
    print(f"  Unknown patient: {result}")

    print(f"\n  ✓ Multi-server agent exercised {len(calls)} tool calls across 3 servers")


if __name__ == "__main__":
    main()
