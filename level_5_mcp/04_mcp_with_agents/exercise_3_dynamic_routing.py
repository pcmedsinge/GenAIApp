"""
Exercise 3: Dynamic Routing
==============================

Skills practiced:
- Building a LangGraph-style workflow with dynamic routing based on query content
- Classifying queries to determine which MCP server to route to
- Implementing conditional edges in a workflow graph
- Handling multi-domain queries that span multiple servers

Healthcare context:
A clinical AI assistant receives diverse queries: medication questions,
lab result inquiries, vital sign checks, and complex questions that span
multiple domains. This exercise builds a dynamic routing workflow that
classifies each query and routes it to the appropriate MCP server.

Routing rules:
- Medication questions → medication_server
- Lab/test questions → lab_server
- Vitals/measurement questions → vitals_server
- Multi-domain queries → multiple servers in sequence

Usage:
    python exercise_3_dynamic_routing.py
"""

import os
import json
import re
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
# Mock MCP Server Data
# ============================================================================

MEDICATION_DB = {
    "metformin": {"class": "Biguanide", "indication": "T2DM", "dose": "500-2000mg daily"},
    "lisinopril": {"class": "ACE Inhibitor", "indication": "HTN", "dose": "10-40mg daily"},
    "atorvastatin": {"class": "Statin", "indication": "Hyperlipidemia", "dose": "10-80mg daily"},
}

PATIENT_LABS = {
    "P001": [
        {"test": "hemoglobin", "value": 14.2, "unit": "g/dL", "flag": "normal"},
        {"test": "glucose", "value": 118, "unit": "mg/dL", "flag": "high"},
        {"test": "hba1c", "value": 7.8, "unit": "%", "flag": "high"},
    ],
}

PATIENT_VITALS = {
    "P001": {"bp": "138/85", "hr": 78, "temp": 98.4, "spo2": 97, "weight_kg": 88.5},
}

DRUG_INTERACTIONS = {
    ("metformin", "lisinopril"): {"severity": "minor", "note": "ACE-I enhances hypoglycemia risk slightly"},
    ("atorvastatin", "amlodipine"): {"severity": "moderate", "note": "Limit atorvastatin to 20mg"},
}


# ============================================================================
# MCP Server Executors
# ============================================================================

def med_server_execute(tool: str, args: dict) -> dict:
    """Medication server tool execution."""
    if tool == "lookup_medication":
        name = args.get("medication_name", "").lower()
        if name in MEDICATION_DB:
            return {"found": True, "medication": name, **MEDICATION_DB[name]}
        return {"error": f"'{name}' not found"}
    elif tool == "check_interaction":
        d1, d2 = args.get("drug_a", "").lower(), args.get("drug_b", "").lower()
        for key in [(d1, d2), (d2, d1)]:
            if key in DRUG_INTERACTIONS:
                return {"drugs": [d1, d2], **DRUG_INTERACTIONS[key]}
        return {"drugs": [d1, d2], "severity": "none", "note": "No interaction"}
    return {"error": f"Unknown med tool: {tool}"}


def lab_server_execute(tool: str, args: dict) -> dict:
    """Lab server tool execution."""
    if tool == "get_lab_results":
        pid = args.get("patient_id", "")
        if pid in PATIENT_LABS:
            return {"patient_id": pid, "results": PATIENT_LABS[pid]}
        return {"error": f"Patient {pid} not found"}
    elif tool == "interpret_lab":
        test = args.get("test_name", "").lower()
        value = args.get("value", 0)
        ranges = {"hemoglobin": (12, 17.5), "glucose": (70, 100), "hba1c": (4, 5.6)}
        if test in ranges:
            low, high = ranges[test]
            flag = "LOW" if value < low else "HIGH" if value > high else "NORMAL"
            return {"test": test, "value": value, "flag": flag}
        return {"error": f"Unknown test: {test}"}
    return {"error": f"Unknown lab tool: {tool}"}


def vitals_server_execute(tool: str, args: dict) -> dict:
    """Vitals server tool execution."""
    if tool == "get_vitals":
        pid = args.get("patient_id", "")
        if pid in PATIENT_VITALS:
            return {"patient_id": pid, **PATIENT_VITALS[pid]}
        return {"error": f"Patient {pid} not found"}
    elif tool == "calculate_bmi":
        w, h = args.get("weight_kg", 0), args.get("height_m", 0)
        if h > 0:
            bmi = round(w / (h ** 2), 1)
            cat = "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
            return {"bmi": bmi, "category": cat}
        return {"error": "Invalid height"}
    return {"error": f"Unknown vitals tool: {tool}"}


# ============================================================================
# Query Classifier
# ============================================================================

class QueryClassifier:
    """Classify clinical queries by domain to determine routing."""

    PATTERNS = {
        "medication": [
            r"\b(medication|drug|dose|dosing|prescri|formulary)\b",
            r"\b(side effect|contraindication|interaction)\b",
            r"\b(metformin|lisinopril|atorvastatin|amlodipine|omeprazole|semaglutide)\b",
        ],
        "lab": [
            r"\b(lab|result|test|value|interpret)\b",
            r"\b(hemoglobin|glucose|creatinine|hba1c|tsh|potassium|sodium|wbc)\b",
            r"\b(blood (count|test|work))\b",
        ],
        "vitals": [
            r"\b(vital|blood pressure|bp|heart rate|pulse|temperature|temp)\b",
            r"\b(oxygen|spo2|saturation|weight|bmi|height)\b",
        ],
    }

    @classmethod
    def classify(cls, query: str) -> list:
        """Return list of matching domains, ordered by relevance."""
        query_lower = query.lower()
        scores = {}
        for domain, patterns in cls.PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                score += len(matches)
            if score > 0:
                scores[domain] = score

        if not scores:
            return ["medication"]  # default

        return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)


# ============================================================================
# Workflow Graph (LangGraph-style)
# ============================================================================

class WorkflowState:
    """State that flows through the workflow graph."""

    def __init__(self, query: str, patient_id: str = "P001"):
        self.query = query
        self.patient_id = patient_id
        self.domains = []
        self.results = {}
        self.steps = []
        self.final_response = ""


class ClinicalWorkflow:
    """LangGraph-style workflow with dynamic routing to MCP servers."""

    def __init__(self):
        self.nodes = {
            "classify": self._classify_node,
            "medication": self._medication_node,
            "lab": self._lab_node,
            "vitals": self._vitals_node,
            "synthesize": self._synthesize_node,
        }

    def _classify_node(self, state: WorkflowState) -> WorkflowState:
        """Classify the query and determine routing."""
        state.domains = QueryClassifier.classify(state.query)
        state.steps.append(f"classify → domains: {state.domains}")
        return state

    def _medication_node(self, state: WorkflowState) -> WorkflowState:
        """Process medication-related queries."""
        state.steps.append("medication_server")
        # Extract medication names from query
        med_results = []
        for med in MEDICATION_DB:
            if med in state.query.lower():
                result = med_server_execute("lookup_medication", {"medication_name": med})
                med_results.append(result)

        # Check for interaction queries
        if "interaction" in state.query.lower() or "together" in state.query.lower():
            meds = [m for m in MEDICATION_DB if m in state.query.lower()]
            if len(meds) >= 2:
                result = med_server_execute("check_interaction",
                                            {"drug_a": meds[0], "drug_b": meds[1]})
                med_results.append(result)

        if not med_results:
            # Default: list available medications
            med_results = [{"available_medications": list(MEDICATION_DB.keys())}]

        state.results["medication"] = med_results
        return state

    def _lab_node(self, state: WorkflowState) -> WorkflowState:
        """Process lab-related queries."""
        state.steps.append("lab_server")
        results = lab_server_execute("get_lab_results", {"patient_id": state.patient_id})
        state.results["lab"] = results
        return state

    def _vitals_node(self, state: WorkflowState) -> WorkflowState:
        """Process vitals-related queries."""
        state.steps.append("vitals_server")
        results = vitals_server_execute("get_vitals", {"patient_id": state.patient_id})
        state.results["vitals"] = results
        return state

    def _synthesize_node(self, state: WorkflowState) -> WorkflowState:
        """Synthesize results into a final response."""
        state.steps.append("synthesize")
        parts = []
        for domain, data in state.results.items():
            parts.append(f"[{domain.upper()}] {json.dumps(data)[:100]}")
        state.final_response = " | ".join(parts) if parts else "No results."
        return state

    def run(self, query: str, patient_id: str = "P001") -> WorkflowState:
        """Execute the workflow for a query."""
        state = WorkflowState(query, patient_id)

        # Step 1: Classify
        state = self._classify_node(state)

        # Step 2: Route to appropriate servers
        for domain in state.domains:
            if domain in self.nodes:
                state = self.nodes[domain](state)

        # Step 3: Synthesize
        state = self._synthesize_node(state)

        return state


# ============================================================================
# Demo / Test
# ============================================================================

def main():
    """Demonstrate dynamic routing workflow."""
    print("=" * 70)
    print("  Exercise 3: Dynamic Routing")
    print("  LangGraph workflow with query-based MCP server routing")
    print("=" * 70)

    workflow = ClinicalWorkflow()

    # 1. Query classification tests
    print("\n--- Query Classification ---")
    test_queries = [
        "What are the side effects of metformin?",
        "Show me lab results for patient P001",
        "What is the patient's blood pressure?",
        "Check interaction between metformin and lisinopril",
        "Review patient P001 labs and vitals",
        "Is the patient's glucose level normal?",
        "What medications is the patient on and what are the latest labs?",
    ]
    for q in test_queries:
        domains = QueryClassifier.classify(q)
        domain_str = ", ".join(domains)
        print(f"  \"{q[:55]}{'...' if len(q) > 55 else ''}\"")
        print(f"    → Domains: {domain_str}")

    # 2. Full workflow execution
    print("\n--- Workflow Execution ---")
    workflow_queries = [
        ("What are the side effects of metformin?", "P001"),
        ("Show me all lab results", "P001"),
        ("What is the patient's blood pressure and heart rate?", "P001"),
        ("Check metformin and lisinopril interaction", "P001"),
        ("Review labs and vitals together", "P001"),
    ]

    for query, pid in workflow_queries:
        print(f"\n{'─' * 60}")
        print(f"  Query: \"{query}\"")
        print(f"  Patient: {pid}")
        print(f"{'─' * 60}")

        state = workflow.run(query, pid)

        print(f"  Routing:  {' → '.join(state.steps)}")
        print(f"  Domains:  {', '.join(state.domains)}")
        for domain, data in state.results.items():
            data_str = json.dumps(data)
            print(f"  [{domain:>12}]: {data_str[:70]}{'...' if len(data_str) > 70 else ''}")

    # 3. Show the workflow graph structure
    print(f"\n{'─' * 60}")
    print("  Workflow Graph Structure")
    print(f"{'─' * 60}")
    print("""
  [CLASSIFY] ──┬── medication? ──→ [MED SERVER] ──┐
               ├── lab?        ──→ [LAB SERVER] ──┤
               └── vitals?     ──→ [VITALS SRV] ──┘
                                                    │
                                              [SYNTHESIZE]

  Conditional routing based on query classification.
  Multi-domain queries visit multiple servers.
    """)

    # 4. LangGraph equivalent code
    print("  LangGraph equivalent:")
    print("""
    from langgraph.graph import StateGraph, END

    def route_by_domain(state):
        domains = state["domains"]
        if "medication" in domains:
            return "medication"
        elif "lab" in domains:
            return "lab"
        elif "vitals" in domains:
            return "vitals"
        return "synthesize"

    workflow = StateGraph(ClinicalState)
    workflow.add_node("classify", classify_node)
    workflow.add_node("medication", med_node)
    workflow.add_node("lab", lab_node)
    workflow.add_node("vitals", vitals_node)
    workflow.add_node("synthesize", synthesize_node)

    workflow.set_entry_point("classify")
    workflow.add_conditional_edges("classify", route_by_domain,
        {"medication": "medication", "lab": "lab",
         "vitals": "vitals", "synthesize": "synthesize"})
    """)

    print(f"  ✓ Dynamic routing tested with {len(workflow_queries)} clinical queries")


if __name__ == "__main__":
    main()
