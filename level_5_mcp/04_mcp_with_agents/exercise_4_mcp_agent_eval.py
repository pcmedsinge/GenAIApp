"""
Exercise 4: MCP Agent Evaluation
===================================

Skills practiced:
- Evaluating MCP-connected agent performance systematically
- Testing tool selection accuracy (did the agent pick the right tool?)
- Testing parameter extraction (did the agent extract correct arguments?)
- Testing response quality (was the agent's answer clinically appropriate?)
- Generating evaluation metrics and reports

Healthcare context:
An AI agent that uses MCP tools in a clinical setting must be rigorously
evaluated. Wrong tool selection could mean checking vitals when the user
asked about medications. Wrong parameter extraction could mean looking up
the wrong patient. Poor response quality could lead to clinical errors.

This exercise builds an evaluation framework with test cases, scoring
rubrics, and metric reporting.

Usage:
    python exercise_4_mcp_agent_eval.py
"""

import os
import json
from datetime import datetime
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
# Mock MCP Tool Implementations
# ============================================================================

MEDICATION_DB = {
    "metformin": {"class": "Biguanide", "indication": "T2DM", "dose": "500-2000mg daily",
                  "side_effects": ["nausea", "diarrhea"]},
    "lisinopril": {"class": "ACE Inhibitor", "indication": "HTN", "dose": "10-40mg daily",
                   "side_effects": ["dry cough", "hyperkalemia"]},
    "atorvastatin": {"class": "Statin", "indication": "Hyperlipidemia", "dose": "10-80mg daily",
                     "side_effects": ["myalgia", "elevated LFTs"]},
}

LAB_RANGES = {
    "hemoglobin": {"low": 12.0, "high": 17.5, "unit": "g/dL"},
    "glucose": {"low": 70, "high": 100, "unit": "mg/dL"},
    "hba1c": {"low": 4.0, "high": 5.6, "unit": "%"},
    "creatinine": {"low": 0.7, "high": 1.3, "unit": "mg/dL"},
}

PATIENT_VITALS = {
    "P001": {"bp": "138/85", "hr": 78, "temp": 98.4, "spo2": 97},
    "P002": {"bp": "125/78", "hr": 72, "temp": 98.2, "spo2": 99},
}


def execute_tool(tool_name: str, args: dict) -> dict:
    """Execute a mock MCP tool."""
    if tool_name == "lookup_medication":
        name = args.get("medication_name", "").lower()
        if name in MEDICATION_DB:
            return {"found": True, **MEDICATION_DB[name]}
        return {"found": False, "error": f"Not found: {name}"}

    elif tool_name == "check_interaction":
        d1, d2 = args.get("drug_a", "").lower(), args.get("drug_b", "").lower()
        if d1 == "metformin" and d2 == "lisinopril" or d2 == "metformin" and d1 == "lisinopril":
            return {"severity": "minor", "note": "ACE-I may enhance hypoglycemia"}
        return {"severity": "none", "note": "No interaction"}

    elif tool_name == "interpret_lab":
        test = args.get("test_name", "").lower()
        value = args.get("value", 0)
        if test in LAB_RANGES:
            rng = LAB_RANGES[test]
            flag = "LOW" if value < rng["low"] else "HIGH" if value > rng["high"] else "NORMAL"
            return {"test": test, "value": value, "flag": flag}
        return {"error": f"Unknown: {test}"}

    elif tool_name == "get_vitals":
        pid = args.get("patient_id", "")
        if pid in PATIENT_VITALS:
            return {"patient_id": pid, **PATIENT_VITALS[pid]}
        return {"error": f"Not found: {pid}"}

    elif tool_name == "calculate_bmi":
        w = args.get("weight_kg", 0)
        h = args.get("height_m", 0)
        if h > 0:
            bmi = round(w / (h ** 2), 1)
            return {"bmi": bmi}
        return {"error": "Invalid height"}

    return {"error": f"Unknown tool: {tool_name}"}


# ============================================================================
# Test Case Definitions
# ============================================================================

EVAL_TEST_CASES = [
    # Tool Selection Tests
    {
        "id": "TS-001",
        "category": "tool_selection",
        "query": "What are the side effects of metformin?",
        "expected_tool": "lookup_medication",
        "expected_args": {"medication_name": "metformin"},
        "expected_in_response": ["nausea", "diarrhea"],
    },
    {
        "id": "TS-002",
        "category": "tool_selection",
        "query": "Check if metformin and lisinopril interact",
        "expected_tool": "check_interaction",
        "expected_args": {"drug_a": "metformin", "drug_b": "lisinopril"},
        "expected_in_response": ["minor"],
    },
    {
        "id": "TS-003",
        "category": "tool_selection",
        "query": "Is a hemoglobin of 10.5 g/dL normal?",
        "expected_tool": "interpret_lab",
        "expected_args": {"test_name": "hemoglobin", "value": 10.5, "unit": "g/dL"},
        "expected_in_response": ["LOW"],
    },
    {
        "id": "TS-004",
        "category": "tool_selection",
        "query": "What is patient P001's blood pressure?",
        "expected_tool": "get_vitals",
        "expected_args": {"patient_id": "P001"},
        "expected_in_response": ["138/85"],
    },
    {
        "id": "TS-005",
        "category": "tool_selection",
        "query": "Calculate BMI for 85kg and 1.75m",
        "expected_tool": "calculate_bmi",
        "expected_args": {"weight_kg": 85, "height_m": 1.75},
        "expected_in_response": ["27.8"],
    },
    # Parameter Extraction Tests
    {
        "id": "PE-001",
        "category": "parameter_extraction",
        "query": "Look up dosing for atorvastatin",
        "expected_tool": "lookup_medication",
        "expected_args": {"medication_name": "atorvastatin"},
        "expected_in_response": ["10-80mg"],
    },
    {
        "id": "PE-002",
        "category": "parameter_extraction",
        "query": "Patient P002 vitals please",
        "expected_tool": "get_vitals",
        "expected_args": {"patient_id": "P002"},
        "expected_in_response": ["125/78"],
    },
    {
        "id": "PE-003",
        "category": "parameter_extraction",
        "query": "Is glucose 155 mg/dL high?",
        "expected_tool": "interpret_lab",
        "expected_args": {"test_name": "glucose", "value": 155, "unit": "mg/dL"},
        "expected_in_response": ["HIGH"],
    },
    # Response Quality Tests
    {
        "id": "RQ-001",
        "category": "response_quality",
        "query": "Tell me about lisinopril",
        "expected_tool": "lookup_medication",
        "expected_args": {"medication_name": "lisinopril"},
        "expected_in_response": ["ACE Inhibitor", "HTN"],
    },
    {
        "id": "RQ-002",
        "category": "response_quality",
        "query": "Is HbA1c 8.2% concerning?",
        "expected_tool": "interpret_lab",
        "expected_args": {"test_name": "hba1c", "value": 8.2, "unit": "%"},
        "expected_in_response": ["HIGH"],
    },
]


# ============================================================================
# Simulated Agent (keyword-based for deterministic testing)
# ============================================================================

class SimulatedAgent:
    """Agent that selects tools based on keyword matching (deterministic)."""

    TOOL_KEYWORDS = {
        "lookup_medication": ["medication", "drug", "dose", "dosing", "side effect",
                              "tell me about", "metformin", "lisinopril", "atorvastatin"],
        "check_interaction": ["interact", "together", "combine", "co-prescribe"],
        "interpret_lab": ["lab", "hemoglobin", "glucose", "hba1c", "creatinine",
                          "normal", "high", "low", "interpret", "g/dL", "mg/dL"],
        "get_vitals": ["vital", "blood pressure", "bp", "heart rate", "spo2",
                        "patient P", "temperature"],
        "calculate_bmi": ["bmi", "body mass"],
    }

    def select_tool(self, query: str) -> str:
        """Select the best tool for a query."""
        query_lower = query.lower()
        scores = {}
        for tool, keywords in self.TOOL_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in query_lower)
            scores[tool] = score
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "unknown"

    def extract_args(self, tool: str, query: str) -> dict:
        """Extract tool arguments from query."""
        query_lower = query.lower()

        if tool == "lookup_medication":
            for med in ["metformin", "lisinopril", "atorvastatin"]:
                if med in query_lower:
                    return {"medication_name": med}
            return {"medication_name": query_lower.split()[-1]}

        elif tool == "check_interaction":
            meds = [m for m in ["metformin", "lisinopril", "atorvastatin"]
                    if m in query_lower]
            if len(meds) >= 2:
                return {"drug_a": meds[0], "drug_b": meds[1]}
            return {"drug_a": "unknown", "drug_b": "unknown"}

        elif tool == "interpret_lab":
            import re
            # Extract test name
            test = "unknown"
            for t in ["hemoglobin", "glucose", "hba1c", "creatinine"]:
                if t in query_lower:
                    test = t
                    break
            # Extract value
            numbers = re.findall(r'[\d]+\.?\d*', query)
            value = float(numbers[0]) if numbers else 0
            # Extract unit
            unit = ""
            for u in ["g/dL", "mg/dL", "mEq/L", "%"]:
                if u.lower() in query_lower or u in query:
                    unit = u
                    break
            return {"test_name": test, "value": value, "unit": unit}

        elif tool == "get_vitals":
            import re
            pid_match = re.search(r'P\d{3}', query)
            pid = pid_match.group(0) if pid_match else "P001"
            return {"patient_id": pid}

        elif tool == "calculate_bmi":
            import re
            numbers = re.findall(r'[\d]+\.?\d*', query)
            if len(numbers) >= 2:
                nums = sorted([float(n) for n in numbers], reverse=True)
                return {"weight_kg": nums[0], "height_m": nums[1]}
            return {"weight_kg": 70, "height_m": 1.7}

        return {}

    def run(self, query: str) -> dict:
        """Run the agent: select tool, extract args, execute."""
        tool = self.select_tool(query)
        args = self.extract_args(tool, query)
        result = execute_tool(tool, args)
        return {
            "query": query,
            "selected_tool": tool,
            "extracted_args": args,
            "result": result,
        }


# ============================================================================
# Evaluation Framework
# ============================================================================

class MCPAgentEvaluator:
    """Evaluate MCP-connected agent performance."""

    def __init__(self, agent):
        self.agent = agent
        self.results = []

    def run_test(self, test_case: dict) -> dict:
        """Run a single test case and score it."""
        query = test_case["query"]
        agent_output = self.agent.run(query)

        # Score tool selection (0 or 1)
        tool_correct = int(agent_output["selected_tool"] == test_case["expected_tool"])

        # Score parameter extraction (fraction of correct params)
        expected_args = test_case["expected_args"]
        extracted_args = agent_output["extracted_args"]
        param_scores = []
        for key, expected_val in expected_args.items():
            if key in extracted_args:
                if isinstance(expected_val, (int, float)):
                    param_scores.append(int(abs(extracted_args[key] - expected_val) < 0.1))
                else:
                    param_scores.append(int(str(extracted_args[key]).lower() == str(expected_val).lower()))
            else:
                param_scores.append(0)
        param_score = sum(param_scores) / len(param_scores) if param_scores else 0

        # Score response quality (fraction of expected items found in result)
        result_str = json.dumps(agent_output["result"]).lower()
        expected_items = test_case.get("expected_in_response", [])
        found = sum(1 for item in expected_items if item.lower() in result_str)
        response_score = found / len(expected_items) if expected_items else 1.0

        result = {
            "test_id": test_case["id"],
            "category": test_case["category"],
            "query": query,
            "tool_selection": {"correct": tool_correct, "selected": agent_output["selected_tool"],
                               "expected": test_case["expected_tool"]},
            "parameter_extraction": {"score": param_score, "extracted": extracted_args,
                                     "expected": expected_args},
            "response_quality": {"score": response_score, "expected_items": expected_items,
                                 "found": found, "total": len(expected_items)},
            "overall_score": round((tool_correct + param_score + response_score) / 3, 3),
        }
        self.results.append(result)
        return result

    def run_all(self, test_cases: list) -> list:
        """Run all test cases."""
        self.results = []
        for tc in test_cases:
            self.run_test(tc)
        return self.results

    def generate_report(self) -> dict:
        """Generate evaluation report with aggregate metrics."""
        if not self.results:
            return {"error": "No results to report"}

        total = len(self.results)
        tool_accuracy = sum(r["tool_selection"]["correct"] for r in self.results) / total
        param_accuracy = sum(r["parameter_extraction"]["score"] for r in self.results) / total
        response_quality = sum(r["response_quality"]["score"] for r in self.results) / total
        overall = sum(r["overall_score"] for r in self.results) / total

        # Per-category breakdown
        categories = {}
        for r in self.results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = {"tests": 0, "tool_acc": 0, "param_acc": 0,
                                    "resp_qual": 0, "overall": 0}
            categories[cat]["tests"] += 1
            categories[cat]["tool_acc"] += r["tool_selection"]["correct"]
            categories[cat]["param_acc"] += r["parameter_extraction"]["score"]
            categories[cat]["resp_qual"] += r["response_quality"]["score"]
            categories[cat]["overall"] += r["overall_score"]

        for cat, data in categories.items():
            n = data["tests"]
            data["tool_acc"] = round(data["tool_acc"] / n, 3)
            data["param_acc"] = round(data["param_acc"] / n, 3)
            data["resp_qual"] = round(data["resp_qual"] / n, 3)
            data["overall"] = round(data["overall"] / n, 3)

        return {
            "summary": {
                "total_tests": total,
                "tool_selection_accuracy": round(tool_accuracy, 3),
                "parameter_extraction_accuracy": round(param_accuracy, 3),
                "response_quality_score": round(response_quality, 3),
                "overall_score": round(overall, 3),
            },
            "by_category": categories,
            "failures": [
                {
                    "test_id": r["test_id"],
                    "query": r["query"],
                    "issue": "tool" if not r["tool_selection"]["correct"]
                             else "params" if r["parameter_extraction"]["score"] < 1
                             else "response",
                    "score": r["overall_score"],
                }
                for r in self.results if r["overall_score"] < 1.0
            ],
            "timestamp": datetime.now().isoformat(),
        }


# ============================================================================
# Demo / Test
# ============================================================================

def main():
    """Demonstrate MCP agent evaluation."""
    print("=" * 70)
    print("  Exercise 4: MCP Agent Evaluation")
    print("  Testing tool selection, parameter extraction, response quality")
    print("=" * 70)

    agent = SimulatedAgent()
    evaluator = MCPAgentEvaluator(agent)

    # 1. Run all test cases
    print(f"\n--- Running {len(EVAL_TEST_CASES)} Test Cases ---\n")
    results = evaluator.run_all(EVAL_TEST_CASES)

    for r in results:
        tool_ok = "✓" if r["tool_selection"]["correct"] else "✗"
        param_ok = "✓" if r["parameter_extraction"]["score"] >= 1.0 else "△"
        resp_ok = "✓" if r["response_quality"]["score"] >= 1.0 else "△"
        print(f"  [{r['test_id']}] {r['query'][:50]}")
        print(f"    Tool: {tool_ok} ({r['tool_selection']['selected']}) "
              f" Params: {param_ok} ({r['parameter_extraction']['score']:.1f}) "
              f" Response: {resp_ok} ({r['response_quality']['score']:.1f}) "
              f" Overall: {r['overall_score']:.2f}")

    # 2. Generate report
    print(f"\n{'─' * 60}")
    print("  Evaluation Report")
    print(f"{'─' * 60}")
    report = evaluator.generate_report()

    summary = report["summary"]
    print(f"\n  Total tests:             {summary['total_tests']}")
    print(f"  Tool selection accuracy: {summary['tool_selection_accuracy']:.1%}")
    print(f"  Parameter extraction:    {summary['parameter_extraction_accuracy']:.1%}")
    print(f"  Response quality:        {summary['response_quality_score']:.1%}")
    print(f"  Overall score:           {summary['overall_score']:.1%}")

    # 3. Per-category breakdown
    print(f"\n  By Category:")
    print(f"  {'Category':<25} {'Tests':>5} {'Tool':>7} {'Params':>7} {'Resp':>7} {'Overall':>8}")
    print(f"  {'─'*25} {'─'*5} {'─'*7} {'─'*7} {'─'*7} {'─'*8}")
    for cat, data in report["by_category"].items():
        print(f"  {cat:<25} {data['tests']:>5} {data['tool_acc']:>7.1%} "
              f"{data['param_acc']:>7.1%} {data['resp_qual']:>7.1%} {data['overall']:>8.1%}")

    # 4. Failures
    failures = report.get("failures", [])
    if failures:
        print(f"\n  Issues ({len(failures)} test(s) with imperfect scores):")
        for f in failures:
            print(f"    • [{f['test_id']}] {f['query'][:45]} — {f['issue']} (score: {f['score']:.2f})")
    else:
        print("\n  ✓ All tests passed perfectly!")

    # 5. Grade scale
    overall = summary["overall_score"]
    if overall >= 0.95:
        grade = "A"
    elif overall >= 0.85:
        grade = "B"
    elif overall >= 0.75:
        grade = "C"
    elif overall >= 0.65:
        grade = "D"
    else:
        grade = "F"
    print(f"\n  Agent Grade: {grade} ({overall:.1%})")
    print(f"\n  ✓ Evaluation complete: {summary['total_tests']} tests, "
          f"overall {overall:.1%}")


if __name__ == "__main__":
    main()
