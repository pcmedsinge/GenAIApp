"""
Exercise 3: Resource Templates
================================

Skills practiced:
- Building parameterized MCP resource templates with multiple parameters
- Implementing age- and sex-adjusted reference ranges for lab tests
- Using URI query parameters for resource customization
- Returning clinically relevant, context-specific data

Healthcare context:
Laboratory reference ranges vary by age, sex, and sometimes ethnicity. A
"normal" hemoglobin for a 25-year-old female (12.0-16.0 g/dL) differs from
a 70-year-old male (13.0-17.0 g/dL). This exercise builds a resource
template that returns adjusted reference ranges:

    lab_reference/{test_name}?age={age}&sex={sex}

The server returns the appropriate reference range based on demographic
parameters, enabling AI agents to make accurate lab interpretations.

Usage:
    python exercise_3_resource_templates.py
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


# ============================================================================
# Age/Sex-Adjusted Lab Reference Ranges
# ============================================================================
# Each test has a base range plus adjustments by age group and sex.

LAB_REFERENCES = {
    "hemoglobin": {
        "unit": "g/dL",
        "description": "Oxygen-carrying protein in red blood cells",
        "ranges": {
            ("male", "adult"): {"low": 13.5, "high": 17.5, "critical_low": 7.0, "critical_high": 20.0},
            ("female", "adult"): {"low": 12.0, "high": 16.0, "critical_low": 7.0, "critical_high": 20.0},
            ("male", "elderly"): {"low": 12.5, "high": 17.0, "critical_low": 7.0, "critical_high": 20.0},
            ("female", "elderly"): {"low": 11.5, "high": 15.5, "critical_low": 7.0, "critical_high": 20.0},
            ("male", "child"): {"low": 11.5, "high": 15.5, "critical_low": 7.0, "critical_high": 20.0},
            ("female", "child"): {"low": 11.5, "high": 15.5, "critical_low": 7.0, "critical_high": 20.0},
        },
    },
    "creatinine": {
        "unit": "mg/dL",
        "description": "Kidney function marker (waste product of muscle metabolism)",
        "ranges": {
            ("male", "adult"): {"low": 0.74, "high": 1.35, "critical_low": 0.4, "critical_high": 10.0},
            ("female", "adult"): {"low": 0.59, "high": 1.04, "critical_low": 0.4, "critical_high": 10.0},
            ("male", "elderly"): {"low": 0.70, "high": 1.50, "critical_low": 0.4, "critical_high": 10.0},
            ("female", "elderly"): {"low": 0.55, "high": 1.20, "critical_low": 0.4, "critical_high": 10.0},
            ("male", "child"): {"low": 0.30, "high": 0.70, "critical_low": 0.2, "critical_high": 5.0},
            ("female", "child"): {"low": 0.30, "high": 0.70, "critical_low": 0.2, "critical_high": 5.0},
        },
    },
    "glucose": {
        "unit": "mg/dL",
        "description": "Fasting blood glucose level",
        "ranges": {
            ("male", "adult"): {"low": 70, "high": 100, "critical_low": 40, "critical_high": 500},
            ("female", "adult"): {"low": 70, "high": 100, "critical_low": 40, "critical_high": 500},
            ("male", "elderly"): {"low": 70, "high": 110, "critical_low": 40, "critical_high": 500},
            ("female", "elderly"): {"low": 70, "high": 110, "critical_low": 40, "critical_high": 500},
            ("male", "child"): {"low": 60, "high": 100, "critical_low": 40, "critical_high": 500},
            ("female", "child"): {"low": 60, "high": 100, "critical_low": 40, "critical_high": 500},
        },
    },
    "potassium": {
        "unit": "mEq/L",
        "description": "Electrolyte critical for cardiac and muscle function",
        "ranges": {
            ("male", "adult"): {"low": 3.5, "high": 5.0, "critical_low": 2.5, "critical_high": 6.5},
            ("female", "adult"): {"low": 3.5, "high": 5.0, "critical_low": 2.5, "critical_high": 6.5},
            ("male", "elderly"): {"low": 3.5, "high": 5.3, "critical_low": 2.5, "critical_high": 6.5},
            ("female", "elderly"): {"low": 3.5, "high": 5.3, "critical_low": 2.5, "critical_high": 6.5},
            ("male", "child"): {"low": 3.4, "high": 4.7, "critical_low": 2.5, "critical_high": 6.5},
            ("female", "child"): {"low": 3.4, "high": 4.7, "critical_low": 2.5, "critical_high": 6.5},
        },
    },
    "hba1c": {
        "unit": "%",
        "description": "Glycated hemoglobin (3-month average blood glucose)",
        "ranges": {
            ("male", "adult"): {"low": 4.0, "high": 5.6, "critical_low": 3.0, "critical_high": 15.0},
            ("female", "adult"): {"low": 4.0, "high": 5.6, "critical_low": 3.0, "critical_high": 15.0},
            ("male", "elderly"): {"low": 4.0, "high": 6.5, "critical_low": 3.0, "critical_high": 15.0},
            ("female", "elderly"): {"low": 4.0, "high": 6.5, "critical_low": 3.0, "critical_high": 15.0},
            ("male", "child"): {"low": 4.0, "high": 5.6, "critical_low": 3.0, "critical_high": 15.0},
            ("female", "child"): {"low": 4.0, "high": 5.6, "critical_low": 3.0, "critical_high": 15.0},
        },
    },
    "tsh": {
        "unit": "mIU/L",
        "description": "Thyroid-stimulating hormone",
        "ranges": {
            ("male", "adult"): {"low": 0.4, "high": 4.0, "critical_low": 0.01, "critical_high": 100.0},
            ("female", "adult"): {"low": 0.4, "high": 4.0, "critical_low": 0.01, "critical_high": 100.0},
            ("male", "elderly"): {"low": 0.4, "high": 5.8, "critical_low": 0.01, "critical_high": 100.0},
            ("female", "elderly"): {"low": 0.4, "high": 5.8, "critical_low": 0.01, "critical_high": 100.0},
            ("male", "child"): {"low": 0.7, "high": 6.4, "critical_low": 0.01, "critical_high": 100.0},
            ("female", "child"): {"low": 0.7, "high": 6.4, "critical_low": 0.01, "critical_high": 100.0},
        },
    },
    "wbc": {
        "unit": "K/uL",
        "description": "White blood cell count (immune function)",
        "ranges": {
            ("male", "adult"): {"low": 4.5, "high": 11.0, "critical_low": 2.0, "critical_high": 30.0},
            ("female", "adult"): {"low": 4.5, "high": 11.0, "critical_low": 2.0, "critical_high": 30.0},
            ("male", "elderly"): {"low": 4.0, "high": 10.5, "critical_low": 2.0, "critical_high": 30.0},
            ("female", "elderly"): {"low": 4.0, "high": 10.5, "critical_low": 2.0, "critical_high": 30.0},
            ("male", "child"): {"low": 5.0, "high": 13.0, "critical_low": 2.0, "critical_high": 30.0},
            ("female", "child"): {"low": 5.0, "high": 13.0, "critical_low": 2.0, "critical_high": 30.0},
        },
    },
}


def _age_group(age: int) -> str:
    """Determine age group from numeric age."""
    if age < 18:
        return "child"
    elif age >= 65:
        return "elderly"
    return "adult"


# ============================================================================
# Resource Functions
# ============================================================================

def get_available_tests() -> dict:
    """List all available lab tests."""
    return {
        "resource": "lab_reference://tests",
        "tests": [
            {"name": name, "unit": info["unit"], "description": info["description"]}
            for name, info in LAB_REFERENCES.items()
        ],
        "total": len(LAB_REFERENCES),
    }


def get_reference_range(test_name: str, age: int = 40, sex: str = "male") -> dict:
    """Get age/sex-adjusted reference range for a lab test."""
    test = test_name.lower().strip()
    sex_norm = sex.lower().strip()
    if sex_norm not in ("male", "female"):
        return {"error": f"Invalid sex '{sex}'. Use 'male' or 'female'."}
    if test not in LAB_REFERENCES:
        return {
            "error": f"Test '{test_name}' not found",
            "available": sorted(LAB_REFERENCES.keys()),
        }
    if age < 0 or age > 120:
        return {"error": f"Invalid age {age}. Must be 0-120."}

    age_grp = _age_group(age)
    lab = LAB_REFERENCES[test]
    key = (sex_norm, age_grp)
    ranges = lab["ranges"].get(key)
    if not ranges:
        return {"error": f"No range data for {sex_norm}/{age_grp}"}

    return {
        "resource": f"lab_reference://{test}?age={age}&sex={sex_norm}",
        "test": test,
        "description": lab["description"],
        "unit": lab["unit"],
        "patient_demographics": {"age": age, "sex": sex_norm, "age_group": age_grp},
        "reference_range": {
            "low": ranges["low"],
            "high": ranges["high"],
            "critical_low": ranges["critical_low"],
            "critical_high": ranges["critical_high"],
        },
        "note": f"Adjusted for {sex_norm} {age_grp} (age {age})",
    }


def interpret_with_range(test_name: str, value: float, age: int = 40, sex: str = "male") -> dict:
    """Interpret a lab value using age/sex-adjusted range."""
    ref = get_reference_range(test_name, age, sex)
    if "error" in ref:
        return ref

    rng = ref["reference_range"]
    if value < rng["critical_low"]:
        flag = "CRITICAL LOW"
    elif value < rng["low"]:
        flag = "LOW"
    elif value <= rng["high"]:
        flag = "NORMAL"
    elif value <= rng["critical_high"]:
        flag = "HIGH"
    else:
        flag = "CRITICAL HIGH"

    return {
        **ref,
        "value": value,
        "flag": flag,
        "interpretation": f"{value} {ref['unit']} is {flag} "
                         f"(range: {rng['low']}-{rng['high']} {ref['unit']} "
                         f"for {sex} age {age})",
    }


# ============================================================================
# MCP Server Definition
# ============================================================================

if MCP_AVAILABLE:
    mcp = FastMCP("Lab Reference Ranges")

    @mcp.resource("lab_reference://tests")
    def mcp_list_tests() -> str:
        """List all available lab tests."""
        return json.dumps(get_available_tests(), indent=2)

    @mcp.resource("lab_reference://{test_name}")
    def mcp_get_range(test_name: str) -> str:
        """Get default reference range for a lab test."""
        return json.dumps(get_reference_range(test_name), indent=2)


# ============================================================================
# Demo / Test
# ============================================================================

def main():
    """Demonstrate parameterized lab reference templates."""
    print("=" * 70)
    print("  Exercise 3: Resource Templates")
    print("  Template: lab_reference/{test_name}?age={age}&sex={sex}")
    print("=" * 70)

    # 1. List available tests
    print("\n--- Available Lab Tests ---")
    tests = get_available_tests()
    for t in tests["tests"]:
        print(f"  • {t['name']:<15} ({t['unit']:<8}) — {t['description']}")

    # 2. Show how ranges differ by demographics
    print("\n--- Age/Sex-Adjusted Ranges for Hemoglobin ---")
    demo_cases = [
        (25, "male"), (25, "female"),
        (45, "male"), (45, "female"),
        (70, "male"), (70, "female"),
        (10, "male"), (10, "female"),
    ]
    print(f"  {'Age':<5} {'Sex':<8} {'Group':<10} {'Low':>6} {'High':>6} {'Unit':<6}")
    print(f"  {'─'*5} {'─'*8} {'─'*10} {'─'*6} {'─'*6} {'─'*6}")
    for age, sex in demo_cases:
        ref = get_reference_range("hemoglobin", age, sex)
        rng = ref["reference_range"]
        grp = ref["patient_demographics"]["age_group"]
        print(f"  {age:<5} {sex:<8} {grp:<10} {rng['low']:>6.1f} {rng['high']:>6.1f} {ref['unit']}")

    # 3. Same for creatinine
    print("\n--- Age/Sex-Adjusted Ranges for Creatinine ---")
    print(f"  {'Age':<5} {'Sex':<8} {'Group':<10} {'Low':>6} {'High':>6} {'Unit':<6}")
    print(f"  {'─'*5} {'─'*8} {'─'*10} {'─'*6} {'─'*6} {'─'*6}")
    for age, sex in demo_cases:
        ref = get_reference_range("creatinine", age, sex)
        rng = ref["reference_range"]
        grp = ref["patient_demographics"]["age_group"]
        print(f"  {age:<5} {sex:<8} {grp:<10} {rng['low']:>6.2f} {rng['high']:>6.2f} {ref['unit']}")

    # 4. Interpret lab values with demographic context
    print("\n--- Lab Interpretation with Demographic Adjustment ---")
    interpretations = [
        ("hemoglobin", 11.0, 30, "female"),
        ("hemoglobin", 11.0, 30, "male"),
        ("creatinine", 1.2, 45, "male"),
        ("creatinine", 1.2, 45, "female"),
        ("glucose", 105, 70, "male"),
        ("glucose", 105, 40, "male"),
        ("tsh", 5.5, 75, "female"),
        ("tsh", 5.5, 35, "female"),
    ]
    for test, value, age, sex in interpretations:
        result = interpret_with_range(test, value, age, sex)
        flag = result["flag"]
        marker = {"NORMAL": "✓", "LOW": "↓", "HIGH": "↑",
                   "CRITICAL LOW": "⚠↓", "CRITICAL HIGH": "⚠↑"}.get(flag, "?")
        print(f"  {marker} {test} = {value} for {sex} age {age}: {flag}")

    # 5. Show the URI template pattern
    print(f"\n{'─' * 60}")
    print("  URI Template Examples:")
    print(f"{'─' * 60}")
    examples = [
        "lab_reference://hemoglobin?age=30&sex=female",
        "lab_reference://creatinine?age=70&sex=male",
        "lab_reference://glucose?age=10&sex=male",
        "lab_reference://tsh?age=45&sex=female",
    ]
    for uri in examples:
        print(f"  {uri}")

    # 6. Error handling
    print(f"\n--- Error Handling ---")
    result = get_reference_range("albumin", 40, "male")
    print(f"  Unknown test:  {result.get('error', '')}")
    result = get_reference_range("hemoglobin", 40, "other")
    print(f"  Invalid sex:   {result.get('error', '')}")

    if MCP_AVAILABLE:
        print("\n  ✓ MCP server defined — run with: mcp run exercise_3_resource_templates.py")


if __name__ == "__main__":
    main()
