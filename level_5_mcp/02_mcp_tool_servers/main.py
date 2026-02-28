"""
Level 5 - Project 02: MCP Tool Servers
========================================

Building robust MCP tool servers for healthcare AI agents. This module
demonstrates how to create production-quality tool servers using the
`mcp` Python SDK with FastMCP.

Builds on: 01_mcp_fundamentals (protocol architecture and message format).

MCP tool servers expose Python functions as tools that any MCP-compatible
agent can discover and invoke. This module covers:

- Building a healthcare tool server with clinical tools
- How tool descriptions and schemas guide agent behavior
- Error handling and graceful failure in MCP tools
- Organizing tools into categories for discoverability

Since running a live MCP server requires a transport connection, we define
the tools using FastMCP (when available) and also provide standalone
callable versions for demonstration and testing.

Usage:
    python main.py
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
    print("Note: 'mcp' package not installed. Install with: pip install mcp")
    print("      Demos will use standalone functions.\n")


# ============================================================================
# Shared healthcare databases
# ============================================================================

MEDICATION_DB = {
    "metformin": {
        "generic": "metformin",
        "brand": "Glucophage",
        "class": "Biguanide",
        "indication": "Type 2 Diabetes Mellitus",
        "mechanism": "Decreases hepatic glucose production, increases insulin sensitivity",
        "common_dose": "500-2000mg daily in divided doses",
        "max_dose": "2550mg/day",
        "side_effects": ["nausea", "diarrhea", "abdominal pain", "lactic acidosis (rare)"],
        "contraindications": ["eGFR < 30", "metabolic acidosis", "severe hepatic impairment"],
    },
    "lisinopril": {
        "generic": "lisinopril",
        "brand": "Zestril / Prinivil",
        "class": "ACE Inhibitor",
        "indication": "Hypertension, Heart Failure, Post-MI",
        "mechanism": "Inhibits angiotensin-converting enzyme, reduces aldosterone",
        "common_dose": "10-40mg once daily",
        "max_dose": "80mg/day",
        "side_effects": ["dry cough", "hyperkalemia", "dizziness", "angioedema (rare)"],
        "contraindications": ["bilateral renal artery stenosis", "pregnancy", "history of angioedema"],
    },
    "atorvastatin": {
        "generic": "atorvastatin",
        "brand": "Lipitor",
        "class": "HMG-CoA Reductase Inhibitor (Statin)",
        "indication": "Hyperlipidemia, Cardiovascular Risk Reduction",
        "mechanism": "Inhibits HMG-CoA reductase, reduces LDL cholesterol synthesis",
        "common_dose": "10-80mg once daily",
        "max_dose": "80mg/day",
        "side_effects": ["myalgia", "elevated liver enzymes", "GI upset", "rhabdomyolysis (rare)"],
        "contraindications": ["active liver disease", "pregnancy", "breastfeeding"],
    },
    "amlodipine": {
        "generic": "amlodipine",
        "brand": "Norvasc",
        "class": "Calcium Channel Blocker (Dihydropyridine)",
        "indication": "Hypertension, Angina",
        "mechanism": "Blocks L-type calcium channels in vascular smooth muscle",
        "common_dose": "5-10mg once daily",
        "max_dose": "10mg/day",
        "side_effects": ["peripheral edema", "dizziness", "flushing", "headache"],
        "contraindications": ["severe aortic stenosis", "cardiogenic shock"],
    },
    "omeprazole": {
        "generic": "omeprazole",
        "brand": "Prilosec",
        "class": "Proton Pump Inhibitor (PPI)",
        "indication": "GERD, Peptic Ulcer Disease, H. pylori (combo)",
        "mechanism": "Irreversibly inhibits H+/K+ ATPase in gastric parietal cells",
        "common_dose": "20-40mg once daily",
        "max_dose": "40mg/day (OTC); 360mg/day (Zollinger-Ellison)",
        "side_effects": ["headache", "diarrhea", "B12 deficiency (long-term)", "C. diff risk"],
        "contraindications": ["rilpivirine co-administration"],
    },
}

DRUG_INTERACTIONS = {
    ("metformin", "lisinopril"): {
        "severity": "minor",
        "description": "ACE inhibitors may slightly enhance hypoglycemic effect",
        "recommendation": "Monitor blood glucose. Usually safe to co-prescribe.",
    },
    ("atorvastatin", "amlodipine"): {
        "severity": "moderate",
        "description": "Amlodipine may increase atorvastatin exposure by ~20%",
        "recommendation": "Limit atorvastatin to 20mg when combined with amlodipine.",
    },
    ("metformin", "omeprazole"): {
        "severity": "minor",
        "description": "Long-term PPI use may reduce B12 absorption, exacerbating metformin B12 effect",
        "recommendation": "Monitor B12 levels annually if long-term combination.",
    },
    ("lisinopril", "atorvastatin"): {
        "severity": "none",
        "description": "No clinically significant interaction",
        "recommendation": "Safe to co-prescribe. Common combination for cardiovascular risk.",
    },
    ("amlodipine", "omeprazole"): {
        "severity": "none",
        "description": "No clinically significant interaction",
        "recommendation": "Safe to co-prescribe.",
    },
}

LAB_REFERENCE_RANGES = {
    "hemoglobin": {"unit": "g/dL", "low": 12.0, "high": 17.5, "critical_low": 7.0, "critical_high": 20.0},
    "glucose": {"unit": "mg/dL", "low": 70, "high": 100, "critical_low": 40, "critical_high": 500},
    "potassium": {"unit": "mEq/L", "low": 3.5, "high": 5.0, "critical_low": 2.5, "critical_high": 6.5},
    "sodium": {"unit": "mEq/L", "low": 136, "high": 145, "critical_low": 120, "critical_high": 160},
    "creatinine": {"unit": "mg/dL", "low": 0.7, "high": 1.3, "critical_low": 0.4, "critical_high": 10.0},
    "wbc": {"unit": "K/uL", "low": 4.5, "high": 11.0, "critical_low": 2.0, "critical_high": 30.0},
    "platelets": {"unit": "K/uL", "low": 150, "high": 400, "critical_low": 50, "critical_high": 1000},
    "alt": {"unit": "U/L", "low": 7, "high": 56, "critical_low": 0, "critical_high": 1000},
    "hba1c": {"unit": "%", "low": 4.0, "high": 5.6, "critical_low": 3.0, "critical_high": 15.0},
    "tsh": {"unit": "mIU/L", "low": 0.4, "high": 4.0, "critical_low": 0.01, "critical_high": 100.0},
}


# ============================================================================
# Helper functions
# ============================================================================

def print_json(label: str, obj, indent: int = 2):
    """Pretty-print a JSON-serializable object with a label."""
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
# Clinical tool implementations
# ============================================================================

def calculate_bmi(weight_kg: float, height_m: float) -> dict:
    """Calculate BMI with WHO classification."""
    if weight_kg <= 0 or height_m <= 0:
        return {"error": "Weight and height must be positive numbers"}
    bmi = round(weight_kg / (height_m ** 2), 1)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    elif bmi < 35:
        category = "Obese Class I"
    elif bmi < 40:
        category = "Obese Class II"
    else:
        category = "Obese Class III"
    return {"bmi": bmi, "category": category, "weight_kg": weight_kg, "height_m": height_m}


def lookup_medication(medication_name: str) -> dict:
    """Look up medication information from the database."""
    name = medication_name.lower().strip()
    if name in MEDICATION_DB:
        return MEDICATION_DB[name]
    # Try partial match
    matches = [k for k in MEDICATION_DB if name in k or k in name]
    if matches:
        return MEDICATION_DB[matches[0]]
    return {"error": f"Medication '{medication_name}' not found in database",
            "available": list(MEDICATION_DB.keys())}


def interpret_lab_value(test_name: str, value: float, unit: str = "") -> dict:
    """Interpret a laboratory result against reference ranges."""
    test_key = test_name.lower().strip()
    if test_key not in LAB_REFERENCE_RANGES:
        return {"error": f"Lab test '{test_name}' not found",
                "available": list(LAB_REFERENCE_RANGES.keys())}

    ref = LAB_REFERENCE_RANGES[test_key]
    if value <= ref["critical_low"]:
        status = "critically low"
        urgency = "critical"
    elif value < ref["low"]:
        status = "low"
        urgency = "abnormal"
    elif value <= ref["high"]:
        status = "normal"
        urgency = "normal"
    elif value < ref["critical_high"]:
        status = "high"
        urgency = "abnormal"
    else:
        status = "critically high"
        urgency = "critical"

    return {
        "test": test_name,
        "value": value,
        "unit": unit or ref["unit"],
        "status": status,
        "urgency": urgency,
        "reference_range": f"{ref['low']}-{ref['high']} {ref['unit']}",
        "critical_range": f"<{ref['critical_low']} or >{ref['critical_high']} {ref['unit']}",
    }


def check_drug_interaction(drug_a: str, drug_b: str) -> dict:
    """Check for known interactions between two medications."""
    a = drug_a.lower().strip()
    b = drug_b.lower().strip()

    # Check both orderings
    key = (a, b) if (a, b) in DRUG_INTERACTIONS else (b, a)
    if key in DRUG_INTERACTIONS:
        interaction = DRUG_INTERACTIONS[key]
        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            **interaction,
        }

    # Check if drugs exist
    if a not in MEDICATION_DB:
        return {"error": f"Drug '{drug_a}' not found in database"}
    if b not in MEDICATION_DB:
        return {"error": f"Drug '{drug_b}' not found in database"}

    return {
        "drug_a": drug_a,
        "drug_b": drug_b,
        "severity": "unknown",
        "description": "No interaction data available for this combination",
        "recommendation": "Consult a pharmacist or drug interaction database",
    }


# ============================================================================
# MCP server registration (when SDK is available)
# ============================================================================

if MCP_AVAILABLE:
    mcp = FastMCP("Healthcare Tool Server")

    @mcp.tool()
    def mcp_calculate_bmi(weight_kg: float, height_m: float) -> str:
        """Calculate Body Mass Index from weight (kg) and height (m).
        Returns BMI value and WHO category (Underweight/Normal/Overweight/Obese).
        Use for nutritional assessment or clinical documentation."""
        return json.dumps(calculate_bmi(weight_kg, height_m))

    @mcp.tool()
    def mcp_lookup_medication(medication_name: str) -> str:
        """Look up medication information: class, indication, dosing, side effects,
        and contraindications. Use when a clinician asks about a specific drug."""
        return json.dumps(lookup_medication(medication_name))

    @mcp.tool()
    def mcp_interpret_lab_value(test_name: str, value: float, unit: str = "") -> str:
        """Interpret a laboratory test result against reference ranges.
        Returns status (normal/low/high/critical) and clinical context.
        Use when reviewing lab results or explaining values to patients."""
        return json.dumps(interpret_lab_value(test_name, value, unit))

    @mcp.tool()
    def mcp_check_drug_interaction(drug_a: str, drug_b: str) -> str:
        """Check for known drug-drug interactions between two medications.
        Returns severity (none/minor/moderate/severe) and recommendation.
        Use before prescribing or when reviewing medication lists."""
        return json.dumps(check_drug_interaction(drug_a, drug_b))


# ============================================================================
# DEMO 1: Healthcare Tool Server
# ============================================================================

def demo_healthcare_tool_server():
    """DEMO 1: Build and demo a healthcare MCP tool server."""
    print_banner("DEMO 1: Healthcare Tool Server")

    print("\n  Server: 'Healthcare Tool Server'")
    print("  Tools registered: 4")
    print("  ─────────────────────────────────────────────")

    # Show server definition
    print("""
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("Healthcare Tool Server")

    @mcp.tool()
    def calculate_bmi(weight_kg: float, height_m: float) -> str:
        ...

    @mcp.tool()
    def lookup_medication(medication_name: str) -> str:
        ...

    @mcp.tool()
    def interpret_lab_value(test_name: str, value: float, unit: str = "") -> str:
        ...

    @mcp.tool()
    def check_drug_interaction(drug_a: str, drug_b: str) -> str:
        ...
    """)

    # Demo each tool
    print("  Calling each tool:")
    print("  ─────────────────────────────────────────────")

    result = calculate_bmi(82.5, 1.75)
    print(f"\n  calculate_bmi(82.5, 1.75)")
    print(f"    → BMI: {result['bmi']}, Category: {result['category']}")

    result = lookup_medication("metformin")
    print(f"\n  lookup_medication('metformin')")
    print(f"    → Class: {result['class']}, Indication: {result['indication']}")
    print(f"    → Dose: {result['common_dose']}")

    result = interpret_lab_value("hemoglobin", 10.2)
    print(f"\n  interpret_lab_value('hemoglobin', 10.2)")
    print(f"    → Status: {result['status']}, Ref: {result['reference_range']}")

    result = check_drug_interaction("atorvastatin", "amlodipine")
    print(f"\n  check_drug_interaction('atorvastatin', 'amlodipine')")
    print(f"    → Severity: {result['severity']}")
    print(f"    → {result['recommendation']}")


# ============================================================================
# DEMO 2: Tool Schemas and Descriptions
# ============================================================================

def demo_tool_schemas():
    """DEMO 2: Show how descriptions guide agent behavior and examine schemas."""
    print_banner("DEMO 2: Tool Schemas and Descriptions")

    print("""
  Tool DESCRIPTIONS guide agent behavior. When an AI agent receives a user
  query, it reads tool descriptions to decide which tool to call.

  Good description: "Calculate Body Mass Index from weight (kg) and height (m).
                     Returns BMI value and WHO category. Use for nutritional
                     assessment or clinical documentation."

  Bad description:  "BMI calculator"  ← Too vague, agent won't know when to use it
    """)

    # Show auto-generated schemas
    tools = {
        "calculate_bmi": {
            "description": "Calculate BMI from weight (kg) and height (m). Returns BMI and WHO category.",
            "parameters": {
                "weight_kg": {"type": "number", "required": True, "description": "Patient weight in kilograms"},
                "height_m": {"type": "number", "required": True, "description": "Patient height in meters"},
            }
        },
        "lookup_medication": {
            "description": "Look up drug info: class, indication, dosing, side effects, contraindications.",
            "parameters": {
                "medication_name": {"type": "string", "required": True, "description": "Name of the medication"},
            }
        },
        "interpret_lab_value": {
            "description": "Interpret a lab result against reference ranges. Returns status and clinical context.",
            "parameters": {
                "test_name": {"type": "string", "required": True, "description": "Lab test name"},
                "value": {"type": "number", "required": True, "description": "Numeric result value"},
                "unit": {"type": "string", "required": False, "description": "Unit of measurement"},
            }
        },
        "check_drug_interaction": {
            "description": "Check for drug-drug interactions. Returns severity and recommendation.",
            "parameters": {
                "drug_a": {"type": "string", "required": True, "description": "First medication"},
                "drug_b": {"type": "string", "required": True, "description": "Second medication"},
            }
        },
    }

    print("  Auto-generated tool schemas (what MCP clients see):")
    print("  ─────────────────────────────────────────────")

    for name, info in tools.items():
        print(f"\n  Tool: {name}")
        print(f"  Description: {info['description']}")
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        for param_name, param_info in info["parameters"].items():
            schema["properties"][param_name] = {
                "type": param_info["type"],
                "description": param_info["description"]
            }
            if param_info.get("required"):
                schema["required"].append(param_name)
        print_json("Input Schema", schema)

    print("\n  Agent decision example:")
    print("  ─────────────────────────────────────────────")
    print("  User: 'What is this patient's BMI? They weigh 78kg and are 1.72m tall.'")
    print("  Agent reads tool descriptions → selects 'calculate_bmi'")
    print("  Agent extracts parameters → {weight_kg: 78.0, height_m: 1.72}")
    print("  Agent calls tool → gets result → formats response for user")


# ============================================================================
# DEMO 3: Error Handling in Tools
# ============================================================================

def demo_error_handling():
    """DEMO 3: Handle invalid inputs, missing data, and failures gracefully."""
    print_banner("DEMO 3: Error Handling in Tools")

    print("""
  MCP tools must handle errors gracefully. A tool that crashes gives the
  agent no information to work with. Instead, return structured error
  messages that the agent can understand and communicate to the user.
    """)

    error_test_cases = [
        {
            "label": "Invalid BMI input (negative weight)",
            "tool": "calculate_bmi",
            "args": {"weight_kg": -5.0, "height_m": 1.75},
            "func": calculate_bmi,
        },
        {
            "label": "Unknown medication",
            "tool": "lookup_medication",
            "args": {"medication_name": "nonexistidol"},
            "func": lookup_medication,
        },
        {
            "label": "Unknown lab test",
            "tool": "interpret_lab_value",
            "args": {"test_name": "unobtanium", "value": 5.0},
            "func": interpret_lab_value,
        },
        {
            "label": "Drug interaction with unknown drug",
            "tool": "check_drug_interaction",
            "args": {"drug_a": "metformin", "drug_b": "fantasymycin"},
            "func": check_drug_interaction,
        },
        {
            "label": "Zero height (division by zero)",
            "tool": "calculate_bmi",
            "args": {"weight_kg": 70.0, "height_m": 0},
            "func": calculate_bmi,
        },
    ]

    for test in error_test_cases:
        print(f"\n  Test: {test['label']}")
        print(f"  Tool: {test['tool']}({test['args']})")

        try:
            result = test["func"](**test["args"])

            if "error" in result:
                print(f"  → Graceful error: {result['error']}")
                if "available" in result:
                    print(f"    Available: {result['available'][:5]}...")
                print("  ✓ Tool returned structured error (good!)")
            else:
                print(f"  → Result: {result}")
                print("  ⚠ Expected an error but got success")
        except Exception as e:
            print(f"  → EXCEPTION: {type(e).__name__}: {e}")
            print("  ✗ Tool crashed instead of returning error (bad!)")

    # Show the MCP error response format
    print("\n\n  MCP Error Response Format (JSON-RPC):")
    print("  ─────────────────────────────────────────────")
    mcp_error = {
        "jsonrpc": "2.0",
        "id": 5,
        "error": {
            "code": -32602,
            "message": "Invalid params",
            "data": {
                "field": "weight_kg",
                "issue": "must be positive",
                "received": -5.0
            }
        }
    }
    print_json("Error response example", mcp_error)

    print("\n  Best practices for MCP tool error handling:")
    print("  ─────────────────────────────────────────────")
    print("  1. Never let tools crash — always return structured error")
    print("  2. Include what went wrong and how to fix it")
    print("  3. List available options when input doesn't match")
    print("  4. Use isError flag in MCP result for tool-level errors")
    print("  5. Validate inputs at the start of every tool function")


# ============================================================================
# DEMO 4: Tool Categories
# ============================================================================

def demo_tool_categories():
    """DEMO 4: Organize tools into categories for discoverability."""
    print_banner("DEMO 4: Tool Categories")

    print("""
  When a server has many tools, organizing them into categories helps:
  - Agents find the right tool faster
  - Developers navigate the tool catalog
  - Deployment can expose tool subsets per role
    """)

    categories = {
        "clinical": {
            "description": "Clinical assessment and calculation tools",
            "tools": [
                {
                    "name": "calculate_bmi",
                    "description": "Body Mass Index calculator with WHO classification",
                    "example_call": "calculate_bmi(weight_kg=78.0, height_m=1.72)",
                },
                {
                    "name": "assess_cardiovascular_risk",
                    "description": "10-year ASCVD risk calculator (ACC/AHA pooled cohort)",
                    "example_call": "assess_cv_risk(age=55, sex='male', systolic_bp=140, ...)",
                },
                {
                    "name": "calculate_egfr",
                    "description": "Estimated GFR using CKD-EPI formula",
                    "example_call": "calculate_egfr(creatinine=1.4, age=65, sex='male')",
                },
            ],
        },
        "medication": {
            "description": "Medication information and safety tools",
            "tools": [
                {
                    "name": "lookup_medication",
                    "description": "Drug information lookup (class, dose, side effects)",
                    "example_call": "lookup_medication(medication_name='metformin')",
                },
                {
                    "name": "check_drug_interaction",
                    "description": "Drug-drug interaction checker with severity",
                    "example_call": "check_interaction(drug_a='warfarin', drug_b='aspirin')",
                },
                {
                    "name": "check_allergy_cross_reactivity",
                    "description": "Cross-reactivity check for drug allergies",
                    "example_call": "check_allergy(allergy='penicillin', proposed='amoxicillin')",
                },
            ],
        },
        "laboratory": {
            "description": "Laboratory result interpretation tools",
            "tools": [
                {
                    "name": "interpret_lab_value",
                    "description": "Interpret a single lab value against reference ranges",
                    "example_call": "interpret_lab(test='hemoglobin', value=10.2, unit='g/dL')",
                },
                {
                    "name": "check_critical_value",
                    "description": "Determine if a lab result is critically abnormal",
                    "example_call": "check_critical(test='potassium', value=6.8)",
                },
                {
                    "name": "trend_lab_values",
                    "description": "Analyze trend across multiple lab results over time",
                    "example_call": "trend_labs(test='creatinine', values=[1.0, 1.2, 1.5, 1.8])",
                },
            ],
        },
    }

    for cat_name, cat_info in categories.items():
        print(f"\n  [{cat_name.upper()}] {cat_info['description']}")
        print("  " + "─" * 55)
        for tool in cat_info["tools"]:
            print(f"    • {tool['name']}")
            print(f"      {tool['description']}")
            print(f"      Example: {tool['example_call']}")

    # Show how to implement categories via naming convention
    print("\n\n  Implementation strategies for categories:")
    print("  ─────────────────────────────────────────────")
    print("""
  Strategy 1: Naming convention (prefix)
    @mcp.tool()
    def clinical_calculate_bmi(...): ...
    def medication_lookup_drug(...): ...
    def lab_interpret_value(...): ...

  Strategy 2: Separate servers per category
    clinical_server = FastMCP("Clinical Tools")
    medication_server = FastMCP("Medication Tools")
    lab_server = FastMCP("Laboratory Tools")
    → Host connects to each as separate MCP clients

  Strategy 3: Description-based categorization
    @mcp.tool()
    def calculate_bmi(...):
        \\"\\"\\"[Clinical] Calculate Body Mass Index...\\"\\"\\"

  Strategy 2 (separate servers) is the MCP-idiomatic approach.
  Each server is a focused microservice with clear responsibility.
    """)


# ============================================================================
# Main Menu
# ============================================================================

def main():
    """Run MCP tool server demos interactively."""
    print("=" * 70)
    print("  Level 5 - Project 02: MCP Tool Servers")
    print("  Building robust MCP tool servers for healthcare AI")
    print("=" * 70)

    if MCP_AVAILABLE:
        print("  ✓ MCP SDK available")
    else:
        print("  ⚠ MCP SDK not installed (demos use standalone functions)")

    demos = {
        "1": ("Healthcare Tool Server", demo_healthcare_tool_server),
        "2": ("Tool Schemas and Descriptions", demo_tool_schemas),
        "3": ("Error Handling in Tools", demo_error_handling),
        "4": ("Tool Categories", demo_tool_categories),
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
