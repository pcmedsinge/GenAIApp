"""
Level 5 - Project 03: MCP Resources
======================================

Understanding MCP resources — read-only data endpoints that servers expose
to AI agents via URI-based addressing.

Builds on: 01_mcp_fundamentals (protocol) and 02_mcp_tool_servers (tools).

MCP resources complement tools:
- **Tools** perform actions (calculate BMI, check interactions)
- **Resources** expose read-only data (formulary list, reference ranges)

Resources use URIs for addressing and support templates for parameterized
access. This module demonstrates static resources, dynamic resources with
URI templates, different content types, and resource catalogs.

We define resources using FastMCP (when available) and provide standalone
simulated versions for demonstration.

Usage:
    python main.py
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Note: 'mcp' package not installed. Install with: pip install mcp")
    print("      Demos will use simulated resource data.\n")


# ============================================================================
# Shared healthcare data
# ============================================================================

FORMULARY = {
    "metformin": {
        "generic": "metformin", "brand": "Glucophage", "class": "Biguanide",
        "tier": 1, "status": "preferred", "prior_auth": False,
        "indication": "Type 2 Diabetes Mellitus",
        "common_dose": "500-2000mg daily",
    },
    "lisinopril": {
        "generic": "lisinopril", "brand": "Zestril", "class": "ACE Inhibitor",
        "tier": 1, "status": "preferred", "prior_auth": False,
        "indication": "Hypertension, Heart Failure",
        "common_dose": "10-40mg daily",
    },
    "atorvastatin": {
        "generic": "atorvastatin", "brand": "Lipitor", "class": "Statin",
        "tier": 1, "status": "preferred", "prior_auth": False,
        "indication": "Hyperlipidemia",
        "common_dose": "10-80mg daily",
    },
    "amlodipine": {
        "generic": "amlodipine", "brand": "Norvasc", "class": "Calcium Channel Blocker",
        "tier": 1, "status": "preferred", "prior_auth": False,
        "indication": "Hypertension, Angina",
        "common_dose": "5-10mg daily",
    },
    "semaglutide": {
        "generic": "semaglutide", "brand": "Ozempic", "class": "GLP-1 Receptor Agonist",
        "tier": 3, "status": "non-preferred", "prior_auth": True,
        "indication": "Type 2 Diabetes, Weight Management",
        "common_dose": "0.25-2mg weekly (injection)",
    },
}

LAB_REFERENCE_RANGES = {
    "hemoglobin": {"unit": "g/dL", "low": 12.0, "high": 17.5, "critical_low": 7.0, "critical_high": 20.0},
    "glucose": {"unit": "mg/dL", "low": 70, "high": 100, "critical_low": 40, "critical_high": 500},
    "potassium": {"unit": "mEq/L", "low": 3.5, "high": 5.0, "critical_low": 2.5, "critical_high": 6.5},
    "sodium": {"unit": "mEq/L", "low": 136, "high": 145, "critical_low": 120, "critical_high": 160},
    "creatinine": {"unit": "mg/dL", "low": 0.7, "high": 1.3, "critical_low": 0.4, "critical_high": 10.0},
    "hba1c": {"unit": "%", "low": 4.0, "high": 5.6, "critical_low": 3.0, "critical_high": 15.0},
    "tsh": {"unit": "mIU/L", "low": 0.4, "high": 4.0, "critical_low": 0.01, "critical_high": 100.0},
    "wbc": {"unit": "K/uL", "low": 4.5, "high": 11.0, "critical_low": 2.0, "critical_high": 30.0},
}

PATIENT_VITALS = {
    "P001": {
        "name": "John Smith", "age": 62,
        "vitals": [
            {"date": "2026-02-20", "bp": "138/85", "hr": 78, "temp": 98.4, "spo2": 97, "weight_kg": 88.5},
            {"date": "2026-02-15", "bp": "142/88", "hr": 82, "temp": 98.6, "spo2": 96, "weight_kg": 89.0},
            {"date": "2026-02-01", "bp": "145/90", "hr": 80, "temp": 98.5, "spo2": 97, "weight_kg": 89.2},
        ],
    },
    "P002": {
        "name": "Maria Garcia", "age": 45,
        "vitals": [
            {"date": "2026-02-22", "bp": "125/78", "hr": 72, "temp": 98.2, "spo2": 99, "weight_kg": 65.0},
            {"date": "2026-02-10", "bp": "122/76", "hr": 70, "temp": 98.4, "spo2": 98, "weight_kg": 65.2},
        ],
    },
    "P003": {
        "name": "Robert Wilson", "age": 71,
        "vitals": [
            {"date": "2026-02-25", "bp": "155/92", "hr": 68, "temp": 97.9, "spo2": 94, "weight_kg": 95.3},
            {"date": "2026-02-18", "bp": "150/88", "hr": 70, "temp": 98.0, "spo2": 95, "weight_kg": 95.8},
            {"date": "2026-02-05", "bp": "148/86", "hr": 72, "temp": 98.1, "spo2": 95, "weight_kg": 96.0},
        ],
    },
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
# DEMO 1: Static Resources
# ============================================================================

def demo_static_resources():
    """DEMO 1: Expose fixed data as MCP resources with URIs."""
    print_banner("DEMO 1: Static Resources")

    print("""
  Static resources expose fixed, read-only data at known URIs.
  Unlike tools (which take arguments and perform actions), resources
  simply return data when read.

  Examples of static healthcare resources:
    • formulary://medications          → list of all formulary medications
    • reference://lab-ranges           → standard lab reference ranges
    • protocol://sepsis-bundle         → sepsis treatment protocol text
    """)

    # Define MCP resource server (if SDK available)
    if MCP_AVAILABLE:
        mcp = FastMCP("Healthcare Resources")

        @mcp.resource("formulary://medications")
        def get_formulary() -> str:
            """Complete formulary medication list with tier and status."""
            return json.dumps({
                "resource": "formulary://medications",
                "description": "Hospital formulary — all approved medications",
                "medications": [
                    {"name": med["generic"], "brand": med["brand"],
                     "class": med["class"], "tier": med["tier"],
                     "status": med["status"]}
                    for med in FORMULARY.values()
                ],
                "last_updated": "2026-02-01",
            }, indent=2)

        @mcp.resource("reference://lab-ranges")
        def get_lab_ranges() -> str:
            """Standard laboratory reference ranges."""
            return json.dumps({
                "resource": "reference://lab-ranges",
                "description": "Standard adult lab reference ranges",
                "ranges": LAB_REFERENCE_RANGES,
                "note": "Ranges may vary by age, sex, and lab methodology",
            }, indent=2)

        print("  ✓ Defined MCP resources using FastMCP:")
        print("    • formulary://medications")
        print("    • reference://lab-ranges")
    else:
        print("  (MCP SDK not available — showing simulated resources)")

    # Simulate reading resources (works with or without SDK)
    print("\n  --- Reading formulary://medications ---")
    formulary_data = {
        "resource": "formulary://medications",
        "description": "Hospital formulary — all approved medications",
        "medications": [
            {"name": med["generic"], "brand": med["brand"],
             "class": med["class"], "tier": med["tier"], "status": med["status"]}
            for med in FORMULARY.values()
        ],
        "last_updated": "2026-02-01",
    }
    print_json("Formulary Resource", formulary_data)

    print("\n  --- Reading reference://lab-ranges ---")
    lab_data = {
        "resource": "reference://lab-ranges",
        "ranges": {k: v for k, v in list(LAB_REFERENCE_RANGES.items())[:3]},
        "note": f"Showing 3 of {len(LAB_REFERENCE_RANGES)} available ranges",
    }
    print_json("Lab Reference Ranges (sample)", lab_data)

    # Show the JSON-RPC protocol message for reading a resource
    print("\n  --- JSON-RPC: resources/read ---")
    request_msg = {
        "jsonrpc": "2.0", "id": 1,
        "method": "resources/read",
        "params": {"uri": "formulary://medications"}
    }
    response_msg = {
        "jsonrpc": "2.0", "id": 1,
        "result": {
            "contents": [{
                "uri": "formulary://medications",
                "mimeType": "application/json",
                "text": json.dumps(formulary_data)[:200] + "..."
            }]
        }
    }
    print_json("Request", request_msg)
    print_json("Response (truncated)", response_msg)

    print("\n  ✓ Static resources provide fixed data at stable URIs")


# ============================================================================
# DEMO 2: Dynamic Resources
# ============================================================================

def demo_dynamic_resources():
    """DEMO 2: Resources with URI templates that accept parameters."""
    print_banner("DEMO 2: Dynamic Resources")

    print("""
  Dynamic resources use URI templates with parameters. The server
  generates content based on the parameter values in the URI.

  Template examples:
    • patient/{id}/vitals       → vitals for a specific patient
    • formulary/{drug_name}     → details for a specific medication
    • lab-range/{test_name}     → reference range for a specific test
    """)

    if MCP_AVAILABLE:
        mcp = FastMCP("Dynamic Resources")

        @mcp.resource("patient/{patient_id}/vitals")
        def get_patient_vitals(patient_id: str) -> str:
            """Get vitals for a specific patient by ID."""
            if patient_id in PATIENT_VITALS:
                data = PATIENT_VITALS[patient_id]
                return json.dumps({
                    "patient_id": patient_id,
                    "name": data["name"],
                    "vitals": data["vitals"],
                }, indent=2)
            return json.dumps({"error": f"Patient {patient_id} not found"})

        print("  ✓ Defined URI template resource: patient/{patient_id}/vitals")

    # Demonstrate reading dynamic resources
    for pid in ["P001", "P002", "P003"]:
        data = PATIENT_VITALS.get(pid, {})
        print(f"\n  --- Reading patient/{pid}/vitals ---")
        print(f"    Patient: {data.get('name', 'Unknown')}, Age: {data.get('age', '?')}")
        if data.get("vitals"):
            latest = data["vitals"][0]
            print(f"    Latest vitals ({latest['date']}):")
            print(f"      BP: {latest['bp']}  HR: {latest['hr']}  "
                  f"Temp: {latest['temp']}°F  SpO2: {latest['spo2']}%")

    # Show JSON-RPC for URI template
    print("\n  --- JSON-RPC: resources/read with template ---")
    request_msg = {
        "jsonrpc": "2.0", "id": 5,
        "method": "resources/read",
        "params": {"uri": "patient/P001/vitals"}
    }
    print_json("Request", request_msg)

    print("\n  Key difference from tools:")
    print("    • Tool:     tools/call → {name: 'get_vitals', arguments: {patient_id: 'P001'}}")
    print("    • Resource: resources/read → {uri: 'patient/P001/vitals'}")
    print("    Resources are read-only; tools can have side effects.")


# ============================================================================
# DEMO 3: Resource Content Types
# ============================================================================

def demo_resource_content_types():
    """DEMO 3: Text, JSON, and markdown resources for different data types."""
    print_banner("DEMO 3: Resource Content Types")

    print("""
  MCP resources can return different content types. The mimeType field
  tells the client how to interpret the content:

    • text/plain          — simple text (notes, summaries)
    • application/json    — structured data (lab results, medication info)
    • text/markdown       — formatted documentation (guidelines, protocols)
    """)

    # Text resource
    print("  --- 1. Text Resource (text/plain) ---")
    text_resource = {
        "uri": "note://patient/P001/assessment",
        "mimeType": "text/plain",
        "text": ("Assessment: 62-year-old male with poorly controlled hypertension. "
                 "BP trending down on current regimen (amlodipine 10mg + lisinopril 20mg). "
                 "Continue current medications. Recheck in 2 weeks.")
    }
    print_json("Text Resource", text_resource)

    # JSON resource
    print("\n  --- 2. JSON Resource (application/json) ---")
    json_resource = {
        "uri": "lab://patient/P001/results/latest",
        "mimeType": "application/json",
        "text": json.dumps({
            "patient_id": "P001",
            "date": "2026-02-20",
            "results": [
                {"test": "hemoglobin", "value": 14.2, "unit": "g/dL", "flag": "normal"},
                {"test": "glucose", "value": 118, "unit": "mg/dL", "flag": "high"},
                {"test": "creatinine", "value": 1.1, "unit": "mg/dL", "flag": "normal"},
                {"test": "potassium", "value": 4.2, "unit": "mEq/L", "flag": "normal"},
            ]
        }, indent=2)
    }
    print_json("JSON Resource", json_resource)

    # Markdown resource
    print("\n  --- 3. Markdown Resource (text/markdown) ---")
    md_content = """# Hypertension Management Protocol

## First-Line Agents
- **ACE Inhibitors** (lisinopril, enalapril) — preferred for DM, CKD
- **ARBs** (losartan, valsartan) — use if ACE-I intolerant (cough)
- **CCBs** (amlodipine) — preferred for isolated systolic HTN
- **Thiazide diuretics** (HCTZ, chlorthalidone) — low cost, effective

## Target Blood Pressure
- General: < 130/80 mmHg (ACC/AHA 2017)
- Age ≥ 65: < 130/80 mmHg (consider frailty)
- CKD with proteinuria: < 130/80 mmHg

## Monitoring
- Recheck BP in 4 weeks after dose change
- Annual: BMP (K+, Cr), lipid panel
"""
    md_resource = {
        "uri": "protocol://cardiology/hypertension",
        "mimeType": "text/markdown",
        "text": md_content
    }
    print_json("Markdown Resource", md_resource)

    # Show how agent would process different types
    print("\n  Content type determines how the agent processes the data:")
    print("    • text/plain       → inject directly into context")
    print("    • application/json → parse and extract specific fields")
    print("    • text/markdown    → render as formatted documentation")
    print("\n  ✓ Choose content type based on the data and how agents will use it")


# ============================================================================
# DEMO 4: Resource Catalog
# ============================================================================

def demo_resource_catalog():
    """DEMO 4: List all available resources with descriptions."""
    print_banner("DEMO 4: Resource Catalog")

    print("""
  An MCP server exposes a catalog of available resources via the
  resources/list method. This lets agents discover what data is
  available before reading it.
    """)

    # Define a comprehensive resource catalog
    resource_catalog = [
        {
            "uri": "formulary://medications",
            "name": "Formulary Medication List",
            "description": "Complete hospital formulary with tier and status for all approved medications",
            "mimeType": "application/json",
        },
        {
            "uri": "formulary://medication/{name}",
            "name": "Medication Detail",
            "description": "Detailed information for a specific medication including dosing, side effects, and contraindications",
            "mimeType": "application/json",
        },
        {
            "uri": "reference://lab-ranges",
            "name": "Lab Reference Ranges",
            "description": "Standard adult laboratory reference ranges with critical values",
            "mimeType": "application/json",
        },
        {
            "uri": "reference://lab-range/{test_name}",
            "name": "Specific Lab Range",
            "description": "Reference range for a specific lab test",
            "mimeType": "application/json",
        },
        {
            "uri": "patient/{id}/vitals",
            "name": "Patient Vitals",
            "description": "Recent vital signs for a patient, ordered by date (most recent first)",
            "mimeType": "application/json",
        },
        {
            "uri": "patient/{id}/medications",
            "name": "Patient Medications",
            "description": "Current medication list for a patient",
            "mimeType": "application/json",
        },
        {
            "uri": "protocol://cardiology/hypertension",
            "name": "Hypertension Protocol",
            "description": "Clinical protocol for hypertension management including targets and monitoring",
            "mimeType": "text/markdown",
        },
        {
            "uri": "protocol://endocrinology/diabetes",
            "name": "Diabetes Protocol",
            "description": "Clinical protocol for Type 2 diabetes management",
            "mimeType": "text/markdown",
        },
        {
            "uri": "note://patient/{id}/assessment",
            "name": "Patient Assessment Note",
            "description": "Most recent clinical assessment note for a patient",
            "mimeType": "text/plain",
        },
    ]

    # Show the resources/list JSON-RPC flow
    print("  --- JSON-RPC: resources/list ---")
    request_msg = {
        "jsonrpc": "2.0", "id": 10,
        "method": "resources/list",
    }
    print_json("Request", request_msg)

    response_msg = {
        "jsonrpc": "2.0", "id": 10,
        "result": {"resources": resource_catalog},
    }
    print_json("Response", response_msg)

    # Display the catalog in a readable table
    print("\n\n  Resource Catalog Summary:")
    print("  " + "─" * 66)
    print(f"  {'URI':<40} {'Type':<20} {'MIME'}")
    print("  " + "─" * 66)
    for r in resource_catalog:
        is_template = "{" in r["uri"]
        rtype = "template" if is_template else "static"
        print(f"  {r['uri']:<40} {rtype:<20} {r['mimeType']}")

    print(f"\n  Total resources: {len(resource_catalog)}")
    print(f"    Static:    {sum(1 for r in resource_catalog if '{' not in r['uri'])}")
    print(f"    Templates: {sum(1 for r in resource_catalog if '{' in r['uri'])}")

    # Show how an agent uses the catalog
    print("\n  Agent workflow with resource catalog:")
    print("    1. Call resources/list to discover available resources")
    print("    2. Match user query to relevant resource URIs")
    print("    3. Call resources/read with the chosen URI")
    print("    4. Process the content based on mimeType")
    print("    5. Incorporate data into response to user")

    # Demonstrate resource selection logic
    print("\n  --- Simulated: Agent selects resource for query ---")
    query = "What are the latest vitals for patient P001?"
    print(f"    Query: \"{query}\"")
    print("    Agent reasoning:")
    print("      → Query mentions 'vitals' and 'patient P001'")
    print("      → Best match: patient/{id}/vitals with id=P001")
    print("      → Reading: patient/P001/vitals")
    if "P001" in PATIENT_VITALS:
        latest = PATIENT_VITALS["P001"]["vitals"][0]
        print(f"      → Result: BP {latest['bp']}, HR {latest['hr']}, "
              f"SpO2 {latest['spo2']}%")


# ============================================================================
# Main Menu
# ============================================================================

def main():
    """Run MCP resource demos interactively."""
    print("=" * 70)
    print("  Level 5 - Project 03: MCP Resources")
    print("  Read-only data endpoints with URI addressing")
    print("=" * 70)

    if MCP_AVAILABLE:
        print("  ✓ MCP SDK available")
    else:
        print("  ⚠ MCP SDK not installed (demos use simulated resources)")

    demos = {
        "1": ("Static Resources", demo_static_resources),
        "2": ("Dynamic Resources", demo_dynamic_resources),
        "3": ("Resource Content Types", demo_resource_content_types),
        "4": ("Resource Catalog", demo_resource_catalog),
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
