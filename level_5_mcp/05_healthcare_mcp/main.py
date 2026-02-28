"""
Level 5 - Project 05: Healthcare MCP Capstone
================================================

A complete healthcare MCP ecosystem with EHR, Lab, and Pharmacy servers
orchestrated by a unified clinical agent.

Builds on: All Level 5 projects (MCP fundamentals, tools, resources, agents).

This capstone combines:
- EHR Server: patient lookup, demographics, encounters, vitals
- Clinical Lab Server: orders, results, interpretations, critical alerts
- Pharmacy Server: formulary, interactions, prior auth, refills
- Unified Agent: single agent that uses all three servers

Usage:
    python main.py
"""

import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Note: 'mcp' package not installed. Install with: pip install mcp")
    print("      Demos will use standalone functions.\n")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ============================================================================
# Shared Patient Database
# ============================================================================

PATIENTS = {
    "P001": {
        "name": "John Smith", "dob": "1963-08-15", "age": 62, "sex": "M",
        "mrn": "MRN-2024-001",
        "problems": ["Type 2 Diabetes", "Hypertension", "Hyperlipidemia"],
        "allergies": ["Penicillin (rash)", "Sulfa drugs (hives)"],
        "pcp": "Dr. Sarah Chen",
    },
    "P002": {
        "name": "Maria Garcia", "dob": "1980-03-22", "age": 45, "sex": "F",
        "mrn": "MRN-2024-002",
        "problems": ["Hypothyroidism", "Iron Deficiency Anemia"],
        "allergies": ["None known"],
        "pcp": "Dr. James Park",
    },
    "P003": {
        "name": "Robert Wilson", "dob": "1954-11-07", "age": 71, "sex": "M",
        "mrn": "MRN-2024-003",
        "problems": ["COPD", "Hypertension", "CKD Stage 3b", "Atrial Fibrillation"],
        "allergies": ["ACE Inhibitors (angioedema)", "Iodine contrast"],
        "pcp": "Dr. Lisa Wong",
    },
}

ENCOUNTERS = {
    "P001": [
        {"date": "2026-02-20", "type": "Office Visit", "provider": "Dr. Sarah Chen",
         "reason": "Diabetes follow-up", "assessment": "HbA1c elevated at 7.8%. Increase metformin."},
        {"date": "2026-01-15", "type": "Office Visit", "provider": "Dr. Sarah Chen",
         "reason": "Hypertension recheck", "assessment": "BP improved to 138/85. Continue amlodipine."},
        {"date": "2025-11-10", "type": "ER Visit", "provider": "Dr. Mike Johnson",
         "reason": "Hypoglycemia episode", "assessment": "Glucose 52. Treated with D50. Adjusted meds."},
    ],
    "P002": [
        {"date": "2026-02-22", "type": "Office Visit", "provider": "Dr. James Park",
         "reason": "Thyroid check", "assessment": "TSH 6.2 — slightly elevated. Increase levothyroxine."},
        {"date": "2026-01-05", "type": "Lab Only", "provider": "Lab",
         "reason": "Routine labs", "assessment": "Hemoglobin 11.5 — low. Start iron supplementation."},
    ],
    "P003": [
        {"date": "2026-02-25", "type": "Office Visit", "provider": "Dr. Lisa Wong",
         "reason": "COPD exacerbation follow-up", "assessment": "SpO2 94%. Continue tiotropium + prednisone taper."},
        {"date": "2026-02-10", "type": "ER Visit", "provider": "Dr. Amy Roberts",
         "reason": "Shortness of breath", "assessment": "COPD exacerbation. Started oral steroids + antibiotics."},
    ],
}

VITALS_HISTORY = {
    "P001": [
        {"date": "2026-02-20", "bp_sys": 138, "bp_dia": 85, "hr": 78, "temp": 98.4, "spo2": 97, "weight_kg": 88.5},
        {"date": "2026-02-15", "bp_sys": 142, "bp_dia": 88, "hr": 82, "temp": 98.6, "spo2": 96, "weight_kg": 89.0},
        {"date": "2026-02-01", "bp_sys": 145, "bp_dia": 90, "hr": 80, "temp": 98.5, "spo2": 97, "weight_kg": 89.2},
        {"date": "2026-01-15", "bp_sys": 148, "bp_dia": 92, "hr": 84, "temp": 98.6, "spo2": 97, "weight_kg": 89.5},
    ],
    "P002": [
        {"date": "2026-02-22", "bp_sys": 125, "bp_dia": 78, "hr": 72, "temp": 98.2, "spo2": 99, "weight_kg": 65.0},
        {"date": "2026-01-05", "bp_sys": 122, "bp_dia": 76, "hr": 70, "temp": 98.4, "spo2": 98, "weight_kg": 65.2},
    ],
    "P003": [
        {"date": "2026-02-25", "bp_sys": 155, "bp_dia": 92, "hr": 68, "temp": 97.9, "spo2": 94, "weight_kg": 95.3},
        {"date": "2026-02-18", "bp_sys": 150, "bp_dia": 88, "hr": 70, "temp": 98.0, "spo2": 95, "weight_kg": 95.8},
        {"date": "2026-02-10", "bp_sys": 160, "bp_dia": 95, "hr": 92, "temp": 100.2, "spo2": 91, "weight_kg": 96.0},
    ],
}


# ============================================================================
# Lab Database
# ============================================================================

LAB_RESULTS = {
    "P001": [
        {"date": "2026-02-20", "test": "HbA1c", "value": 7.8, "unit": "%",
         "range": "4.0-5.6", "flag": "HIGH", "critical": False},
        {"date": "2026-02-20", "test": "Glucose", "value": 118, "unit": "mg/dL",
         "range": "70-100", "flag": "HIGH", "critical": False},
        {"date": "2026-02-20", "test": "Creatinine", "value": 1.1, "unit": "mg/dL",
         "range": "0.7-1.3", "flag": "NORMAL", "critical": False},
        {"date": "2026-02-20", "test": "Potassium", "value": 4.2, "unit": "mEq/L",
         "range": "3.5-5.0", "flag": "NORMAL", "critical": False},
        {"date": "2026-02-20", "test": "Hemoglobin", "value": 14.2, "unit": "g/dL",
         "range": "13.5-17.5", "flag": "NORMAL", "critical": False},
        {"date": "2026-02-20", "test": "LDL", "value": 142, "unit": "mg/dL",
         "range": "<100", "flag": "HIGH", "critical": False},
    ],
    "P002": [
        {"date": "2026-02-22", "test": "TSH", "value": 6.2, "unit": "mIU/L",
         "range": "0.4-4.0", "flag": "HIGH", "critical": False},
        {"date": "2026-02-22", "test": "Hemoglobin", "value": 11.5, "unit": "g/dL",
         "range": "12.0-16.0", "flag": "LOW", "critical": False},
        {"date": "2026-02-22", "test": "Ferritin", "value": 8, "unit": "ng/mL",
         "range": "12-150", "flag": "LOW", "critical": False},
        {"date": "2026-02-22", "test": "Iron", "value": 35, "unit": "mcg/dL",
         "range": "60-170", "flag": "LOW", "critical": False},
    ],
    "P003": [
        {"date": "2026-02-25", "test": "Creatinine", "value": 2.1, "unit": "mg/dL",
         "range": "0.7-1.3", "flag": "HIGH", "critical": False},
        {"date": "2026-02-25", "test": "BNP", "value": 450, "unit": "pg/mL",
         "range": "<100", "flag": "HIGH", "critical": True},
        {"date": "2026-02-25", "test": "Potassium", "value": 5.4, "unit": "mEq/L",
         "range": "3.5-5.0", "flag": "HIGH", "critical": False},
        {"date": "2026-02-25", "test": "INR", "value": 2.5, "unit": "",
         "range": "2.0-3.0", "flag": "NORMAL", "critical": False},
    ],
}


# ============================================================================
# Pharmacy Database
# ============================================================================

PATIENT_MEDICATIONS = {
    "P001": [
        {"name": "metformin", "dose": "1000mg", "frequency": "BID", "status": "active",
         "refills_remaining": 3, "last_fill": "2026-02-01", "prior_auth": False},
        {"name": "amlodipine", "dose": "10mg", "frequency": "daily", "status": "active",
         "refills_remaining": 5, "last_fill": "2026-01-15", "prior_auth": False},
        {"name": "atorvastatin", "dose": "40mg", "frequency": "daily", "status": "active",
         "refills_remaining": 2, "last_fill": "2026-01-20", "prior_auth": False},
    ],
    "P002": [
        {"name": "levothyroxine", "dose": "75mcg", "frequency": "daily", "status": "active",
         "refills_remaining": 6, "last_fill": "2026-02-05", "prior_auth": False},
        {"name": "ferrous sulfate", "dose": "325mg", "frequency": "daily", "status": "active",
         "refills_remaining": 4, "last_fill": "2026-01-10", "prior_auth": False},
    ],
    "P003": [
        {"name": "tiotropium", "dose": "18mcg", "frequency": "daily inhaler", "status": "active",
         "refills_remaining": 2, "last_fill": "2026-02-10", "prior_auth": False},
        {"name": "losartan", "dose": "50mg", "frequency": "daily", "status": "active",
         "refills_remaining": 4, "last_fill": "2026-02-01", "prior_auth": False},
        {"name": "apixaban", "dose": "5mg", "frequency": "BID", "status": "active",
         "refills_remaining": 1, "last_fill": "2026-02-15", "prior_auth": True},
        {"name": "prednisone", "dose": "20mg", "frequency": "daily (taper)", "status": "active",
         "refills_remaining": 0, "last_fill": "2026-02-10", "prior_auth": False},
    ],
}

DRUG_INTERACTIONS = {
    ("metformin", "amlodipine"): {"severity": "none", "note": "Safe combination"},
    ("metformin", "atorvastatin"): {"severity": "none", "note": "Safe combination"},
    ("amlodipine", "atorvastatin"): {"severity": "moderate",
                                      "note": "Limit atorvastatin to 20mg with amlodipine (CYP3A4)"},
    ("losartan", "apixaban"): {"severity": "minor",
                                "note": "Monitor for additive hypotension; generally safe"},
    ("prednisone", "losartan"): {"severity": "moderate",
                                  "note": "Steroids may reduce ARB efficacy; monitor BP and K+"},
}

FORMULARY = {
    "metformin": {"tier": 1, "status": "preferred", "prior_auth": False},
    "amlodipine": {"tier": 1, "status": "preferred", "prior_auth": False},
    "atorvastatin": {"tier": 1, "status": "preferred", "prior_auth": False},
    "levothyroxine": {"tier": 1, "status": "preferred", "prior_auth": False},
    "tiotropium": {"tier": 2, "status": "preferred", "prior_auth": False},
    "losartan": {"tier": 1, "status": "preferred", "prior_auth": False},
    "apixaban": {"tier": 3, "status": "non-preferred", "prior_auth": True},
    "semaglutide": {"tier": 3, "status": "non-preferred", "prior_auth": True},
}


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
# DEMO 1: EHR Server
# ============================================================================

def demo_ehr_server():
    """DEMO 1: MCP server with patient lookup, demographics, encounters, vitals."""
    print_banner("DEMO 1: EHR Server")

    print("""
  The EHR Server provides clinical data for patient management:
  - Patient demographics and problem list
  - Encounter history
  - Vital signs with trending
    """)

    # Define MCP tools (if available)
    if MCP_AVAILABLE:
        ehr = FastMCP("EHR Server")

        @ehr.tool()
        def lookup_patient(patient_id: str) -> str:
            """Look up patient demographics, problems, and allergies."""
            if patient_id in PATIENTS:
                return json.dumps(PATIENTS[patient_id], indent=2)
            return json.dumps({"error": f"Patient {patient_id} not found"})

        @ehr.tool()
        def get_encounters(patient_id: str, limit: int = 5) -> str:
            """Get recent encounters for a patient."""
            if patient_id in ENCOUNTERS:
                return json.dumps(ENCOUNTERS[patient_id][:limit], indent=2)
            return json.dumps({"error": f"No encounters for {patient_id}"})

        @ehr.tool()
        def get_vitals_trend(patient_id: str) -> str:
            """Get vitals trend over time for a patient."""
            if patient_id in VITALS_HISTORY:
                return json.dumps(VITALS_HISTORY[patient_id], indent=2)
            return json.dumps({"error": f"No vitals for {patient_id}"})

        print("  ✓ EHR MCP Server defined with 3 tools")

    # Demonstrate each tool
    for pid in ["P001", "P002", "P003"]:
        pt = PATIENTS[pid]
        print(f"\n  --- Patient: {pt['name']} ({pid}) ---")
        print(f"    Age: {pt['age']}, Sex: {pt['sex']}, PCP: {pt['pcp']}")
        print(f"    Problems: {', '.join(pt['problems'])}")
        print(f"    Allergies: {', '.join(pt['allergies'])}")

        # Latest encounter
        if pid in ENCOUNTERS and ENCOUNTERS[pid]:
            enc = ENCOUNTERS[pid][0]
            print(f"    Latest visit: {enc['date']} — {enc['reason']}")

        # Vitals trend
        if pid in VITALS_HISTORY and len(VITALS_HISTORY[pid]) >= 2:
            vitals = VITALS_HISTORY[pid]
            latest = vitals[0]
            oldest = vitals[-1]
            bp_change = latest["bp_sys"] - oldest["bp_sys"]
            direction = "↓" if bp_change < 0 else "↑" if bp_change > 0 else "→"
            print(f"    BP trend: {oldest['bp_sys']}/{oldest['bp_dia']} → "
                  f"{latest['bp_sys']}/{latest['bp_dia']} ({direction}{abs(bp_change)})")


# ============================================================================
# DEMO 2: Clinical Lab Server
# ============================================================================

def demo_lab_server():
    """DEMO 2: MCP server with lab results, interpretations, critical alerts."""
    print_banner("DEMO 2: Clinical Lab Server")

    print("""
  The Lab Server provides laboratory data and clinical decision support:
  - Recent lab results with reference ranges
  - Abnormal value flagging
  - Critical value alerts requiring immediate attention
    """)

    if MCP_AVAILABLE:
        lab = FastMCP("Lab Server")

        @lab.tool()
        def get_lab_results(patient_id: str, test_name: str = "") -> str:
            """Get lab results for a patient, optionally filtered by test name."""
            if patient_id not in LAB_RESULTS:
                return json.dumps({"error": f"No labs for {patient_id}"})
            results = LAB_RESULTS[patient_id]
            if test_name:
                results = [r for r in results if r["test"].lower() == test_name.lower()]
            return json.dumps(results, indent=2)

        @lab.tool()
        def get_critical_values(patient_id: str) -> str:
            """Get critical lab values that need immediate attention."""
            if patient_id not in LAB_RESULTS:
                return json.dumps({"error": f"No labs for {patient_id}"})
            criticals = [r for r in LAB_RESULTS[patient_id] if r["critical"]]
            return json.dumps({"patient_id": patient_id, "critical_values": criticals}, indent=2)

        print("  ✓ Lab MCP Server defined with 2 tools")

    # Show results for each patient
    for pid in ["P001", "P002", "P003"]:
        pt = PATIENTS[pid]
        labs = LAB_RESULTS.get(pid, [])
        print(f"\n  --- Lab Results: {pt['name']} ({pid}) ---")

        abnormals = [r for r in labs if r["flag"] != "NORMAL"]
        criticals = [r for r in labs if r["critical"]]

        for lab in labs:
            flag_marker = {"NORMAL": "✓", "HIGH": "↑", "LOW": "↓"}.get(lab["flag"], "?")
            crit = " ⚠ CRITICAL" if lab["critical"] else ""
            print(f"    {flag_marker} {lab['test']:<15} {lab['value']:>8} {lab['unit']:<8} "
                  f"(ref: {lab['range']}){crit}")

        if criticals:
            print(f"    *** {len(criticals)} CRITICAL VALUE(S) — immediate notification required ***")
        print(f"    Summary: {len(labs)} tests, {len(abnormals)} abnormal, {len(criticals)} critical")


# ============================================================================
# DEMO 3: Pharmacy Server
# ============================================================================

def demo_pharmacy_server():
    """DEMO 3: MCP server with formulary, interactions, prior auth, refills."""
    print_banner("DEMO 3: Pharmacy Server")

    print("""
  The Pharmacy Server manages medication data:
  - Current medication list per patient
  - Formulary status and tier
  - Drug-drug interaction checking
  - Prior authorization status
  - Refill management
    """)

    if MCP_AVAILABLE:
        pharm = FastMCP("Pharmacy Server")

        @pharm.tool()
        def get_medications(patient_id: str) -> str:
            """Get current medication list for a patient."""
            if patient_id not in PATIENT_MEDICATIONS:
                return json.dumps({"error": f"No medications for {patient_id}"})
            return json.dumps(PATIENT_MEDICATIONS[patient_id], indent=2)

        @pharm.tool()
        def check_interaction(drug_a: str, drug_b: str) -> str:
            """Check for drug-drug interaction between two medications."""
            key1 = (drug_a.lower(), drug_b.lower())
            key2 = (drug_b.lower(), drug_a.lower())
            if key1 in DRUG_INTERACTIONS:
                return json.dumps(DRUG_INTERACTIONS[key1])
            elif key2 in DRUG_INTERACTIONS:
                return json.dumps(DRUG_INTERACTIONS[key2])
            return json.dumps({"severity": "unknown", "note": "Interaction data not available"})

        @pharm.tool()
        def check_formulary(medication_name: str) -> str:
            """Check formulary status for a medication."""
            name = medication_name.lower()
            if name in FORMULARY:
                return json.dumps({"medication": name, **FORMULARY[name]})
            return json.dumps({"error": f"'{name}' not in formulary database"})

        print("  ✓ Pharmacy MCP Server defined with 3 tools")

    # Show medication data per patient
    for pid in ["P001", "P002", "P003"]:
        pt = PATIENTS[pid]
        meds = PATIENT_MEDICATIONS.get(pid, [])
        print(f"\n  --- Medications: {pt['name']} ({pid}) ---")

        for med in meds:
            pa = " [PA REQUIRED]" if med["prior_auth"] else ""
            refill_warn = " ⚠ LOW REFILLS" if med["refills_remaining"] <= 1 else ""
            print(f"    • {med['name']} {med['dose']} {med['frequency']}{pa}{refill_warn}")
            print(f"      Refills: {med['refills_remaining']}, Last fill: {med['last_fill']}")

        # Check interactions within the patient's med list
        if len(meds) >= 2:
            print(f"\n    Interaction check:")
            checked = set()
            for i, m1 in enumerate(meds):
                for m2 in meds[i+1:]:
                    pair = tuple(sorted([m1["name"], m2["name"]]))
                    if pair in checked:
                        continue
                    checked.add(pair)
                    key1 = (m1["name"], m2["name"])
                    key2 = (m2["name"], m1["name"])
                    if key1 in DRUG_INTERACTIONS:
                        info = DRUG_INTERACTIONS[key1]
                    elif key2 in DRUG_INTERACTIONS:
                        info = DRUG_INTERACTIONS[key2]
                    else:
                        continue
                    if info["severity"] != "none":
                        sev = info["severity"].upper()
                        print(f"      [{sev}] {m1['name']} + {m2['name']}: {info['note']}")


# ============================================================================
# DEMO 4: Unified Clinical Agent
# ============================================================================

def demo_unified_agent():
    """DEMO 4: Single agent using all three MCP servers for clinical queries."""
    print_banner("DEMO 4: Unified Clinical Agent")

    print("""
  The Unified Clinical Agent connects to all three MCP servers and
  handles complex clinical queries that span multiple data sources.

  Agent tools: lookup_patient, get_encounters, get_vitals_trend,
               get_lab_results, get_critical_values,
               get_medications, check_interaction, check_formulary
    """)

    def unified_agent_query(patient_id: str, query: str):
        """Simulate a unified agent responding to a clinical query."""
        pt = PATIENTS.get(patient_id, {})
        print(f"\n  Agent Query: \"{query}\"")
        print(f"  Patient: {pt.get('name', patient_id)}")
        print()

        # Determine which servers to call based on query keywords
        results = {}

        # Always get patient demographics
        results["demographics"] = pt

        # Check what data is needed
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["lab", "result", "test", "hba1c", "glucose", "critical"]):
            results["labs"] = LAB_RESULTS.get(patient_id, [])
            criticals = [r for r in results["labs"] if r["critical"]]
            if criticals:
                results["critical_alerts"] = criticals

        if any(kw in query_lower for kw in ["vital", "bp", "blood pressure", "heart rate", "spo2"]):
            results["vitals"] = VITALS_HISTORY.get(patient_id, [])

        if any(kw in query_lower for kw in ["med", "drug", "prescription", "refill", "formulary"]):
            results["medications"] = PATIENT_MEDICATIONS.get(patient_id, [])

        if any(kw in query_lower for kw in ["encounter", "visit", "history"]):
            results["encounters"] = ENCOUNTERS.get(patient_id, [])

        if "review" in query_lower or "comprehensive" in query_lower or "summary" in query_lower:
            results["labs"] = LAB_RESULTS.get(patient_id, [])
            results["vitals"] = VITALS_HISTORY.get(patient_id, [])
            results["medications"] = PATIENT_MEDICATIONS.get(patient_id, [])
            results["encounters"] = ENCOUNTERS.get(patient_id, [])

        # Print what the agent gathered
        servers_used = []
        if "encounters" in results or "vitals" in results:
            servers_used.append("EHR")
        if "labs" in results:
            servers_used.append("Lab")
        if "medications" in results:
            servers_used.append("Pharmacy")
        print(f"  Servers contacted: {', '.join(servers_used)}")

        # Generate synthesized response
        print(f"\n  --- Agent Response ---")
        print(f"  Patient: {pt.get('name', 'Unknown')} ({patient_id}), "
              f"Age: {pt.get('age', '?')}, Sex: {pt.get('sex', '?')}")

        if "labs" in results:
            abnormals = [r for r in results["labs"] if r["flag"] != "NORMAL"]
            if abnormals:
                print(f"\n  Abnormal Labs:")
                for lab in abnormals:
                    print(f"    • {lab['test']}: {lab['value']} {lab['unit']} ({lab['flag']})")

        if "critical_alerts" in results:
            print(f"\n  ⚠ CRITICAL ALERTS:")
            for c in results["critical_alerts"]:
                print(f"    • {c['test']}: {c['value']} {c['unit']} — CRITICAL")

        if "vitals" in results and results["vitals"]:
            latest = results["vitals"][0]
            print(f"\n  Latest Vitals ({latest['date']}):")
            print(f"    BP: {latest['bp_sys']}/{latest['bp_dia']}, HR: {latest['hr']}, "
                  f"SpO2: {latest['spo2']}%, Temp: {latest['temp']}°F")

        if "medications" in results and results["medications"]:
            print(f"\n  Current Medications:")
            for med in results["medications"]:
                print(f"    • {med['name']} {med['dose']} {med['frequency']}")

        if "encounters" in results and results["encounters"]:
            enc = results["encounters"][0]
            print(f"\n  Last Encounter: {enc['date']} — {enc['type']}")
            print(f"    {enc['assessment']}")

    # Run several clinical queries
    queries = [
        ("P001", "Give me a comprehensive review of this patient"),
        ("P002", "What are the latest lab results?"),
        ("P003", "Check for any critical lab values"),
        ("P001", "What medications is this patient on?"),
        ("P003", "Show me vitals trend and current medications"),
    ]

    for pid, query in queries:
        unified_agent_query(pid, query)
        print()


# ============================================================================
# Main Menu
# ============================================================================

def main():
    """Run Healthcare MCP Capstone demos interactively."""
    print("=" * 70)
    print("  Level 5 - Project 05: Healthcare MCP Capstone")
    print("  Complete healthcare MCP ecosystem")
    print("=" * 70)

    if MCP_AVAILABLE:
        print("  ✓ MCP SDK available")
    else:
        print("  ⚠ MCP SDK not installed (demos use standalone functions)")

    demos = {
        "1": ("EHR Server", demo_ehr_server),
        "2": ("Clinical Lab Server", demo_lab_server),
        "3": ("Pharmacy Server", demo_pharmacy_server),
        "4": ("Unified Clinical Agent", demo_unified_agent),
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
