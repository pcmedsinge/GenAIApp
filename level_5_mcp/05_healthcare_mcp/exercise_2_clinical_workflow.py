"""
Exercise 2: Clinical Workflow
================================

Skills practiced:
- Orchestrating a multi-step clinical workflow across MCP servers
- Step-by-step patient encounter: triage → vitals → labs → meds → assessment
- Data routing — each step pulls from the appropriate server
- Synthesizing cross-server data into a clinical summary

Healthcare context:
When a patient presents to a clinic, the encounter follows a structured
workflow: registration, vital signs, lab review, medication reconciliation,
and provider assessment. Each step may query a different backend system
(EHR, lab LIS, pharmacy). This exercise models that workflow, showing
how an MCP-based architecture routes each step to the correct server.

Usage:
    python exercise_2_clinical_workflow.py
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
    print("      Exercise will use standalone functions.\n")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ============================================================================
# Shared Clinical Data (mirrors main.py data)
# ============================================================================

PATIENTS = {
    "P001": {
        "name": "John Smith", "dob": "1963-08-15", "age": 62, "sex": "M",
        "mrn": "MRN-2024-001",
        "problems": ["Type 2 Diabetes", "Hypertension", "Hyperlipidemia"],
        "allergies": ["Penicillin (rash)", "Sulfa drugs (hives)"],
        "pcp": "Dr. Sarah Chen",
        "insurance": "BlueCross PPO",
    },
    "P002": {
        "name": "Maria Garcia", "dob": "1980-03-22", "age": 45, "sex": "F",
        "mrn": "MRN-2024-002",
        "problems": ["Hypothyroidism", "Iron Deficiency Anemia"],
        "allergies": ["None known"],
        "pcp": "Dr. James Park",
        "insurance": "Aetna HMO",
    },
    "P003": {
        "name": "Robert Wilson", "dob": "1954-11-07", "age": 71, "sex": "M",
        "mrn": "MRN-2024-003",
        "problems": ["COPD", "Hypertension", "CKD Stage 3b", "Atrial Fibrillation"],
        "allergies": ["ACE Inhibitors (angioedema)", "Iodine contrast"],
        "pcp": "Dr. Lisa Wong",
        "insurance": "Medicare Part B",
    },
}

VITALS = {
    "P001": {"date": "2026-02-28", "bp_sys": 136, "bp_dia": 84, "hr": 76,
             "temp": 98.4, "spo2": 97, "weight_kg": 88.2, "height_cm": 178, "rr": 16, "pain": 0},
    "P002": {"date": "2026-02-28", "bp_sys": 118, "bp_dia": 74, "hr": 70,
             "temp": 98.1, "spo2": 99, "weight_kg": 64.8, "height_cm": 165, "rr": 14, "pain": 0},
    "P003": {"date": "2026-02-28", "bp_sys": 152, "bp_dia": 90, "hr": 66,
             "temp": 97.8, "spo2": 93, "weight_kg": 95.0, "height_cm": 182, "rr": 20, "pain": 3},
}

LAB_RESULTS = {
    "P001": [
        {"test": "HbA1c", "value": 7.8, "unit": "%", "range": "4.0-5.6", "flag": "HIGH", "critical": False},
        {"test": "Glucose", "value": 118, "unit": "mg/dL", "range": "70-100", "flag": "HIGH", "critical": False},
        {"test": "Creatinine", "value": 1.1, "unit": "mg/dL", "range": "0.7-1.3", "flag": "NORMAL", "critical": False},
        {"test": "LDL", "value": 142, "unit": "mg/dL", "range": "<100", "flag": "HIGH", "critical": False},
        {"test": "Hemoglobin", "value": 14.2, "unit": "g/dL", "range": "13.5-17.5", "flag": "NORMAL", "critical": False},
    ],
    "P002": [
        {"test": "TSH", "value": 6.2, "unit": "mIU/L", "range": "0.4-4.0", "flag": "HIGH", "critical": False},
        {"test": "Hemoglobin", "value": 11.5, "unit": "g/dL", "range": "12.0-16.0", "flag": "LOW", "critical": False},
        {"test": "Ferritin", "value": 8, "unit": "ng/mL", "range": "12-150", "flag": "LOW", "critical": False},
        {"test": "Iron", "value": 35, "unit": "mcg/dL", "range": "60-170", "flag": "LOW", "critical": False},
    ],
    "P003": [
        {"test": "Creatinine", "value": 2.1, "unit": "mg/dL", "range": "0.7-1.3", "flag": "HIGH", "critical": False},
        {"test": "BNP", "value": 450, "unit": "pg/mL", "range": "<100", "flag": "HIGH", "critical": True},
        {"test": "Potassium", "value": 5.4, "unit": "mEq/L", "range": "3.5-5.0", "flag": "HIGH", "critical": False},
        {"test": "INR", "value": 2.5, "unit": "", "range": "2.0-3.0", "flag": "NORMAL", "critical": False},
        {"test": "eGFR", "value": 32, "unit": "mL/min", "range": ">60", "flag": "LOW", "critical": False},
    ],
}

MEDICATIONS = {
    "P001": [
        {"name": "metformin", "dose": "1000mg", "freq": "BID", "status": "active"},
        {"name": "amlodipine", "dose": "10mg", "freq": "daily", "status": "active"},
        {"name": "atorvastatin", "dose": "40mg", "freq": "daily", "status": "active"},
    ],
    "P002": [
        {"name": "levothyroxine", "dose": "75mcg", "freq": "daily", "status": "active"},
        {"name": "ferrous sulfate", "dose": "325mg", "freq": "daily", "status": "active"},
    ],
    "P003": [
        {"name": "tiotropium", "dose": "18mcg", "freq": "daily inhaler", "status": "active"},
        {"name": "losartan", "dose": "50mg", "freq": "daily", "status": "active"},
        {"name": "apixaban", "dose": "5mg", "freq": "BID", "status": "active"},
        {"name": "prednisone", "dose": "20mg", "freq": "daily (taper)", "status": "active"},
    ],
}

DRUG_INTERACTIONS = {
    ("metformin", "amlodipine"): {"severity": "none", "note": "Safe combination"},
    ("metformin", "atorvastatin"): {"severity": "none", "note": "Safe combination"},
    ("amlodipine", "atorvastatin"): {"severity": "moderate",
                                      "note": "Limit atorvastatin to 20mg with amlodipine (CYP3A4)"},
    ("losartan", "apixaban"): {"severity": "minor",
                                "note": "Monitor for additive hypotension"},
    ("prednisone", "losartan"): {"severity": "moderate",
                                  "note": "Steroids may reduce ARB efficacy; monitor BP and K+"},
}


# ============================================================================
# Helper Functions
# ============================================================================

def print_banner(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_step(step_num: int, title: str, server: str):
    """Print a workflow step header."""
    print(f"\n  ┌─ Step {step_num}: {title}")
    print(f"  │  Server: {server}")
    print(f"  │")


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Calculate BMI from weight (kg) and height (cm)."""
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)


# ============================================================================
# Workflow Steps (each maps to a different MCP server)
# ============================================================================

def step_registration(patient_id: str) -> dict:
    """Step 1: Patient registration — data from EHR Server."""
    if patient_id not in PATIENTS:
        return {"error": f"Patient {patient_id} not found"}

    pt = PATIENTS[patient_id]
    return {
        "step": "registration",
        "server": "EHR",
        "patient_id": patient_id,
        "name": pt["name"],
        "mrn": pt["mrn"],
        "dob": pt["dob"],
        "age": pt["age"],
        "sex": pt["sex"],
        "insurance": pt.get("insurance", "Unknown"),
        "pcp": pt["pcp"],
        "problems": pt["problems"],
        "allergies": pt["allergies"],
    }


def step_vitals(patient_id: str) -> dict:
    """Step 2: Vital signs — data from EHR Server (vitals subsystem)."""
    if patient_id not in VITALS:
        return {"error": f"No vitals for {patient_id}"}

    v = VITALS[patient_id]
    bmi = calculate_bmi(v["weight_kg"], v["height_cm"])

    # Flag abnormal vitals
    alerts = []
    if v["bp_sys"] >= 140 or v["bp_dia"] >= 90:
        alerts.append(f"Hypertension: {v['bp_sys']}/{v['bp_dia']}")
    if v["spo2"] < 95:
        alerts.append(f"Hypoxia: SpO2 {v['spo2']}%")
    if v["temp"] >= 100.4:
        alerts.append(f"Fever: {v['temp']}°F")
    if v["hr"] > 100:
        alerts.append(f"Tachycardia: HR {v['hr']}")
    if v["hr"] < 60:
        alerts.append(f"Bradycardia: HR {v['hr']}")
    if v["rr"] > 20:
        alerts.append(f"Tachypnea: RR {v['rr']}")
    if bmi >= 30:
        alerts.append(f"Obesity: BMI {bmi}")

    return {
        "step": "vitals",
        "server": "EHR (vitals)",
        "patient_id": patient_id,
        "vitals": {
            "bp": f"{v['bp_sys']}/{v['bp_dia']}",
            "hr": v["hr"],
            "temp": v["temp"],
            "spo2": v["spo2"],
            "rr": v["rr"],
            "weight_kg": v["weight_kg"],
            "height_cm": v["height_cm"],
            "bmi": bmi,
            "pain": v["pain"],
        },
        "alerts": alerts,
        "alert_count": len(alerts),
    }


def step_lab_review(patient_id: str) -> dict:
    """Step 3: Lab review — data from Lab Server."""
    if patient_id not in LAB_RESULTS:
        return {"error": f"No labs for {patient_id}"}

    labs = LAB_RESULTS[patient_id]
    abnormals = [r for r in labs if r["flag"] != "NORMAL"]
    criticals = [r for r in labs if r.get("critical")]

    interpretations = []
    for lab in abnormals:
        interp = {
            "test": lab["test"],
            "value": f"{lab['value']} {lab['unit']}",
            "flag": lab["flag"],
            "range": lab["range"],
            "critical": lab.get("critical", False),
        }
        # Add clinical context
        if lab["test"] == "HbA1c" and lab["value"] > 7.0:
            interp["interpretation"] = "Suboptimal glycemic control — consider therapy adjustment"
        elif lab["test"] == "LDL" and lab["value"] > 100:
            interp["interpretation"] = "Above target for cardiovascular risk reduction"
        elif lab["test"] == "BNP" and lab["value"] > 100:
            interp["interpretation"] = "Elevated — evaluate for heart failure or volume overload"
        elif lab["test"] == "Creatinine" and lab["value"] > 1.3:
            interp["interpretation"] = "Elevated — assess renal function trend"
        elif lab["test"] == "TSH" and lab["value"] > 4.0:
            interp["interpretation"] = "Elevated — hypothyroidism undertreated"
        elif lab["test"] == "Potassium" and lab["value"] > 5.0:
            interp["interpretation"] = "Elevated — monitor in context of renal function"
        elif lab["test"] == "eGFR" and lab["value"] < 60:
            interp["interpretation"] = "Reduced — consistent with CKD staging"
        else:
            interp["interpretation"] = f"{lab['flag']} — review in clinical context"
        interpretations.append(interp)

    return {
        "step": "lab_review",
        "server": "Lab",
        "patient_id": patient_id,
        "total_tests": len(labs),
        "abnormal_count": len(abnormals),
        "critical_count": len(criticals),
        "abnormal_results": interpretations,
        "critical_alert": len(criticals) > 0,
    }


def step_medication_reconciliation(patient_id: str) -> dict:
    """Step 4: Medication reconciliation — data from Pharmacy Server."""
    if patient_id not in MEDICATIONS:
        return {"error": f"No medications for {patient_id}"}

    meds = MEDICATIONS[patient_id]

    # Check interactions
    interactions = []
    for i, m1 in enumerate(meds):
        for m2 in meds[i+1:]:
            key1 = (m1["name"], m2["name"])
            key2 = (m2["name"], m1["name"])
            info = DRUG_INTERACTIONS.get(key1) or DRUG_INTERACTIONS.get(key2)
            if info and info["severity"] != "none":
                interactions.append({
                    "drug_a": m1["name"],
                    "drug_b": m2["name"],
                    "severity": info["severity"],
                    "note": info["note"],
                })

    # Check allergies
    allergy_warnings = []
    allergies_lower = [a.lower() for a in PATIENTS.get(patient_id, {}).get("allergies", [])]
    for med in meds:
        for allergy in allergies_lower:
            if med["name"].lower() in allergy:
                allergy_warnings.append({
                    "medication": med["name"],
                    "allergy": allergy,
                    "action": "HOLD — allergy conflict",
                })

    return {
        "step": "medication_reconciliation",
        "server": "Pharmacy",
        "patient_id": patient_id,
        "medication_count": len(meds),
        "medications": [
            {"name": m["name"], "dose": m["dose"], "freq": m["freq"], "status": m["status"]}
            for m in meds
        ],
        "interaction_count": len(interactions),
        "interactions": interactions,
        "allergy_warnings": allergy_warnings,
    }


def step_clinical_assessment(patient_id: str, registration: dict, vitals: dict,
                              labs: dict, meds: dict) -> dict:
    """Step 5: Clinical assessment — synthesize all data."""
    # Gather all concerns
    concerns = []

    # From vitals
    for alert in vitals.get("alerts", []):
        concerns.append({"source": "Vitals", "concern": alert, "priority": "medium"})

    # From labs
    for lab in labs.get("abnormal_results", []):
        priority = "high" if lab.get("critical") else "medium"
        concerns.append({
            "source": "Labs",
            "concern": f"{lab['test']}: {lab['value']} ({lab['flag']})",
            "interpretation": lab.get("interpretation", ""),
            "priority": priority,
        })

    # From medications
    for interaction in meds.get("interactions", []):
        concerns.append({
            "source": "Pharmacy",
            "concern": f"Interaction: {interaction['drug_a']} + {interaction['drug_b']}",
            "note": interaction["note"],
            "priority": "medium" if interaction["severity"] == "moderate" else "low",
        })

    for warning in meds.get("allergy_warnings", []):
        concerns.append({
            "source": "Pharmacy",
            "concern": f"ALLERGY: {warning['medication']} — {warning['allergy']}",
            "priority": "critical",
        })

    # Sort by priority
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    concerns.sort(key=lambda x: priority_order.get(x["priority"], 99))

    return {
        "step": "clinical_assessment",
        "server": "Agent (synthesized)",
        "patient_id": patient_id,
        "patient_name": registration.get("name", "Unknown"),
        "total_concerns": len(concerns),
        "concerns": concerns,
        "servers_used": ["EHR", "Lab", "Pharmacy"],
        "assessment_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ============================================================================
# MCP Server Definitions
# ============================================================================

def define_workflow_servers():
    """Define MCP servers for each workflow step."""
    if not MCP_AVAILABLE:
        print("  ⚠ MCP SDK not installed — skipping MCP server definitions")
        return

    ehr = FastMCP("EHR Workflow Server")

    @ehr.tool()
    def register_patient(patient_id: str) -> str:
        """Register a patient and retrieve demographics."""
        return json.dumps(step_registration(patient_id), indent=2)

    @ehr.tool()
    def record_vitals(patient_id: str) -> str:
        """Record and retrieve vital signs for a patient."""
        return json.dumps(step_vitals(patient_id), indent=2)

    lab_server = FastMCP("Lab Workflow Server")

    @lab_server.tool()
    def review_labs(patient_id: str) -> str:
        """Review lab results with interpretations."""
        return json.dumps(step_lab_review(patient_id), indent=2)

    pharmacy_server = FastMCP("Pharmacy Workflow Server")

    @pharmacy_server.tool()
    def reconcile_medications(patient_id: str) -> str:
        """Reconcile medications and check interactions."""
        return json.dumps(step_medication_reconciliation(patient_id), indent=2)

    print("  ✓ Workflow MCP Servers defined:")
    print("    - EHR Workflow Server (register_patient, record_vitals)")
    print("    - Lab Workflow Server (review_labs)")
    print("    - Pharmacy Workflow Server (reconcile_medications)")


# ============================================================================
# Section 1: Single Patient Workflow
# ============================================================================

def section_single_patient_workflow():
    """Run a complete clinical workflow for one patient."""
    print_banner("Section 1: Single Patient Workflow (P001 — John Smith)")

    print("""
  Running a complete encounter workflow for John Smith:
  Patient presents → Registration → Vitals → Labs → Meds → Assessment
  Each step queries the appropriate MCP server.
    """)

    pid = "P001"

    # Step 1: Registration (EHR Server)
    print_step(1, "Patient Registration", "EHR Server")
    reg = step_registration(pid)
    print(f"  │  Name: {reg['name']} | MRN: {reg['mrn']}")
    print(f"  │  Age: {reg['age']}, Sex: {reg['sex']}")
    print(f"  │  PCP: {reg['pcp']}")
    print(f"  │  Insurance: {reg['insurance']}")
    print(f"  │  Problems: {', '.join(reg['problems'])}")
    print(f"  │  Allergies: {', '.join(reg['allergies'])}")
    print(f"  └─ Registration complete ✓")

    # Step 2: Vitals (EHR Server — vitals subsystem)
    print_step(2, "Vital Signs", "EHR Server (Vitals)")
    vit = step_vitals(pid)
    v = vit["vitals"]
    print(f"  │  BP: {v['bp']} | HR: {v['hr']} | Temp: {v['temp']}°F")
    print(f"  │  SpO2: {v['spo2']}% | RR: {v['rr']} | Pain: {v['pain']}/10")
    print(f"  │  Weight: {v['weight_kg']}kg | BMI: {v['bmi']}")
    if vit["alerts"]:
        for alert in vit["alerts"]:
            print(f"  │  ⚠ {alert}")
    else:
        print(f"  │  No vital sign alerts")
    print(f"  └─ Vitals recorded ✓")

    # Step 3: Lab Review (Lab Server)
    print_step(3, "Lab Review", "Lab Server")
    lab = step_lab_review(pid)
    print(f"  │  Total tests: {lab['total_tests']} | Abnormal: {lab['abnormal_count']} | "
          f"Critical: {lab['critical_count']}")
    for result in lab["abnormal_results"]:
        flag = "⚠ CRITICAL" if result["critical"] else f"({result['flag']})"
        print(f"  │  • {result['test']}: {result['value']} {flag}")
        print(f"  │    → {result['interpretation']}")
    print(f"  └─ Lab review complete ✓")

    # Step 4: Medication Reconciliation (Pharmacy Server)
    print_step(4, "Medication Reconciliation", "Pharmacy Server")
    med = step_medication_reconciliation(pid)
    print(f"  │  Active medications: {med['medication_count']}")
    for m in med["medications"]:
        print(f"  │  • {m['name']} {m['dose']} {m['freq']}")
    if med["interactions"]:
        print(f"  │  Interactions found: {med['interaction_count']}")
        for ix in med["interactions"]:
            sev = ix["severity"].upper()
            print(f"  │  ⚠ [{sev}] {ix['drug_a']} + {ix['drug_b']}: {ix['note']}")
    if med["allergy_warnings"]:
        for w in med["allergy_warnings"]:
            print(f"  │  🚨 ALLERGY: {w['medication']} — {w['action']}")
    print(f"  └─ Medication reconciliation complete ✓")

    # Step 5: Clinical Assessment (Agent synthesizes all data)
    print_step(5, "Clinical Assessment", "Agent (Synthesized)")
    assessment = step_clinical_assessment(pid, reg, vit, lab, med)
    print(f"  │  Total concerns: {assessment['total_concerns']}")
    print(f"  │  Servers used: {', '.join(assessment['servers_used'])}")
    for concern in assessment["concerns"]:
        prio = concern["priority"].upper()
        print(f"  │  [{prio}] ({concern['source']}) {concern['concern']}")
    print(f"  └─ Assessment complete ✓")


# ============================================================================
# Section 2: Multi-Patient Comparison
# ============================================================================

def section_multi_patient_comparison():
    """Run workflow for all patients and compare results."""
    print_banner("Section 2: Multi-Patient Workflow Comparison")

    print("""
  Running the workflow for all three patients and comparing the results.
  This demonstrates how the same workflow adapts to different patient data.
    """)

    summaries = []

    for pid in ["P001", "P002", "P003"]:
        reg = step_registration(pid)
        vit = step_vitals(pid)
        lab = step_lab_review(pid)
        med = step_medication_reconciliation(pid)
        assess = step_clinical_assessment(pid, reg, vit, lab, med)

        high_priority = [c for c in assess["concerns"]
                         if c["priority"] in ("critical", "high")]

        summary = {
            "patient": reg["name"],
            "patient_id": pid,
            "age": reg["age"],
            "problems": len(reg["problems"]),
            "vitals_alerts": vit["alert_count"],
            "abnormal_labs": lab["abnormal_count"],
            "critical_labs": lab["critical_count"],
            "medications": med["medication_count"],
            "interactions": med["interaction_count"],
            "total_concerns": assess["total_concerns"],
            "high_priority_concerns": len(high_priority),
        }
        summaries.append(summary)

    # Print comparison table
    print(f"\n  {'Patient':<20} {'Age':>4} {'Dx':>3} {'Vitals':>7} "
          f"{'Abn Lab':>8} {'Crit':>5} {'Meds':>5} {'Int':>4} {'Concerns':>9} {'High':>5}")
    print(f"  {'-'*20} {'-'*4} {'-'*3} {'-'*7} {'-'*8} {'-'*5} {'-'*5} {'-'*4} {'-'*9} {'-'*5}")

    for s in summaries:
        print(f"  {s['patient']:<20} {s['age']:>4} {s['problems']:>3} "
              f"{s['vitals_alerts']:>7} {s['abnormal_labs']:>8} {s['critical_labs']:>5} "
              f"{s['medications']:>5} {s['interactions']:>4} {s['total_concerns']:>9} "
              f"{s['high_priority_concerns']:>5}")

    # Identify highest-risk patient
    riskiest = max(summaries, key=lambda x: x["high_priority_concerns"])
    print(f"\n  Highest-risk patient: {riskiest['patient']} "
          f"({riskiest['high_priority_concerns']} high-priority concerns)")


# ============================================================================
# Section 3: Workflow with AI Summary
# ============================================================================

def section_ai_summary():
    """Use OpenAI to generate a clinical summary from workflow data."""
    print_banner("Section 3: AI-Generated Clinical Summary")

    print("""
  After the workflow gathers data from all servers, an AI model generates
  a structured clinical summary suitable for the provider's review.
    """)

    pid = "P003"

    # Run workflow
    reg = step_registration(pid)
    vit = step_vitals(pid)
    lab = step_lab_review(pid)
    med = step_medication_reconciliation(pid)
    assess = step_clinical_assessment(pid, reg, vit, lab, med)

    # Build context for AI
    context = {
        "patient": f"{reg['name']}, {reg['age']}yo {reg['sex']}",
        "problems": reg["problems"],
        "allergies": reg["allergies"],
        "vitals": vit["vitals"],
        "vital_alerts": vit["alerts"],
        "abnormal_labs": [
            f"{r['test']}: {r['value']} ({r['flag']})"
            for r in lab["abnormal_results"]
        ],
        "medications": [
            f"{m['name']} {m['dose']} {m['freq']}"
            for m in med["medications"]
        ],
        "interactions": [
            f"{ix['drug_a']} + {ix['drug_b']} ({ix['severity']}): {ix['note']}"
            for ix in med["interactions"]
        ],
        "concerns": [
            f"[{c['priority'].upper()}] {c['concern']}"
            for c in assess["concerns"]
        ],
    }

    print(f"\n  Patient: {context['patient']}")
    print(f"  Data gathered from {len(assess['servers_used'])} servers: "
          f"{', '.join(assess['servers_used'])}")
    print(f"  Total concerns: {assess['total_concerns']}")

    if OPENAI_AVAILABLE:
        print(f"\n  Generating AI clinical summary...")
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content":
                     "You are a clinical documentation AI. Generate a concise clinical "
                     "encounter summary from the provided data. Use standard medical "
                     "abbreviations. Structure: Chief Complaint, Vitals, Labs, Medications, "
                     "Assessment/Plan. Keep it under 250 words."},
                    {"role": "user", "content": f"Generate clinical summary:\n{json.dumps(context, indent=2)}"}
                ],
                max_tokens=500,
            )
            summary = response.choices[0].message.content
            print(f"\n  --- AI Clinical Summary ---")
            for line in summary.split("\n"):
                print(f"  {line}")
            print(f"\n  Tokens used: {response.usage.total_tokens}")
        except Exception as e:
            print(f"  (OpenAI error: {e})")
            print(f"  Falling back to structured summary...")
            _print_structured_summary(context)
    else:
        print(f"\n  (OpenAI not available — showing structured summary)")
        _print_structured_summary(context)


def _print_structured_summary(context: dict):
    """Print a structured summary when OpenAI is unavailable."""
    print(f"\n  --- Structured Clinical Summary ---")
    print(f"  Patient: {context['patient']}")
    print(f"  Problems: {', '.join(context['problems'])}")
    print(f"  Allergies: {', '.join(context['allergies'])}")
    print(f"  Vitals: BP {context['vitals']['bp']}, HR {context['vitals']['hr']}, "
          f"SpO2 {context['vitals']['spo2']}%")
    if context["vital_alerts"]:
        print(f"  Vital Alerts: {'; '.join(context['vital_alerts'])}")
    print(f"  Abnormal Labs: {'; '.join(context['abnormal_labs'])}")
    print(f"  Medications: {'; '.join(context['medications'])}")
    if context["interactions"]:
        print(f"  Interactions: {'; '.join(context['interactions'])}")
    print(f"  Concerns: {len(context['concerns'])} identified")
    for c in context["concerns"]:
        print(f"    {c}")


# ============================================================================
# Section 4: Workflow Timing Analysis
# ============================================================================

def section_timing_analysis():
    """Measure and report timing for each workflow step."""
    print_banner("Section 4: Workflow Timing Analysis")

    print("""
  Simulated timing analysis for each workflow step, showing which
  MCP server calls are most expensive. In production, this helps
  identify bottlenecks and optimize server response times.
    """)

    import time

    for pid in ["P001", "P002", "P003"]:
        pt = PATIENTS[pid]
        print(f"\n  --- Workflow Timing: {pt['name']} ({pid}) ---")

        steps = [
            ("Registration", "EHR", lambda: step_registration(pid)),
            ("Vitals", "EHR", lambda: step_vitals(pid)),
            ("Lab Review", "Lab", lambda: step_lab_review(pid)),
            ("Med Reconciliation", "Pharmacy", lambda: step_medication_reconciliation(pid)),
        ]

        total_ms = 0.0
        step_times = []

        for name, server, func in steps:
            start = time.perf_counter()
            result = func()
            elapsed = (time.perf_counter() - start) * 1000

            # Add simulated network latency for realism
            simulated_latency = {"EHR": 12.5, "Lab": 18.3, "Pharmacy": 15.1}
            latency = simulated_latency.get(server, 10.0)
            total_time = elapsed + latency

            step_times.append((name, server, total_time))
            total_ms += total_time

        # Assessment step
        start = time.perf_counter()
        reg = step_registration(pid)
        vit = step_vitals(pid)
        lab = step_lab_review(pid)
        med = step_medication_reconciliation(pid)
        step_clinical_assessment(pid, reg, vit, lab, med)
        elapsed = (time.perf_counter() - start) * 1000
        step_times.append(("Assessment", "Agent", elapsed + 5.0))
        total_ms += elapsed + 5.0

        for name, server, ms in step_times:
            bar = "█" * int(ms / 2)
            print(f"    {name:<22} [{server:<10}] {ms:6.1f}ms {bar}")

        print(f"    {'TOTAL':<22} {'':10}  {total_ms:6.1f}ms")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run the Clinical Workflow exercise."""
    print("=" * 70)
    print("  Exercise 2: Clinical Workflow")
    print("  Complete patient encounter using multiple MCP servers")
    print("=" * 70)

    define_workflow_servers()

    sections = {
        "1": ("Single Patient Workflow", section_single_patient_workflow),
        "2": ("Multi-Patient Comparison", section_multi_patient_comparison),
        "3": ("AI-Generated Clinical Summary", section_ai_summary),
        "4": ("Workflow Timing Analysis", section_timing_analysis),
    }

    while True:
        print("\nSections:")
        for key, (name, _) in sections.items():
            print(f"  {key}. {name}")
        print("  A. Run all sections")
        print("  Q. Quit")

        choice = input("\nSelect section (1-4, A, Q): ").strip().upper()

        if choice == "Q":
            print("\nDone!")
            break
        elif choice == "A":
            for key in sorted(sections.keys()):
                sections[key][1]()
        elif choice in sections:
            sections[choice][1]()
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
