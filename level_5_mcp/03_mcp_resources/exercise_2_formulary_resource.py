"""
Exercise 2: Formulary Resource Server
========================================

Skills practiced:
- Building a comprehensive MCP resource server for medication data
- Implementing multiple resource endpoints with different URI patterns
- Returning structured JSON for medication lookups and interaction checks
- Designing resources that complement (not replace) tool-based lookups

Healthcare context:
A hospital formulary is the master list of approved medications. Clinicians
and AI agents need fast access to formulary status, medication details, and
drug-drug interaction data. This exercise builds a formulary resource server
with three resource types:

    formulary://medications              → full medication list
    formulary://medication/{name}        → details for one medication
    formulary://interactions/{drug1}/{drug2} → interaction check

Usage:
    python exercise_2_formulary_resource.py
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
# Formulary Database
# ============================================================================

MEDICATIONS = {
    "metformin": {
        "generic": "metformin", "brand": "Glucophage",
        "class": "Biguanide", "category": "antidiabetic",
        "indication": "Type 2 Diabetes Mellitus",
        "mechanism": "Decreases hepatic glucose production, increases insulin sensitivity",
        "common_dose": "500-2000mg daily in divided doses",
        "max_dose": "2550mg/day",
        "side_effects": ["nausea", "diarrhea", "abdominal pain", "lactic acidosis (rare)"],
        "contraindications": ["eGFR < 30", "metabolic acidosis", "severe hepatic impairment"],
        "tier": 1, "status": "preferred", "prior_auth": False,
        "cost_30day": "$4-$15",
    },
    "lisinopril": {
        "generic": "lisinopril", "brand": "Zestril",
        "class": "ACE Inhibitor", "category": "cardiovascular",
        "indication": "Hypertension, Heart Failure, Diabetic Nephropathy",
        "mechanism": "Inhibits angiotensin-converting enzyme",
        "common_dose": "10-40mg once daily",
        "max_dose": "80mg/day",
        "side_effects": ["dry cough", "hyperkalemia", "dizziness", "angioedema (rare)"],
        "contraindications": ["bilateral renal artery stenosis", "pregnancy", "angioedema history"],
        "tier": 1, "status": "preferred", "prior_auth": False,
        "cost_30day": "$4-$10",
    },
    "atorvastatin": {
        "generic": "atorvastatin", "brand": "Lipitor",
        "class": "HMG-CoA Reductase Inhibitor", "category": "cardiovascular",
        "indication": "Hyperlipidemia, CV Risk Reduction",
        "mechanism": "Inhibits HMG-CoA reductase, reduces LDL cholesterol synthesis",
        "common_dose": "10-80mg once daily",
        "max_dose": "80mg/day",
        "side_effects": ["myalgia", "elevated LFTs", "GI upset", "rhabdomyolysis (rare)"],
        "contraindications": ["active liver disease", "pregnancy", "breastfeeding"],
        "tier": 1, "status": "preferred", "prior_auth": False,
        "cost_30day": "$4-$20",
    },
    "amlodipine": {
        "generic": "amlodipine", "brand": "Norvasc",
        "class": "Calcium Channel Blocker", "category": "cardiovascular",
        "indication": "Hypertension, Angina",
        "mechanism": "Blocks L-type calcium channels in vascular smooth muscle",
        "common_dose": "5-10mg once daily",
        "max_dose": "10mg/day",
        "side_effects": ["peripheral edema", "dizziness", "flushing", "headache"],
        "contraindications": ["severe aortic stenosis", "cardiogenic shock"],
        "tier": 1, "status": "preferred", "prior_auth": False,
        "cost_30day": "$4-$12",
    },
    "omeprazole": {
        "generic": "omeprazole", "brand": "Prilosec",
        "class": "Proton Pump Inhibitor", "category": "gastrointestinal",
        "indication": "GERD, Peptic Ulcer Disease",
        "mechanism": "Irreversibly inhibits H+/K+ ATPase in gastric parietal cells",
        "common_dose": "20-40mg once daily",
        "max_dose": "40mg/day",
        "side_effects": ["headache", "diarrhea", "B12 deficiency (long‑term)", "C. diff risk"],
        "contraindications": ["rilpivirine co-administration"],
        "tier": 1, "status": "preferred", "prior_auth": False,
        "cost_30day": "$4-$10",
    },
    "semaglutide": {
        "generic": "semaglutide", "brand": "Ozempic / Wegovy",
        "class": "GLP-1 Receptor Agonist", "category": "antidiabetic",
        "indication": "Type 2 Diabetes, Obesity",
        "mechanism": "Mimics GLP-1, enhances glucose-dependent insulin secretion",
        "common_dose": "0.25-2mg weekly (SC injection)",
        "max_dose": "2mg/week",
        "side_effects": ["nausea", "vomiting", "diarrhea", "pancreatitis (rare)"],
        "contraindications": ["personal/family MEN2", "medullary thyroid carcinoma"],
        "tier": 3, "status": "non-preferred", "prior_auth": True,
        "cost_30day": "$900-$1300",
    },
    "dapagliflozin": {
        "generic": "dapagliflozin", "brand": "Farxiga",
        "class": "SGLT2 Inhibitor", "category": "antidiabetic",
        "indication": "Type 2 Diabetes, Heart Failure, CKD",
        "mechanism": "Inhibits SGLT2 in proximal tubule, causes glucosuria",
        "common_dose": "10mg once daily",
        "max_dose": "10mg/day",
        "side_effects": ["UTI", "genital mycotic infections", "DKA (rare)", "dehydration"],
        "contraindications": ["dialysis", "type 1 diabetes (DKA risk)"],
        "tier": 2, "status": "preferred", "prior_auth": False,
        "cost_30day": "$500-$600",
    },
    "losartan": {
        "generic": "losartan", "brand": "Cozaar",
        "class": "Angiotensin II Receptor Blocker", "category": "cardiovascular",
        "indication": "Hypertension, Diabetic Nephropathy",
        "mechanism": "Blocks angiotensin II at AT1 receptor",
        "common_dose": "50-100mg once daily",
        "max_dose": "100mg/day",
        "side_effects": ["dizziness", "hyperkalemia", "hypotension"],
        "contraindications": ["pregnancy", "bilateral renal artery stenosis"],
        "tier": 1, "status": "preferred", "prior_auth": False,
        "cost_30day": "$4-$15",
    },
}

INTERACTIONS = {
    ("metformin", "lisinopril"): {
        "severity": "minor",
        "description": "ACE inhibitors may slightly enhance metformin's hypoglycemic effect.",
        "recommendation": "Monitor blood glucose. Usually safe to co-prescribe.",
    },
    ("atorvastatin", "amlodipine"): {
        "severity": "moderate",
        "description": "Amlodipine increases atorvastatin exposure by ~20% via CYP3A4 inhibition.",
        "recommendation": "Limit atorvastatin to 20mg when combined with amlodipine.",
    },
    ("metformin", "dapagliflozin"): {
        "severity": "minor",
        "description": "Additive glucose-lowering effect. Low hypoglycemia risk individually.",
        "recommendation": "Safe and common combination. Monitor for dehydration.",
    },
    ("lisinopril", "losartan"): {
        "severity": "major",
        "description": "Dual RAAS blockade: increased risk of hyperkalemia, hypotension, renal failure.",
        "recommendation": "AVOID combination. Use one RAAS blocker only.",
    },
    ("lisinopril", "dapagliflozin"): {
        "severity": "moderate",
        "description": "Additive hypotensive effect. SGLT2i-induced volume depletion may worsen.",
        "recommendation": "Monitor BP and renal function closely. May need ACEi dose adjustment.",
    },
    ("omeprazole", "metformin"): {
        "severity": "minor",
        "description": "Long-term PPI use may impair B12 absorption; metformin also reduces B12.",
        "recommendation": "Monitor B12 levels annually if long-term combination.",
    },
    ("semaglutide", "metformin"): {
        "severity": "minor",
        "description": "Additive glucose-lowering. Semaglutide may slow metformin absorption.",
        "recommendation": "Common and effective combination. Monitor for GI side effects.",
    },
}


# ============================================================================
# Resource functions
# ============================================================================

def get_formulary_list() -> dict:
    """Return the complete formulary medication list."""
    return {
        "resource": "formulary://medications",
        "total_medications": len(MEDICATIONS),
        "last_updated": "2026-02-01",
        "medications": [
            {
                "name": m["generic"], "brand": m["brand"], "class": m["class"],
                "category": m["category"], "tier": m["tier"],
                "status": m["status"], "prior_auth": m["prior_auth"],
            }
            for m in MEDICATIONS.values()
        ],
    }


def get_medication_detail(name: str) -> dict:
    """Return detailed information for a specific medication."""
    key = name.lower().strip()
    if key in MEDICATIONS:
        med = MEDICATIONS[key]
        return {
            "resource": f"formulary://medication/{key}",
            "found": True,
            **med,
        }
    # Partial match
    matches = [k for k in MEDICATIONS if key in k or k in key]
    if matches:
        med = MEDICATIONS[matches[0]]
        return {"resource": f"formulary://medication/{matches[0]}", "found": True, **med}
    return {
        "resource": f"formulary://medication/{key}",
        "found": False,
        "error": f"Medication '{name}' not found",
        "available": sorted(MEDICATIONS.keys()),
    }


def get_interaction(drug1: str, drug2: str) -> dict:
    """Check interaction between two drugs."""
    d1 = drug1.lower().strip()
    d2 = drug2.lower().strip()
    key1 = (d1, d2)
    key2 = (d2, d1)
    if key1 in INTERACTIONS:
        info = INTERACTIONS[key1]
    elif key2 in INTERACTIONS:
        info = INTERACTIONS[key2]
    else:
        # Check if both drugs are known
        known1 = d1 in MEDICATIONS
        known2 = d2 in MEDICATIONS
        if not known1 or not known2:
            unknown = [d for d, k in [(d1, known1), (d2, known2)] if not k]
            return {
                "resource": f"formulary://interactions/{d1}/{d2}",
                "found": False,
                "error": f"Unknown medication(s): {', '.join(unknown)}",
            }
        info = {
            "severity": "none",
            "description": "No clinically significant interaction documented.",
            "recommendation": "Safe to co-prescribe based on available evidence.",
        }
    return {
        "resource": f"formulary://interactions/{d1}/{d2}",
        "found": True,
        "drug1": d1, "drug2": d2,
        **info,
    }


# ============================================================================
# MCP Server Definition
# ============================================================================

if MCP_AVAILABLE:
    mcp = FastMCP("Formulary Resources")

    @mcp.resource("formulary://medications")
    def mcp_formulary_list() -> str:
        """Complete hospital formulary medication list."""
        return json.dumps(get_formulary_list(), indent=2)

    @mcp.resource("formulary://medication/{name}")
    def mcp_medication_detail(name: str) -> str:
        """Detailed information for a specific medication."""
        return json.dumps(get_medication_detail(name), indent=2)

    @mcp.resource("formulary://interactions/{drug1}/{drug2}")
    def mcp_interaction_check(drug1: str, drug2: str) -> str:
        """Check for drug-drug interactions."""
        return json.dumps(get_interaction(drug1, drug2), indent=2)


# ============================================================================
# Demo / Test
# ============================================================================

def main():
    """Demonstrate the formulary resource server."""
    print("=" * 70)
    print("  Exercise 2: Formulary Resource Server")
    print("  Resources: formulary://medications, medication/{name}, interactions/{a}/{b}")
    print("=" * 70)

    # 1. Full formulary list
    print("\n--- formulary://medications ---")
    formulary = get_formulary_list()
    print(f"  Total medications: {formulary['total_medications']}")
    print(f"  Last updated: {formulary['last_updated']}")
    print(f"\n  {'Name':<15} {'Brand':<15} {'Class':<25} {'Tier':>4}  {'PA'}")
    print(f"  {'─'*15} {'─'*15} {'─'*25} {'─'*4}  {'─'*3}")
    for m in formulary["medications"]:
        pa = "Yes" if m["prior_auth"] else "No"
        print(f"  {m['name']:<15} {m['brand']:<15} {m['class']:<25} {m['tier']:>4}  {pa}")

    # 2. Individual medication detail
    print("\n--- formulary://medication/{name} ---")
    for name in ["metformin", "semaglutide", "dapagliflozin"]:
        detail = get_medication_detail(name)
        print(f"\n  {detail['generic'].upper()} ({detail['brand']})")
        print(f"    Class:          {detail['class']}")
        print(f"    Indication:     {detail['indication']}")
        print(f"    Common dose:    {detail['common_dose']}")
        print(f"    Tier:           {detail['tier']} ({detail['status']})")
        print(f"    Prior auth:     {'Required' if detail['prior_auth'] else 'Not required'}")
        print(f"    Cost (30-day):  {detail['cost_30day']}")
        print(f"    Side effects:   {', '.join(detail['side_effects'][:3])}")

    # 3. Drug interactions
    print("\n--- formulary://interactions/{drug1}/{drug2} ---")
    test_pairs = [
        ("lisinopril", "losartan"),
        ("atorvastatin", "amlodipine"),
        ("metformin", "semaglutide"),
        ("metformin", "lisinopril"),
        ("lisinopril", "omeprazole"),
    ]
    for d1, d2 in test_pairs:
        result = get_interaction(d1, d2)
        severity = result.get("severity", "unknown")
        sev_marker = {"none": "✓", "minor": "○", "moderate": "△", "major": "✗"}.get(severity, "?")
        print(f"  {sev_marker} {d1} + {d2}: [{severity.upper()}] {result.get('description', '')[:70]}")

    # 4. Error handling
    print("\n--- Error handling ---")
    result = get_medication_detail("aspirin")
    print(f"  formulary://medication/aspirin → {result.get('error', 'found')}")
    result = get_interaction("aspirin", "metformin")
    print(f"  formulary://interactions/aspirin/metformin → {result.get('error', 'found')}")

    # 5. Resource catalog
    print("\n--- Resource Catalog ---")
    catalog = [
        {"uri": "formulary://medications", "type": "static",
         "description": "Complete formulary list"},
        {"uri": "formulary://medication/{name}", "type": "template",
         "description": "Medication detail by name"},
        {"uri": "formulary://interactions/{drug1}/{drug2}", "type": "template",
         "description": "Drug-drug interaction check"},
    ]
    for r in catalog:
        print(f"  [{r['type']:>8}] {r['uri']:<50} {r['description']}")

    if MCP_AVAILABLE:
        print("\n  ✓ MCP server defined — run with: mcp run exercise_2_formulary_resource.py")


if __name__ == "__main__":
    main()
