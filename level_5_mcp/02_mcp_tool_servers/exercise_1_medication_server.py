"""
Exercise 1: Medication MCP Server
===================================

Skills practiced:
- Building a focused MCP tool server for a specific domain
- Defining comprehensive data models for medication information
- Implementing search, lookup, interaction, and formulary tools
- Returning structured data suitable for AI agent consumption

Healthcare context:
Medication information is one of the most commonly needed data sources for
clinical AI agents. A medication MCP server provides standardized access to
drug information, interaction checking, alternative lookups, and formulary
status — all critical for medication reconciliation, prescribing support,
and patient education.

This server includes 10+ medications across common therapeutic categories.

Usage:
    python exercise_1_medication_server.py
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
# Medication Database (10+ medications)
# ============================================================================

MEDICATIONS = {
    "metformin": {
        "generic": "metformin", "brand": "Glucophage",
        "class": "Biguanide", "category": "antidiabetic",
        "indication": "Type 2 Diabetes Mellitus",
        "common_dose": "500-2000mg daily in divided doses", "max_dose": "2550mg/day",
        "side_effects": ["nausea", "diarrhea", "abdominal pain", "lactic acidosis (rare)"],
        "contraindications": ["eGFR < 30", "metabolic acidosis", "severe hepatic impairment"],
        "formulary_status": "preferred", "formulary_tier": 1,
    },
    "lisinopril": {
        "generic": "lisinopril", "brand": "Zestril",
        "class": "ACE Inhibitor", "category": "cardiovascular",
        "indication": "Hypertension, Heart Failure",
        "common_dose": "10-40mg daily", "max_dose": "80mg/day",
        "side_effects": ["dry cough", "hyperkalemia", "dizziness", "angioedema (rare)"],
        "contraindications": ["bilateral renal artery stenosis", "pregnancy", "angioedema history"],
        "formulary_status": "preferred", "formulary_tier": 1,
    },
    "atorvastatin": {
        "generic": "atorvastatin", "brand": "Lipitor",
        "class": "HMG-CoA Reductase Inhibitor (Statin)", "category": "cardiovascular",
        "indication": "Hyperlipidemia, CV Risk Reduction",
        "common_dose": "10-80mg daily", "max_dose": "80mg/day",
        "side_effects": ["myalgia", "elevated LFTs", "GI upset", "rhabdomyolysis (rare)"],
        "contraindications": ["active liver disease", "pregnancy", "breastfeeding"],
        "formulary_status": "preferred", "formulary_tier": 1,
    },
    "amlodipine": {
        "generic": "amlodipine", "brand": "Norvasc",
        "class": "Calcium Channel Blocker", "category": "cardiovascular",
        "indication": "Hypertension, Angina",
        "common_dose": "5-10mg daily", "max_dose": "10mg/day",
        "side_effects": ["peripheral edema", "dizziness", "flushing", "headache"],
        "contraindications": ["severe aortic stenosis", "cardiogenic shock"],
        "formulary_status": "preferred", "formulary_tier": 1,
    },
    "omeprazole": {
        "generic": "omeprazole", "brand": "Prilosec",
        "class": "Proton Pump Inhibitor", "category": "gastrointestinal",
        "indication": "GERD, Peptic Ulcer Disease",
        "common_dose": "20-40mg daily", "max_dose": "40mg/day",
        "side_effects": ["headache", "diarrhea", "B12 deficiency (long-term)", "C. diff risk"],
        "contraindications": ["rilpivirine co-administration"],
        "formulary_status": "preferred", "formulary_tier": 1,
    },
    "levothyroxine": {
        "generic": "levothyroxine", "brand": "Synthroid",
        "class": "Thyroid Hormone", "category": "endocrine",
        "indication": "Hypothyroidism",
        "common_dose": "25-200mcg daily", "max_dose": "300mcg/day",
        "side_effects": ["palpitations", "weight loss", "insomnia", "tremor"],
        "contraindications": ["uncorrected adrenal insufficiency", "acute MI"],
        "formulary_status": "preferred", "formulary_tier": 1,
    },
    "sertraline": {
        "generic": "sertraline", "brand": "Zoloft",
        "class": "SSRI", "category": "psychiatric",
        "indication": "Depression, Anxiety, OCD, PTSD",
        "common_dose": "50-200mg daily", "max_dose": "200mg/day",
        "side_effects": ["nausea", "insomnia", "sexual dysfunction", "serotonin syndrome (rare)"],
        "contraindications": ["MAOIs within 14 days", "pimozide co-administration"],
        "formulary_status": "preferred", "formulary_tier": 1,
    },
    "albuterol": {
        "generic": "albuterol", "brand": "ProAir / Ventolin",
        "class": "Short-Acting Beta-2 Agonist", "category": "respiratory",
        "indication": "Asthma, COPD (acute bronchospasm)",
        "common_dose": "2 puffs q4-6h PRN", "max_dose": "12 puffs/day",
        "side_effects": ["tremor", "tachycardia", "nervousness", "hypokalemia"],
        "contraindications": ["hypersensitivity to albuterol"],
        "formulary_status": "preferred", "formulary_tier": 1,
    },
    "warfarin": {
        "generic": "warfarin", "brand": "Coumadin",
        "class": "Vitamin K Antagonist", "category": "anticoagulant",
        "indication": "DVT/PE Prophylaxis, Atrial Fibrillation, Mechanical Valves",
        "common_dose": "2-10mg daily (INR-guided)", "max_dose": "Varies (INR target 2-3)",
        "side_effects": ["bleeding", "bruising", "skin necrosis (rare)", "purple toe syndrome (rare)"],
        "contraindications": ["active bleeding", "pregnancy", "severe hepatic disease"],
        "formulary_status": "non-preferred", "formulary_tier": 2,
    },
    "apixaban": {
        "generic": "apixaban", "brand": "Eliquis",
        "class": "Direct Oral Anticoagulant (Factor Xa Inhibitor)", "category": "anticoagulant",
        "indication": "DVT/PE Treatment, Atrial Fibrillation (stroke prevention)",
        "common_dose": "5mg BID or 2.5mg BID (reduced dose)", "max_dose": "10mg/day",
        "side_effects": ["bleeding", "bruising", "anemia", "nausea"],
        "contraindications": ["active pathological bleeding", "prosthetic heart valve"],
        "formulary_status": "preferred", "formulary_tier": 2,
    },
    "gabapentin": {
        "generic": "gabapentin", "brand": "Neurontin",
        "class": "Gabapentinoid", "category": "neurological",
        "indication": "Neuropathic Pain, Seizures, Restless Leg Syndrome",
        "common_dose": "300-1200mg TID", "max_dose": "3600mg/day",
        "side_effects": ["dizziness", "somnolence", "peripheral edema", "ataxia"],
        "contraindications": ["hypersensitivity"],
        "formulary_status": "preferred", "formulary_tier": 1,
    },
    "prednisone": {
        "generic": "prednisone", "brand": "Deltasone",
        "class": "Corticosteroid", "category": "anti-inflammatory",
        "indication": "Inflammation, Autoimmune Conditions, Allergic Reactions",
        "common_dose": "5-60mg daily (condition-dependent)", "max_dose": "Varies by indication",
        "side_effects": ["weight gain", "hyperglycemia", "mood changes", "osteoporosis (long-term)"],
        "contraindications": ["systemic fungal infection", "live vaccines during high-dose therapy"],
        "formulary_status": "preferred", "formulary_tier": 1,
    },
}

DRUG_INTERACTIONS = {
    ("warfarin", "omeprazole"): {
        "severity": "moderate",
        "description": "Omeprazole may increase warfarin effect by inhibiting CYP2C19",
        "recommendation": "Monitor INR closely when starting or stopping omeprazole",
    },
    ("warfarin", "sertraline"): {
        "severity": "major",
        "description": "SSRIs increase bleeding risk and may potentiate warfarin anticoagulant effect",
        "recommendation": "Monitor INR closely. Consider alternative SSRI or alternative anticoagulant.",
    },
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
    ("sertraline", "levothyroxine"): {
        "severity": "minor",
        "description": "SSRIs may reduce levothyroxine efficacy in some patients",
        "recommendation": "Monitor TSH after starting/stopping sertraline.",
    },
    ("gabapentin", "prednisone"): {
        "severity": "none",
        "description": "No clinically significant interaction",
        "recommendation": "Safe to co-prescribe.",
    },
    ("warfarin", "apixaban"): {
        "severity": "major",
        "description": "Dual anticoagulation — significantly increased bleeding risk",
        "recommendation": "DO NOT co-prescribe. Choose one anticoagulant.",
    },
    ("albuterol", "prednisone"): {
        "severity": "minor",
        "description": "Both may lower potassium; combined hypokalemia risk",
        "recommendation": "Monitor potassium, especially with high-dose or prolonged use.",
    },
}

# Therapeutic alternatives mapping
ALTERNATIVES = {
    "metformin": ["glipizide", "sitagliptin", "pioglitazone"],
    "lisinopril": ["losartan", "enalapril", "ramipril"],
    "atorvastatin": ["rosuvastatin", "simvastatin", "pravastatin"],
    "amlodipine": ["nifedipine", "felodipine", "diltiazem"],
    "omeprazole": ["pantoprazole", "lansoprazole", "esomeprazole"],
    "levothyroxine": ["liothyronine", "Armour Thyroid"],
    "sertraline": ["escitalopram", "fluoxetine", "citalopram"],
    "albuterol": ["levalbuterol"],
    "warfarin": ["apixaban", "rivaroxaban", "dabigatran"],
    "apixaban": ["rivaroxaban", "dabigatran", "warfarin"],
    "gabapentin": ["pregabalin", "duloxetine", "amitriptyline"],
    "prednisone": ["methylprednisolone", "dexamethasone", "hydrocortisone"],
}


# ============================================================================
# Tool implementations
# ============================================================================

def lookup_medication(medication_name: str) -> dict:
    """Look up detailed medication information."""
    name = medication_name.lower().strip()
    if name in MEDICATIONS:
        return {"found": True, **MEDICATIONS[name]}

    # Partial match
    matches = [k for k in MEDICATIONS if name in k or k in name]
    if matches:
        return {"found": True, **MEDICATIONS[matches[0]], "matched_as": matches[0]}

    # Search brand names
    for key, med in MEDICATIONS.items():
        if name in med["brand"].lower():
            return {"found": True, **med, "matched_as": key}

    return {
        "found": False,
        "error": f"Medication '{medication_name}' not found",
        "suggestion": "Try generic name. Available: " + ", ".join(sorted(MEDICATIONS.keys())),
    }


def check_interaction(drug_a: str, drug_b: str) -> dict:
    """Check for drug-drug interactions between two medications."""
    a = drug_a.lower().strip()
    b = drug_b.lower().strip()

    if a == b:
        return {"error": "Cannot check interaction of a drug with itself"}

    key = (a, b) if (a, b) in DRUG_INTERACTIONS else (b, a)
    if key in DRUG_INTERACTIONS:
        return {"drug_a": drug_a, "drug_b": drug_b, **DRUG_INTERACTIONS[key]}

    if a not in MEDICATIONS:
        return {"error": f"Drug '{drug_a}' not found in database"}
    if b not in MEDICATIONS:
        return {"error": f"Drug '{drug_b}' not found in database"}

    return {
        "drug_a": drug_a, "drug_b": drug_b,
        "severity": "unknown",
        "description": "No specific interaction data available",
        "recommendation": "Consult pharmacist or comprehensive interaction database",
    }


def find_alternative(medication_name: str, reason: str = "general") -> dict:
    """Find therapeutic alternatives for a medication."""
    name = medication_name.lower().strip()

    if name not in MEDICATIONS:
        return {"error": f"Medication '{medication_name}' not found"}

    med = MEDICATIONS[name]
    alts = ALTERNATIVES.get(name, [])

    return {
        "original": medication_name,
        "class": med["class"],
        "reason_for_alternative": reason,
        "alternatives": alts,
        "note": "Alternatives are within the same therapeutic class or indication. "
                "Clinical judgment required for individual patient selection.",
    }


def check_formulary_status(medication_name: str) -> dict:
    """Check if a medication is on the formulary and its tier."""
    name = medication_name.lower().strip()

    if name not in MEDICATIONS:
        return {"error": f"Medication '{medication_name}' not found"}

    med = MEDICATIONS[name]
    status = med.get("formulary_status", "unknown")
    tier = med.get("formulary_tier", "unknown")

    tier_descriptions = {
        1: "Tier 1 — Generic/Preferred: lowest copay",
        2: "Tier 2 — Preferred Brand: moderate copay",
        3: "Tier 3 — Non-Preferred: higher copay",
        4: "Tier 4 — Specialty: highest copay, may require prior authorization",
    }

    requires_pa = status == "non-preferred" or tier >= 3

    return {
        "medication": medication_name,
        "formulary_status": status,
        "tier": tier,
        "tier_description": tier_descriptions.get(tier, "Unknown tier"),
        "requires_prior_authorization": requires_pa,
        "alternatives_if_non_preferred": ALTERNATIVES.get(name, []) if status == "non-preferred" else [],
    }


# ============================================================================
# MCP Server Registration
# ============================================================================

if MCP_AVAILABLE:
    mcp = FastMCP("Medication Server")

    @mcp.tool()
    def mcp_lookup_medication(medication_name: str) -> str:
        """Look up detailed medication information including class, indication,
        dosing, side effects, and contraindications. Use when a clinician
        asks about a specific drug or needs prescribing information."""
        return json.dumps(lookup_medication(medication_name))

    @mcp.tool()
    def mcp_check_interaction(drug_a: str, drug_b: str) -> str:
        """Check for known drug-drug interactions between two medications.
        Returns severity (none/minor/moderate/major) and clinical recommendation.
        Use before prescribing or during medication reconciliation."""
        return json.dumps(check_interaction(drug_a, drug_b))

    @mcp.tool()
    def mcp_find_alternative(medication_name: str, reason: str = "general") -> str:
        """Find therapeutic alternatives for a medication. Useful when a patient
        has an allergy, intolerance, formulary restriction, or treatment failure.
        Returns list of alternatives in the same class or indication."""
        return json.dumps(find_alternative(medication_name, reason))

    @mcp.tool()
    def mcp_check_formulary_status(medication_name: str) -> str:
        """Check formulary status and tier for a medication. Returns whether
        the drug is preferred/non-preferred, copay tier, and whether prior
        authorization is required. Use for cost-effective prescribing."""
        return json.dumps(check_formulary_status(medication_name))


# ============================================================================
# Demo / Test
# ============================================================================

def main():
    """Demo all medication server tools."""
    print("=" * 70)
    print("  Exercise 1: Medication MCP Server")
    print(f"  Database: {len(MEDICATIONS)} medications, {len(DRUG_INTERACTIONS)} interactions")
    print("=" * 70)

    if MCP_AVAILABLE:
        print("  ✓ MCP server registered as 'Medication Server' with 4 tools\n")
    else:
        print("  ⚠ MCP SDK not installed — running tools in standalone mode\n")

    # Tool 1: Lookup
    print("  TOOL 1: lookup_medication")
    print("  " + "─" * 55)
    for med_name in ["metformin", "Zoloft", "apixaban", "unknowndrug"]:
        result = lookup_medication(med_name)
        if result.get("found"):
            print(f"  ✓ {med_name} → {result['class']} | {result['indication']}")
        else:
            print(f"  ✗ {med_name} → {result.get('error', 'Not found')}")

    # Tool 2: Interactions
    print(f"\n  TOOL 2: check_interaction")
    print("  " + "─" * 55)
    interaction_tests = [
        ("warfarin", "sertraline"),
        ("atorvastatin", "amlodipine"),
        ("gabapentin", "prednisone"),
        ("warfarin", "apixaban"),
    ]
    for a, b in interaction_tests:
        result = check_interaction(a, b)
        severity = result.get("severity", "?")
        print(f"  {a} + {b} → [{severity.upper()}] {result.get('recommendation', '')[:60]}")

    # Tool 3: Alternatives
    print(f"\n  TOOL 3: find_alternative")
    print("  " + "─" * 55)
    for med_name in ["lisinopril", "warfarin", "sertraline"]:
        result = find_alternative(med_name, "patient intolerance")
        alts = result.get("alternatives", [])
        print(f"  {med_name} → Alternatives: {', '.join(alts)}")

    # Tool 4: Formulary
    print(f"\n  TOOL 4: check_formulary_status")
    print("  " + "─" * 55)
    for med_name in ["metformin", "warfarin", "apixaban"]:
        result = check_formulary_status(med_name)
        status = result.get("formulary_status", "?")
        tier = result.get("tier", "?")
        pa = "Yes" if result.get("requires_prior_authorization") else "No"
        print(f"  {med_name} → {status} (Tier {tier}), PA Required: {pa}")

    print(f"\n{'=' * 70}")
    print("  ✓ All medication server tools tested")
    print("=" * 70)


if __name__ == "__main__":
    main()
