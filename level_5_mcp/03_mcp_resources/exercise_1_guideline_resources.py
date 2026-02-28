"""
Exercise 1: Guideline Resources
==================================

Skills practiced:
- Defining MCP resources with hierarchical URI patterns
- Organizing clinical guidelines by specialty and topic
- Returning structured markdown content for guideline resources
- Implementing resource listing for discoverability

Healthcare context:
Clinical guidelines are published by medical societies (ACC, ADA, ATS) and
provide evidence-based recommendations for managing specific conditions. An
MCP resource server for guidelines gives AI agents access to up-to-date
clinical references through a standard URI pattern:

    guideline/{specialty}/{topic}

Specialties: cardiology, endocrinology, pulmonology, nephrology, etc.
Topics: hypertension, diabetes, asthma, ckd, etc.

Usage:
    python exercise_1_guideline_resources.py
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


# ============================================================================
# Guideline Database
# ============================================================================

GUIDELINES = {
    "cardiology": {
        "hypertension": {
            "title": "Hypertension Management Guideline",
            "source": "ACC/AHA 2017 (updated 2024)",
            "last_updated": "2024-11-15",
            "summary": "Evidence-based recommendations for diagnosis and management of hypertension.",
            "content": """# Hypertension Management

## Diagnosis
- Stage 1: SBP 130-139 or DBP 80-89 mmHg
- Stage 2: SBP ≥ 140 or DBP ≥ 90 mmHg
- Confirm with out-of-office measurements (ABPM or home BP)

## First-Line Pharmacotherapy
1. **ACE Inhibitors** (lisinopril, enalapril) — preferred with DM, CKD, HF
2. **ARBs** (losartan, valsartan) — if ACE-I intolerant
3. **CCBs** (amlodipine, nifedipine) — preferred in Black patients, elderly
4. **Thiazide diuretics** (chlorthalidone, HCTZ) — cost-effective

## BP Targets
- General population: < 130/80 mmHg
- Age ≥ 65 (non-institutionalized): < 130/80 mmHg
- CKD with albuminuria: < 130/80 mmHg

## Lifestyle Modifications
- Sodium restriction: < 1500 mg/day
- DASH diet, regular exercise (150 min/week)
- Weight loss if BMI > 25, limit alcohol
""",
        },
        "heart_failure": {
            "title": "Heart Failure Management Guideline",
            "source": "ACC/AHA/HFSA 2022",
            "last_updated": "2024-06-01",
            "summary": "Guideline-directed medical therapy for HFrEF and HFpEF.",
            "content": """# Heart Failure Management

## HFrEF (EF ≤ 40%) — Four Pillars of GDMT
1. **ACEi/ARB/ARNI** — Start ARNI (sacubitril/valsartan) if tolerated
2. **Beta-Blocker** — carvedilol, metoprolol succinate, or bisoprolol
3. **MRA** — spironolactone or eplerenone
4. **SGLT2 Inhibitor** — dapagliflozin or empagliflozin

## HFpEF (EF > 40%)
- SGLT2 inhibitor (Class 2a recommendation)
- Diuretics for volume management
- Treat underlying comorbidities

## Monitoring
- BNP/NT-proBNP for diagnosis and prognosis
- Renal function and electrolytes with RAAS inhibitors
- Titrate to target doses over 3-6 months
""",
        },
    },
    "endocrinology": {
        "diabetes": {
            "title": "Type 2 Diabetes Management Guideline",
            "source": "ADA Standards of Care 2026",
            "last_updated": "2026-01-01",
            "summary": "Comprehensive management of Type 2 Diabetes Mellitus.",
            "content": """# Type 2 Diabetes Management

## Glycemic Targets
- HbA1c < 7.0% for most adults
- HbA1c < 8.0% for elderly, limited life expectancy, hypoglycemia risk
- HbA1c < 6.5% if achievable without significant hypoglycemia

## First-Line Therapy
- **Metformin** + lifestyle modifications
- Consider early combination therapy if HbA1c ≥ 1.5% above target

## Second-Line Agents (based on comorbidities)
- **ASCVD**: GLP-1 RA (semaglutide, liraglutide) or SGLT2i
- **Heart Failure**: SGLT2 inhibitor (dapagliflozin, empagliflozin)
- **CKD**: SGLT2 inhibitor (if eGFR ≥ 20) or finerenone
- **Obesity**: GLP-1 RA or tirzepatide

## Monitoring
- HbA1c every 3 months until stable, then every 6 months
- Annual: lipid panel, urine albumin-to-creatinine ratio, eye exam
- eGFR at least annually
""",
        },
        "thyroid": {
            "title": "Thyroid Disorder Management",
            "source": "ATA 2015 (updated 2024)",
            "last_updated": "2024-03-10",
            "summary": "Diagnosis and management of hypothyroidism and hyperthyroidism.",
            "content": """# Thyroid Disorder Management

## Hypothyroidism
- **Diagnosis**: Elevated TSH, low free T4
- **Treatment**: Levothyroxine (start 1.6 mcg/kg/day; lower in elderly)
- **Monitoring**: TSH every 6-8 weeks until stable, then annually
- **Target TSH**: 0.5-2.5 mIU/L for most patients

## Hyperthyroidism
- **Diagnosis**: Suppressed TSH, elevated free T4/T3
- **Graves Disease**: Methimazole (preferred), radioactive iodine, or surgery
- **Toxic nodule**: Radioactive iodine or surgery
- **Monitoring**: TFTs every 4-6 weeks during treatment adjustment
""",
        },
    },
    "pulmonology": {
        "asthma": {
            "title": "Asthma Management Guideline",
            "source": "GINA 2025",
            "last_updated": "2025-05-01",
            "summary": "Stepwise approach to asthma management in adults.",
            "content": """# Asthma Management (Adults)

## Diagnosis
- Variable expiratory airflow limitation
- FEV1/FVC < 0.7 with bronchodilator reversibility (≥ 12% and 200 mL)

## Stepwise Therapy
- **Step 1-2**: Low-dose ICS-formoterol as needed (preferred)
- **Step 3**: Low-dose ICS-LABA maintenance + as-needed
- **Step 4**: Medium-dose ICS-LABA
- **Step 5**: High-dose ICS-LABA ± add-on (tiotropium, biologics)

## Biologic Therapies (Step 5)
- Eosinophilic: mepolizumab, benralizumab, dupilumab
- Allergic: omalizumab
- Type 2 inflammation: tezepelumab

## Monitoring
- ACT score every visit, spirometry annually
- Step down after 3 months of good control
""",
        },
    },
    "nephrology": {
        "ckd": {
            "title": "Chronic Kidney Disease Management",
            "source": "KDIGO 2024",
            "last_updated": "2024-09-01",
            "summary": "Management of CKD progression and complications.",
            "content": """# Chronic Kidney Disease Management

## Staging (by eGFR)
- G1: ≥ 90 (normal or high)
- G2: 60-89 (mildly decreased)
- G3a: 45-59, G3b: 30-44 (moderately decreased)
- G4: 15-29 (severely decreased)
- G5: < 15 (kidney failure)

## Key Interventions
- **RAAS Blockade**: ACEi or ARB for albuminuria
- **SGLT2 Inhibitor**: dapagliflozin/empagliflozin if eGFR ≥ 20
- **Finerenone**: if DM + albuminuria despite RAAS blockade
- **BP Target**: < 120 mmHg systolic (SPRINT)

## Monitoring
- eGFR and UACR every 3-12 months based on stage
- Electrolytes, phosphate, PTH as CKD progresses
- Refer to nephrology if eGFR < 30 or rapid decline
""",
        },
    },
}


# ============================================================================
# MCP Resource Server Definition
# ============================================================================

if MCP_AVAILABLE:
    mcp = FastMCP("Clinical Guidelines")

    @mcp.resource("guideline://specialties")
    def list_specialties() -> str:
        """List all available guideline specialties."""
        return json.dumps({
            "specialties": list(GUIDELINES.keys()),
            "total": len(GUIDELINES),
        }, indent=2)

    @mcp.resource("guideline://{specialty}")
    def list_topics(specialty: str) -> str:
        """List available topics for a specialty."""
        spec = specialty.lower()
        if spec not in GUIDELINES:
            return json.dumps({"error": f"Specialty '{specialty}' not found",
                               "available": list(GUIDELINES.keys())})
        topics = GUIDELINES[spec]
        return json.dumps({
            "specialty": spec,
            "topics": [
                {"topic": t, "title": info["title"], "source": info["source"]}
                for t, info in topics.items()
            ],
        }, indent=2)

    @mcp.resource("guideline://{specialty}/{topic}")
    def get_guideline(specialty: str, topic: str) -> str:
        """Get a specific clinical guideline."""
        spec = specialty.lower()
        top = topic.lower()
        if spec not in GUIDELINES or top not in GUIDELINES[spec]:
            return json.dumps({"error": f"Guideline '{specialty}/{topic}' not found"})
        g = GUIDELINES[spec][top]
        return json.dumps({
            "specialty": spec,
            "topic": top,
            "title": g["title"],
            "source": g["source"],
            "last_updated": g["last_updated"],
            "summary": g["summary"],
            "content": g["content"],
        }, indent=2)


# ============================================================================
# Standalone functions (work without MCP SDK)
# ============================================================================

def list_all_specialties() -> dict:
    """List all guideline specialties."""
    return {"specialties": list(GUIDELINES.keys()), "total": len(GUIDELINES)}


def list_specialty_topics(specialty: str) -> dict:
    """List topics under a specialty."""
    spec = specialty.lower()
    if spec not in GUIDELINES:
        return {"error": f"Specialty '{specialty}' not found",
                "available": list(GUIDELINES.keys())}
    return {
        "specialty": spec,
        "topics": [
            {"topic": t, "title": info["title"], "source": info["source"],
             "last_updated": info["last_updated"]}
            for t, info in GUIDELINES[spec].items()
        ],
    }


def get_guideline_content(specialty: str, topic: str) -> dict:
    """Get a specific guideline by specialty and topic."""
    spec = specialty.lower()
    top = topic.lower()
    if spec not in GUIDELINES:
        return {"error": f"Specialty '{specialty}' not found",
                "available": list(GUIDELINES.keys())}
    if top not in GUIDELINES[spec]:
        return {"error": f"Topic '{topic}' not found in {specialty}",
                "available": list(GUIDELINES[spec].keys())}
    g = GUIDELINES[spec][top]
    return {
        "specialty": spec, "topic": top,
        "title": g["title"], "source": g["source"],
        "last_updated": g["last_updated"],
        "summary": g["summary"],
        "content": g["content"],
    }


# ============================================================================
# Demo / Test
# ============================================================================

def main():
    """Demonstrate the guideline resource server."""
    print("=" * 70)
    print("  Exercise 1: Guideline Resources")
    print("  URI pattern: guideline/{specialty}/{topic}")
    print("=" * 70)

    # 1. List specialties
    print("\n--- Available Specialties ---")
    specs = list_all_specialties()
    for s in specs["specialties"]:
        print(f"  • {s}")

    # 2. List topics per specialty
    for spec in specs["specialties"]:
        print(f"\n--- Topics in {spec.title()} ---")
        topics = list_specialty_topics(spec)
        for t in topics["topics"]:
            print(f"  • {t['topic']}: {t['title']} ({t['source']})")

    # 3. Read specific guidelines
    test_cases = [
        ("cardiology", "hypertension"),
        ("endocrinology", "diabetes"),
        ("pulmonology", "asthma"),
        ("nephrology", "ckd"),
    ]
    for spec, topic in test_cases:
        print(f"\n{'─' * 60}")
        print(f"  Reading: guideline://{spec}/{topic}")
        print(f"{'─' * 60}")
        g = get_guideline_content(spec, topic)
        print(f"  Title:   {g['title']}")
        print(f"  Source:  {g['source']}")
        print(f"  Updated: {g['last_updated']}")
        print(f"  Summary: {g['summary']}")
        # Print first few lines of content
        content_lines = g["content"].strip().split("\n")[:8]
        print("  Content (first 8 lines):")
        for line in content_lines:
            print(f"    {line}")
        print("    ...")

    # 4. Test error handling
    print(f"\n{'─' * 60}")
    print("  Error handling tests:")
    print(f"{'─' * 60}")
    result = get_guideline_content("dermatology", "acne")
    print(f"  guideline://dermatology/acne → {result}")
    result = get_guideline_content("cardiology", "afib")
    print(f"  guideline://cardiology/afib  → {result}")

    # 5. Show resource catalog
    print(f"\n{'─' * 60}")
    print("  Full Resource Catalog")
    print(f"{'─' * 60}")
    catalog = []
    for spec, topics in GUIDELINES.items():
        for topic, info in topics.items():
            catalog.append({
                "uri": f"guideline://{spec}/{topic}",
                "name": info["title"],
                "mimeType": "text/markdown",
                "source": info["source"],
            })
    for r in catalog:
        print(f"  {r['uri']:<45} {r['source']}")
    print(f"\n  Total guideline resources: {len(catalog)}")

    if MCP_AVAILABLE:
        print("\n  ✓ MCP server defined — run with: mcp run exercise_1_guideline_resources.py")
    else:
        print("\n  ⚠ Install 'mcp' package for full MCP server functionality")


if __name__ == "__main__":
    main()
