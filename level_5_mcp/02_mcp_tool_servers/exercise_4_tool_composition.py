"""
Exercise 4: Tool Composition in MCP
=====================================

Skills practiced:
- Building tools that internally call other tools
- Composing multiple tool results into comprehensive assessments
- Aggregating findings with risk scoring
- Structuring composite responses for agent consumption

Healthcare context:
Clinical decision-making rarely relies on a single data point. A comprehensive
patient assessment combines BMI, vital signs, lab results, and medication review.
This exercise builds composite tools that call other tools internally to produce
holistic assessments — mirroring how clinicians synthesize multiple sources.

Example: comprehensive_assessment calls calculate_bmi + interpret_labs +
check_medications + assess_vitals and combines the results into a single
clinical summary with risk scoring.

Usage:
    python exercise_4_tool_composition.py
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
# Base tools (building blocks)
# ============================================================================

def calculate_bmi(weight_kg: float, height_m: float) -> dict:
    """Calculate BMI with WHO classification."""
    if weight_kg <= 0 or height_m <= 0:
        return {"error": "Weight and height must be positive"}
    bmi = round(weight_kg / (height_m ** 2), 1)
    if bmi < 18.5:
        category, risk = "Underweight", "moderate"
    elif bmi < 25:
        category, risk = "Normal", "low"
    elif bmi < 30:
        category, risk = "Overweight", "moderate"
    elif bmi < 35:
        category, risk = "Obese Class I", "high"
    elif bmi < 40:
        category, risk = "Obese Class II", "very high"
    else:
        category, risk = "Obese Class III", "extremely high"
    return {"bmi": bmi, "category": category, "risk_level": risk}


def assess_blood_pressure(systolic: int, diastolic: int) -> dict:
    """Classify blood pressure with risk assessment."""
    if systolic <= 0 or diastolic <= 0:
        return {"error": "BP values must be positive"}
    if systolic < 120 and diastolic < 80:
        category, risk = "Normal", "low"
    elif systolic < 130 and diastolic < 80:
        category, risk = "Elevated", "moderate"
    elif systolic < 140 or diastolic < 90:
        category, risk = "Stage 1 HTN", "high"
    elif systolic < 180 and diastolic < 120:
        category, risk = "Stage 2 HTN", "very high"
    else:
        category, risk = "Hypertensive Crisis", "critical"
    return {
        "reading": f"{systolic}/{diastolic} mmHg",
        "category": category,
        "risk_level": risk,
    }


LAB_RANGES = {
    "glucose": {"unit": "mg/dL", "low": 70, "high": 100, "critical_low": 40, "critical_high": 500},
    "hemoglobin": {"unit": "g/dL", "low": 12.0, "high": 17.5, "critical_low": 7.0, "critical_high": 20.0},
    "hba1c": {"unit": "%", "low": 4.0, "high": 5.6, "critical_low": 3.0, "critical_high": 15.0},
    "potassium": {"unit": "mEq/L", "low": 3.5, "high": 5.0, "critical_low": 2.5, "critical_high": 6.5},
    "creatinine": {"unit": "mg/dL", "low": 0.7, "high": 1.3, "critical_low": 0.4, "critical_high": 10.0},
    "ldl": {"unit": "mg/dL", "low": 0, "high": 100, "critical_low": 0, "critical_high": 500},
    "tsh": {"unit": "mIU/L", "low": 0.4, "high": 4.0, "critical_low": 0.01, "critical_high": 100.0},
    "alt": {"unit": "U/L", "low": 7, "high": 56, "critical_low": 0, "critical_high": 1000},
}


def interpret_lab(test_name: str, value: float) -> dict:
    """Interpret a single lab value."""
    key = test_name.lower().strip()
    if key not in LAB_RANGES:
        return {"error": f"Unknown test: {test_name}"}
    ref = LAB_RANGES[key]
    if value <= ref["critical_low"]:
        status, risk = "critically low", "critical"
    elif value < ref["low"]:
        status, risk = "low", "moderate"
    elif value <= ref["high"]:
        status, risk = "normal", "low"
    elif value >= ref["critical_high"]:
        status, risk = "critically high", "critical"
    else:
        status, risk = "high", "moderate"
    return {
        "test": test_name, "value": value, "unit": ref["unit"],
        "status": status, "risk_level": risk,
        "reference": f"{ref['low']}-{ref['high']} {ref['unit']}",
    }


MEDICATION_DATA = {
    "metformin": {"class": "Biguanide", "indication": "Type 2 Diabetes",
                  "monitoring": ["renal function", "B12 levels", "lactic acid"]},
    "lisinopril": {"class": "ACE Inhibitor", "indication": "Hypertension",
                   "monitoring": ["potassium", "creatinine", "blood pressure"]},
    "atorvastatin": {"class": "Statin", "indication": "Hyperlipidemia",
                     "monitoring": ["LDL cholesterol", "liver enzymes (ALT)", "muscle symptoms"]},
    "amlodipine": {"class": "CCB", "indication": "Hypertension",
                   "monitoring": ["blood pressure", "heart rate", "peripheral edema"]},
    "sertraline": {"class": "SSRI", "indication": "Depression/Anxiety",
                   "monitoring": ["mood", "suicidal ideation", "serotonin syndrome signs"]},
    "levothyroxine": {"class": "Thyroid Hormone", "indication": "Hypothyroidism",
                      "monitoring": ["TSH", "T4", "symptoms"]},
}

INTERACTIONS = {
    ("metformin", "lisinopril"): {"severity": "minor", "note": "Monitor glucose (ACEi may enhance hypoglycemic effect)"},
    ("atorvastatin", "amlodipine"): {"severity": "moderate", "note": "Limit atorvastatin to 20mg with amlodipine"},
    ("sertraline", "metformin"): {"severity": "minor", "note": "SSRIs may affect glucose levels"},
}


def check_medications(medication_list: list[str]) -> dict:
    """Review medications and check for interactions."""
    results = []
    interactions_found = []

    for med_name in medication_list:
        key = med_name.lower().strip()
        if key in MEDICATION_DATA:
            results.append({
                "medication": key,
                "found": True,
                **MEDICATION_DATA[key],
            })
        else:
            results.append({"medication": key, "found": False})

    # Check pairwise interactions
    meds_lower = [m.lower().strip() for m in medication_list]
    for i, a in enumerate(meds_lower):
        for b in meds_lower[i + 1:]:
            key = (a, b) if (a, b) in INTERACTIONS else (b, a)
            if key in INTERACTIONS:
                interactions_found.append({
                    "drug_a": key[0], "drug_b": key[1],
                    **INTERACTIONS[key],
                })

    risk = "low"
    if any(i["severity"] == "major" for i in interactions_found):
        risk = "high"
    elif any(i["severity"] == "moderate" for i in interactions_found):
        risk = "moderate"
    elif interactions_found:
        risk = "low"

    return {
        "medication_count": len(medication_list),
        "medications": results,
        "interactions": interactions_found,
        "interaction_risk": risk,
    }


# ============================================================================
# Composite tools (call base tools internally)
# ============================================================================

def comprehensive_assessment(
    weight_kg: float, height_m: float,
    systolic: int, diastolic: int,
    labs: dict[str, float],
    medications: list[str]
) -> dict:
    """
    Comprehensive patient assessment combining BMI, vitals, labs, and medications.

    This is a COMPOSITE tool that calls:
    - calculate_bmi()
    - assess_blood_pressure()
    - interpret_lab() (for each lab)
    - check_medications()

    And synthesizes the results into a unified assessment with risk scoring.
    """
    assessment = {
        "assessment_type": "comprehensive",
        "components": {},
        "findings": [],
        "risk_factors": [],
    }

    # 1. BMI Assessment
    bmi_result = calculate_bmi(weight_kg, height_m)
    assessment["components"]["bmi"] = bmi_result
    if "error" not in bmi_result:
        if bmi_result["risk_level"] != "low":
            assessment["risk_factors"].append(
                f"BMI {bmi_result['bmi']} ({bmi_result['category']})"
            )
            assessment["findings"].append(
                f"BMI is {bmi_result['bmi']} — {bmi_result['category']}"
            )

    # 2. Blood Pressure Assessment
    bp_result = assess_blood_pressure(systolic, diastolic)
    assessment["components"]["blood_pressure"] = bp_result
    if "error" not in bp_result:
        if bp_result["risk_level"] != "low":
            assessment["risk_factors"].append(
                f"BP {bp_result['reading']} ({bp_result['category']})"
            )
            assessment["findings"].append(
                f"Blood pressure: {bp_result['reading']} — {bp_result['category']}"
            )

    # 3. Lab Assessments
    lab_results = {}
    abnormal_labs = []
    critical_labs = []
    for test_name, value in labs.items():
        result = interpret_lab(test_name, value)
        lab_results[test_name] = result
        if "error" not in result:
            if result["status"] == "normal":
                continue
            if "critical" in result["status"]:
                critical_labs.append(
                    f"{test_name}: {value} {result['unit']} ({result['status']})"
                )
            else:
                abnormal_labs.append(
                    f"{test_name}: {value} {result['unit']} ({result['status']})"
                )

    assessment["components"]["labs"] = lab_results
    if critical_labs:
        assessment["risk_factors"].extend(critical_labs)
        assessment["findings"].append(
            f"CRITICAL labs: {'; '.join(critical_labs)}"
        )
    if abnormal_labs:
        assessment["findings"].append(
            f"Abnormal labs: {'; '.join(abnormal_labs)}"
        )

    # 4. Medication Review
    med_result = check_medications(medications)
    assessment["components"]["medications"] = med_result
    if med_result["interactions"]:
        for interaction in med_result["interactions"]:
            assessment["findings"].append(
                f"Drug interaction: {interaction['drug_a']} + {interaction['drug_b']} "
                f"({interaction['severity']})"
            )
            if interaction["severity"] in ("moderate", "major"):
                assessment["risk_factors"].append(
                    f"Drug interaction: {interaction['drug_a']}/{interaction['drug_b']}"
                )

    # 5. Overall Risk Score
    risk_score = _calculate_risk_score(assessment)
    assessment["overall_risk_score"] = risk_score
    assessment["overall_risk_level"] = _risk_level_from_score(risk_score)
    assessment["risk_factor_count"] = len(assessment["risk_factors"])
    assessment["finding_count"] = len(assessment["findings"])

    return assessment


def _calculate_risk_score(assessment: dict) -> float:
    """Calculate an overall risk score (0-100) from assessment components."""
    score = 0.0

    # BMI contribution (0-20 points)
    bmi_risk = assessment["components"].get("bmi", {}).get("risk_level", "low")
    bmi_scores = {"low": 0, "moderate": 8, "high": 14, "very high": 18, "extremely high": 20}
    score += bmi_scores.get(bmi_risk, 0)

    # BP contribution (0-25 points)
    bp_risk = assessment["components"].get("blood_pressure", {}).get("risk_level", "low")
    bp_scores = {"low": 0, "moderate": 8, "high": 15, "very high": 20, "critical": 25}
    score += bp_scores.get(bp_risk, 0)

    # Lab contribution (0-35 points)
    labs = assessment["components"].get("labs", {})
    lab_score = 0
    for test_name, result in labs.items():
        if isinstance(result, dict) and "risk_level" in result:
            risk = result["risk_level"]
            if risk == "critical":
                lab_score += 15
            elif risk in ("moderate", "high"):
                lab_score += 5
    score += min(lab_score, 35)

    # Medication interaction contribution (0-20 points)
    meds = assessment["components"].get("medications", {})
    interactions = meds.get("interactions", [])
    med_score = 0
    for i in interactions:
        if i.get("severity") == "major":
            med_score += 15
        elif i.get("severity") == "moderate":
            med_score += 8
        else:
            med_score += 3
    score += min(med_score, 20)

    return min(round(score, 1), 100.0)


def _risk_level_from_score(score: float) -> str:
    """Convert numeric risk score to risk level."""
    if score < 10:
        return "low"
    elif score < 25:
        return "moderate"
    elif score < 50:
        return "high"
    elif score < 75:
        return "very high"
    else:
        return "critical"


def metabolic_panel_assessment(
    glucose: float, hba1c: float, creatinine: float,
    potassium: float, alt: float,
    medications: list[str] = None
) -> dict:
    """
    Focused metabolic panel assessment.
    Calls: interpret_lab() for each value + check_medications() if provided.
    """
    results = {
        "assessment_type": "metabolic_panel",
        "labs": {},
        "findings": [],
        "recommendations": [],
    }

    # Interpret each lab
    lab_inputs = {
        "glucose": glucose,
        "hba1c": hba1c,
        "creatinine": creatinine,
        "potassium": potassium,
        "alt": alt,
    }

    for test, value in lab_inputs.items():
        result = interpret_lab(test, value)
        results["labs"][test] = result
        if "error" not in result and result["status"] != "normal":
            results["findings"].append(
                f"{test}: {value} {result['unit']} → {result['status']}"
            )

    # Clinical correlations
    glucose_result = results["labs"].get("glucose", {})
    hba1c_result = results["labs"].get("hba1c", {})

    if (glucose_result.get("status") in ("high", "critically high") and
            hba1c_result.get("status") in ("high", "critically high")):
        results["findings"].append(
            "Elevated glucose AND HbA1c — consistent with diabetes/poor glycemic control"
        )
        results["recommendations"].append("Evaluate diabetes management plan")

    creatinine_result = results["labs"].get("creatinine", {})
    potassium_result = results["labs"].get("potassium", {})

    if creatinine_result.get("status") in ("high", "critically high"):
        results["recommendations"].append("Evaluate renal function (eGFR calculation)")
        if potassium_result.get("status") in ("high", "critically high"):
            results["findings"].append(
                "Elevated creatinine AND potassium — renal impairment with hyperkalemia risk"
            )
            results["recommendations"].append("Urgent nephrology evaluation")

    alt_result = results["labs"].get("alt", {})
    if alt_result.get("status") in ("high", "critically high"):
        results["recommendations"].append("Evaluate liver function — consider hepatic panel")

    # Check medication interactions if provided
    if medications:
        med_result = check_medications(medications)
        results["medications"] = med_result
        if med_result["interactions"]:
            for interaction in med_result["interactions"]:
                results["findings"].append(
                    f"Drug interaction: {interaction['drug_a']}+{interaction['drug_b']} ({interaction['severity']})"
                )

    return results


# ============================================================================
# MCP Server Registration
# ============================================================================

if MCP_AVAILABLE:
    mcp = FastMCP("Clinical Assessment Server")

    @mcp.tool()
    def mcp_comprehensive_assessment(
        weight_kg: float, height_m: float,
        systolic: int, diastolic: int,
        labs: str, medications: str
    ) -> str:
        """Perform a comprehensive patient assessment combining BMI, blood
        pressure, lab results, and medication review. Labs should be JSON
        object mapping test names to values. Medications should be JSON array.
        Returns unified assessment with risk scoring."""
        labs_dict = json.loads(labs) if isinstance(labs, str) else labs
        meds_list = json.loads(medications) if isinstance(medications, str) else medications
        return json.dumps(comprehensive_assessment(
            weight_kg, height_m, systolic, diastolic, labs_dict, meds_list
        ), indent=2)

    @mcp.tool()
    def mcp_metabolic_panel(
        glucose: float, hba1c: float, creatinine: float,
        potassium: float, alt: float,
        medications: str = "[]"
    ) -> str:
        """Assess a metabolic panel with clinical correlations. Interprets
        glucose, HbA1c, creatinine, potassium, and ALT together. Identifies
        patterns like diabetic control, renal impairment, and liver issues."""
        meds = json.loads(medications) if isinstance(medications, str) else medications
        return json.dumps(metabolic_panel_assessment(
            glucose, hba1c, creatinine, potassium, alt, meds
        ), indent=2)


# ============================================================================
# Demo / Test
# ============================================================================

def main():
    """Demo tool composition with comprehensive assessments."""
    print("=" * 70)
    print("  Exercise 4: Tool Composition in MCP")
    print("  Building composite tools from base tools")
    print("=" * 70)

    # --- Composite Tool 1: Comprehensive Assessment ---
    print("\n  COMPOSITE TOOL 1: comprehensive_assessment")
    print("  Calls: calculate_bmi + assess_bp + interpret_lab + check_medications")
    print("  " + "─" * 55)

    # Patient scenario 1: Multiple risk factors
    print("\n  Patient A: 55yo with multiple risk factors")
    result = comprehensive_assessment(
        weight_kg=98.0, height_m=1.70,
        systolic=155, diastolic=98,
        labs={"glucose": 185, "hba1c": 8.2, "creatinine": 1.6,
              "potassium": 5.3, "ldl": 165},
        medications=["metformin", "lisinopril", "atorvastatin"]
    )
    print(f"  Risk Score: {result['overall_risk_score']}/100 "
          f"({result['overall_risk_level'].upper()})")
    print(f"  Risk Factors: {result['risk_factor_count']}")
    for finding in result["findings"]:
        print(f"    • {finding}")

    # Patient scenario 2: Healthy
    print("\n  Patient B: 30yo healthy individual")
    result = comprehensive_assessment(
        weight_kg=68.0, height_m=1.72,
        systolic=118, diastolic=76,
        labs={"glucose": 88, "hemoglobin": 14.5, "creatinine": 0.9},
        medications=[]
    )
    print(f"  Risk Score: {result['overall_risk_score']}/100 "
          f"({result['overall_risk_level'].upper()})")
    print(f"  Risk Factors: {result['risk_factor_count']}")
    if result["findings"]:
        for finding in result["findings"]:
            print(f"    • {finding}")
    else:
        print("    • No significant findings")

    # --- Composite Tool 2: Metabolic Panel Assessment ---
    print(f"\n\n  COMPOSITE TOOL 2: metabolic_panel_assessment")
    print("  Calls: interpret_lab (x5) + check_medications + clinical correlation")
    print("  " + "─" * 55)

    # Scenario: Poorly controlled diabetic with renal concerns
    print("\n  Scenario: Poorly controlled diabetic on multiple medications")
    result = metabolic_panel_assessment(
        glucose=210, hba1c=9.1, creatinine=1.8,
        potassium=5.6, alt=42,
        medications=["metformin", "lisinopril", "atorvastatin", "amlodipine"]
    )
    print("  Findings:")
    for finding in result["findings"]:
        print(f"    • {finding}")
    print("  Recommendations:")
    for rec in result["recommendations"]:
        print(f"    → {rec}")

    # Scenario: Normal metabolic panel
    print("\n  Scenario: Normal metabolic panel")
    result = metabolic_panel_assessment(
        glucose=92, hba1c=5.2, creatinine=0.9,
        potassium=4.2, alt=25,
        medications=["levothyroxine"]
    )
    print("  Findings:")
    if result["findings"]:
        for finding in result["findings"]:
            print(f"    • {finding}")
    else:
        print("    • All values within normal range")

    # --- Architecture diagram ---
    print(f"\n\n  TOOL COMPOSITION ARCHITECTURE")
    print("  " + "─" * 55)
    print("""
    comprehensive_assessment (composite)
    ├── calculate_bmi (base)           → BMI + category
    ├── assess_blood_pressure (base)   → BP category + risk
    ├── interpret_lab (base, x N)      → Lab status per test
    ├── check_medications (base)       → Interactions + review
    └── _calculate_risk_score          → Aggregate risk 0-100

    metabolic_panel_assessment (composite)
    ├── interpret_lab (base, x 5)      → Glucose, HbA1c, Cr, K, ALT
    ├── clinical correlation logic     → Pattern recognition
    └── check_medications (optional)   → Drug interactions
    """)

    print("=" * 70)
    print("  ✓ Tool composition demonstrated — base tools combined into")
    print("    comprehensive assessments with risk scoring")
    print("=" * 70)


if __name__ == "__main__":
    main()
