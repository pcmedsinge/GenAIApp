"""
Exercise 2: Laboratory MCP Server
====================================

Skills practiced:
- Building a domain-specific MCP server for laboratory services
- Interpreting lab values against reference ranges with clinical context
- Identifying critical values that require immediate notification
- Trending lab values over time to detect patterns

Healthcare context:
Laboratory results are central to clinical decision-making. AI agents need
reliable tools to interpret lab values, flag critical results, provide
reference ranges, and identify trends. This server supports 10+ common lab
tests with full reference ranges, critical thresholds, and clinical
interpretation guidance.

Usage:
    python exercise_2_lab_server.py
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
# Laboratory Reference Database (10+ tests)
# ============================================================================

LAB_TESTS = {
    "hemoglobin": {
        "name": "Hemoglobin",
        "abbreviation": "Hgb",
        "unit": "g/dL",
        "category": "hematology",
        "reference_range": {"male": (13.5, 17.5), "female": (12.0, 16.0)},
        "critical_low": 7.0,
        "critical_high": 20.0,
        "clinical_significance": {
            "low": "Anemia — evaluate for blood loss, iron/B12/folate deficiency, chronic disease",
            "high": "Polycythemia — evaluate for dehydration, COPD, polycythemia vera",
            "critical_low": "Severe anemia — consider transfusion, hemodynamic assessment",
            "critical_high": "Hyperviscosity risk — evaluate for polycythemia vera",
        },
    },
    "glucose": {
        "name": "Glucose (Fasting)",
        "abbreviation": "Glu",
        "unit": "mg/dL",
        "category": "chemistry",
        "reference_range": {"general": (70, 100)},
        "critical_low": 40,
        "critical_high": 500,
        "clinical_significance": {
            "low": "Hypoglycemia — assess medications, diet, adrenal function",
            "high": "Hyperglycemia — evaluate for diabetes, stress response, medication effect",
            "critical_low": "Severe hypoglycemia — immediate glucose replacement needed",
            "critical_high": "Diabetic emergency — evaluate for DKA or HHS",
        },
    },
    "potassium": {
        "name": "Potassium",
        "abbreviation": "K+",
        "unit": "mEq/L",
        "category": "chemistry",
        "reference_range": {"general": (3.5, 5.0)},
        "critical_low": 2.5,
        "critical_high": 6.5,
        "clinical_significance": {
            "low": "Hypokalemia — cardiac arrhythmia risk, check diuretics/GI losses",
            "high": "Hyperkalemia — cardiac risk, check renal function, medications (ACEi, K-sparing diuretics)",
            "critical_low": "Severe hypokalemia — cardiac arrest risk, immediate replacement",
            "critical_high": "Severe hyperkalemia — cardiac arrest risk, immediate treatment (calcium, insulin/glucose, kayexalate)",
        },
    },
    "sodium": {
        "name": "Sodium",
        "abbreviation": "Na+",
        "unit": "mEq/L",
        "category": "chemistry",
        "reference_range": {"general": (136, 145)},
        "critical_low": 120,
        "critical_high": 160,
        "clinical_significance": {
            "low": "Hyponatremia — assess volume status, SIADH, medications",
            "high": "Hypernatremia — assess dehydration, diabetes insipidus, Na excess",
            "critical_low": "Severe hyponatremia — seizure risk, careful correction needed",
            "critical_high": "Severe hypernatremia — neurological emergency",
        },
    },
    "creatinine": {
        "name": "Creatinine",
        "abbreviation": "Cr",
        "unit": "mg/dL",
        "category": "chemistry",
        "reference_range": {"male": (0.7, 1.3), "female": (0.6, 1.1)},
        "critical_low": 0.4,
        "critical_high": 10.0,
        "clinical_significance": {
            "low": "Low muscle mass or liver disease (often clinically insignificant)",
            "high": "Renal impairment — calculate eGFR, evaluate AKI vs CKD",
            "critical_low": "Unusually low — verify specimen",
            "critical_high": "Severe renal failure — nephrology consultation, consider dialysis",
        },
    },
    "wbc": {
        "name": "White Blood Cell Count",
        "abbreviation": "WBC",
        "unit": "K/uL",
        "category": "hematology",
        "reference_range": {"general": (4.5, 11.0)},
        "critical_low": 2.0,
        "critical_high": 30.0,
        "clinical_significance": {
            "low": "Leukopenia — infection risk, evaluate medications, bone marrow",
            "high": "Leukocytosis — infection, inflammation, stress, leukemia",
            "critical_low": "Severe leukopenia — neutropenic precautions, hematology consult",
            "critical_high": "Marked leukocytosis — evaluate for leukemia, severe infection",
        },
    },
    "platelets": {
        "name": "Platelet Count",
        "abbreviation": "Plt",
        "unit": "K/uL",
        "category": "hematology",
        "reference_range": {"general": (150, 400)},
        "critical_low": 50,
        "critical_high": 1000,
        "clinical_significance": {
            "low": "Thrombocytopenia — bleeding risk, evaluate HIT, ITP, medications",
            "high": "Thrombocytosis — clotting risk, evaluate reactive vs essential",
            "critical_low": "Severe thrombocytopenia — spontaneous bleeding risk, transfuse if active bleeding",
            "critical_high": "Extreme thrombocytosis — hematology evaluation",
        },
    },
    "alt": {
        "name": "Alanine Aminotransferase",
        "abbreviation": "ALT",
        "unit": "U/L",
        "category": "hepatic",
        "reference_range": {"general": (7, 56)},
        "critical_low": 0,
        "critical_high": 1000,
        "clinical_significance": {
            "low": "Usually not clinically significant",
            "high": "Hepatocellular injury — medications, hepatitis, fatty liver",
            "critical_low": "N/A",
            "critical_high": "Severe hepatic injury — urgent evaluation for acute liver failure",
        },
    },
    "hba1c": {
        "name": "Hemoglobin A1c",
        "abbreviation": "HbA1c",
        "unit": "%",
        "category": "endocrine",
        "reference_range": {"general": (4.0, 5.6)},
        "critical_low": 3.0,
        "critical_high": 15.0,
        "clinical_significance": {
            "low": "Hypoglycemic tendency or hemolytic conditions affecting assay",
            "high": "Prediabetes (5.7-6.4%), Diabetes (≥6.5%) — assess glycemic control",
            "critical_low": "Assay interference or severe hypoglycemia",
            "critical_high": "Very poor glycemic control — DKA risk, urgent management",
        },
    },
    "tsh": {
        "name": "Thyroid Stimulating Hormone",
        "abbreviation": "TSH",
        "unit": "mIU/L",
        "category": "endocrine",
        "reference_range": {"general": (0.4, 4.0)},
        "critical_low": 0.01,
        "critical_high": 100.0,
        "clinical_significance": {
            "low": "Hyperthyroidism or over-replacement with levothyroxine",
            "high": "Hypothyroidism — initiate or adjust thyroid hormone replacement",
            "critical_low": "Thyroid storm risk — evaluate for Graves' disease, toxic nodule",
            "critical_high": "Severe hypothyroidism — evaluate for myxedema coma",
        },
    },
    "troponin_i": {
        "name": "Troponin I (High Sensitivity)",
        "abbreviation": "hs-TnI",
        "unit": "ng/L",
        "category": "cardiac",
        "reference_range": {"general": (0, 14)},
        "critical_low": 0,
        "critical_high": 100,
        "clinical_significance": {
            "low": "Normal — no evidence of myocardial injury",
            "high": "Myocardial injury — evaluate for MI, myocarditis, PE, sepsis, renal failure",
            "critical_low": "N/A",
            "critical_high": "Acute myocardial injury likely — emergent cardiology evaluation",
        },
    },
    "inr": {
        "name": "International Normalized Ratio",
        "abbreviation": "INR",
        "unit": "",
        "category": "coagulation",
        "reference_range": {"general": (0.8, 1.1)},
        "critical_low": 0.5,
        "critical_high": 5.0,
        "clinical_significance": {
            "low": "Hypercoagulable state or subtherapeutic anticoagulation",
            "high": "Elevated bleeding risk — review warfarin dose, vitamin K status",
            "critical_low": "Unusual — verify specimen",
            "critical_high": "High bleeding risk — consider vitamin K, FFP, hold warfarin",
        },
    },
}


# ============================================================================
# Tool implementations
# ============================================================================

def interpret_lab_value(test_name: str, value: float,
                        sex: str = "general") -> dict:
    """Interpret a lab value against reference and critical ranges."""
    key = test_name.lower().strip().replace(" ", "_")
    if key not in LAB_TESTS:
        return {"error": f"Lab test '{test_name}' not found",
                "available": list(LAB_TESTS.keys())}

    lab = LAB_TESTS[key]
    ref_key = sex.lower() if sex.lower() in lab["reference_range"] else "general"
    if ref_key not in lab["reference_range"]:
        ref_key = list(lab["reference_range"].keys())[0]

    low, high = lab["reference_range"][ref_key]

    if value <= lab["critical_low"]:
        status = "critically low"
        significance = lab["clinical_significance"]["critical_low"]
        urgency = "critical"
    elif value < low:
        status = "low"
        significance = lab["clinical_significance"]["low"]
        urgency = "abnormal"
    elif value <= high:
        status = "normal"
        significance = "Within normal reference range"
        urgency = "normal"
    elif value >= lab["critical_high"]:
        status = "critically high"
        significance = lab["clinical_significance"]["critical_high"]
        urgency = "critical"
    else:
        status = "high"
        significance = lab["clinical_significance"]["high"]
        urgency = "abnormal"

    return {
        "test": lab["name"],
        "abbreviation": lab["abbreviation"],
        "value": value,
        "unit": lab["unit"],
        "status": status,
        "urgency": urgency,
        "reference_range": f"{low}-{high} {lab['unit']}",
        "clinical_significance": significance,
        "category": lab["category"],
    }


def check_critical_value(test_name: str, value: float) -> dict:
    """Determine if a lab result is critically abnormal."""
    key = test_name.lower().strip().replace(" ", "_")
    if key not in LAB_TESTS:
        return {"error": f"Lab test '{test_name}' not found"}

    lab = LAB_TESTS[key]
    is_critical = value <= lab["critical_low"] or value >= lab["critical_high"]

    result = {
        "test": lab["name"],
        "value": value,
        "unit": lab["unit"],
        "is_critical": is_critical,
        "critical_low_threshold": lab["critical_low"],
        "critical_high_threshold": lab["critical_high"],
    }

    if is_critical:
        if value <= lab["critical_low"]:
            result["direction"] = "critically low"
            result["clinical_action"] = lab["clinical_significance"]["critical_low"]
        else:
            result["direction"] = "critically high"
            result["clinical_action"] = lab["clinical_significance"]["critical_high"]
        result["notification_required"] = True
        result["notification_message"] = (
            f"CRITICAL VALUE: {lab['name']} = {value} {lab['unit']} "
            f"({result['direction']}). Immediate clinical action required."
        )
    else:
        result["notification_required"] = False

    return result


def get_reference_range(test_name: str, sex: str = "general") -> dict:
    """Get the reference range for a lab test."""
    key = test_name.lower().strip().replace(" ", "_")
    if key not in LAB_TESTS:
        return {"error": f"Lab test '{test_name}' not found",
                "available": list(LAB_TESTS.keys())}

    lab = LAB_TESTS[key]
    ref_key = sex.lower() if sex.lower() in lab["reference_range"] else "general"
    if ref_key not in lab["reference_range"]:
        ref_key = list(lab["reference_range"].keys())[0]

    low, high = lab["reference_range"][ref_key]

    ranges = {"normal_low": low, "normal_high": high}
    if "male" in lab["reference_range"] and "female" in lab["reference_range"]:
        m_low, m_high = lab["reference_range"]["male"]
        f_low, f_high = lab["reference_range"]["female"]
        ranges["male_range"] = f"{m_low}-{m_high} {lab['unit']}"
        ranges["female_range"] = f"{f_low}-{f_high} {lab['unit']}"

    return {
        "test": lab["name"],
        "abbreviation": lab["abbreviation"],
        "unit": lab["unit"],
        "reference_range": f"{low}-{high} {lab['unit']}",
        "critical_low": lab["critical_low"],
        "critical_high": lab["critical_high"],
        "category": lab["category"],
        **ranges,
    }


def trend_lab_values(test_name: str, values: list[float],
                     dates: list[str] = None) -> dict:
    """Analyze a trend across multiple lab values over time."""
    key = test_name.lower().strip().replace(" ", "_")
    if key not in LAB_TESTS:
        return {"error": f"Lab test '{test_name}' not found"}

    if len(values) < 2:
        return {"error": "Need at least 2 values to determine a trend"}

    lab = LAB_TESTS[key]
    ref = lab["reference_range"]
    ref_key = list(ref.keys())[0]
    low, high = ref[ref_key]

    # Calculate trend
    first = values[0]
    last = values[-1]
    change = last - first
    pct_change = round((change / first) * 100, 1) if first != 0 else 0

    if abs(pct_change) < 5:
        direction = "stable"
    elif change > 0:
        direction = "increasing"
    else:
        direction = "decreasing"

    # Classify each value
    statuses = []
    for v in values:
        if v <= lab["critical_low"]:
            statuses.append("critical_low")
        elif v < low:
            statuses.append("low")
        elif v <= high:
            statuses.append("normal")
        elif v >= lab["critical_high"]:
            statuses.append("critical_high")
        else:
            statuses.append("high")

    # Clinical interpretation
    moving_out_of_range = statuses[0] == "normal" and statuses[-1] in ("low", "high")
    moving_into_range = statuses[0] in ("low", "high") and statuses[-1] == "normal"
    any_critical = any(s.startswith("critical") for s in statuses)

    if moving_into_range:
        interpretation = "Improving — values trending toward normal range"
    elif moving_out_of_range:
        interpretation = "Worsening — values trending away from normal range"
    elif any_critical:
        interpretation = "Contains critical values — urgent review needed"
    elif direction == "stable" and statuses[-1] == "normal":
        interpretation = "Stable within normal range"
    elif direction == "stable":
        interpretation = f"Stable but {statuses[-1]} — continued monitoring advised"
    else:
        interpretation = f"Values are {direction} ({pct_change:+.1f}%) — clinical correlation needed"

    data_points = []
    for i, v in enumerate(values):
        point = {"value": v, "status": statuses[i]}
        if dates and i < len(dates):
            point["date"] = dates[i]
        data_points.append(point)

    return {
        "test": lab["name"],
        "unit": lab["unit"],
        "data_points": data_points,
        "trend_direction": direction,
        "change": round(change, 2),
        "percent_change": pct_change,
        "interpretation": interpretation,
        "reference_range": f"{low}-{high} {lab['unit']}",
    }


# ============================================================================
# MCP Server Registration
# ============================================================================

if MCP_AVAILABLE:
    mcp = FastMCP("Laboratory Server")

    @mcp.tool()
    def mcp_interpret_lab_value(test_name: str, value: float,
                                 sex: str = "general") -> str:
        """Interpret a laboratory test result against reference ranges.
        Returns status (normal/low/high/critical), clinical significance,
        and urgency. Use when reviewing lab results or explaining to patients."""
        return json.dumps(interpret_lab_value(test_name, value, sex))

    @mcp.tool()
    def mcp_check_critical_value(test_name: str, value: float) -> str:
        """Determine if a lab result is critically abnormal and requires
        immediate notification. Returns critical status and required actions.
        Use to flag dangerous lab values for urgent clinical attention."""
        return json.dumps(check_critical_value(test_name, value))

    @mcp.tool()
    def mcp_get_reference_range(test_name: str, sex: str = "general") -> str:
        """Get the reference range for a laboratory test. Supports sex-specific
        ranges where applicable. Use when a clinician or patient asks about
        normal values for a specific test."""
        return json.dumps(get_reference_range(test_name, sex))

    @mcp.tool()
    def mcp_trend_lab_values(test_name: str, values: list[float],
                              dates: list[str] = None) -> str:
        """Analyze trends across multiple lab values over time. Returns trend
        direction, percent change, and clinical interpretation. Use when
        reviewing a patient's lab history or monitoring treatment response."""
        return json.dumps(trend_lab_values(test_name, values, dates))


# ============================================================================
# Demo / Test
# ============================================================================

def main():
    """Demo all laboratory server tools."""
    print("=" * 70)
    print("  Exercise 2: Laboratory MCP Server")
    print(f"  Database: {len(LAB_TESTS)} lab tests with reference and critical ranges")
    print("=" * 70)

    if MCP_AVAILABLE:
        print("  ✓ MCP server registered as 'Laboratory Server' with 4 tools\n")
    else:
        print("  ⚠ MCP SDK not installed — running tools in standalone mode\n")

    # Tool 1: Interpret
    print("  TOOL 1: interpret_lab_value")
    print("  " + "─" * 55)
    interpret_tests = [
        ("hemoglobin", 14.5, "male"),
        ("hemoglobin", 10.2, "female"),
        ("glucose", 145, "general"),
        ("potassium", 6.8, "general"),
        ("tsh", 0.02, "general"),
        ("troponin_i", 85, "general"),
    ]
    for test, value, sex in interpret_tests:
        result = interpret_lab_value(test, value, sex)
        print(f"  {test} = {value} {result.get('unit', '')} "
              f"→ [{result.get('urgency', '?').upper()}] {result.get('status', '?')}")

    # Tool 2: Critical values
    print(f"\n  TOOL 2: check_critical_value")
    print("  " + "─" * 55)
    critical_tests = [
        ("hemoglobin", 6.5),
        ("potassium", 6.8),
        ("glucose", 35),
        ("sodium", 125),
        ("platelets", 180),
    ]
    for test, value in critical_tests:
        result = check_critical_value(test, value)
        is_crit = result.get("is_critical", False)
        marker = "🚨 CRITICAL" if is_crit else "✓ Non-critical"
        print(f"  {test} = {value} → {marker}")
        if is_crit:
            print(f"    → {result.get('notification_message', '')[:70]}...")

    # Tool 3: Reference ranges
    print(f"\n  TOOL 3: get_reference_range")
    print("  " + "─" * 55)
    for test in ["hemoglobin", "glucose", "creatinine", "inr"]:
        result = get_reference_range(test)
        print(f"  {test}: {result.get('reference_range', '?')} "
              f"(critical: <{result.get('critical_low', '?')} "
              f"or >{result.get('critical_high', '?')})")

    # Tool 4: Trends
    print(f"\n  TOOL 4: trend_lab_values")
    print("  " + "─" * 55)
    trend_tests = [
        ("creatinine", [0.9, 1.1, 1.4, 1.8, 2.3],
         ["2025-01", "2025-03", "2025-06", "2025-09", "2025-12"]),
        ("hba1c", [8.5, 7.8, 7.2, 6.8],
         ["2025-01", "2025-04", "2025-07", "2025-10"]),
        ("hemoglobin", [13.2, 13.0, 13.1, 13.3],
         ["2025-03", "2025-06", "2025-09", "2025-12"]),
    ]
    for test, values, dates in trend_tests:
        result = trend_lab_values(test, values, dates)
        print(f"  {test}: {values}")
        print(f"    → Trend: {result.get('trend_direction', '?')} "
              f"({result.get('percent_change', 0):+.1f}%)")
        print(f"    → {result.get('interpretation', '?')}")

    print(f"\n{'=' * 70}")
    print("  ✓ All laboratory server tools tested")
    print("=" * 70)


if __name__ == "__main__":
    main()
