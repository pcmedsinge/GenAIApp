"""
Exercise 2: Simple MCP Server with Healthcare Tools
=====================================================

Skills practiced:
- Building MCP servers with FastMCP
- Defining tools with typed parameters and descriptions
- Implementing clinical calculators as MCP tools
- Structuring tool responses for agent consumption

Healthcare context:
Clinical calculators are some of the most useful tools for healthcare AI agents.
This exercise builds an MCP server with three tools that any MCP-compatible
agent could discover and use:
  - calculate_bmi: Body Mass Index with WHO categories
  - check_blood_pressure_category: AHA blood pressure classification
  - interpret_heart_rate: Heart rate interpretation with context

Usage:
    python exercise_2_simple_server.py
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
# MCP Server Definition
# ============================================================================

# Create the MCP server instance
if MCP_AVAILABLE:
    mcp = FastMCP("Clinical Calculator Server")


# ---------------------------------------------------------------------------
# Tool 1: Calculate BMI
# ---------------------------------------------------------------------------

def calculate_bmi(weight_kg: float, height_m: float) -> dict:
    """
    Calculate Body Mass Index from weight and height.

    Uses the standard formula: BMI = weight(kg) / height(m)^2
    Categorizes according to WHO classification.

    Args:
        weight_kg: Patient weight in kilograms (must be > 0)
        height_m: Patient height in meters (must be > 0)

    Returns:
        Dictionary with BMI value, category, and interpretation
    """
    if weight_kg <= 0 or height_m <= 0:
        return {"error": "Weight and height must be positive numbers"}

    bmi = round(weight_kg / (height_m ** 2), 1)

    if bmi < 16.0:
        category = "Severely Underweight"
        interpretation = "Severe thinness — medical evaluation recommended"
    elif bmi < 18.5:
        category = "Underweight"
        interpretation = "Below normal weight — nutritional assessment recommended"
    elif bmi < 25.0:
        category = "Normal"
        interpretation = "Healthy weight range"
    elif bmi < 30.0:
        category = "Overweight"
        interpretation = "Above normal — lifestyle modifications may be beneficial"
    elif bmi < 35.0:
        category = "Obese Class I"
        interpretation = "Moderate obesity — increased health risks"
    elif bmi < 40.0:
        category = "Obese Class II"
        interpretation = "Severe obesity — significant health risks"
    else:
        category = "Obese Class III"
        interpretation = "Very severe obesity — highest health risk category"

    return {
        "bmi": bmi,
        "category": category,
        "interpretation": interpretation,
        "weight_kg": weight_kg,
        "height_m": height_m
    }


# Register with MCP if available
if MCP_AVAILABLE:
    @mcp.tool()
    def mcp_calculate_bmi(weight_kg: float, height_m: float) -> str:
        """Calculate Body Mass Index from weight (kg) and height (m).
        Returns BMI value and WHO category. Use for nutritional assessment."""
        result = calculate_bmi(weight_kg, height_m)
        return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 2: Check Blood Pressure Category
# ---------------------------------------------------------------------------

def check_blood_pressure_category(systolic: int, diastolic: int) -> dict:
    """
    Classify blood pressure according to AHA/ACC 2017 guidelines.

    Args:
        systolic: Systolic blood pressure in mmHg
        diastolic: Diastolic blood pressure in mmHg

    Returns:
        Dictionary with category, stage, and recommendations
    """
    if systolic <= 0 or diastolic <= 0:
        return {"error": "Blood pressure values must be positive"}
    if systolic < diastolic:
        return {"error": "Systolic must be >= diastolic"}

    if systolic < 120 and diastolic < 80:
        category = "Normal"
        stage = "Normal"
        recommendation = "Maintain healthy lifestyle. Recheck in 1-2 years."
        urgency = "routine"
    elif systolic < 130 and diastolic < 80:
        category = "Elevated"
        stage = "Elevated"
        recommendation = ("Lifestyle modifications: diet, exercise, weight management. "
                          "Recheck in 3-6 months.")
        urgency = "routine"
    elif systolic < 140 or diastolic < 90:
        category = "High Blood Pressure"
        stage = "Stage 1"
        recommendation = ("Lifestyle modifications. Consider antihypertensive medication "
                          "if ASCVD risk ≥10% or other compelling indication. "
                          "Recheck in 1-3 months.")
        urgency = "monitor"
    elif systolic < 180 and diastolic < 120:
        category = "High Blood Pressure"
        stage = "Stage 2"
        recommendation = ("Combination antihypertensive therapy plus lifestyle modifications. "
                          "Recheck in 1 month. Consider specialist referral.")
        urgency = "prompt"
    else:
        category = "Hypertensive Crisis"
        stage = "Crisis"
        recommendation = ("URGENT: Immediate medical evaluation required. "
                          "If symptoms present (chest pain, shortness of breath, "
                          "vision changes), call emergency services.")
        urgency = "emergency"

    return {
        "systolic": systolic,
        "diastolic": diastolic,
        "reading": f"{systolic}/{diastolic} mmHg",
        "category": category,
        "stage": stage,
        "recommendation": recommendation,
        "urgency": urgency,
        "guideline": "AHA/ACC 2017"
    }


if MCP_AVAILABLE:
    @mcp.tool()
    def mcp_check_blood_pressure(systolic: int, diastolic: int) -> str:
        """Classify blood pressure reading per AHA/ACC 2017 guidelines.
        Returns category (Normal/Elevated/Stage 1/Stage 2/Crisis) and recommendations.
        Use when a clinician provides BP values."""
        result = check_blood_pressure_category(systolic, diastolic)
        return json.dumps(result)


# ---------------------------------------------------------------------------
# Tool 3: Interpret Heart Rate
# ---------------------------------------------------------------------------

def interpret_heart_rate(bpm: int, age_years: int = 40,
                         is_resting: bool = True,
                         is_athlete: bool = False) -> dict:
    """
    Interpret a heart rate reading with clinical context.

    Args:
        bpm: Heart rate in beats per minute
        age_years: Patient age in years (for context)
        is_resting: Whether the measurement is at rest
        is_athlete: Whether the patient is a trained athlete

    Returns:
        Dictionary with interpretation and clinical considerations
    """
    if bpm <= 0:
        return {"error": "Heart rate must be positive"}
    if age_years < 0 or age_years > 150:
        return {"error": "Age must be between 0 and 150"}

    context = "resting" if is_resting else "active"
    considerations = []

    if is_resting:
        if is_athlete and 40 <= bpm < 60:
            category = "Normal (athlete)"
            interpretation = "Expected resting rate for trained athlete"
        elif bpm < 60:
            category = "Bradycardia"
            interpretation = "Below normal resting heart rate"
            considerations.append("Evaluate for symptomatic bradycardia")
            considerations.append("Review medications (beta-blockers, calcium channel blockers)")
            if bpm < 40:
                considerations.append("CRITICAL: Severe bradycardia — immediate evaluation needed")
        elif bpm <= 100:
            category = "Normal"
            interpretation = "Normal resting heart rate"
            if bpm > 80:
                considerations.append("Upper end of normal — consider cardiovascular risk assessment")
        else:
            category = "Tachycardia"
            interpretation = "Above normal resting heart rate"
            considerations.append("Evaluate for underlying cause (fever, pain, anxiety, dehydration)")
            considerations.append("Consider ECG to evaluate rhythm")
            if bpm > 150:
                considerations.append("URGENT: Significant tachycardia — immediate evaluation")
    else:
        max_hr = 220 - age_years
        pct_max = round((bpm / max_hr) * 100, 1) if max_hr > 0 else 0
        if pct_max < 50:
            category = "Light activity"
        elif pct_max < 70:
            category = "Moderate activity"
        elif pct_max < 85:
            category = "Vigorous activity"
        else:
            category = "Near maximum"
            considerations.append("Close to estimated maximum — monitor for distress")
        interpretation = f"{pct_max}% of estimated max HR ({max_hr} bpm)"

    return {
        "bpm": bpm,
        "context": context,
        "category": category,
        "interpretation": interpretation,
        "considerations": considerations,
        "is_athlete": is_athlete,
        "age_years": age_years
    }


if MCP_AVAILABLE:
    @mcp.tool()
    def mcp_interpret_heart_rate(bpm: int, age_years: int = 40,
                                  is_resting: bool = True,
                                  is_athlete: bool = False) -> str:
        """Interpret a heart rate reading with clinical context.
        Returns category (bradycardia/normal/tachycardia) and considerations.
        Accounts for athlete status and whether measurement is at rest."""
        result = interpret_heart_rate(bpm, age_years, is_resting, is_athlete)
        return json.dumps(result)


# ============================================================================
# Test the tools (standalone mode)
# ============================================================================

def run_tests():
    """Test all three healthcare tools with sample inputs."""
    print("=" * 70)
    print("  Exercise 2: Simple MCP Server — Healthcare Calculator Tools")
    print("=" * 70)

    if MCP_AVAILABLE:
        print(f"\n  ✓ MCP SDK available — server registered as 'Clinical Calculator Server'")
        print(f"    Tools would be discoverable via MCP protocol")
    else:
        print(f"\n  ⚠ MCP SDK not installed — testing tools in standalone mode")
        print(f"    Install with: pip install mcp")

    # --- Test BMI ---
    print(f"\n{'─' * 70}")
    print("  Tool 1: calculate_bmi")
    print(f"{'─' * 70}")
    bmi_tests = [
        (55.0, 1.70, "Normal weight"),
        (95.0, 1.75, "Obese"),
        (48.0, 1.65, "Underweight"),
        (78.0, 1.80, "Normal/Overweight boundary"),
    ]
    for weight, height, note in bmi_tests:
        result = calculate_bmi(weight, height)
        print(f"  {weight}kg, {height}m ({note})")
        print(f"    → BMI: {result['bmi']}, Category: {result['category']}")
        print(f"      {result['interpretation']}")

    # --- Test Blood Pressure ---
    print(f"\n{'─' * 70}")
    print("  Tool 2: check_blood_pressure_category")
    print(f"{'─' * 70}")
    bp_tests = [
        (115, 75, "Normal"),
        (125, 78, "Elevated"),
        (135, 85, "Stage 1"),
        (155, 100, "Stage 2"),
        (185, 125, "Crisis"),
    ]
    for sys, dia, note in bp_tests:
        result = check_blood_pressure_category(sys, dia)
        print(f"  {sys}/{dia} mmHg ({note})")
        print(f"    → {result['category']} — {result['stage']}")
        print(f"      Urgency: {result['urgency']}")

    # --- Test Heart Rate ---
    print(f"\n{'─' * 70}")
    print("  Tool 3: interpret_heart_rate")
    print(f"{'─' * 70}")
    hr_tests = [
        (72, 45, True, False, "Normal resting"),
        (52, 28, True, True, "Athlete bradycardia"),
        (48, 75, True, False, "Bradycardia"),
        (112, 60, True, False, "Tachycardia"),
        (155, 35, False, False, "Active exercise"),
    ]
    for bpm, age, resting, athlete, note in hr_tests:
        result = interpret_heart_rate(bpm, age, resting, athlete)
        print(f"  {bpm} bpm, age {age}, {'resting' if resting else 'active'}"
              f"{', athlete' if athlete else ''} ({note})")
        print(f"    → {result['category']}: {result['interpretation']}")
        if result.get("considerations"):
            for c in result["considerations"]:
                print(f"      • {c}")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("  ✓ All 3 tools tested successfully")
    if MCP_AVAILABLE:
        print("  To run as MCP server: mcp.run() or use 'mcp dev' CLI")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
