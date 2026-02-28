"""
Exercise 3: MCP Server Testing
================================

Skills practiced:
- Building test harnesses for MCP tool servers
- Defining test cases with expected inputs and outputs
- Validating schema compliance for tools
- Reporting test results with pass/fail summaries

Healthcare context:
Healthcare tools must be rigorously tested before deployment. A BMI calculator
that gives the wrong category or a blood pressure classifier that misses a
hypertensive crisis could lead to patient harm. This exercise builds a
systematic test framework that validates tool behavior, edge cases, and
schema compliance.

Usage:
    python exercise_3_server_testing.py
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Import the tools from exercise_2 (or define inline fallbacks)
# ---------------------------------------------------------------------------

try:
    from exercise_2_simple_server import (
        calculate_bmi,
        check_blood_pressure_category,
        interpret_heart_rate,
    )
except ImportError:
    # Inline fallbacks if exercise_2 is not available
    def calculate_bmi(weight_kg: float, height_m: float) -> dict:
        if weight_kg <= 0 or height_m <= 0:
            return {"error": "Weight and height must be positive numbers"}
        bmi = round(weight_kg / (height_m ** 2), 1)
        if bmi < 18.5:
            category = "Underweight"
        elif bmi < 25.0:
            category = "Normal"
        elif bmi < 30.0:
            category = "Overweight"
        else:
            category = "Obese Class I"
        return {"bmi": bmi, "category": category, "weight_kg": weight_kg, "height_m": height_m}

    def check_blood_pressure_category(systolic: int, diastolic: int) -> dict:
        if systolic <= 0 or diastolic <= 0:
            return {"error": "Blood pressure values must be positive"}
        if systolic < 120 and diastolic < 80:
            return {"category": "Normal", "stage": "Normal", "systolic": systolic, "diastolic": diastolic}
        elif systolic < 130:
            return {"category": "Elevated", "stage": "Elevated", "systolic": systolic, "diastolic": diastolic}
        elif systolic < 140 or diastolic < 90:
            return {"category": "High Blood Pressure", "stage": "Stage 1", "systolic": systolic, "diastolic": diastolic}
        elif systolic < 180 and diastolic < 120:
            return {"category": "High Blood Pressure", "stage": "Stage 2", "systolic": systolic, "diastolic": diastolic}
        else:
            return {"category": "Hypertensive Crisis", "stage": "Crisis", "systolic": systolic, "diastolic": diastolic}

    def interpret_heart_rate(bpm: int, age_years: int = 40,
                             is_resting: bool = True, is_athlete: bool = False) -> dict:
        if bpm <= 0:
            return {"error": "Heart rate must be positive"}
        if is_resting:
            if bpm < 60:
                category = "Normal (athlete)" if is_athlete and bpm >= 40 else "Bradycardia"
            elif bpm <= 100:
                category = "Normal"
            else:
                category = "Tachycardia"
        else:
            category = "Active"
        return {"bpm": bpm, "category": category}


# ---------------------------------------------------------------------------
# Tool schemas (what an MCP server would advertise)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = {
    "calculate_bmi": {
        "name": "calculate_bmi",
        "description": "Calculate Body Mass Index from weight and height",
        "inputSchema": {
            "type": "object",
            "properties": {
                "weight_kg": {"type": "number", "description": "Weight in kilograms"},
                "height_m": {"type": "number", "description": "Height in meters"},
            },
            "required": ["weight_kg", "height_m"]
        },
        "required_output_fields": ["bmi", "category"]
    },
    "check_blood_pressure_category": {
        "name": "check_blood_pressure_category",
        "description": "Classify blood pressure per AHA/ACC guidelines",
        "inputSchema": {
            "type": "object",
            "properties": {
                "systolic": {"type": "integer", "description": "Systolic BP in mmHg"},
                "diastolic": {"type": "integer", "description": "Diastolic BP in mmHg"},
            },
            "required": ["systolic", "diastolic"]
        },
        "required_output_fields": ["category", "stage"]
    },
    "interpret_heart_rate": {
        "name": "interpret_heart_rate",
        "description": "Interpret heart rate with clinical context",
        "inputSchema": {
            "type": "object",
            "properties": {
                "bpm": {"type": "integer", "description": "Beats per minute"},
                "age_years": {"type": "integer", "description": "Patient age"},
                "is_resting": {"type": "boolean", "description": "Resting measurement"},
                "is_athlete": {"type": "boolean", "description": "Trained athlete"},
            },
            "required": ["bpm"]
        },
        "required_output_fields": ["bpm", "category"]
    },
}


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

TEST_CASES = {
    "calculate_bmi": [
        {
            "name": "Normal weight",
            "args": {"weight_kg": 68.0, "height_m": 1.72},
            "expect": {"category": "Normal"},
            "expect_bmi_range": (18.5, 24.9),
        },
        {
            "name": "Overweight",
            "args": {"weight_kg": 85.0, "height_m": 1.75},
            "expect": {"category": "Overweight"},
            "expect_bmi_range": (25.0, 29.9),
        },
        {
            "name": "Obese",
            "args": {"weight_kg": 110.0, "height_m": 1.70},
            "expect_category_contains": "Obese",
            "expect_bmi_range": (30.0, 100.0),
        },
        {
            "name": "Underweight",
            "args": {"weight_kg": 45.0, "height_m": 1.70},
            "expect_category_contains": "Underweight",
            "expect_bmi_range": (0.0, 18.5),
        },
        {
            "name": "Invalid input (zero weight)",
            "args": {"weight_kg": 0, "height_m": 1.70},
            "expect_error": True,
        },
        {
            "name": "Invalid input (negative height)",
            "args": {"weight_kg": 70.0, "height_m": -1.0},
            "expect_error": True,
        },
    ],
    "check_blood_pressure_category": [
        {
            "name": "Normal BP",
            "args": {"systolic": 115, "diastolic": 75},
            "expect": {"category": "Normal"},
        },
        {
            "name": "Elevated BP",
            "args": {"systolic": 125, "diastolic": 78},
            "expect": {"category": "Elevated"},
        },
        {
            "name": "Stage 1 HTN",
            "args": {"systolic": 135, "diastolic": 85},
            "expect": {"category": "High Blood Pressure", "stage": "Stage 1"},
        },
        {
            "name": "Stage 2 HTN",
            "args": {"systolic": 155, "diastolic": 100},
            "expect": {"category": "High Blood Pressure", "stage": "Stage 2"},
        },
        {
            "name": "Hypertensive crisis",
            "args": {"systolic": 185, "diastolic": 125},
            "expect": {"category": "Hypertensive Crisis"},
        },
        {
            "name": "Invalid BP (zero)",
            "args": {"systolic": 0, "diastolic": 80},
            "expect_error": True,
        },
    ],
    "interpret_heart_rate": [
        {
            "name": "Normal resting HR",
            "args": {"bpm": 72, "age_years": 45, "is_resting": True},
            "expect": {"category": "Normal"},
        },
        {
            "name": "Bradycardia",
            "args": {"bpm": 48, "age_years": 70, "is_resting": True, "is_athlete": False},
            "expect": {"category": "Bradycardia"},
        },
        {
            "name": "Tachycardia",
            "args": {"bpm": 115, "age_years": 55, "is_resting": True},
            "expect": {"category": "Tachycardia"},
        },
        {
            "name": "Athlete normal bradycardia",
            "args": {"bpm": 52, "age_years": 28, "is_resting": True, "is_athlete": True},
            "expect_category_contains": "Normal",
        },
        {
            "name": "Invalid HR (zero)",
            "args": {"bpm": 0},
            "expect_error": True,
        },
    ],
}

# Tool function mapping
TOOL_FUNCTIONS = {
    "calculate_bmi": calculate_bmi,
    "check_blood_pressure_category": check_blood_pressure_category,
    "interpret_heart_rate": interpret_heart_rate,
}


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test_case(tool_name: str, test: dict) -> dict:
    """Run a single test case and return the result."""
    func = TOOL_FUNCTIONS[tool_name]
    result = {
        "name": test["name"],
        "tool": tool_name,
        "passed": False,
        "details": [],
    }

    try:
        output = func(**test["args"])

        # Check for expected error
        if test.get("expect_error"):
            if "error" in output:
                result["passed"] = True
                result["details"].append(f"Correctly returned error: {output['error']}")
            else:
                result["details"].append("Expected error but got success")
            return result

        # Should not have error
        if "error" in output:
            result["details"].append(f"Unexpected error: {output['error']}")
            return result

        all_checks_passed = True

        # Check exact field matches
        if "expect" in test:
            for field, expected_value in test["expect"].items():
                actual = output.get(field)
                if actual == expected_value:
                    result["details"].append(f"✓ {field} = '{actual}'")
                else:
                    result["details"].append(f"✗ {field}: expected '{expected_value}', got '{actual}'")
                    all_checks_passed = False

        # Check category contains
        if "expect_category_contains" in test:
            actual_cat = output.get("category", "")
            expected_substr = test["expect_category_contains"]
            if expected_substr.lower() in actual_cat.lower():
                result["details"].append(f"✓ category contains '{expected_substr}': '{actual_cat}'")
            else:
                result["details"].append(f"✗ category '{actual_cat}' does not contain '{expected_substr}'")
                all_checks_passed = False

        # Check BMI range
        if "expect_bmi_range" in test:
            bmi_val = output.get("bmi", 0)
            low, high = test["expect_bmi_range"]
            if low <= bmi_val <= high:
                result["details"].append(f"✓ BMI {bmi_val} in range [{low}, {high}]")
            else:
                result["details"].append(f"✗ BMI {bmi_val} NOT in range [{low}, {high}]")
                all_checks_passed = False

        result["passed"] = all_checks_passed

    except Exception as e:
        if test.get("expect_error"):
            result["passed"] = True
            result["details"].append(f"Correctly raised exception: {e}")
        else:
            result["details"].append(f"Unexpected exception: {e}")

    return result


def validate_schema_compliance(tool_name: str, test_args: dict) -> list[str]:
    """Validate that test arguments comply with the tool's schema."""
    issues = []
    schema_info = TOOL_SCHEMAS.get(tool_name, {})
    input_schema = schema_info.get("inputSchema", {})
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    # Check required fields
    for field in required:
        if field not in test_args:
            issues.append(f"Missing required field: {field}")

    # Check types
    type_map = {"number": (int, float), "integer": (int,), "string": (str,), "boolean": (bool,)}
    for field, value in test_args.items():
        if field in properties:
            expected_type = properties[field].get("type")
            if expected_type and expected_type in type_map:
                if not isinstance(value, type_map[expected_type]):
                    issues.append(f"Field '{field}': expected {expected_type}, "
                                  f"got {type(value).__name__}")

    return issues


def validate_output_fields(tool_name: str, output: dict) -> list[str]:
    """Validate that tool output contains required fields."""
    issues = []
    schema_info = TOOL_SCHEMAS.get(tool_name, {})
    required_fields = schema_info.get("required_output_fields", [])
    if "error" not in output:
        for field in required_fields:
            if field not in output:
                issues.append(f"Missing required output field: {field}")
    return issues


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def run_all_tests():
    """Run all test cases and report results."""
    print("=" * 70)
    print("  Exercise 3: MCP Server Testing")
    print("  Systematic test harness for healthcare MCP tools")
    print("=" * 70)

    total = 0
    passed = 0
    failed = 0
    results_by_tool = {}

    for tool_name, tests in TEST_CASES.items():
        print(f"\n  Testing: {tool_name}")
        print(f"  {'─' * 55}")
        tool_results = []

        for test in tests:
            total += 1

            # Schema compliance check
            schema_issues = validate_schema_compliance(tool_name, test["args"])
            if schema_issues and not test.get("expect_error"):
                print(f"    ⚠ Schema warning for '{test['name']}': {schema_issues}")

            # Run the test
            result = run_test_case(tool_name, test)
            tool_results.append(result)

            # Output field validation
            if not test.get("expect_error"):
                func = TOOL_FUNCTIONS[tool_name]
                try:
                    output = func(**test["args"])
                    output_issues = validate_output_fields(tool_name, output)
                    if output_issues:
                        result["details"].append(f"⚠ Output schema: {output_issues}")
                except Exception:
                    pass

            status = "PASS" if result["passed"] else "FAIL"
            if result["passed"]:
                passed += 1
            else:
                failed += 1

            print(f"    [{status}] {test['name']}")
            for detail in result["details"]:
                print(f"           {detail}")

        results_by_tool[tool_name] = tool_results

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"  Test Results Summary")
    print(f"{'=' * 70}")
    print(f"  Total:  {total}")
    print(f"  Passed: {passed} ✓")
    print(f"  Failed: {failed} ✗")
    print(f"  Rate:   {(passed / total * 100) if total > 0 else 0:.1f}%")
    print()

    for tool_name, results in results_by_tool.items():
        tool_passed = sum(1 for r in results if r["passed"])
        tool_total = len(results)
        status = "✓" if tool_passed == tool_total else "✗"
        print(f"  {status} {tool_name}: {tool_passed}/{tool_total} passed")

    print(f"\n{'=' * 70}")
    if failed == 0:
        print("  ✓ All tests passed! Tools are ready for MCP deployment.")
    else:
        print(f"  ✗ {failed} test(s) failed. Review and fix before deployment.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_all_tests()
