"""
Exercise 3: Input Validation for MCP Tools
============================================

Skills practiced:
- Implementing comprehensive input validation for MCP tools
- Validating numeric ranges, enums, required fields, and types
- Providing helpful error messages that guide the agent/user
- Type coercion for common input format variations

Healthcare context:
Clinical tools must validate inputs rigorously. A weight of -50kg, a blood
pressure of 999/999, or a missing required field could produce misleading
results. In healthcare, bad outputs from a calculator or lookup tool could
inform clinical decisions. This exercise builds a validation framework that
catches bad inputs before they reach tool logic.

Usage:
    python exercise_3_input_validation.py
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
# Validation framework
# ============================================================================

class ValidationError(Exception):
    """Raised when input validation fails."""
    def __init__(self, field: str, message: str, received=None):
        self.field = field
        self.message = message
        self.received = received
        super().__init__(f"{field}: {message}")


class Validator:
    """Chainable input validator for MCP tool parameters."""

    def __init__(self):
        self.errors: list[dict] = []

    def require(self, value, field_name: str) -> "Validator":
        """Check that a required field is present and not None/empty."""
        if value is None or (isinstance(value, str) and value.strip() == ""):
            self.errors.append({
                "field": field_name,
                "issue": "required field is missing or empty",
                "received": value,
            })
        return self

    def numeric_range(self, value, field_name: str,
                      min_val: float = None, max_val: float = None) -> "Validator":
        """Validate a numeric value is within an allowed range."""
        if value is None:
            return self

        # Type coercion
        try:
            numeric_val = float(value)
        except (TypeError, ValueError):
            self.errors.append({
                "field": field_name,
                "issue": f"must be a number, got {type(value).__name__}",
                "received": value,
            })
            return self

        if min_val is not None and numeric_val < min_val:
            self.errors.append({
                "field": field_name,
                "issue": f"must be >= {min_val}",
                "received": numeric_val,
            })
        if max_val is not None and numeric_val > max_val:
            self.errors.append({
                "field": field_name,
                "issue": f"must be <= {max_val}",
                "received": numeric_val,
            })
        return self

    def enum(self, value, field_name: str,
             allowed: list[str], case_insensitive: bool = True) -> "Validator":
        """Validate a value is one of the allowed options."""
        if value is None:
            return self

        check_val = str(value).lower() if case_insensitive else str(value)
        allowed_check = [a.lower() for a in allowed] if case_insensitive else allowed

        if check_val not in allowed_check:
            self.errors.append({
                "field": field_name,
                "issue": f"must be one of {allowed}",
                "received": value,
            })
        return self

    def string_length(self, value, field_name: str,
                      min_len: int = None, max_len: int = None) -> "Validator":
        """Validate string length."""
        if value is None:
            return self
        s = str(value)
        if min_len is not None and len(s) < min_len:
            self.errors.append({
                "field": field_name,
                "issue": f"length must be >= {min_len} characters",
                "received": f"'{s}' (length {len(s)})",
            })
        if max_len is not None and len(s) > max_len:
            self.errors.append({
                "field": field_name,
                "issue": f"length must be <= {max_len} characters",
                "received": f"length {len(s)}",
            })
        return self

    def positive(self, value, field_name: str) -> "Validator":
        """Shortcut: validate value is positive (> 0)."""
        return self.numeric_range(value, field_name, min_val=0.001)

    def is_valid(self) -> bool:
        """Return True if no validation errors."""
        return len(self.errors) == 0

    def get_error_response(self) -> dict:
        """Build a structured error response for the MCP tool."""
        return {
            "error": "Input validation failed",
            "validation_errors": self.errors,
            "error_count": len(self.errors),
        }


# ============================================================================
# Type coercion helpers
# ============================================================================

def coerce_float(value, default=None) -> float | None:
    """Try to convert a value to float, handling common string formats."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace(" ", "")
        try:
            return float(cleaned)
        except ValueError:
            return default
    return default


def coerce_int(value, default=None) -> int | None:
    """Try to convert a value to int."""
    f = coerce_float(value)
    if f is not None:
        return int(f)
    return default


def coerce_sex(value) -> str:
    """Normalize sex/gender input to standard values."""
    if value is None:
        return "general"
    s = str(value).lower().strip()
    if s in ("m", "male", "man", "boy"):
        return "male"
    elif s in ("f", "female", "woman", "girl"):
        return "female"
    else:
        return "general"


# ============================================================================
# Validated healthcare tools
# ============================================================================

def calculate_bmi_validated(weight_kg, height_m) -> dict:
    """Calculate BMI with full input validation."""
    # Coerce types
    weight = coerce_float(weight_kg)
    height = coerce_float(height_m)

    # Validate
    v = Validator()
    v.require(weight, "weight_kg")
    v.require(height, "height_m")
    v.numeric_range(weight, "weight_kg", min_val=0.5, max_val=500)
    v.numeric_range(height, "height_m", min_val=0.3, max_val=3.0)

    if not v.is_valid():
        return v.get_error_response()

    bmi = round(weight / (height ** 2), 1)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    return {"bmi": bmi, "category": category, "weight_kg": weight, "height_m": height}


def check_bp_validated(systolic, diastolic) -> dict:
    """Check blood pressure with full input validation."""
    sys_val = coerce_int(systolic)
    dia_val = coerce_int(diastolic)

    v = Validator()
    v.require(sys_val, "systolic")
    v.require(dia_val, "diastolic")
    v.numeric_range(sys_val, "systolic", min_val=40, max_val=300)
    v.numeric_range(dia_val, "diastolic", min_val=20, max_val=200)

    if not v.is_valid():
        return v.get_error_response()

    if sys_val < dia_val:
        return {
            "error": "Input validation failed",
            "validation_errors": [{
                "field": "systolic/diastolic",
                "issue": "systolic must be >= diastolic",
                "received": f"{sys_val}/{dia_val}"
            }],
            "error_count": 1,
        }

    if sys_val < 120 and dia_val < 80:
        category, stage = "Normal", "Normal"
    elif sys_val < 130 and dia_val < 80:
        category, stage = "Elevated", "Elevated"
    elif sys_val < 140 or dia_val < 90:
        category, stage = "High Blood Pressure", "Stage 1"
    elif sys_val < 180 and dia_val < 120:
        category, stage = "High Blood Pressure", "Stage 2"
    else:
        category, stage = "Hypertensive Crisis", "Crisis"

    return {
        "systolic": sys_val, "diastolic": dia_val,
        "reading": f"{sys_val}/{dia_val} mmHg",
        "category": category, "stage": stage,
    }


def interpret_lab_validated(test_name, value, sex="general") -> dict:
    """Interpret lab value with full input validation."""
    VALID_TESTS = [
        "hemoglobin", "glucose", "potassium", "sodium", "creatinine",
        "wbc", "platelets", "alt", "hba1c", "tsh", "troponin_i", "inr"
    ]

    v = Validator()
    v.require(test_name, "test_name")
    v.require(value, "value")

    if test_name:
        v.enum(test_name.lower().strip(), "test_name", VALID_TESTS)

    num_value = coerce_float(value)
    v.numeric_range(num_value, "value", min_val=0, max_val=100000)

    sex_normalized = coerce_sex(sex)
    v.enum(sex_normalized, "sex", ["male", "female", "general"])

    if not v.is_valid():
        return v.get_error_response()

    # Simplified interpretation (production would use full reference ranges)
    return {
        "test": test_name,
        "value": num_value,
        "sex": sex_normalized,
        "status": "interpreted",
        "note": "Validation passed — value would be interpreted against reference ranges",
    }


def lookup_medication_validated(medication_name) -> dict:
    """Look up medication with input validation."""
    v = Validator()
    v.require(medication_name, "medication_name")
    v.string_length(medication_name, "medication_name", min_len=2, max_len=100)

    if not v.is_valid():
        return v.get_error_response()

    KNOWN_MEDICATIONS = [
        "metformin", "lisinopril", "atorvastatin", "amlodipine",
        "omeprazole", "levothyroxine", "sertraline", "albuterol",
        "warfarin", "apixaban", "gabapentin", "prednisone"
    ]

    name = medication_name.lower().strip()
    if name not in KNOWN_MEDICATIONS:
        return {
            "error": f"Medication '{medication_name}' not found",
            "suggestion": f"Available: {', '.join(KNOWN_MEDICATIONS[:6])}...",
        }

    return {"medication": name, "found": True, "note": "Lookup successful"}


# ============================================================================
# MCP Server Registration
# ============================================================================

if MCP_AVAILABLE:
    mcp = FastMCP("Validated Healthcare Tools")

    @mcp.tool()
    def mcp_calculate_bmi(weight_kg: float, height_m: float) -> str:
        """Calculate BMI with validated inputs. Weight must be 0.5-500 kg,
        height must be 0.3-3.0 m. Returns BMI and WHO category."""
        return json.dumps(calculate_bmi_validated(weight_kg, height_m))

    @mcp.tool()
    def mcp_check_bp(systolic: int, diastolic: int) -> str:
        """Classify blood pressure with validated inputs. Systolic 40-300,
        diastolic 20-200. Systolic must be >= diastolic."""
        return json.dumps(check_bp_validated(systolic, diastolic))

    @mcp.tool()
    def mcp_interpret_lab(test_name: str, value: float, sex: str = "general") -> str:
        """Interpret lab value with validated inputs. Test must be a known
        lab test. Value must be numeric and non-negative."""
        return json.dumps(interpret_lab_validated(test_name, value, sex))

    @mcp.tool()
    def mcp_lookup_medication(medication_name: str) -> str:
        """Look up medication with validated input. Name must be 2-100 chars."""
        return json.dumps(lookup_medication_validated(medication_name))


# ============================================================================
# Test / Demo
# ============================================================================

def main():
    """Test the validation framework with good and bad inputs."""
    print("=" * 70)
    print("  Exercise 3: Input Validation for MCP Tools")
    print("  Robust validation with helpful error messages")
    print("=" * 70)

    # --- BMI Validation Tests ---
    print("\n  TOOL: calculate_bmi_validated")
    print("  " + "─" * 55)
    bmi_tests = [
        (78.0, 1.72, "Valid input"),
        (-5.0, 1.72, "Negative weight"),
        (78.0, 0.1, "Height too short"),
        (600, 1.72, "Weight > 500kg"),
        (None, 1.72, "Missing weight"),
        ("78", "1.72", "Strings (should coerce)"),
        ("seventy", 1.72, "Non-numeric string"),
    ]
    for weight, height, label in bmi_tests:
        result = calculate_bmi_validated(weight, height)
        if "error" in result:
            errs = result.get("validation_errors", [])
            err_msg = errs[0]["issue"] if errs else result["error"]
            print(f"  ✗ {label}: {err_msg}")
        else:
            print(f"  ✓ {label}: BMI = {result['bmi']}, {result['category']}")

    # --- BP Validation Tests ---
    print(f"\n  TOOL: check_bp_validated")
    print("  " + "─" * 55)
    bp_tests = [
        (120, 80, "Valid normal"),
        (80, 120, "Systolic < diastolic"),
        (350, 80, "Systolic too high"),
        (120, 10, "Diastolic too low"),
        (None, 80, "Missing systolic"),
        ("135", "88", "Strings (should coerce)"),
    ]
    for sys, dia, label in bp_tests:
        result = check_bp_validated(sys, dia)
        if "error" in result:
            errs = result.get("validation_errors", [])
            err_msg = errs[0]["issue"] if errs else result["error"]
            print(f"  ✗ {label}: {err_msg}")
        else:
            print(f"  ✓ {label}: {result['reading']} → {result['category']}")

    # --- Lab Validation Tests ---
    print(f"\n  TOOL: interpret_lab_validated")
    print("  " + "─" * 55)
    lab_tests = [
        ("hemoglobin", 14.5, "male", "Valid male hemoglobin"),
        ("hemoglobin", 14.5, "M", "Sex coercion (M → male)"),
        ("unknown_test", 5.0, "general", "Invalid test name"),
        ("glucose", None, "general", "Missing value"),
        ("glucose", -5, "general", "Negative value"),
    ]
    for test, value, sex, label in lab_tests:
        result = interpret_lab_validated(test, value, sex)
        if "error" in result:
            errs = result.get("validation_errors", [])
            err_msg = errs[0]["issue"] if errs else result["error"]
            print(f"  ✗ {label}: {err_msg}")
        else:
            print(f"  ✓ {label}: {result.get('status', 'ok')}")

    # --- Medication Validation Tests ---
    print(f"\n  TOOL: lookup_medication_validated")
    print("  " + "─" * 55)
    med_tests = [
        ("metformin", "Valid medication"),
        ("a", "Name too short"),
        (None, "Missing name"),
        ("nonexistentdrug", "Unknown medication"),
    ]
    for name, label in med_tests:
        result = lookup_medication_validated(name)
        if "error" in result:
            errs = result.get("validation_errors", [])
            err_msg = errs[0]["issue"] if errs else result["error"]
            print(f"  ✗ {label}: {err_msg}")
        else:
            print(f"  ✓ {label}: Found {result['medication']}")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("  Validation rules applied:")
    print("    • Numeric ranges: weight 0.5-500kg, height 0.3-3.0m")
    print("    • Enum validation: sex (male/female/general), test names")
    print("    • Required fields: weight, height, test_name, value, medication_name")
    print("    • Type coercion: strings → numbers, sex abbreviations → full")
    print("    • String length: medication name 2-100 characters")
    print("    • Cross-field: systolic >= diastolic")
    print("=" * 70)


if __name__ == "__main__":
    main()
