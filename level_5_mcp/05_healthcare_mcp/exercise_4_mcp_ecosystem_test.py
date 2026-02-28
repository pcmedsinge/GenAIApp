"""
Exercise 4: MCP Ecosystem End-to-End Test Suite
==================================================

Skills practiced:
- Building a comprehensive test suite for an MCP server ecosystem
- Testing individual servers in isolation (unit-level)
- Testing cross-server workflows (integration-level)
- Reporting quality metrics: pass/fail rates, response times, data integrity

Healthcare context:
Before deploying an MCP-based clinical AI system, every server must be
validated individually and as part of the larger ecosystem. This exercise
builds a test harness that verifies each server's tools work correctly,
tests cross-server data consistency, runs workflow-level integration
tests, and produces a quality report.

Usage:
    python exercise_4_mcp_ecosystem_test.py
"""

import os
import json
import time
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
# Clinical Data (shared across MCP servers under test)
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

LAB_RESULTS = {
    "P001": [
        {"test": "HbA1c", "value": 7.8, "unit": "%", "range": "4.0-5.6",
         "flag": "HIGH", "critical": False},
        {"test": "Glucose", "value": 118, "unit": "mg/dL", "range": "70-100",
         "flag": "HIGH", "critical": False},
        {"test": "Creatinine", "value": 1.1, "unit": "mg/dL", "range": "0.7-1.3",
         "flag": "NORMAL", "critical": False},
        {"test": "LDL", "value": 142, "unit": "mg/dL", "range": "<100",
         "flag": "HIGH", "critical": False},
    ],
    "P003": [
        {"test": "BNP", "value": 450, "unit": "pg/mL", "range": "<100",
         "flag": "HIGH", "critical": True},
        {"test": "Potassium", "value": 5.4, "unit": "mEq/L", "range": "3.5-5.0",
         "flag": "HIGH", "critical": False},
        {"test": "INR", "value": 2.5, "unit": "", "range": "2.0-3.0",
         "flag": "NORMAL", "critical": False},
    ],
}

MEDICATIONS = {
    "P001": [
        {"name": "metformin", "dose": "1000mg", "freq": "BID", "status": "active"},
        {"name": "amlodipine", "dose": "10mg", "freq": "daily", "status": "active"},
        {"name": "atorvastatin", "dose": "40mg", "freq": "daily", "status": "active"},
    ],
    "P003": [
        {"name": "tiotropium", "dose": "18mcg", "freq": "daily inhaler", "status": "active"},
        {"name": "losartan", "dose": "50mg", "freq": "daily", "status": "active"},
        {"name": "apixaban", "dose": "5mg", "freq": "BID", "status": "active"},
    ],
}

VITALS = {
    "P001": {"bp_sys": 138, "bp_dia": 85, "hr": 78, "temp": 98.4, "spo2": 97},
    "P002": {"bp_sys": 118, "bp_dia": 74, "hr": 70, "temp": 98.1, "spo2": 99},
    "P003": {"bp_sys": 155, "bp_dia": 92, "hr": 68, "temp": 97.8, "spo2": 94},
}

DRUG_INTERACTIONS = {
    ("amlodipine", "atorvastatin"): {"severity": "moderate",
                                      "note": "Limit atorvastatin to 20mg with amlodipine (CYP3A4)"},
    ("losartan", "apixaban"): {"severity": "minor",
                                "note": "Monitor for additive hypotension"},
}


# ============================================================================
# Simulated MCP Server Functions (under test)
# ============================================================================

def ehr_lookup_patient(patient_id: str) -> dict:
    """EHR Server: look up patient demographics."""
    if patient_id in PATIENTS:
        return {"found": True, **PATIENTS[patient_id]}
    return {"found": False, "error": f"Patient {patient_id} not found"}


def ehr_get_vitals(patient_id: str) -> dict:
    """EHR Server: get latest vitals."""
    if patient_id in VITALS:
        return {"patient_id": patient_id, "vitals": VITALS[patient_id]}
    return {"error": f"No vitals for {patient_id}"}


def lab_get_results(patient_id: str) -> dict:
    """Lab Server: get lab results."""
    if patient_id in LAB_RESULTS:
        return {"patient_id": patient_id, "results": LAB_RESULTS[patient_id]}
    return {"patient_id": patient_id, "results": []}


def lab_get_critical(patient_id: str) -> dict:
    """Lab Server: get critical values."""
    results = LAB_RESULTS.get(patient_id, [])
    criticals = [r for r in results if r.get("critical")]
    return {"patient_id": patient_id, "critical_count": len(criticals),
            "critical_values": criticals}


def pharmacy_get_meds(patient_id: str) -> dict:
    """Pharmacy Server: get medications."""
    if patient_id in MEDICATIONS:
        return {"patient_id": patient_id, "medications": MEDICATIONS[patient_id]}
    return {"patient_id": patient_id, "medications": []}


def pharmacy_check_interaction(drug_a: str, drug_b: str) -> dict:
    """Pharmacy Server: check drug interaction."""
    key1 = (drug_a.lower(), drug_b.lower())
    key2 = (drug_b.lower(), drug_a.lower())
    if key1 in DRUG_INTERACTIONS:
        return DRUG_INTERACTIONS[key1]
    elif key2 in DRUG_INTERACTIONS:
        return DRUG_INTERACTIONS[key2]
    return {"severity": "none", "note": "No significant interaction"}


# ============================================================================
# Test Framework
# ============================================================================

class TestResult:
    """Stores the result of a single test."""

    def __init__(self, test_name: str, category: str, passed: bool,
                 message: str = "", duration_ms: float = 0.0,
                 details: dict = None):
        self.test_name = test_name
        self.category = category
        self.passed = passed
        self.message = message
        self.duration_ms = duration_ms
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "test": self.test_name,
            "category": self.category,
            "passed": self.passed,
            "message": self.message,
            "duration_ms": round(self.duration_ms, 2),
        }


class TestSuite:
    """Collects and reports test results."""

    def __init__(self, name: str):
        self.name = name
        self.results = []
        self.start_time = None
        self.end_time = None

    def add_result(self, result: TestResult):
        self.results.append(result)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0

    @property
    def avg_duration(self) -> float:
        if not self.results:
            return 0
        return sum(r.duration_ms for r in self.results) / len(self.results)

    def get_by_category(self, category: str) -> list:
        return [r for r in self.results if r.category == category]

    def print_report(self):
        """Print a formatted test report."""
        print(f"\n  {'=' * 60}")
        print(f"  TEST REPORT: {self.name}")
        print(f"  {'=' * 60}")
        print(f"  Total: {self.total} | Passed: {self.passed} | "
              f"Failed: {self.failed} | Rate: {self.pass_rate:.1f}%")
        print(f"  Avg Duration: {self.avg_duration:.2f}ms")

        # By category
        categories = sorted(set(r.category for r in self.results))
        print(f"\n  By Category:")
        for cat in categories:
            cat_results = self.get_by_category(cat)
            cat_passed = sum(1 for r in cat_results if r.passed)
            cat_total = len(cat_results)
            pct = cat_passed / cat_total * 100 if cat_total > 0 else 0
            indicator = "✓" if pct == 100 else "⚠" if pct >= 80 else "✗"
            print(f"    {indicator} {cat:<30} {cat_passed}/{cat_total} ({pct:.0f}%)")

        # Failed tests
        failures = [r for r in self.results if not r.passed]
        if failures:
            print(f"\n  Failed Tests:")
            for r in failures:
                print(f"    ✗ [{r.category}] {r.test_name}")
                print(f"      {r.message}")

        print(f"  {'=' * 60}")


def run_test(test_name: str, category: str, test_func) -> TestResult:
    """Run a single test and return the result."""
    start = time.perf_counter()
    try:
        passed, message, details = test_func()
        duration = (time.perf_counter() - start) * 1000
        return TestResult(test_name, category, passed, message, duration, details)
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        return TestResult(test_name, category, False,
                          f"Exception: {str(e)}", duration)


# ============================================================================
# EHR Server Tests
# ============================================================================

def test_ehr_patient_lookup_valid():
    """Test: Look up a valid patient returns correct data."""
    result = ehr_lookup_patient("P001")
    if not result.get("found"):
        return False, "Patient P001 not found", {}
    if result["name"] != "John Smith":
        return False, f"Expected 'John Smith', got '{result['name']}'", {}
    return True, "Patient P001 found with correct name", {}


def test_ehr_patient_lookup_invalid():
    """Test: Look up an invalid patient returns error."""
    result = ehr_lookup_patient("P999")
    if result.get("found", True):
        return False, "Should return found=False for invalid patient", {}
    if "error" not in result:
        return False, "Should include error message", {}
    return True, "Invalid patient handled correctly", {}


def test_ehr_patient_has_required_fields():
    """Test: Patient record contains all required fields."""
    required = ["name", "dob", "age", "sex", "mrn", "problems", "allergies", "pcp"]
    result = ehr_lookup_patient("P001")
    missing = [f for f in required if f not in result]
    if missing:
        return False, f"Missing fields: {missing}", {}
    return True, "All required fields present", {}


def test_ehr_all_patients_accessible():
    """Test: All patients in the database are accessible."""
    errors = []
    for pid in PATIENTS:
        result = ehr_lookup_patient(pid)
        if not result.get("found"):
            errors.append(pid)
    if errors:
        return False, f"Inaccessible patients: {errors}", {}
    return True, f"All {len(PATIENTS)} patients accessible", {}


def test_ehr_vitals_valid_patient():
    """Test: Vitals returned for valid patient."""
    result = ehr_get_vitals("P001")
    if "error" in result:
        return False, result["error"], {}
    vitals = result["vitals"]
    required_vitals = ["bp_sys", "bp_dia", "hr", "temp", "spo2"]
    missing = [v for v in required_vitals if v not in vitals]
    if missing:
        return False, f"Missing vitals: {missing}", {}
    return True, "Vitals returned with all fields", {}


def test_ehr_vitals_ranges():
    """Test: Vital sign values are within physiologically possible ranges."""
    for pid in VITALS:
        v = VITALS[pid]
        if not (50 <= v["bp_sys"] <= 250):
            return False, f"{pid}: BP sys {v['bp_sys']} out of range", {}
        if not (30 <= v["bp_dia"] <= 150):
            return False, f"{pid}: BP dia {v['bp_dia']} out of range", {}
        if not (30 <= v["hr"] <= 200):
            return False, f"{pid}: HR {v['hr']} out of range", {}
        if not (80 <= v["spo2"] <= 100):
            return False, f"{pid}: SpO2 {v['spo2']} out of range", {}
    return True, "All vitals within physiological ranges", {}


# ============================================================================
# Lab Server Tests
# ============================================================================

def test_lab_results_valid_patient():
    """Test: Lab results returned for valid patient."""
    result = lab_get_results("P001")
    if not result["results"]:
        return False, "No results for P001", {}
    return True, f"{len(result['results'])} results returned", {}


def test_lab_results_empty_patient():
    """Test: Empty results for patient with no labs."""
    result = lab_get_results("P002")
    if result["results"]:
        return False, "Expected empty results for P002", {}
    return True, "Empty results returned correctly", {}


def test_lab_result_structure():
    """Test: Each lab result has required fields."""
    required = ["test", "value", "unit", "range", "flag", "critical"]
    result = lab_get_results("P001")
    for lab in result["results"]:
        missing = [f for f in required if f not in lab]
        if missing:
            return False, f"Lab '{lab.get('test', '?')}' missing: {missing}", {}
    return True, "All lab results have required fields", {}


def test_lab_flags_valid():
    """Test: Lab flags are valid values (NORMAL, HIGH, LOW)."""
    valid_flags = {"NORMAL", "HIGH", "LOW"}
    for pid in LAB_RESULTS:
        for lab in LAB_RESULTS[pid]:
            if lab["flag"] not in valid_flags:
                return False, f"{pid} {lab['test']}: invalid flag '{lab['flag']}'", {}
    return True, "All lab flags are valid", {}


def test_lab_critical_values():
    """Test: Critical values are correctly identified for P003."""
    result = lab_get_critical("P003")
    if result["critical_count"] == 0:
        return False, "P003 should have critical values (BNP)", {}
    critical_tests = [c["test"] for c in result["critical_values"]]
    if "BNP" not in critical_tests:
        return False, f"BNP should be critical, found: {critical_tests}", {}
    return True, f"{result['critical_count']} critical value(s) identified", {}


def test_lab_no_criticals_for_normal():
    """Test: No critical values for patient P001."""
    result = lab_get_critical("P001")
    if result["critical_count"] != 0:
        return False, f"P001 should have 0 criticals, got {result['critical_count']}", {}
    return True, "P001 has no critical values (correct)", {}


# ============================================================================
# Pharmacy Server Tests
# ============================================================================

def test_pharmacy_meds_valid():
    """Test: Medications returned for valid patient."""
    result = pharmacy_get_meds("P001")
    if not result["medications"]:
        return False, "No meds for P001", {}
    return True, f"{len(result['medications'])} medications returned", {}


def test_pharmacy_med_structure():
    """Test: Each medication has required fields."""
    required = ["name", "dose", "freq", "status"]
    result = pharmacy_get_meds("P001")
    for med in result["medications"]:
        missing = [f for f in required if f not in med]
        if missing:
            return False, f"Med '{med.get('name', '?')}' missing: {missing}", {}
    return True, "All medications have required fields", {}


def test_pharmacy_interaction_known():
    """Test: Known interaction is detected."""
    result = pharmacy_check_interaction("amlodipine", "atorvastatin")
    if result["severity"] == "none":
        return False, "Should detect moderate interaction", {}
    if result["severity"] != "moderate":
        return False, f"Expected 'moderate', got '{result['severity']}'", {}
    return True, "Moderate interaction detected correctly", {}


def test_pharmacy_interaction_reversed():
    """Test: Interaction detected regardless of drug order."""
    r1 = pharmacy_check_interaction("amlodipine", "atorvastatin")
    r2 = pharmacy_check_interaction("atorvastatin", "amlodipine")
    if r1["severity"] != r2["severity"]:
        return False, f"Asymmetric: {r1['severity']} vs {r2['severity']}", {}
    return True, "Interaction is symmetric (order-independent)", {}


def test_pharmacy_no_interaction():
    """Test: No interaction for safe combination."""
    result = pharmacy_check_interaction("metformin", "levothyroxine")
    if result["severity"] != "none":
        return False, f"Expected 'none', got '{result['severity']}'", {}
    return True, "No interaction returned for safe pair", {}


def test_pharmacy_all_active():
    """Test: All returned medications have active status."""
    for pid in MEDICATIONS:
        result = pharmacy_get_meds(pid)
        for med in result["medications"]:
            if med["status"] != "active":
                return False, f"{pid} {med['name']}: status is '{med['status']}'", {}
    return True, "All medications are active", {}


# ============================================================================
# Integration Tests (cross-server)
# ============================================================================

def test_integration_patient_labs_consistency():
    """Test: Patients with labs exist in EHR."""
    for pid in LAB_RESULTS:
        result = ehr_lookup_patient(pid)
        if not result.get("found"):
            return False, f"Lab patient {pid} not in EHR", {}
    return True, "All lab patients exist in EHR", {}


def test_integration_patient_meds_consistency():
    """Test: Patients with medications exist in EHR."""
    for pid in MEDICATIONS:
        result = ehr_lookup_patient(pid)
        if not result.get("found"):
            return False, f"Meds patient {pid} not in EHR", {}
    return True, "All medication patients exist in EHR", {}


def test_integration_diabetic_has_hba1c():
    """Test: Diabetic patient should have HbA1c lab."""
    for pid, pt in PATIENTS.items():
        is_diabetic = any("diabetes" in p.lower() for p in pt["problems"])
        if is_diabetic:
            labs = lab_get_results(pid)
            has_hba1c = any(r["test"] == "HbA1c" for r in labs["results"])
            if not has_hba1c:
                return False, f"Diabetic patient {pid} missing HbA1c", {}
    return True, "Diabetic patients have HbA1c results", {}


def test_integration_hypertensive_on_antihypertensive():
    """Test: Hypertensive patients should be on an antihypertensive."""
    antihypertensives = {"amlodipine", "losartan", "lisinopril", "metoprolol",
                         "hydrochlorothiazide", "valsartan"}
    for pid, pt in PATIENTS.items():
        is_htn = any("hypertension" in p.lower() for p in pt["problems"])
        if is_htn and pid in MEDICATIONS:
            meds = pharmacy_get_meds(pid)
            med_names = {m["name"].lower() for m in meds["medications"]}
            has_antihtn = bool(med_names & antihypertensives)
            if not has_antihtn:
                return False, f"HTN patient {pid} not on antihypertensive", {}
    return True, "Hypertensive patients on appropriate meds", {}


def test_integration_critical_patient_has_vitals():
    """Test: Patients with critical labs have vitals recorded."""
    for pid in LAB_RESULTS:
        criticals = lab_get_critical(pid)
        if criticals["critical_count"] > 0:
            vitals = ehr_get_vitals(pid)
            if "error" in vitals:
                return False, f"Critical patient {pid} has no vitals", {}
    return True, "Critical patients have vitals recorded", {}


def test_integration_full_workflow():
    """Test: Complete workflow executes for all patients."""
    for pid in PATIENTS:
        # EHR
        pt = ehr_lookup_patient(pid)
        if not pt.get("found"):
            return False, f"Workflow failed at EHR for {pid}", {}

        # Vitals
        vitals = ehr_get_vitals(pid)
        if "error" in vitals and pid in VITALS:
            return False, f"Workflow failed at vitals for {pid}", {}

        # Labs (may be empty)
        labs = lab_get_results(pid)
        if "error" in labs:
            return False, f"Workflow failed at labs for {pid}", {}

        # Meds (may be empty)
        meds = pharmacy_get_meds(pid)
        if "error" in meds:
            return False, f"Workflow failed at pharmacy for {pid}", {}

    return True, f"Full workflow completed for {len(PATIENTS)} patients", {}


# ============================================================================
# Data Quality Tests
# ============================================================================

def test_quality_no_duplicate_patients():
    """Test: No duplicate patient MRNs."""
    mrns = [pt["mrn"] for pt in PATIENTS.values()]
    if len(mrns) != len(set(mrns)):
        return False, "Duplicate MRNs found", {}
    return True, "All MRNs are unique", {}


def test_quality_ages_reasonable():
    """Test: Patient ages are reasonable (0-120)."""
    for pid, pt in PATIENTS.items():
        if not (0 <= pt["age"] <= 120):
            return False, f"{pid}: age {pt['age']} is unreasonable", {}
    return True, "All patient ages are reasonable", {}


def test_quality_lab_values_positive():
    """Test: Lab values are positive numbers."""
    for pid in LAB_RESULTS:
        for lab in LAB_RESULTS[pid]:
            if lab["value"] < 0:
                return False, f"{pid} {lab['test']}: negative value {lab['value']}", {}
    return True, "All lab values are positive", {}


def test_quality_no_empty_med_names():
    """Test: No medications with empty names."""
    for pid in MEDICATIONS:
        for med in MEDICATIONS[pid]:
            if not med["name"].strip():
                return False, f"{pid}: empty medication name", {}
    return True, "All medications have names", {}


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests() -> TestSuite:
    """Run the complete test suite."""
    suite = TestSuite("Healthcare MCP Ecosystem")

    # EHR Server Tests
    ehr_tests = [
        ("EHR: Valid patient lookup", test_ehr_patient_lookup_valid),
        ("EHR: Invalid patient lookup", test_ehr_patient_lookup_invalid),
        ("EHR: Required fields present", test_ehr_patient_has_required_fields),
        ("EHR: All patients accessible", test_ehr_all_patients_accessible),
        ("EHR: Vitals for valid patient", test_ehr_vitals_valid_patient),
        ("EHR: Vitals in valid ranges", test_ehr_vitals_ranges),
    ]
    for name, func in ehr_tests:
        suite.add_result(run_test(name, "EHR Server", func))

    # Lab Server Tests
    lab_tests = [
        ("Lab: Results for valid patient", test_lab_results_valid_patient),
        ("Lab: Empty results for no-lab patient", test_lab_results_empty_patient),
        ("Lab: Result structure valid", test_lab_result_structure),
        ("Lab: Flags are valid values", test_lab_flags_valid),
        ("Lab: Critical values identified", test_lab_critical_values),
        ("Lab: No criticals for normal patient", test_lab_no_criticals_for_normal),
    ]
    for name, func in lab_tests:
        suite.add_result(run_test(name, "Lab Server", func))

    # Pharmacy Server Tests
    pharmacy_tests = [
        ("Pharmacy: Meds for valid patient", test_pharmacy_meds_valid),
        ("Pharmacy: Med structure valid", test_pharmacy_med_structure),
        ("Pharmacy: Known interaction detected", test_pharmacy_interaction_known),
        ("Pharmacy: Interaction is symmetric", test_pharmacy_interaction_reversed),
        ("Pharmacy: No interaction for safe pair", test_pharmacy_no_interaction),
        ("Pharmacy: All meds are active", test_pharmacy_all_active),
    ]
    for name, func in pharmacy_tests:
        suite.add_result(run_test(name, "Pharmacy Server", func))

    # Integration Tests
    integration_tests = [
        ("Integration: Lab patients in EHR", test_integration_patient_labs_consistency),
        ("Integration: Med patients in EHR", test_integration_patient_meds_consistency),
        ("Integration: Diabetic has HbA1c", test_integration_diabetic_has_hba1c),
        ("Integration: HTN on antihypertensive", test_integration_hypertensive_on_antihypertensive),
        ("Integration: Critical patients have vitals", test_integration_critical_patient_has_vitals),
        ("Integration: Full workflow", test_integration_full_workflow),
    ]
    for name, func in integration_tests:
        suite.add_result(run_test(name, "Integration", func))

    # Data Quality Tests
    quality_tests = [
        ("Quality: No duplicate MRNs", test_quality_no_duplicate_patients),
        ("Quality: Ages reasonable", test_quality_ages_reasonable),
        ("Quality: Lab values positive", test_quality_lab_values_positive),
        ("Quality: No empty med names", test_quality_no_empty_med_names),
    ]
    for name, func in quality_tests:
        suite.add_result(run_test(name, "Data Quality", func))

    return suite


# ============================================================================
# Helper Functions
# ============================================================================

def print_banner(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ============================================================================
# Section 1: Individual Server Tests
# ============================================================================

def section_individual_tests():
    """Run and display individual server test results."""
    print_banner("Section 1: Individual Server Tests")

    print("""
  Testing each MCP server in isolation to verify tool correctness,
  data structure, and error handling.
    """)

    suite = run_all_tests()

    # Show individual results
    for category in ["EHR Server", "Lab Server", "Pharmacy Server"]:
        results = suite.get_by_category(category)
        passed = sum(1 for r in results if r.passed)
        print(f"\n  --- {category} ---")
        for r in results:
            indicator = "✓" if r.passed else "✗"
            print(f"    {indicator} {r.test_name:<45} {r.duration_ms:>6.2f}ms  {r.message}")
        print(f"    Result: {passed}/{len(results)} passed")


# ============================================================================
# Section 2: Integration Tests
# ============================================================================

def section_integration_tests():
    """Run cross-server integration tests."""
    print_banner("Section 2: Integration Tests")

    print("""
  Testing cross-server data consistency and end-to-end workflows.
  These verify that data in one server is consistent with related data
  in other servers (e.g., lab patients exist in EHR).
    """)

    suite = run_all_tests()

    results = suite.get_by_category("Integration")
    passed = sum(1 for r in results if r.passed)
    for r in results:
        indicator = "✓" if r.passed else "✗"
        print(f"  {indicator} {r.test_name:<50} {r.message}")

    print(f"\n  Integration result: {passed}/{len(results)} passed")
    if passed == len(results):
        print(f"  ✓ All cross-server integrations are consistent")
    else:
        print(f"  ⚠ Some integration issues detected — review failed tests")


# ============================================================================
# Section 3: Data Quality Tests
# ============================================================================

def section_quality_tests():
    """Run data quality validation tests."""
    print_banner("Section 3: Data Quality Tests")

    print("""
  Validating data quality across the ecosystem: unique identifiers,
  reasonable values, and proper data formatting.
    """)

    suite = run_all_tests()

    results = suite.get_by_category("Data Quality")
    passed = sum(1 for r in results if r.passed)
    for r in results:
        indicator = "✓" if r.passed else "✗"
        print(f"  {indicator} {r.test_name:<45} {r.message}")

    print(f"\n  Quality result: {passed}/{len(results)} passed")


# ============================================================================
# Section 4: Full Test Report
# ============================================================================

def section_full_report():
    """Run all tests and generate a comprehensive report."""
    print_banner("Section 4: Full Test Report")

    print("""
  Running the complete test suite and generating a quality report
  with metrics across all servers and test categories.
    """)

    suite = run_all_tests()
    suite.print_report()

    # Additional metrics
    print(f"\n  --- Performance Metrics ---")
    for category in ["EHR Server", "Lab Server", "Pharmacy Server", "Integration", "Data Quality"]:
        results = suite.get_by_category(category)
        if results:
            avg_ms = sum(r.duration_ms for r in results) / len(results)
            max_ms = max(r.duration_ms for r in results)
            print(f"  {category:<20} avg: {avg_ms:>6.2f}ms  max: {max_ms:>6.2f}ms")

    # Quality score
    print(f"\n  --- Quality Score ---")
    score = suite.pass_rate
    if score == 100:
        grade = "A+ (Excellent)"
    elif score >= 95:
        grade = "A (Very Good)"
    elif score >= 90:
        grade = "B (Good)"
    elif score >= 80:
        grade = "C (Acceptable)"
    else:
        grade = "F (Needs Work)"

    print(f"  Overall: {score:.1f}% — {grade}")
    print(f"  Tests:   {suite.passed}/{suite.total} passed")

    # Generate AI summary if available
    if OPENAI_AVAILABLE and suite.failed > 0:
        print(f"\n  --- AI Analysis of Failures ---")
        failures = [r.to_dict() for r in suite.results if not r.passed]
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a QA engineer analyzing test failures "
                     "in a healthcare MCP system. Provide a brief, actionable analysis (3-5 sentences)."},
                    {"role": "user", "content": f"Analyze these test failures:\n{json.dumps(failures, indent=2)}"}
                ],
                max_tokens=200,
            )
            print(f"  {response.choices[0].message.content}")
        except Exception as e:
            print(f"  (OpenAI unavailable: {e})")
    elif suite.failed == 0:
        print(f"\n  ✓ All tests passed — ecosystem is operating correctly")


# ============================================================================
# Section 5: MCP Server Test Definitions
# ============================================================================

def section_mcp_test_server():
    """Define MCP tools for running tests programmatically."""
    print_banner("Section 5: MCP Test Tools")

    print("""
  Defining MCP tools that expose the test suite, so an agent can
  trigger tests and review results programmatically.
    """)

    if MCP_AVAILABLE:
        test_server = FastMCP("Ecosystem Test Server")

        @test_server.tool()
        def run_ecosystem_tests(category: str = "") -> str:
            """Run ecosystem tests. Optionally filter by category:
            'EHR Server', 'Lab Server', 'Pharmacy Server', 'Integration', 'Data Quality'."""
            suite = run_all_tests()
            if category:
                results = suite.get_by_category(category)
            else:
                results = suite.results

            report = {
                "total": len(results),
                "passed": sum(1 for r in results if r.passed),
                "failed": sum(1 for r in results if not r.passed),
                "pass_rate": round(sum(1 for r in results if r.passed) / len(results) * 100, 1)
                             if results else 0,
                "results": [r.to_dict() for r in results],
            }
            return json.dumps(report, indent=2)

        @test_server.tool()
        def get_test_summary() -> str:
            """Get a high-level summary of the test suite results."""
            suite = run_all_tests()
            return json.dumps({
                "total": suite.total,
                "passed": suite.passed,
                "failed": suite.failed,
                "pass_rate": round(suite.pass_rate, 1),
                "avg_duration_ms": round(suite.avg_duration, 2),
            }, indent=2)

        print("  ✓ Test MCP Server defined with 2 tools")
    else:
        print("  ⚠ MCP SDK not installed — showing test tool schemas")

    # Show what the tools would look like
    print(f"\n  Test tool schemas:")
    print(f"    • run_ecosystem_tests(category?: str) → JSON test report")
    print(f"    • get_test_summary() → JSON summary with pass rate")

    # Run a quick demo
    print(f"\n  Quick test run:")
    suite = run_all_tests()
    print(f"    Total: {suite.total} | Passed: {suite.passed} | "
          f"Failed: {suite.failed} | Rate: {suite.pass_rate:.1f}%")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run the Ecosystem Test Suite exercise."""
    print("=" * 70)
    print("  Exercise 4: MCP Ecosystem End-to-End Test Suite")
    print("  Comprehensive testing of the healthcare MCP ecosystem")
    print("=" * 70)

    sections = {
        "1": ("Individual Server Tests", section_individual_tests),
        "2": ("Integration Tests", section_integration_tests),
        "3": ("Data Quality Tests", section_quality_tests),
        "4": ("Full Test Report", section_full_report),
        "5": ("MCP Test Tools", section_mcp_test_server),
    }

    while True:
        print("\nSections:")
        for key, (name, _) in sections.items():
            print(f"  {key}. {name}")
        print("  A. Run all sections")
        print("  Q. Quit")

        choice = input("\nSelect section (1-5, A, Q): ").strip().upper()

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
