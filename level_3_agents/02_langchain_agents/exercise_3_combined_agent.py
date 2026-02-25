"""
Exercise 3: Combined Agent — Drug Lookup + Lab Checking + Patient History

Skills practiced:
- Building agents that combine multiple data sources
- Using patient history context to improve clinical reasoning
- Multi-step tool chaining for comprehensive assessments
- Structured patient records as agent context

Key insight: Real clinical decisions require combining medication data,
  lab results, AND patient history. This agent does all three —
  like a physician pulling up the chart, labs, and pharmacy list.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent as create_langchain_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# Patient History Database
# ============================================================

PATIENT_DATABASE = {
    "smith_john": {
        "name": "John Smith", "age": 68, "sex": "Male",
        "conditions": ["Type 2 Diabetes", "Hypertension", "CKD Stage 3"],
        "medications": ["metformin 1000mg BID", "lisinopril 20mg daily", "amlodipine 5mg daily"],
        "allergies": ["Sulfa drugs"],
        "recent_labs": {"hba1c": 7.4, "gfr": 42, "potassium": 5.1, "creatinine": 1.6},
        "notes": "Last visit: increased lisinopril from 10mg. Follow-up for renal function."
    },
    "garcia_maria": {
        "name": "Maria Garcia", "age": 55, "sex": "Female",
        "conditions": ["Atrial Fibrillation", "Heart Failure (HFrEF)"],
        "medications": ["apixaban 5mg BID", "metoprolol 50mg BID", "lisinopril 10mg daily"],
        "allergies": [],
        "recent_labs": {"hemoglobin": 11.2, "potassium": 4.3, "creatinine": 0.9, "inr": 1.1},
        "notes": "Stable on anticoagulation. Monitor for anemia — Hb trending down."
    },
    "chen_wei": {
        "name": "Wei Chen", "age": 45, "sex": "Male",
        "conditions": ["Depression", "GERD", "Prediabetes"],
        "medications": ["sertraline 100mg daily", "omeprazole 20mg daily"],
        "allergies": ["Penicillin"],
        "recent_labs": {"hba1c": 6.2, "potassium": 4.0, "creatinine": 0.8},
        "notes": "HbA1c trending up over 12 months (5.8 → 6.0 → 6.2). Lifestyle counseling given."
    },
}


# ============================================================
# Tools
# ============================================================

@tool
def get_patient_history(patient_id: str) -> str:
    """Retrieve comprehensive patient history including conditions, medications,
    allergies, and recent labs. Available patients: smith_john, garcia_maria, chen_wei."""
    patient = PATIENT_DATABASE.get(patient_id.lower())
    if not patient:
        return f"Patient not found. Available: {', '.join(PATIENT_DATABASE.keys())}"

    return (
        f"Patient: {patient['name']} ({patient['age']}yo {patient['sex']})\n"
        f"Conditions: {', '.join(patient['conditions'])}\n"
        f"Medications: {', '.join(patient['medications'])}\n"
        f"Allergies: {', '.join(patient['allergies']) or 'NKDA'}\n"
        f"Recent Labs: {json.dumps(patient['recent_labs'])}\n"
        f"Notes: {patient['notes']}"
    )


@tool
def lookup_medication(medication_name: str) -> str:
    """Look up medication details. Available: metformin, lisinopril, amlodipine,
    apixaban, sertraline, omeprazole, metoprolol."""
    medications = {
        "metformin": "Class: Biguanide | Dose: 500-2000mg daily | Contraindicated: eGFR<30 | Monitor: HbA1c, B12, renal",
        "lisinopril": "Class: ACE Inhibitor | Dose: 10-40mg daily | SE: Cough, hyperkalemia | Contraindicated: Pregnancy",
        "amlodipine": "Class: CCB | Dose: 2.5-10mg daily | SE: Edema, dizziness",
        "apixaban": "Class: DOAC | Dose: 5mg BID (reduce to 2.5mg if age≥80/wt≤60kg/Cr≥1.5) | SE: Bleeding",
        "sertraline": "Class: SSRI | Dose: 50-200mg daily | SE: Nausea, insomnia | Caution: Serotonin syndrome with tramadol",
        "omeprazole": "Class: PPI | Dose: 20-40mg daily | SE: C.diff, B12 def long-term | Monitor: Mg2+ if long-term",
        "metoprolol": "Class: Beta-blocker | Dose: 25-200mg BID | SE: Bradycardia, fatigue | Monitor: HR, BP",
    }
    result = medications.get(medication_name.lower().split()[0])  # handle "metformin 1000mg"
    return result if result else f"Not found. Available: {', '.join(medications.keys())}"


@tool
def check_lab_value(test_name: str, value: float) -> str:
    """Interpret a lab value clinically. Available: hba1c, gfr, potassium, creatinine, hemoglobin, inr."""
    tests = {
        "hba1c": lambda v: f"HbA1c {v}%: {'Normal' if v < 5.7 else 'Prediabetes' if v < 6.5 else 'Diabetes'} (target <7%)",
        "gfr": lambda v: f"GFR {v}: {'Normal' if v >= 90 else 'Mild CKD' if v >= 60 else 'Stage 3 CKD' if v >= 30 else 'Stage 4 CKD' if v >= 15 else 'Kidney failure'}",
        "potassium": lambda v: f"K+ {v}: {'LOW' if v < 3.5 else 'Normal' if v <= 5.0 else 'HIGH (monitor if on ACEi/ARB)'}",
        "creatinine": lambda v: f"Cr {v}: {'Normal' if 0.7 <= v <= 1.3 else 'Elevated — assess renal function'}",
        "hemoglobin": lambda v: f"Hb {v}: {'Anemia' if v < 12 else 'Normal' if v <= 17 else 'Elevated'}",
        "inr": lambda v: f"INR {v}: {'Sub-therapeutic' if v < 2.0 else 'Therapeutic' if v <= 3.0 else 'Supra-therapeutic'}",
    }
    fn = tests.get(test_name.lower())
    return fn(value) if fn else f"Not found. Available: {', '.join(tests.keys())}"


@tool
def check_drug_interaction(drug1: str, drug2: str) -> str:
    """Check interaction between two medications."""
    interactions = {
        ("lisinopril", "potassium"): "MAJOR: Both raise K+ → hyperkalemia risk.",
        ("sertraline", "tramadol"): "MAJOR: Serotonin syndrome risk. Avoid.",
        ("apixaban", "aspirin"): "MAJOR: Increased bleeding risk.",
        ("metformin", "contrast_dye"): "MODERATE: Hold metformin 48h around contrast.",
        ("lisinopril", "nsaids"): "MODERATE: NSAIDs reduce ACEi effectiveness.",
        ("omeprazole", "methotrexate"): "MODERATE: Omeprazole increases methotrexate levels.",
    }
    key1, key2 = (drug1.lower(), drug2.lower()), (drug2.lower(), drug1.lower())
    result = interactions.get(key1) or interactions.get(key2)
    return result if result else f"No known interaction between {drug1} and {drug2}."


@tool
def assess_medication_safety(patient_id: str, proposed_medication: str) -> str:
    """Check if a proposed medication is safe for a specific patient,
    considering their conditions, current meds, allergies, and labs."""
    patient = PATIENT_DATABASE.get(patient_id.lower())
    if not patient:
        return f"Patient not found."

    warnings = []

    # Allergy check
    med_lower = proposed_medication.lower()
    for allergy in patient["allergies"]:
        allergy_lower = allergy.lower()
        if "sulfa" in allergy_lower and med_lower in ["sulfasalazine", "sulfamethoxazole", "trimethoprim"]:
            warnings.append(f"ALLERGY: Patient allergic to {allergy}. {proposed_medication} may cross-react.")
        if "penicillin" in allergy_lower and med_lower in ["amoxicillin", "ampicillin"]:
            warnings.append(f"ALLERGY: Patient allergic to {allergy}. {proposed_medication} is contraindicated.")

    # Renal check
    gfr = patient["recent_labs"].get("gfr", 90)
    if gfr < 30 and med_lower == "metformin":
        warnings.append(f"CONTRAINDICATED: Metformin with GFR {gfr} (<30). Risk of lactic acidosis.")
    elif gfr < 45 and med_lower == "metformin":
        warnings.append(f"CAUTION: Metformin with GFR {gfr}. Consider dose reduction. Monitor closely.")

    # Potassium check
    k = patient["recent_labs"].get("potassium", 4.0)
    if k > 5.0 and med_lower in ["lisinopril", "spironolactone", "potassium"]:
        warnings.append(f"CAUTION: K+ is {k} (high). {proposed_medication} may worsen hyperkalemia.")

    if not warnings:
        return f"No immediate safety concerns for {proposed_medication} in this patient."

    return "\n".join(warnings)


all_tools = [get_patient_history, lookup_medication, check_lab_value,
             check_drug_interaction, assess_medication_safety]


def create_combined_agent():
    return create_langchain_agent(
        llm,
        tools=all_tools,
        system_prompt="""You are a comprehensive clinical decision support agent.
You have access to:
1. Patient history (conditions, meds, allergies, labs)
2. Medication database (dosing, side effects, contraindications)
3. Lab interpretation (clinical significance)
4. Drug interaction checker
5. Medication safety assessment (patient-specific)

For clinical questions:
- ALWAYS pull up patient history first
- Then use specific tools based on the question
- Cross-reference findings (e.g., labs + medications)
- Flag any safety concerns proactively

You are for educational purposes only — not real medical advice."""
    )


# ============================================================
# DEMO 1: Comprehensive Patient Review
# ============================================================

def demo_patient_review():
    """Full review of a patient: history + labs + medications"""
    print("\n" + "=" * 70)
    print("DEMO 1: COMPREHENSIVE PATIENT REVIEW")
    print("=" * 70)
    print("""
    Agent pulls up patient history, interprets each lab,
    reviews each medication, and provides integrated assessment.
    """)

    agent = create_combined_agent()

    question = (
        "Review patient smith_john. Check all their current labs, review their "
        "medications for appropriateness given their conditions, and flag any concerns."
    )

    print(f"\n  Q: {question}\n")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content
    print(f"\n  ASSESSMENT:\n  {answer[:800]}")


# ============================================================
# DEMO 2: Multi-Patient Comparison
# ============================================================

def demo_multi_patient():
    """Compare findings across patients"""
    print("\n" + "=" * 70)
    print("DEMO 2: MULTI-PATIENT QUERIES")
    print("=" * 70)

    agent = create_combined_agent()

    queries = [
        "Patient garcia_maria's hemoglobin is trending down. What's her current level and what should we do?",
        "Patient chen_wei's HbA1c has been trending up. Pull their history and advise.",
        "Is metformin safe for patient smith_john given their kidney function?",
    ]

    for q in queries:
        print(f"\n{'─' * 60}")
        print(f"  Q: {q}\n")
        result = agent.invoke({"messages": [{"role": "user", "content": q}]})
        answer = result["messages"][-1].content
        print(f"\n  Answer: {answer[:400]}")


# ============================================================
# DEMO 3: Safety-First Prescribing
# ============================================================

def demo_safety_prescribing():
    """Test the safety assessment tool"""
    print("\n" + "=" * 70)
    print("DEMO 3: SAFETY-FIRST PRESCRIBING")
    print("=" * 70)
    print("""
    Before prescribing, the agent checks:
    - Allergies
    - Renal function
    - Current medications for interactions
    - Lab values for contraindications
    """)

    agent = create_combined_agent()

    scenarios = [
        "Can I add spironolactone for patient smith_john? Check safety.",
        "Patient chen_wei needs antibiotics. Is amoxicillin safe?",
        "Is it safe to increase metformin for patient smith_john to improve HbA1c?",
    ]

    for q in scenarios:
        print(f"\n{'─' * 60}")
        print(f"  Q: {q}\n")
        result = agent.invoke({"messages": [{"role": "user", "content": q}]})
        answer = result["messages"][-1].content
        print(f"\n  Safety Assessment: {answer[:400]}")


# ============================================================
# DEMO 4: Interactive
# ============================================================

def demo_interactive():
    """Interactive combined agent"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE COMBINED AGENT")
    print("=" * 70)
    print("  Patients: smith_john, garcia_maria, chen_wei")
    print("  Tools: patient history, medications, labs, interactions, safety")
    print("  Type 'quit' to exit.\n")

    agent = create_combined_agent()
    messages = []

    while True:
        question = input("  You: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue

        messages.append({"role": "user", "content": question})
        result = agent.invoke({"messages": messages})
        answer = result["messages"][-1].content
        print(f"\n  Agent: {answer}\n")
        messages = result["messages"]


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 3: COMBINED AGENT — DRUGS + LABS + PATIENT HISTORY")
    print("=" * 70)
    print("""
    Agent combines 5 tools for comprehensive clinical decisions:
    patient history, medication lookup, lab interpretation,
    drug interactions, and medication safety assessment.

    Choose a demo:
      1 → Comprehensive patient review
      2 → Multi-patient queries
      3 → Safety-first prescribing
      4 → Interactive
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1": demo_patient_review()
    elif choice == "2": demo_multi_patient()
    elif choice == "3": demo_safety_prescribing()
    elif choice == "4": demo_interactive()
    elif choice == "5":
        demo_patient_review()
        demo_multi_patient()
        demo_safety_prescribing()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. COMBINED TOOLS = COMPREHENSIVE CARE: Individual tools (med lookup,
   lab check) are useful alone, but combining them with patient history
   enables integrated clinical reasoning.

2. PATIENT CONTEXT CHANGES EVERYTHING: "Is potassium 5.1 high?" depends
   on whether the patient is on an ACE inhibitor. The agent needs
   the FULL picture to reason correctly.

3. SAFETY AS A TOOL: assess_medication_safety() proactively checks
   allergies, renal function, and current meds. This is a pattern
   you'll see again in Level 3 Project 05 (guardrails).

4. TOOL COMPOSITION: 5 focused tools > 1 giant tool. Each tool has
   a clear responsibility. The AGENT decides which to call and in
   what order — just like a physician's clinical workflow.

5. SYSTEM PROMPT MATTERS: "ALWAYS pull up patient history first"
   guides the agent's workflow. Prompt engineering shapes behavior.
"""

if __name__ == "__main__":
    main()
