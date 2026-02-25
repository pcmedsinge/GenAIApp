"""
Exercise 1: Add a New Tool — Allergy Checker

Skills practiced:
- Defining a new tool function with structured output
- Adding the tool's JSON schema to the TOOLS list
- Registering the tool in TOOL_FUNCTIONS dispatch map
- Watching the agent discover and use the new tool autonomously

Key insight: To add a tool to a ReAct agent, you need THREE things:
  1. The Python function (does the actual work)
  2. The JSON schema (tells the LLM what the tool does and expects)
  3. The dispatch mapping (connects the LLM's call to your function)
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# Existing Tools (from main.py)
# ============================================================

def lookup_medication(medication_name: str) -> str:
    """Look up information about a medication"""
    medications = {
        "metformin": {"class": "Biguanide", "indication": "Type 2 Diabetes", "dose": "500-2000mg daily", "side_effects": "GI upset, diarrhea", "contraindications": "eGFR below 30, liver disease"},
        "lisinopril": {"class": "ACE Inhibitor", "indication": "HTN, HF", "dose": "10-40mg daily", "side_effects": "Dry cough, hyperkalemia", "contraindications": "Pregnancy, angioedema history"},
        "amlodipine": {"class": "CCB", "indication": "HTN, Angina", "dose": "2.5-10mg daily", "side_effects": "Peripheral edema, dizziness", "contraindications": "Severe aortic stenosis"},
        "apixaban": {"class": "DOAC", "indication": "AFib, VTE", "dose": "5mg BID (2.5mg if age 80+/wt ≤60kg/Cr≥1.5)", "side_effects": "Bleeding", "contraindications": "Active bleeding, mechanical valve"},
        "sertraline": {"class": "SSRI", "indication": "Depression, Anxiety", "dose": "50-200mg daily", "side_effects": "Nausea, insomnia", "contraindications": "MAOi within 14 days"},
        "amoxicillin": {"class": "Penicillin antibiotic", "indication": "Bacterial infections", "dose": "250-500mg TID", "side_effects": "Diarrhea, rash, nausea", "contraindications": "Penicillin allergy"},
        "ibuprofen": {"class": "NSAID", "indication": "Pain, inflammation, fever", "dose": "200-800mg TID", "side_effects": "GI bleeding, renal impairment", "contraindications": "CKD, GI ulcer, aspirin allergy"},
        "sulfa_antibiotics": {"class": "Sulfonamide antibiotic", "indication": "UTI, bacterial infections", "dose": "Varies by agent", "side_effects": "Rash, GI upset", "contraindications": "Sulfa allergy"},
    }
    med = medications.get(medication_name.lower())
    if med:
        return json.dumps(med, indent=2)
    return f"Medication '{medication_name}' not found. Available: {', '.join(medications.keys())}"


def check_lab_value(test_name: str, value: float) -> str:
    """Interpret a lab value"""
    lab_ranges = {
        "hba1c": {"unit": "%", "interpretation": lambda v: "Normal" if v < 5.7 else "Prediabetes" if v < 6.5 else "Diabetes"},
        "gfr": {"unit": "mL/min", "interpretation": lambda v: "Normal" if v >= 90 else "Mild CKD" if v >= 60 else "Moderate CKD" if v >= 30 else "Severe CKD" if v >= 15 else "Kidney failure"},
        "potassium": {"unit": "mEq/L", "interpretation": lambda v: "LOW" if v < 3.5 else "Normal" if v <= 5.0 else "HIGH" if v <= 6.0 else "CRITICAL HIGH"},
    }
    test = lab_ranges.get(test_name.lower())
    if test:
        return json.dumps({"test": test_name, "value": value, "unit": test["unit"], "interpretation": test["interpretation"](value)}, indent=2)
    return f"Test '{test_name}' not found. Available: {', '.join(lab_ranges.keys())}"


# ============================================================
# NEW TOOL: Allergy Checker
# ============================================================

def check_allergies(patient_allergies: str, proposed_medication: str) -> str:
    """
    Check if a proposed medication is safe given the patient's known allergies.

    This simulates a real allergy-checking system. In production, this would
    query an allergy database or EHR system.
    """
    # Cross-reactivity and allergy rules
    allergy_rules = {
        "penicillin": {
            "unsafe_meds": ["amoxicillin", "ampicillin", "piperacillin"],
            "caution_meds": ["cephalosporins"],  # ~2% cross-reactivity
            "message": "Penicillin allergy: {med} is a penicillin-class antibiotic. "
                       "CONTRAINDICATED. Consider azithromycin or fluoroquinolone alternative."
        },
        "sulfa": {
            "unsafe_meds": ["sulfa_antibiotics", "sulfamethoxazole"],
            "caution_meds": ["thiazide_diuretics"],  # Structural similarity
            "message": "Sulfa allergy: {med} contains sulfonamide. "
                       "CONTRAINDICATED. Consider alternative antibiotic class."
        },
        "nsaid": {
            "unsafe_meds": ["ibuprofen", "naproxen", "aspirin", "ketorolac"],
            "caution_meds": [],
            "message": "NSAID/Aspirin allergy: {med} is an NSAID. "
                       "CONTRAINDICATED. Use acetaminophen for pain. "
                       "Also avoid aspirin and other NSAIDs."
        },
        "ace_inhibitor": {
            "unsafe_meds": ["lisinopril", "enalapril", "ramipril"],
            "caution_meds": [],
            "message": "ACE inhibitor allergy (angioedema): {med} is an ACEi. "
                       "CONTRAINDICATED. Use ARB with caution or consider CCB."
        },
        "statin": {
            "unsafe_meds": [],
            "caution_meds": ["atorvastatin", "simvastatin", "rosuvastatin"],
            "message": "Statin intolerance: Try lower dose, different statin, "
                       "or every-other-day dosing. Consider PCSK9 inhibitor if all statins fail."
        },
    }

    # Parse allergies (comma-separated)
    patient_allergy_list = [a.strip().lower() for a in patient_allergies.split(",")]
    med_lower = proposed_medication.lower()

    alerts = []
    for allergy in patient_allergy_list:
        rule = allergy_rules.get(allergy)
        if rule:
            if med_lower in rule["unsafe_meds"]:
                alerts.append({
                    "severity": "CONTRAINDICATED",
                    "allergy": allergy,
                    "medication": proposed_medication,
                    "detail": rule["message"].format(med=proposed_medication)
                })
            elif med_lower in rule["caution_meds"]:
                alerts.append({
                    "severity": "CAUTION",
                    "allergy": allergy,
                    "medication": proposed_medication,
                    "detail": f"Possible cross-reactivity between {allergy} allergy and {proposed_medication}. "
                              f"Use with caution; monitor for allergic reaction."
                })

    if alerts:
        return json.dumps({"status": "ALERT", "alerts": alerts}, indent=2)
    else:
        return json.dumps({
            "status": "SAFE",
            "message": f"No known allergy conflict between reported allergies "
                       f"({patient_allergies}) and {proposed_medication}."
        }, indent=2)


# ============================================================
# Tool Definitions (JSON schemas for OpenAI function calling)
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_medication",
            "description": "Look up medication info: class, dosing, side effects, contraindications",
            "parameters": {
                "type": "object",
                "properties": {
                    "medication_name": {"type": "string", "description": "Medication name (e.g., metformin, lisinopril)"}
                },
                "required": ["medication_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_lab_value",
            "description": "Interpret a lab test value. Tests: hba1c, gfr, potassium",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_name": {"type": "string", "description": "Lab test name"},
                    "value": {"type": "number", "description": "Numeric test value"}
                },
                "required": ["test_name", "value"]
            }
        }
    },
    # *** NEW TOOL: Allergy Checker ***
    {
        "type": "function",
        "function": {
            "name": "check_allergies",
            "description": "Check if a proposed medication is safe given the patient's known allergies. "
                           "Detects contraindicated medications and cross-reactivity risks. "
                           "Known allergy categories: penicillin, sulfa, nsaid, ace_inhibitor, statin.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_allergies": {
                        "type": "string",
                        "description": "Comma-separated list of patient allergies (e.g., 'penicillin, sulfa')"
                    },
                    "proposed_medication": {
                        "type": "string",
                        "description": "The medication being considered (e.g., 'amoxicillin')"
                    }
                },
                "required": ["patient_allergies", "proposed_medication"]
            }
        }
    },
]

# Dispatch map — connects function names to Python functions
TOOL_FUNCTIONS = {
    "lookup_medication": lambda args: lookup_medication(args["medication_name"]),
    "check_lab_value": lambda args: check_lab_value(args["test_name"], args["value"]),
    "check_allergies": lambda args: check_allergies(args["patient_allergies"], args["proposed_medication"]),
}


# ============================================================
# ReAct Agent Loop (same as main.py)
# ============================================================

def run_agent(user_question, max_steps=6, verbose=True):
    """The ReAct agent loop with our new allergy tool"""
    if verbose:
        print(f"\n{'─' * 70}")
        print(f"  AGENT STARTING")
        print(f"  Question: \"{user_question}\"")
        print(f"  Available tools: {', '.join(TOOL_FUNCTIONS.keys())}")
        print(f"{'─' * 70}")

    messages = [
        {
            "role": "system",
            "content": """You are a clinical decision support agent with tools for medication lookup,
lab interpretation, and ALLERGY CHECKING.

IMPORTANT: When a patient has known allergies, ALWAYS check proposed medications
against their allergies before recommending them. Patient safety is the top priority.

Use tools to gather specific data, reason about results, and provide safe recommendations.
Disclaimer: Educational purposes only."""
        },
        {"role": "user", "content": user_question}
    ]

    step = 0
    while step < max_steps:
        step += 1
        if verbose:
            print(f"\n  Step {step}:")

        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=TOOLS, tool_choice="auto"
        )
        assistant_message = response.choices[0].message

        if assistant_message.tool_calls:
            messages.append(assistant_message)
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                if verbose:
                    print(f"    ACTION: {func_name}({json.dumps(func_args)})")

                result = TOOL_FUNCTIONS.get(func_name, lambda a: "Unknown tool")(func_args)

                if verbose:
                    display = result[:250] + "..." if len(result) > 250 else result
                    print(f"    OBSERVATION: {display}")

                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        else:
            final_answer = assistant_message.content
            if verbose:
                print(f"    FINAL ANSWER (after {step} steps):")
                print(f"\n{final_answer}")
            return final_answer, step, messages

    messages.append({"role": "user", "content": "Provide your best answer with available information."})
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content, step, messages


# ============================================================
# DEMO 1: Allergy Tool in Action
# ============================================================

def demo_allergy_tool():
    """Show the agent using the new allergy tool"""
    print("\n" + "=" * 70)
    print("DEMO 1: ALLERGY CHECKER TOOL IN ACTION")
    print("=" * 70)
    print("""
    The agent now has check_allergies in its tool belt.
    Watch it autonomously decide to check allergies before recommending meds.
    """)

    questions = [
        "Patient has a penicillin allergy and needs antibiotics for a sinus infection. "
        "Can they take amoxicillin?",

        "A patient allergic to NSAIDs has chronic knee pain. "
        "What pain medication can I prescribe? Check ibuprofen safety first.",

        "72-year-old with penicillin and sulfa allergies needs treatment. "
        "Check if amoxicillin and sulfa_antibiotics are safe.",
    ]

    for q in questions:
        answer, steps, _ = run_agent(q, verbose=True)
        print(f"\n{'=' * 70}")


# ============================================================
# DEMO 2: Multi-Tool Allergy Scenario
# ============================================================

def demo_multi_tool_scenario():
    """Agent uses allergy tool + medication lookup + lab together"""
    print("\n" + "=" * 70)
    print("DEMO 2: MULTI-TOOL SCENARIO (Allergies + Meds + Labs)")
    print("=" * 70)
    print("""
    Complex case requiring multiple tools. Watch the agent chain them.
    """)

    question = (
        "65-year-old patient with ACE inhibitor allergy (angioedema), "
        "hypertension (BP 152/94), GFR of 55, and potassium of 5.3. "
        "They need blood pressure medication. Check if lisinopril is safe, "
        "then look up amlodipine as an alternative."
    )

    answer, steps, _ = run_agent(question, verbose=True)

    print(f"""
{'─' * 70}
  KEY LEARNING: The agent used:
    1. check_allergies — found lisinopril contraindicated
    2. check_lab_value — reviewed GFR and potassium
    3. lookup_medication — researched amlodipine alternative
  All three tools combined to give a SAFER recommendation.
""")


# ============================================================
# DEMO 3: Testing Tool Directly
# ============================================================

def demo_tool_direct():
    """Test the allergy tool directly to verify it works"""
    print("\n" + "=" * 70)
    print("DEMO 3: DIRECT TOOL TESTING")
    print("=" * 70)
    print("""
    Before trusting the agent, test the tool function directly.
    This is good practice — validate tools independently.
    """)

    test_cases = [
        ("penicillin", "amoxicillin", "Should be CONTRAINDICATED"),
        ("sulfa", "sulfa_antibiotics", "Should be CONTRAINDICATED"),
        ("nsaid", "ibuprofen", "Should be CONTRAINDICATED"),
        ("penicillin", "sertraline", "Should be SAFE (unrelated)"),
        ("ace_inhibitor", "lisinopril", "Should be CONTRAINDICATED"),
        ("penicillin, sulfa", "amoxicillin", "Should catch penicillin allergy"),
    ]

    for allergies, med, expected in test_cases:
        result = json.loads(check_allergies(allergies, med))
        status = result["status"]
        icon = "🚨" if status == "ALERT" else "✅"
        print(f"\n  {icon} Allergies: {allergies:<20} Med: {med:<20} → {status}")
        print(f"     Expected: {expected}")
        if status == "ALERT":
            for alert in result["alerts"]:
                print(f"     {alert['severity']}: {alert['detail'][:80]}...")


# ============================================================
# DEMO 4: Interactive Session
# ============================================================

def demo_interactive():
    """Ask your own questions — agent now has allergy checking"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE (with Allergy Tool)")
    print("=" * 70)
    print("  Tools: lookup_medication, check_lab_value, check_allergies")
    print("  Try: 'Patient has penicillin allergy, can they take amoxicillin?'")
    print("  Type 'quit' to exit\n")

    while True:
        question = input("  Your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue
        run_agent(question, verbose=True)


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 1: ADD A NEW TOOL — ALLERGY CHECKER")
    print("=" * 70)
    print("""
    This exercise adds a check_allergies tool to the ReAct agent.
    Three things needed to add a tool:
      1. Python function (the logic)
      2. JSON schema (tells LLM when/how to use it)
      3. Dispatch mapping (connects LLM call → function)

    Choose a demo:
      1 → Allergy tool in action (agent uses it autonomously)
      2 → Multi-tool scenario (allergies + meds + labs combined)
      3 → Direct tool testing (test the function independently)
      4 → Interactive session
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1":
        demo_allergy_tool()
    elif choice == "2":
        demo_multi_tool_scenario()
    elif choice == "3":
        demo_tool_direct()
    elif choice == "4":
        demo_interactive()
    elif choice == "5":
        demo_allergy_tool()
        demo_multi_tool_scenario()
        demo_tool_direct()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. THREE PIECES TO ADD A TOOL: Function + JSON schema + dispatch map.
   Miss any one and the agent can't use the tool.

2. SCHEMA IS CRITICAL: The function description in the JSON schema is
   what the LLM reads to decide WHEN to use the tool. A vague description
   means the agent won't know when to call it.

3. AGENT DISCOVERS TOOLS: You don't tell the agent "use check_allergies."
   The agent reads the tool list and DECIDES to use it when relevant.
   This is the power of the ReAct pattern.

4. TOOL COMPOSITION: The agent chains multiple tools (allergy check →
   medication lookup → lab interpretation) to build a complete picture.

5. SAFETY PATTERN: Allergy checking before prescribing is a core patient
   safety principle. Adding it as a tool lets the agent enforce it.
"""

if __name__ == "__main__":
    main()
