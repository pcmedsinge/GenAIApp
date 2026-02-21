"""
Project 1: ReAct Agent from Scratch
Build the core agent pattern (Reasoning + Acting) using ONLY OpenAI API.
No frameworks — understand exactly how agents work.

The pattern:
  THOUGHT:     LLM reasons about what to do next
  ACTION:      LLM decides to call a tool (uses function calling from Level 1!)
  OBSERVATION: Tool returns a result
  ... (repeat) ...
  FINAL:       LLM has enough info → gives the answer

Key Insight: This is function calling (Level 1) in a LOOP.
The magic is the LLM deciding WHEN to call tools and WHEN to stop.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# Healthcare Tools (Functions the Agent Can Call)
# ============================================================

def lookup_medication(medication_name: str) -> str:
    """Look up information about a medication"""
    medications = {
        "metformin": {
            "class": "Biguanide",
            "indication": "Type 2 Diabetes, first-line therapy",
            "dose": "500-2000mg daily with meals",
            "side_effects": "GI upset, diarrhea, nausea (usually improve over time)",
            "contraindications": "eGFR below 30, active liver disease, heavy alcohol use",
            "monitoring": "HbA1c every 3-6 months, annual B12 level, renal function"
        },
        "lisinopril": {
            "class": "ACE Inhibitor",
            "indication": "Hypertension, Heart Failure, Diabetic Nephropathy",
            "dose": "10-40mg daily",
            "side_effects": "Dry cough (10-15%), hyperkalemia, dizziness",
            "contraindications": "Pregnancy, bilateral renal artery stenosis, angioedema history",
            "monitoring": "Blood pressure, potassium, creatinine within 1-2 weeks of starting"
        },
        "amlodipine": {
            "class": "Calcium Channel Blocker",
            "indication": "Hypertension, Angina",
            "dose": "2.5-10mg daily",
            "side_effects": "Peripheral edema, dizziness, flushing",
            "contraindications": "Severe aortic stenosis",
            "monitoring": "Blood pressure, heart rate, edema assessment"
        },
        "apixaban": {
            "class": "Direct Oral Anticoagulant (DOAC)",
            "indication": "Atrial fibrillation, VTE treatment and prevention",
            "dose": "5mg BID standard; 2.5mg BID if age 80+ or weight 60kg or less or Cr 1.5+",
            "side_effects": "Bleeding, bruising",
            "contraindications": "Active major bleeding, mechanical heart valve",
            "monitoring": "Signs of bleeding, renal function annually"
        },
        "sertraline": {
            "class": "SSRI (Selective Serotonin Reuptake Inhibitor)",
            "indication": "Depression, Anxiety disorders, PTSD, OCD",
            "dose": "50-200mg daily",
            "side_effects": "Nausea, insomnia, sexual dysfunction, headache",
            "contraindications": "MAO inhibitor use within 14 days",
            "monitoring": "Mood assessment, suicidal ideation (especially age under 25)"
        }
    }

    med = medications.get(medication_name.lower())
    if med:
        return json.dumps(med, indent=2)
    return f"Medication '{medication_name}' not found in database. Available: {', '.join(medications.keys())}"


def check_lab_value(test_name: str, value: float) -> str:
    """Interpret a lab value and return clinical significance"""
    lab_ranges = {
        "hba1c": {"unit": "%", "normal": "below 5.7", "prediabetes": "5.7-6.4", "diabetes": "6.5 and above",
                   "interpretation": lambda v: "Normal" if v < 5.7 else "Prediabetes" if v < 6.5 else "Diabetes" if v < 8 else "Poorly controlled diabetes"},
        "gfr": {"unit": "mL/min", "normal": "90 or above",
                "interpretation": lambda v: "Normal" if v >= 90 else "Mildly decreased" if v >= 60 else "Moderate CKD (Stage 3)" if v >= 30 else "Severe CKD (Stage 4)" if v >= 15 else "Kidney failure (Stage 5)"},
        "potassium": {"unit": "mEq/L", "normal": "3.5-5.0",
                      "interpretation": lambda v: "Hypokalemia (LOW)" if v < 3.5 else "Normal" if v <= 5.0 else "Hyperkalemia (HIGH)" if v <= 6.0 else "Critical hyperkalemia (DANGEROUS)"},
        "creatinine": {"unit": "mg/dL", "normal": "0.7-1.3",
                       "interpretation": lambda v: "Low" if v < 0.7 else "Normal" if v <= 1.3 else "Elevated" if v <= 2.0 else "Significantly elevated"},
        "systolic_bp": {"unit": "mmHg", "normal": "below 120",
                        "interpretation": lambda v: "Optimal" if v < 120 else "Elevated" if v < 130 else "Stage 1 HTN" if v < 140 else "Stage 2 HTN" if v < 180 else "Hypertensive crisis"},
    }

    test = lab_ranges.get(test_name.lower())
    if test:
        interpretation = test["interpretation"](value)
        return json.dumps({
            "test": test_name,
            "value": value,
            "unit": test["unit"],
            "normal_range": test["normal"],
            "interpretation": interpretation
        }, indent=2)
    return f"Test '{test_name}' not recognized. Available: {', '.join(lab_ranges.keys())}"


def calculate_ckd_risk(gfr: float, albuminuria: float, age: int, has_diabetes: bool) -> str:
    """Calculate CKD risk level based on clinical factors"""
    # Simplified CKD risk assessment
    risk_score = 0
    factors = []

    if gfr < 30:
        risk_score += 3
        factors.append(f"GFR {gfr} (Stage 4-5 CKD)")
    elif gfr < 45:
        risk_score += 2
        factors.append(f"GFR {gfr} (Stage 3b CKD)")
    elif gfr < 60:
        risk_score += 1
        factors.append(f"GFR {gfr} (Stage 3a CKD)")

    if albuminuria > 300:
        risk_score += 2
        factors.append(f"Albuminuria {albuminuria} (Severely elevated)")
    elif albuminuria > 30:
        risk_score += 1
        factors.append(f"Albuminuria {albuminuria} (Moderately elevated)")

    if age > 65:
        risk_score += 1
        factors.append(f"Age {age} (elevated risk)")

    if has_diabetes:
        risk_score += 1
        factors.append("Diabetes present")

    risk_level = "LOW" if risk_score <= 1 else "MODERATE" if risk_score <= 3 else "HIGH" if risk_score <= 5 else "VERY HIGH"

    return json.dumps({
        "risk_level": risk_level,
        "risk_score": risk_score,
        "factors": factors,
        "recommendation": {
            "LOW": "Monitor annually",
            "MODERATE": "Monitor every 6 months, optimize BP and diabetes control",
            "HIGH": "Monitor every 3 months, nephrology referral, start ACE/ARB",
            "VERY HIGH": "Urgent nephrology referral, prepare for renal replacement"
        }[risk_level]
    }, indent=2)


# Tool definitions for OpenAI function calling (from Level 1!)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_medication",
            "description": "Look up detailed information about a medication including class, indication, dosing, side effects, contraindications, and monitoring requirements",
            "parameters": {
                "type": "object",
                "properties": {
                    "medication_name": {"type": "string", "description": "Name of the medication (e.g., metformin, lisinopril, apixaban)"}
                },
                "required": ["medication_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_lab_value",
            "description": "Interpret a lab test value and determine if it is normal, high, or low. Available tests: hba1c, gfr, potassium, creatinine, systolic_bp",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_name": {"type": "string", "description": "Lab test name (hba1c, gfr, potassium, creatinine, systolic_bp)"},
                    "value": {"type": "number", "description": "The numeric test value"}
                },
                "required": ["test_name", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_ckd_risk",
            "description": "Calculate chronic kidney disease risk level based on GFR, albuminuria, age, and diabetes status",
            "parameters": {
                "type": "object",
                "properties": {
                    "gfr": {"type": "number", "description": "GFR in mL/min"},
                    "albuminuria": {"type": "number", "description": "Urine albumin-to-creatinine ratio in mg/g"},
                    "age": {"type": "integer", "description": "Patient age in years"},
                    "has_diabetes": {"type": "boolean", "description": "Whether patient has diabetes"}
                },
                "required": ["gfr", "albuminuria", "age", "has_diabetes"]
            }
        }
    }
]

# Map function names to actual Python functions
TOOL_FUNCTIONS = {
    "lookup_medication": lambda args: lookup_medication(args["medication_name"]),
    "check_lab_value": lambda args: check_lab_value(args["test_name"], args["value"]),
    "calculate_ckd_risk": lambda args: calculate_ckd_risk(
        args["gfr"], args["albuminuria"], args["age"], args["has_diabetes"]
    ),
}


# ============================================================
# The ReAct Agent Loop
# ============================================================

def run_agent(user_question, max_steps=5, verbose=True):
    """
    The ReAct agent loop — THIS IS THE CORE OF ALL AI AGENTS.

    1. Send question + tools to LLM
    2. If LLM returns tool_calls → execute them, add results to messages
    3. Send updated messages back to LLM
    4. Repeat until LLM gives final answer (no more tool calls)
    """

    if verbose:
        print(f"\n{'─' * 70}")
        print(f"🤖 AGENT STARTING")
        print(f"   Question: \"{user_question}\"")
        print(f"   Available tools: {', '.join(TOOL_FUNCTIONS.keys())}")
        print(f"   Max steps: {max_steps}")
        print(f"{'─' * 70}")

    messages = [
        {
            "role": "system",
            "content": """You are a clinical decision support agent. You have access to tools
for looking up medications, interpreting lab values, and calculating disease risk.

When given a clinical question:
1. Think about what information you need
2. Use the appropriate tools to gather data
3. Reason about the results
4. Provide a clear clinical recommendation

Always explain your reasoning. Use tools when you need specific clinical data.
If you have enough information, provide your final answer directly.
Disclaimer: For educational purposes only."""
        },
        {"role": "user", "content": user_question}
    ]

    step = 0
    while step < max_steps:
        step += 1

        if verbose:
            print(f"\n📍 Step {step}:")

        # Call LLM with tools
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"  # LLM decides whether to use tools
        )

        assistant_message = response.choices[0].message

        # Check if LLM wants to call tools
        if assistant_message.tool_calls:
            if verbose:
                print(f"   🧠 THOUGHT: Agent decides to use tool(s)")

            # Add assistant message (with tool calls) to conversation
            messages.append(assistant_message)

            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                if verbose:
                    print(f"   🔧 ACTION: {func_name}({json.dumps(func_args)})")

                # Execute the tool
                if func_name in TOOL_FUNCTIONS:
                    result = TOOL_FUNCTIONS[func_name](func_args)
                else:
                    result = f"Error: Unknown tool '{func_name}'"

                if verbose:
                    # Show truncated observation
                    display_result = result[:200] + "..." if len(result) > 200 else result
                    print(f"   👁️ OBSERVATION: {display_result}")

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        else:
            # No tool calls — LLM is giving final answer
            final_answer = assistant_message.content

            if verbose:
                print(f"   ✅ FINAL ANSWER (after {step} step{'s' if step > 1 else ''}):")
                print(f"\n{final_answer}")

            return final_answer, step, messages

    # Max steps reached
    if verbose:
        print(f"\n   ⚠️ Max steps ({max_steps}) reached. Getting final answer...")

    messages.append({"role": "user", "content": "Please provide your best answer with the information gathered so far."})
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content, step, messages


# ============================================================
# DEMO 1: ReAct Loop Step by Step
# ============================================================

def demo_react_explained():
    """Walk through the ReAct loop with verbose output"""
    print("\n" + "=" * 70)
    print("DEMO 1: ReAct AGENT LOOP (Step by Step)")
    print("=" * 70)
    print("""
💡 Watch the agent THINK, ACT, and OBSERVE in real time.
   This is EXACTLY what LangChain/LangGraph agents do under the hood.
""")

    question = "A 65-year-old patient with diabetes has a GFR of 38 and is currently on metformin. Is this safe?"
    answer, steps, _ = run_agent(question, verbose=True)

    print(f"""
{'─' * 70}
💡 WHAT HAPPENED:
   1. Agent received the question
   2. Agent decided to check lab value (GFR = 38)
   3. Agent decided to look up metformin contraindications
   4. Agent REASONED: GFR 38 + metformin contraindication at GFR < 30
   5. Agent gave a clinical recommendation

   This is the ReAct pattern:
   THOUGHT → ACTION → OBSERVATION → THOUGHT → ACTION → ... → FINAL ANSWER
""")


# ============================================================
# DEMO 2: Medical Lookup Agent
# ============================================================

def demo_medical_agent():
    """Interactive medical lookup agent"""
    print("\n" + "=" * 70)
    print("DEMO 2: MEDICAL LOOKUP AGENT")
    print("=" * 70)

    questions = [
        "What are the monitoring requirements for a patient starting apixaban?",
        "Patient has potassium of 5.8 and is on lisinopril. What should I do?",
        "Calculate CKD risk for a 72-year-old diabetic with GFR 42 and albuminuria 150.",
    ]

    print("\n💬 Running preset clinical questions...\n")

    for q in questions:
        answer, steps, _ = run_agent(q, verbose=True)
        print(f"\n{'=' * 70}")


# ============================================================
# DEMO 3: Agent vs Direct LLM
# ============================================================

def demo_agent_vs_direct():
    """Compare agent reasoning with direct LLM response"""
    print("\n" + "=" * 70)
    print("DEMO 3: AGENT vs DIRECT LLM")
    print("=" * 70)

    question = "Is it safe to give sertraline to a 22-year-old patient with depression and a potassium level of 3.2?"

    print(f"\n❓ Clinical Question:\n   \"{question}\"\n")

    # Direct LLM (no tools)
    print("🔴 DIRECT LLM (no tools, just training knowledge):")
    direct = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical assistant. Answer the clinical question briefly."},
            {"role": "user", "content": question}
        ],
        max_tokens=300
    )
    print(f"   {direct.choices[0].message.content}\n")

    # Agent (with tools)
    print("🟢 AGENT (with clinical tools):")
    answer, steps, _ = run_agent(question, verbose=True)

    print(f"""
{'─' * 70}
💡 THE DIFFERENCE:
   🔴 Direct LLM: Gives general advice from training data
   🟢 Agent: Looks up SPECIFIC medication data and lab interpretations,
      then reasons about the combination

   The agent's answer is:
   • More specific (actual drug data, not generic knowledge)
   • More reliable (grounded in tool outputs)
   • More thorough (checks multiple factors)
   • Traceable (you can see what tools it used)
""")


# ============================================================
# DEMO 4: Interactive Agent
# ============================================================

def demo_interactive():
    """Ask your own questions to the agent"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE AGENT")
    print("=" * 70)
    print("\n💬 Ask clinical questions! The agent has these tools:")
    print("   • lookup_medication (metformin, lisinopril, amlodipine, apixaban, sertraline)")
    print("   • check_lab_value (hba1c, gfr, potassium, creatinine, systolic_bp)")
    print("   • calculate_ckd_risk (GFR + albuminuria + age + diabetes)")
    print("   Type 'quit' to exit\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue
        run_agent(question, verbose=True)
        print()


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n🤖 Level 3, Project 1: ReAct Agent from Scratch")
    print("=" * 70)
    print("Build the core agent pattern — no frameworks needed!\n")

    print("Choose a demo:")
    print("1. ReAct loop explained (verbose step-by-step)")
    print("2. Medical lookup agent (preset clinical questions)")
    print("3. Agent vs Direct LLM (see the difference)")
    print("4. Interactive agent (ask your own questions)")
    print("5. Run all demos (1-3) then interactive")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_react_explained()
    elif choice == "2":
        demo_medical_agent()
    elif choice == "3":
        demo_agent_vs_direct()
    elif choice == "4":
        demo_interactive()
    elif choice == "5":
        demo_react_explained()
        demo_medical_agent()
        demo_agent_vs_direct()
        demo_interactive()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY TAKEAWAYS
{'=' * 70}

🔄 THE ReAct PATTERN:
   THOUGHT → ACTION → OBSERVATION → THOUGHT → ... → FINAL ANSWER
   This is the foundation of ALL agent frameworks!

🧠 HOW AGENTS DECIDE:
   • LLM receives: question + available tools (function definitions)
   • LLM returns: tool_calls (if it needs info) or content (if it's ready)
   • YOUR code: executes the tool, feeds result back
   • Loop continues until LLM has enough info

🔑 KEY INSIGHTS:
   • Agents = function calling in a loop (you learned function calling in Level 1!)
   • The LLM decides WHEN to use tools and WHEN to stop
   • tool_choice="auto" lets the LLM decide; "required" forces tool use
   • max_steps prevents infinite loops

🏥 HEALTHCARE VALUE:
   • Agent checks SPECIFIC data (not just training knowledge)
   • Reasoning is traceable (you see every step)
   • Multiple data sources combined in one answer
   • Safer than direct LLM for clinical questions

🎯 NEXT: Move to 02_langchain_agents to learn the LangChain
   framework, which wraps this pattern with powerful features!
""")


if __name__ == "__main__":
    main()
