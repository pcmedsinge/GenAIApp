"""
Exercise 1: Plan-and-Execute Agent — Treatment Planning

Skills practiced:
- Two-phase agent design: PLAN then EXECUTE
- Structured plan generation (LLM outputs a JSON plan)
- Step-by-step execution with tool calls
- Plan revision when new information changes the plan
- Comparing Plan-and-Execute vs ReAct for the same task

Key insight: ReAct decides what to do ONE step at a time.
  Plan-and-Execute creates the ENTIRE plan upfront, then executes.
  This is better for complex, multi-step tasks where you need
  a structured approach — like creating a treatment plan.

  Think of it like a doctor's workflow:
  ReAct    = "Let me check this... okay now let me check that..."
  Plan-Exe = "Here's my 6-step workup. Let me execute each step."

Architecture:
  ┌────────────────────┐
  │     PLANNER        │ ◄── LLM creates structured plan
  │  "1. Check labs    │
  │   2. Get guideline │
  │   3. Review meds   │     Plan is a JSON array of steps
  │   4. Risk score    │     Each step has: action, tool, params
  │   5. Decide        │
  │   6. Orders"       │
  └────────┬───────────┘
           │
  ┌────────▼───────────┐
  │     EXECUTOR       │ ◄── Executes steps 1-by-1
  │  (loops through    │     Calls tools as specified
  │   plan steps)      │     Collects results
  └────────┬───────────┘
           │
  ┌────────▼───────────┐
  │   RE-PLANNER       │ ◄── Optional: revise plan if
  │  (checks if plan   │     results contradict expectations
  │   needs updating)  │
  └────────┬───────────┘
           │
  ┌────────▼───────────┐
  │   SYNTHESIZER      │ ◄── Combines all results into
  │                    │     a final recommendation
  └────────────────────┘
"""

import os
import json
from datetime import datetime
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

client = OpenAI()


# ============================================================
# Clinical Knowledge Base (tools for the agent)
# ============================================================

MEDICATION_DB = {
    "metformin": {
        "class": "Biguanide", "use": "Type 2 Diabetes",
        "dose": "500-2000mg/day", "route": "oral",
        "cautions": "Hold before contrast, avoid if eGFR<30",
        "interactions": ["contrast dye", "alcohol"],
    },
    "lisinopril": {
        "class": "ACE Inhibitor", "use": "Hypertension, CHF, diabetic nephropathy",
        "dose": "5-40mg/day", "route": "oral",
        "cautions": "Monitor K+ and creatinine, avoid in pregnancy",
        "interactions": ["potassium supplements", "NSAIDs"],
    },
    "aspirin": {
        "class": "Antiplatelet", "use": "ACS, stroke prevention",
        "dose": "81-325mg/day", "route": "oral",
        "cautions": "Bleeding risk, avoid if active GI bleed",
        "interactions": ["warfarin", "NSAIDs"],
    },
    "heparin": {
        "class": "Anticoagulant", "use": "ACS, DVT/PE, AFib",
        "dose": "60 units/kg bolus then 12 units/kg/hr", "route": "IV",
        "cautions": "Monitor aPTT, bleeding risk, HIT",
        "interactions": ["aspirin", "warfarin", "NSAIDs"],
    },
    "nitroglycerin": {
        "class": "Nitrate vasodilator", "use": "Angina, ACS",
        "dose": "0.4mg SL q5min x3", "route": "sublingual/IV",
        "cautions": "Hypotension, avoid with PDE5 inhibitors (sildenafil)",
        "interactions": ["sildenafil", "tadalafil"],
    },
    "atorvastatin": {
        "class": "Statin", "use": "Hyperlipidemia, ASCVD risk reduction",
        "dose": "40-80mg/day", "route": "oral",
        "cautions": "Monitor LFTs, myalgia, avoid with grapefruit",
        "interactions": ["gemfibrozil", "cyclosporine"],
    },
    "clopidogrel": {
        "class": "P2Y12 inhibitor", "use": "ACS, post-PCI",
        "dose": "300-600mg load, then 75mg/day", "route": "oral",
        "cautions": "Bleeding risk, hold 5-7 days before surgery",
        "interactions": ["omeprazole (reduces efficacy)"],
    },
    "insulin glargine": {
        "class": "Long-acting insulin", "use": "Diabetes (basal coverage)",
        "dose": "0.2-0.4 units/kg/day", "route": "subcutaneous",
        "cautions": "Hypoglycemia, renal dose adjustment",
        "interactions": ["sulfonylureas", "beta-blockers (mask hypo symptoms)"],
    },
}

LAB_DATABASE = {
    "troponin": {"unit": "ng/mL", "normal": "0-0.04", "elevated": "0.04-0.4", "critical": ">0.4"},
    "glucose": {"unit": "mg/dL", "normal": "70-100", "elevated": "100-200", "critical": ">200"},
    "hba1c": {"unit": "%", "normal": "<5.7", "elevated": "5.7-6.4", "critical": ">6.5"},
    "creatinine": {"unit": "mg/dL", "normal": "0.7-1.2", "elevated": "1.2-2.0", "critical": ">2.0"},
    "potassium": {"unit": "mEq/L", "normal": "3.5-5.0", "elevated": "5.0-5.5", "critical": ">5.5"},
    "bnp": {"unit": "pg/mL", "normal": "<100", "elevated": "100-400", "critical": ">400"},
    "inr": {"unit": "ratio", "normal": "0.8-1.2", "elevated": "1.2-3.0", "critical": ">3.0"},
}

CLINICAL_GUIDELINES = {
    "acute coronary syndrome": {
        "initial": ["Aspirin 325mg STAT", "Heparin IV bolus + drip", "Nitroglycerin SL", "12-lead ECG", "Serial troponins (0h, 3h, 6h)"],
        "risk_stratify": "Use TIMI or HEART score",
        "high_risk": "Cardiology consult for emergent cath/PCI",
        "moderate_risk": "Admit, serial troponins, stress test",
        "monitoring": ["Telemetry", "Vitals q1h", "Repeat ECG with any change"],
    },
    "diabetes inpatient": {
        "initial": ["Hold metformin if AKI, contrast, or NPO", "Start basal-bolus insulin", "Finger sticks QAC and QHS"],
        "target_glucose": "140-180 mg/dL",
        "hypoglycemia_protocol": "D50 IV if <70 and symptomatic",
        "monitoring": ["Finger stick glucose QAC+HS", "A1c if not done in 3 months"],
    },
    "hypertension management": {
        "initial": ["Continue home antihypertensives unless contraindicated"],
        "target": "BP <130/80",
        "acute_hypertension": "If >180/120 with organ damage: IV labetalol or nicardipine",
        "monitoring": ["BP q4h", "Urine output", "Renal function"],
    },
}


def execute_tool(tool_name: str, params: dict) -> str:
    """Execute a tool call and return the result."""
    if tool_name == "lookup_medication":
        med = params.get("medication", "").lower()
        if med in MEDICATION_DB:
            info = MEDICATION_DB[med]
            return json.dumps(info, indent=2)
        return f"Unknown medication: {med}. Available: {', '.join(MEDICATION_DB.keys())}"

    elif tool_name == "interpret_lab":
        lab = params.get("lab_name", "").lower()
        value = params.get("value", 0)
        if lab in LAB_DATABASE:
            ref = LAB_DATABASE[lab]
            # Determine status based on value ranges
            normal_hi = float(ref["normal"].replace("<", "").replace(">", "").split("-")[-1]) if "-" in ref["normal"] else float(ref["normal"].replace("<", "").replace(">", ""))
            elevated_hi = float(ref["elevated"].replace("<", "").replace(">", "").split("-")[-1]) if "-" in ref["elevated"] else float(ref["elevated"].replace("<", "").replace(">", ""))
            if value <= normal_hi:
                status = "NORMAL"
            elif value <= elevated_hi:
                status = "ELEVATED"
            else:
                status = "CRITICAL"
            return f"{lab}: {value} {ref['unit']} → {status} (ref: normal {ref['normal']}, elevated {ref['elevated']}, critical {ref['critical']})"
        return f"Unknown lab: {lab}. Available: {', '.join(LAB_DATABASE.keys())}"

    elif tool_name == "get_guideline":
        condition = params.get("condition", "").lower()
        for key, guideline in CLINICAL_GUIDELINES.items():
            if key in condition or condition in key:
                return json.dumps(guideline, indent=2)
        return f"No guideline for: {condition}. Available: {', '.join(CLINICAL_GUIDELINES.keys())}"

    elif tool_name == "check_interaction":
        drug1 = params.get("drug1", "").lower()
        drug2 = params.get("drug2", "").lower()
        for d in [drug1, drug2]:
            if d in MEDICATION_DB:
                interactions = [i.lower() for i in MEDICATION_DB[d].get("interactions", [])]
                other = drug2 if d == drug1 else drug1
                if any(other in inter or inter in other for inter in interactions):
                    return f"⚠️ INTERACTION: {drug1} + {drug2} — check cautions for {d}"
        return f"No significant interaction found between {drug1} and {drug2}."

    return f"Unknown tool: {tool_name}"


AVAILABLE_TOOLS = [
    {"name": "lookup_medication", "description": "Look up medication info (class, dose, cautions, interactions)", "params": ["medication"]},
    {"name": "interpret_lab", "description": "Interpret a lab result (normal/elevated/critical)", "params": ["lab_name", "value"]},
    {"name": "get_guideline", "description": "Get clinical guideline for a condition", "params": ["condition"]},
    {"name": "check_interaction", "description": "Check for drug-drug interaction", "params": ["drug1", "drug2"]},
]


# ============================================================
# Plan-and-Execute Agent
# ============================================================

class PlanAndExecuteAgent:
    """
    A two-phase agent:
    1. PLANNER: Creates a structured step-by-step plan
    2. EXECUTOR: Executes each step, calling tools as needed
    3. RE-PLANNER: Optionally revises the plan based on findings
    4. SYNTHESIZER: Combines all results into a final answer
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.plan = []
        self.results = []
        self.execution_log = []

    def create_plan(self, scenario: str) -> list[dict]:
        """Phase 1: Create a structured plan."""
        tools_desc = "\n".join([f"  - {t['name']}: {t['description']} (params: {', '.join(t['params'])})" for t in AVAILABLE_TOOLS])

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are a clinical planner. Given a patient scenario, create a detailed "
                    "step-by-step treatment planning workflow.\n\n"
                    f"Available tools:\n{tools_desc}\n\n"
                    "Output a JSON object with key 'steps' containing an array. Each step:\n"
                    "{\n"
                    '  "step_number": 1,\n'
                    '  "description": "what this step does",\n'
                    '  "tool": "tool name or null if reasoning step",\n'
                    '  "params": {"param_name": "value"} or null,\n'
                    '  "depends_on": [list of step numbers this depends on] or [],\n'
                    '  "rationale": "why this step matters"\n'
                    "}\n\n"
                    "Create 6-10 concrete steps. Include both tool-calling steps and "
                    "reasoning/decision steps. Be specific about parameters."
                )},
                {"role": "user", "content": scenario},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(response.choices[0].message.content)
            self.plan = data.get("steps", [])
        except json.JSONDecodeError:
            self.plan = []

        return self.plan

    def execute_plan(self, scenario: str) -> list[dict]:
        """Phase 2: Execute each step in the plan."""
        self.results = []

        for step in self.plan:
            step_num = step.get("step_number", "?")
            tool = step.get("tool")
            params = step.get("params", {})
            description = step.get("description", "")

            log_entry = {
                "step": step_num,
                "description": description,
                "tool": tool,
                "params": params,
                "result": None,
                "timestamp": datetime.now().isoformat(),
            }

            if tool and params:
                # Tool-calling step
                result = execute_tool(tool, params)
                log_entry["result"] = result
            else:
                # Reasoning step — use LLM
                context = "\n".join([
                    f"Step {r['step']}: {r['description']} → {r['result']}"
                    for r in self.results if r['result']
                ])
                reasoning = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a clinical reasoning agent. Given the gathered information, perform the requested analysis. Be concise (2-3 sentences)."},
                        {"role": "user", "content": f"Scenario: {scenario}\n\nGathered so far:\n{context}\n\nTask: {description}"},
                    ],
                    temperature=0,
                )
                log_entry["result"] = reasoning.choices[0].message.content

            self.results.append(log_entry)
            self.execution_log.append(log_entry)

        return self.results

    def check_replan(self, scenario: str) -> bool:
        """Phase 3: Check if the plan needs revision based on results."""
        results_text = "\n".join([
            f"Step {r['step']}: {r['description']} → {str(r['result'])[:150]}"
            for r in self.results
        ])

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are a clinical quality reviewer. Given the original plan and "
                    "execution results, determine if the plan needs revision.\n"
                    "Output JSON: {\"needs_revision\": true/false, \"reason\": \"...\", "
                    "\"additional_steps\": [{\"description\": \"...\", \"tool\": \"...\", \"params\": {}}]}"
                )},
                {"role": "user", "content": f"Scenario: {scenario}\n\nResults:\n{results_text}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        try:
            review = json.loads(response.choices[0].message.content)
            if review.get("needs_revision") and review.get("additional_steps"):
                print(f"\n    🔄 RE-PLAN: {review.get('reason', 'Plan needs additional steps')}")
                # Add new steps to the plan
                new_steps = review["additional_steps"]
                start_num = len(self.plan) + 1
                for i, step in enumerate(new_steps):
                    step["step_number"] = start_num + i
                    if "tool" not in step:
                        step["tool"] = None
                    if "params" not in step:
                        step["params"] = None
                    self.plan.append(step)

                # Execute new steps
                for step in new_steps:
                    tool = step.get("tool")
                    params = step.get("params", {})
                    log_entry = {
                        "step": step["step_number"],
                        "description": step["description"],
                        "tool": tool,
                        "params": params,
                        "result": None,
                        "timestamp": datetime.now().isoformat(),
                    }
                    if tool and params:
                        log_entry["result"] = execute_tool(tool, params)
                    else:
                        log_entry["result"] = "(reasoning step — handled in synthesis)"
                    self.results.append(log_entry)
                return True
            return False
        except json.JSONDecodeError:
            return False

    def synthesize(self, scenario: str) -> str:
        """Phase 4: Synthesize all results into a final recommendation."""
        results_text = "\n".join([
            f"Step {r['step']}: {r['description']}\n  Result: {str(r['result'])[:200]}"
            for r in self.results
        ])

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are a senior attending physician. Synthesize all the gathered "
                    "information into a comprehensive treatment plan. Structure your response as:\n"
                    "1. ASSESSMENT (2-3 sentences)\n"
                    "2. IMMEDIATE ORDERS (next 30 min)\n"
                    "3. TREATMENT PLAN (next 24h)\n"
                    "4. MONITORING\n"
                    "5. KEY CONSIDERATIONS (diabetes management, drug interactions, etc.)"
                )},
                {"role": "user", "content": f"Patient:\n{scenario}\n\nWorkup Results:\n{results_text}"},
            ],
            temperature=0,
        )

        return response.choices[0].message.content

    def run(self, scenario: str, verbose: bool = True) -> str:
        """Run the full Plan-and-Execute pipeline."""
        if verbose:
            print("\n  ╔══════════════════════════════════════╗")
            print("  ║  PLAN-AND-EXECUTE AGENT              ║")
            print("  ╚══════════════════════════════════════╝")

        # Phase 1: Plan
        if verbose:
            print("\n  📋 Phase 1: PLANNING...")
        plan = self.create_plan(scenario)
        if verbose:
            for step in plan:
                tool_info = f" [tool: {step.get('tool')}]" if step.get('tool') else " [reasoning]"
                print(f"    Step {step.get('step_number', '?')}: {step.get('description', '?')}{tool_info}")

        # Phase 2: Execute
        if verbose:
            print(f"\n  ⚙️ Phase 2: EXECUTING {len(plan)} steps...")
        results = self.execute_plan(scenario)
        if verbose:
            for r in results:
                result_preview = str(r['result'])[:100] if r['result'] else "N/A"
                print(f"    ✓ Step {r['step']}: {result_preview}")

        # Phase 3: Re-plan check
        if verbose:
            print(f"\n  🔍 Phase 3: CHECKING if plan needs revision...")
        revised = self.check_replan(scenario)
        if verbose and not revised:
            print("    No revision needed.")

        # Phase 4: Synthesize
        if verbose:
            print(f"\n  📝 Phase 4: SYNTHESIZING final recommendation...")
        final = self.synthesize(scenario)
        if verbose:
            print(f"\n  {'═' * 60}")
            print("  FINAL TREATMENT PLAN:")
            print(f"  {'═' * 60}")
            for line in final.split("\n"):
                print(f"  {line}")

        return final


# ============================================================
# Demo Scenarios
# ============================================================

SCENARIO_ACS = """
Patient: 55-year-old male
Chief Complaint: Chest pain for 2 hours, radiating to left arm
History: Type 2 diabetes (10 years), hypertension, hyperlipidemia
Current Medications: Metformin 1000mg BID, Lisinopril 20mg daily, Atorvastatin 40mg daily
Vitals: BP 158/92, HR 98, SpO2 96%, Temp 37.1°C
Labs: Troponin I 0.45 ng/mL, Glucose 210 mg/dL, Creatinine 1.3, K+ 4.8
ECG: ST depression in leads V3-V6
""".strip()

SCENARIO_DKA = """
Patient: 38-year-old female
Chief Complaint: Nausea, vomiting, abdominal pain for 12 hours
History: Type 1 diabetes (20 years), hypothyroidism
Current Medications: Insulin glargine 30u daily, Insulin lispro per carb counting, Levothyroxine 100mcg
Vitals: BP 95/60, HR 115, RR 28 (Kussmaul), SpO2 99%, Temp 37.8°C
Labs: Glucose 420 mg/dL, pH 7.18, HCO3 10, K+ 5.8, Creatinine 1.6, BUN 28
Urine: Large ketones
""".strip()


# ============================================================
# Demo Functions
# ============================================================

def demo_basic_plan_execute():
    """Basic Plan-and-Execute for ACS management."""
    print("\n" + "=" * 70)
    print("  DEMO 1: PLAN-AND-EXECUTE FOR ACS MANAGEMENT")
    print("=" * 70)
    print("""
  The agent will:
  1. Create a structured plan for managing the ACS patient
  2. Execute each step (calling tools for info gathering)
  3. Check if the plan needs revision
  4. Synthesize a final treatment recommendation
  """)

    agent = PlanAndExecuteAgent()
    agent.run(SCENARIO_ACS)


def demo_plan_comparison():
    """Compare plans for different scenarios."""
    print("\n" + "=" * 70)
    print("  DEMO 2: PLAN COMPARISON — ACS vs DKA")
    print("=" * 70)
    print("""
  Same architecture, different scenarios → different plans.
  Watch how the planner adapts to the clinical context.
  """)

    print("\n  --- Scenario 1: ACS ---")
    agent1 = PlanAndExecuteAgent()
    plan1 = agent1.create_plan(SCENARIO_ACS)
    print(f"  Plan ({len(plan1)} steps):")
    for s in plan1:
        print(f"    {s.get('step_number', '?')}. {s.get('description', '?')}")

    print("\n  --- Scenario 2: DKA ---")
    agent2 = PlanAndExecuteAgent()
    plan2 = agent2.create_plan(SCENARIO_DKA)
    print(f"  Plan ({len(plan2)} steps):")
    for s in plan2:
        print(f"    {s.get('step_number', '?')}. {s.get('description', '?')}")

    print(f"\n  Key difference: ACS plan focuses on cardiac workup + anticoagulation.")
    print(f"  DKA plan focuses on fluid resuscitation + insulin + electrolyte correction.")


def demo_react_vs_plan():
    """Side-by-side: ReAct vs Plan-and-Execute."""
    print("\n" + "=" * 70)
    print("  DEMO 3: ReAct vs PLAN-AND-EXECUTE — SIDE BY SIDE")
    print("=" * 70)
    print("""
  Same question to both architectures. Watch the difference:
  - ReAct: Figures things out step-by-step (no upfront plan)
  - Plan-Execute: Plans everything first, then executes
  """)

    # Plan-and-Execute
    print("\n  ═══ PLAN-AND-EXECUTE ═══")
    agent = PlanAndExecuteAgent()
    plan = agent.create_plan(SCENARIO_ACS)
    print(f"  Created plan with {len(plan)} steps BEFORE doing anything.")
    for s in plan[:4]:
        print(f"    {s.get('step_number')}: {s.get('description', '')[:70]}")
    if len(plan) > 4:
        print(f"    ... and {len(plan) - 4} more steps")
    agent.execute_plan(SCENARIO_ACS)
    pe_result = agent.synthesize(SCENARIO_ACS)
    print(f"\n  Result (first 200 chars):\n  {pe_result[:200]}...")

    # ReAct
    print("\n\n  ═══ ReAct ═══")
    print("  No upfront plan — decides each step after seeing the previous result.")

    tools = [
        {"type": "function", "function": {"name": "lookup_medication", "description": "Look up medication info", "parameters": {"type": "object", "properties": {"medication": {"type": "string"}}, "required": ["medication"]}}},
        {"type": "function", "function": {"name": "interpret_lab", "description": "Interpret a lab value", "parameters": {"type": "object", "properties": {"lab_name": {"type": "string"}, "value": {"type": "number"}}, "required": ["lab_name", "value"]}}},
        {"type": "function", "function": {"name": "get_guideline", "description": "Get clinical guideline", "parameters": {"type": "object", "properties": {"condition": {"type": "string"}}, "required": ["condition"]}}},
    ]

    messages = [
        {"role": "system", "content": "You are an EM physician. Use tools to gather info, then give a treatment recommendation. Be concise."},
        {"role": "user", "content": f"Assess and create a treatment plan:\n{SCENARIO_ACS}"},
    ]

    react_steps = 0
    for _ in range(6):
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools, temperature=0)
        choice = resp.choices[0]
        react_steps += 1

        if choice.finish_reason == "stop":
            react_result = choice.message.content
            break

        if choice.message.tool_calls:
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                args = json.loads(tc.function.arguments)
                result = execute_tool(tc.function.name, args)
                print(f"    Step {react_steps}: {tc.function.name}({list(args.values())}) → {result[:70]}...")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        elif choice.message.content:
            react_result = choice.message.content
            break
    else:
        react_result = "Max steps reached"

    print(f"\n  Result (first 200 chars):\n  {react_result[:200]}...")

    # Comparison
    print(f"\n\n  ═══ COMPARISON ═══")
    print(f"  Plan-Execute: {len(plan)} planned steps, structured execution")
    print(f"  ReAct: {react_steps} adaptive steps, discovered along the way")
    print(f"\n  Plan-Execute is better when:")
    print(f"    • Task has known, repeatable structure")
    print(f"    • You need an audit trail of the planned approach")
    print(f"    • Multiple steps can be parallelized")
    print(f"\n  ReAct is better when:")
    print(f"    • Task is exploratory (don't know what you'll find)")
    print(f"    • Each step depends heavily on the previous result")
    print(f"    • Speed matters more than structure")


def demo_interactive():
    """Interactive Plan-and-Execute agent."""
    print("\n" + "=" * 70)
    print("  DEMO 4: INTERACTIVE PLAN-AND-EXECUTE")
    print("=" * 70)
    print("""
  Enter a patient scenario and watch the agent plan then execute.
  Type 'quit' to exit.
  """)

    while True:
        print("\n  Enter patient scenario (or 'quit'):")
        scenario = input("  > ").strip()

        if scenario.lower() in ['quit', 'exit', 'q']:
            break

        if len(scenario) < 20:
            print("  Please provide more detail (at least a chief complaint and some history).")
            continue

        agent = PlanAndExecuteAgent()
        agent.run(scenario)

        print(f"\n  Execution log: {len(agent.execution_log)} steps executed")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 1: PLAN-AND-EXECUTE AGENT")
    print("=" * 70)
    print("""
    The Plan-and-Execute pattern: create a structured plan FIRST,
    then execute each step. Better than ReAct for complex tasks.

    Choose a demo:
      1 → Basic Plan-and-Execute (ACS case)
      2 → Plan comparison (ACS vs DKA — different plans)
      3 → ReAct vs Plan-and-Execute (side-by-side)
      4 → Interactive (enter your own scenario)
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1":
        demo_basic_plan_execute()
    elif choice == "2":
        demo_plan_comparison()
    elif choice == "3":
        demo_react_vs_plan()
    elif choice == "4":
        demo_interactive()
    elif choice == "5":
        demo_basic_plan_execute()
        demo_plan_comparison()
        demo_react_vs_plan()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. PLAN-AND-EXECUTE separates WHAT to do from HOW to do it.
   The planner creates the strategy; the executor handles tactics.

2. STRUCTURED PLANS are auditable. You can review the plan before
   execution — critical in healthcare where wrong actions have consequences.

3. RE-PLANNING handles surprises. If a lab comes back unexpected,
   the agent can revise its plan (unlike a static script).

4. PLAN-AND-EXECUTE vs ReAct:
   - Plan-Execute: Better for KNOWN procedures (ACS protocol, DKA workup)
   - ReAct: Better for EXPLORATORY tasks (unusual presentations)

5. PRODUCTION PATTERN: Most clinical decision support systems use
   Plan-and-Execute because treatment protocols ARE plans.
   The LLM adapts the protocol to the specific patient.

6. DEPENDENCY TRACKING: Steps can declare dependencies on other steps.
   This enables parallel execution of independent steps in production.
"""

if __name__ == "__main__":
    main()
