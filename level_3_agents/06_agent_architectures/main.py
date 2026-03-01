"""
Project 6: Agent Architectures — The Complete Catalog

This module teaches the MAJOR agent architecture patterns side-by-side.
Each demo uses the SAME healthcare scenario so you can compare directly.

Scenario: "55-year-old diabetic patient with new chest pain and elevated troponin"

You'll see how EACH architecture approaches this differently:
  1. ReAct         — step-by-step reasoning with tools
  2. Plan-Execute  — creates a plan first, then executes
  3. Reflection    — generates, critiques, and revises output
  4. Tool-Making   — creates new tools on the fly when needed

Architectures covered in exercises:
  Exercise 1 — Plan-and-Execute (detailed)
  Exercise 2 — Reflection / Self-Critique (detailed)
  Exercise 3 — Parallel Fan-Out / Map-Reduce (detailed)
  Exercise 4 — Hierarchical Agents (detailed)

Already covered in earlier projects:
  ReAct                → Project 01 (01_react_agent)
  Router/Supervisor    → Project 04 (04_multi_agent, Demo 2)
  Sequential Pipeline  → Project 04 (04_multi_agent, Demo 1)
  Agent Debate         → Project 04 (04_multi_agent, Demo 3)
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ============================================================
# Shared Healthcare Scenario (same for all architectures)
# ============================================================

CLINICAL_SCENARIO = """
Patient: 55-year-old male
Chief Complaint: Chest pain for 2 hours, radiating to left arm
History: Type 2 diabetes (10 years), hypertension, hyperlipidemia
Current Medications: Metformin 1000mg BID, Lisinopril 20mg daily, Atorvastatin 40mg daily
Vitals: BP 158/92, HR 98, SpO2 96%, Temp 37.1°C
Labs: Troponin I 0.45 ng/mL (elevated), Glucose 210 mg/dL, Creatinine 1.3, K+ 4.8
ECG: ST depression in leads V3-V6
"""

# ============================================================
# Shared Tools (used across architectures)
# ============================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_medication",
            "description": "Look up medication information including dosing, interactions, and contraindications",
            "parameters": {
                "type": "object",
                "properties": {
                    "medication_name": {"type": "string", "description": "Name of the medication"}
                },
                "required": ["medication_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_lab_value",
            "description": "Check if a lab value is normal, high, or critical",
            "parameters": {
                "type": "object",
                "properties": {
                    "lab_name": {"type": "string", "description": "Name of the lab test"},
                    "value": {"type": "number", "description": "The numeric lab value"},
                },
                "required": ["lab_name", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_clinical_guideline",
            "description": "Retrieve a clinical guideline for a specific condition",
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string", "description": "The clinical condition"}
                },
                "required": ["condition"],
            },
        },
    },
]

MEDICATION_DB = {
    "metformin": "Biguanide for T2DM. Dose: 500-2000mg/day. Caution: hold before contrast dye. Contraindicated if eGFR<30.",
    "lisinopril": "ACE inhibitor. Dose: 5-40mg/day. Monitor K+ and creatinine. Contraindicated in pregnancy.",
    "atorvastatin": "Statin, high-intensity. 40-80mg for ASCVD risk. Monitor LFTs. Avoid with grapefruit.",
    "aspirin": "Antiplatelet. 81-325mg for ACS. Contraindicated if active bleed, allergy.",
    "heparin": "Anticoagulant. IV drip for ACS. Monitor aPTT. Risk: bleeding, HIT.",
    "nitroglycerin": "Vasodilator for angina. 0.4mg SL PRN. Avoid with PDE5 inhibitors. May cause hypotension.",
    "morphine": "Opioid analgesic. 2-4mg IV for ACS pain. Caution: respiratory depression, hypotension.",
    "clopidogrel": "Antiplatelet (P2Y12 inhibitor). 300-600mg loading, then 75mg daily. For ACS/PCI.",
}

LAB_RANGES = {
    "troponin": {"normal": (0, 0.04), "elevated": (0.04, 0.4), "critical": (0.4, 100), "unit": "ng/mL"},
    "glucose": {"normal": (70, 100), "elevated": (100, 200), "critical": (200, 800), "unit": "mg/dL"},
    "creatinine": {"normal": (0.7, 1.2), "elevated": (1.2, 2.0), "critical": (2.0, 15), "unit": "mg/dL"},
    "potassium": {"normal": (3.5, 5.0), "elevated": (5.0, 5.5), "critical": (5.5, 10), "unit": "mEq/L"},
}

GUIDELINES = {
    "acute coronary syndrome": "ACS Protocol: 1) Aspirin 325mg STAT, 2) Heparin IV drip, 3) Nitroglycerin SL, 4) Troponin q3h, 5) Cardiology consult, 6) Consider PCI if STEMI or refractory symptoms. TIMI risk stratification.",
    "diabetes management": "Inpatient DM: 1) Hold metformin if contrast/surgery/AKI, 2) Start insulin sliding scale, 3) Target glucose 140-180, 4) A1c on admission if not recent, 5) Endocrine consult for uncontrolled.",
    "hypertensive urgency": "BP >180/120 without organ damage: 1) Oral agents (not IV), 2) Reduce by 25% over hours, 3) Resume home meds, 4) Reassess in 24-48h.",
    "chest pain": "Chest Pain Workup: 1) ECG within 10 min, 2) Troponin on arrival and at 3h/6h, 3) CXR, 4) Risk stratify (HEART score), 5) If ACS suspected: aspirin, nitro, heparin.",
}


def execute_tool(name: str, args: dict) -> str:
    """Execute a tool call and return the result."""
    if name == "lookup_medication":
        med = args["medication_name"].lower()
        return MEDICATION_DB.get(med, f"No data for '{med}'. Available: {', '.join(MEDICATION_DB.keys())}")
    elif name == "check_lab_value":
        lab = args["lab_name"].lower()
        value = args["value"]
        if lab not in LAB_RANGES:
            return f"Unknown lab: {lab}. Available: {', '.join(LAB_RANGES.keys())}"
        ranges = LAB_RANGES[lab]
        if value <= ranges["normal"][1]:
            status = "NORMAL"
        elif value <= ranges["elevated"][1]:
            status = "ELEVATED"
        else:
            status = "CRITICAL"
        return f"{lab}: {value} {ranges['unit']} → {status} (normal: {ranges['normal'][0]}-{ranges['normal'][1]})"
    elif name == "get_clinical_guideline":
        condition = args["condition"].lower()
        for key, guideline in GUIDELINES.items():
            if key in condition or condition in key:
                return guideline
        return f"No guideline for '{condition}'. Available: {', '.join(GUIDELINES.keys())}"
    return f"Unknown tool: {name}"


# ============================================================
# DEMO 1: ReAct Architecture (Quick Recap)
# ============================================================
# Already covered in depth in Project 01.
# Included here for side-by-side comparison.
# Pattern: Think → Act → Observe → Think → ... → Answer
# ============================================================

def demo_react():
    """ReAct: The LLM reasons and acts step-by-step."""
    print("\n" + "=" * 70)
    print("  ARCHITECTURE 1: ReAct (Reason + Act)")
    print("=" * 70)
    print("""
  Pattern: Think → Act → Observe → Think → ... → Answer
  The LLM decides what to do ONE STEP AT A TIME.
  No upfront planning — it figures things out as it goes.

       ┌──────────┐
       │  THINK   │ ◄─── "I need to check the troponin level"
       └────┬─────┘
            │
       ┌────▼─────┐
       │   ACT    │ ◄─── calls check_lab_value(troponin, 0.45)
       └────┬─────┘
            │
       ┌────▼─────┐
       │ OBSERVE  │ ◄─── "troponin 0.45 → CRITICAL"
       └────┬─────┘
            │
       ┌────▼─────┐
       │  THINK   │ ◄─── "Critical troponin + chest pain = ACS. Need guideline."
       └────┬─────┘
            │
           ...       (loop continues)
            │
       ┌────▼─────┐
       │  ANSWER  │ ◄─── Final clinical recommendation
       └──────────┘
  """)

    messages = [
        {"role": "system", "content": "You are an emergency medicine physician. Use tools to look up information. Reason step-by-step. Be concise."},
        {"role": "user", "content": f"Assess this patient and recommend next steps:\n{CLINICAL_SCENARIO}"},
    ]

    print("  Running ReAct agent...")
    steps = 0
    max_steps = 8

    while steps < max_steps:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0,
        )

        choice = response.choices[0]
        steps += 1

        if choice.finish_reason == "stop":
            # Agent is done reasoning
            print(f"\n  [ReAct completed in {steps} steps]")
            print(f"\n  FINAL ANSWER:\n  {'─' * 60}")
            for line in choice.message.content.split("\n"):
                print(f"  {line}")
            break

        if choice.message.tool_calls:
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                args = json.loads(tc.function.arguments)
                result = execute_tool(tc.function.name, args)
                print(f"    Step {steps}: {tc.function.name}({json.dumps(args)}) → {result[:80]}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            # No tool calls, just reasoning text
            if choice.message.content:
                messages.append({"role": "assistant", "content": choice.message.content})
                print(f"    Step {steps} (think): {choice.message.content[:100]}...")


# ============================================================
# DEMO 2: Plan-and-Execute Architecture
# ============================================================
# The agent creates a PLAN first, then executes each step.
# Different from ReAct — ReAct doesn't plan ahead.
# ============================================================

def demo_plan_and_execute():
    """Plan-and-Execute: Plan first, then execute steps in order."""
    print("\n" + "=" * 70)
    print("  ARCHITECTURE 2: Plan-and-Execute")
    print("=" * 70)
    print("""
  Pattern: PLAN (all steps upfront) → EXECUTE step 1 → EXECUTE step 2 → ... → SYNTHESIZE
  The LLM creates a complete plan BEFORE doing anything.

       ┌──────────────────┐
       │   PLAN            │ ◄─── "Here are 5 steps I need to take..."
       │   1. Check labs   │
       │   2. Get guideline│
       │   3. Check meds   │
       │   4. Assess risk  │
       │   5. Recommend    │
       └────────┬─────────┘
                │
       ┌────────▼─────────┐
       │  EXECUTE Step 1   │ ◄─── check_lab_value(troponin, 0.45)
       └────────┬─────────┘
                │
       ┌────────▼─────────┐
       │  EXECUTE Step 2   │ ◄─── get_clinical_guideline(ACS)
       └────────┬─────────┘
                │
               ...
                │
       ┌────────▼─────────┐
       │  SYNTHESIZE       │ ◄─── Combine all results into recommendation
       └──────────────────┘

  vs ReAct: ReAct decides each step after seeing the previous result.
  Plan-and-Execute commits to a plan upfront — more structured, more predictable.
  """)

    # Phase 1: PLANNING
    print("  Phase 1: PLANNING...")
    plan_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a clinical planner. Given a patient scenario, create a step-by-step "
                "plan of exactly what information to gather and decisions to make. "
                "Output a JSON array of steps. Each step has: 'step_number', 'action' (one of: "
                "'check_lab', 'lookup_medication', 'get_guideline', 'assess', 'recommend'), "
                "'description' (what this step does), and 'parameters' (dict of params needed). "
                "Output ONLY valid JSON, no explanation."
            )},
            {"role": "user", "content": f"Create a clinical assessment plan for:\n{CLINICAL_SCENARIO}"},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        plan_data = json.loads(plan_response.choices[0].message.content)
        # Handle both {"steps": [...]} and {"plan": [...]} formats
        if isinstance(plan_data, dict):
            plan = plan_data.get("steps") or plan_data.get("plan") or []
        else:
            plan = plan_data
    except json.JSONDecodeError:
        print("  Failed to parse plan. Using fallback.")
        plan = []

    print(f"\n  📋 PLAN ({len(plan)} steps):")
    for step in plan:
        num = step.get("step_number", "?")
        desc = step.get("description", step.get("action", "?"))
        print(f"    Step {num}: {desc}")

    # Phase 2: EXECUTION
    print(f"\n  Phase 2: EXECUTING plan...")
    results = []

    for step in plan:
        action = step.get("action", "").lower()
        params = step.get("parameters", {})
        num = step.get("step_number", "?")

        if "lab" in action or "check" in action:
            # Execute lab check
            lab_name = params.get("lab_name", params.get("name", ""))
            value = params.get("value", 0)
            if lab_name and value:
                result = execute_tool("check_lab_value", {"lab_name": lab_name, "value": value})
            else:
                result = f"Skipped (missing params for lab check: {params})"
        elif "medication" in action or "lookup" in action:
            med_name = params.get("medication_name", params.get("name", ""))
            if med_name:
                result = execute_tool("lookup_medication", {"medication_name": med_name})
            else:
                result = f"Skipped (missing medication name: {params})"
        elif "guideline" in action:
            condition = params.get("condition", "")
            if condition:
                result = execute_tool("get_clinical_guideline", {"condition": condition})
            else:
                result = f"Skipped (missing condition: {params})"
        else:
            result = f"[Reasoning step — will be handled in synthesis]"

        results.append({"step": num, "action": action, "result": result})
        print(f"    ✓ Step {num}: {result[:90]}{'...' if len(result) > 90 else ''}")

    # Phase 3: SYNTHESIS
    print(f"\n  Phase 3: SYNTHESIZING results...")
    results_text = "\n".join([f"Step {r['step']}: {r['result']}" for r in results])

    synthesis = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an emergency medicine physician. Synthesize the gathered information into a concise clinical recommendation. Be specific and actionable."},
            {"role": "user", "content": f"Patient:\n{CLINICAL_SCENARIO}\n\nGathered Information:\n{results_text}\n\nProvide your clinical assessment and recommendations."},
        ],
        temperature=0,
    )

    print(f"\n  FINAL RECOMMENDATION:\n  {'─' * 60}")
    for line in synthesis.choices[0].message.content.split("\n"):
        print(f"  {line}")


# ============================================================
# DEMO 3: Reflection / Self-Critique Architecture
# ============================================================
# Agent generates output, critiques it, then revises.
# Critical for high-quality outputs like clinical notes.
# ============================================================

def demo_reflection():
    """Reflection: Generate → Critique → Revise loop."""
    print("\n" + "=" * 70)
    print("  ARCHITECTURE 3: Reflection / Self-Critique")
    print("=" * 70)
    print("""
  Pattern: GENERATE → CRITIQUE → REVISE → (CRITIQUE → REVISE)... → FINAL
  The agent checks its own work and improves it iteratively.

       ┌──────────────┐
       │   GENERATE    │ ◄─── First draft of clinical note
       └──────┬───────┘
              │
       ┌──────▼───────┐
       │   CRITIQUE    │ ◄─── "Missing differential, no follow-up plan"
       │   (Score: 6)  │
       └──────┬───────┘
              │
       ┌──────▼───────┐
       │   REVISE      │ ◄─── Improved version addressing critique
       └──────┬───────┘
              │
       ┌──────▼───────┐
       │   CRITIQUE    │ ◄─── "Good, but missing allergy check"
       │   (Score: 8)  │
       └──────┬───────┘
              │
       ┌──────▼───────┐
       │   REVISE      │ ◄─── Final polished version
       │   (Score: 9)  │
       └──────────────┘

  Key: Score threshold determines when to stop (≥8 = good enough).
  This is how you get QUALITY — the agent is its own reviewer.
  """)

    # Phase 1: GENERATE initial clinical note
    print("  Phase 1: GENERATING initial clinical note...")
    generate_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are an emergency medicine physician. Write a clinical assessment note "
                "for this patient. Include: Chief Complaint, History, Vitals, Labs, Assessment, "
                "Differential Diagnosis, and Plan. Be thorough but concise."
            )},
            {"role": "user", "content": CLINICAL_SCENARIO},
        ],
        temperature=0.3,
    )
    draft = generate_response.choices[0].message.content
    print(f"\n  DRAFT (first {200} chars):\n  {'─' * 40}")
    print(f"  {draft[:200]}...")

    # Reflection loop
    max_revisions = 3
    quality_threshold = 8
    current_draft = draft

    for revision in range(max_revisions):
        # Phase 2: CRITIQUE
        print(f"\n  Phase 2.{revision + 1}: CRITIQUING (revision {revision + 1})...")
        critique_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a senior attending physician reviewing a clinical note. "
                    "Evaluate the note on these criteria:\n"
                    "1. Completeness (all sections present?)\n"
                    "2. Clinical accuracy (correct interpretation of labs/vitals?)\n"
                    "3. Differential diagnosis (appropriate breadth?)\n"
                    "4. Plan specificity (actionable, time-bound orders?)\n"
                    "5. Safety (medication interactions, contraindications checked?)\n\n"
                    "Respond with JSON: {\"score\": 1-10, \"strengths\": [...], \"weaknesses\": [...], \"suggestions\": [...]}"
                )},
                {"role": "user", "content": f"Patient scenario:\n{CLINICAL_SCENARIO}\n\nClinical note to review:\n{current_draft}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        try:
            critique = json.loads(critique_response.choices[0].message.content)
        except json.JSONDecodeError:
            critique = {"score": 7, "strengths": ["Good structure"], "weaknesses": ["Could not parse critique"], "suggestions": []}

        score = critique.get("score", 0)
        print(f"    Score: {score}/10")
        print(f"    Strengths: {', '.join(critique.get('strengths', [])[:2])}")
        print(f"    Weaknesses: {', '.join(critique.get('weaknesses', [])[:2])}")

        if score >= quality_threshold:
            print(f"\n  ✓ Score {score} ≥ threshold {quality_threshold}. Accepting note.")
            break

        # Phase 3: REVISE
        print(f"\n  Phase 3.{revision + 1}: REVISING based on critique...")
        revise_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an emergency medicine physician. Revise the clinical note to address all weaknesses and incorporate the suggestions. Keep the good parts."},
                {"role": "user", "content": (
                    f"Original note:\n{current_draft}\n\n"
                    f"Critique:\n{json.dumps(critique, indent=2)}\n\n"
                    f"Please revise the note to address the weaknesses."
                )},
            ],
            temperature=0.2,
        )
        current_draft = revise_response.choices[0].message.content
        print(f"    Revised! (new length: {len(current_draft)} chars)")

    print(f"\n  FINAL NOTE:\n  {'─' * 60}")
    for line in current_draft.split("\n"):
        print(f"  {line}")


# ============================================================
# DEMO 4: Tool-Making Agent
# ============================================================
# Agent generates and uses NEW tools at runtime.
# Impressive but risky — used for dynamic capabilities.
# ============================================================

def demo_tool_making():
    """Tool-Making: Agent creates new tools when existing ones aren't enough."""
    print("\n" + "=" * 70)
    print("  ARCHITECTURE 4: Tool-Making Agent")
    print("=" * 70)
    print("""
  Pattern: Agent encounters a gap → Generates a tool → Uses it
  The agent can extend its own capabilities at runtime.

       ┌───────────────────┐
       │ "I need to compute │
       │  HEART score but   │
       │  I don't have a    │
       │  tool for that"    │
       └────────┬──────────┘
                │
       ┌────────▼──────────┐
       │   GENERATE TOOL    │ ◄─── Writes Python function for HEART score
       │   (Python code)    │
       └────────┬──────────┘
                │
       ┌────────▼──────────┐
       │   VALIDATE TOOL    │ ◄─── Checks syntax, tests with sample input
       └────────┬──────────┘
                │
       ┌────────▼──────────┐
       │   USE NEW TOOL     │ ◄─── heart_score(age=55, troponin=0.45, ...)
       └──────────────────┘

  ⚠️ CAUTION: This involves executing LLM-generated code.
     Production systems use sandboxing (Docker, WASM) for safety.
     This demo uses safe, pre-validated calculations only.
  """)

    # Phase 1: Identify missing capability
    print("  Phase 1: Agent identifies missing capability...")
    identify_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a clinical agent. You have these tools: lookup_medication, check_lab_value, get_clinical_guideline. "
                "Given this patient, identify ONE clinical calculation or scoring tool that would be useful "
                "but is NOT available. Output JSON: {\"tool_name\": \"...\", \"description\": \"...\", "
                "\"reason\": \"why this would help\", \"inputs\": [{\"name\": \"...\", \"type\": \"...\"}], "
                "\"formula\": \"plain English description of how to calculate it\"}"
            )},
            {"role": "user", "content": CLINICAL_SCENARIO},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        tool_spec = json.loads(identify_response.choices[0].message.content)
    except json.JSONDecodeError:
        tool_spec = {
            "tool_name": "heart_score",
            "description": "Calculate HEART score for chest pain risk",
            "reason": "Need risk stratification for ACS",
            "inputs": [{"name": "age", "type": "int"}, {"name": "troponin", "type": "float"}],
            "formula": "HEART score: History + ECG + Age + Risk factors + Troponin"
        }

    print(f"    Missing tool: {tool_spec.get('tool_name', 'unknown')}")
    print(f"    Reason: {tool_spec.get('reason', 'N/A')}")
    print(f"    Inputs: {[i.get('name', '?') for i in tool_spec.get('inputs', [])]}")

    # Phase 2: Generate the tool (Python function)
    print("\n  Phase 2: Generating tool implementation...")
    generate_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a Python developer. Write a Python function that implements the described clinical tool. "
                "The function must:\n"
                "1. Take the specified inputs as parameters\n"
                "2. Return a string with the result and interpretation\n"
                "3. Use only built-in Python (no imports)\n"
                "4. Include input validation\n"
                "Output ONLY the Python function code, nothing else."
            )},
            {"role": "user", "content": json.dumps(tool_spec, indent=2)},
        ],
        temperature=0,
    )

    generated_code = generate_response.choices[0].message.content
    # Clean up markdown code blocks if present
    if "```python" in generated_code:
        generated_code = generated_code.split("```python")[1].split("```")[0]
    elif "```" in generated_code:
        generated_code = generated_code.split("```")[1].split("```")[0]

    print(f"\n  Generated code:\n  {'─' * 40}")
    for line in generated_code.strip().split("\n")[:15]:
        print(f"    {line}")
    if len(generated_code.strip().split("\n")) > 15:
        print(f"    ... ({len(generated_code.strip().split(chr(10)))} lines total)")

    # Phase 3: Validate and execute (with safety)
    print("\n  Phase 3: Validating and executing...")
    try:
        # Safety: compile to check syntax (does NOT execute)
        compile(generated_code.strip(), "<tool>", "exec")
        print("    ✓ Syntax valid")

        # Execute in a restricted namespace
        namespace = {}
        exec(generated_code.strip(), {"__builtins__": {"min": min, "max": max, "int": int, "float": float, "str": str, "round": round, "isinstance": isinstance, "ValueError": ValueError, "range": range, "len": len, "sum": sum}}, namespace)

        # Find the generated function
        tool_func = None
        for name, obj in namespace.items():
            if callable(obj):
                tool_func = obj
                break

        if tool_func:
            print(f"    ✓ Function '{tool_func.__name__}' created")

            # Phase 4: Use the tool
            print("\n  Phase 4: Using the dynamically-created tool...")

            # Let the LLM decide what arguments to pass
            use_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        f"Given this patient data, provide the arguments to call the function '{tool_func.__name__}'. "
                        f"The function accepts: {[i for i in tool_spec.get('inputs', [])]}. "
                        f"Output JSON with the argument names and values."
                    )},
                    {"role": "user", "content": CLINICAL_SCENARIO},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )

            try:
                tool_args = json.loads(use_response.choices[0].message.content)
                print(f"    Arguments: {tool_args}")
                result = tool_func(**tool_args)
                print(f"    Result: {result}")
            except Exception as e:
                print(f"    Execution error: {e}")
                print("    (This is expected — LLM-generated code is not always perfect!)")
        else:
            print("    ✗ No callable function found in generated code")

    except SyntaxError as e:
        print(f"    ✗ Syntax error: {e}")
        print("    (This happens — LLM-generated code needs validation)")

    print(f"\n  {'─' * 60}")
    print("  Key takeaway: Tool-making agents are powerful but require")
    print("  careful sandboxing. Never execute LLM-generated code in")
    print("  production without security boundaries (Docker, WASM, etc.).")


# ============================================================
# DEMO 5: Architecture Comparison — Same Question, 3 Approaches
# ============================================================

def demo_comparison():
    """Compare ReAct vs Plan-and-Execute vs Reflection side-by-side."""
    print("\n" + "=" * 70)
    print("  ARCHITECTURE COMPARISON: SAME QUESTION, DIFFERENT APPROACHES")
    print("=" * 70)
    print("""
  All 3 architectures answer: "What should we do for this patient?"
  Watch how each approaches the problem differently.

  ┌─────────────┬──────────────────┬──────────────────┐
  │   ReAct      │  Plan-Execute    │  Reflection      │
  ├─────────────┼──────────────────┼──────────────────┤
  │ Think        │ PLAN all steps   │ GENERATE draft   │
  │ Act          │ Execute step 1   │ CRITIQUE draft   │
  │ Observe      │ Execute step 2   │ REVISE draft     │
  │ Think        │ Execute step 3   │ CRITIQUE again   │
  │ Act          │ ...              │ REVISE again     │
  │ ...          │ SYNTHESIZE       │ (until good)     │
  │ Answer       │                  │                  │
  └─────────────┴──────────────────┴──────────────────┘
  """)

    question = f"Given this patient, what is the most urgent concern and what should be done in the next 30 minutes?\n{CLINICAL_SCENARIO}"

    # Quick ReAct
    print("  1️⃣  ReAct approach:")
    react = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an EM physician. Think step-by-step, use tools as needed. 3-4 sentences max."},
            {"role": "user", "content": question},
        ],
        tools=TOOL_SCHEMAS,
        temperature=0,
    )
    # Resolve tool calls if any
    if react.choices[0].message.tool_calls:
        msgs = [{"role": "system", "content": "You are an EM physician."}, {"role": "user", "content": question}, react.choices[0].message]
        for tc in react.choices[0].message.tool_calls:
            args = json.loads(tc.function.arguments)
            result = execute_tool(tc.function.name, args)
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        react = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, temperature=0)
    print(f"  {react.choices[0].message.content[:300]}\n")

    # Quick Plan-Execute
    print("  2️⃣  Plan-Execute approach:")
    plan_exec = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an EM physician. First state your 3-step plan, then execute it, then give your final recommendation. 5-6 sentences."},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    print(f"  {plan_exec.choices[0].message.content[:400]}\n")

    # Quick Reflection
    print("  3️⃣  Reflection approach:")
    draft = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an EM physician. Give your initial assessment. 3-4 sentences."},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
    critique = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a senior attending. In ONE sentence, identify what's missing from this assessment."},
            {"role": "user", "content": f"Scenario: {CLINICAL_SCENARIO}\n\nAssessment: {draft.choices[0].message.content}"},
        ],
        temperature=0,
    )
    revised = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an EM physician. Revise your assessment to address the critique. 4-5 sentences."},
            {"role": "user", "content": f"Your draft: {draft.choices[0].message.content}\n\nCritique: {critique.choices[0].message.content}\n\nRevise:"},
        ],
        temperature=0.2,
    )
    print(f"  Draft: {draft.choices[0].message.content[:150]}...")
    print(f"  Critique: {critique.choices[0].message.content[:150]}")
    print(f"  Revised: {revised.choices[0].message.content[:300]}")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  PROJECT 6: AGENT ARCHITECTURES — THE COMPLETE CATALOG")
    print("=" * 70)
    print("""
    Every agent is one of a few core patterns. This module teaches
    them all side-by-side so you can pick the right one for any task.

    Choose a demo:
      1 → ReAct (Reason + Act) — recap from Project 01
      2 → Plan-and-Execute — plan first, then execute
      3 → Reflection / Self-Critique — generate → critique → revise
      4 → Tool-Making Agent — creates new tools at runtime
      5 → Architecture Comparison — same question, 3 approaches
      6 → Run all demos (1-5)
    """)

    choice = input("  Enter choice (1-6): ").strip()

    demos = {
        "1": demo_react,
        "2": demo_plan_and_execute,
        "3": demo_reflection,
        "4": demo_tool_making,
        "5": demo_comparison,
    }

    if choice == "6":
        for demo in [demo_react, demo_plan_and_execute, demo_reflection, demo_tool_making, demo_comparison]:
            demo()
    elif choice in demos:
        demos[choice]()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. REACT: Best for simple tasks. The LLM figures things out as it goes.
   Strengths: Flexible, low token cost. Weaknesses: No upfront planning.

2. PLAN-AND-EXECUTE: Best for complex, multi-step tasks. Plans ahead.
   Strengths: Structured, predictable. Weaknesses: Slow start, rigid plan.

3. REFLECTION: Best for quality-critical outputs. Self-reviews and improves.
   Strengths: High quality. Weaknesses: Slow, expensive (multiple LLM calls).

4. TOOL-MAKING: Best when the agent needs NEW capabilities.
   Strengths: Ultra-flexible. Weaknesses: Security risk (executing generated code).

5. CHOOSING AN ARCHITECTURE:
   - Quick lookup? → ReAct
   - Multi-step procedure? → Plan-and-Execute
   - Clinical report? → Reflection
   - Need multiple perspectives? → Parallel Fan-Out (Exercise 3)
   - Complex case? → Hierarchical (Exercise 4)
   - Route to specialist? → Router/Supervisor (Project 04)
"""

if __name__ == "__main__":
    main()
