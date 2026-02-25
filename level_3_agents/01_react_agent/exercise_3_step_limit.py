"""
Exercise 3: Maximum Step Limit — Preventing Infinite Loops

Skills practiced:
- Adding safety limits to the agent loop
- Handling max-step scenarios gracefully
- Understanding why agents sometimes loop (ambiguous questions, tool failures)
- Comparing behavior with different step limits

Key insight: Without a max step limit, an agent could loop forever —
  calling tools repeatedly without making progress. This is a real
  production concern, similar to how infinite loops crash programs.
  Healthcare systems MUST have safety limits.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# Healthcare Tools (from main.py)
# ============================================================

def lookup_medication(medication_name: str) -> str:
    medications = {
        "metformin": {"class": "Biguanide", "dose": "500-2000mg daily", "contraindications": "eGFR<30"},
        "lisinopril": {"class": "ACE Inhibitor", "dose": "10-40mg daily", "contraindications": "Pregnancy"},
        "amlodipine": {"class": "CCB", "dose": "2.5-10mg daily", "contraindications": "Severe aortic stenosis"},
        "apixaban": {"class": "DOAC", "dose": "5mg BID", "contraindications": "Active bleeding"},
        "sertraline": {"class": "SSRI", "dose": "50-200mg daily", "contraindications": "MAOi within 14 days"},
    }
    med = medications.get(medication_name.lower())
    if med:
        return json.dumps(med, indent=2)
    return f"Not found. Available: {', '.join(medications.keys())}"


def check_lab_value(test_name: str, value: float) -> str:
    tests = {
        "hba1c": lambda v: "Normal" if v < 5.7 else "Prediabetes" if v < 6.5 else "Diabetes",
        "gfr": lambda v: "Normal" if v >= 90 else "Mild CKD" if v >= 60 else "Moderate CKD" if v >= 30 else "Severe CKD",
        "potassium": lambda v: "LOW" if v < 3.5 else "Normal" if v <= 5.0 else "HIGH",
    }
    fn = tests.get(test_name.lower())
    if fn:
        return json.dumps({"test": test_name, "value": value, "interpretation": fn(value)})
    return f"Not found. Available: {', '.join(tests.keys())}"


def calculate_ckd_risk(gfr: float, albuminuria: float, age: int, has_diabetes: bool) -> str:
    risk_score = 0
    if gfr < 30: risk_score += 3
    elif gfr < 60: risk_score += 1
    if albuminuria > 300: risk_score += 2
    elif albuminuria > 30: risk_score += 1
    if age > 65: risk_score += 1
    if has_diabetes: risk_score += 1
    level = "LOW" if risk_score <= 1 else "MODERATE" if risk_score <= 3 else "HIGH" if risk_score <= 5 else "VERY HIGH"
    return json.dumps({"risk_level": level, "risk_score": risk_score})


TOOLS = [
    {"type": "function", "function": {"name": "lookup_medication", "description": "Look up medication info", "parameters": {"type": "object", "properties": {"medication_name": {"type": "string"}}, "required": ["medication_name"]}}},
    {"type": "function", "function": {"name": "check_lab_value", "description": "Interpret a lab value (hba1c, gfr, potassium)", "parameters": {"type": "object", "properties": {"test_name": {"type": "string"}, "value": {"type": "number"}}, "required": ["test_name", "value"]}}},
    {"type": "function", "function": {"name": "calculate_ckd_risk", "description": "Calculate CKD risk from GFR, albuminuria, age, diabetes status", "parameters": {"type": "object", "properties": {"gfr": {"type": "number"}, "albuminuria": {"type": "number"}, "age": {"type": "integer"}, "has_diabetes": {"type": "boolean"}}, "required": ["gfr", "albuminuria", "age", "has_diabetes"]}}},
]

TOOL_FUNCTIONS = {
    "lookup_medication": lambda args: lookup_medication(args["medication_name"]),
    "check_lab_value": lambda args: check_lab_value(args["test_name"], args["value"]),
    "calculate_ckd_risk": lambda args: calculate_ckd_risk(args["gfr"], args["albuminuria"], args["age"], args["has_diabetes"]),
}


# ============================================================
# Agent with Configurable Step Limit and Tracking
# ============================================================

def run_agent_with_limit(user_question, max_steps=5, verbose=True):
    """
    ReAct agent with step limit, timeout handling, and step tracking.

    Returns detailed telemetry about what happened:
    - How many steps taken
    - Whether max_steps was hit
    - Which tools were called at each step
    - Whether the agent completed normally
    """
    if verbose:
        print(f"\n{'─' * 70}")
        print(f"  AGENT (max_steps={max_steps})")
        print(f"  Q: \"{user_question[:80]}\"")
        print(f"{'─' * 70}")

    messages = [
        {
            "role": "system",
            "content": """You are a clinical decision support agent. Use tools when needed.
If you have enough information, provide your answer immediately.
Be efficient — avoid unnecessary tool calls. Educational purposes only."""
        },
        {"role": "user", "content": user_question}
    ]

    step = 0
    step_log = []
    completed = False

    while step < max_steps:
        step += 1

        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=TOOLS, tool_choice="auto"
        )
        assistant_message = response.choices[0].message

        if assistant_message.tool_calls:
            messages.append(assistant_message)
            tools_this_step = []
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                tools_this_step.append(f"{func_name}({json.dumps(func_args)})")

                if verbose:
                    print(f"    Step {step}: TOOL → {func_name}({json.dumps(func_args)})")

                result = TOOL_FUNCTIONS.get(func_name, lambda a: "Unknown")(func_args)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

            step_log.append({"step": step, "action": "tool_call", "tools": tools_this_step})
        else:
            final_answer = assistant_message.content
            step_log.append({"step": step, "action": "final_answer"})
            completed = True

            if verbose:
                print(f"    Step {step}: FINAL ANSWER")
                if final_answer:
                    print(f"\n    {final_answer[:300]}...")

            return {
                "answer": final_answer,
                "steps": step,
                "max_steps": max_steps,
                "completed": True,
                "hit_limit": False,
                "step_log": step_log,
            }

    # Max steps hit — force a final answer
    if verbose:
        print(f"\n    ⚠️  MAX STEPS ({max_steps}) REACHED — forcing final answer")

    messages.append({
        "role": "user",
        "content": "You've reached the maximum number of steps. "
                   "Please provide your best answer with the information gathered so far. "
                   "If you don't have enough data, say so."
    })
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    forced_answer = response.choices[0].message.content
    step_log.append({"step": step + 1, "action": "forced_answer"})

    if verbose:
        print(f"\n    (Forced) {forced_answer[:300]}...")

    return {
        "answer": forced_answer,
        "steps": step,
        "max_steps": max_steps,
        "completed": True,
        "hit_limit": True,
        "step_log": step_log,
    }


# ============================================================
# DEMO 1: Step Limits in Action
# ============================================================

def demo_step_limits():
    """Show how different step limits affect the agent"""
    print("\n" + "=" * 70)
    print("DEMO 1: STEP LIMITS IN ACTION")
    print("=" * 70)
    print("""
    Same question, different max_steps. Watch what happens when
    the agent runs out of steps before finishing.
    """)

    # Complex question that needs multiple tool calls
    question = (
        "Patient is 70 years old with diabetes, GFR of 35, albuminuria of 250. "
        "Calculate their CKD risk, check if metformin is safe, "
        "and interpret their GFR level."
    )

    print(f"\n  Question: \"{question[:80]}...\"\n")

    limits = [1, 2, 3, 5]
    results = []

    for limit in limits:
        print(f"\n{'═' * 60}")
        print(f"  max_steps = {limit}")
        print(f"{'═' * 60}")
        result = run_agent_with_limit(question, max_steps=limit, verbose=True)
        results.append(result)

    # Summary table
    print(f"\n\n{'═' * 70}")
    print("  COMPARISON TABLE")
    print(f"{'═' * 70}")
    print(f"  {'Max Steps':>10} {'Actual':>7} {'Hit Limit':>10} {'Tools Used':>12}")
    print(f"  {'─'*10} {'─'*7} {'─'*10} {'─'*12}")
    for r in results:
        tool_count = sum(1 for s in r["step_log"] if s["action"] == "tool_call")
        print(f"  {r['max_steps']:>10} {r['steps']:>7} {'YES ⚠️' if r['hit_limit'] else 'No':>10} {tool_count:>12}")

    print("""
    INSIGHT: Too few steps → incomplete answers. Too many → wasted cost.
    The right limit depends on your use case complexity.
    For most clinical queries, 5-8 steps is sufficient.
    """)


# ============================================================
# DEMO 2: Simple vs Complex Questions
# ============================================================

def demo_simple_vs_complex():
    """Show that simple questions need fewer steps"""
    print("\n" + "=" * 70)
    print("DEMO 2: SIMPLE vs COMPLEX QUESTIONS")
    print("=" * 70)
    print("""
    Simple questions complete in 1-2 steps.
    Complex multi-part questions need more.
    """)

    questions = [
        ("Simple: single lookup", "What is the dose for metformin?"),
        ("Medium: two lookups", "Compare the side effects of lisinopril and amlodipine."),
        ("Complex: multi-tool", "70-year-old diabetic with GFR 35, albuminuria 200. "
                                "Calculate CKD risk, check metformin safety, and interpret GFR."),
    ]

    results = []
    for label, question in questions:
        print(f"\n{'─' * 60}")
        print(f"  [{label}]")
        result = run_agent_with_limit(question, max_steps=6, verbose=True)
        results.append((label, result))

    print(f"\n\n{'═' * 70}")
    print("  STEP USAGE SUMMARY")
    print(f"{'═' * 70}")
    for label, r in results:
        tool_calls = [s for s in r["step_log"] if s["action"] == "tool_call"]
        all_tools = []
        for s in tool_calls:
            all_tools.extend(s.get("tools", []))
        print(f"\n  {label}")
        print(f"    Steps used: {r['steps']} / {r['max_steps']}")
        print(f"    Tool calls: {len(all_tools)}")
        for t in all_tools:
            print(f"      → {t[:60]}")


# ============================================================
# DEMO 3: Graceful Timeout Handling
# ============================================================

def demo_graceful_timeout():
    """Show how to handle max-step scenarios gracefully"""
    print("\n" + "=" * 70)
    print("DEMO 3: GRACEFUL TIMEOUT HANDLING")
    print("=" * 70)
    print("""
    When max_steps is hit, the agent is forced to give its best answer.
    Compare partial vs complete answers.
    """)

    question = (
        "Full workup: 68-year-old diabetic, hypertensive, on metformin and lisinopril. "
        "GFR is 42, potassium is 5.4, HbA1c is 8.1. "
        "Check all lab values, review both medications, and calculate CKD risk "
        "with albuminuria of 180."
    )

    print(f"\n  Question needs ~5-6 tool calls. Let's see what happens with limits.\n")

    # Very limited
    print(f"{'═' * 60}")
    print(f"  max_steps = 2 (TOO FEW)")
    print(f"{'═' * 60}")
    result_limited = run_agent_with_limit(question, max_steps=2, verbose=True)

    # Adequate
    print(f"\n{'═' * 60}")
    print(f"  max_steps = 8 (ADEQUATE)")
    print(f"{'═' * 60}")
    result_adequate = run_agent_with_limit(question, max_steps=8, verbose=True)

    print(f"\n\n{'─' * 70}")
    print("  COMPARISON:")
    print(f"  Limited (2 steps):  Hit limit = {result_limited['hit_limit']}, "
          f"Tools used = {sum(1 for s in result_limited['step_log'] if s['action'] == 'tool_call')}")
    print(f"  Adequate (8 steps): Hit limit = {result_adequate['hit_limit']}, "
          f"Tools used = {sum(1 for s in result_adequate['step_log'] if s['action'] == 'tool_call')}")

    print("""
    INSIGHT: Graceful timeout means:
      1. Detect when limit is hit
      2. Tell the agent to summarize what it has
      3. Clearly indicate the answer is PARTIAL
      4. In production: flag for human review
    """)


# ============================================================
# DEMO 4: Interactive
# ============================================================

def demo_interactive():
    """Ask questions with configurable step limits"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE (Set Your Own Step Limit)")
    print("=" * 70)
    print("  Type 'quit' to exit\n")

    while True:
        question = input("  Question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue

        limit_str = input("  Max steps (1-10, default 5): ").strip()
        try:
            limit = int(limit_str) if limit_str else 5
            limit = max(1, min(10, limit))
        except ValueError:
            limit = 5

        result = run_agent_with_limit(question, max_steps=limit, verbose=True)
        if result["hit_limit"]:
            print("\n  ⚠️  Agent was cut short. Try a higher step limit.")
        print()


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 3: MAXIMUM STEP LIMIT — PREVENTING INFINITE LOOPS")
    print("=" * 70)
    print("""
    Without a step limit, agents could loop forever. This exercise
    shows how to add limits and handle timeouts gracefully.

    Choose a demo:
      1 → Step limits in action (same question, different limits)
      2 → Simple vs complex questions (step usage comparison)
      3 → Graceful timeout handling (partial vs complete answers)
      4 → Interactive session (set your own limit)
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1":
        demo_step_limits()
    elif choice == "2":
        demo_simple_vs_complex()
    elif choice == "3":
        demo_graceful_timeout()
    elif choice == "4":
        demo_interactive()
    elif choice == "5":
        demo_step_limits()
        demo_simple_vs_complex()
        demo_graceful_timeout()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. ALWAYS SET A MAX STEP LIMIT: Without it, an agent could loop
   indefinitely on ambiguous questions or failed tool calls.

2. RIGHT-SIZE THE LIMIT: Simple queries need 1-2 steps. Complex
   multi-tool queries need 5-8. Over-limiting produces incomplete answers.

3. GRACEFUL TIMEOUTS: When the limit is hit, force the agent to give
   its best answer with available information. Never just crash.

4. STEP TRACKING: Log every step (tool calls, final answer) for
   debugging and monitoring. This is critical in production.

5. HEALTHCARE SAFETY: In clinical systems, an incomplete answer should
   be clearly flagged — never presented as a complete recommendation.
"""

if __name__ == "__main__":
    main()
