"""
Exercise 2: Handle "No Tool Needed" — Direct Answers

Skills practiced:
- Understanding when the agent decides NOT to use tools
- Detecting tool_choice behavior (auto vs required)
- Designing prompts that let the agent answer directly when appropriate
- Comparing tool-assisted vs direct answers

Key insight: A well-designed agent doesn't always use tools.
  Sometimes the question is simple enough to answer from training knowledge.
  The agent should recognize this and respond directly — saving time and cost.
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
    """Look up information about a medication"""
    medications = {
        "metformin": {"class": "Biguanide", "indication": "Type 2 Diabetes", "dose": "500-2000mg daily", "contraindications": "eGFR below 30"},
        "lisinopril": {"class": "ACE Inhibitor", "indication": "HTN, HF", "dose": "10-40mg daily", "contraindications": "Pregnancy, angioedema"},
        "apixaban": {"class": "DOAC", "indication": "AFib, VTE", "dose": "5mg BID", "contraindications": "Active bleeding"},
    }
    med = medications.get(medication_name.lower())
    if med:
        return json.dumps(med, indent=2)
    return f"Medication '{medication_name}' not found."


def check_lab_value(test_name: str, value: float) -> str:
    """Interpret a lab value"""
    lab_ranges = {
        "hba1c": lambda v: "Normal" if v < 5.7 else "Prediabetes" if v < 6.5 else "Diabetes",
        "gfr": lambda v: "Normal" if v >= 90 else "Mild CKD" if v >= 60 else "Moderate CKD" if v >= 30 else "Severe CKD",
        "potassium": lambda v: "LOW" if v < 3.5 else "Normal" if v <= 5.0 else "HIGH",
    }
    fn = lab_ranges.get(test_name.lower())
    if fn:
        return json.dumps({"test": test_name, "value": value, "interpretation": fn(value)})
    return f"Test not found."


TOOLS = [
    {"type": "function", "function": {"name": "lookup_medication", "description": "Look up medication info", "parameters": {"type": "object", "properties": {"medication_name": {"type": "string"}}, "required": ["medication_name"]}}},
    {"type": "function", "function": {"name": "check_lab_value", "description": "Interpret a lab test value. Tests: hba1c, gfr, potassium", "parameters": {"type": "object", "properties": {"test_name": {"type": "string"}, "value": {"type": "number"}}, "required": ["test_name", "value"]}}},
]

TOOL_FUNCTIONS = {
    "lookup_medication": lambda args: lookup_medication(args["medication_name"]),
    "check_lab_value": lambda args: check_lab_value(args["test_name"], args["value"]),
}


# ============================================================
# Agent with Smart Tool Use (answers directly when possible)
# ============================================================

def run_smart_agent(user_question, max_steps=5, verbose=True):
    """
    Agent that can answer directly OR use tools — depending on the question.

    The key is the system prompt: we explicitly tell the agent it CAN
    answer directly if no tool lookup is needed.
    """
    if verbose:
        print(f"\n{'─' * 70}")
        print(f"  AGENT: \"{user_question[:80]}\"")
        print(f"{'─' * 70}")

    messages = [
        {
            "role": "system",
            "content": """You are a clinical decision support agent with access to medication
lookup and lab interpretation tools.

WHEN TO USE TOOLS:
- Use lookup_medication when the user asks about SPECIFIC drug data (doses, side effects, contraindications)
- Use check_lab_value when the user provides a SPECIFIC lab value to interpret

WHEN TO ANSWER DIRECTLY (no tools):
- General medical knowledge questions (e.g., "What is hypertension?")
- Conceptual questions (e.g., "Why do we monitor kidney function?")
- Questions about clinical processes (e.g., "What steps are in a physical exam?")
- Questions that don't require looking up specific data

Be efficient: don't use a tool when you can answer from general medical knowledge.
Disclaimer: Educational purposes only."""
        },
        {"role": "user", "content": user_question}
    ]

    step = 0
    used_tools = []

    while step < max_steps:
        step += 1

        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=TOOLS, tool_choice="auto"
        )
        assistant_message = response.choices[0].message

        if assistant_message.tool_calls:
            messages.append(assistant_message)
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                used_tools.append(func_name)

                if verbose:
                    print(f"    Step {step} — TOOL CALL: {func_name}({json.dumps(func_args)})")

                result = TOOL_FUNCTIONS.get(func_name, lambda a: "Unknown")(func_args)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        else:
            final_answer = assistant_message.content
            mode = "DIRECT (no tools)" if not used_tools else f"TOOL-ASSISTED ({', '.join(used_tools)})"

            if verbose:
                print(f"    Mode: {mode}")
                print(f"    Steps: {step}")
                print(f"\n    Answer: {final_answer[:400]}...")

            return {
                "answer": final_answer,
                "mode": mode,
                "tools_used": used_tools,
                "steps": step,
            }

    return {"answer": "Max steps reached.", "mode": "TIMEOUT", "tools_used": used_tools, "steps": step}


# ============================================================
# DEMO 1: Direct vs Tool-Assisted Answers
# ============================================================

def demo_direct_vs_tools():
    """Compare questions that need tools vs those that don't"""
    print("\n" + "=" * 70)
    print("DEMO 1: DIRECT vs TOOL-ASSISTED ANSWERS")
    print("=" * 70)
    print("""
    Some questions need tool lookups; others don't.
    A smart agent knows the difference.
    """)

    # Questions that should be answered DIRECTLY (no tools)
    direct_questions = [
        "What is hypertension?",
        "Why is smoking cessation important for COPD patients?",
        "What are the components of a SOAP note?",
        "How often should adults get a blood pressure check?",
    ]

    # Questions that should use TOOLS
    tool_questions = [
        "What is the dose range for metformin?",
        "A patient has a GFR of 28 — what stage CKD?",
        "What are the contraindications for apixaban?",
        "Patient has potassium of 5.8 — is this concerning?",
    ]

    print("\n  ─── QUESTIONS EXPECTED TO BE DIRECT (no tools) ───\n")
    direct_results = []
    for q in direct_questions:
        result = run_smart_agent(q, verbose=True)
        direct_results.append(result)
        print()

    print("\n  ─── QUESTIONS EXPECTED TO USE TOOLS ───\n")
    tool_results = []
    for q in tool_questions:
        result = run_smart_agent(q, verbose=True)
        tool_results.append(result)
        print()

    # Summary
    print(f"\n{'═' * 70}")
    print("  SUMMARY")
    print(f"{'═' * 70}")
    direct_count = sum(1 for r in direct_results if not r["tools_used"])
    tool_count = sum(1 for r in tool_results if r["tools_used"])
    print(f"  Direct questions answered without tools: {direct_count}/{len(direct_questions)}")
    print(f"  Tool questions that used tools:          {tool_count}/{len(tool_questions)}")

    print("""
    INSIGHT: The agent learns when to use tools from:
      1. The system prompt (tells it when tools are appropriate)
      2. The tool descriptions (agent matches question → tool)
      3. Its training (knows it can answer general questions directly)
    """)


# ============================================================
# DEMO 2: tool_choice Modes Compared
# ============================================================

def demo_tool_choice_modes():
    """Compare tool_choice="auto" vs "required" vs "none" """
    print("\n" + "=" * 70)
    print('DEMO 2: tool_choice MODES ("auto" vs "required" vs "none")')
    print("=" * 70)
    print("""
    tool_choice controls whether the LLM CAN, MUST, or CANNOT use tools:
      "auto"     → LLM decides (recommended for agents)
      "required" → LLM MUST call a tool (forced tool use)
      "none"     → LLM CANNOT use tools (direct answer only)
    """)

    question = "What is hypertension?"  # Simple question — shouldn't need tools

    modes = ["auto", "none", "required"]

    for mode in modes:
        print(f"\n{'─' * 60}")
        print(f"  tool_choice = \"{mode}\"")
        print(f"{'─' * 60}")

        messages = [
            {"role": "system", "content": "You are a medical assistant. Use tools if needed. Answer briefly."},
            {"role": "user", "content": question}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, tools=TOOLS, tool_choice=mode
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"    → Called tool: {tc.function.name}({tc.function.arguments})")
                print(f"    Result: Agent was FORCED to use a tool (even though it wasn't needed)")
            else:
                print(f"    → Direct answer: {msg.content[:200]}...")
        except Exception as e:
            print(f"    → Error: {e}")

    print("""
    KEY INSIGHT:
      "auto" is best for agents — the LLM decides intelligently.
      "required" forces unnecessary tool calls (wasteful).
      "none" prevents tool use entirely (limits the agent).

    In production, use "auto" and let your prompt guide when tools are useful.
    """)


# ============================================================
# DEMO 3: Cost Comparison
# ============================================================

def demo_cost_comparison():
    """Show the cost difference between direct and tool-assisted answers"""
    print("\n" + "=" * 70)
    print("DEMO 3: COST OF DIRECT vs TOOL-ASSISTED")
    print("=" * 70)
    print("""
    Tool calls cost extra tokens (the tool definition, the call, the result).
    Direct answers save tokens. Let's measure!
    """)

    # Direct answer
    question = "What is diabetes mellitus?"
    messages_direct = [
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": question}
    ]
    resp_direct = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages_direct, tools=TOOLS, tool_choice="none"
    )

    # Tool-assisted answer (forced)
    resp_tool = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages_direct, tools=TOOLS, tool_choice="required"
    )

    print(f"\n  Question: \"{question}\"\n")
    print(f"  {'Mode':<20} {'Input Tokens':>14} {'Output Tokens':>14} {'Total':>8}")
    print(f"  {'─'*20} {'─'*14} {'─'*14} {'─'*8}")
    print(f"  {'Direct (no tools)':<20} {resp_direct.usage.prompt_tokens:>14} {resp_direct.usage.completion_tokens:>14} {resp_direct.usage.total_tokens:>8}")
    print(f"  {'Forced tool use':<20} {resp_tool.usage.prompt_tokens:>14} {resp_tool.usage.completion_tokens:>14} {resp_tool.usage.total_tokens:>8}")

    savings = resp_tool.usage.total_tokens - resp_direct.usage.total_tokens
    print(f"\n  Token savings from direct answer: ~{savings} tokens")
    print(f"  At scale (1000 queries/day): ~{savings * 1000:,} tokens saved")

    print("""
    INSIGHT: Smart agents that answer directly when possible save
    real money at scale. This is why tool_choice="auto" matters —
    the agent should only use tools when they add value.
    """)


# ============================================================
# DEMO 4: Interactive
# ============================================================

def demo_interactive():
    """Try your own questions — see if the agent uses tools or not"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE — Will the Agent Use Tools?")
    print("=" * 70)
    print("  Try general questions (no tools) and specific lookups (tools).")
    print("  Type 'quit' to exit\n")

    while True:
        question = input("  Your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue
        result = run_smart_agent(question, verbose=True)
        print()


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 2: HANDLE 'NO TOOL NEEDED' — DIRECT ANSWERS")
    print("=" * 70)
    print("""
    Not every question needs a tool call. A smart agent answers
    directly when it can, saving time and tokens.

    Choose a demo:
      1 → Direct vs tool-assisted answers (side by side)
      2 → tool_choice modes ("auto" vs "required" vs "none")
      3 → Cost comparison (tokens saved by direct answers)
      4 → Interactive session
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1":
        demo_direct_vs_tools()
    elif choice == "2":
        demo_tool_choice_modes()
    elif choice == "3":
        demo_cost_comparison()
    elif choice == "4":
        demo_interactive()
    elif choice == "5":
        demo_direct_vs_tools()
        demo_tool_choice_modes()
        demo_cost_comparison()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. AGENTS DON'T ALWAYS NEED TOOLS: General knowledge questions can be
   answered directly. The system prompt guides when tools are appropriate.

2. tool_choice PARAMETER:
   - "auto" — LLM decides (best for agents)
   - "required" — forces tool use (wastes tokens on simple questions)
   - "none" — prevents tools (useful for testing direct answers)

3. COST EFFICIENCY: Direct answers use fewer tokens. At scale, smart
   tool-use decisions save significant money.

4. PROMPT DESIGN: Explicitly telling the agent WHEN to use tools
   (and when NOT to) improves its decision-making.

5. TESTING BOTH PATHS: Always test that your agent handles both cases —
   tool-needed and no-tool-needed — correctly.
"""

if __name__ == "__main__":
    main()
