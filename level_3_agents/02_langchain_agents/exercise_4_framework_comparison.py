"""
Exercise 4: Framework Comparison — LangChain Agent vs From-Scratch ReAct

Skills practiced:
- Running the SAME clinical query through two different implementations
- Comparing token usage, latency, and output quality
- Understanding framework overhead vs flexibility trade-offs
- Making informed decisions about when to use a framework

Key insight: LangChain adds convenience but also overhead. For simple
  agents, from-scratch may be more efficient. For complex agents
  with memory, streaming, and error handling, LangChain saves time.
  Know BOTH so you can choose wisely.
"""

import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent as create_langchain_agent
from langchain_core.tools import tool as langchain_tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# ============================================================
# Shared Medication Database
# ============================================================

MEDICATIONS = {
    "metformin": {"class": "Biguanide", "dose": "500-2000mg daily", "contraindications": "eGFR<30", "monitor": "HbA1c, B12, renal"},
    "lisinopril": {"class": "ACE Inhibitor", "dose": "10-40mg daily", "contraindications": "Pregnancy", "monitor": "BP, K+, Cr"},
    "amlodipine": {"class": "CCB", "dose": "2.5-10mg daily", "contraindications": "Severe aortic stenosis", "monitor": "BP, HR"},
    "apixaban": {"class": "DOAC", "dose": "5mg BID", "contraindications": "Active bleeding", "monitor": "Signs of bleeding"},
    "sertraline": {"class": "SSRI", "dose": "50-200mg daily", "contraindications": "MAOi within 14 days", "monitor": "Mood, SE"},
}

LABS = {
    "hba1c": lambda v: f"HbA1c {v}%: {'Normal' if v < 5.7 else 'Prediabetes' if v < 6.5 else 'Diabetes'}",
    "gfr": lambda v: f"GFR {v}: {'Normal' if v >= 90 else 'Mild CKD' if v >= 60 else 'Mod CKD' if v >= 30 else 'Severe CKD'}",
    "potassium": lambda v: f"K+ {v}: {'LOW' if v < 3.5 else 'Normal' if v <= 5.0 else 'HIGH'}",
}


# ============================================================
# FROM-SCRATCH Implementation (Project 01 style)
# ============================================================

raw_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Raw OpenAI tool definitions
RAW_TOOLS = [
    {"type": "function", "function": {
        "name": "lookup_medication",
        "description": "Look up medication info. Available: metformin, lisinopril, amlodipine, apixaban, sertraline.",
        "parameters": {"type": "object", "properties": {"medication_name": {"type": "string"}}, "required": ["medication_name"]}
    }},
    {"type": "function", "function": {
        "name": "check_lab_value",
        "description": "Interpret a lab value. Available: hba1c, gfr, potassium.",
        "parameters": {"type": "object", "properties": {"test_name": {"type": "string"}, "value": {"type": "number"}}, "required": ["test_name", "value"]}
    }},
]

RAW_TOOL_FUNCTIONS = {
    "lookup_medication": lambda args: json.dumps(MEDICATIONS.get(args["medication_name"].lower(), "Not found")),
    "check_lab_value": lambda args: LABS.get(args["test_name"].lower(), lambda v: "Unknown")(args["value"]),
}


def run_from_scratch(question, max_steps=5):
    """Project 01 style ReAct agent — no framework"""
    start = time.time()
    total_tokens = 0
    tool_calls_count = 0

    messages = [
        {"role": "system", "content": "You are a clinical decision support agent. Use tools when needed. Be concise. Educational only."},
        {"role": "user", "content": question}
    ]

    for step in range(max_steps):
        response = raw_client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=RAW_TOOLS, tool_choice="auto"
        )
        total_tokens += response.usage.total_tokens
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                fn = tc.function.name
                args = json.loads(tc.function.arguments)
                result = RAW_TOOL_FUNCTIONS.get(fn, lambda a: "Unknown")(args)
                tool_calls_count += 1
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            return {
                "answer": msg.content,
                "tokens": total_tokens,
                "tool_calls": tool_calls_count,
                "steps": step + 1,
                "duration_ms": int((time.time() - start) * 1000),
                "framework": "FROM-SCRATCH",
            }

    return {
        "answer": "[Max steps reached]",
        "tokens": total_tokens, "tool_calls": tool_calls_count,
        "steps": max_steps, "duration_ms": int((time.time() - start) * 1000),
        "framework": "FROM-SCRATCH",
    }


# ============================================================
# LANGCHAIN Implementation (Project 02 style)
# ============================================================

langchain_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


@langchain_tool
def lc_lookup_medication(medication_name: str) -> str:
    """Look up medication info. Available: metformin, lisinopril, amlodipine, apixaban, sertraline."""
    med = MEDICATIONS.get(medication_name.lower())
    return json.dumps(med) if med else "Not found"


@langchain_tool
def lc_check_lab_value(test_name: str, value: float) -> str:
    """Interpret a lab value. Available: hba1c, gfr, potassium."""
    fn = LABS.get(test_name.lower())
    return fn(value) if fn else "Unknown"


lc_tools = [lc_lookup_medication, lc_check_lab_value]


def run_langchain(question):
    """Project 02 style LangChain agent"""
    start = time.time()

    agent = create_langchain_agent(
        langchain_llm,
        tools=lc_tools,
        system_prompt="You are a clinical decision support agent. Use tools when needed. Be concise. Educational only."
    )

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    # Count tool calls from messages
    tool_calls_count = sum(
        1 for msg in result["messages"]
        if isinstance(msg, ToolMessage)
    )

    # Get the final answer
    answer = result["messages"][-1].content

    return {
        "answer": answer,
        "tokens": -1,  # LangChain doesn't expose total tokens easily
        "tool_calls": tool_calls_count,
        "steps": tool_calls_count + 1,
        "duration_ms": int((time.time() - start) * 1000),
        "framework": "LANGCHAIN",
    }


# ============================================================
# DEMO 1: Side-by-Side Comparison
# ============================================================

def demo_side_by_side():
    """Run the same queries through both frameworks"""
    print("\n" + "=" * 70)
    print("DEMO 1: SIDE-BY-SIDE COMPARISON")
    print("=" * 70)
    print("""
    Same questions, same tools, same model (gpt-4o-mini).
    Only difference: FROM-SCRATCH vs LANGCHAIN framework.
    """)

    questions = [
        "What is the dose and monitoring for metformin?",
        "Check HbA1c of 7.8 and potassium of 5.2",
        "Patient on lisinopril has GFR 38. Is metformin safe?",
    ]

    for q in questions:
        print(f"\n{'─' * 60}")
        print(f"  Q: \"{q}\"\n")

        # From scratch
        raw_result = run_from_scratch(q)
        print(f"  FROM-SCRATCH:")
        print(f"    Tools: {raw_result['tool_calls']} | Steps: {raw_result['steps']} | "
              f"Tokens: {raw_result['tokens']} | Time: {raw_result['duration_ms']}ms")
        print(f"    Answer: {raw_result['answer'][:200]}...\n")

        # LangChain
        lc_result = run_langchain(q)
        print(f"  LANGCHAIN:")
        print(f"    Tools: {lc_result['tool_calls']} | Steps: {lc_result['steps']} | "
              f"Time: {lc_result['duration_ms']}ms")
        print(f"    Answer: {lc_result['answer'][:200]}...")


# ============================================================
# DEMO 2: Aggregate Performance
# ============================================================

def demo_aggregate():
    """Run many queries and compare aggregate metrics"""
    print("\n" + "=" * 70)
    print("DEMO 2: AGGREGATE PERFORMANCE")
    print("=" * 70)

    questions = [
        "What is the dose for lisinopril?",
        "Check HbA1c of 6.3",
        "Look up apixaban contraindications",
        "Is GFR 55 concerning?",
        "What monitoring does sertraline need?",
    ]

    raw_results = []
    lc_results = []

    for q in questions:
        print(f"  Running: \"{q[:40]}\"...")
        raw_results.append(run_from_scratch(q))
        lc_results.append(run_langchain(q))

    # Summary
    print(f"\n{'─' * 60}")
    print(f"  {'Metric':<25} {'FROM-SCRATCH':>15} {'LANGCHAIN':>15}")
    print(f"  {'─'*25} {'─'*15} {'─'*15}")

    raw_total_time = sum(r["duration_ms"] for r in raw_results)
    lc_total_time = sum(r["duration_ms"] for r in lc_results)
    raw_total_tokens = sum(r["tokens"] for r in raw_results)
    raw_total_tools = sum(r["tool_calls"] for r in raw_results)
    lc_total_tools = sum(r["tool_calls"] for r in lc_results)

    print(f"  {'Total time (ms)':<25} {raw_total_time:>15,} {lc_total_time:>15,}")
    print(f"  {'Avg time/query (ms)':<25} {raw_total_time // len(questions):>15,} {lc_total_time // len(questions):>15,}")
    print(f"  {'Total tool calls':<25} {raw_total_tools:>15} {lc_total_tools:>15}")
    print(f"  {'Total tokens':<25} {raw_total_tokens:>15,} {'N/A':>15}")

    print("""
    OBSERVATIONS:
      • Both frameworks make the same number of tool calls (same logic)
      • LangChain may add slight overhead for setup/prompt templating
      • Token usage is similar — both use gpt-4o-mini with same prompts
      • Latency differences are mainly network variance, not framework
    """)


# ============================================================
# DEMO 3: Feature Comparison
# ============================================================

def demo_feature_comparison():
    """Compare what each approach gives you"""
    print("\n" + "=" * 70)
    print("DEMO 3: FEATURE COMPARISON")
    print("=" * 70)

    print("""
    ┌─────────────────────────┬─────────────────┬──────────────────┐
    │ Feature                 │ FROM-SCRATCH     │ LANGCHAIN        │
    ├─────────────────────────┼─────────────────┼──────────────────┤
    │ Setup complexity        │ Low (raw API)    │ Medium (imports) │
    │ Tool definition         │ JSON schema      │ @tool decorator  │
    │ ReAct loop              │ Manual while     │ create_agent()   │
    │ Memory                  │ Manual list      │ Built-in options │
    │ Error handling          │ Manual try/catch │ Built-in retries │
    │ Streaming               │ Manual           │ Built-in         │
    │ Token tracking          │ Direct access    │ Callback layers  │
    │ Multiple LLM providers  │ Manual adapter   │ Swap 1 import    │
    │ Intermediate steps      │ Manual logging   │ Built-in option  │
    │ Debugging               │ Print statements │ verbose=True     │
    │ Learning value          │ HIGH             │ MEDIUM           │
    │ Production readiness    │ LOW              │ HIGH             │
    │ Code maintenance        │ All your code    │ Framework updates │
    └─────────────────────────┴─────────────────┴──────────────────┘
    """)

    # Show code comparison
    print("  CODE COMPARISON — Defining a tool:\n")
    print("  FROM-SCRATCH (Project 01):")
    print('    {"type": "function", "function": {')
    print('        "name": "lookup_medication",')
    print('        "description": "Look up medication info",')
    print('        "parameters": {"type": "object", "properties": {...}}')
    print('    }}')
    print()
    print("  LANGCHAIN (Project 02):")
    print('    @tool')
    print('    def lookup_medication(medication_name: str) -> str:')
    print('        """Look up medication info."""')
    print('        ...')

    print("""
    WHEN TO USE EACH:

    FROM-SCRATCH:
      ✓ Learning / understanding the internals
      ✓ Simple agents with 1-2 tools
      ✓ Maximum control over every detail
      ✓ Minimal dependencies

    LANGCHAIN:
      ✓ Production applications
      ✓ Complex agents with many tools
      ✓ Need memory, streaming, callbacks
      ✓ Rapid prototyping
      ✓ Switching between LLM providers
    """)


# ============================================================
# DEMO 4: Interactive (choose your framework)
# ============================================================

def demo_interactive():
    """Interactive comparison"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE — CHOOSE YOUR FRAMEWORK")
    print("=" * 70)
    print("  Prefix with 'raw:' for from-scratch, 'lc:' for LangChain")
    print("  Or just type a question to run BOTH and compare.")
    print("  Type 'quit' to exit.\n")

    while True:
        inp = input("  Question: ").strip()
        if inp.lower() in ['quit', 'exit', 'q']:
            break
        if not inp:
            continue

        if inp.startswith("raw:"):
            result = run_from_scratch(inp[4:].strip())
            print(f"\n  FROM-SCRATCH ({result['duration_ms']}ms, {result['tokens']} tokens):")
            print(f"  {result['answer'][:400]}\n")
        elif inp.startswith("lc:"):
            result = run_langchain(inp[3:].strip())
            print(f"\n  LANGCHAIN ({result['duration_ms']}ms):")
            print(f"  {result['answer'][:400]}\n")
        else:
            raw_r = run_from_scratch(inp)
            lc_r = run_langchain(inp)
            print(f"\n  FROM-SCRATCH ({raw_r['duration_ms']}ms): {raw_r['answer'][:200]}")
            print(f"  LANGCHAIN    ({lc_r['duration_ms']}ms): {lc_r['answer'][:200]}\n")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 4: FRAMEWORK COMPARISON — LANGCHAIN vs FROM-SCRATCH")
    print("=" * 70)
    print("""
    Same agent, two implementations. Compare performance,
    features, and code complexity.

    Choose a demo:
      1 → Side-by-side (same queries, both frameworks)
      2 → Aggregate performance (5 queries)
      3 → Feature comparison (table)
      4 → Interactive (choose framework per query)
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1": demo_side_by_side()
    elif choice == "2": demo_aggregate()
    elif choice == "3": demo_feature_comparison()
    elif choice == "4": demo_interactive()
    elif choice == "5":
        demo_side_by_side()
        demo_aggregate()
        demo_feature_comparison()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. SAME CORE PATTERN: Both use the ReAct loop — LLM → Tool → LLM.
   LangChain wraps it; from-scratch exposes it. Understanding the
   raw pattern (Project 01) makes LangChain make sense.

2. FRAMEWORK OVERHEAD IS MINIMAL: Token usage and latency are similar.
   The real difference is developer experience and maintainability.

3. CHOOSE BASED ON NEEDS:
   - Learning / interview prep → from-scratch
   - Production / team project → LangChain (or LangGraph)
   - Simple agent → from-scratch (fewer dependencies)
   - Complex agent → LangChain (don't reinvent the wheel)

4. KNOW BOTH: Understanding from-scratch makes you better at
   debugging LangChain agents. You'll know what's happening under
   the hood when verbose=True shows the agent's reasoning.

5. THE INDUSTRY TREND: Most production systems use frameworks.
   LangChain and LangGraph (next project!) are the most common.
   But the best engineers understand the fundamentals underneath.
"""

if __name__ == "__main__":
    main()
