"""
Exercise 4: Agent Step Logger — Debug Trail for Every Action

Skills practiced:
- Logging every agent step (thought, action, observation) to a file
- Structured logging with timestamps, tool calls, and token usage
- Reading logs to debug agent behavior
- Production logging patterns for healthcare AI compliance

Key insight: In healthcare AI, you MUST be able to explain WHY the system
  made a recommendation. Logging creates an audit trail — like medical
  chart notes that document every clinical decision.
"""

import os
import json
import time
from datetime import datetime
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
    return f"Not found."


def calculate_ckd_risk(gfr: float, albuminuria: float, age: int, has_diabetes: bool) -> str:
    risk_score = 0
    if gfr < 30: risk_score += 3
    elif gfr < 60: risk_score += 1
    if albuminuria > 300: risk_score += 2
    elif albuminuria > 30: risk_score += 1
    if age > 65: risk_score += 1
    if has_diabetes: risk_score += 1
    level = "LOW" if risk_score <= 1 else "MODERATE" if risk_score <= 3 else "HIGH"
    return json.dumps({"risk_level": level, "risk_score": risk_score})


TOOLS = [
    {"type": "function", "function": {"name": "lookup_medication", "description": "Look up medication info", "parameters": {"type": "object", "properties": {"medication_name": {"type": "string"}}, "required": ["medication_name"]}}},
    {"type": "function", "function": {"name": "check_lab_value", "description": "Interpret a lab value (hba1c, gfr, potassium)", "parameters": {"type": "object", "properties": {"test_name": {"type": "string"}, "value": {"type": "number"}}, "required": ["test_name", "value"]}}},
    {"type": "function", "function": {"name": "calculate_ckd_risk", "description": "Calculate CKD risk", "parameters": {"type": "object", "properties": {"gfr": {"type": "number"}, "albuminuria": {"type": "number"}, "age": {"type": "integer"}, "has_diabetes": {"type": "boolean"}}, "required": ["gfr", "albuminuria", "age", "has_diabetes"]}}},
]

TOOL_FUNCTIONS = {
    "lookup_medication": lambda args: lookup_medication(args["medication_name"]),
    "check_lab_value": lambda args: check_lab_value(args["test_name"], args["value"]),
    "calculate_ckd_risk": lambda args: calculate_ckd_risk(args["gfr"], args["albuminuria"], args["age"], args["has_diabetes"]),
}


# ============================================================
# Agent Step Logger
# ============================================================

class AgentLogger:
    """
    Logs every step of the agent's execution for debugging and compliance.

    Healthcare parallel: Like a medical chart — every action documented
    with who did what, when, and why. Required for regulatory compliance.
    """

    def __init__(self, log_file=None):
        self.log_file = log_file
        self.entries = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(self, event_type, data):
        """Add a log entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": event_type,
            **data
        }
        self.entries.append(entry)

        # Write to file if configured
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def log_start(self, question, tools_available):
        self.log("agent_start", {
            "question": question,
            "tools_available": tools_available,
        })

    def log_tool_call(self, step, tool_name, tool_args, tool_result, duration_ms):
        self.log("tool_call", {
            "step": step,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": tool_result[:500],  # Truncate for log
            "duration_ms": duration_ms,
        })

    def log_final_answer(self, step, answer, total_tokens, total_duration_ms):
        self.log("final_answer", {
            "step": step,
            "answer": answer[:1000],
            "total_tokens": total_tokens,
            "total_duration_ms": total_duration_ms,
        })

    def log_error(self, step, error_msg):
        self.log("error", {
            "step": step,
            "error": error_msg,
        })

    def log_max_steps(self, max_steps):
        self.log("max_steps_reached", {
            "max_steps": max_steps,
        })

    def get_entries(self):
        return self.entries

    def get_summary(self):
        """Generate a human-readable summary"""
        tool_calls = [e for e in self.entries if e["event_type"] == "tool_call"]
        final = [e for e in self.entries if e["event_type"] == "final_answer"]
        errors = [e for e in self.entries if e["event_type"] == "error"]

        return {
            "session_id": self.session_id,
            "total_entries": len(self.entries),
            "tool_calls": len(tool_calls),
            "tools_used": list(set(e["tool_name"] for e in tool_calls)),
            "errors": len(errors),
            "total_tokens": final[0]["total_tokens"] if final else 0,
            "total_duration_ms": final[0]["total_duration_ms"] if final else 0,
        }

    def format_readable(self):
        """Format log as a readable debug trace"""
        lines = []
        lines.append(f"═══ Agent Session: {self.session_id} ═══")

        for entry in self.entries:
            ts = entry["timestamp"].split("T")[1][:12]

            if entry["event_type"] == "agent_start":
                lines.append(f"\n[{ts}] START")
                lines.append(f"   Question: {entry['question'][:80]}")
                lines.append(f"   Tools: {', '.join(entry['tools_available'])}")

            elif entry["event_type"] == "tool_call":
                lines.append(f"\n[{ts}] STEP {entry['step']} — TOOL CALL")
                lines.append(f"   Tool: {entry['tool_name']}")
                lines.append(f"   Args: {json.dumps(entry['tool_args'])}")
                lines.append(f"   Result: {entry['tool_result'][:150]}...")
                lines.append(f"   Duration: {entry['duration_ms']}ms")

            elif entry["event_type"] == "final_answer":
                lines.append(f"\n[{ts}] STEP {entry['step']} — FINAL ANSWER")
                lines.append(f"   Answer: {entry['answer'][:200]}...")
                lines.append(f"   Total tokens: {entry['total_tokens']}")
                lines.append(f"   Total duration: {entry['total_duration_ms']}ms")

            elif entry["event_type"] == "error":
                lines.append(f"\n[{ts}] ERROR at step {entry['step']}")
                lines.append(f"   {entry['error']}")

            elif entry["event_type"] == "max_steps_reached":
                lines.append(f"\n[{ts}] ⚠️  MAX STEPS REACHED ({entry['max_steps']})")

        return "\n".join(lines)


# ============================================================
# Logged Agent Run
# ============================================================

def run_logged_agent(user_question, max_steps=5, log_file=None, verbose=True):
    """ReAct agent with full step logging"""
    logger = AgentLogger(log_file=log_file)
    logger.log_start(user_question, list(TOOL_FUNCTIONS.keys()))

    start_time = time.time()
    total_tokens = 0

    if verbose:
        print(f"\n{'─' * 70}")
        print(f"  LOGGED AGENT (session: {logger.session_id})")
        print(f"  Q: \"{user_question[:80]}\"")
        print(f"{'─' * 70}")

    messages = [
        {"role": "system", "content": "You are a clinical decision support agent. Use tools when needed. Be efficient. Educational purposes only."},
        {"role": "user", "content": user_question}
    ]

    step = 0
    while step < max_steps:
        step += 1
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=TOOLS, tool_choice="auto"
        )
        total_tokens += response.usage.total_tokens
        assistant_message = response.choices[0].message

        if assistant_message.tool_calls:
            messages.append(assistant_message)
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                tool_start = time.time()
                try:
                    result = TOOL_FUNCTIONS.get(func_name, lambda a: "Unknown")(func_args)
                except Exception as e:
                    result = f"Error: {str(e)}"
                    logger.log_error(step, str(e))
                tool_duration = int((time.time() - tool_start) * 1000)

                logger.log_tool_call(step, func_name, func_args, result, tool_duration)

                if verbose:
                    print(f"    Step {step}: {func_name}({json.dumps(func_args)}) [{tool_duration}ms]")

                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        else:
            final_answer = assistant_message.content
            total_duration = int((time.time() - start_time) * 1000)
            logger.log_final_answer(step, final_answer, total_tokens, total_duration)

            if verbose:
                print(f"    Step {step}: FINAL ANSWER [{total_duration}ms, {total_tokens} tokens]")

            return final_answer, logger

    # Max steps
    logger.log_max_steps(max_steps)
    total_duration = int((time.time() - start_time) * 1000)
    messages.append({"role": "user", "content": "Provide your best answer with available info."})
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    total_tokens += response.usage.total_tokens
    forced = response.choices[0].message.content
    logger.log_final_answer(step + 1, forced, total_tokens, total_duration)

    return forced, logger


# ============================================================
# DEMO 1: Basic Logging
# ============================================================

def demo_basic_logging():
    """Show the logged output for a simple agent run"""
    print("\n" + "=" * 70)
    print("DEMO 1: BASIC AGENT LOGGING")
    print("=" * 70)
    print("""
    Every step is logged with timestamp, tool call, args, result, duration.
    Like a medical record — every action documented.
    """)

    question = "Patient has GFR 38 and is on metformin. Is this safe?"

    answer, logger = run_logged_agent(question, verbose=True)

    print(f"\n\n  ─── READABLE LOG ───\n")
    print(logger.format_readable())

    print(f"\n  ─── LOG SUMMARY ───")
    summary = logger.get_summary()
    for key, val in summary.items():
        print(f"    {key}: {val}")


# ============================================================
# DEMO 2: File Logging
# ============================================================

def demo_file_logging():
    """Log to a file for persistent debugging"""
    print("\n" + "=" * 70)
    print("DEMO 2: FILE-BASED LOGGING")
    print("=" * 70)

    log_path = os.path.join(os.path.dirname(__file__), "agent_debug.log")

    # Clear previous log
    if os.path.exists(log_path):
        os.remove(log_path)

    questions = [
        "What is the dose for lisinopril?",
        "70-year-old diabetic with GFR 35, albuminuria 200. Calculate CKD risk and check metformin.",
    ]

    for q in questions:
        print(f"\n  Running: \"{q[:60]}...\"")
        answer, logger = run_logged_agent(q, log_file=log_path, verbose=True)

    # Read and display the log file
    print(f"\n{'─' * 60}")
    print(f"  LOG FILE: {log_path}")
    print(f"{'─' * 60}")

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            lines = f.readlines()
        print(f"  Total log entries: {len(lines)}\n")
        for i, line in enumerate(lines):
            entry = json.loads(line)
            print(f"  [{i+1}] {entry['event_type']:<20} "
                  f"session={entry['session_id']} "
                  f"ts={entry['timestamp'].split('T')[1][:8]}")
            if entry['event_type'] == 'tool_call':
                print(f"       → {entry['tool_name']}({json.dumps(entry['tool_args'])}) [{entry['duration_ms']}ms]")

    print(f"""
    FILE LOGGING VALUE:
      • Persists across program restarts
      • Can be analyzed with log tools (grep, jq, etc.)
      • Required for healthcare compliance audits
      • Each line is valid JSON (JSONL format) — easy to parse
    """)

    # Clean up
    if os.path.exists(log_path):
        os.remove(log_path)
        print(f"  (Cleaned up log file)")


# ============================================================
# DEMO 3: Log Analysis
# ============================================================

def demo_log_analysis():
    """Run multiple queries and analyze the logs"""
    print("\n" + "=" * 70)
    print("DEMO 3: LOG ANALYSIS ACROSS MULTIPLE QUERIES")
    print("=" * 70)

    questions = [
        "What is the dose for metformin?",
        "Check potassium of 5.6 — is this high?",
        "What is the contraindication for apixaban?",
        "68-year-old diabetic, GFR 42, albuminuria 150. CKD risk?",
        "Compare lisinopril and amlodipine for hypertension.",
    ]

    all_summaries = []
    for q in questions:
        answer, logger = run_logged_agent(q, verbose=False)
        summary = logger.get_summary()
        summary["question"] = q[:50]
        all_summaries.append(summary)

    print(f"\n  {'Question':<52} {'Steps':>5} {'Tools':>5} {'Tokens':>7} {'Time':>7}")
    print(f"  {'─'*52} {'─'*5} {'─'*5} {'─'*7} {'─'*7}")
    for s in all_summaries:
        print(f"  {s['question']:<52} {s['tool_calls']:>5} {len(s['tools_used']):>5} "
              f"{s['total_tokens']:>7} {s['total_duration_ms']:>6}ms")

    # Aggregate stats
    total_tokens = sum(s['total_tokens'] for s in all_summaries)
    total_time = sum(s['total_duration_ms'] for s in all_summaries)
    total_tool_calls = sum(s['tool_calls'] for s in all_summaries)
    all_tools = set()
    for s in all_summaries:
        all_tools.update(s['tools_used'])

    print(f"\n  ─── AGGREGATE STATS ───")
    print(f"  Total queries:    {len(all_summaries)}")
    print(f"  Total tool calls: {total_tool_calls}")
    print(f"  Unique tools:     {', '.join(all_tools)}")
    print(f"  Total tokens:     {total_tokens:,}")
    print(f"  Total time:       {total_time:,}ms")
    print(f"  Avg tokens/query: {total_tokens // len(all_summaries):,}")
    print(f"  Avg time/query:   {total_time // len(all_summaries):,}ms")

    print("""
    LOG ANALYSIS VALUE:
      • Identify expensive queries (high tokens/time)
      • Track tool usage patterns (which tools are most used)
      • Spot inefficiency (unnecessary tool calls)
      • Monitor cost over time
    """)


# ============================================================
# DEMO 4: Interactive with Logging
# ============================================================

def demo_interactive():
    """Interactive session with live logging"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE WITH LIVE LOGGING")
    print("=" * 70)
    print("  Every question is logged. Type 'log' to see session log.")
    print("  Type 'quit' to exit.\n")

    all_loggers = []

    while True:
        question = input("  Question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if question.lower() == 'log':
            if all_loggers:
                for lg in all_loggers:
                    print(lg.format_readable())
                    print()
            else:
                print("  No logs yet.\n")
            continue
        if not question:
            continue

        answer, logger = run_logged_agent(question, verbose=True)
        all_loggers.append(logger)
        print()

    # Final summary
    if all_loggers:
        print(f"\n{'═' * 60}")
        print("  SESSION SUMMARY")
        print(f"{'═' * 60}")
        for lg in all_loggers:
            s = lg.get_summary()
            print(f"  [{s['session_id']}] Tools: {s['tool_calls']}, Tokens: {s['total_tokens']}, Time: {s['total_duration_ms']}ms")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 4: AGENT STEP LOGGER — DEBUG TRAIL")
    print("=" * 70)
    print("""
    Log every agent step for debugging and compliance.
    Like medical charting — every action documented.

    Choose a demo:
      1 → Basic logging (see the readable trace)
      2 → File logging (persist to disk)
      3 → Log analysis (aggregate stats across queries)
      4 → Interactive with logging
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1":
        demo_basic_logging()
    elif choice == "2":
        demo_file_logging()
    elif choice == "3":
        demo_log_analysis()
    elif choice == "4":
        demo_interactive()
    elif choice == "5":
        demo_basic_logging()
        demo_file_logging()
        demo_log_analysis()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. LOG EVERYTHING: Every tool call (with args and result), every step,
   every final answer. This creates an audit trail.

2. STRUCTURED LOGS: JSON format (JSONL = one JSON per line) enables
   automated analysis with tools like jq, pandas, or log aggregators.

3. TIMESTAMPS + DURATIONS: Know WHEN things happened and HOW LONG
   they took. Essential for performance debugging.

4. COMPLIANCE: Healthcare AI systems MUST log every decision for
   regulatory compliance. "Why did the system recommend X?"
   The log should answer that question definitively.

5. AGGREGATE ANALYSIS: Individual logs are useful for debugging.
   Aggregated stats reveal systemic patterns — expensive queries,
   over-used tools, slow tools, error rates.
"""

if __name__ == "__main__":
    main()
