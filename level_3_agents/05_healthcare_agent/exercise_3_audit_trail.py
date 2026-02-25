"""
Exercise 3: Audit Trail — Log Every Agent Action for Compliance

Skills practiced:
- Building comprehensive audit logging for AI agent actions
- Structured logging with timestamps and context
- Compliance-ready output (HIPAA, FDA audit requirements)
- Understanding observability in production AI systems

Key insight: In healthcare, EVERY action by an AI system must be
  logged and traceable. "Why did the system recommend X?" needs a
  clear, timestamped answer. This exercise wraps every agent step
  with audit logging — the foundation for compliance, debugging,
  and trust in clinical AI.
"""

import os
import json
import time
from datetime import datetime
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# Audit Logger
# ============================================================

class AuditLogger:
    """Comprehensive audit trail for clinical AI actions"""

    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.entries: list[dict] = []
        self.start_time = time.time()

    def log(self, action: str, agent: str, details: dict, risk_level: str = "info"):
        """Log an audit entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_ms": round((time.time() - self.start_time) * 1000),
            "session_id": self.session_id,
            "action": action,
            "agent": agent,
            "risk_level": risk_level,
            "details": details,
        }
        self.entries.append(entry)
        return entry

    def log_input(self, patient_data: str):
        """Log patient data input"""
        return self.log("PATIENT_INPUT", "system", {
            "data_length": len(patient_data),
            "preview": patient_data[:100] + "..." if len(patient_data) > 100 else patient_data,
            "note": "Patient data received for processing",
        })

    def log_tool_call(self, tool_name: str, args: dict, result: str):
        """Log a tool invocation"""
        return self.log("TOOL_CALL", "agent", {
            "tool": tool_name,
            "arguments": args,
            "result_length": len(result),
            "result_preview": result[:150] + "..." if len(result) > 150 else result,
        })

    def log_llm_call(self, agent_name: str, prompt_preview: str, response_preview: str):
        """Log an LLM invocation"""
        return self.log("LLM_CALL", agent_name, {
            "prompt_preview": prompt_preview[:100],
            "response_preview": response_preview[:150],
        })

    def log_decision(self, agent_name: str, decision: str, reasoning: str, risk: str = "info"):
        """Log a clinical decision"""
        return self.log("CLINICAL_DECISION", agent_name, {
            "decision": decision,
            "reasoning": reasoning[:200],
        }, risk_level=risk)

    def log_safety_check(self, check_type: str, result: str, passed: bool):
        """Log a safety check"""
        risk = "info" if passed else "warning"
        return self.log("SAFETY_CHECK", "safety_agent", {
            "check_type": check_type,
            "result": result,
            "passed": passed,
        }, risk_level=risk)

    def log_output(self, recommendation: str):
        """Log the final output"""
        return self.log("FINAL_OUTPUT", "system", {
            "recommendation_length": len(recommendation),
            "recommendation_preview": recommendation[:200],
        })

    def get_summary(self) -> dict:
        """Generate audit summary"""
        total_time = round((time.time() - self.start_time) * 1000)
        actions = [e["action"] for e in self.entries]
        return {
            "session_id": self.session_id,
            "total_entries": len(self.entries),
            "total_time_ms": total_time,
            "actions": {a: actions.count(a) for a in set(actions)},
            "risk_events": sum(1 for e in self.entries if e["risk_level"] in ("warning", "critical")),
            "tools_used": [e["details"]["tool"] for e in self.entries if e["action"] == "TOOL_CALL"],
        }

    def format_readable(self) -> str:
        """Format audit trail as readable text"""
        lines = [f"\nAUDIT TRAIL — Session: {self.session_id}"]
        lines.append("=" * 60)
        for e in self.entries:
            risk_flag = " ⚠️" if e["risk_level"] == "warning" else " 🚨" if e["risk_level"] == "critical" else ""
            lines.append(
                f"  [{e['elapsed_ms']:>6}ms] {e['action']:<20s} ({e['agent']}){risk_flag}"
            )
            for k, v in e["details"].items():
                val_str = str(v)
                if len(val_str) > 80:
                    val_str = val_str[:77] + "..."
                lines.append(f"           {k}: {val_str}")
            lines.append("")
        return "\n".join(lines)

    def to_json(self) -> str:
        """Export audit trail as JSON"""
        return json.dumps({
            "session_id": self.session_id,
            "summary": self.get_summary(),
            "entries": self.entries,
        }, indent=2)

    def save(self, filepath: str):
        """Save audit trail to file"""
        with open(filepath, "w") as f:
            f.write(self.to_json())
        return filepath


# ============================================================
# Audited Tools
# ============================================================

# Global audit logger reference
_audit = None

MEDICATION_DATABASE = {
    "metformin": {"class": "Biguanide", "dose": "500-2000mg daily", "contraindications": "eGFR<30", "monitoring": "HbA1c, B12, renal"},
    "lisinopril": {"class": "ACE Inhibitor", "dose": "10-40mg daily", "contraindications": "Pregnancy, angioedema", "monitoring": "BP, K+, Cr"},
    "apixaban": {"class": "DOAC", "dose": "5mg BID", "contraindications": "Active bleeding", "monitoring": "Renal function, bleeding"},
    "carvedilol": {"class": "Beta-blocker", "dose": "3.125-25mg BID", "contraindications": "Severe bradycardia", "monitoring": "HR, BP"},
}


@tool
def lookup_medication(medication: str) -> str:
    """Look up medication information. Available: metformin, lisinopril, apixaban, carvedilol."""
    result = MEDICATION_DATABASE.get(medication.lower())
    if result:
        output = json.dumps({"medication": medication, **result}, indent=2)
    else:
        output = f"Not found: {medication}. Available: {', '.join(MEDICATION_DATABASE.keys())}"
    if _audit:
        _audit.log_tool_call("lookup_medication", {"medication": medication}, output)
    return output


@tool
def interpret_lab(test: str, value: float) -> str:
    """Interpret a lab value. Available: hba1c, gfr, potassium, creatinine."""
    tests = {
        "hba1c": lambda v: f"HbA1c {v}%: {'Normal' if v < 5.7 else 'Prediabetes' if v < 6.5 else 'Diabetes'}",
        "gfr": lambda v: f"GFR {v}: {'Normal' if v >= 90 else 'CKD Stage 2' if v >= 60 else 'CKD Stage 3' if v >= 30 else 'CKD Stage 4' if v >= 15 else 'CKD Stage 5'}",
        "potassium": lambda v: f"K+ {v}: {'Low' if v < 3.5 else 'Normal' if v <= 5.0 else 'HIGH'}",
        "creatinine": lambda v: f"Cr {v}: {'Normal' if 0.7 <= v <= 1.3 else 'Abnormal'}",
    }
    fn = tests.get(test.lower())
    output = fn(value) if fn else f"Unknown test: {test}"
    if _audit:
        _audit.log_tool_call("interpret_lab", {"test": test, "value": value}, output)
    return output


@tool
def safety_check(medications: str, conditions: str, age: int) -> str:
    """Run a safety check. Medications and conditions should be comma-separated."""
    warnings = []
    meds = [m.strip().lower() for m in medications.split(",")]
    conds = [c.strip().lower() for c in conditions.split(",")]

    if age >= 65 and len(meds) >= 5:
        warnings.append("Polypharmacy in elderly — review for deprescribing")
    if "ckd" in " ".join(conds) and "metformin" in meds:
        warnings.append("Metformin with CKD — verify GFR ≥30")
    if "pregnancy" in conds and any(m in meds for m in ["lisinopril", "enalapril"]):
        warnings.append("ACEi in pregnancy — CONTRAINDICATED")

    passed = len(warnings) == 0
    result = "✅ No safety concerns" if passed else "⚠️ " + "; ".join(warnings)

    if _audit:
        _audit.log_safety_check("medication_safety", result, passed)
    return result


audited_tools = [lookup_medication, interpret_lab, safety_check]


# ============================================================
# Audited Pipeline State
# ============================================================

class AuditedState(TypedDict):
    patient_case: str
    clinical_analysis: str
    recommendation: str
    safety_review: str
    final_output: str


# ============================================================
# Audited Pipeline Nodes
# ============================================================

def audited_analyze(state: AuditedState) -> dict:
    """Analyze with audit logging"""
    prompt = f"Analyze this clinical case:\n{state['patient_case']}"
    response = llm.invoke(prompt)
    if _audit:
        _audit.log_llm_call("analyze_agent", prompt[:100], response.content[:150])
    return {"clinical_analysis": response.content}


def audited_recommend(state: AuditedState) -> dict:
    """Recommend with audit logging"""
    prompt = (f"Based on this analysis, provide recommendations with doses:\n"
              f"Case: {state['patient_case'][:200]}\nAnalysis: {state['clinical_analysis'][:300]}")
    response = llm.invoke(prompt)
    if _audit:
        _audit.log_llm_call("recommend_agent", prompt[:100], response.content[:150])
        _audit.log_decision("recommend_agent", "treatment_recommendation",
                           response.content[:200], risk="info")
    return {"recommendation": response.content}


def audited_safety(state: AuditedState) -> dict:
    """Safety review with audit logging"""
    prompt = (f"Review for safety issues:\nCase: {state['patient_case'][:200]}\n"
              f"Recommendation: {state['recommendation'][:300]}")
    response = llm.invoke(prompt)
    has_concern = "⚠" in response.content or "concern" in response.content.lower()
    if _audit:
        _audit.log_llm_call("safety_agent", prompt[:100], response.content[:150])
        _audit.log_safety_check("llm_safety_review", response.content[:200], not has_concern)
    return {"safety_review": response.content}


def audited_finalize(state: AuditedState) -> dict:
    """Finalize with audit logging"""
    output = (f"CLINICAL REPORT\n{'=' * 40}\n"
              f"Analysis: {state['clinical_analysis'][:200]}...\n"
              f"Recommendation: {state['recommendation'][:200]}...\n"
              f"Safety: {state['safety_review'][:200]}...")
    if _audit:
        _audit.log_output(output)
    return {"final_output": output}


def build_audited_pipeline():
    graph = StateGraph(AuditedState)
    graph.add_node("analyze", audited_analyze)
    graph.add_node("recommend", audited_recommend)
    graph.add_node("safety", audited_safety)
    graph.add_node("finalize", audited_finalize)
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "recommend")
    graph.add_edge("recommend", "safety")
    graph.add_edge("safety", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


# ============================================================
# DEMO 1: Audited Pipeline
# ============================================================

def demo_audited_pipeline():
    """Run pipeline with full audit trail"""
    global _audit
    print("\n" + "=" * 70)
    print("DEMO 1: AUDITED CLINICAL PIPELINE")
    print("=" * 70)
    print("""
    Every step is logged: LLM calls, tool calls, decisions, safety checks.
    The audit trail shows EXACTLY what happened and when.
    """)

    _audit = AuditLogger()
    app = build_audited_pipeline()

    case = ("72-year-old male, CKD Stage 3 (GFR 38), T2DM (HbA1c 8.1%), "
            "HTN on lisinopril 20mg, metformin 2000mg. BP 152/88.")

    _audit.log_input(case)
    result = app.invoke({"patient_case": case})

    print(f"\n  CLINICAL OUTPUT (excerpt):")
    print(f"  {result.get('final_output', 'N/A')[:300]}")

    print(_audit.format_readable())

    summary = _audit.get_summary()
    print(f"  AUDIT SUMMARY:")
    print(f"    Total entries:  {summary['total_entries']}")
    print(f"    Total time:     {summary['total_time_ms']}ms")
    print(f"    Risk events:    {summary['risk_events']}")
    print(f"    Actions:        {summary['actions']}")


# ============================================================
# DEMO 2: Audited Agent with Tools
# ============================================================

def demo_audited_agent():
    """Agent with tool calls — each tool call is logged"""
    global _audit
    print("\n" + "=" * 70)
    print("DEMO 2: AUDITED AGENT WITH TOOLS")
    print("=" * 70)
    print("""
    When the agent calls tools, each call is logged in the audit trail.
    This tracks: what tool, what arguments, what result.
    """)

    _audit = AuditLogger()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a clinical agent. Use your tools to look up medications, "
                   "interpret labs, and run safety checks. Be thorough. Educational only."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, audited_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=audited_tools, verbose=False, max_iterations=6)

    question = ("Patient is 75-year-old on metformin and lisinopril with CKD. "
                "GFR is 28, potassium is 5.4. Is this medication list safe?")

    _audit.log_input(question)
    result = executor.invoke({"input": question})
    _audit.log_output(result["output"])

    print(f"\n  AGENT RESPONSE:")
    print(f"  {result['output'][:400]}")
    print(_audit.format_readable())


# ============================================================
# DEMO 3: Audit Export & Compliance
# ============================================================

def demo_audit_export():
    """Show audit trail in different export formats"""
    global _audit
    print("\n" + "=" * 70)
    print("DEMO 3: AUDIT EXPORT FOR COMPLIANCE")
    print("=" * 70)

    _audit = AuditLogger(session_id="COMPLIANCE-DEMO-001")
    app = build_audited_pipeline()

    case = "55-year-old female with depression on sertraline 100mg, now pregnant."
    _audit.log_input(case)
    app.invoke({"patient_case": case})

    # Show JSON format
    print(f"\n  JSON EXPORT (excerpt):")
    json_output = _audit.to_json()
    print(f"  {json_output[:500]}...")

    # Show summary
    summary = _audit.get_summary()
    print(f"\n  COMPLIANCE SUMMARY:")
    print(f"    Session ID:     {summary['session_id']}")
    print(f"    Total entries:  {summary['total_entries']}")
    print(f"    Risk events:    {summary['risk_events']}")
    print(f"    Actions:        {json.dumps(summary['actions'])}")

    # Save to file
    filepath = "/tmp/audit_trail_demo.json"
    _audit.save(filepath)
    print(f"\n  Saved audit trail to: {filepath}")

    print("""
    COMPLIANCE USES:
      • HIPAA: Prove who accessed what patient data and when
      • FDA: Show AI decision-making process for audits
      • Quality: Track tool usage patterns and safety events
      • Legal: Timestamped evidence of AI reasoning
      • Improvement: Analyze where the system struggles
    """)


# ============================================================
# DEMO 4: Interactive with Audit
# ============================================================

def demo_interactive():
    """Interactive agent with live audit trail"""
    global _audit
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE WITH LIVE AUDIT")
    print("=" * 70)
    print("  Every question generates an audit trail.")
    print("  Type 'audit' to see the trail, 'quit' to exit.\n")

    _audit = AuditLogger()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a clinical decision support agent with medication lookup, "
                   "lab interpretation, and safety check tools. Educational only."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, audited_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=audited_tools, verbose=False, max_iterations=6)

    while True:
        question = input("  You: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if question.lower() == 'audit':
            print(_audit.format_readable())
            summary = _audit.get_summary()
            print(f"  Entries: {summary['total_entries']}, Risk events: {summary['risk_events']}")
            continue
        if not question:
            continue

        _audit.log_input(question)
        result = executor.invoke({"input": question})
        _audit.log_output(result["output"])
        print(f"\n  Agent: {result['output'][:400]}\n")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 3: AUDIT TRAIL FOR COMPLIANCE")
    print("=" * 70)
    print("""
    Logs every agent action with timestamps for regulatory compliance.
    Tracks: inputs, LLM calls, tool calls, decisions, safety checks, outputs.

    Choose a demo:
      1 → Audited clinical pipeline
      2 → Audited agent with tools
      3 → Audit export for compliance
      4 → Interactive with live audit
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1": demo_audited_pipeline()
    elif choice == "2": demo_audited_agent()
    elif choice == "3": demo_audit_export()
    elif choice == "4": demo_interactive()
    elif choice == "5":
        demo_audited_pipeline()
        demo_audited_agent()
        demo_audit_export()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. EVERY ACTION LOGGED: In healthcare AI, you must log EVERY input,
   tool call, LLM call, decision, and output. "Show me exactly what
   the AI did" must have a clear answer.

2. STRUCTURED LOGGING: Each audit entry has: timestamp, session ID,
   action type, agent name, risk level, and details. This structure
   enables searching, filtering, and compliance reporting.

3. RISK FLAGGING: Safety checks and high-risk decisions get flagged
   in the audit trail. This lets compliance officers quickly find
   events that need review.

4. EXPORT FORMATS: The audit trail can be exported as:
   - JSON (for programmatic analysis)
   - Readable text (for human review)
   - Saved to file (for long-term retention)

5. PRODUCTION REQUIREMENTS:
   - Store audit trails in a database, not just files
   - Immutable logs (append-only, cannot be deleted)
   - Encrypt PHI in audit entries
   - Retention policies (HIPAA: 6 years minimum)
   - Real-time alerting on critical risk events
   - Aggregate analytics for system improvement
"""

if __name__ == "__main__":
    main()
