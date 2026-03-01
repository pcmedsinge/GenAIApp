"""
Exercise 8: LangGraph Send — Dynamic Fan-Out at Runtime

Skills practiced:
- Using Send() to dynamically create parallel branches at runtime
- Dynamic fan-out: number of branches determined by data, not graph structure
- Collecting results from dynamic branches
- Healthcare use case: parallel multi-specialist consultation

Why this matters:
  In a normal StateGraph, edges are FIXED at compile time. If you have
  3 specialists, you wire 3 edges. But what if the number of specialists
  depends on the patient's conditions? That's DYNAMIC FAN-OUT.

  Send() lets you spawn branches at RUNTIME. The graph structure adapts
  to each input. This is a key pattern for:
  - Multi-specialist consultations (number varies per patient)
  - Parallel lab processing (number of tests varies)
  - Multi-document analysis (number of documents varies)

Architecture:

  Static fan-out (add_edge):        Dynamic fan-out (Send):
  ┌──────────┐                      ┌──────────┐
  │ analyzer  │                      │ analyzer  │
  └─┬──┬──┬──┘                      └─────┬────┘
    │  │  │  (fixed 3)                     │ (Send × N at runtime)
    ▼  ▼  ▼                          ┌─────┼─────┐ ... ┐
  [A] [B] [C]                        ▼     ▼     ▼     ▼
    │  │  │                         [S1]  [S2]  [S3]  [SN]
    ▼  ▼  ▼                          │     │     │     │
  ┌──────────┐                      ┌─────────────────────┐
  │ combiner  │                      │     combiner         │
  └──────────┘                      └─────────────────────┘

  The key: Send("node_name", data) creates a branch to process
  specific data. Multiple Send()s = multiple parallel branches.

Healthcare parallel:
  A triage nurse identifies conditions → spawns parallel specialist
  consultations → collects all recommendations → synthesizes plan.
"""

import os
import json
import operator
from typing import Annotated, TypedDict
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# DEMO 1: Basic Send — Dynamic Parallel Processing
# ============================================================

def demo_basic_send():
    """Show Send() for dynamic parallel processing of lab results."""
    print("\n" + "=" * 70)
    print("  DEMO 1: BASIC SEND — DYNAMIC PARALLEL LAB PROCESSING")
    print("=" * 70)
    print("""
  Scenario: Process a variable number of lab results in parallel.
  The number of labs is NOT known at graph compile time — it depends
  on what was ordered for each patient.

  Send("process_lab", {"lab": lab_data}) spawns one branch per lab.
  """)

    class OverallState(TypedDict):
        labs: list[dict]
        results: Annotated[list[str], operator.add]  # Collected from branches

    class LabState(TypedDict):
        lab: dict

    # Reference ranges
    ranges = {
        "troponin": (0.0, 0.04, "ng/mL"),
        "glucose": (70, 100, "mg/dL"),
        "potassium": (3.5, 5.0, "mEq/L"),
        "sodium": (136, 145, "mEq/L"),
        "creatinine": (0.7, 1.3, "mg/dL"),
        "hemoglobin": (12.0, 17.5, "g/dL"),
        "wbc": (4.5, 11.0, "K/uL"),
    }

    def router(state: OverallState) -> list[Send]:
        """Dynamically send each lab to process_lab node."""
        sends = []
        for lab in state["labs"]:
            sends.append(Send("process_lab", {"lab": lab}))
        print(f"    📤 Router spawning {len(sends)} parallel branches")
        return sends

    def process_lab(state: LabState) -> dict:
        """Process a single lab result (runs in parallel per Send)."""
        lab = state["lab"]
        name = lab["name"].lower()
        value = lab["value"]

        if name in ranges:
            low, high, unit = ranges[name]
            if value < low:
                status = "LOW ⬇️"
            elif value > high:
                status = "HIGH ⬆️"
            else:
                status = "NORMAL ✓"
            result = f"{lab['name']}: {value} {unit} — {status} (ref: {low}-{high})"
        else:
            result = f"{lab['name']}: {value} — no reference range available"

        print(f"    🔬 Processed: {result}")
        return {"results": [result]}

    # Build graph
    graph = StateGraph(OverallState)
    graph.add_node("process_lab", process_lab)
    graph.add_conditional_edges(START, router, ["process_lab"])
    graph.add_edge("process_lab", END)

    app = graph.compile()

    # Test with different numbers of labs
    test_cases = [
        {
            "name": "Chest Pain Workup (3 labs)",
            "labs": [
                {"name": "troponin", "value": 0.45},
                {"name": "glucose", "value": 95},
                {"name": "creatinine", "value": 1.1},
            ],
        },
        {
            "name": "Comprehensive Panel (6 labs)",
            "labs": [
                {"name": "troponin", "value": 0.02},
                {"name": "glucose", "value": 280},
                {"name": "potassium", "value": 5.8},
                {"name": "sodium", "value": 132},
                {"name": "hemoglobin", "value": 8.5},
                {"name": "wbc", "value": 18.2},
            ],
        },
    ]

    for tc in test_cases:
        print(f"\n  ─── {tc['name']} ───")
        result = app.invoke({"labs": tc["labs"], "results": []})
        print(f"  Summary ({len(result['results'])} results):")
        for r in result["results"]:
            print(f"    {r}")

    print(f"\n  KEY INSIGHT: Send() creates branches at runtime.")
    print(f"  3 labs → 3 branches. 6 labs → 6 branches. Graph adapts to data.")


# ============================================================
# DEMO 2: Send + LLM — Multi-Specialist Consultation
# ============================================================

def demo_specialist_consultation():
    """Dynamic multi-specialist consultation using Send + LLM."""
    print("\n" + "=" * 70)
    print("  DEMO 2: SEND + LLM — MULTI-SPECIALIST CONSULTATION")
    print("=" * 70)
    print("""
  Scenario: A triage node identifies which specialists are needed,
  then Send() spawns parallel LLM consultations.

  Patient with chest pain + diabetes + renal issues →
  triage detects 3 conditions → spawns 3 specialist consults.

  Patient with just a sprain → 1 specialist consult.
  The number of consultations adapts dynamically.
  """)

    class TriageState(TypedDict):
        patient_info: str
        identified_conditions: list[str]
        specialist_opinions: Annotated[list[str], operator.add]
        synthesis: str

    class ConsultState(TypedDict):
        patient_info: str
        condition: str
        specialist: str

    specialists = {
        "cardiac": "You are a cardiologist. Give a brief assessment and recommendation.",
        "endocrine": "You are an endocrinologist. Give a brief assessment and recommendation.",
        "renal": "You are a nephrologist. Give a brief assessment and recommendation.",
        "pulmonary": "You are a pulmonologist. Give a brief assessment and recommendation.",
        "orthopedic": "You are an orthopedic specialist. Give a brief assessment.",
        "neurological": "You are a neurologist. Give a brief assessment.",
    }

    condition_to_specialist = {
        "chest pain": "cardiac",
        "diabetes": "endocrine",
        "renal failure": "renal",
        "shortness of breath": "pulmonary",
        "fracture": "orthopedic",
        "seizure": "neurological",
    }

    def triage(state: TriageState) -> dict:
        """Identify conditions — determines how many specialists are needed."""
        patient = state["patient_info"].lower()
        conditions = []
        for condition in condition_to_specialist:
            if condition in patient:
                conditions.append(condition)
        if not conditions:
            conditions = ["general assessment"]
        print(f"    🏥 Triage identified {len(conditions)} condition(s): {conditions}")
        return {"identified_conditions": conditions}

    def route_to_specialists(state: TriageState) -> list[Send]:
        """Spawn one Send per identified condition."""
        sends = []
        for condition in state["identified_conditions"]:
            specialist = condition_to_specialist.get(condition, "cardiac")
            sends.append(Send("consult", {
                "patient_info": state["patient_info"],
                "condition": condition,
                "specialist": specialist,
            }))
        print(f"    📤 Sending to {len(sends)} specialist(s)")
        return sends

    def consult(state: ConsultState) -> dict:
        """One specialist consultation (via LLM)."""
        specialist = state["specialist"]
        prompt = specialists.get(specialist, "You are a general physician.")
        response = llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user",
             "content": f"Patient: {state['patient_info']}\nFocus: {state['condition']}\n"
                        f"Give a 2-3 sentence assessment."},
        ])
        opinion = f"[{specialist.upper()}] {response.content}"
        print(f"    👨‍⚕️ {specialist}: done")
        return {"specialist_opinions": [opinion]}

    def synthesize(state: TriageState) -> dict:
        """Combine all specialist opinions into a care plan."""
        opinions = "\n".join(state["specialist_opinions"])
        response = llm.invoke([
            {"role": "system",
             "content": "You are a care coordinator. Synthesize specialist opinions "
                        "into a brief unified care plan (3-5 bullet points)."},
            {"role": "user",
             "content": f"Patient: {state['patient_info']}\n\nSpecialist opinions:\n{opinions}"},
        ])
        return {"synthesis": response.content}

    # Build graph
    graph = StateGraph(TriageState)
    graph.add_node("triage", triage)
    graph.add_node("consult", consult)
    graph.add_node("synthesize", synthesize)

    graph.set_entry_point("triage")
    graph.add_conditional_edges("triage", route_to_specialists, ["consult"])
    graph.add_edge("consult", "synthesize")
    graph.add_edge("synthesize", END)

    app = graph.compile()

    # Test cases with different numbers of conditions
    patients = [
        "65M with chest pain, diabetes, and renal failure. Troponin elevated.",
        "30F with fracture after fall. No other complaints.",
        "72M with chest pain and shortness of breath. History of CHF.",
    ]

    for patient in patients:
        print(f"\n  ─── Patient: {patient[:60]}... ───")
        result = app.invoke({
            "patient_info": patient,
            "identified_conditions": [],
            "specialist_opinions": [],
            "synthesis": "",
        })
        print(f"\n  Specialist opinions ({len(result['specialist_opinions'])}):")
        for op in result["specialist_opinions"]:
            print(f"    {op[:120]}...")
        print(f"\n  Synthesized plan:\n    {result['synthesis'][:300]}...")

    print(f"\n  KEY INSIGHT: Send() + LLM = dynamic parallel LLM calls.")
    print(f"  Patient complexity determines the number of consultations.")


# ============================================================
# DEMO 3: Send with Aggregation — Voting Pattern
# ============================================================

def demo_voting_pattern():
    """Use Send to create a multi-reviewer voting pattern."""
    print("\n" + "=" * 70)
    print("  DEMO 3: SEND + VOTING — MULTI-REVIEWER CLINICAL REVIEW")
    print("=" * 70)
    print("""
  Scenario: A clinical note is reviewed by multiple AI reviewers,
  each with a different persona. Their votes are aggregated.

  This is the "fan-out → aggregate" pattern using Send().
  Number of reviewers is configurable at runtime.
  """)

    class ReviewState(TypedDict):
        note: str
        reviewers: list[str]
        reviews: Annotated[list[dict], operator.add]
        decision: str

    class SingleReviewState(TypedDict):
        note: str
        reviewer_type: str

    reviewer_prompts = {
        "safety": "You are a patient safety reviewer. Check for safety concerns. "
                  "Respond with JSON: {\"approved\": true/false, \"concern\": \"...\"}",
        "completeness": "You are a documentation reviewer. Check for missing info. "
                       "Respond with JSON: {\"approved\": true/false, \"concern\": \"...\"}",
        "coding": "You are a medical coding reviewer. Check ICD-10 accuracy. "
                  "Respond with JSON: {\"approved\": true/false, \"concern\": \"...\"}",
        "compliance": "You are a HIPAA compliance reviewer. Check for PHI issues. "
                      "Respond with JSON: {\"approved\": true/false, \"concern\": \"...\"}",
    }

    def route_to_reviewers(state: ReviewState) -> list[Send]:
        """Spawn a reviewer per requested reviewer type."""
        sends = []
        for reviewer in state["reviewers"]:
            sends.append(Send("review", {
                "note": state["note"],
                "reviewer_type": reviewer,
            }))
        print(f"    📤 Sending to {len(sends)} reviewer(s): {state['reviewers']}")
        return sends

    def review(state: SingleReviewState) -> dict:
        """Single reviewer assessment."""
        prompt = reviewer_prompts.get(
            state["reviewer_type"],
            "Review this note. Respond with JSON: {\"approved\": true/false, \"concern\": \"...\"}"
        )
        response = llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Clinical note:\n{state['note']}"},
        ])

        try:
            # Try to extract JSON
            text = response.content
            # Handle markdown code blocks
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text.strip())
        except (json.JSONDecodeError, IndexError):
            result = {"approved": True, "concern": "Unable to parse review"}

        result["reviewer"] = state["reviewer_type"]
        print(f"    📋 {state['reviewer_type']}: {'✅ Approved' if result.get('approved') else '❌ Flagged'}")
        return {"reviews": [result]}

    def aggregate(state: ReviewState) -> dict:
        """Aggregate votes and decide."""
        approvals = sum(1 for r in state["reviews"] if r.get("approved", False))
        total = len(state["reviews"])
        concerns = [r for r in state["reviews"] if not r.get("approved", True)]

        if approvals == total:
            decision = f"APPROVED ({approvals}/{total} reviewers approved)"
        elif approvals > total / 2:
            decision = f"CONDITIONALLY APPROVED ({approvals}/{total})"
        else:
            decision = f"REJECTED ({approvals}/{total})"

        if concerns:
            concern_list = "; ".join(
                f"{c['reviewer']}: {c.get('concern', 'N/A')}" for c in concerns
            )
            decision += f". Concerns: {concern_list}"

        return {"decision": decision}

    # Build graph
    graph = StateGraph(ReviewState)
    graph.add_node("review", review)
    graph.add_node("aggregate", aggregate)

    graph.add_conditional_edges(START, route_to_reviewers, ["review"])
    graph.add_edge("review", "aggregate")
    graph.add_edge("aggregate", END)

    app = graph.compile()

    # Test 1: 2 reviewers
    print(f"\n  ─── Test 1: 2 Reviewers ───")
    r1 = app.invoke({
        "note": "Patient: John Doe (DOB 1/1/1960). Chief complaint: chest pain. "
                "Assessment: Acute MI. Plan: Heparin drip, cardiology consult.",
        "reviewers": ["safety", "completeness"],
        "reviews": [],
        "decision": "",
    })
    print(f"  Decision: {r1['decision'][:150]}")

    # Test 2: 4 reviewers
    print(f"\n  ─── Test 2: 4 Reviewers ───")
    r2 = app.invoke({
        "note": "Pt seen today for f/u. Doing better. Continue meds. RTC 2 weeks.",
        "reviewers": ["safety", "completeness", "coding", "compliance"],
        "reviews": [],
        "decision": "",
    })
    print(f"  Decision: {r2['decision'][:200]}")

    print(f"\n  KEY INSIGHT: Send() enables dynamic voting/review patterns.")
    print(f"  Add or remove reviewers without changing the graph structure.")


# ============================================================
# DEMO 4: Send with Map-Reduce — Parallel Document Analysis
# ============================================================

def demo_map_reduce():
    """Map-reduce pattern using Send for parallel document analysis."""
    print("\n" + "=" * 70)
    print("  DEMO 4: MAP-REDUCE — PARALLEL DOCUMENT ANALYSIS")
    print("=" * 70)
    print("""
  Classic map-reduce using Send():
  1. MAP: Send() each document to an analyzer (parallel)
  2. REDUCE: Collect all analyses into a summary

  Healthcare: Analyze multiple clinical notes to find trends.
  """)

    class MapReduceState(TypedDict):
        documents: list[dict]
        analyses: Annotated[list[str], operator.add]
        summary: str

    class DocState(TypedDict):
        doc_id: str
        content: str

    def scatter(state: MapReduceState) -> list[Send]:
        """Map phase: send each document to analyzer."""
        sends = []
        for doc in state["documents"]:
            sends.append(Send("analyze", {
                "doc_id": doc["id"],
                "content": doc["content"],
            }))
        print(f"    📤 Scattering {len(sends)} documents for parallel analysis")
        return sends

    def analyze(state: DocState) -> dict:
        """Analyze a single document."""
        response = llm.invoke([
            {"role": "system",
             "content": "Extract key findings from this clinical note in ONE line. "
                        "Format: 'Doc ID: finding1, finding2, finding3'"},
            {"role": "user", "content": f"Doc {state['doc_id']}:\n{state['content']}"},
        ])
        print(f"    📄 Analyzed doc {state['doc_id']}")
        return {"analyses": [f"[{state['doc_id']}] {response.content}"]}

    def gather(state: MapReduceState) -> dict:
        """Reduce phase: synthesize all analyses."""
        all_analyses = "\n".join(state["analyses"])
        response = llm.invoke([
            {"role": "system",
             "content": "Synthesize these clinical note analyses into a brief "
                        "trend summary (3-4 bullet points)."},
            {"role": "user", "content": f"Analyses:\n{all_analyses}"},
        ])
        return {"summary": response.content}

    # Build graph
    graph = StateGraph(MapReduceState)
    graph.add_node("analyze", analyze)
    graph.add_node("gather", gather)

    graph.add_conditional_edges(START, scatter, ["analyze"])
    graph.add_edge("analyze", "gather")
    graph.add_edge("gather", END)

    app = graph.compile()

    # Test with patient notes
    result = app.invoke({
        "documents": [
            {"id": "visit-1", "content": "Day 1: Patient admitted with chest pain. Troponin 0.45. Started heparin drip. ECG shows ST elevation V2-V4."},
            {"id": "visit-2", "content": "Day 2: Troponin trending down to 0.22. Patient underwent PCI with stent to LAD. Started dual antiplatelet therapy."},
            {"id": "visit-3", "content": "Day 3: Patient stable post-PCI. Troponin 0.10. Ambulating. Echo shows EF 45%. Started ACE inhibitor and beta-blocker."},
            {"id": "visit-4", "content": "Day 4: Discharge. Troponin normal. Patient educated on medications: aspirin, clopidogrel, atorvastatin, metoprolol, lisinopril. Follow-up in 2 weeks."},
        ],
        "analyses": [],
        "summary": "",
    })

    print(f"\n  Individual analyses:")
    for a in result["analyses"]:
        print(f"    {a[:120]}...")
    print(f"\n  Trend summary:\n    {result['summary'][:400]}")

    print(f"\n  KEY INSIGHT: Send() makes map-reduce trivial in LangGraph.")
    print(f"  Each document is processed in parallel, then results are gathered.")


# ============================================================
# DEMO 5: Send with Conditional Fan-Out Count
# ============================================================

def demo_conditional_fanout():
    """Show fan-out count determined by runtime analysis."""
    print("\n" + "=" * 70)
    print("  DEMO 5: CONDITIONAL FAN-OUT — SEVERITY-BASED PARALLELISM")
    print("=" * 70)
    print("""
  The number of branches depends on patient severity:
  - Low severity → 1 reviewer
  - Medium severity → 2 reviewers
  - High severity → 3 reviewers + escalation

  The graph doesn't know how many branches there'll be until runtime.
  """)

    class SeverityState(TypedDict):
        patient_data: str
        severity: str
        checks: Annotated[list[str], operator.add]

    class CheckState(TypedDict):
        patient_data: str
        check_type: str

    def assess_severity(state: SeverityState) -> dict:
        """Determine severity and what checks are needed."""
        data = state["patient_data"].lower()
        if any(w in data for w in ["critical", "code", "arrest", "unresponsive"]):
            severity = "HIGH"
        elif any(w in data for w in ["elevated", "abnormal", "pain"]):
            severity = "MEDIUM"
        else:
            severity = "LOW"
        print(f"    🏥 Severity assessed: {severity}")
        return {"severity": severity}

    def route_by_severity(state: SeverityState) -> list[Send]:
        """Fan-out count based on severity."""
        severity = state["severity"]
        check_types = []

        if severity == "LOW":
            check_types = ["basic_review"]
        elif severity == "MEDIUM":
            check_types = ["clinical_review", "medication_check"]
        else:
            check_types = ["clinical_review", "medication_check", "escalation_alert"]

        sends = [
            Send("perform_check", {
                "patient_data": state["patient_data"],
                "check_type": ct,
            })
            for ct in check_types
        ]
        print(f"    📤 {severity} severity → {len(sends)} parallel check(s): {check_types}")
        return sends

    def perform_check(state: CheckState) -> dict:
        """Perform a single check."""
        check_results = {
            "basic_review": f"Basic review complete for: {state['patient_data'][:40]}...",
            "clinical_review": f"Clinical review: vitals and labs assessed for {state['patient_data'][:30]}...",
            "medication_check": f"Medication check: interactions verified for {state['patient_data'][:30]}...",
            "escalation_alert": f"⚠️ ESCALATION: Senior physician notified for {state['patient_data'][:30]}...",
        }
        result = check_results.get(state["check_type"], "Unknown check")
        print(f"    ✅ {state['check_type']}: done")
        return {"checks": [result]}

    # Build graph
    graph = StateGraph(SeverityState)
    graph.add_node("assess_severity", assess_severity)
    graph.add_node("perform_check", perform_check)

    graph.set_entry_point("assess_severity")
    graph.add_conditional_edges("assess_severity", route_by_severity, ["perform_check"])
    graph.add_edge("perform_check", END)

    app = graph.compile()

    # Test different severities
    patients = [
        ("Low", "Routine follow-up. Patient doing well. Continue current meds."),
        ("Medium", "Elevated blood pressure 165/95. Chest pain with exertion."),
        ("High", "Critical troponin 2.5. Patient unresponsive. Code blue activated."),
    ]

    for expected, data in patients:
        print(f"\n  ─── Expected: {expected} ───")
        result = app.invoke({
            "patient_data": data,
            "severity": "",
            "checks": [],
        })
        print(f"  Checks performed: {len(result['checks'])}")
        for c in result["checks"]:
            print(f"    {c}")

    print(f"\n  KEY INSIGHT: Send() count can be determined by runtime analysis.")
    print(f"  Severity → fan-out count. More critical = more parallel checks.")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 8: LANGGRAPH SEND — DYNAMIC FAN-OUT")
    print("=" * 70)
    print("""
    LangGraph Send() for dynamic parallel branching:
    - Number of branches determined at RUNTIME, not compile time
    - Each Send() spawns a parallel execution branch
    - Results collected via Annotated[list, operator.add]

    Choose a demo:
      1 → Basic Send (dynamic parallel lab processing)
      2 → Multi-specialist consultation (Send + LLM)
      3 → Multi-reviewer voting pattern
      4 → Map-reduce document analysis
      5 → Conditional fan-out by severity
      6 → Run all demos
    """)

    choice = input("  Enter choice (1-6): ").strip()

    demos = {
        "1": demo_basic_send,
        "2": demo_specialist_consultation,
        "3": demo_voting_pattern,
        "4": demo_map_reduce,
        "5": demo_conditional_fanout,
    }

    if choice == "6":
        for d in demos.values():
            d()
    elif choice in demos:
        demos[choice]()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. Send() = DYNAMIC FAN-OUT
   Normal edges are fixed at compile time. Send() creates branches at runtime.
   Send("node_name", {data}) spawns one branch to process that data.
   Return multiple Send()s from a conditional edge → multiple parallel branches.

2. PATTERN: conditional_edges → [Send, Send, ...]
   The routing function returns a LIST of Send objects.
   Each Send targets the SAME node but with DIFFERENT data.
   This is the "map" phase of map-reduce.

3. Annotated[list, operator.add] = RESULT COLLECTION
   When branches return results, operator.add merges them.
   This is the "reduce" (or "gather") phase.

4. RUNTIME ADAPTABILITY:
   - 3 conditions → 3 specialists
   - 6 labs → 6 parallel analyses
   - HIGH severity → 3 checks, LOW → 1 check
   The graph structure is FIXED but the number of branches varies.

5. COMMON PATTERNS WITH Send():
   - Multi-specialist consultation (different experts per condition)
   - Map-reduce (parallel analysis → synthesis)
   - Fan-out voting (multiple reviewers → aggregate vote)
   - Severity-based parallelism (more checks for higher risk)

6. Send() vs Static Parallelism:
   Static: graph.add_edge("node_a", ["b1", "b2", "b3"])  # Always 3
   Dynamic: return [Send("process", d) for d in items]     # N at runtime
"""

if __name__ == "__main__":
    main()
