"""
Exercise 3: Quality Check Agent — Reviews Another Agent's Output

Skills practiced:
- Building a quality assurance agent that validates other agents
- Implementing structured quality criteria
- Understanding feedback loops in multi-agent systems
- Production QA patterns for healthcare AI

Key insight: In healthcare, a second opinion improves quality. This
  exercise adds a "quality check" agent that reviews another agent's
  clinical output against structured criteria — like a peer review
  process that catches errors and improves recommendations.
"""

import os
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# Quality Criteria
# ============================================================

QUALITY_CRITERIA = {
    "completeness": "Does the assessment include differential diagnosis, workup, and treatment plan?",
    "evidence_based": "Are recommendations supported by clinical guidelines?",
    "safety": "Are contraindications, interactions, and monitoring addressed?",
    "patient_specificity": "Is the plan tailored to THIS patient (age, conditions, labs)?",
    "clarity": "Is the recommendation clear enough for a clinician to act on?",
}


# ============================================================
# State
# ============================================================

class QAState(TypedDict):
    patient_case: str
    clinical_output: str          # The output being reviewed
    quality_score: str            # Structured quality assessment
    quality_pass: str             # "pass" or "fail"
    improvement_suggestions: str  # What to improve
    revised_output: str           # Output after QA feedback (if needed)
    final_output: str


# ============================================================
# Pipeline Agents
# ============================================================

def clinical_agent(state: QAState) -> dict:
    """Primary clinical agent — generates the initial assessment"""
    response = llm.invoke(
        f"""You are a clinical decision support agent. Provide a complete
clinical assessment for this patient.

Patient: {state['patient_case']}

Include:
1. Clinical impression
2. Differential diagnosis (top 3)
3. Recommended workup
4. Treatment plan with specific medications
5. Follow-up plan

Educational purposes only."""
    )
    return {"clinical_output": response.content}


def quality_check_agent(state: QAState) -> dict:
    """QA agent — Reviews the clinical output against quality criteria"""

    criteria_text = "\n".join(f"  - {k}: {v}" for k, v in QUALITY_CRITERIA.items())

    response = llm.invoke(
        f"""You are a CLINICAL QUALITY REVIEWER. Your job is to evaluate
the clinical assessment below against quality criteria.

PATIENT CASE:
{state['patient_case']}

CLINICAL ASSESSMENT TO REVIEW:
{state['clinical_output']}

QUALITY CRITERIA:
{criteria_text}

For EACH criterion, rate as:
  PASS (meets standard) or FAIL (does not meet standard)
  with a brief explanation.

Then provide:
  OVERALL: PASS (if all criteria pass) or FAIL (if any criterion fails)
  IMPROVEMENT SUGGESTIONS: specific, actionable changes needed

Be thorough but fair. Be specific about what's missing or incorrect."""
    )

    # Determine if it passes
    content = response.content.lower()
    overall_pass = "pass" if "overall: pass" in content or content.count("fail") <= 1 else "fail"

    return {
        "quality_score": response.content,
        "quality_pass": overall_pass,
    }


def extract_improvements(state: QAState) -> dict:
    """Extract specific improvement suggestions"""
    response = llm.invoke(
        f"Extract the specific improvement suggestions from this QA review. "
        f"List them as numbered action items.\n\n"
        f"QA Review:\n{state['quality_score']}"
    )
    return {"improvement_suggestions": response.content}


def revise_output(state: QAState) -> dict:
    """Revise the clinical output based on QA feedback"""
    if state["quality_pass"] == "pass":
        return {"revised_output": state["clinical_output"]}

    response = llm.invoke(
        f"""You are a clinical agent. Your previous assessment received QA feedback.
Revise your assessment to address the identified issues.

ORIGINAL ASSESSMENT:
{state['clinical_output']}

QA FEEDBACK:
{state['quality_score']}

SPECIFIC IMPROVEMENTS NEEDED:
{state['improvement_suggestions']}

Provide a REVISED, improved assessment that addresses all feedback.
Educational purposes only."""
    )
    return {"revised_output": response.content}


def finalize(state: QAState) -> dict:
    """Generate final output with QA metadata"""
    qa_status = "PASSED" if state["quality_pass"] == "pass" else "REVISED (after QA feedback)"

    response = llm.invoke(
        f"Create a final clinical note with QA status.\n\n"
        f"Assessment: {state['revised_output']}\n"
        f"QA Status: {qa_status}\n\n"
        f"Format as a structured clinical note with QA status footer."
    )
    return {"final_output": response.content}


# ============================================================
# Build Pipeline
# ============================================================

def route_by_quality(state: QAState) -> str:
    return "revise" if state["quality_pass"] == "fail" else "accept"


def build_qa_pipeline():
    graph = StateGraph(QAState)

    graph.add_node("clinical", clinical_agent)
    graph.add_node("quality_check", quality_check_agent)
    graph.add_node("extract_improvements", extract_improvements)
    graph.add_node("revise", revise_output)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("clinical")
    graph.add_edge("clinical", "quality_check")
    graph.add_edge("quality_check", "extract_improvements")

    # Route: if QA fails → revise; if QA passes → finalize directly
    graph.add_conditional_edges(
        "extract_improvements",
        route_by_quality,
        {"revise": "revise", "accept": "finalize"}
    )

    graph.add_edge("revise", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# ============================================================
# DEMO 1: QA Pipeline in Action
# ============================================================

def demo_qa_pipeline():
    """Run a case through the QA pipeline"""
    print("\n" + "=" * 70)
    print("DEMO 1: QUALITY CHECK PIPELINE")
    print("=" * 70)
    print("""
    Pipeline: Clinical Agent → Quality Check → [Revise if needed] → Finalize
    """)

    app = build_qa_pipeline()

    result = app.invoke({
        "patient_case": "68-year-old male with CKD Stage 3, diabetes, HTN. "
                        "Presenting with worsening edema and fatigue. "
                        "Labs: GFR 35, HbA1c 8.2%, K+ 5.4, Cr 1.9. "
                        "Meds: metformin 2000mg, lisinopril 20mg."
    })

    print(f"\n  CLINICAL OUTPUT (original):")
    print(f"  {result.get('clinical_output', 'N/A')[:400]}...")

    print(f"\n  QUALITY CHECK:")
    print(f"  {result.get('quality_score', 'N/A')[:500]}...")

    print(f"\n  QA RESULT: {result.get('quality_pass', '?').upper()}")

    if result.get("quality_pass") == "fail":
        print(f"\n  IMPROVEMENTS NEEDED:")
        print(f"  {result.get('improvement_suggestions', 'N/A')[:300]}...")
        print(f"\n  REVISED OUTPUT:")
        print(f"  {result.get('revised_output', 'N/A')[:400]}...")

    print(f"\n  FINAL OUTPUT:")
    print(f"  {result.get('final_output', 'N/A')[:500]}")


# ============================================================
# DEMO 2: Quality Criteria Breakdown
# ============================================================

def demo_criteria_breakdown():
    """Show how each quality criterion is evaluated"""
    print("\n" + "=" * 70)
    print("DEMO 2: QUALITY CRITERIA BREAKDOWN")
    print("=" * 70)
    print("""
    The QA agent evaluates against 5 criteria:
    """)

    for k, v in QUALITY_CRITERIA.items():
        print(f"    {k.upper()}: {v}")

    app = build_qa_pipeline()

    cases = [
        {
            "label": "Complex case (multiple issues)",
            "patient_case": "75-year-old on warfarin (INR 3.5), new GI bleed, hemoglobin 8.2"
        },
        {
            "label": "Simple case (routine)",
            "patient_case": "30-year-old healthy female with mild seasonal allergies"
        },
    ]

    for case in cases:
        print(f"\n{'─' * 60}")
        print(f"  CASE: {case['label']}")
        result = app.invoke({"patient_case": case["patient_case"]})
        print(f"  QA PASS: {result.get('quality_pass', '?').upper()}")
        print(f"  Criteria review:\n  {result.get('quality_score', 'N/A')[:500]}...")


# ============================================================
# DEMO 3: Before vs After QA
# ============================================================

def demo_before_after():
    """Compare output before and after QA review"""
    print("\n" + "=" * 70)
    print("DEMO 3: BEFORE vs AFTER QA REVIEW")
    print("=" * 70)

    app = build_qa_pipeline()

    case = ("82-year-old female, 50kg, CKD Stage 4 (GFR 18), AFib on apixaban 5mg BID, "
            "diabetes on metformin 1000mg BID, K+ 5.8. Falls risk. New confusion and lethargy.")

    result = app.invoke({"patient_case": case})

    print(f"\n  BEFORE QA:")
    print(f"  {result.get('clinical_output', 'N/A')[:500]}...")

    print(f"\n  QA RESULT: {result.get('quality_pass', '?').upper()}")

    if result.get("quality_pass") == "fail":
        print(f"\n  ISSUES FOUND:")
        print(f"  {result.get('improvement_suggestions', 'N/A')[:300]}...")
        print(f"\n  AFTER QA (REVISED):")
        print(f"  {result.get('revised_output', 'N/A')[:500]}...")
    else:
        print(f"  Assessment passed QA on first try.")

    print("""
    TYPICAL QA CATCHES:
      • Missed renal dose adjustments (metformin with GFR 18)
      • Apixaban dose should be 2.5mg BID (age/weight/Cr criteria)
      • Hyperkalemia (K+ 5.8) not addressed
      • Falls risk medication review needed
    """)


# ============================================================
# DEMO 4: Interactive
# ============================================================

def demo_interactive():
    """Interactive QA pipeline"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE QUALITY CHECK")
    print("=" * 70)
    print("  Type 'quit' to exit.\n")

    app = build_qa_pipeline()

    while True:
        case = input("  Patient case: ").strip()
        if case.lower() in ['quit', 'exit', 'q']:
            break
        if not case:
            continue

        print("\n  Running QA pipeline...\n")
        result = app.invoke({"patient_case": case})

        print(f"  QA RESULT: {result.get('quality_pass', '?').upper()}")
        if result.get("quality_pass") == "fail":
            print(f"  Issues: {result.get('improvement_suggestions', 'N/A')[:300]}")
        print(f"\n  Final: {result.get('final_output', 'N/A')[:500]}\n")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 3: QUALITY CHECK AGENT")
    print("=" * 70)
    print("""
    Adds a QA agent that reviews clinical output against
    structured criteria and triggers revision if needed.

    Pipeline: Clinical → QA → [Revise if fail] → Finalize

    Choose a demo:
      1 → QA pipeline in action
      2 → Quality criteria breakdown
      3 → Before vs after QA
      4 → Interactive
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1": demo_qa_pipeline()
    elif choice == "2": demo_criteria_breakdown()
    elif choice == "3": demo_before_after()
    elif choice == "4": demo_interactive()
    elif choice == "5":
        demo_qa_pipeline()
        demo_criteria_breakdown()
        demo_before_after()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. QA AS A SEPARATE AGENT: The quality check agent has ONE job —
   evaluate the clinical output against structured criteria.
   It doesn't generate clinical content; it REVIEWS it.

2. STRUCTURED CRITERIA: Using defined criteria (completeness,
   evidence-based, safety, specificity, clarity) makes QA
   consistent and auditable. Not just "is this good?"

3. CONDITIONAL REVISION: If QA fails, the clinical agent revises
   with specific feedback. If QA passes, skip revision (efficiency).
   This is a feedback loop within the pipeline.

4. HEALTHCARE QA PARALLEL: In hospitals, clinical decision support
   systems have quality review processes. Peer review, utilization
   review, and clinical pathway adherence are all QA patterns.

5. PRODUCTION ENHANCEMENT: In production, you might:
   - Run QA multiple times (iterative improvement)
   - Use different criteria for different case types
   - Track QA pass rates to monitor system quality
   - Log all QA results for compliance audits
"""

if __name__ == "__main__":
    main()
