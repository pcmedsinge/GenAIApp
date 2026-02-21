"""
Project 5: Healthcare Agent Capstone — Clinical Decision Support System
Combines EVERYTHING: RAG knowledge + Agent tools + Multi-step workflows + Safety guardrails.

This is what a real healthcare AI agent looks like.
It can:
  1. Look up medication info (Agent tools)
  2. Interpret lab values (Agent tools)
  3. Search clinical guidelines (RAG, from Level 2)
  4. Apply safety checks (guardrails)
  5. Generate structured clinical notes (templates)

Builds on: Everything from Levels 1, 2, and 3
"""

import os
import json
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# Healthcare Tools (Agent capabilities)
# ============================================================

# Simulated medical knowledge base (in production: ChromaDB from Level 2)
CLINICAL_GUIDELINES = {
    "hypertension": "Target BP <130/80 for most adults. First-line: ACEi, ARBs, CCBs, thiazides. Start monotherapy, add second agent at 4-6 weeks if not at target. Black patients: CCB or thiazide preferred initially.",
    "diabetes_type2": "First-line: metformin. Add GLP-1 agonist if CVD/obesity, SGLT2i if HF/CKD. HbA1c target <7% for most. Monitor every 3 months until stable.",
    "heart_failure": "Four pillars for HFrEF: ARNI (or ACEi/ARB), beta-blocker, MRA, SGLT2i. All four improve survival. Titrate to target doses. Restrict sodium <2g and fluid 1.5-2L.",
    "depression": "SSRIs first-line (sertraline, escitalopram). Allow 4-6 weeks for response. Combine with CBT for moderate-severe. Monitor suicidal ideation especially <25yo. Continue 6-12 months after remission.",
    "ckd": "Stage by GFR. ACEi/ARB for proteinuria. Avoid NSAIDs. Refer nephrology at Stage 4. Target BP <130/80. Monitor K+, GFR every 3-6 months.",
    "anticoagulation": "AFib: DOACs preferred (apixaban, rivaroxaban). Calculate CHA2DS2-VASc. Dose adjust for renal impairment. Monitor bleeding signs.",
}

MEDICATION_DATABASE = {
    "metformin": {"class": "Biguanide", "dose": "500-2000mg daily", "contraindications": "eGFR<30, liver disease", "interactions": "IV contrast (hold 48h)", "monitoring": "HbA1c, B12, renal function"},
    "lisinopril": {"class": "ACE Inhibitor", "dose": "10-40mg daily", "contraindications": "Pregnancy, angioedema hx, bilateral RAS", "interactions": "K+ supplements, NSAIDs, lithium", "monitoring": "BP, K+, Cr at 1-2 weeks"},
    "amlodipine": {"class": "CCB", "dose": "2.5-10mg daily", "contraindications": "Severe aortic stenosis", "interactions": "Simvastatin (limit dose)", "monitoring": "BP, heart rate, edema"},
    "apixaban": {"class": "DOAC", "dose": "5mg BID (2.5mg BID if criteria met)", "contraindications": "Active bleeding, mechanical valve, severe liver disease", "interactions": "Strong CYP3A4 inhibitors, aspirin", "monitoring": "Renal function, signs of bleeding"},
    "sertraline": {"class": "SSRI", "dose": "50-200mg daily", "contraindications": "MAOi within 14 days", "interactions": "Tramadol (serotonin syndrome), NSAIDs (GI bleeding)", "monitoring": "Mood, suicidal ideation, serotonin symptoms"},
    "carvedilol": {"class": "Beta-blocker", "dose": "3.125-25mg BID", "contraindications": "Severe bradycardia, decompensated HF, severe asthma", "interactions": "Verapamil, digoxin", "monitoring": "HR, BP, weight, symptoms"},
}


@tool
def search_guidelines(condition: str) -> str:
    """Search clinical guidelines for a medical condition. Available: hypertension, diabetes_type2, heart_failure, depression, ckd, anticoagulation."""
    guideline = CLINICAL_GUIDELINES.get(condition.lower().replace(" ", "_"))
    if guideline:
        return f"CLINICAL GUIDELINE for {condition}:\n{guideline}"
    # Try partial match
    for key, value in CLINICAL_GUIDELINES.items():
        if condition.lower() in key:
            return f"CLINICAL GUIDELINE for {key}:\n{value}"
    return f"No guideline found for '{condition}'. Available: {', '.join(CLINICAL_GUIDELINES.keys())}"


@tool
def lookup_medication(medication: str) -> str:
    """Look up medication information including class, dosing, contraindications, interactions, and monitoring. Available: metformin, lisinopril, amlodipine, apixaban, sertraline, carvedilol."""
    med = MEDICATION_DATABASE.get(medication.lower())
    if med:
        return json.dumps({"medication": medication, **med}, indent=2)
    return f"Medication '{medication}' not found. Available: {', '.join(MEDICATION_DATABASE.keys())}"


@tool
def check_interaction(drug1: str, drug2: str) -> str:
    """Check for interactions between two drugs."""
    interactions = {
        frozenset({"lisinopril", "nsaids"}): "MODERATE: NSAIDs reduce ACEi effectiveness, increase renal risk",
        frozenset({"lisinopril", "potassium"}): "MAJOR: Hyperkalemia risk - monitor K+ closely",
        frozenset({"sertraline", "tramadol"}): "MAJOR: Serotonin syndrome risk - avoid combination",
        frozenset({"apixaban", "aspirin"}): "MAJOR: Increased bleeding risk",
        frozenset({"metformin", "contrast"}): "MODERATE: Hold metformin 48h around IV contrast",
        frozenset({"amlodipine", "simvastatin"}): "MODERATE: Limit simvastatin to 20mg with amlodipine",
        frozenset({"carvedilol", "verapamil"}): "MAJOR: Severe bradycardia risk - avoid combination",
    }
    key = frozenset({drug1.lower(), drug2.lower()})
    result = interactions.get(key)
    if result:
        return f"INTERACTION: {drug1} + {drug2}: {result}"
    return f"No significant interaction found between {drug1} and {drug2} in our database."


@tool
def interpret_lab(test: str, value: float) -> str:
    """Interpret a lab value. Available: hba1c, gfr, potassium, creatinine, systolic_bp, hemoglobin, inr."""
    tests = {
        "hba1c": lambda v: f"HbA1c {v}%: {'Normal (<5.7)' if v < 5.7 else 'Prediabetes (5.7-6.4)' if v < 6.5 else 'Diabetes (≥6.5). Target <7% for most diabetics.'}",
        "gfr": lambda v: f"GFR {v}: {'Normal (≥90)' if v >= 90 else 'Mild CKD (60-89)' if v >= 60 else 'Stage 3 CKD (30-59)' if v >= 30 else 'Stage 4 CKD (15-29) - refer nephrology' if v >= 15 else 'Stage 5 CKD (<15) - dialysis evaluation'}",
        "potassium": lambda v: f"K+ {v}: {'HYPOKALEMIA (<3.5) - arrhythmia risk' if v < 3.5 else 'Normal (3.5-5.0)' if v <= 5.0 else 'HYPERKALEMIA (>5.0) - arrhythmia risk'}",
        "creatinine": lambda v: f"Cr {v}: {'Normal (0.7-1.3)' if 0.7 <= v <= 1.3 else 'Abnormal - check GFR'}",
        "systolic_bp": lambda v: f"SBP {v}: {'Normal (<120)' if v < 120 else 'Elevated (120-129)' if v < 130 else 'Stage 1 HTN (130-139)' if v < 140 else 'Stage 2 HTN (≥140)'}",
        "hemoglobin": lambda v: f"Hb {v}: {'Anemia (<12)' if v < 12 else 'Normal (12-17)' if v <= 17 else 'Elevated (>17)'}",
        "inr": lambda v: f"INR {v}: {'Sub-therapeutic (<2)' if v < 2 else 'Therapeutic (2-3)' if v <= 3 else 'Supra-therapeutic (>3) - bleeding risk'}",
    }
    fn = tests.get(test.lower())
    if fn:
        return fn(value)
    return f"Test '{test}' not available. Options: {', '.join(tests.keys())}"


@tool
def safety_check(patient_age: int, medications: str, conditions: str) -> str:
    """Run a safety check for a patient given their age, current medications, and conditions. Medications and conditions should be comma-separated."""
    warnings = []
    meds = [m.strip().lower() for m in medications.split(",")]
    conds = [c.strip().lower() for c in conditions.split(",")]

    # Age-based checks
    if patient_age >= 65:
        if any("nsaid" in m for m in meds):
            warnings.append("⚠️ NSAID use in elderly (≥65) - increased GI bleeding, renal, CV risk")
        if len(meds) >= 5:
            warnings.append("⚠️ Polypharmacy (5+ meds) in elderly - review for deprescribing opportunities")

    # Condition-medication checks
    if "ckd" in conds or "kidney" in " ".join(conds):
        if "metformin" in meds:
            warnings.append("⚠️ Metformin with CKD - verify eGFR ≥30, adjust dose if 30-45")
        if any("nsaid" in m for m in meds):
            warnings.append("🚨 NSAIDs with CKD - AVOID, nephrotoxic")

    if "pregnancy" in conds or "pregnant" in " ".join(conds):
        if any(m in meds for m in ["lisinopril", "enalapril", "losartan", "valsartan"]):
            warnings.append("🚨 ACEi/ARB in pregnancy - CONTRAINDICATED, teratogenic")

    if any("depression" in c for c in conds):
        if "tramadol" in meds and "sertraline" in meds:
            warnings.append("🚨 Sertraline + Tramadol - Serotonin syndrome risk")

    if not warnings:
        return "✅ No safety concerns identified based on provided information."
    return "SAFETY ALERTS:\n" + "\n".join(warnings)


# All tools
clinical_tools = [search_guidelines, lookup_medication, check_interaction, interpret_lab, safety_check]


# ============================================================
# DEMO 1: Clinical Decision Support Agent
# ============================================================

def demo_cds_agent():
    """Full clinical decision support agent"""
    print("\n" + "=" * 70)
    print("DEMO 1: CLINICAL DECISION SUPPORT AGENT")
    print("=" * 70)
    print("""
💡 An agent that combines:
   • Clinical guideline search (RAG-like)
   • Medication database lookup
   • Drug interaction checking
   • Lab value interpretation
   • Safety checking
""")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Clinical Decision Support Agent for healthcare professionals.

You have access to clinical guidelines, medication databases, lab interpretation tools,
drug interaction checkers, and safety validation tools.

When given a clinical question:
1. Search relevant guidelines
2. Look up specific medications if needed
3. Check interactions between drugs
4. Interpret lab values
5. Run safety checks

Always explain your clinical reasoning step by step.
Cite which tools/guidelines you used.
Flag any safety concerns prominently.

DISCLAIMER: For educational purposes only. Not a substitute for clinical judgment."""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, clinical_tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=clinical_tools, verbose=True, max_iterations=8
    )

    cases = [
        "72-year-old with newly diagnosed AFib (CHA2DS2-VASc 4), CKD stage 3 (GFR 42), on aspirin. "
        "What anticoagulation should be started? Any safety concerns?",

        "55-year-old with Type 2 diabetes (HbA1c 8.2%), HTN (BP 148/92), and BMI 34. "
        "Currently on metformin 1000mg BID only. What should be added?",
    ]

    for case in cases:
        print(f"\n{'═' * 70}")
        print(f"📝 CASE: {case}\n")
        result = agent_executor.invoke({"input": case})
        print(f"\n📋 RECOMMENDATION:\n{result['output']}")


# ============================================================
# DEMO 2: Guardrailed Agent (Safety-First Design)
# ============================================================

class GuardedState(TypedDict):
    patient_case: str
    clinical_analysis: str
    recommendation: str
    safety_review: str
    final_output: str
    has_safety_concern: bool


def analyze_case(state: GuardedState) -> dict:
    """Step 1: Clinical analysis"""
    response = llm.invoke(
        f"Analyze this case clinically. Identify key problems, relevant history, "
        f"and factors to consider. Be thorough:\n\n{state['patient_case']}"
    )
    return {"clinical_analysis": response.content}


def generate_recommendation(state: GuardedState) -> dict:
    """Step 2: Generate recommendation"""
    response = llm.invoke(
        f"Based on this analysis, provide specific clinical recommendations "
        f"including medications (with doses), monitoring plan, and follow-up:\n\n"
        f"Case: {state['patient_case']}\n"
        f"Analysis: {state['clinical_analysis']}\n\n"
        f"Be specific with medication names and doses. Educational purposes only."
    )
    return {"recommendation": response.content}


def safety_review_node(state: GuardedState) -> dict:
    """Step 3: Safety guardrail - review for dangerous recommendations"""
    response = llm.invoke(
        f"""You are a PATIENT SAFETY OFFICER. Review this recommendation for safety issues.

CASE: {state['patient_case']}
RECOMMENDATION: {state['recommendation']}

Check for:
1. Drug interactions (dangerous combinations)
2. Contraindications (meds that shouldn't be given)
3. Missing safety steps (labs, monitoring not ordered)
4. Dose errors (too high, too low for patient)
5. Age/weight/renal adjustments not made

If ANY safety concern found, start with "⚠️ SAFETY CONCERN:" and explain.
If safe, start with "✅ SAFETY CLEARED:" and confirm."""
    )

    has_concern = "safety concern" in response.content.lower() or "⚠️" in response.content
    return {"safety_review": response.content, "has_safety_concern": has_concern}


def format_safe_output(state: GuardedState) -> dict:
    """Step 4a: Format output (safe)"""
    output = f"""
{'═' * 60}
CLINICAL DECISION SUPPORT REPORT
{'═' * 60}

📝 CASE: {state['patient_case'][:200]}...

🔬 ANALYSIS:
{state['clinical_analysis']}

💊 RECOMMENDATION:
{state['recommendation']}

🛡️ SAFETY REVIEW:
{state['safety_review']}

⚕️ For educational purposes only.
{'═' * 60}"""
    return {"final_output": output}


def format_flagged_output(state: GuardedState) -> dict:
    """Step 4b: Format output (flagged for review)"""
    output = f"""
{'═' * 60}
🚨 CLINICAL REPORT — FLAGGED FOR REVIEW
{'═' * 60}

📝 CASE: {state['patient_case'][:200]}...

💊 INITIAL RECOMMENDATION:
{state['recommendation']}

🛡️ ⚠️  SAFETY CONCERNS IDENTIFIED:
{state['safety_review']}

⚠️ THIS RECOMMENDATION REQUIRES CLINICIAN REVIEW
   before implementation due to identified safety concerns.

⚕️ For educational purposes only.
{'═' * 60}"""
    return {"final_output": output}


def route_by_safety(state: GuardedState) -> str:
    """Route based on safety review"""
    return "flagged" if state.get("has_safety_concern") else "safe"


def demo_guardrailed_agent():
    """Agent with safety guardrails"""
    print("\n" + "=" * 70)
    print("DEMO 2: GUARDRAILED AGENT (Safety-First Design)")
    print("=" * 70)
    print("""
💡 Every recommendation passes through a SAFETY REVIEW node.
   If safety concerns are found → report is FLAGGED for clinician review.
   This is how production healthcare AI systems should work.
""")

    graph = StateGraph(GuardedState)
    graph.add_node("analyze", analyze_case)
    graph.add_node("recommend", generate_recommendation)
    graph.add_node("safety_review", safety_review_node)
    graph.add_node("format_safe", format_safe_output)
    graph.add_node("format_flagged", format_flagged_output)

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "recommend")
    graph.add_edge("recommend", "safety_review")
    graph.add_conditional_edges("safety_review", route_by_safety, {
        "safe": "format_safe",
        "flagged": "format_flagged",
    })
    graph.add_edge("format_safe", END)
    graph.add_edge("format_flagged", END)

    app = graph.compile()

    cases = [
        "55-year-old male with hypertension and diabetes. Currently well-controlled on metformin and lisinopril. Routine follow-up, no complaints.",
        "28-year-old pregnant woman with blood pressure 152/96 on two readings. History of previous preeclampsia. Currently on lisinopril 10mg daily from before pregnancy.",
    ]

    for case in cases:
        print(f"\n{'─' * 70}")
        print(f"📝 Case: {case[:120]}...\n")
        result = app.invoke({"patient_case": case})
        print(result["final_output"])


# ============================================================
# DEMO 3: Interactive Clinical Assistant
# ============================================================

def demo_interactive():
    """Interactive clinical decision support"""
    print("\n" + "=" * 70)
    print("DEMO 3: INTERACTIVE CLINICAL ASSISTANT")
    print("=" * 70)
    print("\n💬 Ask clinical questions! This agent has memory + tools.")
    print("   Tools: guidelines, medications, interactions, labs, safety check")
    print("   Type 'quit' to exit\n")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Clinical Decision Support assistant with access to clinical guidelines,
medication databases, interaction checkers, lab interpretation, and safety tools.
Use your tools when needed for specific clinical data. Explain your reasoning.
Remember conversation history for follow-up questions. Educational purposes only."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, clinical_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=clinical_tools, verbose=False, max_iterations=6)
    history = []

    while True:
        question = input("Clinician: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue

        result = executor.invoke({"input": question, "chat_history": history})
        print(f"\nCDS Agent: {result['output']}\n")

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=result["output"]))


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n🏥 Level 3, Project 5: Healthcare Agent Capstone")
    print("=" * 70)
    print("Complete clinical decision support — everything combined!\n")

    print("Choose a demo:")
    print("1. Clinical Decision Support Agent (tools + reasoning)")
    print("2. Guardrailed Agent (safety-first design)")
    print("3. Interactive Clinical Assistant (chat + memory)")
    print("4. Run demos 1-2, then interactive")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        demo_cds_agent()
    elif choice == "2":
        demo_guardrailed_agent()
    elif choice == "3":
        demo_interactive()
    elif choice == "4":
        demo_cds_agent()
        demo_guardrailed_agent()
        demo_interactive()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
🎉 LEVEL 3 COMPLETE — CONGRATULATIONS!
{'=' * 70}

You've mastered AI Agents — the most exciting area of GenAI!

📊 WHAT YOU'VE BUILT IN LEVEL 3:
   01. ReAct agent from scratch (no frameworks)
   02. LangChain agent with custom tools & memory
   03. LangGraph stateful workflows with conditional routing
   04. Multi-agent systems (pipeline, router, debate)
   05. Clinical decision support with guardrails (this project!)

🏥 YOUR HEALTHCARE AI CAPABILITIES:
   • Clinical guideline Q&A (RAG)
   • Medication lookup and interaction checking (Tools)
   • Lab value interpretation (Tools)
   • Multi-step clinical reasoning (Agent loops)
   • Multi-agent clinical workflows (LangGraph)
   • Safety guardrails (production pattern)

🔑 THE COMPLETE PICTURE (Levels 1→3):
   Level 1: LLM APIs, embeddings, function calling
   Level 2: RAG — connect LLMs to YOUR documents
   Level 3: Agents — autonomous reasoning + tools + workflows

🎯 WHAT'S NEXT:
   Level 4: Fine-Tuning — customize models for YOUR specific tasks
   Level 5: Production — deploy, scale, monitor, secure

   Or start building REAL applications with what you've learned! 🚀
""")


if __name__ == "__main__":
    main()
