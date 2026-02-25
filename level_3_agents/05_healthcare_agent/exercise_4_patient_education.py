"""
Exercise 4: Patient Education Agent — Explains Diagnoses in Simple Language

Skills practiced:
- Building an agent that translates medical jargon to plain language
- Reading level adaptation (6th grade for patients)
- Structured patient education materials
- Health literacy principles for clinical AI

Key insight: A clinical recommendation is useless if the patient
  doesn't understand it. This exercise builds a patient education
  agent that takes clinical output and produces clear, simple
  explanations that a patient can actually follow. Studies show
  that health literacy directly impacts outcomes.
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
# Medical Terms Dictionary (for the simplify tool)
# ============================================================

MEDICAL_TERMS = {
    "hypertension": "high blood pressure",
    "hyperlipidemia": "high cholesterol",
    "diabetes mellitus": "diabetes (high blood sugar)",
    "type 2 diabetes": "diabetes — your body has trouble using sugar from food for energy",
    "chronic kidney disease": "your kidneys are not working as well as they should",
    "ckd": "kidney disease (your kidneys have trouble filtering your blood)",
    "heart failure": "your heart is weaker than normal and can't pump blood as well",
    "atrial fibrillation": "an irregular heartbeat (your heart beats in an uneven rhythm)",
    "edema": "swelling (usually in your legs or feet)",
    "dyspnea": "feeling short of breath",
    "orthopnea": "trouble breathing when lying flat",
    "hba1c": "a blood test that shows your average blood sugar over 3 months",
    "gfr": "a number that shows how well your kidneys are working (higher is better)",
    "creatinine": "a blood test that checks your kidney function",
    "potassium": "a mineral in your blood — too much or too little can affect your heart",
    "inr": "a blood test that shows how well your blood thinner is working",
    "anticoagulation": "blood thinning treatment (to prevent blood clots)",
    "ace inhibitor": "a blood pressure medicine that also helps protect your kidneys and heart",
    "beta blocker": "a medicine that slows your heart rate and lowers blood pressure",
    "sglt2 inhibitor": "a newer diabetes medicine that also helps your heart and kidneys",
    "diuretic": "a 'water pill' that helps your body get rid of extra fluid",
    "statin": "a cholesterol-lowering medicine",
    "metformin": "a diabetes medicine that helps your body use sugar better",
    "contraindicated": "should NOT be used (it could be dangerous)",
    "prognosis": "what we expect will happen with your health going forward",
    "differential diagnosis": "the list of conditions your doctor is considering",
    "titrate": "slowly adjust the dose of your medicine",
    "renal": "related to your kidneys",
    "hepatic": "related to your liver",
    "cardiac": "related to your heart",
    "pulmonary": "related to your lungs",
}


# ============================================================
# Patient Education Tools
# ============================================================

@tool
def simplify_medical_term(term: str) -> str:
    """Translate a medical term to simple plain language. Use this when you encounter medical jargon that a patient might not understand."""
    simple = MEDICAL_TERMS.get(term.lower())
    if simple:
        return f"'{term}' means: {simple}"
    # Try partial match
    for key, value in MEDICAL_TERMS.items():
        if term.lower() in key or key in term.lower():
            return f"'{term}' relates to: {value}"
    return f"'{term}' — I don't have a simplified version. Explain in simple words."


@tool
def create_medication_card(medication: str, dose: str, purpose: str) -> str:
    """Create a patient-friendly medication card. Include medication name, dose, and purpose."""
    return f"""
    ┌─────────────────────────────────────────┐
    │  💊 MEDICATION CARD                      │
    │                                          │
    │  Medicine: {medication:<28s} │
    │  How much:  {dose:<27s} │
    │  Why:       {purpose:<27s} │
    │                                          │
    │  ⚠️  Take as directed by your doctor     │
    │  📞 Call your doctor if you have         │
    │     questions or side effects            │
    └─────────────────────────────────────────┘"""


@tool
def create_warning_signs_card(condition: str) -> str:
    """Create a simple 'when to call your doctor' card for a condition. Available: heart_failure, diabetes, hypertension, ckd, anticoagulation."""
    warnings = {
        "heart_failure": [
            "You gain 3+ pounds in one day or 5+ pounds in one week",
            "Your legs or feet swell more than usual",
            "You feel more short of breath than normal",
            "You can't lie flat to sleep",
            "You feel dizzy or lightheaded",
        ],
        "diabetes": [
            "Your blood sugar is over 300 or under 70",
            "You feel very thirsty and urinate a lot",
            "You feel confused or very tired",
            "You have fruity-smelling breath",
            "You have a wound that won't heal",
        ],
        "hypertension": [
            "Your blood pressure is over 180/120",
            "You have a severe headache that won't go away",
            "You have chest pain or trouble breathing",
            "You feel numbness or weakness on one side",
            "You have sudden vision changes",
        ],
        "ckd": [
            "Your urine looks foamy or bloody",
            "You're swelling in your face, hands, or feet",
            "You're much more tired than usual",
            "You feel nauseous or can't eat",
            "You're urinating much less than normal",
        ],
        "anticoagulation": [
            "You see blood in your urine or stool",
            "You have nosebleeds that won't stop",
            "You bruise very easily",
            "You cough up blood",
            "You have a bad headache or feel confused",
        ],
    }
    signs = warnings.get(condition.lower().replace(" ", "_"))
    if not signs:
        return f"No warning signs card for '{condition}'. Available: {', '.join(warnings.keys())}"
    lines = [f"🚨 CALL YOUR DOCTOR RIGHT AWAY IF:"]
    for s in signs:
        lines.append(f"  • {s}")
    lines.append("\n  📞 If it feels like an emergency, call 911")
    return "\n".join(lines)


education_tools = [simplify_medical_term, create_medication_card, create_warning_signs_card]


# ============================================================
# Patient Education State
# ============================================================

class EducationState(TypedDict):
    clinical_summary: str       # The clinical content to translate
    patient_name: str
    reading_level: str          # "simple" or "moderate"
    diagnosis_explanation: str
    medication_guide: str
    lifestyle_advice: str
    warning_signs: str
    full_handout: str


# ============================================================
# Pipeline Nodes
# ============================================================

def explain_diagnosis(state: EducationState) -> dict:
    """Explain the diagnosis in simple language"""
    response = llm.invoke(
        f"""You are a patient educator. Explain this clinical information to a patient.
Use simple words (6th-grade reading level). No medical jargon.

Clinical summary: {state['clinical_summary']}

Write a short explanation that answers:
1. What is wrong? (explain the condition in simple terms)
2. Why does it matter? (what could happen if untreated)
3. What are we going to do about it? (brief overview of the plan)

Keep it to 4-5 short paragraphs. Use "you" and "your".
Avoid words like "etiology", "pathophysiology", "prognosis"."""
    )
    return {"diagnosis_explanation": response.content}


def create_med_guide(state: EducationState) -> dict:
    """Create a simple medication guide"""
    response = llm.invoke(
        f"""Create a simple medication guide for this patient.

Clinical summary: {state['clinical_summary']}

For EACH medication, explain in plain language:
• What it's called
• What it does (in simple terms)
• How to take it (when, with food?, etc.)
• Common side effects to watch for
• What to do if you miss a dose

Format as a numbered list. Use simple words.
Patient name: {state.get('patient_name', 'Patient')}"""
    )
    return {"medication_guide": response.content}


def create_lifestyle_section(state: EducationState) -> dict:
    """Create simple lifestyle advice"""
    response = llm.invoke(
        f"""Create simple lifestyle advice for this patient.

Diagnosis explanation: {state['diagnosis_explanation'][:200]}

Provide practical tips for:
1. FOOD: What to eat more of, what to eat less of
2. ACTIVITY: What exercise is safe and helpful
3. DAILY HABITS: Things to do every day for your health
4. THINGS TO AVOID: What makes it worse

Keep each tip to 1-2 sentences. Use everyday language.
Start tips with action words: "Try to...", "Eat more...", "Avoid..."."""
    )
    return {"lifestyle_advice": response.content}


def create_warnings(state: EducationState) -> dict:
    """Create warning signs section"""
    response = llm.invoke(
        f"""Based on this clinical case, create a "When to Call Your Doctor" section.

Clinical summary: {state['clinical_summary'][:200]}

List 5-7 specific warning signs in simple language.
Format as:
🚨 CALL YOUR DOCTOR or go to the ER if:
• [warning sign in simple words]

End with: "If it feels like an emergency, call 911."
Use simple, specific language a patient can understand."""
    )
    return {"warning_signs": response.content}


def assemble_handout(state: EducationState) -> dict:
    """Assemble the complete patient education handout"""
    patient = state.get("patient_name", "Patient")
    handout = f"""
{'═' * 60}
  YOUR HEALTH INFORMATION
  For: {patient}
{'═' * 60}

📋 WHAT'S GOING ON WITH YOUR HEALTH
{'─' * 40}
{state['diagnosis_explanation']}

💊 YOUR MEDICATIONS
{'─' * 40}
{state['medication_guide']}

🥗 HEALTHY LIVING TIPS
{'─' * 40}
{state['lifestyle_advice']}

⚠️ IMPORTANT WARNING SIGNS
{'─' * 40}
{state['warning_signs']}

{'─' * 40}
📞 Questions? Call your doctor's office.
🏥 Emergency? Call 911.
{'═' * 60}
"""
    return {"full_handout": handout}


# ============================================================
# Build Pipeline
# ============================================================

def build_education_pipeline():
    graph = StateGraph(EducationState)

    graph.add_node("explain_diagnosis", explain_diagnosis)
    graph.add_node("med_guide", create_med_guide)
    graph.add_node("lifestyle", create_lifestyle_section)
    graph.add_node("warnings", create_warnings)
    graph.add_node("assemble", assemble_handout)

    graph.set_entry_point("explain_diagnosis")
    graph.add_edge("explain_diagnosis", "med_guide")
    graph.add_edge("med_guide", "lifestyle")
    graph.add_edge("lifestyle", "warnings")
    graph.add_edge("warnings", "assemble")
    graph.add_edge("assemble", END)

    return graph.compile()


# ============================================================
# DEMO 1: Patient Education Handout
# ============================================================

def demo_patient_handout():
    """Generate a complete patient education handout"""
    print("\n" + "=" * 70)
    print("DEMO 1: PATIENT EDUCATION HANDOUT")
    print("=" * 70)
    print("""
    Takes clinical information and produces a patient-friendly handout:
      Diagnosis → Medications → Lifestyle → Warning Signs
    """)

    app = build_education_pipeline()

    result = app.invoke({
        "clinical_summary": "72-year-old male with newly diagnosed heart failure (EF 30%), "
                            "Type 2 diabetes (HbA1c 8.2%), and hypertension. "
                            "Started on carvedilol 3.125mg BID, lisinopril 10mg daily, "
                            "metformin 500mg BID. Needs sodium restriction <2g, "
                            "fluid restriction 2L, daily weights. Follow-up in 2 weeks.",
        "patient_name": "Mr. Johnson",
        "reading_level": "simple",
    })

    print(result.get("full_handout", "No handout generated"))


# ============================================================
# DEMO 2: Medical Jargon Translation
# ============================================================

def demo_jargon_translation():
    """Show how the agent translates medical terms"""
    print("\n" + "=" * 70)
    print("DEMO 2: MEDICAL JARGON TRANSLATION")
    print("=" * 70)
    print("""
    The agent has a dictionary of medical terms and can translate
    clinical language to plain English.
    """)

    # Direct tool use
    clinical_terms = [
        "hypertension", "atrial fibrillation", "hba1c",
        "edema", "dyspnea", "contraindicated",
        "ace inhibitor", "sglt2 inhibitor", "titrate",
    ]

    print(f"\n  MEDICAL TERM → PLAIN ENGLISH:")
    print(f"  {'─' * 50}")
    for term in clinical_terms:
        simple = MEDICAL_TERMS.get(term.lower(), "?")
        print(f"  {term:25s} → {simple}")

    # Agent-based translation
    print(f"\n\n  AGENT TRANSLATION (full clinical note):")
    print(f"  {'─' * 50}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a health educator. Translate medical notes into simple "
                   "language a patient can understand. Use your simplify_medical_term "
                   "tool for medical jargon. Write at a 6th-grade reading level."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, education_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=education_tools, verbose=False, max_iterations=6)

    clinical_note = ("Patient presents with decompensated heart failure (EF 25%), "
                     "dyspnea on exertion and orthopnea. CKD Stage 3 with GFR 42. "
                     "Initiated diuretic therapy. Titrate beta-blocker as tolerated. "
                     "Contraindicated for NSAIDs due to renal impairment.")

    result = executor.invoke({"input": f"Translate this for the patient:\n{clinical_note}"})
    print(f"\n  CLINICAL NOTE:\n  {clinical_note}")
    print(f"\n  PATIENT VERSION:\n  {result['output']}")


# ============================================================
# DEMO 3: Compare Clinical vs Patient Language
# ============================================================

def demo_compare_language():
    """Side-by-side comparison of clinical vs patient language"""
    print("\n" + "=" * 70)
    print("DEMO 3: CLINICAL vs PATIENT LANGUAGE")
    print("=" * 70)

    app = build_education_pipeline()

    cases = [
        {
            "label": "Diabetes Management",
            "clinical": "T2DM with HbA1c 8.5%. Initiate metformin 500mg BID, titrate to 2000mg. "
                        "Add GLP-1 RA given BMI 35 and CVD risk. SGLT2i if eGFR allows. "
                        "Monitor HbA1c q3mo, B12 annually.",
            "patient_name": "Mrs. Garcia",
        },
        {
            "label": "Blood Thinner Start",
            "clinical": "New AFib, CHA2DS2-VASc 4. Initiate anticoagulation with apixaban 5mg BID. "
                        "No dose adjustment needed (Cr 1.1, weight 82kg, age 71). "
                        "Hold aspirin. Monitor for signs of bleeding.",
            "patient_name": "Mr. Williams",
        },
    ]

    for case in cases:
        print(f"\n{'═' * 60}")
        print(f"  {case['label'].upper()}")
        print(f"{'═' * 60}")

        print(f"\n  CLINICAL VERSION:")
        print(f"  {case['clinical']}")

        result = app.invoke({
            "clinical_summary": case["clinical"],
            "patient_name": case["patient_name"],
            "reading_level": "simple",
        })

        print(f"\n  PATIENT VERSION (for {case['patient_name']}):")
        print(f"  {result.get('diagnosis_explanation', 'N/A')[:400]}")
        print(f"\n  MEDICATION GUIDE:")
        print(f"  {result.get('medication_guide', 'N/A')[:300]}")


# ============================================================
# DEMO 4: Interactive Patient Educator
# ============================================================

def demo_interactive():
    """Interactive patient education agent"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE PATIENT EDUCATOR")
    print("=" * 70)
    print("  Enter a clinical summary. The agent will create patient materials.")
    print("  Type 'quit' to exit.\n")

    app = build_education_pipeline()

    while True:
        summary = input("  Clinical summary: ").strip()
        if summary.lower() in ['quit', 'exit', 'q']:
            break
        if not summary:
            continue

        name = input("  Patient name (or press Enter): ").strip() or "Patient"

        print("\n  Generating patient education materials...\n")
        result = app.invoke({
            "clinical_summary": summary,
            "patient_name": name,
            "reading_level": "simple",
        })

        print(result.get("full_handout", "Error generating handout"))
        print()


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 4: PATIENT EDUCATION AGENT")
    print("=" * 70)
    print("""
    Translates clinical recommendations into plain-language
    patient education materials: diagnosis explanation, medication
    guide, lifestyle tips, and warning signs.

    Choose a demo:
      1 → Patient education handout
      2 → Medical jargon translation
      3 → Clinical vs patient language comparison
      4 → Interactive patient educator
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1": demo_patient_handout()
    elif choice == "2": demo_jargon_translation()
    elif choice == "3": demo_compare_language()
    elif choice == "4": demo_interactive()
    elif choice == "5":
        demo_patient_handout()
        demo_jargon_translation()
        demo_compare_language()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. HEALTH LITERACY IS CRITICAL: 36% of US adults have basic or
   below-basic health literacy. Clinical AI that only speaks
   "doctor language" fails patients. Translating to plain language
   directly improves health outcomes.

2. READING LEVEL: Patient materials should target 6th-grade level.
   Avoid jargon. Use "you" and "your". Short sentences. Concrete
   examples. The agent can be prompted to write at specific levels.

3. STRUCTURED EDUCATION: A good patient handout has:
   - What's wrong (simple explanation)
   - What to do (medications + lifestyle)
   - What to watch for (warning signs)
   - Who to call (emergency contacts)
   This structure is the same whether AI or a nurse creates it.

4. TERM TRANSLATION: A dictionary of medical→simple translations
   helps the agent consistently use plain language. In production,
   this would be a comprehensive medical terminology service.

5. PRODUCTION APPLICATIONS:
   - Auto-generate discharge instructions from clinical notes
   - Multi-language patient education (translate to Spanish, etc.)
   - Adapt to different literacy levels per patient
   - Include visual aids and diagrams
   - Integrate with patient portal for at-home access
"""

if __name__ == "__main__":
    main()
