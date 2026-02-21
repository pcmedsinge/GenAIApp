"""
Exercise 4: Prompt Template Library
Build a library of reusable healthcare prompt templates with variable substitution.

Skills practiced:
- Designing reusable prompt templates
- Variable substitution for dynamic prompts
- Building a template registry/catalog
- Template categories for different clinical workflows

Healthcare context:
  In production healthcare AI, you don't write prompts from scratch each time.
  You maintain a LIBRARY of tested, validated templates that your application
  fills in with dynamic data (patient info, lab values, etc.)
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat(messages, temperature=0.3, max_tokens=1000):
    """Helper"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


# ============================================================
# The Template Library
# ============================================================

TEMPLATE_LIBRARY = {

    # ---- CLINICAL DOCUMENTATION ----
    "medication_review": {
        "category": "Clinical Documentation",
        "description": "Review a patient's medication list for interactions, appropriateness, and gaps",
        "system_prompt": "You are a clinical pharmacist performing a comprehensive medication review.",
        "template": """Perform a medication review for this patient:

Patient: {patient_name}, Age: {age}, Sex: {sex}
Conditions: {conditions}
Current Medications: {medications}
Recent Labs: {labs}
Allergies: {allergies}

Review for:
1. DRUG-DRUG INTERACTIONS (severity: major/moderate/minor)
2. DRUG-DISEASE CONTRAINDICATIONS
3. DOSE APPROPRIATENESS (for age, weight, renal function)
4. DUPLICATE THERAPY
5. MISSING MEDICATIONS (per current guidelines for listed conditions)
6. ALLERGY CROSS-REACTIVITY

Format each finding as:
⚠️ [SEVERITY] Finding — Recommendation

End with a summary recommendation.""",
        "variables": ["patient_name", "age", "sex", "conditions", "medications", "labs", "allergies"],
        "temperature": 0.2,
    },

    "discharge_instructions": {
        "category": "Clinical Documentation",
        "description": "Generate patient-friendly discharge instructions",
        "system_prompt": "You are a nurse educator creating discharge instructions that patients can easily understand.",
        "template": """Create discharge instructions for this patient:

Patient: {patient_name}
Condition: {diagnosis}
Procedures done: {procedures}
New Medications: {new_medications}
Restrictions: {restrictions}
Follow-up: {follow_up}
Reading Level: {reading_level}

Instructions must include:
1. WHAT HAPPENED (simple explanation of condition/procedure)
2. MEDICATIONS (name, what it's for, how to take, common side effects)
3. ACTIVITY RESTRICTIONS (specific, time-limited)
4. WARNING SIGNS (when to call doctor vs go to ER — be specific)
5. FOLLOW-UP APPOINTMENTS
6. CONTACT INFORMATION placeholder

Use {reading_level} reading level. Short sentences. Bullet points.
No medical jargon — explain any medical terms in parentheses.""",
        "variables": ["patient_name", "diagnosis", "procedures", "new_medications", "restrictions", "follow_up", "reading_level"],
        "temperature": 0.4,
    },

    # ---- CLINICAL DECISION SUPPORT ----
    "drug_interaction_check": {
        "category": "Clinical Decision Support",
        "description": "Check medications for interactions with clinical context",
        "system_prompt": "You are a clinical pharmacology expert. Be specific about mechanisms and severity.",
        "template": """Check for interactions between these medications:

Medications: {medication_list}
Patient Age: {age}
Renal Function (GFR): {gfr}
Liver Function: {liver_function}
Other relevant factors: {other_factors}

For each interaction found, provide:
- SEVERITY: Major / Moderate / Minor
- MECHANISM: How the interaction occurs
- CLINICAL EFFECT: What could happen to the patient
- MANAGEMENT: How to handle (avoid, monitor, adjust dose, etc.)
- MONITORING: Specific labs or symptoms to watch

If no significant interactions, state that clearly.""",
        "variables": ["medication_list", "age", "gfr", "liver_function", "other_factors"],
        "temperature": 0.1,
    },

    "lab_interpretation": {
        "category": "Clinical Decision Support",
        "description": "Interpret a set of lab values in clinical context",
        "system_prompt": "You are an internal medicine physician interpreting lab results. Be precise with reference ranges.",
        "template": """Interpret these lab results for clinical significance:

Patient: {age} {sex}
Clinical Context: {clinical_context}
Current Medications: {medications}

Lab Results:
{lab_values}

For each abnormal value:
1. Is it HIGH or LOW, and by how much?
2. What are the likely CAUSES given this patient's context?
3. What is the CLINICAL SIGNIFICANCE?
4. What FOLLOW-UP is recommended?

Provide an overall assessment connecting the lab findings together.
Educational purposes only.""",
        "variables": ["age", "sex", "clinical_context", "medications", "lab_values"],
        "temperature": 0.2,
    },

    # ---- PATIENT EDUCATION ----
    "patient_education": {
        "category": "Patient Education",
        "description": "Create condition-specific education material for patients",
        "system_prompt": "You are a patient educator. Create clear, reassuring, actionable content.",
        "template": """Create patient education material:

Condition: {condition}
Target Audience: {audience}
Reading Level: {reading_level}
Language: {language}
Special Considerations: {special_considerations}

Include these sections:
📌 WHAT IS {condition}? (2-3 simple sentences)
🔍 SYMPTOMS TO WATCH FOR (bullet list)
💊 YOUR TREATMENT (what to expect, how long)
🏠 WHAT YOU CAN DO AT HOME
⚠️ WHEN TO CALL YOUR DOCTOR
🚨 WHEN TO GO TO THE ER (emergency signs)
❓ COMMON QUESTIONS

Keep total length under {max_words} words.
Use simple language appropriate for {reading_level} reading level.
Be encouraging and reassuring where appropriate.""",
        "variables": ["condition", "audience", "reading_level", "language", "special_considerations", "max_words"],
        "temperature": 0.5,
    },

    # ---- CLINICAL COMMUNICATION ----
    "referral_letter": {
        "category": "Clinical Communication",
        "description": "Generate a professional referral letter to a specialist",
        "system_prompt": "You are a primary care physician writing a professional referral letter.",
        "template": """Generate a referral letter:

FROM: {referring_provider}, {referring_specialty}
TO: {specialist_name}, {specialist_specialty}
PATIENT: {patient_name}, {age} {sex}
DATE: {date}

REASON FOR REFERRAL: {reason}
CLINICAL SUMMARY: {clinical_summary}
RELEVANT HISTORY: {relevant_history}
CURRENT MEDICATIONS: {medications}
RELEVANT LABS/IMAGING: {results}
SPECIFIC QUESTIONS: {questions}
URGENCY: {urgency}

Use professional medical language. Be concise but thorough.
Include all relevant clinical information the specialist needs.""",
        "variables": ["referring_provider", "referring_specialty", "specialist_name", "specialist_specialty",
                      "patient_name", "age", "sex", "date", "reason", "clinical_summary",
                      "relevant_history", "medications", "results", "questions", "urgency"],
        "temperature": 0.3,
    },
}


# ============================================================
# Template Engine
# ============================================================

def fill_template(template_name, **variables):
    """Fill a template with variables and run it through the LLM"""
    template = TEMPLATE_LIBRARY.get(template_name)
    if not template:
        return f"Template '{template_name}' not found. Available: {', '.join(TEMPLATE_LIBRARY.keys())}"

    # Check for missing required variables
    missing = [v for v in template["variables"] if v not in variables]
    if missing:
        return f"Missing variables: {', '.join(missing)}"

    # Fill the template
    filled_prompt = template["template"].format(**variables)

    # Run through LLM
    response = chat(
        messages=[
            {"role": "system", "content": template["system_prompt"]},
            {"role": "user", "content": filled_prompt}
        ],
        temperature=template.get("temperature", 0.3)
    )
    return response


def list_templates():
    """Show all available templates"""
    print("\n📚 TEMPLATE LIBRARY")
    print("=" * 70)

    categories = {}
    for name, tmpl in TEMPLATE_LIBRARY.items():
        cat = tmpl["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, tmpl))

    for category, templates in categories.items():
        print(f"\n📁 {category}")
        for name, tmpl in templates:
            print(f"   📄 {name}")
            print(f"      {tmpl['description']}")
            print(f"      Variables: {', '.join(tmpl['variables'])}")


# ============================================================
# Main Demo
# ============================================================

def main():
    print("\n📚 Exercise 4: Prompt Template Library")
    print("=" * 70)
    print("Reusable, tested prompt templates for healthcare workflows\n")

    print("Choose a demo:")
    print("1. List all templates in library")
    print("2. Medication review (fill template)")
    print("3. Discharge instructions (fill template)")
    print("4. Patient education (fill template)")
    print("5. Lab interpretation (fill template)")
    print("6. Run demos 2-5 (showcase all)")

    choice = input("\nEnter choice (1-6): ").strip()

    if choice == "1":
        list_templates()

    elif choice == "2" or choice == "6":
        print(f"\n{'═' * 70}")
        print("💊 MEDICATION REVIEW")
        print(f"{'═' * 70}")
        result = fill_template("medication_review",
            patient_name="Robert Johnson",
            age="72",
            sex="Male",
            conditions="Type 2 Diabetes, Hypertension, Atrial Fibrillation, CKD Stage 3 (GFR 38)",
            medications="Metformin 1000mg BID, Lisinopril 20mg daily, Apixaban 5mg BID, Atorvastatin 40mg qhs, Naproxen 500mg PRN",
            labs="GFR 38, K+ 5.1, HbA1c 7.2%, Cr 1.8, INR not checked (on DOAC)",
            allergies="Penicillin (rash), Sulfa (anaphylaxis)"
        )
        print(result)

    if choice == "3" or choice == "6":
        print(f"\n{'═' * 70}")
        print("📋 DISCHARGE INSTRUCTIONS")
        print(f"{'═' * 70}")
        result = fill_template("discharge_instructions",
            patient_name="Maria Garcia",
            diagnosis="Community-acquired pneumonia (lung infection)",
            procedures="Chest X-ray, IV antibiotics for 2 days",
            new_medications="Levofloxacin 750mg once daily for 5 more days, Acetaminophen 500mg every 6 hours as needed for fever",
            restrictions="No strenuous activity for 1 week, rest when tired",
            follow_up="Primary care in 7 days, repeat chest X-ray in 6 weeks",
            reading_level="6th grade"
        )
        print(result)

    if choice == "4" or choice == "6":
        print(f"\n{'═' * 70}")
        print("📖 PATIENT EDUCATION")
        print(f"{'═' * 70}")
        result = fill_template("patient_education",
            condition="Type 2 Diabetes",
            audience="Newly diagnosed adult patient",
            reading_level="8th grade",
            language="English",
            special_considerations="Patient is overweight, works a desk job, limited cooking skills",
            max_words="300"
        )
        print(result)

    if choice == "5" or choice == "6":
        print(f"\n{'═' * 70}")
        print("🔬 LAB INTERPRETATION")
        print(f"{'═' * 70}")
        result = fill_template("lab_interpretation",
            age="68",
            sex="female",
            clinical_context="Admitted for heart failure exacerbation, 3 days into treatment",
            medications="Furosemide 40mg IV BID, Lisinopril 10mg, Carvedilol 12.5mg BID, Spironolactone 25mg",
            lab_values="""BMP:
- Na: 131 (low)
- K: 5.3 (high)
- Cl: 94
- CO2: 22
- BUN: 42 (high)
- Creatinine: 1.6 (baseline 1.1)
- Glucose: 112
- BNP: 850 (was 2400 on admission)
- Hemoglobin: 10.2
- Magnesium: 1.6 (low)"""
        )
        print(result)

    if choice not in ["1", "2", "3", "4", "5", "6"]:
        print("Invalid choice")

    print(f"""
{'═' * 70}
KEY LEARNINGS:
{'═' * 70}
1. Templates SEPARATE the prompt design from the data
   → Prompt engineer designs template, app fills in variables 
2. Each template has its own TEMPERATURE setting
   → Low (0.1) for clinical facts, higher (0.5) for patient education
3. Templates are TESTABLE — run the same template with different data
4. Variable validation catches input errors early
5. Categories organize templates by clinical workflow

🏥 PRODUCTION PATTERN:
   In real healthcare AI systems:
   • Templates are stored in a database (not hardcoded)
   • Each template has a version number and approval status
   • Templates are reviewed by clinical experts before deployment
   • Usage is logged for auditing and quality improvement
   • A/B testing compares template performance
""")


if __name__ == "__main__":
    main()
