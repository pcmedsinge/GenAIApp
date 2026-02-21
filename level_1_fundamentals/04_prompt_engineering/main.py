"""
Project 4: Prompt Engineering
Objective: Master prompting techniques for accurate, structured LLM outputs
Concepts: Zero-shot, few-shot, chain-of-thought, output formatting, templates

Healthcare Use Case: Clinical note structuring, ICD-10 coding, diagnosis reasoning
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat(messages, model="gpt-4o-mini", temperature=0.3, max_tokens=1000):
    """Helper function to make API calls"""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


# ============================================
# TECHNIQUE 1: Zero-Shot Prompting
# ============================================

def demo_zero_shot():
    """
    Zero-shot: Ask directly, no examples provided
    The model uses its training knowledge only
    """
    print("\n" + "="*70)
    print("TECHNIQUE 1: ZERO-SHOT PROMPTING")
    print("="*70)
    print("→ Ask directly, no examples. Model relies on training knowledge.\n")

    # Simple zero-shot
    print("--- Example 1: Simple medical question ---")
    response = chat([
        {"role": "user", "content": "What ICD-10 code would you assign to acute bronchitis?"}
    ])
    print(f"Response:\n{response}\n")

    # Zero-shot with role
    print("--- Example 2: Zero-shot WITH system role ---")
    response = chat([
        {"role": "system", "content": "You are a certified medical coder with 20 years experience. Always provide the ICD-10 code, description, and confidence level."},
        {"role": "user", "content": "What ICD-10 code would you assign to acute bronchitis?"}
    ])
    print(f"Response:\n{response}\n")

    # Zero-shot with format instruction
    print("--- Example 3: Zero-shot with OUTPUT FORMAT ---")
    response = chat([
        {"role": "system", "content": "You are a medical coder. Respond in JSON format with keys: code, description, confidence, reasoning."},
        {"role": "user", "content": "What ICD-10 code would you assign to acute bronchitis?"}
    ])
    print(f"Response:\n{response}\n")

    print("💡 TAKEAWAY: Zero-shot works for straightforward tasks.")
    print("   Adding a role and format improves output quality significantly!")


# ============================================
# TECHNIQUE 2: Few-Shot Prompting
# ============================================

def demo_few_shot():
    """
    Few-shot: Provide examples to guide the model's behavior
    Much more accurate for specific formats or patterns
    """
    print("\n" + "="*70)
    print("TECHNIQUE 2: FEW-SHOT PROMPTING")
    print("="*70)
    print("→ Provide examples first, then ask. Model follows the pattern.\n")

    # Few-shot ICD-10 coding
    print("--- Example 1: Few-shot ICD-10 coding ---")
    response = chat([
        {"role": "system", "content": "You are a medical coder. Follow the exact format shown in the examples."},
        {"role": "user", "content": """Here are examples of clinical notes and their ICD-10 codes:

Note: "Patient presents with type 2 diabetes mellitus without complications"
Code: E11.9 - Type 2 diabetes mellitus without complications

Note: "Acute upper respiratory infection, unspecified"
Code: J06.9 - Acute upper respiratory infection, unspecified

Note: "Patient has essential hypertension"
Code: I10 - Essential (primary) hypertension

Now code this note:
Note: "Patient presents with major depressive disorder, single episode, moderate severity"
Code:"""}
    ])
    print(f"Response:\n{response}\n")

    # Few-shot triage classification
    print("--- Example 2: Few-shot triage classification ---")
    response = chat([
        {"role": "user", "content": """Classify patient urgency based on symptoms. Use these examples:

Symptoms: "Crushing chest pain, shortness of breath, sweating"
Urgency: EMERGENCY - Possible MI, activate code STEMI protocol

Symptoms: "Sore throat for 3 days, mild fever"
Urgency: ROUTINE - Schedule outpatient appointment within 48 hours

Symptoms: "Severe headache, sudden onset, worst of life, stiff neck"
Urgency: EMERGENCY - Rule out subarachnoid hemorrhage, immediate CT

Symptoms: "Knee pain after running, no swelling, able to walk"
Urgency: ROUTINE - Schedule orthopedic follow-up within 1 week

Now classify:
Symptoms: "Sudden facial drooping on right side, can't lift right arm, slurred speech started 30 minutes ago"
Urgency:"""}
    ])
    print(f"Response:\n{response}\n")

    # Few-shot SOAP note extraction
    print("--- Example 3: Few-shot structured extraction ---")
    response = chat([
        {"role": "user", "content": """Extract medications from clinical notes. Follow this format:

Note: "Patient is on metformin 500mg BID and lisinopril 10mg daily for diabetes and hypertension."
Medications:
- metformin 500mg | twice daily | diabetes
- lisinopril 10mg | once daily | hypertension

Note: "Currently taking warfarin 5mg at bedtime, amlodipine 5mg in the morning, and metoprolol 25mg BID."
Medications:
- warfarin 5mg | at bedtime | anticoagulation
- amlodipine 5mg | once daily (morning) | blood pressure
- metoprolol 25mg | twice daily | heart rate/blood pressure

Now extract:
Note: "Patient on atorvastatin 40mg at night, aspirin 81mg daily, omeprazole 20mg before breakfast, and insulin glargine 20 units at bedtime."
Medications:"""}
    ])
    print(f"Response:\n{response}\n")

    print("💡 TAKEAWAY: Few-shot dramatically improves accuracy for specific formats!")
    print("   The model mimics your examples exactly.")


# ============================================
# TECHNIQUE 3: Chain-of-Thought (CoT)
# ============================================

def demo_chain_of_thought():
    """
    Chain-of-thought: Ask the model to reason step by step
    Best for complex decisions requiring logic
    """
    print("\n" + "="*70)
    print("TECHNIQUE 3: CHAIN-OF-THOUGHT (CoT)")
    print("="*70)
    print("→ Ask model to think step by step. Best for complex reasoning.\n")

    # Without CoT
    print("--- WITHOUT Chain-of-Thought ---")
    response = chat([
        {"role": "user", "content": "A 65-year-old male with CKD stage 3 (GFR 40), currently on lisinopril and metformin, presents with acute gout. Can we prescribe NSAIDs?"}
    ])
    print(f"Response:\n{response}\n")

    print("-"*70)

    # With CoT
    print("--- WITH Chain-of-Thought ---")
    response = chat([
        {"role": "system", "content": "You are a clinical pharmacist. Think through each step carefully before giving a recommendation."},
        {"role": "user", "content": """A 65-year-old male with CKD stage 3 (GFR 40), currently on lisinopril and metformin, presents with acute gout. Can we prescribe NSAIDs?

Think step by step:
1. First, assess the patient's kidney function
2. Then, consider how NSAIDs affect the kidneys
3. Consider interactions with current medications
4. List alternative treatments
5. Give your final recommendation with reasoning"""}
    ], max_tokens=1500)
    print(f"Response:\n{response}\n")

    # CoT for differential diagnosis
    print("-"*70)
    print("--- CoT for Differential Diagnosis ---")
    response = chat([
        {"role": "system", "content": "You are an experienced emergency medicine physician. Walk through your clinical reasoning."},
        {"role": "user", "content": """Patient: 45-year-old female
Presenting: Sudden severe headache, neck stiffness, photophobia, temperature 38.5°C

Think through this step by step:
1. What are the key symptoms and their significance?
2. What is the most dangerous diagnosis to rule out first? Why?
3. What other conditions could cause these symptoms?
4. What tests would you order and why?
5. What is your initial management plan?"""}
    ], max_tokens=1500)
    print(f"Response:\n{response}\n")

    print("💡 TAKEAWAY: CoT produces much more thorough and accurate reasoning!")
    print("   Always use for clinical decisions, diagnosis, and complex logic.")


# ============================================
# TECHNIQUE 4: Output Formatting
# ============================================

def demo_output_formatting():
    """
    Control the output format: JSON, tables, specific structures
    Critical for integrating with EHR systems
    """
    print("\n" + "="*70)
    print("TECHNIQUE 4: OUTPUT FORMATTING")
    print("="*70)
    print("→ Control exactly how the model structures its response.\n")

    # JSON output
    print("--- Example 1: JSON Output ---")
    response = chat([
        {"role": "system", "content": """You are a clinical data extractor. Extract patient information from the clinical note and return ONLY valid JSON. No other text."""},
        {"role": "user", "content": """Extract structured data from this note:

"Mrs. Johnson, a 72-year-old female, presented to the ED with complaints of chest pain that started 2 hours ago. She describes it as a pressure-like sensation radiating to her left arm. She has a history of hypertension, type 2 diabetes, and hyperlipidemia. Current medications include metformin 1000mg BID, lisinopril 20mg daily, and atorvastatin 40mg at bedtime. Vitals: BP 165/95, HR 92, RR 20, SpO2 96%, Temp 98.6°F."

Return JSON with keys: patient_name, age, gender, chief_complaint, onset, pain_description, medical_history (array), medications (array of objects with name, dose, frequency), vitals (object)"""}
    ], max_tokens=800)
    print(f"Response:\n{response}\n")

    # Try to parse it
    try:
        # Strip markdown code block if present
        json_str = response.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = json.loads(json_str)
        print(f"✅ Valid JSON! Keys: {list(parsed.keys())}\n")
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON parsing issue: {e}\n")

    # SOAP Note format
    print("--- Example 2: SOAP Note Format ---")
    response = chat([
        {"role": "system", "content": """Convert clinical notes into SOAP format. Use exactly this structure:

**SUBJECTIVE:**
[Patient's reported symptoms and history]

**OBJECTIVE:**
[Vitals, exam findings, lab results]

**ASSESSMENT:**
[Diagnosis and clinical reasoning]

**PLAN:**
[Treatment plan, follow-up, patient education]"""},
        {"role": "user", "content": "Mr. Smith, 55M, came in saying he's been having heartburn after meals for 3 weeks, worse at night when lying down. He tried Tums but they only help temporarily. No difficulty swallowing. He's overweight at 220lbs, 5'10\". BP 135/85, HR 78. Abdomen is soft, mild epigastric tenderness. I think it's GERD. Starting omeprazole 20mg daily, dietary modifications, elevate head of bed, follow up in 4 weeks."}
    ])
    print(f"Response:\n{response}\n")

    # Discharge summary
    print("--- Example 3: Structured Discharge Summary ---")
    response = chat([
        {"role": "system", "content": """Generate a discharge summary using this EXACT format:

DISCHARGE SUMMARY
=================
Patient: [name]
MRN: [assign a sample MRN]
Admission Date: [date]
Discharge Date: [date]

DIAGNOSIS:
- Primary: [main diagnosis]
- Secondary: [other diagnoses]

HOSPITAL COURSE:
[Brief narrative]

DISCHARGE MEDICATIONS:
1. [med] - [dose] - [frequency] - [indication]

FOLLOW-UP:
- [Provider] in [timeframe]

PATIENT INSTRUCTIONS:
- [Key instruction 1]
- [Key instruction 2]"""},
        {"role": "user", "content": "Create a discharge summary for a 68-year-old male admitted for community-acquired pneumonia, treated with IV antibiotics for 3 days, improving, switching to oral antibiotics. He also has COPD and diabetes. Discharge on levofloxacin, continue home meds."}
    ], max_tokens=1000)
    print(f"Response:\n{response}\n")

    print("💡 TAKEAWAY: Be specific about output format in system prompt!")
    print("   JSON output can be parsed by your code for EHR integration.")


# ============================================
# TECHNIQUE 5: Prompt Templates
# ============================================

def demo_prompt_templates():
    """
    Create reusable prompt templates with variables
    Essential for production applications
    """
    print("\n" + "="*70)
    print("TECHNIQUE 5: PROMPT TEMPLATES")
    print("="*70)
    print("→ Reusable prompts with variables. Essential for production apps.\n")

    # Template 1: Medication review
    def medication_review_prompt(patient_age, medications, conditions):
        """Template for medication review"""
        return f"""Review the following patient's medication list for:
1. Drug-drug interactions
2. Dose appropriateness for age
3. Duplicate therapy
4. Missing recommended medications

Patient Age: {patient_age}
Current Medications: {', '.join(medications)}
Conditions: {', '.join(conditions)}

Provide a structured review with:
- INTERACTIONS FOUND (if any)
- DOSE CONCERNS (if any)
- RECOMMENDATIONS
- MISSING MEDICATIONS (based on conditions)"""

    print("--- Template 1: Medication Review ---")
    prompt = medication_review_prompt(
        patient_age=75,
        medications=["lisinopril 20mg daily", "metformin 1000mg BID", "atorvastatin 40mg", "aspirin 81mg daily"],
        conditions=["Type 2 Diabetes", "Hypertension", "Hyperlipidemia", "History of MI"]
    )
    print(f"Generated Prompt:\n{prompt}\n")

    response = chat([
        {"role": "system", "content": "You are a clinical pharmacist performing a medication review."},
        {"role": "user", "content": prompt}
    ], max_tokens=1000)
    print(f"Response:\n{response}\n")

    # Template 2: Patient education
    def patient_education_prompt(condition, reading_level, language_preference="English"):
        """Template for patient education materials"""
        return f"""Create patient education material about {condition}.

Requirements:
- Reading level: {reading_level}
- Language preference: {language_preference}
- Include: What it is, symptoms, treatment, when to seek emergency care
- Use bullet points for easy reading
- Avoid medical jargon (explain any medical terms used)
- Keep it to 200 words or less"""

    print("-"*70)
    print("--- Template 2: Patient Education ---")
    prompt = patient_education_prompt(
        condition="Type 2 Diabetes",
        reading_level="6th grade",
        language_preference="English"
    )

    response = chat([
        {"role": "system", "content": "You are a patient educator who creates easy-to-understand health materials."},
        {"role": "user", "content": prompt}
    ])
    print(f"Response:\n{response}\n")

    # Template 3: Clinical question
    def clinical_question_prompt(question, patient_context, evidence_level="high"):
        """Template for evidence-based clinical questions"""
        return f"""Answer this clinical question using evidence-based medicine:

Question: {question}
Patient Context: {patient_context}
Required Evidence Level: {evidence_level}

Structure your answer as:
1. SHORT ANSWER (1-2 sentences)
2. EVIDENCE (cite guidelines or studies)
3. CLINICAL PEARL (practical tip)
4. CAVEATS (important exceptions)"""

    print("-"*70)
    print("--- Template 3: Clinical Question ---")
    prompt = clinical_question_prompt(
        question="Should this patient be on a statin?",
        patient_context="58-year-old male, diabetic, LDL 145, no history of cardiovascular disease",
        evidence_level="guideline-based"
    )

    response = chat([
        {"role": "system", "content": "You are an evidence-based medicine consultant."},
        {"role": "user", "content": prompt}
    ])
    print(f"Response:\n{response}\n")

    print("💡 TAKEAWAY: Templates make your prompts consistent and reusable!")
    print("   In production, store templates in config files or database.")


# ============================================
# TECHNIQUE 6: Common Pitfalls & Fixes
# ============================================

def demo_pitfalls():
    """
    Common prompt engineering mistakes and how to fix them
    """
    print("\n" + "="*70)
    print("TECHNIQUE 6: COMMON PITFALLS & FIXES")
    print("="*70)

    # Pitfall 1: Vague prompt
    print("\n--- Pitfall 1: Vague vs Specific ---")
    print("\n❌ BAD (vague):")
    bad_response = chat([
        {"role": "user", "content": "Tell me about diabetes medications."}
    ], max_tokens=200)
    print(f"{bad_response[:200]}...\n")

    print("✅ GOOD (specific):")
    good_response = chat([
        {"role": "user", "content": "List the top 5 first-line medications for Type 2 Diabetes with their mechanism of action, typical starting dose, and main side effect. Format as a numbered list."}
    ], max_tokens=500)
    print(f"{good_response}\n")

    # Pitfall 2: No constraints
    print("-"*70)
    print("--- Pitfall 2: No Constraints vs Constrained ---")
    print("\n❌ BAD (no constraints):")
    bad_response = chat([
        {"role": "user", "content": "Explain hypertension treatment."}
    ], max_tokens=150)
    print(f"{bad_response[:150]}...\n")

    print("✅ GOOD (constrained):")
    good_response = chat([
        {"role": "system", "content": "You are a concise clinical reference. Maximum 100 words. Use bullet points."},
        {"role": "user", "content": "First-line treatment for Stage 1 hypertension in a 50-year-old with no comorbidities."}
    ], max_tokens=300)
    print(f"{good_response}\n")

    # Pitfall 3: Temperature too high for medical
    print("-"*70)
    print("--- Pitfall 3: Temperature Setting ---")
    print("\n❌ BAD (temperature=1.0 for medical facts):")
    bad_response = chat([
        {"role": "user", "content": "What is the normal range for adult heart rate?"}
    ], temperature=1.0)
    print(f"{bad_response[:200]}\n")

    print("✅ GOOD (temperature=0.1 for medical facts):")
    good_response = chat([
        {"role": "user", "content": "What is the normal range for adult heart rate?"}
    ], temperature=0.1)
    print(f"{good_response[:200]}\n")

    print("💡 TAKEAWAYS:")
    print("   1. Be SPECIFIC — tell the model exactly what you want")
    print("   2. Add CONSTRAINTS — word limits, format, audience")
    print("   3. Use LOW temperature (0.1-0.3) for medical facts")
    print("   4. Use HIGHER temperature (0.7-0.9) only for creative tasks")


# ============================================
# Main Menu
# ============================================

def main():
    print("\n🎯 Project 4: Prompt Engineering")
    print("="*70)
    print("Master the art of getting accurate, structured outputs from LLMs")

    print("\n\nChoose a technique to explore:")
    print("1. Zero-Shot Prompting (ask directly)")
    print("2. Few-Shot Prompting (provide examples)")
    print("3. Chain-of-Thought (step-by-step reasoning)")
    print("4. Output Formatting (JSON, SOAP, structured)")
    print("5. Prompt Templates (reusable prompts)")
    print("6. Common Pitfalls & Fixes")
    print("7. Run ALL demos")

    choice = input("\nEnter choice (1-7): ").strip()

    demos = {
        "1": demo_zero_shot,
        "2": demo_few_shot,
        "3": demo_chain_of_thought,
        "4": demo_output_formatting,
        "5": demo_prompt_templates,
        "6": demo_pitfalls,
    }

    if choice == "7":
        for demo in demos.values():
            demo()
    elif choice in demos:
        demos[choice]()
    else:
        print("❌ Invalid choice")

    print("\n\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
🎯 PROMPTING TECHNIQUES:
   1. Zero-Shot   → Simple tasks, add role for better results
   2. Few-Shot    → Specific formats, follow-the-pattern tasks
   3. CoT         → Complex reasoning, clinical decisions
   4. Formatting  → JSON for code integration, SOAP for clinical
   5. Templates   → Reusable, consistent prompts for production
   6. Pitfalls    → Be specific, constrain output, low temperature

🏥 HEALTHCARE BEST PRACTICES:
   - Always use low temperature (0.1-0.3) for clinical facts
   - Use few-shot for medical coding (ICD-10, CPT)
   - Use CoT for clinical decision support
   - Use JSON output for EHR system integration
   - Use templates for consistency across the application
   - Always include medical disclaimers

📐 PROMPT STRUCTURE:
   System: WHO the model is (role, expertise, constraints)
   User:   WHAT you want (task, context, format, examples)
   
   The more specific you are, the better the output!
""")


if __name__ == "__main__":
    main()
