"""
Exercise 1: Enhanced BMI Tool with Clinical Interpretation

Skills practiced:
- Creating LangChain tools with the @tool decorator
- Extending tool output with clinical context and recommendations
- Building agents that chain multiple tools together
- Understanding how tool descriptions guide agent behavior

Key insight: main.py already has a basic calculate_bmi tool. This exercise
  extends it to include clinical interpretation, risk stratification, and
  weight management recommendations — showing how richer tool output
  leads to better agent reasoning.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent as create_langchain_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# Enhanced BMI Tool
# ============================================================

@tool
def calculate_bmi_enhanced(weight_kg: float, height_cm: float, age: int, sex: str) -> str:
    """Calculate BMI with full clinical interpretation, risk assessment, and
    weight management recommendations.
    Weight in kilograms, height in centimeters, sex as 'male' or 'female'."""

    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    # Category
    if bmi < 18.5:
        category = "Underweight"
        risk = "Increased risk of nutritional deficiency, osteoporosis, immune dysfunction"
        action = "Evaluate for underlying causes. Nutritional counseling recommended."
    elif bmi < 25:
        category = "Normal weight"
        risk = "Average risk"
        action = "Maintain current lifestyle. Annual wellness screening."
    elif bmi < 30:
        category = "Overweight"
        risk = "Increased risk of HTN, dyslipidemia, type 2 diabetes"
        action = "Lifestyle modifications: diet + 150min/week moderate exercise."
    elif bmi < 35:
        category = "Obesity Class I"
        risk = "High risk of cardiovascular disease, diabetes, sleep apnea"
        action = "Intensive lifestyle intervention. Consider pharmacotherapy if comorbidities."
    elif bmi < 40:
        category = "Obesity Class II"
        risk = "Very high risk. Screen for metabolic syndrome."
        action = "Pharmacotherapy + lifestyle. Consider referral to obesity medicine."
    else:
        category = "Obesity Class III (Severe)"
        risk = "Highest risk category. Significant mortality increase."
        action = "Consider bariatric surgery evaluation. Multidisciplinary approach."

    # Ideal weight range (BMI 18.5-24.9)
    ideal_low = 18.5 * (height_m ** 2)
    ideal_high = 24.9 * (height_m ** 2)

    # Age-adjusted considerations
    age_note = ""
    if age >= 65:
        age_note = "Note: In elderly (65+), slightly higher BMI (25-27) may be protective."
    elif age < 20:
        age_note = "Note: For patients <20, use age-specific BMI percentile charts."

    return (
        f"BMI: {bmi:.1f} — {category}\n"
        f"Weight: {weight_kg}kg | Height: {height_cm}cm | Age: {age} | Sex: {sex}\n"
        f"Risk: {risk}\n"
        f"Ideal weight range: {ideal_low:.1f}-{ideal_high:.1f} kg\n"
        f"Recommendation: {action}\n"
        f"{age_note}"
    )


@tool
def calculate_waist_hip_ratio(waist_cm: float, hip_cm: float, sex: str) -> str:
    """Calculate waist-to-hip ratio for cardiovascular risk assessment.
    Waist and hip circumference in centimeters, sex as 'male' or 'female'."""
    ratio = waist_cm / hip_cm

    if sex.lower() == "male":
        risk = "Low" if ratio < 0.90 else "Moderate" if ratio < 0.95 else "High"
    else:
        risk = "Low" if ratio < 0.80 else "Moderate" if ratio < 0.85 else "High"

    return (
        f"Waist-to-Hip Ratio: {ratio:.2f}\n"
        f"Waist: {waist_cm}cm | Hip: {hip_cm}cm | Sex: {sex}\n"
        f"Cardiovascular Risk: {risk}\n"
        f"Note: WHR captures central obesity risk that BMI alone may miss."
    )


@tool
def lookup_medication(medication_name: str) -> str:
    """Look up medication information. Available: metformin, lisinopril, amlodipine, apixaban, sertraline, omeprazole, orlistat, semaglutide."""
    medications = {
        "metformin": "Class: Biguanide | Indication: Type 2 Diabetes | Dose: 500-2000mg daily | Contraindicated: eGFR<30",
        "lisinopril": "Class: ACE Inhibitor | Indication: HTN, HF | Dose: 10-40mg daily | Monitor: BP, K+, Cr",
        "amlodipine": "Class: CCB | Indication: HTN, Angina | Dose: 2.5-10mg daily",
        "apixaban": "Class: DOAC | Indication: AFib, VTE | Dose: 5mg BID",
        "sertraline": "Class: SSRI | Indication: Depression, Anxiety | Dose: 50-200mg daily",
        "omeprazole": "Class: PPI | Indication: GERD | Dose: 20-40mg daily",
        "orlistat": "Class: Lipase inhibitor | Indication: Obesity (BMI≥30 or BMI≥27 with comorbidities) | Dose: 120mg TID with meals | Side Effects: Steatorrhea, fat-soluble vitamin deficiency",
        "semaglutide": "Class: GLP-1 RA | Indication: Obesity (BMI≥30 or BMI≥27), Type 2 Diabetes | Dose: 2.4mg weekly (obesity) | Side Effects: Nausea, pancreatitis risk | Monitor: Weight, HbA1c",
    }
    result = medications.get(medication_name.lower())
    if result:
        return result
    return f"Not found. Available: {', '.join(medications.keys())}"


@tool
def check_lab_value(test_name: str, value: float) -> str:
    """Interpret a lab value. Available: hba1c, gfr, potassium, cholesterol_total, ldl, triglycerides."""
    tests = {
        "hba1c": lambda v: f"HbA1c {v}%: {'Normal' if v < 5.7 else 'Prediabetes' if v < 6.5 else 'Diabetes'}",
        "gfr": lambda v: f"GFR {v}: {'Normal' if v >= 90 else 'Mild CKD' if v >= 60 else 'Moderate CKD' if v >= 30 else 'Severe CKD'}",
        "potassium": lambda v: f"K+ {v}: {'LOW' if v < 3.5 else 'Normal' if v <= 5.0 else 'HIGH'}",
        "cholesterol_total": lambda v: f"Total Cholesterol {v}: {'Desirable' if v < 200 else 'Borderline' if v < 240 else 'High'}",
        "ldl": lambda v: f"LDL {v}: {'Optimal' if v < 100 else 'Near Optimal' if v < 130 else 'Borderline' if v < 160 else 'High'}",
        "triglycerides": lambda v: f"Triglycerides {v}: {'Normal' if v < 150 else 'Borderline' if v < 200 else 'High'}",
    }
    fn = tests.get(test_name.lower())
    if fn:
        return fn(value)
    return f"Not found. Available: {', '.join(tests.keys())}"


# All tools
all_tools = [calculate_bmi_enhanced, calculate_waist_hip_ratio, lookup_medication, check_lab_value]


def create_agent():
    """Create a healthcare agent with all tools"""
    return create_langchain_agent(
        llm,
        tools=all_tools,
        system_prompt="""You are a clinical decision support agent specializing in metabolic health.
Use your tools to calculate BMI, assess body composition, look up medications,
and interpret labs. Chain tools together to build complete assessments.
Always explain clinical reasoning. Educational purposes only."""
    )


# ============================================================
# DEMO 1: Enhanced BMI in Action
# ============================================================

def demo_enhanced_bmi():
    """Show the enhanced BMI tool vs basic BMI"""
    print("\n" + "=" * 70)
    print("DEMO 1: ENHANCED BMI TOOL")
    print("=" * 70)
    print("""
    The basic BMI tool from main.py returns: 'BMI: 31.2 (Obesity Class I)'
    Our enhanced version adds: risk assessment, ideal weight range,
    age-adjusted notes, and clinical recommendations.
    """)

    agent = create_agent()

    test_cases = [
        "Calculate BMI for a 45-year-old male, 92kg, 175cm",
        "Patient is a 72-year-old female, 58kg, 160cm. BMI?",
        "28-year-old male athlete, 95kg, 188cm. Is he overweight?",
    ]

    for q in test_cases:
        print(f"\n{'─' * 60}")
        print(f"  Q: {q}\n")
        result = agent.invoke({"messages": [{"role": "user", "content": q}]})
        answer = result["messages"][-1].content
        print(f"\n  ANSWER: {answer[:400]}")


# ============================================================
# DEMO 2: Multi-Tool Assessment
# ============================================================

def demo_multi_tool():
    """Agent chains BMI + labs + medication for complete assessment"""
    print("\n" + "=" * 70)
    print("DEMO 2: MULTI-TOOL METABOLIC ASSESSMENT")
    print("=" * 70)
    print("""
    Agent combines multiple tools to build a comprehensive assessment:
    BMI → Lab interpretation → Medication recommendation → Risk summary
    """)

    agent = create_agent()

    question = (
        "55-year-old female, 98kg, 165cm. Labs: HbA1c 7.2%, "
        "total cholesterol 245, LDL 162, triglycerides 210. "
        "She's currently on metformin. Full metabolic assessment."
    )

    print(f"\n{'─' * 60}")
    print(f"  Q: {question}\n")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content
    print(f"\n  COMPREHENSIVE ASSESSMENT:\n  {answer[:800]}")


# ============================================================
# DEMO 3: Waist-Hip Ratio vs BMI
# ============================================================

def demo_body_composition():
    """Compare BMI and WHR for a muscular patient"""
    print("\n" + "=" * 70)
    print("DEMO 3: BMI vs WAIST-HIP RATIO")
    print("=" * 70)
    print("""
    BMI alone can be misleading — a muscular person may have high BMI
    but low cardiovascular risk. WHR captures central obesity better.
    """)

    agent = create_agent()

    question = (
        "A 35-year-old male weightlifter: 102kg, 180cm, waist 82cm, hip 100cm. "
        "Calculate both BMI and waist-hip ratio. Is he truly at risk?"
    )

    print(f"\n{'─' * 60}")
    print(f"  Q: {question}\n")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content
    print(f"\n  ANALYSIS: {answer[:600]}")


# ============================================================
# DEMO 4: Interactive
# ============================================================

def demo_interactive():
    """Interactive chat with the metabolic health agent"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE METABOLIC HEALTH AGENT")
    print("=" * 70)
    print("  Tools: enhanced BMI, waist-hip ratio, medication lookup, labs")
    print("  Type 'quit' to exit.\n")

    agent = create_agent()
    messages = []

    while True:
        question = input("  You: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue

        messages.append({"role": "user", "content": question})
        result = agent.invoke({"messages": messages})
        answer = result["messages"][-1].content
        print(f"\n  Agent: {answer}\n")
        messages = result["messages"]


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 1: ENHANCED BMI TOOL WITH CLINICAL INTERPRETATION")
    print("=" * 70)
    print("""
    Extends the basic BMI tool with risk assessment, recommendations,
    and a new waist-hip ratio tool for cardiovascular risk.

    Choose a demo:
      1 → Enhanced BMI (vs basic)
      2 → Multi-tool metabolic assessment
      3 → BMI vs Waist-Hip Ratio comparison
      4 → Interactive
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1": demo_enhanced_bmi()
    elif choice == "2": demo_multi_tool()
    elif choice == "3": demo_body_composition()
    elif choice == "4": demo_interactive()
    elif choice == "5":
        demo_enhanced_bmi()
        demo_multi_tool()
        demo_body_composition()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. TOOL RICHNESS MATTERS: Richer tool output (risk assessment,
   recommendations) leads to better agent reasoning. Don't just
   return a number — return clinical context.

2. TOOL DESCRIPTIONS GUIDE THE AGENT: The docstring tells the agent
   WHEN and HOW to use each tool. Be specific about what parameters
   to provide and what the tool returns.

3. MULTI-TOOL CHAINING: The agent calls multiple tools to build
   a comprehensive picture — just like a clinician running multiple
   tests before making a diagnosis.

4. LIMITATIONS: BMI doesn't capture muscle mass vs fat. Adding
   waist-hip ratio shows how multiple metrics give a better picture.
   Same principle applies to AI: multiple data sources > one metric.
"""

if __name__ == "__main__":
    main()
