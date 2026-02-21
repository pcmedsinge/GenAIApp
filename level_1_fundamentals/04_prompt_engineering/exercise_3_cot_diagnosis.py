"""
Exercise 3: Chain-of-Thought Diagnosis
Use CoT prompting for step-by-step differential diagnosis reasoning.

Skills practiced:
- Chain-of-thought prompt design
- Structured clinical reasoning
- Comparing CoT vs direct prompting
- Building multi-step reasoning templates

Healthcare context:
  Clinical reasoning is inherently step-by-step: gather data, identify key
  findings, generate differentials, rank by likelihood, determine workup.
  CoT prompting mirrors this natural clinical thought process.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat(messages, temperature=0.3, max_tokens=1500):
    """Helper"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


# ============================================================
# Clinical Cases for Diagnosis
# ============================================================

CASES = {
    "case_1": {
        "title": "Acute Chest Pain",
        "presentation": """45-year-old male presents to the ED with sudden onset substernal chest pain 
for 2 hours. Pain is described as "pressure," 8/10 severity, radiating to left arm and jaw.
Associated with diaphoresis and nausea. History of hypertension and hyperlipidemia.
Smokes 1 pack/day for 25 years. Father had MI at age 50.
Vitals: BP 160/100, HR 110, RR 22, SpO2 97%, Temp 98.6°F.
ECG shows ST elevation in leads V1-V4."""
    },
    "case_2": {
        "title": "Confusion in Elderly",
        "presentation": """78-year-old female brought by family for increasing confusion over 3 days.
Normally lives independently. Family noticed she can't remember what day it is,
has been talking to people who aren't there, and found the stove left on twice.
PMH: hypertension, type 2 diabetes, mild hearing loss.
Medications: metformin 500mg BID, amlodipine 5mg, recently started oxybutynin 5mg for urinary frequency.
Vitals: BP 148/88, HR 82, RR 16, SpO2 96%, Temp 100.2°F.
On exam: oriented to person only. Mini-Cog 1/5. No focal neurologic deficits."""
    },
    "case_3": {
        "title": "Young Woman with Fatigue",
        "presentation": """32-year-old female presents with 3 months of progressive fatigue, unintentional 
weight loss (12 lbs), and increased thirst and urination. She reports blurry vision 
for the past 2 weeks. No significant PMH. Family history: mother has Hashimoto's 
thyroiditis, sister has celiac disease.
Vitals: BP 118/72, HR 88, RR 14, SpO2 99%, Temp 98.2°F.
BMI: 22 (was 24 three months ago).
On exam: dry mucous membranes, no thyromegaly, no lymphadenopathy."""
    },
    "case_4": {
        "title": "Abdominal Pain and Jaundice",
        "presentation": """55-year-old male presents with 1 week of progressively worsening right upper 
quadrant pain, dark urine, pale stools, and yellowing of eyes and skin.
Pain is dull, constant, 6/10, worse after fatty meals. He's had a 15-pound 
weight loss over 2 months. History of heavy alcohol use (6 beers/day for 20 years),
quit 1 year ago. No prior surgeries.
Vitals: BP 132/78, HR 76, RR 14, SpO2 98%, Temp 99.1°F.
On exam: jaundiced, palpable non-tender gallbladder (Courvoisier's sign), 
mild hepatomegaly. No ascites."""
    },
}


# ============================================================
# Approach 1: Direct (No CoT)
# ============================================================

def diagnose_direct(case_presentation):
    """Ask for diagnosis directly — no step-by-step reasoning"""
    response = chat([
        {"role": "system", "content": "You are a physician. Provide your differential diagnosis."},
        {"role": "user", "content": f"What is your differential diagnosis?\n\n{case_presentation}"}
    ], max_tokens=500)
    return response


# ============================================================
# Approach 2: Chain-of-Thought
# ============================================================

COT_TEMPLATE = """Analyze this case using systematic clinical reasoning. Think through each step:

STEP 1 — KEY FINDINGS:
List the most significant symptoms, signs, and history elements. Mark any RED FLAGS with ⚠️.

STEP 2 — PATTERN RECOGNITION:
What clinical syndrome or pattern do these findings suggest?
What organ system(s) are likely involved?

STEP 3 — DIFFERENTIAL DIAGNOSIS:
List top 5 diagnoses ranked by probability:
  #1 (Most Likely): [diagnosis] — [key supporting evidence]
  #2: [diagnosis] — [supporting evidence]
  #3: [diagnosis] — [supporting evidence]
  #4: [diagnosis] — [supporting evidence]
  #5: [diagnosis] — [supporting evidence]

STEP 4 — CRITICAL ACTIONS:
What must be done IMMEDIATELY (if anything)?
What tests/studies would you order and WHY?

STEP 5 — REASONING SUMMARY:
In 2-3 sentences, explain your clinical reasoning for the #1 diagnosis.
What would change your mind (i.e., what findings would point to a different diagnosis)?

Patient Presentation:
{case}"""


def diagnose_cot(case_presentation):
    """Chain-of-thought structured diagnosis"""
    prompt = COT_TEMPLATE.format(case=case_presentation)
    response = chat([
        {"role": "system", "content": """You are an experienced emergency medicine attending physician 
teaching a medical resident. Walk through your clinical reasoning step by step.
Be thorough but practical. This is for educational purposes only."""},
        {"role": "user", "content": prompt}
    ], max_tokens=2000)
    return response


# ============================================================
# Approach 3: Socratic CoT (Teaching Method)
# ============================================================

def diagnose_socratic(case_presentation):
    """Socratic method — asks questions to guide reasoning"""
    response = chat([
        {"role": "system", "content": """You are a clinical teaching attending using the Socratic method.
Present your reasoning as if teaching a medical student, asking and answering 
key clinical questions along the way. Format:

🤔 Question: [clinical question]
💡 Answer: [your reasoning]

Do this for 5-6 key decision points, then give your final assessment."""},
        {"role": "user", "content": f"Walk me through your diagnostic reasoning for this case:\n\n{case_presentation}"}
    ], max_tokens=2000)
    return response


# ============================================================
# Main Demo
# ============================================================

def main():
    print("\n🧠 Exercise 3: Chain-of-Thought Diagnosis")
    print("=" * 70)
    print("Compare direct vs CoT prompting for clinical reasoning\n")

    # Show available cases
    print("Available cases:")
    for key, case in CASES.items():
        print(f"  {key}: {case['title']}")

    print("\nChoose a demo:")
    print("1. Direct vs CoT comparison (one case)")
    print("2. Full CoT analysis (all cases)")
    print("3. Socratic teaching method")
    print("4. Enter your own case (interactive)")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        case_key = input("Choose case (case_1/case_2/case_3/case_4) [case_1]: ").strip() or "case_1"
        case = CASES.get(case_key, CASES["case_1"])

        print(f"\n{'═' * 70}")
        print(f"📝 CASE: {case['title']}")
        print(f"{'═' * 70}")
        print(case["presentation"])

        print(f"\n\n🔴 DIRECT APPROACH (no Chain-of-Thought):")
        print("─" * 70)
        direct_result = diagnose_direct(case["presentation"])
        print(direct_result)

        print(f"\n\n🟢 CHAIN-OF-THOUGHT APPROACH:")
        print("─" * 70)
        cot_result = diagnose_cot(case["presentation"])
        print(cot_result)

        print(f"""
{'═' * 70}
💡 COMPARISON:
   🔴 Direct: Lists diagnoses but may miss reasoning steps
   🟢 CoT:    Systematic — identifies patterns, supports each diagnosis,
              orders appropriate workup, considers alternatives
   
   CoT produces more RELIABLE clinical reasoning because:
   • Forces the model to consider evidence step-by-step
   • Red flags are explicitly identified
   • Each diagnosis has supporting evidence
   • Alternative diagnoses are considered
   • Immediate actions are highlighted
""")

    elif choice == "2":
        for key, case in CASES.items():
            print(f"\n{'═' * 70}")
            print(f"📝 CASE: {case['title']}")
            print(f"{'═' * 70}")
            print(f"{case['presentation'][:120]}...\n")
            print("CoT ANALYSIS:")
            print("─" * 70)
            print(diagnose_cot(case["presentation"]))

    elif choice == "3":
        case_key = input("Choose case (case_1/case_2/case_3/case_4) [case_2]: ").strip() or "case_2"
        case = CASES.get(case_key, CASES["case_2"])

        print(f"\n{'═' * 70}")
        print(f"📝 CASE: {case['title']} — Socratic Teaching")
        print(f"{'═' * 70}")
        print(case["presentation"])
        print(f"\n🎓 SOCRATIC REASONING:")
        print("─" * 70)
        print(diagnose_socratic(case["presentation"]))

    elif choice == "4":
        print("\n💬 Describe the clinical case (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and lines:
                break
            lines.append(line)
        user_case = "\n".join(lines)

        print("\n⏳ Running CoT analysis...\n")
        print(diagnose_cot(user_case))

    else:
        print("Invalid choice")

    print(f"""
{'═' * 70}
KEY LEARNINGS:
{'═' * 70}
1. CoT DRAMATICALLY improves clinical reasoning quality
2. Structured steps (findings → pattern → differential → workup) mirror real clinical thinking
3. Asking to "mark red flags" ensures urgent findings aren't missed
4. "What would change your mind?" tests robustness of reasoning
5. The Socratic method is great for TEACHING — builds understanding
6. For clinical AI, ALWAYS use CoT — never just "give me a diagnosis"

🏥 REAL-WORLD APPLICATION:
   Clinical Decision Support systems use CoT-style reasoning to:
   • Show their work (explainability / auditability)
   • Reduce errors (systematic approach catches oversights)
   • Build clinician trust (reasoning is visible and checkable)
""")


if __name__ == "__main__":
    main()
