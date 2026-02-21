"""
Exercise 2: Few-Shot ICD-10 Coder
Build a few-shot prompt that accurately assigns ICD-10 codes to clinical descriptions.

Skills practiced:
- Few-shot prompt construction with real examples
- Structured output for medical coding
- Confidence scoring
- Handling ambiguous cases

Healthcare context:
  ICD-10 coding is a critical revenue cycle task. Coders read clinical notes
  and assign standardized codes. Few-shot prompting helps LLMs learn the
  exact format and reasoning pattern you want.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat(messages, temperature=0.1, max_tokens=800):
    """Helper — low temperature for coding accuracy"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


# ============================================================
# Few-Shot Examples (the "training" examples in your prompt)
# ============================================================

FEW_SHOT_EXAMPLES = """Here are examples of clinical descriptions and their ICD-10 codes:

Example 1:
Clinical Description: "Patient presents with type 2 diabetes mellitus with diabetic chronic kidney disease, stage 3"
Primary Code: E11.22 - Type 2 diabetes mellitus with diabetic chronic kidney disease
Secondary Code: N18.3 - Chronic kidney disease, stage 3 (stage 3a/3b unspecified)
Reasoning: Diabetes is the underlying cause; CKD is a manifestation. Code both per coding guidelines.
Confidence: HIGH

Example 2:
Clinical Description: "Acute ST-elevation myocardial infarction of anterior wall, initial episode"
Primary Code: I21.09 - ST elevation (STEMI) myocardial infarction involving other coronary artery of anterior wall
Secondary Code: None required
Reasoning: STEMI codes are specific to wall location. "Initial episode" maps to I21 (not I22 for subsequent).
Confidence: HIGH

Example 3:
Clinical Description: "Patient with COPD with acute exacerbation and community-acquired pneumonia"
Primary Code: J44.0 - Chronic obstructive pulmonary disease with (acute) lower respiratory infection
Secondary Code: J18.9 - Pneumonia, unspecified organism
Reasoning: J44.0 captures COPD with infection. Must also code the pneumonia separately per guidelines.
Confidence: HIGH

Example 4:
Clinical Description: "Major depressive disorder, recurrent, moderate, with anxious distress"
Primary Code: F33.1 - Major depressive disorder, recurrent, moderate
Secondary Code: None — anxious distress is a specifier not separately coded
Reasoning: Recurrent + moderate maps to F33.1. Anxious distress is a clinical specifier without its own code.
Confidence: MEDIUM (clinician should verify if anxiety disorder is separate)

Example 5:
Clinical Description: "Fall from ladder at home causing closed fracture of right distal radius"
Primary Code: S52.501A - Unspecified fracture of the lower end of right radius, initial encounter
Secondary Code: W11.XXXA - Fall on and from ladder, initial encounter
Additional: Y93.89 - Activity, other specified (home maintenance)
Reasoning: Code the injury (fracture), external cause (fall from ladder), and activity. "A" = initial encounter.
Confidence: HIGH"""


# ============================================================
# The ICD-10 Coder (Few-Shot)
# ============================================================

def code_clinical_description(description, include_reasoning=True):
    """Use few-shot prompting to assign ICD-10 codes"""

    format_instruction = """
Now code this description using the EXACT same format:
- Primary Code: [code] - [description]
- Secondary Code: [code] - [description] (or "None required")
- Additional Code: [if applicable]
- Reasoning: [explain your code selection]
- Confidence: HIGH / MEDIUM / LOW"""

    response = chat([
        {"role": "system", "content": """You are a certified medical coder (CCS, CPC) with 15 years experience.
Assign ICD-10-CM codes following official coding guidelines.
Be specific — use the most specific code available.
If the description is ambiguous, note what additional info would help."""},
        {"role": "user", "content": f"{FEW_SHOT_EXAMPLES}\n\n{format_instruction}\n\nClinical Description: \"{description}\""}
    ])
    return response


# ============================================================
# Batch Coder
# ============================================================

def batch_code(descriptions):
    """Code multiple descriptions"""
    results = []
    for desc in descriptions:
        result = code_clinical_description(desc)
        results.append({"description": desc, "coding": result})
    return results


# ============================================================
# JSON Output Coder (for system integration)
# ============================================================

def code_to_json(description):
    """Return coding result as structured JSON"""
    response = chat([
        {"role": "system", "content": """You are a certified medical coder. Return ICD-10 codes as JSON ONLY.
Format:
{
  "clinical_description": "",
  "codes": [
    {"code": "", "description": "", "type": "primary|secondary|additional"}
  ],
  "reasoning": "",
  "confidence": "HIGH|MEDIUM|LOW",
  "needs_clarification": null or "string explaining what info is needed"
}"""},
        {"role": "user", "content": f"""{FEW_SHOT_EXAMPLES}

Now code as JSON:
Clinical Description: "{description}" """}
    ])
    return response


# ============================================================
# Test Cases
# ============================================================

TEST_CASES = [
    "Acute bronchitis due to respiratory syncytial virus",
    "Type 1 diabetes with proliferative diabetic retinopathy in both eyes",
    "Patient admitted for chest pain, ruled out for MI, found to have GERD",
    "Chronic low back pain with left-sided sciatica due to lumbar disc herniation L4-L5",
    "Urinary tract infection due to E. coli in a pregnant patient, 28 weeks gestation",
    "Patient fell at nursing home, sustaining hip fracture right side",
]

# Ambiguous cases to test confidence handling
AMBIGUOUS_CASES = [
    "Patient has some abdominal pain",
    "Headache",
    "Elevated blood sugar",
]


# ============================================================
# Main Demo
# ============================================================

def main():
    print("\n🏥 Exercise 2: Few-Shot ICD-10 Coder")
    print("=" * 70)
    print("Assign ICD-10 codes using few-shot prompting\n")

    print("Choose a demo:")
    print("1. Code test cases (see few-shot in action)")
    print("2. Ambiguous cases (see confidence handling)")
    print("3. JSON output for system integration")
    print("4. Code your own clinical description (interactive)")
    print("5. Run all demos")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1" or choice == "5":
        print("\n📋 CODING TEST CASES")
        print("=" * 70)
        for desc in TEST_CASES:
            print(f"\n{'─' * 70}")
            print(f"📝 \"{desc}\"")
            print(f"{'─' * 70}")
            result = code_clinical_description(desc)
            print(result)

    if choice == "2" or choice == "5":
        print(f"\n\n{'═' * 70}")
        print("⚠️  AMBIGUOUS CASES (testing confidence)")
        print("=" * 70)
        for desc in AMBIGUOUS_CASES:
            print(f"\n{'─' * 70}")
            print(f"📝 \"{desc}\"")
            print(f"{'─' * 70}")
            result = code_clinical_description(desc)
            print(result)

        print(f"""
💡 NOTICE:
   • Ambiguous descriptions get LOWER confidence scores
   • The coder notes what additional info is needed
   • This is exactly how human coders handle ambiguity!
""")

    if choice == "3" or choice == "5":
        print(f"\n\n{'═' * 70}")
        print("🔗 JSON OUTPUT FOR SYSTEM INTEGRATION")
        print("=" * 70)
        for desc in TEST_CASES[:3]:
            print(f"\n📝 \"{desc}\"")
            json_result = code_to_json(desc)
            print(json_result)

            try:
                json_str = json_result.strip()
                if json_str.startswith("```"):
                    json_str = json_str.split("\n", 1)[1].rsplit("```", 1)[0]
                parsed = json.loads(json_str)
                print(f"✅ Valid JSON — {len(parsed['codes'])} code(s), Confidence: {parsed['confidence']}")
            except Exception as e:
                print(f"⚠️  Parse issue: {e}")

    if choice == "4":
        print("\n💬 Enter clinical descriptions to code. Type 'quit' to exit.\n")
        while True:
            desc = input("Clinical description: ").strip()
            if desc.lower() in ['quit', 'exit', 'q']:
                break
            if not desc:
                continue
            print(f"\n⏳ Coding...\n")
            print(code_clinical_description(desc))
            print()

    if choice not in ["1", "2", "3", "4", "5"]:
        print("Invalid choice")

    print(f"""
{'═' * 70}
KEY LEARNINGS:
{'═' * 70}
1. Few-shot examples TEACH the format — model follows the pattern exactly
2. More examples (5+) help cover edge cases (fractures, dual codes, etc.)
3. Including "Reasoning" in examples makes the model explain its logic
4. Including "Confidence" teaches the model to flag ambiguous cases
5. Low temperature (0.1) is critical for coding accuracy
6. JSON output enables integration with billing/EHR systems

🏥 REAL-WORLD NOTE:
   In production, you'd validate LLM-suggested codes against the official
   ICD-10 code set and flag any codes not in the database. LLMs can suggest
   codes that don't exist! Always validate.
""")


if __name__ == "__main__":
    main()
