"""
Exercise 1: Clinical Note Structurer
Convert free-text clinical notes into structured SOAP format using prompt engineering.

Skills practiced:
- System prompt design for consistent output
- Output format control (SOAP structure)
- Handling messy real-world clinical text
- Comparing different prompt approaches

Healthcare context:
  Doctors often write quick, unstructured notes during patient visits.
  Structuring these into SOAP format (Subjective, Objective, Assessment, Plan)
  is a tedious but critical task. LLMs can automate this!
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat(messages, temperature=0.2, max_tokens=1000):
    """Helper to make API calls"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


# ============================================================
# Sample Clinical Notes (messy, real-world style)
# ============================================================

CLINICAL_NOTES = [
    {
        "id": "NOTE-001",
        "provider": "Dr. Martinez",
        "raw_note": """Pt is 67yo M came in complaining of SOB x 3 days, getting worse especially when 
climbing stairs. Also noticed ankle swelling both sides. Hx of HTN, DM2, MI 2019. 
On metformin 1g BID, lisinopril 20 daily, aspirin 81, atorvastatin 40 qhs. 
VS: BP 152/94, HR 98, RR 22, O2 sat 93% RA, temp 98.4. Lungs — bilateral crackles 
bases. Heart — regular, S3 gallop. 2+ pitting edema bilat LE. Think CHF exacerbation. 
Will get BNP, CMP, CXR, EKG. Start furosemide 40 IV now. Fluid restrict 2L. Daily weights. 
Cardiology consult. Admit to med-surg."""
    },
    {
        "id": "NOTE-002",
        "provider": "NP Johnson",
        "raw_note": """45F here for f/u on depression management. Started sertraline 50mg 6 wks ago. 
Says sleeping better, mood improving but still low energy. PHQ-9 went from 18 to 11. 
No SI/HI. Appetite returned, gained 3 lbs. Tolerating med well, no major SE. 
A bit of nausea first week but resolved. Exercising 2x/wk now. Good support system.
Assessment — partial response to SSRI. Plan: increase to 100mg, continue therapy, 
recheck in 4 wks with PHQ-9. Discussed importance of exercise and sleep hygiene."""
    },
    {
        "id": "NOTE-003",
        "provider": "Dr. Chen",
        "raw_note": """72F diabetic presents with R foot ulcer x 2 weeks not healing. Noticed redness 
spreading up. On insulin glargine 30u qhs, metformin 500 BID. A1c last month was 8.9. 
exam shows 2cm ulcer plantar R foot with surrounding erythema, purulent drainage, 
no crepitus. Pedal pulses weak bilat. Temp 100.8. WBC 14.2. ABI 0.6 right. 
concern for infected diabetic foot ulcer w likely PAD. need vascular consult, wound 
care, abx — starting augmentin 875 BID, blood cx before abx if possible, 
optimize glucose, possible imaging to r/o osteo."""
    },
]


# ============================================================
# Approach 1: Simple Zero-Shot SOAP Conversion
# ============================================================

def zero_shot_structurer(raw_note):
    """Basic approach — just ask for SOAP format"""
    response = chat([
        {"role": "system", "content": "You are a clinical documentation specialist."},
        {"role": "user", "content": f"Convert this clinical note into SOAP format:\n\n{raw_note}"}
    ])
    return response


# ============================================================
# Approach 2: Detailed System Prompt with SOAP Template
# ============================================================

SOAP_SYSTEM_PROMPT = """You are a clinical documentation specialist who converts free-text provider notes 
into structured SOAP format. Follow this EXACT template:

**SUBJECTIVE:**
- Chief Complaint (CC): [main reason for visit, in patient's words if possible]
- History of Present Illness (HPI): [onset, duration, severity, associated symptoms]
- Past Medical History (PMH): [relevant conditions]
- Medications: [current medication list with doses]
- Review of Systems: [positive and pertinent negative findings]

**OBJECTIVE:**
- Vitals: [BP, HR, RR, SpO2, Temp — structured format]
- Physical Exam: [organized by system]
- Labs/Imaging: [if mentioned, with values]

**ASSESSMENT:**
- Primary Diagnosis: [most likely diagnosis]
- Differential: [if applicable]
- Clinical Reasoning: [brief justification]

**PLAN:**
- Medications: [new or changed medications]
- Orders: [labs, imaging, procedures]
- Referrals: [if any]
- Follow-up: [timeline and conditions]
- Patient Education: [if mentioned]

Rules:
- If information is NOT in the note, write "Not documented" — never invent data
- Expand abbreviations in parentheses, e.g., "SOB (Shortness of Breath)"
- Keep the original clinical meaning; do not add clinical judgment"""


def detailed_structurer(raw_note):
    """Detailed template approach"""
    response = chat([
        {"role": "system", "content": SOAP_SYSTEM_PROMPT},
        {"role": "user", "content": f"Convert this note:\n\n{raw_note}"}
    ])
    return response


# ============================================================
# Approach 3: JSON Output for EHR Integration
# ============================================================

def json_structurer(raw_note):
    """Output structured data as JSON for system integration"""
    response = chat([
        {"role": "system", "content": """Convert clinical notes to SOAP format as JSON.
Return ONLY valid JSON with this structure:
{
  "subjective": {
    "chief_complaint": "",
    "hpi": "",
    "pmh": [],
    "medications": [{"name": "", "dose": "", "frequency": ""}],
    "ros": ""
  },
  "objective": {
    "vitals": {"bp": "", "hr": "", "rr": "", "spo2": "", "temp": ""},
    "physical_exam": "",
    "labs_imaging": ""
  },
  "assessment": {
    "primary_diagnosis": "",
    "clinical_reasoning": ""
  },
  "plan": {
    "medications": [],
    "orders": [],
    "referrals": [],
    "follow_up": "",
    "patient_education": ""
  }
}
If info is missing, use null. Expand medical abbreviations."""},
        {"role": "user", "content": raw_note}
    ], max_tokens=1500)
    return response


# ============================================================
# Main Demo
# ============================================================

def main():
    print("\n📋 Exercise 1: Clinical Note Structurer")
    print("=" * 70)
    print("Convert messy clinical notes into structured SOAP format\n")

    print("Choose a demo:")
    print("1. Compare all 3 approaches on one note")
    print("2. Process all sample notes (detailed approach)")
    print("3. JSON output for EHR integration")
    print("4. Structure your own note (interactive)")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        note = CLINICAL_NOTES[0]
        print(f"\n📝 Original Note (Note {note['id']}, {note['provider']}):")
        print(f"{'─' * 70}")
        print(note["raw_note"])

        print(f"\n\n🔵 APPROACH 1: Zero-Shot (just ask)")
        print("─" * 70)
        print(zero_shot_structurer(note["raw_note"]))

        print(f"\n\n🟢 APPROACH 2: Detailed Template")
        print("─" * 70)
        print(detailed_structurer(note["raw_note"]))

        print(f"\n\n🟡 APPROACH 3: JSON Output")
        print("─" * 70)
        json_result = json_structurer(note["raw_note"])
        print(json_result)

        try:
            json_str = json_result.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("\n", 1)[1].rsplit("```", 1)[0]
            parsed = json.loads(json_str)
            print(f"\n✅ Valid JSON! Can be sent to EHR system.")
            print(f"   Medications found: {len(parsed['subjective']['medications'])}")
        except Exception as e:
            print(f"\n⚠️  JSON parsing note: {e}")

        print(f"""
{'═' * 70}
💡 COMPARISON:
   Zero-Shot:   Quick but inconsistent format
   Detailed:    Consistent SOAP structure, handles abbreviations
   JSON:        Machine-readable, best for EHR integration
   
   The detailed system prompt is the key difference!
""")

    elif choice == "2":
        for note in CLINICAL_NOTES:
            print(f"\n{'═' * 70}")
            print(f"📝 {note['id']} — {note['provider']}")
            print(f"{'═' * 70}")
            print(f"\nRaw: {note['raw_note'][:100]}...\n")
            print("SOAP Output:")
            print("─" * 70)
            print(detailed_structurer(note["raw_note"]))

    elif choice == "3":
        note = CLINICAL_NOTES[0]
        print(f"\n📝 Processing {note['id']} as JSON...\n")
        result = json_structurer(note["raw_note"])
        print(result)

        try:
            json_str = result.strip()
            if json_str.startswith("```"):
                json_str = json_str.split("\n", 1)[1].rsplit("```", 1)[0]
            parsed = json.loads(json_str)
            print(f"\n✅ Parsed successfully!")
            print(f"   Chief Complaint: {parsed['subjective']['chief_complaint']}")
            print(f"   Primary Dx: {parsed['assessment']['primary_diagnosis']}")
            print(f"   Meds: {len(parsed['subjective']['medications'])} listed")
            print(f"   Orders: {parsed['plan']['orders']}")
        except Exception as e:
            print(f"\n⚠️  Parse issue: {e}")

    elif choice == "4":
        print("\n💬 Paste or type a clinical note (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "":
                if lines:
                    break
            else:
                lines.append(line)

        user_note = "\n".join(lines)
        print("\n⏳ Structuring into SOAP format...\n")
        print(detailed_structurer(user_note))

    else:
        print("Invalid choice")

    print(f"""
{'═' * 70}
KEY LEARNINGS:
{'═' * 70}
1. System prompts with EXACT templates produce consistent output
2. Rules like "never invent data" prevent hallucination
3. "Expand abbreviations" catches medical shorthand (SOB → Shortness of Breath)
4. JSON output enables system integration (EHR, databases)
5. The same note can be structured differently based on your prompt design
""")


if __name__ == "__main__":
    main()
