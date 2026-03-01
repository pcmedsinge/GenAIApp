"""
Exercise 2: Reflection / Self-Critique Agent — Clinical Note Quality

Skills practiced:
- Generate → Critique → Revise loop (the core reflection pattern)
- Structured quality scoring with multiple criteria
- Iterative improvement until a quality threshold is met
- Comparing single-pass vs reflection output quality
- Understanding when reflection is worth the extra cost

Key insight: A single LLM call often produces "good enough" output.
  But for HIGH-STAKES content (clinical notes, legal docs, reports),
  you want the agent to REVIEW ITS OWN WORK and improve it.

  This is like a physician writing a note, then the attending
  reviewing it, then the physician revising. Except the same
  LLM plays both roles.

Architecture:
  ┌──────────────────┐
  │    GENERATOR      │ ◄── Writes first draft (clinical note)
  └────────┬─────────┘
           │
  ┌────────▼─────────┐
  │    CRITIC         │ ◄── Evaluates against quality criteria
  │  Score: 6/10      │     Returns: score, strengths, weaknesses
  │  "Missing diff dx"│
  └────────┬─────────┘
           │
           │ Score < threshold?
           │ YES ──────────────────┐
           │                       │
  ┌────────▼─────────┐   ┌───────▼────────┐
  │    REVISER        │   │   ACCEPT       │ ◄── Score ≥ threshold
  │  (addresses gaps) │   │   (done!)      │
  └────────┬─────────┘   └────────────────┘
           │
           └──── back to CRITIC ────┘  (loop until good or max revisions)

  Key parameters:
  - quality_threshold: 8/10 (below = revise, above = accept)
  - max_revisions: 3 (prevent infinite loops)
  - criteria: completeness, accuracy, differential, plan specificity, safety
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

client = OpenAI()


# ============================================================
# Quality Criteria (what makes a good clinical note?)
# ============================================================

QUALITY_CRITERIA = {
    "completeness": {
        "description": "All required sections present: CC, HPI, Vitals, Labs, Assessment, Plan",
        "weight": 2,  # Most important
    },
    "clinical_accuracy": {
        "description": "Lab values interpreted correctly, vitals addressed, medications appropriate",
        "weight": 2,
    },
    "differential_diagnosis": {
        "description": "Appropriate breadth of differential (not just one diagnosis), with reasoning",
        "weight": 1.5,
    },
    "plan_specificity": {
        "description": "Plan has specific, actionable, time-bound orders (not vague recommendations)",
        "weight": 1.5,
    },
    "safety_check": {
        "description": "Drug interactions checked, contraindications noted, allergies considered",
        "weight": 2,
    },
    "documentation_quality": {
        "description": "Clear language, proper medical terminology, logical flow",
        "weight": 1,
    },
}


# ============================================================
# Reflection Agent
# ============================================================

class ReflectionAgent:
    """
    An agent that generates clinical notes, critiques them,
    and revises until quality standards are met.

    The SAME LLM plays 3 roles:
    1. GENERATOR — writes the initial draft
    2. CRITIC — evaluates it against quality criteria
    3. REVISER — improves it based on the critique

    Different system prompts give the LLM different "personas."
    """

    def __init__(self, model: str = "gpt-4o-mini", quality_threshold: int = 8, max_revisions: int = 3):
        self.model = model
        self.quality_threshold = quality_threshold
        self.max_revisions = max_revisions
        self.revision_history = []  # Track every draft and critique

    def generate(self, scenario: str, note_type: str = "clinical assessment") -> str:
        """Phase 1: Generate the initial draft."""
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    f"You are an emergency medicine physician. Write a {note_type} note "
                    f"for the following patient. Include ALL of these sections:\n"
                    f"1. Chief Complaint (CC)\n"
                    f"2. History of Present Illness (HPI)\n"
                    f"3. Past Medical History (PMH)\n"
                    f"4. Current Medications\n"
                    f"5. Vital Signs (with interpretation)\n"
                    f"6. Lab Results (with interpretation — normal/abnormal/critical)\n"
                    f"7. Assessment (clinical impression)\n"
                    f"8. Differential Diagnosis (at least 3 possibilities with reasoning)\n"
                    f"9. Plan (specific, time-bound orders)\n"
                    f"10. Safety Considerations (interactions, contraindications, allergies)\n\n"
                    f"Be thorough and specific. Use medical terminology appropriately."
                )},
                {"role": "user", "content": scenario},
            ],
            temperature=0.3,  # Slight creativity for first draft
        )
        return response.choices[0].message.content

    def critique(self, scenario: str, draft: str) -> dict:
        """Phase 2: Critique the draft against quality criteria."""
        criteria_text = "\n".join([
            f"- {name} (weight {c['weight']}x): {c['description']}"
            for name, c in QUALITY_CRITERIA.items()
        ])

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are a senior attending physician reviewing a clinical note. "
                    "You are STRICT but FAIR. Evaluate the note against these criteria:\n\n"
                    f"{criteria_text}\n\n"
                    "For each criterion, provide:\n"
                    "- A score from 1-10\n"
                    "- A brief explanation\n\n"
                    "Also provide:\n"
                    "- An overall score (1-10, weighted average)\n"
                    "- Top 3 strengths\n"
                    "- Top 3 weaknesses (specific and actionable)\n"
                    "- Specific suggestions for improvement\n\n"
                    "Output as JSON:\n"
                    "{\n"
                    '  "criteria_scores": {"completeness": {"score": 8, "explanation": "..."}, ...},\n'
                    '  "overall_score": 7,\n'
                    '  "strengths": ["...", "...", "..."],\n'
                    '  "weaknesses": ["...", "...", "..."],\n'
                    '  "suggestions": ["...", "...", "..."]\n'
                    "}"
                )},
                {"role": "user", "content": f"Patient scenario:\n{scenario}\n\nClinical note to review:\n{draft}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {
                "overall_score": 5,
                "strengths": ["Could not parse critique"],
                "weaknesses": ["Critique parsing failed"],
                "suggestions": ["Re-evaluate"],
                "criteria_scores": {},
            }

    def revise(self, scenario: str, draft: str, critique: dict) -> str:
        """Phase 3: Revise the draft to address the critique."""
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are an emergency medicine physician revising your clinical note. "
                    "Address ALL weaknesses and incorporate ALL suggestions from the review. "
                    "Keep everything that was good. Improve everything that was flagged.\n\n"
                    "IMPORTANT:\n"
                    "- Do NOT just acknowledge the feedback — actually FIX the issues\n"
                    "- Add missing sections/details\n"
                    "- Make the plan more specific if it was vague\n"
                    "- Add safety checks if they were missing\n"
                    "- Expand the differential if it was too narrow\n"
                    "- Output the COMPLETE revised note, not just the changes"
                )},
                {"role": "user", "content": (
                    f"Patient scenario:\n{scenario}\n\n"
                    f"Your original note:\n{draft}\n\n"
                    f"Attending's review:\n"
                    f"Score: {critique.get('overall_score', '?')}/10\n"
                    f"Weaknesses: {json.dumps(critique.get('weaknesses', []))}\n"
                    f"Suggestions: {json.dumps(critique.get('suggestions', []))}\n\n"
                    f"Please revise the note to address all feedback."
                )},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content

    def run(self, scenario: str, note_type: str = "clinical assessment", verbose: bool = True) -> dict:
        """
        Run the full reflection loop:
        Generate → (Critique → Revise) × N → Final
        """
        if verbose:
            print("\n  ╔══════════════════════════════════════╗")
            print("  ║  REFLECTION / SELF-CRITIQUE AGENT    ║")
            print("  ╚══════════════════════════════════════╝")

        # Phase 1: Generate initial draft
        if verbose:
            print(f"\n  📝 Phase 1: GENERATING initial {note_type}...")
        draft = self.generate(scenario, note_type)
        self.revision_history.append({"version": 0, "type": "draft", "content": draft})

        if verbose:
            print(f"    Draft generated ({len(draft)} chars)")

        current_draft = draft

        # Reflection loop: Critique → Revise → Critique → ...
        for revision_num in range(self.max_revisions):
            # Critique
            if verbose:
                print(f"\n  🔍 Phase 2.{revision_num + 1}: CRITIQUING (revision {revision_num + 1}/{self.max_revisions})...")

            critique = self.critique(scenario, current_draft)
            score = critique.get("overall_score", 0)
            self.revision_history.append({"version": revision_num + 1, "type": "critique", "content": critique})

            if verbose:
                print(f"    Overall Score: {score}/10 (threshold: {self.quality_threshold})")
                criteria_scores = critique.get("criteria_scores", {})
                for name, details in criteria_scores.items():
                    if isinstance(details, dict):
                        print(f"      {name}: {details.get('score', '?')}/10 — {details.get('explanation', '')[:60]}")
                print(f"    Strengths: {', '.join(critique.get('strengths', [])[:2])}")
                print(f"    Weaknesses: {', '.join(critique.get('weaknesses', [])[:2])}")

            # Check if good enough
            if score >= self.quality_threshold:
                if verbose:
                    print(f"\n  ✅ Score {score} ≥ threshold {self.quality_threshold}. ACCEPTING note.")
                break

            # Revise
            if verbose:
                print(f"\n  ✏️ Phase 3.{revision_num + 1}: REVISING to address {len(critique.get('weaknesses', []))} weaknesses...")

            current_draft = self.revise(scenario, current_draft, critique)
            self.revision_history.append({"version": revision_num + 1, "type": "revision", "content": current_draft})

            if verbose:
                print(f"    Revised ({len(current_draft)} chars)")
        else:
            if verbose:
                print(f"\n  ⚠️ Max revisions ({self.max_revisions}) reached. Using best version.")

        if verbose:
            print(f"\n  {'═' * 60}")
            print("  FINAL NOTE:")
            print(f"  {'═' * 60}")
            for line in current_draft.split("\n"):
                print(f"  {line}")

        return {
            "final_note": current_draft,
            "revisions": len(self.revision_history),
            "final_score": score,
            "history": self.revision_history,
        }


# ============================================================
# Demo Scenarios
# ============================================================

SCENARIO_ACS = """
Patient: 55-year-old male
Chief Complaint: Chest pain for 2 hours, radiating to left arm
History: Type 2 diabetes (10 years), hypertension, hyperlipidemia
Current Medications: Metformin 1000mg BID, Lisinopril 20mg daily, Atorvastatin 40mg daily
Vitals: BP 158/92, HR 98, SpO2 96%, Temp 37.1°C
Labs: Troponin I 0.45 ng/mL (elevated), Glucose 210 mg/dL, Creatinine 1.3, K+ 4.8
ECG: ST depression in leads V3-V6
"""

SCENARIO_SEPSIS = """
Patient: 72-year-old female from nursing home
Chief Complaint: Altered mental status and fever for 6 hours
History: COPD, CHF (EF 35%), chronic kidney disease stage 3, dementia
Current Medications: Furosemide 40mg, Carvedilol 12.5mg BID, Donepezil 10mg
Vitals: BP 88/52, HR 110, RR 24, SpO2 91% (on 2L NC), Temp 39.2°C
Labs: WBC 18.5, Lactate 4.2, Creatinine 2.8 (baseline 1.5), Procalcitonin 8.5
Urine: Cloudy, positive nitrites, >100 WBCs
CXR: Left lower lobe infiltrate
"""


# ============================================================
# Demo Functions
# ============================================================

def demo_basic_reflection():
    """Basic reflection loop — generate, critique, revise."""
    print("\n" + "=" * 70)
    print("  DEMO 1: BASIC REFLECTION — GENERATING A CLINICAL NOTE")
    print("=" * 70)
    print("""
  Watch the agent:
  1. Write a clinical note (first draft)
  2. Critique its own work (quality scoring)
  3. Revise to address weaknesses
  4. Repeat until score ≥ 8/10
  """)

    agent = ReflectionAgent(quality_threshold=8, max_revisions=3)
    result = agent.run(SCENARIO_ACS)

    print(f"\n  Summary:")
    print(f"    Revisions: {result['revisions']} versions")
    print(f"    Final score: {result['final_score']}/10")
    print(f"    Note length: {len(result['final_note'])} chars")


def demo_single_vs_reflection():
    """Compare single-pass output vs reflection output."""
    print("\n" + "=" * 70)
    print("  DEMO 2: SINGLE-PASS vs REFLECTION — QUALITY COMPARISON")
    print("=" * 70)
    print("""
  Is reflection worth the extra LLM calls?
  We'll generate the same note TWO ways:
  1. Single pass (one LLM call)
  2. Reflection (generate → critique → revise loop)
  Then compare quality scores.
  """)

    # Single pass
    print("\n  ═══ SINGLE PASS (1 LLM call) ═══")
    single_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an EM physician. Write a complete clinical assessment note with all standard sections."},
            {"role": "user", "content": SCENARIO_SEPSIS},
        ],
        temperature=0.3,
    )
    single_note = single_response.choices[0].message.content
    print(f"  Generated {len(single_note)} chars in 1 call")

    # Score the single pass
    agent = ReflectionAgent(quality_threshold=8, max_revisions=2)
    single_critique = agent.critique(SCENARIO_SEPSIS, single_note)
    single_score = single_critique.get("overall_score", 0)
    print(f"  Score: {single_score}/10")
    print(f"  Weaknesses: {', '.join(single_critique.get('weaknesses', [])[:3])}")

    # Reflection
    print(f"\n  ═══ REFLECTION (generate → critique → revise) ═══")
    agent2 = ReflectionAgent(quality_threshold=8, max_revisions=2)
    result = agent2.run(SCENARIO_SEPSIS, verbose=False)
    reflection_score = result["final_score"]
    print(f"  Generated in {result['revisions']} version(s)")
    print(f"  Score: {reflection_score}/10")

    # Final critique of the reflected version for fair comparison
    reflected_critique = agent2.critique(SCENARIO_SEPSIS, result["final_note"])
    print(f"  Final critique weaknesses: {', '.join(reflected_critique.get('weaknesses', [])[:3])}")

    # Comparison
    print(f"\n  ═══ COMPARISON ═══")
    print(f"  Single Pass:  Score {single_score}/10  ({len(single_note)} chars, 1 LLM call)")
    print(f"  Reflection:   Score {reflection_score}/10  ({len(result['final_note'])} chars, ~{1 + result['revisions'] * 2} LLM calls)")
    improvement = reflection_score - single_score
    print(f"  Improvement:  {'+' if improvement >= 0 else ''}{improvement} points")

    if improvement > 0:
        print(f"\n  ✓ Reflection improved quality by {improvement} points.")
        print(f"    Worth it for: clinical reports, discharge summaries, referral letters")
    else:
        print(f"\n  ≈ Similar quality. Single pass was already good enough.")
        print(f"    Reflection adds more value when the task is harder or the first draft is weak.")


def demo_criteria_deep_dive():
    """Show detailed per-criteria scoring across revisions."""
    print("\n" + "=" * 70)
    print("  DEMO 3: CRITERIA DEEP DIVE — WATCH SCORES IMPROVE")
    print("=" * 70)
    print("""
  Watch how individual quality criteria improve across revisions.
  The agent specifically targets the LOWEST-scoring criteria each round.
  """)

    agent = ReflectionAgent(quality_threshold=9, max_revisions=3)  # High bar!

    # Generate first draft
    print("\n  Generating first draft...")
    draft = agent.generate(SCENARIO_SEPSIS)

    current = draft
    scores_over_time = []

    for revision in range(3):
        critique = agent.critique(SCENARIO_SEPSIS, current)
        score = critique.get("overall_score", 0)
        criteria_scores = critique.get("criteria_scores", {})

        scores_this_round = {"overall": score}
        for name, details in criteria_scores.items():
            if isinstance(details, dict):
                scores_this_round[name] = details.get("score", 0)
        scores_over_time.append(scores_this_round)

        print(f"\n  Revision {revision} — Overall: {score}/10")
        for name in QUALITY_CRITERIA:
            s = scores_this_round.get(name, "?")
            bar = "█" * (int(s) if isinstance(s, (int, float)) else 0) + "░" * (10 - (int(s) if isinstance(s, (int, float)) else 0))
            print(f"    {name:<25} {bar} {s}/10")

        if score >= 9:
            print(f"\n  ✅ Score {score} ≥ 9. Accepted!")
            break

        # Revise
        print(f"\n  Revising to address: {', '.join(critique.get('weaknesses', [])[:2])}")
        current = agent.revise(SCENARIO_SEPSIS, current, critique)

    # Show improvement trajectory
    if len(scores_over_time) > 1:
        print(f"\n  ═══ IMPROVEMENT TRAJECTORY ═══")
        print(f"  {'Criterion':<25} {'Round 1':>10} {'Round 2':>10} {'Change':>10}")
        print(f"  {'─' * 55}")
        for criterion in list(QUALITY_CRITERIA.keys()) + ["overall"]:
            r1 = scores_over_time[0].get(criterion, "?")
            r2 = scores_over_time[-1].get(criterion, "?")
            if isinstance(r1, (int, float)) and isinstance(r2, (int, float)):
                change = r2 - r1
                arrow = "↑" if change > 0 else ("↓" if change < 0 else "→")
                print(f"  {criterion:<25} {r1:>10} {r2:>10} {arrow}{abs(change):>8}")
            else:
                print(f"  {criterion:<25} {str(r1):>10} {str(r2):>10}")


def demo_different_note_types():
    """Generate different types of clinical notes using reflection."""
    print("\n" + "=" * 70)
    print("  DEMO 4: DIFFERENT NOTE TYPES WITH REFLECTION")
    print("=" * 70)
    print("""
  The reflection pattern works for ANY type of clinical document:
  - Assessment notes
  - Discharge summaries
  - Referral letters
  - Patient education handouts

  Watch how the GENERATOR and CRITIC adapt to each type.
  """)

    note_types = [
        ("clinical assessment", "Full ED assessment note"),
        ("discharge summary", "Discharge summary for patient going home"),
        ("referral letter", "Referral letter to cardiology"),
    ]

    for note_type, description in note_types:
        print(f"\n  ─── {description.upper()} ───")

        agent = ReflectionAgent(quality_threshold=7, max_revisions=1)  # Quick — 1 revision max
        draft = agent.generate(SCENARIO_ACS, note_type)
        critique = agent.critique(SCENARIO_ACS, draft)
        score = critique.get("overall_score", 0)

        print(f"  Score: {score}/10")
        print(f"  Length: {len(draft)} chars")
        print(f"  First 150 chars: {draft[:150].replace(chr(10), ' ')}...")
        print(f"  Top weakness: {(critique.get('weaknesses', ['N/A'])[:1] or ['N/A'])[0]}")


def demo_interactive():
    """Interactive reflection agent."""
    print("\n" + "=" * 70)
    print("  DEMO 5: INTERACTIVE REFLECTION AGENT")
    print("=" * 70)
    print("""
  Enter a patient scenario. The agent will generate a note,
  critique it, and revise it until quality ≥ 8/10.
  Type 'quit' to exit.
  """)

    while True:
        print("\n  Enter patient scenario (or 'quit'):")
        scenario = input("  > ").strip()

        if scenario.lower() in ['quit', 'exit', 'q']:
            break

        if len(scenario) < 20:
            print("  Please provide more detail.")
            continue

        threshold = input("  Quality threshold (1-10, default 8): ").strip()
        try:
            threshold = int(threshold) if threshold else 8
        except ValueError:
            threshold = 8

        agent = ReflectionAgent(quality_threshold=threshold, max_revisions=3)
        result = agent.run(scenario)

        print(f"\n  Revisions: {result['revisions']}")
        print(f"  Final score: {result['final_score']}/10")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 2: REFLECTION / SELF-CRITIQUE AGENT")
    print("=" * 70)
    print("""
    The Reflection pattern: Generate → Critique → Revise → Repeat.
    The agent reviews its own work and improves it iteratively.

    Choose a demo:
      1 → Basic reflection (ACS clinical note)
      2 → Single-pass vs Reflection (quality comparison)
      3 → Criteria deep dive (watch scores improve)
      4 → Different note types (assessment, discharge, referral)
      5 → Interactive (enter your own scenario)
      6 → Run demos 1-4
    """)

    choice = input("  Enter choice (1-6): ").strip()

    if choice == "1":
        demo_basic_reflection()
    elif choice == "2":
        demo_single_vs_reflection()
    elif choice == "3":
        demo_criteria_deep_dive()
    elif choice == "4":
        demo_different_note_types()
    elif choice == "5":
        demo_interactive()
    elif choice == "6":
        demo_basic_reflection()
        demo_single_vs_reflection()
        demo_criteria_deep_dive()
        demo_different_note_types()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. REFLECTION IS SIMPLE: Generate → Critique → Revise. That's it.
   The power comes from the LLM reviewing its own output with
   a different "lens" (strict attending vs writing resident).

2. QUALITY THRESHOLD: You control when the loop stops.
   Set threshold=7 for quick results, threshold=9 for polished output.
   Higher thresholds = more revisions = more token cost.

3. STRUCTURED CRITERIA: Don't just say "review this." Give the critic
   specific criteria with weights. This makes feedback actionable.

4. WHEN TO USE REFLECTION:
   ✓ Clinical notes, discharge summaries, referral letters
   ✓ Patient education materials (reading level check)
   ✓ Any document that will be reviewed by a human
   ✗ Quick lookups ("what's the dose of metformin?")
   ✗ Real-time chat (too slow for interactive use)

5. COST vs QUALITY TRADEOFF:
   - 1 LLM call: Fast, cheap, "good enough" for most tasks
   - 3-5 LLM calls (reflection): Slower, 3-5x cost, measurably better
   - For production: Use reflection ONLY for high-stakes outputs

6. PERSONA SEPARATION: The critic and generator are the SAME LLM
   with different system prompts. The prompts create distinct personas:
   - Generator = "eager resident writing the note"  
   - Critic = "strict attending reviewing the note"
   - Reviser = "resident fixing issues the attending found"
"""

if __name__ == "__main__":
    main()
