"""
Exercise 3: Parallel Fan-Out / Map-Reduce Agent — Multi-Specialist Consult

Skills practiced:
- Fan-out pattern: send the SAME question to multiple specialists
- Parallel execution using ThreadPoolExecutor
- Map-Reduce: map (specialists analyze) → reduce (synthesize consensus)
- Handling conflicting opinions from different specialists
- Understanding when parallel execution beats sequential

Key insight: In medicine, complex cases get "second opinions" from
  multiple specialists. The Parallel Fan-Out pattern does this
  programmatically — asking 3-5 specialist agents simultaneously,
  then synthesizing their answers into a consensus.

  Sequential pipeline:  Agent A → Agent B → Agent C (slow, each waits)
  Parallel fan-out:     Agent A ┐
                        Agent B ├→ REDUCE → Consensus
                        Agent C ┘           (fast, all run at once)

Architecture:
  ┌────────────────────────────────────────────────────────┐
  │                    COORDINATOR                          │
  │  "This patient needs cardiology, endocrine, and        │
  │   nephrology input. Let me ask all three at once."     │
  └───────────────┬────────────────────────────────────────┘
                  │
     ┌────────────┼─────────────┐       ← FAN-OUT (parallel)
     │            │             │
     ▼            ▼             ▼
  ┌──────┐   ┌──────┐   ┌──────────┐
  │Cardio│   │Endo  │   │Nephro    │   ← MAP (each specialist
  │Agent │   │Agent │   │Agent     │      analyzes independently)
  └──┬───┘   └──┬───┘   └────┬─────┘
     │           │            │
     └───────────┼────────────┘       ← FAN-IN (collect results)
                 │
     ┌───────────▼────────────┐
     │       REDUCER           │       ← REDUCE (synthesize
     │  (Synthesize consensus, │         into one recommendation)
     │   flag disagreements)   │
     └─────────────────────────┘
"""

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

client = OpenAI()


# ============================================================
# Specialist Agent Definitions
# ============================================================
# Each specialist has a focused system prompt and domain expertise.
# They ALL see the same patient data but analyze it through
# their own specialist lens.
# ============================================================

SPECIALISTS = {
    "cardiology": {
        "title": "Cardiologist",
        "system_prompt": (
            "You are a board-certified cardiologist. Analyze this patient from a "
            "CARDIAC perspective. Focus on:\n"
            "- ECG interpretation and significance\n"
            "- Troponin levels and ACS risk stratification (TIMI/HEART score)\n"
            "- Cardiac medication management (anticoagulation, antiplatelets)\n"
            "- Need for cardiac catheterization or stress testing\n"
            "- Hemodynamic stability assessment\n\n"
            "Provide your assessment in this format:\n"
            "CARDIAC ASSESSMENT: (2-3 sentences)\n"
            "RISK LEVEL: (low/moderate/high/critical)\n"
            "RECOMMENDATIONS: (numbered list of specific actions)\n"
            "URGENCY: (immediate/within hours/within 24h/outpatient)"
        ),
    },
    "endocrinology": {
        "title": "Endocrinologist",
        "system_prompt": (
            "You are a board-certified endocrinologist. Analyze this patient from a "
            "METABOLIC/ENDOCRINE perspective. Focus on:\n"
            "- Glucose management (current control, inpatient management)\n"
            "- Medication adjustments (metformin, insulin needs)\n"
            "- DKA/HHS risk assessment\n"
            "- Long-term diabetes optimization\n"
            "- Thyroid, adrenal considerations if relevant\n\n"
            "Provide your assessment in this format:\n"
            "ENDOCRINE ASSESSMENT: (2-3 sentences)\n"
            "GLUCOSE MANAGEMENT: (specific insulin orders if needed)\n"
            "MEDICATION CHANGES: (what to hold/add/adjust)\n"
            "MONITORING: (what to track and how often)"
        ),
    },
    "nephrology": {
        "title": "Nephrologist",
        "system_prompt": (
            "You are a board-certified nephrologist. Analyze this patient from a "
            "RENAL perspective. Focus on:\n"
            "- Kidney function (eGFR estimation, AKI vs CKD)\n"
            "- Medication dosing adjustments for renal function\n"
            "- Contrast nephropathy risk if cardiac cath planned\n"
            "- Electrolyte management (potassium, bicarb)\n"
            "- Fluid management\n\n"
            "Provide your assessment in this format:\n"
            "RENAL ASSESSMENT: (2-3 sentences)\n"
            "eGFR ESTIMATE: (calculated value and stage)\n"
            "MEDICATION ALERTS: (drugs needing dose adjustment)\n"
            "RENAL PROTECTION: (specific measures)"
        ),
    },
    "emergency_medicine": {
        "title": "Emergency Medicine Physician",
        "system_prompt": (
            "You are a board-certified emergency medicine physician. Analyze this "
            "patient from an ED MANAGEMENT perspective. Focus on:\n"
            "- Immediate life threats (ABC assessment)\n"
            "- Disposition decision (admit vs discharge)\n"
            "- Level of care needed (ICU, stepdown, floor, observation)\n"
            "- Time-sensitive interventions\n"
            "- Differential diagnosis breadth\n\n"
            "Provide your assessment in this format:\n"
            "ED ASSESSMENT: (2-3 sentences)\n"
            "ACUITY: (ESI level 1-5)\n"
            "DISPOSITION: (ICU/stepdown/floor/observation/discharge)\n"
            "IMMEDIATE ACTIONS: (what to do in the next 30 minutes)"
        ),
    },
    "pharmacy": {
        "title": "Clinical Pharmacist",
        "system_prompt": (
            "You are a clinical pharmacist with critical care expertise. Analyze this "
            "patient's MEDICATION profile. Focus on:\n"
            "- Drug-drug interactions\n"
            "- Renal/hepatic dose adjustments needed\n"
            "- Contraindications given current labs and conditions\n"
            "- Medication reconciliation (home meds vs acute treatment)\n"
            "- High-alert medications requiring monitoring\n\n"
            "Provide your assessment in this format:\n"
            "MEDICATION REVIEW: (2-3 sentences)\n"
            "INTERACTIONS/ALERTS: (list any concerns)\n"
            "DOSE ADJUSTMENTS: (specific changes needed)\n"
            "MONITORING: (what to watch for each high-risk med)"
        ),
    },
}


# ============================================================
# Parallel Fan-Out Agent
# ============================================================

class ParallelFanOutAgent:
    """
    Sends the same patient scenario to multiple specialist agents
    simultaneously, then synthesizes a consensus recommendation.

    Map-Reduce pattern:
    - MAP: Each specialist analyzes the case independently (parallel)
    - REDUCE: A synthesizer combines all opinions into consensus
    """

    def __init__(self, model: str = "gpt-4o-mini", specialists: list[str] = None):
        self.model = model
        self.specialist_names = specialists or ["cardiology", "endocrinology", "nephrology", "emergency_medicine"]
        self.results = {}
        self.timing = {}

    def _consult_specialist(self, specialty: str, scenario: str) -> dict:
        """Call a single specialist (runs in parallel)."""
        spec = SPECIALISTS[specialty]
        start = time.time()

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": spec["system_prompt"]},
                {"role": "user", "content": f"Please analyze this patient:\n{scenario}"},
            ],
            temperature=0,
        )

        elapsed = time.time() - start

        return {
            "specialty": specialty,
            "title": spec["title"],
            "opinion": response.choices[0].message.content,
            "elapsed_seconds": round(elapsed, 2),
            "tokens": response.usage.total_tokens if response.usage else 0,
        }

    def fan_out(self, scenario: str, parallel: bool = True) -> dict[str, dict]:
        """
        Phase 1: MAP — Send scenario to all specialists.
        Can run in parallel (ThreadPoolExecutor) or sequentially.
        """
        self.results = {}
        self.timing = {"start": time.time()}

        if parallel:
            # PARALLEL execution — all specialists at once
            with ThreadPoolExecutor(max_workers=len(self.specialist_names)) as executor:
                futures = {
                    executor.submit(self._consult_specialist, spec, scenario): spec
                    for spec in self.specialist_names
                }
                for future in as_completed(futures):
                    result = future.result()
                    self.results[result["specialty"]] = result
        else:
            # SEQUENTIAL execution — one at a time (for comparison)
            for spec in self.specialist_names:
                result = self._consult_specialist(spec, scenario)
                self.results[result["specialty"]] = result

        self.timing["fan_out_elapsed"] = round(time.time() - self.timing["start"], 2)
        return self.results

    def reduce(self, scenario: str) -> dict:
        """
        Phase 2: REDUCE — Synthesize all specialist opinions into consensus.
        Identifies agreements, disagreements, and creates a unified plan.
        """
        # Format all opinions for the synthesizer
        opinions_text = ""
        for spec, result in self.results.items():
            opinions_text += f"\n--- {result['title']} ({spec}) ---\n{result['opinion']}\n"

        reduce_start = time.time()

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are the attending physician synthesizing multiple specialist opinions "
                    "into a unified treatment plan. Your job:\n\n"
                    "1. CONSENSUS: What do ALL specialists agree on? (these are high-confidence actions)\n"
                    "2. DISAGREEMENTS: Where do specialists differ? (these need discussion)\n"
                    "3. PRIORITY ORDER: Rank actions by urgency (what first, what can wait)\n"
                    "4. UNIFIED PLAN: Synthesize into one coherent treatment plan\n"
                    "5. RISK SUMMARY: Overall risk level with justification\n\n"
                    "Be specific. Don't just list — integrate and prioritize."
                )},
                {"role": "user", "content": (
                    f"Patient scenario:\n{scenario}\n\n"
                    f"Specialist opinions:{opinions_text}\n\n"
                    f"Please synthesize these opinions into a unified assessment and plan."
                )},
            ],
            temperature=0,
        )

        reduce_elapsed = round(time.time() - reduce_start, 2)

        return {
            "consensus": response.choices[0].message.content,
            "reduce_elapsed": reduce_elapsed,
            "total_specialists": len(self.results),
            "total_tokens": sum(r["tokens"] for r in self.results.values()) + (response.usage.total_tokens if response.usage else 0),
        }

    def run(self, scenario: str, parallel: bool = True, verbose: bool = True) -> dict:
        """Run the full fan-out → reduce pipeline."""
        if verbose:
            print("\n  ╔══════════════════════════════════════╗")
            print("  ║  PARALLEL FAN-OUT / MAP-REDUCE AGENT ║")
            print("  ╚══════════════════════════════════════╝")

        # Phase 1: Fan-out
        mode = "PARALLEL" if parallel else "SEQUENTIAL"
        if verbose:
            print(f"\n  📡 Phase 1: FAN-OUT to {len(self.specialist_names)} specialists ({mode})...")

        results = self.fan_out(scenario, parallel=parallel)

        if verbose:
            for spec, result in results.items():
                print(f"\n    ─── {result['title']} ({result['elapsed_seconds']}s, {result['tokens']} tokens) ───")
                # Show first 200 chars of opinion
                preview = result["opinion"][:200].replace("\n", "\n    ")
                print(f"    {preview}...")
            print(f"\n    Fan-out completed in {self.timing['fan_out_elapsed']}s")

        # Phase 2: Reduce
        if verbose:
            print(f"\n  🔄 Phase 2: REDUCE — Synthesizing consensus...")

        consensus = self.reduce(scenario)

        if verbose:
            print(f"\n  {'═' * 60}")
            print("  CONSENSUS RECOMMENDATION:")
            print(f"  {'═' * 60}")
            for line in consensus["consensus"].split("\n"):
                print(f"  {line}")
            print(f"\n  Stats: {consensus['total_specialists']} specialists, {consensus['total_tokens']} total tokens")
            print(f"  Fan-out time: {self.timing['fan_out_elapsed']}s | Reduce time: {consensus['reduce_elapsed']}s")

        return {
            "specialist_opinions": results,
            "consensus": consensus,
            "timing": self.timing,
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
""".strip()

SCENARIO_COMPLEX = """
Patient: 68-year-old female
Chief Complaint: Shortness of breath and leg swelling for 5 days
History: Heart failure (EF 30%), Type 2 diabetes, COPD, CKD stage 3, atrial fibrillation
Current Medications: Furosemide 80mg BID, Carvedilol 25mg BID, Apixaban 5mg BID,
  Metformin 500mg BID, Lisinopril 10mg daily, Albuterol PRN
Vitals: BP 142/88, HR 92 (irregular), RR 24, SpO2 89% (room air), Temp 36.8°C
Labs: BNP 1850, Troponin 0.08, Creatinine 2.4 (baseline 1.8), K+ 5.6,
  Glucose 195, Hemoglobin 10.2
CXR: Bilateral pleural effusions, cardiomegaly
ECG: Atrial fibrillation, no acute ST changes
""".strip()


# ============================================================
# Demo Functions
# ============================================================

def demo_basic_fanout():
    """Basic fan-out with 4 specialists."""
    print("\n" + "=" * 70)
    print("  DEMO 1: BASIC FAN-OUT — 4 SPECIALIST CONSULT")
    print("=" * 70)
    print("""
  Send the ACS patient to 4 specialists simultaneously:
  Cardiology, Endocrinology, Nephrology, Emergency Medicine.
  Then synthesize their opinions into a unified plan.
  """)

    agent = ParallelFanOutAgent(specialists=["cardiology", "endocrinology", "nephrology", "emergency_medicine"])
    agent.run(SCENARIO_ACS)


def demo_parallel_vs_sequential():
    """Compare parallel vs sequential execution timing."""
    print("\n" + "=" * 70)
    print("  DEMO 2: PARALLEL vs SEQUENTIAL — TIMING COMPARISON")
    print("=" * 70)
    print("""
  Same 4 specialists, same question.
  First SEQUENTIAL (one at a time), then PARALLEL (all at once).
  Watch the time difference!
  """)

    # Sequential
    print("\n  ═══ SEQUENTIAL (one at a time) ═══")
    seq_agent = ParallelFanOutAgent(specialists=["cardiology", "endocrinology", "nephrology"])
    seq_start = time.time()
    seq_agent.fan_out(SCENARIO_ACS, parallel=False)
    seq_time = round(time.time() - seq_start, 2)

    for spec, result in seq_agent.results.items():
        print(f"    {result['title']}: {result['elapsed_seconds']}s")
    print(f"    TOTAL: {seq_time}s (sum of all specialist times)")

    # Parallel
    print(f"\n  ═══ PARALLEL (all at once) ═══")
    par_agent = ParallelFanOutAgent(specialists=["cardiology", "endocrinology", "nephrology"])
    par_start = time.time()
    par_agent.fan_out(SCENARIO_ACS, parallel=True)
    par_time = round(time.time() - par_start, 2)

    for spec, result in par_agent.results.items():
        print(f"    {result['title']}: {result['elapsed_seconds']}s")
    print(f"    TOTAL: {par_time}s (limited by slowest specialist)")

    # Comparison
    speedup = round(seq_time / par_time, 1) if par_time > 0 else 0
    print(f"\n  ═══ COMPARISON ═══")
    print(f"    Sequential: {seq_time}s")
    print(f"    Parallel:   {par_time}s")
    print(f"    Speedup:    {speedup}x faster")
    print(f"\n    Parallel execution is ~{speedup}x faster because specialists")
    print(f"    run simultaneously. Cost is the SAME (same total tokens).")


def demo_five_specialists():
    """Fan-out to 5 specialists for a complex case."""
    print("\n" + "=" * 70)
    print("  DEMO 3: 5-SPECIALIST CONSULT — COMPLEX PATIENT")
    print("=" * 70)
    print("""
  A complex multi-morbidity patient requiring 5 perspectives:
  Cardiology, Endocrinology, Nephrology, EM, and Pharmacy.
  
  This is the real power of fan-out — getting comprehensive,
  multi-perspective analysis in the time of a single LLM call.
  """)

    agent = ParallelFanOutAgent(
        specialists=["cardiology", "endocrinology", "nephrology", "emergency_medicine", "pharmacy"]
    )
    agent.run(SCENARIO_COMPLEX)


def demo_conflict_detection():
    """Show how the reducer handles conflicting specialist opinions."""
    print("\n" + "=" * 70)
    print("  DEMO 4: CONFLICT DETECTION — WHEN SPECIALISTS DISAGREE")
    print("=" * 70)
    print("""
  Sometimes specialists give conflicting advice.
  Example: Cardiologist wants heparin, Nephrologist is concerned
  about bleeding risk with low kidney function.
  
  Watch the reducer identify and resolve these conflicts.
  """)

    agent = ParallelFanOutAgent(specialists=["cardiology", "nephrology", "pharmacy"])
    results = agent.fan_out(SCENARIO_ACS, parallel=True)

    # Show each specialist's key recommendation
    print("\n  Key recommendations by specialist:")
    for spec, result in results.items():
        print(f"\n    {result['title']}:")
        # Extract first 3 lines of recommendations
        for line in result["opinion"].split("\n"):
            if any(keyword in line.upper() for keyword in ["RECOMMEND", "ACTION", "ALERT", "ADJUST", "PROTECTION"]):
                print(f"      {line.strip()[:100]}")

    # Now reduce — the reducer should flag conflicts
    print("\n  Synthesizing with conflict detection...")
    consensus = agent.reduce(SCENARIO_ACS)

    print(f"\n  {'═' * 60}")
    print("  CONSENSUS WITH CONFLICT RESOLUTION:")
    print(f"  {'═' * 60}")
    for line in consensus["consensus"].split("\n"):
        print(f"  {line}")


def demo_interactive():
    """Interactive fan-out agent."""
    print("\n" + "=" * 70)
    print("  DEMO 5: INTERACTIVE MULTI-SPECIALIST CONSULT")
    print("=" * 70)

    available = list(SPECIALISTS.keys())
    print(f"  Available specialists: {', '.join(available)}")
    print(f"  Enter a patient scenario, choose specialists, get consensus.")
    print(f"  Type 'quit' to exit.\n")

    while True:
        scenario = input("  Patient scenario (or 'quit'): ").strip()
        if scenario.lower() in ['quit', 'exit', 'q']:
            break
        if len(scenario) < 20:
            print("  Please provide more detail.")
            continue

        print(f"  Choose specialists (comma-separated, or 'all'):")
        print(f"    Options: {', '.join(available)}")
        choice = input("  > ").strip().lower()

        if choice == 'all':
            selected = available
        else:
            selected = [s.strip() for s in choice.split(",") if s.strip() in available]

        if not selected:
            print("  No valid specialists selected. Using all.")
            selected = available

        agent = ParallelFanOutAgent(specialists=selected)
        agent.run(scenario)


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 3: PARALLEL FAN-OUT / MAP-REDUCE AGENT")
    print("=" * 70)
    print("""
    Send the same case to multiple specialists simultaneously,
    then synthesize their opinions into a consensus plan.

    Choose a demo:
      1 → Basic fan-out (4 specialists, ACS case)
      2 → Parallel vs Sequential (timing comparison)
      3 → 5-specialist consult (complex patient)
      4 → Conflict detection (when specialists disagree)
      5 → Interactive multi-specialist consult
      6 → Run demos 1-4
    """)

    choice = input("  Enter choice (1-6): ").strip()

    if choice == "1":
        demo_basic_fanout()
    elif choice == "2":
        demo_parallel_vs_sequential()
    elif choice == "3":
        demo_five_specialists()
    elif choice == "4":
        demo_conflict_detection()
    elif choice == "5":
        demo_interactive()
    elif choice == "6":
        demo_basic_fanout()
        demo_parallel_vs_sequential()
        demo_five_specialists()
        demo_conflict_detection()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. FAN-OUT/MAP-REDUCE: Same query → multiple specialists → synthesize.
   This is how real medical teams work (multidisciplinary rounds).

2. PARALLELISM WINS: With ThreadPoolExecutor, 5 specialists take
   the SAME time as 1. Latency = max(specialist_times), not sum.
   Cost is identical (same tokens whether parallel or sequential).

3. CONFLICT RESOLUTION: The REDUCER is the most important component.
   It must identify disagreements and make principled decisions.
   In medicine: "Cardiology wants heparin, Nephrology is concerned
   about bleeding" → Reducer: "Use heparin with reduced dose and
   close monitoring of renal function."

4. WHEN TO USE FAN-OUT:
   ✓ Complex multi-system patients (multiple specialist perspectives)
   ✓ Second opinions (same question, different viewpoints)
   ✓ Comprehensive assessments (medication review + labs + guidelines)
   ✗ Simple questions ("what's the dose of aspirin?")
   ✗ Sequential dependencies (each step needs the previous result)

5. PRODUCTION CONSIDERATIONS:
   - Rate limiting: Fan-out can hit API rate limits fast
   - Cost: N specialists = N × tokens (expensive for large N)
   - Timeout handling: What if one specialist is slow/fails?
   - Quality: More opinions ≠ always better (noise vs signal)

6. FAN-OUT vs SEQUENTIAL PIPELINE vs AGENT DEBATE:
   - Fan-out: Same question, independent answers, synthesize
   - Pipeline: Each step builds on the previous (A → B → C)
   - Debate: Opposing viewpoints argue to find truth
"""

if __name__ == "__main__":
    main()
