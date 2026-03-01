"""
Exercise 4: Hierarchical Agent — Complex Case Management

Skills practiced:
- Supervisor agent that decomposes tasks and delegates to sub-agents
- Sub-agents that can further delegate (true hierarchy, not just routing)
- Task decomposition: breaking a complex case into manageable subtasks
- Result aggregation up the hierarchy
- Understanding when hierarchical beats flat architectures

Key insight: A Router picks ONE specialist and forwards the query.
  A Hierarchical agent DECOMPOSES the problem into subtasks and
  delegates EACH subtask to a different sub-agent — which may
  themselves further decompose and delegate.

  Router:       Query → Pick 1 specialist → Answer
  Hierarchical: Query → Break into 5 subtasks → Delegate each
                        → Sub-agents may further delegate
                        → Aggregate all results → Answer

  Think of it as a hospital chain of command:
    Attending (supervisor) → assigns tasks to:
      Resident (sub-agent) → assigns subtasks to:
        Medical Student → specific focused work

Architecture:
  ┌──────────────────────────────────┐
  │          SUPERVISOR               │
  │  "This is a complex case. I      │
  │   need 4 subtasks completed."    │
  └──────────┬───────────────────────┘
             │
    ┌────────┼──────────┬──────────────┐
    │        │          │              │
    ▼        ▼          ▼              ▼
  ┌─────┐ ┌─────┐ ┌──────────┐ ┌──────────┐
  │ Lab │ │ Med │ │ Clinical │ │ Discharge│  ← Sub-agents
  │Agent│ │Agent│ │ Agent    │ │ Agent    │     (specialists)
  └──┬──┘ └──┬──┘ └─────┬────┘ └──────────┘
     │       │          │
     │       │     ┌────┴────┐
     │       │     ▼         ▼
     │       │  ┌──────┐ ┌──────┐
     │       │  │Diff  │ │Risk  │  ← Sub-sub-agents
     │       │  │Dx    │ │Score │     (further delegation)
     │       │  └──────┘ └──────┘
     │       │
     └───────┴──── All results bubble up ────→ SUPERVISOR
                                                  │
                                           ┌──────▼──────┐
                                           │ FINAL REPORT│
                                           └─────────────┘
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

client = OpenAI()


# ============================================================
# Sub-Agent Definitions
# ============================================================
# Each sub-agent is a focused expert that handles one aspect
# of the case. Some sub-agents can further decompose their work.
# ============================================================

class SubAgent:
    """
    A focused agent that handles one specific subtask.
    Can optionally decompose its task into sub-subtasks.
    """

    def __init__(self, name: str, role: str, system_prompt: str, model: str = "gpt-4o-mini"):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.execution_log = []

    def execute(self, task: str, context: str = "") -> str:
        """Execute the assigned task."""
        self.execution_log.append({
            "agent": self.name,
            "task": task,
            "timestamp": datetime.now().isoformat(),
        })

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nYour task:\n{task}" if context else task},
            ],
            temperature=0,
        )

        result = response.choices[0].message.content
        self.execution_log[-1]["result"] = result[:200]
        return result


# Pre-defined sub-agents
def create_lab_agent():
    return SubAgent(
        name="Lab Interpreter",
        role="lab_analysis",
        system_prompt=(
            "You are a clinical lab specialist. Your job is to:\n"
            "1. Interpret ALL lab values (normal/abnormal/critical)\n"
            "2. Identify clinically significant abnormalities\n"
            "3. Correlate lab findings with the clinical picture\n"
            "4. Recommend additional labs if needed\n\n"
            "Format:\n"
            "LAB INTERPRETATION:\n"
            "- [lab name]: [value] → [status] (reference range)\n"
            "CRITICAL FINDINGS: [list]\n"
            "CORRELATIONS: [how labs relate to clinical picture]\n"
            "ADDITIONAL LABS NEEDED: [list with rationale]"
        ),
    )


def create_medication_agent():
    return SubAgent(
        name="Medication Reviewer",
        role="medication_review",
        system_prompt=(
            "You are a clinical pharmacist. Your job is to:\n"
            "1. Review ALL current medications\n"
            "2. Check for drug-drug interactions\n"
            "3. Identify medications that need adjustment (renal dosing, hold for procedure, etc.)\n"
            "4. Recommend new medications needed for acute management\n"
            "5. Flag high-alert medications\n\n"
            "Format:\n"
            "CURRENT MEDICATIONS REVIEW:\n"
            "- [drug]: [status: continue/hold/adjust/discontinue] — [reason]\n"
            "INTERACTIONS: [list any concerns]\n"
            "NEW MEDICATIONS NEEDED: [list with dose, route, frequency]\n"
            "MONITORING: [what to watch for each medication]"
        ),
    )


def create_clinical_assessment_agent():
    return SubAgent(
        name="Clinical Assessor",
        role="clinical_assessment",
        system_prompt=(
            "You are a senior clinician. Your job is to:\n"
            "1. Synthesize the clinical picture (history + vitals + labs)\n"
            "2. Provide primary differential diagnosis (top 3-5 with reasoning)\n"
            "3. Assess acuity and risk level\n"
            "4. Determine disposition (ICU/floor/observation/discharge)\n\n"
            "Format:\n"
            "CLINICAL IMPRESSION: [2-3 sentence summary]\n"
            "DIFFERENTIAL DIAGNOSIS:\n"
            "  1. [Dx] — [reasoning, supporting evidence]\n"
            "  2. [Dx] — [reasoning]\n"
            "  3. [Dx] — [reasoning]\n"
            "RISK LEVEL: [low/moderate/high/critical]\n"
            "DISPOSITION: [ICU/stepdown/floor/observation/discharge] — [rationale]"
        ),
    )


def create_treatment_plan_agent():
    return SubAgent(
        name="Treatment Planner",
        role="treatment_plan",
        system_prompt=(
            "You are a senior physician creating a treatment plan. Given the lab results, "
            "medication review, and clinical assessment, create a comprehensive plan.\n\n"
            "Format:\n"
            "IMMEDIATE ACTIONS (next 30 min):\n"
            "  1. [specific order with dose/route/frequency]\n"
            "ONGOING MANAGEMENT (next 24h):\n"
            "  1. [specific order]\n"
            "MONITORING:\n"
            "  - [what to monitor, how often]\n"
            "CONSULTS: [which specialties, urgency]\n"
            "CONTINGENCY: [if patient worsens, then...]"
        ),
    )


def create_discharge_planning_agent():
    return SubAgent(
        name="Discharge Planner",
        role="discharge_planning",
        system_prompt=(
            "You are a discharge planning specialist. Based on the clinical assessment "
            "and treatment plan, create a discharge plan.\n\n"
            "Format:\n"
            "ESTIMATED LENGTH OF STAY: [days]\n"
            "DISCHARGE CRITERIA: [what needs to happen before discharge]\n"
            "HOME MEDICATIONS: [list with changes from admission]\n"
            "FOLLOW-UP: [appointments, timing, with whom]\n"
            "PATIENT EDUCATION: [key points the patient must understand]\n"
            "RED FLAGS: [when to return to ED]"
        ),
    )


# Sub-sub-agents (for further decomposition)
def create_differential_dx_agent():
    return SubAgent(
        name="Differential Diagnosis Specialist",
        role="differential_dx",
        system_prompt=(
            "You are a diagnostic reasoning specialist. Your ONLY job is to create "
            "a thorough differential diagnosis. For EACH diagnosis:\n"
            "1. State the diagnosis\n"
            "2. List supporting evidence from the case\n"
            "3. List evidence AGAINST this diagnosis\n"
            "4. State the probability (high/moderate/low)\n"
            "5. State what test would confirm or rule out\n\n"
            "Provide at least 5 differential diagnoses, ranked by probability."
        ),
    )


def create_risk_scoring_agent():
    return SubAgent(
        name="Risk Scoring Specialist",
        role="risk_scoring",
        system_prompt=(
            "You are a clinical risk stratification specialist. Calculate ALL applicable "
            "risk scores for this patient. For each score:\n"
            "1. Name of the score (e.g., HEART, TIMI, Wells, CURB-65)\n"
            "2. Individual components with values\n"
            "3. Total score\n"
            "4. Risk category and interpretation\n"
            "5. Recommended action based on score\n\n"
            "Calculate every applicable score — don't just pick one."
        ),
    )


# ============================================================
# Hierarchical Supervisor Agent
# ============================================================

class HierarchicalSupervisor:
    """
    A supervisor agent that:
    1. Analyzes the case and decomposes it into subtasks
    2. Delegates each subtask to a specialized sub-agent
    3. Some sub-agents may further delegate (creating hierarchy)
    4. Aggregates all results into a comprehensive report
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.sub_agents = {}
        self.task_tree = []  # Track the delegation hierarchy
        self.all_results = {}

    def decompose(self, scenario: str) -> list[dict]:
        """
        Phase 1: Decompose the case into subtasks.
        The supervisor decides what needs to be done and who should do it.
        """
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are a supervising physician. Analyze this case and decompose it "
                    "into subtasks. Available sub-agents:\n"
                    "1. lab_analysis — interprets lab results\n"
                    "2. medication_review — reviews and adjusts medications\n"
                    "3. clinical_assessment — provides differential and risk assessment\n"
                    "4. treatment_plan — creates treatment orders\n"
                    "5. discharge_planning — plans discharge and follow-up\n\n"
                    "For clinical_assessment, it can further decompose into:\n"
                    "  - differential_dx — detailed differential diagnosis\n"
                    "  - risk_scoring — calculate risk scores\n\n"
                    "Output JSON: {\"tasks\": [{\"id\": 1, \"agent\": \"lab_analysis\", "
                    "\"task_description\": \"...\", \"depends_on\": [], \"sub_decompose\": false}, ...]}\n\n"
                    "Set 'sub_decompose': true if the task should be further broken down by the sub-agent.\n"
                    "Set 'depends_on': [task_ids] if a task needs results from earlier tasks."
                )},
                {"role": "user", "content": scenario},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(response.choices[0].message.content)
            self.task_tree = data.get("tasks", [])
        except json.JSONDecodeError:
            # Fallback task list
            self.task_tree = [
                {"id": 1, "agent": "lab_analysis", "task_description": "Interpret all lab values", "depends_on": [], "sub_decompose": False},
                {"id": 2, "agent": "medication_review", "task_description": "Review current medications", "depends_on": [], "sub_decompose": False},
                {"id": 3, "agent": "clinical_assessment", "task_description": "Clinical assessment and differential", "depends_on": [1, 2], "sub_decompose": True},
                {"id": 4, "agent": "treatment_plan", "task_description": "Create treatment plan", "depends_on": [1, 2, 3], "sub_decompose": False},
                {"id": 5, "agent": "discharge_planning", "task_description": "Plan discharge", "depends_on": [3, 4], "sub_decompose": False},
            ]

        return self.task_tree

    def delegate_and_execute(self, scenario: str, verbose: bool = True) -> dict:
        """
        Phase 2: Execute tasks respecting dependencies.
        Tasks without dependencies run first, then dependent tasks.
        """
        # Create all sub-agents
        agent_factory = {
            "lab_analysis": create_lab_agent,
            "medication_review": create_medication_agent,
            "clinical_assessment": create_clinical_assessment_agent,
            "treatment_plan": create_treatment_plan_agent,
            "discharge_planning": create_discharge_planning_agent,
            "differential_dx": create_differential_dx_agent,
            "risk_scoring": create_risk_scoring_agent,
        }

        completed_tasks = {}
        remaining = list(self.task_tree)

        while remaining:
            # Find tasks whose dependencies are all met
            ready = [
                t for t in remaining
                if all(dep in completed_tasks for dep in t.get("depends_on", []))
            ]

            if not ready:
                # Avoid infinite loop — just run everything remaining
                ready = remaining

            for task in ready:
                task_id = task["id"]
                agent_type = task["agent"]
                description = task["task_description"]
                sub_decompose = task.get("sub_decompose", False)

                if verbose:
                    deps = task.get("depends_on", [])
                    dep_str = f" (depends on: {deps})" if deps else ""
                    print(f"\n    📌 Task {task_id}: {description}{dep_str}")
                    print(f"       Assigned to: {agent_type}")

                # Build context from completed dependencies
                context_parts = [scenario]
                for dep_id in task.get("depends_on", []):
                    if dep_id in completed_tasks:
                        context_parts.append(f"\n--- {completed_tasks[dep_id]['agent']} result ---\n{completed_tasks[dep_id]['result'][:500]}")
                context = "\n".join(context_parts)

                # Create and execute sub-agent
                if agent_type in agent_factory:
                    agent = agent_factory[agent_type]()
                    result = agent.execute(description, context)

                    if verbose:
                        preview = result[:150].replace("\n", " | ")
                        print(f"       Result: {preview}...")

                    completed_tasks[task_id] = {
                        "agent": agent_type,
                        "task": description,
                        "result": result,
                    }

                    # Sub-decomposition: the sub-agent delegates further
                    if sub_decompose and agent_type == "clinical_assessment":
                        if verbose:
                            print(f"\n       ↳ Sub-decomposing: clinical_assessment → differential_dx + risk_scoring")

                        # Differential diagnosis sub-agent
                        diff_agent = create_differential_dx_agent()
                        diff_result = diff_agent.execute(
                            "Create a thorough differential diagnosis for this patient",
                            context + f"\n--- Clinical Assessment ---\n{result[:500]}"
                        )
                        completed_tasks[f"{task_id}_diff"] = {
                            "agent": "differential_dx",
                            "task": "Detailed differential diagnosis",
                            "result": diff_result,
                            "parent": task_id,
                        }
                        if verbose:
                            preview = diff_result[:120].replace("\n", " | ")
                            print(f"       ↳ Differential Dx: {preview}...")

                        # Risk scoring sub-agent
                        risk_agent = create_risk_scoring_agent()
                        risk_result = risk_agent.execute(
                            "Calculate all applicable risk scores",
                            context + f"\n--- Clinical Assessment ---\n{result[:500]}"
                        )
                        completed_tasks[f"{task_id}_risk"] = {
                            "agent": "risk_scoring",
                            "task": "Risk score calculations",
                            "result": risk_result,
                            "parent": task_id,
                        }
                        if verbose:
                            preview = risk_result[:120].replace("\n", " | ")
                            print(f"       ↳ Risk Scores: {preview}...")
                else:
                    completed_tasks[task_id] = {
                        "agent": agent_type,
                        "task": description,
                        "result": f"(No agent available for '{agent_type}')",
                    }

                remaining.remove(task)

        self.all_results = completed_tasks
        return completed_tasks

    def aggregate(self, scenario: str) -> str:
        """
        Phase 3: Supervisor aggregates all sub-agent results
        into a comprehensive final report.
        """
        results_text = ""
        for task_id, result in sorted(self.all_results.items(), key=lambda x: str(x[0])):
            parent_note = f" (sub-task of Task {result['parent']})" if 'parent' in result else ""
            results_text += f"\n--- {result['agent']}{parent_note} ---\n{result['result']}\n"

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are the supervising attending physician. Your sub-agents have completed "
                    "their analysis. Now create a COMPREHENSIVE final report that:\n\n"
                    "1. EXECUTIVE SUMMARY (3-4 sentences — the key facts)\n"
                    "2. INTEGRATED ASSESSMENT (combine all sub-agent findings)\n"
                    "3. PRIMARY DIAGNOSIS with confidence level\n"
                    "4. COMPLETE TREATMENT PLAN (immediate + ongoing)\n"
                    "5. CRITICAL ACTIONS (must-do items, time-sensitive)\n"
                    "6. MONITORING & FOLLOW-UP\n"
                    "7. ANTICIPATED COMPLICATIONS\n\n"
                    "This should read as a single coherent report, not a compilation of fragments."
                )},
                {"role": "user", "content": f"Patient:\n{scenario}\n\nSub-agent reports:{results_text}"},
            ],
            temperature=0,
        )

        return response.choices[0].message.content

    def run(self, scenario: str, verbose: bool = True) -> dict:
        """Run the full hierarchical pipeline."""
        if verbose:
            print("\n  ╔══════════════════════════════════════╗")
            print("  ║  HIERARCHICAL AGENT — SUPERVISOR     ║")
            print("  ╚══════════════════════════════════════╝")

        # Phase 1: Decompose
        if verbose:
            print(f"\n  📋 Phase 1: DECOMPOSING case into subtasks...")
        tasks = self.decompose(scenario)
        if verbose:
            for t in tasks:
                deps = f" ← depends on {t.get('depends_on', [])}" if t.get('depends_on') else ""
                sub = " [will sub-decompose]" if t.get('sub_decompose') else ""
                print(f"    Task {t['id']}: [{t['agent']}] {t['task_description']}{deps}{sub}")

        # Phase 2: Delegate and execute
        if verbose:
            print(f"\n  ⚙️ Phase 2: DELEGATING to {len(tasks)} sub-agents...")
        results = self.delegate_and_execute(scenario, verbose)
        if verbose:
            agent_count = len(set(r["agent"] for r in results.values()))
            print(f"\n    Completed: {len(results)} tasks across {agent_count} agent types")

        # Phase 3: Aggregate
        if verbose:
            print(f"\n  📝 Phase 3: AGGREGATING into final report...")
        report = self.aggregate(scenario)
        if verbose:
            print(f"\n  {'═' * 60}")
            print("  FINAL COMPREHENSIVE REPORT:")
            print(f"  {'═' * 60}")
            for line in report.split("\n"):
                print(f"  {line}")

        return {
            "task_tree": tasks,
            "sub_agent_results": results,
            "final_report": report,
            "total_agents_used": len(results),
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
Patient: 70-year-old male, found unresponsive by family
Chief Complaint: Altered mental status, last seen normal 8 hours ago
History: Atrial fibrillation, heart failure (EF 25%), Type 2 diabetes, CKD stage 4,
  prior stroke 2 years ago, COPD on home O2
Current Medications: Apixaban 5mg BID, Metoprolol 50mg BID, Furosemide 40mg BID,
  Insulin glargine 20u QHS, Lisinopril 5mg daily, Tiotropium inhaler
Vitals: BP 92/54, HR 48 (irregular), RR 8, SpO2 82% (on 2L home O2), Temp 35.4°C, GCS 7 (E1V2M4)
Labs: Glucose 42 mg/dL(!), Troponin 0.12, Creatinine 4.8 (baseline 3.2),
  K+ 6.8(!), Lactate 6.2, WBC 2.1, Hgb 7.8, PLT 52
ABG: pH 7.12, pCO2 68, pO2 54, HCO3 12
""".strip()


# ============================================================
# Demo Functions
# ============================================================

def demo_basic_hierarchical():
    """Basic hierarchical agent for ACS."""
    print("\n" + "=" * 70)
    print("  DEMO 1: BASIC HIERARCHICAL AGENT — ACS CASE")
    print("=" * 70)
    print("""
  The supervisor will:
  1. Decompose the case into 5 subtasks
  2. Delegate each to a specialized sub-agent
  3. Some sub-agents further decompose (differential dx, risk scores)
  4. Aggregate everything into a comprehensive report
  """)

    supervisor = HierarchicalSupervisor()
    result = supervisor.run(SCENARIO_ACS)

    print(f"\n  Summary: {result['total_agents_used']} total agents used")
    print(f"  Task tree depth: 2 levels (supervisor → sub-agents → sub-sub-agents)")


def demo_complex_case():
    """Hierarchical agent for a critically ill patient."""
    print("\n" + "=" * 70)
    print("  DEMO 2: COMPLEX CASE — CRITICALLY ILL PATIENT")
    print("=" * 70)
    print("""
  A critically ill patient with MULTIPLE simultaneous problems:
  - Hypoglycemia (glucose 42!)
  - Hyperkalemia (K+ 6.8!)
  - Respiratory failure (SpO2 82%, pH 7.12)
  - Possible sepsis (hypothermia, WBC 2.1)
  - AKI on CKD (creatinine 4.8)
  - GCS 7 (comatose)

  This is where hierarchical architecture shines — too complex
  for a single agent, needs systematic decomposition.
  """)

    supervisor = HierarchicalSupervisor()
    result = supervisor.run(SCENARIO_COMPLEX)

    print(f"\n  This case required {result['total_agents_used']} agents to manage!")
    print(f"  A single ReAct agent would likely miss critical details in a case this complex.")


def demo_hierarchy_vs_flat():
    """Compare hierarchical vs flat single-agent approach."""
    print("\n" + "=" * 70)
    print("  DEMO 3: HIERARCHICAL vs FLAT AGENT — COMPARISON")
    print("=" * 70)
    print("""
  Same complex case, two approaches:
  1. FLAT: Single agent tries to do everything
  2. HIERARCHICAL: Supervisor decomposes and delegates

  Watch which produces a more comprehensive response.
  """)

    # Flat approach (single agent)
    print("\n  ═══ FLAT (single agent) ═══")
    flat_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a physician. Provide a comprehensive assessment including: "
                "lab interpretation, medication review, differential diagnosis, "
                "risk scoring, treatment plan, and discharge planning."
            )},
            {"role": "user", "content": SCENARIO_COMPLEX},
        ],
        temperature=0,
    )
    flat_result = flat_response.choices[0].message.content
    flat_tokens = flat_response.usage.total_tokens if flat_response.usage else 0
    print(f"  Output: {len(flat_result)} chars, {flat_tokens} tokens")
    print(f"  First 300 chars:\n  {flat_result[:300].replace(chr(10), chr(10) + '  ')}...")

    # Hierarchical approach
    print(f"\n\n  ═══ HIERARCHICAL (supervisor + sub-agents) ═══")
    supervisor = HierarchicalSupervisor()
    hier_result = supervisor.run(SCENARIO_COMPLEX, verbose=False)
    hier_output = hier_result["final_report"]
    print(f"  Output: {len(hier_output)} chars, {hier_result['total_agents_used']} agents used")
    print(f"  First 300 chars:\n  {hier_output[:300].replace(chr(10), chr(10) + '  ')}...")

    # Comparison
    print(f"\n\n  ═══ COMPARISON ═══")
    print(f"  {'Metric':<25} {'Flat':>15} {'Hierarchical':>15}")
    print(f"  {'─' * 55}")
    print(f"  {'Output length':<25} {len(flat_result):>12} ch {len(hier_output):>12} ch")
    print(f"  {'Agents used':<25} {'1':>15} {hier_result['total_agents_used']:>15}")
    print(f"  {'LLM calls':<25} {'1':>15} {hier_result['total_agents_used'] + 2:>15}")  # +2 for decompose + aggregate

    print(f"\n  The hierarchical approach typically produces MORE DETAILED output")
    print(f"  because each sub-agent focuses deeply on ONE aspect. The flat approach")
    print(f"  has to spread its attention across everything, often missing details.")
    print(f"\n  Tradeoff: Hierarchical costs ~{hier_result['total_agents_used'] + 2}x more tokens.")


def demo_task_tree_visualization():
    """Visualize the task decomposition tree."""
    print("\n" + "=" * 70)
    print("  DEMO 4: TASK TREE VISUALIZATION")
    print("=" * 70)
    print("""
  See how the supervisor decomposes the case into a tree of tasks.
  Tasks with dependencies must wait for their prerequisites.
  """)

    supervisor = HierarchicalSupervisor()
    tasks = supervisor.decompose(SCENARIO_ACS)

    print("\n  Task Decomposition Tree:")
    print(f"  {'─' * 50}")
    print(f"  SUPERVISOR")

    # Group by dependency level
    levels = {}
    for task in tasks:
        deps = task.get("depends_on", [])
        level = max([levels.get(d, 0) for d in deps], default=-1) + 1
        levels[task["id"]] = level

    max_level = max(levels.values()) if levels else 0
    for level in range(max_level + 1):
        tasks_at_level = [t for t in tasks if levels.get(t["id"]) == level]
        for t in tasks_at_level:
            indent = "  " * (level + 1)
            deps = f" (after: {t.get('depends_on', [])})" if t.get('depends_on') else ""
            sub = " → [will sub-decompose]" if t.get('sub_decompose') else ""
            print(f"  {indent}├── Task {t['id']}: [{t['agent']}] {t['task_description'][:50]}{deps}{sub}")

    print(f"\n  Execution order (respecting dependencies):")
    sorted_tasks = sorted(tasks, key=lambda t: levels.get(t["id"], 0))
    for t in sorted_tasks:
        level = levels.get(t["id"], 0)
        print(f"    Level {level}: Task {t['id']} ({t['agent']})")


def demo_interactive():
    """Interactive hierarchical agent."""
    print("\n" + "=" * 70)
    print("  DEMO 5: INTERACTIVE HIERARCHICAL AGENT")
    print("=" * 70)
    print("  Enter a complex patient scenario. The supervisor will decompose,")
    print("  delegate, and aggregate. Type 'quit' to exit.\n")

    while True:
        scenario = input("  Patient scenario (or 'quit'): ").strip()

        if scenario.lower() in ['quit', 'exit', 'q']:
            break

        if len(scenario) < 30:
            print("  Please provide a detailed scenario (history, vitals, labs).")
            continue

        supervisor = HierarchicalSupervisor()
        result = supervisor.run(scenario)

        print(f"\n  Agents used: {result['total_agents_used']}")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 4: HIERARCHICAL AGENT — COMPLEX CASE MANAGEMENT")
    print("=" * 70)
    print("""
    The Hierarchical pattern: Supervisor decomposes tasks,
    delegates to sub-agents, sub-agents may further delegate,
    supervisor aggregates into final report.

    Choose a demo:
      1 → Basic hierarchical (ACS case)
      2 → Complex case (critically ill patient)
      3 → Hierarchical vs Flat (comparison)
      4 → Task tree visualization
      5 → Interactive (enter your own scenario)
      6 → Run demos 1-4
    """)

    choice = input("  Enter choice (1-6): ").strip()

    if choice == "1":
        demo_basic_hierarchical()
    elif choice == "2":
        demo_complex_case()
    elif choice == "3":
        demo_hierarchy_vs_flat()
    elif choice == "4":
        demo_task_tree_visualization()
    elif choice == "5":
        demo_interactive()
    elif choice == "6":
        demo_basic_hierarchical()
        demo_complex_case()
        demo_hierarchy_vs_flat()
        demo_task_tree_visualization()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. HIERARCHICAL = DECOMPOSE + DELEGATE + AGGREGATE
   The supervisor breaks big problems into small ones.
   Sub-agents solve small problems. Results bubble up.

2. vs ROUTER: A router picks ONE agent. A hierarchy decomposes
   into MANY subtasks and delegates ALL of them.
   Router: "Send this to cardiology"
   Hierarchy: "This needs lab review + med review + clinical assessment + treatment + discharge"

3. vs PARALLEL FAN-OUT: Fan-out sends the SAME question to ALL agents.
   Hierarchy sends DIFFERENT questions to DIFFERENT agents.
   Fan-out: "Everyone analyze this patient" (same task, different lens)
   Hierarchy: "You do labs, you do meds, you do assessment" (different tasks)

4. DEPENDENCY MANAGEMENT: Tasks form a DAG (directed acyclic graph).
   Independent tasks can run in parallel.
   Dependent tasks must wait for their prerequisites.
   This is the same pattern as build systems (Makefile, Gradle).

5. SUB-DECOMPOSITION: Sub-agents can further break down their work.
   The clinical_assessment agent → differential_dx + risk_scoring.
   This creates a true hierarchy, not just a flat delegation.

6. WHEN TO USE:
   ✓ Complex cases requiring multiple types of analysis
   ✓ Comprehensive reports (annual physical, admission workup)
   ✓ Cases with many interacting problems (like the demo)
   ✗ Simple questions ("what's the dose?")
   ✗ Real-time chat (too slow — many LLM calls)

7. COST: Hierarchy is EXPENSIVE.
   5 sub-agents + 2 sub-sub-agents + decompose + aggregate = 10 LLM calls.
   Use ONLY when comprehensiveness justifies the cost.
"""

if __name__ == "__main__":
    main()
