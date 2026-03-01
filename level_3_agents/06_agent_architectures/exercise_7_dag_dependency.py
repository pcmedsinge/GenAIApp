"""
Exercise 7: DAG Dependency Workflows — Graph-Based Task Orchestration

Skills practiced:
- Building a Directed Acyclic Graph (DAG) of tasks
- Topological sort for correct execution order
- Conditional edges: skip/add tasks based on runtime results
- Dynamic replanning when a task fails
- Parallel execution of independent tasks
- Understanding DAGs vs flat sequences vs trees

Key insight: Most real workflows are NOT sequential pipelines.
  They're DAGs: some tasks depend on others, some can run in parallel,
  some are conditional. A DAG captures this naturally.

Architecture:
  ┌────────────────────────────────────────────────────────────┐
  │                    DAG WORKFLOW ENGINE                      │
  │                                                            │
  │      ┌──────────┐                                          │
  │      │  Start   │                                          │
  │      └───┬──┬───┘                                          │
  │          │  │                                               │
  │     ┌────┘  └────┐        ← Fork: 2 independent tasks     │
  │     ▼            ▼                                         │
  │  ┌──────┐   ┌──────┐                                      │
  │  │ Labs │   │ Meds │      ← Can run in parallel           │
  │  └──┬───┘   └──┬───┘                                      │
  │     │          │                                           │
  │     └────┬─────┘          ← Join: both must complete       │
  │          ▼                                                 │
  │     ┌──────────┐                                           │
  │     │ Assess   │                                           │
  │     └───┬──┬───┘                                           │
  │         │  │                                               │
  │    ┌────┘  └────────┐     ← Conditional fork               │
  │    ▼                ▼                                      │
  │  ┌──────┐   ┌────────────┐                                 │
  │  │Treat │   │ Consult    │  ← Only if assessment says      │
  │  │Plan  │   │(condition) │    "needs specialist"           │
  │  └──┬───┘   └──────┬─────┘                                 │
  │     │              │                                       │
  │     └──────┬───────┘                                       │
  │            ▼                                               │
  │       ┌──────────┐                                         │
  │       │Discharge │                                         │
  │       │Planning  │                                         │
  │       └──────────┘                                         │
  │                                                            │
  │  Key: ─── dependency  ═══ conditional dependency           │
  └────────────────────────────────────────────────────────────┘

Healthcare parallel: An admission workup IS a DAG.
  Labs and imaging can run in parallel (no dependency).
  Assessment depends on both labs AND imaging.
  The treatment plan depends on the assessment.
  Specialist consults are conditional (only if needed).
"""

import os
import json
import time
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

client = OpenAI()


# ============================================================
# DAG Engine
# ============================================================

class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DAGTask:
    """
    A single task node in the DAG.
    """

    def __init__(self, task_id: str, name: str, handler, dependencies: list[str] = None,
                 condition=None, retry_count: int = 0):
        """
        Args:
            task_id: Unique identifier
            name: Human-readable name
            handler: Function to execute. Receives (context_dict) → result
            dependencies: List of task_ids that must complete first
            condition: Optional function(context) → bool. If False, skip this task.
            retry_count: How many times to retry on failure
        """
        self.task_id = task_id
        self.name = name
        self.handler = handler
        self.dependencies = dependencies or []
        self.condition = condition
        self.retry_count = retry_count
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.elapsed = 0
        self.attempts = 0


class DAGWorkflow:
    """
    A DAG-based workflow engine that:
    - Accepts tasks with dependencies
    - Validates the DAG (no cycles)
    - Executes tasks in topological order
    - Runs independent tasks in parallel
    - Supports conditional edges
    - Handles failure and retry
    - Provides execution visualization
    """

    def __init__(self, name: str = "Workflow"):
        self.name = name
        self.tasks: dict[str, DAGTask] = {}
        self.execution_log = []
        self.context = {}  # Shared context that all tasks can read/write

    def add_task(self, task: DAGTask):
        """Add a task to the DAG."""
        self.tasks[task.task_id] = task

    def validate(self) -> list[str]:
        """
        Validate the DAG: check for cycles and missing dependencies.
        Returns a list of error messages (empty = valid).
        """
        errors = []

        # Check for missing dependencies
        for task_id, task in self.tasks.items():
            for dep in task.dependencies:
                if dep not in self.tasks:
                    errors.append(f"Task '{task_id}' depends on '{dep}' which doesn't exist")

        # Check for cycles using DFS
        visited = set()
        in_stack = set()

        def has_cycle(node_id):
            visited.add(node_id)
            in_stack.add(node_id)
            for dep in self.tasks[node_id].dependencies:
                if dep in self.tasks:
                    if dep not in visited:
                        if has_cycle(dep):
                            return True
                    elif dep in in_stack:
                        errors.append(f"Cycle detected involving '{node_id}' and '{dep}'")
                        return True
            in_stack.remove(node_id)
            return False

        for task_id in self.tasks:
            if task_id not in visited:
                has_cycle(task_id)

        return errors

    def topological_sort(self) -> list[list[str]]:
        """
        Topological sort of the DAG, grouped into LEVELS.
        Tasks within the same level can run in parallel.

        Returns: [[level0_tasks], [level1_tasks], ...]
        """
        in_degree = {tid: 0 for tid in self.tasks}
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep in in_degree:
                    in_degree[task.task_id] = in_degree.get(task.task_id, 0)

        # Compute in-degrees
        in_degree = {tid: 0 for tid in self.tasks}
        for task in self.tasks.values():
            # This task adds to the in-degree of nothing; its deps add to IT
            pass

        # Proper in-degree computation
        adj = defaultdict(list)  # dep → [tasks that depend on dep]
        in_deg = {tid: 0 for tid in self.tasks}
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep in self.tasks:
                    adj[dep].append(task.task_id)
                    in_deg[task.task_id] += 1

        # BFS with levels (Kahn's algorithm)
        levels = []
        current_level = [tid for tid, deg in in_deg.items() if deg == 0]

        while current_level:
            levels.append(current_level)
            next_level = []
            for tid in current_level:
                for dependent in adj[tid]:
                    in_deg[dependent] -= 1
                    if in_deg[dependent] == 0:
                        next_level.append(dependent)
            current_level = next_level

        return levels

    def _execute_task(self, task: DAGTask) -> bool:
        """Execute a single task with retry logic."""
        # Check condition
        if task.condition and not task.condition(self.context):
            task.status = TaskStatus.SKIPPED
            self.execution_log.append({
                "task": task.task_id,
                "name": task.name,
                "status": "skipped",
                "reason": "condition not met",
            })
            return True

        # Execute with retry
        for attempt in range(task.retry_count + 1):
            try:
                task.status = TaskStatus.RUNNING
                task.attempts = attempt + 1
                start = time.time()

                result = task.handler(self.context)

                task.elapsed = round(time.time() - start, 2)
                task.result = result
                task.status = TaskStatus.COMPLETED

                # Store result in shared context
                self.context[f"result_{task.task_id}"] = result

                self.execution_log.append({
                    "task": task.task_id,
                    "name": task.name,
                    "status": "completed",
                    "elapsed": task.elapsed,
                    "attempts": task.attempts,
                })
                return True

            except Exception as e:
                task.error = str(e)
                if attempt < task.retry_count:
                    time.sleep(1)  # Brief pause before retry
                    continue

        task.status = TaskStatus.FAILED
        self.execution_log.append({
            "task": task.task_id,
            "name": task.name,
            "status": "failed",
            "error": task.error,
            "attempts": task.attempts,
        })
        return False

    def run(self, initial_context: dict = None, verbose: bool = True,
            parallel: bool = True) -> dict:
        """
        Execute the entire DAG workflow.

        Args:
            initial_context: Initial data available to all tasks
            verbose: Print progress
            parallel: Run independent tasks in parallel (True) or sequential (False)
        """
        self.context = initial_context or {}
        self.execution_log = []

        # Validate
        errors = self.validate()
        if errors:
            return {"status": "invalid", "errors": errors}

        # Get execution levels
        levels = self.topological_sort()

        if verbose:
            print(f"\n  ╔══════════════════════════════════════╗")
            print(f"  ║  DAG WORKFLOW: {self.name:<22} ║")
            print(f"  ╚══════════════════════════════════════╝")
            print(f"\n  Execution plan ({len(levels)} levels, {len(self.tasks)} tasks):")
            for i, level in enumerate(levels):
                task_names = [self.tasks[tid].name for tid in level]
                parallel_tag = " [parallel]" if len(level) > 1 else ""
                print(f"    Level {i}: {', '.join(task_names)}{parallel_tag}")

        # Execute level by level
        for level_idx, level in enumerate(levels):
            if verbose:
                print(f"\n  ▶ Level {level_idx}: Executing {len(level)} task(s)...")

            if parallel and len(level) > 1:
                # Run tasks in this level in parallel
                with ThreadPoolExecutor(max_workers=len(level)) as executor:
                    futures = {}
                    for task_id in level:
                        task = self.tasks[task_id]
                        future = executor.submit(self._execute_task, task)
                        futures[future] = task_id

                    for future in as_completed(futures):
                        task_id = futures[future]
                        task = self.tasks[task_id]
                        if verbose:
                            status_icon = {"completed": "✅", "failed": "❌", "skipped": "⏭️"}.get(task.status, "?")
                            print(f"    {status_icon} {task.name} ({task.status}, {task.elapsed}s)")
            else:
                # Run sequentially
                for task_id in level:
                    task = self.tasks[task_id]
                    self._execute_task(task)
                    if verbose:
                        status_icon = {"completed": "✅", "failed": "❌", "skipped": "⏭️"}.get(task.status, "?")
                        print(f"    {status_icon} {task.name} ({task.status}, {task.elapsed}s)")

        # Summary
        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        skipped = sum(1 for t in self.tasks.values() if t.status == TaskStatus.SKIPPED)
        total_time = sum(t.elapsed for t in self.tasks.values())

        result = {
            "status": "completed" if failed == 0 else "partial",
            "tasks_completed": completed,
            "tasks_failed": failed,
            "tasks_skipped": skipped,
            "total_elapsed": round(total_time, 2),
            "execution_log": self.execution_log,
            "context": self.context,
        }

        if verbose:
            print(f"\n  ══ Workflow Complete ══")
            print(f"  Completed: {completed} | Failed: {failed} | Skipped: {skipped}")
            print(f"  Total time: {total_time:.2f}s")

        return result

    def visualize(self) -> str:
        """Generate a text visualization of the DAG."""
        levels = self.topological_sort()
        lines = [f"DAG: {self.name}", ""]

        for i, level in enumerate(levels):
            tasks_str = []
            for tid in level:
                task = self.tasks[tid]
                deps = f" ← [{', '.join(task.dependencies)}]" if task.dependencies else ""
                cond = " (conditional)" if task.condition else ""
                status = f" [{task.status}]" if task.status != TaskStatus.PENDING else ""
                tasks_str.append(f"[{task.name}{status}]{deps}{cond}")

            connector = " & ".join(tasks_str)
            lines.append(f"  Level {i}: {connector}")

            if i < len(levels) - 1:
                lines.append(f"    {'│' * len(level)}")

        return "\n".join(lines)


# ============================================================
# Clinical Workflow Tasks (LLM-powered)
# ============================================================

def make_llm_task(system_prompt: str, task_description: str, context_keys: list[str] = None):
    """
    Factory: create a task handler that calls the LLM.
    The handler reads from context and writes back to context.
    """
    def handler(context: dict) -> str:
        # Build prompt from context
        parts = [context.get("scenario", "")]
        for key in (context_keys or []):
            if key in context:
                parts.append(f"\n{key}:\n{context[key]}")
            elif f"result_{key}" in context:
                val = context[f"result_{key}"]
                parts.append(f"\n{key} result:\n{str(val)[:500]}")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{task_description}\n\nContext:\n{''.join(parts)}"},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    return handler


# ============================================================
# Demo Scenarios
# ============================================================

SCENARIO_ACS = """
Patient: 55-year-old male
Chief Complaint: Chest pain for 2 hours, radiating to left arm
History: Type 2 diabetes, hypertension, hyperlipidemia
Medications: Metformin 1000mg BID, Lisinopril 20mg daily, Atorvastatin 40mg
Vitals: BP 158/92, HR 98, SpO2 96%
Labs: Troponin I 0.45 (elevated), Glucose 210, Creatinine 1.3, K+ 4.8
ECG: ST depression V3-V6
""".strip()


# ============================================================
# Demo Functions
# ============================================================

def build_admission_workflow() -> DAGWorkflow:
    """Build a clinical admission DAG workflow."""
    wf = DAGWorkflow(name="Admission Workup")

    # Level 0: Independent tasks (can run in parallel)
    wf.add_task(DAGTask(
        task_id="labs",
        name="Lab Interpretation",
        handler=make_llm_task(
            "You are a lab specialist. Interpret all lab values. Flag critical results.",
            "Interpret these lab results for the patient."
        ),
    ))

    wf.add_task(DAGTask(
        task_id="meds",
        name="Medication Review",
        handler=make_llm_task(
            "You are a clinical pharmacist. Review medications for interactions and adjustments.",
            "Review current medications for this patient."
        ),
    ))

    wf.add_task(DAGTask(
        task_id="history",
        name="History Synthesis",
        handler=make_llm_task(
            "You are a clinician. Summarize the relevant medical history and risk factors.",
            "Synthesize the medical history and identify key risk factors."
        ),
    ))

    # Level 1: Depends on labs + meds + history
    wf.add_task(DAGTask(
        task_id="assessment",
        name="Clinical Assessment",
        dependencies=["labs", "meds", "history"],
        handler=make_llm_task(
            "You are a senior clinician. Provide differential diagnosis and risk assessment.",
            "Based on the lab results, medication review, and history, provide your clinical assessment.",
            context_keys=["labs", "meds", "history"],
        ),
    ))

    # Level 2: Depends on assessment — conditional: only if high-risk
    def needs_consult(context):
        """Check if the assessment mentions high risk or critical findings."""
        assessment = str(context.get("result_assessment", ""))
        keywords = ["high risk", "critical", "STEMI", "emergent", "ICU", "unstable"]
        return any(kw.lower() in assessment.lower() for kw in keywords)

    wf.add_task(DAGTask(
        task_id="consult",
        name="Specialist Consult",
        dependencies=["assessment"],
        condition=needs_consult,
        handler=make_llm_task(
            "You are a consulting cardiologist. Provide specialist recommendations.",
            "The primary team has requested a cardiology consult for this patient.",
            context_keys=["assessment"],
        ),
    ))

    wf.add_task(DAGTask(
        task_id="treatment",
        name="Treatment Plan",
        dependencies=["assessment"],
        handler=make_llm_task(
            "You are a treating physician. Create a specific treatment plan with orders.",
            "Create a treatment plan based on the clinical assessment.",
            context_keys=["assessment", "meds"],
        ),
    ))

    # Level 3: Depends on treatment (and optionally consult)
    wf.add_task(DAGTask(
        task_id="discharge",
        name="Discharge Planning",
        dependencies=["treatment"],
        handler=make_llm_task(
            "You are a discharge planner. Create discharge criteria and follow-up plan.",
            "Based on the treatment plan, create discharge criteria.",
            context_keys=["treatment", "assessment"],
        ),
    ))

    return wf


def demo_basic_dag():
    """Basic DAG workflow for admission."""
    print("\n" + "=" * 70)
    print("  DEMO 1: BASIC DAG WORKFLOW — ADMISSION WORKUP")
    print("=" * 70)
    print("""
  An admission workup as a DAG:

    Level 0: Labs, Meds, History          ← All independent, run in parallel
    Level 1: Assessment                    ← Depends on ALL of Level 0
    Level 2: Consult (conditional), Treat  ← Depends on Assessment
    Level 3: Discharge Planning            ← Depends on Treatment

  Tasks at the same level run in PARALLEL.
  Tasks at different levels run in ORDER (respecting dependencies).
  """)

    wf = build_admission_workflow()

    # Show the DAG structure
    print(f"\n  DAG Structure:")
    print(f"  {wf.visualize()}")

    # Execute
    result = wf.run(
        initial_context={"scenario": SCENARIO_ACS},
        verbose=True,
        parallel=True,
    )

    # Show key results
    print(f"\n  ── Key Results ──")
    for task_id, task in wf.tasks.items():
        if task.status == TaskStatus.COMPLETED:
            preview = str(task.result)[:150].replace("\n", " | ")
            print(f"\n  [{task.name}]: {preview}...")
        elif task.status == TaskStatus.SKIPPED:
            print(f"\n  [{task.name}]: SKIPPED (condition not met)")


def demo_parallel_vs_sequential():
    """Compare parallel vs sequential execution of the same DAG."""
    print("\n" + "=" * 70)
    print("  DEMO 2: PARALLEL vs SEQUENTIAL EXECUTION")
    print("=" * 70)
    print("""
  Same DAG, two execution strategies:
  1. PARALLEL: Independent tasks run at the same time
  2. SEQUENTIAL: Tasks run one at a time, even when independent

  This demonstrates the performance benefit of DAG-aware execution.
  """)

    # Sequential
    print(f"\n  ═══ SEQUENTIAL EXECUTION ═══")
    wf_seq = build_admission_workflow()
    start = time.time()
    result_seq = wf_seq.run(
        initial_context={"scenario": SCENARIO_ACS},
        verbose=True,
        parallel=False,
    )
    seq_time = time.time() - start

    # Parallel
    print(f"\n\n  ═══ PARALLEL EXECUTION ═══")
    wf_par = build_admission_workflow()
    start = time.time()
    result_par = wf_par.run(
        initial_context={"scenario": SCENARIO_ACS},
        verbose=True,
        parallel=True,
    )
    par_time = time.time() - start

    # Comparison
    speedup = seq_time / par_time if par_time > 0 else 0
    print(f"\n\n  ═══ COMPARISON ═══")
    print(f"  {'Metric':<30} {'Sequential':>12} {'Parallel':>12}")
    print(f"  {'─' * 54}")
    print(f"  {'Wall-clock time':<30} {seq_time:>10.2f}s {par_time:>10.2f}s")
    print(f"  {'Speedup':<30} {'':>12} {speedup:>10.2f}x")
    print(f"\n  Parallel is faster because Level 0 (Labs, Meds, History)")
    print(f"  runs 3 LLM calls simultaneously instead of sequentially.")


def demo_conditional_edges():
    """Show conditional edges in a DAG."""
    print("\n" + "=" * 70)
    print("  DEMO 3: CONDITIONAL EDGES — DYNAMIC WORKFLOW")
    print("=" * 70)
    print("""
  Some tasks only run if a condition is met at runtime.
  The DAG adapts based on intermediate results.

  Example: The specialist consult only triggers if the
  assessment says "high risk" or "critical."
  """)

    wf = build_admission_workflow()

    print(f"\n  DAG has a conditional task: 'Specialist Consult'")
    print(f"  Condition: Only if assessment mentions 'high risk' or 'critical'\n")

    result = wf.run(
        initial_context={"scenario": SCENARIO_ACS},
        verbose=True,
        parallel=True,
    )

    consult_task = wf.tasks.get("consult")
    if consult_task:
        if consult_task.status == TaskStatus.SKIPPED:
            print(f"\n  🔹 Consult was SKIPPED — assessment didn't flag high risk")
        elif consult_task.status == TaskStatus.COMPLETED:
            print(f"\n  🔸 Consult was TRIGGERED — assessment flagged high risk")
            preview = str(consult_task.result)[:200].replace("\n", " | ")
            print(f"     Consult result: {preview}...")


def demo_failure_and_retry():
    """Show failure handling and retry in a DAG."""
    print("\n" + "=" * 70)
    print("  DEMO 4: FAILURE HANDLING & RETRY")
    print("=" * 70)
    print("""
  What happens when a task in the DAG fails?
  - If the task has retries configured, it retries
  - If it still fails, dependent tasks may be affected
  - The workflow reports partial completion

  We'll inject a failing task to demonstrate.
  """)

    wf = DAGWorkflow(name="Failure Demo")

    # Normal task
    wf.add_task(DAGTask(
        task_id="step_a",
        name="Step A (succeeds)",
        handler=lambda ctx: "Step A completed successfully",
    ))

    # Task that fails then succeeds (simulating transient error)
    call_count = {"n": 0}

    def flaky_handler(ctx):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise ConnectionError(f"Simulated failure (attempt {call_count['n']})")
        return "Step B completed after retries"

    wf.add_task(DAGTask(
        task_id="step_b",
        name="Step B (flaky, 3 retries)",
        handler=flaky_handler,
        retry_count=3,
    ))

    # Task that depends on the flaky one
    wf.add_task(DAGTask(
        task_id="step_c",
        name="Step C (depends on B)",
        dependencies=["step_b"],
        handler=lambda ctx: f"Step C using B's result: {ctx.get('result_step_b', 'N/A')}",
    ))

    # Task that always fails
    def always_fails(ctx):
        raise ValueError("This task always fails — permanent error")

    wf.add_task(DAGTask(
        task_id="step_d",
        name="Step D (always fails)",
        handler=always_fails,
        retry_count=1,
    ))

    # Task dependent on the always-failing task
    wf.add_task(DAGTask(
        task_id="step_e",
        name="Step E (depends on D)",
        dependencies=["step_d"],
        handler=lambda ctx: "Step E would run here",
    ))

    result = wf.run(verbose=True, parallel=False)

    print(f"\n  Execution log:")
    for entry in result["execution_log"]:
        status_icon = {"completed": "✅", "failed": "❌", "skipped": "⏭️"}.get(entry["status"], "?")
        attempts = f" ({entry.get('attempts', 1)} attempts)" if entry.get("attempts", 1) > 1 else ""
        error = f" — {entry.get('error', '')}" if entry.get("error") else ""
        print(f"    {status_icon} {entry['name']}: {entry['status']}{attempts}{error}")


def demo_dag_visualization():
    """Visualize different DAG structures."""
    print("\n" + "=" * 70)
    print("  DEMO 5: DAG STRUCTURES COMPARED")
    print("=" * 70)
    print("""
  Not all workflows are the same shape:

  1. PIPELINE (linear): A → B → C → D
  2. FAN-OUT:  A → B, C, D (parallel) → E
  3. DIAMOND:  A → B, C → D  (fork-join)
  4. COMPLEX:  Mixed dependencies + conditionals

  Each structure has different parallelism potential.
  """)

    # Pipeline DAG
    pipeline = DAGWorkflow(name="Pipeline")
    for i, name in enumerate(["Triage", "Labs", "Assessment", "Treatment"]):
        deps = [f"step_{i-1}"] if i > 0 else []
        pipeline.add_task(DAGTask(f"step_{i}", name, lambda c: "done", deps))
    levels = pipeline.topological_sort()
    print(f"\n  PIPELINE: {len(levels)} levels, max parallelism = 1")
    print(f"  {pipeline.visualize()}")

    # Fan-out DAG
    fanout = DAGWorkflow(name="Fan-Out")
    fanout.add_task(DAGTask("start", "Start", lambda c: "done"))
    for name in ["Cardio", "Nephro", "Endo", "Pharm"]:
        fanout.add_task(DAGTask(name.lower(), name, lambda c: "done", ["start"]))
    fanout.add_task(DAGTask("merge", "Merge", lambda c: "done", ["cardio", "nephro", "endo", "pharm"]))
    levels = fanout.topological_sort()
    print(f"\n  FAN-OUT: {len(levels)} levels, max parallelism = 4")
    print(f"  {fanout.visualize()}")

    # Diamond DAG
    diamond = DAGWorkflow(name="Diamond")
    diamond.add_task(DAGTask("a", "Start", lambda c: "done"))
    diamond.add_task(DAGTask("b", "Path-Left", lambda c: "done", ["a"]))
    diamond.add_task(DAGTask("c", "Path-Right", lambda c: "done", ["a"]))
    diamond.add_task(DAGTask("d", "Join", lambda c: "done", ["b", "c"]))
    levels = diamond.topological_sort()
    print(f"\n  DIAMOND: {len(levels)} levels, max parallelism = 2")
    print(f"  {diamond.visualize()}")

    # Complex (the admission workflow)
    admission = build_admission_workflow()
    levels = admission.topological_sort()
    print(f"\n  COMPLEX (Admission): {len(levels)} levels, max parallelism = 3")
    print(f"  {admission.visualize()}")


def demo_interactive():
    """Interactive DAG workflow."""
    print("\n" + "=" * 70)
    print("  DEMO 6: INTERACTIVE — RUN YOUR OWN DAG WORKFLOW")
    print("=" * 70)
    print("  Enter a patient scenario. The admission DAG will execute.")
    print("  Type 'quit' to exit.\n")

    while True:
        scenario = input("  Scenario (or 'quit'): ").strip()

        if scenario.lower() in ['quit', 'exit', 'q']:
            break
        if len(scenario) < 20:
            print("  Please provide a detailed scenario.")
            continue

        wf = build_admission_workflow()
        result = wf.run(
            initial_context={"scenario": scenario},
            verbose=True,
            parallel=True,
        )

        # Show final outputs
        for task_id, task in wf.tasks.items():
            if task.status == TaskStatus.COMPLETED:
                print(f"\n  ── {task.name} ──")
                for line in str(task.result).split("\n")[:5]:
                    print(f"  {line}")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 7: DAG DEPENDENCY WORKFLOWS")
    print("=" * 70)
    print("""
    Build workflows as directed acyclic graphs (DAGs).
    Tasks with dependencies wait. Independent tasks run in parallel.
    Conditional edges make workflows dynamic.

    Choose a demo:
      1 → Basic DAG (admission workup)
      2 → Parallel vs Sequential (timing comparison)
      3 → Conditional edges (dynamic workflow)
      4 → Failure handling & retry
      5 → DAG structures compared (pipeline, fan-out, diamond)
      6 → Interactive (enter your own scenario)
      7 → Run demos 1-5
    """)

    choice = input("  Enter choice (1-7): ").strip()

    demos = {
        "1": demo_basic_dag,
        "2": demo_parallel_vs_sequential,
        "3": demo_conditional_edges,
        "4": demo_failure_and_retry,
        "5": demo_dag_visualization,
        "6": demo_interactive,
    }

    if choice == "7":
        for demo in [demo_basic_dag, demo_parallel_vs_sequential,
                      demo_conditional_edges, demo_failure_and_retry,
                      demo_dag_visualization]:
            demo()
    elif choice in demos:
        demos[choice]()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. WORKFLOWS ARE GRAPHS, NOT LISTS
   Most real workflows have parallel paths, conditional branches, and
   merge points. A flat list (step 1 → 2 → 3) can't express this.
   A DAG can: Lab ┐
               Med ├→ Assessment → Treatment
           History ┘

2. TOPOLOGICAL SORT gives you the correct execution order.
   Kahn's algorithm (BFS) also gives you LEVELS — tasks within
   each level can run in parallel. This is free parallelism.

3. CONDITIONAL EDGES make workflows dynamic.
   "Only run the specialist consult if the assessment says high risk"
   This is like clinical decision trees — not every patient takes
   the same path through the workup.

4. FAILURE PROPAGATION: When a task fails, its dependents are affected.
   Options: skip dependents, retry the failed task, use a fallback,
   or mark the workflow as "partial." The right choice depends on
   which task failed and how critical it is.

5. PARALLEL EXECUTION of independent tasks is the main performance win.
   If Labs, Meds, and History each take 2s:
   - Sequential: 6s
   - Parallel: 2s (all three at once)
   The DAG structure TELLS you what can parallelize.

6. DAG vs OTHER PATTERNS:
   - DAG vs Pipeline: Pipeline is a degenerate DAG (no branches).
   - DAG vs Hierarchical: Hierarchy is a tree (parent → children).
     DAG allows multiple parents (diamond join).
   - DAG vs P2P: DAG has a fixed structure. P2P is unstructured.

7. REAL-WORLD DAGs:
   - Airflow, Prefect, Dagster — all DAG-based workflow engines
   - GitHub Actions — DAG of jobs
   - Terraform — DAG of resources
   - Medical care pathways — DAGs with conditional branches

8. VALIDATE BEFORE EXECUTE: Always check for cycles and missing
   dependencies before running. A cycle = infinite loop.
"""

if __name__ == "__main__":
    main()
