"""
Exercise 9: Saga Pattern — Long-Running Transactions with Compensating Actions

Skills practiced:
- Implementing multi-step transactions that can partially fail
- Designing compensating actions (rollbacks) for each forward step
- Understanding the difference between orchestrated sagas and choreographed sagas
- Handling partial failures gracefully in clinical workflows
- Building audit trails for compliance

What is a Saga?
  A Saga is a sequence of steps where:
  - Each step has a FORWARD action (do the thing)
  - Each step has a COMPENSATING action (undo the thing)
  - If step N fails, you automatically run the compensating actions
    for steps N-1, N-2, ..., 1 — in reverse order

  This is how you handle "transactions" that span multiple services/agents
  when you can't use a simple database rollback.

Architecture:

  Forward:
    Step 1 ──→ Step 2 ──→ Step 3 ──→ Step 4 ──→ SUCCESS
    (order)    (verify)   (approve)  (dispense)

  If Step 3 fails:
    Step 3 FAILS ✗
      │
      ▼
    Compensate Step 2 (cancel verification)
      │
      ▼
    Compensate Step 1 (cancel order)
      │
      ▼
    SAGA FAILED — all steps rolled back

Healthcare parallel:
  Medication ordering saga:
    1. Create order → (compensate: cancel the order)
    2. Verify with pharmacy → (compensate: notify pharmacy of cancel)
    3. Approve by attending → (compensate: revoke approval)
    4. Dispense → (compensate: recall medication)

  If attending rejects at step 3, we must:
    - Notify pharmacy the verification is void (undo step 2)
    - Cancel the original order (undo step 1)

  Without a saga, you end up with orphaned orders, phantom verifications,
  and medications that were "approved" but shouldn't have been.
"""

import os
import json
import time
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

client = OpenAI()


# ============================================================
# Saga Framework
# ============================================================

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"


@dataclass
class SagaStep:
    """A single step in a saga with forward and compensating actions."""
    name: str
    forward_action: callable  # Do the thing
    compensate_action: callable  # Undo the thing
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    error: str = ""
    compensate_result: str = ""
    started_at: str = ""
    completed_at: str = ""


@dataclass
class SagaLog:
    """Audit trail for a saga execution."""
    saga_name: str
    started_at: str = ""
    completed_at: str = ""
    status: str = "pending"
    entries: list = field(default_factory=list)

    def add(self, step_name: str, action: str, status: str, detail: str = ""):
        self.entries.append({
            "timestamp": datetime.now().isoformat(),
            "step": step_name,
            "action": action,
            "status": status,
            "detail": detail[:200],
        })


class SagaOrchestrator:
    """
    Orchestrated Saga: a central coordinator runs steps in order.
    If a step fails, it runs compensating actions in REVERSE order.

    This is the most common saga implementation for LLM agent systems.
    """

    def __init__(self, name: str, steps: list[SagaStep]):
        self.name = name
        self.steps = steps
        self.log = SagaLog(saga_name=name)
        self.completed_steps = []

    def execute(self, context: dict, verbose: bool = True) -> dict:
        """
        Execute the saga. Returns result with status, completed steps,
        and compensation details if any step failed.
        """
        self.log.started_at = datetime.now().isoformat()
        self.log.status = "running"

        if verbose:
            print(f"\n  ╔══════════════════════════════════════════╗")
            print(f"  ║  SAGA: {self.name:<34}║")
            print(f"  ╚══════════════════════════════════════════╝")

        for i, step in enumerate(self.steps):
            step.status = StepStatus.RUNNING
            step.started_at = datetime.now().isoformat()
            self.log.add(step.name, "forward", "started")

            if verbose:
                print(f"\n  Step {i + 1}/{len(self.steps)}: {step.name}")
                print(f"    ⏳ Executing forward action...")

            try:
                result = step.forward_action(context)
                step.result = result
                step.status = StepStatus.COMPLETED
                step.completed_at = datetime.now().isoformat()
                self.completed_steps.append(step)
                self.log.add(step.name, "forward", "completed", result)

                # Make result available in context for subsequent steps
                context[f"step_{i + 1}_result"] = result

                if verbose:
                    print(f"    ✅ Completed: {result[:100].replace(chr(10), ' | ')}")

            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = str(e)
                self.log.add(step.name, "forward", "failed", str(e))

                if verbose:
                    print(f"    ❌ FAILED: {str(e)[:100]}")
                    print(f"\n  ═══ SAGA FAILURE — INITIATING COMPENSATION ═══")
                    print(f"  Must undo {len(self.completed_steps)} completed step(s)")

                # Compensate in REVERSE order
                self._compensate(context, verbose)

                self.log.status = "rolled_back"
                self.log.completed_at = datetime.now().isoformat()

                return {
                    "status": "rolled_back",
                    "failed_step": step.name,
                    "error": str(e),
                    "compensated_steps": [s.name for s in self.completed_steps],
                    "log": self.log,
                }

        self.log.status = "completed"
        self.log.completed_at = datetime.now().isoformat()

        if verbose:
            print(f"\n  ═══ SAGA COMPLETED SUCCESSFULLY ═══")
            print(f"  All {len(self.steps)} steps completed.")

        return {
            "status": "completed",
            "steps": len(self.steps),
            "results": {s.name: s.result for s in self.steps},
            "log": self.log,
        }

    def _compensate(self, context: dict, verbose: bool = True):
        """Run compensating actions in reverse order."""
        for step in reversed(self.completed_steps):
            step.status = StepStatus.COMPENSATING
            self.log.add(step.name, "compensate", "started")

            if verbose:
                print(f"\n    🔄 Compensating: {step.name}")

            try:
                comp_result = step.compensate_action(context)
                step.compensate_result = comp_result
                step.status = StepStatus.COMPENSATED
                self.log.add(step.name, "compensate", "completed", comp_result)

                if verbose:
                    print(f"    ↩️  Undone: {comp_result[:100].replace(chr(10), ' | ')}")

            except Exception as e:
                self.log.add(step.name, "compensate", "failed", str(e))
                if verbose:
                    print(f"    ⚠️  Compensation failed: {str(e)[:100]}")
                    print(f"    (In production, this would trigger a manual review)")


# ============================================================
# Clinical Saga: Medication Ordering
# ============================================================

def build_medication_saga(should_fail_at: int = 0) -> SagaOrchestrator:
    """
    Build a medication ordering saga.

    Args:
        should_fail_at: Step number to simulate failure (0 = no failure)
    """

    def create_order(ctx):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an ordering physician. Create a medication order. Be concise (2-3 lines)."},
                {"role": "user", "content": f"Create order for: {ctx.get('medication', 'Aspirin 325mg')} for patient with {ctx.get('diagnosis', 'ACS')}. Include dose, route, frequency."},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    def cancel_order(ctx):
        return f"ORDER CANCELLED: {ctx.get('medication', 'medication')} order voided. Reason: Saga rollback."

    def verify_order(ctx):
        if should_fail_at == 2:
            raise Exception("PHARMACY VERIFICATION FAILED: Drug interaction detected with existing Metformin — increased lactic acidosis risk.")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a pharmacist. Verify this order for safety. Be concise (2-3 lines)."},
                {"role": "user", "content": f"Verify order: {ctx.get('step_1_result', 'medication order')}\nPatient meds: Metformin, Lisinopril, Atorvastatin"},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    def cancel_verification(ctx):
        return "VERIFICATION CANCELLED: Pharmacy notified that previous verification is void."

    def approve_order(ctx):
        if should_fail_at == 3:
            raise Exception("ATTENDING REJECTED: Patient has documented allergy to this medication class.")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an attending physician. Approve or flag concerns for this order. Be concise (1-2 lines)."},
                {"role": "user", "content": f"Approve order: {ctx.get('step_1_result', 'order')}\nPharmacy verification: {ctx.get('step_2_result', 'verified')}"},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    def revoke_approval(ctx):
        return "APPROVAL REVOKED: Attending's approval marked as void. Clinical note updated."

    def dispense(ctx):
        if should_fail_at == 4:
            raise Exception("DISPENSE FAILED: Medication out of stock. Alternative needed.")

        return f"DISPENSED: {ctx.get('medication', 'Medication')} dispensed per verified order. Barcode scan confirmed."

    def recall_medication(ctx):
        return "MEDICATION RECALLED: Pharmacy notified to retrieve dispensed medication. Patient notified."

    steps = [
        SagaStep("Create Order", create_order, cancel_order),
        SagaStep("Pharmacy Verification", verify_order, cancel_verification),
        SagaStep("Attending Approval", approve_order, revoke_approval),
        SagaStep("Dispense Medication", dispense, recall_medication),
    ]

    return SagaOrchestrator("Medication Ordering", steps)


# ============================================================
# Clinical Saga: Patient Admission
# ============================================================

def build_admission_saga(should_fail_at: int = 0) -> SagaOrchestrator:
    """Build a patient admission saga."""

    def assign_bed(ctx):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a bed manager. Assign a bed for this admission. Be concise (1-2 lines)."},
                {"role": "user", "content": f"Assign bed for: {ctx.get('patient', 'patient')} with {ctx.get('diagnosis', 'ACS')}. Acuity: {ctx.get('acuity', 'high')}."},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    def release_bed(ctx):
        return "BED RELEASED: Bed assignment cancelled, marked available in system."

    def create_admission_orders(ctx):
        if should_fail_at == 2:
            raise Exception("ORDER SYSTEM DOWN: Unable to create admission order set. System timeout.")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an admitting physician. Create admission orders. Be concise (3-4 lines)."},
                {"role": "user", "content": f"Admission orders for ACS patient.\nBed: {ctx.get('step_1_result', 'assigned')}\nInclude: diet, activity, monitoring, key meds."},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    def cancel_admission_orders(ctx):
        return "ADMISSION ORDERS CANCELLED: All pending orders voided. Pharmacy and nursing notified."

    def notify_care_team(ctx):
        if should_fail_at == 3:
            raise Exception("NOTIFICATION FAILED: Unable to reach cardiology on-call.")

        return (
            "CARE TEAM NOTIFIED: "
            "Attending: paged. Cardiology: consulted. "
            "Nursing: assignment received. Case manager: notified."
        )

    def retract_notifications(ctx):
        return "NOTIFICATIONS RETRACTED: Care team notified admission is cancelled."

    def schedule_procedures(ctx):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a procedure scheduler. Schedule needed procedures. Be concise (2 lines)."},
                {"role": "user", "content": f"Schedule for ACS patient: cardiac catheterization if troponin rising, echocardiogram within 24h."},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    def cancel_procedures(ctx):
        return "PROCEDURES CANCELLED: Cath lab and echo lab notified. Slots released."

    steps = [
        SagaStep("Assign Bed", assign_bed, release_bed),
        SagaStep("Create Admission Orders", create_admission_orders, cancel_admission_orders),
        SagaStep("Notify Care Team", notify_care_team, retract_notifications),
        SagaStep("Schedule Procedures", schedule_procedures, cancel_procedures),
    ]

    return SagaOrchestrator("Patient Admission", steps)


# ============================================================
# Demo Functions
# ============================================================

def demo_successful_saga():
    """Show a saga completing successfully."""
    print("\n" + "=" * 70)
    print("  DEMO 1: SUCCESSFUL SAGA — ALL STEPS COMPLETE")
    print("=" * 70)
    print("""
  A medication ordering saga where everything goes right.
  Each step completes → next step runs → saga succeeds.
  """)

    saga = build_medication_saga(should_fail_at=0)
    context = {
        "medication": "Aspirin 325mg",
        "diagnosis": "Acute Coronary Syndrome",
        "patient": "John Smith, 55yo",
    }
    result = saga.execute(context)
    print(f"\n  Final status: {result['status']}")


def demo_failed_saga_rollback():
    """Show a saga failing and rolling back."""
    print("\n" + "=" * 70)
    print("  DEMO 2: SAGA FAILURE — AUTOMATIC ROLLBACK")
    print("=" * 70)
    print("""
  A medication saga where step 3 (Attending Approval) FAILS.
  Watch the saga automatically undo steps 2 and 1 in reverse order.

  Forward:  Create Order ✅ → Pharmacy Verify ✅ → Attending Approve ❌
  Rollback:                    Cancel Verify ↩️  ← Cancel Order ↩️
  """)

    saga = build_medication_saga(should_fail_at=3)
    context = {
        "medication": "Clopidogrel 75mg",
        "diagnosis": "ACS — NSTEMI",
        "patient": "John Smith, 55yo",
    }
    result = saga.execute(context)
    print(f"\n  Final status: {result['status']}")
    print(f"  Failed at: {result.get('failed_step', 'N/A')}")
    print(f"  Error: {result.get('error', 'N/A')[:100]}")
    print(f"  Steps compensated: {result.get('compensated_steps', [])}")


def demo_admission_saga_failure():
    """Show patient admission saga with notification failure."""
    print("\n" + "=" * 70)
    print("  DEMO 3: ADMISSION SAGA — NOTIFICATION FAILURE")
    print("=" * 70)
    print("""
  A patient admission saga where step 3 (Notify Care Team) fails.
  Bed must be released, admission orders must be cancelled.

  Forward:  Assign Bed ✅ → Admission Orders ✅ → Notify Team ❌
  Rollback:                  Cancel Orders ↩️  ← Release Bed ↩️
  """)

    saga = build_admission_saga(should_fail_at=3)
    context = {
        "patient": "John Smith, 55yo",
        "diagnosis": "ACS — NSTEMI",
        "acuity": "high",
    }
    result = saga.execute(context)
    print(f"\n  Final status: {result['status']}")
    print(f"  Failed at: {result.get('failed_step', 'N/A')}")


def demo_audit_trail():
    """Show the audit trail that a saga produces."""
    print("\n" + "=" * 70)
    print("  DEMO 4: SAGA AUDIT TRAIL — COMPLIANCE LOGGING")
    print("=" * 70)
    print("""
  Every saga produces a detailed audit trail:
  - What step ran, when, what it returned
  - If compensation occurred: what was undone, when, by whom

  This is CRITICAL for healthcare compliance (HIPAA, Joint Commission).
  """)

    # Run a saga that fails
    saga = build_medication_saga(should_fail_at=2)
    context = {
        "medication": "Metformin 500mg",
        "diagnosis": "Type 2 DM with ACS",
        "patient": "John Smith, 55yo",
    }
    result = saga.execute(context, verbose=False)

    log = result["log"]
    print(f"\n  SAGA: {log.saga_name}")
    print(f"  Status: {log.status}")
    print(f"  Started: {log.started_at}")
    print(f"  Completed: {log.completed_at}")
    print(f"\n  AUDIT TRAIL ({len(log.entries)} entries):")
    print(f"  {'#':<3} {'Timestamp':<15} {'Step':<25} {'Action':<12} {'Status':<12} {'Detail'}")
    print(f"  {'─' * 100}")

    for i, entry in enumerate(log.entries):
        ts = entry["timestamp"].split("T")[1][:12]
        detail = entry["detail"][:40].replace("\n", " ") if entry["detail"] else ""
        print(f"  {i + 1:<3} {ts:<15} {entry['step']:<25} {entry['action']:<12} {entry['status']:<12} {detail}")

    print(f"\n  This audit trail can be stored for compliance and investigation.")


def demo_comparison_of_failures():
    """Compare saga at different failure points."""
    print("\n" + "=" * 70)
    print("  DEMO 5: COMPARISON — FAILURE AT EACH STEP")
    print("=" * 70)
    print("""
  What happens when the medication saga fails at different steps?
  More completed steps → more compensations needed.
  """)

    for fail_at in [2, 3, 4]:
        saga = build_medication_saga(should_fail_at=fail_at)
        context = {
            "medication": "Heparin drip",
            "diagnosis": "ACS",
            "patient": "John Smith, 55yo",
        }
        result = saga.execute(context, verbose=False)

        compensated = result.get("compensated_steps", [])
        print(f"\n  Failure at step {fail_at} ({result.get('failed_step', 'N/A')}):")
        print(f"    Steps that ran: {fail_at - 1}")
        print(f"    Steps compensated: {len(compensated)} → {compensated}")
        print(f"    Error: {result.get('error', 'N/A')[:80]}")


def demo_interactive():
    """Interactive saga builder."""
    print("\n" + "=" * 70)
    print("  DEMO 6: INTERACTIVE — EXPLORE SAGA FAILURES")
    print("=" * 70)
    print("  Choose a scenario and failure point.\n")

    while True:
        print("  Scenarios: 'med' (medication ordering), 'admit' (admission)")
        print("  Enter: <scenario> <fail_step> (e.g., 'med 3')")
        print("  Or enter 0 for success, 'quit' to exit.")

        user_input = input("\n  > ").strip().lower()
        if user_input in ['quit', 'exit', 'q']:
            break

        parts = user_input.split()
        if len(parts) != 2:
            print("  Format: <scenario> <step>. Example: 'med 3'")
            continue

        scenario_type, fail_step = parts[0], int(parts[1])

        if scenario_type == "med":
            saga = build_medication_saga(should_fail_at=fail_step)
            ctx = {"medication": "Heparin drip", "diagnosis": "ACS", "patient": "John Smith"}
        elif scenario_type == "admit":
            saga = build_admission_saga(should_fail_at=fail_step)
            ctx = {"patient": "John Smith", "diagnosis": "ACS", "acuity": "high"}
        else:
            print("  Unknown scenario. Use 'med' or 'admit'.")
            continue

        saga.execute(ctx)


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 9: SAGA PATTERN")
    print("  Long-Running Transactions with Compensating Actions")
    print("=" * 70)
    print("""
    The Saga Pattern handles multi-step workflows that can partially fail.
    Each step has a FORWARD action and a COMPENSATING action (rollback).
    If step N fails, steps N-1...1 are automatically undone.

    Choose a demo:
      1 → Successful saga (all steps complete)
      2 → Failed saga with rollback (step 3 fails)
      3 → Admission saga failure
      4 → Audit trail (compliance logging)
      5 → Compare failures at each step
      6 → Interactive (choose scenario + failure point)
      7 → Run demos 1-5
    """)

    choice = input("  Enter choice (1-7): ").strip()

    demos = {
        "1": demo_successful_saga,
        "2": demo_failed_saga_rollback,
        "3": demo_admission_saga_failure,
        "4": demo_audit_trail,
        "5": demo_comparison_of_failures,
        "6": demo_interactive,
    }

    if choice == "7":
        for demo in [demo_successful_saga, demo_failed_saga_rollback,
                      demo_admission_saga_failure, demo_audit_trail,
                      demo_comparison_of_failures]:
            demo()
    elif choice in demos:
        demos[choice]()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. SAGA = TRANSACTION ACROSS MULTIPLE AGENTS/SERVICES
   When you can't do a single DB rollback (because you're calling
   multiple APIs, agents, or services), a saga gives you the same
   guarantee: either ALL steps succeed, or ALL are undone.

2. EVERY FORWARD STEP NEEDS A COMPENSATING ACTION
   This is the design discipline:
     Forward: Create order → Compensate: Cancel order
     Forward: Reserve bed → Compensate: Release bed
     Forward: Send notification → Compensate: Retract notification
   If you can't design a compensating action, the step shouldn't
   be in the saga.

3. COMPENSATION ORDER MATTERS: ALWAYS REVERSE
   If steps were: 1 → 2 → 3, compensation must be: 3 → 2 → 1.
   Why? Step 2's compensating action may depend on step 3 being
   undone first (e.g., can't cancel an order if medication was
   already dispensed — must recall medication first).

4. AUDIT TRAIL IS BUILT IN
   Every saga automatically produces a compliance-friendly log:
   what ran, when, what succeeded, what failed, what was undone.
   This is gold for HIPAA/Joint Commission audits.

5. TWO FLAVORS OF SAGA:
   - Orchestrated saga (what we built): central coordinator
     manages the steps. Easier to implement, easier to debug.
   - Choreographed saga: each service publishes events, next
     service reacts. More complex, but no single point of failure.

6. SAGAS ARE NOT ACID:
   - No isolation: other processes can see intermediate states
   - No atomicity: you get "eventual consistency" via compensation
   - Compensating actions may fail too (need manual intervention)
   This is a tradeoff — but for distributed systems, it's often
   the only option.

7. HEALTHCARE-SPECIFIC CONCERNS:
   - Some actions CANNOT be compensated (e.g., medication already
     administered). Design your saga with these boundaries.
   - Always log WHY a saga failed for clinical accountability.
   - Consider human-in-the-loop for high-risk compensations.
"""

if __name__ == "__main__":
    main()
