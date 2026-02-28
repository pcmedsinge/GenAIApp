"""
Exercise 1: Scheduling MCP Server
====================================

Skills practiced:
- Building a domain-specific MCP server for appointment scheduling
- Implementing availability checking with time-slot management
- Booking and cancellation workflows with validation
- Retrieving upcoming appointments with filtering

Healthcare context:
Scheduling is critical infrastructure in any healthcare system. Patients
need to book appointments, check provider availability, cancel or
reschedule, and view upcoming visits. This MCP server exposes scheduling
operations as tools that a clinical agent can invoke on behalf of
patients or front-desk staff.

This exercise builds a complete scheduling MCP server with four tools:
check_availability, book_appointment, cancel_appointment, get_upcoming.

Usage:
    python exercise_1_scheduling_server.py
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Note: 'mcp' package not installed. Install with: pip install mcp")
    print("      Exercise will use standalone functions.\n")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ============================================================================
# Provider & Scheduling Database
# ============================================================================

PROVIDERS = {
    "DR_CHEN": {
        "name": "Dr. Sarah Chen", "specialty": "Internal Medicine",
        "department": "Primary Care", "location": "Building A, Suite 200",
    },
    "DR_PARK": {
        "name": "Dr. James Park", "specialty": "Endocrinology",
        "department": "Specialty Care", "location": "Building B, Suite 310",
    },
    "DR_WONG": {
        "name": "Dr. Lisa Wong", "specialty": "Pulmonology",
        "department": "Specialty Care", "location": "Building B, Suite 420",
    },
    "DR_ROBERTS": {
        "name": "Dr. Amy Roberts", "specialty": "Cardiology",
        "department": "Specialty Care", "location": "Building C, Suite 100",
    },
}

# Pre-booked appointments (some slots are taken)
BOOKED_APPOINTMENTS = [
    {"appt_id": "APT-001", "patient_id": "P001", "provider_id": "DR_CHEN",
     "date": "2026-03-02", "time": "09:00", "duration_min": 30,
     "type": "Follow-up", "reason": "Diabetes management", "status": "confirmed"},
    {"appt_id": "APT-002", "patient_id": "P002", "provider_id": "DR_PARK",
     "date": "2026-03-02", "time": "10:00", "duration_min": 45,
     "type": "New Patient", "reason": "Thyroid evaluation", "status": "confirmed"},
    {"appt_id": "APT-003", "patient_id": "P003", "provider_id": "DR_WONG",
     "date": "2026-03-03", "time": "14:00", "duration_min": 30,
     "type": "Follow-up", "reason": "COPD check", "status": "confirmed"},
    {"appt_id": "APT-004", "patient_id": "P001", "provider_id": "DR_ROBERTS",
     "date": "2026-03-05", "time": "11:00", "duration_min": 45,
     "type": "Consult", "reason": "Cardiology referral", "status": "confirmed"},
    {"appt_id": "APT-005", "patient_id": "P003", "provider_id": "DR_CHEN",
     "date": "2026-03-04", "time": "09:30", "duration_min": 30,
     "type": "Follow-up", "reason": "BP recheck", "status": "confirmed"},
]

# Mutable copy for the exercise
appointments_db = list(BOOKED_APPOINTMENTS)
next_appt_id = 6

# Provider schedule: available time slots per day
SCHEDULE_TEMPLATE = {
    "morning_start": "08:00",
    "morning_end": "12:00",
    "afternoon_start": "13:00",
    "afternoon_end": "17:00",
    "slot_duration_min": 30,
}

# Patient names for display
PATIENT_NAMES = {
    "P001": "John Smith",
    "P002": "Maria Garcia",
    "P003": "Robert Wilson",
}


# ============================================================================
# Helper Functions
# ============================================================================

def generate_time_slots(date_str: str) -> list:
    """Generate available time slots for a given date."""
    slots = []
    # Morning slots
    current = datetime.strptime(f"{date_str} {SCHEDULE_TEMPLATE['morning_start']}", "%Y-%m-%d %H:%M")
    morning_end = datetime.strptime(f"{date_str} {SCHEDULE_TEMPLATE['morning_end']}", "%Y-%m-%d %H:%M")
    while current < morning_end:
        slots.append(current.strftime("%H:%M"))
        current += timedelta(minutes=SCHEDULE_TEMPLATE["slot_duration_min"])

    # Afternoon slots
    current = datetime.strptime(f"{date_str} {SCHEDULE_TEMPLATE['afternoon_start']}", "%Y-%m-%d %H:%M")
    afternoon_end = datetime.strptime(f"{date_str} {SCHEDULE_TEMPLATE['afternoon_end']}", "%Y-%m-%d %H:%M")
    while current < afternoon_end:
        slots.append(current.strftime("%H:%M"))
        current += timedelta(minutes=SCHEDULE_TEMPLATE["slot_duration_min"])

    return slots


def get_booked_slots(provider_id: str, date_str: str) -> list:
    """Get already booked time slots for a provider on a date."""
    return [
        a["time"] for a in appointments_db
        if a["provider_id"] == provider_id
        and a["date"] == date_str
        and a["status"] != "cancelled"
    ]


def print_banner(title: str):
    """Print a section banner."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_json(label: str, obj, indent: int = 2):
    """Pretty-print a JSON-serializable object."""
    print(f"\n  {label}:")
    text = json.dumps(obj, indent=indent) if isinstance(obj, (dict, list)) else str(obj)
    for line in text.split("\n"):
        print(f"    {line}")


# ============================================================================
# Scheduling Tool Implementations
# ============================================================================

def check_availability(provider_id: str, date: str) -> dict:
    """Check provider availability for a given date."""
    if provider_id not in PROVIDERS:
        return {"error": f"Provider '{provider_id}' not found",
                "available_providers": list(PROVIDERS.keys())}

    # Weekday check (no weekends)
    dt = datetime.strptime(date, "%Y-%m-%d")
    if dt.weekday() >= 5:
        return {"provider_id": provider_id, "date": date,
                "available": False, "reason": "Clinic closed on weekends",
                "available_slots": []}

    all_slots = generate_time_slots(date)
    booked = get_booked_slots(provider_id, date)
    available = [s for s in all_slots if s not in booked]

    provider = PROVIDERS[provider_id]
    return {
        "provider_id": provider_id,
        "provider_name": provider["name"],
        "specialty": provider["specialty"],
        "location": provider["location"],
        "date": date,
        "total_slots": len(all_slots),
        "booked_slots": len(booked),
        "available_slots": available,
        "available": len(available) > 0,
    }


def book_appointment(patient_id: str, provider_id: str, date: str,
                     time: str, appointment_type: str, reason: str) -> dict:
    """Book an appointment for a patient."""
    global next_appt_id

    # Validate provider
    if provider_id not in PROVIDERS:
        return {"error": f"Provider '{provider_id}' not found"}

    # Check if slot is available
    booked = get_booked_slots(provider_id, date)
    if time in booked:
        return {"error": f"Slot {time} on {date} is already booked for {provider_id}",
                "suggestion": "Use check_availability to find open slots"}

    # Weekday check
    dt = datetime.strptime(date, "%Y-%m-%d")
    if dt.weekday() >= 5:
        return {"error": "Cannot book on weekends"}

    # Create appointment
    appt_id = f"APT-{next_appt_id:03d}"
    next_appt_id += 1

    duration = 45 if appointment_type.lower() in ["new patient", "consult"] else 30

    appointment = {
        "appt_id": appt_id,
        "patient_id": patient_id,
        "provider_id": provider_id,
        "date": date,
        "time": time,
        "duration_min": duration,
        "type": appointment_type,
        "reason": reason,
        "status": "confirmed",
    }
    appointments_db.append(appointment)

    provider = PROVIDERS[provider_id]
    patient_name = PATIENT_NAMES.get(patient_id, patient_id)
    return {
        "success": True,
        "appointment": appointment,
        "message": f"Appointment {appt_id} booked: {patient_name} with "
                   f"{provider['name']} on {date} at {time} ({appointment_type})",
    }


def cancel_appointment(appt_id: str, reason: str = "") -> dict:
    """Cancel an existing appointment."""
    for appt in appointments_db:
        if appt["appt_id"] == appt_id:
            if appt["status"] == "cancelled":
                return {"error": f"Appointment {appt_id} is already cancelled"}

            appt["status"] = "cancelled"
            appt["cancel_reason"] = reason or "Patient requested"

            patient_name = PATIENT_NAMES.get(appt["patient_id"], appt["patient_id"])
            provider = PROVIDERS.get(appt["provider_id"], {})
            return {
                "success": True,
                "appt_id": appt_id,
                "message": f"Cancelled: {patient_name} with {provider.get('name', '?')} "
                           f"on {appt['date']} at {appt['time']}",
                "cancel_reason": appt["cancel_reason"],
            }

    return {"error": f"Appointment {appt_id} not found"}


def get_upcoming(patient_id: str, days_ahead: int = 30) -> dict:
    """Get upcoming appointments for a patient."""
    today = datetime.strptime("2026-02-28", "%Y-%m-%d")
    cutoff = today + timedelta(days=days_ahead)

    upcoming = []
    for appt in appointments_db:
        if appt["patient_id"] != patient_id:
            continue
        if appt["status"] == "cancelled":
            continue
        appt_date = datetime.strptime(appt["date"], "%Y-%m-%d")
        if today <= appt_date <= cutoff:
            provider = PROVIDERS.get(appt["provider_id"], {})
            upcoming.append({
                **appt,
                "provider_name": provider.get("name", "Unknown"),
                "location": provider.get("location", "Unknown"),
            })

    upcoming.sort(key=lambda x: (x["date"], x["time"]))
    patient_name = PATIENT_NAMES.get(patient_id, patient_id)
    return {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "period": f"Next {days_ahead} days (from 2026-02-28)",
        "appointment_count": len(upcoming),
        "appointments": upcoming,
    }


# ============================================================================
# MCP Server Definition
# ============================================================================

def define_scheduling_mcp_server():
    """Define a Scheduling MCP Server (if SDK is available)."""
    if not MCP_AVAILABLE:
        print("  ⚠ MCP SDK not installed — skipping MCP server definition")
        return None

    sched = FastMCP("Scheduling Server")

    @sched.tool()
    def mcp_check_availability(provider_id: str, date: str) -> str:
        """Check appointment availability for a provider on a given date.
        Returns available time slots."""
        return json.dumps(check_availability(provider_id, date), indent=2)

    @sched.tool()
    def mcp_book_appointment(patient_id: str, provider_id: str, date: str,
                             time: str, appointment_type: str, reason: str) -> str:
        """Book an appointment for a patient with a provider."""
        return json.dumps(book_appointment(patient_id, provider_id, date,
                                           time, appointment_type, reason), indent=2)

    @sched.tool()
    def mcp_cancel_appointment(appt_id: str, reason: str = "") -> str:
        """Cancel an existing appointment by ID."""
        return json.dumps(cancel_appointment(appt_id, reason), indent=2)

    @sched.tool()
    def mcp_get_upcoming(patient_id: str, days_ahead: int = 30) -> str:
        """Get upcoming appointments for a patient within the given number of days."""
        return json.dumps(get_upcoming(patient_id, days_ahead), indent=2)

    print("  ✓ Scheduling MCP Server defined with 4 tools")
    return sched


# ============================================================================
# Section 1: Check Availability
# ============================================================================

def section_check_availability():
    """Demonstrate checking provider availability."""
    print_banner("Section 1: Check Availability")

    print("""
  The check_availability tool lets agents find open appointment slots
  for a given provider and date. It handles weekends, existing bookings,
  and returns structured slot data.
    """)

    # Check availability for Dr. Chen on a weekday
    result = check_availability("DR_CHEN", "2026-03-02")
    print(f"  Provider: {result['provider_name']} ({result['specialty']})")
    print(f"  Date: {result['date']}")
    print(f"  Location: {result['location']}")
    print(f"  Slots: {result['total_slots']} total, {result['booked_slots']} booked, "
          f"{len(result['available_slots'])} available")
    print(f"  Available times: {', '.join(result['available_slots'][:8])}...")

    # Check a weekend
    print()
    result_wknd = check_availability("DR_CHEN", "2026-03-01")  # Sunday
    print(f"  Weekend check (2026-03-01): available={result_wknd['available']}")
    if "reason" in result_wknd:
        print(f"    Reason: {result_wknd['reason']}")

    # Check multiple providers
    print(f"\n  --- Availability Summary for 2026-03-03 ---")
    for prov_id in PROVIDERS:
        avail = check_availability(prov_id, "2026-03-03")
        slots_free = len(avail["available_slots"])
        indicator = "✓" if slots_free > 5 else "⚠" if slots_free > 0 else "✗"
        print(f"    {indicator} {avail['provider_name']:<20} {slots_free} slots available")


# ============================================================================
# Section 2: Book Appointments
# ============================================================================

def section_book_appointments():
    """Demonstrate booking appointments."""
    print_banner("Section 2: Book Appointments")

    print("""
  The book_appointment tool creates confirmed appointments. It validates
  the provider, checks slot availability, and assigns appropriate
  durations based on appointment type.
    """)

    # Book a new appointment
    result = book_appointment(
        patient_id="P002",
        provider_id="DR_CHEN",
        date="2026-03-03",
        time="10:30",
        appointment_type="Follow-up",
        reason="Blood pressure check"
    )
    if result.get("success"):
        appt = result["appointment"]
        print(f"  ✓ Booked: {result['message']}")
        print(f"    Duration: {appt['duration_min']} min, Status: {appt['status']}")
    else:
        print(f"  ✗ Error: {result.get('error')}")

    # Book a new patient consult (longer duration)
    result2 = book_appointment(
        patient_id="P001",
        provider_id="DR_PARK",
        date="2026-03-04",
        time="14:00",
        appointment_type="New Patient",
        reason="Endocrine evaluation"
    )
    if result2.get("success"):
        appt = result2["appointment"]
        print(f"\n  ✓ Booked: {result2['message']}")
        print(f"    Duration: {appt['duration_min']} min (extended for new patient)")

    # Try to double-book (should fail)
    print()
    conflict = book_appointment(
        patient_id="P003",
        provider_id="DR_CHEN",
        date="2026-03-02",
        time="09:00",  # Already booked for P001
        appointment_type="Follow-up",
        reason="Test double-book"
    )
    if conflict.get("error"):
        print(f"  ✗ Double-book prevented: {conflict['error']}")


# ============================================================================
# Section 3: Cancel Appointments
# ============================================================================

def section_cancel_appointments():
    """Demonstrate cancelling appointments."""
    print_banner("Section 3: Cancel Appointments")

    print("""
  The cancel_appointment tool changes appointment status and records
  the cancellation reason. It handles already-cancelled and missing
  appointment IDs gracefully.
    """)

    # Cancel an appointment
    result = cancel_appointment("APT-002", reason="Patient rescheduling")
    if result.get("success"):
        print(f"  ✓ {result['message']}")
        print(f"    Reason: {result['cancel_reason']}")
    else:
        print(f"  ✗ {result.get('error')}")

    # Try to cancel the same one again
    print()
    result2 = cancel_appointment("APT-002")
    if result2.get("error"):
        print(f"  ✗ Re-cancel attempt: {result2['error']}")

    # Try a non-existent appointment
    result3 = cancel_appointment("APT-999")
    if result3.get("error"):
        print(f"  ✗ Non-existent: {result3['error']}")

    # Show that the cancelled slot is now open
    print()
    avail = check_availability("DR_PARK", "2026-03-02")
    slot_freed = "10:00" in avail["available_slots"]
    print(f"  Slot 10:00 on 2026-03-02 for DR_PARK now available: {slot_freed}")


# ============================================================================
# Section 4: Get Upcoming Appointments
# ============================================================================

def section_get_upcoming():
    """Demonstrate retrieving upcoming appointments."""
    print_banner("Section 4: Get Upcoming Appointments")

    print("""
  The get_upcoming tool retrieves all scheduled appointments for a
  patient within a specified time window, sorted chronologically.
    """)

    for pid in ["P001", "P002", "P003"]:
        result = get_upcoming(pid, days_ahead=30)
        print(f"\n  --- {result['patient_name']} ({pid}) ---")
        print(f"  Period: {result['period']}")
        print(f"  Upcoming: {result['appointment_count']} appointment(s)")

        for appt in result["appointments"]:
            print(f"    • {appt['date']} {appt['time']} — {appt['type']} with "
                  f"{appt['provider_name']}")
            print(f"      Reason: {appt['reason']} | Location: {appt['location']}")


# ============================================================================
# Section 5: Agent-Driven Scheduling Workflow
# ============================================================================

def section_agent_scheduling():
    """Demonstrate an agent using the scheduling server end to end."""
    print_banner("Section 5: Agent-Driven Scheduling Workflow")

    print("""
  A clinical agent handles a scheduling request end-to-end:
  1. Patient asks to see a cardiologist
  2. Agent checks cardiologist availability
  3. Agent books the first available slot
  4. Agent confirms the appointment details
    """)

    patient_id = "P002"
    provider_id = "DR_ROBERTS"
    target_date = "2026-03-06"

    print(f"  Patient: {PATIENT_NAMES[patient_id]}")
    print(f"  Request: 'I need to see a cardiologist next week'")

    # Step 1: Agent checks availability
    print(f"\n  Step 1: Checking Dr. Roberts availability on {target_date}...")
    avail = check_availability(provider_id, target_date)
    if not avail["available"]:
        print(f"    No slots available. Reason: {avail.get('reason', 'fully booked')}")
        return

    slots = avail["available_slots"]
    print(f"    Found {len(slots)} open slots")
    print(f"    Available: {', '.join(slots[:6])}...")

    # Step 2: Agent picks first morning slot
    chosen_time = slots[0]
    print(f"\n  Step 2: Booking first available slot — {chosen_time}")
    result = book_appointment(
        patient_id=patient_id,
        provider_id=provider_id,
        date=target_date,
        time=chosen_time,
        appointment_type="Consult",
        reason="Cardiology evaluation"
    )

    if result.get("success"):
        appt = result["appointment"]
        print(f"    ✓ {result['message']}")
    else:
        print(f"    ✗ Booking failed: {result.get('error')}")
        return

    # Step 3: Agent confirms
    print(f"\n  Step 3: Confirmation")
    upcoming = get_upcoming(patient_id, days_ahead=30)
    latest = [a for a in upcoming["appointments"]
              if a["provider_id"] == provider_id and a["date"] == target_date]
    if latest:
        a = latest[0]
        print(f"    Confirmed: {a['type']} with {a['provider_name']}")
        print(f"    Date/Time: {a['date']} at {a['time']}")
        print(f"    Location: {a['location']}")
        print(f"    Reason: {a['reason']}")

    # Use OpenAI to generate a patient-friendly confirmation message
    if OPENAI_AVAILABLE:
        print(f"\n  Step 4: Generating patient-friendly confirmation...")
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a friendly medical office assistant. "
                     "Generate a brief, warm appointment confirmation message for the patient. "
                     "Keep it to 3-4 sentences."},
                    {"role": "user", "content": f"Confirm this appointment: "
                     f"{PATIENT_NAMES[patient_id]} has a {appt['type']} with "
                     f"{PROVIDERS[provider_id]['name']} ({PROVIDERS[provider_id]['specialty']}) "
                     f"on {target_date} at {chosen_time} at {PROVIDERS[provider_id]['location']}. "
                     f"Reason: Cardiology evaluation."}
                ],
                max_tokens=200,
            )
            print(f"    {response.choices[0].message.content}")
        except Exception as e:
            print(f"    (OpenAI unavailable: {e})")
    else:
        print(f"\n  Step 4: (OpenAI not available — skipping confirmation message)")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run the Scheduling Server exercise."""
    print("=" * 70)
    print("  Exercise 1: Scheduling MCP Server")
    print("  Appointment management with availability, booking, and cancellation")
    print("=" * 70)

    # Define MCP server
    define_scheduling_mcp_server()

    sections = {
        "1": ("Check Availability", section_check_availability),
        "2": ("Book Appointments", section_book_appointments),
        "3": ("Cancel Appointments", section_cancel_appointments),
        "4": ("Get Upcoming Appointments", section_get_upcoming),
        "5": ("Agent-Driven Scheduling Workflow", section_agent_scheduling),
    }

    while True:
        print("\nSections:")
        for key, (name, _) in sections.items():
            print(f"  {key}. {name}")
        print("  A. Run all sections")
        print("  Q. Quit")

        choice = input("\nSelect section (1-5, A, Q): ").strip().upper()

        if choice == "Q":
            print("\nDone!")
            break
        elif choice == "A":
            for key in sorted(sections.keys()):
                sections[key][1]()
        elif choice in sections:
            sections[choice][1]()
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
