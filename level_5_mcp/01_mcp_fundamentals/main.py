"""
Level 5 - Project 01: MCP Fundamentals
========================================

Understanding the Model Context Protocol — the universal standard for connecting
AI agents to external tools and data sources.

Builds on: Level 3 (Agents) and Level 4 (Evaluation/Guardrails).

MCP defines a client-server protocol over JSON-RPC 2.0 that lets any AI host
application discover and invoke tools, read resources, and use prompt templates
from any MCP-compatible server. This module walks through:

- The MCP architecture (host → client → server)
- JSON-RPC message format and protocol lifecycle
- Building a minimal MCP server with the Python SDK
- Simulating client-server communication

We simulate protocol messages here because running a live MCP server requires
a transport layer (stdio or SSE). The code shows exactly what the messages
look like on the wire, and how to define servers that produce them.

Usage:
    python main.py
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

# We import FastMCP for server definitions (Demo 2).
# If mcp is not installed, demos 1/3/4 still work with simulated messages.
try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Note: 'mcp' package not installed. Install with: pip install mcp")
    print("      Demos will use simulated protocol messages.\n")


# ============================================================================
# Helper: pretty-print JSON-RPC messages
# ============================================================================

def print_json(label: str, obj: dict, indent: int = 2):
    """Pretty-print a JSON object with a label."""
    print(f"\n  {label}:")
    for line in json.dumps(obj, indent=indent).split("\n"):
        print(f"    {line}")


def print_banner(title: str):
    """Print a section banner."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ============================================================================
# Healthcare data used across demos
# ============================================================================

HEALTHCARE_TOOLS = {
    "calculate_bmi": {
        "description": "Calculate Body Mass Index from weight (kg) and height (m). "
                       "Returns BMI value and WHO category.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "weight_kg": {
                    "type": "number",
                    "description": "Patient weight in kilograms"
                },
                "height_m": {
                    "type": "number",
                    "description": "Patient height in meters"
                }
            },
            "required": ["weight_kg", "height_m"]
        }
    },
    "lookup_medication": {
        "description": "Look up medication information including class, common doses, "
                       "and side effects. Use when a clinician asks about a drug.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "medication_name": {
                    "type": "string",
                    "description": "Name of the medication to look up"
                }
            },
            "required": ["medication_name"]
        }
    },
    "interpret_lab_value": {
        "description": "Interpret a laboratory test result. Returns whether the value "
                       "is normal, low, or high, with clinical significance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "test_name": {
                    "type": "string",
                    "description": "Name of the lab test (e.g., 'hemoglobin', 'glucose')"
                },
                "value": {
                    "type": "number",
                    "description": "The numeric lab result value"
                },
                "unit": {
                    "type": "string",
                    "description": "Unit of measurement (e.g., 'g/dL', 'mg/dL')"
                }
            },
            "required": ["test_name", "value", "unit"]
        }
    },
    "check_drug_interaction": {
        "description": "Check for known interactions between two medications. "
                       "Returns severity and clinical recommendation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "drug_a": {
                    "type": "string",
                    "description": "First medication name"
                },
                "drug_b": {
                    "type": "string",
                    "description": "Second medication name"
                }
            },
            "required": ["drug_a", "drug_b"]
        }
    }
}

MEDICATION_DB = {
    "metformin": {
        "class": "Biguanide",
        "indication": "Type 2 Diabetes",
        "common_dose": "500-2000mg daily",
        "side_effects": ["nausea", "diarrhea", "lactic acidosis (rare)"],
    },
    "lisinopril": {
        "class": "ACE Inhibitor",
        "indication": "Hypertension, Heart Failure",
        "common_dose": "10-40mg daily",
        "side_effects": ["dry cough", "hyperkalemia", "angioedema (rare)"],
    },
    "atorvastatin": {
        "class": "Statin",
        "indication": "Hyperlipidemia",
        "common_dose": "10-80mg daily",
        "side_effects": ["myalgia", "elevated liver enzymes", "rhabdomyolysis (rare)"],
    },
}

LAB_RANGES = {
    "hemoglobin": {"unit": "g/dL", "low": 12.0, "high": 17.5, "critical_low": 7.0, "critical_high": 20.0},
    "glucose": {"unit": "mg/dL", "low": 70, "high": 100, "critical_low": 40, "critical_high": 500},
    "potassium": {"unit": "mEq/L", "low": 3.5, "high": 5.0, "critical_low": 2.5, "critical_high": 6.5},
    "creatinine": {"unit": "mg/dL", "low": 0.7, "high": 1.3, "critical_low": 0.4, "critical_high": 10.0},
}


# ============================================================================
# DEMO 1: MCP Architecture Explained
# ============================================================================

def demo_mcp_architecture():
    """DEMO 1: Walk through MCP protocol layers, message types, JSON-RPC format."""
    print_banner("DEMO 1: MCP Architecture Explained")

    print("""
  The Model Context Protocol defines three layers:

  ┌─────────────────────────────────────────────────────────────┐
  │  HOST APPLICATION (e.g., Claude Desktop, VS Code, your app) │
  │                                                             │
  │   ┌───────────┐   ┌───────────┐   ┌───────────┐           │
  │   │ MCP Client│   │ MCP Client│   │ MCP Client│           │
  │   │     #1    │   │     #2    │   │     #3    │           │
  │   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘           │
  └─────────┼───────────────┼───────────────┼──────────────────┘
            │               │               │
            ▼               ▼               ▼
      ┌──────────┐   ┌──────────┐   ┌──────────┐
      │MCP Server│   │MCP Server│   │MCP Server│
      │ (Tools)  │   │(Resources│   │ (Prompts)│
      └──────────┘   └──────────┘   └──────────┘

  Each MCP Client maintains a 1:1 connection with one MCP Server.
  A Host can have multiple Clients, each connected to a different Server.
    """)

    print("  MCP defines three capability types:")
    print("  ─────────────────────────────────────────────────────")
    print("  1. TOOLS      — Functions the server exposes (actions)")
    print("                   Example: calculate_bmi, lookup_drug")
    print("  2. RESOURCES  — Read-only data the server provides")
    print("                   Example: clinical-guidelines://hypertension")
    print("  3. PROMPTS    — Reusable prompt templates")
    print("                   Example: summarize_note, triage_patient")

    print("\n  All messages use JSON-RPC 2.0 format:")
    print("  ─────────────────────────────────────────────────────")

    example_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "calculate_bmi",
            "arguments": {
                "weight_kg": 82.5,
                "height_m": 1.75
            }
        }
    }
    print_json("Request (Client → Server)", example_request)

    example_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "bmi": 26.9,
                        "category": "Overweight",
                        "interpretation": "BMI 25.0-29.9 indicates overweight"
                    })
                }
            ]
        }
    }
    print_json("Response (Server → Client)", example_response)

    example_notification = {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {
            "progressToken": "task-1",
            "progress": 50,
            "total": 100
        }
    }
    print_json("Notification (no response expected)", example_notification)


# ============================================================================
# DEMO 2: Simple MCP Server
# ============================================================================

def demo_simple_server():
    """DEMO 2: Build a minimal MCP server with FastMCP."""
    print_banner("DEMO 2: Simple MCP Server with FastMCP")

    if MCP_AVAILABLE:
        # Actually define a server using the SDK
        mcp_server = FastMCP("Healthcare Demo Server")

        @mcp_server.tool()
        def greet_patient(patient_name: str, department: str = "General") -> str:
            """Greet a patient by name, optionally specifying the department.
            Use this tool when you need to generate a welcome message."""
            return f"Welcome to the {department} department, {patient_name}! " \
                   f"A healthcare provider will be with you shortly."

        @mcp_server.resource("info://hospital/departments")
        def list_departments() -> str:
            """List all hospital departments."""
            departments = [
                "Emergency", "Cardiology", "Neurology", "Orthopedics",
                "Pediatrics", "Oncology", "Radiology", "Pharmacy"
            ]
            return json.dumps({"departments": departments})

        print("  ✓ Created FastMCP server: 'Healthcare Demo Server'")
        print("  ✓ Registered tool: greet_patient(patient_name, department)")
        print("  ✓ Registered resource: info://hospital/departments")

        print("\n  Server code (this is what you'd write):")
        print("  " + "─" * 55)
        print("""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("Healthcare Demo Server")

    @mcp.tool()
    def greet_patient(patient_name: str, department: str = "General") -> str:
        \\"\\"\\"Greet a patient by name.\\"\\"\\"
        return f"Welcome to {department}, {patient_name}!"

    @mcp.resource("info://hospital/departments")
    def list_departments() -> str:
        \\"\\"\\"List all hospital departments.\\"\\"\\"
        return json.dumps({"departments": [...]})

    # To run: mcp.run()  (starts stdio transport by default)
        """)
    else:
        print("  [mcp package not installed — showing server code only]")
        print("""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("Healthcare Demo Server")

    @mcp.tool()
    def greet_patient(patient_name: str, department: str = "General") -> str:
        \\"\\"\\"Greet a patient by name.\\"\\"\\"
        return f"Welcome to {department}, {patient_name}!"

    @mcp.resource("info://hospital/departments")
    def list_departments() -> str:
        \\"\\"\\"List all hospital departments.\\"\\"\\"
        return json.dumps({"departments": [...]})
        """)

    print("  When this server runs, it advertises:")
    print("    • 1 tool  (greet_patient)")
    print("    • 1 resource (info://hospital/departments)")
    print("  Any MCP client can discover and use them via the protocol.")


# ============================================================================
# DEMO 3: MCP Client Communication
# ============================================================================

def demo_client_communication():
    """DEMO 3: Walk through the full JSON-RPC message flow."""
    print_banner("DEMO 3: MCP Client Communication (Protocol Flow)")

    print("\n  Full protocol lifecycle:")
    print("  " + "─" * 55)

    # Step 1: Initialize
    print("\n  STEP 1: Initialize Handshake")
    print("  Client sends capabilities, server responds with its capabilities.")

    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True}
            },
            "clientInfo": {
                "name": "HealthcareAgent",
                "version": "1.0.0"
            }
        }
    }
    print_json("Client → Server (initialize)", init_request)

    init_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"subscribe": True}
            },
            "serverInfo": {
                "name": "ClinicalToolsServer",
                "version": "1.0.0"
            }
        }
    }
    print_json("Server → Client (initialize result)", init_response)

    # Step 2: Initialized notification
    print("\n  STEP 2: Initialized Notification")
    initialized_notif = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }
    print_json("Client → Server (initialized)", initialized_notif)

    # Step 3: List tools
    print("\n  STEP 3: Discover Available Tools")
    list_tools_req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    }
    print_json("Client → Server (tools/list)", list_tools_req)

    tools_list = []
    for name, info in HEALTHCARE_TOOLS.items():
        tools_list.append({
            "name": name,
            "description": info["description"],
            "inputSchema": info["inputSchema"]
        })

    list_tools_resp = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"tools": tools_list}
    }
    print_json("Server → Client (tools list)", list_tools_resp)

    # Step 4: Call a tool
    print("\n  STEP 4: Call a Tool")
    call_req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "calculate_bmi",
            "arguments": {"weight_kg": 82.5, "height_m": 1.75}
        }
    }
    print_json("Client → Server (tools/call)", call_req)

    bmi = round(82.5 / (1.75 ** 2), 1)
    category = "Overweight" if 25 <= bmi < 30 else "Normal"
    call_resp = {
        "jsonrpc": "2.0",
        "id": 3,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "bmi": bmi,
                        "category": category,
                        "interpretation": f"BMI {bmi} is in the {category} range"
                    })
                }
            ]
        }
    }
    print_json("Server → Client (tool result)", call_resp)

    print("\n  ✓ Complete lifecycle: initialize → discover → call → result")


# ============================================================================
# DEMO 4: Interactive MCP Explorer
# ============================================================================

def demo_interactive_explorer():
    """DEMO 4: Let user explore tools — pick one, see schema, simulate call."""
    print_banner("DEMO 4: Interactive MCP Explorer")

    print("\n  Available tools on the simulated MCP server:")
    tools = list(HEALTHCARE_TOOLS.keys())
    for i, name in enumerate(tools, 1):
        desc = HEALTHCARE_TOOLS[name]["description"][:70]
        print(f"    {i}. {name} — {desc}...")

    choice = input("\n  Select a tool (1-4) or press Enter for #1: ").strip()
    if choice in ("1", "2", "3", "4"):
        selected = tools[int(choice) - 1]
    else:
        selected = tools[0]

    tool_info = HEALTHCARE_TOOLS[selected]
    print(f"\n  Selected: {selected}")
    print(f"  Description: {tool_info['description']}")
    print_json("Auto-generated schema", tool_info["inputSchema"])

    # Simulate a call
    print(f"\n  Simulating a tools/call for '{selected}'...")

    if selected == "calculate_bmi":
        args = {"weight_kg": 78.0, "height_m": 1.72}
        bmi = round(78.0 / (1.72 ** 2), 1)
        if bmi < 18.5:
            cat = "Underweight"
        elif bmi < 25:
            cat = "Normal"
        elif bmi < 30:
            cat = "Overweight"
        else:
            cat = "Obese"
        result_data = {"bmi": bmi, "category": cat}

    elif selected == "lookup_medication":
        args = {"medication_name": "metformin"}
        result_data = MEDICATION_DB["metformin"]

    elif selected == "interpret_lab_value":
        args = {"test_name": "hemoglobin", "value": 10.5, "unit": "g/dL"}
        ref = LAB_RANGES["hemoglobin"]
        status = "low" if 10.5 < ref["low"] else "normal"
        result_data = {"test": "hemoglobin", "value": 10.5, "status": status,
                       "reference_range": f"{ref['low']}-{ref['high']} {ref['unit']}"}

    elif selected == "check_drug_interaction":
        args = {"drug_a": "metformin", "drug_b": "lisinopril"}
        result_data = {
            "drug_a": "metformin", "drug_b": "lisinopril",
            "interaction": "minor",
            "description": "ACE inhibitors may slightly enhance hypoglycemic effect of metformin",
            "recommendation": "Monitor blood glucose; usually safe to co-prescribe"
        }
    else:
        args = {}
        result_data = {"note": "Unknown tool"}

    # Build the JSON-RPC call message
    call_msg = {
        "jsonrpc": "2.0",
        "id": 42,
        "method": "tools/call",
        "params": {"name": selected, "arguments": args}
    }
    print_json("Request sent", call_msg)

    response_msg = {
        "jsonrpc": "2.0",
        "id": 42,
        "result": {
            "content": [{"type": "text", "text": json.dumps(result_data, indent=2)}]
        }
    }
    print_json("Response received", response_msg)

    print(f"\n  ✓ Tool '{selected}' called successfully (simulated)")


# ============================================================================
# Main Menu
# ============================================================================

def main():
    """Run MCP fundamentals demos interactively."""
    print("=" * 70)
    print("  Level 5 - Project 01: MCP Fundamentals")
    print("  Understanding the Model Context Protocol")
    print("=" * 70)

    demos = {
        "1": ("MCP Architecture Explained", demo_mcp_architecture),
        "2": ("Simple MCP Server", demo_simple_server),
        "3": ("MCP Client Communication", demo_client_communication),
        "4": ("Interactive MCP Explorer", demo_interactive_explorer),
    }

    while True:
        print("\nAvailable Demos:")
        for key, (name, _) in demos.items():
            print(f"  {key}. {name}")
        print("  A. Run all demos")
        print("  Q. Quit")

        choice = input("\nSelect demo (1-4, A, Q): ").strip().upper()

        if choice == "Q":
            print("\nGoodbye!")
            break
        elif choice == "A":
            for key in sorted(demos.keys()):
                demos[key][1]()
        elif choice in demos:
            demos[choice][1]()
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
