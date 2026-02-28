"""
Exercise 3: MCP Server Registry
===================================

Skills practiced:
- Building a dynamic server registry for MCP servers
- Service discovery — registering servers and querying capabilities
- Health status monitoring with heartbeat tracking
- Capability matching — find servers that provide specific tools

Healthcare context:
In a hospital with dozens of clinical systems exposed as MCP servers, you
need a registry/catalog to manage them. New servers can register themselves,
agents can discover what servers are available, and administrators can
monitor health status. This is the MCP equivalent of a service mesh or
API gateway for healthcare AI.

Usage:
    python exercise_3_server_registry.py
"""

import os
import json
import time
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
# Server Registry Data Store
# ============================================================================

class MCPServerRegistry:
    """Registry/catalog for MCP servers with health monitoring."""

    def __init__(self):
        self.servers = {}
        self.health_log = []

    def register_server(self, server_id: str, name: str, description: str,
                        endpoint: str, version: str, department: str,
                        tools: list, resources: list = None,
                        contact: str = "") -> dict:
        """Register a new MCP server in the catalog."""
        if server_id in self.servers:
            return {"error": f"Server '{server_id}' already registered. Use update_server."}

        entry = {
            "server_id": server_id,
            "name": name,
            "description": description,
            "endpoint": endpoint,
            "version": version,
            "department": department,
            "tools": tools,
            "resources": resources or [],
            "contact": contact,
            "status": "registered",
            "health": "unknown",
            "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_heartbeat": None,
            "uptime_checks": 0,
            "uptime_successes": 0,
        }
        self.servers[server_id] = entry

        self.health_log.append({
            "timestamp": entry["registered_at"],
            "server_id": server_id,
            "event": "registered",
            "detail": f"Server '{name}' registered with {len(tools)} tools",
        })

        return {"success": True, "server_id": server_id,
                "message": f"Server '{name}' registered with {len(tools)} tools"}

    def unregister_server(self, server_id: str) -> dict:
        """Remove a server from the registry."""
        if server_id not in self.servers:
            return {"error": f"Server '{server_id}' not found in registry"}

        name = self.servers[server_id]["name"]
        del self.servers[server_id]

        self.health_log.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "server_id": server_id,
            "event": "unregistered",
            "detail": f"Server '{name}' removed from registry",
        })

        return {"success": True, "message": f"Server '{name}' unregistered"}

    def update_health(self, server_id: str, status: str, latency_ms: float = 0) -> dict:
        """Record a health check result for a server."""
        if server_id not in self.servers:
            return {"error": f"Server '{server_id}' not found"}

        server = self.servers[server_id]
        server["uptime_checks"] += 1
        server["last_heartbeat"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if status == "healthy":
            server["health"] = "healthy"
            server["status"] = "active"
            server["uptime_successes"] += 1
        elif status == "degraded":
            server["health"] = "degraded"
            server["status"] = "active"
            server["uptime_successes"] += 1
        else:
            server["health"] = "unhealthy"
            server["status"] = "error"

        uptime_pct = (server["uptime_successes"] / server["uptime_checks"] * 100
                      if server["uptime_checks"] > 0 else 0)

        self.health_log.append({
            "timestamp": server["last_heartbeat"],
            "server_id": server_id,
            "event": f"health_check:{status}",
            "latency_ms": latency_ms,
            "detail": f"Uptime: {uptime_pct:.1f}%",
        })

        return {
            "server_id": server_id,
            "health": server["health"],
            "latency_ms": latency_ms,
            "uptime_pct": round(uptime_pct, 1),
        }

    def discover_servers(self, department: str = "", tool_name: str = "",
                         health_filter: str = "") -> dict:
        """Discover servers by department, tool name, or health status."""
        matches = list(self.servers.values())

        if department:
            matches = [s for s in matches
                       if s["department"].lower() == department.lower()]

        if tool_name:
            matches = [s for s in matches
                       if any(t["name"].lower() == tool_name.lower()
                              for t in s["tools"])]

        if health_filter:
            matches = [s for s in matches
                       if s["health"] == health_filter]

        return {
            "query": {"department": department, "tool_name": tool_name,
                      "health_filter": health_filter},
            "result_count": len(matches),
            "servers": [
                {
                    "server_id": s["server_id"],
                    "name": s["name"],
                    "department": s["department"],
                    "health": s["health"],
                    "tool_count": len(s["tools"]),
                    "tools": [t["name"] for t in s["tools"]],
                    "endpoint": s["endpoint"],
                }
                for s in matches
            ],
        }

    def get_server_detail(self, server_id: str) -> dict:
        """Get full details about a registered server."""
        if server_id not in self.servers:
            return {"error": f"Server '{server_id}' not found"}
        return dict(self.servers[server_id])

    def get_capabilities_catalog(self) -> dict:
        """Get a catalog of all available tools across all servers."""
        catalog = {}
        for server in self.servers.values():
            for tool in server["tools"]:
                tool_name = tool["name"]
                if tool_name not in catalog:
                    catalog[tool_name] = []
                catalog[tool_name].append({
                    "server_id": server["server_id"],
                    "server_name": server["name"],
                    "health": server["health"],
                    "description": tool.get("description", ""),
                })

        return {
            "total_tools": len(catalog),
            "total_servers": len(self.servers),
            "tools": catalog,
        }

    def get_health_dashboard(self) -> dict:
        """Get an overview of all server health statuses."""
        total = len(self.servers)
        healthy = sum(1 for s in self.servers.values() if s["health"] == "healthy")
        degraded = sum(1 for s in self.servers.values() if s["health"] == "degraded")
        unhealthy = sum(1 for s in self.servers.values() if s["health"] == "unhealthy")
        unknown = sum(1 for s in self.servers.values() if s["health"] == "unknown")

        servers_status = []
        for s in self.servers.values():
            uptime = (s["uptime_successes"] / s["uptime_checks"] * 100
                      if s["uptime_checks"] > 0 else 0)
            servers_status.append({
                "server_id": s["server_id"],
                "name": s["name"],
                "health": s["health"],
                "status": s["status"],
                "last_heartbeat": s["last_heartbeat"],
                "uptime_pct": round(uptime, 1),
                "checks": s["uptime_checks"],
            })

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_servers": total,
            "healthy": healthy,
            "degraded": degraded,
            "unhealthy": unhealthy,
            "unknown": unknown,
            "overall_health": "healthy" if unhealthy == 0 and degraded == 0
                              else "degraded" if unhealthy == 0
                              else "critical",
            "servers": servers_status,
        }

    def get_health_log(self, server_id: str = "", limit: int = 20) -> list:
        """Get recent health log entries."""
        logs = self.health_log
        if server_id:
            logs = [l for l in logs if l["server_id"] == server_id]
        return logs[-limit:]


# ============================================================================
# Pre-configured Healthcare MCP Servers
# ============================================================================

HEALTHCARE_SERVERS = [
    {
        "server_id": "ehr_server",
        "name": "EHR Server",
        "description": "Electronic Health Record server — patient demographics, "
                       "encounters, vitals, and problem lists.",
        "endpoint": "mcp://hospital.local:8001/ehr",
        "version": "2.1.0",
        "department": "Clinical",
        "contact": "ehr-team@hospital.local",
        "tools": [
            {"name": "lookup_patient", "description": "Look up patient demographics and problems"},
            {"name": "get_encounters", "description": "Get encounter history for a patient"},
            {"name": "get_vitals_trend", "description": "Get vital signs trend over time"},
        ],
        "resources": [
            {"uri": "ehr://patient/{id}", "description": "Patient record resource"},
        ],
    },
    {
        "server_id": "lab_server",
        "name": "Clinical Lab Server",
        "description": "Laboratory information system — orders, results, "
                       "interpretations, and critical value alerts.",
        "endpoint": "mcp://hospital.local:8002/lab",
        "version": "1.8.3",
        "department": "Clinical",
        "contact": "lab-systems@hospital.local",
        "tools": [
            {"name": "get_lab_results", "description": "Get lab results for a patient"},
            {"name": "get_critical_values", "description": "Get critical lab values requiring attention"},
            {"name": "order_lab_test", "description": "Place a new lab order"},
        ],
    },
    {
        "server_id": "pharmacy_server",
        "name": "Pharmacy Server",
        "description": "Pharmacy management — formulary, interactions, prior auth, "
                       "and refill management.",
        "endpoint": "mcp://hospital.local:8003/pharmacy",
        "version": "3.0.1",
        "department": "Pharmacy",
        "contact": "pharmacy-it@hospital.local",
        "tools": [
            {"name": "get_medications", "description": "Get current medication list"},
            {"name": "check_interaction", "description": "Check drug-drug interactions"},
            {"name": "check_formulary", "description": "Check formulary status and tier"},
            {"name": "request_prior_auth", "description": "Submit prior authorization request"},
        ],
    },
    {
        "server_id": "scheduling_server",
        "name": "Scheduling Server",
        "description": "Appointment scheduling — availability, booking, cancellation.",
        "endpoint": "mcp://hospital.local:8004/scheduling",
        "version": "1.5.0",
        "department": "Operations",
        "contact": "scheduling-team@hospital.local",
        "tools": [
            {"name": "check_availability", "description": "Check provider appointment availability"},
            {"name": "book_appointment", "description": "Book a patient appointment"},
            {"name": "cancel_appointment", "description": "Cancel an existing appointment"},
            {"name": "get_upcoming", "description": "Get upcoming appointments for a patient"},
        ],
    },
    {
        "server_id": "radiology_server",
        "name": "Radiology Server",
        "description": "Radiology information — imaging orders, reports, and results.",
        "endpoint": "mcp://hospital.local:8005/radiology",
        "version": "1.2.0",
        "department": "Clinical",
        "contact": "rad-it@hospital.local",
        "tools": [
            {"name": "get_imaging_results", "description": "Get radiology reports for a patient"},
            {"name": "order_imaging", "description": "Place a radiology order"},
        ],
    },
    {
        "server_id": "billing_server",
        "name": "Billing Server",
        "description": "Revenue cycle — claims, coding, and payment status.",
        "endpoint": "mcp://hospital.local:8006/billing",
        "version": "2.0.0",
        "department": "Finance",
        "contact": "billing-it@hospital.local",
        "tools": [
            {"name": "get_claims", "description": "Get claims for a patient encounter"},
            {"name": "validate_coding", "description": "Validate ICD-10 and CPT codes"},
        ],
    },
]


# ============================================================================
# Helper Functions
# ============================================================================

def print_banner(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_json(label: str, obj, indent: int = 2):
    print(f"\n  {label}:")
    text = json.dumps(obj, indent=indent) if isinstance(obj, (dict, list)) else str(obj)
    for line in text.split("\n"):
        print(f"    {line}")


# ============================================================================
# MCP Server Definition for Registry
# ============================================================================

def define_registry_mcp_server(registry: MCPServerRegistry):
    """Define a Registry MCP Server (if SDK is available)."""
    if not MCP_AVAILABLE:
        print("  ⚠ MCP SDK not installed — skipping MCP server definition")
        return None

    reg_server = FastMCP("Server Registry")

    @reg_server.tool()
    def mcp_register_server(server_id: str, name: str, description: str,
                            endpoint: str, version: str, department: str,
                            tools_json: str) -> str:
        """Register a new MCP server in the registry."""
        tools = json.loads(tools_json) if isinstance(tools_json, str) else tools_json
        return json.dumps(registry.register_server(
            server_id, name, description, endpoint, version, department, tools
        ), indent=2)

    @reg_server.tool()
    def mcp_discover_servers(department: str = "", tool_name: str = "",
                             health_filter: str = "") -> str:
        """Discover registered MCP servers by department, tool, or health."""
        return json.dumps(registry.discover_servers(
            department, tool_name, health_filter
        ), indent=2)

    @reg_server.tool()
    def mcp_get_health_dashboard() -> str:
        """Get the health dashboard for all registered servers."""
        return json.dumps(registry.get_health_dashboard(), indent=2)

    @reg_server.tool()
    def mcp_get_capabilities() -> str:
        """Get a catalog of all tools available across all servers."""
        return json.dumps(registry.get_capabilities_catalog(), indent=2)

    print("  ✓ Registry MCP Server defined with 4 tools")
    return reg_server


# ============================================================================
# Section 1: Server Registration
# ============================================================================

def section_registration(registry: MCPServerRegistry):
    """Register all healthcare MCP servers."""
    print_banner("Section 1: Server Registration")

    print("""
  Registering all healthcare MCP servers in the registry.
  Each server declares its tools, resources, and metadata.
    """)

    for server_def in HEALTHCARE_SERVERS:
        result = registry.register_server(
            server_id=server_def["server_id"],
            name=server_def["name"],
            description=server_def["description"],
            endpoint=server_def["endpoint"],
            version=server_def["version"],
            department=server_def["department"],
            tools=server_def["tools"],
            resources=server_def.get("resources", []),
            contact=server_def.get("contact", ""),
        )
        if result.get("success"):
            print(f"  ✓ {result['message']}")
        else:
            print(f"  ✗ {result.get('error')}")

    print(f"\n  Total servers registered: {len(registry.servers)}")


# ============================================================================
# Section 2: Service Discovery
# ============================================================================

def section_discovery(registry: MCPServerRegistry):
    """Demonstrate service discovery capabilities."""
    print_banner("Section 2: Service Discovery")

    print("""
  Agents and clients can discover servers by department, tool capabilities,
  or health status. This enables dynamic tool routing at runtime.
    """)

    # Discover by department
    print(f"\n  --- Discovery by Department ---")
    for dept in ["Clinical", "Pharmacy", "Operations", "Finance"]:
        result = registry.discover_servers(department=dept)
        servers = result["servers"]
        names = [s["name"] for s in servers]
        print(f"  {dept}: {len(servers)} server(s) — {', '.join(names)}")

    # Discover by tool name
    print(f"\n  --- Discovery by Tool ---")
    tools_to_find = ["get_medications", "lookup_patient", "check_availability",
                     "get_critical_values", "validate_coding"]
    for tool_name in tools_to_find:
        result = registry.discover_servers(tool_name=tool_name)
        if result["result_count"] > 0:
            server = result["servers"][0]
            print(f"  '{tool_name}' → {server['name']} ({server['server_id']})")
        else:
            print(f"  '{tool_name}' → not found")

    # Full capabilities catalog
    print(f"\n  --- Capabilities Catalog ---")
    catalog = registry.get_capabilities_catalog()
    print(f"  Total unique tools: {catalog['total_tools']}")
    print(f"  Total servers:      {catalog['total_servers']}")
    print(f"\n  Tools by server:")
    for tool_name, providers in catalog["tools"].items():
        prov_list = ", ".join(p["server_name"] for p in providers)
        print(f"    • {tool_name} → {prov_list}")


# ============================================================================
# Section 3: Health Monitoring
# ============================================================================

def section_health_monitoring(registry: MCPServerRegistry):
    """Simulate health checks and monitoring."""
    print_banner("Section 3: Health Monitoring")

    print("""
  Health monitoring simulates periodic heartbeat checks for each server.
  Servers can be healthy, degraded, or unhealthy based on response times.
    """)

    import random
    random.seed(42)

    # Simulate multiple health check rounds
    for round_num in range(1, 4):
        print(f"\n  --- Health Check Round {round_num} ---")
        for server_id in registry.servers:
            # Simulate various health scenarios
            r = random.random()
            if server_id == "radiology_server" and round_num >= 2:
                # Simulate radiology going down
                status = "unhealthy"
                latency = 5000.0
            elif r < 0.7:
                status = "healthy"
                latency = random.uniform(5, 50)
            elif r < 0.9:
                status = "degraded"
                latency = random.uniform(200, 1000)
            else:
                status = "healthy"
                latency = random.uniform(30, 80)

            result = registry.update_health(server_id, status, latency)
            indicator = {"healthy": "✓", "degraded": "⚠", "unhealthy": "✗"}.get(
                result["health"], "?")
            print(f"    {indicator} {server_id:<25} {result['health']:<12} "
                  f"{result['latency_ms']:>8.1f}ms  uptime: {result['uptime_pct']}%")

    # Show dashboard
    print(f"\n  --- Health Dashboard ---")
    dashboard = registry.get_health_dashboard()
    print(f"  Overall: {dashboard['overall_health'].upper()}")
    print(f"  Healthy: {dashboard['healthy']} | Degraded: {dashboard['degraded']} | "
          f"Unhealthy: {dashboard['unhealthy']} | Unknown: {dashboard['unknown']}")

    print(f"\n  Server Status:")
    for s in dashboard["servers"]:
        indicator = {"healthy": "✓", "degraded": "⚠", "unhealthy": "✗",
                     "unknown": "?"}.get(s["health"], "?")
        hb = s["last_heartbeat"] or "never"
        print(f"    {indicator} {s['name']:<25} {s['health']:<12} "
              f"uptime: {s['uptime_pct']:>5.1f}%  last: {hb}")


# ============================================================================
# Section 4: Intelligent Routing
# ============================================================================

def section_intelligent_routing(registry: MCPServerRegistry):
    """Demonstrate intelligent routing based on registry data."""
    print_banner("Section 4: Intelligent Routing")

    print("""
  An agent uses the registry to find the best server for each query.
  It considers tool availability AND server health when routing.
    """)

    # Clinical queries that need routing
    queries = [
        {"query": "What are the patient's latest lab results?",
         "needed_tool": "get_lab_results", "category": "labs"},
        {"query": "Check for drug interactions with metformin",
         "needed_tool": "check_interaction", "category": "pharmacy"},
        {"query": "Book an appointment with Dr. Chen",
         "needed_tool": "book_appointment", "category": "scheduling"},
        {"query": "Get the radiology report for chest X-ray",
         "needed_tool": "get_imaging_results", "category": "radiology"},
        {"query": "Look up patient demographics",
         "needed_tool": "lookup_patient", "category": "ehr"},
    ]

    for q in queries:
        print(f"\n  Query: \"{q['query']}\"")
        print(f"  Needed tool: {q['needed_tool']}")

        # Find servers with the needed tool
        result = registry.discover_servers(tool_name=q["needed_tool"])
        if result["result_count"] == 0:
            print(f"  → No server found with tool '{q['needed_tool']}'")
            continue

        # Pick the healthiest server
        candidates = result["servers"]
        # Prefer healthy servers
        healthy = [s for s in candidates if s["health"] == "healthy"]
        degraded = [s for s in candidates if s["health"] == "degraded"]

        if healthy:
            chosen = healthy[0]
            print(f"  → Routing to: {chosen['name']} (healthy)")
        elif degraded:
            chosen = degraded[0]
            print(f"  → Routing to: {chosen['name']} (degraded — may be slow)")
        else:
            chosen = candidates[0]
            print(f"  ⚠ Routing to: {chosen['name']} ({chosen['health']}) — "
                  f"WARNING: server may be unavailable")

        print(f"    Endpoint: {chosen.get('endpoint', 'N/A')}")

    # Show routing decision for the unhealthy server
    print(f"\n  --- Failover Scenario ---")
    rad_result = registry.discover_servers(tool_name="get_imaging_results")
    if rad_result["result_count"] > 0:
        server = rad_result["servers"][0]
        if server["health"] == "unhealthy":
            print(f"  Radiology server is UNHEALTHY — agent should:")
            print(f"    1. Notify the user that imaging data is temporarily unavailable")
            print(f"    2. Log the failure for the ops team")
            print(f"    3. Suggest retrying in a few minutes")
            print(f"    4. Offer alternative: check recent encounters for imaging notes")


# ============================================================================
# Section 5: Registry Management
# ============================================================================

def section_registry_management(registry: MCPServerRegistry):
    """Demonstrate registry management operations."""
    print_banner("Section 5: Registry Management")

    print("""
  Administrative operations: add new servers, remove decommissioned
  servers, view audit logs, and generate registry reports.
    """)

    # Register a new server dynamically
    print(f"  --- Adding New Server ---")
    result = registry.register_server(
        server_id="notifications_server",
        name="Notifications Server",
        description="Patient and provider notifications — alerts, reminders, messages.",
        endpoint="mcp://hospital.local:8007/notifications",
        version="1.0.0",
        department="Operations",
        tools=[
            {"name": "send_alert", "description": "Send a clinical alert to a provider"},
            {"name": "send_reminder", "description": "Send an appointment reminder to a patient"},
        ],
        contact="notifications-team@hospital.local",
    )
    print(f"  ✓ {result.get('message', result.get('error'))}")

    # Update its health
    registry.update_health("notifications_server", "healthy", 12.0)

    # Show current registry state
    print(f"\n  --- Registry Summary ---")
    dashboard = registry.get_health_dashboard()
    print(f"  Total servers: {dashboard['total_servers']}")

    for s in dashboard["servers"]:
        tools = registry.servers[s["server_id"]]["tools"]
        tool_names = [t["name"] for t in tools]
        indicator = {"healthy": "✓", "degraded": "⚠", "unhealthy": "✗",
                     "unknown": "?"}.get(s["health"], "?")
        print(f"    {indicator} {s['name']:<28} v{registry.servers[s['server_id']]['version']:<8} "
              f"[{s['health']}]  tools: {', '.join(tool_names)}")

    # Decommission billing server
    print(f"\n  --- Decommissioning Server ---")
    result = registry.unregister_server("billing_server")
    print(f"  {result.get('message', result.get('error'))}")
    print(f"  Servers remaining: {len(registry.servers)}")

    # Show audit log
    print(f"\n  --- Recent Audit Log ---")
    logs = registry.get_health_log(limit=10)
    for log_entry in logs[-8:]:
        print(f"    [{log_entry['timestamp']}] {log_entry['server_id']}: "
              f"{log_entry['event']} — {log_entry['detail']}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run the Server Registry exercise."""
    print("=" * 70)
    print("  Exercise 3: MCP Server Registry")
    print("  Dynamic server catalog with discovery and health monitoring")
    print("=" * 70)

    # Create the registry
    registry = MCPServerRegistry()
    define_registry_mcp_server(registry)

    sections = {
        "1": ("Server Registration", lambda: section_registration(registry)),
        "2": ("Service Discovery", lambda: section_discovery(registry)),
        "3": ("Health Monitoring", lambda: section_health_monitoring(registry)),
        "4": ("Intelligent Routing", lambda: section_intelligent_routing(registry)),
        "5": ("Registry Management", lambda: section_registry_management(registry)),
    }

    # Auto-register if running sections 2-5 without 1
    first_run = True

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
            # Auto-register servers if needed
            if choice != "1" and not registry.servers and first_run:
                print("\n  (Auto-registering servers for demo...)")
                section_registration(registry)
                first_run = False
            sections[choice][1]()
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
