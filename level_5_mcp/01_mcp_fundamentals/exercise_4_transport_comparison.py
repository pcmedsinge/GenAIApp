"""
Exercise 4: MCP Transport Comparison
======================================

Skills practiced:
- Understanding MCP transport mechanisms (stdio vs SSE/HTTP)
- Building server configurations for each transport type
- Comparing latency, security, and deployment characteristics
- Choosing the right transport for different scenarios

Healthcare context:
In healthcare deployments, transport choice matters:
- stdio: Best for local tools on the same machine (clinical workstation apps)
- SSE/HTTP: Required for remote servers (shared lab system, central formulary)
Security requirements (HIPAA, network isolation) often dictate the choice.
This exercise compares both transports and helps decide which fits each scenario.

Usage:
    python exercise_4_transport_comparison.py
"""

import os
import json
import time
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Transport configuration builders
# ---------------------------------------------------------------------------

def build_stdio_config(server_name: str, command: str,
                       args: list[str] = None, env: dict = None) -> dict:
    """
    Build an MCP server configuration for stdio transport.

    stdio transport runs the server as a child process; the client
    communicates via stdin/stdout. Best for local, same-machine tools.

    Args:
        server_name: Name for this server configuration
        command: The command to run (e.g., 'python', 'node')
        args: Command arguments (e.g., ['server.py'])
        env: Environment variables to pass to the server process

    Returns:
        Configuration dictionary compatible with MCP host settings
    """
    config = {
        "name": server_name,
        "transport": "stdio",
        "command": command,
        "args": args or [],
    }
    if env:
        config["env"] = env

    return config


def build_sse_config(server_name: str, url: str,
                     headers: dict = None, timeout_seconds: int = 30) -> dict:
    """
    Build an MCP server configuration for SSE (Server-Sent Events) transport.

    SSE transport connects to a remote HTTP endpoint. The server runs
    independently (perhaps as a microservice), and the client connects
    over the network. Best for shared/remote servers.

    Args:
        server_name: Name for this server configuration
        url: The SSE endpoint URL (e.g., 'http://localhost:8080/sse')
        headers: Optional HTTP headers (e.g., authorization)
        timeout_seconds: Connection timeout

    Returns:
        Configuration dictionary compatible with MCP host settings
    """
    config = {
        "name": server_name,
        "transport": "sse",
        "url": url,
        "timeout": timeout_seconds,
    }
    if headers:
        config["headers"] = headers

    return config


def build_streamable_http_config(server_name: str, url: str,
                                  headers: dict = None) -> dict:
    """
    Build a configuration for the newer Streamable HTTP transport.

    Streamable HTTP is MCP's newer transport that uses standard HTTP
    POST requests with optional streaming responses. It's simpler than
    SSE and better supported by standard HTTP infrastructure.

    Args:
        server_name: Name for this server configuration
        url: The HTTP endpoint URL
        headers: Optional HTTP headers

    Returns:
        Configuration dictionary
    """
    config = {
        "name": server_name,
        "transport": "streamable-http",
        "url": url,
    }
    if headers:
        config["headers"] = headers

    return config


# ---------------------------------------------------------------------------
# Transport characteristic comparison
# ---------------------------------------------------------------------------

TRANSPORT_COMPARISON = {
    "stdio": {
        "name": "Standard I/O (stdio)",
        "how_it_works": "Server runs as child process; client communicates via stdin/stdout",
        "best_for": "Local tools, desktop apps, CLI tools, development",
        "latency": "Very low (in-process pipe)",
        "security": "Inherits OS-level process isolation. No network exposure.",
        "deployment": "Server bundled with the host application",
        "scalability": "One server instance per client (not shared)",
        "healthcare_use_cases": [
            "Clinical workstation calculators",
            "Local EHR data extraction tools",
            "Development and testing of new tools",
            "Single-user desktop AI assistants",
        ],
        "pros": [
            "No network configuration needed",
            "Very low latency",
            "Simple setup — just a command to run",
            "No authentication overhead",
            "Works offline",
        ],
        "cons": [
            "Cannot be shared across machines",
            "Requires server binary on each client machine",
            "No centralized management",
            "Harder to update (must update on each machine)",
        ],
    },
    "sse": {
        "name": "Server-Sent Events (SSE)",
        "how_it_works": "Client connects to HTTP endpoint; server streams events back",
        "best_for": "Remote servers, shared services, microservice architecture",
        "latency": "Moderate (network round-trip)",
        "security": "Requires TLS, authentication headers, network security",
        "deployment": "Server runs as independent service (Docker, cloud, etc.)",
        "scalability": "Shared server serves multiple clients",
        "healthcare_use_cases": [
            "Central hospital formulary server",
            "Shared lab reference range service",
            "Enterprise clinical decision support",
            "Multi-department tool server",
        ],
        "pros": [
            "Shared across many clients",
            "Centralized updates and management",
            "Can run in cloud/container infrastructure",
            "Supports authentication and access control",
        ],
        "cons": [
            "Requires network infrastructure",
            "Higher latency than stdio",
            "Must handle connection drops and reconnection",
            "Requires TLS/authentication for PHI compliance",
        ],
    },
    "streamable-http": {
        "name": "Streamable HTTP",
        "how_it_works": "Standard HTTP POST with optional streaming responses",
        "best_for": "Modern deployments, cloud-native, API gateways",
        "latency": "Moderate (network round-trip)",
        "security": "Standard HTTP security (TLS, auth headers, API keys)",
        "deployment": "Any HTTP server infrastructure",
        "scalability": "Stateless — scales horizontally behind load balancers",
        "healthcare_use_cases": [
            "Cloud-hosted clinical AI services",
            "API gateway for multiple MCP servers",
            "Serverless function deployments",
            "Mobile health application backends",
        ],
        "pros": [
            "Works with standard HTTP infrastructure",
            "Stateless — easy to scale horizontally",
            "Compatible with API gateways and load balancers",
            "Simpler than SSE for request-response patterns",
        ],
        "cons": [
            "Newer transport — less mature tooling",
            "Requires HTTP server setup",
            "Network latency applies",
            "Must implement authentication",
        ],
    },
}


# ---------------------------------------------------------------------------
# Scenario-based recommendations
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "name": "Clinical Workstation AI Assistant",
        "description": "Desktop app on a clinician's workstation that uses local "
                       "clinical calculators and reference tools",
        "factors": {
            "network_required": False,
            "shared_across_users": False,
            "latency_sensitive": True,
            "hipaa_network_concerns": False,
        },
        "recommended_transport": "stdio",
        "reason": "Local tools with no network exposure — simplest and fastest setup",
    },
    {
        "name": "Hospital Formulary Service",
        "description": "Central medication database serving multiple clinical AI "
                       "agents across the hospital network",
        "factors": {
            "network_required": True,
            "shared_across_users": True,
            "latency_sensitive": False,
            "hipaa_network_concerns": True,
        },
        "recommended_transport": "sse",
        "reason": "Shared service with multiple clients — needs network transport "
                  "with proper TLS and authentication for HIPAA",
    },
    {
        "name": "Cloud Clinical Decision Support",
        "description": "Cloud-hosted service providing clinical decision support "
                       "to mobile and web health applications",
        "factors": {
            "network_required": True,
            "shared_across_users": True,
            "latency_sensitive": False,
            "hipaa_network_concerns": True,
        },
        "recommended_transport": "streamable-http",
        "reason": "Cloud-native deployment behind API gateway — Streamable HTTP "
                  "scales horizontally and works with standard infrastructure",
    },
    {
        "name": "Development / Testing",
        "description": "Developer building and testing new MCP tools locally",
        "factors": {
            "network_required": False,
            "shared_across_users": False,
            "latency_sensitive": False,
            "hipaa_network_concerns": False,
        },
        "recommended_transport": "stdio",
        "reason": "Fastest iteration cycle — no server setup, just run the script",
    },
]


# ---------------------------------------------------------------------------
# Configuration examples
# ---------------------------------------------------------------------------

def generate_example_configs() -> dict:
    """Generate example configurations for all transport types."""

    configs = {}

    # stdio config for clinical calculators
    configs["stdio_clinical"] = build_stdio_config(
        server_name="clinical-calculators",
        command="python",
        args=["clinical_calculator_server.py"],
        env={"LOG_LEVEL": "INFO"}
    )

    # SSE config for hospital formulary
    configs["sse_formulary"] = build_sse_config(
        server_name="hospital-formulary",
        url="https://formulary.hospital.internal:8443/sse",
        headers={
            "Authorization": "Bearer ${FORMULARY_API_KEY}",
            "X-Department": "pharmacy"
        },
        timeout_seconds=30
    )

    # Streamable HTTP for cloud CDS
    configs["http_cds"] = build_streamable_http_config(
        server_name="clinical-decision-support",
        url="https://cds.healthsystem.org/mcp",
        headers={
            "Authorization": "Bearer ${CDS_API_KEY}",
            "X-Facility-ID": "hospital-main"
        }
    )

    # Composite host configuration with multiple servers
    configs["multi_server_host"] = {
        "mcpServers": {
            "clinical-calculators": configs["stdio_clinical"],
            "hospital-formulary": configs["sse_formulary"],
            "clinical-decision-support": configs["http_cds"],
        }
    }

    return configs


# ---------------------------------------------------------------------------
# Main display
# ---------------------------------------------------------------------------

def main():
    """Compare MCP transport mechanisms with healthcare examples."""
    print("=" * 70)
    print("  Exercise 4: MCP Transport Comparison")
    print("  Choosing the right transport for your deployment")
    print("=" * 70)

    # --- Transport Overview ---
    print("\n  TRANSPORT OVERVIEW")
    print("  " + "─" * 55)

    for key, info in TRANSPORT_COMPARISON.items():
        print(f"\n  [{key.upper()}] {info['name']}")
        print(f"    How: {info['how_it_works']}")
        print(f"    Best for: {info['best_for']}")
        print(f"    Latency: {info['latency']}")
        print(f"    Security: {info['security']}")
        print(f"    Healthcare uses:")
        for use in info["healthcare_use_cases"]:
            print(f"      • {use}")

    # --- Side-by-side comparison ---
    print(f"\n\n  COMPARISON TABLE")
    print("  " + "─" * 55)
    aspects = ["latency", "security", "deployment", "scalability"]
    header = f"  {'Aspect':<18} {'stdio':<22} {'SSE':<22} {'Streamable HTTP'}"
    print(header)
    print("  " + "─" * 75)
    for aspect in aspects:
        val_stdio = TRANSPORT_COMPARISON["stdio"][aspect]
        val_sse = TRANSPORT_COMPARISON["sse"][aspect]
        val_http = TRANSPORT_COMPARISON["streamable-http"][aspect]
        # Truncate for table display
        print(f"  {aspect:<18} {val_stdio[:20]:<22} {val_sse[:20]:<22} {val_http[:20]}")

    # --- Scenario Recommendations ---
    print(f"\n\n  SCENARIO-BASED RECOMMENDATIONS")
    print("  " + "─" * 55)

    for scenario in SCENARIOS:
        print(f"\n  Scenario: {scenario['name']}")
        print(f"    {scenario['description']}")
        print(f"    → Recommended: {scenario['recommended_transport'].upper()}")
        print(f"    → Why: {scenario['reason']}")
        factors = scenario["factors"]
        print(f"    Factors: network={factors['network_required']}, "
              f"shared={factors['shared_across_users']}, "
              f"latency={factors['latency_sensitive']}, "
              f"HIPAA-net={factors['hipaa_network_concerns']}")

    # --- Example Configurations ---
    print(f"\n\n  EXAMPLE CONFIGURATIONS")
    print("  " + "─" * 55)

    configs = generate_example_configs()

    print("\n  1. stdio server (local clinical calculators):")
    for line in json.dumps(configs["stdio_clinical"], indent=4).split("\n"):
        print(f"     {line}")

    print("\n  2. SSE server (hospital formulary):")
    for line in json.dumps(configs["sse_formulary"], indent=4).split("\n"):
        print(f"     {line}")

    print("\n  3. Streamable HTTP server (cloud CDS):")
    for line in json.dumps(configs["http_cds"], indent=4).split("\n"):
        print(f"     {line}")

    print("\n  4. Multi-server host configuration:")
    for line in json.dumps(configs["multi_server_host"], indent=4).split("\n"):
        print(f"     {line}")

    # --- FastMCP transport code examples ---
    print(f"\n\n  CODE EXAMPLES: Running Server with Different Transports")
    print("  " + "─" * 55)

    print("""
    # stdio (default) — for local tools
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("My Server")

    @mcp.tool()
    def my_tool(x: int) -> str:
        return str(x * 2)

    mcp.run()  # Uses stdio by default

    # -------------------------------------------

    # SSE — for remote/shared servers
    mcp.run(transport="sse", host="0.0.0.0", port=8080)

    # -------------------------------------------

    # Streamable HTTP
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)
    """)

    print("=" * 70)
    print("  ✓ Exercise complete — you understand MCP transports!")
    print("=" * 70)


if __name__ == "__main__":
    main()
