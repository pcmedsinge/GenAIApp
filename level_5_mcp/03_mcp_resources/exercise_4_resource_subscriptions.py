"""
Exercise 4: Resource Subscriptions
=====================================

Skills practiced:
- Simulating MCP resource subscription and update notifications
- Maintaining a change log with timestamps for resource updates
- Implementing subscriber management (subscribe, unsubscribe, notify)
- Understanding the notification pattern in MCP (server → client)

Healthcare context:
Clinical guidelines and formulary data change periodically. When a guideline
is updated (e.g., new HbA1c target) or a medication is added/removed from the
formulary, connected AI agents need to be notified so they can refresh their
cached data and avoid using stale information.

MCP supports resource subscriptions:
  1. Client sends resources/subscribe with a URI
  2. Server tracks the subscription
  3. When the resource changes, server sends notifications/resources/updated
  4. Client re-reads the resource to get fresh data

This exercise simulates the full subscription lifecycle.

Usage:
    python exercise_4_resource_subscriptions.py
"""

import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


# ============================================================================
# Resource Store with Change Tracking
# ============================================================================

class ResourceStore:
    """A resource store that tracks changes and manages subscriptions."""

    def __init__(self):
        self.resources = {}
        self.change_log = []
        self.subscribers = {}  # uri -> set of subscriber IDs
        self.notifications_sent = []

    def add_resource(self, uri: str, content: dict, description: str = ""):
        """Add or update a resource."""
        is_update = uri in self.resources
        self.resources[uri] = {
            "content": content,
            "description": description,
            "created_at": self.resources.get(uri, {}).get("created_at",
                          datetime.now().isoformat()),
            "updated_at": datetime.now().isoformat(),
            "version": self.resources.get(uri, {}).get("version", 0) + 1,
        }
        action = "updated" if is_update else "created"
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "uri": uri,
            "action": action,
            "version": self.resources[uri]["version"],
            "description": description,
        }
        self.change_log.append(log_entry)

        # Notify subscribers if this is an update
        if is_update and uri in self.subscribers:
            self._notify_subscribers(uri, log_entry)

        return log_entry

    def read_resource(self, uri: str) -> dict:
        """Read a resource by URI."""
        if uri not in self.resources:
            return {"error": f"Resource '{uri}' not found",
                    "available": list(self.resources.keys())}
        r = self.resources[uri]
        return {
            "uri": uri,
            "content": r["content"],
            "version": r["version"],
            "updated_at": r["updated_at"],
        }

    def subscribe(self, uri: str, subscriber_id: str) -> dict:
        """Subscribe to updates for a resource URI."""
        if uri not in self.resources:
            return {"error": f"Resource '{uri}' not found"}
        if uri not in self.subscribers:
            self.subscribers[uri] = set()
        self.subscribers[uri].add(subscriber_id)
        return {
            "status": "subscribed",
            "uri": uri,
            "subscriber_id": subscriber_id,
            "current_version": self.resources[uri]["version"],
            "total_subscribers": len(self.subscribers[uri]),
        }

    def unsubscribe(self, uri: str, subscriber_id: str) -> dict:
        """Unsubscribe from resource updates."""
        if uri in self.subscribers:
            self.subscribers[uri].discard(subscriber_id)
            if not self.subscribers[uri]:
                del self.subscribers[uri]
        return {"status": "unsubscribed", "uri": uri, "subscriber_id": subscriber_id}

    def _notify_subscribers(self, uri: str, change_entry: dict):
        """Send notifications to all subscribers of a resource."""
        if uri not in self.subscribers:
            return
        for sub_id in self.subscribers[uri]:
            notification = {
                "jsonrpc": "2.0",
                "method": "notifications/resources/updated",
                "params": {"uri": uri},
                "_meta": {
                    "subscriber_id": sub_id,
                    "change": change_entry,
                    "sent_at": datetime.now().isoformat(),
                },
            }
            self.notifications_sent.append(notification)

    def get_change_log(self, uri: str = None, limit: int = 20) -> list:
        """Get the change log, optionally filtered by URI."""
        log = self.change_log
        if uri:
            log = [e for e in log if e["uri"] == uri]
        return log[-limit:]

    def list_resources(self) -> list:
        """List all resources with metadata."""
        return [
            {
                "uri": uri,
                "description": r["description"],
                "version": r["version"],
                "updated_at": r["updated_at"],
                "subscriber_count": len(self.subscribers.get(uri, set())),
            }
            for uri, r in self.resources.items()
        ]

    def get_subscription_status(self) -> dict:
        """Get an overview of all active subscriptions."""
        return {
            "total_subscriptions": sum(len(subs) for subs in self.subscribers.values()),
            "resources_with_subscribers": len(self.subscribers),
            "total_notifications_sent": len(self.notifications_sent),
            "subscriptions": {
                uri: list(subs) for uri, subs in self.subscribers.items()
            },
        }


# ============================================================================
# Simulation Helpers
# ============================================================================

def simulate_guideline_lifecycle(store: ResourceStore):
    """Simulate a complete guideline resource lifecycle."""

    # Initial creation of guidelines
    guidelines = [
        ("guideline://cardiology/hypertension", {
            "title": "Hypertension Management",
            "source": "ACC/AHA 2024",
            "bp_target": "< 130/80 mmHg",
            "first_line": ["ACE inhibitor", "ARB", "CCB", "Thiazide"],
            "version_note": "Initial publication",
        }, "Hypertension management guideline"),
        ("guideline://endocrinology/diabetes", {
            "title": "Type 2 Diabetes Management",
            "source": "ADA 2026",
            "hba1c_target": "< 7.0%",
            "first_line": "Metformin + lifestyle",
            "version_note": "Initial publication",
        }, "Diabetes management guideline"),
        ("guideline://nephrology/ckd", {
            "title": "CKD Management",
            "source": "KDIGO 2024",
            "key_drugs": ["ACEi/ARB", "SGLT2i", "Finerenone"],
            "version_note": "Initial publication",
        }, "CKD management guideline"),
    ]

    print("\n  Step 1: Creating initial guideline resources")
    for uri, content, desc in guidelines:
        entry = store.add_resource(uri, content, desc)
        print(f"    ✓ Created {uri} (v{entry['version']})")

    # Subscribe agents
    print("\n  Step 2: Agents subscribe to guideline updates")
    agents = [
        ("agent-cardiology", "guideline://cardiology/hypertension"),
        ("agent-primary-care", "guideline://cardiology/hypertension"),
        ("agent-primary-care", "guideline://endocrinology/diabetes"),
        ("agent-endocrinology", "guideline://endocrinology/diabetes"),
        ("agent-nephrology", "guideline://nephrology/ckd"),
        ("agent-primary-care", "guideline://nephrology/ckd"),
    ]
    for agent_id, uri in agents:
        result = store.subscribe(uri, agent_id)
        print(f"    ✓ {agent_id} subscribed to {uri} "
              f"(v{result['current_version']}, {result['total_subscribers']} subscriber(s))")

    # Simulate a guideline update
    print("\n  Step 3: Guideline update — HTN target changed")
    updated_htn = {
        "title": "Hypertension Management",
        "source": "ACC/AHA 2024 (February 2026 Update)",
        "bp_target": "< 120/80 mmHg for high-risk patients",
        "first_line": ["ACE inhibitor", "ARB", "CCB", "Thiazide"],
        "new_recommendation": "ARNI consideration for resistant HTN",
        "version_note": "Updated BP targets for high-risk patients",
    }
    entry = store.add_resource(
        "guideline://cardiology/hypertension",
        updated_htn,
        "Updated hypertension guideline — new BP targets",
    )
    print(f"    ✓ Updated guideline://cardiology/hypertension to v{entry['version']}")

    # Show notifications
    print("\n  Step 4: Notifications sent to subscribers")
    for notif in store.notifications_sent:
        sub = notif["_meta"]["subscriber_id"]
        uri = notif["params"]["uri"]
        print(f"    → Notification to {sub}: {uri} updated")
    print(f"    Total notifications: {len(store.notifications_sent)}")

    # Another update
    print("\n  Step 5: Diabetes guideline update — new HbA1c guidance")
    updated_dm = {
        "title": "Type 2 Diabetes Management",
        "source": "ADA 2026 (Mid-year Update)",
        "hba1c_target": "< 7.0% (< 6.5% if no hypoglycemia risk)",
        "first_line": "Metformin + lifestyle; early GLP-1 RA if ASCVD",
        "new_recommendation": "Tirzepatide added as first-line option for obesity + T2DM",
        "version_note": "Added tirzepatide recommendation",
    }
    store.add_resource(
        "guideline://endocrinology/diabetes",
        updated_dm,
        "Updated diabetes guideline — tirzepatide recommendation",
    )

    # Show change log
    print("\n  Step 6: Full Change Log")
    for entry in store.get_change_log():
        print(f"    [{entry['timestamp'][:19]}] {entry['action'].upper():>8} "
              f"{entry['uri']} (v{entry['version']})")

    return store


# ============================================================================
# Demo / Test
# ============================================================================

def main():
    """Demonstrate resource subscriptions and update notifications."""
    print("=" * 70)
    print("  Exercise 4: Resource Subscriptions")
    print("  Simulating MCP resource update notifications")
    print("=" * 70)

    store = ResourceStore()

    # Run the lifecycle simulation
    store = simulate_guideline_lifecycle(store)

    # Show subscription status
    print("\n" + "─" * 60)
    print("  Subscription Status Overview")
    print("─" * 60)
    status = store.get_subscription_status()
    print(f"  Active subscriptions:    {status['total_subscriptions']}")
    print(f"  Resources with watchers: {status['resources_with_subscribers']}")
    print(f"  Notifications sent:      {status['total_notifications_sent']}")
    for uri, subs in status["subscriptions"].items():
        print(f"\n  {uri}")
        for s in subs:
            print(f"    • {s}")

    # Show resource list with versions
    print("\n" + "─" * 60)
    print("  All Resources (with versions)")
    print("─" * 60)
    for r in store.list_resources():
        print(f"  {r['uri']}")
        print(f"    Version:     {r['version']}")
        print(f"    Updated:     {r['updated_at'][:19]}")
        print(f"    Subscribers: {r['subscriber_count']}")

    # Show JSON-RPC subscription protocol
    print("\n" + "─" * 60)
    print("  MCP Subscription Protocol (JSON-RPC)")
    print("─" * 60)

    subscribe_request = {
        "jsonrpc": "2.0", "id": 1,
        "method": "resources/subscribe",
        "params": {"uri": "guideline://cardiology/hypertension"},
    }
    print(f"\n  1. Client subscribes:")
    print(f"     {json.dumps(subscribe_request, indent=6)}")

    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/resources/updated",
        "params": {"uri": "guideline://cardiology/hypertension"},
    }
    print(f"\n  2. Server sends notification on change:")
    print(f"     {json.dumps(notification, indent=6)}")

    read_request = {
        "jsonrpc": "2.0", "id": 2,
        "method": "resources/read",
        "params": {"uri": "guideline://cardiology/hypertension"},
    }
    print(f"\n  3. Client re-reads resource:")
    print(f"     {json.dumps(read_request, indent=6)}")

    # Demonstrate unsubscribe
    print("\n" + "─" * 60)
    print("  Unsubscribe Test")
    print("─" * 60)
    result = store.unsubscribe("guideline://cardiology/hypertension", "agent-cardiology")
    print(f"  agent-cardiology unsubscribed from HTN guideline: {result['status']}")
    status = store.get_subscription_status()
    print(f"  Remaining subscriptions: {status['total_subscriptions']}")

    print("\n  ✓ Resource subscription simulation complete")


if __name__ == "__main__":
    main()
