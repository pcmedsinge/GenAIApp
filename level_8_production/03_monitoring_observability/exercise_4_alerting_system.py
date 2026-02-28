"""
Exercise 4: Alerting System
============================
Build a multi-signal alerting system for LLM-powered applications.

Requirements:
- Define alert rules: high latency, cost spikes, error rate, unusual patterns
- Set configurable thresholds per alert type
- Evaluate incoming metrics against thresholds
- Generate alerts with severity levels (INFO, WARNING, CRITICAL)
- Maintain alert history and support acknowledgment

Healthcare Context:
  In clinical AI systems, latency spikes can delay patient care, cost
  spikes can drain department budgets, and error-rate increases may
  indicate degraded model quality for safety-critical decisions.

Usage:
    python exercise_4_alerting_system.py
"""

from openai import OpenAI
import time
import json
import os
import random
import statistics
from datetime import datetime, timedelta
from collections import defaultdict

client = OpenAI()


class AlertRule:
    """A single alerting rule with threshold and evaluation logic."""

    def __init__(self, name: str, metric: str, condition: str, threshold: float,
                 severity: str = "WARNING", window_minutes: int = 5,
                 description: str = ""):
        self.name = name
        self.metric = metric
        self.condition = condition  # "gt", "lt", "gte", "pct_change_gt"
        self.threshold = threshold
        self.severity = severity
        self.window_minutes = window_minutes
        self.description = description
        self.enabled = True

    def evaluate(self, current_value: float, historical_values: list = None) -> dict:
        """Evaluate this rule against a current value. Returns alert or None."""
        triggered = False

        if self.condition == "gt":
            triggered = current_value > self.threshold
        elif self.condition == "lt":
            triggered = current_value < self.threshold
        elif self.condition == "gte":
            triggered = current_value >= self.threshold
        elif self.condition == "pct_change_gt" and historical_values:
            avg_hist = statistics.mean(historical_values) if historical_values else 0
            if avg_hist > 0:
                pct_change = ((current_value - avg_hist) / avg_hist) * 100
                triggered = pct_change > self.threshold
            else:
                triggered = False

        if triggered:
            return {
                "rule": self.name,
                "metric": self.metric,
                "severity": self.severity,
                "current_value": current_value,
                "threshold": self.threshold,
                "condition": self.condition,
                "description": self.description,
                "timestamp": datetime.now().isoformat(),
            }
        return None


class AlertingSystem:
    """Multi-signal alerting system for LLM applications."""

    def __init__(self):
        self.rules = []
        self.alerts = []
        self.acknowledged = set()
        self.metrics_buffer = defaultdict(list)

    def add_rule(self, rule: AlertRule):
        """Add an alerting rule."""
        self.rules.append(rule)

    def record_metric(self, metric: str, value: float, timestamp: str = None):
        """Record a metric value for alerting evaluation."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        self.metrics_buffer[metric].append({
            "value": value,
            "timestamp": timestamp,
        })

    def evaluate_all(self) -> list:
        """Evaluate all rules against current metrics. Returns new alerts."""
        new_alerts = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            metric_data = self.metrics_buffer.get(rule.metric, [])
            if not metric_data:
                continue

            # Get recent values within the rule's window
            cutoff = datetime.now() - timedelta(minutes=rule.window_minutes)
            recent = [m for m in metric_data
                      if datetime.fromisoformat(m["timestamp"]) > cutoff]

            if not recent:
                continue

            current_value = recent[-1]["value"]

            # For percentage change, use older data as baseline
            historical = [m["value"] for m in metric_data
                          if datetime.fromisoformat(m["timestamp"]) <= cutoff]

            alert = rule.evaluate(current_value, historical)
            if alert:
                alert["alert_id"] = f"ALT-{len(self.alerts) + len(new_alerts) + 1:04d}"
                new_alerts.append(alert)
                self.alerts.append(alert)

        return new_alerts

    def acknowledge(self, alert_id: str):
        """Acknowledge an alert."""
        self.acknowledged.add(alert_id)

    def get_active_alerts(self) -> list:
        """Get all unacknowledged alerts."""
        return [a for a in self.alerts if a["alert_id"] not in self.acknowledged]

    def get_alert_summary(self) -> dict:
        """Get summary of all alerts."""
        by_severity = defaultdict(int)
        by_rule = defaultdict(int)
        for a in self.alerts:
            by_severity[a["severity"]] += 1
            by_rule[a["rule"]] += 1

        return {
            "total_alerts": len(self.alerts),
            "active": len(self.get_active_alerts()),
            "acknowledged": len(self.acknowledged),
            "by_severity": dict(by_severity),
            "by_rule": dict(by_rule),
        }

    def display_alerts(self):
        """Display all active alerts."""
        active = self.get_active_alerts()
        print("\n" + "=" * 65)
        print("  ACTIVE ALERTS")
        print("=" * 65)

        if not active:
            print("  No active alerts. All clear!")
            return

        severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
        active.sort(key=lambda a: severity_order.get(a["severity"], 99))

        for alert in active:
            sev_marker = {
                "CRITICAL": "[!!!]",
                "WARNING": "[!!]",
                "INFO": "[i]",
            }.get(alert["severity"], "[?]")

            print(f"\n  {sev_marker} {alert['alert_id']} — {alert['rule']}")
            print(f"      Severity:  {alert['severity']}")
            print(f"      Metric:    {alert['metric']} = {alert['current_value']}")
            print(f"      Threshold: {alert['condition']} {alert['threshold']}")
            print(f"      Time:      {alert['timestamp']}")
            if alert.get("description"):
                print(f"      Note:      {alert['description']}")

    def display_summary(self):
        """Display alert summary dashboard."""
        summary = self.get_alert_summary()
        print("\n" + "-" * 65)
        print("  ALERT SUMMARY")
        print("-" * 65)
        print(f"  Total Alerts:   {summary['total_alerts']}")
        print(f"  Active:         {summary['active']}")
        print(f"  Acknowledged:   {summary['acknowledged']}")

        if summary["by_severity"]:
            print("\n  By Severity:")
            for sev, count in sorted(summary["by_severity"].items()):
                print(f"    {sev}: {count}")

        if summary["by_rule"]:
            print("\n  By Rule:")
            for rule, count in sorted(summary["by_rule"].items(), key=lambda x: -x[1]):
                print(f"    {rule}: {count}")


def setup_healthcare_rules() -> AlertingSystem:
    """Create an alerting system with healthcare-relevant rules."""
    system = AlertingSystem()

    system.add_rule(AlertRule(
        name="High Latency",
        metric="latency_ms",
        condition="gt",
        threshold=5000,
        severity="WARNING",
        window_minutes=5,
        description="LLM response time exceeds 5 seconds — may delay clinical decisions",
    ))

    system.add_rule(AlertRule(
        name="Critical Latency",
        metric="latency_ms",
        condition="gt",
        threshold=10000,
        severity="CRITICAL",
        window_minutes=5,
        description="LLM response time exceeds 10 seconds — urgent clinical workflow impact",
    ))

    system.add_rule(AlertRule(
        name="High Error Rate",
        metric="error_rate_pct",
        condition="gt",
        threshold=5.0,
        severity="CRITICAL",
        window_minutes=10,
        description="Error rate above 5% — AI service degradation",
    ))

    system.add_rule(AlertRule(
        name="Cost Spike",
        metric="hourly_cost_usd",
        condition="gt",
        threshold=2.0,
        severity="WARNING",
        window_minutes=60,
        description="Hourly cost exceeds $2 — possible runaway usage",
    ))

    system.add_rule(AlertRule(
        name="High Token Usage",
        metric="avg_tokens_per_request",
        condition="gt",
        threshold=3000,
        severity="INFO",
        window_minutes=15,
        description="Average tokens per request high — consider prompt optimization",
    ))

    system.add_rule(AlertRule(
        name="Low Cache Hit Rate",
        metric="cache_hit_rate_pct",
        condition="lt",
        threshold=30.0,
        severity="INFO",
        window_minutes=30,
        description="Cache hit rate below 30% — caching may need tuning",
    ))

    system.add_rule(AlertRule(
        name="Guardrail Trigger Spike",
        metric="guardrail_triggers_per_hour",
        condition="gt",
        threshold=10,
        severity="WARNING",
        window_minutes=60,
        description="Many guardrail triggers — possible prompt injection attack",
    ))

    return system


def main():
    """Run the alerting system exercise."""
    print("=" * 65)
    print("  Exercise 4: Alerting System")
    print("=" * 65)

    system = setup_healthcare_rules()
    print(f"\nConfigured {len(system.rules)} alerting rules:")
    for rule in system.rules:
        print(f"  - {rule.name} ({rule.severity}): {rule.metric} {rule.condition} {rule.threshold}")

    # Make a real API call to get baseline metrics
    print("\nMaking a live API call for baseline metrics...")
    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical assistant. Be brief."},
            {"role": "user", "content": "What are the symptoms of diabetic ketoacidosis?"},
        ],
    )
    real_latency = (time.time() - start) * 1000
    real_tokens = response.usage.total_tokens
    print(f"  Latency: {real_latency:.0f}ms, Tokens: {real_tokens}")

    # Record real metric
    system.record_metric("latency_ms", real_latency)
    system.record_metric("avg_tokens_per_request", real_tokens)

    # Simulate additional metrics (mix of normal and anomalous)
    print("\nSimulating metric stream with some anomalies...")
    random.seed(42)
    now = datetime.now()

    for i in range(30):
        ts = (now - timedelta(minutes=30 - i)).isoformat()

        # Latency — mostly normal, some spikes
        if i in [12, 13, 25]:
            lat = random.uniform(6000, 12000)  # Spike
        else:
            lat = random.uniform(300, 2000)  # Normal
        system.record_metric("latency_ms", lat, ts)

        # Error rate — mostly low, one spike
        if i == 20:
            system.record_metric("error_rate_pct", 8.5, ts)
        else:
            system.record_metric("error_rate_pct", random.uniform(0, 2), ts)

        # Cost
        cost = random.uniform(0.1, 1.5) if i != 22 else 3.5
        system.record_metric("hourly_cost_usd", cost, ts)

        # Token usage
        tokens = random.randint(200, 2500) if i != 18 else 4500
        system.record_metric("avg_tokens_per_request", tokens, ts)

        # Cache hit rate
        system.record_metric("cache_hit_rate_pct", random.uniform(20, 60), ts)

        # Guardrail triggers
        triggers = random.randint(0, 5) if i != 26 else 15
        system.record_metric("guardrail_triggers_per_hour", triggers, ts)

    print(f"  Recorded {sum(len(v) for v in system.metrics_buffer.values())} metric data points")

    # Evaluate all rules
    print("\nEvaluating alert rules...")
    new_alerts = system.evaluate_all()
    print(f"  Generated {len(new_alerts)} new alerts")

    # Display alerts
    system.display_alerts()

    # Acknowledge one alert (simulate operator response)
    active = system.get_active_alerts()
    if active:
        ack_id = active[0]["alert_id"]
        system.acknowledge(ack_id)
        print(f"\n  Acknowledged alert: {ack_id}")

    # Show summary
    system.display_summary()

    # Display rule effectiveness
    print("\n" + "-" * 65)
    print("  RULE EFFECTIVENESS")
    print("-" * 65)
    for rule in system.rules:
        triggered = sum(1 for a in system.alerts if a["rule"] == rule.name)
        status = "TRIGGERED" if triggered > 0 else "quiet"
        print(f"  {rule.name:<30} {status:<12} ({triggered} alerts)")

    print("\nDone! Alerting system exercised with simulated and real data.")


if __name__ == "__main__":
    main()
