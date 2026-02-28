"""
Exercise 2: Text-Based Cost Dashboard
======================================
Build a comprehensive cost tracking dashboard for LLM API usage.

Requirements:
- Track daily spend with per-model breakdown
- Calculate projected monthly cost from recent trends
- Set budget alerts with configurable thresholds
- Display cost trends over time
- Identify cost optimization opportunities

Healthcare Context:
  Hospital IT departments need to track AI costs per department.
  Budget overruns can threaten program funding and stakeholder trust.

Usage:
    python exercise_2_cost_dashboard.py
"""

from openai import OpenAI
import time
import json
import os
import random
from datetime import datetime, timedelta
from collections import defaultdict

client = OpenAI()

COST_RATES = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
}


class CostDashboard:
    """Text-based cost tracking and budget management dashboard."""

    def __init__(self, monthly_budget: float = 100.0):
        self.monthly_budget = monthly_budget
        self.cost_records = []
        self.budget_alerts = []
        self.alert_thresholds = {
            "daily_limit": monthly_budget / 30,
            "warn_at_pct": 0.75,
            "critical_at_pct": 0.90,
        }

    def record_cost(self, model: str, prompt_tokens: int, completion_tokens: int,
                    endpoint: str = "default", user: str = "unknown",
                    department: str = "general"):
        """Record a single API call cost."""
        rates = COST_RATES.get(model, {"input": 0.005, "output": 0.015})
        cost = (prompt_tokens / 1000) * rates["input"] + (completion_tokens / 1000) * rates["output"]

        record = {
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": round(cost, 8),
            "endpoint": endpoint,
            "user": user,
            "department": department,
        }
        self.cost_records.append(record)
        self._check_alerts(record)
        return record

    def _check_alerts(self, record: dict):
        """Check if the new record triggers any budget alerts."""
        today = datetime.now().strftime("%Y-%m-%d")
        today_cost = sum(r["cost_usd"] for r in self.cost_records if r["date"] == today)

        if today_cost > self.alert_thresholds["daily_limit"]:
            alert = {
                "type": "DAILY_LIMIT_EXCEEDED",
                "severity": "HIGH",
                "message": f"Daily spend ${today_cost:.4f} exceeds limit ${self.alert_thresholds['daily_limit']:.4f}",
                "timestamp": datetime.now().isoformat(),
            }
            self.budget_alerts.append(alert)

        # Monthly projection check
        month_start = datetime.now().replace(day=1).strftime("%Y-%m-%d")
        month_cost = sum(r["cost_usd"] for r in self.cost_records if r["date"] >= month_start)
        days_elapsed = datetime.now().day
        if days_elapsed > 0:
            projected = (month_cost / days_elapsed) * 30
            pct = projected / self.monthly_budget
            if pct > self.alert_thresholds["critical_at_pct"]:
                self.budget_alerts.append({
                    "type": "BUDGET_CRITICAL",
                    "severity": "CRITICAL",
                    "message": f"Projected monthly spend ${projected:.2f} ({pct*100:.0f}% of budget)",
                    "timestamp": datetime.now().isoformat(),
                })

    def load_simulated_history(self, days: int = 14):
        """Load simulated historical data for dashboard demo."""
        random.seed(42)
        models = ["gpt-4o-mini", "gpt-4o", "gpt-4o-mini"]
        endpoints = ["medical_qa", "rag_query", "clinical_agent", "admin_chat"]
        users = ["dr_smith", "dr_jones", "dr_patel", "nurse_chen", "admin_user"]
        departments = ["emergency", "cardiology", "radiology", "general"]

        for days_ago in range(days, -1, -1):
            date = datetime.now() - timedelta(days=days_ago)
            num_calls = random.randint(30, 120)
            for _ in range(num_calls):
                model = random.choice(models)
                pt = random.randint(50, 1500)
                ct = random.randint(20, 800)
                rates = COST_RATES.get(model, {"input": 0.005, "output": 0.015})
                cost = (pt / 1000) * rates["input"] + (ct / 1000) * rates["output"]
                self.cost_records.append({
                    "timestamp": date.isoformat(),
                    "date": date.strftime("%Y-%m-%d"),
                    "model": model,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "cost_usd": round(cost, 8),
                    "endpoint": random.choice(endpoints),
                    "user": random.choice(users),
                    "department": random.choice(departments),
                })

    def display_daily_report(self):
        """Display daily cost breakdown."""
        print("\n" + "=" * 60)
        print("  DAILY COST REPORT")
        print("=" * 60)

        daily = defaultdict(lambda: {"cost": 0.0, "requests": 0, "tokens": 0})
        for r in self.cost_records:
            daily[r["date"]]["cost"] += r["cost_usd"]
            daily[r["date"]]["requests"] += 1
            daily[r["date"]]["tokens"] += r["prompt_tokens"] + r["completion_tokens"]

        max_cost = max(d["cost"] for d in daily.values()) if daily else 1
        print(f"\n{'Date':<14} {'Requests':>10} {'Tokens':>10} {'Cost':>10} {'Bar'}")
        print("-" * 65)
        for date in sorted(daily.keys())[-10:]:
            d = daily[date]
            bar_len = int((d["cost"] / max_cost) * 30) if max_cost > 0 else 0
            bar = "█" * bar_len
            print(f"  {date:<12} {d['requests']:>8} {d['tokens']:>10} ${d['cost']:>8.4f}  {bar}")

    def display_model_breakdown(self):
        """Display cost breakdown by model."""
        print("\n" + "-" * 60)
        print("  COST BY MODEL")
        print("-" * 60)

        by_model = defaultdict(lambda: {"cost": 0.0, "requests": 0, "tokens": 0})
        for r in self.cost_records:
            by_model[r["model"]]["cost"] += r["cost_usd"]
            by_model[r["model"]]["requests"] += 1
            by_model[r["model"]]["tokens"] += r["prompt_tokens"] + r["completion_tokens"]

        total_cost = sum(m["cost"] for m in by_model.values())
        for model in sorted(by_model, key=lambda m: -by_model[m]["cost"]):
            d = by_model[model]
            pct = (d["cost"] / total_cost * 100) if total_cost > 0 else 0
            avg_cost = d["cost"] / d["requests"] if d["requests"] > 0 else 0
            print(f"\n  {model}")
            print(f"    Requests:    {d['requests']:,}")
            print(f"    Tokens:      {d['tokens']:,}")
            print(f"    Total Cost:  ${d['cost']:.4f} ({pct:.1f}%)")
            print(f"    Avg/Request: ${avg_cost:.6f}")

    def display_department_breakdown(self):
        """Display cost breakdown by department."""
        print("\n" + "-" * 60)
        print("  COST BY DEPARTMENT")
        print("-" * 60)

        by_dept = defaultdict(lambda: {"cost": 0.0, "requests": 0})
        for r in self.cost_records:
            by_dept[r["department"]]["cost"] += r["cost_usd"]
            by_dept[r["department"]]["requests"] += 1

        total = sum(d["cost"] for d in by_dept.values())
        print(f"\n{'Department':<20} {'Requests':>10} {'Cost':>12} {'% of Total':>12}")
        print("-" * 56)
        for dept in sorted(by_dept, key=lambda d: -by_dept[d]["cost"]):
            d = by_dept[dept]
            pct = (d["cost"] / total * 100) if total > 0 else 0
            print(f"  {dept:<18} {d['requests']:>8} ${d['cost']:>10.4f} {pct:>10.1f}%")

    def display_projections(self):
        """Display monthly cost projections and budget status."""
        print("\n" + "-" * 60)
        print("  MONTHLY PROJECTION & BUDGET")
        print("-" * 60)

        today = datetime.now()
        month_start = today.replace(day=1).strftime("%Y-%m-%d")
        month_costs = [r["cost_usd"] for r in self.cost_records if r["date"] >= month_start]
        month_total = sum(month_costs)
        days_elapsed = today.day
        days_in_month = 30

        if days_elapsed > 0:
            daily_avg = month_total / days_elapsed
            projected = daily_avg * days_in_month
        else:
            daily_avg = 0
            projected = 0

        pct_used = (month_total / self.monthly_budget * 100) if self.monthly_budget > 0 else 0
        pct_projected = (projected / self.monthly_budget * 100) if self.monthly_budget > 0 else 0

        print(f"\n  Monthly Budget:    ${self.monthly_budget:.2f}")
        print(f"  Spent So Far:      ${month_total:.4f} ({pct_used:.1f}%)")
        print(f"  Daily Average:     ${daily_avg:.4f}")
        print(f"  Projected Monthly: ${projected:.4f} ({pct_projected:.1f}%)")
        print(f"  Remaining Budget:  ${max(0, self.monthly_budget - month_total):.4f}")

        # Visual budget bar
        bar_total = 40
        bar_used = min(int(pct_used / 100 * bar_total), bar_total)
        bar_projected = min(int(pct_projected / 100 * bar_total), bar_total)
        print(f"\n  Budget:   [{'█' * bar_used}{'░' * (bar_total - bar_used)}] {pct_used:.1f}% used")
        print(f"  Projected:[{'█' * bar_projected}{'░' * (bar_total - bar_projected)}] {pct_projected:.1f}%")

        if pct_projected > 100:
            print("\n  ⚠ WARNING: Projected spend EXCEEDS monthly budget!")
            print(f"  Recommendation: Reduce usage by {pct_projected - 100:.0f}% or switch to cheaper models.")
        elif pct_projected > 80:
            print("\n  ⚠ CAUTION: Projected spend approaching budget limit.")

    def display_optimization_tips(self):
        """Identify cost optimization opportunities."""
        print("\n" + "-" * 60)
        print("  OPTIMIZATION OPPORTUNITIES")
        print("-" * 60)

        tips = []

        # Check if expensive models are used for simple tasks
        by_endpoint_model = defaultdict(lambda: defaultdict(int))
        for r in self.cost_records:
            by_endpoint_model[r["endpoint"]][r["model"]] += 1

        for ep, models in by_endpoint_model.items():
            if "gpt-4o" in models and "gpt-4o-mini" in models:
                expensive = models.get("gpt-4o", 0)
                total = sum(models.values())
                if expensive > total * 0.3:
                    tips.append(f"  - Endpoint '{ep}': {expensive}/{total} calls use gpt-4o. "
                                f"Consider gpt-4o-mini for simpler queries.")

        # Check for high-token requests
        high_token = [r for r in self.cost_records if r["prompt_tokens"] > 1000]
        if len(high_token) > len(self.cost_records) * 0.2:
            tips.append(f"  - {len(high_token)} requests use >1000 prompt tokens. "
                        f"Consider caching or shortening prompts.")

        if not tips:
            tips.append("  - No immediate optimization opportunities found. Good job!")

        for tip in tips:
            print(tip)

    def display_alerts(self):
        """Display recent budget alerts."""
        if self.budget_alerts:
            print("\n" + "-" * 60)
            print("  ACTIVE ALERTS")
            print("-" * 60)
            for alert in self.budget_alerts[-5:]:
                print(f"\n  [{alert['severity']}] {alert['type']}")
                print(f"    {alert['message']}")
        else:
            print("\n  No active alerts.")

    def display_full_dashboard(self):
        """Display the complete cost dashboard."""
        print("\n" + "=" * 60)
        print("  LLM COST DASHBOARD")
        print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        self.display_daily_report()
        self.display_model_breakdown()
        self.display_department_breakdown()
        self.display_projections()
        self.display_optimization_tips()
        self.display_alerts()


def main():
    """Run the cost dashboard exercise."""
    print("=" * 60)
    print("  Exercise 2: Text-Based Cost Dashboard")
    print("=" * 60)

    dashboard = CostDashboard(monthly_budget=100.0)

    # Load simulated history
    print("\nLoading 14 days of simulated cost data...")
    dashboard.load_simulated_history(days=14)
    print(f"  Loaded {len(dashboard.cost_records)} historical records.")

    # Make a real API call and record it
    print("\nMaking a live API call to add to tracking...")
    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical cost analyst. Be brief."},
            {"role": "user", "content": "What are the typical costs of running AI models in a hospital setting?"},
        ],
    )
    latency = time.time() - start

    record = dashboard.record_cost(
        model="gpt-4o-mini",
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        endpoint="medical_qa",
        user="admin_user",
        department="administration",
    )
    print(f"  Recorded: ${record['cost_usd']:.6f} ({response.usage.total_tokens} tokens, {latency:.1f}s)")

    # Display the full dashboard
    dashboard.display_full_dashboard()

    print("\n\nDashboard complete!")


if __name__ == "__main__":
    main()
