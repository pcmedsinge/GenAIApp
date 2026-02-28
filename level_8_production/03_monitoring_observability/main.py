"""
Level 8 - Project 03: Monitoring & Observability
=================================================
Production LLM monitoring: request logging, cost tracking, latency monitoring,
and agent trace viewing — all built from scratch with no external dependencies.

Demos:
  1. Request Logging — log every LLM call with full metadata
  2. Cost Tracking — aggregate costs by model, endpoint, user
  3. Latency Monitoring — p50/p95/p99 with anomaly detection
  4. Trace Viewer — visualize agent execution as a timed tree
"""

from openai import OpenAI
import time
import json
import os
import uuid
import statistics
from datetime import datetime, timedelta
from collections import defaultdict

client = OpenAI()

# ============================================================
# Cost rates per 1K tokens (approximate, for tracking)
# ============================================================
MODEL_COSTS = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}

LOG_DIR = "monitoring_logs"
os.makedirs(LOG_DIR, exist_ok=True)


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost for a single LLM call."""
    rates = MODEL_COSTS.get(model, {"input": 0.005, "output": 0.015})
    input_cost = (prompt_tokens / 1000) * rates["input"]
    output_cost = (completion_tokens / 1000) * rates["output"]
    return round(input_cost + output_cost, 6)


def make_monitored_call(
    messages: list,
    model: str = "gpt-4o-mini",
    endpoint: str = "chat",
    user_id: str = "demo_user",
) -> dict:
    """Make an LLM call and return response with monitoring metadata."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    response = client.chat.completions.create(model=model, messages=messages)

    latency = time.time() - start_time
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    cost = calculate_cost(model, prompt_tokens, completion_tokens)

    log_entry = {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "endpoint": endpoint,
        "user_id": user_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": cost,
        "latency_ms": round(latency * 1000, 2),
        "status": "success",
        "content": response.choices[0].message.content[:200],
    }

    return {
        "response": response.choices[0].message.content,
        "metadata": log_entry,
    }


# ============================================================
# DEMO 1: Request Logging
# ============================================================
def demo_request_logging():
    """
    Log every LLM request with full metadata: timestamp, model,
    tokens in/out, latency, cost. Store as JSON Lines.
    """
    print("\n" + "=" * 60)
    print("DEMO 1: Request Logging")
    print("=" * 60)

    log_file = os.path.join(LOG_DIR, "request_log.jsonl")
    print(f"\nLogging requests to: {log_file}")

    medical_queries = [
        "What are the warning signs of a stroke?",
        "Explain the difference between Type 1 and Type 2 diabetes.",
        "What is the standard treatment protocol for hypertension?",
    ]

    all_logs = []

    for i, query in enumerate(medical_queries, 1):
        print(f"\n--- Request {i}/{len(medical_queries)} ---")
        print(f"Query: {query[:60]}...")

        result = make_monitored_call(
            messages=[
                {"role": "system", "content": "You are a medical information assistant. Be concise."},
                {"role": "user", "content": query},
            ],
            model="gpt-4o-mini",
            endpoint="medical_qa",
            user_id=f"dr_smith_{i}",
        )

        meta = result["metadata"]
        all_logs.append(meta)

        print(f"  Request ID:  {meta['request_id']}")
        print(f"  Model:       {meta['model']}")
        print(f"  Tokens:      {meta['prompt_tokens']} in / {meta['completion_tokens']} out")
        print(f"  Cost:        ${meta['cost_usd']:.6f}")
        print(f"  Latency:     {meta['latency_ms']:.0f}ms")
        print(f"  Response:    {result['response'][:100]}...")

        # Append to JSONL log file
        with open(log_file, "a") as f:
            f.write(json.dumps(meta) + "\n")

    # Summary
    print("\n" + "-" * 40)
    print("SESSION SUMMARY")
    print("-" * 40)
    total_tokens = sum(l["total_tokens"] for l in all_logs)
    total_cost = sum(l["cost_usd"] for l in all_logs)
    avg_latency = statistics.mean(l["latency_ms"] for l in all_logs)
    print(f"  Total Requests: {len(all_logs)}")
    print(f"  Total Tokens:   {total_tokens}")
    print(f"  Total Cost:     ${total_cost:.6f}")
    print(f"  Avg Latency:    {avg_latency:.0f}ms")
    print(f"  Log File:       {log_file}")


# ============================================================
# DEMO 2: Cost Tracking
# ============================================================
def demo_cost_tracking():
    """
    Track costs per model, per endpoint, per user. Show daily/weekly
    aggregates and projected monthly costs.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Cost Tracking")
    print("=" * 60)

    # Simulate historical cost data (past 7 days)
    cost_history = []
    models_used = ["gpt-4o-mini", "gpt-4o"]
    endpoints = ["medical_qa", "rag_query", "clinical_agent"]
    users = ["dr_smith", "dr_jones", "dr_patel", "nurse_chen"]

    import random
    random.seed(42)

    for days_ago in range(7):
        date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        num_requests = random.randint(20, 80)
        for _ in range(num_requests):
            model = random.choice(models_used)
            cost_history.append({
                "date": date,
                "model": model,
                "endpoint": random.choice(endpoints),
                "user": random.choice(users),
                "cost_usd": random.uniform(0.0001, 0.005) if "mini" in model else random.uniform(0.002, 0.02),
                "tokens": random.randint(100, 2000),
            })

    # Now make a real call and add it
    print("\nMaking a tracked API call...")
    result = make_monitored_call(
        messages=[
            {"role": "system", "content": "You are a medical assistant. Be brief."},
            {"role": "user", "content": "What are common side effects of metformin?"},
        ],
        model="gpt-4o-mini",
        endpoint="medical_qa",
        user_id="dr_smith",
    )
    cost_history.append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "model": result["metadata"]["model"],
        "endpoint": "medical_qa",
        "user": "dr_smith",
        "cost_usd": result["metadata"]["cost_usd"],
        "tokens": result["metadata"]["total_tokens"],
    })
    print(f"  Live call cost: ${result['metadata']['cost_usd']:.6f}")

    # Daily aggregates
    print("\n--- Daily Cost Summary (Last 7 Days) ---")
    daily_costs = defaultdict(float)
    for entry in cost_history:
        daily_costs[entry["date"]] += entry["cost_usd"]
    for date in sorted(daily_costs.keys()):
        bar = "█" * int(daily_costs[date] * 200)
        print(f"  {date}: ${daily_costs[date]:.4f} {bar}")

    # Per-model breakdown
    print("\n--- Cost by Model ---")
    model_costs = defaultdict(lambda: {"cost": 0.0, "requests": 0})
    for entry in cost_history:
        model_costs[entry["model"]]["cost"] += entry["cost_usd"]
        model_costs[entry["model"]]["requests"] += 1
    for model, data in sorted(model_costs.items()):
        print(f"  {model:20s}: ${data['cost']:.4f} ({data['requests']} requests)")

    # Per-user breakdown
    print("\n--- Cost by User ---")
    user_costs = defaultdict(float)
    for entry in cost_history:
        user_costs[entry["user"]] += entry["cost_usd"]
    for user, cost in sorted(user_costs.items(), key=lambda x: -x[1]):
        print(f"  {user:20s}: ${cost:.4f}")

    # Projected monthly cost
    total_week = sum(daily_costs.values())
    projected_monthly = (total_week / 7) * 30
    print(f"\n--- Projection ---")
    print(f"  Weekly Spend:        ${total_week:.4f}")
    print(f"  Projected Monthly:   ${projected_monthly:.4f}")
    budget = 50.0
    pct = (projected_monthly / budget) * 100
    status = "OK" if pct < 80 else "WARNING" if pct < 100 else "OVER BUDGET"
    print(f"  Budget ($50/month):  {pct:.1f}% [{status}]")


# ============================================================
# DEMO 3: Latency Monitoring
# ============================================================
def demo_latency_monitoring():
    """
    Track p50/p95/p99 latency. Detect anomalies. Alert when
    latency exceeds configurable thresholds.
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Latency Monitoring")
    print("=" * 60)

    ALERT_THRESHOLD_MS = 3000  # Alert if latency > 3 seconds
    latencies = []

    queries = [
        "What is aspirin used for?",
        "Define tachycardia.",
        "List symptoms of pneumonia.",
        "What is the normal blood pressure range?",
        "Explain the mechanism of ACE inhibitors.",
    ]

    print(f"\nMaking {len(queries)} monitored calls...\n")
    print(f"{'#':<4} {'Query':<45} {'Latency':>10} {'Status':>10}")
    print("-" * 72)

    for i, query in enumerate(queries, 1):
        result = make_monitored_call(
            messages=[
                {"role": "system", "content": "You are a medical assistant. Answer in one sentence."},
                {"role": "user", "content": query},
            ],
            model="gpt-4o-mini",
            endpoint="medical_qa",
        )

        latency_ms = result["metadata"]["latency_ms"]
        latencies.append(latency_ms)

        status = "OK" if latency_ms < ALERT_THRESHOLD_MS else "ALERT!"
        print(f"{i:<4} {query[:43]:<45} {latency_ms:>8.0f}ms {status:>10}")

    # Add simulated historical latencies for richer stats
    import random
    random.seed(99)
    historical = [random.gauss(800, 200) for _ in range(50)]
    all_latencies = historical + latencies

    # Percentile calculations
    sorted_lat = sorted(all_latencies)
    n = len(sorted_lat)

    p50 = sorted_lat[int(n * 0.50)]
    p95 = sorted_lat[int(n * 0.95)]
    p99 = sorted_lat[int(n * 0.99)]

    print(f"\n--- Latency Percentiles (n={n}) ---")
    print(f"  p50:  {p50:.0f}ms")
    print(f"  p95:  {p95:.0f}ms")
    print(f"  p99:  {p99:.0f}ms")
    print(f"  min:  {min(all_latencies):.0f}ms")
    print(f"  max:  {max(all_latencies):.0f}ms")

    # Anomaly detection (simple z-score)
    mean_lat = statistics.mean(all_latencies)
    std_lat = statistics.stdev(all_latencies)
    print(f"\n--- Anomaly Detection ---")
    print(f"  Mean: {mean_lat:.0f}ms  |  StdDev: {std_lat:.0f}ms")
    anomalies = [l for l in latencies if abs(l - mean_lat) > 2 * std_lat]
    if anomalies:
        print(f"  ANOMALIES DETECTED: {len(anomalies)} requests")
        for a in anomalies:
            z = (a - mean_lat) / std_lat
            print(f"    - {a:.0f}ms (z-score: {z:.2f})")
    else:
        print("  No anomalies detected (all within 2 std devs)")

    # Threshold check
    print(f"\n--- Threshold Alert (>{ALERT_THRESHOLD_MS}ms) ---")
    violations = [l for l in latencies if l > ALERT_THRESHOLD_MS]
    if violations:
        print(f"  ALERT: {len(violations)} requests exceeded threshold!")
    else:
        print(f"  All {len(latencies)} live requests within threshold.")


# ============================================================
# DEMO 4: Trace Viewer
# ============================================================
def demo_trace_viewer():
    """
    Build a trace viewer for agent interactions: show each step
    (LLM call → tool call → result) with timing.
    """
    print("\n" + "=" * 60)
    print("DEMO 4: Trace Viewer")
    print("=" * 60)

    trace_id = str(uuid.uuid4())[:8]
    trace = {
        "trace_id": trace_id,
        "start_time": datetime.now().isoformat(),
        "steps": [],
        "total_tokens": 0,
        "total_cost": 0.0,
    }

    print(f"\nTrace ID: {trace_id}")
    print("Simulating a clinical agent workflow...\n")

    # Step 1: Initial LLM call — understand the query
    step_start = time.time()
    result1 = make_monitored_call(
        messages=[
            {"role": "system", "content": (
                "You are a clinical decision support agent. "
                "Analyze the query and list what tools you would call. "
                "Available tools: drug_lookup, lab_reference, icd10_search. "
                "Respond in JSON format: {\"tools\": [\"tool_name\"], \"reasoning\": \"...\"}"
            )},
            {"role": "user", "content": "Patient has elevated ALT/AST. On metformin. Should I be concerned?"},
        ],
        model="gpt-4o-mini",
        endpoint="clinical_agent",
    )
    trace["steps"].append({
        "step": 1,
        "type": "llm_call",
        "description": "Query analysis and tool planning",
        "duration_ms": round((time.time() - step_start) * 1000, 2),
        "tokens": result1["metadata"]["total_tokens"],
        "cost": result1["metadata"]["cost_usd"],
    })
    trace["total_tokens"] += result1["metadata"]["total_tokens"]
    trace["total_cost"] += result1["metadata"]["cost_usd"]

    # Step 2: Simulated tool call — drug lookup
    step_start = time.time()
    time.sleep(0.05)  # Simulate tool call latency
    drug_info = {
        "drug": "metformin",
        "hepatic_effects": "Rare hepatotoxicity. Monitor LFTs if elevated.",
        "contraindication": "Severe hepatic impairment",
    }
    trace["steps"].append({
        "step": 2,
        "type": "tool_call",
        "tool": "drug_lookup",
        "description": "Look up metformin hepatic effects",
        "duration_ms": round((time.time() - step_start) * 1000, 2),
        "result_summary": json.dumps(drug_info)[:100],
    })

    # Step 3: Simulated tool call — lab reference
    step_start = time.time()
    time.sleep(0.03)
    lab_info = {
        "test": "ALT/AST",
        "normal_range": "ALT: 7-56 U/L, AST: 10-40 U/L",
        "clinical_significance": "Elevated levels indicate liver damage or disease",
    }
    trace["steps"].append({
        "step": 3,
        "type": "tool_call",
        "tool": "lab_reference",
        "description": "Look up ALT/AST reference ranges",
        "duration_ms": round((time.time() - step_start) * 1000, 2),
        "result_summary": json.dumps(lab_info)[:100],
    })

    # Step 4: Final LLM call — synthesize response
    step_start = time.time()
    result2 = make_monitored_call(
        messages=[
            {"role": "system", "content": "You are a clinical decision support agent. Synthesize findings concisely."},
            {"role": "user", "content": "Patient has elevated ALT/AST. On metformin. Should I be concerned?"},
            {"role": "assistant", "content": f"Tool results: Drug info: {json.dumps(drug_info)}. Lab info: {json.dumps(lab_info)}"},
            {"role": "user", "content": "Based on these tool results, provide your clinical assessment."},
        ],
        model="gpt-4o-mini",
        endpoint="clinical_agent",
    )
    trace["steps"].append({
        "step": 4,
        "type": "llm_call",
        "description": "Synthesize clinical assessment",
        "duration_ms": round((time.time() - step_start) * 1000, 2),
        "tokens": result2["metadata"]["total_tokens"],
        "cost": result2["metadata"]["cost_usd"],
    })
    trace["total_tokens"] += result2["metadata"]["total_tokens"]
    trace["total_cost"] += result2["metadata"]["cost_usd"]

    trace["end_time"] = datetime.now().isoformat()
    total_duration = sum(s["duration_ms"] for s in trace["steps"])
    trace["total_duration_ms"] = total_duration

    # Display trace tree
    print("┌─ TRACE: Clinical Agent Workflow")
    print(f"│  Trace ID: {trace_id}")
    print(f"│  Started:  {trace['start_time']}")
    print("│")
    for step in trace["steps"]:
        icon = "🔷" if step["type"] == "llm_call" else "🔧"
        print(f"├── Step {step['step']}: [{step['type'].upper()}] {step['description']}")
        print(f"│   Duration: {step['duration_ms']:.0f}ms", end="")
        if "tokens" in step:
            print(f"  |  Tokens: {step['tokens']}  |  Cost: ${step['cost']:.6f}")
        else:
            print(f"  |  Result: {step.get('result_summary', 'N/A')[:60]}...")
        print("│")

    print(f"└── TOTAL: {total_duration:.0f}ms  |  {trace['total_tokens']} tokens  |  ${trace['total_cost']:.6f}")

    # Save trace
    trace_file = os.path.join(LOG_DIR, f"trace_{trace_id}.json")
    with open(trace_file, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"\nTrace saved to: {trace_file}")

    # Final response
    print(f"\n--- Agent Response ---")
    print(result2["response"][:500])


# ============================================================
# Main Menu
# ============================================================
def main():
    """Main entry point with interactive demo menu."""
    print("\n" + "=" * 60)
    print("  Level 8 - Project 03: Monitoring & Observability")
    print("=" * 60)
    print("\nDemos:")
    print("  1. Request Logging")
    print("  2. Cost Tracking")
    print("  3. Latency Monitoring")
    print("  4. Trace Viewer")
    print("  5. Run All Demos")
    print("  0. Exit")

    while True:
        choice = input("\nSelect demo (0-5): ").strip()
        if choice == "1":
            demo_request_logging()
        elif choice == "2":
            demo_cost_tracking()
        elif choice == "3":
            demo_latency_monitoring()
        elif choice == "4":
            demo_trace_viewer()
        elif choice == "5":
            demo_request_logging()
            demo_cost_tracking()
            demo_latency_monitoring()
            demo_trace_viewer()
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 0-5.")


if __name__ == "__main__":
    main()
