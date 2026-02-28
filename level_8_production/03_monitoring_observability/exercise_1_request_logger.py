"""
Exercise 1: Comprehensive Request Logger
=========================================
Build a production-grade request logger for LLM API calls.

Requirements:
- Log every request/response with full metadata (tokens, cost, latency, model)
- Store as JSONL (one JSON object per line) for easy streaming/querying
- Query logs by date range, model, endpoint, or user
- Calculate summary statistics from log data
- Support log rotation (new file per day)

Healthcare Context:
  Every AI-assisted clinical decision must be auditable. This logger
  creates the foundation for HIPAA-compliant audit trails.

Usage:
    python exercise_1_request_logger.py
"""

from openai import OpenAI
import time
import json
import os
import uuid
from datetime import datetime, timedelta
from collections import defaultdict

client = OpenAI()

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

MODEL_COSTS_PER_1K = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}


class RequestLogger:
    """Comprehensive JSONL request logger for LLM API calls."""

    def __init__(self, log_dir: str = LOG_DIR):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def _get_log_path(self, date: str = None) -> str:
        """Get log file path for a given date (supports log rotation)."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"requests_{date}.jsonl")

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD for a given call."""
        rates = MODEL_COSTS_PER_1K.get(model, {"input": 0.005, "output": 0.015})
        return round(
            (prompt_tokens / 1000) * rates["input"]
            + (completion_tokens / 1000) * rates["output"],
            8,
        )

    def log_request(
        self,
        messages: list,
        model: str = "gpt-4o-mini",
        endpoint: str = "default",
        user_id: str = "anonymous",
        metadata: dict = None,
    ) -> dict:
        """Make an LLM call, log it, and return response with metadata."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        error_msg = None
        response_content = ""
        prompt_tokens = 0
        completion_tokens = 0

        try:
            response = client.chat.completions.create(model=model, messages=messages)
            response_content = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            status = "success"
        except Exception as e:
            status = "error"
            error_msg = str(e)

        latency_ms = round((time.time() - start_time) * 1000, 2)
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "model": model,
            "endpoint": endpoint,
            "user_id": user_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": cost,
            "latency_ms": latency_ms,
            "status": status,
            "error": error_msg,
            "request_messages": [
                {"role": m["role"], "content": m["content"][:200]} for m in messages
            ],
            "response_preview": response_content[:300] if response_content else None,
            "metadata": metadata or {},
        }

        # Write to daily log file
        log_path = self._get_log_path()
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return {
            "response": response_content,
            "log_entry": log_entry,
        }

    def query_logs(
        self,
        date_from: str = None,
        date_to: str = None,
        model: str = None,
        endpoint: str = None,
        user_id: str = None,
        status: str = None,
    ) -> list:
        """Query logs with optional filters."""
        if date_from is None:
            date_from = datetime.now().strftime("%Y-%m-%d")
        if date_to is None:
            date_to = date_from

        results = []
        current = datetime.strptime(date_from, "%Y-%m-%d")
        end = datetime.strptime(date_to, "%Y-%m-%d")

        while current <= end:
            log_path = self._get_log_path(current.strftime("%Y-%m-%d"))
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        if model and entry["model"] != model:
                            continue
                        if endpoint and entry["endpoint"] != endpoint:
                            continue
                        if user_id and entry["user_id"] != user_id:
                            continue
                        if status and entry["status"] != status:
                            continue
                        results.append(entry)
            current += timedelta(days=1)

        return results

    def get_summary(self, logs: list) -> dict:
        """Calculate summary statistics from a list of log entries."""
        if not logs:
            return {"total_requests": 0}

        total_tokens = sum(l["total_tokens"] for l in logs)
        total_cost = sum(l["cost_usd"] for l in logs)
        latencies = [l["latency_ms"] for l in logs]
        errors = sum(1 for l in logs if l["status"] == "error")

        by_model = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0})
        by_endpoint = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0})
        by_user = defaultdict(lambda: {"requests": 0, "cost": 0.0})

        for l in logs:
            by_model[l["model"]]["requests"] += 1
            by_model[l["model"]]["tokens"] += l["total_tokens"]
            by_model[l["model"]]["cost"] += l["cost_usd"]
            by_endpoint[l["endpoint"]]["requests"] += 1
            by_endpoint[l["endpoint"]]["tokens"] += l["total_tokens"]
            by_endpoint[l["endpoint"]]["cost"] += l["cost_usd"]
            by_user[l["user_id"]]["requests"] += 1
            by_user[l["user_id"]]["cost"] += l["cost_usd"]

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        return {
            "total_requests": len(logs),
            "success_rate": f"{((len(logs) - errors) / len(logs)) * 100:.1f}%",
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "avg_latency_ms": round(sum(latencies) / n, 1),
            "p50_latency_ms": round(sorted_latencies[int(n * 0.50)], 1),
            "p95_latency_ms": round(sorted_latencies[int(n * 0.95)], 1) if n >= 20 else "N/A (need 20+ samples)",
            "by_model": dict(by_model),
            "by_endpoint": dict(by_endpoint),
            "by_user": dict(by_user),
        }

    def display_summary(self, summary: dict):
        """Pretty-print a summary report."""
        print("\n" + "=" * 50)
        print("  REQUEST LOG SUMMARY")
        print("=" * 50)
        print(f"  Total Requests:  {summary['total_requests']}")
        print(f"  Success Rate:    {summary['success_rate']}")
        print(f"  Total Tokens:    {summary['total_tokens']}")
        print(f"  Total Cost:      ${summary['total_cost_usd']:.6f}")
        print(f"  Avg Latency:     {summary['avg_latency_ms']}ms")
        print(f"  p50 Latency:     {summary['p50_latency_ms']}ms")
        print(f"  p95 Latency:     {summary['p95_latency_ms']}")

        if summary.get("by_model"):
            print("\n  By Model:")
            for model, data in summary["by_model"].items():
                print(f"    {model}: {data['requests']} reqs, {data['tokens']} tokens, ${data['cost']:.6f}")

        if summary.get("by_endpoint"):
            print("\n  By Endpoint:")
            for ep, data in summary["by_endpoint"].items():
                print(f"    {ep}: {data['requests']} reqs, ${data['cost']:.6f}")

        if summary.get("by_user"):
            print("\n  By User:")
            for user, data in summary["by_user"].items():
                print(f"    {user}: {data['requests']} reqs, ${data['cost']:.6f}")


def main():
    """Run the request logger exercise."""
    print("=" * 50)
    print("  Exercise 1: Comprehensive Request Logger")
    print("=" * 50)

    logger = RequestLogger()

    # Medical queries to log
    queries = [
        ("What are the symptoms of acute appendicitis?", "symptom_check", "dr_martinez"),
        ("Explain the pharmacokinetics of warfarin.", "drug_info", "pharmacist_lee"),
        ("What is the Glasgow Coma Scale?", "clinical_reference", "nurse_johnson"),
        ("Differential diagnosis for chest pain in a 55-year-old male.", "diagnosis_assist", "dr_martinez"),
        ("What are contraindications for MRI?", "clinical_reference", "tech_williams"),
    ]

    print(f"\nLogging {len(queries)} medical queries...\n")

    for query, endpoint, user in queries:
        print(f"  [{endpoint}] {user}: {query[:50]}...")
        result = logger.log_request(
            messages=[
                {"role": "system", "content": "You are a medical information assistant. Be concise (2-3 sentences)."},
                {"role": "user", "content": query},
            ],
            model="gpt-4o-mini",
            endpoint=endpoint,
            user_id=user,
            metadata={"department": "emergency", "priority": "normal"},
        )
        entry = result["log_entry"]
        print(f"    -> {entry['total_tokens']} tokens, ${entry['cost_usd']:.6f}, {entry['latency_ms']:.0f}ms")

    # Query and summarize
    print("\n\nQuerying all logs for today...")
    today_logs = logger.query_logs()
    summary = logger.get_summary(today_logs)
    logger.display_summary(summary)

    # Query by specific user
    print("\n\nQuerying logs for dr_martinez...")
    dr_logs = logger.query_logs(user_id="dr_martinez")
    if dr_logs:
        dr_summary = logger.get_summary(dr_logs)
        logger.display_summary(dr_summary)

    print("\nDone! Log files stored in:", LOG_DIR)


if __name__ == "__main__":
    main()
