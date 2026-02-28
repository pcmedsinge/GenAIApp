"""
Exercise 4: Load Testing
==========================
Simulate load testing for LLM API endpoints.

Requirements:
- Send concurrent requests at varying concurrency levels
- Measure throughput, latency under load, and error rates
- Identify bottleneck concurrency levels
- Generate a load test report with performance curves
- Implement rate limiting and graceful degradation

Healthcare Context:
  Hospital systems face unpredictable load patterns — ER surges,
  shift changes, and emergency events can spike traffic. Load testing
  ensures the AI system degrades gracefully, not catastrophically.

Usage:
    python exercise_4_load_testing.py
"""

from openai import OpenAI
import time
import json
import os
import statistics
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

client = OpenAI()

MEDICAL_QUERIES = [
    "What are the symptoms of pneumonia?",
    "Explain the mechanism of ACE inhibitors.",
    "What is the normal range for INR?",
    "List contraindications for metformin.",
    "What are the warning signs of sepsis?",
    "Describe the treatment protocol for acute MI.",
    "What labs should be ordered for suspected DKA?",
    "Explain the Glasgow Coma Scale.",
    "What are common side effects of statins?",
    "Describe the ABCDE approach to trauma assessment.",
]


class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, max_requests_per_second: float = 5.0):
        self.max_rps = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second
        self.last_request_time = 0.0
        self._lock = None

    def acquire(self) -> bool:
        """Wait until a request is allowed. Returns True."""
        import threading
        if self._lock is None:
            self._lock = threading.Lock()

        with self._lock:
            now = time.time()
            wait_time = self.min_interval - (now - self.last_request_time)
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_request_time = time.time()
        return True


class LoadTester:
    """Load testing framework for LLM API endpoints."""

    def __init__(self, rate_limiter: RateLimiter = None):
        self.rate_limiter = rate_limiter
        self.results = []

    def _make_request(self, query: str, request_id: int) -> dict:
        """Make a single timed API request."""
        if self.rate_limiter:
            self.rate_limiter.acquire()

        start = time.time()
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical assistant. Answer in 1-2 sentences."},
                    {"role": "user", "content": query},
                ],
            )
            latency = time.time() - start
            return {
                "request_id": request_id,
                "status": "success",
                "latency_ms": round(latency * 1000, 2),
                "tokens": response.usage.total_tokens,
                "query": query[:40],
            }
        except Exception as e:
            latency = time.time() - start
            return {
                "request_id": request_id,
                "status": "error",
                "latency_ms": round(latency * 1000, 2),
                "error": str(e)[:100],
                "query": query[:40],
            }

    def run_load_test(self, concurrency: int, total_requests: int) -> dict:
        """Run a load test at a specific concurrency level."""
        print(f"\n  Testing concurrency={concurrency}, requests={total_requests}...")
        results = []
        queries = [MEDICAL_QUERIES[i % len(MEDICAL_QUERIES)] for i in range(total_requests)]

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(self._make_request, q, i): i
                for i, q in enumerate(queries)
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        total_time = time.time() - start_time

        # Calculate metrics
        latencies = [r["latency_ms"] for r in results if r["status"] == "success"]
        errors = [r for r in results if r["status"] == "error"]

        if latencies:
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            metrics = {
                "concurrency": concurrency,
                "total_requests": total_requests,
                "successful": len(latencies),
                "errors": len(errors),
                "error_rate_pct": round(len(errors) / total_requests * 100, 1),
                "total_time_s": round(total_time, 2),
                "throughput_rps": round(total_requests / total_time, 2),
                "latency_min_ms": round(min(latencies), 0),
                "latency_p50_ms": round(sorted_lat[int(n * 0.50)], 0),
                "latency_p95_ms": round(sorted_lat[int(n * 0.95)], 0) if n >= 20 else round(sorted_lat[-1], 0),
                "latency_p99_ms": round(sorted_lat[int(n * 0.99)], 0) if n >= 100 else round(sorted_lat[-1], 0),
                "latency_max_ms": round(max(latencies), 0),
            }
        else:
            metrics = {
                "concurrency": concurrency,
                "total_requests": total_requests,
                "successful": 0,
                "errors": len(errors),
                "error_rate_pct": 100.0,
                "total_time_s": round(total_time, 2),
                "throughput_rps": 0,
            }

        self.results.append(metrics)
        return metrics

    def run_sweep(self, concurrency_levels: list, requests_per_level: int = 6):
        """Run load tests across multiple concurrency levels."""
        print("\n" + "=" * 60)
        print("  LOAD TEST SWEEP")
        print("=" * 60)
        print(f"  Concurrency levels: {concurrency_levels}")
        print(f"  Requests per level: {requests_per_level}")

        self.results = []
        for c in concurrency_levels:
            metrics = self.run_load_test(c, requests_per_level)
            print(f"    -> Throughput: {metrics['throughput_rps']:.1f} rps, "
                  f"p50: {metrics.get('latency_p50_ms', 'N/A')}ms, "
                  f"Errors: {metrics['error_rate_pct']}%")

    def generate_report(self):
        """Generate comprehensive load test report."""
        if not self.results:
            print("No results to report.")
            return

        print("\n" + "=" * 70)
        print("  LOAD TEST REPORT")
        print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        # Summary table
        print(f"\n{'Conc':>6} {'Reqs':>6} {'OK':>6} {'Err%':>6} {'RPS':>8} "
              f"{'p50ms':>8} {'p95ms':>8} {'maxms':>8}")
        print("-" * 66)
        for m in self.results:
            print(f"{m['concurrency']:>6} {m['total_requests']:>6} "
                  f"{m['successful']:>6} {m['error_rate_pct']:>5.1f}% "
                  f"{m.get('throughput_rps', 0):>7.1f} "
                  f"{m.get('latency_p50_ms', '-'):>8} "
                  f"{m.get('latency_p95_ms', '-'):>8} "
                  f"{m.get('latency_max_ms', '-'):>8}")

        # Throughput chart
        print(f"\n--- Throughput by Concurrency ---")
        max_rps = max(m.get("throughput_rps", 0) for m in self.results)
        for m in self.results:
            rps = m.get("throughput_rps", 0)
            bar_len = int((rps / max_rps) * 30) if max_rps > 0 else 0
            bar = "█" * bar_len
            print(f"  C={m['concurrency']:<3} {bar} {rps:.1f} rps")

        # Latency chart
        print(f"\n--- p50 Latency by Concurrency ---")
        max_lat = max(m.get("latency_p50_ms", 0) for m in self.results)
        for m in self.results:
            lat = m.get("latency_p50_ms", 0)
            bar_len = int((lat / max_lat) * 30) if max_lat > 0 else 0
            bar = "█" * bar_len
            print(f"  C={m['concurrency']:<3} {bar} {lat}ms")

        # Bottleneck analysis
        print(f"\n--- Bottleneck Analysis ---")
        # Find the concurrency level where error rate or latency spikes
        best_throughput = max(self.results, key=lambda m: m.get("throughput_rps", 0))
        print(f"  Peak Throughput: {best_throughput['throughput_rps']:.1f} rps at C={best_throughput['concurrency']}")

        high_error = [m for m in self.results if m.get("error_rate_pct", 0) > 5]
        if high_error:
            first_error = min(high_error, key=lambda m: m["concurrency"])
            print(f"  Error Rate Spike: at C={first_error['concurrency']} ({first_error['error_rate_pct']}% errors)")
        else:
            print(f"  Error Rate: Stable across all concurrency levels")

        # Recommendations
        print(f"\n--- Recommendations ---")
        if best_throughput["concurrency"] == self.results[-1]["concurrency"]:
            print("  - Throughput still increasing — consider testing higher concurrency")
        if high_error:
            print(f"  - Implement rate limiting below C={first_error['concurrency']} to prevent errors")
        print("  - Consider caching for repeated queries to reduce API load")
        print("  - Use async/streaming for better user experience under load")
        print("  - Set up auto-scaling if traffic exceeds peak throughput")


def main():
    """Run the load testing exercise."""
    print("=" * 60)
    print("  Exercise 4: Load Testing")
    print("=" * 60)

    # Create rate limiter (adjust based on your API tier)
    rate_limiter = RateLimiter(max_requests_per_second=3.0)
    tester = LoadTester(rate_limiter=rate_limiter)

    print("\nRate limiter: 3 requests/second")
    print("Starting load test sweep across concurrency levels...")
    print("(Using small request counts to conserve API credits)\n")

    # Test with small numbers to avoid excessive API costs
    tester.run_sweep(
        concurrency_levels=[1, 2, 3],
        requests_per_level=4,
    )

    # Generate report
    tester.generate_report()

    # Save results
    report_path = "load_test_results.json"
    with open(report_path, "w") as f:
        json.dump(tester.results, f, indent=2)
    print(f"\nRaw results saved to: {report_path}")
    print("Load testing exercise complete!")


if __name__ == "__main__":
    main()
