"""
Exercise 4 — API Testing
=========================
Build an automated test suite for the Medical Chat API using httpx + pytest
patterns. Tests cover:

    • Happy-path requests (200 OK)
    • Input validation errors (422)
    • Authentication failures (401)
    • Rate-limit enforcement (429)
    • Concurrent request handling
    • Response schema validation

The test suite can run against a live server or use a mock/OpenAI stub.

Usage:
    # Run tests (no server needed — uses TestClient)
    python exercise_4_api_testing.py

    # Or with pytest (if installed)
    pytest exercise_4_api_testing.py -v
"""

import os
import sys
import time
import json
import asyncio
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, HTTPException, Depends, Header, status
    from fastapi.testclient import TestClient
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️  pip install fastapi uvicorn pydantic httpx")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("⚠️  pip install httpx")


# ---------------------------------------------------------------------------
# Minimal API under test (self-contained for portability)
# ---------------------------------------------------------------------------
if FASTAPI_AVAILABLE:
    test_app = FastAPI(title="Test Medical API")

    API_KEYS = {"valid-key-001": "tester", "valid-key-002": "tester2"}
    _request_counts: dict[str, list[float]] = {}
    RATE_LIMIT_PER_MIN = 5

    async def auth_dep(x_api_key: str = Header(...)):
        if x_api_key not in API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return x_api_key

    async def rate_dep(x_api_key: str = Depends(auth_dep)):
        now = time.time()
        ts = _request_counts.setdefault(x_api_key, [])
        ts[:] = [t for t in ts if now - t < 60]
        if len(ts) >= RATE_LIMIT_PER_MIN:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        ts.append(now)
        return x_api_key

    class ChatReq(BaseModel):
        message: str = Field(..., min_length=1, max_length=2000)

    class ChatResp(BaseModel):
        reply: str
        model: str
        disclaimer: str = "AI-generated — not medical advice."

    @test_app.post("/chat", response_model=ChatResp, dependencies=[Depends(rate_dep)])
    def chat(req: ChatReq):
        return ChatResp(reply=f"Echo: {req.message}", model="mock")

    @test_app.get("/health")
    def health():
        return {"status": "ok"}


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    detail: str = ""


@dataclass
class TestReport:
    results: list[TestResult] = field(default_factory=list)

    def add(self, result: TestResult):
        self.results.append(result)

    def summary(self) -> str:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        lines = [
            "\n" + "=" * 60,
            "TEST REPORT",
            "=" * 60,
            f"Total: {total}   Passed: {passed}   Failed: {failed}",
            "-" * 60,
        ]
        for r in self.results:
            icon = "✅" if r.passed else "❌"
            lines.append(f"  {icon} {r.name:45s} {r.duration_ms:7.1f} ms")
            if not r.passed and r.detail:
                lines.append(f"      ↳ {r.detail}")
        lines.append("-" * 60)
        rate = (passed / total * 100) if total else 0
        lines.append(f"Pass rate: {rate:.0f}%")
        return "\n".join(lines)


def run_test(report: TestReport, name: str, fn):
    """Execute a single test function and record the result."""
    start = time.perf_counter()
    try:
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        report.add(TestResult(name=name, passed=True, duration_ms=elapsed))
    except AssertionError as exc:
        elapsed = (time.perf_counter() - start) * 1000
        report.add(TestResult(name=name, passed=False, duration_ms=elapsed, detail=str(exc)))
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        report.add(TestResult(name=name, passed=False, duration_ms=elapsed, detail=f"{type(exc).__name__}: {exc}"))


# Alias for typo-safe assertion base
AssertionError = AssertionError if False else AssertionError  # noqa


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def test_health(client: "TestClient"):
    """GET /health returns 200 OK."""
    def _test():
        r = client.get("/health")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        assert r.json()["status"] == "ok"
    return _test


def test_chat_happy_path(client: "TestClient"):
    """POST /chat with valid key and body → 200."""
    def _test():
        r = client.post(
            "/chat",
            json={"message": "What are the symptoms of flu?"},
            headers={"X-API-Key": "valid-key-001"},
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        assert "reply" in body
        assert "disclaimer" in body
    return _test


def test_chat_missing_key(client: "TestClient"):
    """POST /chat without an API key → 422 (missing header)."""
    def _test():
        r = client.post("/chat", json={"message": "hello"})
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    return _test


def test_chat_invalid_key(client: "TestClient"):
    """POST /chat with a bad API key → 401."""
    def _test():
        r = client.post(
            "/chat",
            json={"message": "hello"},
            headers={"X-API-Key": "bad-key-999"},
        )
        assert r.status_code == 401, f"Expected 401, got {r.status_code}"
    return _test


def test_chat_empty_message(client: "TestClient"):
    """POST /chat with empty message → 422 (validation error)."""
    def _test():
        r = client.post(
            "/chat",
            json={"message": ""},
            headers={"X-API-Key": "valid-key-001"},
        )
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    return _test


def test_chat_too_long_message(client: "TestClient"):
    """POST /chat with message > 2000 chars → 422."""
    def _test():
        r = client.post(
            "/chat",
            json={"message": "x" * 2500},
            headers={"X-API-Key": "valid-key-001"},
        )
        assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    return _test


def test_rate_limit(client: "TestClient"):
    """Exceed rate limit → 429."""
    def _test():
        _request_counts.clear()
        key = "valid-key-002"
        headers = {"X-API-Key": key}
        for i in range(RATE_LIMIT_PER_MIN):
            r = client.post("/chat", json={"message": f"req {i}"}, headers=headers)
            assert r.status_code == 200, f"Request {i} failed: {r.status_code}"

        # Next request should be rate-limited
        r = client.post("/chat", json={"message": "one too many"}, headers=headers)
        assert r.status_code == 429, f"Expected 429, got {r.status_code}"
    return _test


def test_response_schema(client: "TestClient"):
    """Response body matches ChatResp schema."""
    def _test():
        r = client.post(
            "/chat",
            json={"message": "test"},
            headers={"X-API-Key": "valid-key-001"},
        )
        body = r.json()
        required_fields = {"reply", "model", "disclaimer"}
        missing = required_fields - set(body.keys())
        assert not missing, f"Missing fields: {missing}"
    return _test


def test_concurrent_requests(client: "TestClient"):
    """Multiple sequential requests return correct results."""
    def _test():
        _request_counts.clear()
        messages = ["headache", "fever", "cough"]
        responses = []
        for msg in messages:
            r = client.post(
                "/chat",
                json={"message": msg},
                headers={"X-API-Key": "valid-key-001"},
            )
            assert r.status_code == 200
            responses.append(r.json())
        # Each reply should echo the corresponding message
        for msg, resp in zip(messages, responses):
            assert msg in resp["reply"], f"Expected '{msg}' in reply"
    return _test


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Exercise 4 — API Testing")
    print("=" * 60)

    if not FASTAPI_AVAILABLE:
        print("FastAPI is required. Install it and retry.")
        return

    # Reset rate limiter state
    _request_counts.clear()

    client = TestClient(test_app)
    report = TestReport()

    tests = [
        ("GET /health returns 200", test_health(client)),
        ("POST /chat happy path", test_chat_happy_path(client)),
        ("POST /chat missing API key → 422", test_chat_missing_key(client)),
        ("POST /chat invalid API key → 401", test_chat_invalid_key(client)),
        ("POST /chat empty message → 422", test_chat_empty_message(client)),
        ("POST /chat message too long → 422", test_chat_too_long_message(client)),
        ("Rate limit enforced → 429", test_rate_limit(client)),
        ("Response schema validation", test_response_schema(client)),
        ("Concurrent requests", test_concurrent_requests(client)),
    ]

    for name, fn in tests:
        run_test(report, name, fn)

    print(report.summary())

    # Return exit code for CI
    failed = sum(1 for r in report.results if not r.passed)
    if failed:
        print(f"\n⚠️  {failed} test(s) failed.")
    else:
        print("\n🎉 All tests passed!")
    return failed


if __name__ == "__main__":
    sys.exit(main() or 0)
