"""
Exercise 5: Retry & Error Recovery — Resilient Agent Patterns

Skills practiced:
- Retry with exponential backoff when LLM calls fail
- Fallback agents: if primary model fails, try a simpler/cheaper one
- Circuit breaker pattern: stop retrying after too many failures
- Graceful degradation: return partial results instead of crashing
- Error classification: transient vs permanent failures

Key insight: Production agents WILL fail. LLM APIs have rate limits,
  timeouts, malformed responses, and content filter rejections.
  A resilient agent handles ALL of these gracefully.

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │                    RESILIENT AGENT                   │
  │                                                     │
  │  ┌──────────┐    ┌──────────┐    ┌──────────┐      │
  │  │  Retry   │    │ Fallback │    │ Circuit  │      │
  │  │  Engine  │    │ Chain    │    │ Breaker  │      │
  │  └────┬─────┘    └────┬─────┘    └────┬─────┘      │
  │       │               │               │             │
  │       ▼               ▼               ▼             │
  │  Try primary ──fail──► Try fallback ──► Check if   │
  │  model with           model             circuit     │
  │  backoff                                is open     │
  │       │                   │                │        │
  │       ▼                   ▼                ▼        │
  │  ┌─────────┐    ┌──────────────┐  ┌────────────┐   │
  │  │ Success │    │ Partial      │  │ Graceful   │   │
  │  │ (full   │    │ Result       │  │ Degrade    │   │
  │  │ quality)│    │ (acceptable) │  │ (safe msg) │   │
  │  └─────────┘    └──────────────┘  └────────────┘   │
  │                                                     │
  │  Error Classification:                              │
  │  ├── Transient: timeout, rate_limit → RETRY         │
  │  ├── Recoverable: bad JSON → RETRY with fix         │
  │  └── Permanent: content_filter → FALLBACK/DEGRADE   │
  └─────────────────────────────────────────────────────┘

Healthcare parallel: A hospital doesn't shut down when the CT
scanner breaks. It has backup protocols, alternative imaging,
and escalation procedures. Your agent should too.
"""

import os
import json
import time
import random
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError

client = OpenAI()


# ============================================================
# Error Classification
# ============================================================

class ErrorType(Enum):
    TRANSIENT = "transient"         # Timeout, rate limit → retry
    RECOVERABLE = "recoverable"     # Bad JSON, truncated → retry with fix
    PERMANENT = "permanent"         # Content filter, invalid request → fallback


def classify_error(error: Exception) -> ErrorType:
    """
    Classify an error to determine the right recovery strategy.

    This is critical: retrying a permanent error wastes tokens and time.
    Not retrying a transient error causes unnecessary failures.
    """
    if isinstance(error, (APITimeoutError, APIConnectionError)):
        return ErrorType.TRANSIENT
    elif isinstance(error, RateLimitError):
        return ErrorType.TRANSIENT
    elif isinstance(error, json.JSONDecodeError):
        return ErrorType.RECOVERABLE
    elif isinstance(error, APIError):
        status = getattr(error, 'status_code', None)
        if status and status >= 500:
            return ErrorType.TRANSIENT      # Server error → retry
        elif status == 429:
            return ErrorType.TRANSIENT      # Rate limit → retry
        elif status == 400:
            return ErrorType.PERMANENT      # Bad request → don't retry
        else:
            return ErrorType.PERMANENT
    else:
        return ErrorType.PERMANENT


# ============================================================
# Retry Engine
# ============================================================

class RetryEngine:
    """
    Retry with exponential backoff + jitter.

    Exponential backoff: wait 1s, 2s, 4s, 8s, ...
    Jitter: add random noise so multiple clients don't retry simultaneously
    (the "thundering herd" problem).
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                 max_delay: float = 30.0, jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.attempt_log = []

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff + optional jitter."""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        if self.jitter:
            delay = delay * (0.5 + random.random())  # 50-150% of calculated delay
        return delay

    def execute(self, func, *args, **kwargs):
        """
        Execute a function with retry logic.
        Returns (result, attempt_log) tuple.
        """
        self.attempt_log = []
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                start = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start

                self.attempt_log.append({
                    "attempt": attempt + 1,
                    "status": "success",
                    "elapsed": round(elapsed, 2),
                })
                return result

            except Exception as e:
                elapsed = time.time() - start
                error_type = classify_error(e)

                self.attempt_log.append({
                    "attempt": attempt + 1,
                    "status": "failed",
                    "error": str(e)[:100],
                    "error_type": error_type.value,
                    "elapsed": round(elapsed, 2),
                })

                last_error = e

                # Don't retry permanent errors
                if error_type == ErrorType.PERMANENT:
                    break

                # Don't wait after the last attempt
                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    self.attempt_log[-1]["retry_delay"] = round(delay, 2)
                    time.sleep(delay)

        raise last_error


# ============================================================
# Circuit Breaker
# ============================================================

class CircuitBreaker:
    """
    Circuit breaker: stop calling a failing service after too many failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject immediately (don't even try)
    - HALF_OPEN: After cooldown, try ONE request. If it works → CLOSED.
                 If it fails → OPEN again.

    Like an electrical circuit breaker: trips to protect the system.
    """

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 60.0):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        self.state_log = []

    def _record_state_change(self, new_state: str, reason: str):
        old_state = self.state
        self.state = new_state
        self.state_log.append({
            "from": old_state,
            "to": new_state,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "failure_count": self.failure_count,
        })

    def can_execute(self) -> bool:
        """Check if a request is allowed through."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            # Check if cooldown has elapsed
            if self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.cooldown_seconds:
                    self._record_state_change("HALF_OPEN",
                        f"Cooldown elapsed ({elapsed:.1f}s >= {self.cooldown_seconds}s)")
                    return True
            return False
        elif self.state == "HALF_OPEN":
            return True  # Allow one test request
        return False

    def record_success(self):
        """Record a successful call — reset failure count."""
        if self.state == "HALF_OPEN":
            self._record_state_change("CLOSED", "Test request succeeded")
        self.failure_count = 0

    def record_failure(self):
        """Record a failed call — may trip the breaker."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == "HALF_OPEN":
            self._record_state_change("OPEN",
                f"Test request failed, re-opening")
        elif self.failure_count >= self.failure_threshold:
            self._record_state_change("OPEN",
                f"Failure threshold reached ({self.failure_count}/{self.failure_threshold})")


# ============================================================
# Fallback Chain
# ============================================================

class FallbackChain:
    """
    Try multiple strategies in order. If primary fails, try fallback.

    Chain: Primary Model → Fallback Model → Cached Response → Safe Default

    Like a hospital: If CT scanner breaks → try MRI → try X-ray → clinical exam.
    """

    def __init__(self):
        self.chain = []
        self.execution_log = []

    def add_strategy(self, name: str, func, description: str = ""):
        """Add a fallback strategy to the chain."""
        self.chain.append({
            "name": name,
            "func": func,
            "description": description,
        })

    def execute(self, *args, **kwargs):
        """Try each strategy in order until one succeeds."""
        self.execution_log = []

        for strategy in self.chain:
            try:
                start = time.time()
                result = strategy["func"](*args, **kwargs)
                elapsed = time.time() - start

                self.execution_log.append({
                    "strategy": strategy["name"],
                    "status": "success",
                    "elapsed": round(elapsed, 2),
                })
                return result

            except Exception as e:
                elapsed = time.time() - start
                self.execution_log.append({
                    "strategy": strategy["name"],
                    "status": "failed",
                    "error": str(e)[:100],
                    "elapsed": round(elapsed, 2),
                })

        # All strategies failed — return safe default
        self.execution_log.append({
            "strategy": "safe_default",
            "status": "used",
            "elapsed": 0,
        })
        return {
            "status": "degraded",
            "message": "All strategies failed. Please consult a healthcare professional directly.",
            "strategies_tried": len(self.chain),
        }


# ============================================================
# Resilient Clinical Agent
# ============================================================

class ResilientClinicalAgent:
    """
    A clinical agent with full error recovery:
    - Retry with backoff for transient failures
    - Fallback model chain (gpt-4o-mini → gpt-3.5-turbo → cached)
    - Circuit breaker to avoid hammering a broken service
    - Structured output repair for malformed JSON
    - Graceful degradation when nothing works
    """

    def __init__(self, primary_model: str = "gpt-4o-mini",
                 fallback_model: str = "gpt-4o-mini",  # In production: a different/cheaper model
                 max_retries: int = 3):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.retry_engine = RetryEngine(max_retries=max_retries)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, cooldown_seconds=30)
        self.response_cache = {}
        self.stats = {
            "total_calls": 0,
            "primary_successes": 0,
            "fallback_successes": 0,
            "cache_hits": 0,
            "degraded_responses": 0,
            "total_retries": 0,
        }

    def _call_model(self, model: str, messages: list, expect_json: bool = False) -> str:
        """Make an LLM call — the basic unit that can fail."""
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": 0,
        }
        if expect_json:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content

        # Validate JSON if expected
        if expect_json:
            json.loads(content)  # Raises JSONDecodeError if malformed

        return content

    def _get_cache_key(self, messages: list) -> str:
        """Generate a cache key from the messages."""
        return str(hash(json.dumps(messages, sort_keys=True)))

    def _cached_response(self, messages: list) -> str:
        """Return cached response if available."""
        key = self._get_cache_key(messages)
        if key in self.response_cache:
            self.stats["cache_hits"] += 1
            return self.response_cache[key]
        raise KeyError("No cached response available")

    def call(self, messages: list, expect_json: bool = False) -> dict:
        """
        Make a resilient LLM call with full error recovery.

        Hierarchy:
        1. Check circuit breaker
        2. Try primary model with retry
        3. Try fallback model with retry
        4. Try cached response
        5. Return graceful degradation message
        """
        self.stats["total_calls"] += 1
        result = {
            "content": None,
            "strategy_used": None,
            "attempts": 0,
            "degraded": False,
        }

        # Step 1: Check circuit breaker
        if not self.circuit_breaker.can_execute():
            # Circuit is open — skip directly to cache/fallback
            result["strategy_used"] = "circuit_breaker_tripped"
            # Try cache
            try:
                result["content"] = self._cached_response(messages)
                result["strategy_used"] = "cache (circuit open)"
                return result
            except KeyError:
                result["content"] = self._graceful_degradation()
                result["degraded"] = True
                result["strategy_used"] = "degraded (circuit open)"
                self.stats["degraded_responses"] += 1
                return result

        # Step 2: Try primary model with retry
        try:
            content = self.retry_engine.execute(
                self._call_model, self.primary_model, messages, expect_json
            )
            self.circuit_breaker.record_success()
            self.stats["primary_successes"] += 1
            self.stats["total_retries"] += len(self.retry_engine.attempt_log) - 1

            # Cache the successful response
            key = self._get_cache_key(messages)
            self.response_cache[key] = content

            result["content"] = content
            result["strategy_used"] = "primary"
            result["attempts"] = len(self.retry_engine.attempt_log)
            return result

        except Exception:
            self.circuit_breaker.record_failure()

        # Step 3: Try fallback model with retry
        try:
            content = self.retry_engine.execute(
                self._call_model, self.fallback_model, messages, expect_json
            )
            self.stats["fallback_successes"] += 1
            self.stats["total_retries"] += len(self.retry_engine.attempt_log) - 1

            key = self._get_cache_key(messages)
            self.response_cache[key] = content

            result["content"] = content
            result["strategy_used"] = "fallback"
            result["attempts"] = len(self.retry_engine.attempt_log)
            return result

        except Exception:
            pass

        # Step 4: Try cache
        try:
            result["content"] = self._cached_response(messages)
            result["strategy_used"] = "cache"
            return result
        except KeyError:
            pass

        # Step 5: Graceful degradation
        result["content"] = self._graceful_degradation()
        result["degraded"] = True
        result["strategy_used"] = "degraded"
        self.stats["degraded_responses"] += 1
        return result

    def _graceful_degradation(self) -> str:
        """Return a safe message when all else fails."""
        return (
            "⚠️ SYSTEM NOTICE: The clinical decision support system is temporarily "
            "unavailable. All responses have been attempted through primary, fallback, "
            "and cached pathways.\n\n"
            "RECOMMENDED ACTIONS:\n"
            "1. Proceed with standard clinical protocols\n"
            "2. Consult attending physician directly\n"
            "3. Use hospital's offline reference materials\n"
            "4. Document that CDS was unavailable at time of decision\n\n"
            "This system will automatically retry when service is restored."
        )

    def analyze_patient(self, scenario: str, verbose: bool = True) -> dict:
        """
        Run a clinical analysis with full resilience.
        """
        if verbose:
            print(f"\n  Circuit breaker: {self.circuit_breaker.state}")
            print(f"  Cache entries: {len(self.response_cache)}")

        messages = [
            {"role": "system", "content": (
                "You are a clinical decision support system. Provide a structured "
                "assessment including: differential diagnosis, recommended labs, "
                "treatment plan, and safety considerations.\n"
                "Output JSON: {\"assessment\": \"...\", \"differential\": [...], "
                "\"labs\": [...], \"treatment\": \"...\", \"safety\": \"...\"}"
            )},
            {"role": "user", "content": scenario},
        ]

        result = self.call(messages, expect_json=True)

        if verbose:
            strategy = result["strategy_used"]
            attempts = result.get("attempts", 0)
            degraded = result["degraded"]
            print(f"  Strategy: {strategy} | Attempts: {attempts} | Degraded: {degraded}")

        return result


# ============================================================
# JSON Repair Agent
# ============================================================

class JSONRepairAgent:
    """
    Specialized agent that repairs malformed JSON from LLM responses.

    LLMs frequently produce broken JSON:
    - Trailing commas: {"a": 1,}
    - Missing quotes: {key: "value"}
    - Truncated output: {"a": 1, "b": (end of response)
    - Markdown wrapping: ```json { ... } ```

    This agent tries to fix common issues before asking the LLM to repair.
    """

    @staticmethod
    def attempt_repair(broken_json: str, verbose: bool = False) -> dict:
        """
        Try multiple repair strategies in order.
        """
        repairs_tried = []

        # Strategy 1: Strip markdown code fences
        cleaned = broken_json.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]  # Remove first line
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            repairs_tried.append("strip_markdown_fences")
            try:
                result = json.loads(cleaned)
                if verbose:
                    print(f"    Repair succeeded: strip_markdown_fences")
                return {"data": result, "repairs": repairs_tried, "success": True}
            except json.JSONDecodeError:
                pass

        # Strategy 2: Fix trailing commas
        import re
        fixed = re.sub(r',\s*([}\]])', r'\1', cleaned)
        repairs_tried.append("fix_trailing_commas")
        try:
            result = json.loads(fixed)
            if verbose:
                print(f"    Repair succeeded: fix_trailing_commas")
            return {"data": result, "repairs": repairs_tried, "success": True}
        except json.JSONDecodeError:
            pass

        # Strategy 3: Try to close unclosed brackets/braces
        open_braces = fixed.count("{") - fixed.count("}")
        open_brackets = fixed.count("[") - fixed.count("]")
        if open_braces > 0 or open_brackets > 0:
            patched = fixed
            # Trim to last complete value
            last_comma = patched.rfind(",")
            if last_comma > 0:
                patched = patched[:last_comma]
            patched += "}" * open_braces + "]" * open_brackets
            repairs_tried.append("close_brackets")
            try:
                result = json.loads(patched)
                if verbose:
                    print(f"    Repair succeeded: close_brackets")
                return {"data": result, "repairs": repairs_tried, "success": True}
            except json.JSONDecodeError:
                pass

        # Strategy 4: Ask the LLM to fix it
        repairs_tried.append("llm_repair")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "Fix this broken JSON. Return ONLY valid JSON, nothing else. "
                        "Preserve all data. Fix syntax errors only."
                    )},
                    {"role": "user", "content": broken_json},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            if verbose:
                print(f"    Repair succeeded: llm_repair")
            return {"data": result, "repairs": repairs_tried, "success": True}
        except Exception:
            pass

        return {"data": None, "repairs": repairs_tried, "success": False}


# ============================================================
# Demo Scenarios
# ============================================================

SCENARIO_ACS = """
Patient: 55-year-old male
Chief Complaint: Chest pain for 2 hours, radiating to left arm
History: Type 2 diabetes, hypertension, hyperlipidemia
Vitals: BP 158/92, HR 98, SpO2 96%
Labs: Troponin I 0.45 ng/mL (elevated), Glucose 210
ECG: ST depression V3-V6
""".strip()

SCENARIO_COMPLEX = """
Patient: 70-year-old male, found unresponsive
History: AFib, HF (EF 25%), DM2, CKD stage 4, prior stroke
Vitals: BP 92/54, HR 48, RR 8, SpO2 82%, GCS 7
Labs: Glucose 42, K+ 6.8, Creatinine 4.8, Lactate 6.2, pH 7.12
""".strip()


# ============================================================
# Demo Functions
# ============================================================

def demo_retry_with_backoff():
    """Show retry with exponential backoff."""
    print("\n" + "=" * 70)
    print("  DEMO 1: RETRY WITH EXPONENTIAL BACKOFF")
    print("=" * 70)
    print("""
  When LLM calls fail (timeout, rate limit), retry with increasing delays:
    Attempt 1: try immediately
    Attempt 2: wait ~1s, then retry
    Attempt 3: wait ~2s, then retry
    Attempt 4: wait ~4s, then retry

  The exponential delay prevents overwhelming a struggling service.
  Jitter (random noise) prevents multiple clients from retrying at the same time.
  """)

    # Show the backoff schedule
    engine = RetryEngine(max_retries=5, base_delay=1.0, max_delay=30.0, jitter=True)
    print("  Backoff schedule (with jitter):")
    for attempt in range(6):
        delay = engine.calculate_delay(attempt)
        bar = "█" * int(delay * 3)
        print(f"    Attempt {attempt + 1}: wait {delay:.2f}s {bar}")

    print(f"\n  Without jitter:")
    engine_no_jitter = RetryEngine(max_retries=5, base_delay=1.0, max_delay=30.0, jitter=False)
    for attempt in range(6):
        delay = engine_no_jitter.calculate_delay(attempt)
        bar = "█" * int(delay * 3)
        print(f"    Attempt {attempt + 1}: wait {delay:.2f}s {bar}")

    # Now make a real call (will likely succeed on first try)
    print(f"\n  Running real LLM call with retry protection...")
    agent = ResilientClinicalAgent(max_retries=3)
    result = agent.analyze_patient(SCENARIO_ACS)

    print(f"\n  Result strategy: {result['strategy_used']}")
    print(f"  Attempts needed: {result['attempts']}")
    print(f"  Content preview: {str(result['content'])[:200]}...")

    print(f"\n  Agent stats: {json.dumps(agent.stats, indent=4)}")


def demo_fallback_chain():
    """Show the fallback chain in action."""
    print("\n" + "=" * 70)
    print("  DEMO 2: FALLBACK CHAIN")
    print("=" * 70)
    print("""
  When the primary strategy fails, fall through to alternatives:

    Primary Model (best quality)
         ↓ fail
    Fallback Model (acceptable quality)
         ↓ fail
    Cached Response (stale but available)
         ↓ fail
    Safe Default (graceful degradation)

  Each level trades quality for availability.
  """)

    # Build a fallback chain
    chain = FallbackChain()

    # Strategy 1: Primary model
    def primary_call(scenario):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Provide a brief clinical assessment."},
                {"role": "user", "content": scenario},
            ],
            temperature=0,
        )
        return {"source": "primary", "content": response.choices[0].message.content}

    # Strategy 2: Simulated cached response
    def cached_response(scenario):
        # Simulate a cache that has generic responses
        return {
            "source": "cache",
            "content": (
                "GENERIC ASSESSMENT: Based on the clinical presentation, "
                "recommend standard workup including labs, imaging, and specialist "
                "consultation as appropriate. Refer to institutional protocols."
            ),
        }

    chain.add_strategy("Primary GPT-4o-mini", primary_call, "Best quality, may fail")
    chain.add_strategy("Cached Generic", cached_response, "Pre-computed fallback")

    print(f"  Chain has {len(chain.chain)} strategies. Running...\n")
    result = chain.execute(SCENARIO_ACS)

    print(f"  Result source: {result['source']}")
    print(f"  Content: {result['content'][:200]}...")

    print(f"\n  Execution log:")
    for entry in chain.execution_log:
        status = "✅" if entry["status"] == "success" else "❌"
        elapsed = entry.get("elapsed", 0)
        print(f"    {status} {entry['strategy']}: {entry['status']} ({elapsed}s)")


def demo_circuit_breaker():
    """Show the circuit breaker pattern."""
    print("\n" + "=" * 70)
    print("  DEMO 3: CIRCUIT BREAKER PATTERN")
    print("=" * 70)
    print("""
  The circuit breaker protects against cascading failures:

    CLOSED ──(failures exceed threshold)──► OPEN
       ▲                                      │
       │                                      │ (cooldown)
       │                                      ▼
       └──────(test request succeeds)──── HALF_OPEN

  When OPEN: Requests are rejected immediately (no wasted time/tokens)
  When HALF_OPEN: ONE test request is allowed through
  """)

    breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=5)

    print(f"  Initial state: {breaker.state}")
    print(f"  Failure threshold: {breaker.failure_threshold}")
    print(f"  Cooldown: {breaker.cooldown_seconds}s\n")

    # Simulate a series of failures
    for i in range(5):
        can_exec = breaker.can_execute()
        if can_exec:
            # Simulate failure
            breaker.record_failure()
            print(f"  Call {i + 1}: Allowed → Failed | State: {breaker.state} | Failures: {breaker.failure_count}")
        else:
            print(f"  Call {i + 1}: REJECTED (circuit {breaker.state}) | Failures: {breaker.failure_count}")

    # Wait for cooldown
    print(f"\n  Waiting {breaker.cooldown_seconds}s for cooldown...")
    time.sleep(breaker.cooldown_seconds + 0.5)

    # Try again — should be HALF_OPEN
    can_exec = breaker.can_execute()
    print(f"\n  After cooldown: can_execute={can_exec} | State: {breaker.state}")

    # Simulate success
    breaker.record_success()
    print(f"  Test succeeded! State: {breaker.state} | Failures: {breaker.failure_count}")

    print(f"\n  State transition log:")
    for entry in breaker.state_log:
        print(f"    {entry['from']} → {entry['to']}: {entry['reason']}")


def demo_json_repair():
    """Show JSON repair strategies."""
    print("\n" + "=" * 70)
    print("  DEMO 4: JSON REPAIR — FIXING MALFORMED LLM OUTPUT")
    print("=" * 70)
    print("""
  LLMs frequently produce broken JSON. Instead of crashing,
  a resilient agent tries multiple repair strategies:

    1. Strip markdown fences (```json ... ```)
    2. Fix trailing commas ({a: 1,} → {a: 1})
    3. Close unclosed brackets/braces
    4. Ask LLM to repair (last resort — costs tokens)
  """)

    repair_agent = JSONRepairAgent()

    test_cases = [
        {
            "name": "Markdown wrapped",
            "input": '```json\n{"diagnosis": "ACS", "confidence": 0.85}\n```',
        },
        {
            "name": "Trailing commas",
            "input": '{"diagnosis": "ACS", "labs": ["troponin", "ECG",], "urgent": true,}',
        },
        {
            "name": "Truncated output",
            "input": '{"diagnosis": "ACS", "plan": {"immediate": ["aspirin", "heparin"',
        },
        {
            "name": "Clean JSON (no repair needed)",
            "input": '{"diagnosis": "ACS", "confidence": 0.9, "urgent": true}',
        },
    ]

    for case in test_cases:
        print(f"\n  Case: {case['name']}")
        print(f"  Input:  {case['input'][:80]}...")

        result = repair_agent.attempt_repair(case["input"], verbose=True)

        if result["success"]:
            print(f"  Output: {json.dumps(result['data'])[:80]}...")
            print(f"  Repairs: {' → '.join(result['repairs'])}")
        else:
            print(f"  FAILED: Could not repair")
            print(f"  Tried: {' → '.join(result['repairs'])}")


def demo_full_resilience():
    """Show all patterns working together."""
    print("\n" + "=" * 70)
    print("  DEMO 5: FULL RESILIENCE — ALL PATTERNS COMBINED")
    print("=" * 70)
    print("""
  The complete resilient agent combines all patterns:
    Retry Engine + Circuit Breaker + Fallback Chain + JSON Repair

  We'll run the same agent multiple times to show how
  it builds cache entries and maintains circuit breaker state.
  """)

    agent = ResilientClinicalAgent(max_retries=2)

    scenarios = [SCENARIO_ACS, SCENARIO_COMPLEX, SCENARIO_ACS]  # Repeat ACS to show cache
    labels = ["ACS (first call)", "Complex (first call)", "ACS (may use cache)"]

    for scenario, label in zip(scenarios, labels):
        print(f"\n  ── {label} ──")
        result = agent.analyze_patient(scenario, verbose=True)
        content_preview = str(result["content"])[:150].replace("\n", " ")
        print(f"  Content: {content_preview}...")

    print(f"\n  ══ Final Agent Statistics ══")
    for key, value in agent.stats.items():
        print(f"    {key}: {value}")

    print(f"\n  Circuit breaker state: {agent.circuit_breaker.state}")
    print(f"  Cache entries: {len(agent.response_cache)}")


def demo_interactive():
    """Interactive resilient agent."""
    print("\n" + "=" * 70)
    print("  DEMO 6: INTERACTIVE RESILIENT AGENT")
    print("=" * 70)
    print("  Enter patient scenarios. The agent uses retry, fallback,")
    print("  and circuit breaker. Type 'stats' to see stats, 'quit' to exit.\n")

    agent = ResilientClinicalAgent(max_retries=2)

    while True:
        user_input = input("  > ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        elif user_input.lower() == 'stats':
            print(f"\n  Agent stats: {json.dumps(agent.stats, indent=4)}")
            print(f"  Circuit: {agent.circuit_breaker.state}")
            print(f"  Cache: {len(agent.response_cache)} entries\n")
            continue
        elif len(user_input) < 10:
            print("  Enter a clinical scenario or 'stats' / 'quit'.")
            continue

        result = agent.analyze_patient(user_input)
        content = str(result["content"])
        for line in content.split("\n"):
            print(f"  {line}")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 5: RETRY & ERROR RECOVERY — RESILIENT AGENT PATTERNS")
    print("=" * 70)
    print("""
    Production agents MUST handle failures gracefully.
    This exercise covers every resilience pattern you need.

    Choose a demo:
      1 → Retry with exponential backoff
      2 → Fallback chain
      3 → Circuit breaker pattern
      4 → JSON repair strategies
      5 → Full resilience (all patterns combined)
      6 → Interactive resilient agent
      7 → Run demos 1-5
    """)

    choice = input("  Enter choice (1-7): ").strip()

    demos = {
        "1": demo_retry_with_backoff,
        "2": demo_fallback_chain,
        "3": demo_circuit_breaker,
        "4": demo_json_repair,
        "5": demo_full_resilience,
        "6": demo_interactive,
    }

    if choice == "7":
        for demo in [demo_retry_with_backoff, demo_fallback_chain,
                      demo_circuit_breaker, demo_json_repair, demo_full_resilience]:
            demo()
    elif choice in demos:
        demos[choice]()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. ERROR CLASSIFICATION IS CRITICAL
   - Transient (timeout, rate limit): RETRY — it will probably work next time
   - Recoverable (bad JSON): FIX — repair the output, then use it
   - Permanent (content filter, bad request): DON'T RETRY — switch strategy

2. EXPONENTIAL BACKOFF prevents hammering a struggling service.
   Wait 1s, 2s, 4s, 8s... not 1s, 1s, 1s, 1s.
   Add JITTER (random noise) so 1000 clients don't all retry at second 2.

3. CIRCUIT BREAKER is a state machine:
   CLOSED (normal) → OPEN (failing, reject fast) → HALF_OPEN (cautious test)
   This prevents wasting tokens and time on a dead service.

4. FALLBACK CHAIN: Always have a plan B (and C, and D).
   Primary model → Fallback model → Cached response → Safe default.
   Each level trades quality for availability. In healthcare,
   AVAILABILITY (some answer) often beats QUALITY (perfect answer that never arrives).

5. JSON REPAIR: Don't crash on malformed output.
   Try simple fixes first (strip markdown, fix commas, close brackets).
   Only ask the LLM to repair as a last resort (costs tokens).

6. GRACEFUL DEGRADATION > CRASHING
   A message saying "system unavailable, follow standard protocol" is
   infinitely better than a Python traceback. In healthcare, the safe
   default should ALWAYS recommend human clinical judgment.

7. MONITOR EVERYTHING: Log retries, fallbacks, circuit breaker state.
   If your agent is retrying 50% of the time, something is wrong.
   Metrics first, then fix.
"""

if __name__ == "__main__":
    main()
