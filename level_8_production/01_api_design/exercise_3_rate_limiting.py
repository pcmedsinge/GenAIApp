"""
Exercise 3 — Rate Limiting
===========================
Implement tiered, per-API-key rate limiting for a FastAPI service:

    • Track requests per key per minute using an in-memory sliding window
    • Different rate limits for different subscription tiers (free / pro / enterprise)
    • Return HTTP 429 (Too Many Requests) when the limit is exceeded
    • Include Retry-After header and remaining-request info in responses

Usage:
    uvicorn exercise_3_rate_limiting:app --reload --port 8003

Test:
    # Free tier (5 req/min)
    for i in $(seq 1 7); do
        curl -s -o /dev/null -w "%{http_code}" \
             -X POST http://localhost:8003/chat \
             -H 'Content-Type: application/json' \
             -H 'X-API-Key: free-key-001' \
             -d '{"message": "hello"}'
        echo
    done
"""

import os
import time
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, HTTPException, Request, Depends, Header, status
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️  Install dependencies: pip install fastapi uvicorn pydantic")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tier configuration
# ---------------------------------------------------------------------------
TIERS = {
    "free":       {"rpm": 5,   "daily": 50,   "max_tokens": 256},
    "pro":        {"rpm": 30,  "daily": 1000,  "max_tokens": 1024},
    "enterprise": {"rpm": 120, "daily": 10000, "max_tokens": 4096},
}

# API key → tier mapping (in production use a database)
API_KEYS = {
    "free-key-001":   {"tier": "free",       "owner": "Demo Free User"},
    "pro-key-001":    {"tier": "pro",        "owner": "Dr. Smith"},
    "ent-key-001":    {"tier": "enterprise", "owner": "General Hospital"},
}


# ---------------------------------------------------------------------------
# Rate limiter (sliding window, in-memory)
# ---------------------------------------------------------------------------
class RateLimiter:
    """Per-key sliding-window rate limiter."""

    def __init__(self):
        # key → list of request timestamps
        self._minute_log: dict[str, list[float]] = {}
        self._daily_log: dict[str, list[float]] = {}

    def check(self, api_key: str, tier: str) -> dict:
        """Check limits. Returns usage info or raises HTTPException."""
        now = time.time()
        limits = TIERS[tier]

        # --- Per-minute window ---
        minute_ts = self._minute_log.setdefault(api_key, [])
        minute_ts[:] = [t for t in minute_ts if now - t < 60]
        rpm_remaining = limits["rpm"] - len(minute_ts)

        if rpm_remaining <= 0:
            oldest = min(minute_ts)
            retry_after = int(60 - (now - oldest)) + 1
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "tier": tier,
                    "limit": f"{limits['rpm']} requests/min",
                    "retry_after_seconds": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        # --- Per-day window ---
        daily_ts = self._daily_log.setdefault(api_key, [])
        daily_ts[:] = [t for t in daily_ts if now - t < 86400]
        daily_remaining = limits["daily"] - len(daily_ts)

        if daily_remaining <= 0:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "daily_limit_exceeded",
                    "tier": tier,
                    "limit": f"{limits['daily']} requests/day",
                    "message": "Upgrade your tier or wait until tomorrow.",
                },
            )

        # Record this request
        minute_ts.append(now)
        daily_ts.append(now)

        return {
            "rpm_remaining": rpm_remaining - 1,
            "daily_remaining": daily_remaining - 1,
            "tier": tier,
            "max_tokens": limits["max_tokens"],
        }


rate_limiter = RateLimiter()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
if FASTAPI_AVAILABLE:

    app = FastAPI(title="Rate-Limited Medical API", version="1.0.0")

    # --- Dependencies ---
    async def authenticate(x_api_key: str = Header(...)):
        """Validate API key and return key metadata."""
        if x_api_key not in API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return {**API_KEYS[x_api_key], "key": x_api_key}

    async def apply_rate_limit(auth: dict = Depends(authenticate)):
        """Check rate limit for the authenticated key."""
        usage = rate_limiter.check(auth["key"], auth["tier"])
        return {**auth, **usage}

    # --- Models ---
    class ChatRequest(BaseModel):
        message: str = Field(..., min_length=1, max_length=4000)
        model: str = "gpt-4o-mini"

    class ChatResponse(BaseModel):
        reply: str
        model: str
        tier: str
        rpm_remaining: int
        daily_remaining: int

    # --- Endpoints ---
    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, ctx: dict = Depends(apply_rate_limit)):
        """Rate-limited chat endpoint."""
        if not OPENAI_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
            # Return a mock response for testing rate limiting without OpenAI
            return ChatResponse(
                reply=f"[Mock] Received: {req.message[:80]}",
                model=req.model,
                tier=ctx["tier"],
                rpm_remaining=ctx["rpm_remaining"],
                daily_remaining=ctx["daily_remaining"],
            )

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=req.model,
            messages=[
                {"role": "system", "content": "You are a medical assistant."},
                {"role": "user", "content": req.message},
            ],
            max_tokens=ctx["max_tokens"],
        )
        return ChatResponse(
            reply=resp.choices[0].message.content,
            model=req.model,
            tier=ctx["tier"],
            rpm_remaining=ctx["rpm_remaining"],
            daily_remaining=ctx["daily_remaining"],
        )

    @app.get("/limits")
    def get_limits(ctx: dict = Depends(apply_rate_limit)):
        """Show the caller's current rate-limit status."""
        return {
            "owner": ctx["owner"],
            "tier": ctx["tier"],
            "rpm_remaining": ctx["rpm_remaining"],
            "daily_remaining": ctx["daily_remaining"],
            "max_tokens_per_request": ctx["max_tokens"],
        }

    @app.get("/tiers")
    def list_tiers():
        """Public endpoint — show available tiers and their limits."""
        return TIERS

    @app.get("/health")
    def health():
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
def _local_test():
    print("=" * 60)
    print("Exercise 3 — Rate Limiting (local test)")
    print("=" * 60)

    print("\nTier configuration:")
    for tier, cfg in TIERS.items():
        print(f"  {tier:12s}  {cfg['rpm']:>4d} req/min  {cfg['daily']:>6d} req/day  max_tokens={cfg['max_tokens']}")

    print("\nSimulating 8 requests on the FREE tier (limit = 5/min):")
    test_key = "free-key-001"
    tier = API_KEYS[test_key]["tier"]

    # Reset limiter for clean test
    limiter = RateLimiter()

    for i in range(1, 9):
        try:
            usage = limiter.check(test_key, tier)
            print(f"  Request {i}: ✅ OK  (rpm_remaining={usage['rpm_remaining']})")
        except HTTPException as exc:
            detail = exc.detail
            print(f"  Request {i}: 🛑 429 — {detail}")

    print("\nSimulating requests on the PRO tier (limit = 30/min):")
    test_key_pro = "pro-key-001"
    tier_pro = API_KEYS[test_key_pro]["tier"]
    limiter2 = RateLimiter()
    for i in range(1, 35):
        try:
            usage = limiter2.check(test_key_pro, tier_pro)
            if i % 10 == 0 or i > 28:
                print(f"  Request {i:2d}: ✅ OK  (rpm_remaining={usage['rpm_remaining']})")
        except HTTPException as exc:
            print(f"  Request {i:2d}: 🛑 429 — rate limit exceeded")

    print("\n✅ Rate limiting test complete.")
    print("Run with uvicorn to test the full HTTP API.")


if __name__ == "__main__":
    _local_test()
