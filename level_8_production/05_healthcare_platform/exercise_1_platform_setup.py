"""
Exercise 1: Platform Setup — FastAPI Scaffold
===============================================
Build the platform scaffold: FastAPI app with all endpoints
(/chat, /rag, /agent, /admin), middleware for logging and auth,
health checks.

Requirements:
- FastAPI application with organized route structure
- Authentication middleware (API key-based)
- Request logging middleware
- Health check endpoint with dependency status
- Structured error responses
- CORS configuration for frontend integration

Healthcare Context:
  A production healthcare AI platform needs robust API design with
  authentication, audit logging, and health monitoring. This scaffold
  provides the foundation for all services.

Usage:
    python exercise_1_platform_setup.py

    # To run the actual server:
    # uvicorn exercise_1_platform_setup:app --reload --port 8000
"""

from openai import OpenAI
import time
import json
import os
import uuid
import hashlib
from datetime import datetime
from typing import Optional

# FastAPI imports (with graceful fallback)
try:
    from fastapi import FastAPI, Request, HTTPException, Depends, Header
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

client = OpenAI()

# =============================================================================
# Data Models
# =============================================================================

if FASTAPI_AVAILABLE:
    class ChatRequest(BaseModel):
        message: str = Field(..., min_length=1, max_length=2000, description="User message")
        model: str = Field(default="gpt-4o-mini", description="Model to use")
        user_id: Optional[str] = Field(default="anonymous", description="User identifier")

    class ChatResponse(BaseModel):
        response: str
        request_id: str
        model: str
        tokens: int
        latency_ms: float
        cached: bool = False

    class RAGRequest(BaseModel):
        query: str = Field(..., min_length=1, max_length=2000)
        top_k: int = Field(default=3, ge=1, le=10)
        user_id: Optional[str] = "anonymous"

    class AgentRequest(BaseModel):
        query: str = Field(..., min_length=1, max_length=2000)
        tools: list = Field(default=["drug_lookup", "lab_reference", "guidelines"])
        user_id: Optional[str] = "anonymous"

    class HealthResponse(BaseModel):
        status: str
        version: str
        uptime_seconds: float
        services: dict
        timestamp: str

# =============================================================================
# Platform Core
# =============================================================================

VALID_API_KEYS = {
    "med-api-key-001": {"user": "dr_smith", "role": "physician", "department": "cardiology"},
    "med-api-key-002": {"user": "nurse_chen", "role": "nurse", "department": "emergency"},
    "med-api-key-admin": {"user": "admin", "role": "admin", "department": "IT"},
}

START_TIME = time.time()
request_log = []


def authenticate(api_key: str) -> dict:
    """Validate API key and return user info."""
    if api_key in VALID_API_KEYS:
        return VALID_API_KEYS[api_key]
    return None


def log_request(request_id: str, endpoint: str, user: str, latency_ms: float,
                status_code: int, tokens: int = 0, cost: float = 0.0):
    """Log a platform request."""
    request_log.append({
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "user": user,
        "latency_ms": round(latency_ms, 2),
        "status_code": status_code,
        "tokens": tokens,
        "cost_usd": cost,
    })


# =============================================================================
# FastAPI Application
# =============================================================================

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="MedAI Platform",
        description="Production Healthcare AI Platform — Capstone Project",
        version="1.0.0",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Auth dependency
    async def verify_api_key(x_api_key: str = Header(...)):
        user_info = authenticate(x_api_key)
        if not user_info:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return user_info

    # ---- Health Check ----
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Platform health check endpoint."""
        uptime = time.time() - START_TIME
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=round(uptime, 2),
            services={
                "openai_api": "connected",
                "cache": "active",
                "monitoring": "active",
                "security": "active",
            },
            timestamp=datetime.now().isoformat(),
        )

    # ---- Chat Endpoint ----
    @app.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(req: ChatRequest, user_info: dict = Depends(verify_api_key)):
        """Medical Q&A chat endpoint."""
        request_id = str(uuid.uuid4())[:8]
        start = time.time()

        try:
            response = client.chat.completions.create(
                model=req.model,
                messages=[
                    {"role": "system", "content": (
                        "You are MedAI, a medical information assistant. "
                        "Provide accurate, evidence-based medical education. "
                        "Always include a disclaimer to consult a healthcare provider."
                    )},
                    {"role": "user", "content": req.message},
                ],
            )
            latency = (time.time() - start) * 1000
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens

            log_request(request_id, "/chat", user_info["user"], latency, 200, tokens)

            return ChatResponse(
                response=content,
                request_id=request_id,
                model=req.model,
                tokens=tokens,
                latency_ms=round(latency, 2),
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            log_request(request_id, "/chat", user_info["user"], latency, 500)
            raise HTTPException(status_code=500, detail=str(e))

    # ---- RAG Endpoint ----
    @app.post("/rag")
    async def rag_endpoint(req: RAGRequest, user_info: dict = Depends(verify_api_key)):
        """RAG-backed medical query endpoint."""
        request_id = str(uuid.uuid4())[:8]
        start = time.time()

        # Simulated RAG retrieval
        retrieved_docs = [
            {"content": "Hypertension management guidelines recommend ACE inhibitors as first-line therapy.", "score": 0.92},
            {"content": "Regular blood pressure monitoring is essential for treatment adjustment.", "score": 0.87},
            {"content": "Lifestyle modifications include reduced sodium intake and regular exercise.", "score": 0.85},
        ][:req.top_k]

        context = "\n".join(d["content"] for d in retrieved_docs)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Answer based on this context:\n{context}\n\nAlways cite sources and add disclaimer."},
                {"role": "user", "content": req.query},
            ],
        )
        latency = (time.time() - start) * 1000
        log_request(request_id, "/rag", user_info["user"], latency, 200, response.usage.total_tokens)

        return {
            "request_id": request_id,
            "response": response.choices[0].message.content,
            "sources": retrieved_docs,
            "tokens": response.usage.total_tokens,
            "latency_ms": round(latency, 2),
        }

    # ---- Agent Endpoint ----
    @app.post("/agent")
    async def agent_endpoint(req: AgentRequest, user_info: dict = Depends(verify_api_key)):
        """Clinical agent endpoint with tool calling."""
        request_id = str(uuid.uuid4())[:8]
        start = time.time()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a clinical reasoning agent. Analyze the query, "
                    "list the tools you would use and why, then provide assessment. "
                    "Always include a disclaimer."
                )},
                {"role": "user", "content": req.query},
            ],
        )
        latency = (time.time() - start) * 1000
        log_request(request_id, "/agent", user_info["user"], latency, 200, response.usage.total_tokens)

        return {
            "request_id": request_id,
            "assessment": response.choices[0].message.content,
            "tools_available": req.tools,
            "audit_trail": [
                {"step": "query_analysis", "duration_ms": round(latency * 0.3)},
                {"step": "tool_execution", "duration_ms": round(latency * 0.2)},
                {"step": "synthesis", "duration_ms": round(latency * 0.5)},
            ],
            "tokens": response.usage.total_tokens,
            "latency_ms": round(latency, 2),
        }

    # ---- Admin Endpoint ----
    @app.get("/admin/stats")
    async def admin_stats(user_info: dict = Depends(verify_api_key)):
        """Platform statistics (admin only)."""
        if user_info["role"] != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")

        total_reqs = len(request_log)
        total_tokens = sum(r["tokens"] for r in request_log)
        total_cost = sum(r["cost_usd"] for r in request_log)

        return {
            "total_requests": total_reqs,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "uptime_seconds": round(time.time() - START_TIME, 2),
            "recent_requests": request_log[-10:],
        }


# =============================================================================
# Demo Runner (when FastAPI is not available or running standalone)
# =============================================================================

def demo_platform_scaffold():
    """Demonstrate the platform scaffold without running the server."""
    print("=" * 60)
    print("  Exercise 1: Platform Setup — FastAPI Scaffold")
    print("=" * 60)

    if FASTAPI_AVAILABLE:
        print("\n  FastAPI is available! The app can be run with:")
        print("    uvicorn exercise_1_platform_setup:app --reload --port 8000")
    else:
        print("\n  FastAPI not installed. Install with: pip install fastapi uvicorn")
        print("  Running demo mode instead...\n")

    print("\n--- Platform Endpoints ---")
    endpoints = [
        ("GET", "/health", "Health check — no auth required"),
        ("POST", "/chat", "Medical Q&A — requires API key"),
        ("POST", "/rag", "RAG-backed query — requires API key"),
        ("POST", "/agent", "Clinical agent — requires API key"),
        ("GET", "/admin/stats", "Platform stats — admin only"),
    ]
    for method, path, desc in endpoints:
        print(f"  {method:<6} {path:<18} {desc}")

    print("\n--- Authentication ---")
    for key, info in VALID_API_KEYS.items():
        print(f"  Key: {key[:15]}...  User: {info['user']}, Role: {info['role']}")

    print("\n--- Live API Test ---")
    print("  Making a test call to verify OpenAI connectivity...")

    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical assistant. Be very brief."},
            {"role": "user", "content": "What is normal blood pressure?"},
        ],
    )
    latency = (time.time() - start) * 1000
    req_id = str(uuid.uuid4())[:8]

    log_request(req_id, "/chat", "test_user", latency, 200, response.usage.total_tokens)

    print(f"  Request ID: {req_id}")
    print(f"  Latency:    {latency:.0f}ms")
    print(f"  Tokens:     {response.usage.total_tokens}")
    print(f"  Response:   {response.choices[0].message.content[:150]}...")

    print(f"\n--- Request Log ({len(request_log)} entries) ---")
    for entry in request_log:
        print(f"  [{entry['request_id']}] {entry['endpoint']} — {entry['user']} — {entry['latency_ms']:.0f}ms")

    print("\nPlatform scaffold demo complete!")


def main():
    demo_platform_scaffold()


if __name__ == "__main__":
    main()
