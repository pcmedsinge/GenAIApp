"""
Level 8, Project 01: API Design — Main Demos
=============================================
Four progressive demos showing how to build production-ready FastAPI services
for healthcare AI workloads.

Demos
-----
1. Basic FastAPI App        — /chat endpoint with Pydantic models
2. Async Endpoints          — non-blocking concurrent LLM calls
3. Error Handling           — graceful failures with proper HTTP codes
4. Complete Medical API     — full multi-endpoint authenticated service

Usage
-----
    python main.py            # interactive menu
    uvicorn main:app --port 8000   # run the FastAPI server directly
"""

import os
import json
import time
import asyncio
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Third-party imports (with graceful fallback)
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, HTTPException, Depends, Request, Header, status
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, validator
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️  FastAPI not installed. Run: pip install fastapi uvicorn pydantic")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI SDK not installed. Run: pip install openai")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def get_openai_client() -> "OpenAI":
    """Return an OpenAI client or raise if the key is missing."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY environment variable first.")
    return OpenAI(api_key=api_key)


def quick_chat(client: "OpenAI", message: str, model: str = "gpt-4o-mini") -> str:
    """Blocking convenience wrapper around chat completions."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant. Always include a disclaimer that you are an AI."},
            {"role": "user", "content": message},
        ],
        max_tokens=512,
    )
    return response.choices[0].message.content


# ============================================================
# DEMO 1: Basic FastAPI App
# ============================================================
def demo_basic_fastapi_app():
    """Build a minimal FastAPI app with a /chat endpoint and Pydantic models."""
    print("\n🔹 DEMO 1: Basic FastAPI App")
    print("=" * 60)

    if not FASTAPI_AVAILABLE:
        print("FastAPI is required for this demo. Install it and retry.")
        return

    code = '''
# --- Pydantic request / response models ---
from pydantic import BaseModel, Field
from fastapi import FastAPI
from openai import OpenAI
import os, uvicorn

class ChatRequest(BaseModel):
    """Typed request body for the /chat endpoint."""
    message: str = Field(..., min_length=1, max_length=2000,
                         description="User message to send to the LLM")
    model: str = Field("gpt-4o-mini", description="Model to use")
    max_tokens: int = Field(512, ge=1, le=4096)

class ChatResponse(BaseModel):
    """Typed response body."""
    reply: str
    model: str
    tokens_used: int
    disclaimer: str = "AI-generated content — not a substitute for professional medical advice."

app = FastAPI(title="Medical Chat API", version="0.1.0")

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=req.model,
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": req.message},
        ],
        max_tokens=req.max_tokens,
    )
    return ChatResponse(
        reply=resp.choices[0].message.content,
        model=req.model,
        tokens_used=resp.usage.total_tokens,
    )

# Run with: uvicorn filename:app --reload --port 8000
'''
    print(code)
    print("To start this server:")
    print("  uvicorn main:app --reload --port 8000")
    print("Then POST to http://localhost:8000/chat with JSON body:")
    print('  {"message": "What are the symptoms of diabetes?"}')

    # Quick local test (no server required)
    if OPENAI_AVAILABLE:
        try:
            client = get_openai_client()
            reply = quick_chat(client, "List 3 symptoms of hypertension in one sentence.")
            print(f"\n📨 Quick local test reply:\n{reply}")
        except Exception as exc:
            print(f"⚠️  OpenAI call failed: {exc}")


# ============================================================
# DEMO 2: Async Endpoints
# ============================================================
def demo_async_endpoints():
    """Demonstrate async/await for non-blocking LLM calls."""
    print("\n🔹 DEMO 2: Async Endpoints")
    print("=" * 60)

    code = '''
from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import BaseModel
import os, asyncio

app = FastAPI()
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    latency_ms: float

@app.post("/chat/async", response_model=ChatResponse)
async def async_chat(req: ChatRequest):
    """Non-blocking endpoint — other requests are served while awaiting LLM."""
    import time
    start = time.perf_counter()
    resp = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": req.message}],
        max_tokens=256,
    )
    elapsed = (time.perf_counter() - start) * 1000
    return ChatResponse(reply=resp.choices[0].message.content, latency_ms=round(elapsed, 1))

@app.post("/chat/batch")
async def batch_chat(messages: list[str]):
    """Fire multiple LLM calls concurrently with asyncio.gather."""
    tasks = [
        aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": m}],
            max_tokens=128,
        )
        for m in messages
    ]
    results = await asyncio.gather(*tasks)
    return [r.choices[0].message.content for r in results]
'''
    print(code)

    # Show a local async demo
    if OPENAI_AVAILABLE:
        try:
            from openai import AsyncOpenAI as _AO

            async def _run():
                aclient = _AO(api_key=os.getenv("OPENAI_API_KEY"))
                queries = [
                    "Define tachycardia in one sentence.",
                    "Define bradycardia in one sentence.",
                    "Define arrhythmia in one sentence.",
                ]
                print("\n⏳ Sending 3 concurrent async requests …")
                start = time.perf_counter()
                tasks = [
                    aclient.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": q}],
                        max_tokens=80,
                    )
                    for q in queries
                ]
                results = await asyncio.gather(*tasks)
                elapsed = time.perf_counter() - start
                for q, r in zip(queries, results):
                    print(f"  Q: {q}")
                    print(f"  A: {r.choices[0].message.content[:120]}\n")
                print(f"⏱️  All 3 completed in {elapsed:.2f}s (concurrent)")

            asyncio.run(_run())
        except Exception as exc:
            print(f"⚠️  Async demo error: {exc}")


# ============================================================
# DEMO 3: Error Handling
# ============================================================
def demo_error_handling():
    """Handle API errors gracefully with proper HTTP status codes."""
    print("\n🔹 DEMO 3: Error Handling")
    print("=" * 60)

    code = '''
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from openai import OpenAI, RateLimitError, APITimeoutError
from pydantic import BaseModel, Field
import os

app = FastAPI()

# --- Custom exception handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler so internal errors never leak to the client."""
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "An unexpected error occurred."},
    )

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": req.message}],
            max_tokens=256,
            timeout=30,
        )
        return {"reply": resp.choices[0].message.content}

    except RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="LLM request timed out.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(exc)}")
'''
    print(code)

    # Demonstrate local error scenarios
    print("\n--- Simulated error scenarios ---")
    error_cases = [
        ("Empty message", ""),
        ("Message too long", "x" * 3000),
        ("Invalid model", "nonexistent-model-xyz"),
    ]
    if OPENAI_AVAILABLE:
        client = get_openai_client()
        for label, msg in error_cases:
            try:
                if not msg:
                    raise ValueError("Message must be non-empty")
                if len(msg) > 2000:
                    raise ValueError("Message exceeds 2000 characters")
                quick_chat(client, msg, model="nonexistent-model-xyz" if label == "Invalid model" else "gpt-4o-mini")
                print(f"  ✅ {label}: succeeded (unexpected)")
            except Exception as exc:
                print(f"  🛑 {label}: caught → {type(exc).__name__}: {str(exc)[:100]}")


# ============================================================
# DEMO 4: Complete Medical API
# ============================================================
def demo_complete_medical_api():
    """Full multi-endpoint medical API with auth, rate limiting, and validation."""
    print("\n🔹 DEMO 4: Complete Medical API")
    print("=" * 60)

    code = '''
from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from datetime import datetime
import os, time, json

app = FastAPI(title="Healthcare AI Platform", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------- Auth ----------
VALID_API_KEYS = {"key-abc123": "Dr. Smith", "key-xyz789": "Nurse Jones"}

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return VALID_API_KEYS[x_api_key]

# ---------- Rate limiting (in-memory) ----------
request_log: dict[str, list[float]] = {}
RATE_LIMIT = 20  # requests per minute

async def check_rate_limit(x_api_key: str = Header(...)):
    now = time.time()
    window = [t for t in request_log.get(x_api_key, []) if now - t < 60]
    if len(window) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    window.append(now)
    request_log[x_api_key] = window

# ---------- Models ----------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    model: str = "gpt-4o-mini"

class ChatResponse(BaseModel):
    reply: str
    model: str
    timestamp: str

class RAGQuery(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)

class AssessRequest(BaseModel):
    symptoms: list[str]
    patient_age: int = Field(..., ge=0, le=150)
    patient_sex: str

# ---------- Endpoints ----------
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
def chat_endpoint(req: ChatRequest):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=req.model,
        messages=[{"role": "system", "content": "You are a medical assistant."}, {"role": "user", "content": req.message}],
        max_tokens=512,
    )
    return ChatResponse(reply=resp.choices[0].message.content, model=req.model, timestamp=datetime.utcnow().isoformat())

@app.post("/rag/query", dependencies=[Depends(verify_api_key)])
def rag_query(req: RAGQuery):
    # Placeholder — integrate with a real vector DB in production
    return {"query": req.query, "results": [], "message": "RAG endpoint stub — connect your vector store here."}

@app.post("/agent/assess", dependencies=[Depends(verify_api_key)])
def agent_assess(req: AssessRequest):
    prompt = f"Patient: {req.patient_age}y {req.patient_sex}. Symptoms: {', '.join(req.symptoms)}. Provide a differential diagnosis."
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], max_tokens=512)
    return {"assessment": resp.choices[0].message.content, "disclaimer": "AI-generated — not a substitute for clinical judgment."}

@app.post("/embeddings", dependencies=[Depends(verify_api_key)])
def create_embeddings(texts: list[str]):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return {"embeddings": [e.embedding[:5] for e in resp.data], "dimensions": len(resp.data[0].embedding), "note": "Truncated to first 5 dims for display"}

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
'''
    print(code)

    print("\nEndpoints summary:")
    endpoints = [
        ("POST", "/chat", "General medical chat"),
        ("POST", "/rag/query", "RAG document query"),
        ("POST", "/agent/assess", "Symptom assessment agent"),
        ("POST", "/embeddings", "Generate text embeddings"),
        ("GET", "/health", "Health check"),
    ]
    for method, path, desc in endpoints:
        print(f"  {method:6s} {path:20s} — {desc}")

    if OPENAI_AVAILABLE:
        try:
            client = get_openai_client()
            reply = quick_chat(client, "In one sentence, what is the purpose of a medical API gateway?")
            print(f"\n📨 Sample reply:\n{reply}")
        except Exception as exc:
            print(f"⚠️  OpenAI call failed: {exc}")


# ============================================================
# Main menu
# ============================================================
def main():
    """Interactive demo selector."""
    demos = {
        "1": ("Basic FastAPI App", demo_basic_fastapi_app),
        "2": ("Async Endpoints", demo_async_endpoints),
        "3": ("Error Handling", demo_error_handling),
        "4": ("Complete Medical API", demo_complete_medical_api),
    }

    while True:
        print("\n" + "=" * 60)
        print("LEVEL 8 · PROJECT 01 — API DESIGN")
        print("=" * 60)
        for key, (title, _) in demos.items():
            print(f"  {key}. {title}")
        print("  q. Quit")

        choice = input("\nSelect demo: ").strip().lower()
        if choice == "q":
            print("Goodbye!")
            break
        if choice in demos:
            demos[choice][1]()
        else:
            print("Invalid choice — try again.")


# ---------------------------------------------------------------------------
# FastAPI app instance (used when launched via uvicorn main:app)
# ---------------------------------------------------------------------------
if FASTAPI_AVAILABLE:
    app = FastAPI(title="Medical Chat API", version="1.0.0")

    class ChatRequest(BaseModel):
        message: str = Field(..., min_length=1, max_length=2000)
        model: str = Field("gpt-4o-mini")
        max_tokens: int = Field(512, ge=1, le=4096)

    class ChatResponse(BaseModel):
        reply: str
        model: str
        tokens_used: int
        disclaimer: str = "AI-generated — not a substitute for professional medical advice."

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=req.model,
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": req.message},
            ],
            max_tokens=req.max_tokens,
        )
        return ChatResponse(
            reply=resp.choices[0].message.content,
            model=req.model,
            tokens_used=resp.usage.total_tokens,
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    main()
