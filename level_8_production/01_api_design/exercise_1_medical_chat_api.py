"""
Exercise 1 — Medical Chat API
==============================
Build a medical chat API with FastAPI that includes:

    POST /chat              — single-turn medical Q&A
    POST /chat/stream       — server-sent events (SSE) streaming response
    GET  /models            — list available models

Features:
    • Pydantic request/response validation
    • System prompt with medical disclaimer
    • SSE streaming using OpenAI's stream=True
    • Proper error handling (400, 500)

Usage:
    uvicorn exercise_1_medical_chat_api:app --reload --port 8001

Then test with curl:
    curl -X POST http://localhost:8001/chat \
         -H "Content-Type: application/json" \
         -d '{"message": "What are symptoms of pneumonia?"}'
"""

import os
import time
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.responses import StreamingResponse, JSONResponse
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
    print("⚠️  Install openai: pip install openai")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
if FASTAPI_AVAILABLE:

    class ChatRequest(BaseModel):
        """Request body for /chat and /chat/stream."""
        message: str = Field(
            ...,
            min_length=1,
            max_length=4000,
            description="The user's medical question",
        )
        model: str = Field("gpt-4o-mini", description="LLM model to use")
        max_tokens: int = Field(512, ge=1, le=4096, description="Max tokens in response")
        temperature: float = Field(0.3, ge=0.0, le=2.0, description="Sampling temperature")

    class ChatResponse(BaseModel):
        """Response body for /chat."""
        reply: str
        model: str
        tokens_used: int
        latency_ms: float
        disclaimer: str = (
            "This response is AI-generated and does not constitute medical advice. "
            "Always consult a qualified healthcare professional."
        )
        timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    class ModelInfo(BaseModel):
        """Info about an available model."""
        id: str
        description: str
        max_context: int

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    SYSTEM_PROMPT = (
        "You are a knowledgeable medical assistant. Provide accurate, evidence-based "
        "information. Always remind the user that your answers are for informational "
        "purposes only and should not replace professional medical advice."
    )

    AVAILABLE_MODELS = [
        ModelInfo(id="gpt-4o-mini", description="Fast and affordable for simple queries", max_context=128_000),
        ModelInfo(id="gpt-4o", description="Most capable model for complex medical reasoning", max_context=128_000),
        ModelInfo(id="o1-mini", description="Reasoning model for multi-step clinical analysis", max_context=128_000),
    ]

    def _get_client() -> OpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server misconfigured: OPENAI_API_KEY not set",
            )
        return OpenAI(api_key=api_key)

    # -----------------------------------------------------------------------
    # App
    # -----------------------------------------------------------------------
    app = FastAPI(
        title="Medical Chat API",
        description="Healthcare-focused chat API with streaming support",
        version="1.0.0",
    )

    @app.get("/models", response_model=list[ModelInfo])
    def list_models():
        """Return the list of models the API supports."""
        return AVAILABLE_MODELS

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        """Synchronous single-turn medical chat completion."""
        client = _get_client()
        start = time.perf_counter()

        try:
            resp = client.chat.completions.create(
                model=req.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": req.message},
                ],
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"LLM call failed: {exc}",
            )

        latency = (time.perf_counter() - start) * 1000
        return ChatResponse(
            reply=resp.choices[0].message.content,
            model=req.model,
            tokens_used=resp.usage.total_tokens,
            latency_ms=round(latency, 1),
        )

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest):
        """Stream the LLM response using Server-Sent Events (SSE).

        Returns a text/event-stream response where each data line is a
        JSON object with a ``token`` field.
        """
        client = _get_client()

        def _generate():
            try:
                stream = client.chat.completions.create(
                    model=req.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": req.message},
                    ],
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        # SSE format: data: <json>\n\n
                        yield f"data: {{'token': '{delta.content}'}}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                yield f"data: {{'error': '{exc}'}}\n\n"

        return StreamingResponse(_generate(), media_type="text/event-stream")

    @app.get("/health")
    def health():
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ---------------------------------------------------------------------------
# Standalone test (without a running server)
# ---------------------------------------------------------------------------
def _local_test():
    """Quick test of the core logic without starting uvicorn."""
    print("=" * 60)
    print("Exercise 1 — Medical Chat API (local test)")
    print("=" * 60)

    if not OPENAI_AVAILABLE:
        print("OpenAI SDK not available — skipping live test.")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY to run the live test.")
        return

    client = OpenAI(api_key=api_key)
    questions = [
        "What are common symptoms of pneumonia?",
        "Explain the difference between Type 1 and Type 2 diabetes.",
    ]

    for q in questions:
        print(f"\n📨 Question: {q}")
        start = time.perf_counter()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT if FASTAPI_AVAILABLE else "You are a medical assistant."},
                {"role": "user", "content": q},
            ],
            max_tokens=256,
        )
        elapsed = (time.perf_counter() - start) * 1000
        print(f"💬 Reply ({elapsed:.0f} ms): {resp.choices[0].message.content[:300]}")

    print("\n✅ Local test complete. Run with uvicorn for the full API experience.")


if __name__ == "__main__":
    _local_test()
