# Level 8, Project 01: API Design

## Overview

Build production-ready APIs for AI/LLM services using **FastAPI**. Learn async
endpoint design, Pydantic request/response schemas, rate limiting, API-key
authentication, and robust error handling — all in a healthcare context.

## Why This Matters

Deploying an LLM behind a well-designed API is the standard path to production.
A good API layer gives you:

- **Validation** — reject bad input before it reaches the model
- **Rate limiting** — protect expensive GPU/token budgets
- **Authentication** — control who can call your service
- **Error handling** — return meaningful HTTP status codes
- **Observability** — log every request for auditing & debugging

## Prerequisites

```bash
pip install fastapi uvicorn openai pydantic httpx
```

You also need an `OPENAI_API_KEY` environment variable.

## Files

| File | Description |
|------|-------------|
| `main.py` | Four interactive demos (basic app, async, errors, full medical API) |
| `exercise_1_medical_chat_api.py` | Medical chat API with streaming SSE |
| `exercise_2_rag_api.py` | RAG document-ingestion and query API |
| `exercise_3_rate_limiting.py` | Per-key, per-tier rate limiting middleware |
| `exercise_4_api_testing.py` | Automated test suite with httpx |

## Running

```bash
# Run the demo menu
python main.py

# Start any FastAPI app directly
uvicorn main:app --reload --port 8000
```

## Key Concepts

1. **Pydantic Models** — typed request/response schemas with automatic validation
2. **Dependency Injection** — FastAPI `Depends()` for auth, rate limiting, DB
3. **Async/Await** — non-blocking calls so one slow LLM request doesn't block others
4. **HTTPException** — raise with the right status code (400, 401, 429, 500 …)
5. **Middleware** — cross-cutting concerns like logging, CORS, rate limiting

## Healthcare Considerations

- Always validate clinical input before sending to an LLM
- Log every request/response for compliance auditing
- Add disclaimers to AI-generated medical content
- Implement strict authentication — patient data must be protected

## Next Steps

After completing this project, move on to **02 Caching Optimization** to learn
how to reduce latency and cost with semantic caching and model routing.
