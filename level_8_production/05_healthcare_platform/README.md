# Project 05: Healthcare Platform Capstone

## Overview
This capstone project combines everything from Levels 1-8 into a complete,
production-ready healthcare AI platform. It integrates FastAPI, RAG,
Agents, MCP, Caching, Monitoring, and Security into a unified system.

## What This Platform Includes

### Components (Levels 1-8)
| Component | Source Level | Role |
|-----------|-------------|------|
| LLM Chat | Level 1 | Core medical Q&A |
| Embeddings/RAG | Level 2 | Document-backed answers |
| Agent Workflows | Level 3 | Multi-step clinical reasoning |
| Evaluation | Level 4 | Quality & safety validation |
| MCP Integration | Level 5 | Standardized tool protocol |
| Multimodal | Level 6 | Structured output & reasoning |
| Fine-tuning | Level 7 | Domain-specific models |
| Production Infra | Level 8 | API, caching, monitoring, security |

### Architecture
```
┌─────────────────────────────────────────┐
│            FastAPI Gateway              │
│  /chat  /rag  /agent  /admin  /health   │
├─────────┬──────────┬────────────────────┤
│  Auth   │  Cache   │  Rate Limiter      │
├─────────┴──────────┴────────────────────┤
│            Service Layer                │
│  Medical Q&A │ RAG Engine │ Agent Svc   │
├─────────────────────────────────────────┤
│          Infrastructure                 │
│  Monitoring │ Security │ Audit Trail    │
└─────────────────────────────────────────┘
```

## Demos in main.py
1. **Platform Architecture** — Display full system architecture and config
2. **Medical Q&A Service** — Cached, guarded medical Q&A with monitoring
3. **Clinical Agent Service** — Agent with tools, audit trail, cost tracking
4. **Platform Dashboard** — Live text dashboard with metrics

## Exercises
1. `exercise_1_platform_setup.py` — FastAPI scaffold with all endpoints
2. `exercise_2_integrated_rag.py` — RAG with monitoring and caching
3. `exercise_3_agent_service.py` — Clinical agent with audit and guardrails
4. `exercise_4_platform_test.py` — End-to-end test suite and health report

## Key Features
- **End-to-end monitoring**: Every API call logged with cost/latency tracking
- **Security pipeline**: Input sanitization + output filtering on all endpoints
- **Caching layer**: Semantic deduplication to reduce costs and latency
- **Audit trail**: Full record of every AI-assisted decision for compliance
- **Health checks**: Service health monitoring with automatic alerts
- **Cost control**: Per-user/department budget tracking and enforcement

## Running
```bash
python main.py                        # Interactive demo menu
python exercise_1_platform_setup.py   # Platform scaffold
python exercise_2_integrated_rag.py   # Integrated RAG service
python exercise_3_agent_service.py    # Agent service
python exercise_4_platform_test.py    # End-to-end tests
```

## Prerequisites
- OpenAI API key set as `OPENAI_API_KEY`
- Python 3.9+
- `pip install fastapi uvicorn` (for exercises 1 and 4)
