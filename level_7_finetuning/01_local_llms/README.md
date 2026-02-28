# Level 7, Project 01: Local LLMs with Ollama# Level 7.1: Local LLMs with Ollama


































































```client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")from openai import OpenAI```pythonby simply swapping the `base_url` and `api_key`:OpenAI Chat Completions API. You can reuse **all** your existing OpenAI codeOllama exposes an endpoint at `http://localhost:11434/v1` that mirrors the## Key Concept — OpenAI-Compatible API```python exercise_1_model_benchmarks.py# 4. Try an exercisepython main.py# 3. Run the demosollama pull nomic-embed-textollama pull phi3ollama pull mistralollama pull llama3# 2. Pull the models used in demosollama serve          # in a separate terminal, if not already running# 1. Make sure Ollama is running```bash## Quick Start| 4 | exercise_4_local_rag.py | Full local RAG: Ollama + ChromaDB, zero data leakage || 3 | exercise_3_quantization_comparison.py | Q4 vs Q8 quantization trade-offs || 2 | exercise_2_local_medical_qa.py | Local medical Q&A vs GPT-4o comparison || 1 | exercise_1_model_benchmarks.py | Benchmarking speed, tokens, quality across models ||---|---|---|| # | File | Skills |## Exercises| 4 | Local RAG System | End-to-end RAG — embeddings + generation, zero cloud calls || 3 | Local Embeddings | Generate embeddings with nomic-embed-text || 2 | Model Comparison | Run the same prompt through llama3, mistral, phi3 || 1 | Ollama Basics | OpenAI-compatible API with `base_url` pointed at Ollama ||---|---|---|| # | Demo | What you learn |## Demos (main.py)Verify Ollama is running: `ollama list` should show your pulled models.| ChromaDB (exercise 4) | `pip install chromadb` || Python OpenAI SDK | `pip install openai` || A model pulled | `ollama pull llama3` || Ollama | `curl -fsSL https://ollama.com/install.sh \| sh` ||---|---|| Requirement | Install command |## Prerequisites- **Cost control**: No per-token API charges after hardware investment.- **Offline capability**: Works in air-gapped clinical environments.- **Regulatory compliance**: Simplifies HIPAA / GDPR obligations.- **Data privacy**: PHI never touches a third-party server.## Why Local LLMs Matter in Healthcaremanagement, local embeddings, and building a fully local RAG pipeline.covers Ollama installation, the OpenAI-compatible API it exposes, modelRun large language models **locally** — no data leaves your machine. This project## Overview
## Overview
Run large language models locally on your own machine. No data leaves your network —
critical for healthcare environments with strict data privacy requirements (HIPAA, etc.).

## Why Local LLMs Matter in Healthcare
- **Data Privacy**: Patient data never leaves your infrastructure
- **Compliance**: Easier HIPAA/GDPR compliance with no third-party API calls
- **Cost**: No per-token API fees after initial hardware investment
- **Latency**: No network round-trips; responses from local GPU/CPU
- **Availability**: Works offline, no dependency on external services

## Prerequisites
1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
2. Pull a model: `ollama pull llama3`
3. Verify it's running: `ollama list`
4. Python packages: `pip install openai chromadb`

## Demos (main.py)
| # | Demo | What You Learn |
|---|------|---------------|
| 1 | Ollama Basics | OpenAI-compatible API with local models |
| 2 | Model Comparison | Compare llama3, mistral, phi3 on healthcare queries |
| 3 | Local Embeddings | Generate embeddings locally with nomic-embed-text |
| 4 | Local RAG System | Full RAG pipeline — zero data leaves the machine |

## Exercises
| # | File | Skills |
|---|------|--------|
| 1 | exercise_1_model_benchmarks.py | Benchmark speed, tokens, quality across models |
| 2 | exercise_2_local_medical_qa.py | Local medical Q&A vs GPT-4o quality |
| 3 | exercise_3_quantization_comparison.py | Q4 vs Q8 quantization tradeoffs |
| 4 | exercise_4_local_rag.py | Complete local RAG with ChromaDB |

## Ollama Quick Reference
```bash
ollama pull llama3          # Download a model
ollama pull mistral         # Download another model
ollama pull phi3            # Lightweight model
ollama pull nomic-embed-text # Embedding model
ollama list                 # Show downloaded models
ollama show llama3          # Model details
ollama rm modelname         # Remove a model
ollama serve                # Start the server (usually auto-starts)
```

## Key Concept: OpenAI-Compatible API
Ollama exposes an API identical to OpenAI's, so you can use the `openai` Python
library by simply changing the `base_url`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Running
```bash
python main.py              # Interactive demo menu
python exercise_1_model_benchmarks.py
python exercise_2_local_medical_qa.py
python exercise_3_quantization_comparison.py
python exercise_4_local_rag.py
```
