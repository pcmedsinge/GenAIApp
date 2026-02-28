# GenAI Developer Learning Path
## From Fundamentals to Production Healthcare AI

**Target Audience**: Experienced developers new to GenAI/LLMs  
**Prerequisites**: Python basics, understanding of AI/ML concepts  
**Focus**: Hands-on building, not just "Hello World"  
**Levels**: 8 (Fundamentals → RAG → Agents → Safety → MCP → Multimodal → Fine-Tuning → Production)

---

## 🎯 Learning Philosophy

1. **Build First, Theory Second**: Learn by doing real projects
2. **Progressive Complexity**: Each level builds on the previous
3. **Healthcare Context**: Examples relevant to healthcare IT
4. **Production Ready**: Not just demos, but deployable code
5. **Modern Stack**: MCP, reasoning models, multimodal — not just chat completions

---

## 📚 Learning Levels

### Level 1: GenAI Fundamentals (Week 1-2) ✅
**Goal**: Understand LLM APIs without frameworks

**Projects**:
- `01_basic_chat/`: Simple conversational AI
- `02_embeddings/`: Text similarity and semantic search
- `03_function_calling/`: Make LLMs call your functions
- `04_prompt_engineering/`: Effective prompt patterns
- `05_streaming/`: Real-time response streaming

**Key Concepts**:
- Tokens, context windows, temperature
- Embeddings and vector similarity
- Function/tool calling
- Prompt engineering patterns
- API cost optimization

**Time Investment**: 5-7 hours  
**Output**: 5 working applications

---

### Level 2: RAG Systems (Week 3-4) ✅
**Goal**: Build systems that augment LLMs with your own data

**Projects**:
- `01_simple_rag/`: RAG from scratch — no frameworks, just OpenAI API + cosine similarity
- `02_vector_databases/`: ChromaDB — persistent storage, CRUD, metadata filtering
- `03_document_processing/`: Chunking strategies — fixed, sentence, section-based with overlap
- `04_advanced_retrieval/`: Multi-query, LLM re-ranking, citations, evaluation metrics
- `05_medical_rag/`: Capstone — complete medical guidelines Q&A with confidence scoring

**Key Concepts**:
- The RAG pipeline: Retrieve → Augment → Generate
- Vector databases (ChromaDB)
- Chunking strategies and overlap
- Multi-query retrieval and re-ranking
- Source citations and evaluation

**Time Investment**: 8-10 hours  
**Output**: Production-ready RAG system for healthcare docs

---

### Level 3: AI Agents (Week 5-7) ✅
**Goal**: Build autonomous agents that think, plan, and act

**Projects**:
- `01_react_agent/`: Build the ReAct pattern from scratch — no frameworks, pure OpenAI
- `02_langchain_agents/`: LangChain agents with custom tools, memory, error handling
- `03_langgraph_workflows/`: LangGraph stateful agents — conditional routing, state management
- `04_multi_agent/`: Multi-agent collaboration — pipeline, router, and debate patterns
- `05_healthcare_agent/`: Capstone — clinical decision support with RAG + tools + safety guardrails

**Key Concepts**:
- ReAct pattern (Thought → Action → Observation loop)
- Custom tool creation and agent memory
- Graph-based workflow design (LangGraph)
- Multi-agent orchestration patterns
- Safety guardrails for healthcare AI

**Time Investment**: 12-15 hours  
**Output**: Intelligent healthcare clinical decision support system

---

### Level 4: Evaluation, Safety & Guardrails (Week 8-9)
**Goal**: Systematically evaluate, validate, and secure your AI systems

**Projects**:
- `01_prompt_testing/`: Systematic prompt evaluation — A/B testing, regression suites, scoring rubrics
- `02_rag_evaluation/`: RAG quality metrics — RAGAS (faithfulness, relevancy, context precision)
- `03_output_validation/`: Structured outputs — JSON schema enforcement, Pydantic validation, retry logic
- `04_guardrails/`: Safety layers — NeMo Guardrails, content filtering, hallucination detection
- `05_healthcare_compliance/`: Capstone — HIPAA considerations, PHI de-identification, audit trails, clinical validation

**Key Concepts**:
- RAGAS evaluation framework
- Prompt regression testing
- OpenAI structured outputs + Pydantic
- NeMo Guardrails (NVIDIA)
- HIPAA compliance for LLM usage
- Hallucination detection and mitigation

**Time Investment**: 10-12 hours  
**Output**: Evaluated, guarded, and compliant healthcare AI system

---

### Level 5: Model Context Protocol — MCP (Week 10-11)
**Goal**: Master the universal standard for connecting AI to tools and data

**Projects**:
- `01_mcp_fundamentals/`: Protocol architecture — hosts, clients, servers, transport (stdio/SSE)
- `02_mcp_tool_servers/`: Build MCP servers — expose Python functions as tools via `mcp` SDK
- `03_mcp_resources/`: Data exposure — static/dynamic resources, templates, subscriptions
- `04_mcp_with_agents/`: Integration — connect MCP servers to LangChain/LangGraph agents
- `05_healthcare_mcp/`: Capstone — healthcare MCP ecosystem (EHR, labs, medications, scheduling)

**Key Concepts**:
- MCP protocol (Anthropic's standard, adopted industry-wide)
- Transport: stdio for local, SSE/HTTP for remote
- Tool servers vs resource servers
- Dynamic tool discovery
- Multi-server orchestration

**Time Investment**: 12-15 hours  
**Output**: Healthcare MCP server ecosystem powering clinical agents

**Why MCP matters**: MCP is the "USB-C of AI" — instead of writing custom function-calling integrations for every tool, MCP provides a universal protocol. One agent can connect to any MCP-compatible server. This is the architecture pattern for 2025+ agentic AI.

---

### Level 6: Multimodal & Reasoning Models (Week 14-15)
**Goal**: Go beyond text — vision, audio, structured output, and extended reasoning

**Projects**:
- `01_vision_models/`: Image understanding — GPT-4o vision, Claude vision, document extraction
- `02_audio_transcription/`: Speech-to-text — Whisper, medical terminology, speaker diarization
- `03_structured_outputs/`: Schema-enforced generation — JSON schema, nested Pydantic, FHIR data
- `04_reasoning_models/`: Extended thinking — o1/o3, DeepSeek R1, when reasoning vs standard
- `05_medical_multimodal/`: Capstone — multimodal triage (image + audio + reasoning → clinical note)

**Key Concepts**:
- Vision APIs for image analysis
- Whisper for medical transcription
- Structured output with JSON schema enforcement
- Reasoning models (o1, o3, DeepSeek R1) vs standard models
- Cost/latency tradeoffs for model selection

**Time Investment**: 12-15 hours  
**Output**: Multimodal clinical assistant that sees, hears, and reasons

---

### Level 7: Fine-Tuning & Open-Source Models (Week 16-18)
**Goal**: Customize models for your domain and run locally for HIPAA compliance

**Projects**:
- `01_local_llms/`: Ollama — install, run, and serve Llama/Mistral locally with OpenAI-compatible API
- `02_huggingface_ecosystem/`: Transformers, model hub, tokenizers, inference pipelines
- `03_data_preparation/`: Training datasets — instruction format, quality filtering, synthetic data generation
- `04_lora_finetuning/`: Efficient fine-tuning — LoRA/QLoRA with PEFT + Accelerate
- `05_medical_model/`: Capstone — fine-tuned ICD-10 coder, evaluated and deployed locally via Ollama

**Key Concepts**:
- Ollama for local model deployment
- GGUF format and quantization (Q4, Q8)
- HuggingFace transformers ecosystem
- LoRA/QLoRA parameter-efficient fine-tuning
- Training data preparation and quality
- Model evaluation (accuracy, F1, confusion matrix)

**Time Investment**: 15-20 hours  
**Output**: Custom fine-tuned model running locally — HIPAA-compliant

---

### Level 8: Production & Deployment (Week 19-22)
**Goal**: Ship production-grade healthcare AI applications

**Projects**:
- `01_api_design/`: FastAPI — async endpoints, Pydantic schemas, rate limiting, auth
- `02_caching_optimization/`: Semantic caching, prompt caching (Anthropic/OpenAI), model routing
- `03_monitoring_observability/`: LangSmith/LangFuse tracing, cost dashboards, alerting
- `04_security_scaling/`: Prompt injection defense, input sanitization, load balancing
- `05_healthcare_platform/`: Capstone — complete healthcare AI platform (RAG + agents + MCP + guardrails + monitoring)

**Key Concepts**:
- FastAPI async patterns
- Semantic caching and prompt caching
- LLM observability (LangSmith, LangFuse)
- Prompt injection attacks and defenses
- Horizontal scaling and load balancing
- End-to-end platform architecture

**Time Investment**: 15-20 hours  
**Output**: Production-deployed healthcare AI platform — the final capstone

---

## 🗺️ Level Dependency Map

```
Level 1: Fundamentals
    ↓
Level 2: RAG ──────────────────────────────────────────┐
    ↓                                                   │
Level 3: Agents ───────────────────────────────┐        │
    ↓                                          │        │
Level 4: Evaluation & Safety                   │        │
    ↓                                          │        │
Level 5: MCP ← (uses agents from L3)──────────┘        │
    ↓                                                   │
Level 6: Multimodal & Reasoning                         │
    ↓                                                   │
Level 7: Fine-Tuning & Open Models                      │
    ↓                                                   │
Level 8: Production ← (combines everything)─────────────┘
```

---

## 🛠️ Tech Stack

### Core Libraries
```
openai>=1.0.0              # OpenAI API (GPT-4o, o1, Whisper, embeddings)
anthropic>=0.18.0          # Claude API
langchain>=0.3.0           # LLM framework
langchain-community        # Community integrations
langgraph>=0.2.0           # Stateful agent workflows
```

### Agentic Infrastructure (Level 5)
```
mcp                        # Model Context Protocol SDK
httpx                      # Async HTTP for MCP transports
```

### Vector Databases (Level 2)
```
chromadb                   # Local vector DB
faiss-cpu                  # Facebook AI similarity search
```

### Evaluation & Safety (Level 4)
```
ragas                      # RAG evaluation framework
nemoguardrails             # NVIDIA guardrails framework
pydantic>=2.0              # Output validation
```

### Fine-tuning & Open Models (Level 7)
```
ollama                     # Local model runtime
transformers               # HuggingFace model loading
datasets                   # HuggingFace datasets
peft                       # Parameter-efficient fine-tuning (LoRA)
bitsandbytes               # Quantization
accelerate                 # Training acceleration
```

### Production & Deployment (Level 8)
```
fastapi                    # Async API framework
uvicorn                    # ASGI server
redis                      # Caching layer
langsmith                  # LLM observability
streamlit                  # Quick UIs / dashboards
```

### Supporting Tools
```
python-dotenv              # Environment management
tiktoken                   # Token counting
numpy                      # Numerical computing
pandas                     # Data manipulation
```

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys Setup
Create `.env` file:
```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

### 3. Start with Level 1
```bash
cd level_1_fundamentals/01_basic_chat
python main.py
```

---

## 💡 Learning Tips

### For Healthcare IT Professionals
1. **Start with familiar problems**: Think about repetitive tasks in your current work
2. **Data privacy first**: Learn about de-identification before using patient data
3. **Regulatory awareness**: Understand HIPAA implications of LLM usage
4. **Validation matters**: Always validate LLM outputs, especially in clinical contexts

### General Tips
1. **Read error messages**: LLM APIs give helpful errors
2. **Monitor costs**: Start with small limits, use cheaper models for testing
3. **Iterate prompts**: Spend time on prompt engineering before complex solutions
4. **Test edge cases**: LLMs can hallucinate - test thoroughly

---

## 📖 Recommended Reading Order

1. **Before Level 1**: 
   - OpenAI API documentation
   - Understanding tokens and context windows

2. **Before Level 2**: 
   - Vector databases basics
   - Embedding models comparison

3. **Before Level 3**: 
   - ReAct paper
   - LangChain / LangGraph documentation

4. **Before Level 4**: 
   - RAGAS documentation
   - OpenAI structured outputs guide

5. **Before Level 5**: 
   - MCP specification (modelcontextprotocol.io)
   - MCP Python SDK documentation

6. **Before Level 6**: 
   - OpenAI vision and audio API docs
   - OpenAI reasoning model guide (o1/o3)

7. **Before Level 7**: 
   - Ollama documentation
   - LoRA paper (don't worry if it's dense)
   - HuggingFace PEFT documentation

8. **Before Level 8**: 
   - FastAPI documentation
   - LangSmith/LangFuse getting started

---

## 🎓 Success Metrics

After completing this path, you should be able to:

- ✅ Build conversational AI applications
- ✅ Create RAG systems for domain-specific knowledge
- ✅ Develop autonomous agents that use tools
- ✅ Evaluate and secure AI systems for healthcare
- ✅ Build and connect MCP servers for tool ecosystems
- ✅ Work with vision, audio, and reasoning models
- ✅ Fine-tune models for specialized tasks
- ✅ Run models locally for HIPAA compliance
- ✅ Deploy production-ready GenAI applications
- ✅ Understand cost/performance/safety tradeoffs

---

## 🏥 Healthcare-Specific Outcomes

You'll have built:
1. Medical literature Q&A system (RAG)
2. Clinical decision support agent (Agents)
3. Evaluated and guarded clinical AI (Safety)
4. MCP-powered healthcare tool ecosystem (MCP)
5. Multimodal clinical triage assistant (Multimodal)
6. ICD-10 coding model running locally (Fine-tuning)
7. Complete healthcare AI platform (Production)

---

## 🤝 Community & Resources

- **MCP**: modelcontextprotocol.io — specification, servers, SDKs
- **LangChain**: python.langchain.com — framework docs
- **HuggingFace**: huggingface.co — models, datasets, papers
- **ArXiv**: arxiv.org — latest AI research papers
- **Ollama**: ollama.com — local model runtime
- **OpenAI**: platform.openai.com — API documentation
- **Anthropic**: docs.anthropic.com — Claude API + MCP

---

## 📝 Notes

- **Cost**: Expect ~$45-70 in API costs for the full learning path (local models are free)
- **Time**: ~90-115 hours total (can spread over 4-6 months)
- **Hardware**: Levels 1-6 run on CPU. Level 7 fine-tuning benefits from GPU. Ollama needs 8GB+ RAM
- **Updates**: GenAI moves fast — check for library updates regularly

---

## 🚦 Your Current Status

Track your progress:
- [x] Level 1: Fundamentals ✅
- [x] Level 2: RAG Systems ✅
- [x] Level 3: AI Agents ✅
- [ ] Level 4: Evaluation, Safety & Guardrails
- [ ] Level 5: Model Context Protocol (MCP)
- [ ] Level 6: Multimodal & Reasoning Models
- [ ] Level 7: Fine-Tuning & Open-Source Models
- [ ] Level 8: Production & Deployment

**Next Step**: Level 4 — Evaluate and secure everything you've built

---

Good luck on your GenAI journey! Remember: the best way to learn is to build. 🚀
