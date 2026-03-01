# Module 07: Agent Framework Landscape & Evaluation

## Why This Module Exists

Agent frameworks appear monthly. You can't (and shouldn't) deeply learn each one.
What you CAN do is **evaluate any framework in 30 minutes** by mapping it to
patterns you already understand.

This module teaches you **how to think about frameworks**, not how to use each one.

## The Landscape (as of early 2026)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AGENT FRAMEWORK LANDSCAPE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Model Provider SDKs          Orchestration Frameworks              │
│  ────────────────────         ────────────────────────              │
│  OpenAI Agents SDK            LangGraph (LangChain)                │
│  Anthropic Tool Use           CrewAI                                │
│  Google ADK                   AutoGen (Microsoft)                   │
│  Amazon Bedrock Agents        Semantic Kernel (Microsoft)           │
│                                                                     │
│  RAG-First Frameworks         Lightweight / Minimal                 │
│  ────────────────────         ────────────────────                  │
│  LlamaIndex Workflows         Pydantic AI                          │
│  Haystack (deepset)           Instructor                            │
│                               Marvin                                │
│                                                                     │
│  Multi-Agent Specialized      Emerging / Watch List                 │
│  ────────────────────         ────────────────────                  │
│  CrewAI                       DSPy (prompt optimization)            │
│  AutoGen                      ControlFlow                           │
│  LangGraph (multi-agent)      Julep                                 │
│                               Letta (MemGPT)                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Insight

Every framework implements the SAME core patterns:

| Pattern You Know | LangGraph | CrewAI | AutoGen | OpenAI SDK |
|---|---|---|---|---|
| ReAct loop | StateGraph + ToolNode | Agent + Task | ConversableAgent | Agent + tools |
| Tool calling | @tool + ToolNode | @tool | function_map | function tools |
| Multi-agent | Send / subgraphs | Crew + Agents | GroupChat | Handoffs |
| State management | TypedDict state | Shared memory | Chat history | Context |
| Human-in-loop | interrupt_before | human_input=True | human_input_mode | Guardrails |
| Persistence | Checkpointer | Memory | Cache | Thread storage |

**The patterns are identical. The APIs are different.**

## Exercises

| # | Exercise | What You Learn |
|---|---|---|
| main.py | Framework Overview | The 10+ frameworks, categorized, with key traits |
| exercise_1 | Evaluation Rubric | 8-dimension scoring system to rate any framework |
| exercise_2 | Pattern Mapping | Map YOUR LangGraph knowledge to 5 other frameworks |
| exercise_3 | Decision Guide | "Which framework for which job?" decision tree |

## The Evaluation Mindset

When a new framework appears, ask these 8 questions:

1. **What pattern does it implement?** (ReAct? Plan-and-execute? Multi-agent?)
2. **What's the state model?** (Messages? Custom? Graph?)
3. **How does it handle tools?** (Decorator? Schema? Auto-discovery?)
4. **Can I add human oversight?** (Breakpoints? Approval? Review?)
5. **How does it persist?** (Memory? DB? Files?)
6. **Does it lock me into a model provider?** (OpenAI only? Any LLM?)
7. **What's the debugging story?** (Tracing? Logging? Visualization?)
8. **Is it production-ready?** (Async? Streaming? Error handling?)

If you can answer these 8 questions, you understand the framework.
