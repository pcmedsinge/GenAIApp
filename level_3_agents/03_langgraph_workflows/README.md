# Project 3: LangGraph Workflows — Stateful Agent Design

## What You'll Learn
- Graph-based agent design (nodes = steps, edges = flow)
- State management across steps (TypedDict)
- Conditional routing (different paths based on context)
- Why LangGraph is the production standard for complex agents

## Why LangGraph?
LangChain agents are great for simple tool-use. But real healthcare workflows need:
- **Conditional logic**: If emergency → route to urgent path; else → standard path
- **State tracking**: Accumulate patient data across multiple steps
- **Structured flow**: Triage → Assessment → Plan (not random tool calls)
- **Reliability**: Predictable paths, not hoping the LLM picks the right tool

```
LangChain Agent: LLM decides everything (flexible but unpredictable)
LangGraph:       YOU design the flow, LLM handles reasoning at each step
```

## Running the Code

```bash
cd level_3_agents/03_langgraph_workflows
python main.py
```

## Demos
1. **Simple Graph** — Nodes, edges, and state in a basic workflow
2. **Conditional Routing** — Different paths based on patient urgency
3. **Clinical Triage Workflow** — Full intake → classify → route → respond pipeline
4. **Stateful Multi-Step** — Track patient data across graph nodes

## Exercises
1. Add a "pharmacy check" node that validates medication dosing before final recommendation
2. Implement a "review" node for high-risk recommendations (human-in-the-loop)
3. Build a graph that handles both new patient intake AND follow-up visits
4. Add persistence — save and resume a workflow mid-process
