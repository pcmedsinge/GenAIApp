# Project 6: Agent Architectures — The Complete Catalog

## Why This Module Exists

You've already built agents using ReAct (Project 01), LangChain (02), LangGraph (03), and Multi-Agent patterns (04). But **how do you choose the RIGHT architecture for a given problem?**

This module teaches the **7 major agent architectures** side-by-side, with the same healthcare scenario in each — so you can see exactly how the architecture shapes the solution.

```
┌─────────────────────────────────────────────────────────────┐
│                  AGENT ARCHITECTURE MAP                      │
│                                                              │
│  Simple ◄──────────────────────────────────────► Complex    │
│                                                              │
│  1. ReAct          2. Plan-and-    3. Reflection            │
│  (Think→Act→         Execute        (Generate→              │
│   Observe loop)   (Plan first,      Critique→               │
│                    then execute)     Revise loop)            │
│       │                │                │                    │
│       ▼                ▼                ▼                    │
│  4. Router/        5. Parallel      6. Hierarchical         │
│  Supervisor          Fan-Out        (Supervisor              │
│  (Route to         (Split work,     spawns sub-             │
│   specialists)      aggregate)       agents)                │
│       │                │                │                    │
│       └────────────────┴────────────────┘                   │
│                        │                                     │
│                        ▼                                     │
│              7. Tool-Making Agent                            │
│              (Creates its own tools)                         │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- **Project 01** (ReAct pattern from scratch)
- **Project 03** (LangGraph state graphs)
- **Project 04** (Multi-agent basics)

## What You'll Learn

| Architecture | When to Use | Exercise |
|---|---|---|
| **ReAct** | General Q&A with tool use | (Covered in Project 01) |
| **Plan-and-Execute** | Multi-step tasks that benefit from upfront planning | Exercise 1 |
| **Reflection / Self-Critique** | Quality-critical outputs (clinical notes, reports) | Exercise 2 |
| **Parallel Fan-Out / Map-Reduce** | Same question to multiple perspectives, then merge | Exercise 3 |
| **Hierarchical Agents** | Complex tasks requiring delegation to sub-agents | Exercise 4 |
| **Router/Supervisor** | Multiple specialists, need to pick the right one | (Covered in Project 04) |
| **Tool-Making Agent** | Agent needs capabilities that don't exist yet | `main.py` Demo 4 |

## The Key Decision Framework

```
Q: "How many steps does the task need?"
  → 1-3 steps: ReAct is fine
  → 4+ steps with dependencies: Plan-and-Execute

Q: "Does quality matter more than speed?"
  → Yes (clinical reports, legal docs): Reflection
  → No (quick lookups): ReAct

Q: "Can parts of the work happen in parallel?"
  → Yes (multiple specialist opinions): Parallel Fan-Out
  → No (each step depends on the last): Sequential Pipeline

Q: "Is the task too complex for one agent?"
  → Subtasks are independent: Router/Supervisor
  → Subtasks have sub-subtasks: Hierarchical

Q: "Does the agent need new capabilities dynamically?"
  → Yes: Tool-Making Agent
  → No: Standard tool-use
```

## Architecture Comparison (At a Glance)

| Architecture | Strengths | Weaknesses | Token Cost | Latency |
|---|---|---|---|---|
| ReAct | Simple, flexible | No upfront planning | Low | Low |
| Plan-and-Execute | Structured, predictable | Slow start (planning phase) | Medium | Medium |
| Reflection | High-quality output | Slow (multiple revisions) | High | High |
| Router/Supervisor | Focused expertise | Routing errors cascade | Medium | Medium |
| Parallel Fan-Out | Fast, comprehensive | Hard to merge conflicting results | High | Low* |
| Hierarchical | Handles complexity | Hard to debug, expensive | Very High | High |
| Tool-Making | Ultra-flexible | Risky (executing generated code) | Medium | Medium |

*Low latency because parallel execution

## File Structure

```
06_agent_architectures/
├── README.md                           ← You are here
├── main.py                             ← Architecture overview + visual comparison
├── exercise_1_plan_and_execute.py      ← Plan upfront, then execute step-by-step
├── exercise_2_reflection.py            ← Generate → Critique → Revise loop
├── exercise_3_parallel_fanout.py       ← Fan-out to specialists, reduce to consensus
└── exercise_4_hierarchical.py          ← Supervisor delegates to sub-agents
```

## Estimated Time: 4–5 hours

| Part | Time | What You'll Do |
|------|------|----------------|
| `main.py` | 30 min | Run all 4 architecture demos side-by-side |
| Exercise 1 | 60 min | Build a Plan-and-Execute agent for treatment planning |
| Exercise 2 | 60 min | Build a Reflection agent for clinical note generation |
| Exercise 3 | 60 min | Build a Parallel Fan-Out system for second opinions |
| Exercise 4 | 60 min | Build a Hierarchical agent for complex case management |

## How to Run

```bash
# Overview — see all architectures compared
python level_3_agents/06_agent_architectures/main.py

# Individual exercises
python level_3_agents/06_agent_architectures/exercise_1_plan_and_execute.py
python level_3_agents/06_agent_architectures/exercise_2_reflection.py
python level_3_agents/06_agent_architectures/exercise_3_parallel_fanout.py
python level_3_agents/06_agent_architectures/exercise_4_hierarchical.py
```

## Key Takeaway

There is no "best" architecture. The right one depends on:
1. **Task complexity** — simple lookup vs. multi-step reasoning
2. **Quality requirements** — quick answer vs. reviewed clinical report
3. **Latency budget** — real-time chat vs. background processing
4. **Cost sensitivity** — each architecture has different token costs

Production systems often **combine architectures**: e.g., a Router (pick specialist) → Plan-and-Execute (structured treatment) → Reflection (review quality) → output.
