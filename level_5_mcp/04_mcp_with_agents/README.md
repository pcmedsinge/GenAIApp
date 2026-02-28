# Project 04: MCP with Agents

## Overview

MCP becomes truly powerful when connected to autonomous AI agents. This project
bridges MCP tool/resource servers with LangChain and LangGraph agents, enabling
dynamic tool discovery, multi-server orchestration, and sophisticated clinical
workflows.

Instead of hard-coding tool functions into agents, MCP lets agents discover
available tools at runtime — making them adaptable and extensible without code
changes.

## Key Concepts

- **MCP → LangChain Bridge** — Convert MCP tool schemas to LangChain `@tool` functions
- **Dynamic Tool Discovery** — Agents discover tools from MCP servers at runtime
- **Multi-Server Orchestration** — A single agent uses tools from multiple MCP servers
- **LangGraph + MCP** — Workflow nodes that route to different MCP servers
- **Tool Schema Mapping** — MCP `inputSchema` → LangChain tool parameters
- **Agent Routing** — Query classification determines which MCP server to invoke

## Healthcare Context

A clinical AI assistant needs access to many systems: EHR for patient data,
lab systems for results, pharmacy for medications, scheduling for appointments.
MCP lets one agent seamlessly access all of these through a standard protocol,
with each system exposed as a separate MCP server. This mirrors the real-world
integration challenge in healthcare IT.

## Demos (main.py)

1. **MCP Tools to LangChain** — Convert MCP tool definitions into LangChain-compatible tools
2. **Dynamic Tool Discovery** — Agent discovers available tools from MCP server at runtime
3. **Multi-Server Agent** — Agent connects to medication, lab, and vitals MCP servers
4. **Agent Workflow with MCP** — LangGraph workflow where nodes use different MCP servers

## Exercises

1. **MCP-LangChain Bridge** (`exercise_1_mcp_langchain_bridge.py`) — Systematic bridge from MCP schemas to LangChain tools
2. **Multi-Server Agent** (`exercise_2_multi_server_agent.py`) — Agent routes queries to medication, lab, and vitals servers
3. **Dynamic Routing** (`exercise_3_dynamic_routing.py`) — LangGraph workflow with query-based server routing
4. **MCP Agent Evaluation** (`exercise_4_mcp_agent_eval.py`) — Evaluate tool selection accuracy, parameter extraction, response quality

## Running

```bash
# Run demos
python main.py

# Run exercises
python exercise_1_mcp_langchain_bridge.py
python exercise_2_multi_server_agent.py
python exercise_3_dynamic_routing.py
python exercise_4_mcp_agent_eval.py
```

## Prerequisites

- Python packages: `mcp`, `python-dotenv`, `openai`, `langchain-openai`, `langchain-core`, `langgraph`
- Completion of Level 3 (Agents) and Level 5 Projects 01-03

## Architecture

```
                    ┌──────────────────────┐
                    │   LangGraph Agent     │
                    │   (query router)      │
                    └─────┬───┬───┬────────┘
                          │   │   │
              ┌───────────┘   │   └───────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ MCP: Meds│   │ MCP: Labs│   │MCP:Vitals│
        └──────────┘   └──────────┘   └──────────┘
```
