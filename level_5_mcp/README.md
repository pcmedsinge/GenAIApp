# Level 5: Model Context Protocol (MCP)
**The universal standard for connecting AI to tools and data**

## Overview

MCP (Model Context Protocol) is the "USB-C of AI" — a universal protocol for connecting
LLMs to external tools and data sources. Created by Anthropic, now adopted by OpenAI,
Google, Microsoft, and the broader AI ecosystem.

### Why MCP?

```
Without MCP:  Custom function definitions per model → brittle, non-portable
With MCP:     Universal protocol → any agent connects to any MCP server
```

### The Architecture

```
┌─────────────┐     ┌────────────────┐     ┌─────────────────┐
│  AI Agent    │────▶│  MCP Client    │────▶│  MCP Server     │
│  (LangChain, │     │  (in your app) │     │  (tools + data) │
│   LangGraph) │     └────────────────┘     └─────────────────┘
└─────────────┘            │                        │
                           │ MCP Protocol            │ Your code
                           │ (JSON-RPC)              │ (Python)
                           ▼                        ▼
                    Tools: call functions     Resources: read data
                    Prompts: templates       Notifications: updates
```

## Prerequisites
- **Levels 1-3 Complete**: You'll connect agents to MCP servers
- **Level 4 Recommended**: Safety guardrails for tool calls
- **OpenAI API Key**: Configured in .env
- **MCP SDK**: `pip install mcp` (in requirements.txt)

## Projects

### 01_mcp_fundamentals — Understanding the Protocol
- MCP architecture: hosts, clients, servers
- Transport mechanisms (stdio, SSE/HTTP)
- Protocol lifecycle: initialize → operate → shutdown
- **Healthcare Example**: Interact with a medical reference MCP server

### 02_mcp_tool_servers — Building MCP Tool Servers
- Create MCP servers in Python with the `mcp` SDK
- Expose Python functions as MCP tools
- Tool schemas, parameter validation, error handling
- **Healthcare Example**: BMI calculator + drug lookup MCP server

### 03_mcp_resources — Exposing Data Through MCP
- Resources vs tools (read data vs perform actions)
- Static and dynamic resource templates
- Resource URIs and content types
- **Healthcare Example**: Medical guidelines resource server

### 04_mcp_with_agents — MCP + LangChain/LangGraph Integration
- Connect MCP servers to LangChain agents
- Dynamic tool discovery from MCP servers
- Multi-server orchestration
- **Healthcare Example**: Agent using MCP tools from multiple servers

### 05_healthcare_mcp — Capstone: Healthcare MCP Ecosystem
- EHR data server (patient demographics, encounters, vitals)
- Lab results server (lookup, interpretation, trending)
- Medication server (formulary, interactions, prior auth)
- **Healthcare Example**: Clinical agent powered entirely by MCP servers

## Learning Objectives

After completing Level 5, you will:
- ✅ Understand MCP protocol architecture and transport
- ✅ Build MCP tool servers that expose Python functions
- ✅ Create MCP resource servers for data exposure
- ✅ Connect MCP servers to LangChain/LangGraph agents
- ✅ Orchestrate multiple MCP servers from a single agent
- ✅ Build a healthcare MCP server ecosystem

## Time Estimate
12-15 hours total (2.5-3 hours per project)

## How This Connects to Level 6
Level 6 adds multimodal capabilities (vision, audio, reasoning). The MCP architecture
from this level provides the tool infrastructure — Level 6 adds the AI capabilities.
