# Project 01: MCP Fundamentals

## Overview

The **Model Context Protocol (MCP)** is an open standard created by Anthropic for connecting
AI models to external tools and data sources. Think of it as "USB-C for AI" — a universal
interface that lets any AI agent connect to any MCP-compatible server, regardless of who
built either side.

This project teaches the protocol from the ground up: architecture, transport mechanisms,
the protocol lifecycle (initialize → operate → shutdown), and how to build your first
MCP server using the Python SDK.

## Key Concepts

- **MCP Architecture** — Three-layer model: Host (AI app) → Client (protocol handler) → Server (tools + data)
- **JSON-RPC 2.0** — The wire format for all MCP messages (requests, responses, notifications)
- **Transport** — How messages travel: stdio (local processes) or SSE/HTTP (remote servers)
- **Protocol Lifecycle** — Initialize handshake, capability negotiation, operation, graceful shutdown
- **FastMCP** — The high-level Python SDK for building MCP servers with minimal boilerplate
- **Tools vs Resources** — Tools perform actions (calculate, lookup); Resources expose read-only data

## Healthcare Context

Healthcare systems have many data sources: EHRs, lab systems, formularies, clinical
calculators. MCP provides a standard way for AI agents to access all of these through
a single protocol — no custom integrations per model or vendor.

## Demos (main.py)

1. **MCP Architecture Explained** — Walk through protocol layers, message types, and JSON-RPC format
2. **Simple MCP Server** — Build a minimal MCP server with FastMCP (tool + resource)
3. **MCP Client Communication** — Explore the JSON-RPC message flow: handshake, list tools, call tool
4. **Interactive MCP Explorer** — Pick a tool, examine its schema, simulate a call

## Exercises

1. **Protocol Messages** (`exercise_1_protocol_messages.py`) — Construct MCP JSON-RPC messages by hand, validate structure
2. **Simple Server** (`exercise_2_simple_server.py`) — Build an MCP server with 3 healthcare tools using FastMCP
3. **Server Testing** (`exercise_3_server_testing.py`) — Build a test harness that validates MCP tool behavior and schemas
4. **Transport Comparison** (`exercise_4_transport_comparison.py`) — Compare stdio vs SSE transport, build configs for both

## Running

```bash
# Run demos
python main.py

# Run exercises
python exercise_1_protocol_messages.py
python exercise_2_simple_server.py
python exercise_3_server_testing.py
python exercise_4_transport_comparison.py
```

## Prerequisites

- Python packages: `mcp`, `python-dotenv`
- Completion of Levels 1-3 recommended
- Level 4 (Guardrails) helpful for safety context

## MCP Protocol Quick Reference

```
Client → Server:  initialize, tools/list, tools/call, resources/read
Server → Client:  initialize response, tool results, resource contents
Notifications:    progress, log, resource updates (no response expected)
```
