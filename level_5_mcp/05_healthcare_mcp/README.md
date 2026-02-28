# Project 05: Healthcare MCP Capstone

## Overview

This capstone project brings together everything from Level 5 — MCP protocol,
tool servers, resource servers, and agent integration — to build a **complete
healthcare MCP ecosystem**. You'll create three domain-specific MCP servers
(EHR, Clinical Lab, Pharmacy), connect them to a unified clinical agent, and
build supporting infrastructure (scheduling, workflow orchestration, server
registry, and end-to-end testing).

This project mirrors real-world healthcare AI integration challenges.

## Key Concepts

- **Multi-Server Ecosystem** — Multiple specialized MCP servers working together
- **Unified Clinical Agent** — Single agent that orchestrates all servers
- **EHR Server** — Patient demographics, encounters, vitals, problem lists
- **Clinical Lab Server** — Lab orders, results, interpretations, critical alerts
- **Pharmacy Server** — Formulary, interactions, prior auth, refill management
- **Server Registry** — Dynamic discovery and health monitoring of MCP servers
- **End-to-End Testing** — Comprehensive test suite for the whole ecosystem

## Healthcare Context

A modern hospital runs dozens of clinical systems: EHRs, lab information
systems (LIS), pharmacy management, scheduling, radiology, and more. MCP
provides a universal integration layer where each system becomes an MCP
server, and AI agents can access all of them through a single protocol.
This capstone builds a miniature version of that architecture.

## Demos (main.py)

1. **EHR Server** — Patient lookup, demographics, encounter history, vitals trending
2. **Clinical Lab Server** — Lab orders, results, interpretations, critical value alerts
3. **Pharmacy Server** — Formulary, interactions, prior auth status, refill management
4. **Unified Clinical Agent** — Single agent using all three servers for clinical queries

## Exercises

1. **Scheduling Server** (`exercise_1_scheduling_server.py`) — Appointment scheduling with availability, booking, and cancellation
2. **Clinical Workflow** (`exercise_2_clinical_workflow.py`) — Complete patient encounter workflow using multiple MCP servers
3. **Server Registry** (`exercise_3_server_registry.py`) — Dynamic MCP server registry with health monitoring
4. **Ecosystem Test Suite** (`exercise_4_mcp_ecosystem_test.py`) — End-to-end testing of the healthcare MCP ecosystem

## Running

```bash
# Run demos
python main.py

# Run exercises
python exercise_1_scheduling_server.py
python exercise_2_clinical_workflow.py
python exercise_3_server_registry.py
python exercise_4_mcp_ecosystem_test.py
```

## Prerequisites

- Python packages: `mcp`, `python-dotenv`, `openai`, `langchain-openai`, `langchain-core`
- Completion of Level 5 Projects 01-04

## Architecture

```
                      ┌─────────────────────────────┐
                      │   Unified Clinical Agent      │
                      │   (LangGraph orchestration)   │
                      └──┬──────┬──────┬──────┬──────┘
                         │      │      │      │
          ┌──────────────┘      │      │      └──────────────┐
          ▼                     ▼      ▼                     ▼
   ┌────────────┐     ┌──────────┐  ┌─────────┐    ┌─────────────┐
   │ EHR Server │     │Lab Server│  │Pharmacy │    │ Scheduling  │
   │ (patients, │     │(results, │  │(formulary│    │  (appts,    │
   │  encounters│     │ orders)  │  │ refills) │    │  calendar)  │
   └────────────┘     └──────────┘  └─────────┘    └─────────────┘
```
