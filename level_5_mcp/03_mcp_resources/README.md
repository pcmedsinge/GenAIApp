# Project 03: MCP Resources

## Overview

MCP **Resources** are the read-only data counterpart to **Tools**. While tools
perform actions (calculate, check, create), resources expose stable data that
AI agents can read — formularies, reference ranges, clinical guidelines, and
more. Resources use URI-based addressing and support both static content and
dynamic, parameterized templates.

This project teaches you to build MCP resource servers that expose healthcare
data in a structured, discoverable way.

## Key Concepts

- **Resources vs Tools** — Resources are read-only data; Tools perform actions with side effects
- **Resource URIs** — Every resource has a unique URI (e.g., `formulary://medications/metformin`)
- **Static Resources** — Fixed data exposed at a known URI (formulary list, reference tables)
- **Dynamic Resources / Templates** — URI templates with parameters (e.g., `patient/{id}/vitals`)
- **Content Types** — Resources return typed content: text, JSON, markdown, binary
- **Resource Catalog** — `resources/list` lets agents discover all available resources
- **Subscriptions** — Clients can subscribe to resource change notifications

## Healthcare Context

Healthcare AI agents need access to large bodies of reference data: drug
formularies, lab reference ranges, clinical guidelines, ICD-10 catalogs.
MCP resources provide a standard, cacheable way to expose this data without
the overhead of tool invocations — ideal for frequently-read, rarely-changed
information.

## Demos (main.py)

1. **Static Resources** — Expose fixed data (formulary list, reference ranges) with URIs
2. **Dynamic Resources** — Resources with URI template parameters (`patient/{id}/vitals`)
3. **Resource Content Types** — Text, JSON, and markdown resources for different data types
4. **Resource Catalog** — List all available resources with descriptions and MIME types

## Exercises

1. **Guideline Resources** (`exercise_1_guideline_resources.py`) — URI pattern `guideline/{specialty}/{topic}`
2. **Formulary Resource** (`exercise_2_formulary_resource.py`) — Full formulary server with medication lookup and interactions
3. **Resource Templates** (`exercise_3_resource_templates.py`) — Parameterized lab references with age/sex adjustment
4. **Resource Subscriptions** (`exercise_4_resource_subscriptions.py`) — Simulate change notifications with a changelog

## Running

```bash
# Run demos
python main.py

# Run exercises
python exercise_1_guideline_resources.py
python exercise_2_formulary_resource.py
python exercise_3_resource_templates.py
python exercise_4_resource_subscriptions.py
```

## Prerequisites

- Python packages: `mcp`, `python-dotenv`, `openai`
- Completion of 01_mcp_fundamentals and 02_mcp_tool_servers

## MCP Resources Quick Reference

```
Client → Server:  resources/list          → catalog of available resources
Client → Server:  resources/read { uri }  → content of a specific resource
Client → Server:  resources/subscribe     → watch a resource for changes
Server → Client:  notifications/resources/updated  → resource changed
```
