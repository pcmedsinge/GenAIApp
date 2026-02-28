# Project 02: MCP Tool Servers

## Overview

MCP tool servers are the workhorses of the Model Context Protocol — they expose Python
functions as **tools** that any MCP-compatible AI agent can discover and call. This project
teaches you to build robust, production-quality tool servers using the `mcp` Python SDK.

You'll learn how to define tools with the `@mcp.tool()` decorator, write clear descriptions
that guide agent behavior, add input validation, handle errors gracefully, and organize
tools into logical categories for discoverability.

Builds on: **01_mcp_fundamentals** (protocol architecture and message format).

## Key Concepts

- **@mcp.tool() Decorator** — Register Python functions as MCP tools with auto-generated schemas
- **Tool Schemas** — JSON Schema descriptions of parameters, types, and constraints
- **Tool Descriptions** — Natural language descriptions that guide agent tool selection
- **Input Validation** — Range checks, enum validation, required fields, type coercion
- **Error Handling** — Graceful failures with structured MCP error responses
- **Tool Categories** — Organize tools by domain for better discoverability
- **Tool Composition** — Tools that internally call other tools for complex workflows

## Healthcare Context

Healthcare AI agents need access to clinical calculators, drug databases, lab reference
ranges, and clinical decision support. Each of these can be exposed as an MCP tool server,
giving agents standardized access to validated clinical logic.

## Demos (main.py)

1. **Healthcare Tool Server** — Build a clinical MCP server with BMI, drug lookup, lab interpretation, interaction check
2. **Tool Schemas and Descriptions** — Show how descriptions guide agent behavior, examine auto-generated schemas
3. **Error Handling in Tools** — Handle invalid inputs, missing data, and tool failures gracefully
4. **Tool Categories** — Organize tools into clinical, medication, and laboratory categories

## Exercises

1. **Medication Server** (`exercise_1_medication_server.py`) — Build a medication-focused MCP server with 10+ drugs
2. **Lab Server** (`exercise_2_lab_server.py`) — Build a laboratory MCP server with 10+ lab tests
3. **Input Validation** (`exercise_3_input_validation.py`) — Add robust validation: ranges, enums, required fields
4. **Tool Composition** (`exercise_4_tool_composition.py`) — Build tools that call other tools for comprehensive assessments

## Running

```bash
# Run demos
python main.py

# Run exercises
python exercise_1_medication_server.py
python exercise_2_lab_server.py
python exercise_3_input_validation.py
python exercise_4_tool_composition.py
```

## Prerequisites

- Python packages: `mcp`, `python-dotenv`
- Completion of **01_mcp_fundamentals**
- OpenAI API key configured in `.env`

## Tool Design Quick Reference

```python
@mcp.tool()
def my_tool(param1: str, param2: int = 10) -> str:
    """Clear description of what this tool does and when to use it."""
    # Validate inputs → Process → Return structured result
    return json.dumps({"result": "value"})
```
