"""
Exercise 1: MCP Protocol Messages
===================================

Skills practiced:
- Understanding JSON-RPC 2.0 message format
- Building MCP protocol messages by hand
- Validating message structure against the MCP specification
- Distinguishing requests, responses, and notifications

Healthcare context:
Before using high-level SDKs, it's valuable to understand the raw protocol
messages that flow between MCP clients and servers. This exercise constructs
every major message type by hand — initialize, tools/list, tools/call,
resources/read — and validates their structure. This understanding helps
debug protocol issues and design better tool interfaces.

Usage:
    python exercise_1_protocol_messages.py
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Message structure validators
# ---------------------------------------------------------------------------

def validate_jsonrpc_base(message: dict) -> list[str]:
    """Validate that a message has the base JSON-RPC 2.0 fields."""
    errors = []
    if message.get("jsonrpc") != "2.0":
        errors.append("Missing or incorrect 'jsonrpc' field (must be '2.0')")
    if "method" not in message and "result" not in message and "error" not in message:
        errors.append("Message must have 'method' (request/notification), "
                       "'result' (success response), or 'error' (error response)")
    return errors


def validate_request(message: dict) -> list[str]:
    """Validate a JSON-RPC request message."""
    errors = validate_jsonrpc_base(message)
    if "id" not in message:
        errors.append("Request must have 'id' field")
    if "method" not in message:
        errors.append("Request must have 'method' field")
    elif not isinstance(message["method"], str):
        errors.append("'method' must be a string")
    if "params" in message and not isinstance(message["params"], dict):
        errors.append("'params' must be an object if present")
    return errors


def validate_response(message: dict) -> list[str]:
    """Validate a JSON-RPC response message."""
    errors = validate_jsonrpc_base(message)
    if "id" not in message:
        errors.append("Response must have 'id' field matching the request")
    if "result" not in message and "error" not in message:
        errors.append("Response must have either 'result' or 'error'")
    if "result" in message and "error" in message:
        errors.append("Response cannot have both 'result' and 'error'")
    return errors


def validate_notification(message: dict) -> list[str]:
    """Validate a JSON-RPC notification (request without 'id')."""
    errors = validate_jsonrpc_base(message)
    if "id" in message:
        errors.append("Notification must NOT have 'id' field")
    if "method" not in message:
        errors.append("Notification must have 'method' field")
    return errors


def validate_error_response(message: dict) -> list[str]:
    """Validate an error response message."""
    errors = validate_response(message)
    if "error" in message:
        err = message["error"]
        if not isinstance(err, dict):
            errors.append("'error' must be an object")
        else:
            if "code" not in err:
                errors.append("Error object must have 'code' (integer)")
            elif not isinstance(err["code"], int):
                errors.append("Error 'code' must be an integer")
            if "message" not in err:
                errors.append("Error object must have 'message' (string)")
    return errors


# ---------------------------------------------------------------------------
# MCP-specific message validators
# ---------------------------------------------------------------------------

def validate_initialize_request(message: dict) -> list[str]:
    """Validate an MCP initialize request."""
    errors = validate_request(message)
    if message.get("method") != "initialize":
        errors.append("Method must be 'initialize'")
    params = message.get("params", {})
    if "protocolVersion" not in params:
        errors.append("Initialize params must include 'protocolVersion'")
    if "capabilities" not in params:
        errors.append("Initialize params must include 'capabilities'")
    if "clientInfo" not in params:
        errors.append("Initialize params must include 'clientInfo'")
    else:
        client_info = params["clientInfo"]
        if "name" not in client_info:
            errors.append("clientInfo must include 'name'")
        if "version" not in client_info:
            errors.append("clientInfo must include 'version'")
    return errors


def validate_initialize_response(message: dict) -> list[str]:
    """Validate an MCP initialize response."""
    errors = validate_response(message)
    result = message.get("result", {})
    if "protocolVersion" not in result:
        errors.append("Initialize result must include 'protocolVersion'")
    if "capabilities" not in result:
        errors.append("Initialize result must include 'capabilities'")
    if "serverInfo" not in result:
        errors.append("Initialize result must include 'serverInfo'")
    return errors


def validate_tools_call_request(message: dict) -> list[str]:
    """Validate a tools/call request."""
    errors = validate_request(message)
    if message.get("method") != "tools/call":
        errors.append("Method must be 'tools/call'")
    params = message.get("params", {})
    if "name" not in params:
        errors.append("tools/call params must include 'name'")
    if "arguments" in params and not isinstance(params["arguments"], dict):
        errors.append("'arguments' must be an object")
    return errors


def validate_tool_result(message: dict) -> list[str]:
    """Validate a tools/call response."""
    errors = validate_response(message)
    result = message.get("result", {})
    if "content" not in result:
        errors.append("Tool result must include 'content' array")
    elif not isinstance(result["content"], list):
        errors.append("'content' must be a list")
    else:
        for i, item in enumerate(result["content"]):
            if "type" not in item:
                errors.append(f"Content item {i} must have 'type'")
            if "text" not in item and "data" not in item:
                errors.append(f"Content item {i} must have 'text' or 'data'")
    return errors


# ---------------------------------------------------------------------------
# Build and validate all message types
# ---------------------------------------------------------------------------

def build_and_validate_messages():
    """Construct every major MCP message type and validate them."""

    print("=" * 70)
    print("  Exercise 1: MCP Protocol Messages")
    print("  Building and validating JSON-RPC messages by hand")
    print("=" * 70)

    all_passed = True

    # --- 1. Initialize Request ---
    print("\n  1. Initialize Request (Client → Server)")
    print("  " + "─" * 55)
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True}
            },
            "clientInfo": {
                "name": "HealthcareAgent",
                "version": "1.0.0"
            }
        }
    }
    print(f"    {json.dumps(init_request, indent=4)[:300]}...")
    errors = validate_initialize_request(init_request)
    if errors:
        print(f"    ✗ INVALID: {errors}")
        all_passed = False
    else:
        print("    ✓ Valid initialize request")

    # --- 2. Initialize Response ---
    print("\n  2. Initialize Response (Server → Client)")
    print("  " + "─" * 55)
    init_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"subscribe": True}
            },
            "serverInfo": {
                "name": "ClinicalToolsServer",
                "version": "1.0.0"
            }
        }
    }
    print(f"    {json.dumps(init_response, indent=4)[:300]}...")
    errors = validate_initialize_response(init_response)
    if errors:
        print(f"    ✗ INVALID: {errors}")
        all_passed = False
    else:
        print("    ✓ Valid initialize response")

    # --- 3. tools/list Request ---
    print("\n  3. tools/list Request")
    print("  " + "─" * 55)
    tools_list_req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    }
    print(f"    {json.dumps(tools_list_req)}")
    errors = validate_request(tools_list_req)
    if errors:
        print(f"    ✗ INVALID: {errors}")
        all_passed = False
    else:
        print("    ✓ Valid tools/list request")

    # --- 4. tools/list Response ---
    print("\n  4. tools/list Response")
    print("  " + "─" * 55)
    tools_list_resp = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "tools": [
                {
                    "name": "calculate_bmi",
                    "description": "Calculate Body Mass Index",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "weight_kg": {"type": "number"},
                            "height_m": {"type": "number"}
                        },
                        "required": ["weight_kg", "height_m"]
                    }
                }
            ]
        }
    }
    print(f"    {json.dumps(tools_list_resp, indent=4)[:300]}...")
    errors = validate_response(tools_list_resp)
    if errors:
        print(f"    ✗ INVALID: {errors}")
        all_passed = False
    else:
        print("    ✓ Valid tools/list response")

    # --- 5. tools/call Request ---
    print("\n  5. tools/call Request")
    print("  " + "─" * 55)
    tools_call_req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "calculate_bmi",
            "arguments": {
                "weight_kg": 82.5,
                "height_m": 1.75
            }
        }
    }
    print(f"    {json.dumps(tools_call_req, indent=4)}")
    errors = validate_tools_call_request(tools_call_req)
    if errors:
        print(f"    ✗ INVALID: {errors}")
        all_passed = False
    else:
        print("    ✓ Valid tools/call request")

    # --- 6. tools/call Response (success) ---
    print("\n  6. tools/call Response (success)")
    print("  " + "─" * 55)
    tools_call_resp = {
        "jsonrpc": "2.0",
        "id": 3,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"bmi": 26.9, "category": "Overweight"})
                }
            ]
        }
    }
    print(f"    {json.dumps(tools_call_resp, indent=4)}")
    errors = validate_tool_result(tools_call_resp)
    if errors:
        print(f"    ✗ INVALID: {errors}")
        all_passed = False
    else:
        print("    ✓ Valid tool result")

    # --- 7. Error Response ---
    print("\n  7. Error Response (tool not found)")
    print("  " + "─" * 55)
    error_resp = {
        "jsonrpc": "2.0",
        "id": 4,
        "error": {
            "code": -32601,
            "message": "Method not found",
            "data": {"tool": "nonexistent_tool"}
        }
    }
    print(f"    {json.dumps(error_resp, indent=4)}")
    errors = validate_error_response(error_resp)
    if errors:
        print(f"    ✗ INVALID: {errors}")
        all_passed = False
    else:
        print("    ✓ Valid error response")

    # --- 8. Notification ---
    print("\n  8. Notification (no response expected)")
    print("  " + "─" * 55)
    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }
    print(f"    {json.dumps(notification)}")
    errors = validate_notification(notification)
    if errors:
        print(f"    ✗ INVALID: {errors}")
        all_passed = False
    else:
        print("    ✓ Valid notification")

    # --- 9. resources/read Request ---
    print("\n  9. resources/read Request")
    print("  " + "─" * 55)
    resource_read_req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "resources/read",
        "params": {
            "uri": "clinical://guidelines/hypertension"
        }
    }
    print(f"    {json.dumps(resource_read_req, indent=4)}")
    errors = validate_request(resource_read_req)
    if not resource_read_req.get("params", {}).get("uri"):
        errors.append("resources/read params must include 'uri'")
    if errors:
        print(f"    ✗ INVALID: {errors}")
        all_passed = False
    else:
        print("    ✓ Valid resources/read request")

    # --- 10. resources/read Response ---
    print("\n  10. resources/read Response")
    print("  " + "─" * 55)
    resource_read_resp = {
        "jsonrpc": "2.0",
        "id": 5,
        "result": {
            "contents": [
                {
                    "uri": "clinical://guidelines/hypertension",
                    "mimeType": "text/plain",
                    "text": "Hypertension is defined as systolic BP >= 130 mmHg "
                            "or diastolic BP >= 80 mmHg (ACC/AHA 2017 guidelines)."
                }
            ]
        }
    }
    print(f"    {json.dumps(resource_read_resp, indent=4)[:300]}...")
    errors = validate_response(resource_read_resp)
    result = resource_read_resp.get("result", {})
    if "contents" not in result:
        errors.append("resources/read result must include 'contents'")
    if errors:
        print(f"    ✗ INVALID: {errors}")
        all_passed = False
    else:
        print("    ✓ Valid resources/read response")

    # --- Summary ---
    print("\n" + "=" * 70)
    if all_passed:
        print("  ✓ All 10 message types constructed and validated successfully!")
    else:
        print("  ✗ Some messages had validation errors. Review above.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Bonus: intentionally malformed messages
# ---------------------------------------------------------------------------

def test_malformed_messages():
    """Test the validators catch malformed messages."""
    print("\n  Bonus: Testing validators with MALFORMED messages")
    print("  " + "─" * 55)

    malformed = [
        ("Missing jsonrpc", {"id": 1, "method": "initialize"}),
        ("Missing id on request", {"jsonrpc": "2.0", "method": "tools/call"}),
        ("Notification with id", {"jsonrpc": "2.0", "id": 5, "method": "notifications/progress"}),
        ("Response with both result and error",
         {"jsonrpc": "2.0", "id": 1, "result": {}, "error": {"code": -1, "message": "oops"}}),
        ("Error without code",
         {"jsonrpc": "2.0", "id": 1, "error": {"message": "oops"}}),
    ]

    for label, msg in malformed:
        # Use the most general validator
        if "error" in msg and "result" not in msg:
            errors = validate_error_response(msg)
        elif "result" in msg or "error" in msg:
            errors = validate_response(msg)
        elif "id" not in msg and "method" in msg:
            errors = validate_notification(msg)
        else:
            errors = validate_request(msg)

        if errors:
            print(f"    ✓ Caught: '{label}' → {errors[0]}")
        else:
            print(f"    ✗ Missed: '{label}' was not flagged!")


if __name__ == "__main__":
    build_and_validate_messages()
    test_malformed_messages()
