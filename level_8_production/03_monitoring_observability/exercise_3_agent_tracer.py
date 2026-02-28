"""
Exercise 3: Agent Execution Tracer
====================================
Build a tracer for multi-step agent workflows.

Requirements:
- Record every step in an agent run: LLM calls, tool calls, decisions
- Track timing, token counts, and costs at each step
- Display execution as a tree with cumulative timing
- Support nested traces (sub-agent calls)
- Persist traces for post-hoc analysis

Healthcare Context:
  Clinical agent workflows (e.g., diagnosis support) may involve
  multiple LLM calls, tool lookups, and decision points. Tracing
  each step is essential for debugging and clinical accountability.

Usage:
    python exercise_3_agent_tracer.py
"""

from openai import OpenAI
import time
import json
import os
import uuid
from datetime import datetime

client = OpenAI()

COST_PER_1K = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
}


class TraceSpan:
    """A single span in a trace — represents one operation."""

    def __init__(self, name: str, span_type: str, parent_id: str = None):
        self.span_id = str(uuid.uuid4())[:8]
        self.parent_id = parent_id
        self.name = name
        self.span_type = span_type  # "llm_call", "tool_call", "decision", "sub_agent"
        self.start_time = time.time()
        self.end_time = None
        self.metadata = {}
        self.children = []

    def end(self, **metadata):
        """End this span and record metadata."""
        self.end_time = time.time()
        self.metadata.update(metadata)

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "type": self.span_type,
            "duration_ms": round(self.duration_ms, 2),
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }


class AgentTracer:
    """Traces multi-step agent execution with tree visualization."""

    def __init__(self, agent_name: str = "ClinicalAgent"):
        self.trace_id = str(uuid.uuid4())[:8]
        self.agent_name = agent_name
        self.start_time = datetime.now()
        self.root_spans = []
        self._span_stack = []
        self.total_tokens = 0
        self.total_cost = 0.0

    def start_span(self, name: str, span_type: str) -> TraceSpan:
        """Start a new trace span."""
        parent_id = self._span_stack[-1].span_id if self._span_stack else None
        span = TraceSpan(name, span_type, parent_id)

        if self._span_stack:
            self._span_stack[-1].children.append(span)
        else:
            self.root_spans.append(span)

        self._span_stack.append(span)
        return span

    def end_span(self, **metadata):
        """End the current span."""
        if self._span_stack:
            span = self._span_stack.pop()
            span.end(**metadata)
            return span
        return None

    def traced_llm_call(self, messages: list, model: str = "gpt-4o-mini",
                        span_name: str = "LLM Call") -> dict:
        """Make an LLM call wrapped in a trace span."""
        self.start_span(span_name, "llm_call")

        response = client.chat.completions.create(model=model, messages=messages)
        content = response.choices[0].message.content
        pt = response.usage.prompt_tokens
        ct = response.usage.completion_tokens

        rates = COST_PER_1K.get(model, {"input": 0.005, "output": 0.015})
        cost = (pt / 1000) * rates["input"] + (ct / 1000) * rates["output"]

        self.total_tokens += pt + ct
        self.total_cost += cost

        self.end_span(
            model=model,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
            cost_usd=round(cost, 8),
            response_preview=content[:150],
        )

        return {"content": content, "tokens": pt + ct, "cost": cost}

    def traced_tool_call(self, tool_name: str, tool_input: dict,
                         tool_fn=None) -> dict:
        """Execute a tool call wrapped in a trace span."""
        self.start_span(f"Tool: {tool_name}", "tool_call")

        if tool_fn:
            result = tool_fn(**tool_input)
        else:
            # Simulate tool call
            time.sleep(0.05)
            result = {"status": "simulated", "tool": tool_name, "input": tool_input}

        self.end_span(
            tool=tool_name,
            input=tool_input,
            result_preview=str(result)[:150],
        )

        return result

    def display_trace(self):
        """Display the trace as a visual tree."""
        total_duration = sum(s.duration_ms for s in self.root_spans)

        print("\n" + "=" * 65)
        print(f"  AGENT TRACE: {self.agent_name}")
        print("=" * 65)
        print(f"  Trace ID:    {self.trace_id}")
        print(f"  Started:     {self.start_time.isoformat()}")
        print(f"  Total Time:  {total_duration:.0f}ms")
        print(f"  Total Tokens:{self.total_tokens}")
        print(f"  Total Cost:  ${self.total_cost:.6f}")
        print("=" * 65)

        def print_span(span: TraceSpan, depth: int = 0, is_last: bool = True):
            prefix = "│   " * (depth - 1) if depth > 0 else ""
            connector = "└── " if is_last else "├── "
            if depth > 0:
                prefix += connector

            type_icon = {
                "llm_call": "[LLM]",
                "tool_call": "[TOOL]",
                "decision": "[DECIDE]",
                "sub_agent": "[AGENT]",
            }.get(span.span_type, "[???]")

            print(f"  {prefix}{type_icon} {span.name} ({span.duration_ms:.0f}ms)")

            # Print metadata on next line
            meta_prefix = "│   " * depth + "    " if not is_last else "│   " * (depth - 1) + "    " if depth > 0 else "    "
            if span.span_type == "llm_call" and "total_tokens" in span.metadata:
                print(f"  {meta_prefix}Tokens: {span.metadata['total_tokens']}  Cost: ${span.metadata.get('cost_usd', 0):.6f}")
            elif span.span_type == "tool_call" and "tool" in span.metadata:
                result_preview = span.metadata.get("result_preview", "N/A")[:60]
                print(f"  {meta_prefix}Result: {result_preview}")

            for i, child in enumerate(span.children):
                print_span(child, depth + 1, i == len(span.children) - 1)

        for i, span in enumerate(self.root_spans):
            print_span(span, 0, i == len(self.root_spans) - 1)

        # Timeline bar chart
        print("\n  --- Timing Breakdown ---")
        for span in self.root_spans:
            self._print_timing_bar(span)

    def _print_timing_bar(self, span: TraceSpan, depth: int = 0):
        """Print a timing bar for a span."""
        indent = "    " * depth
        bar_width = max(1, int(span.duration_ms / 50))
        bar = "█" * min(bar_width, 40)
        print(f"  {indent}{span.name[:25]:<25} {bar} {span.duration_ms:.0f}ms")
        for child in span.children:
            self._print_timing_bar(child, depth + 1)

    def save_trace(self, directory: str = "traces"):
        """Save trace to JSON file."""
        os.makedirs(directory, exist_ok=True)
        trace_data = {
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "start_time": self.start_time.isoformat(),
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 8),
            "spans": [s.to_dict() for s in self.root_spans],
        }
        path = os.path.join(directory, f"trace_{self.trace_id}.json")
        with open(path, "w") as f:
            json.dump(trace_data, f, indent=2)
        print(f"\n  Trace saved to: {path}")
        return path


def simulate_drug_lookup(drug_name: str) -> dict:
    """Simulated drug database lookup."""
    time.sleep(0.04)
    drugs = {
        "metformin": {"class": "Biguanide", "use": "Type 2 diabetes", "side_effects": ["GI upset", "Lactic acidosis (rare)"]},
        "lisinopril": {"class": "ACE inhibitor", "use": "Hypertension", "side_effects": ["Cough", "Hyperkalemia"]},
    }
    return drugs.get(drug_name.lower(), {"error": f"Drug '{drug_name}' not found"})


def simulate_lab_lookup(test_name: str) -> dict:
    """Simulated lab reference lookup."""
    time.sleep(0.03)
    labs = {
        "HbA1c": {"normal_range": "4.0-5.6%", "prediabetes": "5.7-6.4%", "diabetes": ">=6.5%"},
        "creatinine": {"normal_range": "0.7-1.3 mg/dL", "elevated": "Suggests kidney impairment"},
    }
    return labs.get(test_name, {"error": f"Lab '{test_name}' not found"})


def main():
    """Run the agent tracer exercise."""
    print("=" * 65)
    print("  Exercise 3: Agent Execution Tracer")
    print("=" * 65)

    tracer = AgentTracer("DiabetesManagementAgent")
    print(f"\nTrace ID: {tracer.trace_id}")
    print("Running clinical agent workflow...\n")

    # Step 1: Analyze patient query
    result1 = tracer.traced_llm_call(
        messages=[
            {"role": "system", "content": (
                "You are a clinical agent. Analyze the query and decide which tools to use. "
                "Available: drug_lookup, lab_reference. "
                "Respond with JSON: {\"tools\": [...], \"reasoning\": \"...\"}"
            )},
            {"role": "user", "content": "Patient on metformin with HbA1c of 7.2%. Creatinine slightly elevated. Assessment?"},
        ],
        span_name="Analyze Query",
    )
    print(f"  Step 1 (Analyze): {result1['tokens']} tokens")

    # Step 2: Tool calls
    drug_result = tracer.traced_tool_call(
        "drug_lookup", {"drug_name": "metformin"}, simulate_drug_lookup
    )
    print(f"  Step 2 (Drug Lookup): {drug_result}")

    lab_result1 = tracer.traced_tool_call(
        "lab_reference", {"test_name": "HbA1c"}, simulate_lab_lookup
    )
    print(f"  Step 3 (Lab - HbA1c): {lab_result1}")

    lab_result2 = tracer.traced_tool_call(
        "lab_reference", {"test_name": "creatinine"}, simulate_lab_lookup
    )
    print(f"  Step 4 (Lab - Creatinine): {lab_result2}")

    # Step 3: Synthesize
    context = json.dumps({"drug": drug_result, "hba1c": lab_result1, "creatinine": lab_result2})
    result2 = tracer.traced_llm_call(
        messages=[
            {"role": "system", "content": "You are a clinical decision support agent. Synthesize tool results into a clinical assessment. Be concise."},
            {"role": "user", "content": f"Patient on metformin, HbA1c 7.2%, elevated creatinine. Tool results: {context}"},
        ],
        span_name="Synthesize Assessment",
    )
    print(f"  Step 5 (Synthesize): {result2['tokens']} tokens")

    # Display the trace tree
    tracer.display_trace()

    # Save trace
    tracer.save_trace()

    # Show final agent response
    print("\n  --- Agent Response ---")
    print(f"  {result2['content'][:500]}")
    print("\nDone!")


if __name__ == "__main__":
    main()
