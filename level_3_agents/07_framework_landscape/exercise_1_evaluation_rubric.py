"""
Exercise 1: The 8-Dimension Evaluation Rubric

Skills practiced:
- Systematically evaluating ANY agent framework
- Scoring across dimensions that matter for production
- Identifying deal-breakers vs nice-to-haves
- Building your own evaluation muscle

Why this matters:
  When a new framework appears, most developers either:
  (a) Ignore it, or (b) Spend a week learning it.

  With this rubric, you can evaluate it in 30 MINUTES by asking
  8 structured questions. You'll know if it's worth your time
  before writing a single line of code.

The 8 Dimensions:
  ┌──────────────────────────────────────────────────────────┐
  │  1. PATTERN      What core pattern does it implement?    │
  │  2. STATE        How is state managed and persisted?     │
  │  3. TOOLS        How do you define and call tools?       │
  │  4. HUMAN LOOP   Can humans intervene at specific steps? │
  │  5. MULTI-AGENT  How do agents collaborate?              │
  │  6. MODEL FREEDOM Can I swap LLM providers freely?      │
  │  7. DEBUGGING    What observability tools exist?         │
  │  8. PRODUCTION   Async? Streaming? Error handling?       │
  └──────────────────────────────────────────────────────────┘
"""

import json


# ============================================================
# THE EVALUATION RUBRIC
# ============================================================

RUBRIC = {
    "pattern": {
        "name": "1. Core Pattern",
        "question": "What core pattern does this framework implement?",
        "what_to_look_for": [
            "ReAct (think-act-observe loop) — most common",
            "Plan-and-execute (plan first, then run steps)",
            "Multi-agent conversation (agents chat with each other)",
            "Graph/workflow (nodes connected by edges)",
            "Event-driven (handlers triggered by events)",
        ],
        "scoring": {
            5: "Supports multiple patterns (ReAct + plan-and-execute + custom)",
            3: "Supports one pattern well, can adapt to others with effort",
            1: "Only one rigid pattern, hard to customize",
        },
        "why_it_matters": (
            "If you understand the pattern, you understand 80% of the framework. "
            "A ReAct agent in LangGraph looks different syntactically from one in "
            "CrewAI, but the LOGIC is identical."
        ),
    },
    "state": {
        "name": "2. State Management",
        "question": "How does the framework manage and persist state?",
        "what_to_look_for": [
            "What IS the state? (messages list? custom dict? typed object?)",
            "Where does state live? (memory? database? file?)",
            "Can I checkpoint and resume?",
            "Can I roll back (time travel)?",
            "Is state shared across threads/sessions?",
        ],
        "scoring": {
            5: "Typed state, native persistence, checkpointing, time-travel, cross-thread store",
            3: "Basic state management, some persistence, no time-travel",
            1: "State is just messages, no persistence, starts fresh each run",
        },
        "why_it_matters": (
            "In healthcare, you MUST persist state (patient context across sessions). "
            "A framework without solid persistence means you build it yourself."
        ),
    },
    "tools": {
        "name": "3. Tool Calling",
        "question": "How do you define and call tools?",
        "what_to_look_for": [
            "Decorator-based? (@tool, @function)",
            "Schema-based? (JSON schema / Pydantic model)",
            "Auto-discovered from docstrings?",
            "Parallel tool calling supported?",
            "Can tools return structured data?",
        ],
        "scoring": {
            5: "Simple decorator, auto-schema, parallel calling, structured returns",
            3: "Decorator or function-based, basic schema, sequential calling",
            1: "Manual schema definition, limited tool integration",
        },
        "why_it_matters": (
            "Tools are how agents interact with the real world. "
            "If defining tools is painful, everything is painful."
        ),
    },
    "human_loop": {
        "name": "4. Human-in-the-Loop",
        "question": "Can humans intervene at specific steps?",
        "what_to_look_for": [
            "Can you pause at a specific node/step?",
            "Can humans approve/reject/modify before continuing?",
            "Can you resume after human input?",
            "Is the intervention point configurable (not just before all tools)?",
            "Does it persist state while waiting for human?",
        ],
        "scoring": {
            5: "Configurable breakpoints, approval flows, persisted state during wait",
            3: "Basic human input flag, can pause but limited control over where",
            1: "No built-in support, must implement manually",
        },
        "why_it_matters": (
            "Healthcare REQUIRES human oversight. A framework without human-in-loop "
            "is a non-starter for clinical applications."
        ),
    },
    "multi_agent": {
        "name": "5. Multi-Agent",
        "question": "How do agents collaborate?",
        "what_to_look_for": [
            "Can agents hand off to each other?",
            "Can agents run in parallel? (fan-out)",
            "Is there a supervisor/orchestrator pattern?",
            "Can agents share context/memory?",
            "How are conflicts resolved?",
        ],
        "scoring": {
            5: "Flexible multi-agent: handoffs, parallel, hierarchical, shared memory",
            3: "Basic multi-agent support (sequential or simple handoff)",
            1: "Single agent only, or multi-agent is an afterthought",
        },
        "why_it_matters": (
            "Complex tasks need multiple agents (triage → specialist → pharmacist). "
            "Multi-agent support determines scalability of your solution."
        ),
    },
    "model_freedom": {
        "name": "6. Model Freedom",
        "question": "Can I swap LLM providers freely?",
        "what_to_look_for": [
            "OpenAI, Anthropic, Google, local models?",
            "Is the model abstracted behind an interface?",
            "Can I use different models for different agents?",
            "Does it work with open-source models (Llama, Mistral)?",
            "Is there vendor lock-in?",
        ],
        "scoring": {
            5: "Works with any LLM, clean abstraction, easy model swapping",
            3: "Supports multiple but best experience with one provider",
            1: "Locked to one provider (e.g., OpenAI only)",
        },
        "why_it_matters": (
            "Model lock-in is a business risk. If your framework only works with "
            "OpenAI, you can't switch when a better/cheaper model appears. "
            "Healthcare orgs may also need on-premise models for PHI."
        ),
    },
    "debugging": {
        "name": "7. Debugging & Observability",
        "question": "What debugging and tracing tools exist?",
        "what_to_look_for": [
            "Can you trace each step (which node ran, what it produced)?",
            "Is there a visual graph/workflow viewer?",
            "Can you inspect intermediate state?",
            "Integration with observability (LangSmith, Arize, etc.)?",
            "Meaningful error messages?",
        ],
        "scoring": {
            5: "Built-in tracing, visual graph, state inspection, observability integration",
            3: "Basic logging, some tracing, no visual tools",
            1: "Black box — hard to see what went wrong",
        },
        "why_it_matters": (
            "When your agent gives a wrong answer, you need to know WHY. "
            "In healthcare, you may need to EXPLAIN the reasoning chain "
            "for audit purposes."
        ),
    },
    "production": {
        "name": "8. Production Readiness",
        "question": "Is this framework production-ready?",
        "what_to_look_for": [
            "Async support (concurrent requests)?",
            "Streaming (token-by-token output)?",
            "Error handling and retry logic?",
            "Rate limiting and cost controls?",
            "Deployment guides (Docker, cloud, serverless)?",
            "Battle-tested by real applications?",
        ],
        "scoring": {
            5: "Async, streaming, error handling, deployed in production by many companies",
            3: "Some production features, but gaps remain",
            1: "Research/prototype quality, not production-tested",
        },
        "why_it_matters": (
            "A framework that works in a notebook may crumble under real load. "
            "Healthcare systems must be reliable — production readiness is not optional."
        ),
    },
}


# ============================================================
# DEMO 1: Explore the Rubric
# ============================================================

def demo_rubric():
    """Walk through all 8 dimensions of the evaluation rubric."""
    print("\n" + "=" * 70)
    print("  THE 8-DIMENSION EVALUATION RUBRIC")
    print("=" * 70)
    print("""
  Use this rubric to evaluate ANY agent framework in 30 minutes.
  Each dimension scores 1-5. Total possible: 40 points.
  """)

    for key, dim in RUBRIC.items():
        print(f"\n  ━━━ {dim['name']} ━━━")
        print(f"  Question: {dim['question']}")
        print(f"\n  What to look for:")
        for item in dim["what_to_look_for"]:
            print(f"    • {item}")
        print(f"\n  Scoring:")
        for score, desc in sorted(dim["scoring"].items(), reverse=True):
            print(f"    {score}/5 — {desc}")
        print(f"\n  Why it matters: {dim['why_it_matters'][:120]}...")

    print(f"\n  {'=' * 60}")
    print(f"  With these 8 questions, you can evaluate ANY framework.")
    print(f"  Try it: next time you see a new tool, score it 1-5 on each.")


# ============================================================
# DEMO 2: Score Existing Frameworks
# ============================================================

def demo_scored_frameworks():
    """Show pre-scored evaluations of major frameworks."""
    print("\n" + "=" * 70)
    print("  PRE-SCORED FRAMEWORK EVALUATIONS")
    print("=" * 70)

    scored = {
        "LangGraph": {
            "pattern": 5, "state": 5, "tools": 5, "human_loop": 5,
            "multi_agent": 5, "model_freedom": 5, "debugging": 5, "production": 5,
        },
        "OpenAI SDK": {
            "pattern": 3, "state": 2, "tools": 5, "human_loop": 3,
            "multi_agent": 4, "model_freedom": 1, "debugging": 4, "production": 4,
        },
        "CrewAI": {
            "pattern": 3, "state": 3, "tools": 4, "human_loop": 3,
            "multi_agent": 5, "model_freedom": 5, "debugging": 2, "production": 3,
        },
        "AutoGen": {
            "pattern": 4, "state": 3, "tools": 4, "human_loop": 4,
            "multi_agent": 5, "model_freedom": 5, "debugging": 3, "production": 3,
        },
        "Pydantic AI": {
            "pattern": 2, "state": 2, "tools": 4, "human_loop": 1,
            "multi_agent": 1, "model_freedom": 5, "debugging": 4, "production": 4,
        },
        "LlamaIndex": {
            "pattern": 3, "state": 3, "tools": 4, "human_loop": 3,
            "multi_agent": 3, "model_freedom": 5, "debugging": 3, "production": 3,
        },
        "Google ADK": {
            "pattern": 3, "state": 4, "tools": 4, "human_loop": 3,
            "multi_agent": 4, "model_freedom": 3, "debugging": 3, "production": 3,
        },
    }

    dims = ["pattern", "state", "tools", "human_loop", "multi_agent",
            "model_freedom", "debugging", "production"]
    dim_labels = ["Pattern", "State", "Tools", "HiL", "Multi", "Model", "Debug", "Prod"]

    # Header
    print(f"\n  {'Framework':<15s}" + "".join(f"{l:<7s}" for l in dim_labels) + "TOTAL")
    print(f"  {'─' * 13}  " + "".join(f"{'─' * 5}  " for _ in dim_labels) + "─────")

    for name, scores in scored.items():
        total = sum(scores.values())
        row = f"  {name:<15s}" + "".join(f"{scores[d]:<7d}" for d in dims) + f"{total}/40"
        print(row)

    print(f"""
  INTERPRETATION:
  35-40: Excellent all-around (LangGraph)
  28-34: Strong with trade-offs (OpenAI SDK, CrewAI, AutoGen)
  20-27: Specialized — great at some things, weak at others
  <20:   Limited — use only for specific niche tasks

  NOTE: These scores reflect general-purpose use.
  For YOUR specific use case, some dimensions matter more than others.
  A healthcare system weights human_loop and state heavily.
  A quick prototype weights ease-of-start over production readiness.
  """)


# ============================================================
# DEMO 3: Evaluate a New Framework (Interactive)
# ============================================================

def demo_evaluate_new():
    """Interactive: evaluate a framework yourself using the rubric."""
    print("\n" + "=" * 70)
    print("  EVALUATE A NEW FRAMEWORK (Interactive)")
    print("=" * 70)
    print("""
  Practice evaluating a framework using the 8-dimension rubric.
  Score each dimension 1-5 based on what you know or discovered.

  Think of a framework you've seen (blog post, tweet, HN).
  Or try: 'SuperAgentX' as a hypothetical exercise.
  """)

    name = input("  Framework name: ").strip() or "Unknown"
    print(f"\n  Evaluating: {name}")
    print(f"  Score each dimension 1-5 (or press Enter by default for 3)\n")

    dims = RUBRIC.items()
    scores = {}
    for key, dim in dims:
        try:
            score = int(input(f"  {dim['name']} (1-5): ").strip() or "3")
            score = max(1, min(5, score))
        except ValueError:
            score = 3
        scores[key] = score

    total = sum(scores.values())
    print(f"\n  ─── Results for {name} ───")
    for key, score in scores.items():
        bar = "█" * score + "░" * (5 - score)
        print(f"  {RUBRIC[key]['name']:<30s} [{bar}] {score}/5")
    print(f"  {'TOTAL':<30s}              {total}/40")

    if total >= 35:
        print(f"\n  Verdict: 🏆 Strong framework. Worth learning deeply.")
    elif total >= 28:
        print(f"\n  Verdict: 👍 Good framework. Evaluate for your specific needs.")
    elif total >= 20:
        print(f"\n  Verdict: ⚠️  Specialized. Good for niche use, not general-purpose.")
    else:
        print(f"\n  Verdict: 🔍 Limited. Use only if it solves one specific problem exceptionally.")

    # Healthcare filter
    healthcare_score = scores.get("human_loop", 0) + scores.get("state", 0) + scores.get("production", 0)
    if healthcare_score >= 12:
        print(f"  Healthcare fit: ✅ Good (HiL + State + Prod = {healthcare_score}/15)")
    elif healthcare_score >= 8:
        print(f"  Healthcare fit: ⚠️  Possible with work (HiL + State + Prod = {healthcare_score}/15)")
    else:
        print(f"  Healthcare fit: ❌ Not suitable (HiL + State + Prod = {healthcare_score}/15)")


# ============================================================
# DEMO 4: Red Flags — When to AVOID a Framework
# ============================================================

def demo_red_flags():
    """Common red flags when evaluating agent frameworks."""
    print("\n" + "=" * 70)
    print("  RED FLAGS — WHEN TO AVOID A FRAMEWORK")
    print("=" * 70)
    print("""
  Not every shiny new framework is worth your time.
  Here are red flags that should make you cautious:

  🚩 RED FLAG 1: "Magic" with no visibility
     If you can't trace what the agent did step-by-step,
     you can't debug it, explain it, or trust it.
     Ask: "Can I see every step the agent took?"

  🚩 RED FLAG 2: Model lock-in
     If it ONLY works with one provider, you're dependent.
     OpenAI raises prices? Model goes down? You're stuck.
     Ask: "Can I swap to Claude or a local model?"

  🚩 RED FLAG 3: No persistence story
     If the framework starts fresh every run, it can't handle
     real applications where conversations span sessions.
     Ask: "How do I save and resume state?"

  🚩 RED FLAG 4: "It just works" marketing
     If the docs show only happy-path demos, what happens when
     a tool call fails? When the LLM hallucinates a tool name?
     Ask: "Show me the error handling documentation."

  🚩 RED FLAG 5: Breaking API changes
     If the API changed 3 times in 6 months, your code will rot.
     Ask: "When was the last breaking change? Is there a stability policy?"

  🚩 RED FLAG 6: Single-contributor project
     If one person maintains it, what happens when they move on?
     Ask: "How many active contributors? Is there an org behind it?"

  🚩 RED FLAG 7: No production users
     Stars on GitHub ≠ production usage.
     Ask: "Who is using this in production? What scale?"

  🚩 RED FLAG 8: Reinventing everything
     If it reimplements tool calling, prompt formatting, streaming,
     etc. from scratch instead of building on standards...
     Ask: "Does it follow OpenAI function calling format or its own?"

  ─── THE LITMUS TEST ───

  Can you answer these 3 questions about the framework?

  1. What happens when the LLM hallucinates a tool call?
  2. How do I resume a conversation after a server restart?
  3. How do I trace WHY the agent gave a wrong answer?

  If the docs don't answer these, proceed with extreme caution.
  """)


# ============================================================
# DEMO 5: Build Your Evaluation Checklist
# ============================================================

def demo_checklist():
    """Generate a personalized evaluation checklist."""
    print("\n" + "=" * 70)
    print("  YOUR PERSONALIZED EVALUATION CHECKLIST")
    print("=" * 70)
    print("""
  Not all dimensions matter equally for YOUR work.
  Let's build a weighted checklist based on your priorities.
  """)

    print("  Rate how important each dimension is for YOUR work (1-5):\n")

    dimensions = [
        ("pattern", "Core Pattern (flexibility of workflows)"),
        ("state", "State Management (persistence, checkpointing)"),
        ("tools", "Tool Calling (ease of defining tools)"),
        ("human_loop", "Human-in-Loop (intervention capability)"),
        ("multi_agent", "Multi-Agent (agent collaboration)"),
        ("model_freedom", "Model Freedom (no vendor lock-in)"),
        ("debugging", "Debugging (tracing, observability)"),
        ("production", "Production (async, streaming, errors)"),
    ]

    weights = {}
    for key, desc in dimensions:
        try:
            w = int(input(f"  {desc}: ").strip() or "3")
            weights[key] = max(1, min(5, w))
        except ValueError:
            weights[key] = 3

    print(f"\n  ─── Your Weighted Checklist ───")
    print(f"  (Use this every time you evaluate a new framework)\n")

    # Sort by weight (most important first)
    sorted_dims = sorted(weights.items(), key=lambda x: x[-1], reverse=True)

    print(f"  Priority  Dimension            Weight  Your Question")
    print(f"  ────────  ─────────            ──────  ─────────────")

    priority_questions = {
        "pattern": "What pattern? Can I customize the workflow?",
        "state": "How is state saved? Can I checkpoint/resume?",
        "tools": "How do I define tools? Parallel tool calls?",
        "human_loop": "Where can humans intervene? Is it configurable?",
        "multi_agent": "How do agents collaborate? Fan-out?",
        "model_freedom": "Which LLM providers? Can I swap freely?",
        "debugging": "Can I trace step-by-step? Visual tools?",
        "production": "Async? Streaming? Error recovery?",
    }

    for i, (key, weight) in enumerate(sorted_dims, 1):
        bar = "█" * weight + "░" * (5 - weight)
        label = [d for k, d in dimensions if k == key][0].split(" (")[0]
        question = priority_questions[key]
        criticality = "CRITICAL" if weight >= 4 else "Important" if weight >= 3 else "Nice-to-have"
        print(f"  {criticality:<10s} {label:<20s} [{bar}]  {question}")

    print(f"\n  Use this checklist when evaluating any new framework.")
    print(f"  Focus on your CRITICAL dimensions first. If those score <3, move on.")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 1: THE 8-DIMENSION EVALUATION RUBRIC")
    print("=" * 70)
    print("""
    A systematic way to evaluate ANY agent framework.

    Choose:
      1 → Explore the rubric (all 8 dimensions explained)
      2 → Pre-scored frameworks (LangGraph vs OpenAI vs CrewAI...)
      3 → Evaluate a new framework yourself (interactive)
      4 → Red flags to watch for
      5 → Build your personalized checklist
      6 → Run all non-interactive (rubric + scores + red flags)
    """)

    choice = input("  Enter choice (1-6): ").strip()

    demos = {
        "1": demo_rubric,
        "2": demo_scored_frameworks,
        "3": demo_evaluate_new,
        "4": demo_red_flags,
        "5": demo_checklist,
    }

    if choice == "6":
        demo_rubric()
        demo_scored_frameworks()
        demo_red_flags()
    elif choice in demos:
        demos[choice]()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. THE 8 DIMENSIONS cover everything that matters:
   Pattern, State, Tools, Human-Loop, Multi-Agent,
   Model Freedom, Debugging, Production Readiness.

2. SCORE 1-5 on each dimension to compare frameworks objectively.
   Total 35+: worth learning deeply. <20: niche only.

3. YOUR PRIORITIES determine which dimensions matter most.
   Healthcare: Human-Loop + State + Production are non-negotiable.
   Prototyping: Pattern + Tools + Ease-of-start matter more.

4. RED FLAGS to watch for: magic black boxes, model lock-in,
   no persistence, no error handling docs, frequent breaking changes.

5. THE 30-MINUTE EVALUATION:
   - Read the "Getting Started" page (5 min)
   - Find their tool calling example (5 min)
   - Search for "persistence" or "checkpoint" in docs (5 min)
   - Search for "human" or "approval" in docs (5 min)
   - Check GitHub: stars, contributors, last commit, issues (5 min)
   - Score on your 8 dimensions (5 min)
   → Now you know if it's worth a deeper look.
"""

if __name__ == "__main__":
    main()
