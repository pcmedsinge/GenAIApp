"""
Module 07: Agent Framework Landscape — Overview

This is NOT a coding exercise. It's a KNOWLEDGE exercise.
It teaches you how to THINK about agent frameworks so you can
evaluate any new one in 30 minutes.

Why this matters:
  New agent frameworks appear monthly. You'll see blog posts saying
  "X is the future" and "Y kills LangChain." 99% of the time, they
  implement the SAME patterns with different syntax.

  Your job is NOT to learn every framework.
  Your job IS to recognize: "Oh, this is just ReAct with a different API."

The Agent Framework Landscape (as of early 2026):

  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  TIER 1: Major Frameworks (production-proven, large community)   │
  │  ─────────────────────────────────────────────────────────────── │
  │  LangGraph ............ Graph-based workflows, most flexible     │
  │  OpenAI Agents SDK .... Provider-native, tightly integrated      │
  │  CrewAI ............... Role-based multi-agent, easiest start    │
  │  AutoGen .............. Microsoft, multi-agent conversations     │
  │  Semantic Kernel ...... Microsoft, enterprise .NET/Python        │
  │                                                                  │
  │  TIER 2: Specialized Frameworks (strong in specific areas)       │
  │  ─────────────────────────────────────────────────────────────── │
  │  Google ADK ........... Google Cloud native agents               │
  │  Amazon Bedrock Agents  AWS-native managed agents                │
  │  LlamaIndex Workflows . RAG-first, document-heavy use cases     │
  │  Haystack ............. RAG pipelines, search-focused            │
  │  Pydantic AI .......... Type-safe, minimal, structured output    │
  │                                                                  │
  │  TIER 3: Emerging / Niche (watch but don't invest yet)           │
  │  ─────────────────────────────────────────────────────────────── │
  │  DSPy ................. Prompt optimization / compilation        │
  │  ControlFlow .......... Prefect team, workflow-native agents     │
  │  Julep ................ Stateful long-running agents             │
  │  Letta (MemGPT) ....... Long-term memory agents                 │
  │  Instructor ........... Structured extraction, minimal           │
  │  Marvin ............... Lightweight AI functions                  │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘

Run this file to explore the landscape interactively.
"""

import json


# ============================================================
# FRAMEWORK DATABASE
# ============================================================

FRAMEWORKS = {
    "langgraph": {
        "name": "LangGraph",
        "by": "LangChain (Harrison Chase)",
        "tier": 1,
        "first_release": "2024",
        "language": "Python, JavaScript/TypeScript",
        "model_lock": "None — works with any LLM via LangChain",
        "core_concept": "Stateful graphs — nodes are functions, edges are transitions",
        "strengths": [
            "Most flexible: any workflow can be expressed as a graph",
            "Fine-grained control over state, routing, persistence",
            "Native checkpointing, human-in-loop, streaming",
            "Subgraphs for modularity, Send() for dynamic parallelism",
            "Large ecosystem (LangChain, LangSmith, LangServe)",
        ],
        "weaknesses": [
            "Steeper learning curve than simpler frameworks",
            "Verbose for simple use cases (create_react_agent helps)",
            "LangChain dependency can feel heavy",
            "Graph visualization requires extra setup",
        ],
        "best_for": "Complex, custom workflows. Production healthcare systems. When you need full control.",
        "pattern_mapping": {
            "ReAct": "create_react_agent() or StateGraph + ToolNode",
            "Multi-agent": "Send(), subgraphs, Command()",
            "State": "TypedDict with Annotated reducers",
            "Tools": "@tool decorator + ToolNode",
            "Persistence": "MemorySaver, SqliteSaver, PostgresSaver",
            "Human-in-loop": "interrupt_before, interrupt_after",
        },
        "hello_world": """
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
agent = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools=[...])
result = agent.invoke({"messages": [("user", "Hello")]})
""",
    },

    "openai_agents_sdk": {
        "name": "OpenAI Agents SDK",
        "by": "OpenAI",
        "tier": 1,
        "first_release": "March 2025 (evolved from Swarm)",
        "language": "Python",
        "model_lock": "OpenAI models only (GPT-4o, o1, o3, etc.)",
        "core_concept": "Agent + Handoffs — agents delegate to other agents",
        "strengths": [
            "Simplest API: Agent() + Runner.run()",
            "Native OpenAI integration (fastest inference)",
            "Built-in tracing and guardrails",
            "Handoff pattern for multi-agent is intuitive",
            "Context variables for type-safe shared state",
        ],
        "weaknesses": [
            "Locked to OpenAI models — no Claude, Gemini, local LLMs",
            "Less flexible than graph-based approaches",
            "Newer, smaller ecosystem than LangChain",
            "Limited persistence options (no native checkpointing)",
            "No equivalent of Send() for dynamic fan-out",
        ],
        "best_for": "OpenAI-only shops. Quick prototypes. When simplicity > flexibility.",
        "pattern_mapping": {
            "ReAct": "Agent(tools=[...]) + Runner.run()",
            "Multi-agent": "Handoffs between Agent objects",
            "State": "RunContext + context variables",
            "Tools": "Python functions with docstrings",
            "Persistence": "Manual (no built-in checkpointing)",
            "Human-in-loop": "Guardrails (input/output validation)",
        },
        "hello_world": """
from agents import Agent, Runner
agent = Agent(name="Helper", instructions="Be helpful.", tools=[...])
result = Runner.run_sync(agent, "Hello")
print(result.final_output)
""",
    },

    "crewai": {
        "name": "CrewAI",
        "by": "CrewAI Inc (João Moura)",
        "tier": 1,
        "first_release": "2024",
        "language": "Python",
        "model_lock": "None — supports OpenAI, Anthropic, local via LiteLLM",
        "core_concept": "Crew of Agents with Roles, Goals, and Tasks",
        "strengths": [
            "Most intuitive for non-engineers (role-based thinking)",
            "Built-in delegation and collaboration between agents",
            "Process types: sequential, hierarchical, consensual",
            "Memory system (short-term, long-term, entity)",
            "CrewAI+ marketplace for pre-built crews",
        ],
        "weaknesses": [
            "Less control over individual steps than LangGraph",
            "Hard to debug when things go wrong (black-box feel)",
            "Performance overhead from role-playing prompts",
            "Limited custom state management",
            "Harder to add surgical human-in-loop breakpoints",
        ],
        "best_for": "Multi-agent role-play scenarios. Quick prototyping. Business users defining agents.",
        "pattern_mapping": {
            "ReAct": "Agent(role=..., goal=..., tools=[...])",
            "Multi-agent": "Crew(agents=[...], tasks=[...], process=...)",
            "State": "Shared memory between agents",
            "Tools": "@tool decorator (LangChain compatible)",
            "Persistence": "Built-in memory types (short/long/entity)",
            "Human-in-loop": "human_input=True on Task",
        },
        "hello_world": """
from crewai import Agent, Task, Crew
researcher = Agent(role="Researcher", goal="Find information", tools=[...])
task = Task(description="Research AI agents", agent=researcher)
crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
""",
    },

    "autogen": {
        "name": "AutoGen",
        "by": "Microsoft Research",
        "tier": 1,
        "first_release": "2023 (v0.2), major rewrite 2025 (v0.4+)",
        "language": "Python, .NET",
        "model_lock": "None — supports multiple providers",
        "core_concept": "Conversable agents that chat with each other",
        "strengths": [
            "Strong research backing (Microsoft Research)",
            "Multi-agent conversation patterns well-developed",
            "Code execution in sandboxed environments",
            "GroupChat for N-agent discussions",
            "Good for code generation + execution loops",
        ],
        "weaknesses": [
            "Major API break between v0.2 and v0.4 (confusing docs)",
            "Less intuitive graph model than LangGraph",
            "Conversation-centric (harder for non-chat workflows)",
            "Heavier setup than simpler frameworks",
            "Community split between old and new API",
        ],
        "best_for": "Multi-agent conversations. Code generation + execution. Research prototyping.",
        "pattern_mapping": {
            "ReAct": "AssistantAgent with tool registration",
            "Multi-agent": "GroupChat + GroupChatManager",
            "State": "Chat messages between agents",
            "Tools": "register_for_llm() / register_for_execution()",
            "Persistence": "Cache system + custom stores",
            "Human-in-loop": "human_input_mode='ALWAYS'/'TERMINATE'",
        },
        "hello_world": """
from autogen import AssistantAgent, UserProxyAgent
assistant = AssistantAgent("helper", llm_config={...})
user = UserProxyAgent("user", human_input_mode="NEVER")
user.initiate_chat(assistant, message="Hello")
""",
    },

    "semantic_kernel": {
        "name": "Semantic Kernel",
        "by": "Microsoft",
        "tier": 1,
        "first_release": "2023",
        "language": "Python, C#, Java",
        "model_lock": "None — connectors for OpenAI, Azure, Hugging Face, etc.",
        "core_concept": "Kernel with Plugins (semantic + native functions)",
        "strengths": [
            "Enterprise-grade (.NET first-class, Java support)",
            "Strong typing and plugin architecture",
            "Process framework for complex workflows",
            "Azure AI integration (enterprise compliance)",
            "Planners (Auto, Stepwise, Handlebars)",
        ],
        "weaknesses": [
            "Python SDK feels secondary to C#",
            "More complex setup than alternatives",
            "Documentation can be confusing (multiple paradigms)",
            "Smaller Python community than LangChain",
            "Heavier abstraction layer",
        ],
        "best_for": "Enterprise .NET shops. Azure-native deployments. When you need Java/C# support.",
        "pattern_mapping": {
            "ReAct": "Kernel + ChatCompletionAgent + plugins",
            "Multi-agent": "AgentGroupChat, agent channels",
            "State": "KernelArguments + ChatHistory",
            "Tools": "kernel_function decorator + plugins",
            "Persistence": "Custom via connectors",
            "Human-in-loop": "Filters (function invocation filter)",
        },
        "hello_world": """
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(model_id="gpt-4o-mini"))
result = await kernel.invoke_prompt("Hello, {{$input}}", input="world")
""",
    },

    "google_adk": {
        "name": "Google Agent Development Kit (ADK)",
        "by": "Google",
        "tier": 2,
        "first_release": "2025",
        "language": "Python",
        "model_lock": "Primarily Google (Gemini), but supports others",
        "core_concept": "Agent with tools, multi-agent orchestration",
        "strengths": [
            "Native Gemini integration (vision, long context)",
            "Built-in multi-agent support",
            "Vertex AI deployment path",
            "Session management and memory",
            "A2A (Agent-to-Agent) protocol support",
        ],
        "weaknesses": [
            "Newer, less battle-tested",
            "Best experience requires Google Cloud",
            "Smaller community than LangChain/CrewAI",
            "Documentation still maturing",
        ],
        "best_for": "Google Cloud shops. Gemini-heavy applications. When you want Vertex AI deployment.",
        "pattern_mapping": {
            "ReAct": "Agent(model=..., tools=[...])",
            "Multi-agent": "Agent with sub_agents=[...]",
            "State": "Session state + memory",
            "Tools": "FunctionTool, Google Search, Code Execution",
            "Persistence": "Session service (in-memory, database)",
            "Human-in-loop": "Callbacks and approval tools",
        },
        "hello_world": """
from google.adk.agents import Agent
from google.adk.runners import Runner
agent = Agent(name="helper", model="gemini-2.0-flash", tools=[...])
runner = Runner(agent=agent, app_name="my_app")
result = runner.run(user_id="user1", session_id="s1", new_message=...)
""",
    },

    "llamaindex_workflows": {
        "name": "LlamaIndex Workflows",
        "by": "LlamaIndex (Jerry Liu)",
        "tier": 2,
        "first_release": "2024",
        "language": "Python, TypeScript",
        "model_lock": "None — supports multiple providers",
        "core_concept": "Event-driven workflow steps (async-first)",
        "strengths": [
            "Best-in-class for RAG (that's where LlamaIndex started)",
            "Event-driven architecture (vs graph-based)",
            "Strong document processing and indexing",
            "Async-first design",
            "LlamaHub for pre-built data connectors",
        ],
        "weaknesses": [
            "Workflow system is newer than the RAG components",
            "Event-driven model less intuitive than graphs for some",
            "Smaller agent community than LangChain",
            "Better for data-heavy tasks than pure agent tasks",
        ],
        "best_for": "RAG-heavy applications. Document processing pipelines. When data retrieval is primary.",
        "pattern_mapping": {
            "ReAct": "ReActAgent or FunctionCallingAgent",
            "Multi-agent": "AgentWorkflow with handoffs",
            "State": "Workflow Context (key-value)",
            "Tools": "FunctionTool(fn=...)",
            "Persistence": "Index stores, docstores, vector stores",
            "Human-in-loop": "InputRequiredEvent (pause + resume)",
        },
        "hello_world": """
from llama_index.core.agent.workflow import AgentWorkflow
agent = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=[...], llm=llm, system_prompt="..."
)
result = await agent.run(user_msg="Hello")
""",
    },

    "pydantic_ai": {
        "name": "Pydantic AI",
        "by": "Pydantic team (Samuel Colvin)",
        "tier": 2,
        "first_release": "2024",
        "language": "Python",
        "model_lock": "None — supports OpenAI, Anthropic, Gemini, Groq, etc.",
        "core_concept": "Type-safe agents with structured outputs (Pydantic models)",
        "strengths": [
            "Excellent structured output (Pydantic models as results)",
            "Minimal, clean API — very Pythonic",
            "Type-safe dependency injection",
            "Built on Pydantic (you probably already use it)",
            "Great for extraction and structured tasks",
        ],
        "weaknesses": [
            "Not designed for complex multi-agent workflows",
            "No built-in graph or workflow engine",
            "Smaller ecosystem than LangChain",
            "Better for structured tasks than open-ended agents",
        ],
        "best_for": "Structured extraction. Type-safe outputs. When Pydantic models are your target.",
        "pattern_mapping": {
            "ReAct": "Agent() with @agent.tool decorator",
            "Multi-agent": "Manual orchestration (not built-in)",
            "State": "Dependencies (type-safe context injection)",
            "Tools": "@agent.tool / @agent.tool_plain decorators",
            "Persistence": "Manual (no built-in persistence)",
            "Human-in-loop": "Manual (pause/resume not built-in)",
        },
        "hello_world": """
from pydantic_ai import Agent
agent = Agent('openai:gpt-4o-mini', system_prompt='Be helpful.')
result = agent.run_sync('Hello')
print(result.data)
""",
    },
}


# ============================================================
# DEMO 1: Framework Overview
# ============================================================

def demo_landscape_overview():
    """Display the full framework landscape."""
    print("\n" + "=" * 70)
    print("  THE AGENT FRAMEWORK LANDSCAPE (Early 2026)")
    print("=" * 70)

    for tier in [1, 2, 3]:
        tier_names = {1: "MAJOR FRAMEWORKS", 2: "SPECIALIZED FRAMEWORKS", 3: "EMERGING / NICHE"}
        tier_fws = {k: v for k, v in FRAMEWORKS.items() if v["tier"] == tier}
        if not tier_fws:
            continue

        print(f"\n  ━━━ TIER {tier}: {tier_names[tier]} ━━━")
        for key, fw in tier_fws.items():
            print(f"\n  {fw['name']} (by {fw['by']})")
            print(f"    Core concept: {fw['core_concept']}")
            print(f"    Model lock: {fw['model_lock']}")
            print(f"    Best for: {fw['best_for']}")
            print(f"    Language: {fw['language']}")


# ============================================================
# DEMO 2: Deep Dive on a Framework
# ============================================================

def demo_deep_dive():
    """Deep dive into any framework."""
    print("\n" + "=" * 70)
    print("  DEEP DIVE — EXPLORE ANY FRAMEWORK")
    print("=" * 70)

    print("\n  Available frameworks:")
    for i, (key, fw) in enumerate(FRAMEWORKS.items(), 1):
        print(f"    {i}. {fw['name']} (Tier {fw['tier']})")

    choice = input("\n  Pick a number: ").strip()
    keys = list(FRAMEWORKS.keys())
    try:
        fw = FRAMEWORKS[keys[int(choice) - 1]]
    except (IndexError, ValueError):
        print("  Invalid choice.")
        return

    print(f"\n  {'=' * 60}")
    print(f"  {fw['name']}")
    print(f"  {'=' * 60}")
    print(f"  By: {fw['by']}")
    print(f"  First release: {fw['first_release']}")
    print(f"  Language: {fw['language']}")
    print(f"  Model lock: {fw['model_lock']}")
    print(f"\n  Core concept: {fw['core_concept']}")
    print(f"  Best for: {fw['best_for']}")

    print(f"\n  STRENGTHS:")
    for s in fw["strengths"]:
        print(f"    ✅ {s}")

    print(f"\n  WEAKNESSES:")
    for w in fw["weaknesses"]:
        print(f"    ⚠️  {w}")

    print(f"\n  PATTERN MAPPING (what you know → their equivalent):")
    for pattern, impl in fw["pattern_mapping"].items():
        print(f"    {pattern:20s} → {impl}")

    print(f"\n  HELLO WORLD:{fw['hello_world']}")


# ============================================================
# DEMO 3: Side-by-Side Comparison
# ============================================================

def demo_comparison():
    """Compare frameworks across dimensions."""
    print("\n" + "=" * 70)
    print("  SIDE-BY-SIDE COMPARISON")
    print("=" * 70)

    dimensions = [
        ("Flexibility",    {"langgraph": "★★★★★", "openai_agents_sdk": "★★★☆☆", "crewai": "★★★☆☆", "autogen": "★★★★☆", "pydantic_ai": "★★☆☆☆"}),
        ("Ease of start",  {"langgraph": "★★★☆☆", "openai_agents_sdk": "★★★★★", "crewai": "★★★★★", "autogen": "★★★☆☆", "pydantic_ai": "★★★★★"}),
        ("Multi-agent",    {"langgraph": "★★★★★", "openai_agents_sdk": "★★★★☆", "crewai": "★★★★★", "autogen": "★★★★★", "pydantic_ai": "★☆☆☆☆"}),
        ("Production-ready",{"langgraph": "★★★★★", "openai_agents_sdk": "★★★★☆", "crewai": "★★★☆☆", "autogen": "★★★☆☆", "pydantic_ai": "★★★★☆"}),
        ("Model freedom",  {"langgraph": "★★★★★", "openai_agents_sdk": "★☆☆☆☆", "crewai": "★★★★★", "autogen": "★★★★★", "pydantic_ai": "★★★★★"}),
        ("RAG support",    {"langgraph": "★★★★☆", "openai_agents_sdk": "★★★☆☆", "crewai": "★★★☆☆", "autogen": "★★★☆☆", "pydantic_ai": "★★☆☆☆"}),
        ("Debugging",      {"langgraph": "★★★★★", "openai_agents_sdk": "★★★★☆", "crewai": "★★☆☆☆", "autogen": "★★★☆☆", "pydantic_ai": "★★★★☆"}),
        ("Community",      {"langgraph": "★★★★★", "openai_agents_sdk": "★★★★☆", "crewai": "★★★★☆", "autogen": "★★★★☆", "pydantic_ai": "★★★☆☆"}),
    ]

    # Header
    names = ["LangGraph", "OpenAI SDK", "CrewAI", "AutoGen", "Pydantic AI"]
    keys = ["langgraph", "openai_agents_sdk", "crewai", "autogen", "pydantic_ai"]
    header = f"  {'Dimension':<20s}" + "".join(f"{n:<14s}" for n in names)
    print(f"\n{header}")
    print(f"  {'─' * 18}" + "".join(f"{'─' * 12}  " for _ in names))

    for dim_name, scores in dimensions:
        row = f"  {dim_name:<20s}" + "".join(f"{scores.get(k, 'N/A'):<14s}" for k in keys)
        print(row)

    print(f"""
  READING THE CHART:
  - No single framework wins every dimension.
  - LangGraph: most flexible + production-ready, but steeper learning curve.
  - OpenAI SDK: easiest start, but locked to OpenAI models.
  - CrewAI: best for multi-agent role-play, less control for custom workflows.
  - AutoGen: strong multi-agent, but API instability in transition.
  - Pydantic AI: cleanest for structured output, not for complex orchestration.
  """)


# ============================================================
# DEMO 4: "What Changed Since I Last Looked?" Tracker
# ============================================================

def demo_changelog():
    """Major events in the agent framework space."""
    print("\n" + "=" * 70)
    print("  FRAMEWORK TIMELINE — MAJOR EVENTS")
    print("=" * 70)
    print("""
  2023
  ────
  Mar  ChatGPT plugins → first "tool use" hype
  May  LangChain agents (AgentExecutor) → first popular framework
  Sep  AutoGen v0.2 released by Microsoft Research
  Nov  OpenAI Assistants API (threads, runs, tools)
  Nov  Semantic Kernel hits GA

  2024
  ────
  Jan  LangGraph launched (graph-based, replaces AgentExecutor)
  Feb  CrewAI explodes in popularity (role-based agents)
  Apr  Anthropic announces tool use for Claude
  Jun  LlamaIndex adds Workflows (event-driven)
  Aug  OpenAI "Swarm" (experimental multi-agent framework)
  Oct  Pydantic AI released (structured + type-safe)
  Nov  DSPy 2.0 (prompt optimization)
  Dec  Google Gemini 2.0 with native tool use

  2025
  ────
  Jan  LangGraph 1.0 stable release
  Mar  OpenAI Agents SDK (replaces Swarm, production-ready)
  Apr  Google ADK launched (Agent Development Kit)
  May  AutoGen v0.4 major rewrite (breaking changes)
  Jun  Amazon Bedrock Agents becomes GA
  Aug  CrewAI 1.0 + CrewAI+ marketplace
  Oct  A2A protocol (Agent-to-Agent, Google) gains traction
  Nov  MCP (Model Context Protocol, Anthropic) widely adopted
  Dec  Semantic Kernel Process Framework for complex workflows

  2026
  ────
  Jan  Multiple frameworks adopt MCP + A2A interop
  Feb  Convergence trend: frameworks start looking more similar

  THE PATTERN:
  Every 6-12 months, a "game-changing" framework appears.
  Every one implements the same core patterns.
  The ones that survive are the ones with real production users.
  """)


# ============================================================
# DEMO 5: Quick Quiz — Test Your Framework Awareness
# ============================================================

def demo_quiz():
    """Test your framework awareness."""
    print("\n" + "=" * 70)
    print("  QUIZ: TEST YOUR FRAMEWORK AWARENESS")
    print("=" * 70)

    questions = [
        {
            "q": "You need a tool-using agent with full control over state, routing,\n"
                 "  and persistence. Which framework?",
            "a": "LangGraph",
            "why": "Graph-based = most control. StateGraph, checkpointers, Send(), Command().",
        },
        {
            "q": "You're at an OpenAI-only company and need to ship an agent ASAP.\n"
                 "  Which framework?",
            "a": "OpenAI Agents SDK",
            "why": "Simplest API, native OpenAI integration, fastest to production.",
        },
        {
            "q": "A non-technical product manager wants to define agents by roles and goals.\n"
                 "  Which framework?",
            "a": "CrewAI",
            "why": "Role-based thinking maps to how PMs think. 'Researcher', 'Writer', 'Reviewer'.",
        },
        {
            "q": "You need agents that generate code, execute it, and iterate.\n"
                 "  Which framework?",
            "a": "AutoGen",
            "why": "Built-in sandboxed code execution. AssistantAgent + UserProxyAgent pattern.",
        },
        {
            "q": "Your app is primarily document search + Q&A with some agent logic.\n"
                 "  Which framework?",
            "a": "LlamaIndex Workflows",
            "why": "RAG-first design. Best document processing, indexing, retrieval.",
        },
        {
            "q": "You need structured JSON output with Pydantic validation, minimal agent.\n"
                 "  Which framework?",
            "a": "Pydantic AI",
            "why": "Type-safe outputs as Pydantic models. Cleanest structured extraction.",
        },
        {
            "q": "You're building a healthcare compliance system that needs human approval\n"
                 "  at specific steps of a multi-step workflow. Which framework?",
            "a": "LangGraph",
            "why": "interrupt_before/after for surgical human-in-loop. Checkpointing for persistence.",
        },
        {
            "q": "A new framework called 'SuperAgent' just launched. What's the FIRST\n"
                 "  question you should ask about it?",
            "a": "What pattern does it implement?",
            "why": "If it's ReAct, you already know 90% of it. The API is just syntax.",
        },
    ]

    score = 0
    for i, question in enumerate(questions, 1):
        print(f"\n  Q{i}: {question['q']}")
        answer = input("  Your answer: ").strip()
        correct = question["a"].lower() in answer.lower()
        if correct:
            print(f"  ✅ Correct! {question['a']}")
            score += 1
        else:
            print(f"  ❌ Answer: {question['a']}")
        print(f"     Why: {question['why']}")

    print(f"\n  Score: {score}/{len(questions)}")
    if score >= 7:
        print("  🏆 Excellent landscape awareness!")
    elif score >= 5:
        print("  👍 Good awareness. Review the ones you missed.")
    else:
        print("  📚 Run demo 1-3 to build your landscape knowledge.")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  MODULE 07: AGENT FRAMEWORK LANDSCAPE")
    print("=" * 70)
    print("""
    Learn to EVALUATE frameworks, not memorize APIs.

    Choose:
      1 → Framework landscape overview (all 8+ frameworks)
      2 → Deep dive into one framework
      3 → Side-by-side comparison chart
      4 → Framework timeline (what happened when)
      5 → Quiz: test your awareness
      6 → Run all (overview → comparison → timeline)
    """)

    choice = input("  Enter choice (1-6): ").strip()

    demos = {
        "1": demo_landscape_overview,
        "2": demo_deep_dive,
        "3": demo_comparison,
        "4": demo_changelog,
        "5": demo_quiz,
    }

    if choice == "6":
        demo_landscape_overview()
        demo_comparison()
        demo_changelog()
    elif choice in demos:
        demos[choice]()
    else:
        print("  Invalid choice.")


if __name__ == "__main__":
    main()
