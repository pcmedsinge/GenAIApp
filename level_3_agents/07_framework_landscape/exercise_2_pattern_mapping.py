"""
Exercise 2: Pattern Mapping — YOUR Knowledge → Every Framework

Skills practiced:
- Mapping patterns you built in LangGraph to other frameworks
- Recognizing the SAME pattern behind different APIs
- Reading framework docs faster by knowing what to look for
- Understanding: it's the pattern that matters, not the syntax

Why this matters:
  You built 30+ exercises in LangGraph. Every pattern you learned
  exists in every other framework — with different names.

  This exercise shows you EXACTLY how your knowledge translates.

  When you see CrewAI's "Crew" — you think "orchestrator pattern."
  When you see OpenAI's "Handoff" — you think "Command(goto=...)."
  When you see AutoGen's "GroupChat" — you think "multi-agent with routing."

  The syntax changes. The THINKING doesn't.
"""

import json


# ============================================================
# PATTERN DATABASE: What You Built → How Others Do It
# ============================================================

PATTERNS = {
    "react_loop": {
        "name": "ReAct Loop (Think → Act → Observe)",
        "your_exercise": "03_langgraph_workflows/exercise_7 → create_react_agent()",
        "what_you_know": (
            "LLM decides to call a tool or respond. If tool: execute it, "
            "feed result back to LLM. Loop until LLM has enough info."
        ),
        "in_other_frameworks": {
            "LangGraph": {
                "code": "agent = create_react_agent(llm, tools)",
                "notes": "One-line prebuilt, or manual with ToolNode + tools_condition.",
            },
            "OpenAI Agents SDK": {
                "code": "agent = Agent(name='helper', tools=[func1, func2])\nresult = Runner.run_sync(agent, 'query')",
                "notes": "Agent() auto-creates the ReAct loop. No graph to build.",
            },
            "CrewAI": {
                "code": "agent = Agent(role='helper', goal='...', tools=[tool1])\ntask = Task(description='query', agent=agent)\ncrew = Crew(agents=[agent], tasks=[task])\nresult = crew.kickoff()",
                "notes": "Each Agent has its own ReAct loop. Crew orchestrates tasks.",
            },
            "AutoGen": {
                "code": "assistant = AssistantAgent('helper', llm_config={...})\nassistant.register_for_llm(name='tool1')(func1)\nuser = UserProxyAgent('user', human_input_mode='NEVER')\nuser.initiate_chat(assistant, message='query')",
                "notes": "ConversableAgent does ReAct internally. Chat-based interface.",
            },
            "Pydantic AI": {
                "code": "agent = Agent('openai:gpt-4o-mini', system_prompt='...')\n@agent.tool\ndef my_tool(ctx, param: str) -> str: ...\nresult = agent.run_sync('query')",
                "notes": "Minimal API. @agent.tool decorator makes tools. Auto ReAct loop.",
            },
        },
    },

    "tool_calling": {
        "name": "Tool Definition & Calling",
        "your_exercise": "02_langchain_agents/exercise_1 → @tool decorator",
        "what_you_know": (
            "Define a Python function. Decorate it as a tool. "
            "The framework generates the JSON schema for the LLM."
        ),
        "in_other_frameworks": {
            "LangGraph": {
                "code": "@tool\ndef check_lab(test: str, value: float) -> str:\n    '''Docstring becomes the description.'''\n    return result",
                "notes": "@tool from langchain_core.tools. Works with ToolNode and create_react_agent.",
            },
            "OpenAI Agents SDK": {
                "code": "def check_lab(test: str, value: float) -> str:\n    '''Docstring becomes description.'''\n    return result\n\nagent = Agent(tools=[check_lab])  # plain function!",
                "notes": "Just pass plain functions. SDK auto-generates schema from type hints.",
            },
            "CrewAI": {
                "code": "@tool\ndef check_lab(test: str, value: float) -> str:\n    '''Docstring.'''\n    return result\n\nagent = Agent(role='...', tools=[check_lab])",
                "notes": "Uses same @tool decorator as LangChain. Drop-in compatible!",
            },
            "AutoGen": {
                "code": "@assistant.register_for_llm(description='Check lab')\n@user.register_for_execution()\ndef check_lab(test: str, value: float) -> str:\n    return result",
                "notes": "Two decorators: one for LLM (schema), one for execution (which agent runs it).",
            },
            "Pydantic AI": {
                "code": "@agent.tool\ndef check_lab(ctx: RunContext[MyDeps], test: str, value: float) -> str:\n    '''Docstring.'''\n    return result",
                "notes": "@agent.tool binds tool to specific agent. RunContext enables dependency injection.",
            },
        },
    },

    "state_management": {
        "name": "State Management & Persistence",
        "your_exercise": "03_langgraph_workflows/exercise_6 → checkpoints",
        "what_you_know": (
            "TypedDict defines state shape. Checkpointer saves at each step. "
            "MemorySaver for dev, SqliteSaver for production. Thread-based isolation."
        ),
        "in_other_frameworks": {
            "LangGraph": {
                "code": "class State(TypedDict):\n    messages: list\n    custom_field: str\n\ngraph.compile(checkpointer=MemorySaver())\nresult = app.invoke(input, {'configurable': {'thread_id': '1'}})",
                "notes": "Most sophisticated: typed state, checkpoints, time-travel, cross-thread store.",
            },
            "OpenAI Agents SDK": {
                "code": "# Context variables for state\nfrom agents import RunContext\n# No built-in persistence; manual save/load",
                "notes": "RunContext for per-run state. NO built-in checkpointing. You build persistence.",
            },
            "CrewAI": {
                "code": "crew = Crew(\n    agents=[...],\n    memory=True,  # enables memory system\n    # short-term, long-term, entity memory\n)",
                "notes": "Built-in memory types but less control than LangGraph's TypedDict + checkpoints.",
            },
            "AutoGen": {
                "code": "# State lives in chat messages between agents\n# Cache system for LLM calls\nassistant = AssistantAgent(..., cache=DiskCache('.cache'))",
                "notes": "State = conversation history. Cache for LLM calls. Custom stores for persistence.",
            },
            "Pydantic AI": {
                "code": "# Dependencies (type-safe context)\nfrom pydantic_ai import Agent, RunContext\n\n@dataclass\nclass Deps:\n    patient_id: str\n    db: Database\n\nagent = Agent('model', deps_type=Deps)",
                "notes": "Dependency injection for context. No built-in persistence or checkpoints.",
            },
        },
    },

    "multi_agent": {
        "name": "Multi-Agent Collaboration",
        "your_exercise": "06_agent_architectures/exercise_4 → hierarchical, exercise_6 → P2P",
        "what_you_know": (
            "Supervisor routes to specialists. Send() for dynamic fan-out. "
            "Subgraphs for modular agents. Command() for in-node routing."
        ),
        "in_other_frameworks": {
            "LangGraph": {
                "code": "# Supervisor pattern\ngraph.add_conditional_edges('supervisor', route_to_specialist)\n\n# Dynamic fan-out\nreturn [Send('specialist', data) for data in conditions]\n\n# Subgraphs\nresult = cardiac_subgraph.invoke(state)",
                "notes": "Most flexible: supervisor, P2P, fan-out (Send), subgraphs, Command routing.",
            },
            "OpenAI Agents SDK": {
                "code": "triage = Agent(name='triage', handoffs=[cardio_agent, neuro_agent])\ncardio = Agent(name='cardio', tools=[...], handoffs=[triage])\n# Agents hand off to each other",
                "notes": "Handoff-based: agents delegate to other agents. Simple but less flexible.",
            },
            "CrewAI": {
                "code": "crew = Crew(\n    agents=[researcher, writer, reviewer],\n    tasks=[research_task, write_task, review_task],\n    process=Process.hierarchical,  # or .sequential\n    manager_llm=ChatOpenAI(model='gpt-4o')\n)",
                "notes": "Role-based crews. Sequential or hierarchical process. Manager agent auto-created.",
            },
            "AutoGen": {
                "code": "group_chat = GroupChat(\n    agents=[triage, cardio, neuro],\n    messages=[],\n    max_round=10,\n    speaker_selection_method='auto'  # LLM picks next speaker\n)\nmanager = GroupChatManager(group_chat, llm_config={...})",
                "notes": "GroupChat: agents take turns speaking. LLM or round-robin speaker selection.",
            },
            "Pydantic AI": {
                "code": "# No built-in multi-agent\n# Manual orchestration:\nresult1 = agent_a.run_sync(query)\nresult2 = agent_b.run_sync(result1.data)",
                "notes": "Single-agent focused. Multi-agent = manual orchestration. Not its strength.",
            },
        },
    },

    "human_in_loop": {
        "name": "Human-in-the-Loop",
        "your_exercise": "03_langgraph_workflows/exercise_2 → human review, exercise_6 → interrupt",
        "what_you_know": (
            "interrupt_before/after pauses at a specific node. State is checkpointed. "
            "Human reviews, modifies state, then resumes. update_state() for overrides."
        ),
        "in_other_frameworks": {
            "LangGraph": {
                "code": "app = graph.compile(\n    checkpointer=MemorySaver(),\n    interrupt_before=['review_node']\n)\n# Pauses at review_node. Human inspects.\napp.update_state(config, {'approved': True})\napp.invoke(None, config)  # resume",
                "notes": "Most granular: interrupt at any node, inspect state, modify, resume. Time-travel.",
            },
            "OpenAI Agents SDK": {
                "code": "# Guardrails for input/output validation\nfrom agents import InputGuardrail, OutputGuardrail\n\nagent = Agent(\n    name='helper',\n    input_guardrails=[my_input_guard],\n    output_guardrails=[my_output_guard]\n)",
                "notes": "Guardrails validate input/output. Not the same as mid-workflow pause-and-review.",
            },
            "CrewAI": {
                "code": "task = Task(\n    description='Review the treatment plan',\n    agent=reviewer,\n    human_input=True  # asks for human input before completing\n)",
                "notes": "human_input=True on Task. Human gets a prompt. Less control over WHEN in the flow.",
            },
            "AutoGen": {
                "code": "user_proxy = UserProxyAgent(\n    'human',\n    human_input_mode='ALWAYS'  # or 'TERMINATE' or 'NEVER'\n)",
                "notes": "human_input_mode controls when human is asked. ALWAYS = every turn.",
            },
            "Pydantic AI": {
                "code": "# No built-in human-in-loop\n# Manual: run agent, check result, decide whether to proceed\nresult = agent.run_sync(query)\nif needs_review(result.data):\n    human_decision = input('Approve?')",
                "notes": "Fully manual. No built-in pause/resume or approval flows.",
            },
        },
    },

    "dynamic_fanout": {
        "name": "Dynamic Parallel Execution (Fan-Out)",
        "your_exercise": "03_langgraph_workflows/exercise_8 → Send()",
        "what_you_know": (
            "Send() spawns branches at runtime. Number of branches depends on data. "
            "Annotated[list, operator.add] collects results. Map-reduce pattern."
        ),
        "in_other_frameworks": {
            "LangGraph": {
                "code": "def route(state) -> list[Send]:\n    return [Send('process', {'item': i}) for i in state['items']]\n\ngraph.add_conditional_edges(START, route, ['process'])",
                "notes": "Send() is unique to LangGraph. Most flexible dynamic fan-out.",
            },
            "OpenAI Agents SDK": {
                "code": "# No native fan-out\n# Manual: use asyncio.gather\nimport asyncio\nresults = await asyncio.gather(\n    Runner.run(agent, item1),\n    Runner.run(agent, item2)\n)",
                "notes": "No built-in Send() equivalent. Use asyncio for manual parallelism.",
            },
            "CrewAI": {
                "code": "# Process.sequential or Process.hierarchical\n# Fan-out = define multiple tasks for multiple agents\ncrew = Crew(agents=[a1, a2, a3], tasks=[t1, t2, t3])",
                "notes": "Tasks can be parallel if agents are different. Not data-driven dynamic fan-out.",
            },
            "AutoGen": {
                "code": "# GroupChat handles multi-agent, but not data-driven fan-out\n# Custom: spawn multiple chat sessions",
                "notes": "No direct equivalent. GroupChat gives multi-agent but not map-reduce style.",
            },
            "Pydantic AI": {
                "code": "# Manual parallelism\nimport asyncio\nresults = await asyncio.gather(\n    agent.run(item1), agent.run(item2)\n)",
                "notes": "No built-in fan-out. Use asyncio.gather for manual parallel execution.",
            },
        },
    },
}


# ============================================================
# DEMO 1: Pattern Translation Table
# ============================================================

def demo_translation_table():
    """Show how YOUR patterns translate to other frameworks."""
    print("\n" + "=" * 70)
    print("  PATTERN TRANSLATION TABLE")
    print("=" * 70)
    print("""
  Everything you built in LangGraph has an equivalent everywhere.
  Here's the translation:
  """)

    for key, pattern in PATTERNS.items():
        print(f"\n  ━━━ {pattern['name']} ━━━")
        print(f"  Your exercise: {pattern['your_exercise']}")
        print(f"  What you know: {pattern['what_you_know'][:100]}...")
        print(f"\n  Framework equivalents:")
        for fw, info in pattern["in_other_frameworks"].items():
            print(f"    {fw}:")
            for line in info["code"].split("\n")[:2]:
                print(f"      {line}")
            print(f"      → {info['notes'][:80]}...")
        print()


# ============================================================
# DEMO 2: Pick a Pattern, See All Implementations
# ============================================================

def demo_pattern_deep_dive():
    """Deep dive into one pattern across all frameworks."""
    print("\n" + "=" * 70)
    print("  PATTERN DEEP DIVE — ONE PATTERN, ALL FRAMEWORKS")
    print("=" * 70)

    print("\n  Which pattern do you want to compare?")
    pattern_list = list(PATTERNS.items())
    for i, (key, p) in enumerate(pattern_list, 1):
        print(f"    {i}. {p['name']}")

    choice = input("\n  Pick a number: ").strip()
    try:
        key, pattern = pattern_list[int(choice) - 1]
    except (IndexError, ValueError):
        print("  Invalid choice.")
        return

    print(f"\n  {'=' * 60}")
    print(f"  {pattern['name']}")
    print(f"  {'=' * 60}")
    print(f"\n  Your exercise: {pattern['your_exercise']}")
    print(f"  What you know: {pattern['what_you_know']}")

    for fw, info in pattern["in_other_frameworks"].items():
        print(f"\n  ─── {fw} ───")
        print(f"  Code:")
        for line in info["code"].split("\n"):
            print(f"    {line}")
        print(f"  Notes: {info['notes']}")


# ============================================================
# DEMO 3: "If I Know LangGraph, How Fast Can I Learn X?"
# ============================================================

def demo_learning_curve():
    """Estimate learning curve for each framework given LangGraph knowledge."""
    print("\n" + "=" * 70)
    print("  LEARNING CURVE: IF YOU KNOW LANGGRAPH")
    print("=" * 70)
    print("""
  You already know LangGraph deeply. How long to be PRODUCTIVE
  (not expert) in each framework?

  ┌──────────────────────────────────────────────────────────────┐
  │ Framework          │ Time to Productive │ Why                │
  ├────────────────────┼────────────────────┼────────────────────┤
  │ OpenAI Agents SDK  │ 2-4 hours          │ Simpler API, same  │
  │                    │                    │ patterns, less to  │
  │                    │                    │ learn (fewer knobs)│
  ├────────────────────┼────────────────────┼────────────────────┤
  │ CrewAI             │ 4-8 hours          │ Different mental   │
  │                    │                    │ model (roles), but │
  │                    │                    │ same patterns under│
  │                    │                    │ the hood           │
  ├────────────────────┼────────────────────┼────────────────────┤
  │ AutoGen            │ 1-2 days           │ Chat-based model   │
  │                    │                    │ requires adjustment│
  │                    │                    │ + API in flux      │
  ├────────────────────┼────────────────────┼────────────────────┤
  │ Pydantic AI        │ 1-2 hours          │ So minimal, mostly │
  │                    │                    │ just read the docs │
  ├────────────────────┼────────────────────┼────────────────────┤
  │ Semantic Kernel    │ 2-3 days           │ Enterprise patterns│
  │                    │                    │ + different naming │
  │                    │                    │ conventions        │
  ├────────────────────┼────────────────────┼────────────────────┤
  │ Google ADK         │ 4-8 hours          │ Similar concepts,  │
  │                    │                    │ Google-specific     │
  │                    │                    │ conventions         │
  ├────────────────────┼────────────────────┼────────────────────┤
  │ LlamaIndex Wkflows │ 4-8 hours          │ Event-driven model │
  │                    │                    │ is different from   │
  │                    │                    │ graph-based         │
  └──────────────────────────────────────────────────────────────┘

  KEY INSIGHT:
  Because you understand PATTERNS, learning new frameworks is
  mostly about:
  1. Mapping concepts ("this is their version of ToolNode")
  2. Learning the new API syntax
  3. Understanding what they DON'T support (gaps)

  The hard part — understanding WHY agents work the way they do —
  you've already done.
  """)


# ============================================================
# DEMO 4: Framework Migration Cheat Sheet
# ============================================================

def demo_migration_cheatsheet():
    """Quick reference for translating LangGraph code to other frameworks."""
    print("\n" + "=" * 70)
    print("  MIGRATION CHEAT SHEET: LANGGRAPH → OTHER FRAMEWORKS")
    print("=" * 70)
    print("""
  Quick translations when reading other frameworks' code:

  ┌─────────────────────────────┬──────────────────────────────────┐
  │ LangGraph                   │ Other Frameworks                 │
  ├─────────────────────────────┼──────────────────────────────────┤
  │ StateGraph(State)           │ The workflow/app definition       │
  │ add_node("name", func)      │ A step / task / handler          │
  │ add_edge("a", "b")          │ Pipeline / sequential flow       │
  │ add_conditional_edges(...)  │ Router / gate / decision point   │
  │ ToolNode(tools)             │ Tool executor (usually implicit) │
  │ tools_condition             │ "Has tool calls?" check          │
  │ create_react_agent(...)     │ Agent() / Agent(tools=[...])     │
  │ @tool decorator             │ @tool / plain function           │
  │ MemorySaver()               │ Memory / cache / persistence     │
  │ interrupt_before            │ Human review / approval step     │
  │ Send("node", data)          │ Fan-out / parallel execution     │
  │ Command(goto=..., update=.) │ Handoff / delegation / routing   │
  │ Subgraph (compiled graph)   │ Sub-workflow / sub-agent         │
  │ InMemoryStore               │ Shared memory / knowledge base   │
  │ MessagesState               │ Chat history / message list      │
  │ thread_id                   │ Session / conversation ID        │
  └─────────────────────────────┴──────────────────────────────────┘

  THE VOCABULARY CHANGES. THE CONCEPTS DON'T.

  When you see "Handoff" in OpenAI SDK, think "Command(goto=...)".
  When you see "Crew" in CrewAI, think "supervisor + subgraphs".
  When you see "GroupChat" in AutoGen, think "multi-agent with routing".
  """)


# ============================================================
# DEMO 5: Interactive — Map a New Framework
# ============================================================

def demo_interactive_mapping():
    """Practice mapping a new framework to patterns you know."""
    print("\n" + "=" * 70)
    print("  EXERCISE: MAP A NEW FRAMEWORK")
    print("=" * 70)
    print("""
  Imagine you just discovered a new framework. Let's practice
  mapping it to patterns you already know.

  Here's a hypothetical framework: "MedAgent"
  Its docs show these features:

    - MedAgent(role="cardiologist", capabilities=[check_ecg, read_labs])
    - MedTeam(agents=[cardio, neuro], protocol="round-robin")
    - med_tool(name="check_ecg", handler=ecg_function)
    - AgentMemory(backend="postgresql")
    - ReviewGate(requires_human=True)
    - ParallelConsult(specialists=["cardio", "neuro", "endo"])

  Map each to what you know:
  """)

    mappings = [
        ("MedAgent(role=..., capabilities=[...])",
         "create_react_agent() or Agent()",
         "A tool-using agent with a persona. ReAct loop under the hood."),
        ("MedTeam(agents=[...], protocol='round-robin')",
         "StateGraph with supervisor or GroupChat",
         "Multi-agent orchestration with a routing strategy."),
        ("med_tool(name=..., handler=...)",
         "@tool decorator",
         "Tool definition. 'handler' = the Python function."),
        ("AgentMemory(backend='postgresql')",
         "Checkpointer (SqliteSaver, PostgresSaver)",
         "State persistence. 'backend' = which store to use."),
        ("ReviewGate(requires_human=True)",
         "interrupt_before, human-in-loop",
         "Pause for human approval before continuing."),
        ("ParallelConsult(specialists=[...])",
         "Send() / dynamic fan-out",
         "Spawn parallel branches, one per specialist."),
    ]

    for feature, answer, explanation in mappings:
        print(f"\n  Feature: {feature}")
        user_answer = input("  This maps to (your LangGraph equivalent): ").strip()
        if user_answer:
            print(f"  ✓ Expected: {answer}")
        else:
            print(f"  Answer: {answer}")
        print(f"    Why: {explanation}")

    print(f"\n  See? You already understood 'MedAgent' by mapping to LangGraph patterns.")
    print(f"  Do this with EVERY new framework you encounter.")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 2: PATTERN MAPPING — YOUR KNOWLEDGE → EVERY FRAMEWORK")
    print("=" * 70)
    print("""
    Map what you know to what other frameworks call things.

    Choose:
      1 → Full translation table (all patterns, all frameworks)
      2 → Deep dive into one pattern across frameworks
      3 → Learning curve estimates (LangGraph → X)
      4 → Migration cheat sheet (vocabulary mapping)
      5 → Interactive: map a new framework (exercise)
      6 → Run all non-interactive
    """)

    choice = input("  Enter choice (1-6): ").strip()

    demos = {
        "1": demo_translation_table,
        "2": demo_pattern_deep_dive,
        "3": demo_learning_curve,
        "4": demo_migration_cheatsheet,
        "5": demo_interactive_mapping,
    }

    if choice == "6":
        demo_translation_table()
        demo_learning_curve()
        demo_migration_cheatsheet()
    elif choice in demos:
        demos[choice]()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. PATTERNS ARE UNIVERSAL:
   ReAct loop, tool calling, state management, human-in-loop,
   multi-agent, fan-out — these exist in EVERY framework.
   The API changes. The pattern doesn't.

2. YOUR LANGGRAPH KNOWLEDGE TRANSLATES:
   StateGraph → workflow definition (in any framework)
   ToolNode → tool executor (usually implicit elsewhere)
   Send() → fan-out/parallel (often manual in other frameworks)
   Command → handoff/routing (named "handoff" in OpenAI SDK)

3. VOCABULARY MAPPING IS THE KEY SKILL:
   When you read a new framework's docs, translate to your known concepts.
   "Crew" = orchestrator. "Handoff" = Command(goto=...). "GroupChat" = multi-agent routing.

4. LEARNING CURVE IS FAST:
   With deep LangGraph knowledge, most frameworks take 2-8 hours to
   become productive. The hard part (understanding agents) is done.

5. WHAT TO LOOK FOR IN DOCS:
   - Their tool calling example → compare to @tool + ToolNode
   - Their multi-agent example → compare to Send/subgraphs
   - Their persistence docs → compare to checkpointers
   - Their human-in-loop → compare to interrupt_before
"""

if __name__ == "__main__":
    main()
