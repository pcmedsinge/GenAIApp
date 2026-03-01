"""
Exercise 3: Decision Guide — Which Framework for Which Job?

Skills practiced:
- Making framework selection decisions based on constraints
- Understanding trade-offs (not just features)
- Thinking in terms of USE CASES, not brand loyalty
- Building a decision tree for future choices

Why this matters:
  The most common mistake: picking a framework because of a blog post,
  then fighting it for months because it doesn't fit the actual need.

  This exercise gives you a DECISION TREE and SCENARIO PRACTICE
  so you pick the right tool for the right job — every time.

The Decision Framework:

  Start with your CONSTRAINTS, not your preferences:
  ┌─────────────────────────────────────────────────┐
  │  1. What LLM provider(s) must you support?      │
  │  2. How complex is the workflow?                 │
  │  3. Is human oversight required?                 │
  │  4. How many agents need to collaborate?         │
  │  5. What's the deployment target?                │
  │  6. What's the team's language preference?       │
  └─────────────────────────────────────────────────┘
"""

import json


# ============================================================
# THE DECISION TREE
# ============================================================

DECISION_TREE = """
                    START HERE
                        │
            ┌───────────┴───────────┐
            │ Must use ONLY OpenAI? │
            └───────────┬───────────┘
                   yes ╱ ╲ no
                      ╱   ╲
           ┌─────────╱     ╲──────────────────────────────────┐
           │ Simple agent?  │                                   │
           │ (1-2 tools,    │                                   │
           │  no complex    │            ┌──────────────────┐   │
           │  workflow)     │            │ Complex workflow? │   │
           └──────┬────────┘            │ (custom routing,  │   │
             yes ╱ ╲ no                 │  fan-out, sub-    │   │
                ╱   ╲                   │  graphs)          │   │
            ┌──╱─┐ ┌─╲──────┐          └──────┬───────────┘   │
            │Open│ │OpenAI   │            yes ╱ ╲ no           │
            │AI  │ │Agents   │               ╱   ╲            │
            │SDK │ │SDK w/   │          ┌───╱─┐ ┌──╲──────┐   │
            │    │ │handoffs │          │Lang │ │Multi-    │   │
            └────┘ └─────────┘          │Graph│ │agent     │   │
                                        │     │ │role-play?│   │
                                        └─────┘ └──┬──────┘   │
                                               yes╱ ╲no        │
                                                 ╱   ╲         │
                                            ┌───╱─┐ ┌─╲─────┐ │
                                            │Crew │ │Simple  │ │
                                            │AI   │ │struct  │ │
                                            │     │ │output? │ │
                                            └─────┘ └──┬─────┘ │
                                                  yes ╱ ╲ no   │
                                                     ╱   ╲     │
                                                ┌───╱─┐ ┌─╲──┐ │
                                                │Pyda │ │Lang│ │
                                                │ntic │ │Gra │ │
                                                │AI   │ │ph  │ │
                                                └─────┘ └────┘ │
                                                               │
  Additional filters:                                          │
  ─────────────────                                            │
  • RAG-heavy? → LlamaIndex Workflows                         │
  • Google Cloud? → Google ADK                                 │
  • AWS? → Amazon Bedrock Agents                               │
  • .NET/Java enterprise? → Semantic Kernel                    │
  • Code gen + execution? → AutoGen                            │
"""


# ============================================================
# SCENARIOS DATABASE
# ============================================================

SCENARIOS = [
    {
        "id": 1,
        "title": "Hospital Clinical Decision Support",
        "description": (
            "Build a CDS system where clinicians input patient data, "
            "the system checks guidelines, labs, drug interactions, "
            "and generates recommendations. Needs human approval before "
            "showing to patients. Must persist across sessions. "
            "Hospital uses Azure OpenAI but wants option for on-prem models later."
        ),
        "constraints": {
            "model_freedom": "Need Azure OpenAI now, on-prem later → HIGH",
            "complexity": "Multi-step workflow with routing → HIGH",
            "human_oversight": "Required for clinical recommendations → YES",
            "multi_agent": "Multiple specialist checks → YES",
            "persistence": "Cross-session patient context → REQUIRED",
        },
        "best_choice": "LangGraph",
        "why": (
            "Most flexible workflow control. interrupt_before for human approval. "
            "Checkpointing for persistence. InMemoryStore for cross-session knowledge. "
            "Model-agnostic (Azure now, local later). Send() for parallel specialist checks."
        ),
        "runner_up": "Semantic Kernel (if heavily invested in Azure/.NET stack)",
    },
    {
        "id": 2,
        "title": "Customer Support Chatbot (Startup)",
        "description": (
            "Build a customer support agent that can look up orders, "
            "check FAQs, and escalate to human agents. OpenAI-only shop. "
            "Need to ship in 2 weeks. Simple tool use, no complex routing."
        ),
        "constraints": {
            "model_freedom": "OpenAI only → NOT NEEDED",
            "complexity": "Simple tool use → LOW",
            "human_oversight": "Escalation only → MINIMAL",
            "multi_agent": "Single agent with tools → NO",
            "persistence": "Per-conversation → BASIC",
        },
        "best_choice": "OpenAI Agents SDK",
        "why": (
            "Simplest API for the job. OpenAI-only = perfect fit. "
            "Agent + tools in 10 lines. Built-in tracing. Ship fast."
        ),
        "runner_up": "Pydantic AI (if you want structured responses)",
    },
    {
        "id": 3,
        "title": "Content Production Pipeline",
        "description": (
            "Build a system with researcher, writer, editor, and fact-checker agents. "
            "Each has a clear role. Sequential pipeline with occasional loops. "
            "Team thinks in terms of roles and responsibilities."
        ),
        "constraints": {
            "model_freedom": "Want flexibility → MODERATE",
            "complexity": "Sequential with roles → MODERATE",
            "human_oversight": "Editor reviews → MODERATE",
            "multi_agent": "4 agents with clear roles → YES",
            "persistence": "Per-project → BASIC",
        },
        "best_choice": "CrewAI",
        "why": (
            "Role-based model matches the mental model perfectly. "
            "Crew with sequential/hierarchical process. "
            "human_input=True on review task. Most intuitive for role-play."
        ),
        "runner_up": "LangGraph (if you need more control over agent interactions)",
    },
    {
        "id": 4,
        "title": "Code Review Assistant",
        "description": (
            "Build an agent that reviews code, suggests improvements, "
            "and can execute test code in a sandbox to verify suggestions. "
            "Needs safe code execution environment."
        ),
        "constraints": {
            "model_freedom": "Prefer flexibility → MODERATE",
            "complexity": "Review + execute loop → MODERATE",
            "human_oversight": "Before execution → YES",
            "multi_agent": "Reviewer + executor → YES",
            "persistence": "Per-review → BASIC",
        },
        "best_choice": "AutoGen",
        "why": (
            "Built-in sandboxed code execution is its killer feature. "
            "AssistantAgent reviews, UserProxyAgent executes code. "
            "Code generation → execution → feedback loop is native."
        ),
        "runner_up": "LangGraph (if you need more workflow control around code execution)",
    },
    {
        "id": 5,
        "title": "Medical Document Q&A System",
        "description": (
            "Build a system that ingests clinical guidelines (PDFs), "
            "research papers, and formulary documents. Users ask questions "
            "and get answers with citations. RAG is the primary functionality."
        ),
        "constraints": {
            "model_freedom": "Flexibility desired → MODERATE",
            "complexity": "RAG pipeline → MODERATE",
            "human_oversight": "Citation verification → MODERATE",
            "multi_agent": "Not needed → NO",
            "persistence": "Document index → YES",
        },
        "best_choice": "LlamaIndex Workflows",
        "why": (
            "RAG-first framework. Best document processing (PDF, HTML, etc.). "
            "LlamaHub for data connectors. Built-in indexing and retrieval. "
            "Agent can use retrieval as a tool."
        ),
        "runner_up": "LangGraph + LangChain retrievers (if you need complex agent logic around RAG)",
    },
    {
        "id": 6,
        "title": "Enterprise API with Structured Extraction",
        "description": (
            "Build an API that takes unstructured clinical notes and extracts "
            "structured data: diagnoses (ICD-10), medications, vitals, "
            "allergies. Output must be validated Pydantic models."
        ),
        "constraints": {
            "model_freedom": "Multiple providers → YES",
            "complexity": "Extraction pipeline → LOW-MODERATE",
            "human_oversight": "Post-extraction QA → MINIMAL",
            "multi_agent": "Single agent → NO",
            "persistence": "Per-request → NONE",
        },
        "best_choice": "Pydantic AI",
        "why": (
            "Pydantic models as output types = matches exactly. "
            "Type-safe extraction. Minimal API, clean code. "
            "If your output is a data model, Pydantic AI is made for this."
        ),
        "runner_up": "Instructor (even simpler for pure extraction)",
    },
    {
        "id": 7,
        "title": "Multi-Cloud Healthcare Platform",
        "description": (
            "Enterprise healthcare platform deployed across Azure AND GCP. "
            "Needs to work with Azure OpenAI AND Gemini. Complex clinical "
            "workflows, human-in-loop for prescriptions, multi-agent for "
            "specialist consultations. Massive scale (1000+ concurrent users)."
        ),
        "constraints": {
            "model_freedom": "Multi-cloud, multi-model → CRITICAL",
            "complexity": "Complex clinical workflows → VERY HIGH",
            "human_oversight": "Prescription approval → REQUIRED",
            "multi_agent": "Specialist teams → YES",
            "persistence": "Cross-session, audit trail → REQUIRED",
        },
        "best_choice": "LangGraph",
        "why": (
            "Only framework that scores 5/5 on ALL critical dimensions. "
            "Model-agnostic, most flexible workflows, interrupt_before for approvals, "
            "checkpointing + InMemoryStore for persistence, Send() for dynamic fan-out. "
            "LangSmith for enterprise observability."
        ),
        "runner_up": "Custom framework (at this scale, some orgs build their own on top of primitives)",
    },
]


# ============================================================
# DEMO 1: The Decision Tree
# ============================================================

def demo_decision_tree():
    """Display and explain the framework decision tree."""
    print("\n" + "=" * 70)
    print("  FRAMEWORK DECISION TREE")
    print("=" * 70)
    print(DECISION_TREE)
    print("""
  HOW TO USE THIS TREE:
  1. Start with your model provider constraint
  2. Follow based on workflow complexity
  3. Consider multi-agent needs
  4. Apply additional filters (RAG, cloud, language)

  THE DEFAULT ANSWER:
  If you're unsure, LangGraph is the safe choice.
  It can do everything (flexibility = no regrets).
  You only choose something else when a simpler tool suffices.
  """)


# ============================================================
# DEMO 2: Scenario Practice
# ============================================================

def demo_scenario_practice():
    """Practice picking the right framework for each scenario."""
    print("\n" + "=" * 70)
    print("  SCENARIO PRACTICE — PICK THE FRAMEWORK")
    print("=" * 70)
    print("  For each scenario, decide which framework is the best fit.\n")

    score = 0
    for scenario in SCENARIOS:
        print(f"\n  ━━━ Scenario {scenario['id']}: {scenario['title']} ━━━")
        print(f"  {scenario['description']}")
        print(f"\n  Constraints:")
        for k, v in scenario["constraints"].items():
            print(f"    {k}: {v}")

        answer = input(f"\n  Your choice: ").strip()

        if scenario["best_choice"].lower() in answer.lower():
            print(f"  ✅ Correct! {scenario['best_choice']}")
            score += 1
        else:
            print(f"  ❌ Best choice: {scenario['best_choice']}")
        print(f"  Why: {scenario['why'][:150]}...")
        print(f"  Runner-up: {scenario['runner_up']}")

    print(f"\n  Score: {score}/{len(SCENARIOS)}")
    if score >= 6:
        print("  🏆 Excellent framework selection skills!")
    elif score >= 4:
        print("  👍 Good instincts. Review the ones you missed.")
    else:
        print("  📚 Review the decision tree and constraint analysis.")


# ============================================================
# DEMO 3: Constraint-Based Selector (Interactive)
# ============================================================

def demo_constraint_selector():
    """Interactive: answer questions, get a framework recommendation."""
    print("\n" + "=" * 70)
    print("  FRAMEWORK SELECTOR — ANSWER 6 QUESTIONS")
    print("=" * 70)
    print("  Answer these questions about YOUR project.\n")

    # Q1: Model provider
    print("  Q1: What LLM provider(s) do you need?")
    print("    a) OpenAI only")
    print("    b) Multiple providers (OpenAI + Anthropic + etc.)")
    print("    c) Google Cloud / Gemini primarily")
    print("    d) AWS Bedrock primarily")
    print("    e) On-premise / local models needed")
    q1 = input("  Answer (a-e): ").strip().lower()

    # Q2: Complexity
    print("\n  Q2: How complex is the workflow?")
    print("    a) Simple: LLM + a few tools, no branching")
    print("    b) Moderate: some conditional routing, 2-3 steps")
    print("    c) Complex: multi-step, branching, loops, parallel")
    print("    d) Very complex: subworkflows, dynamic routing, fan-out")
    q2 = input("  Answer (a-d): ").strip().lower()

    # Q3: Human oversight
    print("\n  Q3: Do you need human-in-the-loop?")
    print("    a) No — fully autonomous")
    print("    b) Basic — human reviews final output")
    print("    c) Required — human approves at specific steps")
    print("    d) Critical — regulatory requirement (healthcare, finance)")
    q3 = input("  Answer (a-d): ").strip().lower()

    # Q4: Multi-agent
    print("\n  Q4: How many agents need to collaborate?")
    print("    a) Single agent with tools")
    print("    b) 2-3 agents with handoffs")
    print("    c) Many agents with clear roles (researcher, writer...)")
    print("    d) Dynamic: number of agents depends on input data")
    q4 = input("  Answer (a-d): ").strip().lower()

    # Q5: Primary task
    print("\n  Q5: What's the primary task?")
    print("    a) Tool-using chatbot / assistant")
    print("    b) Document Q&A / RAG")
    print("    c) Structured data extraction")
    print("    d) Complex workflow / clinical pipeline")
    print("    e) Code generation + execution")
    q5 = input("  Answer (a-e): ").strip().lower()

    # Q6: Speed priority
    print("\n  Q6: What's more important?")
    print("    a) Ship fast (prototype in days)")
    print("    b) Ship right (production quality, maintainable)")
    q6 = input("  Answer (a-b): ").strip().lower()

    # Score frameworks
    scores = {
        "LangGraph": 0, "OpenAI Agents SDK": 0, "CrewAI": 0,
        "AutoGen": 0, "Pydantic AI": 0, "LlamaIndex": 0,
        "Google ADK": 0, "Semantic Kernel": 0,
    }

    # Q1 scoring
    if q1 == "a":
        scores["OpenAI Agents SDK"] += 3
    elif q1 in ["b", "e"]:
        scores["LangGraph"] += 2; scores["CrewAI"] += 2; scores["Pydantic AI"] += 2
    elif q1 == "c":
        scores["Google ADK"] += 3
    elif q1 == "d":
        pass  # Bedrock, not in our main list

    # Q2 scoring
    if q2 == "a":
        scores["OpenAI Agents SDK"] += 2; scores["Pydantic AI"] += 2
    elif q2 == "b":
        scores["CrewAI"] += 2; scores["LangGraph"] += 1
    elif q2 in ["c", "d"]:
        scores["LangGraph"] += 3

    # Q3 scoring
    if q3 in ["c", "d"]:
        scores["LangGraph"] += 3
    elif q3 == "b":
        scores["LangGraph"] += 1; scores["CrewAI"] += 1

    # Q4 scoring
    if q4 == "a":
        scores["Pydantic AI"] += 1; scores["OpenAI Agents SDK"] += 1
    elif q4 == "b":
        scores["OpenAI Agents SDK"] += 2; scores["LangGraph"] += 1
    elif q4 == "c":
        scores["CrewAI"] += 3
    elif q4 == "d":
        scores["LangGraph"] += 3  # Send() for dynamic

    # Q5 scoring
    if q5 == "b":
        scores["LlamaIndex"] += 3
    elif q5 == "c":
        scores["Pydantic AI"] += 3
    elif q5 == "d":
        scores["LangGraph"] += 3
    elif q5 == "e":
        scores["AutoGen"] += 3

    # Q6 scoring
    if q6 == "a":
        scores["OpenAI Agents SDK"] += 2; scores["CrewAI"] += 1; scores["Pydantic AI"] += 1
    elif q6 == "b":
        scores["LangGraph"] += 2; scores["Semantic Kernel"] += 1

    # Sort and display
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  ─── RECOMMENDATION ───")
    print(f"\n  Based on your constraints:\n")
    for i, (name, score) in enumerate(ranked[:4], 1):
        bar = "█" * score + "░" * (max(s for _, s in ranked) - score)
        medal = ["🥇", "🥈", "🥉", "  "][i - 1]
        print(f"  {medal} {name:<22s} [{bar}] ({score} pts)")

    winner = ranked[0][0]
    print(f"\n  Top pick: {winner}")

    # Add context
    context = {
        "LangGraph": "Most flexible. Handles complex workflows and healthcare compliance.",
        "OpenAI Agents SDK": "Simplest to start. Great if OpenAI is your provider.",
        "CrewAI": "Best for role-based multi-agent teams.",
        "AutoGen": "Best for code gen/execution loops.",
        "Pydantic AI": "Best for structured extraction with type safety.",
        "LlamaIndex": "Best for RAG and document-heavy applications.",
        "Google ADK": "Best for Google Cloud / Gemini deployments.",
        "Semantic Kernel": "Best for .NET enterprise environments.",
    }
    print(f"  Why: {context.get(winner, 'N/A')}")


# ============================================================
# DEMO 4: The "Stay or Switch?" Decision
# ============================================================

def demo_stay_or_switch():
    """When to stay with your current framework vs switch."""
    print("\n" + "=" * 70)
    print("  STAY OR SWITCH? — WHEN TO CHANGE FRAMEWORKS")
    print("=" * 70)
    print("""
  You're using LangGraph. A new framework appears. Should you switch?

  ═══ STAY with your current framework when: ═══

  ✅ You have production systems running on it
     Switching cost > benefit. Migration is expensive and risky.

  ✅ Your pain is about the PROBLEM, not the FRAMEWORK
     "This workflow is complex" ≠ "LangGraph is bad."
     Complex workflows ARE complex. Another framework won't fix that.

  ✅ The new framework does the same thing with different syntax
     If it implements the same patterns — you're just learning new syntax.
     That's a cost, not a benefit.

  ✅ The new framework is < 1 year old with < 5 production users
     Wait. Let others find the bugs.

  ═══ SWITCH when: ═══

  🔄 A clear capability gap exists
     Example: You need sandboxed code execution → AutoGen has it native.
     Building it yourself in LangGraph would take weeks.

  🔄 Your constraint changed fundamentally
     Example: Company switched from multi-model to OpenAI-only.
     OpenAI SDK's tight integration becomes a real advantage.

  🔄 The ecosystem matters more than the framework
     Example: Your team is all .NET → Semantic Kernel makes onboarding
     10x faster than teaching everyone Python + LangGraph.

  🔄 The new framework is 10x simpler for YOUR specific use case
     Example: You're only doing structured extraction → Pydantic AI
     does in 5 lines what takes 30 in LangGraph.

  ═══ THE 10X RULE ═══

  Only switch if the new framework is 10x better for YOUR specific
  use case. Not 2x better. Not cooler. Not newer. 10x better.

  Why 10x? Because switching has hidden costs:
  - Learning curve for the team
  - Migration effort
  - New bugs from unfamiliarity
  - Lost muscle memory and debugging skills
  - New framework's bugs you haven't discovered yet

  ═══ THE HYBRID APPROACH ═══

  You don't have to go all-in. Use different frameworks for different tasks:

  - LangGraph for complex clinical workflows (your strength)
  - Pydantic AI for structured extraction endpoints
  - LlamaIndex for document indexing and RAG
  - OpenAI SDK for simple customer-facing chatbots

  Frameworks are tools. Use the right tool for each job.

  ═══ YOUR POSITION ═══

  You have deep LangGraph knowledge. That's your MOAT.
  - You can evaluate new frameworks in 30 minutes (rubric)
  - You can map patterns in 5 minutes (translation table)
  - You can decide in 10 minutes (decision tree)

  Spend 95% of your time building things.
  Spend 5% keeping your landscape awareness current.
  """)


# ============================================================
# DEMO 5: Your 5% Awareness Routine
# ============================================================

def demo_awareness_routine():
    """A practical routine for staying current."""
    print("\n" + "=" * 70)
    print("  YOUR 5% AWARENESS ROUTINE")
    print("=" * 70)
    print("""
  Spend ~2 hours per month on framework awareness.
  Here's a practical routine:

  ═══ MONTHLY (2 hours total) ═══

  📰 Read (30 min):
     - LangChain blog: https://blog.langchain.dev
     - OpenAI blog: https://openai.com/blog
     - Anthropic blog: https://anthropic.com/research
     - Hacker News search: "agent framework" (latest)
     - Twitter/X: follow @LangChainAI, @OpenAI, @CrewAIInc

  🔍 Evaluate ONE new thing (30 min):
     - Pick the most interesting new framework/tool you saw
     - Run the 8-dimension rubric on it (from Exercise 1)
     - Score it. Is it > 28/40? Worth a deeper look?
     - If yes, spend 1 hour on the "Getting Started" tutorial
     - Map it to your known patterns (from Exercise 2)

  📊 Update mental model (15 min):
     - Has anything in the decision tree changed?
     - Did a Tier 2 framework graduate to Tier 1?
     - Did a Tier 1 framework have a major issue?
     - Any new red flags? (company pivot, lead dev left, etc.)

  🧪 Optional: Try ONE new thing (45 min):
     - Build a 20-line hello-world in the new framework
     - Compare the experience to LangGraph
     - Note: what's easier? what's harder? what's missing?

  ═══ QUARTERLY (additional 2-4 hours) ═══

  📋 Review your framework choices:
     - Are your production systems on the right frameworks?
     - Has anything changed that warrants a switch?
     - Apply the 10x rule: is anything 10x better now?

  📚 Deep dive on ONE framework (if needed):
     - If something scored well on the rubric AND fits your needs
     - Spend 4 hours doing a proper tutorial
     - Build a small project (not production, just learning)

  ═══ WHAT THIS LOOKS LIKE IN PRACTICE ═══

  Month 1: Read blog posts → evaluate "SuperAgent v2" → score: 24/40 → skip
  Month 2: Read blog posts → Google ADK update → score: 31/40 → note for later
  Month 3: Read blog posts → OpenAI SDK major update → already using? → check changelog
  Quarter: Review. Nothing 10x better. Stay the course. Maybe try ADK for small RAG project.

  ═══ THE ANTI-PATTERN ═══

  DON'T: Spend 3 days every time a new framework is announced.
  DON'T: Rewrite production code every time something new appears.
  DON'T: Feel guilty for not knowing every framework deeply.

  DO: Have a systematic evaluation process (you now do).
  DO: Know what patterns to look for (you now do).
  DO: Make decisions based on constraints, not hype (you now do).
  """)


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 3: DECISION GUIDE — WHICH FRAMEWORK FOR WHICH JOB?")
    print("=" * 70)
    print("""
    Pick the right framework every time.

    Choose:
      1 → Decision tree (visual framework picker)
      2 → Scenario practice (7 real-world scenarios)
      3 → Constraint-based selector (answer 6 questions)
      4 → Stay or switch? (when to change frameworks)
      5 → Your 5% awareness routine (staying current)
      6 → Run all non-interactive
    """)

    choice = input("  Enter choice (1-6): ").strip()

    demos = {
        "1": demo_decision_tree,
        "2": demo_scenario_practice,
        "3": demo_constraint_selector,
        "4": demo_stay_or_switch,
        "5": demo_awareness_routine,
    }

    if choice == "6":
        demo_decision_tree()
        demo_stay_or_switch()
        demo_awareness_routine()
    elif choice in demos:
        demos[choice]()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. START WITH CONSTRAINTS, NOT PREFERENCES:
   Model lock-in? Complexity? Human oversight? Multi-agent?
   These constraints narrow the field before you even look at features.

2. THE DECISION TREE:
   OpenAI-only + simple → OpenAI SDK
   Complex workflows + human oversight → LangGraph
   Role-based multi-agent → CrewAI
   Code execution → AutoGen
   RAG-heavy → LlamaIndex
   Structured extraction → Pydantic AI

3. THE 10X RULE FOR SWITCHING:
   Only switch if the new framework is 10x better for YOUR case.
   Switching has hidden costs: learning, migration, new bugs.

4. THE HYBRID APPROACH:
   Use different frameworks for different parts of your system.
   LangGraph for workflows, Pydantic AI for extraction, etc.

5. THE 5% ROUTINE:
   2 hours/month: read, evaluate ONE new thing, update mental model.
   95% of time: build with what you know.
   5% of time: stay aware of what's changing.

6. YOUR MOAT:
   Deep pattern knowledge + one framework mastery + evaluation skills
   = you can adopt ANY framework in hours, not weeks.
   That's more valuable than knowing 5 frameworks shallowly.
"""

if __name__ == "__main__":
    main()
