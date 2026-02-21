# Project 1: ReAct Agent from Scratch# Project 1: ReAct Agent from Scratch










































4. Build a simple medication dosing agent with BMI and GFR tools3. Implement a "confidence threshold" — agent admits uncertainty2. Add a maximum step limit to prevent infinite loops1. Add a new tool (e.g., "check_drug_interaction") and test the agent## Exercises4. **Agent vs Direct LLM** — See the difference between agent reasoning and direct answers3. **Multi-Step Reasoning** — Agent chains multiple tool calls to solve complex problems2. **Medical Lookup Agent** — Agent that searches drug info and lab values1. **ReAct Loop Explained** — Step-by-step walkthrough with visible reasoning## Demos```python main.pycd level_3_agents/01_react_agent```bash## Running the Code- You can build simple agents WITHOUT any framework dependencies- Understanding the loop makes debugging frameworks much easier- Frameworks (LangChain, LangGraph) are built ON this pattern## Why Build from Scratch First?```    FINAL ANSWER → LLM gives the answer    (repeat until LLM has enough info)    OBSERVATION → Tool returns a result    ACTION   → LLM decides to call a tool    THOUGHT  → LLM reasons about the problemwhile not done:```## The ReAct Loop- How this connects to function calling you learned in Level 1- How agents decide which tool to use and when to stop- Build a complete agent loop with NO frameworks- The ReAct (Reasoning + Acting) pattern — foundation of ALL AI agents## What You'll Learn
## What You'll Learn
- Build the ReAct (Reason + Act) loop using just OpenAI API
- Understand HOW an agent decides to use tools
- See the Thought → Action → Observation cycle in real time
- No frameworks — demystify what's happening under the hood

## The ReAct Loop

```
while not done:
    THOUGHT = LLM decides what to do next
    ACTION  = LLM calls a function/tool
    OBSERVE = We run the tool, get the result
    # Feed observation back to LLM for next thought
```

This is literally what LangChain/LangGraph do internally. You're building it yourself first!

## Running the Code

```bash
cd level_3_agents/01_react_agent
python main.py
```

## Demos
1. **ReAct Loop Explained** — See every thought/action/observation step
2. **Medical Lookup Agent** — Agent with drug, lab, and guideline tools
3. **Multi-Step Reasoning** — Agent chains multiple tool calls to answer complex questions
4. **Agent vs Direct LLM** — Compare agent reasoning vs single-shot LLM

## Exercises
1. Add a new tool (e.g., `check_allergies`) and watch the agent use it
2. Handle a "no tool needed" case — agent answers directly without tools
3. Add a maximum step limit to prevent infinite loops
4. Log all agent steps to a file for debugging

## Key Learning
After this project, you'll understand that an "agent" is just an LLM in a loop making tool choices. Everything else (LangChain, LangGraph, etc.) is a framework on top of this core pattern.
