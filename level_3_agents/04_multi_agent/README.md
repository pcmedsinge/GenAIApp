# Project 4: Multi-Agent Systems

## What You'll Learn
- Multiple specialized agents working together
- Agent communication and task delegation
- Orchestration patterns (sequential, router)
- Why multi-agent beats single-agent for complex problems

## Why Multi-Agent?

Single Agent: ONE LLM tries to handle everything → gets confused on complex tasks
Multi-Agent:  SPECIALIZED agents collaborate → each expert at its domain

```
Healthcare Example:
  [Triage Agent] → classifies urgency
  [Diagnosis Agent] → generates differential
  [Treatment Agent] → recommends medications
  [Safety Agent] → checks for contraindications

  vs. ONE agent trying to do ALL of these at once
```

## Running the Code

```bash
cd level_3_agents/04_multi_agent
python main.py
```

## Demos
1. **Sequential Pipeline** — Agents hand off to each other in order
2. **Router Pattern** — Supervisor routes tasks to specialist agents
3. **Healthcare Pipeline** — Triage → Diagnosis → Treatment with separate agents
4. **Agent Debate** — Two agents discuss a clinical case from different perspectives

## Exercises
1. Add a "Pharmacist Agent" that validates medication safety before final output
2. Build a router that sends different specialties to different specialist agents
3. Implement a "quality check" agent that reviews another agent's output
4. Create a 4-agent pipeline for a complete patient encounter workflow
