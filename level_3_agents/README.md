# Level 3: AI Agents
**Build autonomous systems that think, plan, and act**

## Overview

Agents are the most exciting frontier in GenAI. Unlike simple chat (Level 1) or search (Level 2),
agents can **reason about what to do**, **take actions**, and **learn from results** — all autonomously.

### What is an Agent?

```
Simple Chat:  User → LLM → Answer (one shot)
RAG:          User → Retrieve docs → LLM → Grounded answer (retrieval + generation)
Agent:        User → LLM THINKS → Chooses a TOOL → Gets result → THINKS again → Maybe uses another TOOL → Final answer
```

An agent is an LLM in a loop: **Think → Act → Observe → Think → Act → ... → Answer**

### Why Agents Matter for Healthcare IT

Healthcare workflows are multi-step:
- **Triage**: Assess symptoms → Check history → Decide urgency → Route to specialist
- **Prior Auth**: Check coverage → Verify criteria → Gather clinical evidence → Submit
- **Clinical Decision**: Review symptoms → Check guidelines → Consider drug interactions → Recommend

These are natural fits for agents.

## Prerequisites
- **Level 1 Complete**: API calls, function calling (Project 03 is especially relevant)
- **Level 2 Complete**: RAG skills (agents USE RAG as a tool)
- **New packages**: `pip install langchain langchain-openai langgraph`

## Projects

### 01_react_agent — ReAct Agent from Scratch
- Build the Think → Act → Observe loop with just OpenAI API
- No frameworks — understand what agents actually DO
- See how the LLM decides which tool to use and when to stop
- **Healthcare Example**: Medical lookup agent with drug/lab/guideline tools

### 02_langchain_agents — LangChain Agent Framework
- Build agents faster using the LangChain framework
- Custom tool creation, built-in tools, memory
- Understand when frameworks help and when they add complexity
- **Healthcare Example**: Medical research assistant

### 03_langgraph_workflows — LangGraph Stateful Workflows
- Build production-grade agents with explicit state and control flow
- Graph-based design: nodes (actions) + edges (decisions)
- Conditional routing, human-in-the-loop, persistence
- **Healthcare Example**: Clinical assessment workflow

### 04_multi_agent — Multi-Agent Collaboration
- Multiple specialized agents working together
- Agent delegation and orchestration
- Supervisor pattern vs peer-to-peer
- **Healthcare Example**: Triage → Diagnosis → Treatment pipeline

### 05_healthcare_agent — Capstone: Clinical Decision Support
- Full clinical agent combining RAG + tools + multi-step reasoning
- Safety guardrails for medical context
- Complete healthcare workflow automation
- **Healthcare Example**: End-to-end clinical decision support system

## Learning Objectives

After completing Level 3, you will:
- ✅ Build ReAct agents from scratch (understand the pattern)
- ✅ Create agents with LangChain (efficient development)
- ✅ Design stateful workflows with LangGraph (production-grade)
- ✅ Orchestrate multi-agent systems
- ✅ Implement safety guardrails for healthcare agents
- ✅ Combine RAG + Agents for knowledge-grounded autonomous systems

## Time Estimate
12-15 hours total (2-3 hours per project)

## Key Concept: The ReAct Pattern

```
User: "Should this patient take ibuprofen?"

THOUGHT: I need to check the patient's medical history first.
ACTION:  check_patient_history(patient_id="12345")
OBSERVE: Patient has CKD Stage 3, on lisinopril.

THOUGHT: CKD is a contraindication for NSAIDs like ibuprofen. Let me verify.
ACTION:  check_drug_contraindications(drug="ibuprofen")
OBSERVE: Contraindicated in CKD (risk of further kidney damage).

THOUGHT: I should suggest an alternative.
ACTION:  suggest_alternative(drug="ibuprofen", condition="CKD")
OBSERVE: Acetaminophen is safe alternative for pain in CKD patients.

ANSWER: Ibuprofen is NOT recommended for this patient due to CKD Stage 3.
        NSAIDs can worsen kidney function. Recommend acetaminophen instead.
        [Sources: Drug database, CKD management guidelines]
```

## Next Steps
After mastering agents, you'll be ready for Level 4 (Fine-Tuning) to customize models for your specific healthcare tasks!
