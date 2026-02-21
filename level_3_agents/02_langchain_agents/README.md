# Project 2: LangChain Agents

## What You'll Learn
- LangChain agent framework (the most popular GenAI framework)
- Custom tool creation with the @tool decorator
- Agent with conversational memory
- Structured tool outputs and error handling

## Why LangChain?
- Most widely-used GenAI framework in the industry
- Huge ecosystem of tools and integrations
- Wraps the ReAct pattern you learned in Project 01
- Great for rapid prototyping

## Running the Code

```bash
cd level_3_agents/02_langchain_agents
python main.py
```

## Demos
1. **Custom Tools** — Create healthcare tools with @tool decorator
2. **Agent with Tools** — LangChain agent that uses your custom tools
3. **Agent with Memory** — Conversational agent that remembers context
4. **Research Assistant** — Medical research agent that chains multiple tools

## Exercises
1. Add a new @tool for calculating BMI and have the agent use it
2. Add conversation memory that persists between program runs (save/load)
3. Build an agent that combines drug lookup + lab checking + patient history
4. Compare LangChain agent vs your from-scratch agent from Project 01
