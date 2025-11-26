# Project 1: Basic Chat

## What You'll Learn
- How to structure API calls to LLMs
- Difference between system, user, and assistant roles
- Managing conversation context
- Impact of parameters (temperature, max_tokens)
- Token usage and cost tracking

## Key Concepts

### Message Roles
- **system**: Sets the behavior/personality of the assistant
- **user**: The human's messages
- **assistant**: The AI's responses (used in multi-turn to maintain context)

### Important Parameters
- **temperature** (0.0 - 2.0): Controls randomness
  - 0.0 = Deterministic, same output every time
  - 0.7 = Balanced (good default)
  - 1.5+ = Creative, more variation
  
- **max_tokens**: Maximum length of response
  - Prevents runaway costs
  - 1 token ≈ 4 characters in English
  
- **model**: Which LLM to use
  - gpt-4o-mini: Fast, cheap, good for most tasks
  - gpt-4o: More capable, more expensive
  - claude-3-5-haiku: Fast, efficient
  - claude-3-5-sonnet: More capable

## Running the Code

```bash
# Make sure you're in the virtual environment
python main.py
```

## Exercises

### Exercise 1: Modify the System Prompt
Change the system message to create different personalities:
- A pediatric nurse (gentle, simple language)
- An ER triage assistant (urgent, efficient)
- A mental health counselor (empathetic, supportive)

### Exercise 2: Add Input Validation
Enhance the code to:
- Detect if user input contains actual medical emergencies
- Warn users that this is not for emergencies
- Add a disclaimer before the conversation starts

### Exercise 3: Conversation Summary
Add a function that:
- Summarizes the entire conversation at the end
- Extracts key symptoms mentioned
- Suggests which medical professional to consult

### Exercise 4: Compare Models
Run the same conversation with:
- OpenAI GPT-4o-mini
- Claude Haiku
- Compare response quality, speed, and cost

## Healthcare Best Practices

⚠️ **IMPORTANT**: 
- Never use real patient data in development
- Always include disclaimers about not being medical advice
- Implement emergency detection (if someone says "chest pain", direct to 911)
- Log conversations for quality review (with proper consent)

## Expected Output

```
User: I've been having headaches for the past week.

OpenAI Response:
I understand headaches can be very uncomfortable. To help you better, 
I'd like to ask a few questions: Where exactly is the pain located 
(front, back, sides)? How would you rate the severity on a scale of 1-10?

Claude Response:
I'm sorry to hear you're experiencing headaches. To better understand 
your symptoms, could you describe: 1) When do they typically occur? 
2) What does the pain feel like? 3) Have you noticed any triggers?
```

## Common Issues

### Issue: "API key not found"
**Solution**: Make sure you've created `.env` file from `.env.example` and added your keys

### Issue: "Rate limit exceeded"
**Solution**: You're making too many requests. Wait a minute or upgrade your API tier

### Issue: "Context length exceeded"
**Solution**: Your conversation is too long. Summarize old messages or start fresh

## Next Steps
Once comfortable with basic chat, move to **02_embeddings** to learn semantic search!
