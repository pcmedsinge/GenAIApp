# 🚀 Quick Start Guide

## Welcome to Your GenAI Learning Journey!

This guide will get you up and running in **under 10 minutes**.

---

## Step 1: Set Up Environment (2 minutes)

### Create Virtual Environment

```bash
cd /home/linuxdev1/PracticeApps/GENAIApp

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# You should see (venv) in your prompt
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install ~50 packages. It takes 2-3 minutes.

---

## Step 2: Get API Keys (5 minutes)

You need at least **one** of these (OpenAI recommended for beginners):

### Option A: OpenAI API (Recommended)

1. Go to: https://platform.openai.com/api-keys
2. Sign up or log in
3. Click "Create new secret key"
4. Copy the key (starts with `sk-...`)

**Cost**: $5 credit for new accounts, then pay-as-you-go
**Pricing**: ~$0.15 per 1M tokens for GPT-4o-mini

### Option B: Anthropic/Claude API

1. Go to: https://console.anthropic.com/
2. Sign up or log in
3. Go to "API Keys"
4. Create a new key
5. Copy the key (starts with `sk-ant-...`)

**Cost**: $5 minimum deposit
**Pricing**: ~$0.25 per 1M input tokens for Claude Haiku

### Both is Better!

Having both APIs lets you:
- Compare model responses
- Use the best model for each task
- Have a backup if one has an outage

---

## Step 3: Configure Environment (1 minute)

```bash
# Copy the example environment file
cp .env.example .env

# Edit it with your favorite editor
nano .env  # or vim, code, etc.
```

Add your API keys:

```bash
OPENAI_API_KEY=sk-proj-...your_actual_key_here...
ANTHROPIC_API_KEY=sk-ant-...your_actual_key_here...
```

Save and exit.

---

## Step 4: Test Your Setup (1 minute)

```bash
cd level_1_fundamentals/01_basic_chat
python main.py
```

You should see:
```
🏥 Level 1.1: Basic Chat Examples

1. SIMPLE CHAT (Single turn)
------------------------------------------------------------
User: I've been having headaches for the past week.

OpenAI Response:
I understand headaches can be very uncomfortable...
```

**✅ If you see this, you're ready to go!**

**❌ If you see an error:**
- "No module named 'openai'" → Run `pip install -r requirements.txt`
- "Invalid API key" → Check your `.env` file has the correct keys
- "Rate limit exceeded" → Wait a minute and try again

---

## Your First Hour: What to Do

### 1. Complete Level 1 Project 1 (15 minutes)

```bash
cd level_1_fundamentals/01_basic_chat
python main.py
```

- Read the code comments
- Try the interactive chat (uncomment line 297)
- Experiment with different temperatures
- Check the cost tracking example

### 2. Learn About Embeddings (20 minutes)

```bash
cd ../02_embeddings
python main.py
```

- See how semantic search works
- Compare to keyword search
- Try the medical case example

### 3. Explore Function Calling (25 minutes)

```bash
cd ../03_function_calling
python main.py
```

- Watch the LLM call your Python functions
- See how it handles multi-step reasoning
- This is the foundation for AI agents!

---

## Learning Path Overview

```
Week 1-2:  Level 1 (Fundamentals) ← YOU ARE HERE
Week 3-4:  Level 2 (RAG Systems)
Week 5-7:  Level 3 (AI Agents)
Week 8-10: Level 4 (Fine-tuning)
Week 11+:  Level 5 (Production)
```

---

## Cost Management

### Set API Limits (Recommended!)

**OpenAI:**
1. Go to https://platform.openai.com/account/limits
2. Set a monthly budget limit (e.g., $10)
3. You'll get email alerts at 75%, 90%, 100%

**Anthropic:**
1. Go to https://console.anthropic.com/settings/limits
2. Set a spending limit

### Expected Costs

- **Level 1** (Fundamentals): $1-2
- **Level 2** (RAG): $2-3
- **Level 3** (Agents): $5-10
- **Level 4** (Fine-tuning): $10-20
- **Level 5** (Production): Variable

**Total for full learning path: ~$20-50**

### Save Money Tips

1. Use `gpt-4o-mini` instead of `gpt-4o` (10x cheaper)
2. Use `claude-3-5-haiku` instead of `sonnet` (5x cheaper)
3. Set `max_tokens` limits
4. Cache embeddings (don't re-embed same text)
5. Use temperature=0 for testing (consistent results)

---

## Troubleshooting

### Virtual Environment Not Activating

```bash
# Make sure you're in the project directory
cd /home/linuxdev1/PracticeApps/GENAIApp

# Try this if 'source venv/bin/activate' doesn't work
. venv/bin/activate

# Or use the full path
source /home/linuxdev1/PracticeApps/GENAIApp/venv/bin/activate
```

### Import Errors

```bash
# Make sure venv is activated (you should see (venv) in prompt)
# Then reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### API Key Issues

```bash
# Check if .env file exists
ls -la .env

# Check if it has your keys (this won't show the actual keys)
grep "OPENAI_API_KEY" .env
grep "ANTHROPIC_API_KEY" .env

# Make sure there are no spaces around the =
# ✅ OPENAI_API_KEY=sk-...
# ❌ OPENAI_API_KEY = sk-...
```

### Permission Errors

```bash
chmod +x venv/bin/activate
```

---

## Healthcare IT Specific Notes

### HIPAA Compliance

⚠️ **NEVER send real PHI (Protected Health Information) to LLM APIs!**

This includes:
- Real patient names
- Medical record numbers
- Actual clinical notes
- Real dates of service

✅ **Always use synthetic/de-identified data for learning**

### De-identification

If you want to experiment with realistic data:
1. Use synthetic data generators
2. Apply de-identification tools
3. Replace all identifiers with fake data
4. Get proper consent/approval

### Production Considerations

Before deploying in healthcare:
- [ ] HIPAA compliance review
- [ ] Security assessment
- [ ] Clinical validation
- [ ] Regulatory review (FDA if applicable)
- [ ] Human-in-the-loop validation
- [ ] Audit logging
- [ ] Access controls

---

## Next Steps

### Today:
1. ✅ Complete setup
2. ✅ Run first three examples
3. ✅ Read the code and comments

### This Week:
1. Complete all of Level 1 (5 projects)
2. Do the exercises in each README
3. Start thinking about a healthcare problem you want to solve

### Next Week:
1. Move to Level 2 (RAG Systems)
2. Build your first knowledge base
3. Learn vector databases

---

## Getting Help

### Common Resources
- **OpenAI Docs**: https://platform.openai.com/docs
- **Anthropic Docs**: https://docs.anthropic.com
- **LangChain Docs**: https://python.langchain.com (for later levels)

### Healthcare AI Communities
- Reddit: r/HealthTech, r/LocalLLaMA
- Discord: LangChain, Hugging Face
- GitHub: Search for "medical-llm", "healthcare-ai"

### Code Issues
- Read error messages carefully (they're usually helpful)
- Check the README in each project folder
- Google the error message
- Check if your API keys have credits

---

## Pro Tips from a 17-Year Veteran

1. **Start small**: Don't try to build a full EHR AI on day 1
2. **Iterate**: Get something working, then improve it
3. **Validate everything**: LLMs can hallucinate, especially with medical info
4. **Think workflows**: What repetitive tasks do you do that AI could help with?
5. **Privacy first**: Build good habits now about data protection
6. **Learn by teaching**: Explain concepts to others to solidify your understanding

---

## Your First Project Ideas

After completing Level 1, try building:

1. **Medical Abbreviation Expander**
   - Input: Clinical note with abbreviations
   - Output: Expanded, clear note

2. **Drug Interaction Checker**
   - Input: List of medications
   - Output: Interaction warnings

3. **Symptom Triager**
   - Input: Patient symptom description
   - Output: Urgency level, suggested specialty

4. **Clinical Note Summarizer**
   - Input: Long clinical note
   - Output: Structured SOAP summary

5. **ICD-10 Code Suggester**
   - Input: Clinical documentation
   - Output: Suggested diagnosis codes

---

## Ready? Let's Go! 🚀

```bash
# Make sure you're in the right directory
cd /home/linuxdev1/PracticeApps/GENAIApp

# Activate virtual environment
source venv/bin/activate

# Start with Project 1
cd level_1_fundamentals/01_basic_chat
python main.py
```

**Remember**: The best way to learn GenAI is to BUILD. Don't just read the code, run it, modify it, break it, fix it!

Good luck! 🏥💻
