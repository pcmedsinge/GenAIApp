# Project 4: Advanced Retrieval Techniques

## What You'll Learn
- Multi-query retrieval (rephrase questions for better coverage)
- Re-ranking with LLM (improve precision after initial retrieval)
- Source citations (ground every answer in specific documents)
- Retrieval evaluation (measure how good your retrieval is)

## Why Advanced Retrieval?

Basic retrieval (embed query → find similar chunks) works well for simple questions.
But for complex medical queries, you need more:

| Technique | Problem It Solves |
|-----------|------------------|
| Multi-query | User's phrasing doesn't match document phrasing |
| Re-ranking | Top results by embedding aren't always the best |
| Citations | Clinicians need to verify sources |
| Evaluation | How do you know your RAG is actually good? |

## Running the Code

```bash
cd level_2_rag/04_advanced_retrieval
python main.py
```

## Demos
1. **Multi-Query Retrieval** — Automatically rephrase query multiple ways
2. **LLM Re-Ranking** — Use the LLM to reorder retrieved results
3. **Cited Answers** — Generate answers with proper source citations
4. **Retrieval Evaluation** — Measure precision and relevance

## Exercises
1. Implement a "query expansion" technique using medical synonyms
2. Build a confidence scoring system (high/medium/low based on retrieval scores)
3. Create a "I don't know" detector — when retrieved chunks aren't relevant enough
4. Compare retrieval quality: single query vs multi-query on 10 test questions
