# Project 1: RAG from Scratch

## What You'll Learn
- The complete RAG pipeline without any frameworks
- How chunking, embedding, retrieval, and generation work together
- Why RAG produces better answers than LLM alone
- How to build on your Level 1 embeddings knowledge

## The RAG Pipeline

```
1. LOAD     → Get your documents
2. CHUNK    → Split into smaller pieces
3. EMBED    → Convert chunks to vectors (Level 1 skill!)
4. STORE    → Keep vectors in memory (for now)
5. RETRIEVE → Find relevant chunks for a question
6. GENERATE → LLM answers using retrieved context
```

## Running the Code

```bash
cd level_2_rag/01_simple_rag
python main.py
```

## Demos
1. **Full RAG Pipeline** — Step-by-step walkthrough of every stage
2. **RAG vs No-RAG** — Side-by-side comparison showing why RAG matters
3. **Interactive Q&A** — Ask your own questions against the knowledge base

## Healthcare Context
Uses 6 medical guideline documents (hypertension, diabetes, asthma, CKD, anticoagulation, depression) as the knowledge base.

## Exercises
1. Add a new medical document to the knowledge base and test retrieval
2. Experiment with chunk sizes (50, 100, 200 words) — how does it affect answer quality?
3. Try different `top_k` values (1, 3, 5) — more context vs more noise?
4. Add a "confidence score" that shows how relevant the retrieved chunks are

## Next Steps
Move to 02_vector_databases to replace in-memory storage with ChromaDB for persistent, scalable vector storage.
