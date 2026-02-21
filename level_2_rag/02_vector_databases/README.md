# Project 2: Vector Databases (ChromaDB)

## What You'll Learn
- Why vector databases exist (limitations of in-memory storage)
- ChromaDB: create collections, add/query/update/delete documents
- Metadata filtering (search by category, date, etc.)
- Persistent storage (data survives program restarts)

## Why Vector Databases?

| Feature | In-Memory (Project 01) | ChromaDB |
|---------|----------------------|----------|
| Speed | Slow (linear scan) | Fast (indexed) |
| Scale | ~1000 docs | Millions |
| Persistence | Lost on restart | Saved to disk |
| Filtering | Manual code | Built-in metadata queries |
| Embedding | You manage | Auto-generates for you |

## Running the Code

```bash
cd level_2_rag/02_vector_databases
python main.py
```

## Demos
1. **ChromaDB Basics** — Create collection, add documents, query
2. **Metadata Filtering** — Filter by category, combine with similarity
3. **Persistent Storage** — Save and reload collections
4. **Full RAG with ChromaDB** — Replace Project 01's in-memory approach

## Exercises
1. Add 3 new medical documents and query across all categories
2. Build a metadata filter for "last_updated" to only search recent guidelines
3. Create separate collections for different departments (cardiology, psychiatry, etc.)
4. Compare query speed: ChromaDB vs the in-memory approach from Project 01
