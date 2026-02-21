# Project 3: Document Processing (Loading & Chunking)

## What You'll Learn
- How to load and process real documents
- Chunking strategies: fixed-size, sentence-based, paragraph-based
- Why chunk overlap matters (and how much to use)
- How chunking quality affects retrieval quality

## Why Chunking Matters

Chunking is where most RAG systems succeed or fail:
- **Too large** → Retrieved chunks contain noise, LLM gets confused
- **Too small** → Important context split across chunks, lost information
- **No overlap** → Critical sentences at boundaries get cut
- **Right size** → Precise retrieval, focused answers

## Running the Code

```bash
cd level_2_rag/03_document_processing
python main.py
```

## Demos
1. **Chunking Strategies Compared** — See fixed-size, sentence-based, and paragraph-based side by side
2. **Overlap Visualization** — See exactly what overlap looks like and why it helps
3. **Retrieval Quality Test** — Same question, different chunking — which finds better answers?
4. **Process Your Own Text** — Input any medical text and see it chunked

## Exercises
1. Chunk a clinical guideline with different sizes (50, 100, 200 words) and compare retrieval
2. Implement semantic chunking (split at topic boundaries, not arbitrary points)
3. Add metadata to each chunk (source document, section header, page number)
4. Compare retrieval accuracy across all chunking strategies using 10 test questions
