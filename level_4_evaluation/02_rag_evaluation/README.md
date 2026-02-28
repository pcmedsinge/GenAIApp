# Project 2: RAG Evaluation with RAGAS

## What You'll Learn
- Measure RAG quality with RAGAS framework
- Evaluate retrieval quality (precision, recall, MRR)
- Evaluate generation quality (faithfulness, relevancy)
- Build end-to-end evaluation pipelines

## Why RAG Evaluation?
Building a RAG system is easy. Building a GOOD RAG system requires measurement.

```
RAG Quality = Retrieval Quality × Generation Quality

Retrieval: Did we find the right documents?
Generation: Did we generate a faithful, relevant answer?
```

## Running the Code

```bash
cd level_4_evaluation/02_rag_evaluation
python main.py
```

## Demos
1. **Retrieval Metrics** — precision@k, recall@k, MRR on medical queries
2. **RAGAS Faithfulness** — Does the answer stay faithful to retrieved context?
3. **RAGAS Relevancy** — Is the answer actually relevant to the question?
4. **End-to-End Pipeline** — Complete evaluation over a test dataset

## Exercises
1. Evaluate your Level 2 medical RAG with RAGAS metrics
2. Compare chunk sizes (256 vs 512 vs 1024) using evaluation metrics
3. Build a retrieval quality dashboard showing precision/recall
4. Create a test dataset with ground truth answers for evaluation

## Key Concepts
- **Faithfulness**: Answer uses ONLY information from retrieved context
- **Answer Relevancy**: Answer directly addresses the question asked
- **Context Precision**: Retrieved docs are relevant (not noise)
- **Context Recall**: All needed information is in retrieved docs
