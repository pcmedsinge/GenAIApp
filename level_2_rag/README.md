# Level 2: RAG Systems (Retrieval-Augmented Generation)
**Connect LLMs to YOUR data**

## Overview

RAG is the most important pattern in enterprise GenAI. It solves the #1 problem with LLMs:
**they don't know YOUR data** (your organization's policies, medical guidelines, patient records, etc.)

### What is RAG?

```
Without RAG:  User → LLM → Answer (from training data only, may be wrong/outdated)
With RAG:     User → Retrieve relevant docs → LLM + docs → Grounded answer with citations
```

### Why RAG Before Agents?
- RAG is the foundation of most real-world GenAI applications
- Agents (Level 3) USE RAG as a tool — you need to understand it first
- In healthcare, RAG ensures answers are grounded in actual guidelines/evidence

## Prerequisites
- **Level 1 Complete**: Especially 02_embeddings (you'll use those skills directly)
- **OpenAI API Key**: Configured in .env
- **ChromaDB**: `pip install chromadb` (already in requirements.txt)

## Projects

### 01_simple_rag — RAG from Scratch
- Build the entire RAG pipeline with just OpenAI API (no frameworks)
- Understand: Chunk → Embed → Store → Retrieve → Generate
- See RAG vs No-RAG comparison side by side
- **Healthcare Example**: Medical guidelines Q&A

### 02_vector_databases — ChromaDB Mastery
- Replace in-memory storage with a proper vector database
- CRUD operations, metadata filtering, persistent storage
- Understand why vector databases matter at scale
- **Healthcare Example**: Medical literature collection

### 03_document_processing — Loading & Chunking
- Load real documents (text files, structured data)
- Chunking strategies: fixed-size, sentence-based, semantic
- Overlap strategies and why they matter
- **Healthcare Example**: Clinical guideline documents

### 04_advanced_retrieval — Beyond Basic Search
- Multi-query retrieval (rephrase for better results)
- Re-ranking with LLM (improve precision)
- Source citations and answer grounding
- **Healthcare Example**: Multi-source medical Q&A

### 05_medical_rag — Capstone: Healthcare Knowledge Base
- Full production-pattern RAG system
- Multiple document types and sources
- Interactive Q&A with citations and confidence
- **Healthcare Example**: Complete clinical guidelines assistant

## Learning Objectives

After completing Level 2, you will:
- ✅ Build RAG systems from scratch AND with frameworks
- ✅ Use vector databases (ChromaDB) for scalable storage
- ✅ Process and chunk documents effectively
- ✅ Apply advanced retrieval techniques (re-ranking, multi-query)
- ✅ Cite sources and ground LLM answers in your data
- ✅ Build a healthcare knowledge base Q&A system

## Time Estimate
8-10 hours total (1.5-2 hours per project)

## How This Connects to Level 3
In Level 3 (Agents), your agents will USE RAG as a tool:
```
Agent receives question → Searches knowledge base (RAG!) → Uses result to reason → Takes action
```

## Next Steps
After mastering RAG, you'll be ready for Level 3 (AI Agents) — where you build autonomous systems that think, plan, and act!
