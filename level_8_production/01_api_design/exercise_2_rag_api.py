"""
Exercise 2 — RAG API
=====================
Build a document-management and retrieval-augmented generation API:

    POST   /documents          — ingest a new document (chunk + embed + store)
    GET    /documents          — list all stored documents
    DELETE /documents/{doc_id} — remove a document
    POST   /query              — semantic search + LLM answer

All endpoints use Pydantic models for request/response validation.

Usage:
    uvicorn exercise_2_rag_api:app --reload --port 8002

Test:
    # Add a document
    curl -X POST http://localhost:8002/documents \
         -H 'Content-Type: application/json' \
         -d '{"title": "Diabetes Guide", "content": "Type 2 diabetes is ..."}'

    # Query
    curl -X POST http://localhost:8002/query \
         -H 'Content-Type: application/json' \
         -d '{"query": "What is type 2 diabetes?"}'
"""

import os
import uuid
import time
import math
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, HTTPException, status
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️  Install FastAPI: pip install fastapi uvicorn pydantic")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  Install openai: pip install openai")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
if FASTAPI_AVAILABLE:

    class DocumentCreate(BaseModel):
        """Body for POST /documents."""
        title: str = Field(..., min_length=1, max_length=200)
        content: str = Field(..., min_length=10, max_length=50_000)
        metadata: Optional[dict] = None

    class DocumentRecord(BaseModel):
        """Stored document representation."""
        id: str
        title: str
        content: str
        metadata: Optional[dict]
        chunk_count: int
        created_at: str

    class QueryRequest(BaseModel):
        """Body for POST /query."""
        query: str = Field(..., min_length=1, max_length=2000)
        top_k: int = Field(3, ge=1, le=10)
        model: str = Field("gpt-4o-mini")

    class QueryResult(BaseModel):
        """A single retrieval result."""
        chunk: str
        score: float
        document_title: str

    class QueryResponse(BaseModel):
        """Full RAG response."""
        answer: str
        sources: list[QueryResult]
        model: str
        latency_ms: float

    class DeleteResponse(BaseModel):
        detail: str

    # -------------------------------------------------------------------
    # In-memory store (replace with a real vector DB in production)
    # -------------------------------------------------------------------
    DOCUMENTS: dict[str, dict] = {}
    CHUNKS: list[dict] = []  # {"doc_id", "text", "embedding", "title"}

    def _get_client() -> OpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
        return OpenAI(api_key=api_key)

    def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def _embed(client: OpenAI, texts: list[str]) -> list[list[float]]:
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [e.embedding for e in resp.data]

    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    # -------------------------------------------------------------------
    # App + endpoints
    # -------------------------------------------------------------------
    app = FastAPI(title="RAG API", version="1.0.0")

    @app.post("/documents", response_model=DocumentRecord, status_code=201)
    def add_document(doc: DocumentCreate):
        """Chunk, embed, and store a document."""
        client = _get_client()
        doc_id = str(uuid.uuid4())
        chunks = _chunk_text(doc.content)
        embeddings = _embed(client, chunks)

        for chunk_text, emb in zip(chunks, embeddings):
            CHUNKS.append({
                "doc_id": doc_id,
                "text": chunk_text,
                "embedding": emb,
                "title": doc.title,
            })

        record = {
            "id": doc_id,
            "title": doc.title,
            "content": doc.content,
            "metadata": doc.metadata,
            "chunk_count": len(chunks),
            "created_at": datetime.utcnow().isoformat(),
        }
        DOCUMENTS[doc_id] = record
        return DocumentRecord(**record)

    @app.get("/documents", response_model=list[DocumentRecord])
    def list_documents():
        """List all ingested documents."""
        return [DocumentRecord(**d) for d in DOCUMENTS.values()]

    @app.delete("/documents/{doc_id}", response_model=DeleteResponse)
    def delete_document(doc_id: str):
        """Remove a document and its chunks."""
        if doc_id not in DOCUMENTS:
            raise HTTPException(status_code=404, detail="Document not found")
        del DOCUMENTS[doc_id]
        # Remove associated chunks
        global CHUNKS
        CHUNKS = [c for c in CHUNKS if c["doc_id"] != doc_id]
        return DeleteResponse(detail=f"Document {doc_id} deleted")

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest):
        """Semantic search over stored chunks + LLM answer."""
        if not CHUNKS:
            raise HTTPException(status_code=400, detail="No documents ingested yet")

        client = _get_client()
        start = time.perf_counter()

        query_emb = _embed(client, [req.query])[0]
        scored = [
            (c, _cosine(query_emb, c["embedding"]))
            for c in CHUNKS
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: req.top_k]

        context = "\n\n".join(f"[{c['title']}]: {c['text']}" for c, _ in top)
        prompt = (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {req.query}\n\n"
            f"Answer concisely. If the context doesn't contain the answer, say so."
        )

        resp = client.chat.completions.create(
            model=req.model,
            messages=[
                {"role": "system", "content": "You are a medical information assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
        )

        latency = (time.perf_counter() - start) * 1000
        sources = [
            QueryResult(chunk=c["text"][:200], score=round(s, 4), document_title=c["title"])
            for c, s in top
        ]
        return QueryResponse(
            answer=resp.choices[0].message.content,
            sources=sources,
            model=req.model,
            latency_ms=round(latency, 1),
        )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "documents": len(DOCUMENTS),
            "chunks": len(CHUNKS),
        }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
def _local_test():
    print("=" * 60)
    print("Exercise 2 — RAG API (local test)")
    print("=" * 60)

    if not OPENAI_AVAILABLE:
        print("OpenAI SDK not available — skipping.")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY to run the test.")
        return

    client = OpenAI(api_key=api_key)

    # Simulate document ingestion
    doc_text = (
        "Hypertension, also known as high blood pressure, is a chronic condition "
        "where the blood pressure in the arteries is persistently elevated. "
        "Normal blood pressure is below 120/80 mmHg. Hypertension is diagnosed "
        "when readings are consistently at or above 130/80 mmHg. Risk factors "
        "include obesity, high sodium intake, lack of exercise, and genetics. "
        "Treatment options include lifestyle changes and medications such as ACE "
        "inhibitors, ARBs, calcium channel blockers, and diuretics."
    )
    chunks = [doc_text[i:i+200] for i in range(0, len(doc_text), 180)]
    embeddings = [e.embedding for e in client.embeddings.create(model="text-embedding-3-small", input=chunks).data]

    print(f"\n📄 Ingested 1 document ({len(chunks)} chunks)")

    query_text = "What medications are used to treat high blood pressure?"
    q_emb = client.embeddings.create(model="text-embedding-3-small", input=[query_text]).data[0].embedding

    scored = sorted(
        [(c, _cosine(q_emb, e)) for c, e in zip(chunks, embeddings)],
        key=lambda x: x[1], reverse=True,
    )
    print(f"\n🔍 Query: {query_text}")
    for i, (chunk, score) in enumerate(scored[:3]):
        print(f"  [{i+1}] score={score:.4f}  {chunk[:80]}…")

    context = "\n".join(c for c, _ in scored[:3])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer using only the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query_text}"},
        ],
        max_tokens=256,
    )
    print(f"\n💬 Answer: {resp.choices[0].message.content}")
    print("\n✅ Local test complete.")


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


if __name__ == "__main__":
    _local_test()
