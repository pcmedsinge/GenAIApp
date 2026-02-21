"""
Project: Embeddings & Semantic Search
Objective: Learn to use embeddings for semantic similarity
Concepts: Vector embeddings, cosine similarity, semantic search

Healthcare Use Case: Find similar medical cases or symptoms
"""

import os
import numpy as np
from openai import OpenAI
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Get embedding vector for a piece of text
    
    Args:
        text: The text to embed
        model: Embedding model to use
            - text-embedding-3-small: 1536 dimensions, $0.02 per 1M tokens
            - text-embedding-3-large: 3072 dimensions, $0.13 per 1M tokens
            
    Returns:
        List of floats representing the embedding vector
    """
    # Remove newlines for better embedding quality
    text = text.replace("\n", " ")
    
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    
    return response.data[0].embedding


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Returns value between -1 and 1:
    - 1.0 = identical
    - 0.0 = unrelated
    - -1.0 = opposite
    """
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    dot_product = np.dot(vec1_np, vec2_np)
    magnitude = np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np)
    
    return dot_product / magnitude


def find_most_similar(
    query: str,
    documents: List[str],
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Find the most similar documents to a query
    
    Args:
        query: The search query
        documents: List of documents to search through
        top_k: Number of results to return
        
    Returns:
        List of (document, similarity_score) tuples
    """
    # Get embedding for the query
    query_embedding = get_embedding(query)
    
    # Get embeddings for all documents
    doc_embeddings = [get_embedding(doc) for doc in documents]
    
    # Calculate similarities
    similarities = []
    for doc, doc_emb in zip(documents, doc_embeddings):
        similarity = cosine_similarity(query_embedding, doc_emb)
        similarities.append((doc, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def medical_case_search():
    """
    Healthcare Example: Search through medical case descriptions
    """
    print("\n" + "="*60)
    print("MEDICAL CASE SIMILARITY SEARCH")
    print("="*60)
    
    # Synthetic medical cases (never use real patient data!)
    medical_cases = [
        "45-year-old male with chest pain radiating to left arm, shortness of breath, and sweating. History of hypertension.",
        "32-year-old female presenting with severe headache, photophobia, and neck stiffness. Fever of 101.5°F.",
        "28-year-old male with sudden onset chest pain that worsens with deep breathing. Recent long flight from Europe.",
        "67-year-old female with gradual onset of chest discomfort, nausea, and unusual fatigue. Diabetic.",
        "19-year-old female with severe unilateral headache, nausea, and visual aura. Family history of migraines.",
        "55-year-old male with crushing chest pain, pain in jaw, extreme anxiety. Heavy smoker.",
        "8-year-old child with fever, severe headache, and confusion. Recently returned from camping trip.",
    ]
    
    # Search queries
    queries = [
        "possible heart attack symptoms",
        "migraine headache",
        "pulmonary embolism"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 60)
        
        results = find_most_similar(query, medical_cases, top_k=3)
        
        for i, (case, score) in enumerate(results, 1):
            print(f"\n{i}. Similarity: {score:.3f}")
            print(f"   {case}")


def symptom_clustering():
    """
    Group similar symptoms together using embeddings
    """
    print("\n" + "="*60)
    print("SYMPTOM CLUSTERING")
    print("="*60)
    
    symptoms = [
        "sharp stabbing chest pain",
        "dull aching chest pain",
        "severe throbbing headache",
        "mild pressure-like headache",
        "crushing chest pressure",
        "pounding headache with nausea"
    ]
    
    print("\nComparing all symptoms:")
    print("-" * 60)
    
    # Get embeddings for all symptoms
    embeddings = {symptom: get_embedding(symptom) for symptom in symptoms}
    
    # Create similarity matrix
    for i, symptom1 in enumerate(symptoms):
        print(f"\n'{symptom1}':")
        similarities = []
        
        for symptom2 in symptoms:
            if symptom1 != symptom2:
                sim = cosine_similarity(
                    embeddings[symptom1],
                    embeddings[symptom2]
                )
                similarities.append((symptom2, sim))
        
        # Sort and show top 2 most similar
        similarities.sort(key=lambda x: x[1], reverse=True)
        for symptom2, score in similarities[:2]:
            print(f"  - {score:.3f}: {symptom2}")


def semantic_vs_keyword():
    """
    Demonstrate semantic search vs keyword search
    Shows why embeddings are powerful
    """
    print("\n" + "="*60)
    print("SEMANTIC vs KEYWORD SEARCH")
    print("="*60)
    
    documents = [
        "Patient reports difficulty breathing and wheezing",
        "Individual experiencing respiratory distress",
        "Person has trouble catching their breath",
        "Patient complains of abdominal pain",
        "Subject has trouble sleeping due to anxiety"
    ]
    
    query = "hard to breathe"
    
    print(f"\n🔍 Query: '{query}'")
    print("\n1. KEYWORD SEARCH (naive string matching):")
    print("-" * 40)
    
    # Keyword search - just check if words appear
    query_words = set(query.lower().split())
    for doc in documents:
        doc_words = set(doc.lower().split())
        matches = query_words & doc_words
        if matches:
            print(f"✓ {doc}")
            print(f"  (matched words: {matches})")
    
    print("\n2. SEMANTIC SEARCH (embeddings):")
    print("-" * 40)
    
    results = find_most_similar(query, documents, top_k=3)
    for doc, score in results:
        print(f"✓ {score:.3f}: {doc}")
    
    print("\n💡 Notice: Semantic search finds related concepts,")
    print("   not just matching keywords!")


def batch_embedding_example():
    """
    Efficiently get embeddings for multiple texts
    Important for performance and cost optimization
    """
    print("\n" + "="*60)
    print("BATCH EMBEDDING (Cost Optimization)")
    print("="*60)
    
    texts = [
        "headache",
        "fever",
        "cough",
        "nausea",
        "fatigue"
    ]
    
    # Batch request - more efficient than individual requests
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    
    print(f"\nEmbedded {len(texts)} texts in one API call")
    print(f"Total tokens used: {response.usage.total_tokens}")
    print(f"Cost: ${(response.usage.total_tokens / 1_000_000) * 0.02:.6f}")
    
    print("\nEmbedding dimensions:", len(response.data[0].embedding))
    
    # Show first few dimensions of each embedding
    print("\nFirst 5 dimensions of each embedding:")
    for i, text in enumerate(texts):
        embedding = response.data[i].embedding[:5]
        print(f"  {text:10} -> {embedding}")


def main():
    """
    Run all examples
    """
    print("\n🏥 Level 1.2: Embeddings & Semantic Search\n")
    
    # Example 1: Medical case search
    #medical_case_search()
    
    # Example 2: Symptom clustering
    # Uncomment to run
    symptom_clustering()
    
    # Example 3: Semantic vs keyword search
    # Uncomment to run
    # semantic_vs_keyword()
    
    # Example 4: Batch embeddings
    # Uncomment to run
    # batch_embedding_example()
    
    print("\n" + "="*60)
    print("✅ Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
