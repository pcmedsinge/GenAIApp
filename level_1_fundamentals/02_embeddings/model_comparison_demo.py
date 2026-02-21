"""
Embedding Model Comparison Demo
Compare text-embedding-3-small vs text-embedding-3-large
"""

import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text: str, model: str = "text-embedding-3-small"):
    """Get embedding for text using specified model"""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / magnitude


def compare_models():
    """
    Compare the two embedding models
    """
    print("\n" + "="*70)
    print("EMBEDDING MODEL COMPARISON")
    print("="*70)
    
    # Test cases - medical terms with subtle differences
    query = "chest pain radiating to left arm"
    
    documents = [
        "Patient has severe chest discomfort spreading to left shoulder",  # Very similar
        "Crushing substernal pain with radiation to jaw",  # Similar
        "Headache and nausea after eating",  # Different
    ]
    
    print(f"\n🔍 Query: '{query}'")
    print(f"\n📚 Documents to compare:")
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. {doc}")
    
    # Test with SMALL model
    print("\n\n" + "="*70)
    print("MODEL 1: text-embedding-3-small")
    print("="*70)
    print(f"Dimensions: 1536")
    print(f"Cost: $0.02 per 1M tokens")
    
    start_time = time.time()
    
    query_embedding_small = get_embedding(query, model="text-embedding-3-small")
    print(f"\n✓ Query embedded: {len(query_embedding_small)} dimensions")
    print(f"  Sample values: {query_embedding_small[:5]}")
    
    results_small = []
    for doc in documents:
        doc_embedding = get_embedding(doc, model="text-embedding-3-small")
        similarity = cosine_similarity(query_embedding_small, doc_embedding)
        results_small.append(similarity)
    
    small_time = time.time() - start_time
    
    print(f"\n📊 Similarity Scores:")
    for i, (doc, score) in enumerate(zip(documents, results_small), 1):
        print(f"   {i}. [{score:.4f}] {doc[:60]}...")
    
    print(f"\n⏱️  Time: {small_time:.3f}s")
    
    # Calculate cost (rough estimate: ~10 tokens per text)
    total_tokens = 10 * (1 + len(documents))  # query + documents
    cost_small = (total_tokens / 1_000_000) * 0.02
    print(f"💰 Cost: ${cost_small:.6f}")
    
    
    # Test with LARGE model
    print("\n\n" + "="*70)
    print("MODEL 2: text-embedding-3-large")
    print("="*70)
    print(f"Dimensions: 3072")
    print(f"Cost: $0.13 per 1M tokens")
    
    start_time = time.time()
    
    query_embedding_large = get_embedding(query, model="text-embedding-3-large")
    print(f"\n✓ Query embedded: {len(query_embedding_large)} dimensions")
    print(f"  Sample values: {query_embedding_large[:5]}")
    
    results_large = []
    for doc in documents:
        doc_embedding = get_embedding(doc, model="text-embedding-3-large")
        similarity = cosine_similarity(query_embedding_large, doc_embedding)
        results_large.append(similarity)
    
    large_time = time.time() - start_time
    
    print(f"\n📊 Similarity Scores:")
    for i, (doc, score) in enumerate(zip(documents, results_large), 1):
        print(f"   {i}. [{score:.4f}] {doc[:60]}...")
    
    print(f"\n⏱️  Time: {large_time:.3f}s")
    
    cost_large = (total_tokens / 1_000_000) * 0.13
    print(f"💰 Cost: ${cost_large:.6f}")
    
    
    # Comparison
    print("\n\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    print(f"\n📏 Dimensions:")
    print(f"   Small: 1536 | Large: 3072 (2x more detailed)")
    
    print(f"\n💰 Cost:")
    print(f"   Small: ${cost_small:.6f} | Large: ${cost_large:.6f} ({cost_large/cost_small:.1f}x more)")
    
    print(f"\n⏱️  Speed:")
    print(f"   Small: {small_time:.3f}s | Large: {large_time:.3f}s ({large_time/small_time:.1f}x slower)")
    
    print(f"\n🎯 Accuracy Difference:")
    for i, (doc, small_score, large_score) in enumerate(zip(documents, results_small, results_large), 1):
        diff = abs(large_score - small_score)
        print(f"   Doc {i}: Small={small_score:.4f}, Large={large_score:.4f} (Δ={diff:.4f})")
    
    avg_diff = np.mean([abs(s - l) for s, l in zip(results_small, results_large)])
    print(f"\n   Average difference: {avg_diff:.4f}")
    
    if avg_diff < 0.02:
        print(f"   💡 Minimal difference - small model is sufficient!")
    elif avg_diff < 0.05:
        print(f"   ⚖️  Small difference - consider use case")
    else:
        print(f"   ⚠️  Significant difference - large model may be worth it")


def storage_comparison():
    """
    Compare storage requirements
    """
    print("\n\n" + "="*70)
    print("STORAGE REQUIREMENTS")
    print("="*70)
    
    num_embeddings = 10_000  # 10,000 medical terms
    
    # Each float is 4 bytes
    small_size = 1536 * 4 * num_embeddings / (1024 * 1024)  # MB
    large_size = 3072 * 4 * num_embeddings / (1024 * 1024)  # MB
    
    print(f"\n📦 Storage for {num_embeddings:,} embeddings:")
    print(f"   Small model: {small_size:.2f} MB")
    print(f"   Large model: {large_size:.2f} MB ({large_size/small_size:.1f}x more)")
    
    print(f"\n💾 For 1 million medical terms:")
    print(f"   Small model: {small_size * 100:.2f} MB ({small_size * 100 / 1024:.2f} GB)")
    print(f"   Large model: {large_size * 100:.2f} MB ({large_size * 100 / 1024:.2f} GB)")


def use_case_recommendations():
    """
    Recommend which model to use for different scenarios
    """
    print("\n\n" + "="*70)
    print("WHEN TO USE EACH MODEL")
    print("="*70)
    
    scenarios = [
        {
            "use_case": "Drug Interaction Search",
            "volume": "High (10,000+ searches/day)",
            "accuracy_need": "Good (85%+)",
            "budget": "Limited",
            "recommendation": "✅ text-embedding-3-small",
            "reason": "High volume, good accuracy sufficient, cost-effective"
        },
        {
            "use_case": "Medical Literature Search",
            "volume": "Medium (1,000 searches/day)",
            "accuracy_need": "High (90%+)",
            "budget": "Moderate",
            "recommendation": "🔄 Consider large model",
            "reason": "Precision matters, moderate volume, can justify cost"
        },
        {
            "use_case": "Clinical Decision Support",
            "volume": "Low (100 searches/day)",
            "accuracy_need": "Critical (95%+)",
            "budget": "High",
            "recommendation": "✅ text-embedding-3-large",
            "reason": "Patient safety critical, low volume, accuracy paramount"
        },
        {
            "use_case": "Patient Education Materials",
            "volume": "High (5,000+ searches/day)",
            "accuracy_need": "Adequate (80%+)",
            "budget": "Low",
            "recommendation": "✅ text-embedding-3-small",
            "reason": "High volume, basic matching sufficient, budget constrained"
        },
        {
            "use_case": "ICD-10 Code Suggestion",
            "volume": "Very High (50,000+ searches/day)",
            "accuracy_need": "High (88%+)",
            "budget": "Limited",
            "recommendation": "✅ text-embedding-3-small",
            "reason": "Massive volume, validated by coders, cost prohibitive at scale"
        },
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'-'*70}")
        print(f"Scenario {i}: {scenario['use_case']}")
        print(f"{'-'*70}")
        print(f"   Volume: {scenario['volume']}")
        print(f"   Accuracy needed: {scenario['accuracy_need']}")
        print(f"   Budget: {scenario['budget']}")
        print(f"\n   🎯 RECOMMENDATION: {scenario['recommendation']}")
        print(f"   💡 Why: {scenario['reason']}")


def main():
    """
    Run all demonstrations
    """
    print("\n🔬 Embedding Model Comparison")
    print("Understanding text-embedding-3-small vs text-embedding-3-large")
    
    print("\n\nChoose a demo:")
    print("1. Compare models on same data")
    print("2. Storage requirements comparison")
    print("3. Use case recommendations")
    print("4. Run all demos")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        compare_models()
    elif choice == "2":
        storage_comparison()
    elif choice == "3":
        use_case_recommendations()
    elif choice == "4":
        compare_models()
        storage_comparison()
        use_case_recommendations()
    else:
        print("❌ Invalid choice")
    
    print("\n\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
📊 text-embedding-3-small (DEFAULT):
   ✅ 1536 dimensions
   ✅ $0.02 per 1M tokens (cheap)
   ✅ Fast processing
   ✅ 85-90% accuracy
   ✅ Good for most use cases
   💡 RECOMMENDED for: high volume, budget-conscious, adequate accuracy

🎯 text-embedding-3-large:
   ✅ 3072 dimensions (2x detail)
   ✅ $0.13 per 1M tokens (6.5x more expensive)
   ✅ Slower processing
   ✅ 92-95% accuracy
   ✅ Better for subtle distinctions
   💡 RECOMMENDED for: critical accuracy, low volume, patient safety

RULE OF THUMB:
- Start with SMALL model
- Measure accuracy for your use case
- Upgrade to LARGE only if:
  * Accuracy is insufficient
  * Volume is low (cost acceptable)
  * Accuracy is business-critical

COST EXAMPLE (10,000 medical terms):
- Small: $0.0002 to embed all
- Large: $0.0013 to embed all
- After embedding, searches are FREE (local computation)
""")


if __name__ == "__main__":
    main()
