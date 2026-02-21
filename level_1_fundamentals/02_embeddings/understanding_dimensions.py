"""
Understanding Embedding Dimensions
What does "1536 dimensions" actually mean?
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text: str, model: str = "text-embedding-3-small"):
    """Get embedding for text"""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def explain_dimensions():
    """
    Show what dimensions actually look like
    """
    print("\n" + "="*70)
    print("UNDERSTANDING EMBEDDING DIMENSIONS")
    print("="*70)
    
    text = "chest pain"
    
    print(f"\n📝 Original text: '{text}'")
    print(f"   Just 2 words, 10 characters")
    
    print("\n🔄 Converting to embedding...")
    embedding = get_embedding(text, model="text-embedding-3-small")
    
    print(f"\n✅ Result: A list of {len(embedding)} numbers!")
    
    print(f"\n📊 First 20 dimensions (out of {len(embedding)}):")
    print("-"*70)
    for i in range(20):
        print(f"   Dimension {i+1:4d}: {embedding[i]:+.6f}")
    
    print(f"\n   ... ({len(embedding) - 40} more dimensions) ...")
    
    print(f"\n📊 Last 20 dimensions:")
    print("-"*70)
    for i in range(len(embedding)-20, len(embedding)):
        print(f"   Dimension {i+1:4d}: {embedding[i]:+.6f}")
    
    print(f"\n💡 Each number is between -1 and +1")
    print(f"   Positive values: Text HAS this feature")
    print(f"   Negative values: Text LACKS this feature")
    print(f"   Close to 0: Feature is neutral")
    
    # Statistics
    print(f"\n📈 Statistics:")
    print(f"   Minimum value: {min(embedding):.6f}")
    print(f"   Maximum value: {max(embedding):.6f}")
    print(f"   Average value: {np.mean(embedding):.6f}")
    print(f"   Standard deviation: {np.std(embedding):.6f}")


def compare_dimensions():
    """
    Compare small model (1536) vs large model (3072) dimensions
    """
    print("\n\n" + "="*70)
    print("SMALL MODEL vs LARGE MODEL DIMENSIONS")
    print("="*70)
    
    text = "patient has severe chest pain"
    
    print(f"\n📝 Text: '{text}'")
    
    # Small model
    print(f"\n🔵 text-embedding-3-small:")
    embedding_small = get_embedding(text, model="text-embedding-3-small")
    print(f"   Dimensions: {len(embedding_small)}")
    print(f"   First 10: {[f'{x:.3f}' for x in embedding_small[:10]]}")
    print(f"   Storage: {len(embedding_small) * 4} bytes ({len(embedding_small) * 4 / 1024:.2f} KB)")
    
    # Large model
    print(f"\n🟣 text-embedding-3-large:")
    embedding_large = get_embedding(text, model="text-embedding-3-large")
    print(f"   Dimensions: {len(embedding_large)}")
    print(f"   First 10: {[f'{x:.3f}' for x in embedding_large[:10]]}")
    print(f"   Storage: {len(embedding_large) * 4} bytes ({len(embedding_large) * 4 / 1024:.2f} KB)")
    
    print(f"\n💡 Large model has {len(embedding_large) - len(embedding_small)} MORE dimensions")
    print(f"   = 2x more detailed representation")
    print(f"   = Can capture more subtle semantic differences")


def why_dimensions_matter():
    """
    Show why dimensions matter with similar texts
    """
    print("\n\n" + "="*70)
    print("WHY MORE DIMENSIONS = BETTER ACCURACY")
    print("="*70)
    
    # Very similar medical terms
    texts = [
        "myocardial infarction",
        "heart attack",
        "cardiac arrest",
    ]
    
    print("\n📝 Medical terms that are SIMILAR but DIFFERENT:")
    for i, text in enumerate(texts, 1):
        print(f"   {i}. {text}")
    
    print("\n" + "-"*70)
    print("Comparing embeddings...")
    print("-"*70)
    
    # Get embeddings
    embeddings_small = [get_embedding(t, "text-embedding-3-small") for t in texts]
    embeddings_large = [get_embedding(t, "text-embedding-3-large") for t in texts]
    
    # Calculate similarities
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    print("\n🔵 Small model (1536 dimensions):")
    sim_small_12 = cosine_similarity(embeddings_small[0], embeddings_small[1])
    sim_small_13 = cosine_similarity(embeddings_small[0], embeddings_small[2])
    print(f"   'myocardial infarction' vs 'heart attack': {sim_small_12:.4f}")
    print(f"   'myocardial infarction' vs 'cardiac arrest': {sim_small_13:.4f}")
    print(f"   Difference: {abs(sim_small_12 - sim_small_13):.4f}")
    
    print("\n🟣 Large model (3072 dimensions):")
    sim_large_12 = cosine_similarity(embeddings_large[0], embeddings_large[1])
    sim_large_13 = cosine_similarity(embeddings_large[0], embeddings_large[2])
    print(f"   'myocardial infarction' vs 'heart attack': {sim_large_12:.4f}")
    print(f"   'myocardial infarction' vs 'cardiac arrest': {sim_large_13:.4f}")
    print(f"   Difference: {abs(sim_large_12 - sim_large_13):.4f}")
    
    print("\n💡 More dimensions = Better at distinguishing subtle differences!")
    print("   MI and heart attack are SAME condition (should be similar)")
    print("   MI and cardiac arrest are DIFFERENT (should be less similar)")


def dimensions_analogy():
    """
    Use an analogy to explain dimensions
    """
    print("\n\n" + "="*70)
    print("ANALOGY: Dimensions are like FEATURES")
    print("="*70)
    
    print("""
🎨 Imagine describing a PAINTING:

Low Dimensions (10 features):
   1. Has blue? (yes/no)
   2. Has red? (yes/no)
   3. Is landscape? (yes/no)
   4. Has people? (yes/no)
   5. Is abstract? (yes/no)
   6. Is dark? (yes/no)
   7. Is large? (yes/no)
   8. Is old? (yes/no)
   9. Has water? (yes/no)
   10. Has buildings? (yes/no)

   ❌ Limited - can't distinguish similar paintings well

High Dimensions (1536 features):
   1. Exact shade of blue at top-left
   2. Exact shade of blue at top-center
   3. Brush stroke style - horizontal
   4. Brush stroke style - vertical
   5. Light intensity - top
   6. Light intensity - bottom
   ... 1,530 more specific features

   ✅ Detailed - can distinguish very similar paintings

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏥 Same for MEDICAL TEXT:

Low Dimensions (10):
   1. Is medical? (0.95)
   2. Is symptom? (0.80)
   3. Is emergency? (0.60)
   ... not enough to capture nuance

High Dimensions (1536):
   1. Relates to cardiovascular? (0.89)
   2. Indicates acute condition? (0.76)
   3. Associated with pain? (0.94)
   4. Mentions left side? (0.45)
   5. Radiating pain pattern? (0.67)
   ... 1,531 more semantic features
   
   ✅ Can distinguish "chest pain" from "chest discomfort"
   ✅ Can distinguish "MI" from "cardiac arrest"
   ✅ Can match "SOB" with "shortness of breath"
""")


def practical_implications():
    """
    Show practical implications of dimensions
    """
    print("\n\n" + "="*70)
    print("PRACTICAL IMPLICATIONS")
    print("="*70)
    
    print("""
📦 STORAGE:
   1536 dimensions × 4 bytes/number = 6,144 bytes (6 KB) per embedding
   
   For 10,000 medical terms:
   - Small model: 10,000 × 6 KB = 60 MB
   - Large model: 10,000 × 12 KB = 120 MB

⚡ SPEED:
   More dimensions = slower computation
   
   Comparing 2 embeddings:
   - 1536 dimensions: ~0.001 seconds (very fast)
   - 3072 dimensions: ~0.002 seconds (still fast)
   
   But for 1 million comparisons:
   - 1536 dimensions: ~1 second
   - 3072 dimensions: ~2 seconds

🎯 ACCURACY:
   More dimensions = more precise matching
   
   1536 dimensions: 85-90% accuracy
   3072 dimensions: 92-95% accuracy
   
   The extra 5% matters for critical applications!

💰 COST:
   Dimensions DON'T affect API cost directly
   Cost is per token (words), not dimensions
   
   But more dimensions → more storage → more hosting costs
""")


def main():
    """
    Run all demonstrations
    """
    print("\n🔢 Understanding Embedding Dimensions")
    print("What does '1536 dimensions' really mean?")
    
    print("\n\nChoose a demo:")
    print("1. See actual dimensions (what they look like)")
    print("2. Compare small (1536) vs large (3072) models")
    print("3. Why more dimensions = better accuracy")
    print("4. Analogy to explain dimensions")
    print("5. Practical implications")
    print("6. Run all demos")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        explain_dimensions()
    elif choice == "2":
        compare_dimensions()
    elif choice == "3":
        why_dimensions_matter()
    elif choice == "4":
        dimensions_analogy()
    elif choice == "5":
        practical_implications()
    elif choice == "6":
        explain_dimensions()
        compare_dimensions()
        why_dimensions_matter()
        dimensions_analogy()
        practical_implications()
    else:
        print("❌ Invalid choice")
    
    print("\n\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
🎯 DIMENSIONS = NUMBER OF NUMBERS IN THE LIST

"1536 dimensions" means:
   ✅ Text converted to list of 1,536 numbers
   ✅ Each number captures a semantic feature
   ✅ Numbers range from -1 to +1
   ✅ Similar texts have similar number patterns

ANALOGIES:
   📍 GPS coordinates: 2 dimensions (latitude, longitude)
   🎨 RGB color: 3 dimensions (red, green, blue)
   📊 Embedding: 1,536 dimensions (semantic features)

WHY SO MANY?
   ✅ Language is complex - needs many features
   ✅ Can distinguish subtle differences
   ✅ "chest pain" ≠ "chest tightness" ≠ "chest discomfort"
   ✅ More dimensions = more precision

SMALL vs LARGE:
   Small (1536): Like describing painting with 1,536 features
   Large (3072): Like describing painting with 3,072 features
   → More features = more detailed = better accuracy

YOU DON'T NEED TO UNDERSTAND EACH DIMENSION:
   ✅ AI figured out the best features automatically
   ✅ You just use the embeddings for similarity
   ✅ Math does the rest (cosine similarity)
""")


if __name__ == "__main__":
    main()
