"""
Exercise 1: Drug Interaction Search System
Find similar drug interactions using semantic search
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding for a text"""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / magnitude


def find_similar_interactions(query, interactions_db, top_k=5):
    """
    Find most similar drug interactions to a query
    
    Args:
        query: Search query (e.g., "blood thinner safety")
        interactions_db: List of drug interaction descriptions
        top_k: Number of results to return
    
    Returns:
        List of (interaction, similarity_score) tuples
    """
    print(f"\n🔍 Searching for: '{query}'")
    print("="*70)
    
    # Get embedding for query
    query_embedding = get_embedding(query)
    
    # Get embeddings for all interactions (in a real app, these would be pre-computed)
    results = []
    for interaction in interactions_db:
        interaction_embedding = get_embedding(interaction)
        similarity = cosine_similarity(query_embedding, interaction_embedding)
        results.append((interaction, similarity))
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top K results
    return results[:top_k]


def demo_basic_search():
    """
    Demo 1: Basic drug interaction search
    """
    print("\n" + "="*70)
    print("DEMO 1: Basic Drug Interaction Search")
    print("="*70)
    
    # Database of known drug interactions
    drug_interactions = [
        "Warfarin and aspirin increase bleeding risk significantly",
        "ACE inhibitors and NSAIDs may cause acute kidney injury",
        "Metformin and contrast dye can lead to lactic acidosis",
        "SSRIs and NSAIDs increase gastrointestinal bleeding risk",
        "Statins and fibrates may cause severe muscle damage",
        "Beta-blockers and calcium channel blockers can cause bradycardia",
        "Lithium and diuretics increase lithium toxicity risk",
        "Digoxin and amiodarone require dose adjustment for toxicity",
        "Macrolides and statins increase risk of rhabdomyolysis",
        "Potassium-sparing diuretics and ACE inhibitors cause hyperkalemia",
        "Clopidogrel and proton pump inhibitors reduce antiplatelet effect",
        "Tamoxifen and SSRIs may reduce cancer treatment efficacy",
    ]
    
    # Search query
    query = "blood thinner safety"
    
    results = find_similar_interactions(query, drug_interactions, top_k=3)
    
    print("\nTop 3 Results:")
    print("-"*70)
    for i, (interaction, score) in enumerate(results, 1):
        print(f"\n{i}. Similarity: {score:.3f}")
        print(f"   {interaction}")


def demo_multiple_queries():
    """
    Demo 2: Multiple related queries
    """
    print("\n\n" + "="*70)
    print("DEMO 2: Multiple Queries - Same Database")
    print("="*70)
    
    drug_interactions = [
        "Warfarin and aspirin increase bleeding risk significantly",
        "ACE inhibitors and NSAIDs may cause acute kidney injury",
        "Metformin and contrast dye can lead to lactic acidosis",
        "SSRIs and NSAIDs increase gastrointestinal bleeding risk",
        "Statins and fibrates may cause severe muscle damage",
        "Beta-blockers and calcium channel blockers can cause bradycardia",
        "Lithium and diuretics increase lithium toxicity risk",
        "Digoxin and amiodarone require dose adjustment for toxicity",
        "Macrolides and statins increase risk of rhabdomyolysis",
        "Potassium-sparing diuretics and ACE inhibitors cause hyperkalemia",
        "Clopidogrel and proton pump inhibitors reduce antiplatelet effect",
        "Tamoxifen and SSRIs may reduce cancer treatment efficacy",
        "MAOIs and SSRIs can cause serotonin syndrome",
        "Aminoglycosides and loop diuretics increase hearing loss risk",
        "Theophylline and fluoroquinolones can cause seizures",
    ]
    
    queries = [
        "heart rate too slow",
        "kidney problems",
        "muscle pain and weakness",
        "increased potassium levels",
    ]
    
    for query in queries:
        results = find_similar_interactions(query, drug_interactions, top_k=2)
        
        print(f"\nTop 2 matches:")
        for i, (interaction, score) in enumerate(results, 1):
            print(f"  {i}. [{score:.3f}] {interaction}")


def demo_patient_medications():
    """
    Demo 3: Check patient's medication list for interactions
    """
    print("\n\n" + "="*70)
    print("DEMO 3: Patient Medication Safety Check")
    print("="*70)
    
    # Known interactions
    interactions_db = [
        "Warfarin and aspirin increase bleeding risk significantly",
        "ACE inhibitors and NSAIDs may cause acute kidney injury",
        "Metformin and contrast dye can lead to lactic acidosis",
        "SSRIs and NSAIDs increase gastrointestinal bleeding risk",
        "Statins and fibrates may cause severe muscle damage",
        "Beta-blockers and calcium channel blockers can cause bradycardia",
        "Lithium and diuretics increase lithium toxicity risk",
        "Digoxin and amiodarone require dose adjustment for toxicity",
        "MAOIs and SSRIs can cause serotonin syndrome",
        "Potassium-sparing diuretics and ACE inhibitors cause hyperkalemia",
    ]
    
    # Patient's current medications
    patient_meds = [
        "Lisinopril",  # ACE inhibitor
        "Ibuprofen",   # NSAID
    ]
    
    print(f"\nPatient Medications: {', '.join(patient_meds)}")
    print("\nChecking for potential interactions...")
    
    # Create query from patient's medications
    query = f"interaction between {' and '.join(patient_meds)}"
    
    results = find_similar_interactions(query, interactions_db, top_k=3)
    
    print("\n⚠️  Potential Interactions Found:")
    print("-"*70)
    for i, (interaction, score) in enumerate(results, 1):
        risk_level = "🔴 HIGH" if score > 0.75 else "🟡 MODERATE" if score > 0.65 else "🟢 LOW"
        print(f"\n{i}. {risk_level} - Similarity: {score:.3f}")
        print(f"   {interaction}")


def demo_category_search():
    """
    Demo 4: Search by interaction category/type
    """
    print("\n\n" + "="*70)
    print("DEMO 4: Search by Interaction Type")
    print("="*70)
    
    interactions_db = [
        "Warfarin and aspirin increase bleeding risk significantly",
        "ACE inhibitors and NSAIDs may cause acute kidney injury",
        "Metformin and contrast dye can lead to lactic acidosis",
        "SSRIs and NSAIDs increase gastrointestinal bleeding risk",
        "Statins and fibrates may cause severe muscle damage (rhabdomyolysis)",
        "Beta-blockers and calcium channel blockers can cause bradycardia",
        "Lithium and diuretics increase lithium toxicity risk",
        "Digoxin and amiodarone require dose adjustment for toxicity",
        "Macrolides and statins increase risk of rhabdomyolysis",
        "Potassium-sparing diuretics and ACE inhibitors cause hyperkalemia",
        "Aminoglycosides and loop diuretics increase ototoxicity (hearing loss)",
        "NSAIDs and corticosteroids increase GI bleeding and ulcer risk",
    ]
    
    categories = {
        "Bleeding complications": "bleeding hemorrhage blood loss",
        "Kidney problems": "renal kidney nephrotoxicity acute injury",
        "Muscle damage": "muscle pain weakness rhabdomyolysis myopathy",
        "Heart rhythm": "bradycardia tachycardia arrhythmia heart rate",
    }
    
    for category, query in categories.items():
        print(f"\n📁 Category: {category}")
        print("-"*70)
        results = find_similar_interactions(query, interactions_db, top_k=2)
        for i, (interaction, score) in enumerate(results, 1):
            print(f"  {i}. [{score:.3f}] {interaction}")


def interactive_search():
    """
    Interactive mode - user can search for drug interactions
    """
    print("\n\n" + "="*70)
    print("INTERACTIVE DRUG INTERACTION SEARCH")
    print("="*70)
    
    interactions_db = [
        "Warfarin and aspirin increase bleeding risk significantly",
        "ACE inhibitors and NSAIDs may cause acute kidney injury",
        "Metformin and contrast dye can lead to lactic acidosis",
        "SSRIs and NSAIDs increase gastrointestinal bleeding risk",
        "Statins and fibrates may cause severe muscle damage",
        "Beta-blockers and calcium channel blockers can cause bradycardia",
        "Lithium and diuretics increase lithium toxicity risk",
        "Digoxin and amiodarone require dose adjustment for toxicity",
        "Macrolides and statins increase risk of rhabdomyolysis",
        "Potassium-sparing diuretics and ACE inhibitors cause hyperkalemia",
        "Clopidogrel and proton pump inhibitors reduce antiplatelet effect",
        "Tamoxifen and SSRIs may reduce cancer treatment efficacy",
        "MAOIs and SSRIs can cause serotonin syndrome",
        "Aminoglycosides and loop diuretics increase hearing loss risk",
        "Theophylline and fluoroquinolones can cause seizures",
        "NSAIDs and corticosteroids increase GI bleeding and ulcer risk",
    ]
    
    print("\n💊 Drug Interaction Database loaded")
    print(f"   {len(interactions_db)} interactions available")
    
    while True:
        print("\n" + "-"*70)
        query = input("\n🔍 Enter search query (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            print("⚠️  Please enter a search query")
            continue
        
        try:
            results = find_similar_interactions(query, interactions_db, top_k=5)
            
            print("\nTop 5 Results:")
            for i, (interaction, score) in enumerate(results, 1):
                risk = "🔴 HIGH" if score > 0.75 else "🟡 MODERATE" if score > 0.65 else "🟢 LOW"
                print(f"\n{i}. {risk} - Similarity: {score:.3f}")
                print(f"   {interaction}")
        
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """
    Run all demonstrations
    """
    print("\n💊 Drug Interaction Search System")
    print("="*70)
    print("Using embeddings to find similar drug interactions")
    
    print("\n\nChoose a demo:")
    print("1. Basic drug interaction search")
    print("2. Multiple queries on same database")
    print("3. Patient medication safety check")
    print("4. Search by interaction category")
    print("5. Interactive search mode")
    print("6. Run all demos")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        demo_basic_search()
    elif choice == "2":
        demo_multiple_queries()
    elif choice == "3":
        demo_patient_medications()
    elif choice == "4":
        demo_category_search()
    elif choice == "5":
        interactive_search()
    elif choice == "6":
        demo_basic_search()
        demo_multiple_queries()
        demo_patient_medications()
        demo_category_search()
        print("\n\n" + "="*70)
        print("Would you like to try interactive search? (y/n): ", end="")
        if input().strip().lower() == 'y':
            interactive_search()
    else:
        print("❌ Invalid choice")
    
    print("\n\n" + "="*70)
    print("KEY LEARNINGS")
    print("="*70)
    print("""
✅ Semantic search finds interactions even with different words
✅ "blood thinner" matches "Warfarin and aspirin" (no exact match!)
✅ "kidney problems" matches "acute kidney injury" automatically
✅ Similarity scores help prioritize high-risk interactions
✅ Can search by drug names, symptoms, or categories

PRODUCTION TIPS:
- Pre-compute and cache embeddings for known interactions
- Use vector database for large interaction databases
- Combine with rule-based checks for critical interactions
- Always review results with clinical pharmacist
- Update database regularly with new interaction data
""")


if __name__ == "__main__":
    main()
