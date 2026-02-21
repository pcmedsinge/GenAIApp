"""
Project 2: Vector Databases with ChromaDB
Replace in-memory storage with a proper vector database.

Why ChromaDB?
  - Free and open-source
  - Runs locally (no cloud account needed)
  - Handles embeddings automatically (or you can provide your own)
  - Supports metadata filtering
  - Persistent storage to disk
  - Production-ready (used by many companies)

Builds on: Project 01 (simple_rag) — same concepts, better storage
"""

import os
import shutil
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ChromaDB can use OpenAI embeddings directly!
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# Sample medical documents (same knowledge, different format for ChromaDB)
DOCUMENTS = [
    {
        "id": "hypertension_1",
        "text": "Hypertension is defined as blood pressure consistently at or above 130/80 mmHg. Stage 1 hypertension is 130-139 systolic or 80-89 diastolic. Stage 2 is 140/90 or higher. First-line treatments include lifestyle modifications: DASH diet, regular exercise 150 minutes per week moderate intensity, weight management, sodium restriction less than 2300mg per day.",
        "metadata": {"category": "cardiology", "topic": "hypertension", "doc_type": "guideline"}
    },
    {
        "id": "hypertension_2",
        "text": "Pharmacological therapy for hypertension: ACE inhibitors (lisinopril, enalapril), ARBs (losartan, valsartan), calcium channel blockers (amlodipine), thiazide diuretics (hydrochlorothiazide). Target BP for most adults less than 130/80. Monitor kidney function and electrolytes when starting ACE inhibitor or ARB therapy. Combination therapy often needed for Stage 2.",
        "metadata": {"category": "cardiology", "topic": "hypertension", "doc_type": "guideline"}
    },
    {
        "id": "diabetes_1",
        "text": "Type 2 Diabetes diagnosis: Fasting glucose 126 mg/dL or higher, HbA1c 6.5 percent or higher, or random glucose 200 or higher with symptoms. First-line therapy is Metformin 500mg starting dose, titrate to 2000mg daily as tolerated. HbA1c target less than 7 percent for most adults.",
        "metadata": {"category": "endocrinology", "topic": "diabetes", "doc_type": "guideline"}
    },
    {
        "id": "diabetes_2",
        "text": "If HbA1c not at target after 3 months on metformin, add second agent. GLP-1 receptor agonists (semaglutide, liraglutide) preferred if cardiovascular disease or obesity. SGLT2 inhibitors (empagliflozin, dapagliflozin) preferred if heart failure or CKD. Monitor HbA1c every 3 months until stable.",
        "metadata": {"category": "endocrinology", "topic": "diabetes", "doc_type": "guideline"}
    },
    {
        "id": "asthma_1",
        "text": "Asthma stepwise therapy: Step 1 mild intermittent uses as-needed low-dose ICS-formoterol or SABA. Step 2 mild persistent uses low-dose ICS daily. Step 3 moderate uses low-dose ICS-LABA combination. Step 4 severe uses medium to high-dose ICS-LABA. Step 5 very severe adds tiotropium or biologics.",
        "metadata": {"category": "pulmonology", "topic": "asthma", "doc_type": "guideline"}
    },
    {
        "id": "ckd_1",
        "text": "Chronic Kidney Disease staging based on GFR: Stage 1 GFR 90+ with kidney damage, Stage 2 GFR 60-89, Stage 3a GFR 45-59, Stage 3b GFR 30-44, Stage 4 GFR 15-29, Stage 5 GFR less than 15. Referral to nephrology at Stage 4 or rapidly declining GFR.",
        "metadata": {"category": "nephrology", "topic": "ckd", "doc_type": "guideline"}
    },
    {
        "id": "ckd_2",
        "text": "CKD management: Control blood pressure target less than 130/80, manage diabetes HbA1c less than 7 percent, ACE inhibitor or ARB for proteinuria. Avoid nephrotoxic drugs: NSAIDs, aminoglycosides, IV contrast. Monitor GFR every 3-6 months. Prepare for dialysis when GFR below 20.",
        "metadata": {"category": "nephrology", "topic": "ckd", "doc_type": "guideline"}
    },
    {
        "id": "depression_1",
        "text": "Depression screening using PHQ-9: Score 0-4 minimal, 5-9 mild, 10-14 moderate, 15-19 moderately severe, 20-27 severe. SSRIs first-line for moderate depression: sertraline 50mg, escitalopram 10mg, fluoxetine 20mg. Allow 4-6 weeks for initial response.",
        "metadata": {"category": "psychiatry", "topic": "depression", "doc_type": "guideline"}
    },
    {
        "id": "anticoag_1",
        "text": "Anticoagulation for atrial fibrillation: DOACs preferred over warfarin. Apixaban 5mg twice daily, rivaroxaban 20mg daily, dabigatran 150mg twice daily. Dose adjust for renal impairment. Warfarin target INR 2-3 for AF, 2.5-3.5 for mechanical valves.",
        "metadata": {"category": "hematology", "topic": "anticoagulation", "doc_type": "guideline"}
    },
]


# ============================================================
# DEMO 1: ChromaDB Basics
# ============================================================

def demo_chromadb_basics():
    """Create a collection, add documents, query"""
    print("\n" + "=" * 70)
    print("DEMO 1: ChromaDB BASICS")
    print("=" * 70)

    # Create an in-memory ChromaDB client
    chroma_client = chromadb.Client()  # In-memory (temporary)

    # Create a collection (like a table in a database)
    print("\n📦 Creating collection 'medical_guidelines'...")
    collection = chroma_client.create_collection(
        name="medical_guidelines",
        embedding_function=openai_ef,  # ChromaDB will embed for us!
        metadata={"description": "Clinical guidelines knowledge base"}
    )

    # Add documents
    print("📄 Adding documents...")
    collection.add(
        ids=[doc["id"] for doc in DOCUMENTS],
        documents=[doc["text"] for doc in DOCUMENTS],
        metadatas=[doc["metadata"] for doc in DOCUMENTS]
    )
    print(f"   ✅ Added {collection.count()} documents to collection")

    # Query!
    print("\n🔍 Querying: 'What medication for diabetes?'")
    results = collection.query(
        query_texts=["What medication should be started for Type 2 diabetes?"],
        n_results=3
    )

    print("\n   Top 3 results:")
    for i in range(len(results["ids"][0])):
        print(f"\n   {i+1}. ID: {results['ids'][0][i]}")
        print(f"      Category: {results['metadatas'][0][i]['category']}")
        print(f"      Distance: {results['distances'][0][i]:.4f}")
        print(f"      Text: {results['documents'][0][i][:120]}...")

    print("""
💡 KEY POINTS:
   • collection.add() — adds documents (ChromaDB auto-embeds them!)
   • collection.query() — finds similar documents by meaning
   • Returns: ids, documents, metadatas, distances
   • Distance = how far apart (lower = more similar)
   • No manual embedding code needed (ChromaDB handles it)
""")
    return collection


# ============================================================
# DEMO 2: Metadata Filtering
# ============================================================

def demo_metadata_filtering():
    """Filter search results by metadata"""
    print("\n" + "=" * 70)
    print("DEMO 2: METADATA FILTERING")
    print("=" * 70)

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_filtered",
        embedding_function=openai_ef
    )
    collection.add(
        ids=[doc["id"] for doc in DOCUMENTS],
        documents=[doc["text"] for doc in DOCUMENTS],
        metadatas=[doc["metadata"] for doc in DOCUMENTS]
    )

    query = "What medications are recommended?"

    # Search ALL documents
    print(f"\n🔍 Query: '{query}'")
    print("\n--- Search ALL categories ---")
    results_all = collection.query(query_texts=[query], n_results=3)
    for i in range(len(results_all["ids"][0])):
        print(f"   {i+1}. [{results_all['metadatas'][0][i]['category']}] "
              f"{results_all['documents'][0][i][:100]}...")

    # Filter by category: only cardiology
    print("\n--- Filter: ONLY cardiology ---")
    results_cardio = collection.query(
        query_texts=[query],
        n_results=3,
        where={"category": "cardiology"}  # ← Metadata filter!
    )
    for i in range(len(results_cardio["ids"][0])):
        print(f"   {i+1}. [{results_cardio['metadatas'][0][i]['category']}] "
              f"{results_cardio['documents'][0][i][:100]}...")

    # Filter by category: only endocrinology
    print("\n--- Filter: ONLY endocrinology ---")
    results_endo = collection.query(
        query_texts=[query],
        n_results=3,
        where={"category": "endocrinology"}
    )
    for i in range(len(results_endo["ids"][0])):
        print(f"   {i+1}. [{results_endo['metadatas'][0][i]['category']}] "
              f"{results_endo['documents'][0][i][:100]}...")

    # Complex filter: multiple conditions
    print("\n--- Filter: cardiology OR nephrology ---")
    results_multi = collection.query(
        query_texts=[query],
        n_results=3,
        where={"$or": [
            {"category": "cardiology"},
            {"category": "nephrology"}
        ]}
    )
    for i in range(len(results_multi["ids"][0])):
        print(f"   {i+1}. [{results_multi['metadatas'][0][i]['category']}] "
              f"{results_multi['documents'][0][i][:100]}...")

    print("""
💡 METADATA FILTERING:
   • where={"category": "cardiology"}     → exact match
   • where={"$or": [{...}, {...}]}        → OR conditions
   • where={"$and": [{...}, {...}]}       → AND conditions
   • Combine semantic search WITH metadata filters
   • Great for: department-specific search, date ranges, doc types
""")


# ============================================================
# DEMO 3: Persistent Storage
# ============================================================

def demo_persistent_storage():
    """Save ChromaDB to disk and reload it"""
    print("\n" + "=" * 70)
    print("DEMO 3: PERSISTENT STORAGE")
    print("=" * 70)

    db_path = "./chroma_medical_db"

    # Clean up from previous runs
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    # Step 1: Create persistent client
    print("\n💾 Step 1: Creating persistent database...")
    persistent_client = chromadb.PersistentClient(path=db_path)

    collection = persistent_client.create_collection(
        name="medical_guidelines",
        embedding_function=openai_ef
    )

    # Add documents
    collection.add(
        ids=[doc["id"] for doc in DOCUMENTS],
        documents=[doc["text"] for doc in DOCUMENTS],
        metadatas=[doc["metadata"] for doc in DOCUMENTS]
    )
    print(f"   ✅ Saved {collection.count()} documents to {db_path}")

    # Step 2: Simulate restart — create a NEW client pointing to same path
    print("\n🔄 Step 2: Simulating application restart...")
    print("   Creating new client from saved database...")
    reloaded_client = chromadb.PersistentClient(path=db_path)
    reloaded_collection = reloaded_client.get_collection(
        name="medical_guidelines",
        embedding_function=openai_ef
    )

    print(f"   ✅ Reloaded collection: {reloaded_collection.count()} documents")

    # Query from reloaded collection
    print("\n🔍 Step 3: Querying the reloaded database...")
    results = reloaded_collection.query(
        query_texts=["blood pressure management"],
        n_results=2
    )
    for i in range(len(results["ids"][0])):
        print(f"   {i+1}. {results['ids'][0][i]}: {results['documents'][0][i][:100]}...")

    # Cleanup
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print(f"\n   🧹 Cleaned up test database")

    print("""
💡 PERSISTENT STORAGE:
   • chromadb.Client()           → In-memory (lost on restart)
   • chromadb.PersistentClient() → Saved to disk (survives restarts)
   • Perfect for: knowledge bases that grow over time
   • In production: embed once, query forever!
""")


# ============================================================
# DEMO 4: Full RAG with ChromaDB
# ============================================================

def demo_rag_with_chromadb():
    """Complete RAG pipeline using ChromaDB instead of in-memory"""
    print("\n" + "=" * 70)
    print("DEMO 4: RAG WITH ChromaDB (Production Pattern)")
    print("=" * 70)

    # Setup ChromaDB
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_rag",
        embedding_function=openai_ef
    )
    collection.add(
        ids=[doc["id"] for doc in DOCUMENTS],
        documents=[doc["text"] for doc in DOCUMENTS],
        metadatas=[doc["metadata"] for doc in DOCUMENTS]
    )
    print(f"\n📦 Knowledge base loaded: {collection.count()} documents")

    print("\n💬 Medical Q&A with ChromaDB RAG (type 'quit' to exit)\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue

        # Retrieve from ChromaDB
        results = collection.query(query_texts=[query], n_results=3)

        # Build context
        context_parts = []
        print(f"\n   📎 Sources found:")
        for i in range(len(results["ids"][0])):
            source = results["metadatas"][0][i]
            text = results["documents"][0][i]
            dist = results["distances"][0][i]
            print(f"      {i+1}. [{source['category']}] {results['ids'][0][i]} (distance: {dist:.4f})")
            context_parts.append(f"[Source {i+1}: {source['topic']}]\n{text}")

        context = "\n\n".join(context_parts)

        # Generate with LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical knowledge assistant. Answer using ONLY the provided context. Cite sources as [Source X]. Educational purposes only."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer with citations:"
                }
            ],
            max_tokens=400, temperature=0.2
        )

        print(f"\n   📋 Answer:\n   {response.choices[0].message.content}")
        print(f"   (Tokens: {response.usage.total_tokens})\n")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n🗄️  Level 2, Project 2: Vector Databases (ChromaDB)")
    print("=" * 70)
    print("Replace in-memory storage with a real vector database\n")

    print("Choose a demo:")
    print("1. ChromaDB basics (create, add, query)")
    print("2. Metadata filtering (search within categories)")
    print("3. Persistent storage (save to disk)")
    print("4. Full RAG with ChromaDB (interactive Q&A)")
    print("5. Run demos 1-3, then interactive Q&A")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_chromadb_basics()
    elif choice == "2":
        demo_metadata_filtering()
    elif choice == "3":
        demo_persistent_storage()
    elif choice == "4":
        demo_rag_with_chromadb()
    elif choice == "5":
        demo_chromadb_basics()
        demo_metadata_filtering()
        demo_persistent_storage()
        demo_rag_with_chromadb()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY TAKEAWAYS
{'=' * 70}

🗄️  ChromaDB OPERATIONS:
   • collection.add()    — Add documents (auto-embeds!)
   • collection.query()  — Semantic search
   • collection.update() — Update existing documents
   • collection.delete() — Remove documents
   • collection.count()  — Number of documents
   • collection.get()    — Get by ID

🔍 METADATA FILTERING:
   • where={{"category": "cardiology"}}
   • where={{"$or": [...]}} / where={{"$and": [...]}}
   • Combine semantic search + metadata for precise results

💾 STORAGE:
   • chromadb.Client()           → In-memory (testing)
   • chromadb.PersistentClient() → Disk (production)

🎯 NEXT: Move to 03_document_processing to learn
   how to load real documents and chunk them effectively!
""")


if __name__ == "__main__":
    main()
