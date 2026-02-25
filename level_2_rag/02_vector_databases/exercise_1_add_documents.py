"""
Exercise 1: Add New Medical Documents and Query Across Categories
Add 3 new medical documents to the ChromaDB collection and verify they
integrate seamlessly with existing documents during cross-category queries.

Skills practiced:
- Adding new documents to an existing ChromaDB collection
- Assigning proper metadata (category, topic, doc_type)
- Querying across all categories to verify integration
- Understanding how ChromaDB auto-embeds new additions

Healthcare context:
  Hospital knowledge bases grow continuously — new protocols arrive,
  formulary updates land, evidence-based guidelines get revised.
  Your vector database must absorb additions instantly and return them
  alongside older content when semantically relevant.
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# ============================================================
# Original 9 documents (from main.py)
# ============================================================

ORIGINAL_DOCUMENTS = [
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
# 3 NEW medical documents to add
# ============================================================

NEW_DOCUMENTS = [
    {
        "id": "heart_failure_1",
        "text": "Heart failure classification: NYHA Class I no symptom limitation, Class II mild limitation with ordinary activity, Class III marked limitation with less-than-ordinary activity, Class IV symptoms at rest. Guideline-directed medical therapy: ACE inhibitor or ARNI (sacubitril-valsartan), beta-blocker (carvedilol, metoprolol succinate), mineralocorticoid receptor antagonist (spironolactone), SGLT2 inhibitor (dapagliflozin, empagliflozin). Target: optimize all four pillars.",
        "metadata": {"category": "cardiology", "topic": "heart_failure", "doc_type": "guideline"}
    },
    {
        "id": "copd_1",
        "text": "COPD diagnosis confirmed by spirometry: FEV1/FVC ratio less than 0.70 post-bronchodilator. GOLD classification based on FEV1: GOLD 1 mild FEV1 80 percent or higher, GOLD 2 moderate 50-79 percent, GOLD 3 severe 30-49 percent, GOLD 4 very severe less than 30 percent. Maintenance therapy: LAMA (tiotropium) or LABA (salmeterol). Dual therapy LAMA plus LABA for persistent symptoms. Add inhaled corticosteroid if frequent exacerbations.",
        "metadata": {"category": "pulmonology", "topic": "copd", "doc_type": "guideline"}
    },
    {
        "id": "thyroid_1",
        "text": "Hypothyroidism: TSH elevated above 4.5 mIU/L with low free T4 indicates overt hypothyroidism. Subclinical if TSH elevated but free T4 normal. Treatment: levothyroxine starting dose 1.6 mcg per kg per day. Elderly or cardiac patients start lower at 25-50 mcg daily. Recheck TSH in 6-8 weeks, adjust dose by 12.5-25 mcg increments. Common symptoms: fatigue, weight gain, cold intolerance, constipation, dry skin.",
        "metadata": {"category": "endocrinology", "topic": "thyroid", "doc_type": "guideline"}
    },
]


# ============================================================
# Demo 1: Query BEFORE adding new documents
# ============================================================

def demo_query_before():
    """Show what the collection returns before new docs are added"""
    print("\n" + "=" * 70)
    print("DEMO 1: QUERY BEFORE ADDING NEW DOCUMENTS")
    print("=" * 70)

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_before",
        embedding_function=openai_ef
    )

    # Add only original 9 documents
    collection.add(
        ids=[doc["id"] for doc in ORIGINAL_DOCUMENTS],
        documents=[doc["text"] for doc in ORIGINAL_DOCUMENTS],
        metadatas=[doc["metadata"] for doc in ORIGINAL_DOCUMENTS]
    )
    print(f"\n📦 Collection has {collection.count()} original documents")

    # Test queries that target the NEW topics
    test_queries = [
        "What is the treatment for heart failure?",
        "How is COPD diagnosed and classified?",
        "What is the treatment for hypothyroidism?",
    ]

    for query in test_queries:
        results = collection.query(query_texts=[query], n_results=2)
        print(f"\n🔍 Query: '{query}'")
        for i in range(len(results["ids"][0])):
            print(f"   {i+1}. [{results['metadatas'][0][i]['category']}] "
                  f"{results['ids'][0][i]} — distance: {results['distances'][0][i]:.4f}")
            print(f"      {results['documents'][0][i][:100]}...")

    print("""
💡 OBSERVATION:
   • Queries about heart failure, COPD, thyroid return nearest available
     docs (hypertension, asthma, diabetes) — NOT the best answers
   • The knowledge base has gaps for these topics
   • Next: we'll add documents and see the difference!
""")


# ============================================================
# Demo 2: Add new documents and query AFTER
# ============================================================

def demo_add_and_query():
    """Add 3 new documents and show improved results"""
    print("\n" + "=" * 70)
    print("DEMO 2: ADD NEW DOCUMENTS, THEN QUERY")
    print("=" * 70)

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_expanded",
        embedding_function=openai_ef
    )

    # Start with original 9
    collection.add(
        ids=[doc["id"] for doc in ORIGINAL_DOCUMENTS],
        documents=[doc["text"] for doc in ORIGINAL_DOCUMENTS],
        metadatas=[doc["metadata"] for doc in ORIGINAL_DOCUMENTS]
    )
    print(f"\n📦 Starting with {collection.count()} documents")

    # Add the 3 new documents
    print("\n➕ Adding 3 new medical documents...")
    collection.add(
        ids=[doc["id"] for doc in NEW_DOCUMENTS],
        documents=[doc["text"] for doc in NEW_DOCUMENTS],
        metadatas=[doc["metadata"] for doc in NEW_DOCUMENTS]
    )
    print(f"   ✅ Collection now has {collection.count()} documents")

    # Same queries — should now return the correct docs
    test_queries = [
        "What is the treatment for heart failure?",
        "How is COPD diagnosed and classified?",
        "What is the treatment for hypothyroidism?",
    ]

    for query in test_queries:
        results = collection.query(query_texts=[query], n_results=2)
        print(f"\n🔍 Query: '{query}'")
        for i in range(len(results["ids"][0])):
            print(f"   {i+1}. [{results['metadatas'][0][i]['category']}] "
                  f"{results['ids'][0][i]} — distance: {results['distances'][0][i]:.4f}")
            print(f"      {results['documents'][0][i][:100]}...")

    print("""
💡 NOW:
   • Heart failure query → heart_failure_1 doc (exact match!)
   • COPD query → copd_1 doc (exact match!)
   • Thyroid query → thyroid_1 doc (exact match!)
   • New docs are IMMEDIATELY searchable after .add()
   • No need to re-index or restart — ChromaDB handles it
""")


# ============================================================
# Demo 3: Cross-category queries with expanded collection
# ============================================================

def demo_cross_category():
    """Show how queries pull from multiple categories at once"""
    print("\n" + "=" * 70)
    print("DEMO 3: CROSS-CATEGORY QUERIES (12 Documents)")
    print("=" * 70)

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_cross",
        embedding_function=openai_ef
    )

    # Add all 12 documents
    all_docs = ORIGINAL_DOCUMENTS + NEW_DOCUMENTS
    collection.add(
        ids=[doc["id"] for doc in all_docs],
        documents=[doc["text"] for doc in all_docs],
        metadatas=[doc["metadata"] for doc in all_docs]
    )
    print(f"\n📦 Full collection: {collection.count()} documents across categories")

    # Cross-cutting queries
    cross_queries = [
        ("What medications need kidney function monitoring?",
         "Spans: cardiology (ACE-I), endocrinology (SGLT2-i), nephrology (CKD), hematology (DOACs)"),
        ("Which conditions use beta-blockers?",
         "Spans: cardiology (hypertension), cardiology (heart failure)"),
        ("What conditions cause fatigue?",
         "Spans: endocrinology (thyroid), psychiatry (depression), nephrology (CKD)"),
        ("What are the first-line medications for newly diagnosed patients?",
         "Spans: multiple categories — metformin, SSRIs, ACE-I, levothyroxine, etc."),
    ]

    for query, expected in cross_queries:
        results = collection.query(query_texts=[query], n_results=4)
        print(f"\n🔍 Query: '{query}'")
        print(f"   Expected: {expected}")
        categories_seen = set()
        for i in range(len(results["ids"][0])):
            cat = results['metadatas'][0][i]['category']
            categories_seen.add(cat)
            print(f"   {i+1}. [{cat}] {results['ids'][0][i]} "
                  f"(dist: {results['distances'][0][i]:.4f})")
        print(f"   → Categories hit: {', '.join(sorted(categories_seen))}")

    print("""
💡 CROSS-CATEGORY SEARCH:
   • Semantic search doesn't care about category boundaries
   • A query about "kidney monitoring" pulls from cardiology, nephrology, etc.
   • This is the POWER of vector databases: meaning-based, not keyword-based
   • Metadata filters let you LIMIT to specific categories when needed
""")


# ============================================================
# Demo 4: Verify with get() and peek()
# ============================================================

def demo_verify_documents():
    """Use ChromaDB's get() and peek() to inspect the collection"""
    print("\n" + "=" * 70)
    print("DEMO 4: VERIFY & INSPECT THE COLLECTION")
    print("=" * 70)

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_inspect",
        embedding_function=openai_ef
    )

    all_docs = ORIGINAL_DOCUMENTS + NEW_DOCUMENTS
    collection.add(
        ids=[doc["id"] for doc in all_docs],
        documents=[doc["text"] for doc in all_docs],
        metadatas=[doc["metadata"] for doc in all_docs]
    )

    # Count
    print(f"\n📦 Total documents: {collection.count()}")

    # Peek — quick look at first few documents
    print("\n👀 Peek at collection (first 5):")
    peek_results = collection.peek(limit=5)
    for i in range(len(peek_results["ids"])):
        print(f"   {i+1}. {peek_results['ids'][i]} "
              f"[{peek_results['metadatas'][i]['category']}] "
              f"— {peek_results['documents'][i][:80]}...")

    # Get by ID — retrieve specific documents
    print("\n🔑 Get by ID (the 3 new documents):")
    new_ids = [doc["id"] for doc in NEW_DOCUMENTS]
    get_results = collection.get(ids=new_ids)
    for i in range(len(get_results["ids"])):
        print(f"   • {get_results['ids'][i]} [{get_results['metadatas'][i]['category']}]")
        print(f"     Topic: {get_results['metadatas'][i]['topic']}")
        print(f"     Preview: {get_results['documents'][i][:100]}...")

    # Get with metadata filter
    print("\n🏷️  Get all cardiology documents (including new heart_failure_1):")
    cardio_results = collection.get(
        where={"category": "cardiology"}
    )
    for i in range(len(cardio_results["ids"])):
        print(f"   • {cardio_results['ids'][i]} — {cardio_results['metadatas'][i]['topic']}")

    # Get all categories
    all_results = collection.get()
    categories = {}
    for meta in all_results["metadatas"]:
        cat = meta["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print(f"\n📊 Documents by category:")
    for cat, count in sorted(categories.items()):
        print(f"   • {cat}: {count} document(s)")

    print("""
💡 COLLECTION INSPECTION:
   • collection.count()         — total documents
   • collection.peek(limit=N)   — quick look at first N docs
   • collection.get(ids=[...])  — retrieve specific documents by ID
   • collection.get(where={})   — retrieve by metadata filter
   • collection.get()           — retrieve ALL documents
   • These are NON-SEARCH operations (no embedding needed)
""")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n📝 Exercise 1: Add New Medical Documents")
    print("=" * 70)
    print("Add 3 new documents and query across all categories\n")

    print("Choose a demo:")
    print("1. Query BEFORE adding new documents (see the gaps)")
    print("2. Add 3 new documents and query AFTER (see improvement)")
    print("3. Cross-category queries (explore 12-doc collection)")
    print("4. Verify & inspect the collection (get, peek, count)")
    print("5. Run all demos in sequence")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_query_before()
    elif choice == "2":
        demo_add_and_query()
    elif choice == "3":
        demo_cross_category()
    elif choice == "4":
        demo_verify_documents()
    elif choice == "5":
        demo_query_before()
        demo_add_and_query()
        demo_cross_category()
        demo_verify_documents()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY LEARNINGS — Exercise 1
{'=' * 70}

1. ADDING DOCUMENTS:
   • collection.add(ids, documents, metadatas) — that's it!
   • ChromaDB auto-embeds text using the collection's embedding function
   • New documents are INSTANTLY searchable (no re-indexing needed)

2. CROSS-CATEGORY SEARCH:
   • Semantic search finds relevant docs regardless of category
   • A "kidney monitoring" query pulls cardiology + nephrology + endocrinology
   • This is the real power of vector databases over keyword search

3. INSPECTION TOOLS:
   • .count() — how many documents
   • .peek()  — quick look at contents
   • .get()   — retrieve by ID or metadata filter

4. HEALTHCARE INSIGHT:
   • Knowledge bases grow organically — your system must handle additions
   • New guidelines, updated protocols, formulary changes
   • Test retrieval BEFORE and AFTER adding to verify quality
""")


if __name__ == "__main__":
    main()
