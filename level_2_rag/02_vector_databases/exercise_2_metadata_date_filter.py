"""
Exercise 2: Metadata Filter for "last_updated" Date
Build metadata filters using a "last_updated" field to search only recent
guidelines, demonstrating ChromaDB's powerful where-clause capabilities.

Skills practiced:
- Adding date-based metadata to documents
- Using ChromaDB where clauses with comparison operators ($gt, $lt, $gte)
- Combining semantic search with date-range filters
- Understanding metadata design for temporal queries

Healthcare context:
  Medical guidelines are updated frequently. A 2019 hypertension guideline
  may conflict with a 2024 update. Clinicians need the MOST RECENT evidence.
  Filtering by "last_updated" ensures stale guidance never surfaces.
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
# Documents WITH last_updated metadata (year as integer)
# ============================================================
# Using integer year for easy comparison with $gt, $lt, $gte, $lte
# ChromaDB where clauses support: $eq, $ne, $gt, $gte, $lt, $lte

DOCUMENTS_WITH_DATES = [
    {
        "id": "hypertension_old",
        "text": "Hypertension was previously defined as blood pressure at or above 140/90 mmHg. JNC 7 guidelines from 2003 recommended thiazide diuretics as first-line for most patients. Target BP was less than 140/90 for most adults, less than 130/80 for diabetes or CKD.",
        "metadata": {"category": "cardiology", "topic": "hypertension",
                      "doc_type": "guideline", "last_updated": 2003, "source": "JNC 7"}
    },
    {
        "id": "hypertension_mid",
        "text": "JNC 8 (2014) relaxed targets for patients over 60: less than 150/90 mmHg. For those under 60, target remained less than 140/90. Initial therapy: ACE inhibitor, ARB, calcium channel blocker, or thiazide. Race-based recommendations were introduced.",
        "metadata": {"category": "cardiology", "topic": "hypertension",
                      "doc_type": "guideline", "last_updated": 2014, "source": "JNC 8"}
    },
    {
        "id": "hypertension_new",
        "text": "ACC/AHA 2017 guidelines redefined hypertension as 130/80 mmHg or higher. Stage 1: 130-139/80-89. Stage 2: 140/90 or higher. SPRINT trial data supports tighter targets for high-risk patients. First-line: lifestyle modifications plus ACE-I, ARB, CCB, or thiazide based on comorbidities.",
        "metadata": {"category": "cardiology", "topic": "hypertension",
                      "doc_type": "guideline", "last_updated": 2017, "source": "ACC/AHA"}
    },
    {
        "id": "hypertension_latest",
        "text": "2024 ESC guidelines for hypertension emphasize individualized targets: less than 130/80 for most adults, less than 140/90 for frail elderly. Combination therapy with single-pill combinations preferred as first-line for Stage 2. SGLT2 inhibitors emerging as adjunctive therapy for hypertension with CKD or heart failure.",
        "metadata": {"category": "cardiology", "topic": "hypertension",
                      "doc_type": "guideline", "last_updated": 2024, "source": "ESC 2024"}
    },
    {
        "id": "diabetes_old",
        "text": "Diabetes management 2010: HbA1c target less than 7 percent for most. Metformin first-line. Sulfonylureas (glipizide, glyburide) as second-line. TZDs (pioglitazone) for insulin resistance. DPP-4 inhibitors emerging.",
        "metadata": {"category": "endocrinology", "topic": "diabetes",
                      "doc_type": "guideline", "last_updated": 2010, "source": "ADA 2010"}
    },
    {
        "id": "diabetes_new",
        "text": "ADA 2024 Standards of Care: Metformin remains first-line. GLP-1 receptor agonists (semaglutide) preferred for patients with cardiovascular disease, obesity, or CKD. SGLT2 inhibitors for heart failure or CKD. Tirzepatide (dual GIP/GLP-1) for obesity and diabetes. Weight management is now a primary treatment goal.",
        "metadata": {"category": "endocrinology", "topic": "diabetes",
                      "doc_type": "guideline", "last_updated": 2024, "source": "ADA 2024"}
    },
    {
        "id": "depression_old",
        "text": "Depression treatment 2008: TCAs (amitriptyline, nortriptyline) and SSRIs both effective. STAR*D trial showed remission rates of 33 percent with first medication. CBT equivalent to medication for mild-moderate depression.",
        "metadata": {"category": "psychiatry", "topic": "depression",
                      "doc_type": "guideline", "last_updated": 2008, "source": "STAR*D"}
    },
    {
        "id": "depression_new",
        "text": "2023 APA guidelines for depression: SSRIs and SNRIs remain first-line. Esketamine nasal spray for treatment-resistant depression. Digital therapeutics and app-based CBT gaining evidence. Psilocybin-assisted therapy in clinical trials. Measurement-based care using PHQ-9 at every visit.",
        "metadata": {"category": "psychiatry", "topic": "depression",
                      "doc_type": "guideline", "last_updated": 2023, "source": "APA 2023"}
    },
    {
        "id": "asthma_old",
        "text": "Asthma management 2007: Step therapy approach. Step 1 SABA as-needed. Step 2 low-dose ICS. Step 3 low-dose ICS plus LABA or medium-dose ICS. Step 4 medium-dose ICS-LABA. SABA recommended as the primary reliever at all steps.",
        "metadata": {"category": "pulmonology", "topic": "asthma",
                      "doc_type": "guideline", "last_updated": 2007, "source": "NAEPP 2007"}
    },
    {
        "id": "asthma_new",
        "text": "GINA 2024: SABA-only therapy NO LONGER recommended. All patients should have ICS-containing therapy. As-needed low-dose ICS-formoterol is the preferred reliever. Biologics (omalizumab, dupilumab, mepolizumab) for severe asthma with specific phenotypes. FeNO testing to guide ICS therapy.",
        "metadata": {"category": "pulmonology", "topic": "asthma",
                      "doc_type": "guideline", "last_updated": 2024, "source": "GINA 2024"}
    },
]


# ============================================================
# Demo 1: Show ALL documents with their dates
# ============================================================

def demo_show_dates():
    """Display all documents sorted by date to see the range"""
    print("\n" + "=" * 70)
    print("DEMO 1: DOCUMENTS WITH LAST_UPDATED DATES")
    print("=" * 70)

    sorted_docs = sorted(DOCUMENTS_WITH_DATES, key=lambda d: d["metadata"]["last_updated"])

    print(f"\n📅 {len(sorted_docs)} documents spanning "
          f"{sorted_docs[0]['metadata']['last_updated']} — "
          f"{sorted_docs[-1]['metadata']['last_updated']}:\n")

    for doc in sorted_docs:
        m = doc["metadata"]
        print(f"   {m['last_updated']} | [{m['category']:15s}] {doc['id']:25s} — {m['source']}")

    print("""
💡 KEY OBSERVATION:
   • Same topics have OLD and NEW versions (hypertension: 2003, 2014, 2017, 2024)
   • Medical guidelines CHANGE over time — older ones may be harmful to follow
   • We need to filter by date to ensure we return CURRENT evidence
""")


# ============================================================
# Demo 2: Search WITHOUT date filter (the problem)
# ============================================================

def demo_no_date_filter():
    """Show what happens without filtering — stale guidance surfaces"""
    print("\n" + "=" * 70)
    print("DEMO 2: SEARCH WITHOUT DATE FILTER (The Problem)")
    print("=" * 70)

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_no_filter",
        embedding_function=openai_ef
    )
    collection.add(
        ids=[doc["id"] for doc in DOCUMENTS_WITH_DATES],
        documents=[doc["text"] for doc in DOCUMENTS_WITH_DATES],
        metadatas=[doc["metadata"] for doc in DOCUMENTS_WITH_DATES]
    )

    query = "What is the blood pressure target for hypertension?"
    print(f"\n🔍 Query: '{query}'")
    print("   (NO date filter applied)\n")

    results = collection.query(query_texts=[query], n_results=4)
    for i in range(len(results["ids"][0])):
        m = results["metadatas"][0][i]
        print(f"   {i+1}. [{m['last_updated']}] {results['ids'][0][i]} "
              f"({m['source']}) — dist: {results['distances'][0][i]:.4f}")
        print(f"      {results['documents'][0][i][:120]}...")

    print("""
⚠️  THE PROBLEM:
   • Without date filtering, OLD and NEW guidelines both appear
   • JNC 7 (2003) says target is 140/90 — that's OUTDATED
   • ACC/AHA (2017) says target is 130/80 — CURRENT
   • ESC (2024) says individualized — LATEST
   • Mixing old and new guidance could lead to clinical errors!
""")


# ============================================================
# Demo 3: Filter by date using $gte (the solution)
# ============================================================

def demo_date_filter():
    """Use where clause with $gte to get only recent guidelines"""
    print("\n" + "=" * 70)
    print("DEMO 3: SEARCH WITH DATE FILTER (The Solution)")
    print("=" * 70)

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_date_filter",
        embedding_function=openai_ef
    )
    collection.add(
        ids=[doc["id"] for doc in DOCUMENTS_WITH_DATES],
        documents=[doc["text"] for doc in DOCUMENTS_WITH_DATES],
        metadatas=[doc["metadata"] for doc in DOCUMENTS_WITH_DATES]
    )

    query = "What is the blood pressure target for hypertension?"

    # Filter: only documents updated 2020 or later
    print(f"\n🔍 Query: '{query}'")
    print("   Filter: last_updated >= 2020 (recent guidelines only)\n")

    results_recent = collection.query(
        query_texts=[query],
        n_results=4,
        where={"last_updated": {"$gte": 2020}}  # ← Date filter!
    )
    for i in range(len(results_recent["ids"][0])):
        m = results_recent["metadatas"][0][i]
        print(f"   {i+1}. [{m['last_updated']}] {results_recent['ids'][0][i]} "
              f"({m['source']}) — dist: {results_recent['distances'][0][i]:.4f}")
        print(f"      {results_recent['documents'][0][i][:120]}...")

    # Filter: only documents from last 5 years (2019+)
    print(f"\n🔍 Same query with filter: last_updated >= 2019\n")
    results_5yr = collection.query(
        query_texts=[query],
        n_results=4,
        where={"last_updated": {"$gte": 2019}}
    )
    for i in range(len(results_5yr["ids"][0])):
        m = results_5yr["metadatas"][0][i]
        print(f"   {i+1}. [{m['last_updated']}] {results_5yr['ids'][0][i]} "
              f"({m['source']}) — dist: {results_5yr['distances'][0][i]:.4f}")

    # Filter: only OLD documents (before 2015)
    print(f"\n🔍 Same query with filter: last_updated < 2015 (historical view)\n")
    results_old = collection.query(
        query_texts=[query],
        n_results=4,
        where={"last_updated": {"$lt": 2015}}
    )
    for i in range(len(results_old["ids"][0])):
        m = results_old["metadatas"][0][i]
        print(f"   {i+1}. [{m['last_updated']}] {results_old['ids'][0][i]} "
              f"({m['source']}) — dist: {results_old['distances'][0][i]:.4f}")

    print("""
💡 DATE FILTERING OPERATORS:
   • where={{"last_updated": {{"$gte": 2020}}}}  → 2020 and newer
   • where={{"last_updated": {{"$gt": 2020}}}}   → after 2020 only
   • where={{"last_updated": {{"$lt": 2015}}}}   → before 2015
   • where={{"last_updated": {{"$lte": 2010}}}}  → 2010 and older
   
   ChromaDB operators: $eq, $ne, $gt, $gte, $lt, $lte
""")


# ============================================================
# Demo 4: Combine date + category filters
# ============================================================

def demo_combined_filters():
    """Combine date and category metadata in one query"""
    print("\n" + "=" * 70)
    print("DEMO 4: COMBINED FILTERS (Date + Category)")
    print("=" * 70)

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_combined",
        embedding_function=openai_ef
    )
    collection.add(
        ids=[doc["id"] for doc in DOCUMENTS_WITH_DATES],
        documents=[doc["text"] for doc in DOCUMENTS_WITH_DATES],
        metadatas=[doc["metadata"] for doc in DOCUMENTS_WITH_DATES]
    )

    query = "What medications are recommended?"

    # Combined: recent cardiology only
    print(f"\n🔍 Query: '{query}'")
    print("   Filter: cardiology AND last_updated >= 2017\n")

    results = collection.query(
        query_texts=[query],
        n_results=4,
        where={
            "$and": [
                {"category": "cardiology"},
                {"last_updated": {"$gte": 2017}}
            ]
        }
    )
    for i in range(len(results["ids"][0])):
        m = results["metadatas"][0][i]
        print(f"   {i+1}. [{m['last_updated']}] [{m['category']}] "
              f"{results['ids'][0][i]} ({m['source']})")

    # Combined: recent across multiple specialties
    print(f"\n🔍 Query: '{query}'")
    print("   Filter: (cardiology OR endocrinology) AND last_updated >= 2020\n")

    results2 = collection.query(
        query_texts=[query],
        n_results=4,
        where={
            "$and": [
                {"$or": [
                    {"category": "cardiology"},
                    {"category": "endocrinology"}
                ]},
                {"last_updated": {"$gte": 2020}}
            ]
        }
    )
    for i in range(len(results2["ids"][0])):
        m = results2["metadatas"][0][i]
        print(f"   {i+1}. [{m['last_updated']}] [{m['category']}] "
              f"{results2['ids'][0][i]} ({m['source']})")

    # Show the difference: same query, no date filter
    print(f"\n🔍 Same query, NO date filter (cardiology OR endocrinology):\n")
    results3 = collection.query(
        query_texts=[query],
        n_results=4,
        where={
            "$or": [
                {"category": "cardiology"},
                {"category": "endocrinology"}
            ]
        }
    )
    for i in range(len(results3["ids"][0])):
        m = results3["metadatas"][0][i]
        print(f"   {i+1}. [{m['last_updated']}] [{m['category']}] "
              f"{results3['ids'][0][i]} ({m['source']})")

    print("""
💡 COMBINED FILTERS:
   • $and: ALL conditions must match
   • $or:  ANY condition matches
   • Nest them: ($or categories) AND ($gte date)
   • This is how production RAG systems work:
     "Show me recent cardiology guidelines about medications"
     = semantic search + category filter + date filter
""")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n📅 Exercise 2: Metadata Date Filter")
    print("=" * 70)
    print("Filter by 'last_updated' to search only recent guidelines\n")

    print("Choose a demo:")
    print("1. Show all documents with their dates")
    print("2. Search WITHOUT date filter (see the problem)")
    print("3. Search WITH date filter (see the solution)")
    print("4. Combined filters (date + category)")
    print("5. Run all demos in sequence")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_show_dates()
    elif choice == "2":
        demo_no_date_filter()
    elif choice == "3":
        demo_date_filter()
    elif choice == "4":
        demo_combined_filters()
    elif choice == "5":
        demo_show_dates()
        demo_no_date_filter()
        demo_date_filter()
        demo_combined_filters()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY LEARNINGS — Exercise 2
{'=' * 70}

1. DATE-BASED METADATA:
   • Store "last_updated" as an integer year for easy comparison
   • ChromaDB supports: $eq, $ne, $gt, $gte, $lt, $lte
   • where={{"last_updated": {{"$gte": 2020}}}} → only 2020 and newer

2. WHY DATE FILTERING MATTERS:
   • Medical guidelines CHANGE — older versions may be harmful
   • A 2003 BP target (140/90) is different from 2024 (130/80)
   • Always filter for recency in production clinical RAG systems

3. COMBINED FILTERS:
   • $and: [{{"category": "cardiology"}}, {{"last_updated": {{"$gte": 2020}}}}]
   • $or + $and nesting for complex queries
   • Semantic search + metadata filters = precise, relevant results

4. PRODUCTION PATTERN:
   • Always include a date field in your document metadata
   • Let users choose the recency window (last 5 years, last year, etc.)
   • Consider flagging or removing superseded guidelines
""")


if __name__ == "__main__":
    main()
