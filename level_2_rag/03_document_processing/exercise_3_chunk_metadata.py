"""
Exercise 3: Add Metadata to Each Chunk
Attach source document, section header, page number, and other metadata
to every chunk — enabling filtered retrieval and source citation.

Skills practiced:
- Designing metadata schemas for document chunks
- Extracting section headers from structured text
- Assigning page numbers and positional information
- Using metadata in ChromaDB queries for filtered search

Healthcare context:
  When a clinician asks "What medications for HF?" and gets an answer,
  they need to know: Which guideline? Which section? What page?
  Without chunk metadata, you can say "ACE inhibitors" but can't say
  "Page 12, Section 3 of the AHA Heart Failure Guideline, 2024 edition."
  Metadata makes your RAG system citable and trustworthy.
"""

import os
import re
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
# Simulated multi-page clinical documents
# ============================================================

# Document 1: Heart Failure Guideline
HF_GUIDELINE = {
    "title": "AHA Heart Failure Management Guideline",
    "source_file": "AHA_HF_Guideline_2024.pdf",
    "year": 2024,
    "author": "American Heart Association",
    "pages": {
        1: {
            "section": "Definition and Classification",
            "text": "Heart failure (HF) is a clinical syndrome caused by structural or functional cardiac abnormalities resulting in reduced cardiac output or elevated intracardiac pressures. Classification by ejection fraction: HFrEF (EF 40% or less), HFmrEF (EF 41-49%), and HFpEF (EF 50% or more). The New York Heart Association (NYHA) functional classification ranges from Class I (no limitation) to Class IV (symptoms at rest)."
        },
        2: {
            "section": "Diagnosis",
            "text": "Diagnosis requires a combination of clinical history, physical exam, and diagnostic testing. Key symptoms: dyspnea, orthopnea, paroxysmal nocturnal dyspnea, fatigue, and peripheral edema. Essential tests include BNP or NT-proBNP (BNP greater than 100 pg/mL suggests HF), echocardiography to assess ejection fraction, ECG to identify arrhythmias, and chest X-ray to evaluate cardiomegaly and pulmonary congestion."
        },
        3: {
            "section": "Pharmacological Treatment",
            "text": "Guideline-directed medical therapy (GDMT) for HFrEF includes four pillars: ACE inhibitor or ARNI (sacubitril-valsartan preferred), beta-blocker (carvedilol, metoprolol succinate), MRA (spironolactone), and SGLT2 inhibitor (dapagliflozin, empagliflozin). Titrate to target doses over weeks to months. Monitor BP, renal function, and potassium. Start low and go slow in elderly patients."
        },
        4: {
            "section": "Device Therapy",
            "text": "Consider ICD for primary prevention if EF 35% or less on optimal GDMT for 3 months. CRT recommended for EF 35% or less with LBBB and QRS 150ms or more. LVAD for advanced HF as bridge to transplant or destination therapy."
        },
        5: {
            "section": "Lifestyle Management",
            "text": "Sodium restriction less than 2000mg daily. Fluid restriction 1.5-2L daily for congestion. Daily weight monitoring — report gain of 2+ pounds overnight. Cardiac rehabilitation recommended. Avoid NSAIDs and most calcium channel blockers in HFrEF."
        },
    }
}

# Document 2: Diabetes Guideline
DM_GUIDELINE = {
    "title": "ADA Standards of Care in Diabetes",
    "source_file": "ADA_Diabetes_Standards_2024.pdf",
    "year": 2024,
    "author": "American Diabetes Association",
    "pages": {
        1: {
            "section": "Diagnosis Criteria",
            "text": "Type 2 Diabetes diagnosis: Fasting glucose 126 mg/dL or higher, HbA1c 6.5% or higher, 2-hour OGTT 200 mg/dL or higher, or random glucose 200 mg/dL or higher with classic symptoms. Prediabetes: FG 100-125, HbA1c 5.7-6.4%, 2hr OGTT 140-199."
        },
        2: {
            "section": "First-Line Treatment",
            "text": "Metformin remains first-line pharmacotherapy. Starting dose 500mg daily, titrate to 2000mg as tolerated. GI side effects common initially. Extended-release formulation improves tolerability. HbA1c target less than 7% for most adults, less than 8% for elderly with comorbidities."
        },
        3: {
            "section": "Second-Line Agents",
            "text": "If HbA1c not at target after 3 months on metformin, add second agent based on comorbidities. GLP-1 receptor agonists (semaglutide, liraglutide) preferred if cardiovascular disease or obesity. SGLT2 inhibitors (empagliflozin, dapagliflozin) preferred if heart failure or CKD. Tirzepatide for obesity plus diabetes."
        },
        4: {
            "section": "Monitoring",
            "text": "Monitor HbA1c every 3 months until stable, then every 6 months. Annual screening: dilated eye exam, comprehensive foot exam, kidney function (eGFR and UACR), lipid panel. Self-monitoring of blood glucose for patients on insulin or sulfonylureas."
        },
    }
}

# Document 3: Hypertension Guideline
HTN_GUIDELINE = {
    "title": "ACC/AHA Hypertension Clinical Practice Guideline",
    "source_file": "ACC_AHA_HTN_2024.pdf",
    "year": 2024,
    "author": "ACC/AHA",
    "pages": {
        1: {
            "section": "Definition and Staging",
            "text": "Hypertension defined as BP 130/80 mmHg or higher. Normal: less than 120/80. Elevated: 120-129 systolic with less than 80 diastolic. Stage 1: 130-139/80-89. Stage 2: 140/90 or higher. Hypertensive crisis: greater than 180/120 with or without target organ damage."
        },
        2: {
            "section": "Pharmacological Treatment",
            "text": "First-line agents: ACE inhibitors (lisinopril), ARBs (losartan), calcium channel blockers (amlodipine), thiazide diuretics (chlorthalidone). Choice depends on comorbidities. Combination therapy for Stage 2 or if target not met. Monitor kidney function with ACE-I/ARB initiation."
        },
        3: {
            "section": "Special Populations",
            "text": "Patients with CKD: ACE-I or ARB preferred, target less than 130/80. Diabetes with HTN: any first-line agent acceptable, but ACE-I/ARB if proteinuria. Elderly over 65: target less than 130/80 if tolerated. Pregnancy: labetalol, nifedipine, or methyldopa."
        },
    }
}

ALL_GUIDELINES = [HF_GUIDELINE, DM_GUIDELINE, HTN_GUIDELINE]


# ============================================================
# Chunk with rich metadata
# ============================================================

def chunk_documents_with_metadata(guidelines):
    """
    Process documents into chunks with full metadata.
    Each chunk gets: source file, title, section, page, year, author, position info.
    """
    all_chunks = []
    chunk_counter = 0

    for guideline in guidelines:
        total_pages = len(guideline["pages"])

        for page_num, page_data in guideline["pages"].items():
            chunk_counter += 1
            chunk = {
                "id": f"chunk_{chunk_counter:03d}",
                "text": page_data["text"],
                "metadata": {
                    # Source identification
                    "source_file": guideline["source_file"],
                    "document_title": guideline["title"],
                    "author": guideline["author"],

                    # Temporal
                    "year": guideline["year"],

                    # Position within document
                    "section": page_data["section"],
                    "page_number": page_num,
                    "total_pages": total_pages,

                    # Derived
                    "word_count": len(page_data["text"].split()),
                }
            }
            all_chunks.append(chunk)

    return all_chunks


# ============================================================
# Demo 1: View chunks with their metadata
# ============================================================

def demo_view_metadata():
    """Display all chunks with their full metadata"""
    print("\n" + "=" * 70)
    print("DEMO 1: CHUNKS WITH RICH METADATA")
    print("=" * 70)

    chunks = chunk_documents_with_metadata(ALL_GUIDELINES)

    print(f"\n📦 Total chunks: {len(chunks)} from {len(ALL_GUIDELINES)} documents\n")

    for chunk in chunks:
        m = chunk["metadata"]
        print(f"   {chunk['id']} | {m['source_file']:35s} | "
              f"p.{m['page_number']}/{m['total_pages']} | "
              f"{m['section']:30s} | {m['word_count']} words")

    print(f"\n   Metadata fields per chunk:")
    sample = chunks[0]["metadata"]
    for key, value in sample.items():
        print(f"      • {key}: {value} ({type(value).__name__})")

    print("""
💡 WHY THIS MATTERS:
   • Every chunk knows WHERE it came from (file, page, section)
   • Every chunk knows WHEN it was published (year)
   • Every chunk knows WHO wrote it (author)
   • This enables: filtered search, source citations, recency filtering
""")


# ============================================================
# Demo 2: Filtered search using metadata
# ============================================================

def demo_filtered_search():
    """Use metadata to filter search results"""
    print("\n" + "=" * 70)
    print("DEMO 2: METADATA-FILTERED SEARCH")
    print("=" * 70)

    chunks = chunk_documents_with_metadata(ALL_GUIDELINES)

    # Build ChromaDB collection
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_with_metadata",
        embedding_function=openai_ef
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks]
    )
    print(f"\n📦 Indexed {collection.count()} chunks with metadata")

    query = "What medications should be prescribed?"

    # Search 1: ALL documents
    print(f"\n🔍 Query: '{query}'")
    print("\n--- ALL documents (no filter) ---")
    results = collection.query(query_texts=[query], n_results=3)
    for i in range(len(results["ids"][0])):
        m = results["metadatas"][0][i]
        print(f"   {i+1}. [{m['source_file'][:20]}] p.{m['page_number']} — "
              f"{m['section']} (dist: {results['distances'][0][i]:.4f})")

    # Search 2: ONLY heart failure guideline
    print("\n--- ONLY Heart Failure guideline ---")
    results_hf = collection.query(
        query_texts=[query],
        n_results=3,
        where={"source_file": "AHA_HF_Guideline_2024.pdf"}
    )
    for i in range(len(results_hf["ids"][0])):
        m = results_hf["metadatas"][0][i]
        print(f"   {i+1}. p.{m['page_number']} — {m['section']} "
              f"(dist: {results_hf['distances'][0][i]:.4f})")

    # Search 3: ONLY "Treatment" sections across all docs
    print("\n--- ONLY Treatment/Pharmacological sections ---")
    results_tx = collection.query(
        query_texts=[query],
        n_results=3,
        where_document={"$contains": "first-line"}
    )
    for i in range(len(results_tx["ids"][0])):
        m = results_tx["metadatas"][0][i]
        print(f"   {i+1}. [{m['source_file'][:20]}] {m['section']} "
              f"(dist: {results_tx['distances'][0][i]:.4f})")

    # Search 4: Page 1 across all guidelines (definitions/staging)
    print("\n--- Page 1 only (definitions) ---")
    results_p1 = collection.query(
        query_texts=["What is the definition and classification?"],
        n_results=3,
        where={"page_number": 1}
    )
    for i in range(len(results_p1["ids"][0])):
        m = results_p1["metadatas"][0][i]
        print(f"   {i+1}. [{m['source_file'][:20]}] {m['section']} "
              f"(dist: {results_p1['distances'][0][i]:.4f})")

    print("""
💡 METADATA FILTERING POWERS:
   • Filter by source document: "Only search the HF guideline"
   • Filter by section type: "Only treatment sections"
   • Filter by page: "Only introductory pages"
   • Filter by year: "Only 2024 guidelines"
   • Combine with semantic search for PRECISE, RELEVANT results
""")


# ============================================================
# Demo 3: Build citations from metadata
# ============================================================

def demo_citations():
    """Show how metadata enables proper source citations"""
    print("\n" + "=" * 70)
    print("DEMO 3: SOURCE CITATIONS FROM METADATA")
    print("=" * 70)

    chunks = chunk_documents_with_metadata(ALL_GUIDELINES)

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="citable_chunks",
        embedding_function=openai_ef
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks]
    )

    query = "What are the four pillars of heart failure treatment?"
    print(f"\n❓ Question: '{query}'")

    results = collection.query(query_texts=[query], n_results=3)

    # Build context with citations
    context_parts = []
    citations = []
    for i in range(len(results["ids"][0])):
        m = results["metadatas"][0][i]
        text = results["documents"][0][i]

        # Build citation string
        citation = (f"{m['author']}. \"{m['document_title']},\" "
                    f"Section: {m['section']}, "
                    f"p.{m['page_number']}, "
                    f"({m['year']}).")
        citations.append(citation)
        context_parts.append(f"[Source {i+1}]\n{text}")

    context = "\n\n".join(context_parts)

    # Generate answer with LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a medical knowledge assistant. Answer using ONLY the provided context. Cite sources as [Source X]. Be concise."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer with citations:"
            }
        ],
        max_tokens=300, temperature=0.2
    )

    print(f"\n📋 Answer:")
    print(f"   {response.choices[0].message.content}")

    print(f"\n📚 Full Citations:")
    for i, citation in enumerate(citations):
        print(f"   [{i+1}] {citation}")

    print(f"\n   (Tokens used: {response.usage.total_tokens})")

    print("""
💡 CITATION QUALITY:
   • Without metadata: "ACE inhibitors are recommended" (says who?)
   • With metadata: "ACE inhibitors are recommended [Source 1]"
     → [1] AHA. "Heart Failure Guideline," Section: Pharmacological Treatment, p.3, (2024).
   • This is how real clinical decision support systems work
   • Clinicians NEED to verify the source — metadata makes this possible
""")


# ============================================================
# Demo 4: Metadata schema design patterns
# ============================================================

def demo_schema_patterns():
    """Show different metadata schema designs for different doc types"""
    print("\n" + "=" * 70)
    print("DEMO 4: METADATA SCHEMA DESIGN PATTERNS")
    print("=" * 70)

    print("""
    📋 CLINICAL GUIDELINE CHUNKS:
    ┌──────────────────────────────────────────────────┐
    │ source_file:     "AHA_HF_Guideline_2024.pdf"     │
    │ document_title:  "AHA Heart Failure Guideline"    │
    │ author:          "American Heart Association"     │
    │ year:            2024                             │
    │ section:         "Pharmacological Treatment"     │
    │ page_number:     3                               │
    │ total_pages:     5                               │
    │ evidence_grade:  "A"                             │
    │ word_count:      85                              │
    └──────────────────────────────────────────────────┘

    📝 CLINICAL NOTE CHUNKS:
    ┌──────────────────────────────────────────────────┐
    │ source_system:   "Epic"                          │
    │ note_type:       "Progress Note"                 │
    │ encounter_date:  "2024-01-15"                    │
    │ department:      "Cardiology"                    │
    │ provider:        "Dr. Smith"                     │
    │ section:         "Assessment and Plan"           │
    │ patient_hash:    "anon_abc123"                   │
    │ chunk_index:     2                               │
    │ total_chunks:    5                               │
    └──────────────────────────────────────────────────┘

    💊 DRUG FORMULARY CHUNKS:
    ┌──────────────────────────────────────────────────┐
    │ drug_name:       "Metformin"                     │
    │ drug_class:      "Biguanide"                     │
    │ formulary_tier:  1                               │
    │ requires_prior_auth: False                       │
    │ section:         "Dosing"                        │
    │ last_reviewed:   2024                            │
    │ therapeutic_area: "Endocrinology"                │
    └──────────────────────────────────────────────────┘

    📰 RESEARCH PAPER CHUNKS:
    ┌──────────────────────────────────────────────────┐
    │ doi:             "10.1234/example.2024"          │
    │ journal:         "NEJM"                          │
    │ first_author:    "Smith et al."                  │
    │ year:            2024                            │
    │ section:         "Results"                       │
    │ study_type:      "RCT"                           │
    │ sample_size:     5000                            │
    │ evidence_level:  "1A"                            │
    └──────────────────────────────────────────────────┘
""")

    print("""
💡 METADATA DESIGN RULES:
   1. Include everything you might want to FILTER by
   2. Include everything you need for CITATION
   3. Use consistent types (ChromaDB: str, int, float, bool only)
   4. Add positional info (page, chunk_index) for context reconstruction
   5. Add temporal info (year, date) for recency filtering
   6. Keep metadata lightweight — the TEXT carries the meaning
""")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n🏷️  Exercise 3: Chunk Metadata")
    print("=" * 70)
    print("Add source, section, page number metadata to every chunk\n")

    print("Choose a demo:")
    print("1. View chunks with their metadata")
    print("2. Metadata-filtered search (by source, section, page)")
    print("3. Source citations from metadata (RAG with references)")
    print("4. Metadata schema design patterns")
    print("5. Run all demos in sequence")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_view_metadata()
    elif choice == "2":
        demo_filtered_search()
    elif choice == "3":
        demo_citations()
    elif choice == "4":
        demo_schema_patterns()
    elif choice == "5":
        demo_view_metadata()
        demo_filtered_search()
        demo_citations()
        demo_schema_patterns()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY LEARNINGS — Exercise 3
{'=' * 70}

1. METADATA MAKES RAG TRUSTWORTHY:
   • Without metadata: "Take ACE inhibitors" (says who? when?)
   • With metadata: "Take ACE inhibitors [AHA HF Guideline 2024, p.3]"
   • Clinicians MUST verify sources — metadata enables this

2. WHAT TO INCLUDE:
   • Source identification: filename, title, author
   • Temporal: publication year, update date
   • Position: page number, section header, chunk index
   • Domain-specific: evidence grade, drug class, department

3. FILTERING USE CASES:
   • "Only search the diabetes guideline" → where source_file = ...
   • "Only recent guidelines" → where year >= 2023
   • "Only treatment sections" → where section = "Treatment"
   • Combine filters + semantic search for precision

4. DESIGN FOR YOUR QUERIES:
   • Think about what users will ask
   • Add metadata for every dimension they might filter by
   • Consistent schema across all documents in a collection
""")


if __name__ == "__main__":
    main()
