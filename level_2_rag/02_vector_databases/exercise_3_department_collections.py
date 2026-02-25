"""
Exercise 3: Separate Collections per Department
Create dedicated ChromaDB collections for different clinical departments and
compare focused (single-department) vs global (cross-department) search.

Skills practiced:
- Creating and managing multiple ChromaDB collections
- Organizing documents by department
- Querying within a specific collection vs across all
- Understanding collection design patterns for large organizations

Healthcare context:
  Hospitals have distinct departments — cardiology, psychiatry, nephrology —
  each with their own protocols. Sometimes you want to search WITHIN a
  specialty (cardiologist reviewing cardiac guidelines) and sometimes ACROSS
  specialties (primary care looking at all relevant guidelines for a patient
  with multiple conditions). This exercise teaches both patterns.
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
# Documents organized by department
# ============================================================

DEPARTMENT_DOCUMENTS = {
    "cardiology": [
        {
            "id": "cardio_htn_1",
            "text": "Hypertension is defined as blood pressure consistently at or above 130/80 mmHg. Stage 1 hypertension is 130-139 systolic or 80-89 diastolic. Stage 2 is 140/90 or higher. First-line treatments include lifestyle modifications: DASH diet, regular exercise 150 minutes per week, sodium restriction less than 2300mg per day.",
            "metadata": {"topic": "hypertension", "doc_type": "guideline", "department": "cardiology"}
        },
        {
            "id": "cardio_htn_2",
            "text": "Pharmacological therapy for hypertension: ACE inhibitors (lisinopril, enalapril), ARBs (losartan, valsartan), calcium channel blockers (amlodipine), thiazide diuretics (hydrochlorothiazide). Target BP for most adults less than 130/80. Combination therapy often needed for Stage 2.",
            "metadata": {"topic": "hypertension", "doc_type": "guideline", "department": "cardiology"}
        },
        {
            "id": "cardio_hf_1",
            "text": "Heart failure classification: NYHA Class I no symptom limitation, Class II mild limitation, Class III marked limitation, Class IV symptoms at rest. GDMT: ACE-I or ARNI, beta-blocker (carvedilol, metoprolol succinate), MRA (spironolactone), SGLT2-i (dapagliflozin). Optimize all four pillars.",
            "metadata": {"topic": "heart_failure", "doc_type": "guideline", "department": "cardiology"}
        },
        {
            "id": "cardio_afib_1",
            "text": "Atrial fibrillation management: Rate control with beta-blockers or calcium channel blockers. Anticoagulation with DOACs preferred over warfarin: apixaban 5mg BID, rivaroxaban 20mg daily. CHA2DS2-VASc score guides anticoagulation decision. Rhythm control with amiodarone or catheter ablation.",
            "metadata": {"topic": "atrial_fibrillation", "doc_type": "guideline", "department": "cardiology"}
        },
    ],
    "psychiatry": [
        {
            "id": "psych_dep_1",
            "text": "Depression screening using PHQ-9: Score 0-4 minimal, 5-9 mild, 10-14 moderate, 15-19 moderately severe, 20-27 severe. SSRIs first-line for moderate depression: sertraline 50mg, escitalopram 10mg, fluoxetine 20mg. Allow 4-6 weeks for initial response.",
            "metadata": {"topic": "depression", "doc_type": "guideline", "department": "psychiatry"}
        },
        {
            "id": "psych_anx_1",
            "text": "Generalized Anxiety Disorder: GAD-7 screening score 10 or higher suggests moderate anxiety. First-line: SSRIs (sertraline, escitalopram) or SNRIs (venlafaxine, duloxetine). Buspirone as adjunct. CBT effective alone for mild-moderate. Avoid long-term benzodiazepine use.",
            "metadata": {"topic": "anxiety", "doc_type": "guideline", "department": "psychiatry"}
        },
        {
            "id": "psych_bipolar_1",
            "text": "Bipolar disorder: Lithium remains first-line mood stabilizer for bipolar I. Target serum level 0.6-1.0 mEq/L. Monitor renal function and thyroid. Valproate alternative for mania. Lamotrigine for bipolar depression maintenance. Antidepressant monotherapy contraindicated — risk of mania switch.",
            "metadata": {"topic": "bipolar", "doc_type": "guideline", "department": "psychiatry"}
        },
        {
            "id": "psych_insomnia_1",
            "text": "Insomnia management: CBT-I (cognitive behavioral therapy for insomnia) is first-line. Sleep hygiene education. If pharmacotherapy needed: low-dose trazodone, melatonin receptor agonists (ramelteon), or orexin receptor antagonists (suvorexant). Avoid chronic benzodiazepines and Z-drugs.",
            "metadata": {"topic": "insomnia", "doc_type": "guideline", "department": "psychiatry"}
        },
    ],
    "endocrinology": [
        {
            "id": "endo_dm_1",
            "text": "Type 2 Diabetes diagnosis: Fasting glucose 126 mg/dL or higher, HbA1c 6.5 percent or higher. First-line: Metformin 500mg starting dose, titrate to 2000mg daily. HbA1c target less than 7 percent for most adults.",
            "metadata": {"topic": "diabetes", "doc_type": "guideline", "department": "endocrinology"}
        },
        {
            "id": "endo_dm_2",
            "text": "If HbA1c not at target after 3 months on metformin, add second agent. GLP-1 agonists (semaglutide) preferred for CV disease or obesity. SGLT2 inhibitors (empagliflozin) preferred for heart failure or CKD. Tirzepatide for obesity plus diabetes.",
            "metadata": {"topic": "diabetes", "doc_type": "guideline", "department": "endocrinology"}
        },
        {
            "id": "endo_thyroid_1",
            "text": "Hypothyroidism: TSH elevated above 4.5 mIU/L with low free T4 is overt hypothyroidism. Treatment: levothyroxine 1.6 mcg/kg/day. Elderly start 25-50 mcg. Recheck TSH in 6-8 weeks. Symptoms: fatigue, weight gain, cold intolerance, constipation.",
            "metadata": {"topic": "thyroid", "doc_type": "guideline", "department": "endocrinology"}
        },
        {
            "id": "endo_osteo_1",
            "text": "Osteoporosis: DEXA scan T-score -2.5 or less defines osteoporosis. Screen women at 65, men at 70, or earlier with risk factors. First-line: bisphosphonates (alendronate 70mg weekly, risedronate). Calcium 1200mg and vitamin D 800-1000 IU daily for all patients.",
            "metadata": {"topic": "osteoporosis", "doc_type": "guideline", "department": "endocrinology"}
        },
    ],
    "nephrology": [
        {
            "id": "neph_ckd_1",
            "text": "Chronic Kidney Disease staging by GFR: Stage 1 GFR 90+ with damage, Stage 2 GFR 60-89, Stage 3a 45-59, Stage 3b 30-44, Stage 4 15-29, Stage 5 less than 15. Refer to nephrology at Stage 4 or rapid decline.",
            "metadata": {"topic": "ckd", "doc_type": "guideline", "department": "nephrology"}
        },
        {
            "id": "neph_ckd_2",
            "text": "CKD management: BP target less than 130/80, ACE-I or ARB for proteinuria. SGLT2 inhibitors now standard for CKD with or without diabetes. Avoid nephrotoxic drugs: NSAIDs, aminoglycosides. Monitor GFR every 3-6 months.",
            "metadata": {"topic": "ckd", "doc_type": "guideline", "department": "nephrology"}
        },
        {
            "id": "neph_aki_1",
            "text": "Acute Kidney Injury: KDIGO staging by creatinine rise. Stage 1: 1.5-1.9x baseline. Stage 2: 2-2.9x. Stage 3: 3x or more, or creatinine over 4.0. Management: identify cause, stop nephrotoxins, maintain euvolemia, avoid contrast if possible. Renal replacement therapy for refractory acidosis, hyperkalemia, or volume overload.",
            "metadata": {"topic": "aki", "doc_type": "guideline", "department": "nephrology"}
        },
    ],
    "pulmonology": [
        {
            "id": "pulm_asthma_1",
            "text": "Asthma stepwise therapy: Step 1 as-needed ICS-formoterol. Step 2 low-dose ICS daily. Step 3 low-dose ICS-LABA. Step 4 medium-high ICS-LABA. Step 5 add biologics (omalizumab, dupilumab, mepolizumab) based on phenotype.",
            "metadata": {"topic": "asthma", "doc_type": "guideline", "department": "pulmonology"}
        },
        {
            "id": "pulm_copd_1",
            "text": "COPD diagnosis: FEV1/FVC ratio less than 0.70 post-bronchodilator. GOLD: mild FEV1 80+, moderate 50-79, severe 30-49, very severe less than 30. Maintenance: LAMA (tiotropium) or LABA. Dual LAMA+LABA for persistent symptoms. Add ICS if frequent exacerbations.",
            "metadata": {"topic": "copd", "doc_type": "guideline", "department": "pulmonology"}
        },
        {
            "id": "pulm_pe_1",
            "text": "Pulmonary embolism: Wells score for pretest probability. CT pulmonary angiography for diagnosis. Anticoagulation: heparin bridge to DOAC (rivaroxaban or apixaban) or warfarin. Duration: provoked 3 months, unprovoked at least 6 months or indefinite. Massive PE: systemic thrombolysis or catheter-directed.",
            "metadata": {"topic": "pulmonary_embolism", "doc_type": "guideline", "department": "pulmonology"}
        },
    ],
}


# ============================================================
# Helper: Create all department collections
# ============================================================

def create_department_collections(chroma_client):
    """Create one collection per department and populate it"""
    collections = {}
    total_docs = 0

    for dept, docs in DEPARTMENT_DOCUMENTS.items():
        collection = chroma_client.create_collection(
            name=f"dept_{dept}",
            embedding_function=openai_ef,
            metadata={"department": dept, "description": f"{dept.title()} department guidelines"}
        )
        collection.add(
            ids=[doc["id"] for doc in docs],
            documents=[doc["text"] for doc in docs],
            metadatas=[doc["metadata"] for doc in docs]
        )
        collections[dept] = collection
        total_docs += len(docs)

    return collections, total_docs


def create_global_collection(chroma_client):
    """Create a single collection with ALL documents"""
    all_docs = []
    for dept, docs in DEPARTMENT_DOCUMENTS.items():
        all_docs.extend(docs)

    collection = chroma_client.create_collection(
        name="all_departments",
        embedding_function=openai_ef,
        metadata={"description": "All departments combined"}
    )
    collection.add(
        ids=[doc["id"] for doc in all_docs],
        documents=[doc["text"] for doc in all_docs],
        metadatas=[doc["metadata"] for doc in all_docs]
    )
    return collection


# ============================================================
# Demo 1: Create and inspect department collections
# ============================================================

def demo_create_collections():
    """Create separate collections and show their contents"""
    print("\n" + "=" * 70)
    print("DEMO 1: CREATE DEPARTMENT COLLECTIONS")
    print("=" * 70)

    chroma_client = chromadb.Client()
    collections, total = create_department_collections(chroma_client)

    print(f"\n📦 Created {len(collections)} department collections ({total} total documents):\n")

    for dept, coll in collections.items():
        docs = coll.get()
        topics = set(m["topic"] for m in docs["metadatas"])
        print(f"   🏥 {dept.upper()} — {coll.count()} documents")
        print(f"      Topics: {', '.join(sorted(topics))}")

    # List all collections
    all_colls = chroma_client.list_collections()
    print(f"\n📋 ChromaDB collections: {[c.name for c in all_colls]}")

    print("""
💡 COLLECTION DESIGN:
   • One collection per department = isolated search spaces
   • Each collection has its own embedding index
   • chroma_client.list_collections() shows all available
   • Like separate "databases" for each specialty
""")


# ============================================================
# Demo 2: Search within a single department
# ============================================================

def demo_single_department_search():
    """Query within one department's collection"""
    print("\n" + "=" * 70)
    print("DEMO 2: SINGLE-DEPARTMENT SEARCH")
    print("=" * 70)

    chroma_client = chromadb.Client()
    collections, _ = create_department_collections(chroma_client)

    query = "What medications help with anxiety and sleep?"

    # Search ONLY psychiatry
    print(f"\n🔍 Query: '{query}'")
    print(f"   Searching: psychiatry collection only\n")

    results = collections["psychiatry"].query(query_texts=[query], n_results=3)
    for i in range(len(results["ids"][0])):
        m = results["metadatas"][0][i]
        print(f"   {i+1}. [{m['topic']}] {results['ids'][0][i]} "
              f"— dist: {results['distances'][0][i]:.4f}")
        print(f"      {results['documents'][0][i][:120]}...")

    # Search ONLY cardiology — same query, different department
    print(f"\n   Searching: cardiology collection (same query)\n")

    results2 = collections["cardiology"].query(query_texts=[query], n_results=3)
    for i in range(len(results2["ids"][0])):
        m = results2["metadatas"][0][i]
        print(f"   {i+1}. [{m['topic']}] {results2['ids'][0][i]} "
              f"— dist: {results2['distances'][0][i]:.4f}")
        print(f"      {results2['documents'][0][i][:120]}...")

    print("""
💡 DEPARTMENT-SPECIFIC SEARCH:
   • Psychiatry finds anxiety, insomnia, depression docs — relevant!
   • Cardiology returns cardiac docs — NOT relevant for this query
   • Notice the distance scores: psychiatry results are MUCH closer
   • Use case: A psychiatrist searching their own department's guidelines
""")


# ============================================================
# Demo 3: Cross-department search (global collection)
# ============================================================

def demo_cross_department_search():
    """Compare single-department vs global search"""
    print("\n" + "=" * 70)
    print("DEMO 3: CROSS-DEPARTMENT (GLOBAL) SEARCH")
    print("=" * 70)

    chroma_client = chromadb.Client()
    collections, _ = create_department_collections(chroma_client)
    global_collection = create_global_collection(chroma_client)

    # Query that spans departments
    query = "What medications require kidney function monitoring?"

    print(f"\n🔍 Query: '{query}'")
    print(f"\n--- Global collection (all {global_collection.count()} documents) ---\n")

    global_results = global_collection.query(query_texts=[query], n_results=5)
    for i in range(len(global_results["ids"][0])):
        m = global_results["metadatas"][0][i]
        print(f"   {i+1}. [{m['department']:15s}] {global_results['ids'][0][i]:20s} "
              f"dist: {global_results['distances'][0][i]:.4f}")

    # Same query across each department separately
    print(f"\n--- Per-department search (top 1 from each) ---\n")
    for dept, coll in collections.items():
        results = coll.query(query_texts=[query], n_results=1)
        if results["ids"][0]:
            m = results["metadatas"][0][0]
            print(f"   [{dept:15s}] {results['ids'][0][0]:20s} "
                  f"dist: {results['distances'][0][0]:.4f}")

    print("""
💡 GLOBAL vs DEPARTMENT:
   • Global search: finds relevant docs across ALL departments
   • Per-department: each department's "best match" for the query
   • Use global when: primary care, cross-discipline queries, comorbidity reviews
   • Use department when: specialist searching their own guidelines
   • Both patterns are valid — your app should support both!
""")


# ============================================================
# Demo 4: Multi-department patient scenario
# ============================================================

def demo_patient_scenario():
    """A patient with multiple conditions spans departments"""
    print("\n" + "=" * 70)
    print("DEMO 4: MULTI-DEPARTMENT PATIENT SCENARIO")
    print("=" * 70)

    chroma_client = chromadb.Client()
    collections, _ = create_department_collections(chroma_client)
    global_collection = create_global_collection(chroma_client)

    print("""
    👤 PATIENT SCENARIO:
    68-year-old with:
      • Atrial fibrillation (cardiology)
      • Type 2 diabetes (endocrinology)
      • CKD Stage 3b (nephrology)
      • Depression (psychiatry)
    
    Query: "What medications need dose adjustment for kidney disease?"
""")

    query = "medications that need dose adjustment for kidney disease"

    # Global search — finds cross-cutting information
    print("--- Global search (all departments) ---\n")
    global_results = global_collection.query(query_texts=[query], n_results=5)
    departments_found = set()
    for i in range(len(global_results["ids"][0])):
        m = global_results["metadatas"][0][i]
        departments_found.add(m["department"])
        print(f"   {i+1}. [{m['department']}] {global_results['ids'][0][i]} "
              f"— dist: {global_results['distances'][0][i]:.4f}")
        print(f"      {global_results['documents'][0][i][:120]}...")

    print(f"\n   📊 Departments represented: {', '.join(sorted(departments_found))}")

    # Targeted search per relevant department
    patient_departments = ["cardiology", "endocrinology", "nephrology", "psychiatry"]
    print(f"\n--- Targeted search (patient's 4 departments) ---\n")

    for dept in patient_departments:
        results = collections[dept].query(query_texts=[query], n_results=1)
        if results["ids"][0]:
            m = results["metadatas"][0][0]
            print(f"   🏥 {dept.upper()}")
            print(f"      Best match: {results['ids'][0][0]} "
                  f"(dist: {results['distances'][0][0]:.4f})")
            print(f"      {results['documents'][0][0][:120]}...")

    print("""
💡 MULTI-DEPARTMENT PATIENT CARE:
   • Real patients have conditions spanning multiple departments
   • Global search catches cross-cutting drug interactions
   • Per-department search ensures nothing is missed per specialty
   • Production pattern: query globally, THEN drill into relevant departments
   • This is exactly how clinical decision support systems work
""")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n🏥 Exercise 3: Department Collections")
    print("=" * 70)
    print("Separate collections per department: cardiology, psychiatry, etc.\n")

    print("Choose a demo:")
    print("1. Create & inspect department collections")
    print("2. Single-department search (focused)")
    print("3. Cross-department (global) search")
    print("4. Multi-department patient scenario")
    print("5. Run all demos in sequence")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_create_collections()
    elif choice == "2":
        demo_single_department_search()
    elif choice == "3":
        demo_cross_department_search()
    elif choice == "4":
        demo_patient_scenario()
    elif choice == "5":
        demo_create_collections()
        demo_single_department_search()
        demo_cross_department_search()
        demo_patient_scenario()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY LEARNINGS — Exercise 3
{'=' * 70}

1. COLLECTION DESIGN PATTERNS:
   • One collection per department = isolated, focused search
   • One global collection = cross-cutting, multi-specialty search
   • Both patterns have valid use cases

2. WHEN TO USE EACH:
   • Department collection: specialist reviewing their own guidelines
   • Global collection: primary care, drug interactions, comorbidity queries
   • Hybrid: global search first, then drill into relevant departments

3. CHROMADB MULTI-COLLECTION:
   • chroma_client.create_collection() — new collection
   • chroma_client.list_collections()  — see all collections
   • chroma_client.get_collection()    — retrieve existing
   • chroma_client.delete_collection() — remove one

4. HEALTHCARE INSIGHT:
   • Real patients don't fit in one department
   • A 68-year-old with AF + DM + CKD + depression needs cross-specialty care
   • Your RAG system should support both focused AND global queries
   • This is how clinical decision support tools are architected
""")


if __name__ == "__main__":
    main()
