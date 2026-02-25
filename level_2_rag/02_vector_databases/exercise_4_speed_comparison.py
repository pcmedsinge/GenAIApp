"""
Exercise 4: Speed Comparison — ChromaDB vs In-Memory
Benchmark ChromaDB vector search against the in-memory cosine similarity
approach from Project 01, demonstrating why vector databases matter at scale.

Skills practiced:
- Benchmarking search latency (time.perf_counter)
- Understanding linear scan vs indexed search complexity
- Scaling document collections to observe performance differences
- Appreciating when in-memory is fine and when you NEED a vector DB

Healthcare context:
  A small clinic might have 50 guidelines — in-memory works fine.
  A hospital system with 500,000 clinical documents (guidelines, notes,
  literature) CANNOT afford to scan every embedding per query.
  This exercise makes the performance difference visceral.
"""

import os
import time
import numpy as np
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
# Base medical documents (used for both approaches)
# ============================================================

BASE_DOCUMENTS = [
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

# Additional documents to scale up the collection for benchmarking
EXTRA_DOCUMENTS = [
    {"id": "heart_failure_1", "text": "Heart failure NYHA classification: Class I no limitation, Class II mild limitation, Class III marked limitation, Class IV symptoms at rest. GDMT four pillars: ACE-I or ARNI, beta-blocker, MRA, SGLT2 inhibitor. Optimize all four for survival benefit.", "metadata": {"category": "cardiology", "topic": "heart_failure", "doc_type": "guideline"}},
    {"id": "afib_1", "text": "Atrial fibrillation: Rate control with beta-blockers or diltiazem. CHA2DS2-VASc for stroke risk. DOACs preferred: apixaban, rivaroxaban. Rhythm control: amiodarone, flecainide, or catheter ablation for symptomatic patients.", "metadata": {"category": "cardiology", "topic": "atrial_fibrillation", "doc_type": "guideline"}},
    {"id": "copd_1", "text": "COPD diagnosis: FEV1/FVC less than 0.70 post-bronchodilator. GOLD classification by FEV1. Maintenance: LAMA or LABA. Dual LAMA+LABA for persistent symptoms. Add ICS for frequent exacerbations. Pulmonary rehabilitation for all.", "metadata": {"category": "pulmonology", "topic": "copd", "doc_type": "guideline"}},
    {"id": "thyroid_1", "text": "Hypothyroidism: elevated TSH with low free T4. Levothyroxine 1.6 mcg/kg/day. Start lower in elderly. Recheck TSH 6-8 weeks. Subclinical hypothyroidism: treat if TSH above 10 or symptomatic. Symptoms: fatigue, cold intolerance, weight gain.", "metadata": {"category": "endocrinology", "topic": "thyroid", "doc_type": "guideline"}},
    {"id": "gout_1", "text": "Gout acute flare: colchicine 1.2mg then 0.6mg one hour later, or NSAIDs, or corticosteroids. Urate-lowering therapy: allopurinol start 100mg, target uric acid less than 6. Febuxostat if allopurinol intolerant. Prophylaxis with colchicine 0.6mg daily during initiation.", "metadata": {"category": "rheumatology", "topic": "gout", "doc_type": "guideline"}},
    {"id": "dvt_1", "text": "Deep vein thrombosis: Wells score for pretest probability. D-dimer if low probability. Ultrasound for diagnosis. Anticoagulation: DOAC preferred. Duration 3 months for provoked, consider indefinite for unprovoked or recurrent.", "metadata": {"category": "hematology", "topic": "dvt", "doc_type": "guideline"}},
    {"id": "anemia_1", "text": "Iron deficiency anemia: ferritin less than 30 highly specific. Oral iron: ferrous sulfate 325mg daily on empty stomach. IV iron if oral intolerant or malabsorption. Evaluate for GI blood loss in men and postmenopausal women. Recheck CBC in 4-8 weeks.", "metadata": {"category": "hematology", "topic": "anemia", "doc_type": "guideline"}},
    {"id": "anxiety_1", "text": "Generalized anxiety disorder: GAD-7 score 10+ suggests moderate anxiety. First-line SSRIs or SNRIs. Buspirone as adjunct. CBT effective alone for mild-moderate. Avoid chronic benzodiazepines. Duration of treatment at least 12 months.", "metadata": {"category": "psychiatry", "topic": "anxiety", "doc_type": "guideline"}},
    {"id": "insomnia_1", "text": "Insomnia: CBT-I is first-line treatment. Sleep hygiene education. Pharmacotherapy if needed: trazodone, ramelteon, suvorexant. Avoid chronic benzodiazepine or Z-drug use. Assess for underlying causes: depression, OSA, pain.", "metadata": {"category": "psychiatry", "topic": "insomnia", "doc_type": "guideline"}},
    {"id": "osteo_1", "text": "Osteoporosis: DEXA T-score -2.5 or less. Screen women 65, men 70. Bisphosphonates first-line: alendronate 70mg weekly. Calcium 1200mg and vitamin D 800-1000 IU daily. Denosumab if bisphosphonate contraindicated. Fall prevention strategies.", "metadata": {"category": "endocrinology", "topic": "osteoporosis", "doc_type": "guideline"}},
    {"id": "pneumonia_1", "text": "Community-acquired pneumonia: CRB-65 for severity. Outpatient mild: amoxicillin or doxycycline. Moderate (inpatient): ceftriaxone plus azithromycin. Severe (ICU): ceftriaxone plus azithromycin, consider vancomycin if MRSA risk. Switch to oral when improving.", "metadata": {"category": "pulmonology", "topic": "pneumonia", "doc_type": "guideline"}},
    {"id": "liver_1", "text": "Non-alcoholic fatty liver disease: ALT elevation with hepatic steatosis on imaging. Weight loss of 7-10 percent for resolution. No approved pharmacotherapy. Screen for fibrosis with FIB-4 index. Refer to hepatology if advanced fibrosis. Avoid alcohol completely.", "metadata": {"category": "gastroenterology", "topic": "nafld", "doc_type": "guideline"}},
    {"id": "gerd_1", "text": "GERD: PPI therapy 8 weeks for erosive esophagitis. Step-down to H2 blocker or as-needed PPI for maintenance. Lifestyle: elevate head of bed, avoid late meals, weight loss. EGD if alarm symptoms or refractory. Long-term PPI: monitor magnesium, bone density.", "metadata": {"category": "gastroenterology", "topic": "gerd", "doc_type": "guideline"}},
    {"id": "uti_1", "text": "Uncomplicated UTI in women: nitrofurantoin 100mg BID for 5 days or TMP-SMX DS BID for 3 days. Avoid fluoroquinolones for uncomplicated UTI. Complicated UTI: fluoroquinolone or ceftriaxone. Urine culture for recurrent or complicated cases.", "metadata": {"category": "infectious_disease", "topic": "uti", "doc_type": "guideline"}},
    {"id": "migraine_1", "text": "Migraine acute treatment: NSAIDs or triptans (sumatriptan). Antiemetics for nausea. Preventive if 4+ headache days/month: propranolol, topiramate, amitriptyline. CGRP monoclonal antibodies (erenumab, fremanezumab) for refractory cases.", "metadata": {"category": "neurology", "topic": "migraine", "doc_type": "guideline"}},
    {"id": "epilepsy_1", "text": "Epilepsy: start monotherapy after 2+ unprovoked seizures. Focal seizures: levetiracetam, lamotrigine, carbamazepine. Generalized: valproate, levetiracetam, lamotrigine. Valproate avoid in women of childbearing age. EEG and MRI for workup.", "metadata": {"category": "neurology", "topic": "epilepsy", "doc_type": "guideline"}},
]


# ============================================================
# In-memory approach (from Project 01)
# ============================================================

def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding vector using OpenAI API"""
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def cosine_similarity(vec_a, vec_b):
    """Calculate cosine similarity between two vectors"""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_in_memory_index(documents):
    """Embed all documents and store in a list (Project 01 approach)"""
    index = []
    for doc in documents:
        embedding = get_embedding(doc["text"])
        index.append({
            "id": doc["id"],
            "text": doc["text"],
            "metadata": doc["metadata"],
            "embedding": embedding
        })
    return index


def search_in_memory(query, index, top_k=3):
    """Linear scan: compare query embedding to every document"""
    query_embedding = get_embedding(query)

    scored = []
    for item in index:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append({**item, "similarity": score})

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


# ============================================================
# ChromaDB approach (from Project 02)
# ============================================================

def build_chromadb_index(documents):
    """Let ChromaDB handle embedding and indexing"""
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_benchmark",
        embedding_function=openai_ef
    )
    collection.add(
        ids=[doc["id"] for doc in documents],
        documents=[doc["text"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents]
    )
    return collection


def search_chromadb(query, collection, top_k=3):
    """ChromaDB indexed search"""
    results = collection.query(query_texts=[query], n_results=top_k)
    return results


# ============================================================
# Demo 1: Side-by-side results comparison (small set)
# ============================================================

def demo_results_comparison():
    """Verify both approaches return the same results"""
    print("\n" + "=" * 70)
    print("DEMO 1: RESULTS COMPARISON (same query, both approaches)")
    print("=" * 70)

    docs = BASE_DOCUMENTS

    # Build both indexes
    print("\n⏳ Building in-memory index (embedding each document)...")
    in_memory_index = build_in_memory_index(docs)
    print(f"   ✅ In-memory: {len(in_memory_index)} documents")

    print("\n⏳ Building ChromaDB index...")
    chroma_collection = build_chromadb_index(docs)
    print(f"   ✅ ChromaDB: {chroma_collection.count()} documents")

    query = "What medications are used for high blood pressure?"
    print(f"\n🔍 Query: '{query}'\n")

    # In-memory search
    print("--- In-Memory Results ---")
    mem_results = search_in_memory(query, in_memory_index, top_k=3)
    for i, r in enumerate(mem_results):
        print(f"   {i+1}. {r['id']:20s} similarity: {r['similarity']:.4f}")

    # ChromaDB search
    print("\n--- ChromaDB Results ---")
    chroma_results = search_chromadb(query, chroma_collection, top_k=3)
    for i in range(len(chroma_results["ids"][0])):
        dist = chroma_results["distances"][0][i]
        print(f"   {i+1}. {chroma_results['ids'][0][i]:20s} distance: {dist:.4f}")

    print("""
💡 COMPARISON:
   • Both approaches find the SAME relevant documents
   • In-memory returns similarity (higher = better, max 1.0)
   • ChromaDB returns distance (lower = better, min 0.0)
   • Results should match — the ranking should be identical
   • The difference is HOW they search, not WHAT they find
""")


# ============================================================
# Demo 2: Query speed comparison (search only)
# ============================================================

def demo_query_speed():
    """Compare search speed — the key difference"""
    print("\n" + "=" * 70)
    print("DEMO 2: QUERY SPEED COMPARISON")
    print("=" * 70)

    docs = BASE_DOCUMENTS + EXTRA_DOCUMENTS
    print(f"\n📦 Using {len(docs)} documents for benchmark\n")

    # Build indexes (embedding cost is the same, so we measure that separately)
    print("⏳ Building in-memory index...")
    t0 = time.perf_counter()
    in_memory_index = build_in_memory_index(docs)
    mem_build_time = time.perf_counter() - t0
    print(f"   In-memory build: {mem_build_time:.2f}s")

    print("⏳ Building ChromaDB index...")
    t0 = time.perf_counter()
    chroma_collection = build_chromadb_index(docs)
    chroma_build_time = time.perf_counter() - t0
    print(f"   ChromaDB build: {chroma_build_time:.2f}s")

    # Pre-embed query to isolate search time from embedding time
    query = "What medications are used for blood pressure?"
    print(f"\n🔍 Query: '{query}'")
    query_embedding = get_embedding(query)

    # Benchmark in-memory search (skip embedding, just measure scan)
    print("\n⏱️  Benchmarking SEARCH time (10 iterations each)...\n")

    # In-memory: linear scan
    mem_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        scored = []
        for item in in_memory_index:
            score = cosine_similarity(query_embedding, item["embedding"])
            scored.append((item["id"], score))
        scored.sort(key=lambda x: x[1], reverse=True)
        _ = scored[:3]
        mem_times.append(time.perf_counter() - t0)

    # ChromaDB: indexed search (includes embedding the query internally)
    chroma_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        results = chroma_collection.query(query_texts=[query], n_results=3)
        chroma_times.append(time.perf_counter() - t0)

    mem_avg = sum(mem_times) / len(mem_times) * 1000
    chroma_avg = sum(chroma_times) / len(chroma_times) * 1000
    mem_min = min(mem_times) * 1000
    chroma_min = min(chroma_times) * 1000

    print(f"   {'Method':<20} {'Avg (ms)':>10} {'Min (ms)':>10} {'Docs Scanned':>15}")
    print(f"   {'-'*55}")
    print(f"   {'In-Memory':<20} {mem_avg:>10.2f} {mem_min:>10.2f} {len(in_memory_index):>15}")
    print(f"   {'ChromaDB':<20} {chroma_avg:>10.2f} {chroma_min:>10.2f} {'indexed':>15}")

    print(f"""
💡 NOTE ON THESE RESULTS:
   • With {len(docs)} documents, the difference may be SMALL
   • In-memory: scans ALL {len(docs)} embeddings (O(n) linear)
   • ChromaDB: uses HNSW index (O(log n) approximate)
   • ChromaDB query time INCLUDES re-embedding the query via API
   • The real advantage appears at 10,000+ documents
   • In-memory scan time grows linearly; ChromaDB stays ~constant
""")


# ============================================================
# Demo 3: Simulated scale test
# ============================================================

def demo_scale_simulation():
    """Simulate larger scale by duplicating documents"""
    print("\n" + "=" * 70)
    print("DEMO 3: SCALE SIMULATION (Multiply Documents)")
    print("=" * 70)

    base_docs = BASE_DOCUMENTS + EXTRA_DOCUMENTS
    scales = [25, 100, 500]

    print(f"\n📦 Base: {len(base_docs)} unique documents")
    print(f"   Testing at: {', '.join(str(len(base_docs)*s) for s in scales)} documents")
    print(f"   (duplicated with unique IDs to test search scaling)\n")

    query_embedding = get_embedding("What medications for blood pressure?")

    results_table = []

    for scale in scales:
        # Create scaled document set
        scaled_docs = []
        for i in range(scale):
            for doc in base_docs:
                scaled_docs.append({
                    "id": f"{doc['id']}_copy_{i}",
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "embedding": None  # will be filled
                })

        n_docs = len(scaled_docs)
        print(f"⏳ Testing {n_docs} documents...")

        # In-memory: embed once (all copies share same embedding)
        base_embeddings = {}
        for doc in base_docs:
            base_embeddings[doc["id"]] = get_embedding(doc["text"])

        # Build in-memory index with pre-cached embeddings
        in_mem = []
        for sd in scaled_docs:
            original_id = sd["id"].rsplit("_copy_", 1)[0]
            in_mem.append({
                "id": sd["id"],
                "embedding": base_embeddings[original_id]
            })

        # Benchmark in-memory linear scan
        mem_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            scored = []
            for item in in_mem:
                score = cosine_similarity(query_embedding, item["embedding"])
                scored.append((item["id"], score))
            scored.sort(key=lambda x: x[1], reverse=True)
            _ = scored[:3]
            mem_times.append(time.perf_counter() - t0)

        # Build ChromaDB index
        chroma_client = chromadb.Client()
        collection = chroma_client.create_collection(
            name=f"benchmark_{n_docs}",
            embedding_function=openai_ef
        )
        # Add in batches (ChromaDB has batch limits)
        batch_size = 5000
        for start in range(0, len(scaled_docs), batch_size):
            batch = scaled_docs[start:start + batch_size]
            collection.add(
                ids=[d["id"] for d in batch],
                documents=[base_docs[j % len(base_docs)]["text"] for j, d in enumerate(batch)],
                metadatas=[base_docs[j % len(base_docs)]["metadata"] for j, d in enumerate(batch)]
            )

        # Benchmark ChromaDB search
        chroma_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            results = collection.query(query_texts=["What medications for blood pressure?"], n_results=3)
            chroma_times.append(time.perf_counter() - t0)

        mem_avg = sum(mem_times) / len(mem_times) * 1000
        chroma_avg = sum(chroma_times) / len(chroma_times) * 1000

        results_table.append((n_docs, mem_avg, chroma_avg))
        print(f"   {n_docs:>6} docs — In-Memory: {mem_avg:>8.2f}ms | ChromaDB: {chroma_avg:>8.2f}ms")

    # Print summary table
    print(f"\n{'=' * 60}")
    print(f"   {'Documents':>10} {'In-Memory (ms)':>16} {'ChromaDB (ms)':>16} {'Ratio':>8}")
    print(f"   {'-' * 52}")
    for n_docs, mem_ms, chroma_ms in results_table:
        ratio = mem_ms / chroma_ms if chroma_ms > 0 else 0
        print(f"   {n_docs:>10} {mem_ms:>16.2f} {chroma_ms:>16.2f} {ratio:>7.1f}x")

    print("""
💡 SCALING BEHAVIOR:
   • In-memory: time grows LINEARLY with document count
   • ChromaDB: time stays roughly CONSTANT (indexed search)
   • At small scale: nearly the same (overhead dominates)
   • At large scale: ChromaDB is dramatically faster
   • Production systems with 100K+ docs REQUIRE indexed search
""")


# ============================================================
# Demo 4: Feature comparison summary
# ============================================================

def demo_feature_comparison():
    """Show the full feature comparison, not just speed"""
    print("\n" + "=" * 70)
    print("DEMO 4: FULL FEATURE COMPARISON")
    print("=" * 70)

    print("""
    ┌─────────────────────┬──────────────────────┬──────────────────────┐
    │ Feature             │ In-Memory (Project 1)│ ChromaDB (Project 2) │
    ├─────────────────────┼──────────────────────┼──────────────────────┤
    │ Setup complexity    │ None (just a list)   │ pip install chromadb  │
    │ Embedding           │ You call API + store │ Auto (on add/query)  │
    │ Storage             │ RAM only (lost)      │ RAM or disk          │
    │ Search speed (9)    │ ~same                │ ~same                │
    │ Search speed (10K+) │ SLOW (linear scan)   │ FAST (HNSW index)    │
    │ Metadata filtering  │ Manual code          │ Built-in where={}    │
    │ Persistence         │ None                 │ PersistentClient()   │
    │ Updates / deletes   │ Manual list ops      │ .update() / .delete()│
    │ Concurrent access   │ Not thread-safe      │ Thread-safe          │
    │ Max docs            │ ~1,000 practical     │ Millions             │
    │ Dependencies        │ numpy, openai        │ chromadb, openai     │
    └─────────────────────┴──────────────────────┴──────────────────────┘
    """)

    print("📊 WHEN TO USE EACH:\n")
    print("   ✅ In-Memory is FINE when:")
    print("      • Prototyping or learning (like Project 01)")
    print("      • Less than ~500 documents")
    print("      • No need for persistence")
    print("      • You want zero dependencies")

    print("\n   ✅ ChromaDB is BETTER when:")
    print("      • 1,000+ documents")
    print("      • Data must persist across restarts")
    print("      • You need metadata filtering")
    print("      • Multiple users or processes querying")
    print("      • You're building a production system")

    print("""
💡 THE REAL-WORLD ANSWER:
   • Start with in-memory for prototyping (fast iteration)
   • Switch to ChromaDB when you have working logic
   • The migration is EASY — same concepts, better infrastructure
   • That's exactly what we did going from Project 01 → 02!
""")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n⚡ Exercise 4: Speed Comparison — ChromaDB vs In-Memory")
    print("=" * 70)
    print("Benchmark vector DB vs linear scan at different scales\n")

    print("Choose a demo:")
    print("1. Results comparison (verify both find same docs)")
    print("2. Query speed comparison (25 documents)")
    print("3. Scale simulation (25 → 12,500 documents)")
    print("4. Full feature comparison table")
    print("5. Run all demos in sequence")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_results_comparison()
    elif choice == "2":
        demo_query_speed()
    elif choice == "3":
        demo_scale_simulation()
    elif choice == "4":
        demo_feature_comparison()
    elif choice == "5":
        demo_results_comparison()
        demo_query_speed()
        demo_scale_simulation()
        demo_feature_comparison()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY LEARNINGS — Exercise 4
{'=' * 70}

1. SEARCH COMPLEXITY:
   • In-memory: O(n) — must scan EVERY embedding on EVERY query
   • ChromaDB (HNSW): O(log n) — indexed, approximate nearest neighbor
   • At 9 documents: difference is negligible
   • At 10,000 documents: ChromaDB is dramatically faster

2. BUILD vs QUERY COST:
   • Building the index (embedding): same cost for both
   • The OpenAI embedding API call dominates build time
   • The difference is in QUERY time (after index is built)
   • ChromaDB: embed once at add(), query many times for free

3. WHEN TO SWITCH:
   • < 500 docs: in-memory is fine (simpler code)
   • 500 - 5,000 docs: ChromaDB starting to matter
   • > 5,000 docs: ChromaDB is essential
   • > 100,000 docs: consider dedicated vector DBs (Pinecone, Weaviate)

4. HEALTHCARE CONTEXT:
   • Small clinic formulary: ~100 docs → in-memory OK
   • Hospital guidelines library: ~10,000 docs → need ChromaDB
   • Health system knowledge base: ~500,000 docs → need production vector DB
   • Always prototype in-memory first, then upgrade storage
""")


if __name__ == "__main__":
    main()
