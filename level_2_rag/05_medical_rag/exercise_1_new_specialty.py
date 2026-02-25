"""
Exercise 1: Add a New Medical Specialty
Extend the medical knowledge base with a complete orthopedics specialty,
then test retrieval across the original + new specialty documents.

Skills practiced:
- Designing document schemas for new medical domains
- Adding structured metadata to new document sets
- Validating retrieval works across original + new documents
- Understanding how knowledge base expansion affects quality

Healthcare context:
  Real clinical knowledge bases grow over time. Every new department,
  protocol update, or specialty addition follows this same pattern:
  create documents → add metadata → verify retrieval quality.
"""

import os
import json
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
# Original Knowledge Base (from main.py)
# ============================================================

ORIGINAL_DOCUMENTS = [
    {"id": "htn_def", "text": "Hypertension is defined as blood pressure consistently at or above 130/80 mmHg. Stage 1 is 130-139 systolic or 80-89 diastolic. Stage 2 is 140/90 or higher. Hypertensive crisis is above 180/120 requiring immediate intervention.", "metadata": {"specialty": "cardiology", "topic": "hypertension", "subtopic": "definition"}},
    {"id": "htn_meds", "text": "First-line antihypertensives: ACE inhibitors (lisinopril 10-40mg, enalapril 5-40mg), ARBs (losartan 50-100mg, valsartan 80-320mg), CCBs (amlodipine 2.5-10mg), thiazides (HCTZ 12.5-25mg). Start monotherapy. If not at target in 4-6 weeks, add second agent from different class or increase dose.", "metadata": {"specialty": "cardiology", "topic": "hypertension", "subtopic": "medications"}},
    {"id": "htn_special", "text": "Special populations in hypertension: Black patients may respond better to CCBs or thiazides as initial therapy. Patients with CKD or proteinuria should receive ACE/ARB. Patients with diabetes benefit from ACE/ARB. Pregnant patients should avoid ACE/ARB; use labetalol or nifedipine instead.", "metadata": {"specialty": "cardiology", "topic": "hypertension", "subtopic": "special_populations"}},
    {"id": "hf_pillars", "text": "Heart failure with reduced EF treatment has four medication pillars that should ALL be started: (1) ARNI (sacubitril-valsartan) preferred over ACEi/ARB, (2) Beta-blocker (carvedilol, metoprolol succinate, or bisoprolol), (3) MRA (spironolactone or eplerenone), (4) SGLT2i (dapagliflozin or empagliflozin). All four improve survival.", "metadata": {"specialty": "cardiology", "topic": "heart_failure", "subtopic": "medications"}},
    {"id": "afib_anticoag", "text": "Atrial fibrillation anticoagulation: Calculate CHA2DS2-VASc score. Score 2+ in men or 3+ in women warrants anticoagulation. DOACs preferred: apixaban 5mg BID, rivaroxaban 20mg daily with food, dabigatran 150mg BID. Warfarin if mechanical valve (target INR 2-3).", "metadata": {"specialty": "cardiology", "topic": "atrial_fibrillation", "subtopic": "anticoagulation"}},
    {"id": "dm_dx", "text": "Type 2 Diabetes diagnosis: fasting glucose 126+ mg/dL on two occasions, HbA1c 6.5%+, 2-hour OGTT 200+ mg/dL, or random glucose 200+ with symptoms. Prediabetes: fasting glucose 100-125, HbA1c 5.7-6.4%.", "metadata": {"specialty": "endocrinology", "topic": "diabetes", "subtopic": "diagnosis"}},
    {"id": "dm_firstline", "text": "Metformin is first-line for Type 2 diabetes. Start 500mg once daily with meals, titrate to 2000mg daily. Contraindicated if eGFR below 30. GI side effects common; extended-release may help. Does not cause hypoglycemia.", "metadata": {"specialty": "endocrinology", "topic": "diabetes", "subtopic": "metformin"}},
    {"id": "dm_addon", "text": "Add-on therapy if HbA1c not at target after 3 months of metformin: GLP-1 agonists (semaglutide, liraglutide) preferred if cardiovascular disease or obesity. SGLT2 inhibitors preferred if heart failure or CKD. DPP-4 inhibitors if cost-sensitive. Insulin if HbA1c very high (10%+).", "metadata": {"specialty": "endocrinology", "topic": "diabetes", "subtopic": "second_line"}},
    {"id": "asthma_steps", "text": "Asthma stepwise management: Step 1 as-needed SABA. Step 2 low-dose ICS daily. Step 3 low-dose ICS-LABA. Step 4 medium-high dose ICS-LABA. Step 5 add tiotropium or biologic. Step down if well-controlled for 3+ months.", "metadata": {"specialty": "pulmonology", "topic": "asthma", "subtopic": "stepwise_therapy"}},
    {"id": "copd_mgmt", "text": "COPD management: Group A bronchodilator PRN. Group B LAMA or LABA. Group E LAMA+LABA, add ICS if eosinophils 300+. Smoking cessation is the ONLY intervention proven to slow FEV1 decline. Pulmonary rehabilitation improves quality of life.", "metadata": {"specialty": "pulmonology", "topic": "copd", "subtopic": "management"}},
    {"id": "ckd_staging", "text": "CKD staged by GFR: Stage 1 (90+), Stage 2 (60-89), Stage 3a (45-59), Stage 3b (30-44), Stage 4 (15-29), Stage 5 (below 15). Also classified by albuminuria: A1 (below 30), A2 (30-300), A3 (above 300).", "metadata": {"specialty": "nephrology", "topic": "ckd", "subtopic": "staging"}},
    {"id": "ckd_mgmt", "text": "CKD management: BP target less than 130/80. ACE/ARB for proteinuria. Avoid nephrotoxins (NSAIDs, aminoglycosides). Monitor potassium, phosphorus, calcium, PTH. Refer to nephrology at Stage 4 or rapid decline.", "metadata": {"specialty": "nephrology", "topic": "ckd", "subtopic": "management"}},
    {"id": "dep_screen", "text": "PHQ-9 depression screening: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-19 moderately severe, 20-27 severe. Screen all adults annually. Assess suicidal ideation, substance use, bipolar symptoms.", "metadata": {"specialty": "psychiatry", "topic": "depression", "subtopic": "screening"}},
    {"id": "dep_tx", "text": "Depression treatment: Mild — psychotherapy or lifestyle. Moderate — SSRI plus CBT. Severe — SSRI plus CBT; consider SNRI or mirtazapine if SSRI fails. Treatment-resistant — augment with aripiprazole, lithium, or bupropion.", "metadata": {"specialty": "psychiatry", "topic": "depression", "subtopic": "treatment"}},
]


# ============================================================
# NEW: Orthopedics Specialty (5 documents)
# ============================================================

ORTHOPEDICS_DOCUMENTS = [
    {
        "id": "ortho_oa_dx",
        "text": "Osteoarthritis (OA) diagnosis is primarily clinical: joint pain worsening with activity "
               "and improving with rest, morning stiffness lasting less than 30 minutes, crepitus on "
               "examination, bony enlargement (Heberden's nodes at DIP, Bouchard's nodes at PIP). "
               "X-ray findings: joint space narrowing, osteophytes, subchondral sclerosis, subchondral "
               "cysts. Labs are NORMAL in OA (unlike inflammatory arthritis). Weight-bearing joints "
               "most affected: knees, hips, spine, first CMC joint.",
        "metadata": {"specialty": "orthopedics", "topic": "osteoarthritis", "subtopic": "diagnosis"}
    },
    {
        "id": "ortho_oa_tx",
        "text": "Osteoarthritis management: First-line is non-pharmacological — weight loss (every 1 lb "
               "lost removes 4 lbs of knee joint force), physical therapy with range-of-motion and "
               "strengthening exercises, assistive devices (cane in contralateral hand). Pharmacological: "
               "topical NSAIDs (diclofenac gel) preferred for localized OA, oral NSAIDs (naproxen 500mg BID, "
               "ibuprofen 400-800mg TID) with PPI gastroprotection, duloxetine for chronic OA pain. "
               "Intra-articular corticosteroid injection for acute flares (max 3-4 per year per joint). "
               "Total joint replacement for severe refractory cases with functional limitation.",
        "metadata": {"specialty": "orthopedics", "topic": "osteoarthritis", "subtopic": "treatment"}
    },
    {
        "id": "ortho_fracture",
        "text": "Common fracture management principles: Assess neurovascular status before and after any "
               "intervention. Ottawa ankle rules: X-ray if bone tenderness at posterior edge or tip of "
               "malleolus OR unable to bear weight 4 steps. Ottawa knee rules: X-ray if age 55+, "
               "isolated patellar tenderness, unable to flex 90 degrees, or unable to bear weight 4 steps. "
               "Distal radius fracture (Colles): short arm cast for non-displaced, surgical fixation if "
               "displaced or intra-articular. Hip fracture: SURGICAL EMERGENCY in elderly — operate within "
               "24-48 hours to reduce mortality. Start DVT prophylaxis.",
        "metadata": {"specialty": "orthopedics", "topic": "fractures", "subtopic": "management"}
    },
    {
        "id": "ortho_backpain",
        "text": "Low back pain evaluation: Red flags requiring urgent workup — saddle anesthesia or "
               "bowel/bladder dysfunction (cauda equina syndrome → emergent MRI), progressive neurological "
               "deficit, cancer history with new back pain, fever/infection concern, trauma with osteoporosis. "
               "No red flags: conservative management for 4-6 weeks — NSAIDs, activity modification (NOT "
               "bed rest), physical therapy. Imaging NOT recommended in first 6 weeks unless red flags. "
               "MRI for persistent symptoms >6 weeks or worsening neurological signs.",
        "metadata": {"specialty": "orthopedics", "topic": "low_back_pain", "subtopic": "evaluation"}
    },
    {
        "id": "ortho_osteoporosis",
        "text": "Osteoporosis screening and treatment: DEXA scan for women 65+, men 70+, or younger with "
               "risk factors. T-score: -1 or above normal, -1 to -2.5 osteopenia, below -2.5 osteoporosis. "
               "FRAX tool estimates 10-year fracture risk. Treatment: calcium 1000-1200mg daily + vitamin D "
               "800-1000 IU daily for ALL patients. Bisphosphonates (alendronate 70mg weekly, risedronate "
               "35mg weekly) first-line pharmacotherapy. Take on empty stomach with full glass of water, "
               "remain upright 30 minutes. Denosumab for bisphosphonate intolerance. Drug holiday after "
               "3-5 years of bisphosphonate for low-risk patients.",
        "metadata": {"specialty": "orthopedics", "topic": "osteoporosis", "subtopic": "screening_treatment"}
    },
]


# ============================================================
# Helper Functions
# ============================================================

def build_combined_kb():
    """Build knowledge base with original + new orthopedics documents"""
    all_docs = ORIGINAL_DOCUMENTS + ORTHOPEDICS_DOCUMENTS
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_kb_expanded",
        embedding_function=openai_ef
    )
    collection.add(
        ids=[d["id"] for d in all_docs],
        documents=[d["text"] for d in all_docs],
        metadatas=[d["metadata"] for d in all_docs]
    )
    return collection, all_docs


def multi_query_retrieve(question, collection, n_results=4):
    """Multi-query retrieval (from main.py)"""
    rephrase = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Generate 2 alternative medical phrasings of this question. Return ONLY a JSON array of strings."},
            {"role": "user", "content": question}
        ],
        max_tokens=150, temperature=0.5
    )
    try:
        alternatives = json.loads(rephrase.choices[0].message.content)
    except json.JSONDecodeError:
        alternatives = []

    all_queries = [question] + alternatives[:2]
    seen = {}
    for q in all_queries:
        results = collection.query(query_texts=[q], n_results=n_results)
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            if doc_id not in seen or results["distances"][0][i] < seen[doc_id]["distance"]:
                seen[doc_id] = {
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
    return sorted(seen.values(), key=lambda x: x["distance"])[:n_results]


def generate_cited_answer(question, sources):
    """Generate answer with citations (from main.py)"""
    context = "\n\n".join([
        f"[Source {i+1}: {s['metadata']['specialty']}/{s['metadata']['topic']}]\n{s['text']}"
        for i, s in enumerate(sources)
    ])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a medical knowledge assistant. Answer ONLY from provided sources.
Cite every claim with [Source X]. If sources are insufficient, say so.
End with confidence: HIGH/MEDIUM/LOW. Add disclaimer about consulting a healthcare provider.
Be specific with medication names, doses, and criteria."""
            },
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=400, temperature=0.2
    )
    return response.choices[0].message.content


# ============================================================
# Demo 1: New Specialty Showcase
# ============================================================

def demo_new_specialty():
    """Show the orthopedics documents and test queries against them"""
    print("\n" + "=" * 70)
    print("DEMO 1: NEW SPECIALTY — ORTHOPEDICS (5 Documents)")
    print("=" * 70)

    collection, all_docs = build_combined_kb()

    # Show what was added
    specialties = {}
    for d in all_docs:
        spec = d["metadata"]["specialty"]
        specialties[spec] = specialties.get(spec, 0) + 1

    print(f"\n📦 Expanded Knowledge Base:")
    for spec, count in sorted(specialties.items()):
        marker = " ← NEW" if spec == "orthopedics" else ""
        print(f"   {spec:<15} {count} documents{marker}")
    print(f"   {'─' * 30}")
    print(f"   {'Total':<15} {sum(specialties.values())} documents\n")

    # New specialty documents
    print(f"📝 Orthopedics documents added:")
    for doc in ORTHOPEDICS_DOCUMENTS:
        print(f"   • [{doc['id']}] {doc['metadata']['topic']}/{doc['metadata']['subtopic']}")
        print(f"     \"{doc['text'][:80]}...\"")

    # Test retrieval
    ortho_questions = [
        "How is osteoarthritis diagnosed?",
        "What are the red flags for low back pain?",
        "When should a hip fracture patient have surgery?",
        "How is osteoporosis screened and treated?",
        "What non-drug treatments help knee osteoarthritis?",
    ]

    print(f"\n\n📊 Testing retrieval for orthopedics questions:\n")
    for question in ortho_questions:
        results = collection.query(query_texts=[question], n_results=1)
        doc_id = results["ids"][0][0]
        dist = results["distances"][0][0]
        specialty = results["metadatas"][0][0]["specialty"]

        is_ortho = specialty == "orthopedics"
        icon = "✅" if is_ortho else "⚠️"
        print(f"   {icon} \"{question}\"")
        print(f"      → [{doc_id}] dist={dist:.4f} (specialty: {specialty})")

    print("""
💡 EXPANDING A KNOWLEDGE BASE:
   1. Create documents with consistent metadata schema
   2. Use the SAME metadata fields (specialty, topic, subtopic)
   3. Test that new documents are retrieved for relevant queries
   4. Verify original queries still work correctly (no regression)
""")


# ============================================================
# Demo 2: Cross-Specialty Questions
# ============================================================

def demo_cross_specialty():
    """Ask questions that span original + new specialty"""
    print("\n" + "=" * 70)
    print("DEMO 2: CROSS-SPECIALTY QUERIES")
    print("=" * 70)

    collection, _ = build_combined_kb()

    cross_questions = [
        "My patient has CKD and needs pain management for osteoarthritis — what can I use?",
        "An elderly patient with osteoporosis fell and broke a hip — what's the urgency?",
        "A patient with heart failure and low back pain — can I use NSAIDs?",
    ]

    for question in cross_questions:
        print(f"\n{'─' * 70}")
        print(f"❓ {question}\n")

        sources = multi_query_retrieve(question, collection, n_results=4)

        # Show which specialties were pulled in
        specialties_found = set(s["metadata"]["specialty"] for s in sources)
        print(f"   📚 Specialties retrieved: {', '.join(sorted(specialties_found))}")

        for i, s in enumerate(sources):
            print(f"   [{i+1}] {s['id']} ({s['metadata']['specialty']}/{s['metadata']['topic']}) "
                  f"dist={s['distance']:.4f}")

        answer = generate_cited_answer(question, sources)
        print(f"\n   📋 {answer}")

    print("""
💡 CROSS-SPECIALTY VALUE:
   • Real clinical questions often span multiple specialties
   • A unified knowledge base retrieves from ALL relevant areas
   • The LLM synthesizes across sources to give a complete answer
   • This is how multi-disciplinary care teams reason
""")


# ============================================================
# Demo 3: Regression Testing (Original Queries Still Work)
# ============================================================

def demo_regression_test():
    """Verify that adding orthopedics doesn't break existing retrieval"""
    print("\n" + "=" * 70)
    print("DEMO 3: REGRESSION TEST — Do Original Queries Still Work?")
    print("=" * 70)

    # Build both original-only and expanded KBs
    chroma_client = chromadb.Client()

    original_coll = chroma_client.create_collection(name="original_only", embedding_function=openai_ef)
    original_coll.add(
        ids=[d["id"] for d in ORIGINAL_DOCUMENTS],
        documents=[d["text"] for d in ORIGINAL_DOCUMENTS],
        metadatas=[d["metadata"] for d in ORIGINAL_DOCUMENTS]
    )

    expanded_coll = chroma_client.create_collection(name="expanded_kb", embedding_function=openai_ef)
    all_docs = ORIGINAL_DOCUMENTS + ORTHOPEDICS_DOCUMENTS
    expanded_coll.add(
        ids=[d["id"] for d in all_docs],
        documents=[d["text"] for d in all_docs],
        metadatas=[d["metadata"] for d in all_docs]
    )

    # Test questions targeting original documents
    original_questions = [
        {"q": "What medications treat hypertension?",         "expected": "htn_meds"},
        {"q": "How is Type 2 diabetes diagnosed?",            "expected": "dm_dx"},
        {"q": "What are the heart failure medication pillars?","expected": "hf_pillars"},
        {"q": "How is CKD staged?",                           "expected": "ckd_staging"},
        {"q": "What is first-line depression treatment?",     "expected": "dep_tx"},
        {"q": "How is asthma managed stepwise?",              "expected": "asthma_steps"},
    ]

    print(f"\n   Testing {len(original_questions)} original queries against both KBs:\n")
    print(f"   {'Question':<45} {'Original':>18} {'Expanded':>18} {'Same?':>6}")
    print(f"   {'─' * 87}")

    regressions = 0
    for test in original_questions:
        orig_results = original_coll.query(query_texts=[test["q"]], n_results=1)
        exp_results = expanded_coll.query(query_texts=[test["q"]], n_results=1)

        orig_id = orig_results["ids"][0][0]
        orig_dist = orig_results["distances"][0][0]
        exp_id = exp_results["ids"][0][0]
        exp_dist = exp_results["distances"][0][0]

        same_doc = orig_id == exp_id
        if not same_doc:
            regressions += 1

        short_q = test["q"][:42] + "..." if len(test["q"]) > 45 else test["q"]
        icon = "✅" if same_doc else "❌"
        print(f"   {short_q:<45} {orig_id:>10} {orig_dist:.3f}  {exp_id:>10} {exp_dist:.3f}  {icon}")

    print(f"\n   {'─' * 87}")
    if regressions == 0:
        print(f"   ✅ ALL CLEAR — No regressions! Adding orthopedics did not break existing retrieval.")
    else:
        print(f"   ⚠️  {regressions} regression(s) detected — some queries now return different top results.")

    print("""
💡 REGRESSION TESTING:
   • ALWAYS test existing queries when adding new documents
   • New specialty docs might "steal" results from existing queries
   • Maintain a test suite of expected query → document pairs
   • Automate this: run after every knowledge base update
   • In healthcare: regression = potential patient safety issue
""")


# ============================================================
# Demo 4: Specialty-Filtered Retrieval
# ============================================================

def demo_filtered_retrieval():
    """Show how metadata filters restrict results to a specific specialty"""
    print("\n" + "=" * 70)
    print("DEMO 4: SPECIALTY-FILTERED RETRIEVAL")
    print("=" * 70)

    collection, _ = build_combined_kb()

    query = "What pain medication is recommended?"

    # Unfiltered
    print(f"\n❓ Query: \"{query}\"\n")

    unfiltered = collection.query(query_texts=[query], n_results=5)
    print(f"   🔍 UNFILTERED results (all specialties):")
    for i in range(len(unfiltered["ids"][0])):
        print(f"      {i+1}. [{unfiltered['ids'][0][i]}] "
              f"{unfiltered['metadatas'][0][i]['specialty']}/{unfiltered['metadatas'][0][i]['topic']} "
              f"dist={unfiltered['distances'][0][i]:.4f}")

    # Filtered by specialty
    for specialty in ["orthopedics", "cardiology", "psychiatry"]:
        filtered = collection.query(
            query_texts=[query],
            n_results=3,
            where={"specialty": specialty}
        )
        print(f"\n   🔬 FILTERED to {specialty}:")
        if filtered["ids"][0]:
            for i in range(len(filtered["ids"][0])):
                print(f"      {i+1}. [{filtered['ids'][0][i]}] "
                      f"{filtered['metadatas'][0][i]['topic']} "
                      f"dist={filtered['distances'][0][i]:.4f}")
        else:
            print(f"      (no results)")

    print("""
💡 FILTERED RETRIEVAL:
   • Metadata filters RESTRICT search to specific specialties
   • Useful when user context is known ("I'm on the ortho floor")
   • ChromaDB where= clause filters BEFORE embedding search
   • Combine: filter by specialty + search by query → precise results
   
   🏥 USE CASES:
   • Specialty-specific clinical decision support
   • Department-level protocol lookup
   • Role-based access to guideline subsets
""")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n🦴 Exercise 1: Add a New Medical Specialty (Orthopedics)")
    print("=" * 70)
    print("Expand the knowledge base and verify quality\n")

    print("Choose a demo:")
    print("1. New specialty showcase (orthopedics documents + retrieval test)")
    print("2. Cross-specialty queries (questions spanning multiple areas)")
    print("3. Regression test (verify original queries still work)")
    print("4. Specialty-filtered retrieval (metadata-based filtering)")
    print("5. Run all demos in sequence")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        demo_new_specialty()
    elif choice == "2":
        demo_cross_specialty()
    elif choice == "3":
        demo_regression_test()
    elif choice == "4":
        demo_filtered_retrieval()
    elif choice == "5":
        demo_new_specialty()
        demo_cross_specialty()
        demo_regression_test()
        demo_filtered_retrieval()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY LEARNINGS — Exercise 1: Adding a New Specialty
{'=' * 70}

1. DOCUMENT DESIGN:
   • Follow the SAME metadata schema (specialty, topic, subtopic)
   • Each document should cover ONE focused concept
   • Include specific details: drug names, doses, criteria, timelines
   • Clinical accuracy is paramount

2. ORTHOPEDICS ADDED (5 docs):
   • Osteoarthritis diagnosis + treatment
   • Fracture management (Ottawa rules, hip fracture urgency)
   • Low back pain (red flags, conservative management)
   • Osteoporosis (DEXA screening, bisphosphonates)

3. CROSS-SPECIALTY QUERIES:
   • Real questions span multiple specialties (CKD + pain, HF + NSAIDs)
   • A unified knowledge base handles these naturally
   • Multi-query retrieval pulls from relevant areas

4. REGRESSION TESTING IS ESSENTIAL:
   • Adding new documents can shift retrieval results
   • Always verify existing queries still return expected documents
   • Automate this as your knowledge base grows
   • In production: run regression suite before every KB update

5. FILTERED RETRIEVAL:
   • Metadata lets you scope queries to a specialty
   • Useful for department-specific or role-specific applications
   • ChromaDB where= filters happen BEFORE vector search
""")


if __name__ == "__main__":
    main()
