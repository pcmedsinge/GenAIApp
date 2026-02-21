"""
Exercise 1: Add New Medical Documents
Add new medical documents to the knowledge base and test retrieval quality.

Skills practiced:
- Expanding a RAG knowledge base with new documents
- Understanding how new content integrates with existing chunks
- Testing retrieval after adding domain-specific content
- Comparing retrieval before and after adding documents

Healthcare context:
  Clinical knowledge bases grow constantly — new guidelines, formulary updates,
  policy changes. Your RAG system must handle additions seamlessly.
  This exercise teaches you to add documents and verify they're retrievable.
"""

import os
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# Original Knowledge Base (from main.py)
# ============================================================

ORIGINAL_KNOWLEDGE_BASE = [
    {
        "id": "hypertension_guidelines",
        "title": "Hypertension Management Guidelines",
        "content": """Hypertension is defined as blood pressure consistently at or above 130/80 mmHg.
Stage 1 hypertension: 130-139/80-89 mmHg. Stage 2: 140/90 mmHg or higher.
First-line treatments include lifestyle modifications: dietary changes (DASH diet),
regular exercise (150 min/week moderate intensity), weight management,
sodium restriction (less than 2300mg/day), and limiting alcohol intake.
Pharmacological therapy first-line agents: ACE inhibitors (lisinopril, enalapril),
ARBs (losartan, valsartan), calcium channel blockers (amlodipine),
thiazide diuretics (hydrochlorothiazide).
Target BP for most adults: less than 130/80 mmHg. For patients over 65: less than 130/80 if tolerated.
Monitor kidney function and electrolytes when starting ACE inhibitor or ARB therapy.
Combination therapy is often needed for Stage 2 or if target not met with monotherapy.
Black patients may respond better to calcium channel blockers or thiazides as initial therapy.""",
        "category": "cardiology"
    },
    {
        "id": "diabetes_type2",
        "title": "Type 2 Diabetes Management",
        "content": """Type 2 Diabetes diagnosis criteria: Fasting glucose 126 mg/dL or higher,
HbA1c 6.5 percent or higher, or random glucose 200 mg/dL or higher with symptoms.
First-line therapy: Metformin 500mg starting dose, titrate to 2000mg/day as tolerated.
HbA1c target: less than 7 percent for most adults, less than 8 percent for elderly or those with comorbidities.
If HbA1c not at target after 3 months on metformin, add second agent.
GLP-1 receptor agonists (semaglutide, liraglutide) preferred if cardiovascular disease or obesity.
SGLT2 inhibitors (empagliflozin, dapagliflozin) preferred if heart failure or CKD.
DPP-4 inhibitors (sitagliptin) if cost is a concern.
Monitor HbA1c every 3 months until stable, then every 6 months.
Annual screening: eye exam, foot exam, kidney function, lipid panel.
Patient education on hypoglycemia recognition and self-monitoring of blood glucose is essential.""",
        "category": "endocrinology"
    },
    {
        "id": "depression_management",
        "title": "Depression Screening and Initial Management",
        "content": """Screen all adults for depression using PHQ-9 questionnaire.
Score interpretation: 0-4 Minimal or none, 5-9 Mild, 10-14 Moderate,
15-19 Moderately severe, 20-27 Severe depression.
Initial treatment for moderate depression: SSRIs are first-line pharmacotherapy.
Common SSRIs: sertraline 50mg daily, escitalopram 10mg daily, fluoxetine 20mg daily.
Allow 4-6 weeks for initial response. Full remission may take 8-12 weeks.
Combination of medication plus psychotherapy (CBT) is most effective for moderate-severe depression.
Monitor for suicidal ideation especially in first 4 weeks and in patients under 25 years.
If no response at 6 weeks: increase dose or switch medication class.
Augmentation strategies include adding bupropion, aripiprazole, or lithium.
Continue medication for at least 6-12 months after remission to prevent relapse.
Referral to psychiatry for treatment-resistant depression (failed 2 or more adequate medication trials).""",
        "category": "psychiatry"
    }
]


# ============================================================
# NEW Documents to Add
# ============================================================

NEW_DOCUMENTS = [
    {
        "id": "heart_failure",
        "title": "Heart Failure Management Guidelines",
        "content": """Heart failure classification uses NYHA functional classes:
Class I: No symptoms with ordinary activity.
Class II: Slight limitation, symptoms with ordinary activity.
Class III: Marked limitation, symptoms with less than ordinary activity.
Class IV: Unable to carry out any physical activity without symptoms.
Heart failure types: HFrEF (ejection fraction 40 percent or less),
HFmrEF (EF 41-49 percent), HFpEF (EF 50 percent or higher).
Foundational therapy for HFrEF (the 4 pillars):
1. ACE inhibitor or ARB or sacubitril-valsartan (preferred)
2. Beta-blocker (carvedilol, metoprolol succinate, bisoprolol)
3. Mineralocorticoid receptor antagonist (spironolactone or eplerenone)
4. SGLT2 inhibitor (dapagliflozin or empagliflozin)
Additional therapy: Diuretics for volume management (furosemide).
Hydralazine plus isosorbide dinitrate if ACE/ARB contraindicated or as add-on in Black patients.
Monitor daily weights, sodium restriction to less than 2g per day.
BNP or NT-proBNP for diagnosis and monitoring treatment response.
Cardiac rehabilitation recommended for stable heart failure patients.
ICD recommended for primary prevention if EF 35 percent or less despite 3 months of optimal therapy.
CRT considered for EF 35 percent or less with LBBB and QRS 150ms or greater.""",
        "category": "cardiology"
    },
    {
        "id": "copd_management",
        "title": "COPD Diagnosis and Management",
        "content": """COPD diagnosis requires spirometry: post-bronchodilator FEV1/FVC ratio less than 0.70.
GOLD classification by FEV1: GOLD 1 Mild (FEV1 80 percent or more),
GOLD 2 Moderate (50-79 percent), GOLD 3 Severe (30-49 percent),
GOLD 4 Very severe (less than 30 percent).
Symptom assessment: mMRC dyspnea scale and CAT questionnaire.
Treatment approach by GOLD group:
Group A (few symptoms, low exacerbation risk): Bronchodilator as needed.
Group B (more symptoms): Long-acting bronchodilator (LAMA or LABA).
Group E (exacerbations): LAMA + LABA, consider adding ICS if eosinophils 300 or higher.
Key medications: LAMA (tiotropium, umeclidinium), LABA (salmeterol, formoterol, indacaterol),
ICS-LABA (fluticasone-salmeterol, budesonide-formoterol),
Triple therapy (fluticasone-umeclidinium-vilanterol).
Smoking cessation is the MOST important intervention.
Pulmonary rehabilitation for GOLD 2-4 patients.
Supplemental oxygen if PaO2 less than 55 mmHg or SpO2 less than 88 percent.
Annual influenza vaccine, pneumococcal vaccine, COVID vaccine recommended.
Exacerbation management: Short course oral prednisone 40mg for 5 days plus antibiotics
(amoxicillin-clavulanate or azithromycin) if purulent sputum.""",
        "category": "pulmonology"
    },
    {
        "id": "thyroid_disorders",
        "title": "Thyroid Disorder Diagnosis and Treatment",
        "content": """Hypothyroidism diagnosis: Elevated TSH with low free T4.
Subclinical hypothyroidism: TSH elevated (4.5-10 mIU/L) with normal free T4.
Treatment: Levothyroxine is the standard replacement therapy.
Starting dose: 1.6 mcg/kg/day for healthy adults under 65.
Elderly or cardiac disease patients: Start low at 25-50 mcg daily, titrate every 6-8 weeks.
Take levothyroxine on empty stomach, 30-60 minutes before breakfast.
Medications that interfere: calcium, iron, PPIs (separate by 4 hours).
Monitoring: TSH every 6-8 weeks until stable, then annually.
Target TSH: 0.5-2.5 mIU/L for most adults, 1-3 mIU/L for elderly.
Hyperthyroidism diagnosis: Low TSH with elevated free T4 or T3.
Common causes: Graves disease (most common), toxic multinodular goiter, toxic adenoma.
Graves disease treatment options: Antithyroid drugs (methimazole preferred, starting 10-30mg daily),
radioactive iodine ablation, or thyroidectomy.
Propylthiouracil reserved for first trimester pregnancy and thyroid storm.
Beta-blockers (propranolol) for symptomatic control of tremor and tachycardia.
Thyroid storm is a life-threatening emergency: high fever, tachycardia, altered mental status.
Thyroid nodules: Evaluate with TSH and ultrasound. FNA biopsy if suspicious features or size over 1cm.""",
        "category": "endocrinology"
    },
]


# ============================================================
# RAG Infrastructure (reused from main.py)
# ============================================================

def simple_chunk(text, chunk_size=80, overlap=15):
    """Split text into overlapping word-based chunks"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding vector for text"""
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def cosine_similarity(vec_a, vec_b):
    """Calculate cosine similarity between two vectors"""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_index(documents):
    """Process documents: chunk → embed → store in memory"""
    index = []
    for doc in documents:
        print(f"  📄 Processing: {doc['title']}...", end=" ")
        chunks = simple_chunk(doc["content"], chunk_size=80, overlap=15)
        for i, chunk_text in enumerate(chunks):
            embedding = get_embedding(chunk_text)
            index.append({
                "doc_id": doc["id"],
                "title": doc["title"],
                "category": doc["category"],
                "chunk_index": i,
                "text": chunk_text,
                "embedding": embedding
            })
        print(f"({len(chunks)} chunks)")
    print(f"  ✅ Index: {len(index)} chunks from {len(documents)} documents")
    return index


def retrieve(query, index, top_k=3):
    """Find the most relevant chunks for a query"""
    query_embedding = get_embedding(query)
    scored_chunks = []
    for item in index:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored_chunks.append({**item, "similarity": score})
    scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
    return scored_chunks[:top_k]


def generate_answer(query, retrieved_chunks):
    """LLM answers using retrieved context"""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(f"[Source {i+1}: {chunk['title']}]\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a medical knowledge assistant. Answer questions using ONLY
the provided context from clinical guidelines. If the context doesn't
contain enough information, say so. Cite sources as [Source X].
Educational purposes only."""},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer with source citations:"}
        ],
        max_tokens=500, temperature=0.2
    )
    return response.choices[0].message.content


# ============================================================
# Demo 1: Before and After Adding Documents
# ============================================================

def demo_before_after():
    """Show retrieval quality before and after adding new documents"""
    print("\n" + "=" * 70)
    print("DEMO 1: BEFORE vs AFTER ADDING NEW DOCUMENTS")
    print("=" * 70)

    # Questions specifically about NEW content
    test_questions = [
        "What are the 4 pillars of heart failure treatment?",
        "How is COPD classified by severity?",
        "What is the starting dose of levothyroxine for hypothyroidism?",
    ]

    # --- BEFORE: Only original 3 docs ---
    print("\n🔴 BEFORE: Knowledge base has only 3 documents")
    print("─" * 70)
    index_before = build_index(ORIGINAL_KNOWLEDGE_BASE)

    for q in test_questions:
        print(f"\n❓ {q}")
        retrieved = retrieve(q, index_before, top_k=2)
        print(f"   Best match: [{retrieved[0]['title']}] (similarity: {retrieved[0]['similarity']:.4f})")
        answer = generate_answer(q, retrieved)
        print(f"   📋 {answer[:200]}...")

    # --- AFTER: Add new documents ---
    print(f"\n\n{'═' * 70}")
    print("🟢 AFTER: Added 3 NEW documents (heart failure, COPD, thyroid)")
    print("─" * 70)
    all_docs = ORIGINAL_KNOWLEDGE_BASE + NEW_DOCUMENTS
    index_after = build_index(all_docs)

    for q in test_questions:
        print(f"\n❓ {q}")
        retrieved = retrieve(q, index_after, top_k=2)
        print(f"   Best match: [{retrieved[0]['title']}] (similarity: {retrieved[0]['similarity']:.4f})")
        answer = generate_answer(q, retrieved)
        print(f"   📋 {answer[:200]}...")

    print(f"""
{'═' * 70}
💡 OBSERVATION:
   BEFORE: The system retrieved WRONG documents (best it could find)
   AFTER:  The system retrieves the CORRECT new documents

   The RAG pipeline immediately benefits from new content —
   no retraining, no model changes, just add documents and rebuild index!
""")
    return index_after


# ============================================================
# Demo 2: Add a Document Interactively
# ============================================================

def demo_add_custom():
    """Let the user add their own document and test it"""
    print("\n" + "=" * 70)
    print("DEMO 2: ADD YOUR OWN DOCUMENT")
    print("=" * 70)

    print("\n📝 Enter a custom medical guideline to add to the knowledge base.\n")

    title = input("Document title (e.g., 'Migraine Management'): ").strip()
    if not title:
        title = "Custom Medical Guideline"

    print("Enter the document content (paste text, then press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line == "":
            if lines:
                break
            continue
        lines.append(line)

    if not lines:
        print("❌ No content entered.")
        return

    content = " ".join(lines)
    category = input("Category (e.g., 'neurology'): ").strip() or "general"

    custom_doc = {
        "id": f"custom_{title.lower().replace(' ', '_')}",
        "title": title,
        "content": content,
        "category": category
    }

    # Build index with original + custom
    all_docs = ORIGINAL_KNOWLEDGE_BASE + [custom_doc]
    print(f"\n🔨 Building index with your new document...")
    index = build_index(all_docs)

    # Test retrieval
    print(f"\n✅ Document added! Let's test retrieval.\n")
    while True:
        query = input("Test question (or 'quit'): ").strip()
        if query.lower() in ['quit', 'q', 'exit']:
            break
        if not query:
            continue

        retrieved = retrieve(query, index, top_k=3)
        print(f"\n   📎 Top matches:")
        for i, r in enumerate(retrieved):
            print(f"      {i+1}. [{r['title']}] (similarity: {r['similarity']:.4f})")

        answer = generate_answer(query, retrieved)
        print(f"\n   📋 {answer}\n")


# ============================================================
# Demo 3: Cross-Document Retrieval
# ============================================================

def demo_cross_document():
    """Questions that span multiple documents"""
    print("\n" + "=" * 70)
    print("DEMO 3: CROSS-DOCUMENT RETRIEVAL")
    print("=" * 70)
    print("\n   Questions that pull context from MULTIPLE documents\n")

    all_docs = ORIGINAL_KNOWLEDGE_BASE + NEW_DOCUMENTS
    print("🔨 Building index with all 6 documents...")
    index = build_index(all_docs)

    cross_questions = [
        "Which medications should a diabetic patient with heart failure be on?",
        "A patient has both COPD and depression — what are the treatment considerations?",
        "Compare how hypertension targets differ for general adults vs CKD patients",
    ]

    for q in cross_questions:
        print(f"\n❓ {q}")
        print("─" * 70)

        retrieved = retrieve(q, index, top_k=4)
        unique_docs = set()
        for r in retrieved:
            unique_docs.add(r['title'])
        print(f"   📎 Retrieved from {len(unique_docs)} different documents:")
        for r in retrieved:
            print(f"      • [{r['title']}] (sim: {r['similarity']:.4f})")

        answer = generate_answer(q, retrieved)
        print(f"\n   📋 {answer}\n")

    print(f"""
💡 KEY INSIGHT:
   RAG shines with cross-document questions!
   A single LLM call can synthesize information from multiple guidelines.
   This is incredibly powerful for clinical decision support.
""")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n📄 Exercise 1: Add New Medical Documents")
    print("=" * 70)
    print("Expand the knowledge base and test retrieval\n")

    print("Choose a demo:")
    print("1. Before vs after adding documents")
    print("2. Add your own custom document (interactive)")
    print("3. Cross-document retrieval")
    print("4. Run demo 1 + 3")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        demo_before_after()
    elif choice == "2":
        demo_add_custom()
    elif choice == "3":
        demo_cross_document()
    elif choice == "4":
        demo_before_after()
        demo_cross_document()
    else:
        print("Invalid choice")

    print(f"""
{'═' * 70}
KEY LEARNINGS:
{'═' * 70}
1. Adding documents is simple: create doc → chunk → embed → add to index
2. New content is IMMEDIATELY retrievable — no model retraining needed
3. The system automatically retrieves from the most relevant documents
4. Cross-document retrieval synthesizes information from multiple sources
5. In production, you'd add documents incrementally (not rebuild entire index)

🏥 REAL-WORLD APPLICATION:
   • New clinical guideline published → add to knowledge base → instant access
   • Formulary updated → add new drug information → clinicians get current answers
   • Hospital policy change → update the document → RAG serves the new policy
""")


if __name__ == "__main__":
    main()
