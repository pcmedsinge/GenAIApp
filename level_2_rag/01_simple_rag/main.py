"""
Project 1: RAG from Scratch
Build a complete Retrieval-Augmented Generation system using ONLY OpenAI API.
No frameworks — understand every step of the RAG pipeline.

What is RAG?
  R - Retrieval:  Find relevant documents from your knowledge base
  A - Augmented:  Add those documents to the LLM's context
  G - Generation: LLM generates an answer using that context

Why RAG?
  - LLMs have a knowledge cutoff and don't know YOUR data
  - RAG lets you connect LLMs to your own documents/databases
  - In healthcare: ground answers in actual clinical guidelines

Builds on: Level 1 Project 02 (Embeddings) — you already know how to embed and compare!
"""

import os
import time
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# STEP 1: Knowledge Base (Your Organization's Documents)
# ============================================================
# In production, these come from files, databases, or APIs.
# For learning, we define them as structured data.

MEDICAL_KNOWLEDGE_BASE = [
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
        "id": "asthma_protocol",
        "title": "Asthma Management Protocol",
        "content": """Asthma diagnosis confirmed by spirometry showing variable airflow obstruction
(FEV1 improvement of 12 percent or more after bronchodilator).
Step 1 Mild intermittent: As-needed low-dose ICS-formoterol or SABA.
Step 2 Mild persistent: Low-dose ICS daily (budesonide 200mcg or fluticasone 100mcg).
Step 3 Moderate persistent: Low-dose ICS-LABA combination (fluticasone-salmeterol or budesonide-formoterol).
Step 4 Severe: Medium to high-dose ICS-LABA combination.
Step 5 Very severe: Add-on therapy including tiotropium or biologics
(omalizumab for allergic asthma, mepolizumab for eosinophilic asthma).
All patients should have a written asthma action plan.
Assess control every 1-3 months. Step down therapy if well-controlled for 3 or more months.
Emergency management: Systemic corticosteroids (prednisone 40-60mg) for exacerbations.
Common triggers to avoid: allergens, smoke, cold air, exercise (use pre-treatment SABA).""",
        "category": "pulmonology"
    },
    {
        "id": "ckd_staging",
        "title": "Chronic Kidney Disease Staging and Management",
        "content": """CKD staging based on GFR: Stage 1 (GFR 90 or higher with kidney damage),
Stage 2 (GFR 60-89), Stage 3a (GFR 45-59), Stage 3b (GFR 30-44),
Stage 4 (GFR 15-29), Stage 5 (GFR less than 15, kidney failure).
Key management principles: Control blood pressure (target less than 130/80),
manage diabetes if present (HbA1c less than 7 percent),
prescribe ACE inhibitor or ARB for proteinuria even if blood pressure is normal.
Limit protein intake in Stage 4-5. Avoid nephrotoxic drugs: NSAIDs,
aminoglycoside antibiotics, IV contrast dye (prepare with hydration if needed).
Monitor: GFR every 3-6 months, urine albumin-to-creatinine ratio,
potassium, phosphorus, PTH levels.
Referral to nephrology recommended at Stage 4 or rapidly declining GFR.
Prepare for renal replacement therapy when GFR drops below 20.
Manage anemia with ESAs when hemoglobin falls below 10 g/dL.""",
        "category": "nephrology"
    },
    {
        "id": "anticoagulation",
        "title": "Anticoagulation Therapy Guidelines",
        "content": """Indications for anticoagulation: Atrial fibrillation (CHA2DS2-VASc score 2 or higher in men,
3 or higher in women), VTE (DVT or PE) treatment and prevention,
mechanical heart valves, antiphospholipid syndrome.
DOACs preferred over warfarin for most non-valvular atrial fibrillation:
apixaban 5mg twice daily, rivaroxaban 20mg daily with food,
dabigatran 150mg twice daily, edoxaban 60mg daily.
Dose adjustments for renal impairment: Apixaban reduced to 2.5mg twice daily
if 2 of 3 criteria met (age 80 or older, weight 60kg or less, creatinine 1.5 or higher).
Dabigatran contraindicated if creatinine clearance below 30.
Warfarin: Target INR 2-3 for atrial fibrillation, 2.5-3.5 for mechanical valves.
Reversal agents: Idarucizumab for dabigatran, andexanet alfa for factor Xa inhibitors,
vitamin K plus fresh frozen plasma or PCC for warfarin.
Assess bleeding risk with HAS-BLED score before starting anticoagulation.""",
        "category": "hematology"
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
# STEP 2: Chunking — Breaking Documents into Pieces
# ============================================================

def simple_chunk(text, chunk_size=100, overlap=20):
    """
    Split text into overlapping chunks of approximately chunk_size words.

    Why chunk?
    - LLM context windows are limited (and expensive)
    - Smaller chunks = more precise retrieval
    - Overlap ensures we don't lose info at chunk boundaries

    Args:
        text: The document text to chunk
        chunk_size: Target number of words per chunk
        overlap: Number of words to overlap between chunks
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


# ============================================================
# STEP 3: Embeddings — Convert Text to Vectors
# ============================================================

def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding vector for text (you mastered this in Level 1!)"""
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def cosine_similarity(vec_a, vec_b):
    """Calculate cosine similarity between two vectors"""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ============================================================
# STEP 4: Build the Index (Prepare for Retrieval)
# ============================================================

def build_index(documents):
    """
    Process all documents: chunk → embed → store in memory.
    This is done ONCE, then reused for many queries.
    """
    index = []
    total_chunks = 0

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
        total_chunks += len(chunks)

    print(f"\n  ✅ Index built: {total_chunks} chunks from {len(documents)} documents")
    return index


# ============================================================
# STEP 5: Retrieve — Find Relevant Chunks
# ============================================================

def retrieve(query, index, top_k=3):
    """
    Find the most relevant chunks for a user's question.
    This is the 'R' in RAG!

    1. Embed the query
    2. Compare to every chunk embedding
    3. Return top_k most similar chunks
    """
    query_embedding = get_embedding(query)

    scored_chunks = []
    for item in index:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored_chunks.append({**item, "similarity": score})

    scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
    return scored_chunks[:top_k]


# ============================================================
# STEP 6: Generate — LLM Answers with Context
# ============================================================

def generate_answer(query, retrieved_chunks):
    """
    Send user's question + retrieved context to LLM.
    This is the 'AG' in RAG!
    """
    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            f"[Source {i+1}: {chunk['title']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a medical knowledge assistant. Answer questions using ONLY
the provided context from clinical guidelines. If the context doesn't
contain enough information to answer fully, say so honestly.
Always cite which source(s) you used in your answer. Format: [Source X]
This is for educational purposes only, not clinical advice."""
            },
            {
                "role": "user",
                "content": f"""Context from medical guidelines:
{context}

Question: {query}

Provide a clear answer based on the context above, citing sources:"""
            }
        ],
        max_tokens=500,
        temperature=0.2  # Low temperature for factual accuracy
    )

    return response.choices[0].message.content, response.usage


# ============================================================
# DEMO 1: Complete RAG Pipeline (Step by Step)
# ============================================================

def demo_full_pipeline():
    """Walk through every step of the RAG pipeline"""
    print("\n" + "=" * 70)
    print("DEMO 1: COMPLETE RAG PIPELINE (Step by Step)")
    print("=" * 70)

    # Step 1: Knowledge Base
    print("\n📚 STEP 1 — LOAD: Your knowledge base")
    print(f"   {len(MEDICAL_KNOWLEDGE_BASE)} medical guideline documents:")
    for doc in MEDICAL_KNOWLEDGE_BASE:
        print(f"   • {doc['title']} [{doc['category']}]")

    # Step 2: Chunking
    print("\n✂️  STEP 2 — CHUNK: Breaking documents into pieces")
    sample = MEDICAL_KNOWLEDGE_BASE[0]
    chunks = simple_chunk(sample["content"], chunk_size=50, overlap=10)
    print(f"   '{sample['title']}' → {len(chunks)} chunks")
    for i, c in enumerate(chunks[:3]):
        print(f"   Chunk {i}: \"{c[:90]}...\"")
    if len(chunks) > 3:
        print(f"   ... and {len(chunks)-3} more chunks")

    # Step 3 & 4: Embed and Index
    print("\n🧮 STEP 3+4 — EMBED & STORE: Creating vector index")
    index = build_index(MEDICAL_KNOWLEDGE_BASE)

    # Step 5: Retrieve
    query = "What medication should I start for a patient with newly diagnosed Type 2 diabetes?"
    print(f"\n🔍 STEP 5 — RETRIEVE: Finding relevant chunks")
    print(f"   Question: \"{query}\"")

    retrieved = retrieve(query, index, top_k=3)
    print(f"\n   Top {len(retrieved)} matches:")
    for i, chunk in enumerate(retrieved):
        print(f"   {i+1}. [{chunk['title']}] (similarity: {chunk['similarity']:.4f})")
        print(f"      \"{chunk['text'][:100]}...\"")

    # Step 6: Generate
    print(f"\n🤖 STEP 6 — GENERATE: LLM answers with retrieved context")
    answer, usage = generate_answer(query, retrieved)

    print(f"\n{'─' * 70}")
    print(f"📋 ANSWER:\n\n{answer}")
    print(f"\n📊 Tokens: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion = {usage.total_tokens} total")
    print(f"   Cost: ~${usage.total_tokens * 0.00000015:.6f}")

    return index


# ============================================================
# DEMO 2: RAG vs No-RAG Comparison
# ============================================================

def demo_rag_vs_no_rag(index):
    """Show why RAG produces better, more trustworthy answers"""
    print("\n\n" + "=" * 70)
    print("DEMO 2: RAG vs NO-RAG — Why RAG Matters")
    print("=" * 70)

    questions = [
        "What is the target blood pressure for a hypertensive patient over 65?",
        "When should a CKD patient be referred to nephrology?",
        "What is the PHQ-9 score range for moderate depression?",
    ]

    for question in questions:
        print(f"\n❓ Question: {question}")
        print("─" * 70)

        # Without RAG
        print("\n🔴 WITHOUT RAG (LLM's training data only):")
        no_rag = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical assistant. Answer briefly in 2-3 sentences."},
                {"role": "user", "content": question}
            ],
            max_tokens=150, temperature=0.3
        )
        print(f"   {no_rag.choices[0].message.content}")

        # With RAG
        print("\n🟢 WITH RAG (LLM + your guidelines):")
        retrieved = retrieve(question, index, top_k=2)
        answer, _ = generate_answer(question, retrieved)
        print(f"   {answer}")

        print()

    print("=" * 70)
    print("""
💡 KEY DIFFERENCES:

🔴 Without RAG:                          🟢 With RAG:
• Uses training data (may be outdated)    • Uses YOUR specific guidelines
• Cannot cite sources                     • Cites exact sources [Source 1], etc.
• May hallucinate specific numbers        • Grounded in your documents
• No control over knowledge               • You control the knowledge base
• Can't update without retraining         • Update docs anytime, instant effect
""")


# ============================================================
# DEMO 3: Interactive RAG Q&A
# ============================================================

def demo_interactive_qa(index):
    """Ask your own questions against the medical knowledge base"""
    print("\n\n" + "=" * 70)
    print("DEMO 3: INTERACTIVE MEDICAL Q&A")
    print("=" * 70)
    print("\n💬 Ask questions about the medical knowledge base!")
    print("   Topics: hypertension, diabetes, asthma, CKD, anticoagulation, depression")
    print("   Type 'quit' to exit\n")

    while True:
        query = input("Your question: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue

        # Retrieve
        retrieved = retrieve(query, index, top_k=3)
        print(f"\n   📎 Retrieved {len(retrieved)} chunks:")
        for i, chunk in enumerate(retrieved):
            print(f"      {i+1}. [{chunk['title']}] (similarity: {chunk['similarity']:.3f})")

        # Generate
        answer, usage = generate_answer(query, retrieved)
        print(f"\n   📋 Answer:\n   {answer}")
        print(f"\n   (Tokens: {usage.total_tokens} | Cost: ~${usage.total_tokens * 0.00000015:.6f})\n")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n📚 Level 2, Project 1: RAG from Scratch")
    print("=" * 70)
    print("Build Retrieval-Augmented Generation with just OpenAI API")
    print("No frameworks — understand every step!\n")

    print("Choose a demo:")
    print("1. Full RAG pipeline (step-by-step walkthrough)")
    print("2. RAG vs No-RAG comparison (see why RAG matters)")
    print("3. Interactive Q&A (ask your own questions)")
    print("4. Run ALL demos")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        demo_full_pipeline()
    elif choice == "2":
        print("\n🔨 Building index first...")
        index = build_index(MEDICAL_KNOWLEDGE_BASE)
        demo_rag_vs_no_rag(index)
    elif choice == "3":
        print("\n🔨 Building index first...")
        index = build_index(MEDICAL_KNOWLEDGE_BASE)
        demo_interactive_qa(index)
    elif choice == "4":
        index = demo_full_pipeline()
        demo_rag_vs_no_rag(index)
        demo_interactive_qa(index)
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY TAKEAWAYS
{'=' * 70}

🏗️  THE RAG PIPELINE:
   1. LOAD    — Get your documents (guidelines, policies, records)
   2. CHUNK   — Split into smaller pieces (~50-200 words each)
   3. EMBED   — Convert each chunk to a vector (OpenAI embeddings)
   4. STORE   — Keep vectors for fast retrieval
   5. RETRIEVE — Find relevant chunks for user's question
   6. GENERATE — LLM answers using retrieved context2
   

📊 WHAT YOU BUILT:
   • Knowledge base: {len(MEDICAL_KNOWLEDGE_BASE)} medical guideline documents
   • Chunking: Word-based with overlap
   • Embeddings: text-embedding-3-small (from Level 1!)
   • Retrieval: Cosine similarity search
   • Generation: gpt-4o-mini with context injection

⚠️  LIMITATIONS (solved in next projects):
   • In-memory storage (data lost when program ends!)
   • Linear search through all chunks (slow for large datasets)
   • Basic chunking (may split sentences mid-thought)

🎯 NEXT: Move to 02_vector_databases to learn ChromaDB
   for persistent, fast, scalable vector storage!
""")


if __name__ == "__main__":
    main()
