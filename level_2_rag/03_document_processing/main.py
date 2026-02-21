"""
Project 3: Document Processing — Loading & Chunking Strategies
The quality of your chunks determines the quality of your RAG system.

Key Insight: Chunking is where most RAG systems succeed or fail.
- Too large → noise in retrieval
- Too small → lost context
- No overlap → broken sentences at boundaries
- Good chunking → precise, relevant retrieval

Builds on: Project 01 (RAG pipeline) + Project 02 (ChromaDB storage)
"""

import os
import re
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# Sample Medical Document (simulates a real clinical guideline)
# ============================================================

CLINICAL_GUIDELINE = """
CLINICAL PRACTICE GUIDELINE: Management of Heart Failure

Section 1: Definition and Classification

Heart failure (HF) is a clinical syndrome caused by structural or functional cardiac
abnormalities resulting in reduced cardiac output or elevated intracardiac pressures.
Classification by ejection fraction: HFrEF (EF 40% or less), HFmrEF (EF 41-49%),
and HFpEF (EF 50% or more). The New York Heart Association (NYHA) functional
classification ranges from Class I (no limitation) to Class IV (symptoms at rest).

Section 2: Diagnosis

Diagnosis requires a combination of clinical history, physical exam, and diagnostic
testing. Key symptoms: dyspnea, orthopnea, paroxysmal nocturnal dyspnea, fatigue,
and peripheral edema. Essential tests include BNP or NT-proBNP (BNP greater than
100 pg/mL or NT-proBNP greater than 300 pg/mL suggests HF), echocardiography
to assess ejection fraction and structural abnormalities, ECG to identify arrhythmias
or ischemia, and chest X-ray to evaluate cardiomegaly and pulmonary congestion.

Section 3: Pharmacological Treatment for HFrEF

Guideline-directed medical therapy (GDMT) for HFrEF includes four pillars:
1. ACE inhibitor or ARB or ARNI (sacubitril-valsartan preferred if tolerated)
2. Beta-blocker (carvedilol, metoprolol succinate, or bisoprolol)
3. Mineralocorticoid receptor antagonist (spironolactone or eplerenone)
4. SGLT2 inhibitor (dapagliflozin or empagliflozin)

Titrate medications to target doses over weeks to months. Monitor blood pressure,
heart rate, kidney function, and potassium levels. Start low and go slow, especially
in elderly patients. Add hydralazine-isosorbide dinitrate for African American patients
or those intolerant to ACE/ARB/ARNI.

Section 4: Device Therapy

Consider ICD for primary prevention if EF 35% or less on optimal GDMT for 3 months.
CRT recommended for EF 35% or less with LBBB and QRS 150ms or more.
CRT may be considered for QRS 120-149ms with LBBB. Left ventricular assist device
(LVAD) for advanced HF as bridge to transplant or destination therapy.

Section 5: Lifestyle and Self-Management

Sodium restriction to less than 2000mg daily. Fluid restriction 1.5-2L daily for
patients with congestion. Daily weight monitoring (report gain of 2+ pounds overnight
or 5+ pounds in a week). Regular exercise: cardiac rehabilitation recommended.
Influenza and pneumococcal vaccination. Avoid NSAIDs, thiazolidinediones, and
most calcium channel blockers (except amlodipine) in HFrEF.

Section 6: Monitoring and Follow-up

Follow-up within 7-14 days after hospital discharge. Monitor BNP/NT-proBNP trends.
Reassess NYHA class and adjust therapy accordingly. Check renal function and
electrolytes 1-2 weeks after medication changes. Annual echocardiography to assess
EF trends. Consider palliative care referral for NYHA Class IV refractory symptoms.
"""


# ============================================================
# Chunking Strategy 1: Fixed-Size (Word Count)
# ============================================================

def chunk_fixed_size(text, chunk_size=100, overlap=20):
    """
    Split text into chunks of approximately chunk_size words.
    Simple and predictable, but may split mid-sentence.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ============================================================
# Chunking Strategy 2: Sentence-Based
# ============================================================

def chunk_by_sentences(text, sentences_per_chunk=3, overlap_sentences=1):
    """
    Split text into chunks of N sentences each.
    Respects sentence boundaries — no mid-sentence cuts.
    """
    # Split into sentences (handle medical abbreviations)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    start = 0

    while start < len(sentences):
        end = min(start + sentences_per_chunk, len(sentences))
        chunk = " ".join(sentences[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += sentences_per_chunk - overlap_sentences

    return chunks


# ============================================================
# Chunking Strategy 3: Section-Based (Semantic)
# ============================================================

def chunk_by_sections(text, section_pattern=r'\nSection \d+:'):
    """
    Split text at section headers — preserves topic coherence.
    Best for structured documents like clinical guidelines.
    """
    parts = re.split(section_pattern, text)
    headers = re.findall(section_pattern, text)

    chunks = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        header = headers[i - 1].strip() if i > 0 and i - 1 < len(headers) else ""
        chunk_text = f"{header}\n{part}" if header else part
        chunks.append(chunk_text.strip())

    return chunks


# ============================================================
# Embedding and Search Helpers
# ============================================================

def get_embedding(text):
    """Get embedding from OpenAI"""
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ============================================================
# DEMO 1: Compare Chunking Strategies
# ============================================================

def demo_compare_strategies():
    """See how different strategies split the same document"""
    print("\n" + "=" * 70)
    print("DEMO 1: CHUNKING STRATEGIES COMPARED")
    print("=" * 70)

    # Strategy 1: Fixed-size
    print("\n📏 STRATEGY 1: Fixed-Size (80 words, 15 word overlap)")
    fixed_chunks = chunk_fixed_size(CLINICAL_GUIDELINE, chunk_size=80, overlap=15)
    print(f"   Produced {len(fixed_chunks)} chunks")
    for i, chunk in enumerate(fixed_chunks[:3]):
        print(f"\n   Chunk {i}: \"{chunk[:120]}...\"")
    print(f"   ... ({len(fixed_chunks) - 3} more)")

    # Strategy 2: Sentence-based
    print(f"\n{'─' * 70}")
    print("\n📝 STRATEGY 2: Sentence-Based (4 sentences, 1 overlap)")
    sentence_chunks = chunk_by_sentences(CLINICAL_GUIDELINE, sentences_per_chunk=4, overlap_sentences=1)
    print(f"   Produced {len(sentence_chunks)} chunks")
    for i, chunk in enumerate(sentence_chunks[:3]):
        print(f"\n   Chunk {i}: \"{chunk[:120]}...\"")
    print(f"   ... ({len(sentence_chunks) - 3} more)")

    # Strategy 3: Section-based
    print(f"\n{'─' * 70}")
    print("\n📑 STRATEGY 3: Section-Based (by document sections)")
    section_chunks = chunk_by_sections(CLINICAL_GUIDELINE)
    print(f"   Produced {len(section_chunks)} chunks")
    for i, chunk in enumerate(section_chunks[:3]):
        first_line = chunk.split('\n')[0] if '\n' in chunk else chunk[:80]
        print(f"\n   Chunk {i}: \"{first_line}...\" ({len(chunk.split())} words)")
    if len(section_chunks) > 3:
        print(f"   ... ({len(section_chunks) - 3} more)")

    print(f"""
{'─' * 70}

💡 COMPARISON:

Strategy        | Chunks | Pros                    | Cons
─────────────────────────────────────────────────────────────────
Fixed-Size      |   {len(fixed_chunks):>2}   | Predictable size        | May split mid-sentence
Sentence-Based  |   {len(sentence_chunks):>2}   | Respects sentences      | Uneven chunk sizes
Section-Based   |   {len(section_chunks):>2}   | Preserves topic context | Large chunks, needs structure

RECOMMENDATION:
• Structured docs (guidelines, protocols) → Section-based
• Unstructured text (notes, articles) → Sentence-based with overlap
• When in doubt → Fixed-size with overlap is a safe default
""")


# ============================================================
# DEMO 2: Overlap Visualization
# ============================================================

def demo_overlap_visualization():
    """See exactly how overlap works and why it matters"""
    print("\n" + "=" * 70)
    print("DEMO 2: OVERLAP — Why It Matters")
    print("=" * 70)

    sample_text = ("Word1 Word2 Word3 Word4 Word5 "
                   "Word6 Word7 Word8 Word9 Word10 "
                   "Word11 Word12 Word13 Word14 Word15 "
                   "Word16 Word17 Word18 Word19 Word20")

    print(f"\n   Sample: \"{sample_text}\"\n")

    # Without overlap
    print("   ❌ WITHOUT overlap (chunk_size=5, overlap=0):")
    no_overlap = chunk_fixed_size(sample_text, chunk_size=5, overlap=0)
    for i, c in enumerate(no_overlap):
        print(f"      Chunk {i}: [{c}]")
    print("      Problem: Word5-Word6 boundary — if a sentence spans this, it's broken!")

    # With overlap
    print(f"\n   ✅ WITH overlap (chunk_size=5, overlap=2):")
    with_overlap = chunk_fixed_size(sample_text, chunk_size=5, overlap=2)
    for i, c in enumerate(with_overlap):
        print(f"      Chunk {i}: [{c}]")
    print("      Word4-Word5 appear in BOTH Chunk 0 and Chunk 1 — no lost context!")

    # Real medical example
    print(f"\n{'─' * 70}")
    print("\n   🏥 Real Example: Splitting a critical medical sentence\n")

    critical_text = ("Start medications at low doses. Monitor kidney function and "
                     "electrolytes when starting ACE inhibitor therapy. Titrate to "
                     "target doses gradually. Check potassium levels regularly.")

    print(f"   Text: \"{critical_text}\"\n")

    no_overlap_real = chunk_fixed_size(critical_text, chunk_size=8, overlap=0)
    print("   ❌ No overlap (chunk_size=8):")
    for i, c in enumerate(no_overlap_real):
        print(f"      Chunk {i}: \"{c}\"")

    with_overlap_real = chunk_fixed_size(critical_text, chunk_size=8, overlap=3)
    print(f"\n   ✅ With overlap (chunk_size=8, overlap=3):")
    for i, c in enumerate(with_overlap_real):
        print(f"      Chunk {i}: \"{c}\"")

    print("""
💡 OVERLAP RULES OF THUMB:
   • 10-20% overlap is typical (e.g., 100-word chunks, 15-word overlap)
   • More overlap = more redundancy but fewer missed connections
   • Too much overlap = wasted storage and slower search
   • For clinical text: err on the side of MORE overlap (critical info at boundaries)
""")


# ============================================================
# DEMO 3: Retrieval Quality Comparison
# ============================================================

def demo_retrieval_quality():
    """Same question, different chunking — which retrieves better?"""
    print("\n" + "=" * 70)
    print("DEMO 3: RETRIEVAL QUALITY — Chunking Matters!")
    print("=" * 70)

    test_questions = [
        "What are the four medication pillars for heart failure treatment?",
        "When should a patient get an ICD implant?",
        "What lifestyle changes are recommended for heart failure?"
    ]

    strategies = {
        "Fixed (80 words)": chunk_fixed_size(CLINICAL_GUIDELINE, 80, 15),
        "Sentences (4)": chunk_by_sentences(CLINICAL_GUIDELINE, 4, 1),
        "Sections": chunk_by_sections(CLINICAL_GUIDELINE),
    }

    # Pre-embed all chunks for each strategy
    print("\n🧮 Embedding all chunk strategies...")
    embedded_strategies = {}
    for name, chunks in strategies.items():
        print(f"   {name}: {len(chunks)} chunks...")
        embedded = [(chunk, get_embedding(chunk)) for chunk in chunks]
        embedded_strategies[name] = embedded

    # Test each question
    for question in test_questions:
        print(f"\n{'─' * 70}")
        print(f"❓ Question: \"{question}\"\n")

        q_embedding = get_embedding(question)

        for strategy_name, embedded_chunks in embedded_strategies.items():
            # Find best match
            best_score = -1
            best_chunk = ""
            for chunk_text, chunk_emb in embedded_chunks:
                score = cosine_similarity(q_embedding, chunk_emb)
                if score > best_score:
                    best_score = score
                    best_chunk = chunk_text

            print(f"   📊 {strategy_name:20s} → similarity: {best_score:.4f}")
            print(f"      Best chunk: \"{best_chunk[:100]}...\"")
            print()

    print("""
💡 WHAT TO NOTICE:
   • Section-based often wins for structured documents (entire topic together)
   • Fixed-size may split the answer across two chunks
   • Sentence-based is a good middle ground
   • The SAME question gets different quality answers depending on chunking!
   • This is why chunking strategy is one of the most impactful RAG decisions
""")


# ============================================================
# DEMO 4: Chunk Your Own Text
# ============================================================

def demo_chunk_your_text():
    """Input any text and see it chunked with all strategies"""
    print("\n" + "=" * 70)
    print("DEMO 4: CHUNK YOUR OWN TEXT")
    print("=" * 70)
    print("\nPaste any medical text (or press Enter for a sample):")
    print("End input with an empty line.\n")

    lines = []
    while True:
        line = input()
        if not line and lines:
            break
        if not line and not lines:
            # Use sample
            text = ("Patients with Type 2 diabetes should be started on metformin as first-line therapy. "
                    "If HbA1c remains above target after three months, a GLP-1 agonist should be added, "
                    "especially if the patient has cardiovascular disease. SGLT2 inhibitors are preferred "
                    "for patients with heart failure or chronic kidney disease. Regular monitoring of "
                    "kidney function is essential when using SGLT2 inhibitors.")
            print(f"   Using sample: \"{text[:80]}...\"\n")
            break
        lines.append(line)

    if lines:
        text = " ".join(lines)

    print(f"\n📏 Fixed-size (15 words):")
    for i, c in enumerate(chunk_fixed_size(text, 15, 3)):
        print(f"   {i}: \"{c}\"")

    print(f"\n📝 Sentence-based (2 sentences):")
    for i, c in enumerate(chunk_by_sentences(text, 2, 0)):
        print(f"   {i}: \"{c}\"")

    print(f"\n📑 Section-based:")
    sections = chunk_by_sections(text)
    if len(sections) <= 1:
        print("   (No section headers found — treating as single chunk)")
        print(f"   0: \"{text[:120]}...\"")
    else:
        for i, c in enumerate(sections):
            print(f"   {i}: \"{c[:120]}...\"")


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n✂️  Level 2, Project 3: Document Processing")
    print("=" * 70)
    print("Learn chunking strategies — the most impactful RAG decision\n")

    print("Choose a demo:")
    print("1. Compare chunking strategies (fixed, sentence, section)")
    print("2. Overlap visualization (why overlap matters)")
    print("3. Retrieval quality comparison (same question, different chunks)")
    print("4. Chunk your own text")
    print("5. Run ALL demos")

    choice = input("\nEnter choice (1-5): ").strip()

    demos = {
        "1": demo_compare_strategies,
        "2": demo_overlap_visualization,
        "3": demo_retrieval_quality,
        "4": demo_chunk_your_text,
    }

    if choice == "5":
        demo_compare_strategies()
        demo_overlap_visualization()
        demo_retrieval_quality()
    elif choice in demos:
        demos[choice]()
    else:
        print("Invalid choice")

    print(f"""
{'=' * 70}
KEY TAKEAWAYS
{'=' * 70}

✂️  CHUNKING STRATEGIES:
   • Fixed-size: Simple, predictable; may split sentences
   • Sentence-based: Respects grammar; uneven sizes
   • Section-based: Preserves topic context; needs structured docs

📏 OVERLAP GUIDELINES:
   • 10-20% overlap is standard
   • Clinical text → use more overlap (15-25%)
   • Overlap prevents critical info loss at boundaries

🎯 CHOOSING A STRATEGY:
   • Structured documents → Section-based (best for guidelines)
   • Unstructured text → Sentence-based with overlap
   • Don't know → Fixed-size with overlap (safe default)
   • ALWAYS test with real questions to validate your choice!

🔑 CHUNKING PARAMETERS TO TUNE:
   • Chunk size (50-500 words depending on content)
   • Overlap amount (10-25% of chunk size)
   • Strategy (fixed, sentence, section, or hybrid)
   • Metadata to attach (source, section header, page number)

🎯 NEXT: Move to 04_advanced_retrieval to learn re-ranking,
   multi-query, and citation techniques!
""")


if __name__ == "__main__":
    main()
