"""
Exercise 1: RAG-Backed Knowledge Base Tool (Level 2 + Level 3 Combined)

Skills practiced:
- Combining RAG retrieval (Level 2) with agent tool use (Level 3)
- Building ChromaDB-backed tools that agents can query
- Bridging vector search and function calling
- Understanding retrieval-augmented agents

Key insight: In the main project, clinical guidelines are stored in
  a simple dict. In production, there are THOUSANDS of guidelines.
  This exercise builds a RAG-backed tool — the agent decides WHEN
  to search and WHAT to search for, then gets relevant documents
  from a vector store. This is how real clinical AI retrieves knowledge.
"""

import os
import json
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    print("  Note: pip install chromadb for full RAG functionality")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# ============================================================
# Clinical Knowledge Base (simulated documents)
# ============================================================

CLINICAL_DOCUMENTS = [
    # Hypertension guidelines
    {"id": "htn_001", "text": "Hypertension Stage 1 (130-139/80-89): First-line therapy includes ACE inhibitors, ARBs, calcium channel blockers, or thiazide diuretics. Start with monotherapy. Reassess at 4-6 weeks. If BP not at target, add second agent from different class.", "category": "hypertension", "source": "AHA/ACC 2023"},
    {"id": "htn_002", "text": "Hypertension in Black patients: Preferred initial therapy is calcium channel blocker or thiazide diuretic. ACE inhibitors may be less effective as monotherapy but recommended if proteinuria present.", "category": "hypertension", "source": "AHA/ACC 2023"},
    {"id": "htn_003", "text": "Hypertension in CKD: ACE inhibitor or ARB recommended, especially with proteinuria. Target BP <130/80. Monitor potassium and creatinine 1-2 weeks after initiation or dose change.", "category": "hypertension", "source": "KDIGO 2024"},
    {"id": "htn_004", "text": "Resistant hypertension (uncontrolled on 3 agents): Add spironolactone 25mg. Check aldosterone-renin ratio. Consider secondary causes: renal artery stenosis, primary aldosteronism, pheochromocytoma, sleep apnea.", "category": "hypertension", "source": "AHA/ACC 2023"},
    # Diabetes guidelines
    {"id": "dm_001", "text": "Type 2 Diabetes first-line: Metformin 500mg daily, titrate to 2000mg. If HbA1c ≥1.5% above target, consider dual therapy from start. Monitor B12 annually.", "category": "diabetes", "source": "ADA 2024"},
    {"id": "dm_002", "text": "Type 2 Diabetes with cardiovascular disease: Add GLP-1 receptor agonist (semaglutide, liraglutide) regardless of HbA1c. Cardiovascular mortality benefit demonstrated.", "category": "diabetes", "source": "ADA 2024"},
    {"id": "dm_003", "text": "Type 2 Diabetes with heart failure or CKD: Add SGLT2 inhibitor (empagliflozin, dapagliflozin). Benefits include HbA1c reduction, weight loss, CV protection, renal protection.", "category": "diabetes", "source": "ADA 2024"},
    {"id": "dm_004", "text": "Insulin initiation in Type 2 Diabetes: Start basal insulin (glargine or detemir) 10 units or 0.1-0.2u/kg at bedtime. Titrate by 2 units every 3 days to fasting glucose target.", "category": "diabetes", "source": "ADA 2024"},
    # Heart failure guidelines  
    {"id": "hf_001", "text": "HFrEF (EF ≤40%) four pillars of therapy: 1) ARNI (sacubitril/valsartan) or ACEi, 2) Beta-blocker (carvedilol, metoprolol succinate, bisoprolol), 3) MRA (spironolactone), 4) SGLT2i. All improve survival.", "category": "heart_failure", "source": "AHA/ACC/HFSA 2023"},
    {"id": "hf_002", "text": "Heart failure decompensation: IV furosemide (2.5x oral dose), daily weights, I&O. Restrict sodium <2g, fluid 1.5-2L. Monitor BMP daily. Resume/titrate GDMT before discharge.", "category": "heart_failure", "source": "AHA/ACC/HFSA 2023"},
    {"id": "hf_003", "text": "HFpEF (EF ≥50%): SGLT2 inhibitor recommended (empagliflozin). Diuretics for congestion. Treat underlying causes (HTN, obesity, AFib). Exercise training improves symptoms.", "category": "heart_failure", "source": "AHA/ACC/HFSA 2023"},
    # CKD guidelines
    {"id": "ckd_001", "text": "CKD staging by GFR: Stage 1 (≥90), Stage 2 (60-89), Stage 3a (45-59), Stage 3b (30-44), Stage 4 (15-29), Stage 5 (<15). Refer nephrology at Stage 4.", "category": "ckd", "source": "KDIGO 2024"},
    {"id": "ckd_002", "text": "CKD medication adjustments: Metformin contraindicated GFR<30, caution 30-45. NSAIDs avoid in all stages. Gabapentin reduce dose. Apixaban 2.5mg BID if Cr≥1.5 + age≥80 or weight≤60kg.", "category": "ckd", "source": "KDIGO 2024"},
    {"id": "ckd_003", "text": "CKD-anemia: Start EPO when Hb <10. Iron first if ferritin <100 or TSAT <20%. Target Hb 10-11.5. Avoid Hb >13 (thrombosis risk).", "category": "ckd", "source": "KDIGO 2024"},
    # Anticoagulation
    {"id": "ac_001", "text": "Atrial fibrillation anticoagulation: Calculate CHA2DS2-VASc. Score ≥2 (men) or ≥3 (women): anticoagulate. DOACs preferred over warfarin (apixaban, rivaroxaban, edoxaban, dabigatran).", "category": "anticoagulation", "source": "AHA/ACC 2023"},
    {"id": "ac_002", "text": "Apixaban dosing: Standard 5mg BID. Reduce to 2.5mg BID if ≥2 of: age ≥80, weight ≤60kg, creatinine ≥1.5. In CKD: safe down to GFR 15. No dose adjustment for CrCl >25.", "category": "anticoagulation", "source": "Package Insert 2024"},
]


# ============================================================
# Build ChromaDB Knowledge Base
# ============================================================

def build_knowledge_base():
    """Create ChromaDB collection from clinical documents"""
    if not HAS_CHROMADB:
        print("  ChromaDB not available — using fallback keyword search")
        return None

    client = chromadb.Client()

    # Delete existing collection if present
    try:
        client.delete_collection("clinical_guidelines")
    except Exception:
        pass

    collection = client.create_collection(
        name="clinical_guidelines",
        metadata={"hnsw:space": "cosine"}
    )

    # Embed and store documents
    texts = [doc["text"] for doc in CLINICAL_DOCUMENTS]
    ids = [doc["id"] for doc in CLINICAL_DOCUMENTS]
    metadatas = [{"category": doc["category"], "source": doc["source"]} for doc in CLINICAL_DOCUMENTS]

    # Use OpenAI embeddings
    embedded = embeddings.embed_documents(texts)

    collection.add(
        documents=texts,
        embeddings=embedded,
        ids=ids,
        metadatas=metadatas,
    )

    print(f"  Knowledge base loaded: {collection.count()} clinical documents")
    return collection


# ============================================================
# RAG-Backed Tools
# ============================================================

# Global reference set by demo functions
_collection = None


def _rag_search(query: str, n_results: int = 3) -> list[dict]:
    """Search the vector store"""
    if _collection is not None:
        query_embedding = embeddings.embed_query(query)
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        docs = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i] if results.get("distances") else 0
            docs.append({"text": doc, "source": meta["source"], "category": meta["category"], "score": round(1 - dist, 3)})
        return docs
    else:
        # Fallback: keyword search
        query_lower = query.lower()
        matches = [d for d in CLINICAL_DOCUMENTS if query_lower in d["text"].lower()]
        return [{"text": d["text"], "source": d["source"], "category": d["category"], "score": 0.8} for d in matches[:n_results]]


@tool
def search_clinical_knowledge(query: str) -> str:
    """Search the clinical knowledge base using semantic search. Returns the most relevant clinical guidelines for a given query. Use this for evidence-based recommendations."""
    results = _rag_search(query, n_results=3)
    if not results:
        return f"No relevant guidelines found for: {query}"
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(f"{i}. [{r['source']}] (relevance: {r['score']})\n   {r['text']}")
    return "CLINICAL EVIDENCE:\n" + "\n\n".join(formatted)


@tool
def search_by_condition(condition: str) -> str:
    """Search guidelines for a specific medical condition (e.g., 'hypertension', 'diabetes', 'heart failure'). Returns all available guidelines for that condition."""
    results = _rag_search(condition, n_results=5)
    condition_results = [r for r in results if condition.lower() in r["category"] or condition.lower() in r["text"].lower()]
    if not condition_results:
        return f"No guidelines found for condition: {condition}"
    formatted = []
    for i, r in enumerate(condition_results, 1):
        formatted.append(f"{i}. [{r['source']}]\n   {r['text']}")
    return f"GUIDELINES FOR {condition.upper()}:\n" + "\n\n".join(formatted)


@tool
def lookup_medication(medication: str) -> str:
    """Look up medication information. Available: metformin, lisinopril, amlodipine, apixaban, sertraline, carvedilol."""
    from level_3_agents.o5_healthcare_agent.main import MEDICATION_DATABASE  # noqa
    # Inline version for standalone use
    meds = {
        "metformin": {"class": "Biguanide", "dose": "500-2000mg daily", "contraindications": "eGFR<30, liver disease", "interactions": "IV contrast (hold 48h)", "monitoring": "HbA1c, B12, renal function"},
        "lisinopril": {"class": "ACE Inhibitor", "dose": "10-40mg daily", "contraindications": "Pregnancy, angioedema hx, bilateral RAS", "interactions": "K+ supplements, NSAIDs, lithium", "monitoring": "BP, K+, Cr at 1-2 weeks"},
        "apixaban": {"class": "DOAC", "dose": "5mg BID (2.5mg BID if criteria met)", "contraindications": "Active bleeding, mechanical valve", "interactions": "Strong CYP3A4 inhibitors, aspirin", "monitoring": "Renal function, signs of bleeding"},
        "carvedilol": {"class": "Beta-blocker", "dose": "3.125-25mg BID", "contraindications": "Severe bradycardia, decompensated HF", "interactions": "Verapamil, digoxin", "monitoring": "HR, BP, weight"},
    }
    med = meds.get(medication.lower())
    if med:
        return json.dumps({"medication": medication, **med}, indent=2)
    return f"Medication '{medication}' not found. Available: {', '.join(meds.keys())}"


@tool
def interpret_lab(test: str, value: float) -> str:
    """Interpret a lab value. Available: hba1c, gfr, potassium, creatinine, systolic_bp."""
    tests = {
        "hba1c": lambda v: f"HbA1c {v}%: {'Normal (<5.7)' if v < 5.7 else 'Prediabetes (5.7-6.4)' if v < 6.5 else 'Diabetes (≥6.5). Target <7%.'}",
        "gfr": lambda v: f"GFR {v}: {'Normal (≥90)' if v >= 90 else 'Mild CKD (60-89)' if v >= 60 else 'Stage 3 CKD (30-59)' if v >= 30 else 'Stage 4 CKD (15-29)' if v >= 15 else 'Stage 5 CKD (<15)'}",
        "potassium": lambda v: f"K+ {v}: {'Low (<3.5)' if v < 3.5 else 'Normal (3.5-5.0)' if v <= 5.0 else 'High (>5.0)'}",
        "creatinine": lambda v: f"Cr {v}: {'Normal (0.7-1.3)' if 0.7 <= v <= 1.3 else 'Abnormal'}",
        "systolic_bp": lambda v: f"SBP {v}: {'Normal (<120)' if v < 120 else 'Elevated (120-129)' if v < 130 else 'HTN Stage 1 (130-139)' if v < 140 else 'HTN Stage 2 (≥140)'}",
    }
    fn = tests.get(test.lower())
    if fn:
        return fn(value)
    return f"Test '{test}' not available. Options: {', '.join(tests.keys())}"


# Combined tools
rag_tools = [search_clinical_knowledge, search_by_condition, lookup_medication, interpret_lab]


# ============================================================
# DEMO 1: RAG Tool in Action
# ============================================================

def demo_rag_tool():
    """Show the RAG tool retrieving relevant guidelines"""
    global _collection
    print("\n" + "=" * 70)
    print("DEMO 1: RAG-BACKED KNOWLEDGE TOOL")
    print("=" * 70)
    print("""
    The agent has a search_clinical_knowledge tool backed by a
    vector database (ChromaDB). Instead of a small dict lookup,
    it does SEMANTIC SEARCH across clinical guidelines.

    Dict lookup:  "hypertension" → one result
    RAG search:   "elderly patient with resistant HTN" → ranked results
    """)

    _collection = build_knowledge_base()

    queries = [
        "How to treat hypertension in a Black patient with CKD?",
        "What diabetes medication for a patient with heart failure?",
        "Apixaban dosing adjustment for elderly with renal impairment",
    ]

    for q in queries:
        print(f"\n{'─' * 60}")
        print(f"  QUERY: {q}")
        result = _rag_search(q, n_results=2)
        for i, r in enumerate(result, 1):
            print(f"\n  Result {i} (score: {r['score']}, source: {r['source']}):")
            print(f"    {r['text'][:200]}")


# ============================================================
# DEMO 2: Agent with RAG Tool
# ============================================================

def demo_agent_with_rag():
    """Agent uses RAG tool alongside other tools"""
    global _collection
    print("\n" + "=" * 70)
    print("DEMO 2: AGENT WITH RAG TOOL")
    print("=" * 70)
    print("""
    The agent decides WHEN to search the knowledge base and
    combines guideline evidence with medication lookups and
    lab interpretation — evidence-based medicine.
    """)

    _collection = build_knowledge_base()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Clinical Decision Support Agent with access to a
RAG-backed clinical knowledge base, medication database, and lab interpretation tools.

When answering clinical questions:
1. ALWAYS search the knowledge base first for relevant guidelines
2. Look up specific medications when needed
3. Interpret lab values in context
4. Cite your sources (guideline name)

Ground your recommendations in the evidence you retrieve.
Educational purposes only."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, rag_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=rag_tools, verbose=True, max_iterations=6)

    case = ("65-year-old Black male with Stage 3b CKD (GFR 38), diabetes (HbA1c 7.8%), "
            "heart failure (EF 35%), and uncontrolled HTN (BP 155/92). "
            "Currently on metformin 1000mg BID, lisinopril 20mg. "
            "What medications should be adjusted or added?")

    print(f"\n  CASE: {case}\n")
    result = executor.invoke({"input": case})
    print(f"\n  RECOMMENDATION:\n  {result['output']}")


# ============================================================
# DEMO 3: RAG vs Dict Comparison
# ============================================================

def demo_rag_vs_dict():
    """Compare RAG search vs simple dict lookup"""
    global _collection
    print("\n" + "=" * 70)
    print("DEMO 3: RAG vs DICT LOOKUP")
    print("=" * 70)

    _collection = build_knowledge_base()

    # Simple dict (from main.py)
    SIMPLE_GUIDELINES = {
        "hypertension": "Target BP <130/80 for most. First-line: ACEi, ARBs, CCBs, thiazides.",
        "diabetes_type2": "First-line: metformin. Add GLP-1 if CVD, SGLT2i if HF/CKD.",
        "heart_failure": "HFrEF four pillars: ARNI, beta-blocker, MRA, SGLT2i.",
    }

    queries = [
        ("hypertension in Black patient with kidney disease", "hypertension"),
        ("best diabetes medication for patient with heart failure", "diabetes_type2"),
        ("anticoagulation dosing for elderly", "anticoagulation"),
    ]

    for semantic_q, dict_key in queries:
        print(f"\n{'─' * 60}")
        print(f"  QUERY: {semantic_q}")

        # Dict lookup
        dict_result = SIMPLE_GUIDELINES.get(dict_key, "Not found")
        print(f"\n  DICT LOOKUP ('{dict_key}'):")
        print(f"    {dict_result}")

        # RAG search
        rag_results = _rag_search(semantic_q, n_results=2)
        print(f"\n  RAG SEARCH (semantic):")
        for i, r in enumerate(rag_results, 1):
            print(f"    {i}. [{r['source']}] (score: {r['score']})")
            print(f"       {r['text'][:150]}...")

    print("""
    KEY DIFFERENCES:
      Dict:  Exact key match → 1 generic result
      RAG:   Semantic match → multiple specific results ranked by relevance
             Handles complex queries, finds related concepts
             Returns source citations for evidence-based practice
    """)


# ============================================================
# DEMO 4: Interactive RAG Agent
# ============================================================

def demo_interactive():
    """Interactive RAG-backed clinical agent"""
    global _collection
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE RAG AGENT")
    print("=" * 70)
    print("  Ask clinical questions. Type 'quit' to exit.\n")

    _collection = build_knowledge_base()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a clinical decision support agent with RAG-backed guidelines, "
                   "medication lookup, and lab interpretation. Always search the knowledge "
                   "base for evidence before answering. Cite sources. Educational only."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, rag_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=rag_tools, verbose=False, max_iterations=6)
    history = []

    while True:
        question = input("  You: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue

        result = executor.invoke({"input": question, "chat_history": history})
        print(f"\n  Agent: {result['output']}\n")
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=result["output"]))


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 1: RAG-BACKED KNOWLEDGE TOOL")
    print("=" * 70)
    print("""
    Combines Level 2 (RAG with ChromaDB) + Level 3 (Agent tools).
    The agent searches a vector database of clinical guidelines
    to find evidence-based answers.

    Choose a demo:
      1 → RAG tool in action (semantic search)
      2 → Agent with RAG tool (decides when to search)
      3 → RAG vs dict lookup comparison
      4 → Interactive RAG agent
      5 → Run demos 1-3
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1": demo_rag_tool()
    elif choice == "2": demo_agent_with_rag()
    elif choice == "3": demo_rag_vs_dict()
    elif choice == "4": demo_interactive()
    elif choice == "5":
        demo_rag_tool()
        demo_agent_with_rag()
        demo_rag_vs_dict()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. RAG AS A TOOL: Instead of giving the agent ALL knowledge upfront,
   RAG lets it SEARCH for relevant info on demand. The agent decides
   what to search for based on the clinical question.

2. SEMANTIC vs KEYWORD: Vector search finds "apixaban dosing in
   elderly with kidney disease" even if those exact words don't
   appear. This is critical for clinical queries that use varied
   medical terminology.

3. LEVEL 2 + LEVEL 3 COMBINED: Level 2 taught you to build RAG
   pipelines. Level 3 taught you to build agents. This exercise
   puts the RAG pipeline INSIDE a tool that the agent calls.
   The agent orchestrates when and what to retrieve.

4. EVIDENCE-BASED PATTERN: In clinical AI, recommendations must
   be grounded in evidence. The RAG tool provides cited sources
   (ADA 2024, AHA/ACC 2023) — the agent can reference these
   in its answer, making it auditable.

5. SCALING: A dict with 6 entries works for a demo. In production,
   you'd have 10,000+ guidelines, formularies, protocols. RAG
   scales because the agent only retrieves what's relevant.
"""

if __name__ == "__main__":
    main()
