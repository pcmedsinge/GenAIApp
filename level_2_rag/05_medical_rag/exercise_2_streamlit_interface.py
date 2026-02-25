"""
Exercise 2: Simple Web Interface with Streamlit
Build a Streamlit web app to interact with the Medical RAG system
through a browser-based UI instead of the command line.

Skills practiced:
- Creating web interfaces with Streamlit
- Connecting a UI to a RAG backend
- Displaying citations and confidence in a user-friendly way
- Session state management for chat history

Healthcare context:
  Clinicians won't use a command-line tool. A web interface makes your
  RAG system accessible to anyone with a browser — the first step
  toward a real clinical decision support tool.

To run:
  pip install streamlit       (if not already installed)
  streamlit run exercise_2_streamlit_interface.py
"""

import os
import json

# ============================================================
# Check for Streamlit availability
# ============================================================
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Medical Knowledge Base (from main.py)
# ============================================================

MEDICAL_DOCUMENTS = [
    {"id": "htn_def", "text": "Hypertension is defined as blood pressure consistently at or above 130/80 mmHg. Stage 1 is 130-139 systolic or 80-89 diastolic. Stage 2 is 140/90 or higher. Hypertensive crisis is above 180/120 requiring immediate intervention.", "metadata": {"specialty": "cardiology", "topic": "hypertension", "subtopic": "definition"}},
    {"id": "htn_meds", "text": "First-line antihypertensives: ACE inhibitors (lisinopril 10-40mg, enalapril 5-40mg), ARBs (losartan 50-100mg, valsartan 80-320mg), CCBs (amlodipine 2.5-10mg), thiazides (HCTZ 12.5-25mg). Start monotherapy. If not at target in 4-6 weeks, add second agent from different class or increase dose.", "metadata": {"specialty": "cardiology", "topic": "hypertension", "subtopic": "medications"}},
    {"id": "htn_special", "text": "Special populations in hypertension: Black patients may respond better to CCBs or thiazides as initial therapy. Patients with CKD or proteinuria should receive ACE/ARB. Patients with diabetes benefit from ACE/ARB. Pregnant patients should avoid ACE/ARB; use labetalol or nifedipine instead.", "metadata": {"specialty": "cardiology", "topic": "hypertension", "subtopic": "special_populations"}},
    {"id": "hf_pillars", "text": "Heart failure with reduced EF treatment has four medication pillars: (1) ARNI (sacubitril-valsartan), (2) Beta-blocker (carvedilol, metoprolol succinate, bisoprolol), (3) MRA (spironolactone or eplerenone), (4) SGLT2i (dapagliflozin or empagliflozin). All four improve survival.", "metadata": {"specialty": "cardiology", "topic": "heart_failure", "subtopic": "medications"}},
    {"id": "afib_anticoag", "text": "Atrial fibrillation anticoagulation: CHA2DS2-VASc score 2+ in men or 3+ in women warrants anticoagulation. DOACs preferred: apixaban 5mg BID, rivaroxaban 20mg daily with food, dabigatran 150mg BID. Warfarin if mechanical valve.", "metadata": {"specialty": "cardiology", "topic": "atrial_fibrillation", "subtopic": "anticoagulation"}},
    {"id": "dm_dx", "text": "Type 2 Diabetes diagnosis: fasting glucose 126+ mg/dL on two occasions, HbA1c 6.5%+, 2-hour OGTT 200+ mg/dL, or random glucose 200+ with symptoms. Prediabetes: fasting glucose 100-125, HbA1c 5.7-6.4%.", "metadata": {"specialty": "endocrinology", "topic": "diabetes", "subtopic": "diagnosis"}},
    {"id": "dm_firstline", "text": "Metformin is first-line for Type 2 diabetes. Start 500mg once daily, titrate to 2000mg daily. Contraindicated if eGFR below 30. GI side effects common; extended-release may help.", "metadata": {"specialty": "endocrinology", "topic": "diabetes", "subtopic": "metformin"}},
    {"id": "dm_addon", "text": "Add-on therapy after metformin: GLP-1 agonists (semaglutide, liraglutide) if cardiovascular disease or obesity. SGLT2 inhibitors if heart failure or CKD. DPP-4 inhibitors if cost-sensitive. Insulin if HbA1c 10%+.", "metadata": {"specialty": "endocrinology", "topic": "diabetes", "subtopic": "second_line"}},
    {"id": "asthma_steps", "text": "Asthma stepwise: Step 1 as-needed SABA. Step 2 low-dose ICS. Step 3 low-dose ICS-LABA. Step 4 medium-high dose ICS-LABA. Step 5 add tiotropium or biologic.", "metadata": {"specialty": "pulmonology", "topic": "asthma", "subtopic": "stepwise_therapy"}},
    {"id": "copd_mgmt", "text": "COPD: Group A bronchodilator PRN. Group B LAMA or LABA. Group E LAMA+LABA, add ICS if eosinophils 300+. Smoking cessation is the ONLY intervention to slow FEV1 decline.", "metadata": {"specialty": "pulmonology", "topic": "copd", "subtopic": "management"}},
    {"id": "ckd_staging", "text": "CKD staged by GFR: Stage 1 (90+), Stage 2 (60-89), Stage 3a (45-59), Stage 3b (30-44), Stage 4 (15-29), Stage 5 (below 15). Also classified by albuminuria.", "metadata": {"specialty": "nephrology", "topic": "ckd", "subtopic": "staging"}},
    {"id": "ckd_mgmt", "text": "CKD management: BP target less than 130/80. ACE/ARB for proteinuria. Avoid nephrotoxins. Monitor potassium, phosphorus, calcium. Refer to nephrology at Stage 4.", "metadata": {"specialty": "nephrology", "topic": "ckd", "subtopic": "management"}},
    {"id": "dep_screen", "text": "PHQ-9 depression screening: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-19 moderately severe, 20-27 severe. Screen all adults annually.", "metadata": {"specialty": "psychiatry", "topic": "depression", "subtopic": "screening"}},
    {"id": "dep_tx", "text": "Depression treatment: Mild — psychotherapy. Moderate — SSRI plus CBT. Severe — SSRI plus CBT; SNRI or mirtazapine if SSRI fails. Treatment-resistant: augment with aripiprazole, lithium, or bupropion.", "metadata": {"specialty": "psychiatry", "topic": "depression", "subtopic": "treatment"}},
]


# ============================================================
# RAG Backend Functions
# ============================================================

@st.cache_resource if STREAMLIT_AVAILABLE else lambda f: f
def init_rag_system():
    """Initialize the RAG system (cached to avoid reloading on every rerun)"""
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_kb_streamlit",
        embedding_function=ef
    )
    collection.add(
        ids=[d["id"] for d in MEDICAL_DOCUMENTS],
        documents=[d["text"] for d in MEDICAL_DOCUMENTS],
        metadatas=[d["metadata"] for d in MEDICAL_DOCUMENTS]
    )
    return openai_client, collection


def retrieve_sources(question, collection, n_results=4):
    """Retrieve relevant documents"""
    results = collection.query(query_texts=[question], n_results=n_results)
    sources = []
    for i in range(len(results["ids"][0])):
        sources.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return sources


def get_confidence(sources):
    """Compute confidence level from retrieval distances"""
    if not sources:
        return "LOW", "🔴"
    best_dist = sources[0]["distance"]
    if best_dist < 0.8:
        return "HIGH", "🟢"
    elif best_dist < 1.2:
        return "MEDIUM", "🟡"
    else:
        return "LOW", "🔴"


def generate_answer(question, sources, openai_client):
    """Generate cited answer"""
    context = "\n\n".join([
        f"[Source {i+1}: {s['metadata']['specialty']}/{s['metadata']['topic']}]\n{s['text']}"
        for i, s in enumerate(sources)
    ])
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a medical knowledge assistant. Answer ONLY from provided sources.
Cite every claim with [Source X]. If sources are insufficient, say so.
Add disclaimer: "For educational purposes only. Consult a healthcare provider for medical decisions."
Be specific with medication names and doses."""
            },
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=500, temperature=0.2
    )
    return response.choices[0].message.content, response.usage.total_tokens


# ============================================================
# Streamlit Web App
# ============================================================

def run_streamlit_app():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Medical RAG Assistant",
        page_icon="🏥",
        layout="wide"
    )

    st.title("🏥 Medical RAG Assistant")
    st.markdown("*Ask clinical questions — powered by RAG with source citations*")

    # Initialize system
    openai_client, collection = init_rag_system()

    # Sidebar
    with st.sidebar:
        st.header("📚 Knowledge Base")

        specialties = set(d["metadata"]["specialty"] for d in MEDICAL_DOCUMENTS)
        topics = set(d["metadata"]["topic"] for d in MEDICAL_DOCUMENTS)

        st.metric("Documents", len(MEDICAL_DOCUMENTS))
        st.metric("Specialties", len(specialties))
        st.metric("Topics", len(topics))

        st.markdown("**Specialties covered:**")
        for spec in sorted(specialties):
            st.markdown(f"- {spec.title()}")

        st.divider()
        st.markdown("**Settings:**")
        n_sources = st.slider("Number of sources", 1, 6, 4)
        show_distances = st.checkbox("Show retrieval distances", value=True)

        st.divider()
        st.caption("⚠️ For educational purposes only.")

    # Chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📎 View Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- **[{s['id']}]** {s['metadata']['specialty']}/{s['metadata']['topic']} "
                                    f"(dist: {s['distance']:.4f})")

    # Chat input
    if question := st.chat_input("Ask a medical question..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Process
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching knowledge base..."):
                sources = retrieve_sources(question, collection, n_results=n_sources)
                confidence, conf_icon = get_confidence(sources)
                answer, tokens = generate_answer(question, sources, openai_client)

            # Show confidence badge
            conf_colors = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}
            st.markdown(f"**Confidence:** :{conf_colors[confidence]}[{conf_icon} {confidence}]")

            # Show answer
            st.markdown(answer)

            # Show sources
            with st.expander("📎 View Sources"):
                for i, s in enumerate(sources):
                    dist_text = f" (distance: {s['distance']:.4f})" if show_distances else ""
                    st.markdown(f"**[Source {i+1}]** `{s['id']}` — "
                                f"{s['metadata']['specialty']}/{s['metadata']['topic']}{dist_text}")
                    st.caption(s["text"][:200] + "...")

            st.caption(f"Tokens used: {tokens}")

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "confidence": confidence,
            })

    # Sample questions section
    if not st.session_state.messages:
        st.markdown("---")
        st.markdown("### 💡 Try these sample questions:")
        cols = st.columns(2)
        sample_questions = [
            "What are the four pillars of heart failure treatment?",
            "How is Type 2 diabetes diagnosed?",
            "What medications treat hypertension in a patient with CKD?",
            "What is first-line treatment for depression?",
        ]
        for i, sq in enumerate(sample_questions):
            with cols[i % 2]:
                st.info(sq)


# ============================================================
# Non-Streamlit Fallback (terminal mode)
# ============================================================

def run_terminal_mode():
    """Run in terminal if Streamlit is not available or not used via 'streamlit run'"""
    print("\n🏥 Medical RAG — Terminal Mode")
    print("=" * 70)
    print("(Install streamlit and run with 'streamlit run' for the web interface)\n")

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_kb_terminal",
        embedding_function=ef
    )
    collection.add(
        ids=[d["id"] for d in MEDICAL_DOCUMENTS],
        documents=[d["text"] for d in MEDICAL_DOCUMENTS],
        metadatas=[d["metadata"] for d in MEDICAL_DOCUMENTS]
    )
    print(f"📦 Loaded {len(MEDICAL_DOCUMENTS)} documents\n")

    print("Choose a demo:")
    print("1. Interactive Q&A (terminal)")
    print("2. Sample questions showcase")
    print("3. Show Streamlit instructions")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        print("\n💬 Ask questions (type 'quit' to exit):\n")
        while True:
            question = input("You: ").strip()
            if question.lower() in ("quit", "exit", "q", ""):
                break

            sources = retrieve_sources(question, collection)
            confidence, conf_icon = get_confidence(sources)
            answer, tokens = generate_answer(question, sources, openai_client)

            print(f"\n   {conf_icon} Confidence: {confidence}")
            print(f"   📋 {answer}")
            print(f"   📎 Sources: {[s['id'] for s in sources]}")
            print(f"   Tokens: {tokens}\n")

    elif choice == "2":
        samples = [
            "What are the four pillars of heart failure treatment?",
            "How is Type 2 diabetes diagnosed?",
            "What medications treat hypertension in a CKD patient?",
        ]
        for q in samples:
            print(f"\n{'─' * 70}")
            print(f"❓ {q}")
            sources = retrieve_sources(q, collection)
            confidence, conf_icon = get_confidence(sources)
            answer, tokens = generate_answer(q, sources, openai_client)
            print(f"   {conf_icon} Confidence: {confidence}")
            print(f"   📋 {answer}\n")

    elif choice == "3":
        print(f"""
   📋 STREAMLIT SETUP:

   1. Install Streamlit:
      pip install streamlit

   2. Run the web app:
      cd level_2_rag/05_medical_rag
      streamlit run exercise_2_streamlit_interface.py

   3. Open your browser to http://localhost:8501

   Features:
   • Chat-style interface with message history
   • Confidence badges (🟢 HIGH / 🟡 MEDIUM / 🔴 LOW)
   • Expandable source citations
   • Sidebar with knowledge base stats and settings
   • Sample questions to get started
""")

    print(f"""
{'=' * 70}
KEY LEARNINGS — Exercise 2: Streamlit Interface
{'=' * 70}

1. STREAMLIT BASICS:
   • st.chat_input / st.chat_message for conversational UI
   • st.session_state for persisting data across reruns
   • st.sidebar for settings and information panels
   • st.cache_resource to avoid reloading the RAG system

2. UI DESIGN FOR HEALTHCARE:
   • Confidence indicators (color-coded) help clinicians trust/distrust answers
   • Expandable sources let users verify without cluttering the view
   • Disclaimer always visible — not optional in healthcare
   • Chat history maintains context for follow-up questions

3. BACKEND SEPARATION:
   • RAG functions (retrieve, generate) are independent of the UI
   • Same functions work in terminal mode or web mode
   • This pattern lets you swap UIs without rewriting logic

4. PRODUCTION CONSIDERATIONS:
   • Add authentication (streamlit-authenticator or SSO)
   • Log all queries and answers for audit trails
   • Rate limiting to control API costs
   • Deploy with streamlit cloud, Docker, or internal hosting
""")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    # If run with 'streamlit run', STREAMLIT_AVAILABLE is True
    # and Streamlit's runtime is active
    try:
        if STREAMLIT_AVAILABLE and hasattr(st, "runtime"):
            run_streamlit_app()
        else:
            run_terminal_mode()
    except Exception:
        run_terminal_mode()
