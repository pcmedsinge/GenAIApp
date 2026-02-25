"""
Exercise 4: User Feedback Loop for Quality Improvement

Skills practiced:
- Collecting and storing structured user feedback on RAG answers
- Analyzing ratings to identify weak topics and low-quality answers
- Tracking feedback trends over time (by specialty, topic, question type)
- Generating actionable improvement recommendations from feedback data
- Connecting quality metrics to healthcare quality improvement (QI) principles

Real-world context:
Healthcare QI programs rely on continuous feedback loops — provider surveys,
patient satisfaction scores, and peer review. This same pattern applies to
AI systems: collect feedback, analyze patterns, and improve iteratively.
A medical RAG system serving clinicians MUST track answer quality so
knowledge gaps and retrieval failures are caught early.
"""

import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict
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
# Medical Knowledge Base (from main.py)
# ============================================================

MEDICAL_DOCUMENTS = [
    {"id": "htn_def", "text": "Hypertension is defined as blood pressure consistently at or above 130/80 mmHg. Stage 1 is 130-139 systolic or 80-89 diastolic. Stage 2 is 140/90 or higher. Hypertensive crisis is above 180/120 requiring immediate intervention.", "metadata": {"specialty": "cardiology", "topic": "hypertension", "subtopic": "definition"}},
    {"id": "htn_meds", "text": "First-line antihypertensives: ACE inhibitors (lisinopril 10-40mg, enalapril 5-40mg), ARBs (losartan 50-100mg, valsartan 80-320mg), CCBs (amlodipine 2.5-10mg), thiazides (HCTZ 12.5-25mg). Start monotherapy. If not at target in 4-6 weeks, add second agent from different class or increase dose. Most Stage 2 patients need combination therapy from the start.", "metadata": {"specialty": "cardiology", "topic": "hypertension", "subtopic": "medications"}},
    {"id": "htn_special", "text": "Special populations in hypertension: Black patients may respond better to CCBs or thiazides as initial therapy. Patients with CKD or proteinuria should receive ACE/ARB. Patients with diabetes benefit from ACE/ARB. Pregnant patients should avoid ACE/ARB; use labetalol or nifedipine instead. Elderly patients start at lower doses.", "metadata": {"specialty": "cardiology", "topic": "hypertension", "subtopic": "special_populations"}},
    {"id": "hf_pillars", "text": "Heart failure with reduced EF treatment has four medication pillars that should ALL be started: (1) ARNI (sacubitril-valsartan) preferred over ACEi/ARB, (2) Beta-blocker (carvedilol, metoprolol succinate, or bisoprolol), (3) MRA (spironolactone or eplerenone), (4) SGLT2i (dapagliflozin or empagliflozin). Start low, titrate to target doses. All four improve survival.", "metadata": {"specialty": "cardiology", "topic": "heart_failure", "subtopic": "medications"}},
    {"id": "afib_anticoag", "text": "Atrial fibrillation anticoagulation: Calculate CHA2DS2-VASc score. Score 2+ in men or 3+ in women warrants anticoagulation. DOACs preferred: apixaban 5mg BID (reduce to 2.5mg if age 80+, weight 60kg or less, Cr 1.5+), rivaroxaban 20mg daily with food, dabigatran 150mg BID. Warfarin if mechanical valve (target INR 2-3).", "metadata": {"specialty": "cardiology", "topic": "atrial_fibrillation", "subtopic": "anticoagulation"}},
    {"id": "dm_dx", "text": "Type 2 Diabetes diagnosis: fasting glucose 126+ mg/dL on two occasions, HbA1c 6.5%+, 2-hour OGTT 200+ mg/dL, or random glucose 200+ with symptoms. Prediabetes: fasting glucose 100-125, HbA1c 5.7-6.4%, 2-hour OGTT 140-199. Screen adults 35-70 who are overweight.", "metadata": {"specialty": "endocrinology", "topic": "diabetes", "subtopic": "diagnosis"}},
    {"id": "dm_firstline", "text": "Metformin is first-line for Type 2 diabetes. Start 500mg once daily with meals, titrate to 2000mg daily. Contraindicated if eGFR below 30. Reduce dose if eGFR 30-45. GI side effects common initially; extended-release may help. Does not cause hypoglycemia. Monitor B12 levels annually with long-term use.", "metadata": {"specialty": "endocrinology", "topic": "diabetes", "subtopic": "metformin"}},
    {"id": "dm_addon", "text": "Add-on therapy if HbA1c not at target after 3 months of metformin: GLP-1 agonists (semaglutide, liraglutide) preferred if cardiovascular disease or obesity — provide weight loss and cardiovascular benefit. SGLT2 inhibitors (empagliflozin, dapagliflozin) preferred if heart failure or CKD. DPP-4 inhibitors (sitagliptin) if cost-sensitive. Insulin if HbA1c very high (10%+) or symptomatic.", "metadata": {"specialty": "endocrinology", "topic": "diabetes", "subtopic": "second_line"}},
    {"id": "asthma_steps", "text": "Asthma stepwise management: Step 1 (intermittent) as-needed SABA or low-dose ICS-formoterol. Step 2 (mild persistent) low-dose ICS daily. Step 3 (moderate) low-dose ICS-LABA. Step 4 (severe) medium-high dose ICS-LABA. Step 5 (very severe) add tiotropium or biologic. Assess control every 1-3 months and step down if well-controlled for 3+ months.", "metadata": {"specialty": "pulmonology", "topic": "asthma", "subtopic": "stepwise_therapy"}},
    {"id": "copd_mgmt", "text": "COPD management based on GOLD classification. Group A: bronchodilator PRN. Group B: LAMA (tiotropium) or LABA. Group E (exacerbation history): LAMA+LABA, add ICS if eosinophils 300+. Smoking cessation is the ONLY intervention proven to slow FEV1 decline. Pulmonary rehabilitation improves quality of life. Annual flu vaccine and pneumococcal vaccine recommended.", "metadata": {"specialty": "pulmonology", "topic": "copd", "subtopic": "management"}},
    {"id": "ckd_staging", "text": "CKD staged by GFR: Stage 1 (GFR 90+, kidney damage present), Stage 2 (60-89), Stage 3a (45-59), Stage 3b (30-44), Stage 4 (15-29), Stage 5 (below 15). Also classified by albuminuria: A1 (normal, below 30), A2 (moderate, 30-300), A3 (severe, above 300). Both GFR and albuminuria determine risk.", "metadata": {"specialty": "nephrology", "topic": "ckd", "subtopic": "staging"}},
    {"id": "ckd_mgmt", "text": "CKD management: BP target less than 130/80. ACE/ARB for proteinuria. Avoid nephrotoxins (NSAIDs, aminoglycosides, IV contrast without preparation). Adjust medication doses for GFR. Monitor potassium, phosphorus, calcium, PTH. Refer to nephrology at Stage 4 or rapid decline. Prepare for dialysis access when GFR below 20.", "metadata": {"specialty": "nephrology", "topic": "ckd", "subtopic": "management"}},
    {"id": "dep_screen", "text": "PHQ-9 depression screening: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-19 moderately severe, 20-27 severe. Screen all adults annually. For positive screens, assess for suicidal ideation (PHQ-9 question 9), substance use, bipolar symptoms (mood disorders questionnaire), and medical causes of depression.", "metadata": {"specialty": "psychiatry", "topic": "depression", "subtopic": "screening"}},
    {"id": "dep_tx", "text": "Depression treatment: Mild — watchful waiting, psychotherapy, or lifestyle changes. Moderate — SSRI (sertraline, escitalopram, fluoxetine) plus psychotherapy (CBT). Severe — SSRI plus CBT; consider SNRI (venlafaxine, duloxetine) or mirtazapine if SSRI fails. Treatment-resistant (failed 2+ adequate trials) — augment with aripiprazole, lithium, or bupropion; refer to psychiatry.", "metadata": {"specialty": "psychiatry", "topic": "depression", "subtopic": "treatment"}},
]


# ============================================================
# RAG System (reused from main.py)
# ============================================================

def build_knowledge_base():
    """Set up ChromaDB collection with all medical documents"""
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="medical_kb_feedback",
        embedding_function=openai_ef
    )
    collection.add(
        ids=[d["id"] for d in MEDICAL_DOCUMENTS],
        documents=[d["text"] for d in MEDICAL_DOCUMENTS],
        metadatas=[d["metadata"] for d in MEDICAL_DOCUMENTS]
    )
    print(f"  Knowledge base loaded: {collection.count()} documents "
          f"across {len(set(d['metadata']['specialty'] for d in MEDICAL_DOCUMENTS))} specialties")
    return collection


def multi_query_retrieve(question, collection, n_results=4):
    """Retrieve using multiple query phrasings for broader coverage"""
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
    """Generate answer with citations and confidence"""
    context = "\n\n".join([
        f"[Source {i+1}: {s['metadata']['specialty']}/{s['metadata']['topic']}]\n{s['text']}"
        for i, s in enumerate(sources)
    ])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a medical knowledge assistant powered by a clinical guidelines database.

RULES:
1. Answer ONLY from the provided sources. Do not use outside knowledge.
2. Cite every clinical claim with [Source X].
3. If sources don't contain enough information, say "Based on available sources, I cannot fully answer this."
4. End with a confidence level: HIGH (directly answered by sources), MEDIUM (partially answered), LOW (sources tangentially related).
5. Add disclaimer: "For educational purposes only. Consult a healthcare provider for medical decisions."

Be specific with medication names, doses, and criteria."""
            },
            {
                "role": "user",
                "content": f"Sources:\n{context}\n\nQuestion: {question}\n\nProvide a cited answer with confidence level:"
            }
        ],
        max_tokens=600, temperature=0.2
    )
    return response.choices[0].message.content, response.usage


# ============================================================
# Feedback System
# ============================================================

class FeedbackStore:
    """
    Collects and stores structured feedback on RAG answers.
    
    In production, this would use a database (Postgres, MongoDB, etc.)
    Here we use an in-memory store for demonstration.
    
    Healthcare analogy: Think of this like a clinical quality database —
    tracking outcomes (answer quality) to identify areas for improvement,
    similar to how hospitals track readmission rates or patient satisfaction.
    """

    def __init__(self):
        self.feedback_records = []
        self.next_id = 1

    def record_feedback(self, question, answer, sources, rating,
                        comment="", specialty_tags=None):
        """
        Store a feedback entry.

        Args:
            question: The user's original question
            answer: The RAG system's generated response
            sources: List of source dicts used for the answer
            rating: Integer 1-5 (1=unhelpful, 5=excellent)
            comment: Optional free-text comment from the user
            specialty_tags: Specialties identified in the sources
        """
        if specialty_tags is None:
            specialty_tags = list(set(
                s["metadata"]["specialty"] for s in sources
            ))

        topic_tags = list(set(
            s["metadata"]["topic"] for s in sources
        ))

        record = {
            "id": self.next_id,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "sources": [
                {"id": s["id"], "distance": s["distance"],
                 "specialty": s["metadata"]["specialty"],
                 "topic": s["metadata"]["topic"]}
                for s in sources
            ],
            "rating": rating,
            "comment": comment,
            "specialty_tags": specialty_tags,
            "topic_tags": topic_tags,
            "source_count": len(sources),
            "avg_distance": sum(s["distance"] for s in sources) / len(sources) if sources else 0,
        }

        self.feedback_records.append(record)
        self.next_id += 1
        return record["id"]

    def get_all(self):
        """Return all feedback records"""
        return self.feedback_records

    def get_by_rating(self, min_rating=None, max_rating=None):
        """Filter feedback by rating range"""
        results = self.feedback_records
        if min_rating is not None:
            results = [r for r in results if r["rating"] >= min_rating]
        if max_rating is not None:
            results = [r for r in results if r["rating"] <= max_rating]
        return results

    def get_by_specialty(self, specialty):
        """Filter feedback by specialty"""
        return [r for r in self.feedback_records
                if specialty in r["specialty_tags"]]

    def get_by_topic(self, topic):
        """Filter feedback by topic"""
        return [r for r in self.feedback_records
                if topic in r["topic_tags"]]

    def count(self):
        return len(self.feedback_records)


# ============================================================
# Feedback Analytics Engine
# ============================================================

class FeedbackAnalytics:
    """
    Analyzes feedback to surface quality patterns and improvement areas.
    
    Healthcare analogy: Like a hospital's quality dashboard that tracks
    infection rates by unit, readmissions by diagnosis, and patient
    satisfaction by department — but for AI answer quality.
    """

    def __init__(self, store: FeedbackStore):
        self.store = store

    def overall_summary(self):
        """High-level quality metrics"""
        records = self.store.get_all()
        if not records:
            return {"total": 0, "message": "No feedback collected yet"}

        ratings = [r["rating"] for r in records]
        return {
            "total_feedback": len(records),
            "avg_rating": round(sum(ratings) / len(ratings), 2),
            "rating_distribution": {
                i: ratings.count(i) for i in range(1, 6)
            },
            "satisfaction_rate": round(
                len([r for r in ratings if r >= 4]) / len(ratings) * 100, 1
            ),
            "low_quality_count": len([r for r in ratings if r <= 2]),
        }

    def ratings_by_specialty(self):
        """Break down quality by medical specialty"""
        specialty_data = defaultdict(list)
        for record in self.store.get_all():
            for spec in record["specialty_tags"]:
                specialty_data[spec].append(record["rating"])

        results = {}
        for spec, ratings in specialty_data.items():
            results[spec] = {
                "count": len(ratings),
                "avg_rating": round(sum(ratings) / len(ratings), 2),
                "low_quality_pct": round(
                    len([r for r in ratings if r <= 2]) / len(ratings) * 100, 1
                ),
            }
        return dict(sorted(results.items(), key=lambda x: x[1]["avg_rating"]))

    def ratings_by_topic(self):
        """Break down quality by topic"""
        topic_data = defaultdict(list)
        for record in self.store.get_all():
            for topic in record["topic_tags"]:
                topic_data[topic].append(record["rating"])

        results = {}
        for topic, ratings in topic_data.items():
            results[topic] = {
                "count": len(ratings),
                "avg_rating": round(sum(ratings) / len(ratings), 2),
            }
        return dict(sorted(results.items(), key=lambda x: x[1]["avg_rating"]))

    def worst_answers(self, n=5):
        """Return the N lowest-rated answers for review"""
        low = self.store.get_by_rating(max_rating=2)
        low.sort(key=lambda r: (r["rating"], r["avg_distance"]))
        return low[:n]

    def distance_vs_rating_correlation(self):
        """
        Check if retrieval distance predicts answer quality.
        If low-distance (close match) answers still get poor ratings,
        the problem is in generation, not retrieval.
        """
        records = self.store.get_all()
        if len(records) < 3:
            return {"message": "Need more data for correlation analysis"}

        # Bucket by distance
        close = [r for r in records if r["avg_distance"] < 0.8]
        medium = [r for r in records if 0.8 <= r["avg_distance"] < 1.2]
        far = [r for r in records if r["avg_distance"] >= 1.2]

        def avg_rating(bucket):
            if not bucket:
                return None
            return round(sum(r["rating"] for r in bucket) / len(bucket), 2)

        return {
            "close_match (<0.8)": {"count": len(close), "avg_rating": avg_rating(close)},
            "medium_match (0.8-1.2)": {"count": len(medium), "avg_rating": avg_rating(medium)},
            "far_match (>1.2)": {"count": len(far), "avg_rating": avg_rating(far)},
            "interpretation": (
                "If close-match answers have low ratings, the problem is in "
                "answer generation (prompt/model). If only far-match answers "
                "have low ratings, the problem is in retrieval or content gaps."
            ),
        }

    def identify_content_gaps(self):
        """
        Find questions where rating was low AND retrieval distance was high.
        These likely represent missing content in the knowledge base.
        """
        records = self.store.get_all()
        gaps = []
        for r in records:
            if r["rating"] <= 2 and r["avg_distance"] > 1.0:
                gaps.append({
                    "question": r["question"],
                    "rating": r["rating"],
                    "avg_distance": round(r["avg_distance"], 4),
                    "closest_topics": r["topic_tags"],
                    "comment": r.get("comment", ""),
                })
        return gaps

    def generate_improvement_report(self):
        """
        AI-generated report: analyze feedback data and create
        specific, actionable recommendations.
        """
        summary = self.overall_summary()
        if summary.get("total_feedback", 0) == 0:
            return "No feedback data to analyze."

        by_specialty = self.ratings_by_specialty()
        by_topic = self.ratings_by_topic()
        worst = self.worst_answers(3)
        gaps = self.identify_content_gaps()

        report_data = {
            "summary": summary,
            "by_specialty": by_specialty,
            "by_topic": by_topic,
            "worst_answers": [
                {"question": w["question"], "rating": w["rating"],
                 "comment": w.get("comment", ""), "topics": w["topic_tags"]}
                for w in worst
            ],
            "content_gaps": gaps,
        }

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a healthcare AI quality improvement analyst. 
Analyze the feedback data and generate a concise improvement report with:
1. Overall system health assessment
2. Top 3 specific problems identified
3. Recommended actions (prioritized)
4. Content gaps that need new documents
5. Topics where answer quality is weakest

Use healthcare QI language: root cause analysis, PDSA cycles, continuous improvement.
Be specific and actionable — name exact specialties, topics, and failure patterns."""
                },
                {
                    "role": "user",
                    "content": f"Feedback data:\n{json.dumps(report_data, indent=2)}\n\nGenerate improvement report:"
                }
            ],
            max_tokens=800, temperature=0.3
        )

        return response.choices[0].message.content


# ============================================================
# Simulated Feedback Data Generator
# ============================================================

def generate_simulated_feedback(collection, store):
    """
    Generate realistic feedback by running questions through the RAG system
    and assigning simulated ratings.
    
    Simulates a real-world scenario where clinicians use the system daily
    and rate answers — some questions match well, others expose gaps.
    """
    # Mix of well-covered and poorly-covered questions
    test_scenarios = [
        # Well-covered: should get high ratings
        {
            "question": "What are the first-line medications for hypertension?",
            "simulated_rating": 5,
            "comment": "Comprehensive answer with specific medications and doses.",
        },
        {
            "question": "How is Type 2 diabetes diagnosed?",
            "simulated_rating": 5,
            "comment": "Clear diagnostic criteria, very helpful.",
        },
        {
            "question": "What are the four pillars of heart failure treatment?",
            "simulated_rating": 4,
            "comment": "Good overview, would like more on titration timing.",
        },
        {
            "question": "When should I start anticoagulation for atrial fibrillation?",
            "simulated_rating": 4,
            "comment": "CHA2DS2-VASc explanation was clear.",
        },
        {
            "question": "What is the stepwise approach to asthma management?",
            "simulated_rating": 5,
            "comment": "Step-by-step breakdown was exactly what I needed.",
        },

        # Partially covered: medium ratings
        {
            "question": "How do you manage a patient with both diabetes and CKD?",
            "simulated_rating": 3,
            "comment": "Found info on both conditions separately but didn't synthesize well.",
        },
        {
            "question": "What blood pressure targets should I use for a diabetic patient?",
            "simulated_rating": 3,
            "comment": "Got CKD BP target but not specific diabetic hypertension guidance.",
        },
        {
            "question": "How is depression managed in elderly patients with heart failure?",
            "simulated_rating": 2,
            "comment": "Generic depression info; no geriatric or cardiac interaction considerations.",
        },

        # Poorly covered / out of scope: low ratings
        {
            "question": "What is the workup for a patient presenting with acute chest pain?",
            "simulated_rating": 1,
            "comment": "No content on acute chest pain differential or workup.",
        },
        {
            "question": "How should thyroid nodules be evaluated and managed?",
            "simulated_rating": 1,
            "comment": "No thyroid content at all in the knowledge base.",
        },
        {
            "question": "What are the drug interactions between SSRIs and DOACs?",
            "simulated_rating": 2,
            "comment": "Has info on both drug classes separately but nothing on interactions.",
        },
        {
            "question": "How do you manage acute kidney injury in the ED setting?",
            "simulated_rating": 1,
            "comment": "Only has CKD content, nothing on AKI.",
        },
        {
            "question": "What is the recommended insulin regimen for Type 1 diabetes?",
            "simulated_rating": 2,
            "comment": "Only covers Type 2 diabetes. No Type 1 content.",
        },
        {
            "question": "What vaccines are recommended for immunocompromised patients?",
            "simulated_rating": 1,
            "comment": "No immunization content in the knowledge base.",
        },
        {
            "question": "CKD staging criteria and management based on GFR?",
            "simulated_rating": 5,
            "comment": "Perfect answer with clear staging and management steps.",
        },
    ]

    print(f"\n  Generating feedback from {len(test_scenarios)} simulated queries...\n")

    for i, scenario in enumerate(test_scenarios):
        question = scenario["question"]
        sources = multi_query_retrieve(question, collection, n_results=4)

        answer, usage = generate_cited_answer(question, sources)

        feedback_id = store.record_feedback(
            question=question,
            answer=answer,
            sources=sources,
            rating=scenario["simulated_rating"],
            comment=scenario["comment"],
        )

        stars = "★" * scenario["simulated_rating"] + "☆" * (5 - scenario["simulated_rating"])
        print(f"  [{i+1:2d}] {stars}  {question[:60]}...")

    print(f"\n  ✅ {store.count()} feedback records stored")
    return store


# ============================================================
# DEMO 1: Feedback Collection & Rating System
# ============================================================

def demo_feedback_collection():
    """Demonstrate collecting ratings on RAG answers interactively"""
    print("\n" + "=" * 70)
    print("DEMO 1: FEEDBACK COLLECTION & RATING SYSTEM")
    print("=" * 70)
    print("""
    Healthcare parallel: Like patient satisfaction surveys (HCAHPS) or
    provider peer review — systematic collection of quality data.
    
    We'll ask the RAG system questions, then rate each answer.
    """)

    collection = build_knowledge_base()
    store = FeedbackStore()

    questions = [
        "What are the first-line medications for hypertension?",
        "How is heart failure with reduced ejection fraction treated?",
        "What is the recommended depression screening tool?",
    ]

    for question in questions:
        print(f"\n{'─' * 60}")
        print(f"  ❓ {question}\n")

        sources = multi_query_retrieve(question, collection, n_results=4)
        answer, usage = generate_cited_answer(question, sources)

        print(f"  📋 ANSWER:\n")
        for line in answer.split('\n'):
            print(f"    {line}")

        print(f"\n  📎 Sources: {', '.join(s['id'] for s in sources)}")
        print(f"  📏 Avg distance: {sum(s['distance'] for s in sources)/len(sources):.4f}")

        # Simulate user rating (in production, this would be interactive)
        # Assign high ratings since these are well-covered topics
        rating = 5 if "hypertension" in question.lower() or "heart failure" in question.lower() else 4
        comment = "Good answer with specific details." if rating == 5 else "Helpful but could be more detailed."

        feedback_id = store.record_feedback(
            question=question,
            answer=answer,
            sources=sources,
            rating=rating,
            comment=comment,
        )

        stars = "★" * rating + "☆" * (5 - rating)
        print(f"\n  Rating: {stars} ({rating}/5)")
        print(f"  Comment: {comment}")
        print(f"  Feedback ID: {feedback_id}")

    # Show collected data
    print(f"\n{'─' * 60}")
    print(f"\n  📊 FEEDBACK SUMMARY")
    print(f"  Total records: {store.count()}")
    for record in store.get_all():
        stars = "★" * record["rating"] + "☆" * (5 - record["rating"])
        print(f"    [{record['id']}] {stars}  "
              f"Topics: {', '.join(record['topic_tags'])}  "
              f"Dist: {record['avg_distance']:.4f}")

    print("""
    💡 KEY INSIGHT: Structured feedback collection enables data-driven
       improvement. Every answer gets a quality score, creating an audit
       trail — just like how clinical quality metrics track outcomes.
    """)

    return collection, store


# ============================================================
# DEMO 2: Analytics Dashboard
# ============================================================

def demo_analytics_dashboard():
    """Analyze feedback to identify patterns and weak areas"""
    print("\n" + "=" * 70)
    print("DEMO 2: FEEDBACK ANALYTICS DASHBOARD")
    print("=" * 70)
    print("""
    Healthcare parallel: Like a quality dashboard showing infection rates
    by unit, mortality by diagnosis, readmissions by service line.
    We identify WHERE the system is weak, not just IF it is weak.
    """)

    collection = build_knowledge_base()
    store = FeedbackStore()

    # Generate substantial feedback data
    generate_simulated_feedback(collection, store)

    analytics = FeedbackAnalytics(store)

    # 1. Overall summary
    print(f"\n{'─' * 60}")
    print("  📊 OVERALL QUALITY METRICS")
    print(f"{'─' * 60}")
    summary = analytics.overall_summary()
    print(f"  Total feedback:      {summary['total_feedback']}")
    print(f"  Average rating:      {summary['avg_rating']}/5.0")
    print(f"  Satisfaction rate:   {summary['satisfaction_rate']}% (rated 4-5)")
    print(f"  Low quality answers: {summary['low_quality_count']}")
    print(f"\n  Rating distribution:")
    for star, count in sorted(summary['rating_distribution'].items()):
        bar = "█" * (count * 3)
        pct = count / summary['total_feedback'] * 100
        print(f"    {'★' * star}{'☆' * (5-star)}  {count:2d} ({pct:4.1f}%)  {bar}")

    # 2. By specialty
    print(f"\n{'─' * 60}")
    print("  🏥 QUALITY BY SPECIALTY")
    print(f"{'─' * 60}")
    by_spec = analytics.ratings_by_specialty()
    print(f"  {'Specialty':<16} {'Count':>6} {'Avg Rating':>11} {'Low Quality %':>14}")
    print(f"  {'─'*16} {'─'*6} {'─'*11} {'─'*14}")
    for spec, data in by_spec.items():
        indicator = "🔴" if data['avg_rating'] < 3.0 else "🟡" if data['avg_rating'] < 4.0 else "🟢"
        print(f"  {indicator} {spec:<14} {data['count']:>5} {data['avg_rating']:>10.2f} {data['low_quality_pct']:>13.1f}%")

    # 3. By topic
    print(f"\n{'─' * 60}")
    print("  📚 QUALITY BY TOPIC")
    print(f"{'─' * 60}")
    by_topic = analytics.ratings_by_topic()
    print(f"  {'Topic':<20} {'Count':>6} {'Avg Rating':>11}")
    print(f"  {'─'*20} {'─'*6} {'─'*11}")
    for topic, data in by_topic.items():
        indicator = "🔴" if data['avg_rating'] < 3.0 else "🟡" if data['avg_rating'] < 4.0 else "🟢"
        print(f"  {indicator} {topic:<18} {data['count']:>5} {data['avg_rating']:>10.2f}")

    # 4. Distance vs rating
    print(f"\n{'─' * 60}")
    print("  📏 RETRIEVAL DISTANCE vs. ANSWER QUALITY")
    print(f"{'─' * 60}")
    dist_cor = analytics.distance_vs_rating_correlation()
    for bucket, data in dist_cor.items():
        if bucket == "interpretation":
            print(f"\n  💡 {data}")
        else:
            rating_str = f"{data['avg_rating']:.2f}" if data['avg_rating'] else "N/A"
            print(f"  {bucket:<25} Count: {data['count']:>3}  Avg Rating: {rating_str}")

    # 5. Worst answers
    print(f"\n{'─' * 60}")
    print("  ⚠️  LOWEST-RATED ANSWERS (need review)")
    print(f"{'─' * 60}")
    worst = analytics.worst_answers(5)
    for w in worst:
        stars = "★" * w["rating"] + "☆" * (5 - w["rating"])
        print(f"\n  {stars}  Rating: {w['rating']}/5")
        print(f"  Q: {w['question'][:70]}")
        print(f"  Comment: {w.get('comment', 'No comment')}")
        print(f"  Topics: {', '.join(w['topic_tags'])}  |  Avg dist: {w['avg_distance']:.4f}")

    # 6. Content gaps
    print(f"\n{'─' * 60}")
    print("  🕳️  CONTENT GAPS (low rating + high distance)")
    print(f"{'─' * 60}")
    gaps = analytics.identify_content_gaps()
    if gaps:
        for g in gaps:
            print(f"\n  ❌ {g['question'][:70]}")
            print(f"     Rating: {g['rating']}/5  |  Avg distance: {g['avg_distance']:.4f}")
            print(f"     Closest topics: {', '.join(g['closest_topics'])}")
            if g['comment']:
                print(f"     Comment: {g['comment']}")
    else:
        print("  No clear content gaps detected (all low-rated answers had close matches)")

    print("""
    💡 KEY INSIGHT: Breaking down quality by specialty and topic reveals
       WHERE to focus improvement efforts — just like how hospital QI
       teams use dashboards to target specific units or diagnoses.
    """)

    return analytics


# ============================================================
# DEMO 3: AI-Generated Improvement Report
# ============================================================

def demo_improvement_report():
    """Use LLM to analyze feedback and generate actionable recommendations"""
    print("\n" + "=" * 70)
    print("DEMO 3: AI-GENERATED IMPROVEMENT REPORT")
    print("=" * 70)
    print("""
    Healthcare parallel: Like a quarterly quality review where data is
    analyzed and turned into a PDSA (Plan-Do-Study-Act) improvement plan.
    The LLM acts as a quality analyst, reviewing all feedback data and
    producing a structured report with root causes and recommendations.
    """)

    collection = build_knowledge_base()
    store = FeedbackStore()

    # Generate feedback data
    generate_simulated_feedback(collection, store)

    analytics = FeedbackAnalytics(store)

    print(f"\n{'─' * 60}")
    print("  🤖 Generating AI improvement report from feedback data...")
    print(f"{'─' * 60}\n")

    report = analytics.generate_improvement_report()
    for line in report.split('\n'):
        print(f"  {line}")

    # Show what actions could be taken
    print(f"\n{'─' * 60}")
    print("  🔧 AUTOMATED IMPROVEMENT ACTIONS (what a system could do)")
    print(f"{'─' * 60}")

    gaps = analytics.identify_content_gaps()
    worst_topics = analytics.ratings_by_topic()

    print("""
    Based on the feedback analysis, a production system could:
    
    1. AUTO-FLAG for review:
       - Queue lowest-rated answers for human clinician review
       - Flag topics with avg rating < 3.0 for content refresh""")

    low_topics = [t for t, d in worst_topics.items() if d["avg_rating"] < 3.0]
    if low_topics:
        print(f"       → Topics needing attention: {', '.join(low_topics)}")

    print("""
    2. CONTENT EXPANSION:
       - Identify questions with no good source matches
       - Generate document requests for knowledge base curators""")

    if gaps:
        print(f"       → {len(gaps)} content gap(s) detected")
        for g in gaps[:3]:
            print(f"         • Need content for: {g['question'][:55]}...")

    print("""
    3. RETRIEVAL TUNING:
       - Adjust embedding model or chunk size for weak topics
       - Add synonyms/aliases for frequently missed queries

    4. PROMPT REFINEMENT:
       - Improve system prompts for topics with close matches
         but low ratings (generation problem, not retrieval)
    """)

    print("""
    💡 KEY INSIGHT: The feedback loop closes the gap between "deployed"
       and "continuously improving." Healthcare AI can't be static —
       guidelines change, new drugs emerge, practice patterns evolve.
       Feedback loops are how you keep a RAG system clinically relevant.
    """)


# ============================================================
# DEMO 4: Interactive Feedback Session
# ============================================================

def demo_interactive_feedback():
    """Let the user ask questions and rate answers in real-time"""
    print("\n" + "=" * 70)
    print("DEMO 4: INTERACTIVE FEEDBACK SESSION")
    print("=" * 70)
    print("""
    Healthcare parallel: Like real-time provider feedback on clinical
    decision support alerts — did this alert help? Was it relevant?
    
    Ask questions, get answers, then rate them. When done, see analytics
    on your feedback to understand system strengths and weaknesses.
    """)

    collection = build_knowledge_base()
    store = FeedbackStore()

    print("\n  Type a medical question, rate the answer (1-5), and optionally comment.")
    print("  Type 'done' to see analytics on your feedback.\n")

    while True:
        question = input("  ❓ Your question (or 'done'): ").strip()
        if question.lower() == 'done':
            break
        if not question:
            continue

        # Get RAG answer
        sources = multi_query_retrieve(question, collection, n_results=4)
        answer, usage = generate_cited_answer(question, sources)

        print(f"\n  📋 ANSWER:\n")
        for line in answer.split('\n'):
            print(f"    {line}")

        print(f"\n  📎 Sources: {', '.join(s['id'] for s in sources)}")
        print(f"  📏 Avg distance: {sum(s['distance'] for s in sources)/len(sources):.4f}")

        # Get rating
        while True:
            try:
                rating_input = input("\n  Rate this answer (1-5, or 's' to skip): ").strip()
                if rating_input.lower() == 's':
                    print("  Skipped.\n")
                    break
                rating = int(rating_input)
                if 1 <= rating <= 5:
                    comment = input("  Optional comment (Enter to skip): ").strip()
                    store.record_feedback(
                        question=question,
                        answer=answer,
                        sources=sources,
                        rating=rating,
                        comment=comment,
                    )
                    stars = "★" * rating + "☆" * (5 - rating)
                    print(f"  ✅ Recorded: {stars}\n")
                    break
                else:
                    print("  Please enter 1-5.")
            except ValueError:
                print("  Please enter a number 1-5.")

    # Show analytics if any feedback was collected
    if store.count() > 0:
        analytics = FeedbackAnalytics(store)
        summary = analytics.overall_summary()

        print(f"\n{'─' * 60}")
        print("  📊 YOUR SESSION ANALYTICS")
        print(f"{'─' * 60}")
        print(f"  Questions rated:   {summary['total_feedback']}")
        print(f"  Average rating:    {summary['avg_rating']}/5.0")
        print(f"  Satisfaction rate:  {summary['satisfaction_rate']}%")

        by_topic = analytics.ratings_by_topic()
        if by_topic:
            print(f"\n  By topic:")
            for topic, data in by_topic.items():
                indicator = "🔴" if data['avg_rating'] < 3 else "🟡" if data['avg_rating'] < 4 else "🟢"
                print(f"    {indicator} {topic}: {data['avg_rating']:.1f}/5 ({data['count']} queries)")

        gaps = analytics.identify_content_gaps()
        if gaps:
            print(f"\n  ⚠️  Identified {len(gaps)} content gap(s):")
            for g in gaps:
                print(f"    • {g['question'][:60]}...")
    else:
        print("\n  No feedback collected. Try rating some answers next time!")

    print("""
    💡 KEY INSIGHT: Interactive feedback captures the clinician's real-time
       judgment — the gold standard for answer quality. Even a few ratings
       per day, aggregated over time, reveal systematic patterns.
    """)


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 4: USER FEEDBACK LOOP FOR QUALITY IMPROVEMENT")
    print("=" * 70)
    print("""
    This exercise builds a feedback collection and analysis system for
    the Medical RAG — enabling continuous quality improvement through
    structured user ratings, analytics dashboards, and AI-generated
    improvement recommendations.
    
    Healthcare QI principle: "You can't improve what you don't measure."

    Choose a demo:
      1 → Feedback Collection & Rating System
      2 → Analytics Dashboard (simulated data)
      3 → AI-Generated Improvement Report
      4 → Interactive Feedback Session (you rate answers!)
      5 → Run all demos (1-3)
    """)

    choice = input("  Enter choice (1-5): ").strip()

    if choice == "1":
        demo_feedback_collection()
    elif choice == "2":
        demo_analytics_dashboard()
    elif choice == "3":
        demo_improvement_report()
    elif choice == "4":
        demo_interactive_feedback()
    elif choice == "5":
        demo_feedback_collection()
        demo_analytics_dashboard()
        demo_improvement_report()
    else:
        print("  Invalid choice. Please enter 1-5.")


"""
KEY LEARNINGS:
=============

1. FEEDBACK COLLECTION: Structured ratings (1-5) with metadata (specialty,
   topic, distance, comment) create an audit trail for answer quality.

2. ANALYTICS BY DIMENSION: Breaking down ratings by specialty, topic, and
   retrieval distance reveals WHERE problems are — not just IF they exist.

3. CONTENT GAP DETECTION: Low ratings + high retrieval distance = missing
   knowledge base content. Low ratings + low distance = generation/prompt
   problem. This distinction drives the right corrective action.

4. AI-POWERED REPORTING: LLMs can analyze structured feedback data and
   produce actionable improvement reports — automating quality review.

5. HEALTHCARE QI PARALLEL: This mirrors PDSA cycles in healthcare —
   Plan (define metrics), Do (collect feedback), Study (analyze patterns),
   Act (implement improvements). Continuous improvement never stops.

6. INTERACTIVE vs. AUTOMATED: Real-time user ratings capture expert
   judgment; automated analysis (distance correlation, gap detection)
   surfaces patterns humans might miss. Both are essential.
"""

if __name__ == "__main__":
    main()
