"""
Exercise 2: Medical Abbreviation Expander
Search and expand medical abbreviations using semantic search
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding for a text"""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / magnitude


def search_abbreviations(query, abbreviations_db, top_k=5):
    """
    Search for medical abbreviations and their meanings
    
    Args:
        query: Search query (plain English or abbreviation)
        abbreviations_db: Dictionary of {abbreviation: full_meaning}
        top_k: Number of results to return
    
    Returns:
        List of (abbrev, meaning, similarity_score) tuples
    """
    print(f"\n🔍 Searching for: '{query}'")
    print("="*70)
    
    # Get embedding for query
    query_embedding = get_embedding(query)
    
    # Search through all abbreviations
    results = []
    for abbrev, meaning in abbreviations_db.items():
        # Create searchable text combining abbreviation and meaning
        search_text = f"{abbrev} - {meaning}"
        
        text_embedding = get_embedding(search_text)
        similarity = cosine_similarity(query_embedding, text_embedding)
        results.append((abbrev, meaning, similarity))
    
    # Sort by similarity
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results[:top_k]


def demo_basic_search():
    """
    Demo 1: Basic abbreviation search
    """
    print("\n" + "="*70)
    print("DEMO 1: Basic Medical Abbreviation Search")
    print("="*70)
    
    abbreviations_db = {
        "ECG": "Electrocardiogram - continuous cardiac monitoring",
        "EKG": "Electrocardiogram - heart electrical activity recording",
        "BP": "Blood Pressure - arterial pressure measurement",
        "HR": "Heart Rate - number of heartbeats per minute",
        "RR": "Respiratory Rate - breathing frequency per minute",
        "SpO2": "Oxygen Saturation - percentage of oxygen in blood",
        "CBC": "Complete Blood Count - full blood cell analysis",
        "CMP": "Comprehensive Metabolic Panel - blood chemistry test",
        "BMP": "Basic Metabolic Panel - essential electrolytes and kidney function",
        "ABG": "Arterial Blood Gas - blood oxygen and acid-base balance",
        "CT": "Computed Tomography - detailed cross-sectional imaging",
        "MRI": "Magnetic Resonance Imaging - detailed soft tissue imaging",
        "CXR": "Chest X-Ray - radiographic imaging of chest",
        "NPO": "Nothing by mouth - fasting before procedure",
        "PRN": "As needed - medication taken when necessary",
        "BID": "Twice a day - medication dosing frequency",
        "TID": "Three times a day - medication dosing frequency",
        "QID": "Four times a day - medication dosing frequency",
    }
    
    queries = [
        "heart rate monitoring",
        "blood oxygen level",
        "take medicine twice daily",
        "chest imaging",
    ]
    
    for query in queries:
        results = search_abbreviations(query, abbreviations_db, top_k=3)
        
        print(f"\nTop 3 matches:")
        for i, (abbrev, meaning, score) in enumerate(results, 1):
            print(f"  {i}. [{score:.3f}] {abbrev} = {meaning}")


def demo_expand_clinical_note():
    """
    Demo 2: Help understand a clinical note with abbreviations
    """
    print("\n\n" + "="*70)
    print("DEMO 2: Expand Abbreviations in Clinical Note")
    print("="*70)
    
    abbreviations_db = {
        "SOB": "Shortness of Breath - difficulty breathing",
        "CP": "Chest Pain - discomfort in chest area",
        "N/V": "Nausea and Vomiting - feeling sick and throwing up",
        "HA": "Headache - pain in head region",
        "DOE": "Dyspnea on Exertion - shortness of breath during activity",
        "PND": "Paroxysmal Nocturnal Dyspnea - sudden nighttime breathlessness",
        "LE": "Lower Extremities - legs and feet",
        "JVD": "Jugular Venous Distension - swollen neck veins",
        "PMH": "Past Medical History - previous health conditions",
        "FH": "Family History - diseases in family members",
        "NKDA": "No Known Drug Allergies - no medication allergies",
        "HTN": "Hypertension - high blood pressure",
        "DM": "Diabetes Mellitus - sugar metabolism disorder",
        "CAD": "Coronary Artery Disease - heart blood vessel disease",
        "CHF": "Congestive Heart Failure - heart pumping weakness",
        "COPD": "Chronic Obstructive Pulmonary Disease - lung airflow obstruction",
        "A&O": "Alert and Oriented - mentally aware and cognizant",
        "NAD": "No Acute Distress - patient appears comfortable",
        "RRR": "Regular Rate and Rhythm - normal heart sounds",
        "CTAB": "Clear to Auscultation Bilaterally - normal lung sounds",
    }
    
    # Sample clinical note with abbreviations
    clinical_note = """
    Patient: 65yo male
    CC: SOB and CP
    HPI: Pt c/o DOE x 3 days, also reports PND and LE edema
    PMH: HTN, DM, CAD
    Meds: NKDA
    PE: A&O x 3, NAD, RRR, CTAB, +JVD, +2 LE edema
    """
    
    print("\n📋 Clinical Note:")
    print("-"*70)
    print(clinical_note)
    
    # Extract potential abbreviations (simple approach - look for uppercase words)
    import re
    potential_abbrevs = re.findall(r'\b[A-Z]{2,}\b', clinical_note)
    unique_abbrevs = list(set(potential_abbrevs))
    
    print("\n\n📖 Abbreviation Meanings:")
    print("-"*70)
    
    for abbrev in sorted(unique_abbrevs):
        if abbrev in abbreviations_db:
            print(f"\n✓ {abbrev:8} = {abbreviations_db[abbrev]}")
        else:
            # Try semantic search
            results = search_abbreviations(abbrev, abbreviations_db, top_k=1)
            if results and results[0][2] > 0.8:
                best_abbrev, meaning, score = results[0]
                print(f"\n≈ {abbrev:8} ≈ {best_abbrev} = {meaning} (similarity: {score:.3f})")
            else:
                print(f"\n? {abbrev:8} = Not found in database")


def demo_context_aware_search():
    """
    Demo 3: Context-aware abbreviation expansion
    Many abbreviations have multiple meanings
    """
    print("\n\n" + "="*70)
    print("DEMO 3: Context-Aware Abbreviation Expansion")
    print("="*70)
    
    # Some abbreviations have multiple meanings depending on context
    abbreviations_db = {
        # MS can mean:
        "MS_neuro": "Multiple Sclerosis - neurological disease affecting brain and spinal cord",
        "MS_cardiac": "Mitral Stenosis - narrowing of heart's mitral valve",
        "MS_mental": "Mental Status - cognitive and psychological state",
        
        # RA can mean:
        "RA_rheum": "Rheumatoid Arthritis - autoimmune joint disease",
        "RA_cardiac": "Right Atrium - upper right chamber of heart",
        "RA_air": "Room Air - breathing without supplemental oxygen",
        
        # PT can mean:
        "PT_therapy": "Physical Therapy - rehabilitation treatment",
        "PT_lab": "Prothrombin Time - blood clotting test",
        "PT_person": "Patient - person receiving medical care",
        
        # BS can mean:
        "BS_glucose": "Blood Sugar - glucose level in blood",
        "BS_sounds": "Bowel Sounds - intestinal activity sounds",
        "BS_breath": "Breath Sounds - lung sounds during breathing",
    }
    
    # Different contexts
    contexts = [
        ("neurological examination with MS", "MS"),
        ("cardiac echo showing MS", "MS"),
        ("joint pain and RA diagnosis", "RA"),
        ("patient on RA, no supplemental oxygen", "RA"),
        ("PT evaluation for mobility", "PT"),
        ("PT prolonged, check INR", "PT"),
    ]
    
    for context, abbrev in contexts:
        print(f"\n📄 Context: '{context}'")
        results = search_abbreviations(context, abbreviations_db, top_k=1)
        
        if results:
            best_abbrev, meaning, score = results[0]
            print(f"   Best match: {best_abbrev} = {meaning}")
            print(f"   Confidence: {score:.3f}")


def demo_specialty_abbreviations():
    """
    Demo 4: Search by medical specialty
    """
    print("\n\n" + "="*70)
    print("DEMO 4: Search by Medical Specialty")
    print("="*70)
    
    abbreviations_db = {
        # Cardiology
        "CAD": "Coronary Artery Disease - heart vessel narrowing",
        "MI": "Myocardial Infarction - heart attack",
        "CHF": "Congestive Heart Failure - heart pump weakness",
        "AF": "Atrial Fibrillation - irregular heart rhythm",
        "EF": "Ejection Fraction - heart pumping efficiency percentage",
        
        # Pulmonology
        "COPD": "Chronic Obstructive Pulmonary Disease - airflow limitation",
        "PE": "Pulmonary Embolism - blood clot in lung",
        "PFT": "Pulmonary Function Test - lung capacity measurement",
        "ARDS": "Acute Respiratory Distress Syndrome - severe lung inflammation",
        
        # Neurology
        "CVA": "Cerebrovascular Accident - stroke",
        "TIA": "Transient Ischemic Attack - mini-stroke",
        "MS": "Multiple Sclerosis - neurological disease",
        "ALS": "Amyotrophic Lateral Sclerosis - motor neuron disease",
        
        # Gastroenterology
        "GERD": "Gastroesophageal Reflux Disease - acid reflux",
        "IBD": "Inflammatory Bowel Disease - intestinal inflammation",
        "PUD": "Peptic Ulcer Disease - stomach or duodenal ulcer",
        "LFT": "Liver Function Test - liver enzyme measurement",
        
        # Endocrinology
        "DM": "Diabetes Mellitus - blood sugar disorder",
        "DKA": "Diabetic Ketoacidosis - diabetes complication",
        "TSH": "Thyroid Stimulating Hormone - thyroid function test",
        "HbA1c": "Hemoglobin A1c - average blood sugar over 3 months",
    }
    
    specialties = {
        "Cardiology": "heart cardiovascular cardiac myocardial",
        "Pulmonology": "lung respiratory breathing pulmonary",
        "Neurology": "brain neurological stroke seizure",
        "Gastroenterology": "stomach intestine bowel liver digestive",
        "Endocrinology": "diabetes thyroid hormone metabolic",
    }
    
    for specialty, keywords in specialties.items():
        print(f"\n🏥 {specialty}")
        print("-"*70)
        results = search_abbreviations(keywords, abbreviations_db, top_k=3)
        for i, (abbrev, meaning, score) in enumerate(results, 1):
            print(f"  {i}. {abbrev:10} = {meaning}")


def interactive_mode():
    """
    Interactive abbreviation lookup
    """
    print("\n\n" + "="*70)
    print("INTERACTIVE MEDICAL ABBREVIATION EXPANDER")
    print("="*70)
    
    # Comprehensive abbreviation database
    abbreviations_db = {
        # Vital signs
        "BP": "Blood Pressure - arterial pressure measurement",
        "HR": "Heart Rate - heartbeats per minute",
        "RR": "Respiratory Rate - breaths per minute",
        "Temp": "Temperature - body temperature",
        "SpO2": "Oxygen Saturation - blood oxygen percentage",
        
        # Cardiac
        "ECG": "Electrocardiogram - heart electrical activity",
        "EKG": "Electrocardiogram - heart rhythm tracing",
        "MI": "Myocardial Infarction - heart attack",
        "CHF": "Congestive Heart Failure - heart pump failure",
        "CAD": "Coronary Artery Disease - heart vessel disease",
        "AF": "Atrial Fibrillation - irregular heartbeat",
        
        # Respiratory
        "SOB": "Shortness of Breath - breathing difficulty",
        "DOE": "Dyspnea on Exertion - breathless with activity",
        "COPD": "Chronic Obstructive Pulmonary Disease - lung disease",
        "PE": "Pulmonary Embolism - lung blood clot",
        
        # Labs
        "CBC": "Complete Blood Count - full blood cell analysis",
        "CMP": "Comprehensive Metabolic Panel - chemistry panel",
        "BMP": "Basic Metabolic Panel - electrolytes and kidney",
        "LFT": "Liver Function Test - liver enzyme test",
        "PT": "Prothrombin Time - blood clotting test",
        "INR": "International Normalized Ratio - warfarin monitoring",
        
        # Imaging
        "CT": "Computed Tomography - CAT scan",
        "MRI": "Magnetic Resonance Imaging - detailed scan",
        "CXR": "Chest X-Ray - chest radiograph",
        "US": "Ultrasound - sound wave imaging",
        
        # Medications
        "NPO": "Nothing by mouth - no food or drink",
        "PRN": "As needed - take when necessary",
        "BID": "Twice a day - two times daily",
        "TID": "Three times a day - three times daily",
        "QID": "Four times a day - four times daily",
        "PO": "By mouth - oral administration",
        "IV": "Intravenous - into vein",
        
        # Common diagnoses
        "HTN": "Hypertension - high blood pressure",
        "DM": "Diabetes Mellitus - diabetes",
        "GERD": "Gastroesophageal Reflux Disease - acid reflux",
        "UTI": "Urinary Tract Infection - bladder infection",
        "DVT": "Deep Vein Thrombosis - blood clot in leg",
    }
    
    print(f"\n📚 Database contains {len(abbreviations_db)} abbreviations")
    
    while True:
        print("\n" + "-"*70)
        query = input("\n🔍 Enter medical term or abbreviation (or 'quit'): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            continue
        
        try:
            results = search_abbreviations(query, abbreviations_db, top_k=5)
            
            print(f"\nTop 5 matches:")
            for i, (abbrev, meaning, score) in enumerate(results, 1):
                confidence = "✓" if score > 0.8 else "≈" if score > 0.6 else "?"
                print(f"  {i}. {confidence} [{score:.3f}] {abbrev:10} = {meaning}")
        
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """
    Run all demonstrations
    """
    print("\n📖 Medical Abbreviation Expander")
    print("="*70)
    print("Using semantic search to find and explain medical abbreviations")
    
    print("\n\nChoose a demo:")
    print("1. Basic abbreviation search")
    print("2. Expand abbreviations in clinical note")
    print("3. Context-aware expansion (MS, RA, PT, etc.)")
    print("4. Search by medical specialty")
    print("5. Interactive lookup mode")
    print("6. Run all demos")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        demo_basic_search()
    elif choice == "2":
        demo_expand_clinical_note()
    elif choice == "3":
        demo_context_aware_search()
    elif choice == "4":
        demo_specialty_abbreviations()
    elif choice == "5":
        interactive_mode()
    elif choice == "6":
        demo_basic_search()
        demo_expand_clinical_note()
        demo_context_aware_search()
        demo_specialty_abbreviations()
        print("\n\nWould you like to try interactive mode? (y/n): ", end="")
        if input().strip().lower() == 'y':
            interactive_mode()
    else:
        print("❌ Invalid choice")
    
    print("\n\n" + "="*70)
    print("KEY LEARNINGS")
    print("="*70)
    print("""
✅ Semantic search works with plain English OR abbreviations
✅ "heart rate monitoring" automatically finds ECG, EKG, HR
✅ Context helps disambiguate multi-meaning abbreviations
✅ Can search by specialty to find relevant abbreviations
✅ Much better than keyword matching (ECG matches even if you say "heart monitoring")

PRODUCTION USE CASES:
- EHR systems to help staff understand notes
- Medical student learning tools
- Clinical documentation assistance
- Patient portal translations (medical → plain English)
- Voice dictation error correction

TIPS:
- Build comprehensive abbreviation database for your specialty
- Include context in descriptions for better matching
- Cache embeddings for faster lookups
- Consider multiple meanings for ambiguous abbreviations
""")


if __name__ == "__main__":
    main()
