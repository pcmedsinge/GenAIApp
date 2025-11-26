"""
Exercise 3: Symptom Checker
Create a symptom similarity system for differential diagnosis support
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


def find_similar_symptom_patterns(patient_symptoms, known_patterns, top_k=5):
    """
    Find most similar symptom patterns from medical knowledge base
    
    Args:
        patient_symptoms: Patient's symptom description
        known_patterns: Dictionary of {condition: symptom_pattern}
        top_k: Number of results to return
    
    Returns:
        List of (condition, pattern, similarity) tuples
    """
    print(f"\n🔍 Analyzing symptoms: '{patient_symptoms}'")
    print("="*70)
    
    # Get embedding for patient's symptoms
    patient_embedding = get_embedding(patient_symptoms)
    
    # Compare with known patterns
    results = []
    for condition, pattern in known_patterns.items():
        pattern_embedding = get_embedding(pattern)
        similarity = cosine_similarity(patient_embedding, pattern_embedding)
        results.append((condition, pattern, similarity))
    
    # Sort by similarity
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results[:top_k]


def demo_basic_symptom_matching():
    """
    Demo 1: Basic symptom pattern matching
    """
    print("\n" + "="*70)
    print("DEMO 1: Basic Symptom Pattern Matching")
    print("="*70)
    
    # Known symptom patterns for common conditions
    symptom_patterns = {
        "Acute Myocardial Infarction": "crushing chest pain radiating to left arm and jaw, shortness of breath, nausea, diaphoresis, sense of impending doom",
        "Pulmonary Embolism": "sudden onset shortness of breath, chest pain worse with breathing, rapid heart rate, cough with blood, leg swelling",
        "Pneumonia": "fever, productive cough with colored sputum, shortness of breath, chest pain with breathing, fatigue",
        "Congestive Heart Failure": "shortness of breath worse when lying down, leg swelling, fatigue, rapid weight gain, orthopnea",
        "GERD": "burning chest pain after meals, worse when lying down, sour taste in mouth, difficulty swallowing",
        "Panic Attack": "sudden intense fear, chest tightness, rapid heartbeat, shortness of breath, dizziness, feeling of losing control",
        "Costochondritis": "sharp chest pain worse with movement or pressing on chest, pain with deep breathing",
        "Angina": "pressure-like chest pain with exertion, relieved by rest, jaw or arm discomfort",
    }
    
    # Patient presentations
    test_cases = [
        "I have really bad chest pain that feels like an elephant sitting on my chest, and my left arm hurts",
        "Can't catch my breath, just came back from long flight, calf is swollen and painful",
        "Chest feels tight whenever I climb stairs, goes away when I sit down",
        "Bad heartburn after eating, especially at night when I lie down",
    ]
    
    for patient_symptoms in test_cases:
        results = find_similar_symptom_patterns(patient_symptoms, symptom_patterns, top_k=3)
        
        print("\nTop 3 differential diagnoses:")
        print("-"*70)
        for i, (condition, pattern, score) in enumerate(results, 1):
            urgency = "🔴 URGENT" if score > 0.75 and "Myocardial" in condition or "Embolism" in condition else "🟡 EVALUATE" if score > 0.65 else "🟢 ROUTINE"
            print(f"\n{i}. {urgency} - Similarity: {score:.3f}")
            print(f"   Condition: {condition}")
            print(f"   Pattern: {pattern[:100]}...")


def demo_triage_system():
    """
    Demo 2: Emergency department triage support
    """
    print("\n\n" + "="*70)
    print("DEMO 2: Emergency Department Triage Support")
    print("="*70)
    
    # Critical symptom patterns requiring immediate attention
    critical_patterns = {
        "STEMI (Heart Attack)": "severe crushing chest pain, radiation to arm or jaw, diaphoresis, nausea, shortness of breath",
        "Stroke": "sudden facial drooping, arm weakness, speech difficulty, severe headache, confusion, vision changes",
        "Sepsis": "fever with confusion, rapid breathing, rapid heart rate, low blood pressure, decreased urine output",
        "Pulmonary Embolism": "sudden severe shortness of breath, chest pain, coughing blood, leg swelling, recent surgery",
        "Ruptured AAA": "sudden severe abdominal or back pain, pulsating mass in abdomen, low blood pressure, syncope",
        "Acute Abdomen": "severe sudden abdominal pain, rigid abdomen, rebound tenderness, nausea, vomiting",
        "Anaphylaxis": "sudden difficulty breathing, swelling of face or throat, hives, low blood pressure, after exposure to allergen",
        "Tension Pneumothorax": "severe shortness of breath, chest pain, decreased breath sounds, tracheal deviation, hypotension",
    }
    
    # Incoming ED patients
    ed_patients = [
        {
            "id": "PT001",
            "complaint": "I have the worst headache of my life, started suddenly, and I feel confused",
            "age": 55,
            "vitals": "BP 180/110, HR 95"
        },
        {
            "id": "PT002", 
            "complaint": "Ate shrimp 20 minutes ago, now my throat is swelling and hard to breathe",
            "age": 28,
            "vitals": "BP 85/50, HR 120, RR 28"
        },
        {
            "id": "PT003",
            "complaint": "Been having back pain for 3 days, getting worse, feels like a pulse",
            "age": 72,
            "vitals": "BP 90/60, HR 105"
        },
    ]
    
    print("\n🚨 Incoming ED Patients - Triage Analysis")
    
    for patient in ed_patients:
        print("\n" + "="*70)
        print(f"Patient: {patient['id']} | Age: {patient['age']} | Vitals: {patient['vitals']}")
        print(f"Chief Complaint: {patient['complaint']}")
        
        results = find_similar_symptom_patterns(patient['complaint'], critical_patterns, top_k=2)
        
        print("\nTop 2 Critical Conditions to Rule Out:")
        for i, (condition, pattern, score) in enumerate(results, 1):
            if score > 0.70:
                triage_level = "ESI 1 - IMMEDIATE"
                color = "🔴"
            elif score > 0.60:
                triage_level = "ESI 2 - EMERGENT"
                color = "🟠"
            else:
                triage_level = "ESI 3 - URGENT"
                color = "🟡"
            
            print(f"\n  {i}. {color} {triage_level}")
            print(f"     Condition: {condition}")
            print(f"     Match: {score:.3f}")


def demo_symptom_clustering():
    """
    Demo 3: Group patients by similar symptom patterns
    """
    print("\n\n" + "="*70)
    print("DEMO 3: Symptom Clustering for Pattern Recognition")
    print("="*70)
    
    # Multiple patients with various presentations
    patients = [
        "Patient A: Fever, cough with yellow sputum, chest pain when breathing",
        "Patient B: High fever, productive cough, shortness of breath, chills",
        "Patient C: Crushing chest pain, left arm numbness, sweating profusely",
        "Patient D: Severe chest pressure, jaw pain, nausea, shortness of breath",
        "Patient E: Fever, severe cough, greenish mucus, fatigue, chest discomfort",
        "Patient F: Sudden shortness of breath, coughing blood, calf pain and swelling",
    ]
    
    print("\n👥 Patient Presentations:")
    for i, patient in enumerate(patients, 1):
        print(f"{i}. {patient}")
    
    # Common condition patterns
    condition_patterns = {
        "Pneumonia": "fever, productive cough, colored sputum, chest pain with breathing, shortness of breath",
        "Acute MI": "crushing chest pain, arm or jaw pain, nausea, diaphoresis, pressure sensation",
        "Pulmonary Embolism": "sudden shortness of breath, chest pain, hemoptysis, leg swelling or pain",
    }
    
    print("\n\n📊 Clustering Results:")
    print("="*70)
    
    # Group patients by their most likely condition
    clusters = {condition: [] for condition in condition_patterns}
    
    for patient in patients:
        results = find_similar_symptom_patterns(patient, condition_patterns, top_k=1)
        if results:
            best_match = results[0][0]
            clusters[best_match].append((patient, results[0][2]))
    
    for condition, patient_list in clusters.items():
        if patient_list:
            print(f"\n🏥 {condition} ({len(patient_list)} patients):")
            print("-"*70)
            for patient, score in patient_list:
                print(f"  • [{score:.3f}] {patient}")


def demo_differential_diagnosis():
    """
    Demo 4: Generate differential diagnosis list
    """
    print("\n\n" + "="*70)
    print("DEMO 4: Differential Diagnosis Generator")
    print("="*70)
    
    # Comprehensive symptom pattern database
    conditions = {
        # Cardiovascular
        "Acute MI": "crushing substernal chest pain, left arm radiation, diaphoresis, nausea, dyspnea",
        "Unstable Angina": "chest pressure with exertion, relieved by rest, crescendo pattern",
        "Pericarditis": "sharp chest pain relieved by sitting forward, worse lying down, recent viral illness",
        "Aortic Dissection": "sudden tearing chest or back pain, blood pressure difference in arms",
        
        # Pulmonary
        "Pulmonary Embolism": "sudden dyspnea, pleuritic chest pain, hemoptysis, DVT risk factors",
        "Pneumonia": "fever, productive cough, pleuritic pain, dyspnea, crackles on exam",
        "Pneumothorax": "sudden sharp chest pain, dyspnea, decreased breath sounds, trauma or tall thin male",
        "COPD Exacerbation": "increased dyspnea, cough, sputum production, wheezing, smoking history",
        
        # GI
        "GERD": "burning retrosternal pain after meals, worse supine, sour taste",
        "Esophageal Spasm": "chest pain with swallowing, intermittent, relieved by nitroglycerin",
        "Peptic Ulcer": "epigastric pain, relation to meals, relieved by antacids",
        
        # Musculoskeletal
        "Costochondritis": "chest wall tenderness, pain with movement or palpation",
        "Muscle Strain": "chest pain with specific movements, history of exertion or trauma",
        
        # Psychiatric
        "Panic Attack": "sudden anxiety, chest tightness, palpitations, hyperventilation, sense of doom",
    }
    
    # Complex patient case
    patient_case = """
    64-year-old male with history of hypertension and diabetes.
    Chief Complaint: "Having chest discomfort for past hour"
    
    HPI: Patient describes pressure-like sensation in center of chest
    that started while shoveling snow. Pain radiates to left shoulder.
    Associated with shortness of breath and nausea. No relief with rest.
    Diaphoretic. Denies fever, cough, or recent illness.
    
    Risk Factors: Smoker (30 pack-years), family history of early CAD
    """
    
    print("\n📋 Patient Case:")
    print("-"*70)
    print(patient_case)
    
    results = find_similar_symptom_patterns(patient_case, conditions, top_k=8)
    
    print("\n\n📊 Differential Diagnosis (ranked by probability):")
    print("="*70)
    
    for i, (condition, pattern, score) in enumerate(results, 1):
        # Assign likelihood based on similarity score
        if score > 0.75:
            likelihood = "VERY LIKELY"
            color = "🔴"
        elif score > 0.65:
            likelihood = "LIKELY"
            color = "🟠"
        elif score > 0.55:
            likelihood = "POSSIBLE"
            color = "🟡"
        else:
            likelihood = "UNLIKELY"
            color = "🟢"
        
        print(f"\n{i}. {color} {likelihood} - Match: {score:.3f}")
        print(f"   Diagnosis: {condition}")
        print(f"   Pattern: {pattern}")


def interactive_symptom_checker():
    """
    Interactive symptom checker
    """
    print("\n\n" + "="*70)
    print("INTERACTIVE SYMPTOM CHECKER")
    print("="*70)
    
    conditions = {
        "Acute MI": "crushing chest pain, left arm or jaw radiation, nausea, diaphoresis, dyspnea",
        "Pulmonary Embolism": "sudden shortness of breath, chest pain, hemoptysis, leg swelling",
        "Pneumonia": "fever, productive cough, colored sputum, chest pain with breathing",
        "CHF": "shortness of breath worse lying down, leg swelling, fatigue, weight gain",
        "GERD": "burning chest pain after meals, worse lying down, sour taste",
        "Panic Attack": "sudden fear, chest tightness, rapid heartbeat, hyperventilation",
        "Costochondritis": "chest wall pain worse with movement or palpation",
        "Pneumothorax": "sudden sharp chest pain, dyspnea, decreased breath sounds",
        "Pericarditis": "sharp chest pain relieved by leaning forward",
        "Aortic Dissection": "tearing chest or back pain, sudden onset, severe",
        "Stroke": "facial drooping, arm weakness, speech difficulty, sudden headache",
        "Sepsis": "fever with confusion, rapid breathing, low blood pressure",
        "Appendicitis": "right lower abdominal pain, nausea, fever, rebound tenderness",
        "Meningitis": "severe headache, neck stiffness, fever, photophobia",
        "UTI": "burning urination, frequent urination, lower abdominal pain",
    }
    
    print(f"\n💊 Knowledge base contains {len(conditions)} conditions")
    print("\n⚠️  DISCLAIMER: This is for educational purposes only.")
    print("   Always seek professional medical evaluation for symptoms.")
    
    while True:
        print("\n" + "="*70)
        symptoms = input("\n🩺 Describe your symptoms (or 'quit' to exit): ").strip()
        
        if symptoms.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye! Remember to consult healthcare provider for actual symptoms.")
            break
        
        if not symptoms:
            continue
        
        try:
            results = find_similar_symptom_patterns(symptoms, conditions, top_k=5)
            
            print("\n📊 Top 5 Possible Conditions:")
            print("-"*70)
            
            for i, (condition, pattern, score) in enumerate(results, 1):
                if score > 0.70:
                    urgency = "🔴 SEEK IMMEDIATE CARE"
                elif score > 0.60:
                    urgency = "🟠 SEE DOCTOR SOON"
                else:
                    urgency = "🟡 MONITOR"
                
                print(f"\n{i}. {condition}")
                print(f"   Match: {score:.3f} - {urgency}")
                print(f"   Typical pattern: {pattern}")
            
            print("\n" + "-"*70)
            print("⚠️  This is NOT a medical diagnosis. Consult a healthcare provider.")
        
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """
    Run all demonstrations
    """
    print("\n🩺 Symptom Checker - Differential Diagnosis Support")
    print("="*70)
    print("Using embeddings to match symptom patterns")
    
    print("\n\nChoose a demo:")
    print("1. Basic symptom pattern matching")
    print("2. Emergency department triage support")
    print("3. Symptom clustering (group similar patients)")
    print("4. Differential diagnosis generator")
    print("5. Interactive symptom checker")
    print("6. Run all demos")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        demo_basic_symptom_matching()
    elif choice == "2":
        demo_triage_system()
    elif choice == "3":
        demo_symptom_clustering()
    elif choice == "4":
        demo_differential_diagnosis()
    elif choice == "5":
        interactive_symptom_checker()
    elif choice == "6":
        demo_basic_symptom_matching()
        demo_triage_system()
        demo_symptom_clustering()
        demo_differential_diagnosis()
        print("\n\nWould you like to try interactive mode? (y/n): ", end="")
        if input().strip().lower() == 'y':
            interactive_symptom_checker()
    else:
        print("❌ Invalid choice")
    
    print("\n\n" + "="*70)
    print("KEY LEARNINGS")
    print("="*70)
    print("""
✅ Semantic search matches symptoms even with different wording
✅ "Can't breathe" matches "shortness of breath" patterns
✅ "Elephant on chest" matches MI patterns automatically
✅ Similarity scores help rank differential diagnoses
✅ Can identify critical conditions requiring immediate attention

CLINICAL APPLICATIONS:
- Triage support in ED (prioritize critical conditions)
- Clinical decision support systems
- Pattern recognition for diagnostic assistance
- Teaching tool for medical students
- Quality assurance (missed diagnosis detection)

IMPORTANT LIMITATIONS:
⚠️  NOT a replacement for clinical judgment
⚠️  Requires validation by medical professionals
⚠️  Should be used as decision support, not sole diagnostic tool
⚠️  Atypical presentations may not match well
⚠️  Always consider full clinical context

PRODUCTION CONSIDERATIONS:
- Use comprehensive symptom pattern database
- Include demographic and risk factor weighting
- Integrate with clinical guidelines
- Log all recommendations for review
- Regular validation against outcomes
- Clear disclaimers about limitations
""")


if __name__ == "__main__":
    main()
