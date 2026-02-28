"""
Exercise 3 — Medical Spelling Corrector
=========================================
Post-process transcriptions to fix medical terminology errors.
Uses embeddings for similar-term matching and GPT for contextual
correction.

Objectives
----------
* Build a medical terminology dictionary with embeddings
* Find closest matching medical terms for misspelled words
* Use GPT for context-aware spelling correction
* Compare rule-based vs. embedding-based vs. LLM approaches
"""

import json
import re
import math
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Medical terminology dictionary
# ---------------------------------------------------------------------------

MEDICAL_DICTIONARY = [
    # Medications
    "Acetaminophen", "Amoxicillin", "Atorvastatin", "Amlodipine",
    "Aspirin", "Azithromycin", "Carvedilol", "Clopidogrel",
    "Dexamethasone", "Diazepam", "Enalapril", "Furosemide",
    "Gabapentin", "Glipizide", "Hydrochlorothiazide", "Ibuprofen",
    "Insulin Glargine", "Levothyroxine", "Lisinopril", "Losartan",
    "Metformin", "Metoprolol", "Nitroglycerin", "Omeprazole",
    "Pantoprazole", "Prednisone", "Sertraline", "Simvastatin",
    "Sumatriptan", "Warfarin",
    # Conditions
    "hypertension", "hypotension", "tachycardia", "bradycardia",
    "atrial fibrillation", "myocardial infarction", "angina",
    "diabetes mellitus", "hypothyroidism", "hyperthyroidism",
    "pneumonia", "bronchitis", "asthma", "COPD",
    "neuropathy", "nephropathy", "retinopathy",
    "osteoarthritis", "rheumatoid arthritis",
    "migraine", "seizure", "stroke", "aneurysm",
    # Procedures & tests
    "echocardiogram", "electrocardiogram", "spirometry",
    "colonoscopy", "endoscopy", "bronchoscopy",
    "MRI", "CT scan", "X-ray", "ultrasound",
    "troponin", "hemoglobin A1c", "creatinine",
    "complete blood count", "basic metabolic panel",
    "comprehensive metabolic panel", "urinalysis",
    # Anatomy
    "thorax", "abdomen", "bilateral", "peripheral",
    "sublingual", "intramuscular", "intravenous",
    "subcutaneous", "tympanic", "auscultation",
]


# ---------------------------------------------------------------------------
# Embedding-based spell checker
# ---------------------------------------------------------------------------

class MedicalSpellChecker:
    """Spell checker using OpenAI embeddings for medical term matching."""

    def __init__(self, dictionary: list[str]):
        self.dictionary = dictionary
        self.embeddings: dict[str, list[float]] = {}
        self._build_index()

    def _build_index(self):
        """Generate embeddings for all dictionary terms."""
        print(f"  Building embedding index for {len(self.dictionary)} terms …")
        # Batch embed for efficiency
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=self.dictionary,
        )
        for i, item in enumerate(response.data):
            self.embeddings[self.dictionary[i]] = item.embedding
        print(f"  Index built. Dimensions: {len(response.data[0].embedding)}")

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def find_closest(self, term: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Find the closest dictionary terms for a given word.

        Parameters
        ----------
        term : str
            The potentially misspelled term.
        top_k : int
            Number of top matches to return.

        Returns
        -------
        list of (term, similarity_score) tuples, sorted descending.
        """
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[term],
        )
        query_emb = response.data[0].embedding

        scores = []
        for dict_term, emb in self.embeddings.items():
            sim = self.cosine_similarity(query_emb, emb)
            scores.append((dict_term, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def suggest_correction(self, term: str, threshold: float = 0.80) -> str | None:
        """Suggest a correction if the best match exceeds the threshold.

        Returns None if no confident match is found.
        """
        matches = self.find_closest(term, top_k=1)
        if matches and matches[0][1] >= threshold:
            return matches[0][0]
        return None


# ---------------------------------------------------------------------------
# Rule-based corrections (fast, no API calls)
# ---------------------------------------------------------------------------

COMMON_MISSPELLINGS = {
    "lissinopril": "Lisinopril",
    "lisinipril": "Lisinopril",
    "metforman": "Metformin",
    "metformen": "Metformin",
    "atorvastain": "Atorvastatin",
    "atorvastin": "Atorvastatin",
    "amlodapine": "Amlodipine",
    "amlodipin": "Amlodipine",
    "summatriptan": "Sumatriptan",
    "sumatriptin": "Sumatriptan",
    "gabapenten": "Gabapentin",
    "gabapenten": "Gabapentin",
    "glipzide": "Glipizide",
    "nitroglycerine": "Nitroglycerin",
    "hypertention": "hypertension",
    "diabeties": "diabetes",
    "neropathy": "neuropathy",
    "nuropathy": "neuropathy",
    "echocardigram": "echocardiogram",
    "electrocadiogram": "electrocardiogram",
    "troponon": "troponin",
    "hemogloben": "hemoglobin",
    "creatanine": "creatinine",
}


def rule_based_correct(text: str) -> tuple[str, list[dict]]:
    """Apply rule-based corrections using the misspelling dictionary.

    Returns
    -------
    (corrected_text, list of corrections made)
    """
    corrections = []
    result = text
    for wrong, right in COMMON_MISSPELLINGS.items():
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        if pattern.search(result):
            result = pattern.sub(right, result)
            corrections.append({"original": wrong, "corrected": right, "method": "rule"})
    return result, corrections


# ---------------------------------------------------------------------------
# GPT-based contextual correction
# ---------------------------------------------------------------------------

CORRECTION_PROMPT = """\
You are a medical transcription editor.  Fix any misspelled medical terms
in the following text.  This includes:
- Drug names (capitalize properly)
- Medical conditions and diagnoses
- Anatomical terms
- Lab tests and procedures

Rules:
- Only fix medical term spelling.  Do not change grammar, style, or meaning.
- If a word is correct, leave it unchanged.
- Return the corrected text AND a JSON list of changes.

Format your response as:
CORRECTED TEXT:
<the full corrected text>

CHANGES:
[{"original": "wrong", "corrected": "right"}]
"""


def gpt_based_correct(text: str) -> tuple[str, list[dict]]:
    """Use GPT to contextually correct medical spelling errors.

    Returns
    -------
    (corrected_text, list of corrections)
    """
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": CORRECTION_PROMPT},
            {"role": "user", "content": text},
        ],
        max_tokens=1000,
        temperature=0,
    )

    output = response.choices[0].message.content.strip()

    # Parse response
    corrected_text = text
    changes = []

    if "CORRECTED TEXT:" in output:
        parts = output.split("CHANGES:")
        corrected_text = parts[0].replace("CORRECTED TEXT:", "").strip()
        if len(parts) > 1:
            try:
                json_str = parts[1].strip()
                if json_str.startswith("```"):
                    json_str = json_str.split("\n", 1)[1].rsplit("```", 1)[0]
                changes = json.loads(json_str)
            except json.JSONDecodeError:
                pass

    return corrected_text, changes


# ---------------------------------------------------------------------------
# Sample misspelled transcriptions
# ---------------------------------------------------------------------------

MISSPELLED_SAMPLES = [
    (
        "Patient is on lissinopril 20mg and metforman 1000mg twice daily. "
        "His hemogloben A1c is 8.9 and creatanine is 1.2. "
        "History of hypertention and diabeties with early neropathy."
    ),
    (
        "Ordered echocardigram and electrocadiogram. Troponon levels pending. "
        "Started nitroglycerine sublingual PRN for chest pain. "
        "Continue atorvastain 40mg and amlodapine 5mg daily."
    ),
]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_medical_spelling():
    print("=" * 70)
    print("  Exercise 3 — Medical Spelling Corrector")
    print("=" * 70)

    # --- Part A: Rule-based correction ---
    print("\n--- Part A: Rule-Based Correction ---\n")
    for i, sample in enumerate(MISSPELLED_SAMPLES, 1):
        print(f"Sample {i} (original):")
        print(f"  {sample}\n")
        corrected, fixes = rule_based_correct(sample)
        print(f"Sample {i} (corrected):")
        print(f"  {corrected}\n")
        print(f"  Corrections: {len(fixes)}")
        for f in fixes:
            print(f"    '{f['original']}' → '{f['corrected']}'")
        print()

    # --- Part B: Embedding-based matching ---
    print("-" * 70)
    print("\n--- Part B: Embedding-Based Term Matching ---\n")

    checker = MedicalSpellChecker(MEDICAL_DICTIONARY)

    test_terms = ["lissinopril", "metforman", "echocardigram", "neropathy", "gabapenten"]
    for term in test_terms:
        matches = checker.find_closest(term, top_k=3)
        suggestion = checker.suggest_correction(term)
        print(f"  '{term}' → suggested: {suggestion}")
        for m, score in matches:
            print(f"      {m}: {score:.4f}")
        print()

    # --- Part C: GPT-based contextual correction ---
    print("-" * 70)
    print("\n--- Part C: GPT-Based Contextual Correction ---\n")

    for i, sample in enumerate(MISSPELLED_SAMPLES, 1):
        print(f"Sample {i} — sending to GPT for correction …\n")
        corrected, changes = gpt_based_correct(sample)
        print(f"Corrected: {corrected}\n")
        if changes:
            print(f"Changes ({len(changes)}):")
            for c in changes:
                print(f"  '{c.get('original', '?')}' → '{c.get('corrected', '?')}'")
        print()

    # --- Interactive ---
    print("-" * 70)
    user_text = input("\nEnter text to spell-check (or Enter to skip) → ").strip()
    if user_text:
        corrected, changes = gpt_based_correct(user_text)
        print(f"\nCorrected: {corrected}")
        if changes:
            for c in changes:
                print(f"  '{c.get('original')}' → '{c.get('corrected')}'")


if __name__ == "__main__":
    demo_medical_spelling()
