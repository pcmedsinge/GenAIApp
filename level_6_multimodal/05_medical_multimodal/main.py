"""
Project 05: Medical Multimodal Capstone — Main Demo
====================================================
Combines vision, audio transcription, reasoning models, and structured outputs
into integrated clinical workflows. This capstone demonstrates real-world
healthcare AI pipelines that process multiple input modalities.

Demos:
  1. Image + Text Analysis — medical image with patient history context
  2. Audio + Structured Output — clinical audio → structured SOAP note
  3. Reasoning + Multiple Inputs — complex case via reasoning model
  4. Full Multimodal Pipeline — all modalities in single clinical workflow
"""

import base64
import json
import os
import time
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI()
MODEL = "gpt-4o"
REASONING_MODEL = "o1-mini"


# ============================================================================
# SHARED SCHEMAS
# ============================================================================

class SOAPNote(BaseModel):
    """Structured SOAP note schema."""
    subjective: str = Field(..., description="Patient's reported symptoms and history")
    objective: str = Field(..., description="Examination findings and vitals")
    assessment: str = Field(..., description="Clinical assessment and diagnoses")
    plan: List[str] = Field(..., description="Treatment plan items")


class TriageResult(BaseModel):
    """Triage classification result."""
    urgency: Literal["emergent", "urgent", "semi-urgent", "non-urgent"] = Field(
        ..., description="Urgency classification"
    )
    recommended_action: str = Field(..., description="Recommended next step")
    reasoning: str = Field(..., description="Clinical reasoning for classification")
    confidence: float = Field(..., description="Confidence score 0-1")


class ClinicalFinding(BaseModel):
    """A single clinical finding from any modality."""
    source: str = Field(..., description="Data source: image, audio, text, lab")
    finding: str = Field(..., description="The clinical finding")
    significance: Literal["critical", "significant", "minor", "normal"] = Field(
        ..., description="Clinical significance level"
    )


class MultimodalAssessment(BaseModel):
    """Combined assessment from multiple modalities."""
    patient_summary: str = Field(..., description="Brief patient summary")
    findings: List[ClinicalFinding] = Field(..., description="All findings by modality")
    primary_assessment: str = Field(..., description="Primary clinical assessment")
    differential_diagnoses: List[str] = Field(
        default_factory=list, description="Differential diagnoses ranked by likelihood"
    )
    recommended_actions: List[str] = Field(
        default_factory=list, description="Recommended next steps"
    )
    urgency: Literal["emergent", "urgent", "semi-urgent", "non-urgent"] = Field(
        ..., description="Overall urgency"
    )
    confidence: float = Field(..., description="Overall confidence 0-1")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def simulate_audio_transcription(audio_description: str) -> str:
    """
    Simulate audio transcription using GPT-4o.
    In production, use client.audio.transcriptions.create() with Whisper.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are simulating an audio transcription of a clinical encounter. "
                    "Generate a realistic verbatim transcription based on the description. "
                    "Include natural speech patterns, filler words, and realistic dialogue."
                ),
            },
            {
                "role": "user",
                "content": f"Generate a verbatim transcription for: {audio_description}",
            },
        ],
        max_tokens=800,
    )
    return response.choices[0].message.content


def analyze_with_reasoning(prompt: str, max_tokens: int = 3000) -> dict:
    """Call a reasoning model and return response with metadata."""
    start = time.time()
    response = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_tokens,
    )
    elapsed = time.time() - start
    return {
        "content": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
        "elapsed": elapsed,
    }


# ============================================================================
# DEMO 1: IMAGE + TEXT ANALYSIS
# ============================================================================

def demo_1_image_text_analysis():
    """
    Analyze a medical image with patient history context.
    Uses vision capabilities combined with clinical text.
    """
    print("\n" + "=" * 70)
    print("DEMO 1: Image + Text Analysis")
    print("=" * 70)

    # Simulate image analysis (describe what we'd see in an image)
    # In production, you'd send an actual image via base64 or URL
    patient_history = """
    Patient: Margaret Wilson, 62-year-old female
    History: Type 2 diabetes (15 years), peripheral neuropathy, PAD
    Current concern: Non-healing wound on right foot, present for 3 weeks
    Current medications: Metformin, Insulin glargine, Gabapentin, Aspirin
    Recent HbA1c: 9.2%
    """

    image_description = """
    [SIMULATED IMAGE DESCRIPTION — In production, send actual image via base64]
    Right foot, plantar surface: 2.5 cm diameter ulcer on the first metatarsal head.
    Wound base has pink granulation tissue with some yellow fibrinous material.
    Surrounding skin shows callus formation. Mild erythema extending 1 cm from wound edge.
    No visible bone or tendon. Foot appears warm. Toes are intact with thickened nails.
    """

    print(f"\n--- Patient History ---\n{patient_history.strip()}")
    print(f"\n--- Image Description ---\n{image_description.strip()}")

    print("\n--- Multimodal Analysis (Image + Text) ---")
    start = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a wound care specialist analyzing a wound image in the "
                    "context of the patient's medical history. Provide:\n"
                    "1. Wound classification (Wagner grade)\n"
                    "2. Key findings from image\n"
                    "3. Risk factors from history\n"
                    "4. Recommended treatment plan\n"
                    "5. Urgency level"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Patient History:\n{patient_history}\n\n"
                    f"Wound Image Findings:\n{image_description}\n\n"
                    "Provide comprehensive wound assessment."
                ),
            },
        ],
        max_tokens=1200,
    )
    elapsed = time.time() - start

    print(f"\n{response.choices[0].message.content}")
    print(f"\n  [Tokens: {response.usage.total_tokens} | Time: {elapsed:.2f}s]")

    # Show how you'd send a real image
    print("\n--- How to Send Real Images ---")
    print("""
    # With base64-encoded image:
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this wound..."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"  # high detail for medical images
                }}
            ]
        }]
    )
    """)


# ============================================================================
# DEMO 2: AUDIO + STRUCTURED OUTPUT
# ============================================================================

def demo_2_audio_structured_output():
    """
    Transcribe clinical audio → generate structured SOAP note.
    Combines audio processing with structured output extraction.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Audio + Structured Output — Clinical Audio → SOAP Note")
    print("=" * 70)

    audio_scenario = (
        "A primary care physician examining a 45-year-old male patient with "
        "complaints of persistent lower back pain for 2 weeks after lifting "
        "heavy furniture. The doctor asks about pain location, severity (7/10), "
        "radiation, numbness/tingling (none), performs physical exam, and "
        "discusses treatment plan including NSAIDs, physical therapy referral, "
        "and activity modification."
    )

    print(f"\n--- Audio Scenario ---\n  {audio_scenario}")

    # Step 1: Simulate transcription
    print("\n--- Step 1: Audio Transcription ---")
    transcript = simulate_audio_transcription(audio_scenario)
    print(f"\n  Transcript (first 300 chars):\n  {transcript[:300]}...")

    # Step 2: Extract structured SOAP note
    print("\n--- Step 2: Structured SOAP Note Extraction ---")
    start = time.time()
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical documentation specialist. Convert the "
                    "transcribed clinical encounter into a structured SOAP note. "
                    "Be thorough but concise. Use proper medical terminology."
                ),
            },
            {
                "role": "user",
                "content": f"Convert this transcription to a SOAP note:\n\n{transcript}",
            },
        ],
        response_format=SOAPNote,
    )
    elapsed = time.time() - start

    soap = completion.choices[0].message.parsed
    print(f"\n  SOAP NOTE")
    print(f"  {'=' * 50}")
    print(f"\n  SUBJECTIVE:")
    print(f"  {soap.subjective}")
    print(f"\n  OBJECTIVE:")
    print(f"  {soap.objective}")
    print(f"\n  ASSESSMENT:")
    print(f"  {soap.assessment}")
    print(f"\n  PLAN:")
    for i, item in enumerate(soap.plan, 1):
        print(f"  {i}. {item}")

    print(f"\n  [Tokens: {completion.usage.total_tokens} | Time: {elapsed:.2f}s]")

    print("\n--- How to Transcribe Real Audio ---")
    print("""
    # With actual audio file:
    with open("clinical_audio.mp3", "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en",
            prompt="Medical clinical encounter transcription"
        )
    # Then pass transcript.text to structured output extraction
    """)


# ============================================================================
# DEMO 3: REASONING + MULTIPLE INPUTS
# ============================================================================

def demo_3_reasoning_multiple_inputs():
    """
    Complex case with image description + labs + history → differential
    diagnosis via reasoning model.
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Reasoning + Multiple Inputs — Complex Case Analysis")
    print("=" * 70)

    # Multiple input sources
    patient_history = """
    PATIENT HISTORY:
    - 28-year-old female, previously healthy
    - 2-week history of progressive fatigue, fever, and joint pain
    - New rash on face and arms appearing with sun exposure
    - Recent hair loss noticed over past month
    - No significant past medical history
    - Family history: mother with autoimmune thyroiditis
    """

    lab_results = """
    LABORATORY RESULTS:
    - CBC: WBC 3.1 (L), Hgb 10.2 (L), Platelets 128 (L)
    - ESR: 62 mm/hr (H)
    - CRP: 3.4 mg/dL (H)
    - BMP: Creatinine 1.4 (H), BUN 22
    - Urinalysis: Protein 2+, RBC 8-10/hpf
    - ANA: Positive, 1:1280, speckled pattern
    - Anti-dsDNA: Positive (elevated)
    - C3: 58 (L), C4: 6 (L)
    - Anti-Sm: Positive
    """

    imaging_description = """
    IMAGING FINDINGS (simulated):
    - Chest X-ray: Small bilateral pleural effusions
    - Echocardiogram: Small pericardial effusion, normal EF
    - Skin: Malar rash photograph showing erythematous, butterfly-shaped
      rash across cheeks and nasal bridge, sparing nasolabial folds
    """

    combined_input = (
        "You are an expert diagnostician. Analyze the following multi-source "
        "clinical data and provide a comprehensive assessment. Use systematic "
        "reasoning to work through the differential diagnosis.\n\n"
        f"{patient_history}\n{lab_results}\n{imaging_description}\n\n"
        "Provide:\n"
        "1. Ranked differential diagnosis with reasoning for each\n"
        "2. Which ACR/EULAR criteria are met (if applicable)\n"
        "3. Most critical next steps\n"
        "4. Urgency assessment"
    )

    print(f"--- Input Sources ---")
    print(f"\n{patient_history.strip()}")
    print(f"\n{lab_results.strip()}")
    print(f"\n{imaging_description.strip()}")

    print(f"\n--- Reasoning Model Analysis ---")
    result = analyze_with_reasoning(combined_input, max_tokens=4000)
    print(f"\n{result['content']}")
    print(f"\n  [Tokens: {result['tokens']} | Time: {result['elapsed']:.2f}s]")
    print(f"\n  💡 Reasoning model synthesized data from 3 sources:")
    print(f"     - Patient history (text)")
    print(f"     - Lab results (structured data)")
    print(f"     - Imaging findings (simulated visual)")


# ============================================================================
# DEMO 4: FULL MULTIMODAL PIPELINE
# ============================================================================

def demo_4_full_pipeline():
    """
    Combine all modalities into a single clinical assessment workflow.
    Pipeline: Input → Process each modality → Combine → Reason → Structured Output.
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Full Multimodal Pipeline — End-to-End Clinical Workflow")
    print("=" * 70)

    # Simulate multiple input modalities
    print("\n--- STAGE 1: Input Collection ---")

    # Text input (patient history)
    text_input = """
    72-year-old male, history of COPD (on tiotropium, albuterol PRN),
    CHF (EF 35%, on carvedilol, lisinopril, furosemide), and type 2 diabetes.
    Presents with worsening dyspnea over 3 days, productive cough with
    yellow-green sputum, and low-grade fever.
    """
    print(f"  📝 Text Input (History): Captured")

    # Audio input (simulated clinical encounter)
    audio_description = (
        "A pulmonologist examining a 72-year-old male with acute exacerbation "
        "of COPD. Lung exam reveals bilateral wheezing and right-base crackles. "
        "SpO2 is 89% on room air. Patient reports increased sputum production "
        "and worsening shortness of breath over 3 days."
    )
    print(f"  🎤 Audio Input (Encounter): Captured")

    # Lab results (structured)
    lab_data = {
        "WBC": {"value": 14.2, "unit": "K/uL", "flag": "H", "ref": "4.5-11.0"},
        "Procalcitonin": {"value": 0.8, "unit": "ng/mL", "flag": "H", "ref": "<0.1"},
        "BNP": {"value": 850, "unit": "pg/mL", "flag": "H", "ref": "<100"},
        "Creatinine": {"value": 1.6, "unit": "mg/dL", "flag": "H", "ref": "0.7-1.3"},
        "ABG_pH": {"value": 7.32, "unit": "", "flag": "L", "ref": "7.35-7.45"},
        "ABG_pCO2": {"value": 52, "unit": "mmHg", "flag": "H", "ref": "35-45"},
    }
    print(f"  🔬 Lab Data: {len(lab_data)} results captured")

    # Image description (simulated chest X-ray)
    imaging_desc = (
        "Chest X-ray: Hyperinflated lungs, flattened diaphragms consistent with COPD. "
        "Right lower lobe infiltrate suggesting pneumonia. Bilateral costophrenic angle "
        "blunting suggesting small pleural effusions. Cardiomegaly."
    )
    print(f"  🖼️  Imaging: Chest X-ray findings captured")

    # STAGE 2: Process each modality
    print("\n--- STAGE 2: Modality Processing ---")

    # Process audio → transcript
    print("  Processing audio transcription...")
    transcript = simulate_audio_transcription(audio_description)
    print(f"  ✓ Transcript generated ({len(transcript)} chars)")

    # Format labs
    lab_text = "LABORATORY RESULTS:\n"
    for name, data in lab_data.items():
        flag = f" ({data['flag']})" if data.get("flag") else ""
        lab_text += f"  {name}: {data['value']} {data['unit']}{flag} [Ref: {data['ref']}]\n"
    print(f"  ✓ Labs formatted")

    # STAGE 3: Combined analysis
    print("\n--- STAGE 3: Combined Analysis (Structured Output) ---")

    combined_prompt = (
        f"Patient History:\n{text_input}\n\n"
        f"Clinical Encounter Transcript:\n{transcript}\n\n"
        f"{lab_text}\n"
        f"Imaging:\n{imaging_desc}\n\n"
        "Synthesize all information into a comprehensive clinical assessment."
    )

    start = time.time()
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior clinical decision support system. Analyze ALL "
                    "input sources (history, exam, labs, imaging) and provide a "
                    "comprehensive multimodal assessment. Identify which findings "
                    "came from which source."
                ),
            },
            {"role": "user", "content": combined_prompt},
        ],
        response_format=MultimodalAssessment,
    )
    elapsed = time.time() - start

    assessment = completion.choices[0].message.parsed

    print(f"\n  ╔{'═' * 58}╗")
    print(f"  ║  MULTIMODAL CLINICAL ASSESSMENT                         ║")
    print(f"  ╚{'═' * 58}╝")

    print(f"\n  Patient Summary: {assessment.patient_summary}")

    print(f"\n  Findings ({len(assessment.findings)} total):")
    for f in assessment.findings:
        icon = {
            "critical": "🔴", "significant": "🟠",
            "minor": "🟡", "normal": "🟢"
        }.get(f.significance, "⚪")
        print(f"    {icon} [{f.source}] {f.finding}")

    print(f"\n  Primary Assessment: {assessment.primary_assessment}")

    print(f"\n  Differential Diagnoses:")
    for i, dx in enumerate(assessment.differential_diagnoses, 1):
        print(f"    {i}. {dx}")

    print(f"\n  Recommended Actions:")
    for action in assessment.recommended_actions:
        print(f"    → {action}")

    print(f"\n  Urgency: {assessment.urgency.upper()}")
    print(f"  Confidence: {assessment.confidence:.0%}")
    print(f"\n  [Tokens: {completion.usage.total_tokens} | Time: {elapsed:.2f}s]")

    # STAGE 4: Pipeline summary
    print("\n--- PIPELINE SUMMARY ---")
    print(f"  Modalities processed: text, audio, labs, imaging")
    print(f"  Pipeline stages: input → process → combine → structured output")
    print(f"  Output: MultimodalAssessment (typed Pydantic model)")
    print(f"  This structured output can be fed into EHR systems, CDSSs, or dashboards.")


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    """Interactive menu for multimodal capstone demos."""
    demos = {
        "1": ("Image + Text Analysis", demo_1_image_text_analysis),
        "2": ("Audio + Structured Output (→ SOAP Note)", demo_2_audio_structured_output),
        "3": ("Reasoning + Multiple Inputs", demo_3_reasoning_multiple_inputs),
        "4": ("Full Multimodal Pipeline", demo_4_full_pipeline),
    }

    while True:
        print("\n" + "=" * 70)
        print("PROJECT 05: MEDICAL MULTIMODAL CAPSTONE — DEMO MENU")
        print("=" * 70)
        for key, (desc, _) in demos.items():
            print(f"  [{key}] {desc}")
        print(f"  [a] Run all demos")
        print(f"  [q] Quit")

        choice = input("\nSelect demo: ").strip().lower()

        if choice == "q":
            print("Goodbye!")
            break
        elif choice == "a":
            for key in sorted(demos.keys()):
                demos[key][1]()
        elif choice in demos:
            demos[choice][1]()
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
