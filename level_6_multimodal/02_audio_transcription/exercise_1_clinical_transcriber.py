"""




























































- Monitor token usage carefully — reasoning tokens add up fast- Always benchmark: reasoning models aren't always better- o1-mini is more cost-effective than o1 for most tasks- Reasoning model access may require specific API tier## Important Notes```python exercise_1_differential_diagnosis.py # Run exercise directlypython main.py                              # Interactive demo menu```bash## Running```pip install openai python-dotenv```bash## Prerequisites4. `exercise_4_reasoning_evaluation.py` — Evaluate reasoning model performance3. `exercise_3_model_selector.py` — Automatic query → model routing system2. `exercise_2_treatment_planning.py` — Complex treatment with comorbidities1. `exercise_1_differential_diagnosis.py` — Ranked differential with reasoning model## Exercises4. **When to Use Reasoning** — decision framework with examples3. **Cost/Latency Analysis** — token cost and latency comparison2. **Extended Thinking** — multi-step differential diagnosis case1. **Reasoning vs Standard** — compare GPT-4o vs o1-mini on complex medical question## Demos in main.py- Route queries to cheapest capable model for cost optimization- Latency: 10-60 seconds vs 1-5 seconds for standard models- o1-mini: ~3-5x cost of GPT-4o for complex tasks- Reasoning tokens are billed but not visible in output### Cost/Latency Tradeoffs  with multiple comorbidities, complex clinical decision-making- **Reasoning**: Differential diagnosis, drug interaction analysis, treatment planning- **Standard**: Simple lookups, formatting, summarization, classification### When to Use Reasoning Models- No streaming support for some reasoning models- Higher latency, higher cost per token- `reasoning_effort` parameter: low, medium, high- Use `max_completion_tokens` instead of `max_tokens`- No `system` message — use `user` or `developer` role instead### API Differences for Reasoning Models- Reasoning models "think" before answering — better for differential diagnosis, treatment planning- **Reasoning models** (o1-mini, o1, o3-mini): Internal chain-of-thought for complex logic- **Standard models** (GPT-4o, GPT-4o-mini): Fast, cost-effective for routine tasks### Reasoning vs Standard Models## Key Conceptsand how to manage cost/latency tradeoffs in healthcare applications.multi-step clinical reasoning. Learn when to use them versus standard models,Explore OpenAI's reasoning models (o1, o3-mini) that excel at complex,## OverviewExercise 1 — Clinical Encounter Transcriber
=============================================
Build a clinical encounter transcriber that processes audio → cleans
the text → identifies speakers → formats as structured dialogue.

Objectives
----------
* Transcribe audio with Whisper (or process simulated text)
* Detect speaker turns from contextual cues
* Clean up medical terminology and numbers
* Output as structured dialogue JSON
"""

import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# ---------------------------------------------------------------------------
# Simulated raw transcript (as if returned by Whisper without speaker labels)
# ---------------------------------------------------------------------------

SIMULATED_UNLABELED_TRANSCRIPT = """\
Good morning, I'm Dr. Patel. What brings you in today? \
I've been having chest pain on and off for about three days now. \
Can you describe the pain? Is it sharp, dull, or pressure-like? \
It feels like pressure, right in the center of my chest. \
Sometimes it goes to my left arm. \
Does anything make it better or worse? \
It gets worse when I walk up stairs and better when I rest. \
How long does each episode last? \
Maybe five to ten minutes each time. \
Any shortness of breath, nausea, or sweating? \
A little short of breath during the episodes. No nausea or sweating though. \
Do you have any history of heart disease? \
My father had a heart attack at fifty five. \
I see. And what medications are you currently taking? \
I take atorvastatin twenty milligrams, aspirin eighty one milligrams, \
and amlodipine five milligrams for my blood pressure. \
Any allergies? \
Just penicillin. I get a rash. \
Let me listen to your heart and check your vitals. \
Blood pressure is one fifty over ninety five. Heart rate is eighty eight. \
Oxygen sat is ninety seven percent. \
Heart sounds are regular, no murmurs. Lungs are clear. \
I'm concerned about possible angina given your symptoms and family history. \
I'd like to order an EKG, troponin levels, and a stress test. \
In the meantime, continue your current medications. \
If you experience severe chest pain, come to the ER immediately. \
Let's follow up in one week with the test results. \
Okay doctor, thank you.\
"""


# ---------------------------------------------------------------------------
# Speaker detection with GPT
# ---------------------------------------------------------------------------

SPEAKER_DETECTION_PROMPT = """\
You are a medical transcription specialist.  The following text is a raw
transcription of a clinical encounter between a doctor and a patient.
Speaker labels are missing.

Your task: Add speaker labels to each sentence or logical utterance.
Use these labels: PROVIDER, PATIENT.

Rules:
- Each utterance should start with "PROVIDER: " or "PATIENT: "
- Determine the speaker from context (questions about symptoms = provider,
  answers about how they feel = patient, exam findings = provider, etc.)
- Preserve the original wording exactly
- Put each labeled utterance on its own line

Return ONLY the labeled transcript, nothing else.
"""


def detect_speakers_gpt(raw_text: str) -> str:
    """Use GPT to add speaker labels to unlabeled transcription text."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SPEAKER_DETECTION_PROMPT},
            {"role": "user", "content": raw_text},
        ],
        max_tokens=1200,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Rule-based speaker detection (fallback / lightweight)
# ---------------------------------------------------------------------------

PROVIDER_CUES = [
    r"\b(I'm Dr\.|doctor|let me|I'd like to order|I'm going to|"
    r"let's|I see|I'm concerned|blood pressure is|heart rate|"
    r"oxygen sat|heart sounds|lungs are|no murmurs)\b",
]
PATIENT_CUES = [
    r"\b(I've been|I feel|it feels|I take|my father|I get|"
    r"okay doctor|thank you|sometimes it|a little|maybe five)\b",
]


def detect_speakers_rules(raw_text: str) -> list[dict]:
    """Simple rule-based speaker detection using keyword patterns."""
    sentences = re.split(r'(?<=[.?!])\s+', raw_text)
    dialogue = []
    provider_re = re.compile("|".join(PROVIDER_CUES), re.IGNORECASE)
    patient_re = re.compile("|".join(PATIENT_CUES), re.IGNORECASE)

    prev_speaker = "PROVIDER"  # assume doctor starts

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        p_match = provider_re.search(sentence)
        pt_match = patient_re.search(sentence)

        if p_match and not pt_match:
            speaker = "PROVIDER"
        elif pt_match and not p_match:
            speaker = "PATIENT"
        else:
            speaker = prev_speaker  # carry forward

        dialogue.append({"speaker": speaker, "text": sentence})
        prev_speaker = speaker

    return dialogue


# ---------------------------------------------------------------------------
# Transcription cleanup with GPT
# ---------------------------------------------------------------------------

CLEANUP_PROMPT = """\
Clean up this medical transcription:
1. Convert spelled-out numbers to digits (e.g., "twenty milligrams" → "20 mg")
2. Capitalize drug names properly (e.g., "atorvastatin" → "Atorvastatin")
3. Normalize vital signs (e.g., "one fifty over ninety five" → "150/95")
4. Fix medical abbreviations and terminology
5. Preserve speaker labels if present

Return only the cleaned text.
"""


def cleanup_transcription(text: str) -> str:
    """Use GPT to clean medical transcription text."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": CLEANUP_PROMPT},
            {"role": "user", "content": text},
        ],
        max_tokens=1200,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Full transcription pipeline
# ---------------------------------------------------------------------------

def transcribe_encounter(audio_path: str = None) -> dict:
    """Process a clinical encounter from audio to structured dialogue.

    Steps:
    1. Transcribe audio (Whisper) — or use simulated text
    2. Detect speakers
    3. Clean up medical terminology
    4. Return structured dialogue

    Parameters
    ----------
    audio_path : str, optional
        Path to an audio file.  If None, simulated text is used.

    Returns
    -------
    dict with keys: raw_text, labeled_text, cleaned_text, dialogue
    """
    # Step 1: Transcribe
    if audio_path and os.path.exists(audio_path):
        print("  [1/4] Transcribing audio with Whisper …")
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en",
                prompt="atorvastatin, amlodipine, troponin, EKG, angina",
                response_format="text",
            )
        raw_text = result if isinstance(result, str) else result.text
    else:
        print("  [1/4] Using simulated transcription (no audio file) …")
        raw_text = SIMULATED_UNLABELED_TRANSCRIPT

    # Step 2: Speaker detection
    print("  [2/4] Detecting speakers with GPT …")
    labeled_text = detect_speakers_gpt(raw_text)

    # Step 3: Cleanup
    print("  [3/4] Cleaning up medical terminology …")
    cleaned_text = cleanup_transcription(labeled_text)

    # Step 4: Structure as dialogue
    print("  [4/4] Structuring as dialogue …")
    dialogue = []
    for line in cleaned_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("PROVIDER:"):
            dialogue.append({"speaker": "PROVIDER", "text": line[9:].strip()})
        elif line.startswith("PATIENT:"):
            dialogue.append({"speaker": "PATIENT", "text": line[8:].strip()})
        else:
            dialogue.append({"speaker": "UNKNOWN", "text": line})

    return {
        "raw_text": raw_text,
        "labeled_text": labeled_text,
        "cleaned_text": cleaned_text,
        "dialogue": dialogue,
        "turn_count": len(dialogue),
        "speakers": list(set(d["speaker"] for d in dialogue)),
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_clinical_transcriber():
    print("=" * 70)
    print("  Exercise 1 — Clinical Encounter Transcriber")
    print("=" * 70)

    # --- Part A: Rule-based detection ---
    print("\n--- Part A: Rule-Based Speaker Detection ---\n")
    rules_result = detect_speakers_rules(SIMULATED_UNLABELED_TRANSCRIPT)
    print(f"Detected {len(rules_result)} utterances\n")
    for turn in rules_result[:6]:
        print(f"  [{turn['speaker']}] {turn['text'][:70]}")
    print("  …\n")

    # --- Part B: Full pipeline ---
    print("-" * 70)
    print("\n--- Part B: Full Transcription Pipeline ---\n")
    result = transcribe_encounter()

    print(f"\nTurn count: {result['turn_count']}")
    print(f"Speakers:   {', '.join(result['speakers'])}\n")

    print("=== Structured Dialogue (first 8 turns) ===\n")
    for turn in result["dialogue"][:8]:
        print(f"  [{turn['speaker']}] {turn['text'][:80]}")

    # --- Save result ---
    output_path = "transcription_output.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull result saved to {output_path}")

    # --- Interactive ---
    print("\n" + "-" * 70)
    user_file = input("\nEnter audio file path (or Enter to skip) → ").strip()
    if user_file:
        r = transcribe_encounter(user_file)
        print(json.dumps(r["dialogue"][:5], indent=2))


if __name__ == "__main__":
    demo_clinical_transcriber()
