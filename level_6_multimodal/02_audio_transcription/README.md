# Level 6, Project 02: Audio Transcription

## Whisper API & Medical Transcription

Learn to transcribe audio using OpenAI's Whisper model, handle medical
terminology, and generate structured clinical notes from transcriptions.

## Key Concepts

- **Whisper API**: Transcribe audio files to text with high accuracy
- **Medical Terminology**: Use prompts to improve medical term recognition
- **Speaker Diarization**: Identify and label different speakers
- **SOAP Notes**: Generate structured clinical notes from transcriptions
- **Post-Processing**: Clean, correct, and format raw transcriptions

## API Pattern

```python
# Transcription
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    language="en",
    response_format="text",
    prompt="Medical terms: hypertension, metformin, HbA1c"
)

# SOAP note generation from transcript
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": f"Generate SOAP note:\n{transcript}"}]
)
```

## Demos (main.py)

| # | Demo | Description |
|---|------|-------------|
| 1 | Whisper Basics | Transcribe audio, explore API parameters |
| 2 | Medical Terminology | Improve medical term accuracy with prompts |
| 3 | Transcription Post-Processing | Clean and format raw transcriptions |
| 4 | SOAP Note Generation | Generate structured SOAP notes from text |

## Exercises

| # | File | Description |
|---|------|-------------|
| 1 | exercise_1_clinical_transcriber.py | Full clinical encounter transcriber |
| 2 | exercise_2_soap_generator.py | Raw transcription to SOAP note |
| 3 | exercise_3_medical_spelling.py | Medical terminology spell-checker |
| 4 | exercise_4_transcription_pipeline.py | End-to-end transcription pipeline |

## Running

```bash
# Run all demos
python main.py

# Run individual exercises
python exercise_1_clinical_transcriber.py
python exercise_2_soap_generator.py
python exercise_3_medical_spelling.py
python exercise_4_transcription_pipeline.py
```

## Prerequisites

- OpenAI API key with Whisper and GPT-4o access
- `pip install openai python-dotenv`

## Notes

- Whisper supports mp3, mp4, mpeg, mpga, m4a, wav, and webm formats
- Maximum audio file size is 25 MB
- The `prompt` parameter helps with domain-specific vocabulary
- Demos simulate audio input with sample text where real audio is unavailable
