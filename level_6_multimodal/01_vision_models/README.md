# Level 6, Project 01: Vision Models

## Image Understanding with GPT-4o

Learn to analyze, classify, and extract data from images using OpenAI's vision
capabilities. GPT-4o and GPT-4o-mini can accept images as input alongside text,
enabling powerful multimodal applications in healthcare.

## Key Concepts

- **Vision API**: Send images (URLs or base64) in the messages array
- **Image Analysis**: Describe, interpret, and reason about visual content
- **Document Extraction**: Pull structured data from scanned forms and documents
- **Image Classification**: Categorize images into predefined classes
- **Image Comparison**: Identify differences between multiple images

## API Pattern

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "https://..."}}
        ]
    }]
)
```

## Demos (main.py)

| # | Demo | Description |
|---|------|-------------|
| 1 | Image Analysis Basics | Send an image URL, get a description |
| 2 | Image Classification | Classify images into medical categories |
| 3 | Document Extraction | Extract structured data from form images |
| 4 | Image Comparison | Compare two images for differences |

## Exercises

| # | File | Description |
|---|------|-------------|
| 1 | exercise_1_medical_form_reader.py | Extract patient data from form images as JSON |
| 2 | exercise_2_chart_analyzer.py | Analyze medical charts and graphs |
| 3 | exercise_3_image_classification.py | Multi-category image classifier |
| 4 | exercise_4_vision_pipeline.py | End-to-end vision processing pipeline |

## Running

```bash
# Run all demos
python main.py

# Run individual exercises
python exercise_1_medical_form_reader.py
python exercise_2_chart_analyzer.py
python exercise_3_image_classification.py
python exercise_4_vision_pipeline.py
```

## Prerequisites

- OpenAI API key with GPT-4o or GPT-4o-mini access
- `pip install openai python-dotenv`

## Notes

- Vision is available on gpt-4o, gpt-4o-mini, and gpt-4-turbo models
- Images can be passed as URLs or base64-encoded strings
- The `detail` parameter controls image resolution (low/high/auto)
- Demos use public image URLs; replace with your own images as needed
