"""
Exercise 2 — Medical Chart / Graph Analyzer
============================================
Analyze medical charts, graphs, and tracings from images.  Describe
trends, identify potential abnormalities, and summarize findings.

Objectives
----------
* Send a chart / graph image to GPT-4o-mini
* Prompt the model to provide structured clinical analysis
* Detect trends (increasing, decreasing, stable)
* Flag potential abnormalities against normal reference ranges
* Produce a concise findings summary suitable for a clinician
"""

import json
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

VISION_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_image_message(text: str, image_source: str, detail: str = "high") -> dict:
    if image_source.startswith(("http://", "https://", "data:")):
        url = image_source
    else:
        ext = os.path.splitext(image_source)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        with open(image_source, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        url = f"data:{mime};base64,{b64}"
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": url, "detail": detail}},
        ],
    }


# ---------------------------------------------------------------------------
# Analysis prompts
# ---------------------------------------------------------------------------

CHART_ANALYSIS_SYSTEM = """\
You are a clinical data analyst specializing in reading medical charts,
graphs, and diagnostic tracings.

When presented with a medical chart or graph image, you must:

1. **Identify** — What type of chart is this? (ECG, vital-signs trend,
   lab-value graph, growth chart, etc.)
2. **Describe** — Describe the axes, units, time range, and data points
   or waveforms visible.
3. **Trends** — Identify overall trends: increasing, decreasing, stable,
   or cyclical patterns.
4. **Abnormalities** — Flag any values or waveform features that fall
   outside normal reference ranges.  Cite the normal range when possible.
5. **Summary** — Write a concise 2-3 sentence clinical summary suitable
   for a physician's review.

Respond in the following JSON format (no markdown fences):

{
  "chart_type": "<string>",
  "description": "<string>",
  "axes": {"x": "<label and unit>", "y": "<label and unit>"},
  "data_points_observed": ["<list of notable values or features>"],
  "trends": ["<list of identified trends>"],
  "abnormalities": ["<list of flagged issues with normal ranges>"],
  "clinical_summary": "<string>"
}
"""


def analyze_chart(image_source: str) -> dict:
    """Analyze a medical chart image and return structured findings.

    Parameters
    ----------
    image_source : str
        URL or local path to a chart / graph image.

    Returns
    -------
    dict — structured analysis results.
    """
    user_message = build_image_message(
        "Analyze this medical chart or graph image.",
        image_source,
    )

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": CHART_ANALYSIS_SYSTEM},
            user_message,
        ],
        max_tokens=800,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw_response": raw, "_error": "JSON parse failed"}


# ---------------------------------------------------------------------------
# Follow-up Q&A about a chart
# ---------------------------------------------------------------------------

def ask_about_chart(image_source: str, question: str) -> str:
    """Ask a follow-up clinical question about a chart image.

    Parameters
    ----------
    image_source : str
        URL or local path to the chart image.
    question : str
        Free-text clinical question about the chart.

    Returns
    -------
    str — the model's answer.
    """
    user_message = build_image_message(
        f"You are a clinical analyst.  Here is a medical chart.\n\n"
        f"Question: {question}\n\n"
        f"Answer concisely with clinical reasoning.",
        image_source,
    )

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[user_message],
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Trend comparison helper
# ---------------------------------------------------------------------------

NORMAL_RANGES = {
    "heart_rate_bpm": (60, 100),
    "systolic_bp_mmhg": (90, 140),
    "diastolic_bp_mmhg": (60, 90),
    "temperature_f": (97.0, 99.5),
    "respiratory_rate": (12, 20),
    "spo2_pct": (95, 100),
    "glucose_mg_dl": (70, 100),
    "potassium_meq_l": (3.5, 5.0),
}


def flag_abnormals(values: dict) -> list[dict]:
    """Compare extracted numeric values against normal ranges.

    Parameters
    ----------
    values : dict
        Mapping of metric name → numeric value.

    Returns
    -------
    list[dict] — list of flagged abnormalities.
    """
    flags = []
    for metric, value in values.items():
        metric_key = metric.lower().replace(" ", "_")
        if metric_key in NORMAL_RANGES:
            lo, hi = NORMAL_RANGES[metric_key]
            if value < lo:
                flags.append({
                    "metric": metric,
                    "value": value,
                    "status": "LOW",
                    "normal_range": f"{lo}–{hi}",
                })
            elif value > hi:
                flags.append({
                    "metric": metric,
                    "value": value,
                    "status": "HIGH",
                    "normal_range": f"{lo}–{hi}",
                })
    return flags


# ---------------------------------------------------------------------------
# Sample images
# ---------------------------------------------------------------------------

SAMPLE_ECG_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "1/1b/Sinus_rhythm_labels.svg/"
    "600px-Sinus_rhythm_labels.svg.png"
)

SAMPLE_ANATOMY_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "d/d5/Anatomical_chart%2C_Cyclopaedia%2C_1728%2C_Volume_1.jpg/"
    "440px-Anatomical_chart%2C_Cyclopaedia%2C_1728%2C_Volume_1.jpg"
)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_chart_analyzer():
    print("=" * 70)
    print("  Exercise 2 — Medical Chart Analyzer")
    print("=" * 70)

    # --- Part A: Structured analysis of an ECG image ---
    print("\n--- Part A: Structured Chart Analysis (ECG) ---\n")
    print(f"Image: {SAMPLE_ECG_URL}\n")
    print("Analyzing chart …\n")

    result = analyze_chart(SAMPLE_ECG_URL)
    print(json.dumps(result, indent=2))

    # --- Part B: Follow-up question ---
    print("\n--- Part B: Follow-up Question ---\n")
    question = "Is the heart rate within normal limits based on this ECG tracing?"
    print(f"Q: {question}\n")
    answer = ask_about_chart(SAMPLE_ECG_URL, question)
    print(f"A: {answer}\n")

    # --- Part C: Abnormal flagging with sample values ---
    print("-" * 70)
    print("\n--- Part C: Abnormal Value Flagging (simulated) ---\n")
    sample_vitals = {
        "heart_rate_bpm": 112,
        "systolic_bp_mmhg": 155,
        "diastolic_bp_mmhg": 72,
        "spo2_pct": 91,
        "temperature_f": 98.6,
    }
    print(f"Vitals: {sample_vitals}\n")
    flags = flag_abnormals(sample_vitals)
    if flags:
        print("Flagged abnormalities:")
        for f in flags:
            print(f"  ⚠ {f['metric']}: {f['value']} ({f['status']}) "
                  f"— normal: {f['normal_range']}")
    else:
        print("All values within normal range.")

    # --- Interactive ---
    print("\n" + "-" * 70)
    user_url = input("\nEnter a chart image URL to analyze (or Enter to skip) → ").strip()
    if user_url:
        print("\nAnalyzing …\n")
        res = analyze_chart(user_url)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    demo_chart_analyzer()
