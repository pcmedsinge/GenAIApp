"""
Exercise 4: End-to-End Multimodal Pipeline with Evaluation
============================================================
Build a complete multimodal clinical AI pipeline that processes multiple
input types, combines them through reasoning, generates structured output,
and evaluates the quality of results.

Pipeline: Input → Process Each Modality → Combine → Reason → Structured
          Output → Quality Score

Learning Objectives:
  - Design end-to-end multimodal pipelines
  - Implement pipeline stage evaluation
  - Combine all modalities: vision, audio, text, structured data
  - Build quality scoring for clinical AI outputs
  - Handle errors across pipeline stages gracefully

Usage:
  python exercise_4_multimodal_pipeline.py
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI()
MODEL = "gpt-4o"
REASONING_MODEL = "o1-mini"


# ============================================================================
# PIPELINE SCHEMAS
# ============================================================================

class ModalityInput(BaseModel):
    """A single input modality."""
    modality_type: Literal["text", "image", "audio", "lab_data", "structured_data"]
    content: str = Field(..., description="Content or description of the input")
    metadata: Dict[str, str] = Field(default_factory=dict)


class ProcessedModality(BaseModel):
    """Processed output from a single modality."""
    modality_type: str
    extracted_findings: List[str]
    key_data_points: Dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(..., ge=0, le=1)
    processing_notes: Optional[str] = None


class ClinicalDiagnosis(BaseModel):
    """A diagnosis with supporting evidence."""
    name: str
    confidence: float = Field(..., ge=0, le=1)
    supporting_evidence: List[str]
    evidence_sources: List[str] = Field(
        default_factory=list, description="Which modalities support this"
    )


class PipelineOutput(BaseModel):
    """Final structured output from the multimodal pipeline."""
    patient_id: str
    timestamp: str
    modalities_processed: List[str]
    patient_summary: str
    primary_diagnosis: ClinicalDiagnosis
    differential_diagnoses: List[ClinicalDiagnosis]
    critical_findings: List[str]
    recommended_actions: List[str]
    urgency: Literal["emergent", "urgent", "semi-urgent", "routine"]
    overall_confidence: float = Field(..., ge=0, le=1)
    pipeline_notes: List[str] = Field(default_factory=list)


class StageEvaluation(BaseModel):
    """Evaluation of a single pipeline stage."""
    stage_name: str
    score: int = Field(..., ge=1, le=5)
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class PipelineEvaluation(BaseModel):
    """Complete pipeline quality evaluation."""
    stage_evaluations: List[StageEvaluation]
    overall_quality_score: float = Field(..., ge=0, le=1)
    clinical_accuracy_score: float = Field(..., ge=0, le=1)
    completeness_score: float = Field(..., ge=0, le=1)
    safety_score: float = Field(..., ge=0, le=1)
    key_strengths: List[str]
    key_weaknesses: List[str]
    improvement_suggestions: List[str]


# ============================================================================
# PIPELINE TRACKING
# ============================================================================

@dataclass
class PipelineMetrics:
    """Track metrics across pipeline stages."""
    stages: List[dict] = field(default_factory=list)
    total_tokens: int = 0
    total_time: float = 0
    errors: List[str] = field(default_factory=list)

    def record_stage(self, name: str, tokens: int, elapsed: float,
                     success: bool, notes: str = ""):
        self.stages.append({
            "name": name,
            "tokens": tokens,
            "elapsed": elapsed,
            "success": success,
            "notes": notes,
        })
        self.total_tokens += tokens
        self.total_time += elapsed

    def summary(self) -> str:
        lines = [f"Pipeline: {len(self.stages)} stages"]
        for s in self.stages:
            status = "✓" if s["success"] else "✗"
            lines.append(f"  {status} {s['name']}: {s['elapsed']:.2f}s, {s['tokens']} tokens")
        lines.append(f"  Total: {self.total_time:.2f}s, {self.total_tokens} tokens")
        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
        return "\n".join(lines)


# ============================================================================
# PIPELINE STAGES
# ============================================================================

def stage_process_text(text_input: str, metrics: PipelineMetrics) -> ProcessedModality:
    """Stage 1a: Process text input (patient history)."""
    start = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract key clinical findings from the patient history text. "
                    "Return as JSON with fields: findings (list of strings), "
                    "key_data (dict of name:value pairs), confidence (0-1)."
                ),
            },
            {"role": "user", "content": f"Extract findings:\n\n{text_input}"},
        ],
        max_tokens=800,
    )
    elapsed = time.time() - start
    content = response.choices[0].message.content
    tokens = response.usage.total_tokens
    metrics.record_stage("process_text", tokens, elapsed, True)

    # Parse response into structured format
    try:
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            data = json.loads(content[start_idx:end_idx])
        else:
            data = {"findings": [content], "key_data": {}, "confidence": 0.7}
    except json.JSONDecodeError:
        data = {"findings": [content], "key_data": {}, "confidence": 0.7}

    return ProcessedModality(
        modality_type="text",
        extracted_findings=data.get("findings", []),
        key_data_points=data.get("key_data", {}),
        confidence=data.get("confidence", 0.7),
    )


def stage_process_image(image_description: str,
                        metrics: PipelineMetrics) -> ProcessedModality:
    """Stage 1b: Process image input (simulated via description)."""
    start = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Analyze the medical image findings. Extract clinical findings "
                    "and rate confidence. Return JSON: {findings: [...], "
                    "key_data: {...}, confidence: 0-1}."
                ),
            },
            {"role": "user", "content": f"Analyze image:\n\n{image_description}"},
        ],
        max_tokens=600,
    )
    elapsed = time.time() - start
    content = response.choices[0].message.content
    tokens = response.usage.total_tokens
    metrics.record_stage("process_image", tokens, elapsed, True)

    try:
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            data = json.loads(content[start_idx:end_idx])
        else:
            data = {"findings": [content], "key_data": {}, "confidence": 0.6}
    except json.JSONDecodeError:
        data = {"findings": [content], "key_data": {}, "confidence": 0.6}

    return ProcessedModality(
        modality_type="image",
        extracted_findings=data.get("findings", []),
        key_data_points=data.get("key_data", {}),
        confidence=data.get("confidence", 0.6),
    )


def stage_process_audio(audio_description: str,
                        metrics: PipelineMetrics) -> ProcessedModality:
    """Stage 1c: Process audio input (simulated transcription)."""
    start = time.time()
    # Simulate transcription
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Simulate transcribing and extracting clinical findings from "
                    "the described audio. Return JSON: {transcript_summary: str, "
                    "findings: [...], key_data: {...}, confidence: 0-1}."
                ),
            },
            {"role": "user", "content": f"Process audio:\n\n{audio_description}"},
        ],
        max_tokens=600,
    )
    elapsed = time.time() - start
    content = response.choices[0].message.content
    tokens = response.usage.total_tokens
    metrics.record_stage("process_audio", tokens, elapsed, True)

    try:
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            data = json.loads(content[start_idx:end_idx])
        else:
            data = {"findings": [content], "key_data": {}, "confidence": 0.6}
    except json.JSONDecodeError:
        data = {"findings": [content], "key_data": {}, "confidence": 0.6}

    return ProcessedModality(
        modality_type="audio",
        extracted_findings=data.get("findings", []),
        key_data_points=data.get("key_data", {}),
        confidence=data.get("confidence", 0.6),
        processing_notes=data.get("transcript_summary", ""),
    )


def stage_process_labs(lab_data: dict, metrics: PipelineMetrics) -> ProcessedModality:
    """Stage 1d: Process structured lab data."""
    start = time.time()
    lab_text = json.dumps(lab_data, indent=2)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Analyze lab results. Identify all abnormal values, critical "
                    "values, and clinically significant patterns. Return JSON: "
                    "{findings: [...], abnormals: {...}, critical: [...], confidence: 0-1}."
                ),
            },
            {"role": "user", "content": f"Analyze labs:\n\n{lab_text}"},
        ],
        max_tokens=600,
    )
    elapsed = time.time() - start
    content = response.choices[0].message.content
    tokens = response.usage.total_tokens
    metrics.record_stage("process_labs", tokens, elapsed, True)

    try:
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            data = json.loads(content[start_idx:end_idx])
        else:
            data = {"findings": [content], "abnormals": {}, "confidence": 0.8}
    except json.JSONDecodeError:
        data = {"findings": [content], "abnormals": {}, "confidence": 0.8}

    return ProcessedModality(
        modality_type="lab_data",
        extracted_findings=data.get("findings", []),
        key_data_points=data.get("abnormals", {}),
        confidence=data.get("confidence", 0.8),
    )


def stage_combine_and_reason(processed: List[ProcessedModality],
                             metrics: PipelineMetrics) -> str:
    """Stage 2: Combine all processed modalities and apply reasoning."""
    # Build combined context
    combined = "MULTIMODAL CLINICAL FINDINGS:\n\n"
    for p in processed:
        combined += f"[{p.modality_type.upper()}] (confidence: {p.confidence:.0%}):\n"
        for finding in p.extracted_findings:
            combined += f"  - {finding}\n"
        if p.key_data_points:
            combined += f"  Key data: {json.dumps(p.key_data_points)}\n"
        combined += "\n"

    combined += (
        "\nSynthesize all findings above into a unified clinical assessment. "
        "Consider how findings from different sources corroborate or contradict "
        "each other. Provide primary diagnosis, differential, critical findings, "
        "and recommended actions."
    )

    start = time.time()
    response = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a senior clinical expert. Analyze these multimodal "
                    "clinical findings and provide a comprehensive assessment.\n\n"
                    + combined
                ),
            },
        ],
        max_completion_tokens=4000,
    )
    elapsed = time.time() - start
    tokens = response.usage.total_tokens
    metrics.record_stage("reasoning", tokens, elapsed, True)

    return response.choices[0].message.content


def stage_structured_output(reasoning_text: str, patient_id: str,
                            modalities: List[str],
                            metrics: PipelineMetrics) -> PipelineOutput:
    """Stage 3: Generate final structured output."""
    start = time.time()
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Convert the clinical reasoning analysis into a structured "
                    "pipeline output. Include all diagnoses, evidence, and actions. "
                    "Set appropriate urgency and confidence levels."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Patient ID: {patient_id}\n"
                    f"Modalities: {', '.join(modalities)}\n\n"
                    f"Analysis:\n{reasoning_text}\n\n"
                    "Generate structured pipeline output."
                ),
            },
        ],
        response_format=PipelineOutput,
    )
    elapsed = time.time() - start
    tokens = completion.usage.total_tokens
    metrics.record_stage("structured_output", tokens, elapsed, True)

    return completion.choices[0].message.parsed


def stage_evaluate(pipeline_output: PipelineOutput,
                   metrics: PipelineMetrics) -> PipelineEvaluation:
    """Stage 4: Evaluate pipeline quality."""
    output_json = json.dumps(pipeline_output.model_dump(), indent=2, default=str)
    metrics_text = metrics.summary()

    start = time.time()
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a clinical AI quality evaluator. Assess the pipeline "
                    "output for clinical accuracy, completeness, safety, and quality. "
                    "Evaluate each pipeline stage and the overall output."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Evaluate this pipeline output:\n\n{output_json}\n\n"
                    f"Pipeline metrics:\n{metrics_text}"
                ),
            },
        ],
        response_format=PipelineEvaluation,
    )
    elapsed = time.time() - start
    tokens = completion.usage.total_tokens
    metrics.record_stage("evaluation", tokens, elapsed, True)

    return completion.choices[0].message.parsed


# ============================================================================
# TEST CASES
# ============================================================================

PIPELINE_CASES = [
    {
        "patient_id": "MP-2025-001",
        "title": "Multimodal Emergency Case",
        "inputs": {
            "text": (
                "68-year-old male with history of atrial fibrillation on warfarin, "
                "COPD, and type 2 diabetes. Presented to ED after a fall at home. "
                "Wife reports he's been confused since yesterday, not eating, and "
                "had a fever. He fell when trying to get to the bathroom and "
                "hit his head on the nightstand. INR was 4.2 at last check (2 weeks ago)."
            ),
            "image": (
                "[Simulated CT Head] Right temporal/parietal subdural hematoma, "
                "crescent-shaped, approximately 12 mm thickness. 6 mm midline shift "
                "to the left. No herniation. Right lateral ventricle compressed. "
                "Left ventricle mildly dilated. Soft tissue swelling overlying "
                "right temporal bone. No skull fracture."
            ),
            "audio": (
                "EMS radio report: 'Medic 7, we have a 68-year-old male, found on "
                "the floor by his wife. GCS is 12, eyes 3, verbal 4, motor 5. "
                "Right pupil is slightly larger than left but both reactive. "
                "BP is 178/96, heart rate 88 irregular, SpO2 is 94 on 2 liters. "
                "He's on Coumadin. Wife says his INR has been hard to control.'"
            ),
            "labs": {
                "INR": "3.8 (CRITICAL, ref 0.8-1.1, therapeutic 2-3)",
                "PT": "42 seconds (CRITICAL)",
                "Hemoglobin": "11.8 g/dL (L)",
                "Platelets": "142 K/uL (L-normal)",
                "Glucose": "245 mg/dL (H)",
                "Creatinine": "1.4 mg/dL (H)",
                "Troponin": "0.03 ng/mL (N)",
                "Lactate": "2.4 mmol/L (H)",
                "Sodium": "148 mEq/L (H)",
                "Potassium": "3.8 mEq/L (N)",
            },
        },
    },
    {
        "patient_id": "MP-2025-002",
        "title": "Multimodal Chronic Disease Assessment",
        "inputs": {
            "text": (
                "52-year-old female presenting for comprehensive evaluation. "
                "History of SLE (diagnosed 10 years ago, on hydroxychloroquine "
                "and low-dose prednisone), type 2 diabetes, and hypertension. "
                "Reports increasing fatigue, joint pain flare-up in hands and "
                "wrists for 2 weeks, new facial rash with sun exposure, and "
                "foamy urine noticed for past month. Also reports blurry vision "
                "in right eye for 1 week."
            ),
            "image": (
                "[Simulated fundoscopy image — right eye] Cotton-wool spots in "
                "the posterior pole, a few dot-blot hemorrhages, mild optic disc "
                "edema. Hard exudates forming a partial macular star pattern. "
                "No neovascularization. Consistent with hypertensive retinopathy "
                "grade III or lupus retinopathy."
            ),
            "audio": (
                "Rheumatology dictation: 'Physical examination reveals a malar "
                "rash across both cheeks and the nose bridge, active synovitis "
                "of bilateral MCP and PIP joints with joint line tenderness, "
                "mild bilateral knee effusions, no oral ulcers today, alopecia "
                "noted at the temples, blood pressure 162/98, trace pedal edema.'"
            ),
            "labs": {
                "CBC": {
                    "WBC": "3.2 K/uL (L)",
                    "Hgb": "10.4 g/dL (L)",
                    "Plt": "118 K/uL (L)",
                },
                "Chemistry": {
                    "Creatinine": "1.6 mg/dL (H, baseline 0.9)",
                    "BUN": "28 (H)",
                    "Albumin": "2.8 g/dL (L)",
                    "Glucose": "195 mg/dL (H)",
                    "HbA1c": "8.4% (H)",
                },
                "Inflammatory": {
                    "ESR": "68 mm/hr (H)", "CRP": "3.2 (H)",
                    "C3": "48 mg/dL (L, ref 90-180)",
                    "C4": "5 mg/dL (L, ref 10-40)",
                },
                "Autoimmune": {
                    "dsDNA": "280 IU/mL (H, rising from 85)",
                    "ANA": "1:1280",
                },
                "Urine": {
                    "Protein_Cr_ratio": "2.8 g/g (H, ref <0.2)",
                    "RBC": "25/hpf with casts",
                },
            },
        },
    },
]


# ============================================================================
# PIPELINE RUNNER
# ============================================================================

def run_pipeline(case: dict) -> tuple:
    """Run the complete multimodal pipeline for a case."""
    metrics = PipelineMetrics()
    inputs = case["inputs"]

    print(f"\n  ┌── STAGE 1: MODALITY PROCESSING ─────────────────────────")
    processed = []

    # Process each available modality
    if "text" in inputs:
        print(f"  │ Processing: text...")
        result = stage_process_text(inputs["text"], metrics)
        processed.append(result)
        print(f"  │   ✓ {len(result.extracted_findings)} findings extracted")

    if "image" in inputs:
        print(f"  │ Processing: image...")
        result = stage_process_image(inputs["image"], metrics)
        processed.append(result)
        print(f"  │   ✓ {len(result.extracted_findings)} findings extracted")

    if "audio" in inputs:
        print(f"  │ Processing: audio...")
        result = stage_process_audio(inputs["audio"], metrics)
        processed.append(result)
        print(f"  │   ✓ {len(result.extracted_findings)} findings extracted")

    if "labs" in inputs:
        print(f"  │ Processing: lab data...")
        result = stage_process_labs(inputs["labs"], metrics)
        processed.append(result)
        print(f"  │   ✓ {len(result.extracted_findings)} findings extracted")

    modalities = [p.modality_type for p in processed]
    print(f"  │ Total modalities: {len(processed)}")

    # Stage 2: Combine and reason
    print(f"\n  ┌── STAGE 2: REASONING ─────────────────────────────────────")
    reasoning = stage_combine_and_reason(processed, metrics)
    print(f"  │ ✓ Reasoning complete ({len(reasoning)} chars)")
    print(f"  │ Preview: {reasoning[:150]}...")

    # Stage 3: Structured output
    print(f"\n  ┌── STAGE 3: STRUCTURED OUTPUT ─────────────────────────────")
    output = stage_structured_output(
        reasoning, case["patient_id"], modalities, metrics
    )
    print(f"  │ ✓ Structured output generated")
    print(f"  │ Primary: {output.primary_diagnosis.name} "
          f"({output.primary_diagnosis.confidence:.0%})")
    print(f"  │ Urgency: {output.urgency}")

    # Stage 4: Evaluation
    print(f"\n  ┌── STAGE 4: QUALITY EVALUATION ────────────────────────────")
    evaluation = stage_evaluate(output, metrics)
    print(f"  │ ✓ Evaluation complete")

    return output, evaluation, metrics


def display_pipeline_results(output: PipelineOutput, evaluation: PipelineEvaluation,
                             metrics: PipelineMetrics, title: str):
    """Display complete pipeline results."""
    print(f"\n  {'═' * 60}")
    print(f"  PIPELINE RESULTS — {title}")
    print(f"  {'═' * 60}")

    # Output summary
    print(f"\n  Patient: {output.patient_id}")
    print(f"  Modalities: {', '.join(output.modalities_processed)}")
    print(f"  Summary: {output.patient_summary[:200]}")

    # Primary diagnosis
    dx = output.primary_diagnosis
    print(f"\n  PRIMARY DIAGNOSIS: {dx.name} ({dx.confidence:.0%})")
    for ev in dx.supporting_evidence[:3]:
        print(f"    ✓ {ev}")
    if dx.evidence_sources:
        print(f"    Sources: {', '.join(dx.evidence_sources)}")

    # Differential
    if output.differential_diagnoses:
        print(f"\n  DIFFERENTIAL:")
        for d in output.differential_diagnoses[:5]:
            sources = f" [{', '.join(d.evidence_sources)}]" if d.evidence_sources else ""
            print(f"    • {d.name} ({d.confidence:.0%}){sources}")

    # Critical findings
    if output.critical_findings:
        print(f"\n  🚨 CRITICAL FINDINGS:")
        for cf in output.critical_findings:
            print(f"    🔴 {cf}")

    # Actions
    print(f"\n  RECOMMENDED ACTIONS:")
    for action in output.recommended_actions:
        print(f"    → {action}")

    urgency_icons = {
        "emergent": "🔴", "urgent": "🟠",
        "semi-urgent": "🟡", "routine": "🟢",
    }
    icon = urgency_icons.get(output.urgency, "⚪")
    print(f"\n  Urgency: {icon} {output.urgency.upper()}")
    print(f"  Overall Confidence: {output.overall_confidence:.0%}")

    # Quality evaluation
    print(f"\n  {'─' * 55}")
    print(f"  QUALITY EVALUATION")
    print(f"  {'─' * 55}")

    for dim, score in [
        ("Clinical Accuracy", evaluation.clinical_accuracy_score),
        ("Completeness", evaluation.completeness_score),
        ("Safety", evaluation.safety_score),
        ("Overall Quality", evaluation.overall_quality_score),
    ]:
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"    {dim:<20} {bar} {score:.0%}")

    if evaluation.stage_evaluations:
        print(f"\n    Stage Scores:")
        for se in evaluation.stage_evaluations:
            sbar = "█" * se.score + "░" * (5 - se.score)
            print(f"      {se.stage_name:<25} {sbar} {se.score}/5")

    if evaluation.key_strengths:
        print(f"\n    Strengths:")
        for s in evaluation.key_strengths[:3]:
            print(f"      ✓ {s}")

    if evaluation.key_weaknesses:
        print(f"\n    Weaknesses:")
        for w in evaluation.key_weaknesses[:3]:
            print(f"      ✗ {w}")

    # Pipeline metrics
    print(f"\n  {'─' * 55}")
    print(f"  PIPELINE METRICS")
    print(f"  {'─' * 55}")
    print(f"  {metrics.summary()}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run end-to-end multimodal pipelines with evaluation."""
    print("=" * 65)
    print("EXERCISE 4: End-to-End Multimodal Pipeline with Evaluation")
    print("=" * 65)
    print("\nThis exercise runs a complete multimodal clinical AI pipeline:")
    print("  Input → Process Modalities → Reason → Structure → Evaluate\n")

    all_metrics = []

    for case in PIPELINE_CASES:
        print(f"\n{'#' * 65}")
        print(f"  CASE: {case['title']} (Patient {case['patient_id']})")
        print(f"{'#' * 65}")

        # Run pipeline
        output, evaluation, metrics = run_pipeline(case)

        # Display results
        display_pipeline_results(output, evaluation, metrics, case["title"])
        all_metrics.append(metrics)

    # Overall summary
    print(f"\n\n{'=' * 65}")
    print("PIPELINE SUMMARY ACROSS ALL CASES")
    print(f"{'=' * 65}")

    total_tokens = sum(m.total_tokens for m in all_metrics)
    total_time = sum(m.total_time for m in all_metrics)
    total_stages = sum(len(m.stages) for m in all_metrics)
    total_errors = sum(len(m.errors) for m in all_metrics)

    print(f"  Cases processed:    {len(PIPELINE_CASES)}")
    print(f"  Total stages run:   {total_stages}")
    print(f"  Total tokens used:  {total_tokens:,}")
    print(f"  Total time:         {total_time:.2f}s")
    print(f"  Avg time per case:  {total_time / len(PIPELINE_CASES):.2f}s")
    print(f"  Errors:             {total_errors}")

    # Cost estimate
    cost_estimate = total_tokens * 10 / 1_000_000  # rough average cost
    print(f"  Est. cost:          ${cost_estimate:.4f}")

    print(f"\n\nKey takeaways:")
    print("  • End-to-end multimodal pipelines synthesize diverse clinical data")
    print("  • Stage-by-stage processing enables modular testing and improvement")
    print("  • Quality evaluation catches issues before clinical deployment")
    print("  • Pipeline metrics help optimize cost and latency")
    print("  • Always include human oversight for clinical AI systems")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
