"""
Exercise 3: Intelligent Model Routing System
=============================================
Build a model routing system that analyzes query complexity and routes
to the cheapest model capable of handling it:
  gpt-4o-mini → gpt-4o → o1-mini

Learning Objectives:
  - Classify query complexity programmatically
  - Route to appropriate model tier based on complexity
  - Optimize cost while maintaining quality
  - Build a practical model selection pipeline

Usage:
  python exercise_3_model_selector.py
"""

import json
import os
import time
from typing import List, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI()


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_TIERS = {
    "gpt-4o-mini": {
        "cost_per_1m_input": 0.15,
        "cost_per_1m_output": 0.60,
        "max_tokens_param": "max_tokens",
        "supports_system": True,
        "description": "Fast, cheap — simple lookups, formatting, classification",
    },
    "gpt-4o": {
        "cost_per_1m_input": 2.50,
        "cost_per_1m_output": 10.00,
        "max_tokens_param": "max_tokens",
        "supports_system": True,
        "description": "Balanced — moderate complexity, good reasoning",
    },
    "o1-mini": {
        "cost_per_1m_input": 3.00,
        "cost_per_1m_output": 12.00,
        "max_tokens_param": "max_completion_tokens",
        "supports_system": False,
        "description": "Reasoning — complex multi-step problems, safety-critical",
    },
}


# ============================================================================
# COMPLEXITY CLASSIFICATION
# ============================================================================

class ComplexityAnalysis(BaseModel):
    """Analysis of query complexity for model routing."""
    complexity_level: Literal["simple", "moderate", "complex"] = Field(
        ..., description="Overall complexity level"
    )
    recommended_model: Literal["gpt-4o-mini", "gpt-4o", "o1-mini"] = Field(
        ..., description="Recommended model tier"
    )
    reasoning: str = Field(..., description="Why this complexity level was assigned")
    factors: List[str] = Field(
        default_factory=list, description="Factors that influenced the classification"
    )
    estimated_tokens: Optional[int] = Field(
        None, description="Estimated output tokens needed"
    )


# Complexity indicators (used for rule-based fallback)
COMPLEXITY_INDICATORS = {
    "simple": [
        "what is", "define", "list", "name", "dose of", "generic name",
        "abbreviation", "normal range", "mechanism of action",
    ],
    "moderate": [
        "compare", "classify", "summarize", "explain", "side effects",
        "drug interaction between two", "treatment for", "first-line",
        "differential for", "interpret these labs",
    ],
    "complex": [
        "multiple comorbidities", "contraindicated", "drug interactions with",
        "considering all", "complex case", "differential diagnosis",
        "treatment plan for patient with", "step by step reasoning",
        "multiple medications", "competing risks", "immunosuppressed",
    ],
}


def classify_complexity_rule_based(query: str) -> str:
    """Rule-based complexity classification (fast fallback)."""
    query_lower = query.lower()

    # Check complex indicators first
    complex_score = sum(1 for ind in COMPLEXITY_INDICATORS["complex"] if ind in query_lower)
    moderate_score = sum(1 for ind in COMPLEXITY_INDICATORS["moderate"] if ind in query_lower)
    simple_score = sum(1 for ind in COMPLEXITY_INDICATORS["simple"] if ind in query_lower)

    # Word count heuristic
    word_count = len(query.split())
    if word_count > 150:
        complex_score += 2
    elif word_count > 50:
        moderate_score += 1

    # Question mark count (more sub-questions = more complex)
    q_count = query.count("?")
    if q_count > 3:
        complex_score += 1
    elif q_count > 1:
        moderate_score += 1

    if complex_score >= 2:
        return "complex"
    elif moderate_score >= 2 or complex_score >= 1:
        return "moderate"
    else:
        return "simple"


def classify_complexity_llm(query: str) -> ComplexityAnalysis:
    """LLM-based complexity classification (more accurate, costs tokens)."""
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",  # Use cheapest model for classification
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a query complexity classifier for a medical AI system. "
                    "Classify the incoming query's complexity to route it to the right model.\n\n"
                    "Classification rules:\n"
                    "- 'simple' → gpt-4o-mini: Simple factual lookups, definitions, "
                    "normal ranges, single drug info, formatting tasks\n"
                    "- 'moderate' → gpt-4o: Comparisons, moderate clinical reasoning, "
                    "treatment for a single condition, lab interpretation, summarization\n"
                    "- 'complex' → o1-mini: Multi-step reasoning, multiple comorbidities, "
                    "complex drug interactions, differential diagnosis with many factors, "
                    "competing clinical risks, safety-critical decisions\n\n"
                    "When in doubt, prefer routing to a more capable model."
                ),
            },
            {"role": "user", "content": f"Classify this query:\n\n{query}"},
        ],
        response_format=ComplexityAnalysis,
    )
    return completion.choices[0].message.parsed


# ============================================================================
# MODEL ROUTER
# ============================================================================

class ModelRouter:
    """Routes queries to appropriate models based on complexity."""

    def __init__(self, use_llm_classifier: bool = True):
        self.use_llm_classifier = use_llm_classifier
        self.history = []

    def classify(self, query: str) -> dict:
        """Classify query and determine model."""
        if self.use_llm_classifier:
            analysis = classify_complexity_llm(query)
            return {
                "complexity": analysis.complexity_level,
                "model": analysis.recommended_model,
                "reasoning": analysis.reasoning,
                "factors": analysis.factors,
            }
        else:
            complexity = classify_complexity_rule_based(query)
            model_map = {
                "simple": "gpt-4o-mini",
                "moderate": "gpt-4o",
                "complex": "o1-mini",
            }
            return {
                "complexity": complexity,
                "model": model_map[complexity],
                "reasoning": "Rule-based classification",
                "factors": [],
            }

    def route_and_call(self, query: str) -> dict:
        """Classify query, route to model, and get response."""
        # Step 1: Classify
        classification = self.classify(query)
        model = classification["model"]
        tier = MODEL_TIERS[model]

        # Step 2: Call the appropriate model
        start = time.time()

        if tier["supports_system"]:
            messages = [
                {
                    "role": "system",
                    "content": "You are a clinical expert. Be accurate and concise.",
                },
                {"role": "user", "content": query},
            ]
            kwargs = {"max_tokens": 1500}
            if tier["supports_system"]:
                kwargs["temperature"] = 0.2
        else:
            # Reasoning model — no system message
            messages = [
                {
                    "role": "user",
                    "content": (
                        "You are a clinical expert. Be accurate and concise.\n\n"
                        + query
                    ),
                },
            ]
            kwargs = {"max_completion_tokens": 3000}

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        elapsed = time.time() - start

        usage = response.usage
        estimated_cost = (
            usage.prompt_tokens * tier["cost_per_1m_input"] / 1_000_000
            + usage.completion_tokens * tier["cost_per_1m_output"] / 1_000_000
        )

        result = {
            "query": query[:80] + "..." if len(query) > 80 else query,
            "classification": classification,
            "model_used": model,
            "response": response.choices[0].message.content,
            "tokens": usage.total_tokens,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "estimated_cost": estimated_cost,
            "elapsed": elapsed,
        }

        self.history.append(result)
        return result

    def get_cost_summary(self) -> dict:
        """Summarize costs across all routed queries."""
        total_cost = sum(r["estimated_cost"] for r in self.history)
        total_tokens = sum(r["tokens"] for r in self.history)
        by_model = {}
        for r in self.history:
            m = r["model_used"]
            if m not in by_model:
                by_model[m] = {"count": 0, "cost": 0, "tokens": 0}
            by_model[m]["count"] += 1
            by_model[m]["cost"] += r["estimated_cost"]
            by_model[m]["tokens"] += r["tokens"]

        return {
            "total_queries": len(self.history),
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "by_model": by_model,
        }


# ============================================================================
# TEST QUERIES
# ============================================================================

TEST_QUERIES = [
    # Simple queries
    "What is the normal range for serum potassium?",
    "What is the generic name for Zocor?",
    "Define 'tachycardia'.",

    # Moderate queries
    "Compare ACE inhibitors and ARBs for a hypertensive patient with diabetes. "
    "Which is preferred and why?",
    "A patient's labs show TSH 8.2 and free T4 0.7. Interpret these results "
    "and suggest next steps.",
    "What is the first-line treatment for community-acquired pneumonia in an "
    "outpatient adult with no comorbidities?",

    # Complex queries
    (
        "A 65-year-old with CKD stage 4 (eGFR 20), heart failure (EF 25%), "
        "and new-onset atrial fibrillation needs anticoagulation. However, "
        "she had a GI bleed 4 months ago and her platelets are 95K. "
        "Design an anticoagulation strategy considering all competing risks."
    ),
    (
        "Patient on tacrolimus (post-kidney transplant), fluconazole (fungal "
        "prophylaxis), and carbamazepine (seizure disorder) presents with "
        "subtherapeutic tacrolimus levels despite dose increases. Analyze all "
        "drug interactions and provide a management plan."
    ),
]


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Demonstrate intelligent model routing."""
    print("=" * 60)
    print("EXERCISE 3: Intelligent Model Routing System")
    print("=" * 60)

    router = ModelRouter(use_llm_classifier=True)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'─' * 60}")
        print(f"  QUERY {i}: {query[:70]}{'...' if len(query) > 70 else ''}")
        print(f"{'─' * 60}")

        result = router.route_and_call(query)
        cls = result["classification"]

        print(f"  Complexity:  {cls['complexity'].upper()}")
        print(f"  Model:       {result['model_used']}")
        print(f"  Reasoning:   {cls['reasoning'][:80]}")
        print(f"  Tokens:      {result['tokens']}")
        print(f"  Cost:        ${result['estimated_cost']:.6f}")
        print(f"  Latency:     {result['elapsed']:.2f}s")
        print(f"\n  Response (preview):")
        print(f"  {result['response'][:200]}...")

    # Cost summary
    summary = router.get_cost_summary()
    print(f"\n\n{'=' * 60}")
    print("ROUTING COST SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total queries: {summary['total_queries']}")
    print(f"  Total tokens:  {summary['total_tokens']}")
    print(f"  Total cost:    ${summary['total_cost']:.6f}")
    print(f"\n  By model:")
    for model, stats in summary["by_model"].items():
        print(f"    {model:<15} {stats['count']} queries, "
              f"{stats['tokens']} tokens, ${stats['cost']:.6f}")

    # Estimate savings vs. always using o1-mini
    all_o1_cost = sum(
        (r["prompt_tokens"] * MODEL_TIERS["o1-mini"]["cost_per_1m_input"]
         + r["completion_tokens"] * MODEL_TIERS["o1-mini"]["cost_per_1m_output"])
        / 1_000_000
        for r in router.history
    )
    savings = all_o1_cost - summary["total_cost"]
    savings_pct = (savings / all_o1_cost * 100) if all_o1_cost > 0 else 0

    print(f"\n  Cost if all queries used o1-mini: ${all_o1_cost:.6f}")
    print(f"  Savings from routing:             ${savings:.6f} ({savings_pct:.1f}%)")

    print(f"\n\nKey takeaways:")
    print("  • Smart routing saves 50-80% on API costs")
    print("  • Simple queries don't need expensive reasoning models")
    print("  • LLM-based classification is more accurate than rules")
    print("  • The classifier itself should use the cheapest model (gpt-4o-mini)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
