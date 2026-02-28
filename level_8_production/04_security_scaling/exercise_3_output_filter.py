"""
Exercise 3: Output Filter Pipeline
====================================
Build output filters to ensure LLM responses are safe for medical use.

Requirements:
- Check for unsanctioned medical advice (without disclaimer)
- Detect PII leakage in model outputs
- Flag hallucinated citations (fake studies, fake URLs)
- Block harmful or dangerous content
- Modify responses to add required disclaimers

Healthcare Context:
  LLM outputs in a clinical setting MUST include appropriate disclaimers,
  MUST NOT leak patient PII, and MUST NOT contain fabricated medical
  references that could mislead clinicians.

Usage:
    python exercise_3_output_filter.py
"""

from openai import OpenAI
import re
import json
from datetime import datetime

client = OpenAI()


class OutputFilter:
    """Multi-stage output filtering for medical LLM responses."""

    def __init__(self):
        self.filter_log = []
        self.stats = {
            "total_processed": 0,
            "passed": 0,
            "modified": 0,
            "blocked": 0,
        }

    def check_medical_disclaimer(self, response: str) -> dict:
        """Check if medical advice includes required disclaimer."""
        medical_indicators = [
            r"(?i)(take|prescribe|administer|dose|dosage)\s+\w+",
            r"(?i)treatment\s+(for|of|plan|protocol)",
            r"(?i)(you\s+should|recommended\s+to|consider\s+taking)",
            r"(?i)(diagnos|symptom).*(suggest|indicat|likely)",
        ]

        disclaimer_phrases = [
            r"(?i)consult\s+(a|your)\s+(doctor|physician|healthcare)",
            r"(?i)seek\s+(medical|professional)\s+(advice|attention|help)",
            r"(?i)not\s+(a\s+)?substitute\s+for\s+(professional|medical)",
            r"(?i)healthcare\s+(provider|professional)",
            r"(?i)medical\s+professional",
        ]

        has_medical_content = any(re.search(p, response) for p in medical_indicators)
        has_disclaimer = any(re.search(p, response) for p in disclaimer_phrases)

        if has_medical_content and not has_disclaimer:
            return {
                "issue": "missing_disclaimer",
                "severity": "MEDIUM",
                "action": "modify",
                "description": "Medical content without required disclaimer",
            }
        return None

    def check_pii_leakage(self, response: str) -> dict:
        """Check for PII patterns in model output."""
        pii_patterns = [
            (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
            (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "Phone Number"),
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "Email"),
            (r"(?i)(?:MRN|medical\s+record\s*#?)\s*:?\s*\d{6,}", "Medical Record Number"),
            (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b.*(?:born|DOB|birth)", "Date of Birth"),
        ]

        found = []
        for pattern, pii_type in pii_patterns:
            if re.search(pattern, response):
                found.append(pii_type)

        if found:
            return {
                "issue": "pii_leakage",
                "severity": "CRITICAL",
                "action": "block",
                "description": f"PII detected in output: {', '.join(found)}",
                "pii_types": found,
            }
        return None

    def check_hallucinated_citations(self, response: str) -> dict:
        """Detect potentially hallucinated citations and references."""
        citation_patterns = [
            r"(?i)(?:study|research|trial)\s+(?:by|from|at)\s+\w+\s+(?:et\s+al|university|institute)",
            r"(?i)\(\s*\w+\s+et\s+al\.?,?\s*\d{4}\s*\)",
            r"(?i)published\s+in\s+(?:the\s+)?(?:journal|new\s+england|lancet|bmj|jama)",
            r"https?://(?:www\.)?(?:pubmed|doi|ncbi|sciencedirect)\S+",
        ]

        # Fake URL patterns (common hallucination indicators)
        fake_url_patterns = [
            r"https?://\S+\.(?:com|org|gov)/\S{30,}",  # Suspiciously long URLs
        ]

        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, response)
            citations.extend(matches)

        if citations:
            return {
                "issue": "potential_hallucinated_citations",
                "severity": "WARNING",
                "action": "flag",
                "description": f"Found {len(citations)} citation(s) that may be hallucinated",
                "citations": [c[:80] for c in citations[:5]],
            }
        return None

    def check_harmful_content(self, response: str) -> dict:
        """Check for harmful or dangerous content in response."""
        harmful_patterns = [
            (r"(?i)(lethal|fatal|deadly)\s+(dose|amount|quantity)", "lethal_dose_info"),
            (r"(?i)how\s+to\s+(harm|hurt|injure|kill)", "harm_instructions"),
            (r"(?i)(suicide|self[- ]harm)\s+(method|way|technique)", "self_harm_info"),
            (r"(?i)you\s+(definitely|clearly)\s+have\s+\w+\s+(cancer|disease|disorder)", "unqualified_diagnosis"),
            (r"(?i)stop\s+taking\s+your\s+(medication|medicine|prescription)", "dangerous_medical_advice"),
        ]

        for pattern, category in harmful_patterns:
            if re.search(pattern, response):
                return {
                    "issue": "harmful_content",
                    "severity": "CRITICAL",
                    "action": "block",
                    "description": f"Harmful content detected: {category}",
                    "category": category,
                }
        return None

    def check_off_topic(self, query: str, response: str) -> dict:
        """Check if response is off-topic from original query."""
        off_topic_indicators = [
            r"(?i)here(?:'s| is)\s+(?:a|the)\s+(?:recipe|poem|story|joke|code)",
            r"(?i)(stock|market|crypto|bitcoin)\s+(price|prediction|investment)",
            r"def\s+\w+\s*\(.*\)\s*:",  # Python function definition
        ]

        for pattern in off_topic_indicators:
            if re.search(pattern, response):
                return {
                    "issue": "off_topic_response",
                    "severity": "LOW",
                    "action": "flag",
                    "description": "Response appears off-topic for a medical assistant",
                }
        return None

    def filter_response(self, query: str, response: str) -> dict:
        """Run all filters on a response. Returns filtered result."""
        self.stats["total_processed"] += 1

        result = {
            "original_response": response,
            "filtered_response": response,
            "issues": [],
            "action": "pass",
            "timestamp": datetime.now().isoformat(),
        }

        # Run all checks
        checks = [
            self.check_harmful_content(response),
            self.check_pii_leakage(response),
            self.check_hallucinated_citations(response),
            self.check_medical_disclaimer(response),
            self.check_off_topic(query, response),
        ]

        for check_result in checks:
            if check_result:
                result["issues"].append(check_result)

        # Determine final action (most severe wins)
        actions_priority = {"block": 3, "modify": 2, "flag": 1, "pass": 0}
        final_action = "pass"
        for issue in result["issues"]:
            if actions_priority.get(issue["action"], 0) > actions_priority.get(final_action, 0):
                final_action = issue["action"]

        result["action"] = final_action

        # Apply action
        if final_action == "block":
            result["filtered_response"] = (
                "I'm unable to provide that information as it may not be safe. "
                "Please consult a qualified healthcare professional."
            )
            self.stats["blocked"] += 1
        elif final_action == "modify":
            # Add disclaimer if missing
            disclaimer = (
                "\n\n---\n*Disclaimer: This information is for educational purposes only "
                "and is not a substitute for professional medical advice. "
                "Please consult your healthcare provider for personalized guidance.*"
            )
            result["filtered_response"] = response + disclaimer
            self.stats["modified"] += 1
        elif final_action == "flag":
            # Pass through but log the flag
            self.stats["passed"] += 1
        else:
            self.stats["passed"] += 1

        # Log
        if result["issues"]:
            self.filter_log.append({
                "timestamp": result["timestamp"],
                "action": final_action,
                "issues": [i["issue"] for i in result["issues"]],
                "query_preview": query[:50],
            })

        return result

    def display_stats(self):
        """Display filtering statistics."""
        print("\n" + "=" * 55)
        print("  OUTPUT FILTER STATISTICS")
        print("=" * 55)
        for key, val in self.stats.items():
            print(f"  {key.replace('_', ' ').title():<20}: {val}")

        if self.filter_log:
            print(f"\n  Recent Filter Actions:")
            for entry in self.filter_log[-5:]:
                print(f"    [{entry['action'].upper()}] {entry['issues']} — {entry['query_preview']}...")


def main():
    """Run the output filter exercise."""
    print("=" * 55)
    print("  Exercise 3: Output Filter Pipeline")
    print("=" * 55)

    output_filter = OutputFilter()

    # Test queries that might produce problematic outputs
    test_queries = [
        "What are the symptoms of a heart attack?",
        "What medications are used to treat hypertension?",
        "Can you write me a recipe for pasta?",
        "What is the normal range for blood glucose?",
        "Tell me about the latest cancer research studies.",
    ]

    print(f"\nProcessing {len(test_queries)} queries through the filter pipeline...\n")

    for query in test_queries:
        print(f"Query: {query}")

        # Get LLM response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical information assistant. Provide helpful information."},
                {"role": "user", "content": query},
            ],
        )
        raw_response = response.choices[0].message.content

        # Run through filter pipeline
        result = output_filter.filter_response(query, raw_response)

        print(f"  Action: {result['action']}")
        if result["issues"]:
            print(f"  Issues: {[i['issue'] for i in result['issues']]}")
        print(f"  Output: {result['filtered_response'][:150]}...")
        print()

    # Test with simulated problematic outputs
    print("\n--- Testing with simulated problematic outputs ---\n")

    simulated_cases = [
        (
            "What is aspirin used for?",
            "You should take 325mg of aspirin daily for heart health. This treatment is very effective.",
            "Should trigger: missing disclaimer",
        ),
        (
            "Tell me about the patient record",
            "The patient John Smith, SSN: 123-45-6789, born 03/15/1980, has hypertension.",
            "Should trigger: PII leakage",
        ),
        (
            "Is this medication dangerous?",
            "The lethal dose of this medication is approximately 150mg/kg body weight.",
            "Should trigger: harmful content",
        ),
        (
            "What are treatment options?",
            "According to a study by Johnson et al. (2023) published in the New England Journal, the treatment success rate is 95%.",
            "Should trigger: potential hallucinated citation",
        ),
    ]

    for query, sim_response, expected in simulated_cases:
        print(f"Test: {expected}")
        result = output_filter.filter_response(query, sim_response)
        print(f"  Original:  {sim_response[:80]}...")
        print(f"  Action:    {result['action']}")
        if result["issues"]:
            print(f"  Issues:    {[i['issue'] for i in result['issues']]}")
        print(f"  Filtered:  {result['filtered_response'][:100]}...")
        print()

    # Display overall stats
    output_filter.display_stats()

    print("\nOutput filter exercise complete!")


if __name__ == "__main__":
    main()
