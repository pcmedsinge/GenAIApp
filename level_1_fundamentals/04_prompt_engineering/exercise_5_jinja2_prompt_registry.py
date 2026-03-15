"""
Exercise 5: Jinja2 Prompt Registry + Versioning + Rollback

Skills practiced:
- Rendering prompts with Jinja2 templates
- Managing prompt versions in a registry
- Promoting / rolling back active prompt versions
- Running quick side-by-side checks across versions

Why this matters:
  In production, prompts should be managed like code:
  - versioned
  - testable
  - auditable
  - reversible

  This exercise gives a practical workflow for that lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


try:
    from jinja2 import Environment, StrictUndefined
except Exception:
    Environment = None
    StrictUndefined = None


# ============================================================
# Data Models
# ============================================================

@dataclass
class PromptVersion:
    prompt_name: str
    version: str
    template: str
    created_at: str
    author: str
    notes: str = ""


@dataclass
class PromptRegistry:
    """In-memory prompt registry with active-version tracking."""

    versions: dict[str, list[PromptVersion]] = field(default_factory=dict)
    active: dict[str, str] = field(default_factory=dict)

    def add_version(
        self,
        prompt_name: str,
        version: str,
        template: str,
        author: str,
        notes: str = "",
    ) -> PromptVersion:
        pv = PromptVersion(
            prompt_name=prompt_name,
            version=version,
            template=template,
            created_at=datetime.now().isoformat(timespec="seconds"),
            author=author,
            notes=notes,
        )
        self.versions.setdefault(prompt_name, []).append(pv)

        if prompt_name not in self.active:
            self.active[prompt_name] = version

        return pv

    def list_versions(self, prompt_name: str) -> list[PromptVersion]:
        return self.versions.get(prompt_name, [])

    def set_active(self, prompt_name: str, version: str) -> bool:
        if not any(v.version == version for v in self.list_versions(prompt_name)):
            return False
        self.active[prompt_name] = version
        return True

    def get_active(self, prompt_name: str) -> PromptVersion | None:
        active_version = self.active.get(prompt_name)
        if not active_version:
            return None
        for v in self.list_versions(prompt_name):
            if v.version == active_version:
                return v
        return None

    def rollback(self, prompt_name: str, target_version: str) -> bool:
        return self.set_active(prompt_name, target_version)


# ============================================================
# Jinja2 Renderer
# ============================================================

class PromptRenderer:
    def __init__(self):
        if Environment is None:
            raise RuntimeError(
                "Jinja2 is not installed. Install with: pip install jinja2"
            )
        self.env = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)

    def render(self, template_text: str, variables: dict[str, Any]) -> str:
        template = self.env.from_string(template_text)
        return template.render(**variables)


# ============================================================
# Seed Data
# ============================================================

CLINICAL_SUMMARY_V1 = """
You are a clinical summarization assistant.

Summarize this encounter in {{ style }} style.
Maximum {{ max_words }} words.

Patient Context:
- Age: {{ age }}
- Symptoms: {{ symptoms }}
- Current medications: {{ medications | join(', ') }}

Output sections:
1. Key Findings
2. Assessment
3. Plan
""".strip()


CLINICAL_SUMMARY_V2 = """
You are a clinical summarization assistant.

Task:
Create a concise, clinically safe summary in {{ style }} style.
Limit to {{ max_words }} words.

Patient Context:
- Age: {{ age }}
- Symptoms: {{ symptoms }}
- Medications: {{ medications | join(', ') }}
- Comorbidities: {{ comorbidities | join(', ') }}

Requirements:
- Mention red flags if present
- Do not invent missing facts
- Keep wording suitable for handoff to another clinician

Output sections:
1) Key Findings
2) Assessment
3) Plan
4) Safety Notes
""".strip()


CLINICAL_SUMMARY_V3 = """
You are a clinical summarization assistant for emergency medicine.

Task:
Generate a {{ style }} handoff summary in <= {{ max_words }} words.

Patient Context:
- Age: {{ age }}
- Symptoms: {{ symptoms }}
- Medications: {{ medications | join(', ') }}
- Comorbidities: {{ comorbidities | join(', ') }}
- Allergies: {{ allergies | join(', ') }}

Hard constraints:
- Explicitly list uncertainty where data is incomplete
- Include medication/allergy safety risks
- Include immediate next action
- Never fabricate lab/imaging findings not provided

Output sections:
1) Key Findings
2) Assessment
3) Plan
4) Safety Notes
5) Unknowns / Follow-up Needed
""".strip()


# ============================================================
# Demo Utilities
# ============================================================

def seed_registry() -> PromptRegistry:
    registry = PromptRegistry()
    registry.add_version(
        prompt_name="clinical_summary",
        version="v1.0.0",
        template=CLINICAL_SUMMARY_V1,
        author="team-foundation",
        notes="Initial simple template",
    )
    registry.add_version(
        prompt_name="clinical_summary",
        version="v1.1.0",
        template=CLINICAL_SUMMARY_V2,
        author="team-foundation",
        notes="Added safety + comorbidities section",
    )
    registry.add_version(
        prompt_name="clinical_summary",
        version="v1.2.0",
        template=CLINICAL_SUMMARY_V3,
        author="team-safety",
        notes="Added allergies + uncertainty constraints",
    )
    registry.set_active("clinical_summary", "v1.2.0")
    return registry


def sample_variables() -> dict[str, Any]:
    return {
        "style": "SOAP-like",
        "max_words": 180,
        "age": 67,
        "symptoms": "chest tightness, dyspnea on exertion, mild diaphoresis",
        "medications": ["metformin", "lisinopril", "atorvastatin"],
        "comorbidities": ["type 2 diabetes", "hypertension", "hyperlipidemia"],
        "allergies": ["penicillin"],
    }


# ============================================================
# Demos
# ============================================================

def demo_list_versions(registry: PromptRegistry):
    print("\n" + "=" * 70)
    print("  DEMO 1: LIST PROMPT VERSIONS")
    print("=" * 70)

    prompt_name = "clinical_summary"
    active = registry.active.get(prompt_name)

    for v in registry.list_versions(prompt_name):
        marker = "(ACTIVE)" if v.version == active else ""
        print(
            f"  - {v.version:8s} {marker:9s} | by {v.author:15s} "
            f"| {v.created_at} | {v.notes}"
        )


def demo_render_active(registry: PromptRegistry, renderer: PromptRenderer):
    print("\n" + "=" * 70)
    print("  DEMO 2: RENDER ACTIVE VERSION")
    print("=" * 70)

    active = registry.get_active("clinical_summary")
    if not active:
        print("  No active version found.")
        return

    rendered = renderer.render(active.template, sample_variables())
    print(f"  Active version: {active.version}\n")
    print(rendered)


def demo_compare_versions(registry: PromptRegistry, renderer: PromptRenderer):
    print("\n" + "=" * 70)
    print("  DEMO 3: SIDE-BY-SIDE VERSION RENDER")
    print("=" * 70)

    variables = sample_variables()
    versions = ["v1.0.0", "v1.2.0"]

    for version in versions:
        candidate = next(
            (v for v in registry.list_versions("clinical_summary") if v.version == version),
            None,
        )
        if not candidate:
            continue

        rendered = renderer.render(candidate.template, variables)
        print("\n" + "-" * 70)
        print(f"  VERSION: {version}")
        print("-" * 70)
        print(rendered)

    print("\n  Observation:")
    print("  - v1.2.0 has stronger safety/uncertainty constraints than v1.0.0")


def demo_rollback(registry: PromptRegistry, renderer: PromptRenderer):
    print("\n" + "=" * 70)
    print("  DEMO 4: ROLLBACK ACTIVE VERSION")
    print("=" * 70)

    before = registry.active.get("clinical_summary")
    ok = registry.rollback("clinical_summary", "v1.1.0")
    after = registry.active.get("clinical_summary")

    print(f"  Before: {before}")
    print(f"  Rollback success: {ok}")
    print(f"  After:  {after}")

    active = registry.get_active("clinical_summary")
    if active:
        print("\n  Rendered prompt after rollback:\n")
        print(renderer.render(active.template, sample_variables()))


def demo_add_new_version(registry: PromptRegistry, renderer: PromptRenderer):
    print("\n" + "=" * 70)
    print("  DEMO 5: REGISTER NEW VERSION + PROMOTE")
    print("=" * 70)

    new_template = """
You are a discharge counseling assistant.

Create a {{ style }} patient-facing summary (<= {{ max_words }} words).

Context:
- Symptoms: {{ symptoms }}
- Medications: {{ medications | join(', ') }}
- Allergies: {{ allergies | join(', ') }}

Must include:
1) What happened
2) What to do at home
3) Red flags for ER return
4) Follow-up reminder
""".strip()

    registry.add_version(
        prompt_name="clinical_summary",
        version="v1.3.0",
        template=new_template,
        author="team-patient-ed",
        notes="Patient-facing discharge orientation",
    )
    registry.set_active("clinical_summary", "v1.3.0")

    active = registry.get_active("clinical_summary")
    print(f"  New active version: {active.version if active else 'N/A'}")
    print("\n  Rendered:\n")
    print(renderer.render(active.template, sample_variables()))


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 5: JINJA2 PROMPT REGISTRY + VERSIONING")
    print("=" * 70)

    if Environment is None:
        print("\n  Jinja2 is not installed.")
        print("  Install it first: pip install jinja2")
        return

    registry = seed_registry()
    renderer = PromptRenderer()

    print("""
    Choose:
      1 → List versions (registry view)
      2 → Render active template
      3 → Compare v1.0.0 vs v1.2.0
      4 → Rollback active version (to v1.1.0)
      5 → Add new version and promote
      6 → Run all demos
    """)

    choice = input("  Enter choice (1-6): ").strip()
    demos = {
        "1": lambda: demo_list_versions(registry),
        "2": lambda: demo_render_active(registry, renderer),
        "3": lambda: demo_compare_versions(registry, renderer),
        "4": lambda: demo_rollback(registry, renderer),
        "5": lambda: demo_add_new_version(registry, renderer),
    }

    if choice == "6":
        demo_list_versions(registry)
        demo_render_active(registry, renderer)
        demo_compare_versions(registry, renderer)
        demo_rollback(registry, renderer)
        demo_add_new_version(registry, renderer)
    elif choice in demos:
        demos[choice]()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS
=============

1) Jinja2 turns prompts into maintainable templates with variables and filters.
2) Registry gives a single place to manage prompt history and active versions.
3) Version promotion + rollback enables safe production operations.
4) Prompt changes should be testable and reversible, not ad-hoc edits.
5) Treat prompts like code: version, review, evaluate, deploy, rollback.
"""


if __name__ == "__main__":
    main()
