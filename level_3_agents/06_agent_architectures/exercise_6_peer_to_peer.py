"""
Exercise 6: Peer-to-Peer Agent Collaboration

Skills practiced:
- Agents that communicate directly without a central supervisor
- Shared context (blackboard) that any agent can read/write
- Negotiation: agents propose, critique, and revise each other's work
- Consensus building through iterative discussion rounds
- Understanding when P2P beats hierarchical

Key insight: In hierarchical systems, the supervisor is the bottleneck
  and single point of failure. In P2P, agents collaborate as equals —
  like a medical team conference where anyone can speak up.

  Hierarchical:  Supervisor → assigns tasks → collects results
  Peer-to-Peer:  Agents ←→ talk to each other ←→ converge on consensus

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │              PEER-TO-PEER COLLABORATION              │
  │                                                     │
  │       ┌──────────┐                                  │
  │       │ Agent A   │◄───────────────┐                │
  │       │(Cardiolog)│                │                │
  │       └───┬──┬────┘                │                │
  │     reads │  │ writes         critiques             │
  │           ▼  ▼                     │                │
  │    ┌──────────────────┐     ┌──────┴─────┐          │
  │    │   SHARED         │     │  Agent C   │          │
  │    │   BLACKBOARD     │◄────│ (Pharmacy) │          │
  │    │                  │     └──────┬─────┘          │
  │    │  - proposals     │            │                │
  │    │  - critiques     │      reads │ writes         │
  │    │  - revisions     │            │                │
  │    │  - votes         │     ┌──────┴─────┐          │
  │    └──────┬───────────┘     │  Agent D   │          │
  │           │                 │ (ED Phys)  │          │
  │     reads │ writes          └────────────┘          │
  │           ▼                                         │
  │       ┌──────────┐                                  │
  │       │ Agent B   │                                 │
  │       │(Hospitali)│                                 │
  │       └──────────┘                                  │
  │                                                     │
  │  Round 1: Each agent proposes (writes to blackboard)│
  │  Round 2: Each agent critiques others' proposals    │
  │  Round 3: Each agent revises based on critiques     │
  │  Final:   Vote on consensus or moderator synthesizes│
  └─────────────────────────────────────────────────────┘

Healthcare parallel: A multidisciplinary team (MDT) conference.
  The cardiologist, hospitalist, pharmacist, and ED physician all
  discuss the case as equals — no one "owns" the conversation.
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

client = OpenAI()


# ============================================================
# Shared Blackboard
# ============================================================

class Blackboard:
    """
    A shared workspace that all agents can read from and write to.

    This is the "blackboard pattern" from AI literature:
    - Any agent can post information
    - Any agent can read all information
    - No central coordinator decides who speaks when
    - The blackboard IS the communication channel

    Think of it as the shared whiteboard in a conference room.
    """

    def __init__(self):
        self.entries = []       # All messages posted
        self.proposals = {}     # Agent name → their proposal
        self.critiques = {}     # Agent name → list of critiques
        self.revisions = {}     # Agent name → revised proposal
        self.votes = {}         # Agent name → vote dict
        self.round_history = [] # Summary of each round

    def post(self, agent_name: str, entry_type: str, content: str):
        """Post a message to the blackboard."""
        entry = {
            "agent": agent_name,
            "type": entry_type,  # "proposal", "critique", "revision", "vote", "comment"
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "index": len(self.entries),
        }
        self.entries.append(entry)

        if entry_type == "proposal":
            self.proposals[agent_name] = content
        elif entry_type == "critique":
            if agent_name not in self.critiques:
                self.critiques[agent_name] = []
            self.critiques[agent_name].append(content)
        elif entry_type == "revision":
            self.revisions[agent_name] = content

    def read_all(self) -> str:
        """Read the entire blackboard as formatted text."""
        output = []
        for entry in self.entries:
            output.append(f"[{entry['agent']}] ({entry['type']}): {entry['content']}")
        return "\n\n".join(output)

    def read_proposals(self) -> str:
        """Read all proposals."""
        parts = []
        for agent, proposal in self.proposals.items():
            parts.append(f"--- {agent}'s Proposal ---\n{proposal}")
        return "\n\n".join(parts)

    def read_critiques_for(self, agent_name: str) -> str:
        """Read all critiques directed at a specific agent."""
        relevant = []
        for entry in self.entries:
            if entry["type"] == "critique" and agent_name in entry["content"]:
                relevant.append(f"[{entry['agent']}]: {entry['content']}")
        return "\n\n".join(relevant) if relevant else "No critiques received."

    def read_all_critiques(self) -> str:
        """Read all critiques posted."""
        parts = []
        for entry in self.entries:
            if entry["type"] == "critique":
                parts.append(f"[{entry['agent']}]: {entry['content']}")
        return "\n\n".join(parts) if parts else "No critiques posted."

    def get_summary(self) -> dict:
        """Get a summary of blackboard activity."""
        return {
            "total_entries": len(self.entries),
            "proposals": len(self.proposals),
            "critiques_posted": sum(len(v) for v in self.critiques.values()),
            "revisions": len(self.revisions),
            "votes": len(self.votes),
            "agents_active": list(set(e["agent"] for e in self.entries)),
        }


# ============================================================
# Peer Agent
# ============================================================

class PeerAgent:
    """
    An agent that participates in peer-to-peer collaboration.
    It can propose, critique, revise, and vote — all through the blackboard.
    No agent is "in charge" — they're equals.
    """

    def __init__(self, name: str, specialty: str, system_prompt: str,
                 model: str = "gpt-4o-mini"):
        self.name = name
        self.specialty = specialty
        self.system_prompt = system_prompt
        self.model = model

    def _call(self, user_prompt: str, context: str = "") -> str:
        """Make an LLM call with the agent's persona."""
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        if context:
            messages.append({"role": "user", "content": f"Current discussion:\n{context}"})
        messages.append({"role": "user", "content": user_prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,  # Slight creativity for diverse perspectives
        )
        return response.choices[0].message.content

    def propose(self, scenario: str, blackboard: Blackboard) -> str:
        """
        Round 1: Read the case and propose your assessment/plan.
        """
        proposal = self._call(
            f"Based on this patient case, provide your assessment and recommendations "
            f"from your perspective as a {self.specialty}.\n\nPatient:\n{scenario}\n\n"
            f"Focus on YOUR specialty's concerns. Be specific with recommendations.",
        )
        blackboard.post(self.name, "proposal", proposal)
        return proposal

    def critique(self, scenario: str, blackboard: Blackboard) -> str:
        """
        Round 2: Read others' proposals and provide constructive critique.
        """
        all_proposals = blackboard.read_proposals()
        critique_text = self._call(
            f"Review your colleagues' proposals for this patient. "
            f"For EACH proposal that isn't yours, provide:\n"
            f"1. What you AGREE with\n"
            f"2. What you DISAGREE with or find MISSING\n"
            f"3. Specific concerns from your {self.specialty} perspective\n"
            f"4. Suggestions for improvement\n\n"
            f"Be constructive but honest. Patient safety is the priority.\n\n"
            f"Patient:\n{scenario}",
            context=all_proposals,
        )
        blackboard.post(self.name, "critique", critique_text)
        return critique_text

    def revise(self, scenario: str, blackboard: Blackboard) -> str:
        """
        Round 3: Read critiques of your proposal and revise it.
        """
        my_proposal = blackboard.proposals.get(self.name, "")
        all_critiques = blackboard.read_all_critiques()

        revision = self._call(
            f"Your original proposal:\n{my_proposal}\n\n"
            f"Critiques from colleagues:\n{all_critiques}\n\n"
            f"Revise your proposal based on valid critiques. "
            f"You may:\n"
            f"- Accept critiques and modify your recommendation\n"
            f"- Respectfully disagree with reasoning (explain why)\n"
            f"- Add new points raised by colleagues\n"
            f"- Flag unresolved disagreements that need the team's attention\n\n"
            f"Patient:\n{scenario}",
        )
        blackboard.post(self.name, "revision", revision)
        return revision

    def vote(self, scenario: str, blackboard: Blackboard) -> dict:
        """
        Final: Vote on the consensus. Rate each aspect.
        """
        all_revisions = "\n\n".join(
            f"--- {name}'s Revised Plan ---\n{rev}"
            for name, rev in blackboard.revisions.items()
        )

        vote_text = self._call(
            f"All revised proposals are in. As {self.name} ({self.specialty}), "
            f"evaluate the overall plan.\n\n"
            f"Provide a JSON vote:\n"
            f'{{"agreement_level": "full|partial|disagree", '
            f'"key_concerns": ["..."], '
            f'"must_change": ["..."], '
            f'"safe_to_proceed": true/false}}\n\n'
            f"Revised proposals:\n{all_revisions}\n\n"
            f"Patient:\n{scenario}",
        )

        # Try to parse as JSON, fall back to text
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[^{}]*\}', vote_text, re.DOTALL)
            if json_match:
                vote_data = json.loads(json_match.group())
            else:
                vote_data = {"raw_vote": vote_text, "agreement_level": "partial"}
        except json.JSONDecodeError:
            vote_data = {"raw_vote": vote_text, "agreement_level": "partial"}

        vote_data["agent"] = self.name
        blackboard.votes[self.name] = vote_data
        blackboard.post(self.name, "vote", json.dumps(vote_data))
        return vote_data


# ============================================================
# P2P Collaboration Orchestrator
# ============================================================

class PeerToPeerCollaboration:
    """
    Orchestrates a peer-to-peer discussion.

    Note: There IS a lightweight orchestrator here, but it ONLY manages
    turn-taking (who speaks when). It does NOT assign tasks, decompose
    work, or make decisions. The agents themselves drive the content.

    Think of it as a meeting facilitator who says "your turn to speak"
    but doesn't tell anyone WHAT to say.
    """

    def __init__(self, agents: list[PeerAgent]):
        self.agents = agents
        self.blackboard = Blackboard()

    def run(self, scenario: str, verbose: bool = True) -> dict:
        """Run a full P2P collaboration session."""
        if verbose:
            print(f"\n  ╔══════════════════════════════════════╗")
            print(f"  ║  PEER-TO-PEER COLLABORATION          ║")
            print(f"  ║  {len(self.agents)} agents, no supervisor              ║")
            print(f"  ╚══════════════════════════════════════╝")

        # ROUND 1: PROPOSALS
        if verbose:
            print(f"\n  📋 ROUND 1: PROPOSALS — Each agent assesses independently")
        for agent in self.agents:
            proposal = agent.propose(scenario, self.blackboard)
            if verbose:
                preview = proposal[:120].replace("\n", " | ")
                print(f"    [{agent.name}]: {preview}...")

        # ROUND 2: CRITIQUES
        if verbose:
            print(f"\n  🔍 ROUND 2: CRITIQUES — Each agent reviews others' proposals")
        for agent in self.agents:
            critique = agent.critique(scenario, self.blackboard)
            if verbose:
                preview = critique[:120].replace("\n", " | ")
                print(f"    [{agent.name}]: {preview}...")

        # ROUND 3: REVISIONS
        if verbose:
            print(f"\n  ✏️ ROUND 3: REVISIONS — Each agent incorporates feedback")
        for agent in self.agents:
            revision = agent.revise(scenario, self.blackboard)
            if verbose:
                preview = revision[:120].replace("\n", " | ")
                print(f"    [{agent.name}]: {preview}...")

        # FINAL: VOTES
        if verbose:
            print(f"\n  🗳️ FINAL: VOTING — Each agent evaluates the consensus")
        votes = []
        for agent in self.agents:
            vote = agent.vote(scenario, self.blackboard)
            votes.append(vote)
            if verbose:
                agreement = vote.get("agreement_level", "unknown")
                safe = vote.get("safe_to_proceed", "unknown")
                print(f"    [{agent.name}]: agreement={agreement}, safe_to_proceed={safe}")

        # Synthesize consensus
        consensus = self._synthesize_consensus(scenario, verbose)

        summary = self.blackboard.get_summary()
        if verbose:
            print(f"\n  Session summary: {summary['total_entries']} messages, "
                  f"{summary['agents_active']} agents active")

        return {
            "consensus": consensus,
            "votes": votes,
            "blackboard_summary": summary,
            "full_blackboard": self.blackboard.read_all(),
        }

    def _synthesize_consensus(self, scenario: str, verbose: bool = True) -> str:
        """
        A neutral moderator (not a supervisor!) synthesizes the final consensus.
        The moderator doesn't add opinions — just summarizes where agents agreed/disagreed.
        """
        all_content = self.blackboard.read_all()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a neutral meeting moderator. You do NOT have medical opinions. "
                    "Your ONLY job is to summarize the discussion:\n\n"
                    "1. CONSENSUS POINTS: What ALL agents agreed on\n"
                    "2. MAJORITY VIEWS: What most (but not all) agents agreed on\n"
                    "3. UNRESOLVED DISAGREEMENTS: Where agents could not agree\n"
                    "4. CRITICAL SAFETY POINTS: Any safety concerns raised by ANY agent\n"
                    "5. RECOMMENDED PLAN: The plan that reflects the group's consensus\n\n"
                    "Do NOT add your own medical opinion. Only reflect what the agents said."
                )},
                {"role": "user", "content": f"Patient:\n{scenario}\n\nFull discussion:\n{all_content}"},
            ],
            temperature=0,
        )

        consensus = response.choices[0].message.content

        if verbose:
            print(f"\n  {'═' * 60}")
            print(f"  CONSENSUS (synthesized by neutral moderator):")
            print(f"  {'═' * 60}")
            for line in consensus.split("\n"):
                print(f"  {line}")

        return consensus


# ============================================================
# Pre-built Agent Teams
# ============================================================

def create_cardiac_team() -> list[PeerAgent]:
    """Create a cardiology-focused MDT."""
    return [
        PeerAgent(
            name="Dr. Chen",
            specialty="Cardiology",
            system_prompt=(
                "You are Dr. Chen, a cardiologist. You focus on cardiac pathology, "
                "risk stratification (HEART, TIMI scores), catheterization decisions, "
                "and cardiac medication management. You advocate for aggressive workup "
                "when cardiac symptoms are present. You believe in early intervention."
            ),
        ),
        PeerAgent(
            name="Dr. Patel",
            specialty="Emergency Medicine",
            system_prompt=(
                "You are Dr. Patel, an emergency medicine physician. You focus on "
                "immediate stabilization, ruling out life-threatening diagnoses, "
                "and disposition decisions. You balance thoroughness with ED throughput. "
                "You're cautious about unnecessary admissions but aggressive about "
                "ruling out dangerous conditions."
            ),
        ),
        PeerAgent(
            name="Dr. Kim",
            specialty="Clinical Pharmacy",
            system_prompt=(
                "You are Dr. Kim, a clinical pharmacist. You focus on drug interactions, "
                "dosing adjustments (renal/hepatic), contraindications, and medication "
                "reconciliation. You flag high-risk medications and advocate for "
                "evidence-based prescribing. You catch things other clinicians miss."
            ),
        ),
        PeerAgent(
            name="Dr. Okafor",
            specialty="Hospital Medicine",
            system_prompt=(
                "You are Dr. Okafor, a hospitalist. You focus on the big picture: "
                "admission criteria, care coordination, comorbidity management, "
                "and discharge planning. You think about the WHOLE patient, not just "
                "one organ system. You push back on unnecessary tests and advocate "
                "for cost-effective care."
            ),
        ),
    ]


def create_complex_case_team() -> list[PeerAgent]:
    """Create a broader MDT for complex cases."""
    return [
        PeerAgent(
            name="Dr. Rivera",
            specialty="Critical Care",
            system_prompt=(
                "You are Dr. Rivera, an intensivist. You focus on immediate stabilization, "
                "airway management, hemodynamic support, and ICU-level interventions. "
                "You think in terms of: what will kill this patient in the next hour?"
            ),
        ),
        PeerAgent(
            name="Dr. Nakamura",
            specialty="Nephrology",
            system_prompt=(
                "You are Dr. Nakamura, a nephrologist. You focus on renal function, "
                "electrolyte management, acid-base disorders, and dialysis decisions. "
                "You advocate for renal-safe approaches and worry about drug clearance."
            ),
        ),
        PeerAgent(
            name="Dr. Williams",
            specialty="Endocrinology",
            system_prompt=(
                "You are Dr. Williams, an endocrinologist. You focus on glycemic control, "
                "insulin management, thyroid, and adrenal issues. You advocate for tight "
                "glucose control while avoiding hypoglycemia."
            ),
        ),
        PeerAgent(
            name="Dr. Kim",
            specialty="Clinical Pharmacy",
            system_prompt=(
                "You are Dr. Kim, a clinical pharmacist. You review all medications for "
                "interactions, dosing errors, and contraindications. In critically ill "
                "patients, you focus on ICU drug protocols and renal dose adjustments."
            ),
        ),
    ]


# ============================================================
# Demo Scenarios
# ============================================================

SCENARIO_ACS = """
Patient: 55-year-old male
Chief Complaint: Chest pain for 2 hours, radiating to left arm
History: Type 2 diabetes (10 years), hypertension, hyperlipidemia, smoker (30 pack-years)
Current Medications: Metformin 1000mg BID, Lisinopril 20mg daily, Atorvastatin 40mg daily
Vitals: BP 158/92, HR 98, SpO2 96%, Temp 37.1°C
Labs: Troponin I 0.45 ng/mL (elevated), Glucose 210 mg/dL, Creatinine 1.3, K+ 4.8
ECG: ST depression in leads V3-V6

Question for the team: What is the optimal management plan?
""".strip()

SCENARIO_COMPLEX = """
Patient: 70-year-old male, found unresponsive by family
History: AFib on apixaban, HF (EF 25%), Type 2 diabetes, CKD stage 4, prior stroke
Medications: Apixaban 5mg BID, Metoprolol 50mg BID, Furosemide 40mg BID,
  Insulin glargine 20u QHS, Lisinopril 5mg daily
Vitals: BP 92/54, HR 48 (irregular), RR 8, SpO2 82%, Temp 35.4°C, GCS 7
Labs: Glucose 42(!), K+ 6.8(!), Creatinine 4.8, Lactate 6.2, pH 7.12, pCO2 68

Multiple simultaneous crises:
  - Hypoglycemia (glucose 42)
  - Hyperkalemia (K+ 6.8)
  - Respiratory failure
  - AKI on CKD
  - Possible new stroke vs metabolic encephalopathy

Question for the team: What are the priorities and how do we manage simultaneously?
""".strip()


# ============================================================
# Demo Functions
# ============================================================

def demo_basic_p2p():
    """Basic P2P collaboration for ACS."""
    print("\n" + "=" * 70)
    print("  DEMO 1: BASIC PEER-TO-PEER — ACS CASE")
    print("=" * 70)
    print("""
  4 physicians discuss an ACS case as equals.
  No one is "in charge" — they propose, critique, revise, and vote.

  Team: Cardiologist, ED Physician, Pharmacist, Hospitalist
  Rounds: Propose → Critique → Revise → Vote
  """)

    team = create_cardiac_team()
    collab = PeerToPeerCollaboration(team)
    result = collab.run(SCENARIO_ACS)

    # Show vote summary
    print(f"\n  Vote Summary:")
    for vote in result["votes"]:
        agent = vote.get("agent", "?")
        agreement = vote.get("agreement_level", "?")
        safe = vote.get("safe_to_proceed", "?")
        print(f"    {agent}: agreement={agreement}, safe={safe}")


def demo_complex_p2p():
    """P2P collaboration for a critically ill patient."""
    print("\n" + "=" * 70)
    print("  DEMO 2: COMPLEX CASE — MULTI-CRISIS PATIENT")
    print("=" * 70)
    print("""
  A critically ill patient with 5 simultaneous problems.
  This is where P2P shines — each specialist brings their expertise
  and the team negotiates priorities together.

  Team: Intensivist, Nephrologist, Endocrinologist, Pharmacist
  """)

    team = create_complex_case_team()
    collab = PeerToPeerCollaboration(team)
    result = collab.run(SCENARIO_COMPLEX)


def demo_p2p_vs_hierarchical():
    """Compare P2P vs hierarchical approach."""
    print("\n" + "=" * 70)
    print("  DEMO 3: P2P vs HIERARCHICAL — COMPARISON")
    print("=" * 70)
    print("""
  Same case, two approaches:
  1. HIERARCHICAL: One supervisor assigns tasks, collects results
  2. PEER-TO-PEER: Agents discuss, critique, and converge

  Watch for differences in:
  - How disagreements are handled
  - Whether minority opinions are preserved
  - Quality of the final consensus
  """)

    # Hierarchical (single agent does everything)
    print(f"\n  ═══ HIERARCHICAL (supervisor-driven) ═══")
    hier_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a supervising physician. Provide a comprehensive assessment "
                "covering: cardiology, emergency medicine, pharmacy, and hospital "
                "medicine perspectives. Integrate all perspectives into one plan."
            )},
            {"role": "user", "content": SCENARIO_ACS},
        ],
        temperature=0,
    )
    hier_result = hier_response.choices[0].message.content
    print(f"  Output: {len(hier_result)} chars (single LLM call)")
    print(f"  Preview: {hier_result[:200].replace(chr(10), ' | ')}...")

    # Peer-to-peer
    print(f"\n\n  ═══ PEER-TO-PEER (agent discussion) ═══")
    team = create_cardiac_team()
    collab = PeerToPeerCollaboration(team)
    p2p_result = collab.run(SCENARIO_ACS, verbose=False)
    consensus = p2p_result["consensus"]
    summary = p2p_result["blackboard_summary"]
    print(f"  Output: {len(consensus)} chars ({summary['total_entries']} messages exchanged)")
    print(f"  Preview: {consensus[:200].replace(chr(10), ' | ')}...")

    # Comparison
    print(f"\n\n  ═══ COMPARISON ═══")
    print(f"  {'Metric':<30} {'Hierarchical':>15} {'Peer-to-Peer':>15}")
    print(f"  {'─' * 60}")
    print(f"  {'Output length':<30} {len(hier_result):>12} ch {len(consensus):>12} ch")
    print(f"  {'LLM calls':<30} {'1':>15} {summary['total_entries']:>15}")
    print(f"  {'Perspectives represented':<30} {'1 (blended)':>15} {len(team):>15}")
    print(f"  {'Disagreements visible':<30} {'No':>15} {'Yes':>15}")


def demo_blackboard_walkthrough():
    """Show all blackboard entries step by step."""
    print("\n" + "=" * 70)
    print("  DEMO 4: BLACKBOARD WALKTHROUGH")
    print("=" * 70)
    print("""
  See exactly what gets written to the shared blackboard at each round.
  This demonstrates how P2P agents communicate through shared state.
  """)

    team = create_cardiac_team()[:2]  # Just 2 agents for brevity
    collab = PeerToPeerCollaboration(team)
    result = collab.run(SCENARIO_ACS, verbose=False)

    print(f"\n  FULL BLACKBOARD TRANSCRIPT:")
    print(f"  {'═' * 60}")
    for entry in collab.blackboard.entries:
        agent = entry["agent"]
        etype = entry["type"].upper()
        content = entry["content"][:200].replace("\n", " | ")
        print(f"\n  [{etype}] {agent}:")
        print(f"  {content}...")
        print(f"  {'─' * 40}")

    print(f"\n  Blackboard stats: {json.dumps(collab.blackboard.get_summary(), indent=4)}")


def demo_interactive():
    """Interactive P2P collaboration."""
    print("\n" + "=" * 70)
    print("  DEMO 5: INTERACTIVE PEER-TO-PEER")
    print("=" * 70)
    print("  Enter a case. The team will discuss it. Type 'quit' to exit.\n")

    while True:
        scenario = input("  Patient case (or 'quit'): ").strip()

        if scenario.lower() in ['quit', 'exit', 'q']:
            break
        if len(scenario) < 30:
            print("  Please provide a detailed case.")
            continue

        team = create_cardiac_team()
        collab = PeerToPeerCollaboration(team)
        result = collab.run(scenario)


# ============================================================
# Main Menu
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("  EXERCISE 6: PEER-TO-PEER AGENT COLLABORATION")
    print("=" * 70)
    print("""
    Agents collaborate as equals through a shared blackboard.
    No supervisor — just propose, critique, revise, and vote.

    Choose a demo:
      1 → Basic P2P (ACS case, 4 agents)
      2 → Complex case (critically ill, 4 specialists)
      3 → P2P vs Hierarchical (comparison)
      4 → Blackboard walkthrough (see all messages)
      5 → Interactive (enter your own case)
      6 → Run demos 1-4
    """)

    choice = input("  Enter choice (1-6): ").strip()

    demos = {
        "1": demo_basic_p2p,
        "2": demo_complex_p2p,
        "3": demo_p2p_vs_hierarchical,
        "4": demo_blackboard_walkthrough,
        "5": demo_interactive,
    }

    if choice == "6":
        for demo in [demo_basic_p2p, demo_complex_p2p,
                      demo_p2p_vs_hierarchical, demo_blackboard_walkthrough]:
            demo()
    elif choice in demos:
        demos[choice]()
    else:
        print("  Invalid choice.")


"""
KEY LEARNINGS:
=============

1. P2P vs HIERARCHICAL: It's not about which is "better."
   - Hierarchical: One supervisor decomposes and assigns. Good when the problem
     is well-understood and decomposition is clear. Fast, cheap.
   - P2P: Agents discuss as equals. Good when multiple perspectives are needed
     and disagreements should be surfaced. Slower, more expensive, richer.

2. THE BLACKBOARD PATTERN: Agents communicate through shared state.
   No direct agent-to-agent messaging. Instead:
   - Agent A writes proposal to blackboard
   - Agent B reads blackboard, writes critique
   - Agent A reads critique, writes revision
   This decouples agents — they don't need to know about each other.

3. NEGOTIATION = PROPOSE → CRITIQUE → REVISE
   This mirrors real medical team conferences:
   - Each specialist gives their assessment (propose)
   - Others push back on things they disagree with (critique)
   - Specialists update their recommendations (revise)
   - The team aligns on a consensus (vote)

4. MINORITY OPINIONS MATTER
   In hierarchical, the supervisor might override a dissenting agent.
   In P2P, every agent's concerns are visible on the blackboard.
   In healthcare, a pharmacist's drug interaction warning shouldn't
   be silently overridden by a supervisor agent.

5. THE MODERATOR IS NOT A SUPERVISOR
   The consensus synthesizer at the end ONLY summarizes — it does not
   add its own opinions or override any agent. It's a reporter, not a boss.

6. WHEN TO USE P2P:
   ✓ Multidisciplinary decisions (need diverse expertise)
   ✓ Ethical dilemmas (need multiple perspectives)
   ✓ Complex cases where no one specialty "owns" the answer
   ✓ When you want disagreements to be visible and documented
   ✗ Simple, well-defined tasks
   ✗ Time-critical decisions (P2P is slow — many LLM calls)
   ✗ When one agent clearly has authority

7. COST: P2P is the most expensive pattern.
   4 agents × 4 rounds = 16+ LLM calls, plus synthesis.
   Use when the decision quality justifies the cost.
"""

if __name__ == "__main__":
    main()
