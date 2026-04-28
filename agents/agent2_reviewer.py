"""
agents/agent2_reviewer.py
Agent 2 — Feasibility & Peer Review Agent
"""

from __future__ import annotations

import json
import logging
from typing import Any

from utils.api_client import call_claude
from utils.schema_validator import extract_json, validate_schema, FEASIBILITY_SCHEMA

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior computational neuroscience reviewer with 20+ years of experience,
serving on editorial boards of Nature Neuroscience, PLOS Computational Biology, and NeurIPS.

Your role is to provide a rigorous, honest peer review of a computational neuroscience research proposal.
You are NOT here to be encouraging — you are here to ensure scientific rigor and feasibility.

## REVIEW DIMENSIONS (score each 0–10):
1. **Scientific Novelty (0-10)**: Is the hypothesis truly novel? Does it go beyond incremental improvement?
   - 0-3: Clearly derivative, already published, or trivially incremental
   - 4-6: Modest novelty, builds on existing work without major new insight  
   - 7-9: Genuinely novel angle, addresses real gap in literature
   - 10: Groundbreaking, would reshape the field

2. **Technical Feasibility (0-10)**: Can a single researcher actually execute this with the stated resources?
   - 0-3: Methods don't exist, or would require 5+ years to implement
   - 4-6: Feasible but highly challenging; significant implementation risk
   - 7-9: Well-established methods, achievable with stated resources
   - 10: Trivially feasible with existing tools

3. **Data Availability (0-10)**: Are suitable, open-access datasets accessible?
   - 0-3: No suitable public datasets exist
   - 4-6: Data exists but has restrictive access or significant preprocessing burden
   - 7-9: High-quality public datasets readily available (Allen, OpenNeuro, CRCNS, etc.)
   - 10: Data is pre-curated and immediately usable

4. **Computational Requirements (0-10)**: Are resource requirements matched to available compute?
   - 0-3: Would require supercomputer cluster or >$100k compute budget
   - 4-6: Feasible but resource-intensive; needs careful optimization
   - 7-9: Manageable within typical academic compute budget
   - 10: Runs on a laptop

5. **Publication Viability (0-10)**: Is the scope and impact appropriate for publication?
   - 0-3: Too narrow/broad, or findings would not meet journal standards
   - 4-6: Publishable in minor venues if all goes well
   - 7-9: Strong candidate for target journals with solid execution
   - 10: Near-certain publication at top venues

## DECISION RULES:
- overall_score = mean of all 5 dimension scores
- overall_score ≥ 7.0 → APPROVED
- 5.0 ≤ overall_score < 7.0 → REVISE (provide specific, actionable revision instructions)
- overall_score < 5.0 → REJECTED (explain why and propose a better direction)

## OUTPUT FORMAT:
Respond with ONLY a valid JSON object. No preamble, no markdown, no explanation.

{
  "decision": "APPROVED | REVISE | REJECTED",
  "scores": {
    "novelty": number,
    "technical_feasibility": number,
    "data_availability": number,
    "computational_requirements": number,
    "publication_viability": number
  },
  "overall_score": number,
  "strengths": ["string", ...],
  "weaknesses": ["string", ...],
  "revision_instructions": "string | null",
  "approved_proposal": { /* include the full proposal object verbatim if APPROVED, else null */ }
}"""


def run(
    proposal: dict,
    init_params: dict,
    revision_cycle: int = 0,
) -> dict:
    """
    Execute Agent 2: Feasibility & Peer Review.

    Parameters
    ----------
    proposal : dict
        Output from Agent 1.
    init_params : dict
        Pipeline configuration for resource context.
    revision_cycle : int
        Which revision cycle this is (0 = first review).

    Returns
    -------
    dict
        Feasibility review JSON with decision.
    """
    resources = init_params.get("available_resources", {})

    user_message = f"""
## RESEARCH PROPOSAL FOR REVIEW (Revision cycle: {revision_cycle}):
{json.dumps(proposal, indent=2)}

## AVAILABLE COMPUTATIONAL RESOURCES (for feasibility assessment):
{json.dumps(resources, indent=2)}

## YOUR TASK:
Provide a rigorous peer review of this proposal. Be specific and critical.
For each weakness, explain the exact problem and why it matters scientifically.
If REVISE: revision_instructions must be specific enough that Agent 1 can act on them.
If APPROVED: include the full proposal in approved_proposal field verbatim.
If REJECTED: revision_instructions should explain the fundamental problems.

Evaluate whether the cited works in "background" are real and relevant.
Check whether the hypothesis is truly falsifiable with the proposed methods.

Respond with ONLY the JSON object. No preamble, no markdown fences.
"""

    logger.info(f"Agent 2: Reviewing proposal (revision_cycle={revision_cycle})...")
    raw_response = call_claude(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        max_tokens=3000,
        temperature=0.2,  # Low temp for consistent scoring
    )

    logger.debug(f"Agent 2 raw response (first 500 chars): {raw_response[:500]}")

    review = extract_json(raw_response)

    # Ensure approved_proposal is populated when APPROVED
    if review.get("decision") == "APPROVED" and not review.get("approved_proposal"):
        review["approved_proposal"] = proposal

    # Ensure revision_instructions is present
    if "revision_instructions" not in review:
        review["revision_instructions"] = None

    validate_schema(review, FEASIBILITY_SCHEMA, "Agent2_Reviewer")

    # Recompute overall_score as sanity check
    scores = review.get("scores", {})
    if scores:
        computed = sum(scores.values()) / len(scores)
        stated = review.get("overall_score", 0)
        if abs(computed - stated) > 1.0:
            logger.warning(
                f"Agent 2: overall_score mismatch — stated={stated:.2f}, computed={computed:.2f}. "
                "Using computed value."
            )
            review["overall_score"] = round(computed, 2)

    decision = review["decision"]
    score = review["overall_score"]
    logger.info(f"Agent 2: Decision={decision}, overall_score={score:.2f}")

    return review
