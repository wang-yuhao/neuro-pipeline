"""
agents/agent1_proposer.py
Agent 1 — Research Topic Proposer
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from utils.api_client import call_claude
from utils.schema_validator import extract_json, validate_schema, PROPOSAL_SCHEMA

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior computational neuroscience researcher with deep expertise in:
- Spiking neural networks (SNNs) and rate-coded models
- Neural population dynamics and manifold geometry
- Brain-computer interfaces and neural decoding
- Connectomics and synaptic plasticity rules
- Neural coding theory (sparse coding, predictive coding, efficient coding)
- Current open datasets: Allen Brain Atlas, OpenNeuro, CRCNS, NWB ecosystem

Your task is to select the single most promising research topic from the provided candidates and
formulate a rigorous, publication-ready research proposal.

CRITICAL REQUIREMENTS:
1. The hypothesis MUST be falsifiable (testable with quantitative methods).
2. The background MUST cite at least 3 real, specific peer-reviewed works (author, year, title/journal).
3. The proposed methodology must be technically concrete and feasible.
4. You MUST respond with ONLY a valid JSON object matching the schema exactly — no preamble, no markdown, no explanation.

JSON SCHEMA (respond with this exact structure):
{
  "topic_title": "string (precise, ≤15 words)",
  "hypothesis": "string (1-2 sentences, falsifiable)",
  "background": "string (150-200 words citing ≥3 key works)",
  "research_questions": ["string", "string", "string"],
  "proposed_methodology": "string (concrete technical overview, ≥100 words)",
  "expected_contribution": "string (what new knowledge this adds)",
  "keywords": ["string", ...],
  "estimated_timeline_weeks": number,
  "candidate_journals": ["string", ...]
}"""


def run(
    init_params: dict,
    revision_instructions: str | None = None,
    previous_proposal: dict | None = None,
) -> dict:
    """
    Execute Agent 1: Research Topic Proposer.

    Parameters
    ----------
    init_params : dict
        Loaded init_params.json content.
    revision_instructions : str | None
        If this is a revision cycle, instructions from Agent 2.
    previous_proposal : dict | None
        The previous proposal being revised (if any).

    Returns
    -------
    dict
        Validated proposal JSON.
    """
    topics = init_params.get("research_topics", [])
    references = init_params.get("reference_papers", [])
    resources = init_params.get("available_resources", {})
    constraints = init_params.get("constraints", {})
    journals = init_params.get("target_journals", [])

    topics_str = json.dumps(topics, indent=2)
    refs_str = "\n".join(f"  - {r}" for r in references)
    resources_str = json.dumps(resources, indent=2)

    if revision_instructions and previous_proposal:
        mode_block = f"""
## REVISION MODE
You previously submitted a proposal that requires revisions. 

Previous proposal:
{json.dumps(previous_proposal, indent=2)}

Revision instructions from peer reviewer:
{revision_instructions}

Address ALL revision instructions thoroughly while maintaining scientific rigor.
"""
    else:
        mode_block = "## INITIAL PROPOSAL MODE — Select the best topic from candidates."

    user_message = f"""
{mode_block}

## CANDIDATE RESEARCH TOPICS:
{topics_str}

## REFERENCE PAPERS TO BUILD UPON:
{refs_str}

## AVAILABLE COMPUTATIONAL RESOURCES:
{resources_str}

## CONSTRAINTS:
- Maximum timeline: {constraints.get('timeline_weeks_max', 24)} weeks
- Must use open data: {constraints.get('must_use_open_data', True)}
- Preferred framework: {constraints.get('preferred_framework', 'PyTorch')}
- Target journals (preference): {', '.join(journals)}

## YOUR TASK:
Evaluate all candidate topics. Select the ONE with the highest combined score of:
  (a) Scientific novelty relative to 2020-2025 literature
  (b) Technical feasibility given the resources
  (c) Societal/scientific impact potential
  (d) Publication viability for the target journals

Then produce a rigorous research proposal for the selected topic.

IMPORTANT: Respond with ONLY the JSON object. No preamble, no markdown fences, no explanation.
"""

    logger.info("Agent 1: Calling Claude API for research proposal...")
    raw_response = call_claude(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        max_tokens=4000,
        temperature=0.4,
    )

    logger.debug(f"Agent 1 raw response (first 500 chars): {raw_response[:500]}")

    # Parse and validate
    proposal = extract_json(raw_response)
    validate_schema(proposal, PROPOSAL_SCHEMA, "Agent1_Proposer")

    # Gate check: falsifiable hypothesis + ≥3 citations in background
    _gate_check(proposal)

    logger.info(f"Agent 1: Proposal generated — '{proposal['topic_title']}'")
    return proposal


def _gate_check(proposal: dict) -> None:
    """
    Enforce Agent 1 quality gate:
    - Hypothesis must be present and non-trivial
    - Background must contain at least 3 citations (look for year patterns)
    """
    import re

    hyp = proposal.get("hypothesis", "")
    if len(hyp.split()) < 10:
        raise ValueError(f"Gate FAIL: Hypothesis too short or missing: '{hyp}'")

    background = proposal.get("background", "")
    # Look for citation patterns like (Author, 2020) or Author et al., 2021
    citation_patterns = [
        r"\(\w[\w\s]+,\s*\d{4}\)",          # (Author, 2020)
        r"\w[\w\s]+et al\.,?\s*\d{4}",      # Author et al., 2020
        r"\w[\w\s]+\(\d{4}\)",              # Author (2020)
        r"\d{4}[a-z]?[;,\)]",              # bare year in citation context
    ]
    citations_found = set()
    for pat in citation_patterns:
        matches = re.findall(pat, background)
        citations_found.update(matches)

    if len(citations_found) < 2:
        logger.warning(
            f"Agent 1 Gate: Only {len(citations_found)} citation patterns found in background. "
            "Proceeding but reviewer will scrutinize. Consider this a soft warning."
        )

    rqs = proposal.get("research_questions", [])
    if len(rqs) < 3:
        raise ValueError(f"Gate FAIL: Need at least 3 research questions, got {len(rqs)}")

    logger.info("Agent 1: Gate check PASSED ✓")
