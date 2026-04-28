"""
agents/agent4_validator.py
Agent 4 — Logic & Resource Validation Agent
"""

from __future__ import annotations

import json
import logging
from typing import Any

from utils.api_client import call_claude
from utils.schema_validator import extract_json, validate_schema, VALIDATION_SCHEMA

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research infrastructure and scientific logic auditor with expertise in:
- Computational reproducibility and open science standards
- Dataset licensing, access controls, and FAIR data principles
- GPU/CPU resource estimation for deep learning workloads
- Common methodological pitfalls: data leakage, selection bias, p-hacking, inappropriate baselines
- Neuroscience-specific issues: circular inference, overfitting to small neural datasets,
  misuse of cross-validation in time-series neural data

Your job is to AUDIT the experiment design for correctness. You are a skeptic, not a cheerleader.

## VALIDATION CHECKLIST:
1. **Internal Consistency**: Do metrics directly test the hypothesis? Are baselines appropriate comparisons?
2. **Dataset Accessibility**: Is the dataset URL real? Is it publicly accessible? What's the license?
3. **Compute Feasibility**: Are GPU hour estimates realistic? (e.g., training a Transformer on 100GB data ≠ 2 hours on one A100)
4. **Methodological Soundness**: 
   - No data leakage between train/val/test
   - Statistical tests appropriate for sample size
   - Baselines are genuine alternatives (not strawmen)
   - Ablations test specific, meaningful components
5. **Reproducibility**: Are random seeds specified? Are dependencies pinnable?

## DECISION:
- VALIDATED: All checks pass — include validated_experiment with the full design
- RETURN: Issues found — provide specific corrections in "corrections" field

## OUTPUT (ONLY JSON, no preamble):
{
  "validation_decision": "VALIDATED | RETURN",
  "dataset_accessible": boolean,
  "compute_feasible": boolean,
  "no_methodological_flaws": boolean,
  "issues": ["string describing each issue found"],
  "corrections": "string with specific corrections (null if VALIDATED)",
  "gate_summary": {
    "dataset_accessible": "✓ | ✗",
    "compute_feasible": "✓ | ✗",
    "no_methodological_flaws": "✓ | ✗"
  },
  "validated_experiment": { /* full experiment design if VALIDATED, null if RETURN */ }
}"""


def run(
    experiment_design: dict,
    approved_proposal: dict,
    init_params: dict,
) -> dict:
    """
    Execute Agent 4: Logic & Resource Validation.

    Parameters
    ----------
    experiment_design : dict
        Output from Agent 3.
    approved_proposal : dict
        The approved proposal (for consistency checks).
    init_params : dict
        Pipeline configuration.

    Returns
    -------
    dict
        Validation result JSON.
    """
    resources = init_params.get("available_resources", {})

    user_message = f"""
## EXPERIMENT DESIGN TO VALIDATE:
{json.dumps(experiment_design, indent=2)}

## ORIGINAL APPROVED PROPOSAL (for consistency checking):
{json.dumps(approved_proposal, indent=2)}

## DECLARED AVAILABLE RESOURCES:
{json.dumps(resources, indent=2)}

## YOUR AUDIT TASKS:

### 1. Internal Consistency Check
- Do the proposed METRICS directly measure what the HYPOTHESIS claims?
- Are the BASELINES genuine scientific alternatives (not trivially weak strawmen)?
- Does the ABLATION PLAN test meaningful components of the proposed model?
- Are the DATA SPLITS appropriate (no temporal leakage for time-series data)?

### 2. Dataset Accessibility Check
- Is the source_url a real, accessible public repository?
- Is the dataset actually suitable for the stated hypothesis?
- Is the license compatible with academic publication?
- Estimate realistic download size and preprocessing time.

### 3. Compute Feasibility Check
- Compare estimated_hours against known benchmarks for similar architectures
- Verify RAM requirements are realistic for the dataset and batch sizes
- Flag if estimates are wildly off (>3x in either direction)

### 4. Methodological Soundness Check
- Check for data leakage opportunities
- Verify statistical tests match data properties (paired vs unpaired, parametric vs non-parametric)
- Assess if sample size supports the statistical tests proposed
- Check for any circular inference or confounds

### 5. Reproducibility Check
- Are all components of the reproducibility checklist sufficient?
- Are there missing critical elements (seeds, hardware pinning, etc.)?

If ALL checks pass: VALIDATED
If ANY check fails: RETURN with specific, actionable corrections

Respond with ONLY the JSON object. No preamble, no markdown fences.
"""

    logger.info("Agent 4: Validating experiment design...")
    raw_response = call_claude(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        max_tokens=3000,
        temperature=0.1,  # Very deterministic for auditing
    )

    logger.debug(f"Agent 4 raw response (first 500 chars): {raw_response[:500]}")

    validation = extract_json(raw_response)

    # Ensure required fields
    if "validated_experiment" not in validation:
        validation["validated_experiment"] = None
    if "corrections" not in validation:
        validation["corrections"] = None

    # If VALIDATED but validated_experiment is null, populate it
    if validation.get("validation_decision") == "VALIDATED" and not validation.get("validated_experiment"):
        validation["validated_experiment"] = experiment_design

    validate_schema(validation, VALIDATION_SCHEMA, "Agent4_Validator")

    decision = validation["validation_decision"]
    ds_ok = validation["dataset_accessible"]
    compute_ok = validation["compute_feasible"]
    methods_ok = validation["no_methodological_flaws"]

    logger.info(
        f"Agent 4: Decision={decision} | "
        f"dataset_accessible={ds_ok} | compute_feasible={compute_ok} | "
        f"no_methodological_flaws={methods_ok}"
    )

    if validation.get("issues"):
        for issue in validation["issues"]:
            logger.warning(f"Agent 4 Issue: {issue}")

    return validation
