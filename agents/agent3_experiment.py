"""
agents/agent3_experiment.py
Agent 3 — Experiment Design Agent
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from utils.api_client import call_claude
from utils.schema_validator import extract_json, validate_schema, EXPERIMENT_SCHEMA

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a computational neuroscience experimental design specialist with 15+ years
of experience designing reproducible ML/neuroscience experiments. You have deep expertise in:

- PyTorch and JAX-based neural network implementations
- Neuroscience datasets: Allen Brain Atlas, OpenNeuro (BIDS format), CRCNS, NWB ecosystem,
  Human Connectome Project, UK Biobank, DANDI Archive
- Statistical methodology: paired t-tests, Wilcoxon signed-rank, effect sizes (Cohen's d),
  bootstrap confidence intervals, permutation tests, multiple comparison correction (FDR/Bonferroni)
- Ablation study design and systematic hyperparameter optimization
- Reproducibility best practices: seed control, version pinning, Docker containerization

Your designs must be:
1. COMPLETE: Every step specified with enough detail for independent replication
2. REALISTIC: Resource estimates based on actual benchmark performance
3. RIGOROUS: Statistical tests appropriate for the data type and sample size
4. REPRODUCIBLE: Full seed control, hardware specification, and dependency versioning

You MUST respond with ONLY a valid JSON object. No preamble, no markdown, no explanation.

The experiment_id field MUST be a valid UUID (version 4) string.

JSON SCHEMA:
{
  "experiment_id": "UUID string",
  "dataset": {
    "name": "string",
    "source_url": "string",
    "license": "string",
    "preprocessing_steps": ["string", ...],
    "splits": {"train": 0.7, "val": 0.15, "test": 0.15}
  },
  "models": [
    {"name": "string", "architecture": "string (detailed)", "justification": "string", "hyperparameters": {}}
  ],
  "baselines": [
    {"name": "string", "reference": "string (Author, Year)", "implementation": "string"}
  ],
  "metrics": ["string", ...],
  "statistical_tests": ["string", ...],
  "ablation_plan": ["string (specific component to ablate)", ...],
  "compute_spec": {
    "gpu": "string",
    "ram_gb": number,
    "estimated_hours": number,
    "estimated_cost_usd": number
  },
  "risks": [{"risk": "string", "likelihood": "HIGH|MEDIUM|LOW", "mitigation": "string"}],
  "reproducibility_checklist": ["string", ...]
}"""


def run(
    approved_proposal: dict,
    init_params: dict,
    corrections: str | None = None,
    previous_design: dict | None = None,
) -> dict:
    """
    Execute Agent 3: Experiment Design.

    Parameters
    ----------
    approved_proposal : dict
        The approved proposal from Agent 1/2.
    init_params : dict
        Pipeline configuration.
    corrections : str | None
        Corrections from Agent 4 (if revision).
    previous_design : dict | None
        Previous experiment design (if revision).

    Returns
    -------
    dict
        Validated experiment design JSON.
    """
    resources = init_params.get("available_resources", {})
    constraints = init_params.get("constraints", {})

    if corrections and previous_design:
        mode_block = f"""
## REVISION MODE — Apply these corrections from the validation agent:
{corrections}

## PREVIOUS DESIGN (revise this):
{json.dumps(previous_design, indent=2)}
"""
    else:
        mode_block = "## INITIAL DESIGN MODE"

    user_message = f"""
{mode_block}

## APPROVED RESEARCH PROPOSAL:
{json.dumps(approved_proposal, indent=2)}

## AVAILABLE RESOURCES:
{json.dumps(resources, indent=2)}

## CONSTRAINTS:
- Framework: {constraints.get('preferred_framework', 'PyTorch')}
- Must use open data: {constraints.get('must_use_open_data', True)}
- Language: {constraints.get('language', 'Python 3.10+')}

## YOUR TASK:
Design a COMPLETE, REPRODUCIBLE experimental framework. Requirements:
1. Dataset: Specify a REAL, publicly accessible dataset matching the hypothesis. Include exact URL.
2. Models: Minimum 1 proposed model with full architecture details (layer sizes, activations, etc.)
3. Baselines: Minimum 2 ESTABLISHED baselines from literature (with citations)
4. Metrics: All metrics must directly evaluate the hypothesis
5. Statistical tests: Choose tests appropriate for the data distribution and sample size
6. Ablation plan: At least 4 specific components to ablate
7. Compute: Realistic estimate for the stated GPU/RAM
8. Risks: Identify top 3-5 failure modes with concrete mitigations
9. Reproducibility: Comprehensive checklist including seeds, versions, hardware

Generate a UUID for experiment_id (format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx).

Respond with ONLY the JSON object. No preamble, no markdown fences, no explanation.
"""

    logger.info("Agent 3: Designing experiment...")
    raw_response = call_claude(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        max_tokens=5000,
        temperature=0.3,
    )

    logger.debug(f"Agent 3 raw response (first 500 chars): {raw_response[:500]}")

    design = extract_json(raw_response)

    # Ensure valid UUID
    if not design.get("experiment_id"):
        design["experiment_id"] = str(uuid.uuid4())
    else:
        # Try to parse/normalize the UUID
        try:
            design["experiment_id"] = str(uuid.UUID(design["experiment_id"]))
        except ValueError:
            design["experiment_id"] = str(uuid.uuid4())
            logger.warning("Agent 3: Invalid UUID in response, generated a new one.")

    validate_schema(design, EXPERIMENT_SCHEMA, "Agent3_ExperimentDesign")

    # Gate: must have ≥2 baselines
    if len(design.get("baselines", [])) < 2:
        raise ValueError("Agent 3 Gate FAIL: Must specify at least 2 baselines.")

    logger.info(f"Agent 3: Experiment design complete — ID={design['experiment_id']}")
    return design
