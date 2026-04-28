"""
utils/schema_validator.py
JSON schema definitions and validation for all agent handoffs.
"""

from __future__ import annotations

import json
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definitions (JSON Schema Draft-7 compatible)
# ---------------------------------------------------------------------------

PROPOSAL_SCHEMA = {
    "type": "object",
    "required": [
        "topic_title", "hypothesis", "background", "research_questions",
        "proposed_methodology", "expected_contribution", "keywords",
        "estimated_timeline_weeks", "candidate_journals"
    ],
    "properties": {
        "topic_title": {"type": "string", "minLength": 5},
        "hypothesis": {"type": "string", "minLength": 20},
        "background": {"type": "string", "minLength": 100},
        "research_questions": {"type": "array", "items": {"type": "string"}, "minItems": 3},
        "proposed_methodology": {"type": "string", "minLength": 20},
        "expected_contribution": {"type": "string", "minLength": 10},
        "keywords": {"type": "array", "items": {"type": "string"}, "minItems": 3},
        "estimated_timeline_weeks": {"type": "number", "minimum": 1},
        "candidate_journals": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    }
}

FEASIBILITY_SCHEMA = {
    "type": "object",
    "required": ["decision", "scores", "overall_score", "strengths", "weaknesses", "revision_instructions"],
    "properties": {
        "decision": {"type": "string", "enum": ["APPROVED", "REVISE", "REJECTED"]},
        "scores": {
            "type": "object",
            "required": ["novelty", "technical_feasibility", "data_availability",
                         "computational_requirements", "publication_viability"],
            "properties": {
                "novelty": {"type": "number", "minimum": 0, "maximum": 10},
                "technical_feasibility": {"type": "number", "minimum": 0, "maximum": 10},
                "data_availability": {"type": "number", "minimum": 0, "maximum": 10},
                "computational_requirements": {"type": "number", "minimum": 0, "maximum": 10},
                "publication_viability": {"type": "number", "minimum": 0, "maximum": 10},
            }
        },
        "overall_score": {"type": "number", "minimum": 0, "maximum": 10},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "revision_instructions": {},  # string or null
        "approved_proposal": {},  # object or null
    }
}

EXPERIMENT_SCHEMA = {
    "type": "object",
    "required": [
        "experiment_id", "dataset", "models", "baselines", "metrics",
        "statistical_tests", "ablation_plan", "compute_spec", "risks",
        "reproducibility_checklist"
    ],
    "properties": {
        "experiment_id": {"type": "string"},
        "dataset": {
            "type": "object",
            "required": ["name", "source_url", "preprocessing_steps", "splits"],
        },
        "models": {"type": "array", "minItems": 1},
        "baselines": {"type": "array", "minItems": 2},
        "metrics": {"type": "array", "minItems": 1},
        "statistical_tests": {"type": "array", "minItems": 1},
        "ablation_plan": {"type": "array", "minItems": 1},
        "compute_spec": {"type": "object"},
        "risks": {"type": "array"},
        "reproducibility_checklist": {"type": "array", "minItems": 1},
    }
}

VALIDATION_SCHEMA = {
    "type": "object",
    "required": ["validation_decision", "dataset_accessible", "compute_feasible",
                 "no_methodological_flaws", "issues", "validated_experiment"],
    "properties": {
        "validation_decision": {"type": "string", "enum": ["VALIDATED", "RETURN"]},
        "dataset_accessible": {"type": "boolean"},
        "compute_feasible": {"type": "boolean"},
        "no_methodological_flaws": {"type": "boolean"},
        "issues": {"type": "array"},
        "corrections": {},
        "validated_experiment": {},
    }
}

# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Any:
    """
    Extract the first valid JSON object or array from a text string.

    Handles cases where the model wraps JSON in markdown code blocks.

    Parameters
    ----------
    text : str
        Raw text that may contain JSON.

    Returns
    -------
    Any
        Parsed JSON object.

    Raises
    ------
    ValueError
        If no valid JSON can be extracted.
    """
    # Try to find JSON in markdown code blocks first
    pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)
    for m in matches:
        try:
            return json.loads(m.strip())
        except json.JSONDecodeError:
            continue

    # Try to find raw JSON object
    brace_start = text.find("{")
    bracket_start = text.find("[")

    if brace_start == -1 and bracket_start == -1:
        raise ValueError("No JSON found in response")

    # Pick whichever comes first
    if brace_start == -1:
        start = bracket_start
    elif bracket_start == -1:
        start = brace_start
    else:
        start = min(brace_start, bracket_start)

    # Find matching closing bracket
    depth = 0
    in_string = False
    escape = False
    open_char = text[start]
    close_char = "}" if open_char == "{" else "]"

    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    break

    raise ValueError(f"Could not parse JSON from text (first 200 chars): {text[:200]}")


def validate_schema(data: Any, schema: dict, agent_name: str) -> bool:
    """
    Validate data against a simplified schema.

    Parameters
    ----------
    data : Any
        Parsed JSON data.
    schema : dict
        JSON Schema dict.
    agent_name : str
        For logging purposes.

    Returns
    -------
    bool
        True if valid.

    Raises
    ------
    ValueError
        If validation fails.
    """
    try:
        import jsonschema
        jsonschema.validate(instance=data, schema=schema)
        logger.info(f"[{agent_name}] Schema validation passed ✓")
        return True
    except ImportError:
        # Fallback: manual required field check
        required = schema.get("required", [])
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"[{agent_name}] Missing required fields: {missing}")
        logger.info(f"[{agent_name}] Basic schema validation passed ✓ (jsonschema not installed)")
        return True
    except Exception as e:
        raise ValueError(f"[{agent_name}] Schema validation failed: {e}") from e
