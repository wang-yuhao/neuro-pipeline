"""
agents/agent5_developer.py
Agent 5 — Developer & Researcher Code Agent

Generates a complete, modular Python research codebase based on the validated experiment design,
then writes it to disk.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from utils.api_client import call_claude
from utils.schema_validator import extract_json

logger = logging.getLogger(__name__)

CODEBASE_DIR = Path("outputs/codebase")

SYSTEM_PROMPT = """You are a senior ML research engineer with expertise in:
- PyTorch (2.x), PyTorch Lightning, Weights & Biases
- Neuroscience data pipelines: NWB format, MNE-Python, Neo, h5py, nibabel
- Research-grade Python: PEP 8, type annotations, NumPy docstrings
- Testing: pytest, unittest.mock
- Reproducible research: MLflow, Sacred, hydra configs

## CODE QUALITY STANDARDS (non-negotiable):
1. PEP 8 compliant (max 100 chars/line)
2. Full type annotations (Python 3.10+ syntax)
3. NumPy-style docstrings on ALL public functions/classes
4. Logging with Python's standard `logging` module (not print statements)
5. Configuration via dataclasses or hydra (no hardcoded hyperparameters)
6. Checkpointing: save best model + latest checkpoint
7. Early stopping implemented
8. All results saved to structured JSON/CSV

## CODEBASE STRUCTURE:
```
codebase/
├── data/
│   ├── __init__.py
│   ├── dataset.py       # Dataset class(es)
│   └── preprocessing.py # Preprocessing pipeline
├── models/
│   ├── __init__.py
│   ├── main_model.py    # Primary model
│   └── baselines.py     # Baseline implementations
├── experiments/
│   ├── __init__.py
│   ├── train.py         # Training loop
│   ├── evaluate.py      # Evaluation logic
│   └── config.py        # Experiment configuration (dataclass)
├── utils/
│   ├── __init__.py
│   ├── metrics.py       # Metric implementations
│   ├── statistical.py   # Statistical tests
│   └── visualization.py # Plotting utilities
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_metrics.py
├── notebooks/
│   └── exploration.ipynb  # Skeleton notebook
├── main.py              # Entry point
├── requirements.txt
├── README.md
└── reproduce.sh         # One-command reproduction script
```

## CRITICAL INSTRUCTIONS:
- Generate COMPLETE, RUNNABLE code (not stubs or pseudocode)
- Every file must be importable independently
- Include a smoke_test() function in main.py that runs 10 training steps
- Save results to results.json with all metrics, timestamps, and config
- The code must work with the actual dataset specified in the experiment design

You will output a JSON object where each key is a file path and the value is the complete file content.
Respond with ONLY the JSON. No preamble, no markdown.

Format:
{
  "codebase/data/__init__.py": "content...",
  "codebase/data/dataset.py": "content...",
  ... (all files)
}
"""


def run(
    validated_experiment: dict,
    approved_proposal: dict,
    debug_attempt: int = 0,
    previous_error: str | None = None,
    previous_files: dict | None = None,
) -> dict:
    """
    Execute Agent 5: Code generation and writing.

    Parameters
    ----------
    validated_experiment : dict
        Validated experiment design from Agent 4.
    approved_proposal : dict
        The approved research proposal.
    debug_attempt : int
        Which debug cycle this is (0 = first attempt).
    previous_error : str | None
        Error output from previous attempt (for debugging).
    previous_files : dict | None
        Previously generated files (for targeted fixing).

    Returns
    -------
    dict
        {"files": {path: content, ...}, "test_report": str, "results_json": dict}
    """
    if debug_attempt > 0 and previous_error and previous_files:
        mode_block = f"""
## DEBUG MODE (Attempt {debug_attempt + 1}/5)
The previously generated code had errors. Fix ALL issues.

## ERROR OUTPUT:
{previous_error}

## PREVIOUSLY GENERATED FILES (fix these):
{json.dumps(previous_files, indent=2)[:8000]}  <!-- truncated if long -->
"""
    else:
        mode_block = "## INITIAL CODE GENERATION"

    user_message = f"""
{mode_block}

## VALIDATED EXPERIMENT DESIGN:
{json.dumps(validated_experiment, indent=2)}

## APPROVED RESEARCH PROPOSAL:
{json.dumps(approved_proposal, indent=2)}

## YOUR TASK:
Generate a COMPLETE Python research codebase implementing the experiment design.

Key requirements:
1. The dataset loading code must handle the actual data format ({validated_experiment.get('dataset', {}).get('name', 'specified dataset')})
2. Implement ALL models listed in the experiment design
3. Implement ALL baselines listed in the experiment design  
4. Implement ALL metrics listed in the experiment design
5. Implement ALL statistical tests listed in the experiment design
6. smoke_test() in main.py must run 10 training steps with synthetic/random data (no download needed for smoke test)
7. results.json must capture: metrics for each model, p-values for statistical tests, timestamps, config hash

The smoke test should:
- Use synthetic data (torch.randn or similar) to avoid needing to download the real dataset
- Run exactly 10 training steps
- Verify the model forward pass works
- Verify loss decreases (or at least runs without error)
- Save results to outputs/results.json

Respond with ONLY the JSON object mapping file paths to file contents.
No preamble, no markdown fences, no explanation.
"""

    logger.info(f"Agent 5: Generating codebase (attempt {debug_attempt + 1}/5)...")
    raw_response = call_claude(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        max_tokens=8096,
        temperature=0.2,
    )

    logger.debug(f"Agent 5 raw response length: {len(raw_response)} chars")

    # Parse the file map
    files = extract_json(raw_response)

    if not isinstance(files, dict):
        raise ValueError(f"Agent 5: Expected dict of files, got {type(files)}")

    # Write files to disk
    _write_codebase(files)

    logger.info(f"Agent 5: Wrote {len(files)} files to {CODEBASE_DIR}")
    return files


def _write_codebase(files: dict[str, str]) -> None:
    """Write all generated files to the codebase directory."""
    CODEBASE_DIR.mkdir(parents=True, exist_ok=True)

    for rel_path, content in files.items():
        # Normalize path — strip leading slashes or 'codebase/' prefix if doubled
        rel_path = rel_path.lstrip("/")
        if not rel_path.startswith("codebase/"):
            rel_path = f"codebase/{rel_path}"

        full_path = Path("outputs") / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.debug(f"  Wrote: {full_path}")


def run_smoke_test() -> tuple[bool, str]:
    """
    Attempt to run the smoke test in the generated codebase.

    Returns
    -------
    tuple[bool, str]
        (success, output_or_error_message)
    """
    import subprocess
    import sys

    main_py = Path("outputs/codebase/main.py")
    if not main_py.exists():
        return False, f"main.py not found at {main_py}"

    try:
        result = subprocess.run(
            [sys.executable, str(main_py), "--smoke-test"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path("outputs/codebase"),
        )
        output = result.stdout + result.stderr
        if result.returncode == 0:
            return True, output
        else:
            return False, output
    except subprocess.TimeoutExpired:
        return False, "Smoke test timed out after 120 seconds"
    except Exception as e:
        return False, f"Smoke test error: {e}"


def run_unit_tests() -> tuple[bool, str]:
    """
    Run pytest on the generated codebase.

    Returns
    -------
    tuple[bool, str]
        (success, output)
    """
    import subprocess
    import sys

    tests_dir = Path("outputs/codebase/tests")
    if not tests_dir.exists():
        return False, "tests/ directory not found"

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(tests_dir), "-v", "--tb=short", "--timeout=60"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "pytest timed out after 300 seconds"
    except Exception as e:
        return False, f"pytest error: {e}"
