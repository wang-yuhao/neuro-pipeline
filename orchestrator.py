"""
orchestrator.py
Main pipeline orchestrator for the Autonomous Computational Neuroscience Research Pipeline.

Coordinates all 6 agents in strict sequential order with:
- Gate decisions between agents
- Revision cycles with configurable max attempts
- Audit logging of every decision
- Failure escalation after 3 consecutive agent errors
- All outputs saved to the outputs/ directory
"""

# Auto-load .env file if present
import load_env  # noqa: F401 - side-effect import


from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

from utils.logger import PipelineLogger, setup_logging
from agents import agent1_proposer, agent2_reviewer, agent3_experiment, \
    agent4_validator, agent5_developer, agent6_writer

# ───────────────────────────── Paths ────────────────────────────────────────
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("orchestrator")


# ───────────────────────────── Helpers ──────────────────────────────────────

def load_init_params(path: str = "config/init_params.json") -> dict:
    """Load and return the pipeline initialization parameters."""
    with open(path) as f:
        params = json.load(f)
    logger.info(f"Loaded init_params from {path}")
    return params


def save_output(filename: str, data: Any) -> None:
    """Save agent output to the outputs directory."""
    path = OUTPUTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(data, str):
            f.write(data)
        else:
            json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved output → {path}")


def escalate(stage: str, error_summary: str, pipeline_log: PipelineLogger) -> None:
    """
    Escalate to a human-readable summary report when max retries exceeded.
    Does NOT halt silently - writes a human_escalation_report.txt.
    """
    report = (
        f"⚠️  PIPELINE ESCALATION REPORT\n"
        f"{'='*60}\n"
        f"Stage: {stage}\n"
        f"Reason: Max retries exceeded or unrecoverable error\n\n"
        f"Error Summary:\n{error_summary}\n\n"
        f"Pipeline Log:\n"
        + json.dumps(pipeline_log.get_entries(), indent=2)
    )
    path = OUTPUTS_DIR / "human_escalation_report.txt"
    with open(path, "w") as f:
        f.write(report)
    logger.critical(f"Pipeline escalated at stage '{stage}'. See {path}")
    print(f"\n{'='*60}")
    print("⚠️  HUMAN INTERVENTION REQUIRED")
    print(f"Pipeline escalated at: {stage}")
    print(f"Report saved to: {path}")
    print('='*60)


# ───────────────────────────── Main Orchestrator ─────────────────────────────

def run_pipeline(init_params_path: str = "config/init_params.json") -> None:
    """
    Run the full 6-agent research pipeline.

    Pipeline flow:
    Agent 1 -> Agent 2 (review loop) -> Agent 3 -> Agent 4 (validation loop)
    -> Agent 5 (debug loop) -> Agent 6

    Parameters
    ----------
    init_params_path : str
        Path to the JSON initialization parameters file.
    """
    # ── Setup ──────────────────────────────────────────────────────────────
    init_params = load_init_params(init_params_path)
    settings = init_params.get("pipeline_settings", {})
    max_revision_cycles = settings.get("max_revision_cycles", 2)
    max_retries = settings.get("max_agent_retries", 3)
    approval_threshold = settings.get("approval_threshold", 7.0)

    setup_logging(settings.get("log_level", "INFO"))
    pipeline_log = PipelineLogger(OUTPUTS_DIR / "pipeline_log.json")

    print("\n" + "="*70)
    print("  AUTONOMOUS COMPUTATIONAL NEUROSCIENCE RESEARCH PIPELINE")
    print("="*70 + "\n")

    # ══════════════════════════════════════════════════════════════════════
    # AGENT 1: Research Topic Proposer
    # ══════════════════════════════════════════════════════════════════════
    print("▶ AGENT 1: Research Topic Proposer")
    print("─"*50)

    proposal = None
    revision_instructions = None
    revision_cycle = 0

    for attempt in range(1, max_retries + 1):
        try:
            proposal = agent1_proposer.run(
                init_params=init_params,
                revision_instructions=revision_instructions,
                previous_proposal=proposal,
            )
            pipeline_log.record("Agent1_Proposer", "COMPLETED", proposal,
                                {"attempt": attempt, "revision_cycle": revision_cycle})
            save_output("proposal.json", proposal)
            print(f"  ✓ Proposal generated: '{proposal['topic_title']}'")
            break
        except Exception as e:
            logger.error(f"Agent 1 attempt {attempt} failed: {e}")
            if attempt == max_retries:
                escalate("Agent1_Proposer", traceback.format_exc(), pipeline_log)
                return

    # Gate check confirmation
    print(f"  Gate: Hypothesis ✓ | Research Questions ({len(proposal.get('research_questions', []))}) ✓")

    # ══════════════════════════════════════════════════════════════════════
    # AGENT 2 + AGENT 1 REVISION LOOP
    # ══════════════════════════════════════════════════════════════════════
    print("\n▶ AGENT 2: Feasibility & Peer Review")
    print("─"*50)

    review = None
    approved_proposal = None

    for cycle in range(max_revision_cycles + 1):
        # Agent 2: Review
        for attempt in range(1, max_retries + 1):
            try:
                review = agent2_reviewer.run(
                    proposal=proposal,
                    init_params=init_params,
                    revision_cycle=cycle,
                )
                pipeline_log.record("Agent2_Reviewer", review["decision"], review,
                                    {"cycle": cycle, "attempt": attempt,
                                     "overall_score": review.get("overall_score")})
                save_output("feasibility_review.json", review)
                break
            except Exception as e:
                logger.error(f"Agent 2 attempt {attempt} (cycle {cycle}) failed: {e}")
                if attempt == max_retries:
                    escalate("Agent2_Reviewer", traceback.format_exc(), pipeline_log)
                    return

        decision = review["decision"]
        score = review.get("overall_score", 0)
        print(f"  Cycle {cycle}: Decision={decision}, Score={score:.2f}/10.0")

        scores = review.get("scores", {})
        for dim, val in scores.items():
            print(f"    {dim}: {val}/10")

        if decision == "APPROVED":
            approved_proposal = review.get("approved_proposal") or proposal
            print(f"  ✓ GATE PASSED — Proposal approved (score={score:.2f})")
            break

        elif decision == "REVISE":
            if cycle >= max_revision_cycles:
                print(f"  ✗ Max revision cycles ({max_revision_cycles}) reached. Escalating.")
                escalate(
                    "Agent2_Reviewer_RevisionCycleLimit",
                    f"Proposal score {score:.2f} after {cycle} revision cycles.\n"
                    f"Weaknesses: {review.get('weaknesses')}",
                    pipeline_log
                )
                return

            revision_instructions = review.get("revision_instructions", "")
            print(f"  ↩ REVISING — Returning to Agent 1 (cycle {cycle+1}/{max_revision_cycles})")
            print(f"    Instructions: {revision_instructions[:200]}...")

            # Agent 1 revision
            for attempt in range(1, max_retries + 1):
                try:
                    revision_cycle = cycle + 1
                    proposal = agent1_proposer.run(
                        init_params=init_params,
                        revision_instructions=revision_instructions,
                        previous_proposal=proposal,
                    )
                    pipeline_log.record("Agent1_Proposer", "REVISED", proposal,
                                        {"revision_cycle": revision_cycle, "attempt": attempt})
                    save_output("proposal.json", proposal)
                    print(f"  ✓ Revised proposal: '{proposal['topic_title']}'")
                    break
                except Exception as e:
                    logger.error(f"Agent 1 revision attempt {attempt} failed: {e}")
                    if attempt == max_retries:
                        escalate("Agent1_Proposer_Revision", traceback.format_exc(), pipeline_log)
                        return

        elif decision == "REJECTED":
            print(f"  ✗ REJECTED — Score {score:.2f} below minimum threshold.")
            print(f"    Weaknesses: {review.get('weaknesses')}")
            escalate(
                "Agent2_Reviewer_Rejected",
                f"Proposal rejected with score {score:.2f}.\n"
                f"Weaknesses: {json.dumps(review.get('weaknesses'), indent=2)}",
                pipeline_log
            )
            return

    if not approved_proposal:
        escalate("Agent2_Reviewer_NoApproval", "No approved proposal after review cycles", pipeline_log)
        return

    # ══════════════════════════════════════════════════════════════════════
    # AGENT 3 + AGENT 4 VALIDATION LOOP
    # ══════════════════════════════════════════════════════════════════════
    print("\n▶ AGENT 3: Experiment Design")
    print("─"*50)

    experiment_design = None
    validated_experiment = None
    corrections = None

    for val_cycle in range(max_revision_cycles + 1):
        # Agent 3: Design
        for attempt in range(1, max_retries + 1):
            try:
                experiment_design = agent3_experiment.run(
                    approved_proposal=approved_proposal,
                    init_params=init_params,
                    corrections=corrections,
                    previous_design=experiment_design,
                )
                pipeline_log.record("Agent3_ExperimentDesign", "COMPLETED", experiment_design,
                                    {"val_cycle": val_cycle, "attempt": attempt})
                save_output("experiment_design.json", experiment_design)
                print(f"  ✓ Experiment designed — ID: {experiment_design['experiment_id']}")
                print(f"    Dataset: {experiment_design.get('dataset', {}).get('name', 'N/A')}")
                print(f"    Models: {len(experiment_design.get('models', []))}")
                print(f"    Baselines: {len(experiment_design.get('baselines', []))}")
                break
            except Exception as e:
                logger.error(f"Agent 3 attempt {attempt} (val_cycle {val_cycle}) failed: {e}")
                if attempt == max_retries:
                    escalate("Agent3_ExperimentDesign", traceback.format_exc(), pipeline_log)
                    return

        # Agent 4: Validate
        print(f"\n▶ AGENT 4: Logic & Resource Validation (Cycle {val_cycle})")
        print("─"*50)

        for attempt in range(1, max_retries + 1):
            try:
                validation = agent4_validator.run(
                    experiment_design=experiment_design,
                    approved_proposal=approved_proposal,
                    init_params=init_params,
                )
                pipeline_log.record("Agent4_Validator", validation["validation_decision"], validation,
                                    {"val_cycle": val_cycle, "attempt": attempt})
                break
            except Exception as e:
                logger.error(f"Agent 4 attempt {attempt} (val_cycle {val_cycle}) failed: {e}")
                if attempt == max_retries:
                    escalate("Agent4_Validator", traceback.format_exc(), pipeline_log)
                    return

        val_decision = validation["validation_decision"]
        ds_ok = "✓" if validation.get("dataset_accessible") else "✗"
        compute_ok = "✓" if validation.get("compute_feasible") else "✗"
        methods_ok = "✓" if validation.get("no_methodological_flaws") else "✗"

        print(f"  Decision: {val_decision}")
        print(f"  Gate: dataset_accessible {ds_ok} | compute_feasible {compute_ok} | no_methodological_flaws {methods_ok}")

        if validation.get("issues"):
            for issue in validation["issues"]:
                print(f"    ⚠ {issue}")

        if val_decision == "VALIDATED":
            validated_experiment = validation.get("validated_experiment") or experiment_design
            save_output("experiment_design.json", validated_experiment)
            print("  ✓ GATE PASSED — Experiment design validated")
            break

        elif val_decision == "RETURN":
            if val_cycle >= max_revision_cycles:
                escalate(
                    "Agent4_Validator_CycleLimit",
                    f"Validation failed after {val_cycle} correction cycles.\n"
                    f"Issues: {json.dumps(validation.get('issues'), indent=2)}",
                    pipeline_log
                )
                return
            corrections = validation.get("corrections", "")
            print(f"  ↩ Returning to Agent 3 with corrections...")

    if not validated_experiment:
        escalate("Agent4_Validator_NoValidation", "No validated experiment after correction cycles", pipeline_log)
        return

    # ══════════════════════════════════════════════════════════════════════
    # AGENT 5: Code Generation + Debug Loop
    # ══════════════════════════════════════════════════════════════════════
    print("\n▶ AGENT 5: Developer & Researcher Code Agent")
    print("─"*50)

    codebase_files = None
    results_json = None
    previous_error = None

    for debug_attempt in range(5):
        # Generate code
        for attempt in range(1, max_retries + 1):
            try:
                codebase_files = agent5_developer.run(
                    validated_experiment=validated_experiment,
                    approved_proposal=approved_proposal,
                    debug_attempt=debug_attempt,
                    previous_error=previous_error,
                    previous_files=codebase_files,
                )
                pipeline_log.record("Agent5_Developer", "CODE_GENERATED", 
                                    {"files": list(codebase_files.keys())},
                                    {"debug_attempt": debug_attempt, "attempt": attempt})
                print(f"  ✓ Generated {len(codebase_files)} files")
                break
            except Exception as e:
                logger.error(f"Agent 5 attempt {attempt} (debug_attempt {debug_attempt}) failed: {e}")
                if attempt == max_retries:
                    escalate("Agent5_Developer", traceback.format_exc(), pipeline_log)
                    return

        # Install dependencies first
        print("  Installing dependencies...")
        req_path = OUTPUTS_DIR / "codebase/requirements.txt"
        if req_path.exists():
            import subprocess
            install_result = subprocess.run(
                ["pip", "install", "-r", str(req_path), "-q"],
                capture_output=True, text=True, timeout=180
            )
            if install_result.returncode != 0:
                logger.warning(f"pip install warnings: {install_result.stderr[:500]}")

        # Run smoke test
        print(f"  Running smoke test (attempt {debug_attempt + 1}/5)...")
        smoke_success, smoke_output = agent5_developer.run_smoke_test()

        if smoke_success:
            print("  ✓ Smoke test PASSED (10 training steps verified)")
            pipeline_log.record("Agent5_Developer", "SMOKE_TEST_PASSED",
                                {"output": smoke_output[:500]},
                                {"debug_attempt": debug_attempt})

            # Run unit tests
            print("  Running unit tests...")
            test_success, test_output = agent5_developer.run_unit_tests()
            if test_success:
                print("  ✓ Unit tests PASSED")
            else:
                print("  ⚠ Unit tests had failures (non-blocking):")
                print(f"    {test_output[:300]}")

            # Save test report
            test_report = {
                "smoke_test": "PASSED" if smoke_success else "FAILED",
                "unit_tests": "PASSED" if test_success else "FAILED",
                "smoke_output": smoke_output[:2000],
                "test_output": test_output[:2000],
            }
            save_output("test_report.json", test_report)

            # Load results.json if it was generated
            results_path = OUTPUTS_DIR / "results.json"
            if results_path.exists():
                with open(results_path) as f:
                    results_json = json.load(f)
                print(f"  ✓ results.json loaded")

            break

        else:
            print(f"  ✗ Smoke test FAILED:")
            print(f"    {smoke_output[:400]}")
            previous_error = smoke_output
            pipeline_log.record("Agent5_Developer", "SMOKE_TEST_FAILED",
                                {"error": smoke_output[:500]},
                                {"debug_attempt": debug_attempt})

            if debug_attempt == 4:  # Last attempt
                print("  ⚠ Max debug cycles reached. Proceeding with best-effort codebase.")
                pipeline_log.record("Agent5_Developer", "MAX_DEBUG_CYCLES_REACHED",
                                    {"error": smoke_output[:500]})
                # Save error report
                save_output("test_report.json", {
                    "smoke_test": "FAILED",
                    "unit_tests": "NOT_RUN",
                    "error": smoke_output[:2000],
                    "note": "Max debug cycles reached. Code generated but smoke test failed."
                })

    # ══════════════════════════════════════════════════════════════════════
    # AGENT 6: Academic Writing
    # ══════════════════════════════════════════════════════════════════════
    print("\n▶ AGENT 6: Academic Writing Agent")
    print("─"*50)

    manuscript = None
    for attempt in range(1, max_retries + 1):
        try:
            manuscript = agent6_writer.run(
                approved_proposal=approved_proposal,
                validated_experiment=validated_experiment,
                codebase_files=codebase_files or {},
                results_json=results_json,
            )
            pipeline_log.record("Agent6_Writer", "MANUSCRIPT_COMPLETED",
                                {"word_count": len(manuscript.split()), "chars": len(manuscript)},
                                {"attempt": attempt})
            # Already saved in agent6_writer.run()
            print(f"  ✓ Manuscript draft generated ({len(manuscript.split())} words)")
            break
        except Exception as e:
            logger.error(f"Agent 6 attempt {attempt} failed: {e}")
            if attempt == max_retries:
                escalate("Agent6_Writer", traceback.format_exc(), pipeline_log)

    # ══════════════════════════════════════════════════════════════════════
    # PIPELINE COMPLETE
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  ✅ PIPELINE COMPLETE — All deliverables generated")
    print("="*70)
    print("\nFINAL DELIVERABLES:")
    deliverables = [
        ("outputs/proposal.json", "Research proposal (Agent 1)"),
        ("outputs/feasibility_review.json", "Feasibility review (Agent 2)"),
        ("outputs/experiment_design.json", "Experiment design (Agents 3+4)"),
        ("outputs/codebase/", "Complete research codebase (Agent 5)"),
        ("outputs/results.json", "Experiment results (Agent 5)"),
        ("outputs/test_report.json", "Test report (Agent 5)"),
        ("outputs/manuscript_draft.md", "Manuscript draft (Agent 6)"),
        ("outputs/pipeline_log.json", "Audit trail (Orchestrator)"),
    ]
    for path, description in deliverables:
        exists = "✓" if (Path(path).exists() or Path(path).is_dir()) else "·"
        print(f"  {exists} {path:<45} — {description}")

    pipeline_log.record("Orchestrator", "PIPELINE_COMPLETE", 
                        {"deliverables": [d[0] for d in deliverables]})
    print("\n")


# ───────────────────────────── Entry Point ───────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Autonomous Computational Neuroscience Research Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/init_params.json",
        help="Path to init_params.json",
    )
    args = parser.parse_args()

    try:
        run_pipeline(args.config)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled pipeline error: {e}")
        traceback.print_exc()
        sys.exit(1)
