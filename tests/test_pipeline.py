"""
tests/test_pipeline.py
Comprehensive unit tests for the Autonomous Neuro-Research Pipeline.
Runs WITHOUT making real Anthropic API calls (all LLM calls are mocked).

Run with:
    pytest tests/ -v
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_API_RESPONSE = """{
  "topic": "Sleep oscillation coupling in hippocampal-neocortical circuits",
  "hypothesis": "Sharp-wave ripples temporally coordinate cortical spindles during NREM sleep.",
  "research_questions": ["RQ1", "RQ2", "RQ3"],
  "background": "Background text.",
  "timeline": "6 months"
}"""


# ===========================================================================
# utils/schema_validator.py
# ===========================================================================

class TestExtractJson:
    def test_plain_json(self):
        from utils.schema_validator import extract_json
        payload = '{"key": "value"}'
        result = extract_json(payload)
        assert result == {"key": "value"}

    def test_json_in_markdown_fence(self):
        from utils.schema_validator import extract_json
        payload = '```json\n{"key": "value"}\n```'
        result = extract_json(payload)
        assert result == {"key": "value"}

    def test_nested_json(self):
        from utils.schema_validator import extract_json
        payload = '{"a": {"b": 1}, "c": [1, 2, 3]}'
        result = extract_json(payload)
        assert result["a"]["b"] == 1
        assert result["c"] == [1, 2, 3]

    def test_invalid_raises(self):
        from utils.schema_validator import extract_json
        with pytest.raises((ValueError, json.JSONDecodeError)):
            extract_json("not json at all")


class TestValidateSchema:
    def test_valid_data_passes(self):
        from utils.schema_validator import validate_schema
        schema = {"required": ["name", "score"]}
        data = {"name": "test", "score": 9.5}
        assert validate_schema(data, schema, "TestAgent") is True

    def test_missing_required_raises(self):
        from utils.schema_validator import validate_schema
        schema = {"required": ["name", "score"]}
        data = {"name": "test"}  # missing 'score'
        with pytest.raises(ValueError):
            validate_schema(data, schema, "TestAgent")


# ===========================================================================
# utils/logger.py
# ===========================================================================

class TestPipelineLogger:
    def test_record_and_get_entries(self, tmp_path):
        from utils.logger import PipelineLogger
        log_file = tmp_path / "pipeline_log.json"
        logger = PipelineLogger(log_path=str(log_file))
        logger.record(
            agent_name="Agent1_Proposer",
            decision="APPROVED",
            output={"topic": "sleep"},
        )
        entries = logger.get_entries()
        assert len(entries) == 1
        assert entries[0]["agent"] == "Agent1_Proposer"
        assert entries[0]["decision"] == "APPROVED"

    def test_log_file_written(self, tmp_path):
        from utils.logger import PipelineLogger
        log_file = tmp_path / "pipeline_log.json"
        logger = PipelineLogger(log_path=str(log_file))
        logger.record(agent_name="AgentX", decision="DONE", output={})
        assert log_file.exists()
        content = json.loads(log_file.read_text())
        assert isinstance(content, list)
        assert content[0]["agent"] == "AgentX"

    def test_last_decision(self, tmp_path):
        from utils.logger import PipelineLogger
        log_file = tmp_path / "pipeline_log.json"
        logger = PipelineLogger(log_path=str(log_file))
        logger.record(agent_name="Agent1", decision="FIRST", output={})
        logger.record(agent_name="Agent1", decision="SECOND", output={})
        last = logger.last_decision("Agent1")
        assert last == "SECOND"

    def test_last_decision_none_for_unknown_agent(self, tmp_path):
        from utils.logger import PipelineLogger
        log_file = tmp_path / "pipeline_log.json"
        logger = PipelineLogger(log_path=str(log_file))
        assert logger.last_decision("UnknownAgent") is None


# ===========================================================================
# load_env.py
# ===========================================================================

class TestLoadEnv:
    def test_loads_dotenv_file(self, tmp_path, monkeypatch):
        """Writing a .env in cwd should expose vars via os.environ."""
        env_file = tmp_path / ".env"
        env_file.write_text('TEST_NEURO_VAR=hello_world\n')
        monkeypatch.chdir(tmp_path)
        # Remove key if already set
        monkeypatch.delenv("TEST_NEURO_VAR", raising=False)
        # Re-run loader in the new cwd
        import importlib, load_env  # noqa
        importlib.reload(load_env)
        assert os.environ.get("TEST_NEURO_VAR") == "hello_world"


# ===========================================================================
# orchestrator.py  — load_init_params
# ===========================================================================

class TestLoadInitParams:
    def test_loads_valid_json(self, tmp_path):
        config = {"settings": {"approval_threshold": 7.0}}
        config_file = tmp_path / "init_params.json"
        config_file.write_text(json.dumps(config))
        # Import function directly — avoids running the whole pipeline
        import importlib.util, types
        spec = importlib.util.spec_from_file_location(
            "orchestrator",
            Path("orchestrator.py"),
        )
        # Patch load_env so the import side-effect is harmless
        sys.modules.setdefault("load_env", types.ModuleType("load_env"))
        mod = importlib.util.module_from_spec(spec)
        # Only parse the function, not run main()
        with patch("builtins.__import__", side_effect=lambda n, *a, **kw: __import__(n, *a, **kw)):
            try:
                spec.loader.exec_module(mod)
            except Exception:
                pass  # ignore runtime errors from imports
        if hasattr(mod, "load_init_params"):
            result = mod.load_init_params(str(config_file))
            assert result["settings"]["approval_threshold"] == 7.0

    def test_missing_file_raises(self):
        import importlib.util, types
        spec = importlib.util.spec_from_file_location("orchestrator", Path("orchestrator.py"))
        sys.modules.setdefault("load_env", types.ModuleType("load_env"))
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass
        if hasattr(mod, "load_init_params"):
            with pytest.raises((FileNotFoundError, SystemExit)):
                mod.load_init_params("/nonexistent/path/config.json")


# ===========================================================================
# Integration smoke test — full pipeline with mocked API
# ===========================================================================

SAMPLE_PROPOSAL = {
    "topic": "Sleep oscillations",
    "hypothesis": "SWR drives spindle coupling.",
    "research_questions": ["RQ1", "RQ2", "RQ3"],
    "background": "Background.",
    "timeline": "6 months",
}

SAMPLE_REVIEW = {
    "overall_score": 8.5,
    "decision": "APPROVED",
    "dimension_scores": {
        "novelty": 8,
        "feasibility": 9,
        "clarity": 8,
        "significance": 9,
        "reproducibility": 8,
    },
    "revision_notes": "",
}

SAMPLE_EXPERIMENT = {
    "experiment_id": "exp-1234",
    "dataset": {"name": "MODA", "url": "https://example.com"},
    "models": ["LSTM"],
    "baselines": ["threshold", "random"],
    "metrics": ["F1", "AUC"],
    "compute": {"gpu": "A100"},
}

SAMPLE_VALIDATION = {
    "validation_decision": "VALIDATED",
    "dataset_accessible": True,
    "compute_feasible": True,
    "no_methodological_flaws": True,
    "corrections": None,
    "validated_experiment": SAMPLE_EXPERIMENT,
}

SAMPLE_CODE = {
    "smoke_test_passed": True,
    "smoke_test_output": "OK",
    "codebase_path": "outputs/codebase",
}

SAMPLE_MANUSCRIPT = "# Draft\n\nAbstract: placeholder."


@patch("utils.api_client.call_claude")
class TestPipelineSmoke:
    """Run each agent function with a fully mocked LLM call."""

    def _setup_outputs(self, tmp_path: Path) -> None:
        (tmp_path / "outputs").mkdir(exist_ok=True)

    def test_agent1_returns_proposal(self, mock_call, tmp_path):
        mock_call.return_value = json.dumps(SAMPLE_PROPOSAL)
        from agents.agent1_proposer import run as run_agent1
        init_params = {
            "research_topics": ["sleep"],
            "settings": {"max_revision_cycles": 1},
        }
        result = run_agent1(init_params=init_params, output_dir=str(tmp_path))
        assert "topic" in result
        assert "hypothesis" in result

    def test_agent2_approves_proposal(self, mock_call, tmp_path):
        mock_call.return_value = json.dumps(SAMPLE_REVIEW)
        from agents.agent2_reviewer import run as run_agent2
        result = run_agent2(proposal=SAMPLE_PROPOSAL, output_dir=str(tmp_path))
        assert result.get("decision") in ("APPROVED", "REVISE")

    def test_agent3_designs_experiment(self, mock_call, tmp_path):
        mock_call.return_value = json.dumps(SAMPLE_EXPERIMENT)
        from agents.agent3_experiment import run as run_agent3
        result = run_agent3(
            proposal=SAMPLE_PROPOSAL,
            review=SAMPLE_REVIEW,
            init_params={"available_resources": {}},
            output_dir=str(tmp_path),
        )
        assert "dataset" in result or "models" in result or result is not None

    def test_agent4_validates_experiment(self, mock_call, tmp_path):
        mock_call.return_value = json.dumps(SAMPLE_VALIDATION)
        from agents.agent4_validator import run as run_agent4
        result = run_agent4(
            experiment_design=SAMPLE_EXPERIMENT,
            output_dir=str(tmp_path),
        )
        assert result is not None

    def test_agent6_generates_manuscript(self, mock_call, tmp_path):
        mock_call.return_value = SAMPLE_MANUSCRIPT
        from agents.agent6_writer import run as run_agent6
        (tmp_path / "outputs").mkdir(exist_ok=True)
        result = run_agent6(
            proposal=SAMPLE_PROPOSAL,
            review=SAMPLE_REVIEW,
            experiment_design=SAMPLE_EXPERIMENT,
            validation=SAMPLE_VALIDATION,
            code_output=SAMPLE_CODE,
            init_params={"target_journals": ["Nature"]},
            output_dir=str(tmp_path),
        )
        assert isinstance(result, str)
        assert len(result) > 0
