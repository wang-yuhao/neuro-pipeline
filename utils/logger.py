"""
utils/logger.py
Pipeline audit logger — writes pipeline_log.json incrementally.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_PATH = Path("outputs/pipeline_log.json")


def _hash(obj: Any) -> str:
    """Compute a short SHA-256 hash of a JSON-serializable object."""
    raw = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger for console + file output."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("outputs/pipeline.log", mode="a"),
        ],
    )


class PipelineLogger:
    """
    Maintains an audit trail for the entire pipeline.

    Each agent call is recorded as a log entry with:
    - agent_name, timestamp, decision, output_hash, metadata
    """

    def __init__(self, log_path: Path = LOG_PATH) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[dict] = []

        # Load existing log if present (for resume support)
        if self.log_path.exists():
            with open(self.log_path) as f:
                self._entries = json.load(f)

    def record(
        self,
        agent_name: str,
        decision: str,
        output: Any,
        metadata: dict | None = None,
    ) -> None:
        """
        Record an agent's completion to the audit log.

        Parameters
        ----------
        agent_name : str
            E.g. "Agent1_Proposer"
        decision : str
            E.g. "APPROVED", "REVISE", "COMPLETED"
        output : Any
            The agent's output (JSON-serializable).
        metadata : dict | None
            Any extra fields (revision_cycle, attempt, etc.).
        """
        entry = {
            "agent": agent_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "output_hash": _hash(output),
            "metadata": metadata or {},
        }
        self._entries.append(entry)
        self._flush()
        logging.getLogger("pipeline").info(
            f"[{agent_name}] decision={decision} hash={entry['output_hash']}"
        )

    def _flush(self) -> None:
        """Write current log to disk (atomic-ish via temp file)."""
        tmp = self.log_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._entries, f, indent=2, default=str)
        os.replace(tmp, self.log_path)

    def get_entries(self) -> list[dict]:
        return list(self._entries)

    def last_decision(self, agent_name: str) -> str | None:
        """Return the most recent decision for a given agent."""
        for entry in reversed(self._entries):
            if entry["agent"] == agent_name:
                return entry["decision"]
        return None
