"""
load_env.py
Auto-loads .env file if present, before any other imports.
Import this module as the very first import in orchestrator.py.
"""
import os
from pathlib import Path

def _load_env() -> None:
    """Load .env file if python-dotenv is available and .env exists."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=env_path, override=False)
        print(f"[load_env] Loaded environment variables from {env_path.resolve()}")
    except ImportError:
        # python-dotenv not installed; read manually
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
        print(f"[load_env] Loaded .env manually (dotenv not installed)")

_load_env()
