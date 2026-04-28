# ============================================================
# Dockerfile — Autonomous Neuro-Research Pipeline
# ============================================================
# Build:  docker build -t neuro-pipeline .
# Run:    docker run --rm -e ANTHROPIC_API_KEY=sk-ant-xxx \
#                 -v $(pwd)/outputs:/app/outputs \
#                 neuro-pipeline
# Tests:  docker run --rm neuro-pipeline pytest tests/ -v
# ============================================================

FROM python:3.11-slim

# --- System deps ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        git \
    && rm -rf /var/lib/apt/lists/*

# --- Working directory ---
WORKDIR /app

# --- Python dependencies (cached layer) ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytest pytest-timeout pytest-cov flake8

# --- Copy source code ---
COPY . .

# --- Create output directories ---
RUN mkdir -p outputs/codebase

# --- Healthcheck: verify imports ---
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import orchestrator" || exit 1

# --- Default command: run the full pipeline ---
CMD ["python", "orchestrator.py", "--config", "config/init_params.json"]
