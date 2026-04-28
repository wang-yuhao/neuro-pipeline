#!/usr/bin/env bash
# setup.sh — One-command environment setup for the neuro-pipeline
# Usage: bash setup.sh

set -euo pipefail

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Neuro-Pipeline Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Check Python version ────────────────────────────────────────────────
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

echo "▸ Python version: $PYTHON_VERSION"

if [[ $PYTHON_MAJOR -lt 3 || ($PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 10) ]]; then
    echo "ERROR: Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi

# ── 2. Create virtual environment ─────────────────────────────────────────
if [[ ! -d "venv" ]]; then
    echo "▸ Creating virtual environment..."
    python3 -m venv venv
fi

echo "▸ Activating virtual environment..."
source venv/bin/activate

# ── 3. Install dependencies ───────────────────────────────────────────────
echo "▸ Installing pipeline dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "▸ Dependencies installed ✓"

# ── 4. Create outputs directory ───────────────────────────────────────────
mkdir -p outputs/codebase

# ── 5. Check for API key ──────────────────────────────────────────────────
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo ""
    echo "⚠  ANTHROPIC_API_KEY is not set."
    echo ""
    echo "   Option A: Set it in your shell:"
    echo "     export ANTHROPIC_API_KEY='your-key-here'"
    echo ""
    echo "   Option B: Create a .env file:"
    echo "     echo \"ANTHROPIC_API_KEY=your-key-here\" > .env"
    echo "     (the pipeline will auto-load .env if python-dotenv is installed)"
    echo ""
else
    echo "▸ ANTHROPIC_API_KEY detected ✓"
fi

# ── 6. Validate project structure ─────────────────────────────────────────
echo ""
echo "▸ Project structure:"
find . -name "*.py" -not -path "./venv/*" -not -path "./.git/*" | sort | head -40

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete! Run the pipeline with:"
echo ""
echo "    source venv/bin/activate"
echo "    python orchestrator.py --config config/init_params.json"
echo ""
echo "  To customize research topics, edit: config/init_params.json"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
