#!/usr/bin/env bash
set -euo pipefail

# Create venv if not present
if [ ! -d "env" ]; then
    python -m venv env
fi

# Activate venv (works for bash/zsh)
source env/bin/activate

# Upgrade pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

# Install dependencies if requirements.txt exists
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# Install from pyproject.toml if using PEP 517/518
if [ -f pyproject.toml ]; then
    pip install .
fi

# Install your package in editable mode
pip install -e .

# Optionally: install dev dependencies
if [ -f requirements-dev.txt ]; then
    pip install -r requirements-dev.txt
fi

echo "Setup complete. To activate the environment, run: source env/bin/activate"