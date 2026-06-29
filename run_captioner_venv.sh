#!/usr/bin/env bash
#
# ---------------------------------------------------------------------------
#  Auto-launcher (Linux / macOS / venv) for the Ideogram captioner.
#
#  On first run this creates a local .venv, installs the requirements, then
#  starts the app. On later runs it just launches.
#
#  Requires Python 3.10+ on your PATH.
#
#  Usage:
#    chmod +x run_captioner_venv.sh   # once
#    ./run_captioner_venv.sh
#
#  (For a conda environment instead, use run_captioner_conda.sh.)
# ---------------------------------------------------------------------------

set -euo pipefail

# Resolve this script's own directory (following symlinks) and work from there.
SOURCE="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
    SOURCE="$(readlink -f "$SOURCE" 2>/dev/null || echo "$SOURCE")"
fi
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
PYEXE="$VENV_DIR/bin/python"

# Pick a base interpreter to build the venv with.
BASE_PY=""
for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        BASE_PY="$cand"
        break
    fi
done
if [ -z "$BASE_PY" ]; then
    echo "Error: no 'python3' or 'python' found on PATH. Install Python 3.10+ first." >&2
    exit 1
fi

if [ ! -x "$PYEXE" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    "$BASE_PY" -m venv "$VENV_DIR"
    echo "Installing dependencies (first run only, this may take a minute) ..."
    "$PYEXE" -m pip install --upgrade pip
    "$PYEXE" -m pip install -r requirements.txt
fi

exec "$PYEXE" -m ideogram_captioner "$@"
