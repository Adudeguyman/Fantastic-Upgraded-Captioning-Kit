#!/usr/bin/env bash
#
# Launch the PySide6 (Qt) build of the Ideogram captioner in the
# `id4caption` conda environment.
#
# Usage:
#   chmod +x run_captioner_conda.sh      # once
#   ./run_captioner_conda.sh             # launch the Qt GUI

set -euo pipefail

ENV_NAME="id4caption"

SOURCE="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
    SOURCE="$(readlink -f "$SOURCE" 2>/dev/null || echo "$SOURCE")"
fi
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

CONDA_SH=""
if command -v conda >/dev/null 2>&1; then
    CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
fi
if [ -z "$CONDA_SH" ] || [ ! -f "$CONDA_SH" ]; then
    for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" "/opt/conda"; do
        if [ -f "$base/etc/profile.d/conda.sh" ]; then
            CONDA_SH="$base/etc/profile.d/conda.sh"
            break
        fi
    done
fi
if [ -z "$CONDA_SH" ] || [ ! -f "$CONDA_SH" ]; then
    echo "Error: could not find conda. Edit CONDA_SH near the top of this script." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"

if ! conda activate "$ENV_NAME" 2>/dev/null; then
    echo "Error: conda environment '$ENV_NAME' not found." >&2
    conda env list >&2
    exit 1
fi

exec python -m ideogram_captioner "$@"
