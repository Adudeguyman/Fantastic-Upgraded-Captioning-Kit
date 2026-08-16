#!/usr/bin/env bash
#
# cuda_fix.sh — install the CUDA runtime libraries into your captioner environment.
#
# The prebuilt CUDA llama-server dynamically links the CUDA runtime (libcublas,
# libcudart, libnccl and friends) but the release archives don't ship them, so a
# CUDA launch can fail to even load with, for example
#   "error while loading shared libraries: libcublas.so.12"
#   "error while loading shared libraries: libnccl.so.2"
# even on a single GPU. This installs requirements-cuda.txt (the nvidia-*-cu12
# wheels) into your environment so the server can find them.
#
# Note: libcuda.so.1 is NOT covered — that one comes from the NVIDIA driver, so if
# it's the missing library, install or repair the driver instead.
#
# Linux x86_64 only — Windows/macOS and the Vulkan/CPU backends don't need these.
#
# Usage:
#   chmod +x cuda_fix.sh   # once
#   ./cuda_fix.sh

set -euo pipefail

# Resolve this script's own directory (following symlinks) and work from there.
SOURCE="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
    SOURCE="$(readlink -f "$SOURCE" 2>/dev/null || echo "$SOURCE")"
fi
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

REQ="requirements-cuda.txt"
if [ ! -f "$REQ" ]; then
    echo "Error: $REQ not found next to this script." >&2
    exit 1
fi

run_pip() {
    # $1 = python executable to install into
    echo
    echo "Installing $REQ ..."
    "$1" -m pip install -r "$REQ"
    echo
    echo "Done. Restart the captioner (or its llama.cpp server) to pick up the libraries."
}

echo "CUDA runtime fix — installs $REQ into your captioner environment."
echo
echo "Which environment does the captioner run in?"
echo "  1) conda"
echo "  2) venv  (.venv in this folder)"
printf "Enter 1 or 2: "
read -r choice

case "$choice" in
  1)
    # Locate and source conda the same way the launcher scripts do.
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
        echo "Error: could not find conda on this system." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$CONDA_SH"

    # Prompt for the env name and verify it exists before touching anything.
    while true; do
        printf "Conda environment name [fantastic-captioner]: "
        read -r ENV_NAME
        ENV_NAME="${ENV_NAME:-fantastic-captioner}"
        if conda env list | awk '{print $1}' | grep -qxF "$ENV_NAME"; then
            break
        fi
        echo "Environment '$ENV_NAME' not found. Available environments:"
        conda env list
        echo
    done

    conda activate "$ENV_NAME"
    echo "Using conda environment '$ENV_NAME'."
    run_pip "python"
    ;;
  2)
    PYEXE=".venv/bin/python"
    if [ ! -x "$PYEXE" ]; then
        echo "Error: $PYEXE not found. Create the venv first with install_venv.sh." >&2
        exit 1
    fi
    echo "Using venv at $PYEXE."
    run_pip "$PYEXE"
    ;;
  *)
    echo "Invalid choice. Run again and enter 1 or 2." >&2
    exit 1
    ;;
esac
