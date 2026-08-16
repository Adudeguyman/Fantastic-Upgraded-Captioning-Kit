#!/usr/bin/env bash
#
# ---------------------------------------------------------------------------
#  Fantastic Upgraded Captioning Kit — set up and launch.
#
#  First run:  asks whether to use conda or a venv, installs everything, and
#              remembers the choice in .captioner_env.
#  Later runs: reads .captioner_env and launches straight away.
#
#  Usage:
#    chmod +x start.sh      # once
#    ./start.sh             # set up (first time) then launch
#    ./start.sh --setup     # redo the setup / switch environment type
#    ./start.sh --repair    # reinstall dependencies into the current env
#    ./start.sh --cuda      # install the CUDA runtime libs (Linux/NVIDIA)
#
#  Anything else is passed through to the app.
# ---------------------------------------------------------------------------

set -euo pipefail

SOURCE="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
    SOURCE="$(readlink -f "$SOURCE" 2>/dev/null || echo "$SOURCE")"
fi
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

CONFIG=".captioner_env"
CONDA_ENV_DEFAULT="fantastic-captioner"
VENV_DIR_DEFAULT=".venv"
PY_VERSION="3.11"

DO_SETUP=0
DO_REPAIR=0
WITH_CUDA=0
APP_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --setup|--reconfigure) DO_SETUP=1; shift ;;
        --repair) DO_REPAIR=1; shift ;;
        --cuda) WITH_CUDA=1; shift ;;
        -h|--help) sed -n '3,19p' "$0"; exit 0 ;;
        *) APP_ARGS+=("$1"); shift ;;
    esac
done

# --- config ---------------------------------------------------------------
write_config() {
    {
        echo "# Written by start.sh — delete this file (or run ./start.sh --setup)"
        echo "# to choose a different environment."
        echo "ENV_TYPE=$1"
        echo "ENV_NAME=$2"
    } > "$CONFIG"
}

read_config() {
    ENV_TYPE=""
    ENV_NAME=""
    [ -f "$CONFIG" ] || return 1
    while IFS='=' read -r key value; do
        case "$key" in
            ENV_TYPE) ENV_TYPE="$value" ;;
            ENV_NAME) ENV_NAME="$value" ;;
        esac
    done < "$CONFIG"
    [ -n "$ENV_TYPE" ]
}

# --- conda helpers --------------------------------------------------------
find_conda_sh() {
    if command -v conda >/dev/null 2>&1; then
        local base
        base="$(conda info --base 2>/dev/null || true)"
        if [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ]; then
            echo "$base/etc/profile.d/conda.sh"
            return 0
        fi
    fi
    for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
                "$HOME/mambaforge" "/opt/conda"; do
        if [ -f "$base/etc/profile.d/conda.sh" ]; then
            echo "$base/etc/profile.d/conda.sh"
            return 0
        fi
    done
    return 1
}

conda_available() { find_conda_sh >/dev/null 2>&1; }

install_conda_env() {
    local env_name="$1" conda_sh
    conda_sh="$(find_conda_sh)" || {
        echo "Error: could not find conda." >&2
        return 1
    }
    # shellcheck disable=SC1090
    source "$conda_sh"
    if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qxF "$env_name"; then
        echo "Conda environment '$env_name' already exists — reusing it."
    else
        echo "Creating conda environment '$env_name' (Python $PY_VERSION) ..."
        conda create -y -n "$env_name" "python=$PY_VERSION"
    fi
    conda activate "$env_name"
    echo
    echo "Installing dependencies ..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if [ "$WITH_CUDA" -eq 1 ]; then
        echo
        echo "Installing the CUDA runtime libraries ..."
        python -m pip install -r requirements-cuda.txt
    fi
}

# --- venv helpers ---------------------------------------------------------
install_venv_env() {
    local venv_dir="$1" base_py=""
    for cand in python3 python; do
        if command -v "$cand" >/dev/null 2>&1; then
            base_py="$cand"
            break
        fi
    done
    if [ -z "$base_py" ]; then
        echo "Error: no 'python3' or 'python' found on PATH." >&2
        echo "Install Python 3.10+ from https://www.python.org/downloads/" >&2
        return 1
    fi
    if ! "$base_py" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
        echo "Error: Python 3.10+ is required. Found: $("$base_py" --version 2>&1)" >&2
        return 1
    fi
    if [ -x "$venv_dir/bin/python" ]; then
        echo "Virtual environment '$venv_dir' already exists — reusing it."
    else
        echo "Creating virtual environment in '$venv_dir' ..."
        "$base_py" -m venv "$venv_dir"
    fi
    echo
    echo "Installing dependencies ..."
    "$venv_dir/bin/python" -m pip install --upgrade pip
    "$venv_dir/bin/python" -m pip install -r requirements.txt
    if [ "$WITH_CUDA" -eq 1 ]; then
        echo
        echo "Installing the CUDA runtime libraries ..."
        "$venv_dir/bin/python" -m pip install -r requirements-cuda.txt
    fi
}

# --- first-run setup ------------------------------------------------------
run_setup() {
    echo "=================================================================="
    echo " Fantastic Upgraded Captioning Kit — setup"
    echo "=================================================================="
    echo
    echo "How would you like to install the app's Python environment?"
    echo
    if conda_available; then
        echo "  1) conda  — creates the '$CONDA_ENV_DEFAULT' environment (conda detected)"
    else
        echo "  1) conda  — NOT AVAILABLE (conda wasn't found on this system)"
    fi
    echo "  2) venv   — creates a local $VENV_DIR_DEFAULT folder (needs Python 3.10+)"
    echo

    local choice default
    if conda_available; then default="1"; else default="2"; fi
    while true; do
        printf "Enter 1 or 2 [%s]: " "$default"
        read -r choice || choice=""
        choice="${choice:-$default}"
        case "$choice" in
            1)
                if ! conda_available; then
                    echo "conda isn't available on this system — please choose 2 (venv)."
                    continue
                fi
                echo
                install_conda_env "$CONDA_ENV_DEFAULT"
                write_config conda "$CONDA_ENV_DEFAULT"
                return 0
                ;;
            2)
                echo
                install_venv_env "$VENV_DIR_DEFAULT"
                write_config venv "$VENV_DIR_DEFAULT"
                return 0
                ;;
            *) echo "Please enter 1 or 2." ;;
        esac
    done
}

# --- decide what to do ----------------------------------------------------
if [ "$DO_SETUP" -eq 1 ] || ! read_config; then
    run_setup
    read_config || { echo "Error: setup did not complete." >&2; exit 1; }
    echo
fi

# --cuda on its own used to be silently ignored: WITH_CUDA was only read inside the
# install functions, which run only for --setup/--repair. On a configured install it
# fell straight through to launch and installed nothing.
if [ "$WITH_CUDA" -eq 1 ] && [ "$DO_REPAIR" -eq 0 ] && [ "$DO_SETUP" -eq 0 ]; then
    echo "Installing the CUDA runtime libraries into the $ENV_TYPE environment ..."
    if [ "$ENV_TYPE" = "conda" ]; then
        conda_sh="$(find_conda_sh)" || {
            echo "Error: could not find conda." >&2
            exit 1
        }
        # shellcheck disable=SC1090
        source "$conda_sh"
        conda activate "$ENV_NAME"
        python -m pip install -r requirements-cuda.txt
    else
        if [ ! -x "$ENV_NAME/bin/python" ]; then
            echo "Error: virtual environment '$ENV_NAME' is missing." >&2
            echo "Run  ./start.sh --repair --cuda  to rebuild it." >&2
            exit 1
        fi
        "$ENV_NAME/bin/python" -m pip install -r requirements-cuda.txt
    fi
    echo
fi

if [ "$DO_REPAIR" -eq 1 ]; then
    echo "Reinstalling dependencies into the $ENV_TYPE environment ..."
    if [ "$ENV_TYPE" = "conda" ]; then
        install_conda_env "$ENV_NAME"
    else
        install_venv_env "$ENV_NAME"
    fi
    echo
fi

# --- launch ---------------------------------------------------------------
case "$ENV_TYPE" in
    conda)
        conda_sh="$(find_conda_sh)" || {
            echo "Error: could not find conda (was it uninstalled?)." >&2
            echo "Run  ./start.sh --setup  to switch to a venv instead." >&2
            exit 1
        }
        # shellcheck disable=SC1090
        source "$conda_sh"
        if ! conda activate "$ENV_NAME" 2>/dev/null; then
            echo "Conda environment '$ENV_NAME' is missing — rebuilding it ..."
            install_conda_env "$ENV_NAME"
            conda activate "$ENV_NAME"
            echo
        fi
        exec python -m captioning_kit "${APP_ARGS[@]+"${APP_ARGS[@]}"}"
        ;;
    venv)
        if [ ! -x "$ENV_NAME/bin/python" ]; then
            echo "The virtual environment '$ENV_NAME' is missing — rebuilding it ..."
            install_venv_env "$ENV_NAME"
            echo
        fi
        exec "$ENV_NAME/bin/python" -m captioning_kit "${APP_ARGS[@]+"${APP_ARGS[@]}"}"
        ;;
    *)
        echo "Error: unrecognised ENV_TYPE '$ENV_TYPE' in $CONFIG." >&2
        echo "Run  ./start.sh --setup  to reconfigure." >&2
        exit 1
        ;;
esac
