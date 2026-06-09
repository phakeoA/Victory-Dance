#!/bin/bash
# =============================================================================
# ShowdownProject Environment Setup Script
# Run this from Git Bash inside D:\ShowdownProject\Victory-Dance
# Usage: bash setup.sh
#
# Stack: poke-env (Showdown client) + PyTorch (policy/value network)
#        + MCTS layer (Phase 2)
# =============================================================================

# NOTE: We intentionally do NOT use "set -e" here because some steps
# produce non-zero exit codes that are harmless.
# Each critical step has its own explicit error check instead.

# --- Colors for output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()    { echo -e "${GREEN}[SETUP]${NC} $1"; }
warn()   { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
header() { echo -e "\n${BLUE}========================================${NC}";
           echo -e "${BLUE} $1${NC}";
           echo -e "${BLUE}========================================${NC}"; }

# =============================================================================
# 0. VERIFY LOCATION
# =============================================================================
header "Checking Location"

EXPECTED_DIR="/d/ShowdownProject/Victory-Dance"
CURRENT_DIR=$(pwd)

if [[ "$CURRENT_DIR" != "$EXPECTED_DIR" ]]; then
    warn "Expected: $EXPECTED_DIR"
    warn "Current:  $CURRENT_DIR"
    read -p "Continue anyway? (y/n): " confirm
    if [[ "$confirm" != "y" ]]; then
        error "Aborted. Navigate to D:\\ShowdownProject\\Victory-Dance and rerun."
    fi
fi

log "Working directory: $CURRENT_DIR"

# =============================================================================
# 1. FIND PYTHON 3.10+ (poke-env requires 3.10+, project targets 3.12)
# =============================================================================
header "Checking Python"

REQUIRED_MAJOR=3
REQUIRED_MINOR=10
PYTHON_INSTALLED=false
PYTHON_CMD=""

# Helper — returns 0 if the given python is 3.10+
check_python() {
    local cmd="$1"
    if [[ -f "$cmd" ]] || command -v "$cmd" &>/dev/null 2>&1; then
        local MAJOR MINOR
        MAJOR=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        MINOR=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [[ "$MAJOR" -eq "$REQUIRED_MAJOR" && "$MINOR" -ge "$REQUIRED_MINOR" ]]; then
            return 0
        fi
    fi
    return 1
}

# Priority order:
# 1. Your custom D:\Python\3.12 install
# 2. Standard PATH names
# 3. Common C drive fallbacks

CUSTOM_PATHS=(
    "/d/Python/3.12/python.exe"
    "/d/Python/3.12/python"
    "/d/Python/Python312/python.exe"
    "/d/Python/3.11/python.exe"
)

PATH_CMDS=(
    "python3.12"
    "python3.11"
    "python3.13"
    "python3"
    "python"
)

STANDARD_PATHS=(
    "/c/Python312/python.exe"
    "/c/Python311/python.exe"
    "/c/Users/$USERNAME/AppData/Local/Programs/Python/Python312/python.exe"
    "/c/Users/$USERNAME/AppData/Local/Programs/Python/Python311/python.exe"
)

for path in "${CUSTOM_PATHS[@]}"; do
    if check_python "$path"; then
        PYTHON_CMD="$path"
        PYTHON_INSTALLED=true
        log "Found Python (D drive custom): $("$path" --version) at $path"
        break
    fi
done

if [[ "$PYTHON_INSTALLED" == false ]]; then
    for cmd in "${PATH_CMDS[@]}"; do
        if check_python "$cmd"; then
            PYTHON_CMD="$cmd"
            PYTHON_INSTALLED=true
            log "Found Python (PATH): $($cmd --version) at $(which $cmd 2>/dev/null)"
            break
        fi
    done
fi

if [[ "$PYTHON_INSTALLED" == false ]]; then
    for path in "${STANDARD_PATHS[@]}"; do
        if check_python "$path"; then
            PYTHON_CMD="$path"
            PYTHON_INSTALLED=true
            log "Found Python (C drive): $("$path" --version) at $path"
            break
        fi
    done
fi

if [[ "$PYTHON_INSTALLED" == false ]]; then
    echo ""
    warn "Python 3.10+ not found. Checked:"
    warn "  D:\\Python\\3.12\\python.exe  (your custom install)"
    warn "  Standard PATH commands"
    warn "  C:\\Python312\\"
    echo ""
    echo -e "${YELLOW}Fix: Add Python 3.12 to your PATH:${NC}"
    echo "  Windows Search → 'Edit environment variables'"
    echo "  → Path → New → D:\\Python\\3.12"
    echo "  → New → D:\\Python\\3.12\\Scripts"
    echo "  Then close Git Bash, reopen, and rerun setup.sh"
    echo ""
    error "Cannot continue without Python 3.10+."
fi

log "Prerequisites OK."

# =============================================================================
# 2. SET UP PYTHON VENV
# =============================================================================
header "Setting Up Python Virtual Environment"

VENV_DIR="$(pwd)/.venv"

if [[ -d "$VENV_DIR" ]]; then
    log ".venv already exists. Skipping creation."
else
    log "Creating Python virtual environment using: $PYTHON_CMD"
    "$PYTHON_CMD" -m venv "$VENV_DIR"

    if [[ ! -d "$VENV_DIR" ]]; then
        error "venv creation failed. Python may not have the venv module. Try: $PYTHON_CMD -m pip install virtualenv"
    fi

    log "Virtual environment created at $VENV_DIR"
fi

# Activate — Windows Git Bash uses Scripts/, Unix uses bin/
if [[ -f "$VENV_DIR/Scripts/activate" ]]; then
    source "$VENV_DIR/Scripts/activate"
elif [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
else
    error "Cannot find venv activation script. venv creation may have silently failed."
fi

log "Python venv activated."
log "Python: $(python --version) at $(which python)"

log "Upgrading pip..."
python -m pip install --upgrade pip --quiet

# =============================================================================
# 2b. INSTALL NODE.JS 10+ INTO .venv
# =============================================================================
# Node.js is installed as a portable (no-install) binary inside .venv/node/
# so it stays project-local and doesn't require system-level changes.
# The .venv/bin (or Scripts) directory gets wrapper shims for node and npm.
# =============================================================================
header "Installing Node.js into .venv"

NODE_DIR="$VENV_DIR/node"
NODE_MIN_MAJOR=10
NODE_VERSION="22.3.0"   # LTS — change if you need a different 10+ release
NODE_WIN_ZIP="node-v${NODE_VERSION}-win-x64.zip"
NODE_WIN_URL="https://nodejs.org/dist/v${NODE_VERSION}/${NODE_WIN_ZIP}"
NODE_LINUX_TAR="node-v${NODE_VERSION}-linux-x64.tar.xz"
NODE_LINUX_URL="https://nodejs.org/dist/v${NODE_VERSION}/${NODE_LINUX_TAR}"

# Helper — checks if a binary satisfies Node 10+
check_node() {
    local cmd="$1"
    if command -v "$cmd" &>/dev/null 2>&1 || [[ -x "$cmd" ]]; then
        local major
        major=$("$cmd" -e "process.stdout.write(String(process.versions.node.split('.')[0]))" 2>/dev/null)
        if [[ -n "$major" && "$major" -ge "$NODE_MIN_MAJOR" ]]; then
            return 0
        fi
    fi
    return 1
}

# Detect OS / shell environment
detect_os() {
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
        Darwin)               echo "macos"   ;;
        Linux)                echo "linux"   ;;
        *)                    echo "unknown" ;;
    esac
}
OS=$(detect_os)

# Shim dir — prefer bin/ (Unix) otherwise Scripts/ (Windows venv)
if [[ -d "$VENV_DIR/bin" ]]; then
    SHIM_DIR="$VENV_DIR/bin"
else
    SHIM_DIR="$VENV_DIR/Scripts"
fi

# Check if we already have a good Node inside .venv
NODE_BIN="$NODE_DIR/node"
[[ "$OS" == "windows" ]] && NODE_BIN="$NODE_DIR/node.exe"

if check_node "$NODE_BIN"; then
    NODE_VER=$("$NODE_BIN" --version 2>/dev/null)
    log "Node.js already installed in .venv: $NODE_VER — skipping."
else
    log "Downloading Node.js v${NODE_VERSION} (portable) into $NODE_DIR ..."

    mkdir -p "$NODE_DIR"

    if [[ "$OS" == "windows" ]]; then
        # ---- Windows: download zip, extract, flatten ----
        TMP_ZIP="/tmp/${NODE_WIN_ZIP}"
        if ! curl -fsSL "$NODE_WIN_URL" -o "$TMP_ZIP"; then
            error "Failed to download Node.js. Check your internet connection.\n  URL: $NODE_WIN_URL"
        fi

        log "Extracting Node.js zip..."
        # unzip is available in Git for Windows
        if command -v unzip &>/dev/null 2>&1; then
            unzip -q "$TMP_ZIP" -d "/tmp/node_extract"
        else
            error "unzip not found. Install Git for Windows (includes unzip) or add it to PATH."
        fi

        EXTRACTED_DIR=$(ls -d /tmp/node_extract/node-v*/ 2>/dev/null | head -1)
        if [[ -z "$EXTRACTED_DIR" ]]; then
            error "Could not locate extracted Node.js directory."
        fi

        # Move contents into $NODE_DIR (flatten one level)
        cp -r "$EXTRACTED_DIR"* "$NODE_DIR/"
        rm -rf "/tmp/node_extract" "$TMP_ZIP"

        NODE_BIN="$NODE_DIR/node.exe"
        NPM_BIN="$NODE_DIR/npm.cmd"

        # Create shims in Scripts/ so `node` / `npm` work while venv is active
        cat > "$SHIM_DIR/node" << SHIM
#!/bin/bash
exec "$NODE_BIN" "\$@"
SHIM
        cat > "$SHIM_DIR/npm" << SHIM
#!/bin/bash
exec "$NPM_BIN" "\$@"
SHIM
        chmod +x "$SHIM_DIR/node" "$SHIM_DIR/npm"

    elif [[ "$OS" == "linux" || "$OS" == "macos" ]]; then
        # ---- Linux / macOS: download tar.xz, extract, flatten ----
        [[ "$OS" == "macos" ]] && NODE_LINUX_TAR="node-v${NODE_VERSION}-darwin-x64.tar.gz" \
                                && NODE_LINUX_URL="https://nodejs.org/dist/v${NODE_VERSION}/${NODE_LINUX_TAR}"

        TMP_TAR="/tmp/${NODE_LINUX_TAR}"
        if ! curl -fsSL "$NODE_LINUX_URL" -o "$TMP_TAR"; then
            error "Failed to download Node.js. Check your internet connection.\n  URL: $NODE_LINUX_URL"
        fi

        log "Extracting Node.js archive..."
        mkdir -p /tmp/node_extract
        tar -xf "$TMP_TAR" -C /tmp/node_extract --strip-components=1

        cp -r /tmp/node_extract/* "$NODE_DIR/"
        rm -rf /tmp/node_extract "$TMP_TAR"

        NODE_BIN="$NODE_DIR/bin/node"
        NPM_BIN="$NODE_DIR/bin/npm"

        # Symlink into venv bin/ so they're on PATH while venv is active
        ln -sf "$NODE_BIN" "$SHIM_DIR/node"
        ln -sf "$NPM_BIN"  "$SHIM_DIR/npm"

    else
        warn "Unrecognised OS '$OS'. Skipping automatic Node.js install."
        warn "Manually copy a Node.js 10+ binary to $NODE_DIR and add it to PATH."
    fi

    # Final check
    if check_node "$NODE_BIN"; then
        log "Node.js installed: $("$NODE_BIN" --version) at $NODE_BIN ✅"
        log "npm:               $("$SHIM_DIR/npm" --version 2>/dev/null)"
    else
        warn "Node.js install may have failed — verify manually: $NODE_BIN --version"
    fi
fi

# =============================================================================
# 3. CUDA PRE-FLIGHT CHECK
# =============================================================================
# IMPORTANT: setup.sh CANNOT install the CUDA toolkit or GPU drivers for you.
# These are Windows system-level installers that must be done manually.
# This step checks what you have and tells you exactly what to do if anything
# is missing before attempting the PyTorch install.
# =============================================================================
header "CUDA Pre-Flight Check"

REQUIRED_DRIVER_MIN=570
CUDA_READY=true

# --- Check nvidia-smi (confirms GPU driver is installed) ---
if command -v nvidia-smi &>/dev/null 2>&1; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d'.' -f1)
    log "NVIDIA driver found: ${DRIVER_VERSION}"

    if [[ -n "$DRIVER_MAJOR" && "$DRIVER_MAJOR" -ge "$REQUIRED_DRIVER_MIN" ]]; then
        log "Driver version OK (>= ${REQUIRED_DRIVER_MIN}.x required for CUDA 13.2) ✅"
    else
        warn "Driver version ${DRIVER_VERSION} is too old for CUDA 13.2."
        warn "Minimum required: ${REQUIRED_DRIVER_MIN}.x"
        warn "Update your driver at: https://www.nvidia.com/drivers"
        warn "  Select: RTX 3070 Ti / Windows 11 / Game Ready Driver"
        CUDA_READY=false
    fi
else
    warn "nvidia-smi not found — NVIDIA driver may not be installed or not in PATH."
    warn "Install the latest Game Ready Driver for your RTX 3070 Ti:"
    warn "  https://www.nvidia.com/drivers"
    warn "After installing, restart Git Bash and rerun setup.sh."
    CUDA_READY=false
fi

# --- Check CUDA Toolkit (nvcc) ---
# nvcc is NOT required for PyTorch — the cu132 wheel bundles its own CUDA runtime.
# You only need the toolkit if building custom CUDA extensions in the future.
if command -v nvcc &>/dev/null 2>&1; then
    NVCC_RAW=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | tr -d ',')
    log "CUDA Toolkit found: ${NVCC_RAW} ✅"
    log "  (Not required for PyTorch — bundled runtime will be used)"
else
    log "CUDA Toolkit (nvcc) not found — this is OK for PyTorch."
    log "  PyTorch cu132 wheel includes its own CUDA runtime libraries."
    log "  You only need the toolkit later if building custom CUDA extensions."
    log "  If needed: https://developer.nvidia.com/cuda-downloads"
    log "    -> Windows -> x86_64 -> 11 -> exe (local) -> CUDA 13.2"
fi

# --- Abort or warn if driver check failed ---
if [[ "$CUDA_READY" == false ]]; then
    echo ""
    warn "============================================================"
    warn "  GPU driver check failed."
    warn "  PyTorch will install but CUDA will NOT be available."
    warn "  Training will run on CPU only — extremely slow."
    warn "  Fix your drivers and rerun setup.sh for full GPU support."
    warn "============================================================"
    echo ""
    read -p "Continue without GPU support? (y/n): " cuda_confirm
    if [[ "$cuda_confirm" != "y" ]]; then
        error "Aborted. Fix GPU drivers and rerun setup.sh."
    fi
fi

# =============================================================================
# 3b. INSTALL PYTORCH (stable + CUDA 13.2 runtime bundled)
# =============================================================================
header "Installing PyTorch with CUDA 13.2"

# Must install via PyTorch's index-url — PyPI only has CPU wheels.
# The cu132 wheel bundles its own CUDA runtime (no system toolkit needed).
TORCH_CUDA_TAG="cu132"
TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"

if python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" &>/dev/null 2>&1; then
    log "PyTorch with CUDA already installed: $(python -c 'import torch; print(torch.__version__)'). Skipping."
else
    log "Installing PyTorch stable with CUDA 13.2 bundled runtime..."
    log "This downloads ~2GB — may take several minutes."
    pip install \
        torch \
        torchvision \
        --index-url "${TORCH_INDEX_URL}"

    if ! python -c "import torch" &>/dev/null 2>&1; then
        error "PyTorch install failed. Check your internet connection and retry."
    fi
    log "PyTorch installed: $(python -c 'import torch; print(torch.__version__)')"
fi

# Verify CUDA is reachable via PyTorch
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [[ "$CUDA_AVAILABLE" == "True" ]]; then
    CUDA_VER=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    log "CUDA accessible via PyTorch ✅"
    log "  CUDA runtime: ${CUDA_VER}"
    log "  GPU:          ${GPU_NAME}"
else
    warn "CUDA not accessible in PyTorch after install."
    warn "If your driver is up to date, try restarting Git Bash and running:"
    warn "  python -c \"import torch; print(torch.cuda.is_available())\""
fi

# =============================================================================
# 3c. REMAINING PYTHON DEPENDENCIES
# =============================================================================
header "Installing Remaining Dependencies"

log "Installing numpy, poke-env, and utilities..."
pip install \
    "numpy==2.4.6" \
    "poke-env>=0.14.0" \
    "python-dotenv>=1.0.0" \
    "aiohttp>=3.9.0" \
    "requests>=2.33.0" \
    "pytest>=8.0.0" \
    "black>=24.0.0" \
    "flake8>=7.0.0"

log "Dependencies installed."

# =============================================================================
# 4. CREATE .env FILE IF MISSING
# =============================================================================
header "Setting Up Environment Config"

if [[ -f ".env" ]]; then
    log ".env already exists. Skipping."
else
    cat > .env << 'EOF'
# =============================================
# ShowdownProject Bot Configuration
# DO NOT COMMIT THIS FILE
# =============================================

# Pokemon Showdown Connection
WEBSOCKET_URI=wss://sim.smogon.com/showdown/websocket
PS_USERNAME=YourBotUsername
PS_PASSWORD=YourBotPassword
PS_AVATAR=cynthia

# Battle Config
POKEMON_FORMAT=gen9championsvgc2026regma
REGULATION=reg_ma
TEAM_NAME=regma_team1

# Bot Mode: ACCEPT_CHALLENGE | LADDER | CHALLENGE_USER
BOT_MODE=ACCEPT_CHALLENGE
TARGET_USERNAME=

# AI Config
# PHASE: heuristic | nn | mcts
# Start with heuristic, swap to nn once network is trained, mcts for Phase 2
AI_PHASE=heuristic
MCTS_ROLLOUTS=800
MCTS_TIME_LIMIT_SEC=8
MODEL_CHECKPOINT=models/latest.pt

# Logging
LOG_LEVEL=INFO
EOF
    warn ".env created — fill in PS_USERNAME and PS_PASSWORD before running."
fi

# =============================================================================
# 5. CREATE / UPDATE .gitignore
# =============================================================================
header "Setting Up .gitignore"

REQUIRED_ENTRIES=(".venv/" ".env" "__pycache__/" "*.pyc" "logs/" "models/*.pt" "models/*.pth")

if [[ -f ".gitignore" ]]; then
    log ".gitignore exists. Checking for missing entries..."
    for entry in "${REQUIRED_ENTRIES[@]}"; do
        if ! grep -qF "$entry" .gitignore; then
            echo "$entry" >> .gitignore
            log "  Added: $entry"
        fi
    done
else
    cat > .gitignore << 'EOF'
# Python venv
.venv/

# Credentials — NEVER COMMIT
.env

# Python artifacts
*.pyc
*.pyo
__pycache__/
*.egg-info/
dist/
build/
.pytest_cache/
.mypy_cache/

# Logs
logs/
*.log

# Trained models
models/*.pt
models/*.pth
models/*.ckpt

# OS
.DS_Store
Thumbs.db
desktop.ini
EOF
    log ".gitignore created."
fi

# =============================================================================
# 6. VS CODE SETTINGS
# =============================================================================
header "Setting Up VS Code"

mkdir -p .vscode

if [[ -f ".vscode/settings.json" ]]; then
    log ".vscode/settings.json already exists. Skipping."
else
    cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
    "python.terminal.activateEnvironment": true,
    "terminal.integrated.defaultProfile.windows": "Git Bash",
    "editor.formatOnSave": true,
    "editor.tabSize": 4,
    "python.analysis.typeCheckingMode": "basic",
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".venv": true
    },
    "search.exclude": {
        ".venv": true
    }
}
EOF
    log ".vscode/settings.json created."
fi

# =============================================================================
# 7. FINAL SUMMARY
# =============================================================================
header "Setup Complete"

echo ""
echo -e "${GREEN}✅ Python:${NC}          $(python --version) — $(which python)"
echo -e "${GREEN}✅ Virtual env:${NC}     $VENV_DIR"
if check_node "$NODE_BIN" 2>/dev/null || check_node "node" 2>/dev/null; then
    echo -e "${GREEN}✅ Node.js:${NC}         $(node --version 2>/dev/null) — $NODE_DIR"
else
    echo -e "${YELLOW}⚠️  Node.js:${NC}         not found in .venv — check install logs above"
fi
echo -e "${GREEN}✅ PyTorch:${NC}         $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [[ "$CUDA_OK" == "True" ]]; then
    echo -e "${GREEN}✅ CUDA:${NC}            $(python -c 'import torch; print(torch.version.cuda)') — $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
else
    echo -e "${YELLOW}⚠️  CUDA:${NC}            Not available — check NVIDIA drivers"
fi
echo -e "${GREEN}✅ numpy:${NC}           $(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null)"
echo -e "${GREEN}✅ Dependencies:${NC}    installed"
echo -e "${GREEN}✅ .env:${NC}            ready — fill in credentials"
echo -e "${GREEN}✅ .gitignore:${NC}      configured"
echo -e "${GREEN}✅ VS Code:${NC}         .vscode/settings.json ready"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Edit .env — fill in PS_USERNAME and PS_PASSWORD"
echo "  2. Open VS Code:  code ."
echo "  3. VS Code terminal auto-activates Python venv"
echo "  4. Scaffold src/bot/player.py  — VGCBot(Player) subclass"
echo "  5. Scaffold src/bot/embedder.py — embed_battle() state encoding"
echo "  6. Run the bot:   python run.py"
echo ""
echo -e "${BLUE}Happy building! — Victory-Dance VGC Bot${NC}"
