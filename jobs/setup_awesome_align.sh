#!/usr/bin/env bash
# Install awesome-align and its dependencies into a dedicated venv.
# Run once from the project root:
#   bash jobs/setup_awesome_align.sh

set -euo pipefail
REPO_DIR="vendor/awesome-align"
VENV_DIR="vendor/awesome-align-venv"

echo "=== 1. Cloning awesome-align ==="
if [ -d "$REPO_DIR" ]; then
    echo "  Already cloned at $REPO_DIR — skipping."
else
    git clone https://github.com/neulab/awesome-align.git "$REPO_DIR"
fi

echo "=== 2. Creating virtual environment at $VENV_DIR ==="
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "=== 3. Installing PyTorch (MPS-enabled for Apple Silicon) ==="
pip install --upgrade pip
pip install torch torchvision torchaudio

echo "=== 4. Installing awesome-align ==="
pip install -e "$REPO_DIR"

echo "=== 5. Verifying installation ==="
awesome-align --help > /dev/null && echo "  awesome-align: OK"
awesome-train  --help > /dev/null && echo "  awesome-train:  OK"

deactivate
echo ""
echo "Done. Activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
