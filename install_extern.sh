#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XFAC_QUAD_DIR="$SCRIPT_DIR/extern/xfac_quad"

echo "==> Initializing submodules..."
git -C "$SCRIPT_DIR" submodule update --init --recursive

echo "==> Installing xfac_quad extern dependencies:"
cd "$XFAC_QUAD_DIR"
bash install_extern.sh

echo "==> Done!"