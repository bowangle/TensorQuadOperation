#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XFAC_QUAD_RUNNER_DIR="$SCRIPT_DIR/extern/xfac_quad_runner"

echo "==> Initializing submodules..."
git -C "$SCRIPT_DIR" submodule update --init --recursive

echo "==> Building xfac_quad_runner:"
cd "$XFAC_QUAD_RUNNER_DIR"
bash install_extern.sh

echo "==> Done!"