#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QD_DIR="$SCRIPT_DIR/extern/qd"
QD_INSTALL_DIR="$SCRIPT_DIR/extern/qd-install"

echo "==> Initializing submodules..."
git -C "$SCRIPT_DIR" submodule update --init --recursive

echo "==> Building QD..."
cd "$QD_DIR"
autoreconf -fi
./configure --prefix="$QD_INSTALL_DIR"
make -j"$(nproc)"
make install

echo "==> QD installed to: $QD_INSTALL_DIR"
echo "==> Done!"