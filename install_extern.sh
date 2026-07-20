#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QD_DIR="$SCRIPT_DIR/extern/qd"
QD_INSTALL_DIR="$SCRIPT_DIR/extern/qd-install"
TCI_QUAD_DIR="$SCRIPT_DIR/extern/tci_quad"
TYPING_DIR="$SCRIPT_DIR/extern/numeric-type-quad"

echo "==> Cleaning extern/ ..."
rm -rf "$SCRIPT_DIR"/extern/*

echo "==> Initializing submodules..."
git -C "$SCRIPT_DIR" submodule update --init --recursive

echo "==> Building TCI quad:"
cd "$TCI_QUAD_DIR"
bash install_extern.sh
# bash compile.sh

echo "==> Done!"