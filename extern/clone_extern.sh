#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EIGEN_DIR="$SCRIPT_DIR/eigen"
PYBIND_DIR="$SCRIPT_DIR/pybind11"

echo "==> Cloning Eigen..."
git clone https://gitlab.com/libeigen/eigen.git "$EIGEN_DIR"

echo ""
echo "Eigen cloned to:"
echo "$EIGEN_DIR"

git clone https://github.com/pybind/pybind11.git "$PYBIND_DIR"

echo ""
echo "pybind11 cloned to:"
echo "$PYBIND_DIR"

echo "==> Done!"