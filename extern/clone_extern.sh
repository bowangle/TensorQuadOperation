#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EIGEN_DIR="$SCRIPT_DIR/eigen"
PYBIND_DIR="$SCRIPT_DIR/pybind11"
QD_DIR="$SCRIPT_DIR/qd"
QD_INSTALL_DIR="$SCRIPT_DIR/qd-install"

echo "==> Cloning Eigen..."
git clone https://gitlab.com/libeigen/eigen.git "$EIGEN_DIR"

echo "Eigen cloned to:"
echo "$EIGEN_DIR"

echo "==> Cloning pybind11..."
git clone https://github.com/pybind/pybind11.git "$PYBIND_DIR"

echo "pybind11 cloned to:"
echo "$PYBIND_DIR"

echo "==> Cloning QD..."
git clone https://github.com/BL-highprecision/QD.git "$QD_DIR"

echo "QD cloned to:"
echo "$QD_DIR"

echo "==> Building QD..."
cd "$QD_DIR"

autoreconf -fi
./configure --prefix="$QD_INSTALL_DIR"
make -j"$(nproc)"
make install

echo "QD installed to:"
echo "$QD_INSTALL_DIR"

echo "==> Done!"