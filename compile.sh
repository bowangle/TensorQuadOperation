#!/bin/bash

# Paths
PROJECT_ROOT=$(pwd)
BUILD_DIR="$PROJECT_ROOT/build"

# Clean old build
rm -rf "$BUILD_DIR"

# Configure CMake (out-of-source build)
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native"\

# cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

# Build everything
cmake --build "$BUILD_DIR" -j 8

cat build/CMakeCache.txt | grep CMAKE_CXX_COMPILER
