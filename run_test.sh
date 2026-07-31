#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Remove existing test output files (if any)
rm -f "$SCRIPT_DIR/test/test_Mat_QR.txt" "$SCRIPT_DIR/test/test_Mat_SVDDecomp.txt" "$SCRIPT_DIR/test/test_Tensor3D.txt" "$SCRIPT_DIR/test/test_TT_base.txt" "$SCRIPT_DIR/test/test_MPS_base.txt" "$SCRIPT_DIR/test/test_compress_svd.txt" "$SCRIPT_DIR/test/test_mpo.txt"

cd "$SCRIPT_DIR"

# Run tests in parallel and redirect output
./build/tq_test_test_Mat_QR       > "$SCRIPT_DIR/test/test_Mat_QR.txt"      2>&1 &
./build/tq_test_test_Mat_SVDDecomp > "$SCRIPT_DIR/test/test_Mat_SVDDecomp.txt" 2>&1 &
./build/tq_test_test_Tensor3D     > "$SCRIPT_DIR/test/test_Tensor3D.txt"    2>&1 &
./build/tq_test_test_tt_base      > "$SCRIPT_DIR/test/test_TT_base.txt"     2>&1 &
./build/tq_test_test_mps          > "$SCRIPT_DIR/test/test_MPS_base.txt"    2>&1 &
./build/tq_test_test_compress_svd  > "$SCRIPT_DIR/test/test_compress_svd.txt" 2>&1 &
./build/tq_test_test_mpo          > "$SCRIPT_DIR/test/test_mpo.txt"         2>&1 &

# Wait for all background processes to finish
wait
echo "All tests done. Output saved to individual files."