#!/bin/bash
# Remove existing test files (if any)
rm -f test/test_Mat_QR.txt test/test_Mat_SVDDecomp.txt test/test_Tensor3D.txt test/test_TT_base.txt test/test_MPS_base.txt test/test_compress_svd.txt
# Run tests in parallel and redirect output
./build/Test_Mat_QR      > test/test_Mat_QR.txt      2>&1 &
./build/Test_Mat_SVDDecomp > test/test_Mat_SVDDecomp.txt 2>&1 &
./build/Test_Tensor3D    > test/test_Tensor3D.txt    2>&1 &
./build/Test_TT_base     > test/test_TT_base.txt     2>&1 &
./build/Test_MPS_base    > test/test_MPS_base.txt    2>&1 &
./build/Test_compress_svd > test/test_compress_svd.txt 2>&1 &
# Wait for all background processes to finish
wait
echo "All tests done. Output saved to individual files."