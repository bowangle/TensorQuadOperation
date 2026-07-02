#!/bin/bash

./build/Test_Mat_QR      > test/test_Mat_QR.txt      2>&1 &
./build/Test_Mat_SVDDecomp > test/test_Mat_SVDDecomp.txt 2>&1 &
./build/Test_Tensor3D    > test/test_Tensor3D.txt    2>&1 &
./build/Test_TT_base     > test/test_TT_base.txt     2>&1 &

wait
echo "All tests done. Output saved to individual files."