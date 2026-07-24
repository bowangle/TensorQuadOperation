cmake_minimum_required(VERSION 3.16)
project(TensorQuadOperation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- dependency: tci_quad manages itself (transitively QTgrid, numeric-type) -
add_subdirectory(extern/tci_quad)

# ---- own paths ----------------------------------------------------------------
set(PROJECT_INCLUDE_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- interface target: carries our headers + transitive deps ------------------
add_library(tensorquad_deps INTERFACE)
target_include_directories(tensorquad_deps INTERFACE ${PROJECT_INCLUDE_DIR})
target_link_libraries(tensorquad_deps INTERFACE tci_quad_deps)

# ---- test targets -------------------------------------------------------------
set(TEST_TARGETS
    Test1
    Test_Tensor3D
    Test_Mat_QR
    Test_Mat_SVDDecomp
    Test_TT_base
    Test_MPS_base
    Test_compress_svd
    Test_MPO
)

add_executable(Test1                    test/test.cpp)
add_executable(Test_Tensor3D            test/test_Tensor3D.cpp)
add_executable(Test_Mat_QR              test/test_Mat_QR.cpp)
add_executable(Test_Mat_SVDDecomp       test/test_Mat_SVDDecomp.cpp)
add_executable(Test_TT_base             test/test_tt_base.cpp)
add_executable(Test_MPS_base            test/test_mps.cpp)
add_executable(Test_compress_svd        test/test_compress_svd.cpp)
add_executable(Test_MPO                 test/test_mpo.cpp)

foreach(_tgt IN LISTS TEST_TARGETS)
    target_link_libraries(${_tgt} PRIVATE tensorquad_deps)
    target_compile_options(${_tgt} PRIVATE -Wall -Wextra -Wreorder -Wshadow)
endforeach()
