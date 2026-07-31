cmake_minimum_required(VERSION 3.21)
project(TensorQuadOperation LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TQ_BUILD_TESTS
       "Build TensorQuadOperation's own tests"
       ${PROJECT_IS_TOP_LEVEL})

# ---- dependencies -------------------------------------------------------------
find_package(OpenMP REQUIRED)

# xfac_quad_runner manages itself (and transitively xfac_quad, QTgrid-quad,
# numeric-type-quad, QD, spdlog). Its tests, and those of every layer below it,
# stay off unless the matching <LAYER>_BUILD_TESTS option is explicitly set.
if(NOT TARGET xf_qd_runner_deps)
    add_subdirectory(extern/xfac_quad_runner)
endif()

# ---- own paths ----------------------------------------------------------------
set(TQ_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TQ_TEST_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/test)

# ---- interface target: carries our headers + transitive deps ------------------
add_library(tensorquad_deps INTERFACE)

target_include_directories(tensorquad_deps INTERFACE
    ${TQ_INCLUDE_DIR})

target_compile_features(tensorquad_deps INTERFACE cxx_std_20)

target_link_libraries(tensorquad_deps INTERFACE
    xf_qd_runner_deps
    OpenMP::OpenMP_CXX)

# ---- test-only interface: shared test helpers in test/ ------------------------
# Kept out of tensorquad_deps so consumers never get the test directory on
# their include path. Link this only from test targets.
add_library(tensorquad_testing INTERFACE)

target_include_directories(tensorquad_testing INTERFACE
    ${TQ_TEST_DIR})

# ---- tests (leaf targets: nothing may ever depend on these) -------------------
if(TQ_BUILD_TESTS)
    enable_testing()

    # One entry per test/<name>.cpp -> target tq_test_<name>
    set(_tq_tests
        test
        test_Tensor3D
        test_Mat_QR
        test_Mat_SVDDecomp
        test_tt_base
        test_mps
        test_compress_svd
        test_mpo)

    foreach(_name IN LISTS _tq_tests)
        add_executable(tq_test_${_name} test/${_name}.cpp)
        target_link_libraries(tq_test_${_name}
            PRIVATE tensorquad_deps tensorquad_testing)
        target_compile_options(tq_test_${_name} PRIVATE
            -Wall -Wextra -Wreorder -Wshadow)
        add_test(NAME tq_test_${_name}
                 COMMAND tq_test_${_name})
    endforeach()
endif()
