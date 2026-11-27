cmake_minimum_required(VERSION 3.21)
project(TensorQuadOperation LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TQ_BUILD_TESTS
       "Build TensorQuadOperation's own tests"
       ${PROJECT_IS_TOP_LEVEL})

# ---- dependencies -------------------------------------------------------------
find_package(OpenMP REQUIRED)

# xfac_quad provides grids, numerical headers, and runners, and manages its
# external dependencies. Its tests stay off unless XFAC_QUAD_BUILD_TESTS is
# explicitly enabled; numeric-type-quad likewise controls its own tests.
if(NOT TARGET xfac_quad_deps)
    add_subdirectory(extern/xfac_quad)
endif()

# ---- own paths ----------------------------------------------------------------
set(TQ_INCLUDE_DIR       ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TQ_TEST_DIR          ${CMAKE_CURRENT_SOURCE_DIR}/test)
set(CATCH2_INCLUDE_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/extern/Catch2/single_include)

# ---- interface target: carries our headers + transitive deps ------------------
add_library(tensorquad_deps INTERFACE)

target_include_directories(tensorquad_deps INTERFACE
    ${TQ_INCLUDE_DIR})

target_compile_features(tensorquad_deps INTERFACE cxx_std_20)

target_link_libraries(tensorquad_deps INTERFACE
    xfac_quad_deps
    OpenMP::OpenMP_CXX)

# ---- test-only interface: shared test helpers in test/ ------------------------
# Kept out of tensorquad_deps so consumers never get the test directory on
# their include path. Link this only from test targets.
add_library(tensorquad_testing INTERFACE)

target_include_directories(tensorquad_testing INTERFACE
    ${TQ_TEST_DIR})

target_include_directories(tensorquad_testing SYSTEM INTERFACE
    ${CATCH2_INCLUDE_DIR})

# ---- FFT/QFT executable -----------------------------------------------------
add_executable(precompute_qft src/fft-tn-quadp/precompute_qft.cpp)
target_link_libraries(precompute_qft PRIVATE tensorquad_deps)
target_compile_options(precompute_qft PRIVATE -Wall -Wextra -Wreorder -Wshadow)

# ---- Quad integration executable -------------------------------------------
add_executable(integration_method src/quad_integration/integration_method.cpp)
target_link_libraries(integration_method PRIVATE tensorquad_deps)
target_compile_options(integration_method PRIVATE -Wall -Wextra -Wreorder -Wshadow)

# ---- tests (leaf targets: nothing may ever depend on these) -------------------
if(TQ_BUILD_TESTS)
    if(NOT EXISTS ${CATCH2_INCLUDE_DIR}/catch2/catch.hpp)
        message(FATAL_ERROR
            "Catch2 not found at ${CATCH2_INCLUDE_DIR}.\n"
            "Run: git submodule update --init --recursive")
    endif()

    enable_testing()

    # Paths are relative to test/; keep the existing tq_test_<name> targets.
    set(_tq_tests
        TensorQuadOperation/test
        TensorQuadOperation/test_Tensor3D
        TensorQuadOperation/test_Mat_QR
        TensorQuadOperation/test_Mat_SVDDecomp
        TensorQuadOperation/test_tt_base
        TensorQuadOperation/test_mps
        TensorQuadOperation/test_ttn
        TensorQuadOperation/test_compress_svd
        TensorQuadOperation/test_mpo
        TensorQuadOperation/test_analytics_construction
        TensorQuadOperation/test_convolution
        QuanticsIntegration/test_quantics_integrator
        QuanticsIntegration/test_quantics_integrator_kD
        QuanticsIntegration/test_indefinite_integral
        QuanticsIntegration/test_simplex_integration)

    foreach(_test IN LISTS _tq_tests)
        get_filename_component(_name "${_test}" NAME)
        set(_target tq_test_${_name})
        add_executable(${_target} test/${_test}.cpp)
        target_link_libraries(${_target}
            PRIVATE tensorquad_deps tensorquad_testing)
        target_compile_options(${_target} PRIVATE
            -Wall -Wextra -Wreorder -Wshadow)
        add_test(NAME ${_target} COMMAND ${_target})
        # Shared fixtures and generated data remain relative to the project root.
        set_tests_properties(${_target} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endforeach()

    set(_fft_tn_tests
        magic_qft_mpo
        general_integrator
        fft_param
        fft_vanilla
        fft_double
        fft_convergence_double
        fft_convergence_dd128
        fft_phase_dd128
        fft_dd128
        fft_float128)

    foreach(_name IN LISTS _fft_tn_tests)
        set(_target fft_tn_test_${_name})
        add_executable(${_target} test/fft-tn-quadp/test_${_name}.cpp)
        target_link_libraries(${_target} PRIVATE tensorquad_deps tensorquad_testing)
        target_compile_options(${_target} PRIVATE -Wall -Wextra -Wreorder -Wshadow -UNDEBUG)
        add_test(NAME ${_target} COMMAND ${_target})
        set_tests_properties(${_target} PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endforeach()
endif()
