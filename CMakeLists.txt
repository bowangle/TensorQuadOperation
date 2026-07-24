cmake_minimum_required(VERSION 3.16)
project(TCIcpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)

# ---- paths ----------------------------------------------------------------------
# rename for shorter path:
set(TCI_QUAD_DIR            ${CMAKE_CURRENT_SOURCE_DIR}/extern/tci_quad)
set(QTGRID_DIR              ${TCI_QUAD_DIR}/extern/QTgrid-quad)
set(NUMERIC_TYPE_DIR        ${QTGRID_DIR}/extern/numeric-type-quad)

# include project:
set(PROJECT_INCLUDE_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TYPING_INCLUDE_DIR      ${NUMERIC_TYPE_DIR}/include)
set(TCI_QUAD_INCLUDE_DIR    ${TCI_QUAD_DIR}/include)
set(QTGRID_INCLUDE_DIR      ${QTGRID_DIR}/include)

# include dependance:
set(EIGEN_INCLUDE_DIR       ${NUMERIC_TYPE_DIR}/extern/eigen)
set(QD_INCLUDE_DIR          ${NUMERIC_TYPE_DIR}/extern/qd-install/include)
set(QD_LIBRARY_DIR          ${NUMERIC_TYPE_DIR}/extern/qd-install/lib)

set(JSON_INCLUDE_DIR        ${QTGRID_DIR}/extern/json/include)
set(spdlog_INCLUDE_DIR      ${TCI_QUAD_DIR}/extern/spdlog/include)

# ---- sanity check: QD must be built before configuring --------------------------
if(NOT EXISTS ${QD_INCLUDE_DIR}/qd/qd_real.h)
    message(FATAL_ERROR
        "QD not found at ${QD_INCLUDE_DIR}.\n"
        "Run install_extern.sh first to build the QD library.")
endif()

# ---- common dependency bundle ---------------------------------------------------
add_library(project_deps INTERFACE)

target_include_directories(project_deps INTERFACE
    ${PROJECT_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
    ${QD_INCLUDE_DIR}
    ${TCI_QUAD_INCLUDE_DIR}
    ${QTGRID_INCLUDE_DIR}
    ${JSON_INCLUDE_DIR}
    ${spdlog_INCLUDE_DIR}
    ${TYPING_INCLUDE_DIR}
)
target_link_directories(project_deps INTERFACE ${QD_LIBRARY_DIR})
target_link_libraries(project_deps INTERFACE qd quadmath OpenMP::OpenMP_CXX)

# ---- test targets ---------------------------------------------------------------
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
    target_link_libraries(${_tgt} PRIVATE project_deps)
    target_compile_options(${_tgt} PRIVATE -Wall -Wextra -Wreorder -Wshadow)
endforeach()