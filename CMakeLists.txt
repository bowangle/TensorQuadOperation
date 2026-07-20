cmake_minimum_required(VERSION 3.5)
project(TCIcpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)

# rename for shorter path:
set(TCI_QUAD_DIR            ${CMAKE_SOURCE_DIR}/extern/tci_quad)
set(NUMERIC_TYPE_DIR        ${TCI_QUAD_DIR}/extern/numeric-type-quad)

# include project:
set(PROJECT_INCLUDE_DIR     ${CMAKE_SOURCE_DIR}/include)
set(TYPING_INCLUDE_DIR      ${NUMERIC_TYPE_DIR}/include)
set(TCI_QUAD_INCLUDE_DIR    ${TCI_QUAD_DIR}/include)

# include dependance:
set(EIGEN_INCLUDE_DIR       ${NUMERIC_TYPE_DIR}/extern/eigen)
set(QD_INCLUDE_DIR          ${NUMERIC_TYPE_DIR}/extern/qd-install/include)
set(QD_LIBRARY_DIR          ${NUMERIC_TYPE_DIR}/extern/qd-install/lib)

set(JSON_INCLUDE_DIR        ${TCI_QUAD_DIR}/extern/nlohmann_json/include)
set(spdlog_INCLUDE_DIR      ${TCI_QUAD_DIR}/extern/spdlog/include)

add_executable(Test1                    test/test.cpp)
add_executable(Test_Tensor3D            test/test_Tensor3D.cpp)
add_executable(Test_Mat_QR              test/test_Mat_QR.cpp)
add_executable(Test_Mat_SVDDecomp       test/test_Mat_SVDDecomp.cpp)
add_executable(Test_TT_base             test/test_tt_base.cpp)
add_executable(Test_MPS_base            test/test_mps.cpp)
add_executable(Test_compress_svd        test/test_compress_svd.cpp)
add_executable(Test_MPO                 test/test_mpo.cpp)

# ---- include dirs (same for all targets) ----------------------------------------
foreach(_tgt Test1 Test_Tensor3D Test_Mat_QR Test_Mat_SVDDecomp Test_TT_base Test_MPS_base Test_compress_svd Test_MPO)
    target_include_directories(${_tgt} PUBLIC
        ${PROJECT_INCLUDE_DIR}
        ${EIGEN_INCLUDE_DIR}
        ${QD_INCLUDE_DIR}
        ${TCI_QUAD_INCLUDE_DIR}
        ${JSON_INCLUDE_DIR}
        ${spdlog_INCLUDE_DIR}
        ${TYPING_INCLUDE_DIR}
    )
    target_link_directories(${_tgt} PRIVATE ${QD_LIBRARY_DIR})
    target_link_libraries(${_tgt} PRIVATE qd quadmath OpenMP::OpenMP_CXX)
    target_compile_options(${_tgt} PRIVATE -Wall -Wextra -Wreorder -Wshadow)
endforeach()