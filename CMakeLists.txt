cmake_minimum_required(VERSION 3.5)

project(TCIcpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(EIGEN_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/extern/eigen)

add_executable(Test1
    test/test.cpp
)

add_executable(Test_Tensor3D
    test/test_Tensor3D.cpp
)

target_include_directories(Test1 PUBLIC
    ${PROJECT_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
)

target_include_directories(Test_Tensor3D PUBLIC
    ${PROJECT_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
)

target_link_libraries(Test1 PRIVATE quadmath)
target_link_libraries(Test_Tensor3D PRIVATE quadmath)