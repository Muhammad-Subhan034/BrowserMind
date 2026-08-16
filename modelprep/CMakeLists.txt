cmake_minimum_required(VERSION 3.16)
project(modelprep CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra)
endif()

add_library(modelprep_core STATIC
    src/raw_io.cpp
    src/quantize.cpp
    src/pack.cpp
    src/validate.cpp
)
target_include_directories(modelprep_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(modelprep src/main.cpp)
target_link_libraries(modelprep PRIVATE modelprep_core)

enable_testing()
add_executable(modelprep_tests tests/test_quantize.cpp)
target_link_libraries(modelprep_tests PRIVATE modelprep_core)
add_test(NAME modelprep_tests COMMAND modelprep_tests)
