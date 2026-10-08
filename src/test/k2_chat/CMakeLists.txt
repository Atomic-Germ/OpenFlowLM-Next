cmake_minimum_required(VERSION 3.22)
project(k2_chat VERSION 1.0.0 LANGUAGES CXX)

# Header-only parser test (no NPU, XRT or model), so it skips ../CMakeLists.txt's npu_test_setup().

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/test/k2_chat)

add_executable(test_k2_chat test.cpp)
target_include_directories(test_k2_chat PRIVATE ${CMAKE_SOURCE_DIR}/../../include)
if(MSVC)
    target_compile_options(test_k2_chat PRIVATE /utf-8 /EHsc)
endif()

enable_testing()
add_test(NAME k2_chat COMMAND test_k2_chat)
