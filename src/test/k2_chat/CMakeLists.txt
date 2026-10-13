cmake_minimum_required(VERSION 3.22)
# Wired into the root build (src/test/CMakeLists.txt adds this directory) as
# well as configured standalone; only the standalone case declares a project.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(k2_chat VERSION 1.0.0 LANGUAGES CXX)
endif()

# Header-only parser test (no NPU, XRT or model), so it skips ../CMakeLists.txt's npu_test_setup().

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(test_k2_chat test.cpp)
target_include_directories(test_k2_chat PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../../include")
if(MSVC)
    target_compile_options(test_k2_chat PRIVATE /utf-8 /EHsc)
endif()

enable_testing()
add_test(NAME k2_chat COMMAND test_k2_chat)
