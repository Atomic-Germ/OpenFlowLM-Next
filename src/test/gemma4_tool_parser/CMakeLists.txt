cmake_minimum_required(VERSION 3.22)
project(gemma4_tool_parser VERSION 1.0.0 LANGUAGES CXX)

# Header-only parser test: no NPU, no XRT, no model. Configure and run with
#   cmake -S src/test/gemma4_tool_parser -B src/build/test_gemma4_tool_parser -G Ninja
#   cmake --build src/build/test_gemma4_tool_parser
#   ctest --test-dir src/build/test_gemma4_tool_parser --output-on-failure

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/test/gemma4_tool_parser)

add_executable(test_gemma4_tool_parser test.cpp)
target_include_directories(test_gemma4_tool_parser PRIVATE ${CMAKE_SOURCE_DIR}/../../include)
if(MSVC)
    target_compile_options(test_gemma4_tool_parser PRIVATE /utf-8 /EHsc)
endif()

enable_testing()
add_test(NAME gemma4_tool_parser COMMAND test_gemma4_tool_parser)
