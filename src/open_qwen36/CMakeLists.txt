# open_qwen36_cli — the open Qwen3.6-MoE engine driven without the FLM app
# (see cli.cpp). The engine's core (core.cpp, pools.cpp, q4nx_file.cpp) is
# also compiled into flm by src/CMakeLists.txt; engine.cpp (the causal_lm
# adapter) only there.
#
# Standalone:  cmake -S . -B build && cmake --build build
# Windows:     build.cmd (MSVC against the system XRT)
#
# XRT is located the same way open_kernels/harness does it: XRT_INCLUDE_DIR /
# XRT_LIB_DIR cache vars, else $XILINX_XRT, else /opt/xilinx/xrt.
cmake_minimum_required(VERSION 3.22)
project(open_qwen36 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED XRT_INCLUDE_DIR OR NOT DEFINED XRT_LIB_DIR)
    set(_xrt_root "$ENV{XILINX_XRT}")
    if(_xrt_root STREQUAL "")
        set(_xrt_root "/opt/xilinx/xrt")
    endif()
    if(NOT DEFINED XRT_INCLUDE_DIR)
        set(XRT_INCLUDE_DIR "${_xrt_root}/include" CACHE PATH "XRT headers")
    endif()
    if(NOT DEFINED XRT_LIB_DIR)
        set(XRT_LIB_DIR "${_xrt_root}/lib" CACHE PATH "XRT libraries")
    endif()
endif()

add_executable(open_qwen36_cli cli.cpp core.cpp pools.cpp q4nx_file.cpp)
target_include_directories(open_qwen36_cli PRIVATE
    "${XRT_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."            # open_qwen36/*.hpp
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"    # nlohmann/json.hpp
    "${CMAKE_CURRENT_SOURCE_DIR}/../../open_kernels/harness"  # stream_patch.hpp
)
target_link_directories(open_qwen36_cli PRIVATE "${XRT_LIB_DIR}")
target_link_libraries(open_qwen36_cli PRIVATE xrt_coreutil)
if(MSVC)
    target_compile_options(open_qwen36_cli PRIVATE /EHsc /Zc:__cplusplus /bigobj)
    target_compile_definitions(open_qwen36_cli PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
