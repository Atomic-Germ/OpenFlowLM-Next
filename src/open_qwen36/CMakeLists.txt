# open_qwen36_cli — the open Qwen3.6-MoE engine driven without the OFLM app
# (see cli.cpp). The engine's core (core.cpp, pools.cpp, manifest.cpp,
# q4nx_file.cpp) is also compiled into oflm by src/CMakeLists.txt; engine.cpp
# (the causal_lm adapter) only there. manifest_test (no XRT) is the
# OPEN-MANIFEST unit test: `ctest` runs it against the checked-in fixture.
#
# Standalone:  cmake -S . -B build && cmake --build build && ctest --test-dir build
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

set(_inc
    "${CMAKE_CURRENT_SOURCE_DIR}/.."            # open_qwen36/*.hpp
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"    # nlohmann/json.hpp
    "${CMAKE_CURRENT_SOURCE_DIR}/../../open_kernels/harness"  # stream_patch.hpp
)

add_executable(open_qwen36_cli cli.cpp core.cpp pools.cpp manifest.cpp q4nx_file.cpp)
target_include_directories(open_qwen36_cli PRIVATE "${XRT_INCLUDE_DIR}" ${_inc})
target_link_directories(open_qwen36_cli PRIVATE "${XRT_LIB_DIR}")
target_link_libraries(open_qwen36_cli PRIVATE xrt_coreutil)

add_executable(manifest_test manifest_test.cpp manifest.cpp)
target_include_directories(manifest_test PRIVATE ${_inc})

# vit_test (no XRT): the host vision tower against open_kernels/model/replica_vit.py's fixture
add_executable(vit_test vision/vit_test.cpp vision/vit.cpp q4nx_file.cpp)
target_include_directories(vit_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" ${_inc})
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(vit_test PRIVATE OpenMP::OpenMP_CXX)
endif()

foreach(t open_qwen36_cli manifest_test vit_test)
    if(MSVC)
        target_compile_options(${t} PRIVATE /EHsc /Zc:__cplusplus /bigobj)
        target_compile_definitions(${t} PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
endforeach()

enable_testing()
add_test(NAME OPEN-MANIFEST COMMAND manifest_test
         "${CMAKE_CURRENT_SOURCE_DIR}/../../specs/open-engine/tests/fixtures/manifest_qwen36.json"
         "${CMAKE_CURRENT_SOURCE_DIR}/../../specs/open-engine/tests/fixtures/manifest_qwen3_4b.json"
         "${CMAKE_CURRENT_SOURCE_DIR}/../../specs/open-engine/tests/fixtures/manifest_gemma3_4b.json"
         "${CMAKE_CURRENT_SOURCE_DIR}/../../specs/open-engine/tests/fixtures/manifest_hy_mt2_7b.json"
         "${CMAKE_CURRENT_SOURCE_DIR}/../../specs/open-engine/tests/fixtures/manifest_qwen35_9b.json"
         "${CMAKE_CURRENT_SOURCE_DIR}/../../specs/open-engine/tests/fixtures/manifest_phi4_mini_4b.json")
