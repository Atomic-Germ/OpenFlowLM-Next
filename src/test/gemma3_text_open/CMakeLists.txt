cmake_minimum_required(VERSION 3.22)
# Standalone open Gemma3 text engine validation.
#
# This target intentionally does NOT use test/CMakeLists.txt: that helper always
# links the closed q4_npu_eXpress/mha/dequant/gemm/lm_head stack. The open
# engine is CPU-only at this stage and needs none of it.
project(gemma3_text_open VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
find_package(Threads REQUIRED)

# utils.hpp -> typedef.hpp -> buffer.hpp -> device_runtime.hpp pulls in XRT
# types, so even this CPU-only target needs the XRT headers to compile the
# shared utility code. No XRT library is linked and no NPU is used.
find_path(XRT_HEADER_DIR xrt/xrt_bo.h
    HINTS "$ENV{XILINX_XRT}"
    PATHS /opt/xilinx/xrt
    PATH_SUFFIXES include
    REQUIRED
)

add_subdirectory(
    "${CMAKE_CURRENT_LIST_DIR}/../../../third_party/tokenizers-cpp"
    "${CMAKE_BINARY_DIR}/tokenizers-cpp"
    EXCLUDE_FROM_ALL
)

add_executable(test_gemma3_text_open
    test.cpp
    "${CMAKE_CURRENT_LIST_DIR}/../../open_gemma3/engine.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../../common/utils.cpp"
)

target_include_directories(test_gemma3_text_open PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../.."
    "${CMAKE_CURRENT_LIST_DIR}/../../include"
    "${XRT_HEADER_DIR}"
)

target_compile_definitions(test_gemma3_text_open PRIVATE
    DISABLE_ABI_CHECK=1
    USEAVX2=1
    CMAKE_INSTALL_PREFIX="${CMAKE_BINARY_DIR}"
    # utils.cpp (find_xclbin_path) requires this; the engine never uses it.
    CMAKE_XCLBIN_PREFIX="${CMAKE_CURRENT_LIST_DIR}/../../xclbins"
    __FLM_VERSION__="test"
    __NPU_VERSION__="test"
)

target_compile_options(test_gemma3_text_open PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2 /O2>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mavx2 -mfma -O2>
)

target_link_libraries(test_gemma3_text_open PRIVATE
    Boost::program_options
    Threads::Threads
    tokenizers_cpp
)
