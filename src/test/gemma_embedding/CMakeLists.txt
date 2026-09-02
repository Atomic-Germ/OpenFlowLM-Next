cmake_minimum_required(VERSION 3.22)
# Standalone open-engine embedding test.
#
# This target intentionally does NOT use test/CMakeLists.txt: that helper always
# links the closed q4_npu_eXpress/mha/dequant/gemm/lm_head stack. The open
# engine only needs XRT for the shared typed-buffer headers.
project(gemma_embedding VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
find_package(Threads REQUIRED)

# XRT is required only for the shared typed-buffer headers, not for NPU
# dispatch. Discover it the same way the main build does (pkg-config first,
# then the documented /opt/xilinx/xrt fallback, then XILINX_XRT).
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(XRT QUIET xrt)
endif()

if(XRT_FOUND)
    set(XRT_INCLUDE_DIR "${XRT_INCLUDE_DIRS}")
    set(XRT_LIBRARY_DIR "${XRT_LIBRARY_DIRS}")
else()
    if(DEFINED ENV{XILINX_XRT} AND NOT "$ENV{XILINX_XRT}" STREQUAL "")
        set(XRT_PREFIX "$ENV{XILINX_XRT}")
    else()
        set(XRT_PREFIX "/opt/xilinx/xrt")
    endif()
    set(XRT_INCLUDE_DIR "${XRT_PREFIX}/include")
    set(XRT_LIBRARY_DIR "${XRT_PREFIX}/lib")
endif()

find_path(XRT_HEADER_DIR xrt/xrt_bo.h
    HINTS "${XRT_INCLUDE_DIR}" "${XRT_PREFIX}"
    PATHS /opt/xilinx/xrt
    PATH_SUFFIXES include
    REQUIRED
)
find_library(XRT_COREUTIL_LIBRARY NAMES xrt_coreutil
    HINTS "${XRT_LIBRARY_DIR}" "${XRT_PREFIX}"
    PATHS /opt/xilinx/xrt
    PATH_SUFFIXES lib lib64
    REQUIRED
)

add_subdirectory(
    "${CMAKE_CURRENT_LIST_DIR}/../../../third_party/tokenizers-cpp"
    "${CMAKE_BINARY_DIR}/tokenizers-cpp"
    EXCLUDE_FROM_ALL
)

add_executable(test_gemma_embedding
    test.cpp
    "${CMAKE_CURRENT_LIST_DIR}/../../open_embedding/engine.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../../common/utils.cpp"
)

target_include_directories(test_gemma_embedding PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../.."
    "${CMAKE_CURRENT_LIST_DIR}/../../include"
    "${XRT_HEADER_DIR}"
)

target_compile_definitions(test_gemma_embedding PRIVATE
    DISABLE_ABI_CHECK=1
    USEAVX2=1
    CMAKE_INSTALL_PREFIX="${CMAKE_BINARY_DIR}"
    CMAKE_XCLBIN_PREFIX="${CMAKE_CURRENT_LIST_DIR}/../../xclbins"
    __FLM_VERSION__="test"
    __NPU_VERSION__="test"
)

target_compile_options(test_gemma_embedding PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2 /O2 /fp:fast>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mavx2 -mfma -O2>
)

target_link_libraries(test_gemma_embedding PRIVATE
    Boost::program_options
    Threads::Threads
    tokenizers_cpp
    "${XRT_COREUTIL_LIBRARY}"
)

# Add test target
add_custom_target(test_gemma_embedding_target
    DEPENDS test_gemma_embedding
    COMMENT "Building test_gemma_embedding executable"
)
