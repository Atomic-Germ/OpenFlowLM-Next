cmake_minimum_required(VERSION 3.22)
project(gemma_embedding VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
find_package(Threads REQUIRED)
find_path(XRT_INCLUDE_DIR xrt/xrt_bo.h
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

add_executable(test_gemma_embedding
    test.cpp
    "${CMAKE_CURRENT_LIST_DIR}/../../open_embedding/engine.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../../common/utils.cpp"
)

target_include_directories(test_gemma_embedding PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../.."
    "${CMAKE_CURRENT_LIST_DIR}/../../include"
    "${XRT_INCLUDE_DIR}"
)

target_compile_definitions(test_gemma_embedding PRIVATE
    DISABLE_ABI_CHECK=1
    CMAKE_INSTALL_PREFIX="${CMAKE_BINARY_DIR}"
    CMAKE_XCLBIN_PREFIX="${CMAKE_CURRENT_LIST_DIR}/../../xclbins"
    __FLM_VERSION__="test"
    __NPU_VERSION__="test"
)

target_link_libraries(test_gemma_embedding PRIVATE
    Boost::program_options
    Threads::Threads
    tokenizers_cpp
)

# Add test target
add_custom_target(test_gemma_embedding_target
    DEPENDS test_gemma_embedding
    COMMENT "Building test_gemma_embedding executable"
)
