# run_kernel — N-buffer opcode-3 runner for the open kernels (see run_kernel.cpp).
#
# Standalone:  cmake -S . -B build && cmake --build build
# Inside flm:  cmake -DFLM_BUILD_OPEN_KERNELS_HARNESS=ON ... (src/CMakeLists.txt)
#
# XRT is located the same way src/CMakeLists.txt does it: XRT_INCLUDE_DIR /
# XRT_LIB_DIR cache vars, else $XILINX_XRT, else /opt/xilinx/xrt.
cmake_minimum_required(VERSION 3.22)
project(open_kernels_harness LANGUAGES CXX)

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

add_executable(run_kernel run_kernel.cpp)
target_include_directories(run_kernel PRIVATE "${XRT_INCLUDE_DIR}")
target_link_directories(run_kernel PRIVATE "${XRT_LIB_DIR}")
target_link_libraries(run_kernel PRIVATE xrt_coreutil)
if(MSVC)
    target_compile_options(run_kernel PRIVATE /EHsc /Zc:__cplusplus)
    target_compile_definitions(run_kernel PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
