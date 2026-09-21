# Helper-only tests: no NPU, no prebuilt engine library, so this builds and
# runs on Linux CI. The XRT headers are still needed because
# npu_platform.hpp reaches oflm_rt through the device runtime.
cmake_minimum_required(VERSION 3.22)
project(npu_platform_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OFLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

if(WIN32)
    set(XRT_INCLUDE_DIR "C:/dev/XRT/src/runtime_src/core/include"
        CACHE PATH "Where XRT headers live")
else()
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(XRT xrt)
    endif()
    if(XRT_FOUND)
        set(XRT_INCLUDE_DIR "${XRT_INCLUDE_DIRS}")
    else()
        set(XRT_INCLUDE_DIR /opt/xilinx/xrt/include CACHE PATH "Where XRT headers live")
    endif()
endif()

add_executable(test_npu_platform
    test_npu_platform.cpp
    "${OFLM_SOURCE_DIR}/common/npu_platform.cpp"
    "${OFLM_SOURCE_DIR}/common/utils.cpp")
target_include_directories(test_npu_platform PRIVATE
    "${OFLM_SOURCE_DIR}/include"
    "${XRT_INCLUDE_DIR}")
# utils.cpp expects the install-location macros the main build defines.
target_compile_definitions(test_npu_platform PRIVATE
    CMAKE_INSTALL_PREFIX="/opt/openflowlm"
    CMAKE_XCLBIN_PREFIX="/opt/openflowlm/share/oflm/xclbins")

if(WIN32)
    target_link_directories(test_npu_platform PRIVATE "${XRT_INCLUDE_DIR}/../lib")
elseif(XRT_FOUND)
    target_link_directories(test_npu_platform PRIVATE "${XRT_LIBRARY_DIRS}")
else()
    target_link_directories(test_npu_platform PRIVATE /opt/xilinx/xrt/lib)
endif()
target_link_libraries(test_npu_platform PRIVATE xrt_coreutil)

enable_testing()
add_test(NAME test_npu_platform COMMAND test_npu_platform)
