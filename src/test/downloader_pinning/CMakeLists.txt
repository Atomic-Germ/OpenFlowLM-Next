# Pinning-logic tests: no network, no NPU, so this builds and runs on
# Linux CI. The XRT headers are still needed because model_downloader.hpp
# reaches causal_lm.hpp (via lm_config.hpp) through the device runtime.
cmake_minimum_required(VERSION 3.22)
project(downloader_pinning_tests LANGUAGES CXX)

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

find_package(CURL REQUIRED)

add_executable(test_downloader_pinning
    test_downloader_pinning.cpp
    "${OFLM_SOURCE_DIR}/pull/model_downloader.cpp"
    "${OFLM_SOURCE_DIR}/pull/download_model.cpp"
    "${OFLM_SOURCE_DIR}/common/utils.cpp")
target_include_directories(test_downloader_pinning PRIVATE
    "${OFLM_SOURCE_DIR}/include"
    "${OFLM_SOURCE_DIR}/pull"
    "${XRT_INCLUDE_DIR}"
    "${CURL_INCLUDE_DIRS}")
# utils.cpp expects the install-location macros the main build defines.
# model_downloader.cpp compares against the app version the same way.
target_compile_definitions(test_downloader_pinning PRIVATE
    CMAKE_INSTALL_PREFIX="/opt/openflowlm"
    CMAKE_XCLBIN_PREFIX="/opt/openflowlm/share/oflm/xclbins"
    __OFLM_VERSION__="0.9.34"
    __NPU_VERSION__="32.0.203.304"
    DISABLE_ABI_CHECK=1)

if(WIN32)
    target_link_directories(test_downloader_pinning PRIVATE "${XRT_INCLUDE_DIR}/../lib")
elseif(XRT_FOUND)
    target_link_directories(test_downloader_pinning PRIVATE "${XRT_LIBRARY_DIRS}")
else()
    target_link_directories(test_downloader_pinning PRIVATE /opt/xilinx/xrt/lib)
endif()
target_link_libraries(test_downloader_pinning PRIVATE xrt_coreutil CURL::libcurl)

enable_testing()
add_test(NAME test_downloader_pinning COMMAND test_downloader_pinning)
