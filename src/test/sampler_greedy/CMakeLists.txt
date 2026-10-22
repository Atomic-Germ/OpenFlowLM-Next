# Greedy sampler tie rule, AVX2 path; host-only, like ../model_backend.
cmake_minimum_required(VERSION 3.22)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sampler_greedy_tests LANGUAGES CXX)
endif()

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

add_executable(test_sampler_greedy test.cpp "${OFLM_SOURCE_DIR}/common/modules/sampler.cpp")
# the served build has no USEAVX2, so this is the only place its vector argmax is exercised
target_compile_definitions(test_sampler_greedy PRIVATE USEAVX2=1)
target_compile_options(test_sampler_greedy PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mavx2 -mfma>)
target_include_directories(test_sampler_greedy PRIVATE
    "${OFLM_SOURCE_DIR}/include"
    "${XRT_INCLUDE_DIR}")

if(WIN32)
    target_link_directories(test_sampler_greedy PRIVATE "${XRT_INCLUDE_DIR}/../lib")
elseif(XRT_FOUND)
    target_link_directories(test_sampler_greedy PRIVATE "${XRT_LIBRARY_DIRS}")
else()
    target_link_directories(test_sampler_greedy PRIVATE /opt/xilinx/xrt/lib)
endif()
target_link_libraries(test_sampler_greedy PRIVATE xrt_coreutil)

enable_testing()
add_test(NAME test_sampler_greedy COMMAND test_sampler_greedy)
