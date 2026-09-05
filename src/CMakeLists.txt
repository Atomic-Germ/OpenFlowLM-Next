cmake_minimum_required(VERSION 3.22)
project(flm LANGUAGES CXX VERSION 0.1.0)

if(POLICY CMP0207)
    cmake_policy(SET CMP0207 NEW)
endif()

if(NOT DEFINED FLM_VERSION)
    message(FATAL_ERROR "FLM_VERSION must be specified externally. Use -DFLM_VERSION=<version> when running cmake.")
endif()

if(NOT DEFINED NPU_VERSION)
    message(FATAL_ERROR "NPU_VERSION must be specified externally. Use -DNPU_VERSION=<version> when running cmake.")
endif()

# Set build type to Release
set(CMAKE_BUILD_TYPE Release)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure CMake to stop at first error
set(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION ON)
set(CMAKE_ERROR_DEPRECATED ON)


# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/)


# Force output directories to be absolute
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# ———————————————————————————————————————————————
# NPU runtime backend selection.
#
# FLM ships two interchangeable NPU dispatch backends:
#   * XRT  — the Xilinx Run Time (the historical default), and
#   * HRX  — the HRX amdxdna runtime.
#
# The reviewer-requested build flag selects between them (0 = XRT, 1 = HRX) with
# XRT as the default, so the XRT build is unchanged and HRX is strictly opt-in.
# FLM_USE_HRX=OFF selects XRT; FLM_USE_HRX=ON selects HRX. The choice drives:
#   - which runtime headers/namespace alias the sources see (FLM_USE_HRX define),
#   - which prebuilt engine libs are consumed (lib/xrt vs lib/hrx), and
#   - which runtime is discovered, linked and bundled below.
# ———————————————————————————————————————————————
option(FLM_USE_HRX "Use the HRX amdxdna NPU runtime instead of XRT (0=XRT default, 1=HRX)" OFF)
option(FLM_PORTABLE_BUILD "Build portable distribution with bundled runtime libraries" OFF)

if(FLM_USE_HRX)
    set(FLM_RUNTIME_NAME "hrx")
    message(STATUS "FLM NPU runtime backend: HRX (FLM_USE_HRX=ON)")
else()
    set(FLM_RUNTIME_NAME "xrt")
    message(STATUS "FLM NPU runtime backend: XRT (FLM_USE_HRX=OFF, default)")
endif()

# Prebuilt engine libraries live in a per-backend subdirectory so the XRT-built
# and HRX-built .so/.dll/.lib never collide (see src/lib/xrt and src/lib/hrx).
set(FLM_ENGINE_LIB_DIR "${CMAKE_SOURCE_DIR}/lib/${FLM_RUNTIME_NAME}")

# ———————————————————————————————————————————————
# NPU runtime discovery.
#   HRX: consumed from its public CMake package via find_package(hrx).
#   XRT: discovered via pkg-config (system install), a manual /opt/xilinx/xrt
#        fallback, or fetched from source for a portable build.
# ———————————————————————————————————————————————
if(FLM_USE_HRX)
    find_package(hrx CONFIG REQUIRED)
else()
    if(WIN32)
        set(XRT_INCLUDE_DIR C:/dev/XRT/src/runtime_src/core/include CACHE PATH "Where XRT headers live")
        set(XRT_LIB_DIR     C:/dev/xrtNPUfromDLL                    CACHE PATH "Where XRT libs live")
    else()
        find_package(PkgConfig)
        if(PkgConfig_FOUND)
            pkg_check_modules(XRT xrt)
            if(XRT_FOUND)
                message(STATUS "Found XRT via pkg-config")
                message(STATUS "  XRT include dirs: ${XRT_INCLUDE_DIRS}")
                message(STATUS "  XRT library dirs: ${XRT_LIBRARY_DIRS}")
                message(STATUS "  XRT libraries: ${XRT_LIBRARIES}")
            endif()
        endif()

        # Portable build: fetch XRT from source if it is not installed on the host.
        if(FLM_PORTABLE_BUILD AND NOT XRT_FOUND)
            message(STATUS "XRT not found on system; fetching from source for portable build")
            include(FetchContent)
            set(XRT_GIT_REPO "https://github.com/Xilinx/XRT.git" CACHE STRING "XRT Git repository URL")
            set(XRT_GIT_TAG "2.21.75" CACHE STRING "XRT Git tag/branch to fetch")
            FetchContent_Declare(xrt_source
                GIT_REPOSITORY ${XRT_GIT_REPO}
                GIT_TAG ${XRT_GIT_TAG}
                GIT_SHALLOW TRUE)
            set(XRT_NATIVE_BUILD ON CACHE BOOL "Native XRT build" FORCE)
            set(BUILD_DOCS OFF CACHE BOOL "Build documentation" FORCE)
            set(XRT_BASE 1 CACHE STRING "Build base XRT only" FORCE)
            set(XRT_ALVEO 0 CACHE STRING "Disable Alveo support" FORCE)
            set(XRT_BUILD_XBMGMT OFF CACHE BOOL "Build xbmgmt" FORCE)
            set(XRT_BUILD_XBUTIL OFF CACHE BOOL "Build xbutil" FORCE)
            set(XRT_BUILD_XBT OFF CACHE BOOL "Build xbt" FORCE)
            FetchContent_MakeAvailable(xrt_source)
            set(XRT_REPO_DIR ${xrt_source_SOURCE_DIR})
            set(XRT_BUILD_DIR ${xrt_source_BINARY_DIR})
            set(XRT_INCLUDE_DIRS ${XRT_REPO_DIR}/src/runtime_src/core/include)
            set(XRT_FOUND TRUE)
            set(XRT_BUILT_FROM_SOURCE TRUE)
            set(XRT_LIBRARY_DIRS ${XRT_BUILD_DIR}/src/runtime_src)
            message(STATUS "Note: XDNA plugin (libxrt_driver_xdna.so.2) must be provided separately")
        endif()

        # Manual fallback if pkg-config didn't find XRT and it wasn't fetched.
        if(NOT XRT_FOUND AND NOT XRT_BUILT_FROM_SOURCE)
            set(XRT_INCLUDE_DIR /opt/xilinx/xrt/include CACHE PATH "Where XRT headers live")
            set(XRT_LIB_DIR     /opt/xilinx/xrt/lib     CACHE PATH "Where XRT libs live")
        endif()

        # Portable XRT build: statically link FFmpeg (and zlib) so the self-
        # contained tarball does not depend on the host having libavformat/
        # libavcodec/... installed. FFmpeg and zlib are fetched and built from
        # source; the resulting .a archives are linked into flm below. Non-
        # portable builds keep using the system FFmpeg via pkg-config.
        if(FLM_PORTABLE_BUILD)
            include(FetchContent)
            message(STATUS "Portable build: FFmpeg and zlib will be built statically")

            # ——— FFmpeg (static) ———
            set(FFMPEG_GIT_REPO "https://github.com/FFmpeg/FFmpeg.git" CACHE STRING "FFmpeg Git repository URL")
            set(FFMPEG_GIT_TAG  "n7.1" CACHE STRING "FFmpeg Git tag/branch to fetch")
            FetchContent_Declare(ffmpeg_source
                GIT_REPOSITORY ${FFMPEG_GIT_REPO}
                GIT_TAG ${FFMPEG_GIT_TAG}
                GIT_SHALLOW TRUE)
            message(STATUS "Fetching FFmpeg sources...")
            FetchContent_Populate(ffmpeg_source)

            set(FFMPEG_SOURCE_DIR ${ffmpeg_source_SOURCE_DIR})
            set(FFMPEG_BUILD_DIR ${CMAKE_BINARY_DIR}/ffmpeg-build)
            set(FFMPEG_INSTALL_DIR ${CMAKE_BINARY_DIR}/ffmpeg-install)
            file(MAKE_DIRECTORY ${FFMPEG_BUILD_DIR})

            message(STATUS "Configuring FFmpeg for static build...")
            execute_process(
                COMMAND ${FFMPEG_SOURCE_DIR}/configure
                    --prefix=${FFMPEG_INSTALL_DIR}
                    --enable-static
                    --disable-shared
                    --disable-programs
                    --disable-doc
                    --disable-htmlpages
                    --disable-manpages
                    --disable-podpages
                    --disable-txtpages
                    --enable-pic
                    --enable-zlib
                    --enable-avcodec
                    --enable-avformat
                    --enable-avutil
                    --enable-swscale
                    --enable-swresample
                    --disable-vaapi
                    --disable-libdrm
                    --disable-bzlib
                    --disable-lzma
                WORKING_DIRECTORY ${FFMPEG_BUILD_DIR}
                RESULT_VARIABLE FFMPEG_CONFIGURE_RESULT
                OUTPUT_FILE ${FFMPEG_BUILD_DIR}/configure.log
                ERROR_FILE ${FFMPEG_BUILD_DIR}/configure.log)
            if(NOT FFMPEG_CONFIGURE_RESULT EQUAL 0)
                message(FATAL_ERROR "FFmpeg configure failed. Check ${FFMPEG_BUILD_DIR}/configure.log")
            endif()

            message(STATUS "Building FFmpeg static libraries...")
            # FFmpeg compilation is memory-intensive, so cap parallelism at 4.
            if(DEFINED CMAKE_BUILD_PARALLEL_LEVEL AND CMAKE_BUILD_PARALLEL_LEVEL GREATER 4)
                set(FFMPEG_BUILD_JOBS 4)
            elseif(DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
                set(FFMPEG_BUILD_JOBS ${CMAKE_BUILD_PARALLEL_LEVEL})
            else()
                set(FFMPEG_BUILD_JOBS 2)
            endif()
            execute_process(
                COMMAND make -j${FFMPEG_BUILD_JOBS}
                WORKING_DIRECTORY ${FFMPEG_BUILD_DIR}
                RESULT_VARIABLE FFMPEG_BUILD_RESULT
                OUTPUT_FILE ${FFMPEG_BUILD_DIR}/build.log
                ERROR_FILE ${FFMPEG_BUILD_DIR}/build.log)
            if(NOT FFMPEG_BUILD_RESULT EQUAL 0)
                message(FATAL_ERROR "FFmpeg build failed. Check ${FFMPEG_BUILD_DIR}/build.log")
            endif()

            message(STATUS "Installing FFmpeg static libraries...")
            execute_process(
                COMMAND make install
                WORKING_DIRECTORY ${FFMPEG_BUILD_DIR}
                RESULT_VARIABLE FFMPEG_INSTALL_RESULT)
            if(NOT FFMPEG_INSTALL_RESULT EQUAL 0)
                message(FATAL_ERROR "FFmpeg install failed")
            endif()

            set(FFMPEG_BUILT_FROM_SOURCE TRUE)
            set(FFMPEG_INCLUDE_DIRS ${FFMPEG_INSTALL_DIR}/include)
            set(FFMPEG_LIBRARY_DIRS ${FFMPEG_INSTALL_DIR}/lib)
            message(STATUS "FFmpeg built successfully (static): ${FFMPEG_LIBRARY_DIRS}")

            # ——— zlib (static) ———
            message(STATUS "Fetching zlib source...")
            FetchContent_Declare(zlib_source
                GIT_REPOSITORY "https://github.com/madler/zlib.git"
                GIT_TAG "v1.3.1"
                GIT_SHALLOW TRUE)
            FetchContent_MakeAvailable(zlib_source)
            set(ZLIB_BUILT_FROM_SOURCE TRUE)
            set(ZLIB_STATIC_LIB ${zlib_source_BINARY_DIR}/libz.a)
            message(STATUS "zlib built successfully (static): ${ZLIB_STATIC_LIB}")
        endif()
    endif()
endif()

# ———————————————————————————————————————————————
# Add tokenizers-cpp subproject
# ———————————————————————————————————————————————
# Ensure C++17 is available for tokenizers-cpp
if(CMAKE_CXX_STANDARD LESS 17)
    set(CMAKE_CXX_STANDARD 17)
endif()

# sentencepiece links third_party/absl -> third_party/abseil-cpp/absl with
# file(CREATE_LINK ... SYMBOLIC), and on Windows a symbolic link needs either
# Developer Mode or an elevated shell. On a stock box a FRESH CLONE therefore
# dies at configure with
#
#   file failed to create symbolic link '.../sentencepiece/third_party/absl':
#   A required privilege is not held by the client.
#
# and the message points at a third-party CMakeLists, so it reads as somebody
# else's bug. It is not fixable there -- that is an upstream submodule -- but
# it is avoidable here: sentencepiece guards the call with
# `if(NOT EXISTS .../third_party/absl)`, and a DIRECTORY JUNCTION satisfies
# that guard while needing no privileges at all. Same trick as the xclbins
# junction further down.
#
# NATIVE paths: mklink is a cmd builtin and silently rejects the forward
# slashes CMake uses internally.
if(WIN32)
    set(FLM_SPM_TP
        "${CMAKE_SOURCE_DIR}/../third_party/tokenizers-cpp/sentencepiece/third_party")
    if(EXISTS "${FLM_SPM_TP}/abseil-cpp/absl" AND NOT EXISTS "${FLM_SPM_TP}/absl")
        file(TO_NATIVE_PATH "${FLM_SPM_TP}/absl" FLM_SPM_LINK)
        file(TO_NATIVE_PATH "${FLM_SPM_TP}/abseil-cpp/absl" FLM_SPM_TARGET)
        execute_process(
            COMMAND cmd /c mklink /J "${FLM_SPM_LINK}" "${FLM_SPM_TARGET}"
            RESULT_VARIABLE FLM_SPM_RC
            OUTPUT_QUIET ERROR_QUIET)
        if(FLM_SPM_RC EQUAL 0)
            message(STATUS "sentencepiece: junction third_party/absl -> abseil-cpp/absl")
        endif()
        # No else(): if the junction fails, sentencepiece's own symlink attempt
        # runs next and produces the real diagnostic. Do not mask it.
    elseif(NOT EXISTS "${FLM_SPM_TP}/absl")
        # HONEST LIMITATION: abseil-cpp is fetched by FetchContent INSIDE
        # sentencepiece's own CMakeLists, i.e. during the add_subdirectory()
        # below -- so on a genuinely fresh clone the target does not exist yet
        # when this runs, and there is nothing to point a junction at. This
        # configure will fetch abseil and then fail on the symlink; the NEXT
        # one takes the branch above and succeeds.
        #
        # Fixing the first pass would mean cloning abseil here ourselves, which
        # means copying sentencepiece's GIT_TAG into this file -- a duplicated
        # version pin, which is a worse problem than one repeated command.
        message(STATUS
            "sentencepiece: abseil-cpp not fetched yet. If this configure "
            "fails with 'failed to create symbolic link', simply run it again: "
            "abseil will be present by then and a junction is used instead of "
            "a symlink (no Developer Mode or elevation needed).")
    endif()
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/../third_party/tokenizers-cpp
                 ${CMAKE_BINARY_DIR}/tokenizers-cpp
                 EXCLUDE_FROM_ALL)

# ———————————————————————————————————————————————
# Gather your sources
# ———————————————————————————————————————————————
file(GLOB SOURCES "src/*.cpp" "runner/*.cpp" "common/*.cpp" "common/*/*.cpp" "server/*.cpp" "pull/*.cpp")
file(GLOB HEADERS "include/*.hpp" "runner/*.hpp" "common/*.hpp" "common/*/*.hpp" "server/*.hpp" "pull/*.hpp")

list(APPEND SOURCES "${CMAKE_SOURCE_DIR}/open_embedding/engine.cpp")
if(NOT FLM_USE_HRX)
    list(APPEND SOURCES "${CMAKE_SOURCE_DIR}/open_embedding/npu_matmul.cpp")
    # The open Qwen3.6-MoE engine (open_qwen36/, open XDNA2 kernels under
    # xclbins/<model>/open_kernels). XRT only: it drives the kernels through
    # xrt::hw_context / xrt::kernel directly, like the open embedding backend.
    list(APPEND SOURCES
        "${CMAKE_SOURCE_DIR}/open_qwen36/q4nx_file.cpp"
        "${CMAKE_SOURCE_DIR}/open_qwen36/pools.cpp"
        "${CMAKE_SOURCE_DIR}/open_qwen36/core.cpp"
        "${CMAKE_SOURCE_DIR}/open_qwen36/engine.cpp")
endif()

# ———————————————————————————————————————————————
# open_npue: a second embedding backend (NpuEmbeddings)
# ———————————————————————————————————————————————
#
# ADDITIVE. open_embedding above stays exactly where it is and keeps serving
# embed-gemma:300m; this adds encoders for models the tree does not have.
# It is a synced copy -- see open_npue/SYNCED.md for the source commit and
# per-file hashes. Edit it upstream, not here: the accuracy gates that make
# its numbers mean anything (a HuggingFace golden gate, cross-lane bitwise
# agreement, an MTEB bridge, a p99 tail gate and four per-tokenizer
# byte-exactness harnesses) live in that repository and want an NPU.
#
# Sources are listed EXPLICITLY, not globbed, per AGENTS.md.
#
# NOT FLM_USE_HRX, exactly like npu_matmul.cpp above: open_npue/npu_device.cpp
# talks to XRT directly, and the encoders are unusable without it.
if(NOT FLM_USE_HRX)
    set(OPEN_NPUE_SOURCES
        "${CMAKE_SOURCE_DIR}/open_npue/npue_encoder.cpp"
        "${CMAKE_SOURCE_DIR}/open_npue/npue.cpp"
        "${CMAKE_SOURCE_DIR}/open_npue/npue_pack.cpp"
        "${CMAKE_SOURCE_DIR}/open_npue/npu_device.cpp"
        "${CMAKE_SOURCE_DIR}/open_npue/json_min.cpp"
        "${CMAKE_SOURCE_DIR}/open_npue/tokenizer.cpp"
        "${CMAKE_SOURCE_DIR}/open_npue/tokenizer_gemma.cpp"
        "${CMAKE_SOURCE_DIR}/open_npue/gemma_tokenizer_gen.cpp"
        "${CMAKE_SOURCE_DIR}/open_npue/tokenizer_xlmr.cpp"
        "${CMAKE_SOURCE_DIR}/open_npue/xlmr_tokenizer_gen.cpp"
        "${CMAKE_SOURCE_DIR}/open_npue/tokenizer_bbpe.cpp"
        "${CMAKE_SOURCE_DIR}/open_npue/bbpe_tokenizer_gen.cpp"
        "${CMAKE_SOURCE_DIR}/open_npue/gemma_kernels.cpp"
        "${CMAKE_SOURCE_DIR}/open_npue/gemma_encode.cpp"
        # Fork-owned, not synced: the ONLY translation unit that includes the
        # engine's header. It is in this list rather than in the glob because
        # it must carry the engine's flags -- see the /arch:AVX2 note below and
        # the long comment in npue_embedding.hpp, which records what a host TU
        # instantiating those inline functions at a different ISA level costs.
        "${CMAKE_SOURCE_DIR}/open_npue_adapter/npue_embedding.cpp"
    )
    list(APPEND SOURCES ${OPEN_NPUE_SOURCES})

    # THE INCLUDE PATH IS PER-SOURCE, NOT GLOBAL, AND THAT IS DELIBERATE.
    # open_npue/tokenizer.hpp and src/include/tokenizer/tokenizer.hpp are
    # different files with the same basename. Putting open_npue on the global
    # include path would let the wrong one win for some translation unit
    # depending on directory order -- a failure that shows up as a confusing
    # compile error at best and as the wrong tokenizer at worst. Scoping it to
    # these fourteen files means nothing else in the tree can even see them.
    set_source_files_properties(${OPEN_NPUE_SOURCES} PROPERTIES
        INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/open_npue")

    # /arch:AVX2 IS LOAD-BEARING AND IS LOST IN SILENCE.
    #
    # Roughly half of an encode is host-side AVX2 intrinsics, every one of them
    # behind `#if defined(__AVX2__)` with a correct scalar `#else`. Without the
    # flag this compiles, runs, returns the right vectors and is 2.1-2.6x
    # slower -- measured upstream on three models, with every accuracy gate
    # still green. MSVC x64 defaults to the SSE2 baseline and defines no
    # __AVX2__, so the flag is not optional here.
    #
    # Same precedent as the AVX-512 image/audio sources below: per-source
    # options, so raising the ISA for these files does not raise it for the
    # whole binary.
    #
    # One consequence worth knowing before writing a comparison test: THE
    # EMBEDDING BYTES MOVE WITH THE HOST ISA. The scalar and AVX2 paths reduce
    # in different orders, so any check comparing vectors against an upstream
    # binary must fix the /arch: level on both sides or it fails for a reason
    # that has nothing to do with the port.
    if(MSVC)
        set_source_files_properties(${OPEN_NPUE_SOURCES} PROPERTIES
            COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(${OPEN_NPUE_SOURCES} PROPERTIES
            COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
endif()

# Exclude files that depend on missing libraries for Linux
if(NOT WIN32)
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gemma3\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gemma3_image\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gemma3_text\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gpt_oss\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_lfm2\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_phi4\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_qwen2\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_qwen3\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_qwen3vl\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_qwen3vl_image\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_whisper\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_whisper_audio\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*modeling_gemma_embedding\\.cpp$")
    # list(FILTER SOURCES EXCLUDE REGEX ".*auto_embedding_model\\.cpp$")

    set_source_files_properties(
        ${CMAKE_SOURCE_DIR}/common/image_process_utils/imageprocAVX512.cpp
        ${CMAKE_SOURCE_DIR}/common/audio_process_utils/audioprocAVX512.cpp
        PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512dq;-mavx512vl;-mavx512bw;-mfma"
    )

    # Define a macro to indicate limited model support on Linux
    # add_compile_definitions(FASTFLOWLM_LINUX_LIMITED_MODELS=1)
endif()


add_executable(flm ${SOURCES} ${HEADERS})

if(WIN32)
    if(VCPKG_TOOLCHAIN)
        # A vcpkg toolchain is active (e.g. the rocm-npu-staging dev.py build or
        # a local CMakePresets build): the vcpkg tree ships CMake package configs,
        # so resolve the native deps in CONFIG mode. find_package auto-detects the
        # correct versioned import-lib names, so this works regardless of the
        # installed Boost/FFTW versions.
        find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
        find_package(CURL CONFIG REQUIRED)
        find_package(FFMPEG REQUIRED)
        find_package(FFTW3 CONFIG REQUIRED)
        find_package(FFTW3f CONFIG REQUIRED)
        find_package(FFTW3l CONFIG REQUIRED)
    endif()
    # Otherwise (the bare self-hosted CI runner) there is no vcpkg toolchain and
    # no Boost in vcpkg: Boost is a standalone b2 build under C:/dev/boost_1_88_0
    # and curl/ffmpeg/fftw are linked by raw name from the vcpkg lib dir. That
    # path is wired via the include/link dirs and raw library names guarded by
    # NOT VCPKG_TOOLCHAIN below.
else()
    find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
    # The Linux build environment installs these via apt, which provides the
    # CMake FindCURL module and pkg-config files rather than CMake package
    # configs. Resolve CURL via the module and FFmpeg/FFTW via pkg-config.
    find_package(CURL REQUIRED)
    find_package(Threads REQUIRED)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)
    pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)
    pkg_check_modules(FFTW3L REQUIRED IMPORTED_TARGET fftw3l)
    # Portable builds link FFmpeg statically (built from source above), so the
    # system FFmpeg is only resolved via pkg-config for non-portable builds.
    if(NOT FFMPEG_BUILT_FROM_SOURCE)
        pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
            libavformat libavcodec libavutil libswscale libswresample)
    endif()
    # readline is GPL; skip it for portable builds (the CLI falls back to a
    # plain line reader via the FASTFLOWLM_USE_READLINE guard) so the portable
    # tarball carries no readline/ncurses runtime dependency.
    if(NOT FLM_PORTABLE_BUILD)
        pkg_check_modules(readline REQUIRED IMPORTED_TARGET readline)
        pkg_check_modules(ncurses REQUIRED IMPORTED_TARGET ncursesw)
    endif()
endif()

# ———————————————————————————————————————————————
# Library specific settings
# ———————————————————————————————————————————————
target_include_directories(flm PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/runner
    ${CMAKE_SOURCE_DIR}/server
    ${CMAKE_SOURCE_DIR}/pull
    ${FFMPEG_INCLUDE_DIRS}
)

# XRT backend: add the XRT headers (HRX carries its own via the hrx::hrx target).
if(NOT FLM_USE_HRX)
    if(NOT WIN32 AND XRT_FOUND)
        target_include_directories(flm PUBLIC ${XRT_INCLUDE_DIRS})
    else()
        target_include_directories(flm PUBLIC ${XRT_INCLUDE_DIR})
    endif()
endif()

if(WIN32 AND NOT VCPKG_TOOLCHAIN)
    # Bare CI runner: standalone Boost + vcpkg headers by absolute path.
    target_include_directories(flm PUBLIC
        C:/dev/boost_1_88_0
        C:/dev/vcpkg/installed/x64-windows/include/
    )
endif()

target_compile_definitions(flm PUBLIC
    DISABLE_ABI_CHECK=1
    CMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}"
    CMAKE_XCLBIN_PREFIX="${CMAKE_XCLBIN_PREFIX}"
    __FLM_VERSION__=\"${FLM_VERSION}\"
    __NPU_VERSION__=\"${NPU_VERSION}\"
    FLM_USE_OPEN_EMBEDDING=1
)

# Select the NPU runtime backend seen by the sources (device_runtime.hpp et al.).
if(FLM_USE_HRX)
    target_compile_definitions(flm PUBLIC FLM_USE_HRX=1)
endif()

# The open embedding NPU backend uses XRT directly. HRX retains the CPU path.
if(NOT FLM_USE_HRX)
    target_compile_definitions(flm PUBLIC FLM_USE_OPEN_EMBEDDING_NPU=1)
    # Qwen3.6-MoE on the open kernels (open_qwen36/); the closed engine stays
    # reachable with FLM_QWEN36_ENGINE=closed.
    target_compile_definitions(flm PUBLIC FLM_USE_OPEN_QWEN36=1)
    target_include_directories(flm PUBLIC "${CMAKE_SOURCE_DIR}/../open_kernels/harness")
endif()

if(WIN32)
    target_compile_definitions(flm PUBLIC
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        # Handle legacy stdio functions
        _CRT_SECURE_NO_WARNINGS
        _CRT_NONSTDC_NO_DEPRECATE
        __WINDOWS__
    )
endif()

if(WIN32 AND NOT VCPKG_TOOLCHAIN)
    # Bare CI runner: statically link the standalone Boost + curl.
    target_compile_definitions(flm PUBLIC
        CURL_STATICLIB
        BOOST_ALL_NO_LIB
        BOOST_ALL_STATIC_LINK
    )
endif()

target_link_directories(flm PUBLIC
    ${FLM_ENGINE_LIB_DIR}       # per-backend prebuilt engine libs (lib/xrt or lib/hrx)
    ${CMAKE_SOURCE_DIR}/lib     # shared third-party runtime libs (ffmpeg/curl/fftw/... on Windows)
    ${FFMPEG_LIBRARY_DIRS}
)

# XRT backend: add the XRT runtime library dir (aiebu ships in lib/<backend>).
if(NOT FLM_USE_HRX)
    if(NOT WIN32 AND XRT_FOUND)
        target_link_directories(flm PUBLIC ${XRT_LIBRARY_DIRS})
    else()
        target_link_directories(flm PUBLIC ${XRT_LIB_DIR})
    endif()
endif()

if(WIN32 AND NOT VCPKG_TOOLCHAIN)
    # Bare CI runner: standalone Boost stage libs + vcpkg import libs.
    target_link_directories(flm PUBLIC
        C:/dev/boost_1_88_0/stage/lib
        C:/dev/vcpkg/installed/x64-windows/lib
    )
endif()

# Link static libraries first
if(MSVC)
    target_link_libraries(flm PUBLIC ${STATIC_LIBS})
endif()

# Link your custom libraries (these may still be DLLs if no static versions available)
target_link_libraries(flm PUBLIC
    q4_npu_eXpress
    llama_npu
    qwen2_npu
    qwen2vl_npu
    qwen3_npu
    qwen3vl_npu
    qwen3_5vl_npu
    qwen3_5_omni_npu
    qwen3_6_moe_npu
    gemma_npu
    gemma_text_npu
    gemma4e_npu
    gemma4_12b_npu
    gpt_oss_npu
    whisper_npu
    lfm2_npu
    phi4_npu
    nanbeige_npu
    dequant
    gemm
    lm_head
    mha
)

# Link the selected NPU runtime after the detail model libraries above so their
# runtime references (libhrx / libxrt_coreutil) resolve.
if(FLM_USE_HRX)
    # (Windows: hrx.lib import lib for hrx.dll; Linux: libhrx.so.)
    target_link_libraries(flm PUBLIC hrx::hrx)
    if(NOT WIN32)
        set_target_properties(flm PROPERTIES
            BUILD_RPATH "$<TARGET_FILE_DIR:hrx::hrx>;${FLM_ENGINE_LIB_DIR}")
    endif()
else()
    # XRT: link xrt_coreutil (+ aiebu for the ELF assembler used by npu_utils_xrt).
    if(XRT_BUILT_FROM_SOURCE AND TARGET xrt_coreutil)
        target_link_libraries(flm PUBLIC xrt_coreutil)
    elseif(NOT WIN32 AND XRT_FOUND)
        target_link_libraries(flm PUBLIC ${XRT_LIBRARIES})
    else()
        target_link_libraries(flm PUBLIC xrt_coreutil)
    endif()
    if(WIN32)
        target_link_libraries(flm PUBLIC aiebu_static)
    else()
        target_link_libraries(flm PUBLIC aiebu)
    endif()
    if(NOT WIN32)
        set_target_properties(flm PROPERTIES
            BUILD_RPATH "${FLM_ENGINE_LIB_DIR}")
    endif()
endif()

if(WIN32 AND VCPKG_TOOLCHAIN)
    # Local/managed vcpkg: link the imported targets from the CONFIG packages
    # found above (versioned import-lib names resolved automatically).
    target_link_libraries(flm PUBLIC
        ${FFMPEG_LIBRARIES}
        Boost::program_options
        CURL::libcurl
        FFTW3::fftw3
        FFTW3::fftw3f
        FFTW3::fftw3l
    )
elseif(WIN32)
    # Bare CI runner: link the native deps by raw name from the standalone Boost
    # + vcpkg lib dirs wired above (no CMake package configs there).
    target_link_libraries(flm PUBLIC
        avformat
        avcodec
        avutil
        swscale
        swresample
        libcurl
        libboost_program_options-vc143-mt-x64-1_88
        libfftw3-3
        libfftw3f-3
        libfftw3l-3
    )
elseif(FFMPEG_BUILT_FROM_SOURCE)
    # Portable Linux: link the static FFmpeg archives built above (plus zlib) so
    # flm carries no libav*/libsw* runtime dependency.
    if(ZLIB_BUILT_FROM_SOURCE)
        set(_flm_zlib_lib ${ZLIB_STATIC_LIB})
    else()
        set(_flm_zlib_lib z)
    endif()
    target_link_libraries(flm PUBLIC
        ${FFMPEG_LIBRARY_DIRS}/libavformat.a
        ${FFMPEG_LIBRARY_DIRS}/libavcodec.a
        ${FFMPEG_LIBRARY_DIRS}/libavutil.a
        ${FFMPEG_LIBRARY_DIRS}/libswscale.a
        ${FFMPEG_LIBRARY_DIRS}/libswresample.a
        ${_flm_zlib_lib}
        Boost::program_options
        CURL::libcurl
        PkgConfig::FFTW3
        PkgConfig::FFTW3F
        PkgConfig::FFTW3L
    )
else()
    target_link_libraries(flm PUBLIC
        ${FFMPEG_LIBRARIES}
        Boost::program_options
        CURL::libcurl
        PkgConfig::FFTW3
        PkgConfig::FFTW3F
        PkgConfig::FFTW3L
    )
endif()

if(NOT WIN32)
    target_compile_options(flm PUBLIC -mavx -mavx2)

    # readline is only linked for non-portable builds (see the discovery block);
    # the CLI compiles a plain line-reader fallback when the macro is undefined.
    if(NOT FLM_PORTABLE_BUILD)
        target_compile_definitions(flm PUBLIC FASTFLOWLM_USE_READLINE=1)
        target_link_libraries(flm PUBLIC PkgConfig::readline PkgConfig::ncurses)
    endif()

    # Link with dl library for dlopen() support (used for preloading bundled libraries)
    target_link_libraries(flm PUBLIC dl)
endif()

# ———————————————————————————————————————————————
# Link tokenizers-cpp libraries
# ———————————————————————————————————————————————
# The tokenizers-cpp subproject provides these targets:
# - tokenizers_cpp: main tokenizers C++ library
# - tokenizers_c: tokenizers C FFI bindings  
# - sentencepiece: sentencepiece tokenizer library
target_link_libraries(flm PRIVATE
    tokenizers_cpp
)

if(WIN32)
    target_link_libraries(flm PRIVATE
        ntdll wsock32 ws2_32 Bcrypt
        iphlpapi userenv psapi
        crypt32 secur32 advapi32 normaliz wldap32
    )
else()
    target_link_libraries(flm PRIVATE Threads::Threads)
endif()


# ———————————————————————————————————————————————
# Copy the build flm.exe into the out directory (for local dev)
# ———————————————————————————————————————————————
if(WIN32)
    add_custom_command(TARGET flm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:flm>
            ${CMAKE_SOURCE_DIR}/out/flm.exe
    )

    # AND THE ENGINE DLLs BESIDE IT, or a freshly built flm.exe cannot start
    # -- and fails in a way that sends you looking in the wrong place.
    #
    # The 22 *_npu.dll / q4_npu_eXpress.dll / gemm.dll / dequant.dll live in
    # src/lib/<backend>/ and are load-time imports, so without them the process
    # dies at 0xC0000135 (DLL not found) BEFORE main(), printing nothing.
    # Windows then falls through to the next PATH entry -- which on a machine
    # with FastFlowLM installed is C:/Program Files/flm/flm.exe. So
    # `flm.exe serve ... --embeddingmodel x` run from the build directory
    # reports "unrecognised option", from a completely different binary than
    # the one you just built. The symptom names a flag; the cause is a
    # missing DLL two steps earlier.
    #
    # Both destinations are gitignored build trees, so this copies nothing
    # into the repository and redistributes nothing: it puts the files this
    # checkout already has next to the executable that needs them.

    # AND THE MODEL CATALOGUE, for the same reason. utils.cpp looks for
    # model_list.json next to the executable (after FLM_CONFIG_PATH), and
    # nothing put one there, so a freshly built flm.exe stops at
    #
    #   model_list.json not found. Please set FLM_CONFIG_PATH or place it
    #   next to the executable.
    #
    # before it can do anything at all. copy_if_different rather than a
    # configure-time copy, so editing the catalogue -- which adding a model
    # means doing -- takes effect on the next build instead of the next
    # configure.
    add_custom_command(TARGET flm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/model_list.json
            ${CMAKE_SOURCE_DIR}/model_info.json
            $<TARGET_FILE_DIR:flm>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/model_list.json
            ${CMAKE_SOURCE_DIR}/model_info.json
            ${CMAKE_SOURCE_DIR}/out
        COMMENT "Copying model_list.json / model_info.json beside flm.exe"
    )

    file(GLOB FLM_RUNTIME_DLLS "${CMAKE_SOURCE_DIR}/lib/${FLM_RUNTIME_NAME}/*.dll")
    if(FLM_RUNTIME_DLLS)
        add_custom_command(TARGET flm POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${FLM_RUNTIME_DLLS} $<TARGET_FILE_DIR:flm>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${FLM_RUNTIME_DLLS} ${CMAKE_SOURCE_DIR}/out
            COMMENT "Copying ${FLM_RUNTIME_NAME} engine DLLs beside flm.exe"
        )
    endif()

    # AND THE XCLBIN TREE, as a junction rather than a copy.
    #
    # find_xclbin_path() already looks for "<exe_dir>/xclbins" -- it calls that
    # the "portable development-tree location" -- but nothing puts one there,
    # so a freshly built flm finds no kernels and every model fails with "no
    # design set" until FLM_XCLBIN_PATH is set by hand.
    #
    # A JUNCTION, not a copy: src/xclbins is hundreds of megabytes, and a
    # junction also means a rebuilt kernel is visible immediately rather than
    # after the next build. mklink /J needs no privileges (unlike a symlink,
    # which needs Developer Mode), and if it fails this is a warning rather
    # than an error -- FLM_XCLBIN_PATH still works.
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/xclbins")
        # NATIVE paths: mklink is a cmd builtin and rejects the forward slashes
        # CMake uses internally, silently enough that the first version of this
        # produced no junction and no message.
        file(TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/xclbins" FLM_XCLBIN_LINK)
        file(TO_NATIVE_PATH "${CMAKE_SOURCE_DIR}/xclbins" FLM_XCLBIN_TARGET)
        execute_process(
            COMMAND cmd /c mklink /J "${FLM_XCLBIN_LINK}" "${FLM_XCLBIN_TARGET}"
            RESULT_VARIABLE FLM_XCLBIN_LINK_RC
            OUTPUT_QUIET ERROR_QUIET)
        if(FLM_XCLBIN_LINK_RC EQUAL 0)
            message(STATUS "xclbins: junction ${CMAKE_BINARY_DIR}/xclbins -> ${CMAKE_SOURCE_DIR}/xclbins")
        else()
            message(WARNING "could not link xclbins next to the executable; "
                            "set FLM_XCLBIN_PATH=${CMAKE_SOURCE_DIR} to run from the build tree")
        endif()
    endif()
endif()

# Default install location for model_list.json / xclbins (matches the app's
# relocatable "<exe_dir>/../share/flm" lookup). Overridden to the prefix root for
# the flat portable layout below.
set(FLM_SHARE_DESTINATION "share/flm")

# Add a custom target to check for remaining DLL dependencies
if(WIN32)
    add_custom_target(check_dependencies ALL
        COMMAND ${CMAKE_COMMAND} -E echo "Checking for DLL dependencies..."
        COMMAND dumpbin /dependents $<TARGET_FILE:flm> | findstr /i ".dll"
        COMMENT "Checking for remaining DLL dependencies"
        DEPENDS flm
    )
else()
    if(NOT CMAKE_INSTALL_LIBDIR)
        set(CMAKE_INSTALL_LIBDIR "lib")
    endif()

    # ————————————————————————————————————————————————————————————
    # Install layout. Portable = flat, self-contained tree rooted at the install
    # prefix (consumed by the portable tarball workflow); otherwise an FHS layout
    # (bin/, lib/, share/) consumed by the .deb packaging.
    # ————————————————————————————————————————————————————————————
    if(FLM_PORTABLE_BUILD)
        set(FLM_BIN_DESTINATION ".")
        set(FLM_ENGINE_LIB_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
        set(FLM_SHARE_DESTINATION ".")
        set(FLM_FLM_INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
        # Engine libs sit next to libhrx in <root>/lib.
        set(FLM_ENGINE_INSTALL_RPATH "$ORIGIN")
    elseif(FLM_USE_HRX)
        # HRX FHS layout: engine libs sit in <prefix>/lib/flm and the bundled NPU
        # runtime (libhrx) one level up in <prefix>/lib, so engines resolve it via
        # $ORIGIN/.. — this separation is what lets the HRX .deb ship libhrx.
        set(FLM_BIN_DESTINATION "bin")
        set(FLM_ENGINE_LIB_DESTINATION "${CMAKE_INSTALL_LIBDIR}/flm")
        set(FLM_SHARE_DESTINATION "share/flm")
        set(FLM_FLM_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/flm;$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
        set(FLM_ENGINE_INSTALL_RPATH "$ORIGIN:$ORIGIN/..")
    else()
        # XRT FHS layout: identical to upstream main — engine libs install directly
        # into <prefix>/lib and flm resolves them via $ORIGIN/../lib. XRT itself is
        # an external dependency (libxrt-npu2) resolved from the system.
        set(FLM_BIN_DESTINATION "bin")
        set(FLM_ENGINE_LIB_DESTINATION "${CMAKE_INSTALL_LIBDIR}")
        set(FLM_SHARE_DESTINATION "share/flm")
        set(FLM_FLM_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
        set(FLM_ENGINE_INSTALL_RPATH "$ORIGIN")
    endif()

    file(GLOB so_libs "${FLM_ENGINE_LIB_DIR}/*.so*")
    install(FILES ${so_libs} DESTINATION "${FLM_ENGINE_LIB_DESTINATION}")
    set_target_properties(flm PROPERTIES INSTALL_RPATH "${FLM_FLM_INSTALL_RPATH}")

    # Engine .so file names, used below to keep them out of the flm dependency
    # closure (they are installed explicitly, above).
    set(_flm_engine_names "")
    set(_flm_engine_post_exclude "")
    foreach(_flm_src ${so_libs})
        get_filename_component(_flm_name "${_flm_src}" NAME)
        list(APPEND _flm_engine_names "${_flm_name}")
        string(REPLACE "." "[.]" _flm_name_re "${_flm_name}")
        list(APPEND _flm_engine_post_exclude ".*/${_flm_name_re}$")
    endforeach()

    # HRX only: the prebuilt HRX engine .so ship with an absolute build-machine
    # RUNPATH and a libhrx.so.0 NEEDED entry. Rewrite the RUNPATH to a
    # relocatable $ORIGIN-based one so libhrx resolves at runtime and, crucially,
    # so dpkg-shlibdeps can locate it during .deb packaging. The XRT engine .so
    # are shipped unmodified: the portable wrapper makes ./lib discoverable via
    # LD_LIBRARY_PATH, and for the .deb XRT is an external dependency resolved
    # from the system.
    if(FLM_USE_HRX)
        find_program(PATCHELF_EXECUTABLE patchelf REQUIRED)
        install(CODE "
            set(_flm_engine_names \"${_flm_engine_names}\")
            foreach(_flm_name \${_flm_engine_names})
                set(_flm_lib \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${FLM_ENGINE_LIB_DESTINATION}/\${_flm_name}\")
                if(EXISTS \"\${_flm_lib}\" AND NOT IS_SYMLINK \"\${_flm_lib}\")
                    execute_process(
                        COMMAND \"${PATCHELF_EXECUTABLE}\" --set-rpath \"${FLM_ENGINE_INSTALL_RPATH}\" \"\${_flm_lib}\"
                        RESULT_VARIABLE _flm_patchelf_rc)
                    if(NOT _flm_patchelf_rc EQUAL 0)
                        message(FATAL_ERROR \"patchelf --set-rpath failed for \${_flm_lib}\")
                    endif()
                endif()
            endforeach()
        ")
    endif()

    # Portable build ships a small wrapper as `flm`; the packaging step renames the
    # real binary to `flm-real` and this wrapper to `flm`.
    if(FLM_PORTABLE_BUILD)
        configure_file(
            "${CMAKE_SOURCE_DIR}/flm-wrapper.sh.in"
            "${CMAKE_BINARY_DIR}/flm-wrapper.sh"
            @ONLY)
        install(PROGRAMS "${CMAKE_BINARY_DIR}/flm-wrapper.sh" DESTINATION "${FLM_SHARE_DESTINATION}")
    endif()
endif()

# Directory holding the selected NPU runtime shared library, used by the Windows
# dependency set, the portable XRT bundling, and the HRX dependency closure. HRX
# ships libhrx via its imported target; XRT is a system/library-dir dependency.
if(FLM_USE_HRX)
    set(_flm_rt_dep_dir "$<TARGET_FILE_DIR:hrx::hrx>")
    # Sentinel that never matches a real library name (an empty PRE_EXCLUDE regex
    # would match — and thus exclude — everything).
    set(_flm_rt_pre_exclude "__flm_no_such_lib__")
elseif(NOT WIN32 AND XRT_FOUND)
    set(_flm_rt_dep_dir "${XRT_LIBRARY_DIRS}")
else()
    set(_flm_rt_dep_dir "${XRT_LIB_DIR}")
endif()

if(WIN32)
    install(TARGETS flm
        RUNTIME_DEPENDENCY_SET fastflowlm_runtime_dependencies
        RUNTIME DESTINATION bin)
    install(RUNTIME_DEPENDENCY_SET fastflowlm_runtime_dependencies
        DESTINATION bin
        DIRECTORIES
            "${FLM_ENGINE_LIB_DIR}"
            "${CMAKE_SOURCE_DIR}/lib"
            "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin"
            "${_flm_rt_dep_dir}"
        PRE_EXCLUDE_REGEXES "api-ms-.*" "ext-ms-.*"
            # Optional Windows feature-on-demand / attestation / lockdown DLLs
            # that are not present as files on all hosts, so the resolver would
            # report them unresolved. They are OS components, never bundled.
            # Patterns are case-insensitive because PE import names are lowercase.
            "[Aa][Zz][Uu][Rr][Ee][Aa][Tt][Tt][Ee][Ss][Tt].*"
            "[Hh][Vv][Ss][Ii][Ff][Ii][Ll][Ee][Tt][Rr][Uu][Ss][Tt].*"
            "[Pp][Dd][Mm][Uu][Tt][Ii][Ll][Ii][Tt][Ii][Ee][Ss].*"
            "[Ww][Pp][Aa][Xx][Hh][Oo][Ll][Dd][Ee][Rr].*"
        POST_EXCLUDE_REGEXES ".*[Ww]indows[/\\\\][Ss]ystem32[/\\\\].*")
elseif(NOT FLM_USE_HRX AND FLM_PORTABLE_BUILD)
    # Portable XRT: bundle the XRT runtime explicitly. XRT is deliberately
    # handled here instead of via the dependency closure below: flm only lists
    # libxrt_coreutil as a link-time NEEDED entry, while XRT dlopens libxrt_core,
    # the libxrt_driver_xdna plugin and others at run time via a path it builds
    # as $XILINX_XRT/lib/x86_64-linux-gnu/<lib>. Those are invisible to
    # file(GET_RUNTIME_DEPENDENCIES), so copy the whole libxrt*.so* set (plus
    # boost_program_options), re-root each real lib at $ORIGIN, and mirror them
    # under lib/x86_64-linux-gnu for XRT's internal lookup. FFTW is bundled the
    # same way just below. Other shared deps (libcurl, libgomp, ...) resolve from
    # the host.
    install(TARGETS flm RUNTIME DESTINATION "${FLM_BIN_DESTINATION}")
    find_program(PATCHELF_EXECUTABLE patchelf REQUIRED)
    install(CODE "
        # Locate a directory that actually holds the XRT runtime libraries.
        set(_flm_xrt_src \"\")
        foreach(_cand \"${_flm_rt_dep_dir}\" \"/usr/lib/x86_64-linux-gnu\" \"/usr/lib\")
            if(_cand AND EXISTS \"\${_cand}/libxrt_coreutil.so.2\")
                set(_flm_xrt_src \"\${_cand}\")
                break()
            endif()
        endforeach()
        if(NOT _flm_xrt_src)
            message(FATAL_ERROR
                \"Portable XRT build: could not locate libxrt_coreutil.so.2 to bundle\")
        endif()
        message(STATUS \"Bundling XRT runtime libraries from \${_flm_xrt_src}\")

        set(_flm_libdir \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\")
        file(MAKE_DIRECTORY \"\${_flm_libdir}\")

        # Copy every libxrt*.so* (from the XRT dir) plus libboost_program_options.so*
        # (searched across the standard system dirs, since Boost may live apart
        # from XRT — e.g. XRT under /opt/xilinx but Boost in the multiarch dir),
        # preserving the symlink chain.
        file(GLOB _flm_xrt_files \"\${_flm_xrt_src}/libxrt*.so*\")
        foreach(_bdir \"\${_flm_xrt_src}\" \"/usr/lib/x86_64-linux-gnu\" \"/usr/lib\")
            file(GLOB _flm_boost_files \"\${_bdir}/libboost_program_options.so*\")
            if(_flm_boost_files)
                list(APPEND _flm_xrt_files \${_flm_boost_files})
                break()
            endif()
        endforeach()
        foreach(_f \${_flm_xrt_files})
            get_filename_component(_n \"\${_f}\" NAME)
            if(IS_SYMLINK \"\${_f}\")
                file(READ_SYMLINK \"\${_f}\" _t)
                execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink
                    \"\${_t}\" \"\${_flm_libdir}/\${_n}\")
            else()
                file(COPY \"\${_f}\" DESTINATION \"\${_flm_libdir}\"
                    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                     GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
            endif()
        endforeach()

        # Re-root the real XRT libs at \$ORIGIN so they find each other in lib/.
        file(GLOB _flm_xrt_real \"\${_flm_libdir}/libxrt*.so.*\")
        foreach(_l \${_flm_xrt_real})
            if(NOT IS_SYMLINK \"\${_l}\")
                execute_process(
                    COMMAND \"${PATCHELF_EXECUTABLE}\" --set-rpath \"\$ORIGIN\" \"\${_l}\"
                    RESULT_VARIABLE _rc)
                if(NOT _rc EQUAL 0)
                    message(FATAL_ERROR \"patchelf --set-rpath failed for \${_l}\")
                endif()
            endif()
        endforeach()

        # Mirror the bundled XRT libs under lib/x86_64-linux-gnu for XRT's lookup.
        set(_flm_multiarch \"\${_flm_libdir}/x86_64-linux-gnu\")
        file(MAKE_DIRECTORY \"\${_flm_multiarch}\")
        file(GLOB _flm_xrt_all \"\${_flm_libdir}/libxrt*.so*\")
        foreach(_l \${_flm_xrt_all})
            get_filename_component(_n \"\${_l}\" NAME)
            if(IS_SYMLINK \"\${_l}\")
                file(READ_SYMLINK \"\${_l}\" _t)
                execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink
                    \"../\${_t}\" \"\${_flm_multiarch}/\${_n}\")
            else()
                execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink
                    \"../\${_n}\" \"\${_flm_multiarch}/\${_n}\")
            endif()
        endforeach()
    ")
    # Bundle FFTW so the portable tree does not depend on libfftw3 on the host.
    install(CODE "
        set(_flm_libdir \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\")
        set(_flm_fftw_files \"\")
        foreach(_dir \"${_flm_rt_dep_dir}\" \"/usr/lib/x86_64-linux-gnu\" \"/usr/lib\")
            if(_dir)
                file(GLOB _dir_fftw \"\${_dir}/libfftw3*.so*\")
                list(APPEND _flm_fftw_files \${_dir_fftw})
            endif()
        endforeach()
        foreach(_f \${_flm_fftw_files})
            get_filename_component(_n \"\${_f}\" NAME)
            if(IS_SYMLINK \"\${_f}\")
                file(READ_SYMLINK \"\${_f}\" _t)
                execute_process(COMMAND \"\${CMAKE_COMMAND}\" -E create_symlink
                    \"\${_t}\" \"\${_flm_libdir}/\${_n}\")
            else()
                file(COPY \"\${_f}\" DESTINATION \"\${_flm_libdir}\"
                    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                     GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
            endif()
        endforeach()
        file(GLOB _flm_fftw_real \"\${_flm_libdir}/libfftw3*.so.*\")
        foreach(_l \${_flm_fftw_real})
            if(NOT IS_SYMLINK \"\${_l}\")
                execute_process(
                    COMMAND \"${PATCHELF_EXECUTABLE}\" --set-rpath \"\$ORIGIN\" \"\${_l}\"
                    RESULT_VARIABLE _rc)
            endif()
        endforeach()
    ")
elseif(NOT FLM_USE_HRX)
    # System XRT (.deb): XRT is an external dependency (the libxrt-npu2 package)
    # and flm's other shared libraries are resolved from the system via
    # ${shlibs:Depends}. Nothing is bundled here.
    install(TARGETS flm RUNTIME DESTINATION "${FLM_BIN_DESTINATION}")
else()
    # HRX: bundle flm's third-party runtime dependencies into
    # ${CMAKE_INSTALL_LIBDIR} (libhrx plus the engine .so private deps).
    #
    # Subtlety: some private dependencies (libgomp, libmvec, ...) are pulled in
    # ONLY by the prebuilt engine .so, not by flm itself. The engine .so are
    # installed and RUNPATH-patched explicitly into FLM_ENGINE_LIB_DESTINATION
    # above, so the resolver must NOT emit a second, unpatched copy of them
    # (which would retain the absolute build-machine RUNPATH) — but their private
    # deps DO need to be in the closure.
    #
    # A RUNTIME_DEPENDENCY_SET rooted only at flm cannot express this: listing
    # the engine .so in POST_EXCLUDE_REGEXES also stops traversal INTO them, so
    # their private deps silently drop out of the bundle (that is exactly how
    # libgomp/libmvec went missing). Instead, scan flm together with the engine
    # .so as ROOTS via file(GET_RUNTIME_DEPENDENCIES): roots are traversed for
    # their dependencies but are never themselves emitted, so libgomp/libmvec are
    # captured while the engine .so stay solely under the patched install above.
    # The engine names are still POST_EXCLUDEd as a belt-and-suspenders guard for
    # the case of one engine lib depending on another.
    install(TARGETS flm RUNTIME DESTINATION "${FLM_BIN_DESTINATION}")
    install(CODE "
        set(_flm_engine_libs \"${so_libs}\")
        set(_flm_engine_post_exclude \"${_flm_engine_post_exclude}\")
        file(GET_RUNTIME_DEPENDENCIES
            EXECUTABLES \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${FLM_BIN_DESTINATION}/flm\"
            LIBRARIES \${_flm_engine_libs}
            RESOLVED_DEPENDENCIES_VAR _flm_resolved
            UNRESOLVED_DEPENDENCIES_VAR _flm_unresolved
            DIRECTORIES
                \"${FLM_ENGINE_LIB_DIR}\"
                \"${CMAKE_SOURCE_DIR}/lib\"
                \"${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib\"
                \"${_flm_rt_dep_dir}\"
            PRE_EXCLUDE_REGEXES
                \"linux-vdso\\\\.so.*\"
                \"ld-linux.*\"
                \"lib(c|dl|m|pthread|rt)\\\\.so.*\"
                \"${_flm_rt_pre_exclude}\"
            POST_EXCLUDE_REGEXES \${_flm_engine_post_exclude})
        if(_flm_unresolved)
            message(FATAL_ERROR
                \"FastFlowLM runtime dependency closure is incomplete; unresolved: \${_flm_unresolved}\")
        endif()
        # file(INSTALL) prepends \$ENV{DESTDIR} itself, so DESTINATION must not.
        file(INSTALL
            DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\"
            TYPE SHARED_LIBRARY
            FOLLOW_SYMLINK_CHAIN
            FILES \${_flm_resolved})
    ")
endif()

install(FILES model_list.json DESTINATION "${FLM_SHARE_DESTINATION}")
install(FILES model_info.json DESTINATION "${FLM_SHARE_DESTINATION}")

# xclbins, which are loaded by shared libraries need to be in location
# relative to the executable, so we install them relative to the binary.
install(DIRECTORY xclbins DESTINATION "${FLM_SHARE_DESTINATION}")

# Open-kernel harness (open_kernels/harness/run_kernel): an N-buffer opcode-3
# runner for IRON designs, driven by the same .cfg language phlegm's driver
# speaks. A developer tool, not part of flm; off by default.
option(FLM_BUILD_OPEN_KERNELS_HARNESS "Build open_kernels/harness/run_kernel" OFF)
if(FLM_BUILD_OPEN_KERNELS_HARNESS)
    add_subdirectory("${CMAKE_SOURCE_DIR}/../open_kernels/harness"
                     "${CMAKE_BINARY_DIR}/open_kernels_harness")
endif()
