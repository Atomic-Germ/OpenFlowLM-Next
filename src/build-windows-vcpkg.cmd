@echo off
REM Configure + build flm.exe on Windows with MSVC and a vcpkg toolchain.
REM
REM This is the path that worked on a bare box (2026-09-05): no standalone
REM Boost b2 build, no hand-copied import libs — every native dependency comes
REM from vcpkg in classic mode, and CMakeLists.txt's VCPKG_TOOLCHAIN branch
REM resolves them via find_package(CONFIG). One-time setup:
REM
REM   git clone https://github.com/microsoft/vcpkg C:\dev\vcpkg
REM   C:\dev\vcpkg\bootstrap-vcpkg.bat -disableMetrics
REM   C:\dev\vcpkg\vcpkg install boost-program-options boost-asio boost-beast curl ffmpeg fftw3 --triplet x64-windows
REM       (~25 min; ffmpeg is the long one)
REM   git clone --recurse-submodules https://github.com/mlc-ai/tokenizers-cpp third_party\tokenizers-cpp
REM       (.gitmodules lists it but the gitlink is not in the index)
REM   cargo (rustup, x86_64-pc-windows-msvc) on PATH — tokenizers-cpp builds a Rust crate
REM   XRT headers + xrt_coreutil.lib: see WinSetup.md (or a phlegm checkout's copies, the defaults below)
REM
REM sentencepiece's CMake creates a symlink that needs a privilege a normal
REM console session lacks; the junction below is the unprivileged equivalent.
REM
REM Overrides: VCPKG_ROOT, XRT_INCLUDE_DIR, XRT_LIB_DIR, VS_INSTANCE (the VS
REM install that has the C++ toolset — Community without "Desktop development
REM with C++" does not; BuildTools does).
setlocal
cd /d "%~dp0"
if "%VCPKG_ROOT%"=="" set "VCPKG_ROOT=C:\dev\vcpkg"
if "%XRT_INCLUDE_DIR%"=="" set "XRT_INCLUDE_DIR=C:\code\phlegm\npu-engine\deps\XRT\src\runtime_src\core\include"
if "%XRT_LIB_DIR%"=="" set "XRT_LIB_DIR=C:\code\phlegm\npu-engine\m0\out"
if "%VS_INSTANCE%"=="" set "VS_INSTANCE=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools"
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%USERPROFILE%\.cargo\bin;%PATH%"

if not exist "..\third_party\tokenizers-cpp\CMakeLists.txt" (
    echo [flm] third_party\tokenizers-cpp is missing: clone it first ^(see the header^)
    exit /b 1
)
set "SPM_TP=..\third_party\tokenizers-cpp\sentencepiece\third_party"
if exist "%SPM_TP%\abseil-cpp\absl" if not exist "%SPM_TP%\absl" (
    mklink /J "%SPM_TP%\absl" "%SPM_TP%\abseil-cpp\absl" >nul
)

cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_GENERATOR_INSTANCE="%VS_INSTANCE:\=/%" ^
  -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT:\=/%/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DXRT_INCLUDE_DIR="%XRT_INCLUDE_DIR:\=/%" -DXRT_LIB_DIR="%XRT_LIB_DIR:\=/%" ^
  -DFLM_VERSION=1.0.4 -DNPU_VERSION=32.0.203.304 -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 (
    REM sentencepiece fetches abseil during the first configure; the junction
    REM can only be made after that, so a first attempt may need a second pass.
    if exist "%SPM_TP%\abseil-cpp\absl" if not exist "%SPM_TP%\absl" (
        mklink /J "%SPM_TP%\absl" "%SPM_TP%\abseil-cpp\absl" >nul
        cmake -S . -B build
    )
    if errorlevel 1 exit /b 1
)
cmake --build build --config Release --parallel 8 --target flm
if errorlevel 1 exit /b 1

REM Stage a runnable tree in out\ (what the Makefile's `run` target does, plus
REM the vcpkg runtime DLLs).
if not exist out mkdir out
copy /y build\flm.exe out\ >nul
copy /y lib\xrt\*.dll out\ >nul
copy /y lib\*.dll out\ >nul
copy /y "%VCPKG_ROOT%\installed\x64-windows\bin\*.dll" out\ >nul
copy /y model_list.json out\ >nul
copy /y model_info.json out\ >nul
xcopy /e /i /y /q xclbins out\xclbins >nul
echo [flm] OK -^> out\flm.exe   (set FLM_MODEL_PATH=%%USERPROFILE%%\.flm from a non-interactive shell)
exit /b 0
