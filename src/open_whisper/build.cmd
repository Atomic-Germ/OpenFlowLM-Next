@echo off
REM Build open_whisper_cli.exe -- the phase-2b Whisper encoder gate (issue #72).
REM Standalone build, modelled on ..\open_qwen36\build.cmd: no CMake target yet
REM (this is intentionally not wired into src\CMakeLists.txt -- see the phase-2b
REM brief), just cl.exe against the system XRT.
REM
REM Required from the environment (defaults below match the dev machine this
REM was built on; override if yours differs):
REM   XRT_INCLUDE_DIR  .../XRT/src/runtime_src/core/include
REM   XRT_LIB_DIR      directory holding xrt_coreutil.lib
REM   VCVARS64         vcvars64.bat
setlocal
cd /d "%~dp0"
if "%VCVARS64%"=="" set "VCVARS64=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if "%XRT_INCLUDE_DIR%"=="" set "XRT_INCLUDE_DIR=C:/dev/XRT/src/runtime_src/core/include"
if "%XRT_LIB_DIR%"=="" set "XRT_LIB_DIR=C:/dev/xrtNPUfromDLL"
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%VCVARS64%" >nul
if errorlevel 1 goto :vcfail
if not exist out mkdir out

echo [open_whisper] XRT_INCLUDE_DIR=%XRT_INCLUDE_DIR%
echo [open_whisper] XRT_LIB_DIR=%XRT_LIB_DIR%
if not exist "%XRT_LIB_DIR%\xrt_coreutil.lib" goto :noxrt

REM DISABLE_ABI_CHECK=1: a raw C:\dev\XRT source checkout has no generated
REM version-slim.h unless a build step ran in it; without the define,
REM xrt/detail/abi.h wants that header and cl fails with C1083 before ever
REM reaching this file's own code (see ..\open_qwen36\build.cmd).
cl /nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS ^
   /DDISABLE_ABI_CHECK=1 /bigobj /openmp /arch:AVX2 ^
   /I "%XRT_INCLUDE_DIR%" /I "." /I ".." /I "..\include" /I "..\open_npue" ^
   weights.cpp kernels.cpp guards.cpp host_ops.cpp encoder.cpp decoder.cpp cli.cpp ^
   "..\open_npue\npu_device.cpp" "..\open_qwen36\q4nx_file.cpp" ^
   "%XRT_LIB_DIR%\xrt_coreutil.lib" ^
   /Fe:out\open_whisper_cli.exe /Fo:out\
if errorlevel 1 goto :clfail

REM The guards: no device, no XRT, no 1.6 GB container -- guards.cpp and the
REM safetensors reader are all it needs, which is why guards.cpp is its own
REM translation unit. Built and RUN here, because a test nothing runs is the
REM same failure one step earlier.
echo [open_whisper] guards_test
cl /nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS ^
   /DDISABLE_ABI_CHECK=1 /bigobj ^
   /I "%XRT_INCLUDE_DIR%" /I "." /I ".." /I "..\include" /I "..\open_npue" ^
   guards_test.cpp guards.cpp "..\open_qwen36\q4nx_file.cpp" ^
   /Fe:out\guards_test.exe /Fo:out\
if errorlevel 1 goto :clfail
out\guards_test.exe out
if errorlevel 1 goto :testfail

echo [open_whisper] OK -^> out\open_whisper_cli.exe
exit /b 0

:testfail
echo [open_whisper] guards_test FAILED
exit /b 1

:noxrt
echo [open_whisper] %XRT_LIB_DIR%\xrt_coreutil.lib not found -- set XRT_LIB_DIR
exit /b 1
:vcfail
echo [open_whisper] vcvars64 failed: "%VCVARS64%"
exit /b 1
:clfail
echo [open_whisper] compile FAILED
exit /b 1
