@echo off
REM Build open_diffusion_cli.exe -- FLUX.2 [klein] text-to-image on the NPU (Phase 6).
REM Standalone, like ..\open_whisper\build.cmd: cl.exe against the system XRT, no CMake
REM target yet.
REM
REM Required from the environment (defaults match the dev machine):
REM   XRT_INCLUDE_DIR  .../XRT/src/runtime_src/core/include
REM   XRT_LIB_DIR      directory holding xrt_coreutil.lib
REM   VCVARS64         vcvars64.bat (default: VS 2022 BuildTools)
setlocal
cd /d "%~dp0"
if "%VCVARS64%"=="" set "VCVARS64=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if "%XRT_INCLUDE_DIR%"=="" set "XRT_INCLUDE_DIR=C:/dev/XRT/src/runtime_src/core/include"
if "%XRT_LIB_DIR%"=="" set "XRT_LIB_DIR=C:/dev/xrtNPUfromDLL"
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%VCVARS64%" >nul
if errorlevel 1 goto :vcfail
if not exist out mkdir out
if not exist "%XRT_LIB_DIR%\xrt_coreutil.lib" goto :noxrt

REM DISABLE_ABI_CHECK=1: a raw XRT source checkout has no generated version-slim.h
REM (see ..\open_qwen36\build.cmd).
cl /nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS ^
   /DDISABLE_ABI_CHECK=1 /bigobj ^
   /I "%XRT_INCLUDE_DIR%" /I "." /I "..\include" /I "..\..\third_party\stb" ^
   engine.cpp cli.cpp "%XRT_LIB_DIR%\xrt_coreutil.lib" ^
   /Fe:out\open_diffusion_cli.exe /Fo:out\
if errorlevel 1 goto :clfail
echo [open_diffusion] OK -^> out\open_diffusion_cli.exe
exit /b 0

:noxrt
echo [open_diffusion] %XRT_LIB_DIR%\xrt_coreutil.lib not found -- set XRT_LIB_DIR
exit /b 1
:vcfail
echo [open_diffusion] vcvars64 failed: "%VCVARS64%"
exit /b 1
:clfail
echo [open_diffusion] compile FAILED
exit /b 1
