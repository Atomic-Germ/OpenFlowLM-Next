@echo off
REM Build run_kernel.exe on Windows with MSVC against the system XRT.
REM
REM Needs: an XRT header tree and an import lib for xrt_coreutil.dll. Neither
REM ships with this repo; both are required, from the environment:
REM   XRT_INCLUDE_DIR  .../XRT/src/runtime_src/core/include  (a Xilinx/XRT checkout)
REM   XRT_LIB_DIR      directory holding xrt_coreutil.lib   (an import lib made from
REM                    C:\Windows\System32\xrt_coreutil.dll -- see src\WinSetup.md)
REM   VCVARS64         vcvars64.bat (default: VS 2022 BuildTools)
setlocal
cd /d "%~dp0"
if "%VCVARS64%"=="" set "VCVARS64=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if "%XRT_INCLUDE_DIR%"=="" goto :noxrt
if "%XRT_LIB_DIR%"=="" goto :noxrt
REM vcvarsall locates the toolchain with vswhere; make sure it is reachable.
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%VCVARS64%" >nul
if errorlevel 1 goto :vcfail
if not exist out mkdir out
echo [harness] XRT_INCLUDE_DIR=%XRT_INCLUDE_DIR%
echo [harness] XRT_LIB_DIR=%XRT_LIB_DIR%
cl /nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS ^
   /I "%XRT_INCLUDE_DIR%" run_kernel.cpp "%XRT_LIB_DIR%\xrt_coreutil.lib" ^
   /Fe:out\run_kernel.exe /Fo:out\
if errorlevel 1 goto :clfail
echo [harness] OK -^> out\run_kernel.exe
exit /b 0
:noxrt
echo [harness] set XRT_INCLUDE_DIR (XRT\src\runtime_src\core\include) and XRT_LIB_DIR (dir with xrt_coreutil.lib)
exit /b 1
:vcfail
echo [harness] vcvars64 failed: "%VCVARS64%"
exit /b 1
:clfail
echo [harness] compile FAILED
exit /b 1
