@echo off
REM Build the open Qwen3.6 engine's standalone CLI on Windows with MSVC against
REM the system XRT — the same setup open_kernels/harness/build.cmd uses — and
REM the manifest unit test (no XRT), which it then runs against the fixture.
REM Required from the environment (neither ships with this repo):
REM   XRT_INCLUDE_DIR  .../XRT/src/runtime_src/core/include  (a Xilinx/XRT checkout)
REM   XRT_LIB_DIR      directory holding xrt_coreutil.lib   (import lib from the system
REM                    xrt_coreutil.dll -- see ..\WinSetup.md)
REM   VCVARS64         vcvars64.bat (default: VS 2022 BuildTools)
setlocal
cd /d "%~dp0"
if "%VCVARS64%"=="" set "VCVARS64=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%VCVARS64%" >nul
if errorlevel 1 goto :vcfail
if not exist out mkdir out
echo [open_qwen36] manifest_test
cl /nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS /bigobj ^
   /I ".." /I "..\include" /I "..\..\open_kernels\harness" ^
   manifest_test.cpp manifest.cpp /Fe:out\manifest_test.exe /Fo:out\
if errorlevel 1 goto :clfail
out\manifest_test.exe ..\..\specs\open-engine\tests\fixtures\manifest_qwen36.json ..\..\specs\open-engine\tests\fixtures\manifest_qwen3_4b.json ..\..\specs\open-engine\tests\fixtures\manifest_gemma3_4b.json ..\..\specs\open-engine\tests\fixtures\manifest_hy_mt2_7b.json ..\..\specs\open-engine\tests\fixtures\manifest_qwen35_9b.json
if errorlevel 1 goto :testfail
echo [open_qwen36] pools_test
cl /nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS /bigobj ^
   /I ".." /I "..\include" /I "..\..\open_kernels\harness" ^
   pools_test.cpp pools.cpp manifest.cpp q4nx_file.cpp /Fe:out\pools_test.exe /Fo:out\
if errorlevel 1 goto :clfail
out\pools_test.exe
if errorlevel 1 goto :testfail
echo [open_qwen36] vit_test
cl /nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS /bigobj /openmp /arch:AVX2 /fp:fast ^
   /I "." /I ".." /I "..\include" ^
   vision\vit_test.cpp vision\vit.cpp q4nx_file.cpp /Fe:out\vit_test.exe /Fo:out\
if errorlevel 1 goto :clfail
REM vit_test needs the model's vision_weight.q4nx and a fixture from replica_vit.py; run it by hand:
REM   python ..\..\open_kernels\model\replica_vit.py --grid 16 16 --no-hf --fixture <dir>
REM   out\vit_test.exe %USERPROFILE%\.flm\models\Qwen3.6-35B-A3B-NPU2 <dir>
REM The two unit tests above need no XRT; the CLI does.
if "%XRT_INCLUDE_DIR%"=="" goto :noxrt
if "%XRT_LIB_DIR%"=="" goto :noxrt
echo [open_qwen36] XRT_INCLUDE_DIR=%XRT_INCLUDE_DIR%
cl /nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS /bigobj ^
   /I "%XRT_INCLUDE_DIR%" /I ".." /I "..\include" /I "..\..\open_kernels\harness" ^
   q4nx_file.cpp pools.cpp manifest.cpp core.cpp cli.cpp "%XRT_LIB_DIR%\xrt_coreutil.lib" ^
   /Fe:out\open_qwen36_cli.exe /Fo:out\
if errorlevel 1 goto :clfail
echo [open_qwen36] OK -^> out\open_qwen36_cli.exe
exit /b 0
:noxrt
echo [open_qwen36] set XRT_INCLUDE_DIR (XRT\src\runtime_src\core\include) and XRT_LIB_DIR (dir with xrt_coreutil.lib)
exit /b 1
:vcfail
echo [open_qwen36] vcvars64 failed: "%VCVARS64%"
exit /b 1
:clfail
echo [open_qwen36] compile FAILED
exit /b 1
:testfail
echo [open_qwen36] manifest_test FAILED
exit /b 1
