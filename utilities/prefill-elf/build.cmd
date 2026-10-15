@echo off
REM Build elf_probe.exe (MSVC against the system XRT, as src\open_qwen36\build.cmd does).
setlocal
cd /d "%~dp0"
if "%XRT_INCLUDE_DIR%"=="" set "XRT_INCLUDE_DIR=C:\dev\XRT\src\runtime_src\core\include"
if "%XRT_LIB_DIR%"=="" set "XRT_LIB_DIR=C:\dev\xrtNPUfromDLL"
if "%VCVARS64%"=="" set "VCVARS64=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS64%" >nul
if not exist out mkdir out
cl /nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS /DDISABLE_ABI_CHECK=1 ^
   /I "%XRT_INCLUDE_DIR%" /I "..\..\src\include" elf_probe.cpp "%XRT_LIB_DIR%\xrt_coreutil.lib" /Fe:out\elf_probe.exe /Fo:out\
if errorlevel 1 exit /b 1
echo OK out\elf_probe.exe
