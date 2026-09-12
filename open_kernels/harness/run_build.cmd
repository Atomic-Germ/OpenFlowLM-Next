@echo off
set "XRT_INCLUDE_DIR=C:\dev\XRT\src\runtime_src\core\include"
set "XRT_LIB_DIR=C:\dev\xrtNPUfromDLL"
call "%~dp0build.cmd" > "%~dp0build_out.txt" 2>&1
echo EXIT %ERRORLEVEL% >> "%~dp0build_out.txt"
