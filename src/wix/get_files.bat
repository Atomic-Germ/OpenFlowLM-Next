@echo off
REM Stage the MSI payload into .\package\, which is what oflm.wxs globs.
REM
REM TWO THINGS THIS DOES THAT THE INNO STAGING SCRIPT (src\inno\get_files.bat)
REM DID NOT, both of which were silent on a fresh machine:
REM
REM   1. It empties package\ first. oflm.wxs installs whatever is in there, so
REM      a file left by an earlier build of a different shape -- an oflm.exe from
REM      before a rename, a DLL nobody ships any more -- ships in the MSI.
REM   2. It copies the vcpkg runtime DLLs, and copies them LAST. A build with
REM      the vcpkg toolchain (src\build-windows-vcpkg.cmd, and the release
REM      workflow) links oflm.exe against the import libraries in
REM      %VCPKG_ROOT%\installed\x64-windows\lib, so those are the DLLs it has to
REM      find at run time. The checked-in src\lib\*.dll are an older set --
REM      libcurl, FFmpeg 7, the CRT -- and are the fallback for a build that
REM      linked the checked-in import libraries instead. Same order as
REM      build-windows-vcpkg.cmd's out\ staging, deliberately.

setlocal
cd /d "%~dp0"

if exist package rmdir /s /q package
mkdir package

echo Copying oflm.exe...
if not exist "..\build\oflm.exe" (
    echo ERROR: ..\build\oflm.exe is not there. Build first: cmake --build build --config Release
    exit /b 1
)
copy /y "..\build\oflm.exe" "package\oflm.exe" >nul || exit /b 1

echo Copying the checked-in DLLs (engine libraries, then shared runtime)...
copy /y "..\lib\xrt\*.dll" "package\" >nul 2>&1
copy /y "..\lib\*.dll"     "package\" >nul 2>&1

echo Copying the vcpkg runtime DLLs...
if defined VCPKG_ROOT (
    copy /y "%VCPKG_ROOT%\installed\x64-windows\bin\*.dll" "package\" >nul 2>&1
) else (
    echo WARNING: VCPKG_ROOT is not set, so no vcpkg runtime DLLs are staged.
    echo          If oflm.exe was built with the vcpkg toolchain, the installed
    echo          MSI will fail to start with a missing DLL.
)

echo Copying the model registry and the icon...
copy /y "..\model_list.json" "package\model_list.json" >nul || exit /b 1
copy /y "..\model_info.json" "package\model_info.json" >nul || exit /b 1
copy /y "..\inno\logo.ico"   "package\logo.ico"        >nul || exit /b 1

REM terms.rtf is deliberately NOT staged: oflm.wxs reads it straight out of
REM src\inno for WixUILicenseRtf, and the payload glob would otherwise install
REM the licence text into Program Files.

echo.
echo Done: %CD%\package
endlocal
exit /b 0
