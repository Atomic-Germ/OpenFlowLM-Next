@echo off
setlocal enabledelayedexpansion
::
:: Build flm from a clean CMake cache, on Windows.
::
::   Open "x64 Native Tools Command Prompt for VS", then:
::       clean_build.bat [vcpkg-root]
::
:: It does NOT set up the MSVC environment -- the Developer Command Prompt has
:: already done that, and a script that re-derives it is liability rather than
:: convenience. If cl, cmake or ninja are missing it says which and stops.
::
:: WHY "CLEAN" IS THE WHOLE POINT. If a configure fails partway -- and on a
:: fresh clone the first one does, see below -- CMake leaves a cache behind
:: that records CMAKE_TOOLCHAIN_FILE but never ran it. CMake does NOT re-apply
:: a toolchain file to an existing cache, so every later configure of that
:: directory runs with VCPKG_TOOLCHAIN false. src\CMakeLists.txt then takes its
:: "bare self-hosted CI runner" branch, which hardcodes C:/dev/boost_1_88_0 and
:: links libboost_program_options-vc143-mt-x64-1_88 by raw name. That file does
:: not exist on a normal machine -- vcpkg installs 1.91/vc145, shared -- so 332
:: files compile for ten minutes and the link fails on a Boost nobody asked for.
::
:: A failed configure does not merely cost a retry: it POISONS the build tree
:: into silently selecting different dependencies. Deleting it is the fix, and
:: doing that unconditionally is cheaper than explaining when to.
::
:: The other traps this exists to absorb:
::
::   * The wrong vcpkg. Visual Studio ships its own under VC\vcpkg and sets
::     VCPKG_ROOT to it; that tree has the toolchain file and none of the
::     packages, so trusting the variable picks the broken one on exactly the
::     machines that have VS. Candidates are checked for boost_program_options,
::     not merely for vcpkg.cmake.
::   * sentencepiece links third_party/absl with a SYMBOLIC link, which Windows
::     allows only under Developer Mode or elevation. src\CMakeLists.txt makes a
::     junction instead -- but abseil is fetched inside that same add_subdirectory,
::     so on a truly fresh clone the target does not exist on the first pass.
::     This script therefore retries the configure ONCE.
::   * flm.exe on PATH. If the build produced nothing, typing flm.exe in
::     src\build silently runs the INSTALLED FastFlowLM instead, and you get
::     "unrecognised option '--embeddingmodel'" -- an error about a flag, which
::     sends you looking for the flag rather than for the binary.
::
:: It does NOT build the AIE design sets. Those need the IRON toolchain, a
:: different environment entirely, and about twenty minutes:
::     cd C:\dev\mlir-aie ^& . .\iron_env.ps1        (PowerShell, dot-sourced)
::     npu_offload\gemm_rtp\build.ps1
:: This script checks whether they are there and says so at the end.

set "REPO=%~dp0"
if "%REPO:~-1%"=="\" set "REPO=%REPO:~0,-1%"

:: ---------------------------------------------------------------- vcpkg
::
:: DO NOT simply trust %VCPKG_ROOT%. Visual Studio ships its own vcpkg at
:: ...\VC\vcpkg and sets VCPKG_ROOT to it, and that tree has
:: scripts\buildsystems\vcpkg.cmake but none of the packages this build needs.
:: Preferring the variable therefore picks the WRONG vcpkg on exactly the
:: machines that have Visual Studio -- which is all of them.
::
:: So each candidate is checked for the package we actually need, not merely
:: for the toolchain file. A vcpkg without boost_program_options fails here, at
:: second zero, instead of as LNK1181 ten minutes into a build.
set "VCPKG="
if not "%~1"=="" (
    call :try_vcpkg "%~1"
    if not defined VCPKG (
        echo ERROR: "%~1" is not a usable vcpkg root.
        echo        It needs installed\x64-windows\share\boost_program_options,
        echo        i.e. `vcpkg install boost-program-options:x64-windows`.
        exit /b 1
    )
)
if not defined VCPKG call :try_vcpkg "C:\dev\vcpkg"
if not defined VCPKG if defined VCPKG_ROOT call :try_vcpkg "%VCPKG_ROOT%"
if not defined VCPKG (
    echo ERROR: no vcpkg tree found with boost_program_options installed for
    echo        x64-windows. Looked at C:\dev\vcpkg and %%VCPKG_ROOT%%
    echo        ^(currently "%VCPKG_ROOT%"^).
    echo.
    echo        Note that Visual Studio's bundled vcpkg under VC\vcpkg sets
    echo        VCPKG_ROOT but ships none of the packages, so it will not do.
    echo        Pass a usable root as the first argument.
    exit /b 1
)
echo Using vcpkg at %VCPKG%

:: ---------------------------------------------------------------- toolchain
::
:: RUN THIS FROM A DEVELOPER COMMAND PROMPT. It does not call vcvars64.bat and
:: deliberately does not go looking for one: the Developer Command Prompt
:: already puts cl, cmake and ninja on PATH, and an earlier version of this
:: script that located and called vcvars itself was pure liability -- it bought
:: nothing the shell does not already provide, and vcvars64.bat prints
:: "'vswhere.exe' is not recognized" on this machine all by itself, which then
:: looked like a fault in this script.
::
:: So: check, name what is missing, and stop.
set "MISSINGTOOL="
where cl.exe    >nul 2>&1 || set "MISSINGTOOL=!MISSINGTOOL! cl"
where cmake.exe >nul 2>&1 || set "MISSINGTOOL=!MISSINGTOOL! cmake"
where ninja.exe >nul 2>&1 || set "MISSINGTOOL=!MISSINGTOOL! ninja"
if defined MISSINGTOOL (
    echo ERROR: not on PATH:!MISSINGTOOL!
    echo.
    echo        Run this from "x64 Native Tools Command Prompt for VS"
    echo        ^(Start menu, under Visual Studio^). A plain cmd will not do:
    echo        without the MSVC environment the compile fails with
    echo        "Cannot open include file: 'cstdint'", which reads as a broken
    echo        checkout rather than a shell that was never set up.
    echo.
    echo        If ninja alone is missing, install the Visual Studio component
    echo        "C++ CMake tools for Windows".
    exit /b 1
)

:: ---------------------------------------------------------------- configure
if exist "%REPO%\src\build" (
    echo Removing the existing build tree -- see the note at the top of this file.
    rmdir /s /q "%REPO%\src\build"
)

echo.
echo === configure, pass 1 ===
call :configure
if not errorlevel 1 goto :configured

echo.
echo Pass 1 failed. On a fresh clone this is expected once: sentencepiece
echo fetches abseil-cpp during configure and only then links third_party\absl,
echo so the junction has nothing to point at yet. Retrying now that abseil
echo is present.
echo.
echo === configure, pass 2 ===
rmdir /s /q "%REPO%\src\build" 2>nul
call :configure
if errorlevel 1 (
    echo.
    echo ERROR: configure failed twice. The output above is the real
    echo        diagnostic -- this script has nothing to add to it.
    exit /b 1
)
:configured

:: ---------------------------------------------------------------- build
echo.
echo === build ===
cmake --build "%REPO%\src\build" --target flm
if errorlevel 1 (
    echo.
    echo ERROR: build failed. If the link asks for a Boost that is not
    echo        installed, the vcpkg toolchain did not take effect -- which
    echo        this script exists to prevent, so please report it.
    exit /b 1
)

if not exist "%REPO%\src\build\flm.exe" (
    echo ERROR: the build reported success but there is no flm.exe.
    exit /b 1
)

:: ---------------------------------------------------------------- report
echo.
for %%A in ("%REPO%\src\build\flm.exe") do echo Built %%~fA  ^(%%~zA bytes^)
echo.
echo Run it BY FULL PATH the first time:
echo     "%REPO%\src\build\flm.exe" --version
echo Typing bare `flm.exe` runs whichever one PATH finds first, which on a
echo machine with FastFlowLM installed is the OTHER one -- and it fails with
echo "unrecognised option '--embeddingmodel'", an error about a flag rather
echo than about the binary.
echo.

set "MISSING="
for %%F in (BERT-h384-bfp16 BERT-h384-bf16 BERT-h768-bfp16 BERT-h768-gated-bfp16 BERT-h1024-bfp16) do (
    if not exist "%REPO%\src\xclbins\%%F\gemm_rtp\design.json" set "MISSING=!MISSING! %%F"
)
if defined MISSING (
    echo The AIE design sets are NOT built:!MISSING!
    echo An open_npue model will refuse to load until they are. In a PowerShell
    echo with the IRON toolchain dot-sourced:
    echo     cd C:\dev\mlir-aie; . .\iron_env.ps1
    echo     %REPO%\npu_offload\gemm_rtp\build.ps1
) else (
    echo All five AIE design sets are present.
)
endlocal
exit /b 0

:: ---------------------------------------------------------------- subroutines
::
:: The cmake line lives here rather than in a variable. Building it as a string
:: means quoting quotes, and the toolchain path routinely contains spaces --
:: which produced `Could not find toolchain file: "C:/Program"` and a warning
:: about an "extra path from command line".
:configure
cmake -S "%REPO%\src" -B "%REPO%\src\build" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DFLM_VERSION=0.9.25 -DNPU_VERSION=0.9.25 ^
    -DFLM_USE_HRX=OFF ^
    -DCMAKE_TOOLCHAIN_FILE="%VCPKG%\scripts\buildsystems\vcpkg.cmake"
exit /b %errorlevel%

:: Accept a vcpkg root only if it has both the toolchain file AND the package
:: this build actually needs. Sets VCPKG on success, leaves it undefined
:: otherwise, so the caller can fall through to the next candidate.
:try_vcpkg
if not exist "%~1\scripts\buildsystems\vcpkg.cmake" exit /b 0
if not exist "%~1\installed\x64-windows\share\boost_program_options" exit /b 0
set "VCPKG=%~1"
exit /b 0
