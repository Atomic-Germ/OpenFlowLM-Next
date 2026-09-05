@echo off
setlocal enabledelayedexpansion
::
:: Build flm from a clean CMake cache, on Windows, from a plain cmd prompt.
::
::   clean_build.bat [vcpkg-root] [--ironbuild]
::
:: --ironbuild also builds the NPU kernels with the IRON toolchain after flm
:: links: the five BERT design sets via npu_offload\gemm_rtp\build.ps1
:: (~20 min, skipping sets that are already built) and the six Qwen3.6-MoE
:: sets via open_kernels\build.ps1. Same flag as clean_build.sh --ironbuild
:: on Linux. build.ps1 needs `import aie.iron` to work in the PowerShell it
:: runs under -- i.e. the iron_env.ps1 environment -- and checks that itself
:: before spending any build time. The Qwen sets additionally need aiebu-asm
:: on PATH (XRT tool; the BERT sets never ask for it).
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
::   * vcvars64.bat. Without it the compile fails with "Cannot open include
::     file: 'cstdint'", which reads as a broken checkout rather than a shell
::     that was never set up. Located via vswhere and loaded automatically,
::     so a plain cmd prompt works -- no Developer Command Prompt needed.
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
:: Without --ironbuild this script does NOT build the AIE design sets. Those
:: need the IRON toolchain, a different environment entirely, and about
:: twenty minutes:
::     cd C:\dev\mlir-aie ^& . .\iron_env.ps1        (PowerShell, dot-sourced)
::     npu_offload\gemm_rtp\build.ps1
:: With --ironbuild it runs that build.ps1 itself after flm links, and skips
:: the presence report below (build.ps1 ends by checking the sets itself).
:: Without the flag it checks whether the sets are there and says so.

set "REPO=%~dp0"
if "%REPO:~-1%"=="\" set "REPO=%REPO:~0,-1%"

:: First non-flag argument is the vcpkg root, so --ironbuild may come first
:: or last. A shift loop rather than `%~1`: empty-safe when no arguments are
:: given, and quote-safe for vcpkg paths containing spaces. Flag comparisons
:: only below: no ')' anywhere in here, per the note in the MSVC section
:: about %ProgramFiles(x86)% and parenthesised blocks.
set "VCPKG_ARG="
set "IRONBUILD="
:parse_args
if "%~1"=="" goto parse_done
if /i "%~1"=="--ironbuild" set "IRONBUILD=1"
if /i "%~1"=="/ironbuild" set "IRONBUILD=1"
if /i not "%~1"=="--ironbuild" if /i not "%~1"=="/ironbuild" if not defined VCPKG_ARG set "VCPKG_ARG=%~1"
shift
goto parse_args
:parse_done

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
if not "%VCPKG_ARG%"=="" (
    call :try_vcpkg "%VCPKG_ARG%"
    if not defined VCPKG (
        echo ERROR: "%VCPKG_ARG%" is not a usable vcpkg root.
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

:: ---------------------------------------------------------------- MSVC
::
:: VSWHERE is set OUTSIDE the if-block on purpose. %ProgramFiles(x86)% contains
:: a literal ')' and cmd parses a parenthesised block by scanning for the first
:: unescaped ')', so setting it inside one closes the block early -- which
:: showed up as a spurious "'vswhere.exe' is not recognized" while the script
:: otherwise worked.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

where cl.exe >nul 2>&1
if errorlevel 1 (
    if not exist "!VSWHERE!" (
        echo ERROR: vswhere.exe not found, and cl.exe is not on PATH.
        echo        Open a "Developer Command Prompt for VS" and re-run.
        exit /b 1
    )
    :: Via a temp file rather than `for /f "usebackq"`. Backquoting a command
    :: whose executable path is quoted makes cmd print
    :: "'vswhere.exe' is not recognized" even when it resolves the path
    :: correctly -- noise that reads like a real failure in an otherwise
    :: working script.
    "!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath > "%TEMP%\flm_vspath.txt" 2>nul
    set "VSPATH="
    if exist "%TEMP%\flm_vspath.txt" set /p VSPATH=<"%TEMP%\flm_vspath.txt"
    del "%TEMP%\flm_vspath.txt" 2>nul
    if not exist "!VSPATH!\VC\Auxiliary\Build\vcvars64.bat" (
        echo ERROR: no vcvars64.bat under "!VSPATH!"
        exit /b 1
    )
    echo Loading MSVC environment from !VSPATH!
    call "!VSPATH!\VC\Auxiliary\Build\vcvars64.bat" >nul
    if errorlevel 1 ( echo ERROR: vcvars64.bat failed & exit /b 1 )
)

where cmake.exe >nul 2>&1
if errorlevel 1 ( echo ERROR: cmake.exe is not on PATH. & exit /b 1 )
where ninja.exe >nul 2>&1
if errorlevel 1 ( echo ERROR: ninja.exe is not on PATH ^(it ships with the VS "C++ CMake tools" component^). & exit /b 1 )

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

if defined IRONBUILD (
    echo.
    echo === design sets [IRON] ===
    powershell -NoProfile -ExecutionPolicy Bypass -File "%REPO%\npu_offload\gemm_rtp\build.ps1"
    if errorlevel 1 (
        echo.
        echo ERROR: design-set build failed. The output above is the real
        echo        diagnostic -- if it says the IRON toolchain is not on the
        echo        shell's path, dot-source iron_env.ps1 first, see above.
        exit /b 1
    )
    echo.
    echo === kernels [IRON] ===
    powershell -NoProfile -ExecutionPolicy Bypass -File "%REPO%\open_kernels\build.ps1"
    if errorlevel 1 (
        echo.
        echo ERROR: Qwen kernel build failed. The output above is the real
        echo        diagnostic -- if it names aiebu-asm, that XRT tool is not
        echo        on PATH (the BERT sets never ask for it, so a box that
        echo        builds those can still miss it).
        exit /b 1
    )
    echo.
    echo flm.exe, all five AIE design sets and all six Qwen kernels are built.
) else (
    set "MISSING="
    for %%F in (BERT-h384-bfp16 BERT-h384-bf16 BERT-h768-bfp16 BERT-h768-gated-bfp16 BERT-h1024-bfp16) do (
        if not exist "%REPO%\src\xclbins\%%F\gemm_rtp\design.json" set "MISSING=!MISSING! %%F"
    )
    if defined MISSING (
        echo The AIE design sets are NOT built:!MISSING!
        echo An open_npue model will refuse to load until they are. Either
        echo re-run with --ironbuild to build them now ^(IRON toolchain,
        echo ~20 minutes^), or in a PowerShell with the IRON toolchain
        echo dot-sourced:
        echo     cd C:\dev\mlir-aie; . .\iron_env.ps1
        echo     %REPO%\npu_offload\gemm_rtp\build.ps1
    ) else (
        echo All five AIE design sets are present.
    )
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
