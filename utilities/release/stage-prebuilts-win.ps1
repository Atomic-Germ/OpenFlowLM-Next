<#
.SYNOPSIS
    Stage the Windows-only build dependencies of a release into a prebuilts bundle.

.DESCRIPTION
    Runs on a Windows machine that has the Ryzen AI NPU driver installed. It
    produces, in -Dest\win:

      xrt-include\xrt\...        the XRT headers oflms compiles against
      xrt-lib\xrt_coreutil.lib   the import library for the driver's DLL

    which is everything a Windows CI runner cannot produce for itself: the
    driver is the NPU, and neither the headers nor the import library are in
    the repository. The release workflow reads them from the prebuilts branch
    (utilities/release/stage-prebuilts.sh --require-windows) and builds
    oflm-setup.msi against them.

    Run this FIRST, then stage-prebuilts.sh with the same -Dest, and push. The
    recipe is the one in src/WinSetup.md, section "The standalone open-engine
    CLI: what XRT actually has to supply"; the comments here say which step of
    that recipe is load-bearing and why.

.PARAMETER Dest
    The bundle directory stage-prebuilts.sh will assemble into. Defaults to
    .\out-prebuilts next to this script.

.PARAMETER XrtTag
    XRT tag to take the headers from. 2.21.75 is the tag src/CMakeLists.txt
    pins for portable builds, and the compile-time version in
    version-slim.h has to agree with the tree.

.PARAMETER DriverDll
    xrt_coreutil.dll from the installed driver. The driver puts it in
    System32 or System32\AMD depending on the version; both are looked for.

.EXAMPLE
    .\stage-prebuilts-win.ps1 -Dest ..\..\out-prebuilts
#>
[CmdletBinding()]
param(
    [string]$Dest,
    [string]$XrtTag = '2.21.75',
    [string]$DriverDll
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
if (-not $Dest) { $Dest = Join-Path $ScriptDir 'out-prebuilts' }
$Win = Join-Path $Dest 'win'
$Include = Join-Path $Win 'xrt-include'
$LibDir = Join-Path $Win 'xrt-lib'
New-Item -ItemType Directory -Force -Path $Include, $LibDir | Out-Null

# --------------------------------------------------------------------- driver
# The import library is made FROM the driver's DLL: that is the only place
# xrt_coreutil.dll exists, and the ABI has to be the driver's.
if (-not $DriverDll) {
    $candidates = @(
        "$env:SystemRoot\System32\xrt_coreutil.dll",
        "$env:SystemRoot\System32\AMD\xrt_coreutil.dll"
    ) | Where-Object { Test-Path $_ }
    if ($candidates.Count -eq 0) {
        throw @"
xrt_coreutil.dll not found. This is the NPU runtime that ships with the Ryzen AI
driver, so this step needs a machine with the driver installed (and it is why the
Windows build cannot happen on a CI runner). Install the driver, or pass
-DriverDll <path to xrt_coreutil.dll>.
"@
    }
    $DriverDll = $candidates[0]
}
Write-Host "Driver DLL: $DriverDll"
Write-Host "  (the recorded driver version is what oflm's NPU_VERSION floor is"
Write-Host "   checked against at run time, so a stale driver here is a release"
Write-Host "   that cannot run on a current machine)"

# ------------------------------------------------------------- MSVC toolchain
# lib.exe and dumpbin.exe, found rather than assumed: PATH in a plain
# PowerShell is not a developer prompt, and the release workflow is not going to
# open one.
function Find-MsvcTool([string]$Tool) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found; is Visual Studio installed?" }
    $vs = & $vswhere -latest -products * `
                       -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
                       -property installationPath
    if (-not $vs) { throw "no Visual Studio with the C++ toolset (VC.Tools.x86.x64)" }
    $exe = Get-ChildItem -Path (Join-Path $vs 'VC\Tools\MSVC') -Filter "$Tool.exe" `
                         -Recurse -ErrorAction SilentlyContinue |
           Where-Object { $_.FullName -match 'Hostx64\\x64' } |
           Sort-Object FullName -Descending | Select-Object -First 1
    if (-not $exe) { throw "$Tool.exe not found under $vs" }
    return $exe.FullName
}

$dumpbin = Find-MsvcTool 'dumpbin'
$lib     = Find-MsvcTool 'lib'
Write-Host "dumpbin: $dumpbin"
Write-Host "lib:     $lib"

# ------------------------------------------------------------- import library
# dumpbin /exports prints an ordinal table; the .def wants one name per line.
# (gendef from mingw-w64-tools produces the .def directly and is the route
# src/WinSetup.md documents for WSL, but it is not something a Windows box has
# by default, and this way needs nothing but Visual Studio.)
Write-Host "Reading exports from the driver DLL..."
$exports = & $dumpbin /exports $DriverDll
if ($LASTEXITCODE -ne 0) { throw "dumpbin /exports failed ($LASTEXITCODE)" }

# The table is "  ordinal  RVA   name"; the header lines and the summary line
# ("N exports") are not names. A name is an identifier at the end of a line.
$names = $exports |
    ForEach-Object {
        if ($_ -match '^\s+[0-9]+\s+[0-9A-Fa-f]+\s+([A-Za-z_][A-Za-z0-9_@]*)\s*$') { $Matches[1] }
        elseif ($_ -match '^\s+[0-9]+\s+(\S+)\s*$') { $null }   # ordinal-only row: data, skip
    } | Where-Object { $_ } | Sort-Object -Unique
if ($names.Count -lt 100) {
    throw "parsed only $($names.Count) export names out of the driver DLL; the dumpbin output format changed and this script would write a broken .def"
}

$def = Join-Path $LibDir 'xrt_coreutil.def'
@("LIBRARY xrt_coreutil.dll", "EXPORTS") + $names | Set-Content -Encoding ascii $def
Write-Host "  $($names.Count) exports -> $(Split-Path -Leaf $def)"

$libOut = Join-Path $LibDir 'xrt_coreutil.lib'
& $lib /def:$def /machine:x64 /out:$libOut
if ($LASTEXITCODE -ne 0) { throw "lib /def failed ($LASTEXITCODE)" }
Remove-Item $def
if (-not (Test-Path $libOut)) { throw "lib reported success but wrote no $libOut" }
Write-Host "  import library: $libOut ($((Get-Item $libOut).Length) bytes)"

# -------------------------------------------------------------------- headers
# Sparse, shallow, pinned: the headers are a few MB and the repository is not.
$scratch = Join-Path $env:TEMP ("xrt-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
Write-Host "Checking out XRT $XrtTag headers..."
git clone --depth 1 --filter=blob:none --sparse `
    --branch $XrtTag https://github.com/Xilinx/XRT.git $scratch
if ($LASTEXITCODE -ne 0) { throw "git clone of XRT failed" }
try {
    Push-Location $scratch
    git sparse-checkout set src/runtime_src/core/include src/CMake/config
    if ($LASTEXITCODE -ne 0) { throw "git sparse-checkout failed" }
    Pop-Location

    $src = Join-Path $scratch 'src\runtime_src\core\include'
    if (-not (Test-Path $src)) { throw "the sparse checkout has no $src" }

    # version-slim.h is the load-bearing step and the one that is easy to skip.
    # It is GENERATED by XRT's own CMake from src/CMake/config/version-slim.h.in,
    # and building only the headers skips that. xrt/detail/abi.h includes it, so
    # without it every translation unit that touches an XRT header fails with
    # C1083 -- an error about a header that the repository plainly contains,
    # which is the worst kind.
    $slim = Join-Path $src 'xrt\detail\version-slim.h'
    if (-not (Test-Path $slim)) {
        $tmpl = Join-Path $scratch 'src\CMake\config\version-slim.h.in'
        if (-not (Test-Path $tmpl)) {
            throw @"
Neither xrt\detail\version-slim.h nor its template src\CMake\config\version-slim.h.in
is in the $XrtTag checkout. The tag moved; see the recipe in src/WinSetup.md.
"@
        }
        $version = $XrtTag -replace '^v', ''
        $parts = $version.Split('.')
        $body = (Get-Content $tmpl -Raw) `
            -replace '@XRT_VERSION_MAJOR@', $parts[0] `
            -replace '@XRT_VERSION_MINOR@', $parts[1] `
            -replace '@XRT_VERSION@', $version
        # Do not trust the substitution to have covered the file: a placeholder
        # that survived becomes a compile error about an undefined macro, far
        # from the cause.
        $left = [regex]::Matches($body, '@[A-Za-z0-9_]+@') | ForEach-Object { $_.Value } |
                Sort-Object -Unique
        if ($left) {
            throw "version-slim.h still has unsubstituted placeholders: $($left -join ', '). XRT's template changed; the recipe is in src/WinSetup.md."
        }
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $slim) | Out-Null
        Set-Content -Encoding ascii -NoNewline $slim $body
        Write-Host "  generated xrt/detail/version-slim.h from the template ($version)"
    } else {
        Write-Host "  xrt/detail/version-slim.h came from the checkout"
    }

    Copy-Item -Recurse -Force (Join-Path $src 'xrt') (Join-Path $Include 'xrt')
}
finally {
    if ((Get-Location).Path -eq $scratch) { Pop-Location }
    Remove-Item -Recurse -Force $scratch -ErrorAction SilentlyContinue
}

# --------------------------------------------------------------------- report
$n = (Get-ChildItem -Recurse -File $Include).Count
Write-Host ""
Write-Host "win\ staged in $Win"
Write-Host "  xrt-include: $n header files"
Write-Host "  xrt-lib:     xrt_coreutil.lib from $DriverDll"
Write-Host ""
Write-Host "Next:  utilities/release/stage-prebuilts.sh --dest `"$Dest`" --require-windows"
Write-Host "from the Linux NPU machine (or WSL), and push the npu-prebuilts branch."
