<#
.SYNOPSIS
    Check that the staged MSI payload actually satisfies oflm.exe.

.DESCRIPTION
    oflm.wxs installs whatever get_files.bat put in src\wix\package, so the MSI
    is as complete as the staging step was. That step is a copy of wildcards
    and a vcpkg tree, and the failure mode is not a build error: a DLL that was
    not copied is not referenced by anything at install time, the MSI builds,
    it installs, and oflm.exe dies on first run with "the code execution cannot
    proceed because X.dll was not found" -- on the user's machine, after an
    admin install, with the logs saying the installation succeeded.

    So: read the import table of oflm.exe and of every engine library beside it,
    and require each imported DLL to be either in the payload or a Windows
    system DLL. This is the check the old hand-written <File> list could not be:
    that list was a claim about the payload, and nothing compared it to the
    binary's imports.

    Run from the repository root after src\wix\get_files.bat. Exits non-zero on
    a missing DLL.
#>
[CmdletBinding()]
param(
    [string]$PackageDir = 'src/wix/package',
    [string]$XclbinDir = 'src/xclbins'
)

$ErrorActionPreference = 'Stop'

# Windows' own DLLs, and the CRT that ships with the OS. api-ms-win-* and
# ext-ms-* are API sets: contracts resolved by the loader, never files.
$SystemDlls = @(
    'kernel32.dll', 'kernelbase.dll', 'ntdll.dll', 'user32.dll', 'gdi32.dll',
    'advapi32.dll', 'shell32.dll', 'ole32.dll', 'oleaut32.dll', 'comdlg32.dll',
    'comctl32.dll', 'ws2_32.dll', 'mswsock.dll', 'iphlpapi.dll', 'crypt32.dll',
    'bcrypt.dll', 'secur32.dll', 'netapi32.dll', 'userenv.dll', 'wtsapi32.dll',
    'setupapi.dll', 'shlwapi.dll', 'psapi.dll', 'dbghelp.dll', 'winmm.dll',
    'imm32.dll', 'version.dll', 'rpcrt4.dll', 'sechost.dll', 'sspicli.dll',
    'cfgmgr32.dll', 'powrprof.dll', 'pdh.dll', 'mscoree.dll', 'ucrtbase.dll',
    'vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll', 'concrt140.dll',
    'msvcp140_1.dll', 'msvcp140_2.dll', 'msvcp140_codecvt_ids.dll'
)
function Test-SystemDll([string]$name) {
    $n = $name.ToLowerInvariant()
    return ($SystemDlls -contains $n) -or $n.StartsWith('api-ms-win-') `
        -or $n.StartsWith('ext-ms-')
}

function Find-MsvcTool([string]$Tool) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found" }
    $vs = & $vswhere -latest -products * `
                       -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
                       -property installationPath
    $exe = Get-ChildItem -Path (Join-Path $vs 'VC\Tools\MSVC') -Filter "$Tool.exe" `
                         -Recurse -ErrorAction SilentlyContinue |
           Where-Object { $_.FullName -match 'Hostx64\\x64' } |
           Sort-Object FullName -Descending | Select-Object -First 1
    if (-not $exe) { throw "$Tool.exe not found under $vs" }
    return $exe.FullName
}

$exe = Join-Path $PackageDir 'oflm.exe'
if (-not (Test-Path $exe)) {
    throw "$exe is not there. Run cmd /c src\wix\get_files.bat first (or src\wix\build.ps1, which does it for you)."
}

$dumpbin = Find-MsvcTool 'dumpbin'
$staged = @(Get-ChildItem -File $PackageDir | ForEach-Object { $_.Name.ToLowerInvariant() })

Write-Host "Payload: $($staged.Count) files in $PackageDir"

# The binaries whose imports have to be satisfiable: oflm.exe and every engine
# library, because the engine dlopen()s those and a missing dependency fails
# later and less legibly than a missing one next to the exe.
$targets = @($exe) + @(Get-ChildItem -File $PackageDir -Filter *.dll |
                       ForEach-Object { $_.FullName })

$missing = @{}
$checked = 0
foreach ($target in $targets) {
    $out = & $dumpbin /dependents $target 2>$null
    if ($LASTEXITCODE -ne 0) { continue }
    foreach ($line in $out) {
        $m = $line -match '^\s+([A-Za-z0-9_.\-]+\.dll)\s*$'
        if (-not $m) { continue }
        $dep = $Matches[1]
        if (Test-SystemDll $dep) { continue }
        if ($staged -contains $dep.ToLowerInvariant()) { continue }
        if (-not $missing.ContainsKey($dep)) { $missing[$dep] = @() }
        $missing[$dep] += (Split-Path -Leaf $target)
    }
    $checked++
}
Write-Host "Checked the import table of $checked binaries."

# The kernel sets, the other thing the MSI ships by glob. A release whose MSI
# has no xclbins installs an engine that cannot open a model, and the xclbins
# are ~40x the size of everything else in the MSI, so an empty one is a
# staging mistake rather than an intent.
$familyCount = 0
if (Test-Path $XclbinDir) {
    $familyCount = @(Get-ChildItem -Directory $XclbinDir).Count
}
if ($familyCount -eq 0) {
    Write-Error @"
No xclbin families under $XclbinDir. The MSI would install an engine and no
kernels. Stage the prebuilts (utilities/release/stage-prebuilts.sh) before
building the installer.
"@
    exit 1
}
Write-Host "xclbins: $familyCount families"

if ($missing.Count -gt 0) {
    foreach ($dep in ($missing.Keys | Sort-Object)) {
        Write-Error ("{0} is imported by {1} and is in neither the payload nor Windows" -f
                     $dep, (($missing[$dep] | Sort-Object -Unique) -join ', '))
    }
    Write-Error @"
$($missing.Count) DLL(s) the payload does not contain. get_files.bat copies
src\lib\*.dll, src\lib\xrt\*.dll and %VCPKG_ROOT%\installed\x64-windows\bin\*.dll;
one of those is the set this build actually linked against, and a DLL missing from
it is invisible until the installed oflm.exe is run. Add the port to
utilities/release/vcpkg-ports.txt, or the file to src\lib.
"@
    exit 1
}

Write-Host "Every imported DLL is either in the payload or a Windows system DLL."
exit 0
