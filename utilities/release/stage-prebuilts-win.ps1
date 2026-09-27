<#
.SYNOPSIS
    Stage the Windows build dependencies of a release: the XRT headers and the
    xrt_coreutil import library, checked in and pushed to the staging branch.

.DESCRIPTION
    Runs on a Windows machine that has the Ryzen AI NPU driver installed, and
    produces, in the repository:

      prebuilts\win\xrt-include\xrt\...       the XRT headers oflm compiles against
      prebuilts\win\xrt-lib\xrt_coreutil.lib  the import library for the driver's DLL
      prebuilts\manifest.json                 this script's section of it

    which is everything a GitHub Windows runner cannot produce for itself: the
    driver IS the NPU, the import library is made from the driver's own DLL, and
    the headers come from XRT with one generated header filled in by hand. The
    release workflow builds oflm-setup.msi against them, from the tag.

    THE CONTRACT WITH THE LINUX MACHINE. utilities/release/stage-prebuilts.sh
    does the same dance for the open kernel sets, and the two of you take turns
    on the same branch. What is shared is the layout and the manifest, so neither
    script needs to know what the other staged:

      src\xclbins\<family>\open_kernels\**   stage-prebuilts.sh (the kernels)
      prebuilts\win\**                       this script (the XRT inputs)
      prebuilts\manifest.json                both, one section each

    Each script owns exactly one section of the manifest and rewrites only that
    section, so running them in either order, any number of times, is safe.

    GUARDS. The branch must be the staging branch, and the working tree must have
    no changes outside the paths the two scripts own: a release commit that
    happens to carry an uncommitted source edit is a release nobody reviewed and
    no CI run tested.

    The recipe this follows, and the reason each step is load-bearing, is
    src/WinSetup.md, section "The standalone open-engine CLI: what XRT actually
    has to supply".

.PARAMETER XrtTag
    XRT tag to take the headers from. 2.21.75 is the tag src/CMakeLists.txt pins
    for portable builds, and the compile-time version in version-slim.h has to
    agree with the tree.

.PARAMETER DriverDll
    xrt_coreutil.dll from the installed driver. The driver puts it in System32
    or System32\AMD depending on the version; both are looked for.

.PARAMETER Branch
    The staging branch. Default: staging.

.PARAMETER NoCommit
    Write prebuilts\win\ and the manifest, stage them, and stop.

.PARAMETER NoPush
    Commit locally and stop.

.EXAMPLE
    .\stage-prebuilts-win.ps1
    .\stage-prebuilts-win.ps1 -NoCommit      # look before you commit
#>
[CmdletBinding()]
param(
    [string]$XrtTag = '2.21.75',
    [string]$DriverDll,
    [string]$Branch = $(if ($env:OFLM_STAGING_BRANCH) { $env:OFLM_STAGING_BRANCH } else { 'staging' }),
    [switch]$NoCommit,
    [switch]$NoPush
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$Manifest = Join-Path $RepoRoot 'prebuilts\manifest.json'
# Captured into a scratch directory first and only moved into the repository once
# it has passed the checks below: a half-captured bundle that is committed is a
# broken MSI on the next tag.
# GetTempPath() rather than $env:TEMP: it is TEMP on Windows and TMPDIR
# elsewhere, and this script is also run under pwsh on Linux and macOS when
# checking the half of it that is not Windows-only.
$Dest = Join-Path ([System.IO.Path]::GetTempPath()) ('oflm-win-prebuilts-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
$Win = Join-Path $Dest 'win\'

# --------------------------------------------------------------------- guards
$current = (git -C $RepoRoot branch --show-current).Trim()
if ($current -ne $Branch) {
    throw @"
you are on '$current', not '$Branch'. The prebuilts are committed to a branch that
is not main, and a release is tagged from it. One-time setup, from a clean clone
of main:

    git checkout -b $Branch
    git push -u origin $Branch

Then merge main into it at the start of each release cycle:

    git merge main
"@
}

# Changes under the two paths the prebuild scripts own are expected (a re-run
# after a rebuild, a half-finished capture); anything else is not.
$owned = '^(src/xclbins/[^/]+/open_kernels[^/]*/|src/xclbins/BERT-h[^/]*/|prebuilts/)'
$status = @(git -C $RepoRoot status --porcelain)
$foreign = @($status | ForEach-Object { $_ -replace '^...', '' } | Where-Object { $_ -notmatch $owned })
if ($foreign.Count -gt 0) {
    $foreign | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    throw @"
the working tree has changes outside the prebuilt kernels and prebuilts\.
Commit or stash them first: a release must not be the commit that happens to
carry uncommitted work.
"@
}
if ($status.Count -gt 0) {
    Write-Host "==> $($status.Count) existing change(s) under the prebuilt paths"
}

git -C $RepoRoot fetch --quiet origin $Branch
if ($LASTEXITCODE -ne 0) { throw "cannot fetch origin/$Branch" }
$behind = (git -C $RepoRoot rev-list --count "HEAD..origin/$Branch").Trim()
if ($behind -ne '0') {
    throw @"
you are $behind commit(s) behind origin/${Branch}. The other developer pushed
first. Pull and re-run:

    git pull origin ${Branch}
"@
}
Write-Host "==> up to date with origin/$Branch ($((git -C $RepoRoot rev-parse --short "origin/$Branch").Trim()))"
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
$scratch = Join-Path ([System.IO.Path]::GetTempPath()) ("xrt-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
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

# -------------------------------------------------------------------- verify
# Everything the release workflow needs, checked here, on the one machine that
# can check it. A bundle that is only wrong in a way nobody can see until the MSI
# is installed on a user's laptop is the failure mode this exists to prevent.
$headers = @(Get-ChildItem -Recurse -File $Include)
$n = $headers.Count
if ($n -lt 10) {
    throw @"
only $n header file(s) under $Include. The sparse checkout is meant to pull the
whole xrt/ tree; a handful of files means the XRT layout moved at $XrtTag.
See the recipe in src/WinSetup.md.
"@
}
foreach ($required in 'xrt\xrt.h', 'xrt\detail\abi.h', 'xrt\detail\version-slim.h') {
    $p = Join-Path $Include $required
    if (-not (Test-Path $p)) { throw "xrt-include is missing $required" }
    if ($required -like '*version-slim.h') {
        # A placeholder that survived the substitution is a compile error about an
        # undefined macro, a long way from the cause.
        $left = [regex]::Matches((Get-Content $p -Raw), '@[A-Za-z0-9_]+@') |
                ForEach-Object { $_.Value } | Sort-Object -Unique
        if ($left) {
            throw "version-slim.h still has unsubstituted placeholders: $($left -join ', ')"
        }
    }
}
$libBytes = (Get-Item $libOut).Length
if ($libBytes -lt 1024) {
    throw @"
xrt_coreutil.lib is $libBytes bytes. An import library with a handful of exports is
kilobytes; this is one the lib.exe run above did not finish.
"@
}
$driverVersion = (Get-Item $DriverDll).VersionInfo.FileVersion
if (-not $driverVersion) { $driverVersion = 'unknown' }
Write-Host ""
Write-Host "==> $n header files, xrt_coreutil.lib $libBytes bytes, driver $driverVersion"

# -------------------------------------------------------------- into the repo
# Only now, after the checks. A half-captured bundle that is committed is a broken
# MSI on the next tag; one that is still sitting in TEMP can simply be re-run.
$RepoWin = Join-Path $RepoRoot 'prebuilts\win'
if (Test-Path $RepoWin) { Remove-Item -Recurse -Force $RepoWin }
New-Item -ItemType Directory -Force -Path $RepoWin | Out-Null
foreach ($sub in 'xrt-include', 'xrt-lib') {
    Copy-Item -Recurse -Force (Join-Path $Win $sub) $RepoWin
}

# ------------------------------------------------------------------ manifest
# Read-modify-write, one key. This script owns "windows" and nothing else, so it
# never clobbers the "linux" section stage-prebuilts.sh wrote, whichever order
# the two of you run in and however often.
$versions = @()
foreach ($rel in 'CMakePresets.json', 'src\CMakePresets.json') {
    $presets = Get-Content (Join-Path $RepoRoot $rel) -Raw | ConvertFrom-Json
    $common = $presets.configurePresets | Where-Object { $_.name -eq 'common-default' }
    $versions += [string]$common.cacheVariables.OFLM_VERSION
}
if (($versions | Sort-Object -Unique).Count -ne 1) {
    throw @"
the presets disagree about OFLM_VERSION ($($versions -join ' vs ')). The release
workflow checks this too, but finding it here is twenty minutes cheaper than
finding it in CI. One of them was not bumped.
"@
}
$version = $versions[0]
if ($version -notmatch '^\d+\.\d+\.\d+$') {
    throw "OFLM_VERSION is '$version', which is not X.Y.Z. The release tag is v$version."
}

if (Test-Path $Manifest) {
    $doc = Get-Content $Manifest -Raw | ConvertFrom-Json
} else {
    $doc = [pscustomobject]@{ schema = 2; platforms = [pscustomobject]@{} }
}
if ($doc.schema -ne 2) {
    throw @"
${Manifest} is schema $($doc.schema) but this script writes schema 2.
Delete it and re-stage both platforms, or check out the version of
stage-prebuilts.sh that agrees with this one.
"@
}

$section = [pscustomobject]@{
    staged_utc       = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    for_version      = $version
    xrt_version      = ($XrtTag -replace '^v', '')
    npu_driver       = $driverVersion
    import_lib_bytes = $libBytes
    header_files     = $n
}
$doc.platforms | Add-Member -MemberType NoteProperty -Name 'windows' -Value $section -Force
# Sorted, so the file reads the same whoever wrote it and a diff of two releases
# is a diff of values rather than of key order.
$sorted = [pscustomobject]@{}
foreach ($prop in ($doc.platforms.PSObject.Properties | Sort-Object Name)) {
    $sorted | Add-Member -MemberType NoteProperty -Name $prop.Name -Value $prop.Value
}
$doc.platforms = $sorted
$doc | ConvertTo-Json -Depth 8 | Set-Content -Encoding ascii $Manifest

# Round-trip it immediately. ConvertTo-Json silently truncates nested objects at
# its -Depth and ConvertFrom-Json happily reads the truncated result back, so a
# typo in the depth turns the linux section into a string, or drops it, with no
# error at all. Compare against what we just wrote.
$check = Get-Content $Manifest -Raw | ConvertFrom-Json
if ($check.schema -ne 2) { throw 'manifest round-trip lost the schema' }
if (-not $check.platforms.windows) { throw 'manifest round-trip lost the windows section' }
if ([int]$check.platforms.windows.header_files -ne $n) {
    throw 'manifest round-trip lost fields: the serialization was truncated, not
   staged. Raise -Depth in the ConvertTo-Json above.'
}
if ($doc.platforms.PSObject.Properties.Name.Count -ne $check.platforms.PSObject.Properties.Name.Count) {
    throw 'manifest round-trip lost a platform section'
}
Write-Host "==> ${Manifest}: $($check.platforms.PSObject.Properties.Name -join ', ')"

# --------------------------------------------------------------------- commit
git -C $RepoRoot add -A -- prebuilts
$staged = @(git -C $RepoRoot diff --cached --name-only)
if ($staged.Count -eq 0) {
    Write-Host ""
    Write-Host "Nothing to stage: prebuilts\win is byte-identical to what is on $Branch."
    Write-Host "That is usually right -- a re-run after no driver or XRT change."
    exit 0
}
Write-Host ""
Write-Host "==> staging $($staged.Count) path(s) under prebuilts\"
if ($NoCommit) {
    Write-Host "==> -NoCommit: stopping with the files staged"
    exit 0
}
$msg = "prebuilts(windows): xrt headers + import lib for $version"
git -C $RepoRoot commit -m $msg
if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
Write-Host "==> committed $((git -C $RepoRoot rev-parse --short HEAD).Trim()) on $Branch"
if ($NoPush) {
    Write-Host "==> -NoPush: stopping before the push"
    exit 0
}

# ----------------------------------------------------------------------- push
# The other developer staged their side while this capture ran, which is the
# normal case rather than the exception. Merge theirs in, and if the manifest is
# the conflict, resolve it the only way that can be right: take the version from
# origin, then re-apply this machine's section on top of it. Taking it from
# origin explicitly (git show <ref>:<path>) rather than with --ours/--theirs
# because the two flags mean OPPOSITE things in a merge and a rebase, and this
# script should not be able to get it wrong.
git -C $RepoRoot push origin $Branch
if ($LASTEXITCODE -ne 0) {
    Write-Host "==> push refused: origin/$Branch moved. Merging it in..."
    git -C $RepoRoot fetch --quiet origin $Branch
    git -C $RepoRoot merge --no-edit "origin/$Branch"
    if ($LASTEXITCODE -ne 0) {
        $inConflict = @(git -C $RepoRoot diff --name-only --diff-filter=U)
        if ($inConflict -notcontains 'prebuilts/manifest.json') {
            throw @"
merge failed on $($inConflict -join ', '), which this script does not know how to
resolve. Merge by hand, then re-run.
"@
        }
        Write-Host "==> manifest conflict: taking origin's sections, re-applying windows"
        $originManifest = git -C $RepoRoot show "origin/${Branch}:prebuilts/manifest.json"
        if ($LASTEXITCODE -ne 0) {
            throw @"
origin/${Branch} has no prebuilts/manifest.json to merge with. Merge by hand,
then re-run.
"@
        }
        # git show gives the bytes as text lines; rejoin them for ConvertFrom-Json
        # so a CRLF/LF difference in the repo file is not a parse error here.
        $merged = ($originManifest -join "`n") | ConvertFrom-Json
        $merged.platforms | Add-Member -MemberType NoteProperty -Name 'windows' -Value $section -Force
        $merged | ConvertTo-Json -Depth 8 | Set-Content -Encoding ascii $Manifest
        git -C $RepoRoot add -- prebuilts/manifest.json
        git -C $RepoRoot commit --no-edit
        if ($LASTEXITCODE -ne 0) {
            throw @"
could not finish the merge after resolving the manifest. Merge by hand:

    git merge origin/$Branch
"@
        }
    }
    git -C $RepoRoot push origin $Branch
    if ($LASTEXITCODE -ne 0) { throw "push failed again. Something else moved; pull and re-run." }
}
Write-Host "==> pushed $Branch"

# The scratch capture is redundant now that the verified copy is in the
# repository. If a check above threw, it is still there to look at.
Remove-Item -Recurse -Force $Dest -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Done. Check that the manifest has both sections, then tag from ${Branch}:"
Write-Host ""
Write-Host "    git tag -a v$version -m 'OpenFlowLM $version' && git push origin v$version"
