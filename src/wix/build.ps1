<#
.SYNOPSIS
    Builds the oflm MSI installer with WiX v5.

.DESCRIPTION
    Runs get_files.bat to stage the payload and then `wix build` against
    oflm.wxs, dropping the resulting oflm-setup.msi into <repo root>/output/.
    Requires WiX Toolset v5 (wix.exe) on PATH.

.PARAMETER OutputDir
    Directory to write oflm-setup.msi into. Defaults to src/wix/output.

.PARAMETER Version
    The MSI ProductVersion, as MAJOR.MINOR.PATCH. Passed to oflm.wxs as
    ProductVersion. The default reads OFLM_VERSION out of the repository-root
    CMakePresets.json, which is the documented single source of truth for the
    version, so a local build is the same version the release would be.

.EXAMPLE
    ./build.ps1
    ./build.ps1 -Version 1.2.3 -OutputDir C:\artifacts
#>

[CmdletBinding()]
param(
    [string]$OutputDir,
    [string]$Version
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
if (-not $OutputDir) {
    $OutputDir = Join-Path $ScriptDir "output"
}

if (-not $Version) {
    $presets = Get-Content (Join-Path $RepoRoot "CMakePresets.json") -Raw | ConvertFrom-Json
    $Version = ($presets.configurePresets |
        Where-Object { $_.name -eq "common-default" }).cacheVariables.OFLM_VERSION
}
if ($Version -notmatch '^\d{1,3}\.\d{1,3}\.\d{1,3}$') {
    throw "'$Version' is not MAJOR.MINOR.PATCH. The MSI ProductVersion is three numeric fields, and an MSI whose version is 0.0.0 installs over nothing and upgrades nothing."
}
Write-Host "ProductVersion: $Version"

if (-not (Get-Command wix -ErrorAction SilentlyContinue)) {
    throw "wix.exe not found on PATH. Install WiX Toolset v5 (https://wixtoolset.org/) first."
}

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}
$OutputDir = (Resolve-Path $OutputDir).Path
$msiPath = Join-Path $OutputDir "oflm-setup.msi"

Push-Location $ScriptDir
try {
    Write-Host "Running get_files.bat to stage package\"
    cmd.exe /c "$ScriptDir\get_files.bat"
    if ($LASTEXITCODE -ne 0) {
        throw "get_files.bat failed with exit code $LASTEXITCODE"
    }

    Write-Host "Building oflm.wxs -> $msiPath"
    wix build oflm.wxs -arch x64 -ext WixToolset.UI.wixext `
        -d "ProductVersion=$Version" -out $msiPath
    if ($LASTEXITCODE -ne 0) {
        throw "wix build failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host "Built $msiPath"
