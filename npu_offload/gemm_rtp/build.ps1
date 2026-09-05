# Build every open_npue design set, in order.
#
#   cd C:\dev\mlir-aie; . .\iron_env.ps1        # MUST be dot-sourced
#   cd <repo>; .\npu_offload\gemm_rtp\build.ps1
#
# ~3-4 minutes per family, five families, so budget ~20 minutes. Families that
# are already built are skipped; -Force rebuilds, -Only <name> does one.
#
# THE FLAGS ARE IN families.json, NOT HERE. One machine-readable source that
# this script builds from and check_design_sets.py verifies against, so there
# is no second copy to drift. They used to be five pasteable commands in
# README.md, and every way of pasting them wrong has now happened to somebody:
#
#   * `<dst>` -- PowerShell rejects `<` as a reserved operator during PARSING,
#     so it dies with "The '<' operator is reserved for future use", naming
#     neither the placeholder nor the substitution that was forgotten.
#   * `$dst` -- PowerShell does NOT error on an undefined variable, it expands
#     it to nothing, so `--out $dst\BERT-...` becomes `--out \BERT-...` and
#     resolves to the DRIVE ROOT. Builds fine, three minutes a family, and puts
#     the set where nothing will ever look for it. Four of them, once.
#   * Two at once -- purge() deletes matching entries from the SHARED
#     ~/.npu/cache on content markers, and qkv/attn_out depend on neither
#     --gated-ffn nor --intermediate, so the two hidden-768 families own
#     identical markers for 8 of their 16 entries and each deletes the other's
#     builds. export_gemm_rtp.py holds a lock now and refuses in under a
#     second; this script is how not to meet it.

[CmdletBinding()]
param(
    # Build one family instead of all five, by name (e.g. BERT-h1024-bfp16).
    [string] $Only = "",
    # Where the sets go. Defaults to the tree this script lives in.
    [string] $Dst = "",
    # Rebuild even if the set is already there.
    [switch] $Force
)

$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
if (-not $Dst) { $Dst = (Join-Path $here '..\..\src\xclbins') }
$Dst = [IO.Path]::GetFullPath($Dst)

$spec = Get-Content (Join-Path $here 'families.json') -Raw | ConvertFrom-Json
$common = $spec.common

# The IRON toolchain, checked before four minutes of work rather than after.
# Without `. .\iron_env.ps1` the failure is `ModuleNotFoundError: No module
# named 'aie'`, which reads as a broken checkout rather than a shell that was
# never set up.
& python -c "import aie.iron" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "The IRON toolchain is not on this shell's path." -ForegroundColor Red
    Write-Host ""
    Write-Host "    cd C:\dev\mlir-aie"
    Write-Host "    . .\iron_env.ps1        # MUST be dot-sourced"
    Write-Host ""
    Write-Host "Then re-run this script. (Also: XILINX_XRT must stay UNSET --"
    Write-Host "it poisons Windows builds. Use XRT_ROOT.)"
    exit 1
}

$families = $spec.families
if ($Only) {
    $families = @($families | Where-Object { $_.name -eq $Only })
    if (-not $families) {
        Write-Host "No family named '$Only'. Known:" -ForegroundColor Red
        $spec.families | ForEach-Object { Write-Host "  $($_.name)" }
        exit 1
    }
}

Write-Host "Building into $Dst"
Write-Host ""
$t_all = Get-Date
$built = 0; $skipped = 0; $failed = @()

foreach ($f in $families) {
    $out = Join-Path $Dst $f.name
    if ((Test-Path (Join-Path $out 'gemm_rtp\design.json')) -and -not $Force) {
        Write-Host ("  {0,-24} already built (use -Force to rebuild)" -f $f.name)
        $skipped++
        continue
    }
    Write-Host ("  {0,-24} {1}" -f $f.name, ($f.serves -join ', '))
    if (Test-Path $out) { Remove-Item $out -Recurse -Force }

    $t0 = Get-Date
    Push-Location $here
    # 2>&1 | Out-String rather than a redirection: PowerShell 5.1 wraps a
    # native command's stderr in ErrorRecords, which under -ErrorActionPreference
    # Stop aborts the whole script on a build that merely printed a warning.
    $ErrorActionPreference = 'Continue'
    $log = & python export_gemm_rtp.py @($f.args + $common) --out $out 2>&1 | Out-String
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    Pop-Location

    $secs = [int]((Get-Date) - $t0).TotalSeconds
    $n = 0
    if (Test-Path (Join-Path $out 'gemm_rtp')) {
        $n = (Get-ChildItem (Join-Path $out 'gemm_rtp') -File).Count
    }
    if ($code -ne 0) {
        Write-Host "    FAILED (exit $code) after ${secs}s" -ForegroundColor Red
        ($log -split "`n" | Select-Object -Last 15) | ForEach-Object { "      $_" }
        $failed += $f.name
    } else {
        Write-Host "    ok  $n files, ${secs}s" -ForegroundColor Green
        $built++
    }
}

Write-Host ""
Write-Host ("built {0}, skipped {1}, failed {2}  ({3:N0} s total)" -f `
    $built, $skipped, $failed.Count, ((Get-Date) - $t_all).TotalSeconds)

if ($failed.Count) {
    Write-Host "failed: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}

# The spec and the sets it produced must agree. This catches a flag edited in
# families.json without a rebuild, and a set built by some other route.
Write-Host ""
Write-Host "Checking the built sets against families.json:"
Push-Location $here
& python check_design_sets.py --xclbins $Dst
$rc = $LASTEXITCODE
Pop-Location
exit $rc
