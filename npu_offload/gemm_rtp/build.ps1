# Build every open_npue design set, in order.
#
#   cd C:\dev\mlir-aie; . .\iron_env.ps1        # MUST be dot-sourced
#   cd <repo>; .\npu_offload\gemm_rtp\build.ps1
#
# ~3-4 minutes per family, five families, so budget ~20 minutes.
#
# WHY A SCRIPT AND NOT FIVE PASTED COMMANDS. Both ways of getting this wrong
# have now actually happened:
#
#   * The README used a `<dst>` placeholder. PowerShell rejects `<` as a
#     reserved operator during PARSING, so a pasted command dies with
#     "The '<' operator is reserved for future use" -- naming neither the
#     placeholder nor the substitution that was forgotten.
#   * Running two of them in parallel corrupts both. `purge()` deletes matching
#     entries from the SHARED ~/.npu/cache and matches on content markers, and
#     `qkv`/`attn_out` depend on neither --gated-ffn nor --intermediate -- so
#     the two hidden-768 families own identical markers for 8 of their 16
#     entries and each deletes the other's builds. It surfaces minutes later as
#     a FileNotFoundError on a cache hash. export_gemm_rtp.py holds a lock now
#     and refuses in under a second, but the way not to meet it is this script.
#
# EVERY FLAG IS LOAD-BEARING. A set is selected at load time by hidden,
# intermediate, gated_ffn AND the datapath, so a set built with the wrong flags
# is not a slower design -- it is one the wrong model loads, or none does. The
# two that are easy to get wrong are called out at their families below.

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

$families = @(
    @{
        name = 'BERT-h384-bfp16'
        note = 'all-minilm:l6-v2'
        args = @('--hidden','384','--intermediate','1536','--qkv-n','1152',
                 '--emulate-bfp16','--c-bf16','-n','48')
    }
    @{
        name = 'BERT-h384-bf16'
        # THE ONE FAMILY WITH NEITHER FLAG. bge-small failed the bfp16 MTEB
        # gate bit-reproducibly, so it stays on the plain datapath -- and its C
        # stays fp32. Adding --c-bf16 here builds a design that loads and is
        # not the one that passed the gates.
        note = 'bge-small:en-v1.5  (plain bf16, C as fp32 -- note BOTH flags absent)'
        args = @('--hidden','384','--intermediate','1536','--qkv-n','1152',
                 '-n','48')
    }
    @{
        name = 'BERT-h768-bfp16'
        note = 'bge-base:en-v1.5'
        args = @('--hidden','768','--intermediate','3072','--qkv-n','2304',
                 '--emulate-bfp16','--c-bf16','-n','48')
    }
    @{
        name = 'BERT-h768-gated-bfp16'
        note = 'nomic-embed-text:v1.5 AND gte-multilingual:base'
        args = @('--hidden','768','--intermediate','3072','--qkv-n','2304',
                 '--gated-ffn','--emulate-bfp16','--c-bf16','-n','48')
    }
    @{
        name = 'BERT-h1024-bfp16'
        # -n 32, not 48: the design asserts N % (tile_n * n_cols) == 0 and
        # bge-large's N is in {1024, 3072, 4096}. 64 divides them but needs
        # 65,536 B of a 63 KB L1 budget.
        note = 'bge-large:en-v1.5  (tile_n 32)'
        args = @('--hidden','1024','--intermediate','4096','--qkv-n','3072',
                 '--emulate-bfp16','--c-bf16','-n','32')
    }
)

# Shared by every family: the four batch tiers, and the software-pipelined
# runtime sequence. --tg-depth 2 is worth 1.034-1.141x of array time,
# bit-identical; --tg-depth 3 compiles clean and then TIMES OUT on hardware.
$common = @('--batches','4,16,32,128','--tg-depth','2','--tb-rows','4')

if ($Only) {
    $families = @($families | Where-Object { $_.name -eq $Only })
    if (-not $families) {
        Write-Host "No family named '$Only'. Known:" -ForegroundColor Red
        (Get-Variable families -ValueOnly) | ForEach-Object { "  $($_.name)" }
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
    Write-Host ("  {0,-24} {1}" -f $f.name, $f.note)
    if (Test-Path $out) { Remove-Item $out -Recurse -Force }

    $t0 = Get-Date
    Push-Location $here
    # 2>&1 | Out-String rather than a redirection: PowerShell 5.1 wraps a
    # native command's stderr in ErrorRecords, which with -ErrorActionPreference
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

# The commands above and the sets they produced must agree. This is the check
# that catches a flag edited in one place and not the other -- and the README
# is the artifact here, since the sets are not in git.
Write-Host ""
Write-Host "Checking the README against what was built:"
Push-Location $here
& python check_readme.py --xclbins $Dst
$rc = $LASTEXITCODE
Pop-Location
exit $rc
