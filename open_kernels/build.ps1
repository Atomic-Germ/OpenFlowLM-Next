# Build the six open Qwen3.6-MoE kernel sets (Windows).
#
#   cd C:\dev\mlir-aie; . .\iron_env.ps1        # MUST be dot-sourced
#   cd <repo>; .\open_kernels\build.ps1
#
# A few minutes for all six. Sets that are already built (final.xclbin +
# insts.bin present) are skipped; -Force rebuilds, -Only takes a
# comma-separated subset, -Dst redirects. The one command behind this script
# is export_qwen36_kernels.py, which owns the SETS table (design source,
# build dir, compile-time knobs) -- see also src/open_qwen36/README.md. The
# device needs no flag: build_design.py pins npu2 itself (without it IRON
# silently targets NPU1).
#
# The set names are NOT repeated here: they are read out of the exporter's
# SETS table, so there is no second copy to drift. (Same discipline as
# npu_offload/gemm_rtp/build.ps1 reading families.json.)
#
# Windows-specific preflight, beyond the IRON check below: aiebu-asm. The
# exporter wraps every set's instructions into insts.elf, and aiecc resolves
# that tool from PATH. The BERT sets never ask for ELF, so a box that builds
# those can still miss this one -- check first, fail fast with the fix.

[CmdletBinding()]
param(
    # Build a comma-separated subset instead of all six (e.g. -Only "lx0,ln").
    [string] $Only = "",
    # Where the sets go. Defaults to the tree this script lives in.
    [string] $Dst = "",
    # Rebuild even if the set is already there.
    [switch] $Force
)

$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
if (-not $Dst) { $Dst = (Join-Path $here '..\..\src\xclbins\Qwen3.6-35B-A3B-NPU2\open_kernels') }
$Dst = [IO.Path]::GetFullPath($Dst)

# The IRON toolchain, checked before minutes of work rather than after.
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
    Write-Host "Then re-run this script."
    exit 1
}

# aiebu-asm (and xclbinutil): resolved from PATH by aiecc. xclbinutil is
# proven by any BERT build on this box; aiebu-asm is new with these sets
# (insts.elf per set) and will not be. Both ship with XRT.
$missingTools = @()
foreach ($t in @('xclbinutil', 'aiebu-asm')) {
    if (-not (Get-Command $t -ErrorAction SilentlyContinue)) { $missingTools += $t }
}
if ($missingTools.Count -gt 0) {
    Write-Host ("Not on PATH: {0}." -f ($missingTools -join ', ')) -ForegroundColor Red
    Write-Host "They ship with XRT -- make sure its bin directory is on PATH"
    Write-Host "in the shell that dot-sourced iron_env.ps1."
    exit 1
}

# The set names, out of the exporter's SETS table rather than a copy here.
Push-Location $here
$ErrorActionPreference = 'Continue'
$setsOut = & python -c "import export_qwen36_kernels as e; print(','.join(e.SETS))" 2>&1 | Out-String
$code = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
Pop-Location
if ($code -ne 0) {
    Write-Host "Could not read the set list from export_qwen36_kernels.py:" -ForegroundColor Red
    ($setsOut -split "`n" | Select-Object -Last 5) | ForEach-Object { "      $_" }
    exit 1
}
$all = @($setsOut.Trim() -split ',' | Where-Object { $_ -ne '' })
if ($Only) {
    $want = @($Only -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' })
    $bad = @($want | Where-Object { $all -notcontains $_ })
    if ($bad.Count -gt 0) {
        Write-Host ("No set(s) named '{0}'. Known: {1}" -f ($bad -join ', '), ($all -join ', ')) -ForegroundColor Red
        exit 1
    }
} else {
    $want = $all
}

Write-Host "Building into $Dst"
Write-Host ""
$t_all = Get-Date
$skipped = 0
$todo = @()
foreach ($n in $want) {
    if ((Test-Path (Join-Path $Dst "$n/final.xclbin")) -and (Test-Path (Join-Path $Dst "$n/insts.bin")) -and -not $Force) {
        Write-Host ("  {0,-12} already built (use -Force to rebuild)" -f $n)
        $skipped++
        continue
    }
    $todo += $n
}

if ($todo.Count -eq 0) {
    Write-Host ("built 0, skipped {0}, failed 0" -f $skipped)
    exit 0
}

Write-Host ("  building: {0}" -f ($todo -join ','))
$t0 = Get-Date
Push-Location $here
$ErrorActionPreference = 'Continue'
$log = & python export_qwen36_kernels.py --out $Dst --only ($todo -join ',') 2>&1 | Out-String
$code = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
Pop-Location

$secs = [int]((Get-Date) - $t0).TotalSeconds
if ($code -ne 0) {
    Write-Host "    FAILED (exit $code) after ${secs}s" -ForegroundColor Red
    ($log -split "`n" | Select-Object -Last 15) | ForEach-Object { "      $_" }
    exit 1
}
Write-Host ("    ok  ({0:N0} s total)" -f ((Get-Date) - $t_all).TotalSeconds) -ForegroundColor Green
Write-Host ("built {0}, skipped {1}, failed 0" -f $todo.Count, $skipped)
