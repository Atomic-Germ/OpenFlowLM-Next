# Does per-step decode cost grow with context, and is that specific to a family?
#
# Runs ONE decode step at several context positions and records the timing the
# CLI prints. --at-position seeks the KV cache to P without decoding P tokens,
# so each measurement is a single dispatch and the sweep is cheap.
#
# Why: Granite 4.2 3B measured part0 91.9 ms at position 18 and 235.4 ms at
# position 80 (40 layers), about +2.3 ms per position, while lm_head stayed
# flat at 4.1 ms -- so the growth is the KV scan in attention. Granite is the
# first family measured at head_dim 64 / 40 heads AND the first measured at
# more than one position, so nothing yet says which of those two facts explains
# it. Running the same sweep on an already-validated family settles it:
#
#   grows there too  -> the dense recipe's attention at long context; Granite is
#                       merely the first anyone looked at
#   flat there       -> specific to 40 heads over 8 kv heads at head_dim 64
#
#   powershell -File open_kernels\model\sweep_positions.ps1
#
# Each family's first token is its own (make_decode.py's map): qwen3 151644
# (<|im_start|>), granite 100264 (<|start_of_role|>).

param(
    [int[]] $Positions = @(0, 256, 1024, 2048),
    [string] $Cli = "src\open_qwen36\out\open_qwen36_cli.exe",
    [int] $MaxCtx = 4096
)

$ErrorActionPreference = 'Stop'
$models = @(
    @{ name = "Qwen3-4B";   dir = "$HOME\.flm\models\Qwen3-4B-NPU2";
       kernels = "src\xclbins\Qwen3-4B-NPU2\open_kernels";        id = 151644 },
    @{ name = "Granite-3B"; dir = "$HOME\.flm\models\Granite-4.2-3B-NPU2";
       kernels = "src\xclbins\Granite-4.2-3B-NPU2\open_kernels";  id = 100264 }
)

if (-not (Test-Path $Cli)) {
    Write-Host "No $Cli -- build it first:" -ForegroundColor Red
    Write-Host '    $env:XRT_INCLUDE_DIR = "C:\Xilinx\XRT\include"'
    Write-Host '    $env:XRT_LIB_DIR     = "C:\Xilinx\XRT\lib"'
    Write-Host '    $env:VCVARS64        = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"'
    Write-Host '    cmd /c src\open_qwen36\build.cmd'
    exit 1
}

$rows = @()
foreach ($m in $models) {
    if (-not (Test-Path $m.kernels)) {
        Write-Host "skipping $($m.name): no kernel set at $($m.kernels)" -ForegroundColor Yellow
        Write-Host "  build it:  python open_kernels\export_qwen36_kernels.py --model-dir $($m.dir)"
        continue
    }
    if (-not (Test-Path $m.dir)) {
        Write-Host "skipping $($m.name): no model at $($m.dir)" -ForegroundColor Yellow
        continue
    }
    Write-Host "`n=== $($m.name) ===" -ForegroundColor Cyan
    foreach ($p in $Positions) {
        # stderr carries the timings; 2>&1 merges it so we can parse.
        $out = & $Cli --model $m.dir --kernels $m.kernels --ids $m.id `
                      --max-tokens 1 --max-ctx $MaxCtx --at-position $p 2>&1 | Out-String
        # "  step @123: 96.1 ms (part0 91.9, route 0.00, part1 0.0, lm_head 4.0)"
        $hit = [regex]::Match($out, 'step @\d+:\s*([\d.]+) ms \(part0 ([\d.]+).*?lm_head ([\d.]+)')
        if ($hit.Success) {
            $row = [pscustomobject]@{
                model    = $m.name
                position = $p
                step_ms  = [double]$hit.Groups[1].Value
                part0_ms = [double]$hit.Groups[2].Value
                lmhead_ms= [double]$hit.Groups[3].Value
            }
            $rows += $row
            "{0,6}  step {1,7:N1} ms   part0 {2,7:N1}   lm_head {3,5:N1}" -f $p, $row.step_ms, $row.part0_ms, $row.lmhead_ms
        } else {
            Write-Host ("{0,6}  no step line -- output was:" -f $p) -ForegroundColor Yellow
            ($out -split "`n" | Select-Object -Last 6) | ForEach-Object { "        $_" }
        }
    }
}

if ($rows.Count) {
    Write-Host "`n=== part0 growth per position (the number that decides it) ===" -ForegroundColor Cyan
    foreach ($g in $rows | Group-Object model) {
        $s = $g.Group | Sort-Object position
        if ($s.Count -ge 2) {
            $d = ($s[-1].part0_ms - $s[0].part0_ms) / ($s[-1].position - $s[0].position)
            "{0,-12} {1,7:N3} ms per position   ({2:N1} -> {3:N1} ms over {4} positions)" -f `
                $g.Name, $d, $s[0].part0_ms, $s[-1].part0_ms, ($s[-1].position - $s[0].position)
        }
    }
    $rows | Export-Csv -NoTypeInformation open_kernels\model\sweep_positions.csv
    Write-Host "`nwrote open_kernels\model\sweep_positions.csv"
}
