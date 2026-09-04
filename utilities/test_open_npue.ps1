# Test every open_npue model end to end against the upstream binary.
#
# One model per process on purpose: the engine's geometry is process-wide and
# a ShapeLease refuses a second, so a server serves one embedding model. The
# loop therefore restarts flm for each.
#
# The comparison is BIT-IDENTITY against `npuembed --embed` from the
# NpuEmbeddings tree -- and it only holds when both are compiled at the same
# /arch: level, because the host ISA changes the reduction order and therefore
# the bytes. Both are /arch:AVX2 here.
#
#   pwsh -File utilities/test_open_npue.ps1 -Upstream C:\path\to\NpuEmbeddings

param(
  [string]$Upstream = "C:\Users\vegar\Documents\GitHub\NpuEmbeddings",
  [int]$Port = 52625
)

$ErrorActionPreference = 'Stop'
$fork = (Resolve-Path "$PSScriptRoot/..").Path
$exe  = Join-Path $fork 'src/build/flm.exe'
$text = "A man is playing a guitar on stage."

# tag -> the upstream container name and design set to compare against
$cases = @(
  @{ tag = 'all-minilm:l6-v2';      model = 'all-MiniLM-L6-v2';      art = 'artifacts_minilm_tgp'; dims = 384 }
  @{ tag = 'bge-small:en-v1.5';     model = 'bge-small-en-v1.5';     art = 'artifacts_small_tgp';  dims = 384 }
  @{ tag = 'bge-base:en-v1.5';      model = 'bge-base-en-v1.5';      art = 'artifacts_base_tgp';   dims = 768 }
  @{ tag = 'bge-large:en-v1.5';     model = 'bge-large-en-v1.5';     art = 'artifacts_large_tgp';  dims = 1024 }
  @{ tag = 'nomic-embed-text:v1.5'; model = 'nomic-embed-text-v1.5'; art = 'artifacts_nomic_tgp';  dims = 768; prefix = 'search_query' }
  @{ tag = 'gte-multilingual:base'; model = 'gte-multilingual-base'; art = 'artifacts_nomic_tgp';  dims = 768 }
)

$env:PATH = (Join-Path $fork 'src/lib/xrt') + ";C:\dev\vcpkg\installed\x64-windows\bin;C:\Xilinx\XRT\bin;" + $env:PATH
$env:FLM_XCLBIN_PATH = Join-Path $fork 'src'

$in = Join-Path $env:TEMP 'npue_one.txt'
[System.IO.File]::WriteAllText($in, "$text`n", (New-Object System.Text.UTF8Encoding $false))

$fail = 0
foreach ($c in $cases) {
    Write-Host ("`n===== {0}" -f $c.tag) -ForegroundColor Cyan

    # KILL ANY RUNNING SERVER FIRST, before the reference and not after it.
    #
    # The first version killed it afterwards, so `npuembed --embed` ran while
    # the PREVIOUS model's server still held an hw_context on the array. That
    # wedged: the reference produced no file and no process, and the harness
    # sat for half an hour. Two co-resident contexts on one NPU is a known
    # unmeasured question upstream (T63 item 4); this harness reproduced it by
    # accident, which is worth knowing but is not what it is here to test.
    Get-Process flm -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 3

    # The reference: the same engine in its own binary.
    $ref = Join-Path $env:TEMP ("ref_" + $c.model + ".f32")
    $a = @($Upstream, '--model', $c.model, '--artifacts', (Join-Path $Upstream "runtime/$($c.art)"),
           '--embed', $in, $ref, '--threads', '24', '--pipeline', '2')
    if ($c.prefix) { $a += @('--prefix', $c.prefix) }
    & (Join-Path $Upstream 'runtime/build/npuembed.exe') @a 2>&1 | Out-Null
    if (-not (Test-Path $ref)) { Write-Host "  reference FAILED"; $fail++; continue }

    $log = Join-Path $env:TEMP ("flm_" + ($c.tag -replace '[:\.]', '_') + ".txt")
    $p = Start-Process -FilePath $exe -WorkingDirectory (Split-Path $exe) `
         -ArgumentList 'serve','llama3.2:1b','--embed','1','--embeddingmodel',$c.tag `
         -NoNewWindow -PassThru -RedirectStandardOutput $log `
         -RedirectStandardError ($log + '.err')

    # A first run packs the container, which is minutes for a 335M model.
    $loaded = $false
    for ($w = 0; $w -lt 60; $w++) {
        Start-Sleep -Seconds 10
        if (Select-String -Path $log -Pattern 'WebServer started' -Quiet -ErrorAction SilentlyContinue) { $loaded = $true; break }
        if ($p.HasExited) { break }
    }
    if (-not $loaded) {
        Write-Host "  server did not come up; last lines:"
        Get-Content $log -ErrorAction SilentlyContinue | Select-Object -Last 4 | ForEach-Object { "    $_" }
        $fail++; continue
    }
    (Select-String -Path $log -Pattern 'NPUE\]\s+loaded').Line | ForEach-Object { "  $_" }

    $body = @{ model = $c.tag; input = @($text) } | ConvertTo-Json -Compress
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/embeddings" -Method Post `
             -Body $body -ContentType 'application/json' -TimeoutSec 600
    } catch {
        Write-Host ("  request FAILED: {0}" -f $_.Exception.Message); $fail++; continue
    }
    $got = $r.data[0].embedding
    if ($got.Count -ne $c.dims) {
        Write-Host ("  WRONG DIMS: {0}, expected {1}" -f $got.Count, $c.dims); $fail++; continue
    }

    $bytes = [System.IO.File]::ReadAllBytes($ref)
    $want = New-Object float[] $c.dims
    [Buffer]::BlockCopy($bytes, 0, $want, 0, $c.dims * 4)
    $exact = 0; $maxd = 0.0
    for ($i = 0; $i -lt $c.dims; $i++) {
        if ($want[$i] -eq [float]$got[$i]) { $exact++ }
        $d = [Math]::Abs($want[$i] - [float]$got[$i])
        if ($d -gt $maxd) { $maxd = $d }
    }
    if ($exact -eq $c.dims) {
        Write-Host ("  {0}/{1} components exact -- BIT-IDENTICAL" -f $exact, $c.dims) -ForegroundColor Green
    } else {
        Write-Host ("  {0}/{1} exact, max abs diff {2:E3} -- NOT bit-identical" -f $exact, $c.dims, $maxd) -ForegroundColor Red
        $fail++
    }
}

Get-Process flm -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host ""
if ($fail) { Write-Host "$fail of $($cases.Count) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "all $($cases.Count) models bit-identical to the upstream binary" -ForegroundColor Green
