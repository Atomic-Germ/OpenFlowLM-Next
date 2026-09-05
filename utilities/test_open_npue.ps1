# Test the open_npue embedding backend end to end.
#
#   pwsh -File utilities/test_open_npue.ps1
#       SELF-CONTAINED. Needs nothing but this repository and a built flm.
#       Starts a server per model and checks the OpenAI response shape, that
#       vectors are unit length, that a paraphrase is nearer than an unrelated
#       sentence, that gte-multilingual really is multilingual, and that a
#       malformed request does not take the server down.
#
#   pwsh -File utilities/test_open_npue.ps1 -Upstream <path-to-NpuEmbeddings>
#       All of the above, PLUS bit-identity against `npuembed --embed` from the
#       NpuEmbeddings tree.
#
# WHY THE SECOND MODE EXISTS, given the first passes on its own: every check in
# the first mode is one a WRONG answer can satisfy. Unit norm, sane geometry and
# plausible cosines are exactly what a subtly incorrect vector looks like -- and
# one was, during this work. Including the engine's header in a translation unit
# compiled without /arch:AVX2 mixed the scalar and AVX2 reduction orders and
# produced 1-cos 1.04e-04 against the same engine in its own binary. EVERY
# self-contained check here passed on those vectors. Only comparison against an
# independent build caught it.
#
# So the first mode tells you it WORKS; the second tells you it is RIGHT. Run
# the first when building the fork, the second when changing the engine, the
# compile flags, or anything they touch.
#
# Bit-identity only holds when both binaries are compiled at the SAME /arch:
# level -- the host ISA changes the reduction order and therefore the bytes.
# That is measured, not feared: a mismatch there is informative, not a bug.
#
# One model per process on purpose. The engine's geometry is process-wide and a
# ShapeLease refuses a second, so one server serves one embedding model; the
# loop restarts flm for each.

param(
  [string]$Upstream = "",
  [int]$Port = 52625,
  [string]$Xclbins = ""
)

$ErrorActionPreference = 'Stop'
$fork = (Resolve-Path "$PSScriptRoot/..").Path
$exe  = Join-Path $fork 'src/build/flm.exe'
if (-not (Test-Path $exe)) { throw "no flm.exe at $exe -- build it first" }

$EN   = "A man is playing a guitar on stage."
$PARA = "Someone plays a guitar at a concert."
$FAR  = "The stock market closed lower on Tuesday."
$FR   = "Un homme joue de la guitare sur scene."

$cases = @(
  @{ tag = 'all-minilm:l6-v2';      model = 'all-MiniLM-L6-v2';      art = 'artifacts_minilm_tgp'; dims = 384 }
  @{ tag = 'bge-small:en-v1.5';     model = 'bge-small-en-v1.5';     art = 'artifacts_small_tgp';  dims = 384 }
  @{ tag = 'bge-base:en-v1.5';      model = 'bge-base-en-v1.5';      art = 'artifacts_base_tgp';   dims = 768 }
  @{ tag = 'bge-large:en-v1.5';     model = 'bge-large-en-v1.5';     art = 'artifacts_large_tgp';  dims = 1024 }
  @{ tag = 'nomic-embed-text:v1.5'; model = 'nomic-embed-text-v1.5'; art = 'artifacts_nomic_tgp';  dims = 768; prefix = 'search_query' }
  @{ tag = 'gte-multilingual:base'; model = 'gte-multilingual-base'; art = 'artifacts_nomic_tgp';  dims = 768; multilingual = $true }
)

$env:PATH = (Join-Path $fork 'src/lib/xrt') + ";C:\dev\vcpkg\installed\x64-windows\bin;C:\Xilinx\XRT\bin;" + $env:PATH
if (-not $Xclbins) { $Xclbins = Join-Path $fork 'src' }
$env:FLM_XCLBIN_PATH = $Xclbins

$in = Join-Path $env:TEMP 'npue_one.txt'
[System.IO.File]::WriteAllText($in, "$EN`n", (New-Object System.Text.UTF8Encoding $false))

function Post([string]$tag, [string[]]$texts, [int]$timeout = 600) {
    # UTF-8 BYTES, not a string. Invoke-RestMethod encodes a string body as
    # ISO-8859-1 by default, which turns any non-ASCII text into a malformed
    # request -- and then you are testing your own harness, not the server.
    $obj  = @{ model = $tag; input = $texts }
    $body = [System.Text.Encoding]::UTF8.GetBytes(($obj | ConvertTo-Json -Compress))
    Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/embeddings" -Method Post `
        -Body $body -ContentType 'application/json; charset=utf-8' -TimeoutSec $timeout
}

function Dot($a, $b, $n) { $s = 0.0; for ($i = 0; $i -lt $n; $i++) { $s += [double]$a[$i] * $b[$i] }; $s }

if ($Upstream) { Write-Host "oracle: $Upstream" }
else { Write-Host "self-contained mode -- pass -Upstream <NpuEmbeddings> to also check bit-identity" -ForegroundColor DarkYellow }

$fail = 0
foreach ($c in $cases) {
    Write-Host ("`n===== {0}" -f $c.tag) -ForegroundColor Cyan

    # Stop any server FIRST, before the oracle runs. The first version of this
    # script stopped it afterwards, so npuembed ran while the previous model's
    # flm still held an hw_context -- and hung indefinitely, no output and no
    # error, until the server was killed. Two processes each holding a context
    # on this NPU do not queue, they block.
    Get-Process flm -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 3

    $ref = $null
    if ($Upstream) {
        $ref = Join-Path $env:TEMP ("ref_" + $c.model + ".f32")
        Remove-Item $ref -Force -ErrorAction SilentlyContinue
        $a = @($Upstream, '--model', $c.model,
               '--artifacts', (Join-Path $Upstream "runtime/$($c.art)"),
               '--embed', $in, $ref, '--threads', '24', '--pipeline', '2')
        if ($c.prefix) { $a += @('--prefix', $c.prefix) }
        # NO `2>&1`. In PowerShell 5.1 that merge wraps every stderr line from a
        # native executable in an ErrorRecord, and under $ErrorActionPreference
        # = 'Stop' the first becomes a TERMINATING error even though the process
        # exited 0. npuembed prints its resolved prefix to stderr, so this
        # aborted the run on nomic -- the only case with a --prefix -- after
        # four models had already passed.
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & (Join-Path $Upstream 'runtime/build/npuembed.exe') @a > $null 2> $null
        $ErrorActionPreference = $prev
        if (-not (Test-Path $ref)) { Write-Host "  oracle FAILED" -ForegroundColor Red; $fail++; continue }
    }

    $log = Join-Path $env:TEMP ("flm_" + ($c.tag -replace '[:\.]', '_') + ".txt")
    Remove-Item $log -Force -ErrorAction SilentlyContinue
    $p = Start-Process -FilePath $exe -WorkingDirectory (Split-Path $exe) `
         -ArgumentList 'serve', 'llama3.2:1b', '--embed', '1', '--embeddingmodel', $c.tag `
         -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')

    # A first run PACKS the container from the checkpoint, which is minutes for
    # a 335M model. Later runs mmap it.
    $up = $false
    for ($w = 0; $w -lt 90; $w++) {
        Start-Sleep -Seconds 10
        if (Select-String -Path $log -Pattern 'WebServer started' -Quiet -ErrorAction SilentlyContinue) { $up = $true; break }
        if ($p.HasExited) { break }
    }
    if (-not $up) {
        Write-Host "  server did not come up:" -ForegroundColor Red
        Get-Content $log -ErrorAction SilentlyContinue | Select-Object -Last 5 | ForEach-Object { "    $_" }
        $fail++; continue
    }
    (Select-String -Path $log -Pattern 'NPUE\]\s+loaded').Line | ForEach-Object { "  $_" }

    # ---- the OpenAI contract, and geometry a broken pipeline fails
    $texts = @($EN, $PARA, $FAR)
    if ($c.multilingual) { $texts += $FR }
    try { $r = Post $c.tag $texts } catch {
        Write-Host ("  request FAILED: {0}" -f $_.Exception.Message) -ForegroundColor Red; $fail++; continue
    }
    $bad = @()
    if ($r.object -ne 'list')                   { $bad += "object=$($r.object)" }
    if ($r.model -ne $c.tag)                    { $bad += "model=$($r.model)" }
    if ($r.data.Count -ne $texts.Count)         { $bad += "n=$($r.data.Count)" }
    if ($r.data[0].embedding.Count -ne $c.dims) { $bad += "dims=$($r.data[0].embedding.Count)" }
    for ($i = 0; $i -lt $r.data.Count; $i++) {
        if ($r.data[$i].index -ne $i) { $bad += "index[$i]=$($r.data[$i].index)" }
        $n2 = Dot $r.data[$i].embedding $r.data[$i].embedding $c.dims
        if ($n2 -lt 0.99 -or $n2 -gt 1.01) { $bad += ("norm[{0}]={1:F4}" -f $i, $n2) }
    }
    $near = Dot $r.data[0].embedding $r.data[1].embedding $c.dims
    $far  = Dot $r.data[0].embedding $r.data[2].embedding $c.dims
    if ($near -le $far) { $bad += ("paraphrase {0:F4} not nearer than unrelated {1:F4}" -f $near, $far) }
    if ($bad) {
        Write-Host ("  CONTRACT FAILED: {0}" -f ($bad -join ', ')) -ForegroundColor Red; $fail++
    } else {
        Write-Host ("  shape ok; cos(paraphrase)={0:F4} > cos(unrelated)={1:F4}" -f $near, $far)
    }
    if ($c.multilingual) {
        $x = Dot $r.data[0].embedding $r.data[3].embedding $c.dims
        Write-Host ("  cross-lingual: cos(en, fr-same-meaning)={0:F4}" -f $x)
        if ($x -le $far) { Write-Host "  the multilingual model is not behaving multilingually" -ForegroundColor Red; $fail++ }
    }

    # ---- a malformed body must not take the server down
    #
    # It used to. json::parse throws on invalid UTF-8; the catch built an error
    # object from e.what(), which CONTAINS the offending bytes; .dump() threw
    # again; that second exception escaped, so the NPU lock taken before the
    # handler was never released and every later request hung. One byte.
    $lone = [byte[]](0x7b,0x22,0x69,0x6e,0x70,0x75,0x74,0x22,0x3a,0x5b,0x22,0xE5,0x22,0x5d,0x7d)
    try { $null = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/embeddings" -Method Post `
                  -Body $lone -ContentType 'application/json' -TimeoutSec 30 } catch { }
    Start-Sleep -Seconds 2
    try {
        $null = Post $c.tag @($EN) 60
        Write-Host "  survived a malformed request"
    } catch {
        Write-Host "  WEDGED by a malformed request -- the NPU lock was not released" -ForegroundColor Red
        $fail++
    }

    # ---- bit-identity, when an oracle was given
    if ($ref) {
        $bytes = [System.IO.File]::ReadAllBytes($ref)
        $want = New-Object float[] $c.dims
        [Buffer]::BlockCopy($bytes, 0, $want, 0, $c.dims * 4)
        $got = $r.data[0].embedding
        $exact = 0; $maxd = 0.0
        for ($i = 0; $i -lt $c.dims; $i++) {
            if ($want[$i] -eq [float]$got[$i]) { $exact++ }
            $d = [Math]::Abs($want[$i] - [float]$got[$i]); if ($d -gt $maxd) { $maxd = $d }
        }
        if ($exact -eq $c.dims) {
            Write-Host ("  {0}/{1} components exact -- BIT-IDENTICAL to the upstream binary" -f $exact, $c.dims) -ForegroundColor Green
        } else {
            Write-Host ("  {0}/{1} exact, max abs diff {2:E3} -- NOT bit-identical (same /arch: level?)" -f $exact, $c.dims, $maxd) -ForegroundColor Red
            $fail++
        }
    }
}

Get-Process flm -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host ""
if ($fail) { Write-Host "$fail check(s) FAILED across $($cases.Count) models" -ForegroundColor Red; exit 1 }
if ($Upstream) { Write-Host "all $($cases.Count) models pass, and are bit-identical to the upstream binary" -ForegroundColor Green }
else { Write-Host "all $($cases.Count) models pass the self-contained checks" -ForegroundColor Green }
