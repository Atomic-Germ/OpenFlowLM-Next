# contention_trial.ps1: open_diffusion_cli images while another process uses the NPU.
#
# Starts utilities\dit-chain\switch_probe.py (six xclbin contexts, switching nonstop) as the
# contender, waits for it to reach its timed loops, runs the engine, then reports whether
# the images match a reference PNG and whether the contender survived. One trial per call.
#
#   . C:\dev\mlir-aie\iron_env.ps1
#   utilities\reconfig-probe\contention_trial.ps1 -Size 512 -Runs 4 -Priority 0x100 -Ref C:\dev\switch-work\elf512.png
param(
    [int]$Size = 512,
    [int]$Runs = 4,
    [string]$Priority = "",
    [Parameter(Mandatory = $true)][string]$Ref,
    [string]$Model = "$env:USERPROFILE\.flm\models\FLUX.2-klein-4B-NPU2",
    [string]$Kernels = "C:\dev\klein-kernels-elf",
    [string]$BuildKernels = "C:\dev\klein-kernels",
    [string]$Goldens = "C:\dev\ditref-out",
    [string]$Work = "C:\dev\switch-work"
)
$root = Resolve-Path "$PSScriptRoot\..\.."
$log = Join-Path $Work "contender_$PID.log"
$contender = Start-Process -PassThru -WindowStyle Hidden -FilePath python `
    -ArgumentList "-u", "$root\utilities\dit-chain\switch_probe.py", "--kernels", $BuildKernels, "--size", "512", "--reps", "3000" `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err"
Start-Sleep -Seconds 30
if ($contender.HasExited) { Write-Output "contender exited before the trial"; exit 2 }

if ($Priority) { $env:OFLM_DIFFUSION_QOS_PRIORITY = $Priority } else { Remove-Item Env:OFLM_DIFFUSION_QOS_PRIORITY -ErrorAction SilentlyContinue }
$out = Join-Path $Work "trial_$PID.png"
Remove-Item $out -ErrorAction SilentlyContinue
& "$root\src\open_diffusion\out\open_diffusion_cli.exe" --model $Model --kernels $Kernels --size $Size `
    --ids "$Goldens\goldens_pipe_$Size\ids_0.npy" --noise "$Goldens\goldens_pipe_$Size\noise_0.npy" `
    --runs $Runs --out $out 2>&1 | Where-Object { $_ -match "run |state|open_diffusion_cli:" } | ForEach-Object { "$_" }
$same = (Test-Path $out) -and ((Get-FileHash $out).Hash -eq (Get-FileHash $Ref).Hash)
$alive = -not $contender.HasExited
if ($alive) { Stop-Process -Id $contender.Id -Force; Get-CimInstance Win32_Process -Filter "ParentProcessId=$($contender.Id)" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
Write-Output ("trial size={0} priority={1}: image {2}, contender {3}" -f $Size, ($(if ($Priority) { $Priority } else { "default" })),
    $(if ($same) { "byte-identical" } else { "WRONG OR MISSING" }), $(if ($alive) { "alive" } else { "DIED" }))
