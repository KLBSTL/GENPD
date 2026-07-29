[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = 'paper-20260729-xpbd-residency-scaling-r2',
    [string]$RunRoot = '',
    [int[]]$ClothDimensions = @(128, 256, 386),
    [int]$IterationsPerFrame = 32,
    [int]$TimingFrames = 300,
    [int]$TimingWarmup = 30,
    [int]$TimingRepetitions = 3,
    [int]$TrajectoryFrames = 120,
    [int]$TrajectoryWarmup = 20,
    [int]$TrajectoryCheckpointStride = 10,
    [int]$RenderWidth = 1600,
    [int]$RenderHeight = 900,
    [int]$ProcessTimeoutSeconds = 1800,
    [string]$PythonExe = '',
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') { $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir '..')) }
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($RunRoot -eq '') { $RunRoot = Join-Path $ProjectRoot (Join-Path 'results' $RunLabel) }
$RunRoot = [System.IO.Path]::GetFullPath($RunRoot)

if ($RunLabel -notmatch '^paper-' -or $ClothDimensions.Count -eq 0 -or ($ClothDimensions | Where-Object { $_ -lt 2 }).Count -ne 0 `
    -or $IterationsPerFrame -lt 1 -or $TimingFrames -ne 300 -or $TimingWarmup -ne 30 -or $TimingRepetitions -ne 3 `
    -or $TrajectoryFrames -lt 1 -or $TrajectoryWarmup -lt 0 -or $TrajectoryCheckpointStride -lt 1) {
    throw 'Formal XPBD residency scaling requires a paper label, 30 warm-up + 300 measured frames, three repetitions, and valid trajectory settings.'
}

$singleStudy = Join-Path $scriptDir 'run_xpbd_residency_study.ps1'
$analysisScript = Join-Path $scriptDir 'analyze_xpbd_residency_scaling_paper_study.py'
foreach ($path in @($singleStudy, $analysisScript)) { if (-not (Test-Path -LiteralPath $path)) { throw "Missing XPBD scaling-study input: $path" } }

function Read-HardwareMetadata {
    $gpu = ''; $driver = ''
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
        $driver = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
    }
    if ([string]::IsNullOrWhiteSpace($gpu) -or [string]::IsNullOrWhiteSpace($driver)) { throw 'nvidia-smi did not provide GPU and driver metadata.' }
    return [ordered]@{ gpu_name = $gpu; nvidia_driver_version = $driver }
}

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$commit = (& git -c "safe.directory=$ProjectRoot" -C $ProjectRoot rev-parse HEAD 2>$null | Select-Object -First 1).Trim()
$manifest = [ordered]@{
    protocol_version = 'xpbd-residency-scaling-v1'
    label = $RunLabel
    git_commit = $commit
    hardware = Read-HardwareMetadata
    solver_variant = 'gpu-xpbd-jacobi'
    scene_id = 'moving-sphere'
    scene_path = 'scenes\moving_sphere_cloth.xml'
    cloth_dimensions = @($ClothDimensions | Sort-Object -Unique)
    iterations_per_frame = $IterationsPerFrame
    measurement = [ordered]@{ mode = 'rendered-end-to-end'; render_width = $RenderWidth; render_height = $RenderHeight; sync_gpu = $true; disable_vsync = $true; quality_readback_during_timing = $false }
    timing = [ordered]@{ frames = $TimingFrames; warmup = $TimingWarmup; repetitions = $TimingRepetitions }
    trajectory = [ordered]@{ frames = $TrajectoryFrames; warmup = $TrajectoryWarmup; checkpoint_stride = $TrajectoryCheckpointStride; rendered = $true; quality_metrics = $true; timing_separate = $true }
    conditions = @('resident', 'forced-cpu-state-roundtrip')
    acceptance = [ordered]@{ require_finite = $true; require_rendered = $true; require_equal_xpbd_dispatches = $true; position_checkpoint_rel_l2_p95 = 1.0e-3; velocity_checkpoint_rel_l2_p95 = 1.0e-3; require_trajectory_quality_metrics = $true }
    comparison_scope = 'Each resolution uses the same XPBD shader dispatch sequence in both conditions. The forced condition alone reads finalized position/velocity state to CPU and uploads it for the next frame.'
}
[System.IO.File]::WriteAllText((Join-Path $RunRoot 'manifest.json'), ($manifest | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))

$planned = foreach ($dimension in $manifest.cloth_dimensions) {
    foreach ($condition in $manifest.conditions) {
        for ($repetition = 1; $repetition -le $TimingRepetitions; ++$repetition) {
            [pscustomobject]@{ cloth_dimension = $dimension; phase = 'timing'; condition = $condition; repetition = $repetition }
        }
        [pscustomobject]@{ cloth_dimension = $dimension; phase = 'trajectory'; condition = $condition; repetition = 0 }
    }
}
$planned | Export-Csv -LiteralPath (Join-Path $RunRoot 'planned_runs.csv') -NoTypeInformation
if ($DryRun) { Write-Host "Formal XPBD residency scaling dry run: $RunRoot"; exit 0 }

foreach ($dimension in $manifest.cloth_dimensions) {
    $dimensionRoot = Join-Path $RunRoot (Join-Path 'dimensions' ("d{0}" -f $dimension))
    $parameters = @{
        ProjectRoot = $ProjectRoot; RunLabel = "$RunLabel-d$dimension"; RunRoot = $dimensionRoot
        SceneId = 'moving-sphere'; Scene = 'scenes\moving_sphere_cloth.xml'; ClothDimension = $dimension; IterationsPerFrame = $IterationsPerFrame
        TimingFrames = $TimingFrames; TimingWarmup = $TimingWarmup; TimingRepetitions = $TimingRepetitions
        TrajectoryFrames = $TrajectoryFrames; TrajectoryWarmup = $TrajectoryWarmup; TrajectoryCheckpointStride = $TrajectoryCheckpointStride
        RenderWidth = $RenderWidth; RenderHeight = $RenderHeight; ProcessTimeoutSeconds = $ProcessTimeoutSeconds; PythonExe = $PythonExe
    }
    if ($Force) { $parameters.Force = $true }
    Write-Host "Formal XPBD residency scaling: ${dimension}x${dimension}"
    & $singleStudy @parameters | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "XPBD residency study failed at ${dimension}x${dimension}." }
}

if ($PythonExe -eq '') { $PythonExe = if (Test-Path -LiteralPath 'E:\Anaconda\envs\DL\python.exe') { 'E:\Anaconda\envs\DL\python.exe' } else { 'python' } }
& $PythonExe $analysisScript --run-root $RunRoot
if ($LASTEXITCODE -ne 0) { throw "XPBD residency scaling analysis failed with exit code $LASTEXITCODE." }
Write-Host "Formal XPBD residency scaling complete: $RunRoot"
