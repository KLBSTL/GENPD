[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = 'paper-20260729-xpbd-residency-r1',
    [string]$RunRoot = '',
    [string]$SceneId = 'moving-sphere',
    [string]$Scene = 'scenes\moving_sphere_cloth.xml',
    [int]$ClothDimension = 256,
    [int]$IterationsPerFrame = 32,
    [int]$TimingFrames = 300,
    [int]$TimingWarmup = 30,
    [int]$TimingRepetitions = 3,
    [int]$TrajectoryFrames = 120,
    [int]$TrajectoryWarmup = 20,
    [int]$TrajectoryCheckpointStride = 10,
    [int]$RenderWidth = 1600,
    [int]$RenderHeight = 900,
    [int]$ProcessTimeoutSeconds = 600,
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

if ($SceneId -notin @('hanging', 'moving-sphere') -or $ClothDimension -lt 2 -or $IterationsPerFrame -lt 1 `
    -or $TimingFrames -lt 1 -or $TimingWarmup -lt 0 -or $TimingRepetitions -lt 1 `
    -or $TrajectoryFrames -lt 1 -or $TrajectoryWarmup -lt 0 -or $TrajectoryCheckpointStride -lt 1) {
    throw 'XPBD residency study arguments are invalid.'
}
if ($RunLabel -match '^paper-' -and ($TimingFrames -ne 300 -or $TimingWarmup -ne 30 -or $TimingRepetitions -ne 3)) {
    throw 'Paper-labelled XPBD residency runs require 30 warm-up frames, 300 measured frames, and three timing repetitions.'
}

$benchmarkScript = Join-Path $scriptDir 'run_benchmark.ps1'
$analysisScript = Join-Path $scriptDir 'analyze_xpbd_residency.py'
foreach ($path in @($benchmarkScript, $analysisScript)) { if (-not (Test-Path -LiteralPath $path)) { throw "Missing XPBD residency input: $path" } }

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
    protocol_version = 'xpbd-residency-v1'
    label = $RunLabel
    git_commit = $commit
    hardware = Read-HardwareMetadata
    solver_variant = 'gpu-xpbd-jacobi'
    scene_id = $SceneId
    scene_path = $Scene
    cloth_dimension = $ClothDimension
    iterations_per_frame = $IterationsPerFrame
    comparison_scope = 'same XPBD shader dispatches; conditions differ only by the finalized position/velocity CPU roundtrip and next-frame state upload.'
    measurement = [ordered]@{ mode = 'rendered-end-to-end'; render_width = $RenderWidth; render_height = $RenderHeight; sync_gpu = $true; disable_vsync = $true; quality_readback_during_timing = $false }
    timing = [ordered]@{ frames = $TimingFrames; warmup = $TimingWarmup; repetitions = $TimingRepetitions }
    trajectory = [ordered]@{ frames = $TrajectoryFrames; warmup = $TrajectoryWarmup; checkpoint_stride = $TrajectoryCheckpointStride; rendered = $true; timing_separate = $true }
    conditions = @('resident', 'forced-cpu-state-roundtrip')
    acceptance = [ordered]@{ require_finite = $true; require_rendered = $true; position_checkpoint_rel_l2_p95 = 1.0e-6; velocity_checkpoint_rel_l2_p95 = 1.0e-6; require_equal_xpbd_dispatches = $true }
}
[System.IO.File]::WriteAllText((Join-Path $RunRoot 'manifest.json'), ($manifest | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))

$planned = @()
foreach ($condition in $manifest.conditions) {
    for ($rep = 1; $rep -le $TimingRepetitions; ++$rep) {
        $planned += [pscustomobject]@{ phase = 'timing'; condition = $condition; repetition = $rep }
    }
    $planned += [pscustomobject]@{ phase = 'trajectory'; condition = $condition; repetition = 0 }
}
$planned | Export-Csv -LiteralPath (Join-Path $RunRoot 'planned_runs.csv') -NoTypeInformation
if ($DryRun) { Write-Host "XPBD residency dry run: $RunRoot"; exit 0 }

function Invoke-XPBDRun {
    param([string]$Condition, [string]$OutputDir, [int]$Frames, [int]$Warmup, [int]$Repetition, [string]$ReferenceExportDir = '')
    if ((Test-Path -LiteralPath (Join-Path $OutputDir 'frame_profile.csv')) -and -not $Force) { return }
    $labelSuffix = if ($Repetition -gt 0) { "$Condition-rep$('{0:D2}' -f $Repetition)" } else { "$Condition-trajectory" }
    $benchmarkLabel = if ($Condition -eq 'forced-cpu-state-roundtrip') { "diagnostic-$RunLabel-$labelSuffix" } else { "$RunLabel-$labelSuffix" }
    $params = @{
        ProjectRoot = $ProjectRoot; RunLabel = $benchmarkLabel; OutputDir = $OutputDir
        SolverVariant = 'gpu-xpbd-jacobi'; IterationsPerFrame = $IterationsPerFrame
        Frames = $Frames; Warmup = $Warmup; Uncapped = $true; SyncGpu = $true; DisableVsync = $true
        RenderWidth = $RenderWidth; RenderHeight = $RenderHeight; ProcessTimeoutSeconds = $ProcessTimeoutSeconds
        ExtraArgs = @('--scene', $Scene, '--cloth-dimension', $ClothDimension)
    }
    if ($ReferenceExportDir -ne '') { $params.ReferenceExportDir = $ReferenceExportDir; $params.QualityCheckpointStride = $TrajectoryCheckpointStride }
    if ($Condition -eq 'forced-cpu-state-roundtrip') { $params.ForceCpuStateRoundtrip = $true }
    & $benchmarkScript @params | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "XPBD residency run failed: $OutputDir" }
}

foreach ($condition in $manifest.conditions) {
    for ($rep = 1; $rep -le $TimingRepetitions; ++$rep) {
        Invoke-XPBDRun -Condition $condition -OutputDir (Join-Path $RunRoot (Join-Path 'timing' (Join-Path $condition ('rep{0:D2}' -f $rep)))) `
            -Frames $TimingFrames -Warmup $TimingWarmup -Repetition $rep
    }
    $trajectoryDir = Join-Path $RunRoot (Join-Path 'trajectory' $condition)
    Invoke-XPBDRun -Condition $condition -OutputDir $trajectoryDir -Frames $TrajectoryFrames -Warmup $TrajectoryWarmup -Repetition 0 `
        -ReferenceExportDir (Join-Path $trajectoryDir 'reference_checkpoints')
}

if ($PythonExe -eq '') { $PythonExe = if (Test-Path -LiteralPath 'E:\Anaconda\envs\DL\python.exe') { 'E:\Anaconda\envs\DL\python.exe' } else { 'python' } }
& $PythonExe $analysisScript --run-root $RunRoot
if ($LASTEXITCODE -ne 0) { throw "XPBD residency analysis failed with exit code $LASTEXITCODE." }
Write-Host "XPBD residency study complete: $RunRoot"
