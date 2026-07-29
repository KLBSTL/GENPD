[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = 'diagnostic-20260729-xpbd-atomic-causality-r1',
    [string]$RunRoot = '',
    [int]$ClothDimension = 128,
    [int]$IterationsPerFrame = 32,
    [int]$Frames = 120,
    [int]$Warmup = 20,
    [int]$CheckpointStride = 10,
    [int]$RenderWidth = 1600,
    [int]$RenderHeight = 900,
    [int]$ProcessTimeoutSeconds = 600,
    [int]$InterRunDelayMilliseconds = 500,
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

if ($RunLabel -notmatch '^diagnostic-' -or $ClothDimension -lt 2 -or $IterationsPerFrame -lt 1 `
    -or $Frames -lt 30 -or $Warmup -lt 0 -or $CheckpointStride -lt 1 `
    -or $RenderWidth -lt 1 -or $RenderHeight -lt 1 -or $ProcessTimeoutSeconds -lt 1 `
    -or $InterRunDelayMilliseconds -lt 0) {
    throw 'XPBD atomic-causality probe arguments are invalid.'
}

$benchmarkScript = Join-Path $scriptDir 'run_benchmark.ps1'
$analysisScript = Join-Path $scriptDir 'analyze_xpbd_atomic_causality_probe.py'
foreach ($path in @($benchmarkScript, $analysisScript)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing atomic-causality input: $path" }
}

function Read-HardwareMetadata {
    $gpu = ''; $driver = ''
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
        $driver = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
    }
    if ([string]::IsNullOrWhiteSpace($gpu) -or [string]::IsNullOrWhiteSpace($driver)) {
        throw 'nvidia-smi did not provide GPU and driver metadata.'
    }
    return [ordered]@{ gpu_name = $gpu; nvidia_driver_version = $driver }
}

# A/B/B/A keeps atomic checks on both sides of the vertex-owned control.
$runs = @(
    [ordered]@{ id = 'atomic-a'; solver_variant = 'gpu-xpbd-jacobi'; reduction = 'float-atomic-scatter' },
    [ordered]@{ id = 'gather-a'; solver_variant = 'gpu-xpbd-vertex-gather'; reduction = 'fixed-order-vertex-gather' },
    [ordered]@{ id = 'gather-b'; solver_variant = 'gpu-xpbd-vertex-gather'; reduction = 'fixed-order-vertex-gather' },
    [ordered]@{ id = 'atomic-b'; solver_variant = 'gpu-xpbd-jacobi'; reduction = 'float-atomic-scatter' }
)

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$commit = (& git -c "safe.directory=$ProjectRoot" -C $ProjectRoot rev-parse HEAD 2>$null | Select-Object -First 1).Trim()
$manifest = [ordered]@{
    protocol_version = 'xpbd-atomic-causality-probe-v1'
    label = $RunLabel
    git_commit = $commit
    hardware = Read-HardwareMetadata
    scene_id = 'moving-sphere'
    scene_path = 'scenes\moving_sphere_cloth.xml'
    cloth_dimension = $ClothDimension
    iterations_per_frame = $IterationsPerFrame
    runs = $runs
    measurement = [ordered]@{
        mode = 'rendered-trajectory-diagnostic'
        render_width = $RenderWidth
        render_height = $RenderHeight
        sync_gpu = $true
        disable_vsync = $true
        timing_claim = 'none'
    }
    trajectory = [ordered]@{ frames = $Frames; warmup = $Warmup; checkpoint_stride = $CheckpointStride }
    causality = [ordered]@{
        atomic_repeat = 'atomic-a:atomic-b'
        gather_repeat = 'gather-a:gather-b'
        gather_hash_requirement = 'all matched checkpoint files must have equal SHA-256 digests'
        atomic_signal_velocity_rel_l2_p95 = 1.0e-3
        gather_repeat_velocity_rel_l2_p95_max = 1.0e-7
        physical_mean_stretch_relative_difference_max = 2.0e-2
        physical_max_stretch_relative_difference_max = 2.0e-2
        physical_penetration_absolute_difference_max = 1.0e-5
        conclusion_rule = 'Strong support requires a nonidentical atomic repeat above the signal threshold, an exact gather repeat below the gather threshold, and comparable physical metrics.'
    }
    acceptance = [ordered]@{ require_finite = $true; require_rendered = $true; min_matched_measured_checkpoints = 3 }
}
[System.IO.File]::WriteAllText((Join-Path $RunRoot 'manifest.json'), ($manifest | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))

$planned = foreach ($run in $runs) {
    [pscustomobject]@{
        run_order = [array]::IndexOf($runs, $run) + 1
        id = $run.id
        solver_variant = $run.solver_variant
        reduction = $run.reduction
        result_dir = (Join-Path 'runs' $run.id)
    }
}
$planned | Export-Csv -LiteralPath (Join-Path $RunRoot 'planned_runs.csv') -NoTypeInformation
if ($DryRun) { Write-Host "XPBD atomic-causality probe dry run: $RunRoot"; exit 0 }

foreach ($run in $runs) {
    $outputDir = Join-Path $RunRoot (Join-Path 'runs' $run.id)
    if ((Test-Path -LiteralPath (Join-Path $outputDir 'frame_profile.csv')) -and -not $Force) {
        Write-Host "Keeping existing causality result: $outputDir"
        continue
    }
    $params = @{
        ProjectRoot = $ProjectRoot
        RunLabel = "$RunLabel-$($run.id)"
        OutputDir = $outputDir
        SolverVariant = $run.solver_variant
        IterationsPerFrame = $IterationsPerFrame
        Frames = $Frames
        Warmup = $Warmup
        ReferenceExportDir = (Join-Path $outputDir 'reference_checkpoints')
        QualityCheckpointStride = $CheckpointStride
        QualityMetrics = $true
        Uncapped = $true
        SyncGpu = $true
        DisableVsync = $true
        RenderWidth = $RenderWidth
        RenderHeight = $RenderHeight
        ProcessTimeoutSeconds = $ProcessTimeoutSeconds
        ExtraArgs = @('--scene', 'scenes\moving_sphere_cloth.xml', '--cloth-dimension', $ClothDimension, '--xpbd-fuse-apply-collision', '0', '--xpbd-cached-pins', '1')
    }
    Write-Host "XPBD atomic-causality probe: $($run.id)"
    & $benchmarkScript @params | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "XPBD atomic-causality run failed: $outputDir" }
    if ($InterRunDelayMilliseconds -gt 0) { Start-Sleep -Milliseconds $InterRunDelayMilliseconds }
}

if ($PythonExe -eq '') { $PythonExe = if (Test-Path -LiteralPath 'E:\Anaconda\envs\DL\python.exe') { 'E:\Anaconda\envs\DL\python.exe' } else { 'python' } }
& $PythonExe $analysisScript --run-root $RunRoot
if ($LASTEXITCODE -ne 0) { throw "XPBD atomic-causality analysis failed with exit code $LASTEXITCODE." }
Write-Host "XPBD atomic-causality probe complete: $RunRoot"
