[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = 'diagnostic-20260729-xpbd-velocity-repro-r1',
    [string]$RunRoot = '',
    [int]$ClothDimension = 128,
    [int]$IterationsPerFrame = 32,
    [int]$Frames = 120,
    [int]$Warmup = 20,
    [int]$CheckpointStride = 10,
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

if ($RunLabel -notmatch '^diagnostic-' -or $ClothDimension -lt 2 -or $IterationsPerFrame -lt 1 `
    -or $Frames -lt 30 -or $Warmup -lt 0 -or $CheckpointStride -lt 1 `
    -or $RenderWidth -lt 1 -or $RenderHeight -lt 1 -or $ProcessTimeoutSeconds -lt 1) {
    throw 'XPBD velocity reproducibility probe arguments are invalid.'
}

$benchmarkScript = Join-Path $scriptDir 'run_benchmark.ps1'
$analysisScript = Join-Path $scriptDir 'analyze_xpbd_velocity_reproducibility_probe.py'
foreach ($path in @($benchmarkScript, $analysisScript)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing probe input: $path" }
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

$scenes = @(
    [ordered]@{
        id = 'hanging'
        scene_path = 'scenes\test_scene.xml'
        control_role = 'No external collision primitive; tests whether the discrepancy exists without moving-contact branching.'
    },
    [ordered]@{
        id = 'moving-sphere'
        scene_path = 'scenes\moving_sphere_cloth.xml'
        control_role = 'Moving sphere and plane; tests whether contact dynamics amplify a residency-condition discrepancy.'
    }
)
$conditions = @(
    [ordered]@{ id = 'resident-a'; force_cpu_state_roundtrip = $false; repetition = 'a' },
    [ordered]@{ id = 'resident-b'; force_cpu_state_roundtrip = $false; repetition = 'b' },
    [ordered]@{ id = 'forced-a'; force_cpu_state_roundtrip = $true; repetition = 'a' },
    [ordered]@{ id = 'forced-b'; force_cpu_state_roundtrip = $true; repetition = 'b' }
)

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$commit = (& git -c "safe.directory=$ProjectRoot" -C $ProjectRoot rev-parse HEAD 2>$null | Select-Object -First 1).Trim()
$manifest = [ordered]@{
    protocol_version = 'xpbd-velocity-reproducibility-probe-v1'
    label = $RunLabel
    git_commit = $commit
    hardware = Read-HardwareMetadata
    solver_variant = 'gpu-xpbd-jacobi'
    cloth_dimension = $ClothDimension
    iterations_per_frame = $IterationsPerFrame
    scenes = $scenes
    conditions = $conditions
    measurement = [ordered]@{
        mode = 'rendered-trajectory-diagnostic'
        render_width = $RenderWidth
        render_height = $RenderHeight
        sync_gpu = $true
        disable_vsync = $true
        quality_readback_during_run = $true
        timing_claim = 'none'
    }
    trajectory = [ordered]@{ frames = $Frames; warmup = $Warmup; checkpoint_stride = $CheckpointStride }
    comparison_definition = [ordered]@{
        repeat_pairs = @('resident-a:resident-b', 'forced-a:forced-b')
        cross_condition_pairs = @('resident-a:forced-a', 'resident-a:forced-b', 'resident-b:forced-a', 'resident-b:forced-b')
        interpretation = 'Compare repeated runs within one residency condition against all cross-condition pairs. Compare hanging and moving-sphere results to assess collision-associated amplification.'
    }
    acceptance = [ordered]@{ require_finite = $true; require_rendered = $true; min_matched_measured_checkpoints = 3 }
}
[System.IO.File]::WriteAllText((Join-Path $RunRoot 'manifest.json'), ($manifest | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))

$planned = foreach ($scene in $scenes) {
    foreach ($condition in $conditions) {
        [pscustomobject]@{
            scene_id = $scene.id
            scene_path = $scene.scene_path
            condition = $condition.id
            force_cpu_state_roundtrip = $condition.force_cpu_state_roundtrip
            result_dir = (Join-Path (Join-Path 'runs' $scene.id) $condition.id)
        }
    }
}
$planned | Export-Csv -LiteralPath (Join-Path $RunRoot 'planned_runs.csv') -NoTypeInformation
if ($DryRun) { Write-Host "XPBD velocity reproducibility probe dry run: $RunRoot"; exit 0 }

foreach ($scene in $scenes) {
    foreach ($condition in $conditions) {
        $outputDir = Join-Path $RunRoot (Join-Path (Join-Path 'runs' $scene.id) $condition.id)
        if ((Test-Path -LiteralPath (Join-Path $outputDir 'frame_profile.csv')) -and -not $Force) {
            Write-Host "Keeping existing probe result: $outputDir"
            continue
        }
        $params = @{
            ProjectRoot = $ProjectRoot
            RunLabel = "$RunLabel-$($scene.id)-$($condition.id)"
            OutputDir = $outputDir
            SolverVariant = 'gpu-xpbd-jacobi'
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
            ExtraArgs = @('--scene', $scene.scene_path, '--cloth-dimension', $ClothDimension, '--xpbd-fuse-apply-collision', '0', '--xpbd-cached-pins', '1')
        }
        if ($condition.force_cpu_state_roundtrip) { $params.ForceCpuStateRoundtrip = $true }
        Write-Host "XPBD velocity probe: $($scene.id), $($condition.id)"
        & $benchmarkScript @params | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "XPBD velocity probe run failed: $outputDir" }
    }
}

if ($PythonExe -eq '') { $PythonExe = if (Test-Path -LiteralPath 'E:\Anaconda\envs\DL\python.exe') { 'E:\Anaconda\envs\DL\python.exe' } else { 'python' } }
& $PythonExe $analysisScript --run-root $RunRoot
if ($LASTEXITCODE -ne 0) { throw "XPBD velocity probe analysis failed with exit code $LASTEXITCODE." }
Write-Host "XPBD velocity reproducibility probe complete: $RunRoot"
