[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = 'diagnostic-20260729-xpbd-residency-scaling-r1',
    [string]$RunRoot = '',
    [int[]]$ClothDimensions = @(128, 256, 386),
    [int]$XpbdIterationsPerFrame = 32,
    [int]$CpuIterationsPerFrame = 32,
    [int]$Frames = 300,
    [int]$Warmup = 30,
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

if ($RunLabel -match '^paper-') { throw 'This single-repetition diagnostic must not use a paper-labelled run label.' }
if ($ClothDimensions.Count -eq 0 -or ($ClothDimensions | Where-Object { $_ -lt 2 }).Count -ne 0 `
    -or $XpbdIterationsPerFrame -lt 1 -or $CpuIterationsPerFrame -lt 1 -or $Frames -lt 1 -or $Warmup -lt 0) {
    throw 'Scaling-study arguments are invalid.'
}

$benchmarkScript = Join-Path $scriptDir 'run_benchmark.ps1'
$analysisScript = Join-Path $scriptDir 'analyze_xpbd_residency_scaling_study.py'
foreach ($path in @($benchmarkScript, $analysisScript)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing scaling-study input: $path" }
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

$conditions = @(
    [ordered]@{ id = 'cpu-ncg'; solver_variant = 'cpu-ncg'; force_cpu_state_roundtrip = $false; iterations_per_frame = $CpuIterationsPerFrame; comparison_role = 'CPU NCG timing context only; it is not an equal-quality XPBD comparison.' },
    [ordered]@{ id = 'xpbd-resident'; solver_variant = 'gpu-xpbd-jacobi'; force_cpu_state_roundtrip = $false; iterations_per_frame = $XpbdIterationsPerFrame; comparison_role = 'Causal XPBD residency reference.' },
    [ordered]@{ id = 'xpbd-forced-roundtrip'; solver_variant = 'gpu-xpbd-jacobi'; force_cpu_state_roundtrip = $true; iterations_per_frame = $XpbdIterationsPerFrame; comparison_role = 'Same XPBD computation with finalized position/velocity CPU roundtrip between frames.' },
    [ordered]@{ id = 'xpbd-signed-incidence-gather'; solver_variant = 'gpu-xpbd-vertex-gather'; force_cpu_state_roundtrip = $false; iterations_per_frame = $XpbdIterationsPerFrame; comparison_role = 'Current signed-incidence vertex-gather XPBD implementation.' }
)

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$commit = (& git -c "safe.directory=$ProjectRoot" -C $ProjectRoot rev-parse HEAD 2>$null | Select-Object -First 1).Trim()
$manifest = [ordered]@{
    protocol_version = 'xpbd-residency-scaling-diagnostic-v1'
    label = $RunLabel
    git_commit = $commit
    hardware = Read-HardwareMetadata
    scene_id = 'moving-sphere'
    scene_path = 'scenes\moving_sphere_cloth.xml'
    cloth_dimensions = @($ClothDimensions | Sort-Object -Unique)
    conditions = $conditions
    measurement = [ordered]@{ mode = 'rendered-end-to-end'; render_width = $RenderWidth; render_height = $RenderHeight; sync_gpu = $true; disable_vsync = $true; quality_readback_during_timing = $false; repetitions = 1 }
    timing = [ordered]@{ frames = $Frames; warmup = $Warmup }
    xpbd_controls = [ordered]@{ fuse_apply_collision = 0; cached_pins = 1 }
    comparison_scope = [ordered]@{
        residency = 'GPU-resident and forced-roundtrip use the same gpu-xpbd-jacobi shader sequence and iteration count.'
        gather = 'Signed-incidence vertex gather is compared with atomic XPBD at the same iteration count; this timing study validates finite execution, while trajectory equivalence is separately established only for the 256x256 short diagnostic.'
        cpu = 'CPU NCG is retained as a fixed-budget timing context. Its iteration method and quality target differ from XPBD, so it is not used for an equal-quality speedup claim.'
    }
    acceptance = [ordered]@{ require_finite = $true; require_rendered = $true; expected_measured_frames = $Frames }
}
[System.IO.File]::WriteAllText((Join-Path $RunRoot 'manifest.json'), ($manifest | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))

$planned = foreach ($dimension in $manifest.cloth_dimensions) {
    foreach ($condition in $conditions) {
        [pscustomobject]@{ cloth_dimension = $dimension; condition = $condition.id; solver_variant = $condition.solver_variant; force_cpu_state_roundtrip = $condition.force_cpu_state_roundtrip; iterations_per_frame = $condition.iterations_per_frame; result_dir = (Join-Path (Join-Path 'timing' $dimension) $condition.id) }
    }
}
$planned | Export-Csv -LiteralPath (Join-Path $RunRoot 'planned_runs.csv') -NoTypeInformation
if ($DryRun) { Write-Host "XPBD scaling-study dry run: $RunRoot"; exit 0 }

foreach ($dimension in $manifest.cloth_dimensions) {
    foreach ($condition in $conditions) {
        $outputDir = Join-Path $RunRoot (Join-Path (Join-Path 'timing' $dimension) $condition.id)
        if ((Test-Path -LiteralPath (Join-Path $outputDir 'frame_profile.csv')) -and -not $Force) {
            Write-Host "Keeping existing result: $outputDir"
            continue
        }
        $params = @{
            ProjectRoot = $ProjectRoot; RunLabel = "$RunLabel-$($condition.id)-$dimension"; OutputDir = $outputDir
            SolverVariant = $condition.solver_variant; IterationsPerFrame = $condition.iterations_per_frame
            Frames = $Frames; Warmup = $Warmup; Uncapped = $true; SyncGpu = $true; DisableVsync = $true
            RenderWidth = $RenderWidth; RenderHeight = $RenderHeight; ProcessTimeoutSeconds = $ProcessTimeoutSeconds
            ExtraArgs = @('--scene', 'scenes\moving_sphere_cloth.xml', '--cloth-dimension', $dimension, '--xpbd-fuse-apply-collision', '0', '--xpbd-cached-pins', '1')
        }
        if ($condition.force_cpu_state_roundtrip) { $params.ForceCpuStateRoundtrip = $true }
        Write-Host "Scaling-study: $($condition.id), ${dimension}x${dimension}"
        & $benchmarkScript @params | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Scaling-study run failed: $outputDir" }
    }
}

if ($PythonExe -eq '') { $PythonExe = if (Test-Path -LiteralPath 'E:\Anaconda\envs\DL\python.exe') { 'E:\Anaconda\envs\DL\python.exe' } else { 'python' } }
& $PythonExe $analysisScript --run-root $RunRoot
if ($LASTEXITCODE -ne 0) { throw "Scaling-study analysis failed with exit code $LASTEXITCODE." }
Write-Host "XPBD scaling study complete: $RunRoot"
