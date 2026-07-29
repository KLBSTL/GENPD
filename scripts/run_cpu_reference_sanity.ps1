[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = 'paper-20260729-cpu-reference-sanity-r1',
    [string]$RunRoot = '',
    [int]$ClothDimension = 256,
    [int[]]$Iterations = @(100, 200, 400),
    [int]$Frames = 30,
    [int]$Warmup = 5,
    [int]$CheckpointStride = 1,
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
if ($ClothDimension -lt 2 -or $Frames -lt 1 -or $Warmup -lt 0 -or $CheckpointStride -lt 1 -or @($Iterations).Count -ne 3 -or (@($Iterations) -join ',') -ne '100,200,400') {
    throw 'CPU reference sanity requires a valid grid and exactly the 100,200,400 iteration ladder.'
}

$referenceScript = Join-Path $scriptDir 'run_reference.ps1'
$analysisScript = Join-Path $scriptDir 'analyze_cpu_reference_sanity.py'
foreach ($path in @($referenceScript, $analysisScript)) { if (-not (Test-Path -LiteralPath $path)) { throw "Missing CPU reference sanity input: $path" } }

function Read-HardwareMetadata {
    $gpu = ''; $driver = ''
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
        $driver = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
    }
    if ([string]::IsNullOrWhiteSpace($gpu) -or [string]::IsNullOrWhiteSpace($driver)) { throw 'nvidia-smi did not provide GPU and driver metadata.' }
    return [ordered]@{ gpu_name = $gpu; nvidia_driver_version = $driver }
}

$sceneMap = [ordered]@{ hanging = 'scenes\test_scene.xml'; 'moving-sphere' = 'scenes\moving_sphere_cloth.xml' }
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$commit = (& git -c "safe.directory=$ProjectRoot" -C $ProjectRoot rev-parse HEAD 2>$null | Select-Object -First 1).Trim()
$manifest = [ordered]@{
    protocol_version = 'cpu-reference-sanity-v1'
    label = $RunLabel
    git_commit = $commit
    hardware = Read-HardwareMetadata
    solver_variant = 'cpu-ncg'
    scenes = $sceneMap
    cloth_dimension = $ClothDimension
    iterations = $Iterations
    measurement = [ordered]@{ mode = 'rendered-checkpoint-export'; render_width = $RenderWidth; render_height = $RenderHeight; sync_gpu = $true; disable_vsync = $true }
    timing = [ordered]@{ frames = $Frames; warmup = $Warmup; checkpoint_stride = $CheckpointStride }
    decision = [ordered]@{ comparison_iteration = 400; primary_iteration = 100; p95_position_rel_l2 = 1.0e-3; velocity_error_reported = $true }
}
[System.IO.File]::WriteAllText((Join-Path $RunRoot 'manifest.json'), ($manifest | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
$planned = foreach ($scene in $sceneMap.Keys) { foreach ($iteration in $Iterations) { [pscustomobject]@{ scene_id = $scene; scene_path = $sceneMap[$scene]; iterations_per_frame = $iteration } } }
$planned | Export-Csv -LiteralPath (Join-Path $RunRoot 'planned_runs.csv') -NoTypeInformation
if ($DryRun) { Write-Host "CPU reference sanity dry run: $RunRoot"; exit 0 }

foreach ($case in $planned) {
    $outputDir = Join-Path $RunRoot (Join-Path 'references' (Join-Path ("{0}-d{1}" -f $case.scene_id, $ClothDimension) ("i{0:D3}" -f $case.iterations_per_frame)))
    $checkpoint = Join-Path $outputDir 'reference_checkpoints\reference_state_000000.bin'
    if ((Test-Path -LiteralPath $checkpoint) -and -not $Force) { continue }
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    $stdoutLog = Join-Path $outputDir 'cpu_reference_stdout.log'
    & $referenceScript -ProjectRoot $ProjectRoot -RunLabel "$RunLabel-$($case.scene_id)-i$($case.iterations_per_frame)" -OutputDir $outputDir `
        -Frames $Frames -Warmup $Warmup -ReferenceIterations $case.iterations_per_frame -CheckpointStride $CheckpointStride `
        -SyncGpu -DisableVsync -RenderWidth $RenderWidth -RenderHeight $RenderHeight -ProcessTimeoutSeconds $ProcessTimeoutSeconds `
        -ExtraArgs @('--scene', $case.scene_path, '--cloth-dimension', $ClothDimension) *> $stdoutLog
    if ($LASTEXITCODE -ne 0) {
        Get-Content -LiteralPath $stdoutLog -Tail 80 | Write-Host
        throw "CPU reference run failed: $outputDir"
    }
}

if ($PythonExe -eq '') { $PythonExe = if (Test-Path -LiteralPath 'E:\Anaconda\envs\DL\python.exe') { 'E:\Anaconda\envs\DL\python.exe' } else { 'python' } }
& $PythonExe $analysisScript --run-root $RunRoot
if ($LASTEXITCODE -ne 0) { throw "CPU reference sanity analysis failed with exit code $LASTEXITCODE." }
Write-Host "CPU reference sanity study complete: $RunRoot"
