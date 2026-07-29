[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = 'diagnostic-20260729-xpbd-stage-profile-r1',
    [string]$RunRoot = '',
    [int]$Frames = 60,
    [int]$Warmup = 5,
    [int]$ClothDimension = 256,
    [int]$IterationsPerFrame = 32,
    [int]$RenderWidth = 1600,
    [int]$RenderHeight = 900,
    [int]$ProcessTimeoutSeconds = 300,
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
if ($Frames -lt 1 -or $Warmup -lt 0 -or $ClothDimension -lt 2 -or $IterationsPerFrame -lt 1) {
    throw 'XPBD stage-profile arguments are invalid.'
}

$benchmarkScript = Join-Path $scriptDir 'run_benchmark.ps1'
$analysisScript = Join-Path $scriptDir 'analyze_xpbd_stage_profile.py'
foreach ($path in @($benchmarkScript, $analysisScript)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing stage-profile input: $path" }
}

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$commit = (& git -c "safe.directory=$ProjectRoot" -C $ProjectRoot rev-parse HEAD 2>$null | Select-Object -First 1).Trim()
$manifest = [ordered]@{
    protocol_version = 'xpbd-stage-profile-v1'
    git_commit = $commit
    solver_variant = 'gpu-xpbd-jacobi'
    scene = 'scenes\moving_sphere_cloth.xml'
    cloth_dimension = $ClothDimension
    iterations_per_frame = $IterationsPerFrame
    rendered = $true
    warmup = $Warmup
    frames = $Frames
    render_width = $RenderWidth
    render_height = $RenderHeight
    fusion = $false
    cached_pins = $true
    diagnostic_only = $true
    timing_note = 'Stage queries are read after each XPBD substage. They measure GPU elapsed time but intentionally perturb host scheduling and are not frame-time evidence.'
    go_threshold_constraint_share = 0.60
}
[System.IO.File]::WriteAllText((Join-Path $RunRoot 'manifest.json'), ($manifest | ConvertTo-Json -Depth 6), [System.Text.UTF8Encoding]::new($false))
if ($DryRun) { Write-Host "XPBD stage-profile dry run: $RunRoot"; exit 0 }

$outputDir = Join-Path $RunRoot 'profile'
if (-not ((Test-Path -LiteralPath (Join-Path $outputDir 'frame_profile_experiment.csv')) -and -not $Force)) {
    & $benchmarkScript -ProjectRoot $ProjectRoot -RunLabel $RunLabel -OutputDir $outputDir `
        -SolverVariant 'gpu-xpbd-jacobi' -IterationsPerFrame $IterationsPerFrame `
        -Frames $Frames -Warmup $Warmup -Uncapped -SyncGpu -DisableVsync -ProfileGpuQueries `
        -RenderWidth $RenderWidth -RenderHeight $RenderHeight -ProcessTimeoutSeconds $ProcessTimeoutSeconds `
        -ExtraArgs @('--scene', 'scenes\moving_sphere_cloth.xml', '--cloth-dimension', $ClothDimension, '--xpbd-fuse-apply-collision', '0', '--xpbd-cached-pins', '1') | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "XPBD stage profile failed: $outputDir" }
}

if ($PythonExe -eq '') { $PythonExe = if (Test-Path -LiteralPath 'E:\Anaconda\envs\DL\python.exe') { 'E:\Anaconda\envs\DL\python.exe' } else { 'python' } }
& $PythonExe $analysisScript --run-root $RunRoot
if ($LASTEXITCODE -ne 0) { throw "XPBD stage-profile analysis failed with exit code $LASTEXITCODE." }
Write-Host "XPBD stage profile complete: $RunRoot"
