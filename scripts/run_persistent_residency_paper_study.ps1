[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = 'paper-20260724-residency-r1',
    [string]$RunRoot = '',
    [string]$CalibrationRoot = '',
    [int]$TimingFrames = 300,
    [int]$TimingWarmup = 30,
    [int]$TimingRepetitions = 3,
    [int]$QualityFrames = 120,
    [int]$QualityWarmup = 20,
    [int]$RenderWidth = 1600,
    [int]$RenderHeight = 900,
    [int]$ProcessTimeoutSeconds = 600,
    [int]$InterRunDelayMilliseconds = 1000,
    [string]$PythonExe = '',
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') { $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir '..')) }
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($RunRoot -eq '') { $RunRoot = Join-Path $ProjectRoot (Join-Path 'results' $RunLabel) }
$RunRoot = [System.IO.Path]::GetFullPath($RunRoot)
if ($CalibrationRoot -eq '') { $CalibrationRoot = Join-Path $ProjectRoot 'results\paper-20260724-r2' }
$CalibrationRoot = [System.IO.Path]::GetFullPath($CalibrationRoot)
if ($TimingFrames -lt 1 -or $TimingWarmup -lt 0 -or $TimingRepetitions -ne 3 -or $QualityFrames -lt 1 -or $QualityWarmup -lt 0) {
    throw 'Formal residency study requires positive frames, nonnegative warm-up, and exactly three timing repetitions.'
}

$benchmarkScript = Join-Path $scriptDir 'run_benchmark.ps1'
$analysisScript = Join-Path $scriptDir 'analyze_persistent_residency_paper_study.py'
$budgetPath = Join-Path $CalibrationRoot 'selected_budgets.csv'
foreach ($path in @($benchmarkScript, $budgetPath)) { if (-not (Test-Path -LiteralPath $path)) { throw "Missing residency-study input: $path" } }

$sceneMap = [ordered]@{
    'hanging' = 'scenes\test_scene.xml'
    'moving-sphere' = 'scenes\moving_sphere_cloth.xml'
}
$dimensions = @(256, 386)
function Resolve-ReferenceCheckpointDirectory {
    param([string]$Path)
    $root = [System.IO.Path]::GetFullPath($Path)
    foreach ($candidate in @((Join-Path $root 'reference_checkpoints'), $root)) {
        if (Test-Path -LiteralPath (Join-Path $candidate 'reference_state_000000.bin')) { return $candidate }
    }
    throw "Reference checkpoints were not found below: $root"
}
$budgetRows = @(Import-Csv -LiteralPath $budgetPath)
$cases = @()
foreach ($sceneId in $sceneMap.Keys) {
    foreach ($dimension in $dimensions) {
        $matches = @($budgetRows | Where-Object {
            $_.scene_id -eq $sceneId -and [int]$_.cloth_dimension -eq $dimension `
                -and $_.solver_variant -eq 'gpu-gather-fusion-batched-ls-persistent' -and $_.qualified -eq '1'
        })
        if ($matches.Count -ne 1) { throw "Expected one qualified persistent calibration row for $sceneId d$dimension." }
        $referenceDir = Resolve-ReferenceCheckpointDirectory -Path $matches[0].reference_dir
        $cases += [pscustomobject]@{
            scene_id = $sceneId
            scene_path = $sceneMap[$sceneId]
            cloth_dimension = $dimension
            iterations_per_frame = [int]$matches[0].iterations_per_frame
            quality_reference_dir = $referenceDir
            calibration_commit = 'c73d2bb'
        }
    }
}

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$commit = (& git -c "safe.directory=$ProjectRoot" -C $ProjectRoot rev-parse --short HEAD 2>$null | Select-Object -First 1).Trim()
$manifest = [ordered]@{
    protocol_version = 'persistent-residency-paper-v1'
    label = $RunLabel
    git_commit = $commit
    calibration_root = $CalibrationRoot
    calibration_commit = 'c73d2bb'
    solver_variant = 'gpu-gather-fusion-batched-ls-persistent'
    line_search = [ordered]@{ schedule = 'fixed-batched'; k = 4; beta = 0.5 }
    measurement = [ordered]@{ mode = 'rendered-end-to-end'; render_width = $RenderWidth; render_height = $RenderHeight; sync_gpu = $true; disable_vsync = $true; quality_readback_during_timing = $false }
    timing = [ordered]@{ frames = $TimingFrames; warmup = $TimingWarmup; repetitions = $TimingRepetitions }
    quality = [ordered]@{ frames = $QualityFrames; warmup = $QualityWarmup; checkpoint_stride = 1; timing_separate = $true }
    scenes = @($sceneMap.Keys)
    cloth_dimensions = $dimensions
    conditions = @('resident', 'forced-cpu-state-roundtrip')
    state_traffic_scope = 'position-and-velocity SSBO uploads/readbacks only; excludes capture and solver statistics.'
}
[System.IO.File]::WriteAllText((Join-Path $RunRoot 'manifest.json'), ($manifest | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
$cases | Export-Csv -LiteralPath (Join-Path $RunRoot 'planned_cases.csv') -NoTypeInformation
if ($DryRun) { Write-Host "Persistent residency paper dry-run: $RunRoot"; exit 0 }

foreach ($case in $cases) {
    $caseId = "$($case.scene_id)-d$($case.cloth_dimension)"
    foreach ($condition in @('resident', 'forced-cpu-state-roundtrip')) {
        for ($rep = 1; $rep -le $TimingRepetitions; ++$rep) {
            $outputDir = Join-Path $RunRoot (Join-Path 'timing' (Join-Path $caseId (Join-Path $condition ('rep{0:D2}' -f $rep))))
            if ((Test-Path -LiteralPath (Join-Path $outputDir 'frame_presentation.csv')) -and -not $Force) { continue }
            $params = @{
                ProjectRoot = $ProjectRoot; RunLabel = "$RunLabel-$caseId-$condition-rep$('{0:D2}' -f $rep)"; OutputDir = $outputDir
                SolverVariant = 'gpu-gather-fusion-batched-ls-persistent'; IterationsPerFrame = $case.iterations_per_frame
                Frames = $TimingFrames; Warmup = $TimingWarmup; Uncapped = $true; SyncGpu = $true; DisableVsync = $true
                RenderWidth = $RenderWidth; RenderHeight = $RenderHeight; ProcessTimeoutSeconds = $ProcessTimeoutSeconds
                ExtraArgs = @('--scene', $case.scene_path, '--cloth-dimension', $case.cloth_dimension, '--batched-ls-k', '4', '--armijo-beta', '0.5')
            }
            if ($condition -eq 'forced-cpu-state-roundtrip') { $params.ForceCpuStateRoundtrip = $true }
            & $benchmarkScript @params | Out-Host
            if ($LASTEXITCODE -ne 0) { throw "Timing run failed: $outputDir" }
            if ($InterRunDelayMilliseconds -gt 0) { Start-Sleep -Milliseconds $InterRunDelayMilliseconds }
        }
        $qualityDir = Join-Path $RunRoot (Join-Path 'quality' (Join-Path $caseId $condition))
        if (-not (Test-Path -LiteralPath (Join-Path $qualityDir 'quality_metrics.csv')) -or $Force) {
            $qualityParams = @{
                ProjectRoot = $ProjectRoot; RunLabel = "$RunLabel-quality-$caseId-$condition"; OutputDir = $qualityDir
                SolverVariant = 'gpu-gather-fusion-batched-ls-persistent'; IterationsPerFrame = $case.iterations_per_frame
                Frames = $QualityFrames; Warmup = $QualityWarmup; Uncapped = $true; SyncGpu = $true; DisableVsync = $true
                RenderWidth = $RenderWidth; RenderHeight = $RenderHeight; ProcessTimeoutSeconds = $ProcessTimeoutSeconds
                QualityMetrics = $true; QualityReferenceDir = $case.quality_reference_dir; QualityCheckpointStride = 1
                ExtraArgs = @('--scene', $case.scene_path, '--cloth-dimension', $case.cloth_dimension, '--batched-ls-k', '4', '--armijo-beta', '0.5')
            }
            if ($condition -eq 'forced-cpu-state-roundtrip') { $qualityParams.ForceCpuStateRoundtrip = $true }
            & $benchmarkScript @qualityParams | Out-Host
            if ($LASTEXITCODE -ne 0) { throw "Quality run failed: $qualityDir" }
        }
    }
}

if ($PythonExe -eq '') { $PythonExe = if (Test-Path -LiteralPath 'E:\Anaconda\envs\DL\python.exe') { 'E:\Anaconda\envs\DL\python.exe' } else { 'python' } }
if (-not (Test-Path -LiteralPath $analysisScript)) { throw "Missing residency analysis script: $analysisScript" }
& $PythonExe $analysisScript --run-root $RunRoot --report-path (Join-Path $ProjectRoot 'docs\experiments\2026-07-24-persistent-residency-study.md')
if ($LASTEXITCODE -ne 0) { throw "Persistent residency analysis failed with exit code $LASTEXITCODE." }
Write-Host "Persistent residency study complete: $RunRoot"
