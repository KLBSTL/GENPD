[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = 'minimal-optimizer-effects-r1',
    [string]$RunRoot = '',
    [string]$CalibrationRoot = '',
    [int]$TimingFrames = 150,
    [int]$TimingWarmup = 30,
    [int]$TimingRepetitions = 6,
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
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') { $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir '..')) }
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($RunRoot -eq '') { $RunRoot = Join-Path $ProjectRoot (Join-Path 'results' $RunLabel) }
$RunRoot = [System.IO.Path]::GetFullPath($RunRoot)
if ($CalibrationRoot -eq '') { $CalibrationRoot = Join-Path $ProjectRoot 'results\paper-20260724-r2' }
$CalibrationRoot = [System.IO.Path]::GetFullPath($CalibrationRoot)

if ($TimingFrames -lt 1 -or $TimingWarmup -lt 0 -or $TimingRepetitions -ne 6 -or $QualityFrames -lt 1 -or $QualityWarmup -lt 0) {
    throw 'The minimal recheck requires positive frame counts, nonnegative warm-up, and exactly six timing repetitions.'
}
if ($RenderWidth -le 0 -or $RenderHeight -le 0 -or $ProcessTimeoutSeconds -le 0 -or $InterRunDelayMilliseconds -lt 0) {
    throw 'Render dimensions, timeout, and inter-run delay are invalid.'
}

$benchmarkScript = Join-Path $scriptDir 'run_benchmark.ps1'
$analysisScript = Join-Path $scriptDir 'analyze_minimal_optimizer_effects.py'
$budgetPath = Join-Path $CalibrationRoot 'selected_budgets.csv'
foreach ($path in @($benchmarkScript, $budgetPath)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing minimal recheck input: $path" }
}
if (-not $DryRun -and -not (Test-Path -LiteralPath $analysisScript)) {
    throw "Missing minimal recheck analysis script: $analysisScript"
}

function Resolve-ReferenceCheckpointDirectory {
    param([string]$Path)
    $root = [System.IO.Path]::GetFullPath($Path)
    foreach ($candidate in @((Join-Path $root 'reference_checkpoints'), $root)) {
        if (Test-Path -LiteralPath (Join-Path $candidate 'reference_state_000000.bin')) { return $candidate }
    }
    throw "Reference checkpoints were not found below: $root"
}

function New-Condition {
    param(
        [string]$Suite,
        [string]$ConditionId,
        [string]$SolverVariant,
        [int]$IterationsPerFrame,
        [int]$BatchedLsK,
        [ValidateSet('none', 'frame')][string]$AdaptiveHistory,
        [bool]$DecisionTrace,
        [string]$ComparisonGroup
    )
    return [pscustomobject]@{
        suite = $Suite
        condition_id = $ConditionId
        solver_variant = $SolverVariant
        iterations_per_frame = $IterationsPerFrame
        batched_ls_k = $BatchedLsK
        armijo_beta = 0.5
        ncg_restart_mode = 'non-descent'
        adaptive_ls_history = $AdaptiveHistory
        decision_trace = $DecisionTrace
        comparison_group = $ComparisonGroup
    }
}

$scene = 'scenes\moving_sphere_cloth.xml'
$dimension = 386
$budgetRows = @(Import-Csv -LiteralPath $budgetPath)
$referenceMatches = @($budgetRows | Where-Object {
    $_.scene_id -eq 'moving-sphere' -and [int]$_.cloth_dimension -eq $dimension `
        -and $_.solver_variant -eq 'gpu-gather-fusion-batched-ls-persistent' -and $_.qualified -eq '1'
})
if ($referenceMatches.Count -ne 1) {
    throw 'Expected one qualified persistent calibration row for moving-sphere d386.'
}
$referenceDir = Resolve-ReferenceCheckpointDirectory -Path $referenceMatches[0].reference_dir

$baselineConditions = @(
    (New-Condition -Suite 'baseline' -ConditionId 'edge-scatter-k8' -SolverVariant 'gpu-edge-scatter' -IterationsPerFrame 1 -BatchedLsK 8 -AdaptiveHistory 'none' -DecisionTrace $false -ComparisonGroup 'traversal'),
    (New-Condition -Suite 'baseline' -ConditionId 'vertex-gather-k8' -SolverVariant 'gpu-gather-no-fusion' -IterationsPerFrame 1 -BatchedLsK 8 -AdaptiveHistory 'none' -DecisionTrace $false -ComparisonGroup 'traversal'),
    (New-Condition -Suite 'baseline' -ConditionId 'gather-fusion-k8' -SolverVariant 'gpu-gather-fusion' -IterationsPerFrame 1 -BatchedLsK 8 -AdaptiveHistory 'none' -DecisionTrace $false -ComparisonGroup 'traversal')
)
$stressConditions = @(
    (New-Condition -Suite 'stress' -ConditionId 'edge-scatter-k8' -SolverVariant 'gpu-edge-scatter' -IterationsPerFrame 8 -BatchedLsK 8 -AdaptiveHistory 'none' -DecisionTrace $false -ComparisonGroup 'traversal'),
    (New-Condition -Suite 'stress' -ConditionId 'vertex-gather-k8' -SolverVariant 'gpu-gather-no-fusion' -IterationsPerFrame 8 -BatchedLsK 8 -AdaptiveHistory 'none' -DecisionTrace $false -ComparisonGroup 'traversal'),
    (New-Condition -Suite 'stress' -ConditionId 'gather-fusion-k8' -SolverVariant 'gpu-gather-fusion' -IterationsPerFrame 8 -BatchedLsK 8 -AdaptiveHistory 'none' -DecisionTrace $false -ComparisonGroup 'traversal'),
    (New-Condition -Suite 'stress' -ConditionId 'fixed-k4' -SolverVariant 'gpu-gather-fusion-batched-ls-persistent' -IterationsPerFrame 8 -BatchedLsK 4 -AdaptiveHistory 'none' -DecisionTrace $true -ComparisonGroup 'line-search'),
    (New-Condition -Suite 'stress' -ConditionId 'adaptive-k4-no-history' -SolverVariant 'gpu-gather-fusion-adaptive-ls-persistent' -IterationsPerFrame 8 -BatchedLsK 4 -AdaptiveHistory 'none' -DecisionTrace $true -ComparisonGroup 'line-search'),
    (New-Condition -Suite 'stress' -ConditionId 'adaptive-k4-frame-history' -SolverVariant 'gpu-gather-fusion-adaptive-ls-persistent' -IterationsPerFrame 8 -BatchedLsK 4 -AdaptiveHistory 'frame' -DecisionTrace $true -ComparisonGroup 'line-search')
)

function Rotate-Conditions {
    param([object[]]$Conditions, [int]$Offset)
    $result = @()
    for ($index = 0; $index -lt $Conditions.Count; ++$index) {
        $result += $Conditions[($index + $Offset) % $Conditions.Count]
    }
    return $result
}

$timingPlan = @()
for ($block = 1; $block -le $TimingRepetitions; ++$block) {
    $orderIndex = 1
    foreach ($condition in (Rotate-Conditions -Conditions $baselineConditions -Offset (($block - 1) % $baselineConditions.Count))) {
        $timingPlan += [pscustomobject]@{
            block = $block; suite = $condition.suite; order_index = $orderIndex; condition_id = $condition.condition_id
            solver_variant = $condition.solver_variant; iterations_per_frame = $condition.iterations_per_frame
            batched_ls_k = $condition.batched_ls_k; adaptive_ls_history = $condition.adaptive_ls_history
            comparison_group = $condition.comparison_group
        }
        ++$orderIndex
    }
    $orderIndex = 1
    foreach ($condition in (Rotate-Conditions -Conditions $stressConditions -Offset ($block - 1))) {
        $timingPlan += [pscustomobject]@{
            block = $block; suite = $condition.suite; order_index = $orderIndex; condition_id = $condition.condition_id
            solver_variant = $condition.solver_variant; iterations_per_frame = $condition.iterations_per_frame
            batched_ls_k = $condition.batched_ls_k; adaptive_ls_history = $condition.adaptive_ls_history
            comparison_group = $condition.comparison_group
        }
        ++$orderIndex
    }
}

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$commit = (& git -c "safe.directory=$ProjectRoot" -C $ProjectRoot rev-parse --short HEAD 2>$null | Select-Object -First 1).Trim()
$manifest = [ordered]@{
    protocol_version = 'minimal-optimizer-effects-r1'
    label = $RunLabel
    git_commit = $commit
    calibration_root = $CalibrationRoot
    calibration_commit = 'c73d2bb'
    reference_dir = $referenceDir
    scene = $scene
    practical_effect_threshold = 0.03
    measurement = [ordered]@{ mode = 'rendered-end-to-end'; render_width = $RenderWidth; render_height = $RenderHeight; sync_gpu = $true; disable_vsync = $true; quality_readback_during_timing = $false }
    timing = [ordered]@{ frames = $TimingFrames; warmup = $TimingWarmup; repetitions = $TimingRepetitions; order = 'cyclic-balanced-within-suite' }
    quality = [ordered]@{ frames = $QualityFrames; warmup = $QualityWarmup; checkpoint_stride = 1; timing_separate = $true; position_gate_p95 = 1.0e-3 }
    baseline = [ordered]@{ scene = $scene; cloth_dimension = $dimension; iterations_per_frame = 1; conditions = $baselineConditions }
    stress = [ordered]@{ scene = $scene; cloth_dimension = $dimension; iterations_per_frame = 8; conditions = $stressConditions }
    comparisons = @(
        [ordered]@{ id = 'baseline-gather-vs-scatter'; suite = 'baseline'; control = 'edge-scatter-k8'; treatment = 'vertex-gather-k8'; family = 'traversal' },
        [ordered]@{ id = 'baseline-fusion-vs-gather'; suite = 'baseline'; control = 'vertex-gather-k8'; treatment = 'gather-fusion-k8'; family = 'fusion' },
        [ordered]@{ id = 'baseline-fusion-vs-scatter'; suite = 'baseline'; control = 'edge-scatter-k8'; treatment = 'gather-fusion-k8'; family = 'traversal+fusion' },
        [ordered]@{ id = 'stress-gather-vs-scatter'; suite = 'stress'; control = 'edge-scatter-k8'; treatment = 'vertex-gather-k8'; family = 'traversal' },
        [ordered]@{ id = 'stress-fusion-vs-gather'; suite = 'stress'; control = 'vertex-gather-k8'; treatment = 'gather-fusion-k8'; family = 'fusion' },
        [ordered]@{ id = 'stress-fusion-vs-scatter'; suite = 'stress'; control = 'edge-scatter-k8'; treatment = 'gather-fusion-k8'; family = 'traversal+fusion' },
        [ordered]@{ id = 'stress-adaptive-no-history-vs-fixed'; suite = 'stress'; control = 'fixed-k4'; treatment = 'adaptive-k4-no-history'; family = 'line-search' },
        [ordered]@{ id = 'stress-adaptive-frame-history-vs-fixed'; suite = 'stress'; control = 'fixed-k4'; treatment = 'adaptive-k4-frame-history'; family = 'line-search' },
        [ordered]@{ id = 'stress-frame-history-vs-no-history'; suite = 'stress'; control = 'adaptive-k4-no-history'; treatment = 'adaptive-k4-frame-history'; family = 'line-search-history' }
    )
}
[System.IO.File]::WriteAllText((Join-Path $RunRoot 'manifest.json'), ($manifest | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))
$timingPlan | Export-Csv -LiteralPath (Join-Path $RunRoot 'planned_timing_runs.csv') -NoTypeInformation
if ($DryRun) {
    Write-Host "Minimal optimizer-effects dry-run: $RunRoot"
    exit 0
}

function Invoke-BenchmarkRun {
    param([pscustomobject]$Condition, [string]$Kind, [int]$Block = 0)
    $suffix = if ($Kind -eq 'timing') { ('block{0:D2}' -f $Block) } else { 'trace' }
    $outputDir = Join-Path $RunRoot (Join-Path $Kind (Join-Path $Condition.suite (Join-Path $Condition.condition_id $suffix)))
    $requiredFile = if ($Kind -eq 'timing') { 'frame_presentation.csv' } else { 'quality_metrics.csv' }
    if ((Test-Path -LiteralPath (Join-Path $outputDir $requiredFile)) -and -not $Force) { return }

    $extraArgs = @('--scene', $scene, '--cloth-dimension', $dimension, '--batched-ls-k', $Condition.batched_ls_k, '--armijo-beta', '0.5', '--ncg-restart-mode', 'non-descent')
    if ($Kind -eq 'quality' -and $Condition.decision_trace) { $extraArgs += '--profile-line-search-decisions' }
    $parameters = @{
        ProjectRoot = $ProjectRoot
        RunLabel = "$RunLabel-$($Condition.suite)-$($Condition.condition_id)-$suffix"
        OutputDir = $outputDir
        SolverVariant = $Condition.solver_variant
        IterationsPerFrame = $Condition.iterations_per_frame
        AdaptiveLsHistory = $Condition.adaptive_ls_history
        Frames = if ($Kind -eq 'timing') { $TimingFrames } else { $QualityFrames }
        Warmup = if ($Kind -eq 'timing') { $TimingWarmup } else { $QualityWarmup }
        Uncapped = $true
        SyncGpu = $true
        DisableVsync = $true
        RenderWidth = $RenderWidth
        RenderHeight = $RenderHeight
        ProcessTimeoutSeconds = $ProcessTimeoutSeconds
        ExtraArgs = $extraArgs
    }
    if ($Kind -eq 'quality') {
        $parameters.QualityMetrics = $true
        $parameters.QualityReferenceDir = $referenceDir
        $parameters.QualityCheckpointStride = 1
    }
    & $benchmarkScript @parameters | Out-Host
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $outputDir $requiredFile))) {
        throw "Benchmark run failed: $outputDir"
    }
}

$allConditions = @($baselineConditions + $stressConditions)
for ($block = 1; $block -le $TimingRepetitions; ++$block) {
    foreach ($suite in @('baseline', 'stress')) {
        $blockRows = @($timingPlan | Where-Object { [int]$_.block -eq $block -and $_.suite -eq $suite } | Sort-Object order_index)
        foreach ($row in $blockRows) {
            $condition = @($allConditions | Where-Object { $_.suite -eq $row.suite -and $_.condition_id -eq $row.condition_id })
            if ($condition.Count -ne 1) { throw "Could not resolve planned condition: $($row.suite) $($row.condition_id)" }
            Invoke-BenchmarkRun -Condition $condition[0] -Kind 'timing' -Block $block
            if ($InterRunDelayMilliseconds -gt 0) { Start-Sleep -Milliseconds $InterRunDelayMilliseconds }
        }
    }
}
foreach ($condition in $allConditions) {
    Invoke-BenchmarkRun -Condition $condition -Kind 'quality'
}

if ($PythonExe -eq '') { $PythonExe = if (Test-Path -LiteralPath 'E:\Anaconda\envs\DL\python.exe') { 'E:\Anaconda\envs\DL\python.exe' } else { 'python' } }
& $PythonExe $analysisScript --run-root $RunRoot --report-path (Join-Path $ProjectRoot 'docs\experiments\2026-07-27-minimal-optimizer-effects-recheck.md')
if ($LASTEXITCODE -ne 0) { throw "Minimal optimizer-effects analysis failed with exit code $LASTEXITCODE." }
Write-Host "Minimal optimizer-effects recheck complete: $RunRoot"
