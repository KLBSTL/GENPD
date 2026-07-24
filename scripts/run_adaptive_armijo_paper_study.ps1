[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = 'paper-20260724-adaptive-armijo-r1',
    [string]$RunRoot = '',
    [string]$CalibrationRoot = '',
    [ValidateSet('archived', 'regenerate')]
    [string]$ReferenceMode = 'archived',
    [ValidateSet('all', 'references', 'core', 'sensitivity', 'analyze')]
    [string]$Stage = 'all',
    [int]$TimingFrames = 300,
    [int]$TimingWarmup = 30,
    [int]$TimingRepetitions = 3,
    [int]$TraceFrames = 120,
    [int]$TraceWarmup = 20,
    [int]$ReferenceIterations = 100,
    [int]$ReferenceCheckpointStride = 10,
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
if ($TimingFrames -lt 1 -or $TimingWarmup -lt 0 -or $TimingRepetitions -ne 3 -or $TraceFrames -lt 1 -or $TraceWarmup -lt 0 -or $ReferenceIterations -lt 1 -or $ReferenceCheckpointStride -lt 1) {
    throw 'Formal adaptive Armijo study requires valid frame counts, reference settings, and exactly three timing repetitions.'
}

$lineSearchScript = Join-Path $scriptDir 'run_line_search_sweep.ps1'
$referenceScript = Join-Path $scriptDir 'run_reference.ps1'
$analysisScript = Join-Path $scriptDir 'analyze_adaptive_armijo_paper_study.py'
$budgetPath = Join-Path $CalibrationRoot 'selected_budgets.csv'
foreach ($path in @($lineSearchScript, $referenceScript, $analysisScript, $budgetPath)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing adaptive Armijo study input: $path" }
}

$sceneMap = [ordered]@{
    'hanging' = 'scenes\test_scene.xml'
    'moving-sphere' = 'scenes\moving_sphere_cloth.xml'
}
$dimensions = @(128, 256, 386)
$coreMethods = @(
    [pscustomobject]@{ method_id = 'serial'; schedule = 'serial'; k = 1; beta = 0.5; restart = 'non-descent'; history = 'none' },
    [pscustomobject]@{ method_id = 'fixed-k8'; schedule = 'fixed'; k = 8; beta = 0.5; restart = 'non-descent'; history = 'none' },
    [pscustomobject]@{ method_id = 'adaptive-k4-history-none'; schedule = 'adaptive'; k = 4; beta = 0.5; restart = 'non-descent'; history = 'none' },
    [pscustomobject]@{ method_id = 'adaptive-k4-history-iteration'; schedule = 'adaptive'; k = 4; beta = 0.5; restart = 'non-descent'; history = 'iteration' },
    [pscustomobject]@{ method_id = 'adaptive-k4-history-frame'; schedule = 'adaptive'; k = 4; beta = 0.5; restart = 'non-descent'; history = 'frame' }
)
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
        $caseId = "$sceneId-d$dimension"
        $archivedReferenceDir = Resolve-ReferenceCheckpointDirectory -Path $matches[0].reference_dir
        $referenceDir = if ($ReferenceMode -eq 'archived') { $archivedReferenceDir } else { Join-Path $RunRoot (Join-Path 'references' (Join-Path $caseId 'reference_checkpoints')) }
        $cases += [pscustomobject]@{
            case_id = $caseId; scene_id = $sceneId; scene_path = $sceneMap[$sceneId]; cloth_dimension = $dimension
            iterations_per_frame = [int]$matches[0].iterations_per_frame
            reference_dir = $referenceDir; archived_reference_dir = $archivedReferenceDir
        }
    }
}
$sensitivityCases = @()
foreach ($dimension in @(256, 386)) {
    $case = @($cases | Where-Object { $_.scene_id -eq 'moving-sphere' -and $_.cloth_dimension -eq $dimension })
    if ($case.Count -ne 1) { throw "Missing moving-sphere sensitivity case at d$dimension." }
    foreach ($k in @(2, 4, 8)) {
        foreach ($beta in @(0.25, 0.5, 0.75)) {
            foreach ($history in @('none', 'iteration', 'frame')) {
                $sensitivityCases += [pscustomobject]@{
                    case_id = $case[0].case_id; scene_id = $case[0].scene_id; scene_path = $case[0].scene_path; cloth_dimension = $dimension
                    iterations_per_frame = $case[0].iterations_per_frame; reference_dir = $case[0].reference_dir
                    k = $k; beta = $beta; history = $history; restart = 'non-descent'
                }
            }
        }
    }
}

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$commit = (& git -c "safe.directory=$ProjectRoot" -C $ProjectRoot rev-parse --short HEAD 2>$null | Select-Object -First 1).Trim()
$manifest = [ordered]@{
    protocol_version = 'adaptive-armijo-paper-v1'
    label = $RunLabel
    git_commit = $commit
    calibration_root = $CalibrationRoot
    calibration_commit = 'c73d2bb'
    measurement = [ordered]@{ mode = 'rendered-end-to-end'; render_width = $RenderWidth; render_height = $RenderHeight; sync_gpu = $true; disable_vsync = $true; quality_readback_during_timing = $false }
    timing = [ordered]@{ frames = $TimingFrames; warmup = $TimingWarmup; repetitions = $TimingRepetitions }
    trace = [ordered]@{ frames = $TraceFrames; warmup = $TraceWarmup; decision_trace = $true; timing_separate = $true; position_gate_p95 = 1.0e-3 }
    reference = [ordered]@{ source = $ReferenceMode; calibration_commit = 'c73d2bb'; solver_variant = 'cpu-ncg'; iterations_per_frame = $ReferenceIterations; frames = $TraceFrames; warmup = $TraceWarmup; checkpoint_stride = $ReferenceCheckpointStride }
    scenes = @($sceneMap.Keys)
    cloth_dimensions = $dimensions
    core_methods = $coreMethods
    sensitivity = [ordered]@{ scene = 'moving-sphere'; cloth_dimensions = @(256, 386); k_values = @(2, 4, 8); betas = @(0.25, 0.5, 0.75); histories = @('none', 'iteration', 'frame'); restart = 'non-descent' }
    go_no_go = [ordered]@{
        candidate_reduction_vs_fixed = 0.25
        adaptive_end_to_end_win_one_case = 0.05
        adaptive_end_to_end_regression_other_case = 0.03
        history_line_search_reduction = 0.05
        p95_position_rel_l2 = 1.0e-3
        require_zero_invalid_frames = $true
        require_zero_armijo_failures = $true
    }
}
[System.IO.File]::WriteAllText((Join-Path $RunRoot 'manifest.json'), ($manifest | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))
$cases | Export-Csv -LiteralPath (Join-Path $RunRoot 'planned_core_cases.csv') -NoTypeInformation
$sensitivityCases | Export-Csv -LiteralPath (Join-Path $RunRoot 'planned_sensitivity_cases.csv') -NoTypeInformation
if ($DryRun) { Write-Host "Adaptive Armijo paper dry-run: $RunRoot"; exit 0 }

function Invoke-Reference {
    param($Case)
    if ($ReferenceMode -eq 'archived') {
        if (-not (Test-Path -LiteralPath (Join-Path $Case.reference_dir 'reference_state_000000.bin'))) { throw "Missing archived reference for $($Case.case_id)." }
        return
    }
    if ((Test-Path -LiteralPath (Join-Path $Case.reference_dir 'reference_state_000000.bin')) -and -not $Force) { return }
    $referenceOutput = Split-Path -Parent $Case.reference_dir
    & $referenceScript -ProjectRoot $ProjectRoot -RunLabel "$RunLabel-reference-$($Case.case_id)" -OutputDir $referenceOutput `
        -Frames $TraceFrames -Warmup $TraceWarmup -ReferenceIterations $ReferenceIterations -CheckpointStride $ReferenceCheckpointStride `
        -SyncGpu -DisableVsync -RenderWidth $RenderWidth -RenderHeight $RenderHeight -ProcessTimeoutSeconds $ProcessTimeoutSeconds `
        -ExtraArgs @('--scene', $Case.scene_path, '--cloth-dimension', $Case.cloth_dimension)
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $Case.reference_dir 'reference_state_000000.bin'))) {
        throw "Reference generation failed for $($Case.case_id)."
    }
}

function Invoke-Sweep {
    param([string]$OutputDir, $Case, [string]$Schedule, [int]$K, [double]$Beta, [string]$History)
    & $lineSearchScript -ProjectRoot $ProjectRoot -RunLabel "$RunLabel-$($Case.case_id)-$Schedule-k$K-b$Beta-h$History" -OutputDir $OutputDir `
        -Scene $Case.scene_path -ClothDimension $Case.cloth_dimension -IterationsPerFrame $Case.iterations_per_frame `
        -TimingFrames $TimingFrames -TimingWarmup $TimingWarmup -TimingRepetitions $TimingRepetitions `
        -TraceFrames $TraceFrames -TraceWarmup $TraceWarmup -ProcessTimeoutSeconds $ProcessTimeoutSeconds -InterRunDelayMilliseconds $InterRunDelayMilliseconds `
        -KValues @($K) -Betas @($Beta) -RestartModes @('non-descent') -Schedules @($Schedule) -AdaptiveHistoryModes @($History) `
        -QualityReferenceDir $Case.reference_dir -RequireReference -Force:$Force
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $OutputDir 'line_search_summary.csv'))) {
        throw "Line-search sweep failed: $OutputDir"
    }
}

if ($Stage -in @('all', 'references')) {
    foreach ($case in $cases) { Invoke-Reference -Case $case }
}
if ($Stage -in @('all', 'core')) {
    foreach ($case in $cases) {
        foreach ($method in $coreMethods) {
            $outputDir = Join-Path $RunRoot (Join-Path 'core' (Join-Path $case.case_id $method.method_id))
            Invoke-Sweep -OutputDir $outputDir -Case $case -Schedule $method.schedule -K $method.k -Beta $method.beta -History $method.history
        }
    }
}
if ($Stage -in @('all', 'sensitivity')) {
    foreach ($case in $sensitivityCases) {
        $betaText = ('{0:R}' -f [double]$case.beta).Replace('.', 'p')
        $outputDir = Join-Path $RunRoot (Join-Path 'sensitivity' (Join-Path $case.case_id ("adaptive-k$($case.k)-b$betaText-h$($case.history)")))
        Invoke-Sweep -OutputDir $outputDir -Case $case -Schedule 'adaptive' -K $case.k -Beta $case.beta -History $case.history
    }
}
if ($Stage -in @('all', 'analyze')) {
    if ($PythonExe -eq '') { $PythonExe = if (Test-Path -LiteralPath 'E:\Anaconda\envs\DL\python.exe') { 'E:\Anaconda\envs\DL\python.exe' } else { 'python' } }
    & $PythonExe $analysisScript --run-root $RunRoot --report-path (Join-Path $ProjectRoot 'docs\experiments\2026-07-24-adaptive-armijo-study.md')
    if ($LASTEXITCODE -ne 0) { throw "Adaptive Armijo analysis failed with exit code $LASTEXITCODE." }
}
Write-Host "Adaptive Armijo study stage '$Stage' complete: $RunRoot"
