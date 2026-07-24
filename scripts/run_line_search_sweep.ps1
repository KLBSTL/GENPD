[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = ('line-search-' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [string]$OutputDir = '',
    [string]$Scene = 'scenes\moving_sphere_cloth.xml',
    [int]$ClothDimension = 256,
    [int]$IterationsPerFrame = 16,
    [int]$TimingFrames = 300,
    [int]$TimingWarmup = 30,
    [int]$TimingRepetitions = 3,
    [int]$TraceFrames = 120,
    [int]$TraceWarmup = 20,
    [int]$ProcessTimeoutSeconds = 180,
    [int]$InterRunDelayMilliseconds = 1000,
    [int[]]$KValues = @(1, 2, 4, 8),
    [double[]]$Betas = @(0.25, 0.5, 0.75),
    [ValidateSet('none', 'periodic', 'non-descent')]
    [string[]]$RestartModes = @('none', 'periodic', 'non-descent'),
    [int]$RestartPeriod = 8,
    [ValidateSet('serial', 'fixed', 'adaptive')]
    [string[]]$Schedules = @('fixed', 'adaptive'),
    [ValidateSet('none', 'iteration', 'frame')]
    [string[]]$AdaptiveHistoryModes = @('none', 'iteration', 'frame'),
    [string]$QualityReferenceDir = '',
    [switch]$RequireReference,
    [switch]$IncludePersistent,
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') { $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir '..')) }
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($OutputDir -eq '') { $OutputDir = Join-Path $ProjectRoot (Join-Path 'results' $RunLabel) }
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$benchmarkScript = Join-Path $scriptDir 'run_benchmark.ps1'

if (-not (Test-Path -LiteralPath $benchmarkScript)) { throw "Missing benchmark wrapper: $benchmarkScript" }
if ($ClothDimension -lt 2 -or $IterationsPerFrame -lt 1) { throw 'ClothDimension and IterationsPerFrame must be positive.' }
if ($TimingFrames -lt 1 -or $TraceFrames -lt 1 -or $TimingWarmup -lt 0 -or $TraceWarmup -lt 0 -or $TimingRepetitions -lt 1) { throw 'Frame counts are invalid.' }
if ($ProcessTimeoutSeconds -lt 1 -or $InterRunDelayMilliseconds -lt 0) { throw 'Timeout and inter-run delay values are invalid.' }
if ($RestartModes -contains 'periodic' -and $RestartPeriod -lt 1) { throw 'RestartPeriod must be positive for periodic restart.' }
foreach ($k in $KValues) { if ($k -lt 1) { throw 'All K values must be positive.' } }
foreach ($beta in $Betas) { if ($beta -le 0 -or $beta -ge 1) { throw 'All beta values must be in (0, 1).' } }

$scenePath = if ([System.IO.Path]::IsPathRooted($Scene)) { [System.IO.Path]::GetFullPath($Scene) } else { Join-Path $ProjectRoot $Scene }
if (-not (Test-Path -LiteralPath $scenePath)) { throw "Scene was not found: $scenePath" }
if ($QualityReferenceDir -ne '') {
    $QualityReferenceDir = [System.IO.Path]::GetFullPath($QualityReferenceDir)
    if (-not (Test-Path -LiteralPath $QualityReferenceDir)) { throw "Quality reference directory was not found: $QualityReferenceDir" }
}
if ($RequireReference -and $QualityReferenceDir -eq '') { throw 'RequireReference needs QualityReferenceDir.' }

function Format-Double { param([double]$Value) return $Value.ToString('R', $invariant) }
function Read-Double {
    param($Value)
    $parsed = 0.0
    if ($null -eq $Value -or -not [double]::TryParse([string]$Value, [System.Globalization.NumberStyles]::Float, $invariant, [ref]$parsed)) { return $null }
    if ([double]::IsNaN($parsed) -or [double]::IsInfinity($parsed)) { return $null }
    return $parsed
}
function Get-Mean { param([double[]]$Values) if ($Values.Count -eq 0) { return $null }; return [double](($Values | Measure-Object -Average).Average) }
function Get-Std {
    param([double[]]$Values)
    if ($Values.Count -lt 2) { return 0.0 }
    $mean = Get-Mean $Values; $sum = 0.0
    foreach ($value in $Values) { $sum += ($value - $mean) * ($value - $mean) }
    return [math]::Sqrt($sum / [double]($Values.Count - 1))
}
function Get-Percentile {
    param([double[]]$Values, [double]$Quantile)
    if ($Values.Count -eq 0) { return $null }
    $sorted = @($Values | Sort-Object); $position = ($sorted.Count - 1) * $Quantile
    $lo = [math]::Floor($position); $hi = [math]::Ceiling($position)
    if ($lo -eq $hi) { return [double]$sorted[$lo] }
    return [double]$sorted[$lo] + ($position - $lo) * ([double]$sorted[$hi] - [double]$sorted[$lo])
}
function Sum-Field {
    param([object[]]$Rows, [string]$Field)
    $values = @($Rows | ForEach-Object { Read-Double $_.$Field } | Where-Object { $null -ne $_ })
    if ($values.Count -eq 0) { return 0.0 }
    return [double](($values | Measure-Object -Sum).Sum)
}
function Get-ScheduleVariant {
    param([string]$Schedule)
    if ($Schedule -eq 'serial') { return 'gpu-gather-fusion-serial-ls-persistent' }
    if ($Schedule -eq 'fixed') { return 'gpu-gather-fusion-batched-ls-persistent' }
    return 'gpu-gather-fusion-adaptive-ls-persistent'
}
function Get-CaseId {
    param([string]$Schedule, [int]$K, [double]$Beta, [string]$RestartMode, [string]$HistoryMode)
    $betaText = (Format-Double $Beta).Replace('.', 'p')
    if ($Schedule -eq 'adaptive') { return "adaptive-k$K-b$betaText-h$HistoryMode-r$RestartMode" }
    if ($Schedule -eq 'serial') { return "serial-b$betaText-r$RestartMode" }
    return "fixed-k$K-b$betaText-r$RestartMode"
}
function Get-RunExtraArgs {
    param([int]$K, [double]$Beta, [string]$RestartMode)
    $args = @('--cloth-dimension', $ClothDimension, '--scene', $Scene, '--batched-ls-k', $K, '--armijo-beta', (Format-Double $Beta), '--ncg-restart-mode', $RestartMode)
    if ($RestartMode -eq 'periodic') { $args += @('--ncg-restart-period', $RestartPeriod) }
    return $args
}
function Test-RenderedRun {
    param([string]$Directory, [int]$Frames, [int]$Warmup, [switch]$RequireQuality, [switch]$RequireDecisionTrace, [switch]$RequireAdaptiveTrace)
    $metadataPath = Join-Path $Directory 'run_metadata.json'
    $extendedPath = Join-Path $Directory 'frame_profile_extended.csv'
    $presentationPath = Join-Path $Directory 'frame_presentation.csv'
    foreach ($path in @($metadataPath, $extendedPath, $presentationPath)) { if (-not (Test-Path -LiteralPath $path)) { return $false } }
    if ($RequireQuality -and -not (Test-Path -LiteralPath (Join-Path $Directory 'quality_metrics.csv'))) { return $false }
    if ($RequireAdaptiveTrace -and -not (Test-Path -LiteralPath (Join-Path $Directory 'line_search_trace.csv'))) { return $false }
    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    if ([bool]$metadata.benchmark.no_render -or -not [bool]$metadata.benchmark.sync_gpu -or -not [bool]$metadata.benchmark.disable_vsync) { return $false }
    if ($RequireDecisionTrace -and $metadata.solver_controls.line_search_decisions_profiled -ne '1') { return $false }
    $extended = @(Import-Csv -LiteralPath $extendedPath | Where-Object { [int]$_.frame -ge $Warmup })
    $presentation = @(Import-Csv -LiteralPath $presentationPath | Where-Object { [int]$_.frame -ge $Warmup })
    if ($extended.Count -ne $Frames -or $presentation.Count -ne $Frames) { return $false }
    foreach ($field in @('adaptive_history_uses', 'adaptive_history_resets', 'adaptive_first_batch_accepts', 'adaptive_second_batch_accepts', 'adaptive_candidate_evaluations', 'adaptive_batch_count')) {
        if (-not ($extended[0].PSObject.Properties.Name -contains $field)) { return $false }
    }
    foreach ($row in $extended) { if ($row.frame_valid -ne '1' -or $row.termination_reason -ne 'none') { return $false } }
    foreach ($row in $presentation) {
        if ($row.rendered -ne '1' -or $row.gpu_sync_enabled -ne '1' -or $null -eq (Read-Double $row.frame_wall_ms)) { return $false }
    }
    return $true
}
function Invoke-Run {
    param([string]$Kind, [string]$Schedule, [string]$HistoryMode, [int]$K, [double]$Beta, [string]$RestartMode, [int]$Repetition = 0)
    $variant = Get-ScheduleVariant $Schedule
    $caseId = Get-CaseId -Schedule $Schedule -K $K -Beta $Beta -RestartMode $RestartMode -HistoryMode $HistoryMode
    $directory = Join-Path $OutputDir (Join-Path $Kind $caseId)
    if ($Kind -eq 'timing') { $directory = Join-Path $directory ('rep{0:D2}' -f $Repetition) }
    $frames = if ($Kind -eq 'timing') { $TimingFrames } else { $TraceFrames }
    $warmup = if ($Kind -eq 'timing') { $TimingWarmup } else { $TraceWarmup }
    $trace = $Kind -eq 'trace'
    $adaptiveTrace = $trace -and $Schedule -eq 'adaptive'
    if (-not $Force -and (Test-RenderedRun -Directory $directory -Frames $frames -Warmup $warmup -RequireQuality:$trace -RequireDecisionTrace:$trace -RequireAdaptiveTrace:$adaptiveTrace)) { return $directory }
    if ($DryRun) { Write-Host "[dry-run] $Kind $caseId$(if ($Kind -eq 'timing') { " rep$Repetition" })"; return $directory }
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    $extra = Get-RunExtraArgs -K $K -Beta $Beta -RestartMode $RestartMode
    if ($trace) { $extra += '--profile-line-search-decisions' }
    $parameters = @{
        ProjectRoot = $ProjectRoot; RunLabel = "$RunLabel-$Kind-$caseId$(if ($Kind -eq 'timing') { "-rep$('{0:D2}' -f $Repetition)" })"
        Frames = $frames; Warmup = $warmup; SolverVariant = $variant; AdaptiveLsHistory = $(if ($Schedule -eq 'adaptive') { $HistoryMode } else { 'none' })
        IterationsPerFrame = $IterationsPerFrame; OutputDir = $directory; Uncapped = $true; SyncGpu = $true; DisableVsync = $true
        RenderWidth = 1600; RenderHeight = 900; ProcessTimeoutSeconds = $ProcessTimeoutSeconds; ExtraArgs = $extra
    }
    if ($trace) {
        $parameters.QualityMetrics = $true
        if ($QualityReferenceDir -ne '') { $parameters.QualityReferenceDir = $QualityReferenceDir; $parameters.QualityCheckpointStride = 1 }
    }
    & $benchmarkScript @parameters | Out-Host
    if ($InterRunDelayMilliseconds -gt 0) { Start-Sleep -Milliseconds $InterRunDelayMilliseconds }
    if (-not (Test-RenderedRun -Directory $directory -Frames $frames -Warmup $warmup -RequireQuality:$trace -RequireDecisionTrace:$trace -RequireAdaptiveTrace:$adaptiveTrace)) {
        throw "Incomplete or invalid $Kind run: $directory"
    }
    return $directory
}
function Get-CaseSummary {
    param([string[]]$TimingDirs, [string]$TraceDir, [string]$Schedule, [string]$HistoryMode, [int]$K, [double]$Beta, [string]$RestartMode)
    $timingPresentation = @(); $timingProfile = @(); $timingMeans = @(); $totalMeans = @()
    foreach ($timingDir in $TimingDirs) {
        $presentation = @(Import-Csv -LiteralPath (Join-Path $timingDir 'frame_presentation.csv') | Where-Object { [int]$_.frame -ge $TimingWarmup })
        $profile = @(Import-Csv -LiteralPath (Join-Path $timingDir 'frame_profile.csv') | Where-Object { [int]$_.frame -ge $TimingWarmup })
        $timingPresentation += $presentation; $timingProfile += $profile
        $timingMeans += Get-Mean @($presentation | ForEach-Object { Read-Double $_.frame_wall_ms } | Where-Object { $null -ne $_ })
        $totalMeans += Get-Mean @($profile | ForEach-Object { Read-Double $_.total_ms } | Where-Object { $null -ne $_ })
    }
    $extended = @(Import-Csv -LiteralPath (Join-Path $TraceDir 'frame_profile_extended.csv') | Where-Object { [int]$_.frame -ge $TraceWarmup })
    $traceProfile = @(Import-Csv -LiteralPath (Join-Path $TraceDir 'frame_profile.csv') | Where-Object { [int]$_.frame -ge $TraceWarmup })
    $quality = @(Import-Csv -LiteralPath (Join-Path $TraceDir 'quality_metrics.csv') | Where-Object { [int]$_.frame -ge $TraceWarmup })
    $searches = Sum-Field $extended 'cs_full_ls'
    $candidateEvaluations = Sum-Field $extended 'adaptive_candidate_evaluations'
    $historyUses = Sum-Field $extended 'adaptive_history_uses'
    $firstAccepts = Sum-Field $extended 'adaptive_first_batch_accepts'
    $secondAccepts = Sum-Field $extended 'adaptive_second_batch_accepts'
    $fallbacks = Sum-Field $extended 'armijo_fallbacks'
    $invalid = @($extended | Where-Object { $_.frame_valid -ne '1' -or $_.termination_reason -ne 'none' }).Count
    $positionError = @($quality | Where-Object { $_.has_reference -eq '1' } | ForEach-Object { Read-Double $_.position_rel_l2 } | Where-Object { $null -ne $_ })
    [pscustomobject]@{
        schedule = $Schedule; solver_variant = Get-ScheduleVariant $Schedule; adaptive_ls_history = if ($Schedule -eq 'adaptive') { $HistoryMode } else { 'none' }
        batched_ls_k = $K; armijo_beta = Format-Double $Beta; ncg_restart_mode = $RestartMode; ncg_restart_period = if ($RestartMode -eq 'periodic') { $RestartPeriod } else { 0 }
        timing_dirs = ($TimingDirs -join ';'); trace_dir = $TraceDir; timing_repetitions = $TimingDirs.Count; timing_frames_per_repetition = $TimingFrames
        timing_frame_samples = $timingPresentation.Count; trace_frame_samples = $extended.Count
        rendered_frame_wall_ms_mean = Get-Mean $timingMeans; rendered_frame_wall_ms_std = Get-Std $timingMeans
        rendered_frame_wall_ms_p95 = Get-Percentile @($timingPresentation | ForEach-Object { Read-Double $_.frame_wall_ms } | Where-Object { $null -ne $_ }) 0.95
        total_ms_mean = Get-Mean $totalMeans; line_search_ms = Get-Mean @($traceProfile | ForEach-Object { Read-Double $_.cs_linesearch_ms } | Where-Object { $null -ne $_ })
        candidates_per_search = if ($Schedule -eq 'serial') { 0.0 } elseif ($searches -gt 0 -and $Schedule -eq 'adaptive') { $candidateEvaluations / $searches } else { [double]$K }
        candidate_evaluation_measurement = if ($Schedule -eq 'serial') { 'not-applicable-serial' } else { 'batched-candidates-per-search' }
        history_use_ratio = if ($searches -gt 0) { $historyUses / $searches } else { 0.0 }
        first_batch_accept_ratio = if ($searches -gt 0) { $firstAccepts / $searches } else { 0.0 }
        second_batch_accept_ratio = if ($searches -gt 0) { $secondAccepts / $searches } else { 0.0 }
        fallback_ratio = if ($searches -gt 0) { $fallbacks / $searches } else { 0.0 }
        adaptive_history_uses = $historyUses; adaptive_candidate_evaluations = $candidateEvaluations; adaptive_batch_count = Sum-Field $extended 'adaptive_batch_count'
        armijo_rejections = Sum-Field $extended 'armijo_rejections'; armijo_failures = Sum-Field $extended 'armijo_failures'; armijo_fallbacks = $fallbacks
        invalid_trace_frames = $invalid; trace_failure_rate = if ($extended.Count -gt 0) { [double]$invalid / $extended.Count } else { 1.0 }
        p95_position_rel_l2 = Get-Percentile $positionError 0.95; timing_has_decision_tracing = 0; trace_has_decision_tracing = 1
    }
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$manifest = [ordered]@{
    protocol_version = 3; label = $RunLabel; measurement = 'rendered-end-to-end'; timing_run_has_decision_tracing = $false; trace_run_has_decision_tracing = $true
    scene = $Scene; cloth_dimension = $ClothDimension; iterations_per_frame = $IterationsPerFrame
    timing = [ordered]@{ frames = $TimingFrames; warmup = $TimingWarmup; repetitions = $TimingRepetitions; render_width = 1600; render_height = 900 }
    trace = [ordered]@{ frames = $TraceFrames; warmup = $TraceWarmup; quality_reference_dir = $QualityReferenceDir }
    k_values = $KValues; armijo_betas = $Betas; restart_modes = $RestartModes; restart_period = $RestartPeriod
    schedules = $Schedules; adaptive_history_modes = $AdaptiveHistoryModes; serial_variant = 'gpu-gather-fusion-serial-ls-persistent'; fixed_variant = 'gpu-gather-fusion-batched-ls-persistent'; adaptive_variant = 'gpu-gather-fusion-adaptive-ls-persistent'
    process_timeout_seconds = $ProcessTimeoutSeconds; inter_run_delay_milliseconds = $InterRunDelayMilliseconds
}
[System.IO.File]::WriteAllText((Join-Path $OutputDir 'manifest.json'), ($manifest | ConvertTo-Json -Depth 6), [System.Text.UTF8Encoding]::new($false))

$rows = @()
foreach ($schedule in $Schedules) {
    $scheduleKValues = if ($schedule -eq 'serial') { @(1) } else { $KValues }
    foreach ($k in $scheduleKValues) {
        foreach ($beta in $Betas) {
            foreach ($restartMode in $RestartModes) {
                $histories = if ($schedule -eq 'adaptive') { $AdaptiveHistoryModes } else { @('none') }
                foreach ($history in $histories) {
                    $timingDirs = @()
                    for ($rep = 1; $rep -le $TimingRepetitions; ++$rep) {
                        $timingDirs += Invoke-Run -Kind 'timing' -Schedule $schedule -HistoryMode $history -K $k -Beta $beta -RestartMode $restartMode -Repetition $rep
                    }
                    $traceDir = Invoke-Run -Kind 'trace' -Schedule $schedule -HistoryMode $history -K $k -Beta $beta -RestartMode $restartMode
                    if (-not $DryRun) { $rows += Get-CaseSummary -TimingDirs $timingDirs -TraceDir $traceDir -Schedule $schedule -HistoryMode $history -K $k -Beta $beta -RestartMode $restartMode }
                }
            }
        }
    }
}
if (-not $DryRun) {
    $rows | Export-Csv -LiteralPath (Join-Path $OutputDir 'line_search_summary.csv') -NoTypeInformation
    Write-Host "Line-search summary: $(Join-Path $OutputDir 'line_search_summary.csv')"
}
