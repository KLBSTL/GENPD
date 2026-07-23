[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$RunRoot = '',
    [switch]$AllowIncomplete
)

$ErrorActionPreference = 'Stop'
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') { $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir '..')) }
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($RunRoot -eq '') { $RunRoot = Join-Path $ProjectRoot 'results\paper-20260723' }
$RunRoot = [System.IO.Path]::GetFullPath($RunRoot)
$manifestPath = Join-Path $RunRoot 'manifest.json'
$selectedPath = Join-Path $RunRoot 'selected_budgets.csv'
if (-not (Test-Path -LiteralPath $manifestPath)) { throw "Missing manifest: $manifestPath" }
if (-not (Test-Path -LiteralPath $selectedPath)) { throw "Missing selected equal-quality budgets: $selectedPath" }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ([int]$manifest.performance.repetitions -ne 3) { throw 'Formal performance protocol requires exactly three repetitions.' }
if ([double]$manifest.quality_target.position_rel_l2_p95 -ne 0.001) { throw 'Unexpected equal-quality threshold in manifest.' }

function To-Double {
    param($Value)
    $parsed = 0.0
    if ($null -eq $Value -or -not [double]::TryParse([string]$Value, [System.Globalization.NumberStyles]::Float, $invariant, [ref]$parsed)) { return $null }
    if ([double]::IsNaN($parsed) -or [double]::IsInfinity($parsed)) { return $null }
    return $parsed
}

function Get-Percentile {
    param([double[]]$Values, [double]$Percentile)
    if ($Values.Count -eq 0) { return $null }
    $valuesSorted = @($Values | Sort-Object)
    $position = ($valuesSorted.Count - 1) * $Percentile
    $lo = [math]::Floor($position)
    $hi = [math]::Ceiling($position)
    if ($lo -eq $hi) { return [double]$valuesSorted[$lo] }
    return [double]$valuesSorted[$lo] + ($position - $lo) * ([double]$valuesSorted[$hi] - [double]$valuesSorted[$lo])
}

function Get-Mean {
    param([double[]]$Values)
    if ($Values.Count -eq 0) { return $null }
    return [double](($Values | Measure-Object -Average).Average)
}

function Get-SampleStd {
    param([double[]]$Values)
    if ($Values.Count -lt 2) { return 0.0 }
    $mean = Get-Mean $Values
    $sum = 0.0
    foreach ($value in $Values) { $sum += ($value - $mean) * ($value - $mean) }
    return [math]::Sqrt($sum / ($Values.Count - 1))
}

function Get-FieldValues {
    param([object[]]$Rows, [string]$Field)
    return @($Rows | ForEach-Object { To-Double $_.$Field } | Where-Object { $null -ne $_ })
}

function Get-RequiredRows {
    param([string]$OutputDir, [int]$ExpectedFrames, [int]$Warmup)
    $profilePath = Join-Path $OutputDir 'frame_profile.csv'
    $experimentPath = Join-Path $OutputDir 'frame_profile_experiment.csv'
    $metadataPath = Join-Path $OutputDir 'run_metadata.json'
    foreach ($path in @($profilePath, $experimentPath, $metadataPath)) {
        if (-not (Test-Path -LiteralPath $path)) { throw "Required output is missing: $path" }
    }
    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    foreach ($field in @('git_commit', 'gpu_name', 'nvidia_driver_version')) {
        if ([string]::IsNullOrWhiteSpace([string]$metadata.$field)) { throw "Required metadata field '$field' is missing: $metadataPath" }
    }
    $profileRows = @(Import-Csv -LiteralPath $profilePath | Where-Object { [int]$_.frame -ge $Warmup })
    $experimentRows = @(Import-Csv -LiteralPath $experimentPath | Where-Object { [int]$_.frame -ge $Warmup })
    if ($profileRows.Count -ne $ExpectedFrames -or $experimentRows.Count -ne $ExpectedFrames) {
        throw "Unexpected measured-frame count in $OutputDir (profile=$($profileRows.Count), experiment=$($experimentRows.Count), expected=$ExpectedFrames)."
    }
    foreach ($row in $profileRows) {
        if ($row.exploded -eq '1') { throw "Exploded frame in formal performance run: $OutputDir" }
        foreach ($field in @('total_ms', 'optimization_ms', 'transfer_ms', 'gradient_norm', 'max_position')) {
            if ($null -eq (To-Double $row.$field)) { throw "Non-finite '$field' in formal performance run: $OutputDir" }
        }
    }
    return [pscustomobject]@{ profile = $profileRows; experiment = $experimentRows; metadata = $metadata }
}

$selected = @(Import-Csv -LiteralPath $selectedPath)
if ($selected.Count -eq 0) { throw 'No equal-quality case was selected.' }
$frameSummaryRows = @()
$groupRows = @{}
foreach ($case in $selected) {
    $caseKey = '{0}|{1}|{2}' -f $case.scene_id, $case.cloth_dimension, $case.solver_variant
    $groupRows[$caseKey] = @()
    for ($rep = 1; $rep -le [int]$manifest.performance.repetitions; ++$rep) {
        $caseId = '{0}-d{1}-{2}-rep{3:D2}' -f $case.scene_id, $case.cloth_dimension, $case.solver_variant, $rep
        $outputDir = Join-Path $RunRoot (Join-Path 'performance' $caseId)
        $run = Get-RequiredRows -OutputDir $outputDir -ExpectedFrames ([int]$manifest.performance.frames) -Warmup ([int]$manifest.performance.warmup_frames)
        $profile = $run.profile
        $experiment = $run.experiment
        $summary = [pscustomobject]@{
            scene_id = $case.scene_id
            scene_path = $case.scene_path
            cloth_dimension = [int]$case.cloth_dimension
            solver_variant = $case.solver_variant
            iterations_per_frame = [int]$case.iterations_per_frame
            repetition = $rep
            result_dir = $outputDir
            git_commit = $run.metadata.git_commit
            gpu_name = $run.metadata.gpu_name
            nvidia_driver_version = $run.metadata.nvidia_driver_version
            measured_frames = $profile.Count
            total_ms_mean = Get-Mean (Get-FieldValues $profile 'total_ms')
            optimization_ms_mean = Get-Mean (Get-FieldValues $profile 'optimization_ms')
            transfer_ms_mean = Get-Mean (Get-FieldValues $profile 'transfer_ms')
            cs_gradient_gpu_ms_mean = Get-Mean (Get-FieldValues $profile 'cs_gradient_gpu_ms')
            cs_reduction_gpu_ms_mean = Get-Mean (Get-FieldValues $profile 'cs_reduction_gpu_ms')
            cs_stats_readback_ms_mean = Get-Mean (Get-FieldValues $profile 'cs_stats_readback_ms')
            cs_gradstats_ms_mean = Get-Mean (Get-FieldValues $profile 'cs_gradstats_ms')
            gradient_dispatches_mean = Get-Mean (Get-FieldValues $experiment 'gradient_dispatches')
            stats_dispatches_mean = Get-Mean (Get-FieldValues $experiment 'stats_dispatches')
            reduction_dispatches_mean = Get-Mean (Get-FieldValues $experiment 'reduction_dispatches')
            xupdate_dispatches_mean = Get-Mean (Get-FieldValues $experiment 'xupdate_dispatches')
            descent_dispatches_mean = Get-Mean (Get-FieldValues $experiment 'descent_dispatches')
            host_readbacks_mean = Get-Mean (Get-FieldValues $experiment 'host_readbacks')
            tracked_buffer_bytes_mean = Get-Mean (Get-FieldValues $experiment 'tracked_buffer_bytes')
            persistent_buffers_active = (Get-FieldValues $experiment 'persistent_buffers_active' | Select-Object -First 1)
        }
        $frameSummaryRows += $summary
        $groupRows[$caseKey] += [pscustomobject]@{ run = $summary; profile = $profile; case = $case }
    }
}

$frameSummaryRows | Export-Csv -LiteralPath (Join-Path $RunRoot 'paper_frame_summary.csv') -NoTypeInformation
$paperRows = @()
foreach ($key in $groupRows.Keys) {
    $runs = @($groupRows[$key])
    $first = $runs[0].run
    $caseInfo = $runs[0].case
    $allFrames = @($runs | ForEach-Object { $_.profile })
    $totalMeans = @($runs | ForEach-Object { [double]$_.run.total_ms_mean })
    $dispatchMeans = @($runs | ForEach-Object { [double]$_.run.gradient_dispatches_mean + [double]$_.run.stats_dispatches_mean + [double]$_.run.reduction_dispatches_mean + [double]$_.run.xupdate_dispatches_mean + [double]$_.run.descent_dispatches_mean })
    $paperRows += [pscustomobject]@{
        scene_id = $first.scene_id
        scene_path = $first.scene_path
        cloth_dimension = $first.cloth_dimension
        solver_variant = $first.solver_variant
        iterations_per_frame = $first.iterations_per_frame
        repetitions = $runs.Count
        git_commit = $first.git_commit
        gpu_name = $first.gpu_name
        nvidia_driver_version = $first.nvidia_driver_version
        total_ms_mean = Get-Mean $totalMeans
        total_ms_std = Get-SampleStd $totalMeans
        total_ms_p50 = Get-Percentile -Values (Get-FieldValues $allFrames 'total_ms') -Percentile 0.50
        total_ms_p95 = Get-Percentile -Values (Get-FieldValues $allFrames 'total_ms') -Percentile 0.95
        optimization_ms_mean = Get-Mean @($runs | ForEach-Object { [double]$_.run.optimization_ms_mean })
        transfer_ms_mean = Get-Mean @($runs | ForEach-Object { [double]$_.run.transfer_ms_mean })
        cs_gradient_gpu_ms_mean = Get-Mean @($runs | ForEach-Object { [double]$_.run.cs_gradient_gpu_ms_mean })
        cs_reduction_gpu_ms_mean = Get-Mean @($runs | ForEach-Object { [double]$_.run.cs_reduction_gpu_ms_mean })
        cs_stats_readback_ms_mean = Get-Mean @($runs | ForEach-Object { [double]$_.run.cs_stats_readback_ms_mean })
        cs_gradstats_ms_mean = Get-Mean @($runs | ForEach-Object { [double]$_.run.cs_gradstats_ms_mean })
        dispatches_mean = Get-Mean $dispatchMeans
        host_readbacks_mean = Get-Mean @($runs | ForEach-Object { [double]$_.run.host_readbacks_mean })
        tracked_buffer_bytes_mean = Get-Mean @($runs | ForEach-Object { [double]$_.run.tracked_buffer_bytes_mean })
        persistent_buffers_active = $first.persistent_buffers_active
        p95_position_rel_l2 = $caseInfo.p95_position_rel_l2
        p95_velocity_rel_l2 = $caseInfo.p95_velocity_rel_l2
        p95_energy_rel_error = $caseInfo.p95_energy_rel_error
        max_penetration_depth = $caseInfo.max_penetration_depth
        calibration_failure_rate = $caseInfo.failure_rate
    }
}
$paperRows | Sort-Object scene_id, cloth_dimension, solver_variant | Export-Csv -LiteralPath (Join-Path $RunRoot 'paper_summary.csv') -NoTypeInformation

$stabilityReplicatePath = Join-Path $RunRoot 'stability_replicates.csv'
if (-not $AllowIncomplete -and -not (Test-Path -LiteralPath $stabilityReplicatePath)) { throw "Missing formal stability data: $stabilityReplicatePath" }
if (Test-Path -LiteralPath $stabilityReplicatePath) {
    $stabilityRows = @(Import-Csv -LiteralPath $stabilityReplicatePath)
    if (-not $AllowIncomplete -and $stabilityRows.Count -ne 27) { throw "Formal stability data must contain 27 repetitions, found $($stabilityRows.Count)." }
}

Write-Host "Per-repetition summary: $(Join-Path $RunRoot 'paper_frame_summary.csv')"
Write-Host "Formal paper summary: $(Join-Path $RunRoot 'paper_summary.csv')"
