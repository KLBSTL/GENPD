[CmdletBinding()]
param(
    [string] $ProjectRoot = '',
    [string] $RunLabel = 'paper-20260729-ppm-r1',
    [string] $RunRoot = '',
    [string[]] $Scenes = @('hanging', 'moving-sphere'),
    [int[]] $Dimensions = @(128),
    [int] $Warmup = 30,
    [int] $Frames = 300,
    [int] $Repetitions = 3,
    [int] $SolverIterations = 10,
    [float] $Dt = 0.001,
    [int] $RenderWidth = 1600,
    [int] $RenderHeight = 900,
    [string] $PeridynoRoot = '',
    [string] $BuildDir = '',
    [switch] $DryRun,
    [switch] $Force
)

$ErrorActionPreference = 'Stop'
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') { $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir '..')) }
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ($RunRoot -eq '') { $RunRoot = Join-Path $ProjectRoot (Join-Path 'results' $RunLabel) }
$RunRoot = [System.IO.Path]::GetFullPath($RunRoot)
if ($PeridynoRoot -eq '') { $PeridynoRoot = Join-Path $ProjectRoot '..\external\peridyno-ppm' }
$PeridynoRoot = [System.IO.Path]::GetFullPath($PeridynoRoot)
if ($BuildDir -eq '') { $BuildDir = Join-Path $PeridynoRoot 'build-genpd-ppm-cmake331' }
$BuildDir = [System.IO.Path]::GetFullPath($BuildDir)
$runner = Join-Path $scriptDir 'run_peridyno_ppm_operating_point.ps1'
$Scenes = @($Scenes | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' })

if ($Scenes.Count -lt 1 -or $Dimensions.Count -lt 1 -or $Warmup -lt 0 -or $Frames -lt 1 -or $Repetitions -lt 1 -or
    $SolverIterations -lt 1 -or $Dt -le 0 -or $RenderWidth -lt 1 -or $RenderHeight -lt 1) {
    throw 'Formal PPM operating-point arguments are invalid.'
}
foreach ($scene in $Scenes) { if ($scene -notin @('hanging', 'moving-sphere')) { throw "Unsupported PPM scene: $scene" } }
foreach ($dimension in $Dimensions) { if ($dimension -lt 2) { throw 'PPM dimensions must be at least 2.' } }
if ($RunLabel -match '^paper-' -and ($Warmup -ne 30 -or $Frames -ne 300 -or $Repetitions -ne 3)) {
    throw 'Paper-labelled PPM runs require exactly 30 warm-up frames, 300 measured frames, and three repetitions.'
}
if (-not (Test-Path -LiteralPath $runner)) { throw "Missing PPM runner: $runner" }
if (-not (Test-Path -LiteralPath (Join-Path $PeridynoRoot 'CMakeLists.txt'))) { throw "Missing PeriDyno source: $PeridynoRoot" }

function Convert-ToFiniteDouble {
    param($Value)
    $parsed = 0.0
    if ($null -eq $Value -or -not [double]::TryParse([string]$Value, [System.Globalization.NumberStyles]::Float, $invariant, [ref]$parsed)) { return $null }
    if ([double]::IsNaN($parsed) -or [double]::IsInfinity($parsed)) { return $null }
    return $parsed
}

function Get-Mean {
    param([double[]] $Values)
    if ($Values.Count -eq 0) { return $null }
    return [double](($Values | Measure-Object -Average).Average)
}

function Get-SampleStd {
    param([double[]] $Values)
    if ($Values.Count -lt 2) { return 0.0 }
    $mean = Get-Mean $Values
    $sum = 0.0
    foreach ($value in $Values) { $sum += ($value - $mean) * ($value - $mean) }
    return [math]::Sqrt($sum / ($Values.Count - 1))
}

function Get-Percentile {
    param([double[]] $Values, [double] $Quantile)
    if ($Values.Count -eq 0) { return $null }
    $sorted = @($Values | Sort-Object)
    $position = ($sorted.Count - 1) * $Quantile
    $low = [math]::Floor($position); $high = [math]::Ceiling($position)
    if ($low -eq $high) { return [double]$sorted[$low] }
    return [double]$sorted[$low] + ($position - $low) * ([double]$sorted[$high] - [double]$sorted[$low])
}

function Read-HardwareMetadata {
    $gpu = ''; $driver = ''; $memory = ''
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        $line = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null | Select-Object -First 1
        if ($line) {
            $parts = @($line -split ',' | ForEach-Object { $_.Trim() })
            if ($parts.Count -ge 3) { $gpu = $parts[0]; $driver = $parts[1]; $memory = $parts[2] }
        }
    }
    if ([string]::IsNullOrWhiteSpace($gpu) -or [string]::IsNullOrWhiteSpace($driver)) { throw 'nvidia-smi did not provide a GPU model and driver version.' }
    return [ordered]@{ gpu_name = $gpu; nvidia_driver_version = $driver; gpu_memory_total = $memory }
}

function Read-ValidatedRun {
    param([string] $Directory, [string] $Scene, [int] $Dimension, [int] $Repetition)
    $csvPath = Join-Path $Directory 'frame_profile.csv'
    $metadataPath = Join-Path $Directory 'run_metadata.json'
    foreach ($path in @($csvPath, $metadataPath)) { if (-not (Test-Path -LiteralPath $path)) { throw "Required PPM artifact is missing: $path" } }
    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    if ($metadata.scene -ne $Scene -or [int]$metadata.vertex_count -ne ($Dimension * $Dimension) -or
        [int]$metadata.warmup_frames -ne $Warmup -or [int]$metadata.measured_frames -ne $Frames -or
        [int]$metadata.solver_iterations -ne $SolverIterations -or -not [bool]$metadata.rendered -or
        [int]$metadata.render_width -ne $RenderWidth -or [int]$metadata.render_height -ne $RenderHeight -or
        -not [bool]$metadata.final_state_finite) {
        throw "PPM run metadata does not satisfy the formal rendered contract: $Directory"
    }
    $rows = @(Import-Csv -LiteralPath $csvPath)
    if ($rows.Count -ne $Frames) { throw "PPM run has $($rows.Count) measured records, expected ${Frames}: $Directory" }
    $hostValues = @(); $simulation = @()
    foreach ($row in $rows) {
        if ($row.scene -ne $Scene -or [int]$row.vertex_count -ne ($Dimension * $Dimension) -or $row.rendered -ne '1') {
            throw "PPM frame record violates the formal rendered contract: $Directory"
        }
        $hostValue = Convert-ToFiniteDouble $row.frame_host_ms
        $simulationValue = Convert-ToFiniteDouble $row.simulation_gpu_ms
        if ($null -eq $hostValue -or $null -eq $simulationValue) { throw "PPM frame record is non-finite: $Directory" }
        $hostValues += $hostValue; $simulation += $simulationValue
    }
    return [pscustomobject]@{
        scene = $Scene; dimension = $Dimension; repetition = $Repetition; output_dir = $Directory; metadata = $metadata
        host_ms = $hostValues; simulation_gpu_ms = $simulation
    }
}

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$gitCommit = (& git -c "safe.directory=$ProjectRoot" -C $ProjectRoot rev-parse HEAD 2>$null | Select-Object -First 1).Trim()
$peridynoCommit = (& git -c "safe.directory=$PeridynoRoot" -C $PeridynoRoot rev-parse HEAD 2>$null | Select-Object -First 1).Trim()
$hardware = Read-HardwareMetadata
$adapterDir = Join-Path $ProjectRoot 'external_baselines\peridyno_ppm'
$sourceHashes = [ordered]@{
    formal_runner_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()
    base_runner_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $runner).Hash.ToLowerInvariant()
    adapter_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $adapterDir 'GenPD_PPM_OperatingPoint.cpp')).Hash.ToLowerInvariant()
    peridyno_patch_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $adapterDir 'peridyno-ppm-genpd-operating-point.patch')).Hash.ToLowerInvariant()
}
$manifest = [ordered]@{
    protocol_version = 'peridyno-ppm-operating-point-v1'
    label = $RunLabel
    git_commit = $gitCommit
    peridyno_commit = $peridynoCommit
    source_hashes = $sourceHashes
    hardware = $hardware
    baseline = 'Projective Peridynamic Modeling of Hyperelastic Membranes With Contact (2024)'
    comparison_scope = 'same-hardware operating point; not an equal-model or equal-quality ranking'
    measurement = [ordered]@{
        mode = 'rendered-end-to-end'; render_width = $RenderWidth; render_height = $RenderHeight; disable_vsync = $true
        primary_metric = 'frame_host_ms'; timing_scope = 'PPM substeps plus graphics update, draw, and buffer swap; capture and process lifecycle excluded'
    }
    simulation = [ordered]@{ dt_seconds = $Dt; substeps_per_nominal_frame = 33; solver_iterations = $SolverIterations; self_contact = $false; moving_sdf_refresh = $true }
    timing = [ordered]@{ warmup_frames = $Warmup; measured_frames = $Frames; repetitions = $Repetitions }
    scenes = $Scenes
    square_dimensions = $Dimensions
}
[System.IO.File]::WriteAllText((Join-Path $RunRoot 'manifest.json'), ($manifest | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))

$planned = @()
foreach ($scene in $Scenes) {
    foreach ($dimension in $Dimensions) {
        for ($rep = 1; $rep -le $Repetitions; ++$rep) {
            $planned += [pscustomobject]@{ scene = $scene; width = $dimension; height = $dimension; repetition = $rep }
        }
    }
}
$planned | Export-Csv -LiteralPath (Join-Path $RunRoot 'planned_runs.csv') -NoTypeInformation
if ($DryRun) { Write-Host "PPM formal operating-point dry run: $RunRoot"; exit 0 }

$validatedRuns = @(); $buildCompleted = $false
foreach ($case in $planned) {
    $caseDir = Join-Path $RunRoot (Join-Path $case.scene (Join-Path ("{0}x{0}" -f $case.width) ('rep{0:D2}' -f $case.repetition)))
    $capturePath = if ($case.repetition -eq 1) { Join-Path $caseDir 'proof.bmp' } else { '' }
    if ((Test-Path -LiteralPath (Join-Path $caseDir 'frame_profile.csv')) -and -not $Force) {
        $validatedRuns += Read-ValidatedRun -Directory $caseDir -Scene $case.scene -Dimension $case.width -Repetition $case.repetition
        continue
    }
    New-Item -ItemType Directory -Force -Path $caseDir | Out-Null
    $logPath = Join-Path $caseDir 'ppm_stdout.log'
    $runParams = @{
        Scene = $case.scene; Width = $case.width; Height = $case.height; Warmup = $Warmup; Frames = $Frames
        SolverIterations = $SolverIterations; Dt = $Dt; RenderWidth = $RenderWidth; RenderHeight = $RenderHeight
        PeridynoRoot = $PeridynoRoot; BuildDir = $BuildDir; OutputDir = $caseDir
    }
    if ($capturePath -ne '') { $runParams.CaptureOutput = $capturePath }
    if ($buildCompleted) { $runParams.SkipBuild = $true }
    & $runner @runParams *> $logPath
    if ($LASTEXITCODE -ne 0) { Get-Content -LiteralPath $logPath -Tail 80 -ErrorAction SilentlyContinue | Write-Host; throw "PPM formal run failed: $caseDir" }
    $buildCompleted = $true
    $validatedRuns += Read-ValidatedRun -Directory $caseDir -Scene $case.scene -Dimension $case.width -Repetition $case.repetition
}

$runRows = @()
foreach ($run in $validatedRuns) {
    $runRows += [pscustomobject]@{
        scene = $run.scene; cloth_width = $run.dimension; cloth_height = $run.dimension; vertex_count = $run.dimension * $run.dimension
        repetition = $run.repetition; result_dir = $run.output_dir; measured_frames = $run.host_ms.Count
        frame_host_ms_mean = Get-Mean $run.host_ms; simulation_gpu_ms_mean = Get-Mean $run.simulation_gpu_ms
        rendered = 1; final_state_finite = [int][bool]$run.metadata.final_state_finite; moving_sdf_refresh = [int][bool]$run.metadata.moving_sdf_refresh
    }
}
$runRows | Sort-Object scene, vertex_count, repetition | Export-Csv -LiteralPath (Join-Path $RunRoot 'per_repetition_summary.csv') -NoTypeInformation

$summaryRows = @()
foreach ($group in @($validatedRuns | Group-Object { '{0}|{1}' -f $_.scene, $_.dimension })) {
    $runs = @($group.Group); $first = $runs[0]
    $hostMeans = @($runs | ForEach-Object { Get-Mean $_.host_ms })
    $gpuMeans = @($runs | ForEach-Object { Get-Mean $_.simulation_gpu_ms })
    $allHost = @($runs | ForEach-Object { $_.host_ms })
    $allGpu = @($runs | ForEach-Object { $_.simulation_gpu_ms })
    $summaryRows += [pscustomobject]@{
        scene = $first.scene; cloth_width = $first.dimension; cloth_height = $first.dimension; vertex_count = $first.dimension * $first.dimension
        repetitions = $runs.Count; measured_frames_per_repetition = $Frames; measured_frames_total = $allHost.Count
        frame_host_ms_mean = Get-Mean $hostMeans; frame_host_ms_std = Get-SampleStd $hostMeans
        frame_host_ms_p50 = Get-Percentile $allHost 0.50; frame_host_ms_p95 = Get-Percentile $allHost 0.95
        simulation_gpu_ms_mean = Get-Mean $gpuMeans; simulation_gpu_ms_std = Get-SampleStd $gpuMeans
        simulation_gpu_ms_p50 = Get-Percentile $allGpu 0.50; simulation_gpu_ms_p95 = Get-Percentile $allGpu 0.95
        rendered = 1; final_state_finite = 1; moving_sdf_refresh = [int][bool]$first.metadata.moving_sdf_refresh
        git_commit = $gitCommit; peridyno_commit = $peridynoCommit; gpu_name = $hardware.gpu_name; nvidia_driver_version = $hardware.nvidia_driver_version
    }
}
$summaryRows | Sort-Object scene, vertex_count | Export-Csv -LiteralPath (Join-Path $RunRoot 'operating_point_summary.csv') -NoTypeInformation
Write-Host "PPM per-repetition summary: $(Join-Path $RunRoot 'per_repetition_summary.csv')"
Write-Host "PPM formal operating-point summary: $(Join-Path $RunRoot 'operating_point_summary.csv')"
