[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$RunLabel = 'paper-20260729-fixed-batched-armijo-r2',
    [string]$RunRoot = '',
    [string]$CalibrationRoot = '',
    [int[]]$ClothDimensions = @(256, 386),
    [int]$TimingFrames = 300,
    [int]$TimingWarmup = 30,
    [int]$TimingRepetitions = 3,
    [int]$TraceFrames = 120,
    [int]$TraceWarmup = 20,
    [int]$RenderWidth = 1600,
    [int]$RenderHeight = 900,
    [int]$ProcessTimeoutSeconds = 1800,
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
if ($RunLabel -notmatch '^paper-' -or $ClothDimensions.Count -eq 0 -or ($ClothDimensions | Where-Object { $_ -lt 2 }).Count -ne 0 `
    -or $TimingFrames -ne 300 -or $TimingWarmup -ne 30 -or $TimingRepetitions -ne 3 -or $TraceFrames -lt 1 -or $TraceWarmup -lt 0) {
    throw 'Fixed-batched Armijo revalidation requires a paper label, 30 warm-up + 300 measured frames, three repetitions, and valid traces.'
}

$sweepScript = Join-Path $scriptDir 'run_line_search_sweep.ps1'
$analysisScript = Join-Path $scriptDir 'analyze_fixed_batched_armijo_revalidation.py'
$budgetPath = Join-Path $CalibrationRoot 'selected_budgets.csv'
foreach ($path in @($sweepScript, $analysisScript, $budgetPath)) { if (-not (Test-Path -LiteralPath $path)) { throw "Missing Armijo revalidation input: $path" } }

function Resolve-ReferenceCheckpointDirectory {
    param([string]$Path)
    foreach ($candidate in @((Join-Path $Path 'reference_checkpoints'), $Path)) {
        if (Test-Path -LiteralPath (Join-Path $candidate 'reference_state_000000.bin')) { return [System.IO.Path]::GetFullPath($candidate) }
    }
    throw "Reference checkpoints were not found below: $Path"
}

$sceneMap = [ordered]@{ hanging = 'scenes\test_scene.xml'; 'moving-sphere' = 'scenes\moving_sphere_cloth.xml' }
$budgetRows = @(Import-Csv -LiteralPath $budgetPath)
$cases = @()
foreach ($sceneId in $sceneMap.Keys) {
    foreach ($dimension in @($ClothDimensions | Sort-Object -Unique)) {
        $matches = @($budgetRows | Where-Object {
            $_.scene_id -eq $sceneId -and [int]$_.cloth_dimension -eq $dimension `
                -and $_.solver_variant -eq 'gpu-gather-fusion-batched-ls-persistent' -and $_.qualified -eq '1'
        })
        if ($matches.Count -ne 1) { throw "Expected one qualified persistent budget for $sceneId d$dimension." }
        $cases += [pscustomobject]@{
            case_id = "$sceneId-d$dimension"; scene_id = $sceneId; scene_path = $sceneMap[$sceneId]; cloth_dimension = $dimension
            iterations_per_frame = [int]$matches[0].iterations_per_frame; reference_dir = Resolve-ReferenceCheckpointDirectory -Path $matches[0].reference_dir
        }
    }
}

function Read-HardwareMetadata {
    $gpu = ''; $driver = ''
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
        $driver = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
    }
    if ([string]::IsNullOrWhiteSpace($gpu) -or [string]::IsNullOrWhiteSpace($driver)) { throw 'nvidia-smi did not provide GPU and driver metadata.' }
    return [ordered]@{ gpu_name = $gpu; nvidia_driver_version = $driver }
}

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
$commit = (& git -c "safe.directory=$ProjectRoot" -C $ProjectRoot rev-parse HEAD 2>$null | Select-Object -First 1).Trim()
$manifest = [ordered]@{
    protocol_version = 'fixed-batched-armijo-revalidation-v1'
    label = $RunLabel
    git_commit = $commit
    calibration_root = $CalibrationRoot
    hardware = Read-HardwareMetadata
    measurement = [ordered]@{ mode = 'rendered-end-to-end'; render_width = $RenderWidth; render_height = $RenderHeight; sync_gpu = $true; disable_vsync = $true; quality_readback_during_timing = $false }
    timing = [ordered]@{ frames = $TimingFrames; warmup = $TimingWarmup; repetitions = $TimingRepetitions }
    trace = [ordered]@{ frames = $TraceFrames; warmup = $TraceWarmup; rendered = $true; quality_reference = 'archived-cpu-ncg-100-iteration-checkpoints'; p95_position_rel_l2 = 1.0e-3 }
    methods = @(
        [ordered]@{ id = 'serial'; schedule = 'serial'; solver_variant = 'gpu-gather-fusion-serial-ls-persistent'; batched_ls_k = 1; armijo_beta = 0.5; ncg_restart_mode = 'non-descent' },
        [ordered]@{ id = 'fixed-k8'; schedule = 'fixed'; solver_variant = 'gpu-gather-fusion-batched-ls-persistent'; batched_ls_k = 8; armijo_beta = 0.5; ncg_restart_mode = 'non-descent' }
    )
    cases = $cases
    acceptance = [ordered]@{ require_zero_invalid_frames = $true; require_zero_armijo_failures = $true; p95_position_rel_l2 = 1.0e-3 }
}
[System.IO.File]::WriteAllText((Join-Path $RunRoot 'manifest.json'), ($manifest | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))
$cases | Export-Csv -LiteralPath (Join-Path $RunRoot 'planned_cases.csv') -NoTypeInformation
if ($DryRun) { Write-Host "Fixed-batched Armijo revalidation dry run: $RunRoot"; exit 0 }

foreach ($case in $cases) {
    $outputDir = Join-Path $RunRoot (Join-Path 'cases' $case.case_id)
    Write-Host "Fixed-batched Armijo revalidation: $($case.case_id)"
    & $sweepScript -ProjectRoot $ProjectRoot -RunLabel "$RunLabel-$($case.case_id)" -OutputDir $outputDir `
        -Scene $case.scene_path -ClothDimension $case.cloth_dimension -IterationsPerFrame $case.iterations_per_frame `
        -TimingFrames $TimingFrames -TimingWarmup $TimingWarmup -TimingRepetitions $TimingRepetitions `
        -TraceFrames $TraceFrames -TraceWarmup $TraceWarmup -ProcessTimeoutSeconds $ProcessTimeoutSeconds -InterRunDelayMilliseconds $InterRunDelayMilliseconds `
        -KValues @(8) -Betas @(0.5) -RestartModes @('non-descent') -Schedules @('serial', 'fixed') -AdaptiveHistoryModes @('none') `
        -QualityReferenceDir $case.reference_dir -RequireReference -Force:$Force
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $outputDir 'line_search_summary.csv'))) {
        throw "Armijo revalidation failed: $($case.case_id)"
    }
}

if ($PythonExe -eq '') { $PythonExe = if (Test-Path -LiteralPath 'E:\Anaconda\envs\DL\python.exe') { 'E:\Anaconda\envs\DL\python.exe' } else { 'python' } }
& $PythonExe $analysisScript --run-root $RunRoot
if ($LASTEXITCODE -ne 0) { throw "Armijo revalidation analysis failed with exit code $LASTEXITCODE." }
Write-Host "Fixed-batched Armijo revalidation complete: $RunRoot"
