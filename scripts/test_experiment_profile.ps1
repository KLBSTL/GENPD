param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Configuration = 'Release',
    [string]$Platform = 'x64'
)

$ErrorActionPreference = 'Stop'

$executablePath = Join-Path $ProjectRoot ("x64\\{0}\\GenPD.exe" -f $Configuration)
$outputDir = Join-Path $ProjectRoot 'results\test-experiment-profile'
if (-not (Test-Path -LiteralPath $executablePath)) {
    throw "GenPD executable was not found: $executablePath"
}

& $executablePath --benchmark --headless --uncapped --frames 1 --warmup 0 --output-dir $outputDir --solver-variant gpu-gather-no-fusion 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Experiment-profile smoke run failed with exit code $LASTEXITCODE"
}

$profilePath = Join-Path $outputDir 'frame_profile_experiment.csv'
if (-not (Test-Path -LiteralPath $profilePath)) {
    throw "Missing experiment profile: $profilePath"
}

$row = Import-Csv -LiteralPath $profilePath | Select-Object -Last 1
if ($row.solver_variant -ne 'gpu-gather-no-fusion') {
    throw "Expected gpu-gather-no-fusion, got '$($row.solver_variant)'."
}

@('gradient_dispatches', 'stats_dispatches', 'reduction_dispatches', 'host_readbacks', 'tracked_buffer_bytes') | ForEach-Object {
    if ($null -eq $row.$_) {
        throw "Experiment profile is missing column $_"
    }
}

if ([int]$row.gradient_dispatches -lt 1 -or [int]$row.stats_dispatches -lt 1) {
    throw 'No-fusion variant did not report both gradient and stats dispatches.'
}

Write-Output 'Experiment profile contract passed.'
