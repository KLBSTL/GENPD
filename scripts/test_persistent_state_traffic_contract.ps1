param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $ProjectRoot 'scripts\run_benchmark.ps1'
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Missing benchmark runner: $runner"
}

$root = Join-Path $ProjectRoot (Join-Path 'results' ('test-persistent-state-traffic-' + $PID))
foreach ($condition in @('resident', 'roundtrip')) {
    $outputDir = Join-Path $root $condition
    $params = @{
        ProjectRoot = $ProjectRoot
        RunLabel = "test-persistent-state-traffic-$condition"
        OutputDir = $outputDir
        SolverVariant = 'gpu-gather-fusion-batched-ls-persistent'
        IterationsPerFrame = 1
        Frames = 2
        Warmup = 0
        NoRender = $true
        Uncapped = $true
        ExtraArgs = @('--cloth-dimension', '128')
    }
    if ($condition -eq 'roundtrip') { $params.ForceCpuStateRoundtrip = $true }
    & $runner @params | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Persistent traffic benchmark failed for $condition." }
}

function Read-ExperimentRows([string]$Directory) {
    $path = Join-Path $Directory 'frame_profile_experiment.csv'
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing experiment profile: $path" }
    $rows = @(Import-Csv -LiteralPath $path)
    if ($rows.Count -ne 2) { throw "Expected two profile rows in $path" }
    foreach ($field in @('state_h2d_bytes', 'state_d2h_bytes', 'state_upload_calls', 'state_readback_calls')) {
        if (-not ($rows[0].PSObject.Properties.Name -contains $field)) { throw "Missing state traffic field '$field' in $path" }
    }
    return $rows
}

$resident = Read-ExperimentRows (Join-Path $root 'resident')
$roundtrip = Read-ExperimentRows (Join-Path $root 'roundtrip')
if ([double]$resident[1].state_d2h_bytes -ne 0 -or [double]$resident[1].state_h2d_bytes -ne 0) {
    throw 'Resident-state second frame unexpectedly performed a full state transfer.'
}
if ([double]$roundtrip[1].state_d2h_bytes -le 0 -or [double]$roundtrip[1].state_h2d_bytes -le 0 `
    -or [int]$roundtrip[1].state_readback_calls -lt 2 -or [int]$roundtrip[1].state_upload_calls -lt 2) {
    throw 'Forced roundtrip second frame did not report position/velocity state traffic.'
}

Write-Host "Persistent-state traffic contract passed: $root"
