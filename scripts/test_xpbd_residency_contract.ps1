param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $ProjectRoot 'scripts\run_xpbd_residency_study.ps1'
$analysis = Join-Path $ProjectRoot 'scripts\analyze_xpbd_residency.py'
$benchmark = Join-Path $ProjectRoot 'scripts\run_benchmark.ps1'
foreach ($path in @($runner, $analysis, $benchmark)) { if (-not (Test-Path -LiteralPath $path)) { throw "Missing XPBD residency artifact: $path" } }

$runnerText = Get-Content -LiteralPath $runner -Raw
foreach ($required in @("'gpu-xpbd-jacobi'", "[int]`$TimingFrames = 300", "[int]`$TimingWarmup = 30", "[int]`$TimingRepetitions = 3", "'rendered-end-to-end'", "forced-cpu-state-roundtrip", "reference_checkpoints")) {
    if ($runnerText -notmatch [regex]::Escape($required)) { throw "XPBD residency runner is missing $required" }
}
$benchmarkText = Get-Content -LiteralPath $benchmark -Raw
if ($benchmarkText -notmatch "'gpu-xpbd-jacobi'" -or $benchmarkText -notmatch 'GPU-resident NCG and XPBD variants') {
    throw 'The benchmark wrapper does not permit the XPBD forced-roundtrip counterfactual.'
}

$root = Join-Path $ProjectRoot (Join-Path 'results' ('test-xpbd-residency-contract-' + $PID))
& $runner -ProjectRoot $ProjectRoot -RunRoot $root -RunLabel ('test-xpbd-residency-contract-' + $PID) -DryRun
if ($LASTEXITCODE -ne 0) { throw 'XPBD residency dry run failed.' }
$manifest = Get-Content -LiteralPath (Join-Path $root 'manifest.json') -Raw | ConvertFrom-Json
if ($manifest.protocol_version -ne 'xpbd-residency-v1' -or $manifest.solver_variant -ne 'gpu-xpbd-jacobi' `
    -or $manifest.measurement.mode -ne 'rendered-end-to-end' -or $manifest.timing.repetitions -ne 3 `
    -or @($manifest.conditions).Count -ne 2 -or $manifest.acceptance.position_checkpoint_rel_l2_p95 -ne 1e-6) {
    throw 'XPBD residency manifest does not encode the controlled protocol.'
}
Write-Host "XPBD residency contract passed: $root"
