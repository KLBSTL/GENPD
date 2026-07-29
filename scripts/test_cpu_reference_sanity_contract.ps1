param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $ProjectRoot 'scripts\run_cpu_reference_sanity.ps1'
$analysis = Join-Path $ProjectRoot 'scripts\analyze_cpu_reference_sanity.py'
foreach ($path in @($runner, $analysis)) { if (-not (Test-Path -LiteralPath $path)) { throw "Missing CPU reference sanity artifact: $path" } }
$runnerText = Get-Content -LiteralPath $runner -Raw
foreach ($required in @("@(100, 200, 400)", "[int]`$Frames = 30", "[int]`$Warmup = 5", "'cpu-ncg'", "'rendered-checkpoint-export'", "1.0e-3")) {
    if ($runnerText -notmatch [regex]::Escape($required)) { throw "CPU reference sanity runner is missing $required" }
}
$root = Join-Path $ProjectRoot (Join-Path 'results' ('test-cpu-reference-sanity-contract-' + $PID))
& $runner -ProjectRoot $ProjectRoot -RunRoot $root -RunLabel ('test-cpu-reference-sanity-contract-' + $PID) -DryRun
if ($LASTEXITCODE -ne 0) { throw 'CPU reference sanity dry run failed.' }
$manifest = Get-Content -LiteralPath (Join-Path $root 'manifest.json') -Raw | ConvertFrom-Json
if ($manifest.protocol_version -ne 'cpu-reference-sanity-v1' -or $manifest.solver_variant -ne 'cpu-ncg' `
    -or @($manifest.iterations).Count -ne 3 -or $manifest.iterations[0] -ne 100 -or $manifest.iterations[2] -ne 400 `
    -or $manifest.measurement.mode -ne 'rendered-checkpoint-export' -or $manifest.decision.p95_position_rel_l2 -ne 1e-3) {
    throw 'CPU reference sanity manifest does not encode the 100/200/400 protocol.'
}
Write-Host "CPU reference sanity contract passed: $root"
