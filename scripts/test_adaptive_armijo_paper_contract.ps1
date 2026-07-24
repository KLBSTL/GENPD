param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $ProjectRoot 'scripts\run_adaptive_armijo_paper_study.ps1'
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Missing adaptive Armijo paper runner: $runner"
}

$root = Join-Path $ProjectRoot (Join-Path 'results' ('test-adaptive-armijo-paper-' + $PID))
& $runner -ProjectRoot $ProjectRoot -RunRoot $root -DryRun
if ($LASTEXITCODE -ne 0) { throw 'Adaptive Armijo paper dry-run failed.' }

$manifestPath = Join-Path $root 'manifest.json'
$corePath = Join-Path $root 'planned_core_cases.csv'
$sensitivityPath = Join-Path $root 'planned_sensitivity_cases.csv'
foreach ($path in @($manifestPath, $corePath, $sensitivityPath)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Adaptive Armijo dry-run did not write $path" }
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.protocol_version -ne 'adaptive-armijo-paper-v1' -or $manifest.measurement.mode -ne 'rendered-end-to-end' `
    -or $manifest.timing.frames -ne 300 -or $manifest.timing.warmup -ne 30 -or $manifest.timing.repetitions -ne 3 `
    -or @($manifest.core_methods).Count -ne 5 -or @($manifest.scenes).Count -ne 2 -or @($manifest.cloth_dimensions).Count -ne 3 `
    -or @($manifest.sensitivity.k_values).Count -ne 3 -or @($manifest.sensitivity.betas).Count -ne 3) {
    throw 'Adaptive Armijo formal protocol is incomplete.'
}
if (@(Import-Csv -LiteralPath $corePath).Count -ne 6 -or @(Import-Csv -LiteralPath $sensitivityPath).Count -ne 54) {
    throw 'Adaptive Armijo plan dimensions are incomplete.'
}
Write-Host "Adaptive Armijo paper contract passed: $root"
