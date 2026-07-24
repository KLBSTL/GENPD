param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $ProjectRoot 'scripts\run_gather_regression.ps1'
if (-not (Test-Path -LiteralPath $runner)) { throw "Missing gather regression runner: $runner" }
$root = Join-Path $ProjectRoot (Join-Path 'results' ('test-gather-regression-contract-' + $PID))
& $runner -ProjectRoot $ProjectRoot -OutputDir $root -IterationBudgets 1 -Frames 1 -Warmup 0 -DryRun
$manifest = Get-Content -LiteralPath (Join-Path $root 'manifest.json') -Raw | ConvertFrom-Json
if (-not [bool]$manifest.measurement.rendered -or $manifest.variants.Count -ne 4 -or $manifest.cloth_dimension -ne 386) {
    throw 'Gather regression manifest does not encode the rendered 386^2 four-variant regression.'
}
Write-Host "Gather regression contract passed: $root"
