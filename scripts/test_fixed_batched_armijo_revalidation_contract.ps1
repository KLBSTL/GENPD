$ErrorActionPreference = 'Stop'
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir '..'))
$runner = Join-Path $scriptDir 'run_fixed_batched_armijo_revalidation.ps1'
$analysis = Join-Path $scriptDir 'analyze_fixed_batched_armijo_revalidation.py'
foreach ($path in @($runner, $analysis, (Join-Path $projectRoot 'results\paper-20260724-r2\selected_budgets.csv'))) { if (-not (Test-Path -LiteralPath $path)) { throw "Missing fixed-batched revalidation artifact: $path" } }
$root = [System.IO.Path]::GetFullPath((Join-Path $projectRoot 'results\diagnostic-fixed-batched-armijo-contract'))
& $runner -ProjectRoot $projectRoot -RunLabel paper-fixed-batched-armijo-contract -RunRoot $root -DryRun
if ($LASTEXITCODE -ne 0) { throw 'Fixed-batched Armijo revalidation dry run failed.' }
$manifest = Get-Content -LiteralPath (Join-Path $root 'manifest.json') -Raw | ConvertFrom-Json
if ($manifest.protocol_version -ne 'fixed-batched-armijo-revalidation-v1' -or $manifest.cases.Count -ne 4 -or $manifest.methods.Count -ne 2 -or $manifest.timing.repetitions -ne 3) {
    throw 'Fixed-batched Armijo manifest does not encode the expected formal protocol.'
}
if (-not $root.StartsWith((Join-Path $projectRoot 'results') + [System.IO.Path]::DirectorySeparatorChar) -or (Split-Path -Leaf $root) -ne 'diagnostic-fixed-batched-armijo-contract') {
    throw "Unexpected contract cleanup target: $root"
}
Remove-Item -LiteralPath $root -Recurse -Force
Write-Host 'Fixed-batched Armijo revalidation contract passed.'
