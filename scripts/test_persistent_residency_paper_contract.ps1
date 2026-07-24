param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $ProjectRoot 'scripts\run_persistent_residency_paper_study.ps1'
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Missing persistent residency paper runner: $runner"
}

$root = Join-Path $ProjectRoot (Join-Path 'results' ('test-persistent-residency-paper-' + $PID))
& $runner -ProjectRoot $ProjectRoot -RunRoot $root -DryRun
if ($LASTEXITCODE -ne 0) { throw 'Persistent residency paper dry-run failed.' }

$manifestPath = Join-Path $root 'manifest.json'
$planPath = Join-Path $root 'planned_cases.csv'
if (-not (Test-Path -LiteralPath $manifestPath) -or -not (Test-Path -LiteralPath $planPath)) {
    throw 'Persistent residency paper dry-run did not write manifest and planned cases.'
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.protocol_version -ne 'persistent-residency-paper-v1' -or $manifest.measurement.mode -ne 'rendered-end-to-end' `
    -or $manifest.timing.frames -ne 300 -or $manifest.timing.warmup -ne 30 -or $manifest.timing.repetitions -ne 3 `
    -or @($manifest.scenes).Count -ne 2 -or @($manifest.cloth_dimensions).Count -ne 2 `
    -or @($manifest.conditions).Count -ne 2) {
    throw 'Persistent residency paper protocol is incomplete.'
}
$cases = @(Import-Csv -LiteralPath $planPath)
if ($cases.Count -ne 4 -or @($cases | Where-Object { $_.iterations_per_frame -lt 1 -or [string]::IsNullOrWhiteSpace($_.quality_reference_dir) }).Count -ne 0) {
    throw 'Persistent residency paper plan does not contain four calibrated quality-reference cases.'
}
Write-Host "Persistent residency paper contract passed: $root"
