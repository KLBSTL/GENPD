param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $ProjectRoot 'scripts\run_vertex_owned_microstudy.ps1'
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Missing vertex-owned microstudy runner: $runner"
}

$root = Join-Path $ProjectRoot (Join-Path 'results' ('test-vertex-owned-paper-' + $PID))
& $runner -ProjectRoot $ProjectRoot -RunRoot $root -DryRun
if ($LASTEXITCODE -ne 0) { throw 'Vertex-owned microstudy dry-run failed.' }

$manifestPath = Join-Path $root 'manifest.json'
$planPath = Join-Path $root 'planned_cases.csv'
if (-not (Test-Path -LiteralPath $manifestPath) -or -not (Test-Path -LiteralPath $planPath)) {
    throw 'Vertex-owned microstudy dry-run did not write its manifest and plan.'
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.protocol_version -ne 'vertex-owned-microstudy-v1' -or $manifest.measurement.mode -ne 'rendered-end-to-end' `
    -or $manifest.timing.frames -ne 300 -or $manifest.timing.warmup -ne 30 -or $manifest.timing.repetitions -ne 3 `
    -or @($manifest.variants).Count -ne 3 -or @($manifest.scenes).Count -ne 2 -or @($manifest.cloth_dimensions).Count -ne 2) {
    throw 'Vertex-owned microstudy protocol is incomplete.'
}
$cases = @(Import-Csv -LiteralPath $planPath)
if ($cases.Count -ne 4 -or @($cases | Where-Object { [int]$_.iterations_per_frame -lt 1 -or [string]::IsNullOrWhiteSpace($_.quality_reference_dir) }).Count -ne 0) {
    throw 'Vertex-owned microstudy plan does not contain four calibrated quality-reference cases.'
}
Write-Host "Vertex-owned microstudy paper contract passed: $root"
