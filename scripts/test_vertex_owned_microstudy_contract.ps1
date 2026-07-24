param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $ProjectRoot 'scripts\run_benchmark.ps1'
if (-not (Test-Path -LiteralPath $runner)) { throw "Missing benchmark runner: $runner" }
$root = Join-Path $ProjectRoot (Join-Path 'results' ('test-vertex-owned-microstudy-' + $PID))
$variants = @('gpu-edge-scatter', 'gpu-gather-no-fusion', 'gpu-gather-fusion')
$rowsByVariant = @{}
foreach ($variant in $variants) {
    $outputDir = Join-Path $root $variant
    & $runner -ProjectRoot $ProjectRoot -RunLabel "test-vertex-owned-$variant" -OutputDir $outputDir `
        -SolverVariant $variant -IterationsPerFrame 1 -Frames 1 -Warmup 0 -NoRender $true -Uncapped $true `
        -ExtraArgs @('--cloth-dimension', '128') | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Vertex-owned contract run failed for $variant." }
    $experimentPath = Join-Path $outputDir 'frame_profile_experiment.csv'
    if (-not (Test-Path -LiteralPath $experimentPath)) { throw "Missing experiment profile for $variant" }
    $row = @(Import-Csv -LiteralPath $experimentPath)
    if ($row.Count -ne 1) { throw "Unexpected profile row count for $variant" }
    foreach ($field in @('constraint_spring_count', 'constraint_attachment_count')) {
        if (-not ($row[0].PSObject.Properties.Name -contains $field)) { throw "Missing $field for $variant" }
    }
    $rowsByVariant[$variant] = $row[0]
}

$edge = $rowsByVariant['gpu-edge-scatter']
$gather = $rowsByVariant['gpu-gather-no-fusion']
$fusion = $rowsByVariant['gpu-gather-fusion']
if ([int]$edge.constraint_spring_count -le 0 -or [int]$edge.constraint_attachment_count -le 0) {
    throw 'Constraint counts were not recorded.'
}
if ([int]$edge.gradient_dispatches + [int]$edge.stats_dispatches -ne 3 `
    -or [int]$gather.gradient_dispatches + [int]$gather.stats_dispatches -ne 2 `
    -or [int]$fusion.gradient_dispatches + [int]$fusion.stats_dispatches -ne 1) {
    throw 'Gradient/stats dispatch structure does not match edge scatter, gather, and fused gather.'
}
Write-Host "Vertex-owned microstudy contract passed: $root"
