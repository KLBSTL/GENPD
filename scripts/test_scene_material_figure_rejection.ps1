param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $ProjectRoot 'scripts\run_scene_material_matrix.ps1'
$figure = Join-Path $ProjectRoot 'scripts\generate_scene_material_figures.py'
$root = Join-Path $ProjectRoot (Join-Path 'results' ('test-scene-material-figure-rejection-' + $PID))
& $runner -ProjectRoot $ProjectRoot -OutputDir $root -Stage manifest -DryRun
if (-not (Test-Path -LiteralPath (Join-Path $root 'manifest.json'))) { throw 'Scene/material figure rejection fixture was not created.' }
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { throw 'Python is required for the scene/material figure contract.' }
$rejected = $false
try {
    & $python.Source $figure --run-root $root --paper-figure-dir (Join-Path $root 'paper-figures') *> $null
    $rejected = $LASTEXITCODE -ne 0
}
catch {
    $rejected = $true
}
if (-not $rejected) { throw 'Scene/material figure script accepted incomplete evidence.' }
$global:LASTEXITCODE = 0
Write-Host "Scene/material figure rejection contract passed: $root"
