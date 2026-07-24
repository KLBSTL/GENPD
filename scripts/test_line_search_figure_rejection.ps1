param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $ProjectRoot 'scripts\run_line_search_sweep.ps1'
$figure = Join-Path $ProjectRoot 'scripts\generate_line_search_figures.py'
$root = Join-Path $ProjectRoot (Join-Path 'results' ('test-line-search-figure-rejection-' + $PID))
& $runner -ProjectRoot $ProjectRoot -OutputDir $root -DryRun -KValues 1,2 -Betas 0.5 -RestartModes none
if (-not (Test-Path -LiteralPath (Join-Path $root 'manifest.json'))) { throw 'Line-search figure rejection fixture was not created.' }
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { throw 'Python is required for the line-search figure contract.' }
$rejected = $false
try {
    & $python.Source $figure --run-root $root --paper-figure-dir (Join-Path $root 'paper-figures') *> $null
    $rejected = $LASTEXITCODE -ne 0
}
catch {
    $rejected = $true
}
if (-not $rejected) { throw 'Line-search figure script accepted incomplete evidence.' }
$global:LASTEXITCODE = 0
Write-Host "Line-search figure rejection contract passed: $root"
