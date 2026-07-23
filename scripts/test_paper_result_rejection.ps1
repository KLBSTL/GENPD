param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$stamp = 'test-paper-rejection-' + (Get-Random -Minimum 1000 -Maximum 99999)
$runRoot = Join-Path $ProjectRoot (Join-Path 'results' $stamp)
$runner = Join-Path $ProjectRoot 'scripts\run_paper_experiments.ps1'
$summary = Join-Path $ProjectRoot 'scripts\summarize_paper_experiments.ps1'
$figures = Join-Path $ProjectRoot 'scripts\generate_paper_figures.py'

& $runner -ProjectRoot $ProjectRoot -RunRoot $runRoot -Stage manifest -DryRun
if (-not (Test-Path -LiteralPath (Join-Path $runRoot 'manifest.json'))) {
    throw 'Manifest setup failed.'
}

$summaryFailed = $false
try {
    & $summary -ProjectRoot $ProjectRoot -RunRoot $runRoot
}
catch {
    $summaryFailed = $true
}
if (-not $summaryFailed) {
    throw 'Summary accepted an incomplete formal experiment directory.'
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { throw 'Python is required for the figure contract test.' }
$figureFailed = $false
try {
    & $python.Source $figures --run-root $runRoot --paper-figure-dir (Join-Path $runRoot 'paper-figures') *> $null
    $figureFailed = $LASTEXITCODE -ne 0
}
catch {
    $figureFailed = $true
}
if (-not $figureFailed) {
    throw 'Figure script accepted incomplete formal evidence.'
}

Write-Host "Paper result rejection contract passed: $runRoot"
