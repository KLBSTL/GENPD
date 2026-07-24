param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Configuration = 'Release',
    [string]$Platform = 'x64'
)

$ErrorActionPreference = 'Stop'

$executablePath = Join-Path $ProjectRoot ("x64\\{0}\\GenPD.exe" -f $Configuration)
$outputDir = Join-Path $ProjectRoot 'results\test-solver-variant-metadata'
if (-not (Test-Path -LiteralPath $executablePath)) {
    throw "GenPD executable was not found: $executablePath"
}

& $executablePath --benchmark --headless --uncapped --frames 1 --warmup 0 --output-dir $outputDir --solver-variant gpu-gather-fusion 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Solver-variant metadata smoke run failed with exit code $LASTEXITCODE"
}

$metadataPath = Join-Path $outputDir 'run_metadata.json'
if (-not (Test-Path -LiteralPath $metadataPath)) {
    throw "Missing run metadata: $metadataPath"
}

$metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
if ($metadata.solver_variant -ne 'gpu-gather-fusion') {
    throw "Expected solver_variant gpu-gather-fusion, got '$($metadata.solver_variant)'."
}

Write-Output 'Solver variant metadata check passed.'
