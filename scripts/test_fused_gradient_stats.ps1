param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Configuration = 'Release',
    [string]$Platform = 'x64'
)

$ErrorActionPreference = 'Stop'

$executablePath = Join-Path $ProjectRoot ("x64\\{0}\\GenPD.exe" -f $Configuration)
$outputDir = Join-Path $ProjectRoot 'results\test-fused-gradient-stats'
if (-not (Test-Path -LiteralPath $executablePath)) {
    throw "GenPD executable was not found: $executablePath"
}

& $executablePath --benchmark --headless --uncapped --frames 1 --warmup 0 --output-dir $outputDir --solver-variant gpu-gather-fusion 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Fused-gradient smoke run failed with exit code $LASTEXITCODE"
}

$row = Import-Csv -LiteralPath (Join-Path $outputDir 'frame_profile.csv') | Select-Object -Last 1
if ([double]$row.gradient_norm -le 1e-12) {
    throw "Fused gradient statistics were not initialized (gradient_norm=$($row.gradient_norm))."
}

Write-Output 'Fused gradient statistics check passed.'
