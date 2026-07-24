param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Configuration = 'Release',
    [string]$Platform = 'x64'
)

$ErrorActionPreference = 'Stop'

$executablePath = Join-Path $ProjectRoot ("x64\\{0}\\GenPD.exe" -f $Configuration)
if (-not (Test-Path -LiteralPath $executablePath)) {
    throw "GenPD executable was not found: $executablePath"
}

$testRoot = Join-Path $ProjectRoot ("results\\test-phase4-runtime-" + $PID)
$configPath = Join-Path $ProjectRoot 'config\config.txt'
$configHashBefore = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash

& $executablePath --benchmark --headless --uncapped --frames 2 --warmup 0 `
    --project-root $ProjectRoot --output-dir $testRoot --run-label 'test-phase4-runtime' `
    --solver-variant gpu-gather-fusion-batched-ls-persistent --quality-metrics `
    --timestep 0.02 --stretch-stiffness 50 --bending-stiffness 12 --cloth-dimension 16 `
    --scene 'scenes\moving_sphere_cloth.xml' 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Phase 4 runtime smoke failed with exit code $LASTEXITCODE"
}

$qualityPath = Join-Path $testRoot 'quality_metrics.csv'
$metadataPath = Join-Path $testRoot 'run_metadata.json'
if (-not (Test-Path -LiteralPath $qualityPath) -or -not (Test-Path -LiteralPath $metadataPath)) {
    throw "Phase 4 run did not produce quality/metadata outputs: $testRoot"
}

$qualityRows = @(Import-Csv -LiteralPath $qualityPath)
if ($qualityRows.Count -ne 2 -or @($qualityRows | Where-Object { $_.finite -ne '1' }).Count -ne 0) {
    throw 'Phase 4 quality metrics are missing, incomplete, or non-finite.'
}

$metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
if ($metadata.quality.metrics_enabled -ne '1' -or $metadata.experiment_overrides.cloth_dimension -ne '16') {
    throw 'Phase 4 metadata did not record quality mode and mesh override.'
}
if ($metadata.experiment_overrides.scene -notmatch 'moving_sphere_cloth\.xml$') {
    throw 'Phase 4 metadata did not record the moving obstacle scene.'
}

$configHashAfter = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash
if ($configHashBefore -ne $configHashAfter) {
    throw 'Phase 4 CLI run modified config/config.txt.'
}

Write-Output "Phase 4 runtime contract passed: $testRoot"
