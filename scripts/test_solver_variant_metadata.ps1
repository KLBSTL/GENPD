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

& $executablePath --benchmark --headless --uncapped --frames 1 --warmup 0 --output-dir $outputDir --solver-variant gpu-gather-fusion-adaptive-ls-persistent 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Solver-variant metadata smoke run failed with exit code $LASTEXITCODE"
}

$metadataPath = Join-Path $outputDir 'run_metadata.json'
if (-not (Test-Path -LiteralPath $metadataPath)) {
    throw "Missing run metadata: $metadataPath"
}

$metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
if ($metadata.solver_variant -ne 'gpu-gather-fusion-adaptive-ls-persistent') {
	throw "Expected solver_variant gpu-gather-fusion-adaptive-ls-persistent, got '$($metadata.solver_variant)'."
}

$roundtripOutputDir = Join-Path $ProjectRoot 'results\test-adaptive-variant-roundtrip-metadata'
$runner = Join-Path $ProjectRoot 'scripts\run_benchmark.ps1'
& $runner `
    -ProjectRoot $ProjectRoot `
    -RunLabel 'test-adaptive-variant-roundtrip-metadata' `
    -OutputDir $roundtripOutputDir `
    -SolverVariant gpu-gather-fusion-adaptive-ls-persistent `
    -ForceCpuStateRoundtrip `
    -Frames 1 `
    -Warmup 0 `
    -NoRender $true `
    -ProcessTimeoutSeconds 120 | Out-Null

$roundtripMetadataPath = Join-Path $roundtripOutputDir 'run_metadata.json'
if (-not (Test-Path -LiteralPath $roundtripMetadataPath)) {
	throw "Missing roundtrip metadata: $roundtripMetadataPath"
}

$roundtripMetadata = Get-Content -LiteralPath $roundtripMetadataPath -Raw | ConvertFrom-Json
if ($roundtripMetadata.solver_controls.force_cpu_state_roundtrip -ne '1') {
	throw 'Adaptive persistent variant did not accept ForceCpuStateRoundtrip.'
}

Write-Output 'Solver variant metadata check passed.'
