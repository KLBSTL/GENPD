param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Configuration = 'Release',
    [string]$Platform = 'x64'
)

$ErrorActionPreference = 'Stop'

$runReferenceScript = Join-Path $ProjectRoot 'scripts\run_reference.ps1'
$runLabel = "test-reference-" + $PID
$outputDir = Join-Path $ProjectRoot (Join-Path 'results' $runLabel)

& $runReferenceScript -ProjectRoot $ProjectRoot -RunLabel $runLabel -OutputDir $outputDir `
    -Frames 1 -Warmup 0 -ReferenceIterations 2 -CheckpointStride 1 -ExtraArgs '--headless' 2>&1 | Out-Null
if (-not $?) {
    throw 'Reference-run wrapper failed.'
}

$checkpointPath = Join-Path $outputDir 'reference_checkpoints\reference_state_000000.bin'
if (-not (Test-Path -LiteralPath $checkpointPath)) {
    throw "Reference wrapper did not produce a checkpoint: $checkpointPath"
}

$metadata = Get-Content -LiteralPath (Join-Path $outputDir 'run_metadata.json') -Raw | ConvertFrom-Json
if ($metadata.solver_variant -ne 'cpu-ncg' -or $metadata.quality.iterations_per_frame -ne '2') {
    throw 'Reference wrapper metadata does not identify the requested CPU reference run.'
}

Write-Output "Reference-run wrapper passed: $outputDir"
