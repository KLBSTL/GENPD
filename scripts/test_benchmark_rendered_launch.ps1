param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$runner = Join-Path $ProjectRoot 'scripts\run_benchmark.ps1'
$outputDir = Join-Path $ProjectRoot (Join-Path 'results' ('test-rendered-launch-' + $PID))

& $runner -ProjectRoot $ProjectRoot -RunLabel ('test-rendered-launch-' + $PID) `
    -Frames 1 -Warmup 0 -SolverVariant cpu-ncg -IterationsPerFrame 1 -OutputDir $outputDir `
    -NoRender:$false -Uncapped:$true -SyncGpu -DisableVsync -RenderWidth 1600 -RenderHeight 900 `
    -ProcessTimeoutSeconds 60 -ExtraArgs @('--cloth-dimension', '128', '--scene', 'scenes\test_scene.xml')

foreach ($name in @('frame_profile.csv', 'frame_presentation.csv', 'run_metadata.json')) {
    if (-not (Test-Path -LiteralPath (Join-Path $outputDir $name))) {
        throw "Rendered-launch regression is missing $name."
    }
}
$metadata = Get-Content -LiteralPath (Join-Path $outputDir 'run_metadata.json') -Raw | ConvertFrom-Json
$presentation = @(Import-Csv -LiteralPath (Join-Path $outputDir 'frame_presentation.csv'))
if ([bool]$metadata.benchmark.no_render -or $presentation.Count -ne 1 -or $presentation[0].rendered -ne '1' `
    -or [int]$presentation[0].screen_width -ne 1600 -or [int]$presentation[0].screen_height -ne 900) {
    throw 'Rendered-launch regression did not exercise the expected 1600x900 presentation path.'
}

foreach ($path in @(
    (Join-Path $ProjectRoot 'scripts\run_benchmark.ps1'),
    (Join-Path $ProjectRoot 'scripts\run_nsys.ps1'),
    (Join-Path $ProjectRoot 'scripts\run_ncu.ps1')
)) {
    if (-not (Select-String -LiteralPath $path -SimpleMatch 'Push-Location $ProjectRoot' -Quiet)) {
        throw "Runtime path contract missing from $path."
    }
}

$global:LASTEXITCODE = 0
Write-Output "Rendered benchmark launch passed: $outputDir"
