param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$benchmark = Join-Path $ProjectRoot 'scripts\run_benchmark.ps1'
$runDir = Join-Path $ProjectRoot ('results\test-rendered-presentation-' + $PID)

& $benchmark -ProjectRoot $ProjectRoot -RunLabel 'test-rendered-presentation' `
    -Frames 2 -Warmup 0 -SolverVariant 'gpu-gather-fusion-batched-ls-persistent' `
    -OutputDir $runDir -NoRender:$false -Uncapped:$true -SyncGpu -DisableVsync `
    -RenderWidth 320 -RenderHeight 240 -ExtraArgs @('--cloth-dimension', 16, '--scene', 'scenes\test_scene.xml')

$metadata = Get-Content -LiteralPath (Join-Path $runDir 'run_metadata.json') -Raw | ConvertFrom-Json
if ([bool]$metadata.benchmark.no_render -or -not [bool]$metadata.benchmark.sync_gpu -or -not [bool]$metadata.benchmark.disable_vsync) {
    throw 'Rendered benchmark metadata does not record the required presentation protocol.'
}

$rows = @(Import-Csv -LiteralPath (Join-Path $runDir 'frame_presentation.csv'))
if ($rows.Count -ne 2) { throw "Expected 2 presentation rows, found $($rows.Count)." }
foreach ($row in $rows) {
    if ($row.rendered -ne '1' -or $row.gpu_sync_enabled -ne '1' -or [int]$row.screen_width -ne 320 -or [int]$row.screen_height -ne 240) {
        throw 'Presentation profile does not match the rendered benchmark contract.'
    }
    if ([double]$row.frame_wall_ms -le 0.0 -or [double]$row.render_and_present_wall_ms -le 0.0) {
        throw 'Presentation timings must be positive.'
    }
}

Write-Host "Rendered presentation contract passed: $runDir"
