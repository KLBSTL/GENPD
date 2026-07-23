param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'

@(
    'run_paper_experiments.ps1',
    'summarize_paper_experiments.ps1',
    'generate_paper_figures.py'
) | ForEach-Object {
    $path = Join-Path $ProjectRoot (Join-Path 'scripts' $_)
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing paper experiment tool: $path"
    }
}

$runRoot = Join-Path $ProjectRoot ("results\\test-paper-contract-" + $PID)
& (Join-Path $ProjectRoot 'scripts\run_paper_experiments.ps1') -ProjectRoot $ProjectRoot -RunRoot $runRoot -Stage manifest -DryRun 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw 'Paper experiment manifest dry run failed.'
}

$manifestPath = Join-Path $runRoot 'manifest.json'
if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw 'Paper experiment manifest was not created.'
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.quality_target.position_rel_l2_p95 -ne 0.001 -or $manifest.performance.repetitions -ne 3 `
    -or $manifest.measurement.mode -ne 'rendered-end-to-end' `
    -or $manifest.measurement.primary_metric -ne 'frame_wall_ms' `
    -or $manifest.measurement.render_width -ne 1600 -or $manifest.measurement.render_height -ne 900) {
    throw 'Paper experiment manifest does not encode the approved protocol.'
}

Write-Output "Paper experiment contract passed: $manifestPath"
