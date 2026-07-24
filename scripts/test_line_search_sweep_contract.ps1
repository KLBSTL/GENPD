param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $ProjectRoot 'scripts\run_line_search_sweep.ps1'
if (-not (Test-Path -LiteralPath $runner)) { throw "Missing line-search sweep runner: $runner" }
$root = Join-Path $ProjectRoot (Join-Path 'results' ('test-line-search-contract-' + $PID))
$dryRunOutput = @(& $runner -ProjectRoot $ProjectRoot -OutputDir $root -DryRun -KValues 1,2 -Betas 0.5 -RestartModes none,periodic -RestartPeriod 3 `
    -Schedules serial,fixed,adaptive -AdaptiveHistoryModes none,iteration,frame -IncludePersistent 6>&1)
if (-not (Test-Path -LiteralPath (Join-Path $root 'manifest.json'))) { throw 'Line-search manifest was not generated.' }
$manifest = Get-Content -LiteralPath (Join-Path $root 'manifest.json') -Raw | ConvertFrom-Json
if ($manifest.protocol_version -ne 3 -or $manifest.measurement -ne 'rendered-end-to-end' -or $manifest.timing_run_has_decision_tracing -or -not $manifest.trace_run_has_decision_tracing -or $manifest.timing.repetitions -ne 3 `
    -or @($manifest.schedules).Count -ne 3 -or @($manifest.adaptive_history_modes).Count -ne 3 `
    -or $manifest.serial_variant -ne 'gpu-gather-fusion-serial-ls-persistent' -or $manifest.fixed_variant -ne 'gpu-gather-fusion-batched-ls-persistent' -or $manifest.adaptive_variant -ne 'gpu-gather-fusion-adaptive-ls-persistent') {
    throw 'Line-search runner does not separate rendered timing and diagnostic trace runs.'
}
$serialLines = @($dryRunOutput | Where-Object { ("$_") -match '^\[dry-run\] (timing|trace) serial-' })
if ($serialLines.Count -ne 8) {
    throw "Serial schedule expanded into $($serialLines.Count) dry-run entries instead of 8."
}
Write-Host "Line-search sweep contract passed: $root"
