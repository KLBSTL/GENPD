param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $ProjectRoot 'scripts\run_line_search_sweep.ps1'
if (-not (Test-Path -LiteralPath $runner)) { throw "Missing line-search sweep runner: $runner" }
$root = Join-Path $ProjectRoot (Join-Path 'results' ('test-line-search-contract-' + $PID))
& $runner -ProjectRoot $ProjectRoot -OutputDir $root -DryRun -KValues 1,2 -Betas 0.5 -RestartModes none,periodic -RestartPeriod 3 `
    -Schedules fixed,adaptive -AdaptiveHistoryModes none,iteration,frame -IncludePersistent
if (-not (Test-Path -LiteralPath (Join-Path $root 'manifest.json'))) { throw 'Line-search manifest was not generated.' }
$manifest = Get-Content -LiteralPath (Join-Path $root 'manifest.json') -Raw | ConvertFrom-Json
if ($manifest.protocol_version -ne 3 -or $manifest.measurement -ne 'rendered-end-to-end' -or $manifest.timing_run_has_decision_tracing -or -not $manifest.trace_run_has_decision_tracing -or $manifest.timing.repetitions -ne 3 `
    -or @($manifest.schedules).Count -ne 2 -or @($manifest.adaptive_history_modes).Count -ne 3 `
    -or $manifest.fixed_variant -ne 'gpu-gather-fusion-batched-ls-persistent' -or $manifest.adaptive_variant -ne 'gpu-gather-fusion-adaptive-ls-persistent') {
    throw 'Line-search runner does not separate rendered timing and diagnostic trace runs.'
}
Write-Host "Line-search sweep contract passed: $root"
