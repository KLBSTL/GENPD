param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$runner = Join-Path $ProjectRoot 'scripts\run_minimal_optimizer_effects_recheck.ps1'
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Missing minimal optimizer-effects runner: $runner"
}

$runRoot = Join-Path $ProjectRoot (Join-Path 'results' ('test-minimal-optimizer-effects-' + $PID))
& $runner -ProjectRoot $ProjectRoot -RunRoot $runRoot -DryRun
if ($LASTEXITCODE -ne 0) {
    throw 'Minimal optimizer-effects dry-run failed.'
}

$manifestPath = Join-Path $runRoot 'manifest.json'
$planPath = Join-Path $runRoot 'planned_timing_runs.csv'
foreach ($path in @($manifestPath, $planPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Minimal optimizer-effects dry-run did not write $path"
    }
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.protocol_version -ne 'minimal-optimizer-effects-r1') {
    throw 'Unexpected minimal optimizer-effects protocol version.'
}
if ($manifest.measurement.mode -ne 'rendered-end-to-end' -or $manifest.measurement.render_width -ne 1600 -or $manifest.measurement.render_height -ne 900 `
    -or -not $manifest.measurement.sync_gpu -or -not $manifest.measurement.disable_vsync) {
    throw 'Minimal optimizer-effects timing configuration is not rendered and synchronized.'
}
if ($manifest.timing.frames -ne 150 -or $manifest.timing.warmup -ne 30 -or $manifest.timing.repetitions -ne 6 `
    -or $manifest.quality.frames -ne 120 -or $manifest.quality.warmup -ne 20 -or $manifest.quality.position_gate_p95 -ne 0.001) {
    throw 'Minimal optimizer-effects frame counts or quality gate are incorrect.'
}
if ($manifest.practical_effect_threshold -ne 0.03) {
    throw 'Minimal optimizer-effects practical threshold must be 3%.'
}
if (@($manifest.baseline.conditions).Count -ne 3 -or @($manifest.stress.conditions).Count -ne 6) {
    throw 'Expected three baseline and six stress conditions.'
}
if ($manifest.baseline.cloth_dimension -ne 386 -or $manifest.baseline.iterations_per_frame -ne 1 `
    -or $manifest.stress.cloth_dimension -ne 386 -or $manifest.stress.iterations_per_frame -ne 8 `
    -or $manifest.baseline.scene -ne 'scenes\moving_sphere_cloth.xml' -or $manifest.stress.scene -ne 'scenes\moving_sphere_cloth.xml') {
    throw 'Baseline and stress cells do not match the pre-registered workload.'
}

$runs = @(Import-Csv -LiteralPath $planPath)
if ($runs.Count -ne 54) {
    throw "Expected 54 timing runs, found $($runs.Count)."
}
if (@($runs | Where-Object { [int]$_.block -lt 1 -or [int]$_.block -gt 6 -or [int]$_.order_index -lt 1 }).Count -ne 0) {
    throw 'Timing plan lacks valid interleaved blocks.'
}
if (@($runs | Where-Object { $_.suite -eq 'baseline' }).Count -ne 18 -or @($runs | Where-Object { $_.suite -eq 'stress' }).Count -ne 36) {
    throw 'Timing plan does not have balanced baseline and stress coverage.'
}

$global:LASTEXITCODE = 0
Write-Host "Minimal optimizer-effects contract passed: $runRoot"
