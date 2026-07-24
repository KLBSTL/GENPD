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
if ($manifest.protocol_version -ne 2 -or $manifest.quality_target.position_rel_l2_p95 -ne 0.001 -or $manifest.performance.repetitions -ne 3 `
    -or $manifest.measurement.mode -ne 'rendered-end-to-end' `
    -or $manifest.measurement.primary_metric -ne 'frame_wall_ms' `
    -or $manifest.measurement.render_width -ne 1600 -or $manifest.measurement.render_height -ne 900) {
    throw 'Paper experiment manifest does not encode the approved protocol.'
}
if ($manifest.variants -notcontains 'gpu-xpbd-jacobi' -or $manifest.quality_target.xpbd.p95_max_stretch_strain -ne 0.1) {
    throw 'Paper experiment manifest does not encode the GPU XPBD quality gate.'
}
if (-not [bool]$manifest.validity_policy.invalid_frame_blocks_performance -or $manifest.validity_policy.required_gather_dimension -ne 386) {
    throw 'Paper experiment manifest does not encode the required gather validity policy.'
}
if ($manifest.calibration.candidate_iterations.Count -ne 13 -or
    $manifest.calibration.candidate_iterations[-1] -ne 64 -or
    $manifest.validity_policy.validity_matrix_schema -ne 'qualified-invalid-termination-reason') {
    throw 'Paper experiment manifest does not encode the R2 calibration and validity contract.'
}
if ($manifest.execution.process_timeout_seconds -ne 600 -or $manifest.execution.inter_run_delay_milliseconds -ne 1000) {
    throw 'Paper experiment manifest does not encode the R2 execution protection contract.'
}

Write-Output "Paper experiment contract passed: $manifestPath"
