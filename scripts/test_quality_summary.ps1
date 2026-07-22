param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'

$summaryScript = Join-Path $ProjectRoot 'scripts\summarize_quality_metrics.ps1'
$testRoot = Join-Path $ProjectRoot ("results\\test-quality-summary-" + $PID)
$inputCsv = Join-Path $testRoot 'quality_metrics.csv'
$outputCsv = Join-Path $testRoot 'quality_summary.csv'
New-Item -ItemType Directory -Force -Path $testRoot | Out-Null

@'
frame,has_reference,finite,exploded,position_rel_l2,velocity_rel_l2,constraint_energy,constraint_energy_rel_error,mean_stretch_strain,max_stretch_strain,max_penetration_depth
0,1,1,0,0.1,0.2,1.0,0.3,0.01,0.02,0.0
1,1,0,1,0.3,0.4,3.0,0.5,0.03,0.04,0.2
2,0,1,0,0.0,0.0,2.0,0.0,0.02,0.03,0.1
'@ | Set-Content -LiteralPath $inputCsv -NoNewline

& $summaryScript -InputCsv $inputCsv -OutputCsv $outputCsv 2>&1 | Out-Null
if (-not $?) {
    throw 'Quality summary failed.'
}

$summary = Import-Csv -LiteralPath $outputCsv
$position = $summary | Where-Object metric -eq 'position_rel_l2'
if ($null -eq $position -or [int]$position.count -ne 2 -or [math]::Abs(([double]$position.mean) - 0.2) -gt 1e-9) {
    throw 'Position relative-error summary is incorrect.'
}

$failure = $summary | Where-Object metric -eq 'failure_rate'
if ($null -eq $failure -or [int]$failure.count -ne 3 -or [math]::Abs(([double]$failure.mean) - (1.0 / 3.0)) -gt 1e-9) {
    throw 'Failure-rate summary is incorrect.'
}

Write-Output "Quality summary contract passed: $testRoot"
