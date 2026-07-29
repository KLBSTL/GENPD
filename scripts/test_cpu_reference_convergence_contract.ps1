$ErrorActionPreference = 'Stop'
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir '..'))
$analysis = Join-Path $scriptDir 'analyze_cpu_reference_convergence.py'
$runRoot = Join-Path $projectRoot 'results\paper-20260729-cpu-reference-sanity-r1'
foreach ($path in @($analysis, (Join-Path $runRoot 'manifest.json'))) { if (-not (Test-Path -LiteralPath $path)) { throw "Missing CPU convergence audit input: $path" } }
$python = if (Test-Path -LiteralPath 'E:\Anaconda\envs\DL\python.exe') { 'E:\Anaconda\envs\DL\python.exe' } else { 'python' }
& $python $analysis --run-root $runRoot
if ($LASTEXITCODE -ne 0) { throw 'CPU convergence audit failed.' }
$summary = @(Import-Csv -LiteralPath (Join-Path $runRoot 'cpu_reference_convergence_summary.csv'))
if ($summary.Count -ne 6 -or @($summary | Where-Object { [int]$_.converged_before_cap_frames -ne 30 }).Count -ne 0) {
    throw 'CPU convergence audit summary violates the expected reference-ladder contract.'
}
Write-Host 'CPU reference convergence contract passed.'
