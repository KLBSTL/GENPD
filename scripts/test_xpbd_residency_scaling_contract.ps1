$ErrorActionPreference = 'Stop'
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir '..'))
$runner = Join-Path $scriptDir 'run_xpbd_residency_scaling_paper_study.ps1'
$analysis = Join-Path $scriptDir 'analyze_xpbd_residency_scaling_paper_study.py'
foreach ($path in @($runner, $analysis)) { if (-not (Test-Path -LiteralPath $path)) { throw "Missing XPBD scaling artifact: $path" } }
$root = [System.IO.Path]::GetFullPath((Join-Path $projectRoot 'results\diagnostic-xpbd-residency-scaling-contract'))
& $runner -ProjectRoot $projectRoot -RunLabel paper-xpbd-residency-scaling-contract -RunRoot $root -DryRun
if ($LASTEXITCODE -ne 0) { throw 'XPBD residency scaling dry run failed.' }
$manifest = Get-Content -LiteralPath (Join-Path $root 'manifest.json') -Raw | ConvertFrom-Json
if ($manifest.protocol_version -ne 'xpbd-residency-scaling-v1' -or $manifest.cloth_dimensions.Count -ne 3 -or $manifest.timing.frames -ne 300 -or $manifest.timing.repetitions -ne 3) {
    throw 'XPBD residency scaling manifest does not encode the formal protocol.'
}
if (-not $root.StartsWith((Join-Path $projectRoot 'results') + [System.IO.Path]::DirectorySeparatorChar) -or (Split-Path -Leaf $root) -ne 'diagnostic-xpbd-residency-scaling-contract') {
    throw "Unexpected contract cleanup target: $root"
}
Remove-Item -LiteralPath $root -Recurse -Force
Write-Host 'XPBD residency scaling contract passed.'
