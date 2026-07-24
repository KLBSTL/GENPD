param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'

@('run_stability_sweep.ps1', 'plot_stability_heatmap.ps1', 'test_phase4_runtime.ps1') | ForEach-Object {
    $scriptPath = Join-Path $ProjectRoot (Join-Path 'scripts' $_)
    if (-not (Test-Path -LiteralPath $scriptPath)) {
        throw "Missing Phase 4 experiment tool: $scriptPath"
    }
}

$scenePath = Join-Path $ProjectRoot 'scenes\moving_sphere_cloth.xml'
if (-not (Test-Path -LiteralPath $scenePath)) {
    throw "Missing moving-obstacle scene: $scenePath"
}

$scene = Get-Content -LiteralPath $scenePath -Raw
if ($scene -notmatch '<sphere ' -or $scene -notmatch 'v[xyz]="') {
    throw 'Moving-obstacle scene must declare a sphere with a nonzero velocity attribute.'
}

$sweep = Get-Content -LiteralPath (Join-Path $ProjectRoot 'scripts\run_stability_sweep.ps1') -Raw
if ($sweep -notmatch 'IsPathRooted\(\$Scene\)') {
    throw 'Stability sweep must support both relative and absolute scene paths.'
}

Write-Output 'Phase 4 stability tool contract passed.'
