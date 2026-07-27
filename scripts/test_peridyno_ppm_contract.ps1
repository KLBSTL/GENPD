param(
    [string] $ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$adapter = Join-Path $ProjectRoot 'external_baselines\peridyno_ppm\GenPD_PPM_OperatingPoint.cpp'
$peridynoPatch = Join-Path $ProjectRoot 'external_baselines\peridyno_ppm\peridyno-ppm-genpd-operating-point.patch'
$runner = Join-Path $ProjectRoot 'scripts\run_peridyno_ppm_operating_point.ps1'
$protocol = Join-Path $ProjectRoot 'docs\experiments\2026-07-27-peridyno-ppm-operating-point.md'

foreach ($path in @($adapter, $peridynoPatch, $runner, $protocol)) {
    if (!(Test-Path $path)) {
        throw "Missing PPM operating-point artifact: $path"
    }
}

$source = Get-Content -Raw $adapter
foreach ($required in @(
    'CodimensionalPD<DataType3f>',
    'FixedPoints<DataType3f>',
    'SphereModel<DataType3f>',
    'BasicShapeToVolume<DataType3f>',
    'VolumeBoundary<DataType3f>',
    'setSelfContact(false)',
    'UpdateMovingSphere',
    'GLSurfaceVisualModule',
    'GLSurfaceVisualNode<DataType3f>',
    'setVisible(true)',
    'objects.sphere->reset()',
    'sphere_render_vertex_count',
    'sphere_render_min',
    'renderOneFrame',
    'captureFrame',
    'app.release()',
    'same-hardware operating point')) {
    if ($source -notmatch [regex]::Escape($required)) {
        throw "PPM adapter is missing required mapped-scene component: $required"
    }
}

$runnerText = Get-Content -Raw $runner
if ($runnerText -notmatch 'GenPD_PPM_OperatingPoint' -or $runnerText -notmatch '--peridyno-commit' -or
    $runnerText -notmatch 'peridyno-ppm-genpd-operating-point.patch' -or
    $runnerText -notmatch '--render-width') {
    throw 'PPM runner does not build the tracked adapter or record its source revision.'
}

$cmakeText = Get-Content -Raw (Join-Path $ProjectRoot 'external_baselines\peridyno_ppm\CMakeLists.txt')
if ($cmakeText -notmatch 'GenPD_PPM_OperatingPoint_Rendered') {
    throw 'PPM adapter CMake target does not use the rendered operating-point executable name.'
}

$patchText = Get-Content -Raw $peridynoPatch
foreach ($required in @('selfContact', 'mContactRule', 'renderOneFrame', 'captureFrame')) {
    if ($patchText -notmatch [regex]::Escape($required)) {
        throw "PeriDyno patch is missing required compatibility behavior: $required"
    }
}

$protocolText = Get-Content -Raw $protocol
foreach ($required in @('not an equal-model or equal-quality ranking', 'moving-sphere', 'headless', 'frame_host_ms')) {
    if ($protocolText -notmatch [regex]::Escape($required)) {
        throw "PPM protocol is missing required evidence boundary: $required"
    }
}

Write-Host 'PeriDyno PPM operating-point contract passed.'
