param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $ProjectRoot 'scripts\run_scene_material_matrix.ps1'
if (-not (Test-Path -LiteralPath $runner)) { throw "Missing scene/material runner: $runner" }
$root = Join-Path $ProjectRoot (Join-Path 'results' ('test-scene-material-contract-' + $PID))
& $runner -ProjectRoot $ProjectRoot -OutputDir $root -Stage manifest -DryRun
$manifestPath = Join-Path $root 'manifest.json'
if (-not (Test-Path -LiteralPath $manifestPath)) { throw 'Scene/material manifest was not produced.' }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.measurement -ne 'rendered-end-to-end' -or $manifest.meshes.Count -ne 2 -or $manifest.meshes[1].width -ne 512 -or $manifest.meshes[1].height -ne 128) {
    throw 'Scene/material manifest does not encode the required rendered square/rectangular matrix.'
}
if ($manifest.stretch_stiffnesses.Count -ne 3 -or $manifest.bending_stiffnesses.Count -ne 3) { throw 'Scene/material manifest does not encode the material matrix.' }
Write-Host "Scene/material matrix contract passed: $root"
