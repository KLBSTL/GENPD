param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $ProjectRoot 'scripts\run_gradient_diagnostics.ps1'
if (-not (Test-Path -LiteralPath $runner)) { throw "Missing gradient diagnostics runner: $runner" }
$root = Join-Path $ProjectRoot (Join-Path 'results' ('test-gradient-diagnostics-contract-' + $PID))
& $runner -ProjectRoot $ProjectRoot -OutputDir $root -ClothDimensions 128 -SceneIds hanging -DryRun
$manifestPath = Join-Path $root 'manifest.json'
if (-not (Test-Path -LiteralPath $manifestPath)) { throw 'Gradient diagnostic manifest was not generated.' }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if (-not [bool]$manifest.diagnostic_only -or $manifest.rendering -notmatch 'not paper' -or $manifest.cloth_dimensions.Count -ne 1 -or $manifest.scenes.Count -ne 1) {
    throw 'Gradient diagnostics manifest does not distinguish diagnostics from paper evidence.'
}
Write-Host "Gradient diagnostics contract passed: $root"
