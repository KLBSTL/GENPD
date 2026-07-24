param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$runner = Join-Path $ProjectRoot 'scripts\run_paper_experiments.ps1'
$runRoot = Join-Path $ProjectRoot (Join-Path 'results' ('test-paper-partial-' + $PID))

& $runner -ProjectRoot $ProjectRoot -RunRoot $runRoot -Stage manifest -DryRun `
    -SceneIds hanging -ClothDimensions 128 -AllowPartialMatrix 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Partial-matrix R2 manifest dry run failed.' }

$manifestPath = Join-Path $runRoot 'manifest.json'
if (-not (Test-Path -LiteralPath $manifestPath)) { throw 'Partial-matrix R2 manifest was not created.' }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ([bool]$manifest.scope.complete_matrix -or [bool]$manifest.scope.paper_figure_eligible `
    -or $manifest.scope.purpose -ne 'protocol-preflight-not-paper-evidence' `
    -or @($manifest.scope.selected_scene_ids).Count -ne 1 -or $manifest.scope.selected_scene_ids[0] -ne 'hanging' `
    -or @($manifest.scope.selected_resolutions).Count -ne 1 -or [int]$manifest.scope.selected_resolutions[0] -ne 128) {
    throw 'Partial-matrix R2 manifest is not explicitly marked as non-paper evidence.'
}

$rejected = $false
try {
    & $runner -ProjectRoot $ProjectRoot -RunRoot (Join-Path $ProjectRoot (Join-Path 'results' ('test-paper-filter-reject-' + $PID))) `
        -Stage manifest -DryRun -SceneIds hanging -ClothDimensions 128 2>&1 | Out-Null
    $rejected = $LASTEXITCODE -ne 0
}
catch {
    $rejected = $true
}
if (-not $rejected) { throw 'Filtered R2 invocation did not require -AllowPartialMatrix.' }

$global:LASTEXITCODE = 0
Write-Output "Partial-matrix R2 contract passed: $manifestPath"
