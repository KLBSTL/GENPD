param(
    [string]$ProjectRoot = ''
)

$ErrorActionPreference = 'Stop'
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $PSCommandPath }
if ($ProjectRoot -eq '') {
    $ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $ScriptDir '..'))
}
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$Runner = Join-Path $ScriptDir 'run_benchmark.ps1'
$ExePath = Join-Path $ProjectRoot 'x64\Release\GenPD.exe'

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$invalidOutput = & $ExePath '--benchmark' '--project-root' $ProjectRoot '--no-render' `
    '--solver-variant' 'gpu-gather-fusion-adaptive-ls-persistent' `
    '--adaptive-ls-history' 'invalid' '--frames' '1' '--warmup' '0' 2>&1
$invalidExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
if ($invalidExitCode -eq 0 -or ($invalidOutput -join "`n") -notmatch 'Unknown --adaptive-ls-history') {
    throw 'Invalid adaptive history mode was not rejected by GenPD.exe.'
}

foreach ($mode in @('none', 'iteration', 'frame')) {
    $label = "test-adaptive-ls-history-$mode"
    $output = Join-Path $ProjectRoot (Join-Path 'results' $label)
    & $Runner -ProjectRoot $ProjectRoot -RunLabel $label -OutputDir $output `
        -SolverVariant 'gpu-gather-fusion-adaptive-ls-persistent' `
        -AdaptiveLsHistory $mode -Frames 1 -Warmup 0 -NoRender $true | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Adaptive history mode '$mode' did not complete."
    }
    $metadataPath = Join-Path $output 'run_metadata.json'
    if (-not (Test-Path -LiteralPath $metadataPath)) {
        throw "Missing metadata for adaptive history mode '$mode'."
    }
    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    if ($metadata.solver_controls.adaptive_ls_history -ne $mode) {
        throw "Metadata history mode mismatch for '$mode'."
    }
}

Write-Host 'Adaptive line-search CLI and metadata contract passed.'
