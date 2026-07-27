param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonExe = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$analysis = Join-Path $ProjectRoot 'scripts\analyze_minimal_optimizer_effects.py'
if (-not (Test-Path -LiteralPath $analysis)) {
    throw "Missing minimal optimizer-effects analysis script: $analysis"
}
if ($PythonExe -eq '') { $PythonExe = if (Test-Path -LiteralPath 'E:\Anaconda\envs\DL\python.exe') { 'E:\Anaconda\envs\DL\python.exe' } else { 'python' } }

$help = & $PythonExe $analysis --help 2>&1
$helpText = [string]::Join("`n", @($help))
if ($LASTEXITCODE -ne 0 -or $helpText -notmatch '--run-root' -or $helpText -notmatch '--report-path') {
    throw 'Minimal optimizer-effects analyzer does not expose the required CLI.'
}

$global:LASTEXITCODE = 0
Write-Host 'Minimal optimizer-effects analyzer contract passed.'
