param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Configuration = 'Release',
    [string]$Platform = 'x64'
)

$ErrorActionPreference = 'Stop'

$executablePath = Join-Path $ProjectRoot ("x64\\{0}\\GenPD.exe" -f $Configuration)
if (-not (Test-Path -LiteralPath $executablePath)) {
    throw "GenPD executable was not found: $executablePath"
}

$help = (& $executablePath --help 2>&1) -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw "GenPD --help failed with exit code $LASTEXITCODE"
}

@(
    '--iterations-per-frame N',
    '--reference-export-dir PATH',
    '--quality-reference-dir PATH',
    '--quality-checkpoint-stride N',
    '--quality-metrics'
) | ForEach-Object {
    if ($help -notmatch [regex]::Escape($_)) {
        throw "Missing quality experiment option in --help: $_"
    }
}

Write-Output 'Quality metrics CLI contract passed.'
