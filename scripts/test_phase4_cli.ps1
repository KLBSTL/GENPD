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
    '--timestep FLOAT',
    '--stretch-stiffness FLOAT',
    '--bending-stiffness FLOAT',
    '--cloth-dimension N',
    '--scene PATH'
) | ForEach-Object {
    if ($help -notmatch [regex]::Escape($_)) {
        throw "Missing Phase 4 option in --help: $_"
    }
}

Write-Output 'Phase 4 CLI contract passed.'
